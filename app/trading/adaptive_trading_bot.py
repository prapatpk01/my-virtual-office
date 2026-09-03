"""Adaptive SMC Execution Bot V7.2.

Entry: 4H TSS -> M15 structure -> M5 AMD -> M1 IFVG -> micro BOS -> pullback.
V7.2 adds three protections:
- no-chase pullback entry is supplied by indicator_engine,
- an SL blocks re-entry on the same AMD cycle,
- after TP1, the remaining runner trails confirmed M5 structure while M15 CHOCH
  remains the hard structural invalidation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, Optional
import json
import logging
import os
import sys
import time

TP1_R = float(os.getenv("SMC_TP1_R", "1.0"))
TP2_R = float(os.getenv("SMC_TP2_R", "2.0"))
BE_LOCK_R = float(os.getenv("SMC_BE_LOCK_R", "0.10"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
MIN_SL_PCT = float(os.getenv("SMC_MIN_SL_PCT", "0.0025"))
MAX_SL_PCT = float(os.getenv("SMC_MAX_SL_PCT", "0.030"))
COOLDOWN_BARS = int(os.getenv("SMC_COOLDOWN_1M_BARS", "5"))
SUPPORTED_SCHEMAS = {"adaptive-smc-mtf-v1"}


@dataclass
class Position:
    direction: str
    entry: float
    sl: float
    initial_sl: float
    tp: float
    tp1: float
    tp2: float
    size: float
    initial_size: float
    strategy: str
    trigger: str
    opened_at: float
    tp1_hit: bool = False
    be_moved: bool = False
    style: str = "LEGACY"
    tp2_hit: bool = False
    runner_active: bool = False
    tss_bias: str = ""
    structure: str = ""
    amd_phase: str = ""
    ifvg_low: float = 0.0
    ifvg_high: float = 0.0
    manipulation_low: float = 0.0
    manipulation_high: float = 0.0
    amd_cycle_id: str = ""
    sl_algo_id: str = ""


class TradingBot:
    def __init__(self, symbol: str, margin_usdt: float = 20.0, leverage: int = 20,
                 paper: bool = True, state_file: str = "",
                 execution_callback: Optional[Callable] = None,
                 risk_usdt: float = RISK_USDT, **_kwargs):
        self.symbol = symbol
        self.margin_usdt = float(margin_usdt)
        self.leverage = int(leverage)
        self.paper = bool(paper)
        self.state_file = state_file
        self.execution_callback = execution_callback
        self.risk_usdt = float(risk_usdt)
        self.position: Optional[Position] = None
        self.cooldown_remaining = 0
        self.blocked_amd_cycle_id = ""
        self.last_signal = "WARMUP"
        self.counts = {"scans": 0, "entries": 0, "cooldown": 0, "wait": 0, "rearm": 0}
        self._identity()
        self.load_state()

    @staticmethod
    def _identity() -> None:
        try:
            runner = sys.modules.get("run_bot") or sys.modules.get("__main__")
            if runner is not None and hasattr(runner, "logger"):
                runner.logger = logging.getLogger("adaptive_smc_v7_2")
            if runner is not None and hasattr(runner, "BUILD_ID"):
                runner.BUILD_ID = "adaptive-smc-v7.2-2026-09-03"
        except Exception:
            pass

    @property
    def position_open(self) -> bool:
        return self.position is not None

    def load_state(self) -> None:
        if not self.state_file or not os.path.exists(self.state_file): return
        try:
            with open(self.state_file, encoding="utf-8") as handle: raw = json.load(handle)
            position_raw = raw.get("position")
            if position_raw:
                allowed = {field.name for field in fields(Position)}
                self.position = Position(**{k:v for k,v in position_raw.items() if k in allowed})
            self.cooldown_remaining = int(raw.get("cooldown_remaining", 0))
            self.blocked_amd_cycle_id = str(raw.get("blocked_amd_cycle_id", "") or "")
        except Exception as exc:
            logging.getLogger("adaptive_smc_v7_2").warning("state load failed: %s", exc)
            self.position = None; self.cooldown_remaining = 0; self.blocked_amd_cycle_id = ""

    def save_state(self) -> None:
        if not self.state_file: return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({"position": asdict(self.position) if self.position else None,
                       "cooldown_remaining": self.cooldown_remaining,
                       "blocked_amd_cycle_id": self.blocked_amd_cycle_id}, handle)
        os.replace(temp, self.state_file)

    def _debug(self, i: Dict, reason: str) -> str:
        s=self.symbol.split("/")[0]
        if reason=="COOLDOWN": return f"SMC V7.2 · {s} · COOLDOWN {self.cooldown_remaining}x1M · WAIT"
        if reason=="REARM": return f"SMC V7.2 · {s} · WAIT NEW AMD CYCLE"
        tss=str(i.get("tss_bias","NEUTRAL")); st=str(i.get("structure","UNKNOWN")); amd=str(i.get("amd_phase","WAIT"))
        ml=i.get("micro_long") or {}; ms=i.get("micro_short") or {}
        return f"SMC V7.2 · {s} · 4H={tss} · M15={st} · M5={amd} · M1 PB L={int(bool(ml.get('entry_ready')))}/S={int(bool(ms.get('entry_ready')))} · WAIT"

    def _build(self, i: Dict, direction: str, trigger: str, live_price: float):
        entry=float(live_price or i.get("close",0.0)); raw_sl=float(i.get("sl",0.0) or 0.0)
        if entry<=0 or raw_sl<=0:return None
        if direction=="LONG":
            if raw_sl>=entry:return None
            risk=entry-raw_sl
        else:
            if raw_sl<=entry:return None
            risk=raw_sl-entry
        min_risk=entry*MIN_SL_PCT
        if risk<min_risk:
            raw_sl=entry-min_risk if direction=="LONG" else entry+min_risk; risk=min_risk
        sl_pct=risk/entry
        if sl_pct>MAX_SL_PCT:
            self.last_signal=f"WAIT SMC stop too wide {sl_pct*100:.2f}% > {MAX_SL_PCT*100:.2f}%"; return None
        tp1=entry+TP1_R*risk if direction=="LONG" else entry-TP1_R*risk
        tp2=entry+TP2_R*risk if direction=="LONG" else entry-TP2_R*risk
        size=min(self.risk_usdt/max(risk,1e-12),(self.margin_usdt*self.leverage)/max(entry,1e-12))
        if size<=0:return None
        return {"direction":direction,"style":"SMC_MTF_V1","strategy":"adaptive_smc_mtf_v7_2","trigger":trigger,
                "entry":entry,"sl":raw_sl,"tp":tp2,"tp1":tp1,"tp2":tp2,"size":size,
                "risk_usdt":size*risk,"sl_pct":100*sl_pct,"tss_bias":str(i.get("tss_bias","")),
                "tss_score":float(i.get("tss_score",0)),"structure":str(i.get("structure","")),
                "structure_bias":str(i.get("structure_bias","")),"amd_phase":str(i.get("amd_phase","")),
                "amd_cycle_id":str(i.get("amd_cycle_id","") or ""),"ifvg_low":float(i.get("ifvg_low",0)),
                "ifvg_high":float(i.get("ifvg_high",0)),"manipulation_low":float(i.get("manipulation_low",0)),
                "manipulation_high":float(i.get("manipulation_high",0)),"rsi1":float(i.get("rsi1",50)),
                "atr1":float(i.get("atr1",0))}

    def _close(self, price: float, reason: str):
        p=self.position; assert p
        pnl=(price-p.entry)*p.size if p.direction=="LONG" else (p.entry-price)*p.size
        initial_risk=abs(p.entry-p.initial_sl)*max(p.initial_size,1e-12); r=pnl/initial_risk if initial_risk else 0.0
        payload={"symbol":self.symbol,"direction":p.direction,"style":p.style,"price":price,"entry":p.entry,
                 "sl":p.sl,"tp":p.tp2,"tp1":p.tp1,"tp2":p.tp2,"size":p.size,"strategy":p.strategy,
                 "trigger":p.trigger,"reason":reason,"pnl":pnl,"r_multiple":r,"amd_cycle_id":p.amd_cycle_id}
        if self.execution_callback:self.execution_callback("CLOSE_"+p.direction,payload)
        if reason in {"SL","LOCKED_SL"}:
            self.cooldown_remaining=COOLDOWN_BARS
            if p.amd_cycle_id:self.blocked_amd_cycle_id=p.amd_cycle_id
        self.position=None; self.save_state(); self.last_signal=f"CLOSE {reason} pnl=${pnl:+.2f} r={r:+.2f}R"
        return {"event":"CLOSE",**payload}

    def _partial(self, price: float, qty: float, reason: str, r_multiple: float):
        p=self.position; assert p
        requested_qty=min(max(float(qty),0.0),p.size); requested_price=float(price)
        estimated_pnl=((requested_price-p.entry)*requested_qty if p.direction=="LONG" else (p.entry-requested_price)*requested_qty)
        initial_risk=abs(p.entry-p.initial_sl)*max(p.initial_size,1e-12); estimated_r=estimated_pnl/initial_risk if initial_risk else r_multiple
        payload={"symbol":self.symbol,"direction":p.direction,"style":p.style,"price":requested_price,"entry":p.entry,
                 "size":requested_qty,"trigger":p.trigger,"reason":reason,"pnl":estimated_pnl,"r_multiple":estimated_r}
        result=self.execution_callback("CLOSE_PARTIAL",payload) if self.execution_callback else None
        actual_qty=requested_qty; actual_price=requested_price; actual_pnl=None; already_flat=False
        if isinstance(result,dict):
            already_flat=bool(result.get("_already_flat")); filled=float(result.get("_exit_fill_sz") or 0); fill_px=float(result.get("_exit_avg_px") or 0)
            if already_flat:actual_qty=p.size
            elif filled>0:actual_qty=min(p.size,filled)
            if fill_px>0:actual_price=fill_px
            if "_realized_pnl" in result:actual_pnl=float(result.get("_realized_pnl") or 0)
        pnl=actual_pnl if actual_pnl is not None else ((actual_price-p.entry)*actual_qty if p.direction=="LONG" else (p.entry-actual_price)*actual_qty)
        rr=pnl/initial_risk if initial_risk else r_multiple
        return {"event":"PARTIAL",**payload,"price":actual_price,"size":actual_qty,"pnl":pnl,"r_multiple":rr,"already_flat":already_flat}

    def _amend_exchange_sl(self,new_sl:float)->None:
        p=self.position
        if not p or not p.sl_algo_id or not self.execution_callback:return
        try:self.execution_callback("AMEND_SL",{"symbol":self.symbol,"direction":p.direction,"sl_algo_id":p.sl_algo_id,"new_sl":float(new_sl)})
        except Exception as exc:logging.getLogger("adaptive_smc_v7_2").warning("[%s] exchange SL amend failed %.4f: %s",self.symbol,new_sl,exc)

    def check_price(self,price:float):
        p=self.position
        if not p:return None
        if (p.direction=="LONG" and price<=p.sl) or (p.direction=="SHORT" and price>=p.sl):
            return self._close(price,"LOCKED_SL" if p.be_moved else "SL")
        if not p.tp1_hit and ((p.direction=="LONG" and price>=p.tp1) or (p.direction=="SHORT" and price<=p.tp1)):
            qty=min(p.initial_size*.50,p.size); event=self._partial(price,qty,"TP1",TP1_R); p.size=max(0.0,p.size-float(event.get("size",qty)))
            if event.get("already_flat") or p.size<=1e-12:
                self.position=None; self.save_state(); event["event"]="CLOSE"; event["reason"]="TP1_EXCHANGE_FULL"; return event
            p.tp1_hit=True; p.runner_active=True; risk=abs(p.entry-p.initial_sl); lock=BE_LOCK_R*risk
            p.sl=p.entry+lock if p.direction=="LONG" else p.entry-lock; p.be_moved=True; self._amend_exchange_sl(p.sl); self.save_state(); return event
        if p.tp1_hit and not p.tp2_hit and ((p.direction=="LONG" and price>=p.tp2) or (p.direction=="SHORT" and price<=p.tp2)):
            p.tp2_hit=True; return self._close(price,"TP2")
        return None

    def reconcile_flat(self,price:float,reason:str="EXCHANGE_CLOSED"):
        return self.reconcile_exchange_closed(price,reason) if self.position else None

    def reconcile_exchange_closed(self,price:float,reason:str="EXCHANGE_CLOSED"):
        p=self.position
        if not p:return None
        px=float(price or p.entry); pnl=(px-p.entry)*p.size if p.direction=="LONG" else (p.entry-px)*p.size
        ir=abs(p.entry-p.initial_sl)*max(p.initial_size,1e-12); rr=pnl/ir if ir else 0.0
        payload={"symbol":self.symbol,"direction":p.direction,"style":p.style,"price":px,"entry":p.entry,"sl":p.sl,
                 "tp":p.tp2,"tp1":p.tp1,"tp2":p.tp2,"size":p.size,"strategy":p.strategy,"trigger":p.trigger,
                 "reason":reason,"pnl":pnl,"r_multiple":rr}
        self.position=None; self.save_state(); self.last_signal=f"RECONCILED {reason} @ {px:.4f}"; return {"event":"CLOSE",**payload}

    def adopt_exchange_position(self,direction:str,entry:float,size:float,sl:float,tp2:float,sl_algo_id:str="")->None:
        direction=direction.upper(); risk=abs(float(entry)-float(sl))
        if direction not in {"LONG","SHORT"} or entry<=0 or size<=0 or risk<=0:raise ValueError("invalid exchange position for adoption")
        tp1=entry+risk*TP1_R if direction=="LONG" else entry-risk*TP1_R; target=float(tp2 or 0)
        if target<=0:target=entry+risk*TP2_R if direction=="LONG" else entry-risk*TP2_R
        self.position=Position(direction=direction,entry=float(entry),sl=float(sl),initial_sl=float(sl),tp=target,tp1=tp1,tp2=target,
                               size=float(size),initial_size=float(size),strategy="recovered_exchange",trigger="Recovered after restart",
                               opened_at=time.time(),style="RECOVERED",sl_algo_id=str(sl_algo_id or ""))
        self.save_state(); self.last_signal=f"RECOVERED {direction} entry={entry:.4f} size={size:.6g}"

    def _manage_runner(self,i:Dict,live_price:float):
        p=self.position
        if not p or not p.runner_active:return None
        if p.style=="SMC_MTF_V1":
            trail=float(i.get("runner_trail_long" if p.direction=="LONG" else "runner_trail_short",0) or 0)
            if trail>0:
                better=(p.direction=="LONG" and trail>p.sl and trail<live_price) or (p.direction=="SHORT" and trail<p.sl and trail>live_price)
                if better:
                    p.sl=trail; p.be_moved=True; self._amend_exchange_sl(p.sl); self.save_state()
            invalid=(p.direction=="LONG" and bool(i.get("runner_exit_long"))) or (p.direction=="SHORT" and bool(i.get("runner_exit_short")))
            if invalid:return self._close(float(i.get("close",live_price)),"RUNNER_M15_CHOCH")
        else:
            mc=float(i.get("m15_close",0) or 0); me=float(i.get("m15_ema20",0) or 0)
            if mc and me and ((p.direction=="LONG" and mc<me) or (p.direction=="SHORT" and mc>me)):
                return self._close(mc,"RUNNER_EMA20_EXIT")
        return None

    def on_bar(self,i:Dict,_i1=None,_i4=None,price:float=0.0):
        if not i:self.last_signal="WAIT INDICATOR_WARMUP"; return None
        if i.get("schema") not in SUPPORTED_SCHEMAS:raise RuntimeError(f"ADAPTIVE_SMC_SCHEMA_MISMATCH: {i.get('schema')}")
        live_price=float(price or i.get("close",0.0))
        if self.position:
            event=self.check_price(live_price)
            if event:return event
            event=self._manage_runner(i,live_price)
            if event:return event
            p=self.position
            self.last_signal=f"MANAGE {p.style} {p.direction} | SL={p.sl:.4f} | TP1={p.tp1:.4f} | TP2={p.tp2:.4f} | Runner={int(p.runner_active)}"
            return None
        self.counts["scans"]+=1
        if self.cooldown_remaining>0:
            self.cooldown_remaining-=1; self.counts["cooldown"]+=1; self.save_state(); self.last_signal=self._debug(i,"COOLDOWN"); return None
        direction="LONG" if i.get("long_signal") else "SHORT" if i.get("short_signal") else "NONE"
        if direction=="NONE":self.counts["wait"]+=1; self.last_signal=self._debug(i,"WAIT"); return None
        cycle=str(i.get("amd_cycle_id","") or "")
        if cycle and cycle==self.blocked_amd_cycle_id:
            self.counts["rearm"]+=1; self.last_signal=self._debug(i,"REARM"); return None
        if cycle and self.blocked_amd_cycle_id and cycle!=self.blocked_amd_cycle_id:
            self.blocked_amd_cycle_id=""; self.save_state()
        trigger=str(i.get("trigger") or "4H → M15 → M5 → M1 IFVG → BOS → PULLBACK")
        payload=self._build(i,direction,trigger,live_price)
        if not payload:
            if not self.last_signal.startswith("WAIT SMC stop"):self.last_signal=f"SMC V7.2 · {self.symbol.split('/')[0]} · WAIT RISK BUILD"
            return None
        payload["symbol"]=self.symbol; result=self.execution_callback("OPEN_"+direction,payload) if self.execution_callback else None
        actual_size=payload["size"]
        if isinstance(result,dict):actual_size=float(result.get("_filled_coins") or actual_size)
        self.position=Position(direction=direction,entry=payload["entry"],sl=payload["sl"],initial_sl=payload["sl"],tp=payload["tp2"],
                               tp1=payload["tp1"],tp2=payload["tp2"],size=actual_size,initial_size=actual_size,strategy=payload["strategy"],
                               trigger=payload["trigger"],opened_at=time.time(),style="SMC_MTF_V1",tss_bias=payload["tss_bias"],
                               structure=payload["structure"],amd_phase=payload["amd_phase"],ifvg_low=payload["ifvg_low"],ifvg_high=payload["ifvg_high"],
                               manipulation_low=payload["manipulation_low"],manipulation_high=payload["manipulation_high"],amd_cycle_id=payload["amd_cycle_id"],
                               sl_algo_id=(str(result.get("_sl_algo_id") or "") if isinstance(result,dict) else ""))
        self.counts["entries"]+=1; self.save_state(); self.last_signal=f"ENTRY {direction} · {trigger}"; return {"event":"OPEN",**payload}
