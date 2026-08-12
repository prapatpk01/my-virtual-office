"""Adaptive Momentum v3.3 trading bot."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json, logging, os, sys, time

TP1_R=float(os.getenv("MOM_TP1_R","1.0")); TP2_R=float(os.getenv("MOM_TP2_R","2.0")); TP_R=TP2_R
SL_ATR=float(os.getenv("MOM_SL_ATR","1.0")); MIN_SL_PCT=float(os.getenv("MOM_MIN_SL_PCT","0.004")); RISK_USDT=float(os.getenv("MOM_RISK_USDT","5.0"))
ADX_MIN=float(os.getenv("MOM_ADX_MIN","15")); CHOP_MAX=float(os.getenv("MOM_CHOP_MAX","55")); COOLDOWN_BARS=int(os.getenv("MOM_COOLDOWN_BARS","3"))
SUPPORTED_SCHEMAS={"adaptive-momentum-v3.2-15m","adaptive-momentum-v3.3-15m"}

@dataclass
class Position:
    direction:str; entry:float; sl:float; initial_sl:float; tp:float; tp1:float; tp2:float; size:float; initial_size:float; strategy:str; trigger:str; opened_at:float
    tp1_hit:bool=False; be_moved:bool=False

class TradingBot:
    def __init__(self,symbol:str,margin_usdt:float=20.0,leverage:int=20,paper:bool=True,state_file:str="",execution_callback:Optional[Callable]=None,risk_usdt:float=RISK_USDT,**_kwargs):
        self.symbol=symbol; self.margin_usdt=float(margin_usdt); self.leverage=int(leverage); self.paper=bool(paper); self.state_file=state_file; self.execution_callback=execution_callback; self.risk_usdt=float(risk_usdt)
        self.position:Optional[Position]=None; self.cooldown_remaining=0; self.last_signal="WARMUP"
        self.counts={k:0 for k in ("scans","entries","cooldown","trend","quality","momentum","location","trigger")}
        self._identity(); self.load_state()

    @staticmethod
    def _identity():
        try:
            r=sys.modules.get("run_bot") or sys.modules.get("__main__")
            if r is not None and hasattr(r,"logger"): r.logger=logging.getLogger("adaptive_momentum_v3_3")
            if r is not None and hasattr(r,"BUILD_ID"): r.BUILD_ID="adaptive-momentum-v3.3-2026-08-12"
        except Exception: pass

    @property
    def position_open(self): return self.position is not None

    def load_state(self):
        if not self.state_file or not os.path.exists(self.state_file): return
        try:
            with open(self.state_file,encoding="utf-8") as f: raw=json.load(f)
            if raw.get("position"): self.position=Position(**raw["position"])
            self.cooldown_remaining=int(raw.get("cooldown_remaining",0))
        except Exception: self.position=None; self.cooldown_remaining=0

    def save_state(self):
        if not self.state_file:return
        os.makedirs(os.path.dirname(self.state_file) or ".",exist_ok=True); tmp=self.state_file+".tmp"
        with open(tmp,"w",encoding="utf-8") as f: json.dump({"position":asdict(self.position) if self.position else None,"cooldown_remaining":self.cooldown_remaining},f)
        os.replace(tmp,self.state_file)

    def _debug(self,i:Dict,result:str,reason:str)->str:
        bull=bool(i.get("trend_bull")); bear=bool(i.get("trend_bear")); d="LONG" if bull else "SHORT" if bear else "NEUTRAL"; s=self.symbol.split("/")[0]
        adx=float(i.get("adx",0)); chop=float(i.get("chop",100)); roc=float(i.get("roc9",0)); hist=float(i.get("macd_hist",0)); dist=float(i.get("distance_ema20_atr",99))
        mom=int(i.get("momentum_score_long" if d=="LONG" else "momentum_score_short",0)); loc=int(i.get("location_score_long" if d=="LONG" else "location_score_short",0))
        structure="HL" if d=="LONG" and i.get("structure_hl") else "LH" if d=="SHORT" and i.get("structure_lh") else "-"
        if reason=="COOLDOWN": return f"MOMENTUM V3.3 · {s} · ⏳ COOLDOWN {self.cooldown_remaining} bars · RESULT: WAIT"
        parts=[f"MOMENTUM V3.3 · {s} · 1H {d}"]
        if bull or bear: parts.append("✅ Trend EMA20/50")
        if reason=="TREND": parts.append("❌ No 1H trend")
        elif reason=="QUALITY": parts.append(f"❌ Quality ADX {adx:.1f} rising={'YES' if i.get('adx_rising') else 'NO'} · CHOP {chop:.1f}")
        else:
            parts.append(f"✅ Quality ADX {adx:.1f}↑ · CHOP {chop:.1f}")
            if reason=="MOMENTUM": parts.append(f"❌ Momentum 0/2 · Hist {hist:.4f} · ROC9 {roc:.2f}%")
            else:
                parts.append(f"✅ Momentum {mom}/2{' STRONG' if mom==2 else ''} · Hist {hist:.4f} · ROC9 {roc:.2f}%")
                if reason=="LOCATION": parts.append(f"❌ Location 0/2 · EMA20 {dist:.2f}ATR · Structure {structure}")
                else:
                    parts.append(f"✅ Location {loc}/2 · EMA20 {dist:.2f}ATR · Structure {structure}")
                    if reason=="TRIGGER": parts.append("❌ WAIT EMA8/13 CROSS")
                    elif result=="ENTRY": parts.append("✅ EMA8/13 CROSS")
        labels={"TREND":"WAIT TREND","QUALITY":"WAIT QUALITY","MOMENTUM":"WAIT MOMENTUM","LOCATION":"WAIT LOCATION","TRIGGER":"WAIT EMA CROSS","LONG":"ENTRY LONG","SHORT":"ENTRY SHORT","RISK":"WAIT RISK"}
        parts.append(f"RESULT: {labels.get(reason,reason)}"); return " · ".join(parts)

    def _build(self,i:Dict,d:str):
        entry=float(i["close"]); atr=max(float(i["atr"]),entry*.0005); minimum=entry*MIN_SL_PCT
        if d=="LONG": sl=min(float(i.get("recent_low",entry-atr)),entry-SL_ATR*atr,entry-minimum); risk=entry-sl; tp1=entry+TP1_R*risk; tp2=entry+TP2_R*risk
        else: sl=max(float(i.get("recent_high",entry+atr)),entry+SL_ATR*atr,entry+minimum); risk=sl-entry; tp1=entry-TP1_R*risk; tp2=entry-TP2_R*risk
        if risk<=0:return None
        size=min(self.risk_usdt/risk,(self.margin_usdt*self.leverage)/max(entry,1e-12))
        if size<=0:return None
        return {"direction":d,"strategy":"momentum_v3_3_dual_layer","trigger":"EMA8/13 cross","entry":entry,"sl":sl,"tp":tp2,"tp1":tp1,"tp2":tp2,"size":size,"risk_usdt":size*risk,"sl_pct":100*risk/max(entry,1e-12),"ema8":float(i["ema8"]),"ema13":float(i["ema13"]),"ema20":float(i["ema20"]),"ema50":float(i["ema50"]),"macd_hist":float(i["macd_hist"]),"roc9":float(i["roc9"]),"adx":float(i["adx"]),"chop":float(i["chop"]),"distance_ema20_atr":float(i["distance_ema20_atr"])}

    def _close(self,price:float,reason:str):
        p=self.position; assert p
        pnl=(price-p.entry)*p.size if p.direction=="LONG" else (p.entry-price)*p.size; initial=abs(p.entry-p.initial_sl)*max(p.initial_size,1e-12); r=pnl/initial if initial else 0
        payload={"symbol":self.symbol,"direction":p.direction,"price":price,"entry":p.entry,"sl":p.sl,"tp":p.tp2,"tp1":p.tp1,"tp2":p.tp2,"size":p.size,"strategy":p.strategy,"trigger":p.trigger,"reason":reason,"pnl":pnl,"r_multiple":r}
        if self.execution_callback:self.execution_callback("CLOSE_"+p.direction,payload)
        self.position=None
        if reason in {"EMA_CROSS_BACK","HISTOGRAM_WEAK_3"}:self.cooldown_remaining=COOLDOWN_BARS
        self.save_state(); self.last_signal=f"CLOSE {reason} pnl=${pnl:+.2f} r={r:+.2f}R"; return {"event":"CLOSE",**payload}

    def check_price(self,price:float):
        p=self.position
        if not p:return None
        if (p.direction=="LONG" and price<=p.sl) or (p.direction=="SHORT" and price>=p.sl):return self._close(price,"BE" if p.be_moved else "SL")
        if not p.tp1_hit and ((p.direction=="LONG" and price>=p.tp1) or (p.direction=="SHORT" and price<=p.tp1)):
            q=p.size*.5; pnl=(price-p.entry)*q if p.direction=="LONG" else (p.entry-price)*q; payload={"symbol":self.symbol,"direction":p.direction,"price":price,"entry":p.entry,"size":q,"reason":"TP1","pnl":pnl,"r_multiple":TP1_R}
            if self.execution_callback:self.execution_callback("CLOSE_PARTIAL",payload)
            p.size-=q; p.tp1_hit=True; p.sl=p.entry; p.be_moved=True; self.save_state(); return {"event":"PARTIAL",**payload}
        if (p.direction=="LONG" and price>=p.tp2) or (p.direction=="SHORT" and price<=p.tp2):return self._close(price,"TP2")
        return None

    def reconcile_flat(self,price:float,reason:str="EXCHANGE_CLOSED"): return self._close(price,reason) if self.position else None

    def on_bar(self,i:Dict,_i1=None,_i4=None,price:float=0.0):
        if not i:self.last_signal="WAIT INDICATOR_WARMUP"; return None
        if i.get("schema") not in SUPPORTED_SCHEMAS: raise RuntimeError(f"MOMENTUM_V33_SCHEMA_MISMATCH: {i.get('schema')}")
        if self.position:
            ev=self.check_price(price or float(i["close"]))
            if ev:return ev
            p=self.position; px=price or float(i["close"])
            if p.direction=="LONG" and i.get("ema_cross_down"):return self._close(px,"EMA_CROSS_BACK")
            if p.direction=="SHORT" and i.get("ema_cross_up"):return self._close(px,"EMA_CROSS_BACK")
            if p.direction=="LONG" and i.get("macd_hist_weaken_long_3"):return self._close(px,"HISTOGRAM_WEAK_3")
            if p.direction=="SHORT" and i.get("macd_hist_weaken_short_3"):return self._close(px,"HISTOGRAM_WEAK_3")
            self.last_signal=f"MANAGE {p.direction} | SL={p.sl:.4f} | TP1={p.tp1:.4f} | TP2={p.tp2:.4f}"; return None
        self.counts["scans"]+=1
        if self.cooldown_remaining>0:
            self.cooldown_remaining-=1; self.counts["cooldown"]+=1; self.save_state(); self.last_signal=self._debug(i,"WAIT","COOLDOWN"); return None
        bull=bool(i.get("trend_bull")); bear=bool(i.get("trend_bear"))
        if not bull and not bear:self.counts["trend"]+=1; self.last_signal=self._debug(i,"WAIT","TREND"); return None
        d="LONG" if bull else "SHORT"
        # Quality is the hard gate: both ADX and CHOP must pass.
        if float(i["adx"])<ADX_MIN or not i.get("adx_rising") or float(i["chop"])>CHOP_MAX:
            self.counts["quality"]+=1; self.last_signal=self._debug(i,"WAIT","QUALITY"); return None
        # Momentum is permissive: MACD histogram OR ROC9 is enough (1/2).
        mom=int(i.get("momentum_score_long" if d=="LONG" else "momentum_score_short",0))
        if mom<1:self.counts["momentum"]+=1; self.last_signal=self._debug(i,"WAIT","MOMENTUM"); return None
        # Location is permissive: EMA20 distance OR HL/LH structure is enough (1/2).
        loc=int(i.get("location_score_long" if d=="LONG" else "location_score_short",0))
        if loc<1:self.counts["location"]+=1; self.last_signal=self._debug(i,"WAIT","LOCATION"); return None
        # The only execution trigger is a fresh EMA8/13 cross on the current closed 15M bar.
        trigger=bool(i.get("ema_cross_up")) if d=="LONG" else bool(i.get("ema_cross_down"))
        if not trigger:self.counts["trigger"]+=1; self.last_signal=self._debug(i,"WAIT","TRIGGER"); return None
        payload=self._build(i,d)
        if not payload:self.last_signal=self._debug(i,"WAIT","RISK"); return None
        payload["symbol"]=self.symbol
        if self.execution_callback:self.execution_callback("OPEN_"+d,payload)
        self.position=Position(direction=d,entry=payload["entry"],sl=payload["sl"],initial_sl=payload["sl"],tp=payload["tp2"],tp1=payload["tp1"],tp2=payload["tp2"],size=payload["size"],initial_size=payload["size"],strategy=payload["strategy"],trigger=payload["trigger"],opened_at=time.time())
        self.counts["entries"]+=1; self.save_state(); self.last_signal=self._debug(i,"ENTRY",d); return {"event":"OPEN",**payload}
