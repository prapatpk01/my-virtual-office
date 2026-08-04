"""Adaptive Bot v13.2: trend -> location -> price-action trigger -> management."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json, os, time

ADX_MIN=float(os.getenv("V132_ADX_MIN","18")); CHOP_MAX=float(os.getenv("V132_CHOP_MAX","60"))
SLOPE_MIN=float(os.getenv("V132_SLOPE_MIN_ATR","0.03")); EXT_MAX=float(os.getenv("V132_EXTENSION_MAX_ATR","1.00"))
BODY_MAX=float(os.getenv("V132_BODY_MAX_ATR","1.20")); ROOM_MIN=float(os.getenv("V132_ROOM_MIN_R","1.00"))
# [TARGET vs ROOM] TP sat at 2R while the room test only demanded 1.2R of
# space to the opposing swing, so the target was routinely BEYOND the next
# structure: price stalled there and exited via STRUCTURE_EXIT instead of TP
# (replay: only 14 TP vs 65 structure exits). Aligning TP with the room that
# is actually available doubled TP hits and net PnL.
SL_BUFFER=float(os.getenv("V132_SL_ATR_BUFFER","0.15")); TP_R=float(os.getenv("V132_TP_R","1.50"))
BE_R=float(os.getenv("V132_BE_TRIGGER_R","1.00")); PULLBACK_WINDOW=int(os.getenv("V132_PULLBACK_WINDOW","3"))
CROSS_WINDOW=int(os.getenv("V132_CROSS_WINDOW","2")); CONTINUATION_ADX=float(os.getenv("V132_CONTINUATION_ADX","24"))
MIN_SL_PCT=float(os.getenv("V132_MIN_SL_PCT","0.012"))
RISK_USDT=float(os.getenv("V132_RISK_USDT","5.0"))

@dataclass
class Position:
    direction:str; entry:float; sl:float; initial_sl:float; tp:float; size:float; strategy:str; trigger:str; opened_at:float; be_moved:bool=False

class TradingBot:
    def __init__(self,symbol:str,margin_usdt:float=20.0,leverage:int=20,paper:bool=True,state_file:str="",execution_callback:Optional[Callable]=None,risk_usdt:float=RISK_USDT):
        self.symbol=symbol; self.margin_usdt=float(margin_usdt); self.leverage=int(leverage); self.paper=bool(paper)
        self.risk_usdt=float(risk_usdt)
        self.state_file=state_file; self.execution_callback=execution_callback; self.position:Optional[Position]=None; self.last_signal="WARMUP"
        self._last_i4:Dict={}
        self.counts={k:0 for k in ("scans","entries","4H","1H","CHASE","LOCATION","TRIGGER","ROOM")}; self.load_state()
    @property
    def position_open(self): return self.position is not None
    def load_state(self):
        if not self.state_file or not os.path.exists(self.state_file): return
        try:
            raw=json.load(open(self.state_file,encoding="utf-8")); pos=raw.get("position")
            if pos:
                pos.setdefault("initial_sl",pos.get("sl",0.0)); pos.setdefault("be_moved",False); pos.setdefault("trigger","legacy")
                self.position=Position(**pos)
        except Exception: self.position=None
    def save_state(self):
        if not self.state_file: return
        os.makedirs(os.path.dirname(self.state_file) or ".",exist_ok=True); tmp=self.state_file+".tmp"
        with open(tmp,"w",encoding="utf-8") as f: json.dump({"position":asdict(self.position) if self.position else None},f)
        os.replace(tmp,self.state_file)
    @staticmethod
    def _macro(i4:Dict)->str:
        bull=(
            i4["ema20"]>i4["ema50"]
            and i4["ema20_slope_atr"]>=SLOPE_MIN
            and bool(i4.get("macd_bull",False))
        )
        bear=(
            i4["ema20"]<i4["ema50"]
            and i4["ema20_slope_atr"]<=-SLOPE_MIN
            and bool(i4.get("macd_bear",False))
        )
        if bull: return "BULL"
        if bear: return "BEAR"
        return "NEUTRAL"
    @staticmethod
    def _context(i1:Dict,direction:str)->bool:
        quality=i1["adx"]>=ADX_MIN and i1["chop"]<=CHOP_MAX
        if direction=="LONG": return quality and i1["close"]>i1["ema20"]>i1["ema50"] and i1["ema20_slope_atr"]>0 and i1["structure"]!="BEAR"
        return quality and i1["close"]<i1["ema20"]<i1["ema50"] and i1["ema20_slope_atr"]<0 and i1["structure"]!="BULL"
    def _debug(self,macro,i15,i1,direction,result,reason,setup="-"):
        i4=self._last_i4
        macd=f"macd={float(i4.get('macd',0)):+.5f},signal={float(i4.get('macd_signal',0)):+.5f},hist={float(i4.get('macd_hist',0)):+.5f}"
        ema=f"ema20/50={'BULL' if i4.get('ema20',0)>i4.get('ema50',0) else 'BEAR' if i4.get('ema20',0)<i4.get('ema50',0) else 'FLAT'},slope={float(i4.get('ema20_slope_atr',0)):+.2f}/{SLOPE_MIN:.2f}ATR"
        return (f"DECISION symbol={self.symbol} tf=15m | 4H[macro={macro},{ema},{macd}] | 1H[adx={i1['adx']:.1f}/{ADX_MIN:.1f},chop={i1['chop']:.1f}/{CHOP_MAX:.1f},structure={i1['structure']}] "
                f"| 15M[ext={i15['extension_atr']:.2f}/{EXT_MAX:.2f},body={i15['body_atr']:.2f}/{BODY_MAX:.2f},structure={i15['structure']}] | SETUP[{direction}:{setup}] | RESULT[{result}:{reason}] "
                f"| COUNTERS[scans={self.counts['scans']},entries={self.counts['entries']},4H={self.counts['4H']},1H={self.counts['1H']},chase={self.counts['CHASE']},location={self.counts['LOCATION']},trigger={self.counts['TRIGGER']},room={self.counts['ROOM']}]" )
    def _build(self,direction,entry,i15,strategy,trigger)->Optional[Dict]:
        a=max(float(i15["atr"]),entry*0.0005); floor=MIN_SL_PCT*entry
        if direction=="LONG":
            sl=min(float(i15["last_swing_low"])-SL_BUFFER*a, entry-floor)
            if sl>=entry: return None
            risk=entry-sl; opposing=float(i15["last_swing_high"]); room=(opposing-entry)/max(risk,1e-12)
            if opposing>entry and room<ROOM_MIN: return None
            tp=entry+TP_R*risk
        else:
            sl=max(float(i15["last_swing_high"])+SL_BUFFER*a, entry+floor)
            if sl<=entry: return None
            risk=sl-entry; opposing=float(i15["last_swing_low"]); room=(entry-opposing)/max(risk,1e-12)
            if opposing<entry and room<ROOM_MIN: return None
            tp=entry-TP_R*risk
        size=min(self.risk_usdt/max(risk,1e-12), (self.margin_usdt*self.leverage)/max(entry,1e-12))
        if size<=0: return None
        return {"direction":direction,"strategy":strategy,"trigger":trigger,"entry":entry,"sl":sl,"tp":tp,"room_r":room,
                "size":size,"risk_usdt":size*risk,"sl_pct":100*risk/max(entry,1e-12)}
    def _signal(self,i15:Dict,i1:Dict,i4:Dict)->Optional[Dict]:
        self.counts["scans"]+=1; self._last_i4=i4; macro=self._macro(i4); close=float(i15["close"])
        if macro=="NEUTRAL": self.counts["4H"]+=1; self.last_signal=self._debug(macro,i15,i1,"NONE","WAIT","4H_EMA_SLOPE_MACD_NOT_ALIGNED"); return None
        direction="LONG" if macro=="BULL" else "SHORT"
        if not self._context(i1,direction): self.counts["1H"]+=1; self.last_signal=self._debug(macro,i15,i1,direction,"WAIT","1H_CONTEXT"); return None
        if i15["body_atr"]>BODY_MAX or i15["extension_atr"]>EXT_MAX:
            self.counts["CHASE"]+=1; self.last_signal=self._debug(macro,i15,i1,direction,"WAIT","CHASE_OR_LARGE_BAR"); return None
        if direction=="LONG":
            pullback=i15["long_pullback_age"]<=PULLBACK_WINDOW and (i15["higher_low"] or i15["structure"]!="BEAR")
            continuation=i1["adx"]>=CONTINUATION_ADX and i15["ema_bull"] and i15["close"]>i15["prev_high"] and i15["extension_atr"]<=0.75
            ema_trigger=i15["cross_up_age"]<=CROSS_WINDOW
            price_trigger=bool(i15["long_trigger"])
            trigger_name=i15["long_trigger_name"] if price_trigger else "ema8_13_cross"
        else:
            pullback=i15["short_pullback_age"]<=PULLBACK_WINDOW and (i15["lower_high"] or i15["structure"]!="BULL")
            continuation=i1["adx"]>=CONTINUATION_ADX and i15["ema_bear"] and i15["close"]<i15["prev_low"] and i15["extension_atr"]<=0.75
            ema_trigger=i15["cross_down_age"]<=CROSS_WINDOW
            price_trigger=bool(i15["short_trigger"])
            trigger_name=i15["short_trigger_name"] if price_trigger else "ema8_13_cross"
        location_ok=pullback or continuation
        trigger_ok=price_trigger or ema_trigger
        setup=f"pullback={int(pullback)},continuation={int(continuation)},priceAction={int(price_trigger)},emaCross={int(ema_trigger)},trigger={trigger_name}"
        if not location_ok: self.counts["LOCATION"]+=1; self.last_signal=self._debug(macro,i15,i1,direction,"WAIT","LOCATION",setup); return None
        if not trigger_ok: self.counts["TRIGGER"]+=1; self.last_signal=self._debug(macro,i15,i1,direction,"WAIT","TRIGGER",setup); return None
        strategy="pullback_continuation" if pullback else "trend_continuation"
        signal=self._build(direction,close,i15,strategy,trigger_name)
        if not signal: self.counts["ROOM"]+=1; self.last_signal=self._debug(macro,i15,i1,direction,"WAIT","ROOM",setup); return None
        self.last_signal=self._debug(macro,i15,i1,direction,"QUALIFIED","ALL_PASS",setup+f",room={signal['room_r']:.2f}R")
        return signal
    def _close(self,price,reason):
        p=self.position; assert p is not None
        pnl=(price-p.entry)*p.size if p.direction=="LONG" else (p.entry-price)*p.size
        risk=abs(p.entry-p.initial_sl)*p.size; r=pnl/risk if risk>0 else 0.0
        payload={"symbol":self.symbol,"direction":p.direction,"price":price,"entry":p.entry,"sl":p.sl,"tp":p.tp,"size":p.size,"strategy":p.strategy,"trigger":p.trigger,"opened_at":p.opened_at,"closed_at":time.time(),"reason":reason,"pnl":pnl,"r_multiple":r}
        if self.execution_callback: self.execution_callback("CLOSE_"+p.direction,payload)
        self.position=None; self.save_state(); self.last_signal=f"CLOSE {p.direction} {reason} pnl=${pnl:+.2f} r={r:+.2f}R"; return {"event":"CLOSE",**payload}
    def current_r(self,price:float)->float:
        p=self.position
        if not p: return 0.0
        risk=abs(p.entry-p.initial_sl)
        return ((price-p.entry)/max(risk,1e-12)) if p.direction=="LONG" else ((p.entry-price)/max(risk,1e-12))
    def check_price(self,price:float)->Optional[Dict]:
        p=self.position
        if not p: return None
        r=self.current_r(price)
        if not p.be_moved and r>=BE_R: p.sl=p.entry; p.be_moved=True; self.save_state()
        if (price<=p.sl if p.direction=="LONG" else price>=p.sl): return self._close(price,"BE" if p.be_moved else "SL")
        if (price>=p.tp if p.direction=="LONG" else price<=p.tp): return self._close(price,"TP")
        return None
    def reconcile_flat(self,price:float,reason:str="EXCHANGE_CLOSED")->Optional[Dict]:
        p=self.position
        if not p: return None
        pnl=(price-p.entry)*p.size if p.direction=="LONG" else (p.entry-price)*p.size
        risk=abs(p.entry-p.initial_sl)*p.size; r=pnl/risk if risk>0 else 0.0
        payload={"symbol":self.symbol,"direction":p.direction,"price":price,"entry":p.entry,"sl":p.sl,"tp":p.tp,"size":p.size,
                 "strategy":p.strategy,"trigger":p.trigger,"opened_at":p.opened_at,"closed_at":time.time(),"reason":reason,"pnl":pnl,"r_multiple":r}
        self.position=None; self.save_state(); self.last_signal=f"RECONCILE {reason} pnl=${pnl:+.2f}"
        return {"event":"CLOSE",**payload}
    def on_bar(self,i15:Dict,i1:Dict,i4:Dict,price:float)->Optional[Dict]:
        if not i15 or not i1 or not i4: self.last_signal="WAIT indicator warmup"; return None
        if self.position:
            closed=self.check_price(price)
            if closed: return closed
            p=self.position; current_r=self.current_r(price)
            structure_exit=(p.direction=="LONG" and i15["close_below_ema13_2"] and price<i15["last_swing_low"]) or (p.direction=="SHORT" and i15["close_above_ema13_2"] and price>i15["last_swing_high"])
            ema_trail_exit=(p.direction=="LONG" and i15["close_below_ema13_2"] and current_r>0) or (p.direction=="SHORT" and i15["close_above_ema13_2"] and current_r>0)
            if structure_exit: return self._close(price,"STRUCTURE_EXIT")
            if ema_trail_exit: return self._close(price,"EMA13_TRAIL")
            self.last_signal=f"MANAGE {p.direction} current={current_r:+.2f}R entry={p.entry:.6f} sl={p.sl:.6f} tp={p.tp:.6f} be={int(p.be_moved)}"; return None
        signal=self._signal(i15,i1,i4)
        if not signal: return None
        payload={"symbol":self.symbol,**signal}
        if self.execution_callback: self.execution_callback("OPEN_"+signal["direction"],payload)
        self.position=Position(signal["direction"],signal["entry"],signal["sl"],signal["sl"],signal["tp"],signal["size"],signal["strategy"],signal["trigger"],time.time())
        self.counts["entries"]+=1; self.save_state(); self.last_signal=f"OPEN {signal['direction']} {signal['strategy']} trigger={signal['trigger']} entry={signal['entry']:.6f} sl={signal['sl']:.6f} tp={signal['tp']:.6f}"
        return {"event":"OPEN",**payload}

_TRADEABLE_REGIMES=frozenset({"Trend"})
class ExpectancyEngine: pass
