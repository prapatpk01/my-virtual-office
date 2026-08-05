"""Adaptive SMC v14: 4H bias -> sweep -> CHoCH/BOS -> OB/FVG -> trigger."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json, os, time

TP_R=float(os.getenv("V14_TP2_R","2.0")); TP1_R=float(os.getenv("V14_TP1_R","1.0"))
MIN_ROOM_R=float(os.getenv("V14_MIN_ROOM_R","1.5")); SL_BUFFER_ATR=float(os.getenv("V14_SL_BUFFER_ATR","0.15"))
MIN_SL_PCT=float(os.getenv("V14_MIN_SL_PCT","0.006")); RISK_USDT=float(os.getenv("V14_RISK_USDT","5.0"))
SETUP_EXPIRY_BARS=int(os.getenv("V14_SETUP_EXPIRY_BARS","12")); MIN_SCORE=int(os.getenv("V14_MIN_SCORE","60")); FULL_RISK_SCORE=int(os.getenv("V14_FULL_RISK_SCORE","75"))

@dataclass
class SetupState:
    direction:str; phase:str; sweep_level:float; choch_level:float=0.0; bos_level:float=0.0
    zone_low:float=0.0; zone_high:float=0.0; zone_type:str="NONE"; age:int=0; score:int=0

@dataclass
class Position:
    direction:str; entry:float; sl:float; initial_sl:float; tp:float; tp1:float; tp2:float
    size:float; initial_size:float; strategy:str; trigger:str; opened_at:float; quality:int
    sweep_level:float; choch_level:float; bos_level:float; zone_low:float; zone_high:float; zone_type:str
    tp1_hit:bool=False; be_moved:bool=False

class TradingBot:
    def __init__(self,symbol:str,margin_usdt:float=20.0,leverage:int=20,paper:bool=True,state_file:str="",execution_callback:Optional[Callable]=None,risk_usdt:float=RISK_USDT):
        self.symbol=symbol; self.margin_usdt=float(margin_usdt); self.leverage=int(leverage); self.paper=bool(paper)
        self.state_file=state_file; self.execution_callback=execution_callback; self.risk_usdt=float(risk_usdt)
        self.position:Optional[Position]=None; self.setup:Optional[SetupState]=None; self.last_signal="WARMUP"
        self.counts={k:0 for k in ("scans","entries","bias","sweep","structure","zone","trigger","room","expired")}; self.load_state()
    @property
    def position_open(self): return self.position is not None
    def load_state(self):
        if not self.state_file or not os.path.exists(self.state_file): return
        try:
            raw=json.load(open(self.state_file,encoding="utf-8")); p=raw.get("position"); s=raw.get("setup")
            if p:
                defaults={"tp1":p.get("entry",0),"tp2":p.get("tp",0),"initial_size":p.get("size",0),"quality":0,"sweep_level":p.get("sl",0),"choch_level":0,"bos_level":0,"zone_low":0,"zone_high":0,"zone_type":"LEGACY","tp1_hit":False,"be_moved":False}
                for k,v in defaults.items(): p.setdefault(k,v)
                self.position=Position(**p)
            if s:self.setup=SetupState(**s)
        except Exception:self.position=None;self.setup=None
    def save_state(self):
        if not self.state_file:return
        os.makedirs(os.path.dirname(self.state_file) or ".",exist_ok=True); tmp=self.state_file+".tmp"
        with open(tmp,"w",encoding="utf-8") as f:json.dump({"position":asdict(self.position) if self.position else None,"setup":asdict(self.setup) if self.setup else None},f)
        os.replace(tmp,self.state_file)
    @staticmethod
    def _macro(i4:Dict)->str:
        if i4["ema20"]>i4["ema50"] and i4["ema20_slope_atr"]>0:return "BULL"
        if i4["ema20"]<i4["ema50"] and i4["ema20_slope_atr"]<0:return "BEAR"
        return "NEUTRAL"
    def _debug(self,macro,i1,i15,result,reason):
        s=self.setup; state="NONE" if not s else f"{s.direction}:{s.phase},age={s.age}/{SETUP_EXPIRY_BARS},zone={s.zone_type}[{s.zone_low:.6f}-{s.zone_high:.6f}],score={s.score}"
        return (f"SMC_DECISION symbol={self.symbol} | 4H[bias={macro}] | 1H[sellSweep={int(i1.get('recent_sell_sweep',False))},buySweep={int(i1.get('recent_buy_sweep',False))},bullCHoCH={int(i1.get('bullish_choch',False))},bearCHoCH={int(i1.get('bearish_choch',False))},bullBOS={int(i1.get('bullish_bos',False))},bearBOS={int(i1.get('bearish_bos',False))}] | 15M[OBbull={int(i15.get('ob_bull',False))},OBbear={int(i15.get('ob_bear',False))},FVGbull={int(i15.get('fvg_bull',False))},FVGbear={int(i15.get('fvg_bear',False))}] | STATE[{state}] | RESULT[{result}:{reason}] | COUNTERS[{','.join(f'{k}={v}' for k,v in self.counts.items())}]" )
    def _reset(self,reason): self.setup=None;self.last_signal=f"SETUP_RESET {reason}";self.save_state()
    def _advance(self,macro,i1,i15):
        if self.setup:
            self.setup.age+=1
            if self.setup.age>SETUP_EXPIRY_BARS:self.counts["expired"]+=1;self._reset("EXPIRED");return
            if (self.setup.direction=="LONG" and i1.get("bearish_choch")) or (self.setup.direction=="SHORT" and i1.get("bullish_choch")):self._reset("OPPOSITE_CHOCH");return
        if not self.setup:
            if macro=="BULL" and i1.get("recent_sell_sweep"):self.setup=SetupState("LONG","SWEEP_DETECTED",float(i1["last_swing_low"]),score=40)
            elif macro=="BEAR" and i1.get("recent_buy_sweep"):self.setup=SetupState("SHORT","SWEEP_DETECTED",float(i1["last_swing_high"]),score=40)
            else:self.counts["sweep"]+=1;return
        s=self.setup
        if s.direction=="LONG":
            if s.phase=="SWEEP_DETECTED" and i1.get("bullish_choch"):s.phase="CHOCH_CONFIRMED";s.choch_level=float(i1["last_swing_high"]);s.score=55
            if s.phase in {"SWEEP_DETECTED","CHOCH_CONFIRMED"} and i1.get("bullish_bos"):s.phase="BOS_CONFIRMED";s.bos_level=float(i1["last_swing_high"]);s.score=max(s.score,70)
            if s.phase in {"CHOCH_CONFIRMED","BOS_CONFIRMED"} and (i15.get("ob_bull") or i15.get("fvg_bull")):
                s.zone_low=float(i15.get("bull_zone_low",0));s.zone_high=float(i15.get("bull_zone_high",0));s.zone_type="OB+FVG" if i15.get("bull_zone_overlap") else "OB" if i15.get("ob_bull") else "FVG";s.phase="WAIT_RETRACE";s.score=max(s.score,80 if s.zone_type=="OB+FVG" else 70)
        else:
            if s.phase=="SWEEP_DETECTED" and i1.get("bearish_choch"):s.phase="CHOCH_CONFIRMED";s.choch_level=float(i1["last_swing_low"]);s.score=55
            if s.phase in {"SWEEP_DETECTED","CHOCH_CONFIRMED"} and i1.get("bearish_bos"):s.phase="BOS_CONFIRMED";s.bos_level=float(i1["last_swing_low"]);s.score=max(s.score,70)
            if s.phase in {"CHOCH_CONFIRMED","BOS_CONFIRMED"} and (i15.get("ob_bear") or i15.get("fvg_bear")):
                s.zone_low=float(i15.get("bear_zone_low",0));s.zone_high=float(i15.get("bear_zone_high",0));s.zone_type="OB+FVG" if i15.get("bear_zone_overlap") else "OB" if i15.get("ob_bear") else "FVG";s.phase="WAIT_RETRACE";s.score=max(s.score,80 if s.zone_type=="OB+FVG" else 70)
        self.save_state()
    @staticmethod
    def _in_zone(i15,s):return s.zone_low>0 and s.zone_high>s.zone_low and float(i15["low"])<=s.zone_high and float(i15["high"])>=s.zone_low
    @staticmethod
    def _trigger(i15,direction):
        if direction=="LONG":
            if i15.get("bull_engulf"):return True,"bullish_engulfing",10
            if i15.get("bull_pin"):return True,"bullish_pinbar",10
            if i15.get("break_high"):return True,"break_reversal_high",10
            if i15.get("bull_volume"):return True,"positive_volume_proxy",5
        else:
            if i15.get("bear_engulf"):return True,"bearish_engulfing",10
            if i15.get("bear_pin"):return True,"bearish_pinbar",10
            if i15.get("break_low"):return True,"break_reversal_low",10
            if i15.get("bear_volume"):return True,"negative_volume_proxy",5
        return False,"none",0
    def _build(self,i15,trigger,trigger_score):
        s=self.setup
        if not s:return None
        entry=float(i15["close"]);a=max(float(i15["atr"]),entry*0.0005);floor=entry*MIN_SL_PCT
        if s.direction=="LONG":
            invalid=min(s.sweep_level,s.zone_low if s.zone_low>0 else s.sweep_level);sl=min(invalid-SL_BUFFER_ATR*a,entry-floor)
            if sl>=entry:return None
            risk=entry-sl;target=float(i15["last_swing_high"]);room=(target-entry)/max(risk,1e-12) if target>entry else TP_R;tp1=entry+TP1_R*risk;tp2=entry+TP_R*risk
        else:
            invalid=max(s.sweep_level,s.zone_high if s.zone_high>0 else s.sweep_level);sl=max(invalid+SL_BUFFER_ATR*a,entry+floor)
            if sl<=entry:return None
            risk=sl-entry;target=float(i15["last_swing_low"]);room=(entry-target)/max(risk,1e-12) if target<entry else TP_R;tp1=entry-TP1_R*risk;tp2=entry-TP_R*risk
        if room<MIN_ROOM_R:return None
        score=min(100,s.score+trigger_score+(5 if i15.get("volume_ratio",0)>=1.2 else 0))
        if score<MIN_SCORE:return None
        budget=self.risk_usdt*(0.5 if score<FULL_RISK_SCORE else 1.0);size=min(budget/max(risk,1e-12),(self.margin_usdt*self.leverage)/max(entry,1e-12))
        if size<=0:return None
        return {"direction":s.direction,"strategy":"smc_liquidity_retrace","trigger":trigger,"entry":entry,"sl":sl,"tp":tp2,"tp1":tp1,"tp2":tp2,"room_r":room,"size":size,"risk_usdt":size*risk,"sl_pct":100*risk/max(entry,1e-12),"quality":score,"sweep_level":s.sweep_level,"choch_level":s.choch_level,"bos_level":s.bos_level,"zone_low":s.zone_low,"zone_high":s.zone_high,"zone_type":s.zone_type}
    def _close(self,price,reason):
        p=self.position;assert p
        pnl=(price-p.entry)*p.size if p.direction=="LONG" else (p.entry-price)*p.size;risk=abs(p.entry-p.initial_sl)*max(p.initial_size,1e-12);r=pnl/risk if risk>0 else 0
        payload={"symbol":self.symbol,"direction":p.direction,"price":price,"entry":p.entry,"sl":p.sl,"tp":p.tp2,"tp1":p.tp1,"tp2":p.tp2,"size":p.size,"strategy":p.strategy,"trigger":p.trigger,"opened_at":p.opened_at,"closed_at":time.time(),"reason":reason,"pnl":pnl,"r_multiple":r,"quality":p.quality}
        if self.execution_callback:self.execution_callback("CLOSE_"+p.direction,payload)
        self.position=None;self.save_state();self.last_signal=f"CLOSE {reason} pnl=${pnl:+.2f} r={r:+.2f}R";return {"event":"CLOSE",**payload}
    def current_r(self,price):
        p=self.position
        if not p:return 0.0
        risk=abs(p.entry-p.initial_sl);return (price-p.entry)/max(risk,1e-12) if p.direction=="LONG" else (p.entry-price)/max(risk,1e-12)
    def check_price(self,price):
        p=self.position
        if not p:return None
        if (price<=p.sl if p.direction=="LONG" else price>=p.sl):return self._close(price,"BE" if p.be_moved else "SL")
        if not p.tp1_hit and (price>=p.tp1 if p.direction=="LONG" else price<=p.tp1):
            close_size=p.size*0.5;pnl=(price-p.entry)*close_size if p.direction=="LONG" else (p.entry-price)*close_size;payload={"symbol":self.symbol,"direction":p.direction,"price":price,"entry":p.entry,"size":close_size,"reason":"TP1","pnl":pnl,"r_multiple":TP1_R}
            if self.execution_callback:self.execution_callback("CLOSE_PARTIAL",payload)
            p.size-=close_size;p.tp1_hit=True;p.sl=p.entry;p.be_moved=True;self.save_state();self.last_signal=f"TP1 HIT close=50% pnl=${pnl:+.2f} SL->BE";return {"event":"PARTIAL",**payload}
        if (price>=p.tp2 if p.direction=="LONG" else price<=p.tp2):return self._close(price,"TP2")
        return None
    def reconcile_flat(self,price,reason="EXCHANGE_CLOSED"):
        return self._close(price,reason) if self.position else None
    def on_bar(self,i15,i1,i4,price):
        if not i15 or not i1 or not i4:self.last_signal="WAIT INDICATOR_WARMUP";return None
        if i15.get("schema")!="adaptive-smc-v14-structure-v1":raise RuntimeError(f"V14_SCHEMA_MISMATCH: {i15.get('schema')}")
        if self.position:
            event=self.check_price(price)
            if event:return event
            p=self.position;invalid=(p.direction=="LONG" and (i15.get("bearish_choch") or float(i15["close"])<p.zone_low)) or (p.direction=="SHORT" and (i15.get("bullish_choch") or float(i15["close"])>p.zone_high))
            if invalid:return self._close(price,"STRUCTURE_INVALIDATION")
            self.last_signal=f"MANAGE {p.direction} {self.current_r(price):+.2f}R TP1={int(p.tp1_hit)} SL={p.sl:.6f} TP2={p.tp2:.6f}";return None
        self.counts["scans"]+=1;macro=self._macro(i4)
        if macro=="NEUTRAL":self.counts["bias"]+=1;self.last_signal=self._debug(macro,i1,i15,"WAIT","4H_NEUTRAL");return None
        self._advance(macro,i1,i15);s=self.setup
        if not s:self.last_signal=self._debug(macro,i1,i15,"WAIT","LIQUIDITY_SWEEP");return None
        if s.direction!=("LONG" if macro=="BULL" else "SHORT"):self._reset("4H_BIAS_CHANGED");return None
        if s.phase not in {"WAIT_RETRACE","WAIT_TRIGGER"}:self.counts["structure"]+=1;self.last_signal=self._debug(macro,i1,i15,"WAIT",s.phase);return None
        if not self._in_zone(i15,s):self.counts["zone"]+=1;self.last_signal=self._debug(macro,i1,i15,"WAIT","RETRACE_TO_OB_FVG");return None
        s.phase="WAIT_TRIGGER";passed,trigger,score=self._trigger(i15,s.direction)
        if not passed:self.counts["trigger"]+=1;self.last_signal=self._debug(macro,i1,i15,"WAIT","PRICE_ACTION_TRIGGER");self.save_state();return None
        signal=self._build(i15,trigger,score)
        if not signal:self.counts["room"]+=1;self.last_signal=self._debug(macro,i1,i15,"WAIT","SCORE_OR_ROOM");return None
        payload={"symbol":self.symbol,**signal}
        if self.execution_callback:self.execution_callback("OPEN_"+signal["direction"],payload)
        self.position=Position(signal["direction"],signal["entry"],signal["sl"],signal["sl"],signal["tp2"],signal["tp1"],signal["tp2"],signal["size"],signal["size"],signal["strategy"],signal["trigger"],time.time(),signal["quality"],signal["sweep_level"],signal["choch_level"],signal["bos_level"],signal["zone_low"],signal["zone_high"],signal["zone_type"])
        self.setup=None;self.counts["entries"]+=1;self.save_state();self.last_signal=f"OPEN {signal['direction']} SMC score={signal['quality']} trigger={signal['trigger']}";return {"event":"OPEN",**payload}

_TRADEABLE_REGIMES=frozenset({"Trend"})
class ExpectancyEngine:pass
