"""Adaptive v13.2 indicators: trend, structure, EMA, MACD and price-action triggers."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
import numpy as np

ENGINE_SCHEMA = "adaptive-v13.2-price-action-macd-v1"

def _v(c: Any, name: str, idx: int) -> float:
    value = getattr(c, name, None)
    if value is None and isinstance(c, dict): value = c.get(name)
    if value is None and isinstance(c, (list, tuple)) and len(c) > idx: value = c[idx]
    return float(value or 0.0)

def _s(candles: List[Any], name: str, idx: int) -> List[float]:
    return [_v(c, name, idx) for c in candles]

def ema(values: List[float], length: int) -> List[float]:
    if not values: return []
    alpha = 2.0 / (length + 1.0); out = [float(values[0])]
    for value in values[1:]: out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out

def atr(candles: List[Any], length: int = 14) -> float:
    if len(candles) < 2: return 0.0
    h, l, c = _s(candles, "high", 2), _s(candles, "low", 3), _s(candles, "close", 4)
    tr = [h[0] - l[0]]
    for i in range(1, len(c)): tr.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    return float(np.mean(tr[-length:]))

def adx(candles: List[Any], length: int = 14) -> float:
    if len(candles) < length + 2: return 0.0
    h, l, c = _s(candles, "high", 2), _s(candles, "low", 3), _s(candles, "close", 4)
    pdm, mdm, tr = [], [], []
    for i in range(1, len(c)):
        up, down = h[i]-h[i-1], l[i-1]-l[i]
        pdm.append(up if up > down and up > 0 else 0.0); mdm.append(down if down > up and down > 0 else 0.0)
        tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    total = max(sum(tr[-length:]), 1e-12); pdi = 100*sum(pdm[-length:])/total; mdi = 100*sum(mdm[-length:])/total
    return float(100*abs(pdi-mdi)/max(pdi+mdi, 1e-12))

def choppiness(candles: List[Any], length: int = 14) -> float:
    if len(candles) < length + 1: return 100.0
    window = candles[-length:]; highs, lows = _s(window, "high", 2), _s(window, "low", 3)
    previous = _v(candles[-length-1], "close", 4); total = 0.0
    for candle in window:
        high, low = _v(candle, "high", 2), _v(candle, "low", 3)
        total += max(high-low, abs(high-previous), abs(low-previous)); previous = _v(candle, "close", 4)
    span = max(max(highs)-min(lows), 1e-12)
    return float(100*math.log10(max(total/span, 1e-12))/math.log10(length))

def _pivots(values: List[float], high: bool, left: int = 2, right: int = 2) -> List[Tuple[int,float]]:
    out=[]
    for i in range(left, len(values)-right):
        window=values[i-left:i+right+1]; value=values[i]
        if high and value==max(window) and window.count(value)==1: out.append((i,value))
        if not high and value==min(window) and window.count(value)==1: out.append((i,value))
    return out

def _age(flags: List[bool], maximum: int = 9) -> int:
    for age, flag in enumerate(reversed(flags[-maximum:])):
        if flag: return age
    return 999

def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 80: return {}
    o,h,l,c,vol = _s(candles,"open",1),_s(candles,"high",2),_s(candles,"low",3),_s(candles,"close",4),_s(candles,"volume",5)
    e8,e13,e20,e50 = (ema(c,n) for n in (8,13,20,50)); a=max(atr(candles),c[-1]*0.0005)
    macd_fast, macd_slow = ema(c,12), ema(c,26)
    macd_series = [fast-slow for fast,slow in zip(macd_fast,macd_slow)]
    macd_signal_series = ema(macd_series,9)
    macd_value = macd_series[-1]; macd_signal = macd_signal_series[-1]; macd_hist = macd_value-macd_signal
    ph,pl=_pivots(h[-60:],True),_pivots(l[-60:],False); hs=[v for _,v in ph[-2:]]; ls=[v for _,v in pl[-2:]]
    last_high=hs[-1] if hs else max(h[-12:-1]); prev_high=hs[-2] if len(hs)>1 else max(h[-24:-12])
    last_low=ls[-1] if ls else min(l[-12:-1]); prev_low=ls[-2] if len(ls)>1 else min(l[-24:-12])
    hh,hl,lh,ll=last_high>prev_high,last_low>prev_low,last_high<prev_high,last_low<prev_low
    structure="BULL" if hh and hl else "BEAR" if lh and ll else "MIXED"
    up_flags=[]; down_flags=[]
    for i in range(max(1,len(c)-10),len(c)):
        up_flags.append(e8[i-1]<=e13[i-1] and e8[i]>e13[i]); down_flags.append(e8[i-1]>=e13[i-1] and e8[i]<e13[i])
    bull_engulf=c[-1]>o[-1] and c[-2]<o[-2] and c[-1]>=o[-2] and o[-1]<=c[-2]
    bear_engulf=c[-1]<o[-1] and c[-2]>o[-2] and c[-1]<=o[-2] and o[-1]>=c[-2]
    body=abs(c[-1]-o[-1]); lower=min(o[-1],c[-1])-l[-1]; upper=h[-1]-max(o[-1],c[-1])
    hammer=c[-1]>o[-1] and lower>=max(body*1.5,a*0.15); shooting=c[-1]<o[-1] and upper>=max(body*1.5,a*0.15)
    strong_bull=c[-1]>o[-1] and body>=0.45*a and h[-1]-c[-1]<=0.20*a
    strong_bear=c[-1]<o[-1] and body>=0.45*a and c[-1]-l[-1]<=0.20*a
    inside=h[-2]<h[-3] and l[-2]>l[-3]; inside_up=inside and c[-1]>h[-2]; inside_down=inside and c[-1]<l[-2]
    break_up=c[-1]>h[-2] and c[-2]>o[-2]; break_down=c[-1]<l[-2] and c[-2]<o[-2]
    long_trigger=bull_engulf or hammer or strong_bull or inside_up or break_up
    short_trigger=bear_engulf or shooting or strong_bear or inside_down or break_down
    long_name="bull_engulf" if bull_engulf else "hammer" if hammer else "inside_break" if inside_up else "break_high" if break_up else "strong_bull" if strong_bull else "none"
    short_name="bear_engulf" if bear_engulf else "shooting_star" if shooting else "inside_break" if inside_down else "break_low" if break_down else "strong_bear" if strong_bear else "none"
    lp=[]; sp=[]
    for i in range(len(c)-6,len(c)):
        lp.append(l[i]<=e20[i]+0.35*a and c[i]>=e13[i]); sp.append(h[i]>=e20[i]-0.35*a and c[i]<=e13[i])
    return {
        "schema":ENGINE_SCHEMA,"open":o[-1],"high":h[-1],"low":l[-1],"close":c[-1],"prev_open":o[-2],"prev_high":h[-2],"prev_low":l[-2],"prev_close":c[-2],
        "ema8":e8[-1],"ema13":e13[-1],"ema20":e20[-1],"ema50":e50[-1],"ema8_series":e8[-80:],"ema13_series":e13[-80:],"ema20_series":e20[-80:],
        "ema20_slope_atr":(e20[-1]-e20[-4])/a,"macd":macd_value,"macd_signal":macd_signal,"macd_hist":macd_hist,
        "macd_bull":macd_value>macd_signal and macd_hist>0,"macd_bear":macd_value<macd_signal and macd_hist<0,
        "cross_up":up_flags[-1],"cross_down":down_flags[-1],"cross_up_age":_age(up_flags),"cross_down_age":_age(down_flags),
        "atr":a,"adx":adx(candles),"chop":choppiness(candles),"body_atr":body/a,"extension_atr":abs(c[-1]-e20[-1])/a,"volume":vol[-1],"vol_avg":float(np.mean(vol[-20:])),
        "last_swing_high":last_high,"previous_swing_high":prev_high,"last_swing_low":last_low,"previous_swing_low":prev_low,"higher_high":hh,"higher_low":hl,"lower_high":lh,"lower_low":ll,"structure":structure,
        "long_pullback_age":_age(lp),"short_pullback_age":_age(sp),"long_trigger":long_trigger,"short_trigger":short_trigger,"long_trigger_name":long_name,"short_trigger_name":short_name,
        "ema_bull":e8[-1]>e13[-1]>e20[-1],"ema_bear":e8[-1]<e13[-1]<e20[-1],"close_below_ema13_2":c[-1]<e13[-1] and c[-2]<e13[-2],"close_above_ema13_2":c[-1]>e13[-1] and c[-2]>e13[-2]
    }

class IndicatorEngine:
    @staticmethod
    def _candle(candles: List[Any]) -> Dict[str, Any]:
        if not candles: return {}
        c=candles[-1]; return {"open":_v(c,"open",1),"high":_v(c,"high",2),"low":_v(c,"low",3),"close":_v(c,"close",4),"volume":_v(c,"volume",5)}
    def compute(self,c15m,c1h,c4h):
        return self._candle(c15m),self._candle(c1h),self._candle(c4h),compute(c15m),compute(c1h),compute(c4h)
