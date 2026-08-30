"""Adaptive SMC MTF V7.1 indicator engine.

Closed-candle pipeline:
  4H  TSS-style direction (EMA20/50 + HMA16 slope)
  15M Market Structure (HH/HL, LH/LL, BOS/CHOCH)
  5M  AMD liquidity setup
  1M  IFVG location -> micro BOS/CHOCH confirmation -> execution

V7.1 fixes the main V7 weakness: an IFVG retest is a location, not proof that
short-term order flow has actually turned.  Entry therefore requires a fresh
M1 break of micro structure after the IFVG retest.  Stops are built from the
M1 confirmation swing plus the M5 manipulation extreme and an ATR buffer.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math

ENGINE_SCHEMA = "adaptive-smc-mtf-v1"


def _v(c: Any, name: str, idx: int) -> float:
    v = getattr(c, name, None)
    if v is None and isinstance(c, dict): v = c.get(name)
    if v is None and isinstance(c, (list, tuple)) and len(c) > idx: v = c[idx]
    return float(v or 0.0)


def _series(c, name, idx): return [_v(x, name, idx) for x in c]


def ema(values: List[float], length: int) -> List[float]:
    if not values: return []
    a = 2.0 / (length + 1.0); out = [float(values[0])]
    for v in values[1:]: out.append(a * float(v) + (1-a) * out[-1])
    return out


def _wma(values, length):
    if not values: return []
    length=max(1,int(length)); out=[float(values[0])]*len(values); ws=list(range(1,length+1)); d=float(sum(ws))
    for i in range(length-1,len(values)):
        out[i]=sum(float(v)*w for v,w in zip(values[i-length+1:i+1],ws))/d
    if length>1:
        seed=out[length-1] if len(out)>=length else float(values[-1])
        for i in range(min(length-1,len(out))): out[i]=seed
    return out


def _hma(values,length):
    if not values:return []
    wh=_wma(values,max(2,length//2)); wf=_wma(values,length)
    return _wma([2*a-b for a,b in zip(wh,wf)],max(2,int(round(math.sqrt(length)))))


def _rma(values,length):
    if not values:return []
    a=1.0/max(length,1); out=[float(values[0])]
    for v in values[1:]:out.append(a*float(v)+(1-a)*out[-1])
    return out


def _atr(candles,length=14):
    h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    if not c:return [],0.0
    tr=[max(h[0]-l[0],0.0)]
    for i in range(1,len(c)):tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    a=_rma(tr,length); return a,max(a[-1],c[-1]*0.00005)


def _rsi(c,length=14):
    if not c:return []
    g=[0.0]; d=[0.0]
    for i in range(1,len(c)):
        x=c[i]-c[i-1]; g.append(max(x,0)); d.append(max(-x,0))
    ag=_rma(g,length); ad=_rma(d,length); out=[]
    for x,y in zip(ag,ad): out.append(100.0 if y<=1e-12 and x>0 else 50.0 if y<=1e-12 else 100-100/(1+x/y))
    return out


def _pivots(values,kind,left=2,right=2):
    out=[]
    for i in range(left,len(values)-right):
        w=values[i-left:i+right+1]; v=values[i]
        if kind=="high" and v==max(w) and w.count(v)==1:out.append((i,v))
        if kind=="low" and v==min(w) and w.count(v)==1:out.append((i,v))
    return out


def _tss_4h(candles):
    c=_series(candles,"close",4)
    if len(c)<55:return {"bias":"NEUTRAL","score":0.0}
    e20=ema(c,20); e50=ema(c,50); h=_hma(c,16); span=min(3,len(c)-1)
    lv=sum((c[-1]>e20[-1]>e50[-1],e20[-1]>e20[-1-span],h[-1]>h[-2]))
    sv=sum((c[-1]<e20[-1]<e50[-1],e20[-1]<e20[-1-span],h[-1]<h[-2]))
    bias="LONG" if lv>=2 and lv>sv else "SHORT" if sv>=2 and sv>lv else "NEUTRAL"
    score=min(100.0,55+15*max(lv,sv)) if bias!="NEUTRAL" else 40+5*max(lv,sv)
    return {"bias":bias,"score":score,"close":c[-1],"ema20":e20[-1],"ema50":e50[-1],"hma16":h[-1],"long_votes":lv,"short_votes":sv}


def _structure_15m(candles):
    h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    ph=_pivots(h,"high"); pl=_pivots(l,"low")
    if len(ph)<2 or len(pl)<2:return {"state":"UNKNOWN","bias":"NEUTRAL","allow_long":False,"allow_short":False}
    h1,h2=ph[-2][1],ph[-1][1]; l1,l2=pl[-2][1],pl[-1][1]
    if h2>h1 and l2>l1:state,bias="HH/HL","LONG"
    elif h2<h1 and l2<l1:state,bias="LH/LL","SHORT"
    else:state,bias="TRANSITION","NEUTRAL"
    bos_up=c[-1]>h2; bos_down=c[-1]<l2; cu=bias=="SHORT" and bos_up; cd=bias=="LONG" and bos_down
    return {"state":state,"bias":bias,"last_swing_high":h2,"last_swing_low":l2,"bos_up":bos_up,"bos_down":bos_down,"choch_up":cu,"choch_down":cd,"allow_long":bias=="LONG" or cu,"allow_short":bias=="SHORT" or cd}


def _amd_5m(candles):
    o=_series(candles,"open",1); h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    if len(c)<40:return {"phase":"WAIT","long_ready":False,"short_ready":False}
    _,atr=_atr(candles); a0,a1=len(c)-26,len(c)-7; hi=max(h[a0:a1]); lo=min(l[a0:a1]); width=hi-lo
    ok=width<=max(4*atr,c[-1]*0.018); tol=.05*atr; li=si=None
    for i in range(a1,len(c)):
        if l[i]<lo-tol and c[i]>lo:li=i
        if h[i]>hi+tol and c[i]<hi:si=i
    lr=sr=False
    if ok and li is not None and li<len(c)-1:
        lr=any(c[i]>h[li] or (c[i]>o[i] and c[i]-o[i]>=.55*atr and c[i]>(hi+lo)/2) for i in range(li+1,len(c)))
    if ok and si is not None and si<len(c)-1:
        sr=any(c[i]<l[si] or (c[i]<o[i] and o[i]-c[i]>=.55*atr and c[i]<(hi+lo)/2) for i in range(si+1,len(c)))
    phase="DISTRIBUTION_LONG" if lr and not sr else "DISTRIBUTION_SHORT" if sr and not lr else "MANIPULATION" if li is not None or si is not None else "ACCUMULATION" if ok else "WAIT"
    return {"phase":phase,"long_ready":lr,"short_ready":sr,"accumulation_ok":ok,"range_high":hi,"range_low":lo,"manipulation_low":min(l[a1:]),"manipulation_high":max(h[a1:]),"atr":atr}


def _ifvg_1m(candles,direction):
    o=_series(candles,"open",1); h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    if len(c)<50:return {"valid":False,"direction":direction}
    _,atr=_atr(candles); tol=.12*atr; candidates=[]
    for i in range(max(2,len(c)-70),len(c)-3):
        if direction=="LONG":
            if h[i]>=l[i-2]:continue
            zl,zh=h[i],l[i-2]; inv=next((j for j in range(i+1,len(c)) if c[j]>zh),None)
            if inv is None or inv>=len(c)-1:continue
            rs=[j for j in range(max(inv+1,len(c)-5),len(c)) if l[j]<=zh+tol and c[j]>=zh]
            if not rs:continue
            j=rs[-1]
            if not (c[j]>o[j] or c[-1]>zh):continue
        else:
            if l[i]<=h[i-2]:continue
            zl,zh=h[i-2],l[i]; inv=next((j for j in range(i+1,len(c)) if c[j]<zl),None)
            if inv is None or inv>=len(c)-1:continue
            rs=[j for j in range(max(inv+1,len(c)-5),len(c)) if h[j]>=zl-tol and c[j]<=zl]
            if not rs:continue
            j=rs[-1]
            if not (c[j]<o[j] or c[-1]<zl):continue
        candidates.append({"valid":True,"direction":direction,"zone_low":zl,"zone_high":zh,"fvg_index":i,"invert_index":inv,"retest_index":j,"age":len(c)-1-j,"atr":atr})
    return candidates[-1] if candidates else {"valid":False,"direction":direction,"atr":atr}


def _micro_confirm(candles,direction,ifvg):
    """Require a fresh M1 BOS after the IFVG retest; returns confirmation swing."""
    if not ifvg.get("valid"):return {"confirmed":False,"direction":direction,"reason":"NO_IFVG"}
    h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    ret=int(ifvg.get("retest_index",len(c)-1)); start=max(3,ret-12)
    ph=[p for p in _pivots(h,"high",1,1) if start<=p[0]<ret]
    pl=[p for p in _pivots(l,"low",1,1) if start<=p[0]<ret]
    if direction=="LONG":
        if not ph:return {"confirmed":False,"direction":direction,"reason":"NO_MICRO_HIGH"}
        level=ph[-1][1]; breaks=[i for i in range(ret+1,len(c)) if c[i]>level]
        if not breaks:return {"confirmed":False,"direction":direction,"reason":"WAIT_BOS_UP","level":level}
        bi=breaks[0]; swing=min(l[max(start,ret-2):bi+1])
    else:
        if not pl:return {"confirmed":False,"direction":direction,"reason":"NO_MICRO_LOW"}
        level=pl[-1][1]; breaks=[i for i in range(ret+1,len(c)) if c[i]<level]
        if not breaks:return {"confirmed":False,"direction":direction,"reason":"WAIT_BOS_DOWN","level":level}
        bi=breaks[0]; swing=max(h[max(start,ret-2):bi+1])
    # Confirmation must remain fresh; otherwise we are chasing a move that left the IFVG.
    age=len(c)-1-bi
    return {"confirmed":age<=2,"direction":direction,"reason":"MICRO_BOS" if age<=2 else "BOS_STALE","level":level,"break_index":bi,"break_age":age,"swing":swing}


def compute(c1m,c5m=None,c15m=None,c4h=None):
    if c5m is None or c15m is None or c4h is None:return {}
    if len(c1m)<70 or len(c5m)<50 or len(c15m)<50 or len(c4h)<60:return {}
    tss=_tss_4h(c4h); ms=_structure_15m(c15m); amd=_amd_5m(c5m)
    il=_ifvg_1m(c1m,"LONG"); is_=_ifvg_1m(c1m,"SHORT"); ml=_micro_confirm(c1m,"LONG",il); ms1=_micro_confirm(c1m,"SHORT",is_)
    c1=_series(c1m,"close",4); o1=_series(c1m,"open",1); h1=_series(c1m,"high",2); l1=_series(c1m,"low",3); v1=_series(c1m,"volume",5)
    c15=_series(c15m,"close",4); e15=ema(c15,20); _,a1=_atr(c1m); r=_rsi(c1); _,a5=_atr(c5m)
    long_sig=tss.get("bias")=="LONG" and ms.get("allow_long") and amd.get("long_ready") and il.get("valid") and ml.get("confirmed")
    short_sig=tss.get("bias")=="SHORT" and ms.get("allow_short") and amd.get("short_ready") and is_.get("valid") and ms1.get("confirmed")
    d="LONG" if long_sig else "SHORT" if short_sig else "NONE"; chosen=il if d=="LONG" else is_ if d=="SHORT" else {}; micro=ml if d=="LONG" else ms1 if d=="SHORT" else {}
    sl=0.0
    if d=="LONG":
        # Lower of M5 manipulation and M1 confirmation swing, then volatility buffer.
        structural=min(float(amd["manipulation_low"]),float(micro["swing"])); sl=structural-.15*a5
    elif d=="SHORT":
        structural=max(float(amd["manipulation_high"]),float(micro["swing"])); sl=structural+.15*a5
    trigger=f"4H {tss['bias']} → M15 {ms['state']} → M5 {amd['phase']} → M1 IFVG → MICRO BOS {d}" if d!="NONE" else ""
    return {"schema":ENGINE_SCHEMA,"timeframe":"1M_EXECUTION_V7_1","open":o1[-1],"high":h1[-1],"low":l1[-1],"close":c1[-1],"volume":v1[-1],"atr1":a1,"atr5":a5,"rsi1":r[-1],"m15_close":c15[-1],"m15_ema20":e15[-1],"tss_bias":tss.get("bias","NEUTRAL"),"tss_score":float(tss.get("score",0)),"tss":tss,"structure":ms.get("state","UNKNOWN"),"structure_bias":ms.get("bias","NEUTRAL"),"m15":ms,"amd_phase":amd.get("phase","WAIT"),"amd":amd,"ifvg_long":il,"ifvg_short":is_,"micro_long":ml,"micro_short":ms1,"micro_confirmed":bool(micro.get("confirmed")),"micro_level":float(micro.get("level",0) or 0),"micro_swing":float(micro.get("swing",0) or 0),"ifvg_valid":bool(chosen.get("valid")),"ifvg_low":float(chosen.get("zone_low",0) or 0),"ifvg_high":float(chosen.get("zone_high",0) or 0),"manipulation_low":float(amd.get("manipulation_low",0) or 0),"manipulation_high":float(amd.get("manipulation_high",0) or 0),"sl":sl,"long_signal":long_sig,"short_signal":short_sig,"direction":d,"trigger":trigger,"runner_exit_long":bool(ms.get("choch_down")) or amd.get("phase")=="DISTRIBUTION_SHORT","runner_exit_short":bool(ms.get("choch_up")) or amd.get("phase")=="DISTRIBUTION_LONG"}


class IndicatorEngine:
    def compute(self,c1m,c5m,c15m,c4h):return compute(c1m,c5m,c15m,c4h)
