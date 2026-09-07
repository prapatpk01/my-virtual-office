"""Adaptive SMC MTF V7.3 indicator engine.

Pipeline (closed candles only):
  4H  TSS-style direction (EMA20/50 + HMA16 slope)
  15M Market Structure (HH/HL, LH/LL, BOS/CHOCH)
  5M  AMD liquidity setup
  5M/1M Momentum Quality (RSI/SMA, MACD histogram, ADX, Bollinger)
  1M  IFVG -> micro BOS -> pullback/no-chase execution

KDJ is intentionally a veto only.  It is not another hard trigger.  This keeps
V7.3 selective without returning to the over-filtered behaviour of older bots.
"""
from __future__ import annotations
from typing import Any, Dict, List
import math

ENGINE_SCHEMA = "adaptive-smc-mtf-v1"
NO_CHASE_ATR1 = 0.50
BOS_ARM_BARS = 8
RUNNER_ATR5_BUFFER = 0.12
QUALITY_MIN = 65.0


def _v(c: Any, name: str, idx: int) -> float:
    v = getattr(c, name, None)
    if v is None and isinstance(c, dict): v = c.get(name)
    if v is None and isinstance(c, (list, tuple)) and len(c) > idx: v = c[idx]
    return float(v or 0.0)


def _ts(c: Any):
    v = getattr(c, "timestamp", None)
    if v is None and isinstance(c, dict): v = c.get("timestamp")
    if v is None and isinstance(c, (list, tuple)) and c: v = c[0]
    return str(v or "0")


def _series(c, name, idx): return [_v(x, name, idx) for x in c]


def ema(values: List[float], length: int) -> List[float]:
    if not values: return []
    a = 2.0 / (length + 1.0); out = [float(values[0])]
    for v in values[1:]: out.append(a * float(v) + (1-a) * out[-1])
    return out


def _sma(values, length):
    if not values: return []
    length=max(1,int(length)); out=[]
    for i in range(len(values)):
        w=values[max(0,i-length+1):i+1]; out.append(sum(w)/len(w))
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


def _macd_hist(c):
    if not c:return []
    e12=ema(c,12); e26=ema(c,26); mac=[a-b for a,b in zip(e12,e26)]; sig=ema(mac,9)
    return [a-b for a,b in zip(mac,sig)]


def _adx(candles,length=14):
    h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    if len(c)<2:return []
    tr=[max(h[0]-l[0],0.0)]; pdm=[0.0]; mdm=[0.0]
    for i in range(1,len(c)):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm.append(up if up>dn and up>0 else 0.0); mdm.append(dn if dn>up and dn>0 else 0.0)
        tr.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    atr=_rma(tr,length); p=_rma(pdm,length); m=_rma(mdm,length); dx=[]
    for a,x,y in zip(atr,p,m):
        pdi=100*x/max(a,1e-12); mdi=100*y/max(a,1e-12); dx.append(100*abs(pdi-mdi)/max(pdi+mdi,1e-12))
    return _rma(dx,length)


def _bollinger(c,length=20,mult=2.0):
    mid=_sma(c,length); upper=[]; lower=[]; width=[]
    for i,v in enumerate(c):
        w=c[max(0,i-length+1):i+1]; mean=mid[i]; sd=math.sqrt(sum((x-mean)**2 for x in w)/max(len(w),1))
        u=mean+mult*sd; lo=mean-mult*sd; upper.append(u); lower.append(lo); width.append((u-lo)/max(abs(mean),1e-12))
    return mid,upper,lower,width


def _kdj(candles,length=9):
    h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    k=50.0; d=50.0; out=[]
    for i,px in enumerate(c):
        hi=max(h[max(0,i-length+1):i+1]); lo=min(l[max(0,i-length+1):i+1]); rsv=50.0 if hi<=lo else 100*(px-lo)/(hi-lo)
        k=(2*k+rsv)/3; d=(2*d+k)/3; j=3*k-2*d; out.append((k,d,j))
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
    long_id=f"L:{_ts(candles[li])}:{hi:.8g}:{lo:.8g}" if li is not None else ""
    short_id=f"S:{_ts(candles[si])}:{hi:.8g}:{lo:.8g}" if si is not None else ""
    return {"phase":phase,"long_ready":lr,"short_ready":sr,"accumulation_ok":ok,"range_high":hi,"range_low":lo,"manipulation_low":min(l[a1:]),"manipulation_high":max(h[a1:]),"atr":atr,"long_cycle_id":long_id,"short_cycle_id":short_id}


def _ifvg_1m(candles,direction):
    o=_series(candles,"open",1); h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4)
    if len(c)<50:return {"valid":False,"direction":direction}
    _,atr=_atr(candles); tol=.12*atr; candidates=[]
    for i in range(max(2,len(c)-70),len(c)-3):
        if direction=="LONG":
            if h[i]>=l[i-2]:continue
            zl,zh=h[i],l[i-2]; inv=next((j for j in range(i+1,len(c)) if c[j]>zh),None)
            if inv is None or inv>=len(c)-1:continue
            rs=[j for j in range(max(inv+1,len(c)-8),len(c)) if l[j]<=zh+tol and c[j]>=zl]
            if not rs:continue
            j=rs[-1]
        else:
            if l[i]<=h[i-2]:continue
            zl,zh=h[i-2],l[i]; inv=next((j for j in range(i+1,len(c)) if c[j]<zl),None)
            if inv is None or inv>=len(c)-1:continue
            rs=[j for j in range(max(inv+1,len(c)-8),len(c)) if h[j]>=zl-tol and c[j]<=zh]
            if not rs:continue
            j=rs[-1]
        candidates.append({"valid":True,"direction":direction,"zone_low":zl,"zone_high":zh,"fvg_index":i,"invert_index":inv,"retest_index":j,"age":len(c)-1-j,"atr":atr})
    return candidates[-1] if candidates else {"valid":False,"direction":direction,"atr":atr}


def _micro_confirm(candles,direction,ifvg):
    if not ifvg.get("valid"):return {"confirmed":False,"armed":False,"entry_ready":False,"direction":direction,"reason":"NO_IFVG"}
    h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4); _,atr=_atr(candles)
    ret=int(ifvg.get("retest_index",len(c)-1)); start=max(3,ret-12)
    ph=[p for p in _pivots(h,"high",1,1) if start<=p[0]<ret]; pl=[p for p in _pivots(l,"low",1,1) if start<=p[0]<ret]
    if direction=="LONG":
        if not ph:return {"confirmed":False,"armed":False,"entry_ready":False,"direction":direction,"reason":"NO_MICRO_HIGH"}
        level=ph[-1][1]; breaks=[i for i in range(ret+1,len(c)) if c[i]>level]
        if not breaks:return {"confirmed":False,"armed":False,"entry_ready":False,"direction":direction,"reason":"WAIT_BOS_UP","level":level}
        bi=breaks[0]; swing=min(l[max(start,ret-2):bi+1]); age=len(c)-1-bi; armed=age<=BOS_ARM_BARS
        distance=max(0.0,c[-1]-float(ifvg["zone_high"])); near=distance<=NO_CHASE_ATR1*atr
        retest_now=l[-1]<=float(ifvg["zone_high"])+NO_CHASE_ATR1*atr and c[-1]>=float(ifvg["zone_low"])
    else:
        if not pl:return {"confirmed":False,"armed":False,"entry_ready":False,"direction":direction,"reason":"NO_MICRO_LOW"}
        level=pl[-1][1]; breaks=[i for i in range(ret+1,len(c)) if c[i]<level]
        if not breaks:return {"confirmed":False,"armed":False,"entry_ready":False,"direction":direction,"reason":"WAIT_BOS_DOWN","level":level}
        bi=breaks[0]; swing=max(h[max(start,ret-2):bi+1]); age=len(c)-1-bi; armed=age<=BOS_ARM_BARS
        distance=max(0.0,float(ifvg["zone_low"])-c[-1]); near=distance<=NO_CHASE_ATR1*atr
        retest_now=h[-1]>=float(ifvg["zone_low"])-NO_CHASE_ATR1*atr and c[-1]<=float(ifvg["zone_high"])
    ready=armed and near and retest_now; reason="PULLBACK_READY" if ready else "WAIT_PULLBACK" if armed else "BOS_STALE"
    return {"confirmed":True,"armed":armed,"entry_ready":ready,"direction":direction,"reason":reason,"level":level,"break_index":bi,"break_age":age,"swing":swing,"distance_atr":distance/max(atr,1e-12)}


def _momentum_quality(c1m,c5m,direction):
    c1=_series(c1m,"close",4); c5=_series(c5m,"close",4)
    r1=_rsi(c1); r5=_rsi(c5); rs1=_sma(r1,14); rs5=_sma(r5,14)
    mh=_macd_hist(c5); adx=_adx(c5m); mid,up,lo,bw=_bollinger(c5); kdj=_kdj(c5m)
    long=direction=="LONG"; score=0.0; parts={}

    # RSI/SMA = 35 points. M5 alignment is primary; fresh M1 alignment can rescue timing.
    rsi_ok=(r5[-1]>rs5[-1] and r5[-1]<70) if long else (r5[-1]<rs5[-1] and r5[-1]>30)
    cross1=(r1[-1]>rs1[-1] and r1[-2]<=rs1[-2]) if long else (r1[-1]<rs1[-1] and r1[-2]>=rs1[-2])
    m1_ok=(r1[-1]>rs1[-1] and r1[-1]<72) if long else (r1[-1]<rs1[-1] and r1[-1]>28)
    rsi_points=35.0 if rsi_ok else 25.0 if (m1_ok or cross1) else 0.0; score+=rsi_points; parts["rsi_sma"]=rsi_points

    # MACD histogram slope = 25 points. Cross above/below zero is not required.
    macd_ok=(mh[-1]>mh[-2] and mh[-2]>=mh[-3]) if long else (mh[-1]<mh[-2] and mh[-2]<=mh[-3])
    macd_soft=(mh[-1]>mh[-2]) if long else (mh[-1]<mh[-2])
    macd_points=25.0 if macd_ok else 15.0 if macd_soft else 0.0; score+=macd_points; parts["macd_hist"]=macd_points

    # ADX = 20 points: trending or clearly rising is enough.
    adx_now=adx[-1] if adx else 0.0; adx_rising=len(adx)>=3 and adx[-1]>adx[-2]>adx[-3]
    adx_points=20.0 if adx_now>=18 else 12.0 if adx_rising else 0.0; score+=adx_points; parts["adx"]=adx_points

    # Bollinger = 20 points: avoid extreme chase and dead compression.
    width_ok=bw[-1]>=0.003
    bb_ok=(c5[-1]<=up[-1] and c5[-1]>=mid[-1]-0.35*(mid[-1]-lo[-1])) if long else (c5[-1]>=lo[-1] and c5[-1]<=mid[-1]+0.35*(up[-1]-mid[-1]))
    bb_points=20.0 if width_ok and bb_ok else 10.0 if width_ok else 0.0; score+=bb_points; parts["bollinger"]=bb_points

    k,d,j=kdj[-1] if kdj else (50.0,50.0,50.0)
    veto=(long and k>90 and j>100) or ((not long) and k<10 and j<0)
    return {"score":score,"pass":score>=QUALITY_MIN and not veto,"veto":veto,"parts":parts,
            "rsi1":r1[-1],"rsi1_sma":rs1[-1],"rsi5":r5[-1],"rsi5_sma":rs5[-1],
            "macd_hist":mh[-1],"macd_hist_prev":mh[-2],"adx":adx_now,"bb_width":bw[-1],
            "bb_mid":mid[-1],"bb_upper":up[-1],"bb_lower":lo[-1],"kdj_k":k,"kdj_d":d,"kdj_j":j}


def _runner_trails(c5m, atr5):
    h=_series(c5m,"high",2); l=_series(c5m,"low",3); ph=_pivots(h,"high",1,1); pl=_pivots(l,"low",1,1)
    return ((pl[-1][1]-RUNNER_ATR5_BUFFER*atr5) if pl else 0.0,
            (ph[-1][1]+RUNNER_ATR5_BUFFER*atr5) if ph else 0.0)


def compute(c1m,c5m=None,c15m=None,c4h=None):
    if c5m is None or c15m is None or c4h is None:return {}
    if len(c1m)<70 or len(c5m)<50 or len(c15m)<50 or len(c4h)<60:return {}
    tss=_tss_4h(c4h); ms=_structure_15m(c15m); amd=_amd_5m(c5m)
    il=_ifvg_1m(c1m,"LONG"); is_=_ifvg_1m(c1m,"SHORT"); ml=_micro_confirm(c1m,"LONG",il); ms1=_micro_confirm(c1m,"SHORT",is_)
    ql=_momentum_quality(c1m,c5m,"LONG"); qs=_momentum_quality(c1m,c5m,"SHORT")
    c1=_series(c1m,"close",4); o1=_series(c1m,"open",1); h1=_series(c1m,"high",2); l1=_series(c1m,"low",3); v1=_series(c1m,"volume",5)
    c15=_series(c15m,"close",4); e15=ema(c15,20); _,a1=_atr(c1m); _,a5=_atr(c5m); rtl,rts=_runner_trails(c5m,a5)
    long_base=tss.get("bias")=="LONG" and ms.get("allow_long") and amd.get("long_ready") and il.get("valid") and ml.get("entry_ready")
    short_base=tss.get("bias")=="SHORT" and ms.get("allow_short") and amd.get("short_ready") and is_.get("valid") and ms1.get("entry_ready")
    long_sig=long_base and ql.get("pass"); short_sig=short_base and qs.get("pass")
    d="LONG" if long_sig else "SHORT" if short_sig else "NONE"; chosen=il if d=="LONG" else is_ if d=="SHORT" else {}; micro=ml if d=="LONG" else ms1 if d=="SHORT" else {}; quality=ql if d=="LONG" else qs if d=="SHORT" else {}
    sl=0.0; cycle=""
    if d=="LONG":
        structural=min(float(amd["manipulation_low"]),float(micro["swing"])); sl=structural-.15*a5; cycle=amd.get("long_cycle_id","")
    elif d=="SHORT":
        structural=max(float(amd["manipulation_high"]),float(micro["swing"])); sl=structural+.15*a5; cycle=amd.get("short_cycle_id","")
    trigger=f"4H {tss['bias']} → M15 {ms['state']} → M5 {amd['phase']} → Q {quality.get('score',0):.0f} → M1 IFVG → MICRO BOS → PULLBACK {d}" if d!="NONE" else ""
    return {"schema":ENGINE_SCHEMA,"timeframe":"1M_EXECUTION_V7_3","open":o1[-1],"high":h1[-1],"low":l1[-1],"close":c1[-1],"volume":v1[-1],"atr1":a1,"atr5":a5,
            "m15_close":c15[-1],"m15_ema20":e15[-1],"tss_bias":tss.get("bias","NEUTRAL"),"tss_score":float(tss.get("score",0)),"tss":tss,
            "structure":ms.get("state","UNKNOWN"),"structure_bias":ms.get("bias","NEUTRAL"),"m15":ms,"amd_phase":amd.get("phase","WAIT"),"amd":amd,
            "amd_cycle_id":cycle,"ifvg_long":il,"ifvg_short":is_,"micro_long":ml,"micro_short":ms1,"quality_long":ql,"quality_short":qs,
            "quality_score":float(quality.get("score",0) or 0),"quality_pass":bool(quality.get("pass")),"quality_veto":bool(quality.get("veto")),
            "micro_confirmed":bool(micro.get("confirmed")),"micro_armed":bool(micro.get("armed")),"pullback_ready":bool(micro.get("entry_ready")),
            "micro_level":float(micro.get("level",0) or 0),"micro_swing":float(micro.get("swing",0) or 0),"ifvg_valid":bool(chosen.get("valid")),
            "ifvg_low":float(chosen.get("zone_low",0) or 0),"ifvg_high":float(chosen.get("zone_high",0) or 0),"manipulation_low":float(amd.get("manipulation_low",0) or 0),
            "manipulation_high":float(amd.get("manipulation_high",0) or 0),"runner_trail_long":rtl,"runner_trail_short":rts,
            "runner_momentum_long":bool(ql.get("score",0)>=65 and not ql.get("veto")),"runner_momentum_short":bool(qs.get("score",0)>=65 and not qs.get("veto")),
            "sl":sl,"long_signal":long_sig,"short_signal":short_sig,"direction":d,"trigger":trigger,
            "runner_exit_long":bool(ms.get("choch_down")),"runner_exit_short":bool(ms.get("choch_up"))}


class IndicatorEngine:
    def compute(self,c1m,c5m,c15m,c4h):return compute(c1m,c5m,c15m,c4h)
