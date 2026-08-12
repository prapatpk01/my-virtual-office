"""Adaptive Momentum v3.3 indicator engine.
1H EMA20/50 selects direction. 15M uses ADX+CHOP quality, MACD histogram+ROC9 momentum,
EMA20 distance+market structure location, and EMA8/13 cross as the only entry trigger.
"""
from __future__ import annotations
from typing import Any, Dict, List
import math

ENGINE_SCHEMA = "adaptive-momentum-v3.3-15m"


def _v(c: Any, name: str, idx: int) -> float:
    v = getattr(c, name, None)
    if v is None and isinstance(c, dict): v = c.get(name)
    if v is None and isinstance(c, (list, tuple)) and len(c) > idx: v = c[idx]
    return float(v or 0.0)


def _ts(c: Any) -> float:
    v = getattr(c, "timestamp", None)
    if v is None and isinstance(c, dict): v = c.get("timestamp")
    if v is None and isinstance(c, (list, tuple)) and c: v = c[0]
    return float(v or 0.0)


def _series(cs, name, idx): return [_v(c, name, idx) for c in cs]


def ema(xs: List[float], n: int) -> List[float]:
    if not xs: return []
    a = 2.0 / (n + 1.0); out = [float(xs[0])]
    for x in xs[1:]: out.append(a * float(x) + (1-a) * out[-1])
    return out


def _rma(xs: List[float], n: int) -> List[float]:
    if not xs: return []
    a = 1.0 / max(n, 1); out = [float(xs[0])]
    for x in xs[1:]: out.append(a * float(x) + (1-a) * out[-1])
    return out


def _tr(h, l, c):
    out = [max(h[0]-l[0], 0.0)]
    for i in range(1, len(c)): out.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    return out


def _adx(h, l, c, n=14):
    tr = _tr(h,l,c); pd=[0.0]; md=[0.0]
    for i in range(1,len(c)):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pd.append(up if up>dn and up>0 else 0.0); md.append(dn if dn>up and dn>0 else 0.0)
    ar,pr,mr=_rma(tr,n),_rma(pd,n),_rma(md,n); dx=[]
    for a,p,m in zip(ar,pr,mr):
        if a<=1e-12: dx.append(0.0); continue
        pdi=100*p/a; mdi=100*m/a; s=pdi+mdi
        dx.append(100*abs(pdi-mdi)/s if s>1e-12 else 0.0)
    return _rma(dx,n)


def _chop(h,l,c,n=14):
    s=sum(_tr(h,l,c)[-n:]); span=max(h[-n:])-min(l[-n:])
    return 100.0 if s<=0 or span<=1e-12 else 100*math.log10(s/span)/math.log10(n)


def _closed_1h(cs):
    groups={}
    for c in cs:
        t=int(_ts(c));
        if t<=0: continue
        if t<10_000_000_000: t*=1000
        groups.setdefault(t//3_600_000,[]).append(c)
    out=[]
    for b in sorted(groups):
        rows=sorted(groups[b],key=_ts)
        if len(rows)!=4: continue
        out.append([b*3_600_000,_v(rows[0],"open",1),max(_v(x,"high",2) for x in rows),min(_v(x,"low",3) for x in rows),_v(rows[-1],"close",4),sum(_v(x,"volume",5) for x in rows)])
    return out


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles)<240: return {}
    o=_series(candles,"open",1); h=_series(candles,"high",2); l=_series(candles,"low",3); c=_series(candles,"close",4); v=_series(candles,"volume",5)
    e8,e13,e20,e50=ema(c,8),ema(c,13),ema(c,20),ema(c,50)
    ml=[a-b for a,b in zip(ema(c,12),ema(c,26))]; ms=ema(ml,9); mh=[a-b for a,b in zip(ml,ms)]
    atrs=_rma(_tr(h,l,c),14); atr=max(atrs[-1],c[-1]*0.0005); adxs=_adx(h,l,c,14); chop=_chop(h,l,c,14)
    h1=_closed_1h(candles)
    if len(h1)<50: return {}
    hc=[x[4] for x in h1]; te20=ema(hc,20); te50=ema(hc,50)
    trend_bull=te20[-1]>te50[-1]; trend_bear=te20[-1]<te50[-1]

    # Momentum: 1/2 is enough to pass; 2/2 is STRONG.
    roc9=((c[-1]/c[-10])-1.0)*100.0 if c[-10] else 0.0
    hist_long=mh[-1]>mh[-2]; hist_short=mh[-1]<mh[-2]
    roc_long=roc9>0; roc_short=roc9<0
    mom_long=int(hist_long)+int(roc_long); mom_short=int(hist_short)+int(roc_short)

    # Location tool 1: distance from 15M EMA20, not EMA13.
    dist20=abs(c[-1]-e20[-1])/atr
    ema20_loc_long=c[-1]>=e20[-1] and dist20<=1.2
    ema20_loc_short=c[-1]<=e20[-1] and dist20<=1.2

    # Location tool 2: simple confirmed swing structure. It is supportive, not a hard 2/2 gate.
    prior_low=min(l[-7:-2]); prior_high=max(h[-7:-2])
    recent_low=min(l[-3:]); recent_high=max(h[-3:])
    hl=recent_low>prior_low; lh=recent_high<prior_high
    location_score_long=int(ema20_loc_long)+int(hl)
    location_score_short=int(ema20_loc_short)+int(lh)

    cross_up=e8[-1]>e13[-1] and e8[-2]<=e13[-2]
    cross_down=e8[-1]<e13[-1] and e8[-2]>=e13[-2]
    recent_low5=min(l[-6:-1]); recent_high5=max(h[-6:-1])

    return {
        "schema":ENGINE_SCHEMA,"trend_tf":"1H","trend_ema20":te20[-1],"trend_ema50":te50[-1],"trend_bull":trend_bull,"trend_bear":trend_bear,
        "open":o[-1],"high":h[-1],"low":l[-1],"close":c[-1],"prev_high":h[-2],"prev_low":l[-2],"prev_close":c[-2],
        "ema8":e8[-1],"ema13":e13[-1],"ema20":e20[-1],"ema50":e50[-1],"ema8_prev":e8[-2],"ema13_prev":e13[-2],
        "ema_cross_up":cross_up,"ema_cross_down":cross_down,"entry_bull":e8[-1]>e13[-1],"entry_bear":e8[-1]<e13[-1],
        "macd":ml[-1],"macd_signal":ms[-1],"macd_hist":mh[-1],"macd_hist_prev":mh[-2],
        "macd_hist_improving_long":hist_long,"macd_hist_improving_short":hist_short,
        "macd_hist_weaken_long_3":mh[-1]<mh[-2]<mh[-3]<mh[-4],"macd_hist_weaken_short_3":mh[-1]>mh[-2]>mh[-3]>mh[-4],
        "roc9":roc9,"roc_long":roc_long,"roc_short":roc_short,"momentum_score_long":mom_long,"momentum_score_short":mom_short,
        "adx":adxs[-1],"adx_prev":adxs[-2],"adx_rising":adxs[-1]>adxs[-2],"chop":chop,"atr":atr,
        "distance_ema20_atr":dist20,"ema20_location_long":ema20_loc_long,"ema20_location_short":ema20_loc_short,
        "structure_hl":hl,"structure_lh":lh,"location_score_long":location_score_long,"location_score_short":location_score_short,
        "location_long":location_score_long>=1,"location_short":location_score_short>=1,
        "trigger_long":cross_up,"trigger_short":cross_down,
        "recent_low":recent_low5,"recent_high":recent_high5,"volume":v[-1],
    }


class IndicatorEngine:
    def compute(self,c15m:List[Any],c1h:List[Any],c4h:List[Any]):
        return compute(c15m),{},{}
