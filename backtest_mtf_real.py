"""
MTF Backtest Optimizer — Real Binance Data (Jan–May 2026)
Timeframes: 15m (entry confirm) | 1h (signal) | 4h (trend bias)

Strategies: WaveTrend, Momentum Score, UT Bot, MACD/EMA
MTF gate: BUY only when 4H bullish + 15M micro-confirm
Target: WR ≥ 67%, ≥ 8 trades/month per strategy
"""
import csv, glob, math, itertools, os
import numpy as np
from datetime import datetime, timezone

# ── Data paths ─────────────────────────────────────────────────────────────
BASE = "/tmp/chart_data"

# ── Candle loader ──────────────────────────────────────────────────────────

class C:
    __slots__ = ("ts","open","high","low","close","volume")
    def __init__(self, ts, o, h, l, c, v):
        self.ts = int(ts); self.open = float(o); self.high = float(h)
        self.low = float(l); self.close = float(c); self.volume = float(v)

def load_tf(tf: str) -> list:
    pattern = os.path.join(BASE, tf, tf, f"*.csv")
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 6: continue
                # Binance: ts(μs), o, h, l, c, vol, ...
                ts_us = int(parts[0])
                ts_ms = ts_us // 1000   # convert μs → ms
                rows.append(C(ts_ms, parts[1], parts[2], parts[3], parts[4], parts[5]))
    rows.sort(key=lambda x: x.ts)
    return rows

def ts_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d")

# ── Indicator helpers ──────────────────────────────────────────────────────

def ema(v, p):
    a = np.array(v, dtype=float); r = np.full(len(a), np.nan)
    if len(a) < p: return r
    k = 2/(p+1); r[p-1] = a[:p].mean()
    for i in range(p, len(a)): r[i] = a[i]*k + r[i-1]*(1-k)
    return r

def sma(v, p):
    a = np.array(v, dtype=float); r = np.full(len(a), np.nan)
    for i in range(p-1, len(a)): r[i] = a[i-p+1:i+1].mean()
    return r

def wma(v, p):
    a = np.array(v, dtype=float); r = np.full(len(a), np.nan)
    w = np.arange(1, p+1, dtype=float); d = w.sum()
    for i in range(p-1, len(a)): r[i] = np.dot(a[i-p+1:i+1], w)/d
    return r

def hma(v, p):
    half=max(2,p//2); sq=max(2,int(round(math.sqrt(p))))
    raw = 2*wma(v,half) - wma(v,p)
    val = [float(x) for x in raw if not np.isnan(x)]
    if len(val) < sq: return np.full(len(v), np.nan)
    h = wma(val, sq); pad = len(v)-len(h)
    return np.concatenate([np.full(pad, np.nan), h])

def atr_calc(candles, p=14):
    n = len(candles); tr = np.full(n, np.nan)
    for i in range(1, n):
        h=candles[i].high; l=candles[i].low; pc=candles[i-1].close
        tr[i] = max(h-l, abs(h-pc), abs(l-pc))
    r = np.full(n, np.nan)
    if n > p:
        r[p] = float(np.nanmean(tr[1:p+1]))
        for i in range(p+1, n): r[i] = (r[i-1]*(p-1)+tr[i])/p
    return r

def macd_calc(v, fast=12, slow=26, sig=9):
    fe=ema(v,fast); se=ema(v,slow); ml=fe-se
    vals=[x for x in ml if not np.isnan(x)]
    if len(vals)<sig: sl=np.full(len(ml),np.nan)
    else:
        s=ema(vals,sig); pad=len(ml)-len(s)
        sl=np.concatenate([np.full(pad,np.nan),s])
    return ml, sl, ml-sl

def rsi_w(closes, p=14):
    n=len(closes); r=np.full(n,np.nan)
    if n<p+1: return r
    d=np.diff(closes)
    g=np.where(d>0,d,0.); ls=np.where(d<0,-d,0.)
    ag=float(np.mean(g[:p])); al=float(np.mean(ls[:p]))
    r[p]=100. if al<1e-10 else 100-100/(1+ag/al)
    for i in range(p+1,n):
        ag=(ag*(p-1)+g[i-1])/p; al=(al*(p-1)+ls[i-1])/p
        r[i]=100. if al<1e-10 else 100-100/(1+ag/al)
    return r

def ha(candles):
    n=len(candles); ho=np.zeros(n); hc=np.zeros(n)
    hh=np.zeros(n); hl=np.zeros(n)
    for i,c in enumerate(candles):
        hc[i]=(c.open+c.high+c.low+c.close)/4
        ho[i]=((c.open+c.close)/2 if i==0 else (ho[i-1]+hc[i-1])/2)
        hh[i]=max(c.high,ho[i],hc[i]); hl[i]=min(c.low,ho[i],hc[i])
    class _C:
        __slots__=("ts","open","high","low","close","volume")
        def __init__(s,i):
            s.ts=candles[i].ts; s.open=ho[i]; s.high=hh[i]
            s.low=hl[i]; s.close=hc[i]; s.volume=candles[i].volume
    return [_C(i) for i in range(n)], ho, hc, hh, hl

# ══════════════════════════════════════════════════════════════════════════════
# MTF GATE — 4H trend + 15M micro confirm
# ══════════════════════════════════════════════════════════════════════════════

def build_4h_bias(c4h: list) -> np.ndarray:
    """
    Returns array indexed by 4H bar: +1 bull, -1 bear, 0 neutral.
    Bull: ha_close > EMA20(ha_close) AND ADX > 18 uptrend
    """
    _, _, hc4, hh4, hl4 = ha(c4h)
    em20 = ema(hc4.tolist(), 20)
    n = len(c4h)
    # Simple ADX proxy: directional strength via DM
    plus_dm  = np.zeros(n); minus_dm = np.zeros(n)
    tr4 = np.full(n, np.nan)
    for i in range(1, n):
        h=c4h[i].high; l=c4h[i].low; ph=c4h[i-1].high; pl=c4h[i-1].low
        hm=h-ph; lm=pl-l; pc=c4h[i-1].close
        tr4[i] = max(h-l, abs(h-pc), abs(l-pc))
        plus_dm[i]  = hm if hm>lm and hm>0 else 0
        minus_dm[i] = lm if lm>hm and lm>0 else 0
    atr14 = np.full(n, np.nan)
    if n > 14:
        atr14[14] = float(np.nanmean(tr4[1:15]))
        for i in range(15, n): atr14[i] = (atr14[i-1]*13+tr4[i])/14
    di_plus = np.full(n, np.nan); di_minus = np.full(n, np.nan)
    for i in range(14, n):
        if atr14[i] > 0:
            di_plus[i]  = plus_dm[i] /atr14[i]*100
            di_minus[i] = minus_dm[i]/atr14[i]*100
    # Smooth DI with 14-bar EMA
    dp = ema([float(x) if not np.isnan(x) else 0 for x in di_plus],  14)
    dm = ema([float(x) if not np.isnan(x) else 0 for x in di_minus], 14)

    bias = np.zeros(n, dtype=int)
    for i in range(n):
        if np.isnan(em20[i]): continue
        bull_ema  = float(hc4[i]) > float(em20[i])
        di_bull   = float(dp[i]) > float(dm[i]) if not (np.isnan(dp[i]) or np.isnan(dm[i])) else False
        bear_ema  = float(hc4[i]) < float(em20[i])
        di_bear   = float(dm[i]) > float(dp[i]) if not (np.isnan(dp[i]) or np.isnan(dm[i])) else False
        if bull_ema and di_bull: bias[i] = 1
        elif bear_ema and di_bear: bias[i] = -1
    return bias

def build_15m_confirm(c15: list) -> np.ndarray:
    """
    Returns array indexed by 15m bar: +1 bull confirm, -1 bear confirm, 0 neutral.
    Bull: RSI14 > 50 AND RSI trending up (> EMA9 of RSI) AND last HA bar bullish
    """
    _, ho15, hc15, hh15, hl15 = ha(c15)
    rs = rsi_w(hc15, 14)
    valid = np.where(~np.isnan(rs))[0]
    rs_ema = np.full(len(rs), np.nan)
    if len(valid) >= 9:
        s = int(valid[0])
        re = ema([float(rs[i]) for i in range(s, len(rs))], 9)
        rs_ema[s:s+len(re)] = re[:len(rs)-s]
    n = len(c15)
    conf = np.zeros(n, dtype=int)
    for i in range(1, n):
        if np.isnan(rs[i]) or np.isnan(rs_ema[i]): continue
        bull_rsi  = float(rs[i]) > 50 and float(rs[i]) > float(rs_ema[i])
        bull_ha   = float(hc15[i]) > float(ho15[i])
        bear_rsi  = float(rs[i]) < 50 and float(rs[i]) < float(rs_ema[i])
        bear_ha   = float(hc15[i]) < float(ho15[i])
        if bull_rsi and bull_ha: conf[i] = 1
        elif bear_rsi and bear_ha: conf[i] = -1
    return conf

# Build lookup maps: ts_ms → nearest index for each TF
def build_ts_index(candles: list) -> dict:
    return {c.ts: i for i, c in enumerate(candles)}

def find_tf_idx(ts_1h_ms: int, tf_candles: list, ts_map: dict, tf_minutes: int):
    """Find the bar in tf_candles whose window contains ts_1h_ms."""
    # Align to tf bar start: floor(ts / tf_ms) * tf_ms
    tf_ms = tf_minutes * 60 * 1000
    aligned = (ts_1h_ms // tf_ms) * tf_ms
    if aligned in ts_map:
        return ts_map[aligned]
    # fallback: nearest prior bar
    for delta in range(0, tf_minutes*2 + 1, tf_minutes):
        key = aligned - delta * 60 * 1000
        if key in ts_map:
            return ts_map[key]
    return None

# ══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate(c1h, buy_s, sell_s, atr14,
             sl_m, rr, use_pct=False, sl_pct=0, tp_pct=0,
             ha_c=None, ha_h=None, ha_l=None):
    LF = 60
    cls = np.array([c.close for c in c1h])
    hig = np.array([c.high  for c in c1h])
    low = np.array([c.low   for c in c1h])
    ec = ha_c if ha_c is not None else cls
    eh = ha_h if ha_h is not None else hig
    el = ha_l if ha_l is not None else low
    n = len(c1h)
    wins = losses = 0; total_r = 0.0; trades = []
    for i in range(n):
        d = 1 if buy_s[i] else (-1 if sell_s[i] else 0)
        if d == 0: continue
        av = float(atr14[i]) if not np.isnan(atr14[i]) else 0
        if av <= 0 and not use_pct: continue
        ep = float(ec[i])
        if use_pct:
            sl_p = ep*(1-sl_pct) if d==1 else ep*(1+sl_pct)
            tp_p = ep*(1+tp_pct) if d==1 else ep*(1-tp_pct)
            rr_local = tp_pct/sl_pct if sl_pct > 0 else rr
        else:
            sl_p = ep-sl_m*av if d==1 else ep+sl_m*av
            tp_p = ep+sl_m*rr*av if d==1 else ep-sl_m*rr*av
            rr_local = rr
        out = 0; xp = ep
        for j in range(i+1, min(i+LF, n)):
            if d == 1:
                if el[j] <= sl_p: out=-1; xp=sl_p; break
                if eh[j] >= tp_p: out=1;  xp=tp_p; break
            else:
                if eh[j] >= sl_p: out=-1; xp=sl_p; break
                if el[j] <= tp_p: out=1;  xp=tp_p; break
        if out == 1:  wins+=1;   total_r+=rr_local
        elif out==-1: losses+=1; total_r-=1.0
        if out != 0: trades.append((i,d,out,ep,xp,c1h[i].ts))
    return wins, losses, total_r, trades

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY SIGNAL BUILDERS  (1H candles only — no MTF yet)
# ══════════════════════════════════════════════════════════════════════════════

def sig_wt(c1h, n1=2, n2=4):
    hac, _, hc, hh, hl = ha(c1h)
    hh_a=np.array([c.high for c in hac]); hl_a=np.array([c.low for c in hac])
    ap=(hh_a+hl_a+hc)/3
    esa=ema(ap.tolist(),n1)
    dabs=np.abs(ap-esa); fo=np.where(~np.isnan(dabs))[0]
    df=dabs.copy()
    if len(fo): df[:fo[0]]=dabs[fo[0]]
    d=ema(df.tolist(),n1)
    ci=np.where(d>1e-10,(ap-esa)/(0.015*d),0.)
    wt1=ema(ci.tolist(),n2); wt2=sma(wt1.tolist(),4)
    n=len(c1h); buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool)
    pd=0
    for i in range(1,n):
        if np.isnan(wt1[i]) or np.isnan(wt2[i]): continue
        cu=float(wt1[i-1])<=float(wt2[i-1]) and float(wt1[i])>float(wt2[i])
        cd=float(wt1[i-1])>=float(wt2[i-1]) and float(wt1[i])<float(wt2[i])
        if cu and pd!=1: buy_s[i]=True; pd=1
        elif cd and pd!=-1: sell_s[i]=True; pd=-1
        elif not cu and not cd: pd=0
    atr14=atr_calc(hac,14)
    return buy_s,sell_s,atr14,hc,hh_a,hl_a

def sig_momentum(c1h, threshold=2, reentry=10):
    hac,ho,hc,hh,hl=ha(c1h)
    cl=hc.tolist()
    rs=rsi_w(hc,14)
    re=np.full(len(hc),np.nan); valid=np.where(~np.isnan(rs))[0]
    if len(valid)>=9:
        s=int(valid[0])
        rv=ema([float(rs[i]) for i in range(s,len(rs))],9)
        re[s:s+len(rv)]=rv[:len(rs)-s]
    em14=ema(cl,14); ml,sl,_=macd_calc(cl,12,26,9)
    n=len(c1h); buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool)
    pb=0; ps=0; lb=-999; ls=-999
    for i in range(1,n):
        if any(np.isnan(x) for x in [rs[i],re[i],em14[i],ml[i],sl[i]]): continue
        rv_=float(rs[i]); re_=float(re[i])
        ho_=float(ho[i]); hc_=float(hc[i]); em_=float(em14[i])
        ml_=float(ml[i]); sl_=float(sl[i])
        bull=int(rv_>50)+int(rv_>re_)+int(hc_>ho_)+int(hc_>em_)+int(ml_>sl_)
        bear=int(rv_<50)+int(rv_<re_)+int(hc_<ho_)+int(hc_<em_)+int(ml_<sl_)
        if bull>=threshold:
            if pb<threshold or (i-lb)>=reentry: buy_s[i]=True; lb=i
        if bear>=threshold:
            if ps<threshold or (i-ls)>=reentry: sell_s[i]=True; ls=i
        pb=bull; ps=bear
    atr14=atr_calc(hac,14)
    return buy_s,sell_s,atr14,hc,np.array([c.high for c in hac]),np.array([c.low for c in hac])

def sig_utbot(c1h, ut_mult=0.30, ut_len=14):
    hac,_,hc,hh,hl=ha(c1h)
    ut_atr=atr_calc(hac,ut_len); atr14=atr_calc(hac,14)
    n=len(c1h); tsl=np.full(n,np.nan)
    for i in range(n):
        sv=float(ut_atr[i])*ut_mult if not np.isnan(ut_atr[i]) else 0
        sc=float(hc[i])
        if i==0 or np.isnan(tsl[i-1]): tsl[i]=sc+sv; continue
        tp=float(tsl[i-1]); sp=float(hc[i-1])
        if sc>tp and sp>tp:    tsl[i]=max(tp,sc-sv)
        elif sc<tp and sp<tp:  tsl[i]=min(tp,sc+sv)
        elif sc>tp:            tsl[i]=sc-sv
        else:                  tsl[i]=sc+sv
    buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool); pd=0
    for i in range(1,n):
        if np.isnan(tsl[i]) or np.isnan(tsl[i-1]): continue
        bu=float(hc[i-1])<float(tsl[i-1]) and float(hc[i])>float(tsl[i])
        se=float(hc[i-1])>float(tsl[i-1]) and float(hc[i])<float(tsl[i])
        if bu and pd!=1: buy_s[i]=True; pd=1
        elif se and pd!=-1: sell_s[i]=True; pd=-1
        elif not bu and not se: pd=0
    return buy_s,sell_s,atr14,hc,np.array([c.high for c in hac]),np.array([c.low for c in hac])

def sig_macd_ema(c1h):
    hac,ho,hc,hh,hl=ha(c1h)
    cl=hc.tolist(); hm9=hma(cl,9)
    ml,sl,_=macd_calc(cl,5,13,4); atr14=atr_calc(hac,14)
    n=len(c1h); buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool); pd=0
    for i in range(1,n):
        if any(np.isnan(x) for x in [hm9[i],ml[i],ml[i-1],sl[i],sl[i-1]]): continue
        hc_=float(ml[i])-float(sl[i]); hp_=float(ml[i-1])-float(sl[i-1])
        d=0
        if hc_>0 and hp_<=0 and float(hc[i])>float(hm9[i]): d=1
        elif hc_<0 and hp_>=0: d=-1
        if d==1 and pd!=1: buy_s[i]=True; pd=1
        elif d==-1 and pd!=-1: sell_s[i]=True; pd=-1
        elif d==0: pd=0
    return buy_s,sell_s,atr14,hc,np.array([c.high for c in hac]),np.array([c.low for c in hac])

# ══════════════════════════════════════════════════════════════════════════════
# MTF FILTER — apply 4H bias + 15M confirm to 1H signals
# ══════════════════════════════════════════════════════════════════════════════

def apply_mtf(buy_s, sell_s, c1h,
              bias_4h, c4h, ts4h_map,
              conf_15m, c15m, ts15m_map,
              require_4h=True, require_15m=True):
    n = len(c1h)
    new_buy  = np.zeros(n, bool)
    new_sell = np.zeros(n, bool)
    for i in range(n):
        ts = c1h[i].ts
        # 4H bias
        if require_4h:
            idx4 = find_tf_idx(ts, c4h, ts4h_map, 240)
            b4 = int(bias_4h[idx4]) if idx4 is not None and idx4 < len(bias_4h) else 0
        else:
            b4 = 0  # neutral (pass-through)
        # 15M confirm
        if require_15m:
            idx15 = find_tf_idx(ts, c15m, ts15m_map, 15)
            c15 = int(conf_15m[idx15]) if idx15 is not None and idx15 < len(conf_15m) else 0
        else:
            c15 = 0
        if buy_s[i]:
            ok4 = (not require_4h) or (b4 >= 0)   # 4H bull or neutral
            ok15 = (not require_15m) or (c15 >= 0)  # 15M bull or neutral
            if ok4 and ok15:
                new_buy[i] = True
        if sell_s[i]:
            ok4 = (not require_4h) or (b4 <= 0)   # 4H bear or neutral
            ok15 = (not require_15m) or (c15 <= 0) # 15M bear or neutral
            if ok4 and ok15:
                new_sell[i] = True
    return new_buy, new_sell

def apply_mtf_strict(buy_s, sell_s, c1h,
                     bias_4h, c4h, ts4h_map,
                     conf_15m, c15m, ts15m_map):
    """Strict: BUY only when 4H=bull AND 15M=bull. SELL only when 4H=bear AND 15M=bear."""
    n = len(c1h)
    new_buy  = np.zeros(n, bool)
    new_sell = np.zeros(n, bool)
    for i in range(n):
        ts = c1h[i].ts
        idx4 = find_tf_idx(ts, c4h, ts4h_map, 240)
        b4 = int(bias_4h[idx4]) if idx4 is not None and idx4 < len(bias_4h) else 0
        idx15 = find_tf_idx(ts, c15m, ts15m_map, 15)
        c15 = int(conf_15m[idx15]) if idx15 is not None and idx15 < len(conf_15m) else 0
        if buy_s[i]  and b4 == 1  and c15 == 1:  new_buy[i]  = True
        if sell_s[i] and b4 == -1 and c15 == -1: new_sell[i] = True
    return new_buy, new_sell

# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH PER STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def fmt(wins,losses,total_r,trades,months,risk=100):
    t=wins+losses; wr=wins/t*100 if t else 0
    pnl=total_r*risk; tmo=t/months if months else 0
    return t,wr,pnl,tmo

def print_result(tag, wins, losses, total_r, trades, months, best_tag=""):
    t,wr,pnl,tmo = fmt(wins,losses,total_r,trades,months)
    star = " *** WR≥67%" if wr>=67 and t>=5 else ""
    print(f"  {tag:<40} T={t:>3} WR={wr:>5.1f}% P&L=${pnl:>+7.0f} {tmo:>5.1f}/mo{star}")

def grid_wt(c1h, months, bias4, c4h, ts4map, conf15, c15m, ts15map):
    print("\n[WaveTrend] Grid search...")
    best = None
    param_grid = list(itertools.product(
        [2,4,8,10],        # n1
        [4,8,12,18,21],    # n2
        [1.5,2.0,2.5,3.0], # sl_mult
        [1.0,1.2,1.5,2.0], # rr_ratio
        ["none","loose","strict"],  # mtf_mode
    ))
    results = []
    for n1, n2, sl_m, rr, mtf_mode in param_grid:
        buy_s, sell_s, atr14, hc, hh, hl = sig_wt(c1h, n1=n1, n2=n2)
        if mtf_mode == "loose":
            buy_s, sell_s = apply_mtf(buy_s, sell_s, c1h, bias4,c4h,ts4map, conf15,c15m,ts15map)
        elif mtf_mode == "strict":
            buy_s, sell_s = apply_mtf_strict(buy_s, sell_s, c1h, bias4,c4h,ts4map, conf15,c15m,ts15map)
        w,l,r,tr = simulate(c1h, buy_s, sell_s, atr14, sl_m, rr,
                            ha_c=hc, ha_h=hh, ha_l=hl)
        t=w+l; wr=w/t*100 if t else 0
        results.append((wr, t, r, w, l, tr, n1,n2,sl_m,rr,mtf_mode))
    # Sort by WR desc, then trades desc
    results.sort(key=lambda x: (-x[0], -x[1]))
    print(f"  {'Params':<38} {'T':>3} {'WR%':>6} {'P&L$':>8} {'t/mo':>6}")
    shown = set()
    for wr,t,r,w,l,tr,n1,n2,sl_m,rr,mtf in results[:30]:
        if t < 3: continue
        tag = f"n1={n1} n2={n2} sl={sl_m} rr={rr} [{mtf}]"
        pnl=r*100; tmo=t/months
        star=" ***" if wr>=67 else ""
        print(f"  {tag:<38} {t:>3} {wr:>5.1f}% ${pnl:>+7.0f} {tmo:>5.1f}/mo{star}")
    # Best ≥67% WR
    target = [(x) for x in results if x[0]>=67 and x[1]>=3]
    if target:
        wr,t,r,w,l,tr,n1,n2,sl_m,rr,mtf = target[0]
        print(f"\n  >>> BEST WR≥67%: n1={n1} n2={n2} sl={sl_m} rr={rr} MTF={mtf}")
        print(f"      T={t} WR={wr:.1f}% P&L=${r*100:+.0f} {t/months:.1f}/mo")
        return dict(strategy="WaveTrend", n1=n1, n2=n2, sl_m=sl_m, rr=rr, mtf=mtf,
                    wr=wr, trades=t, tmo=t/months, pnl=r*100)
    else:
        # Best overall
        wr,t,r,w,l,tr,n1,n2,sl_m,rr,mtf = results[0]
        print(f"\n  >>> BEST (no WR≥67%): n1={n1} n2={n2} sl={sl_m} rr={rr} MTF={mtf}")
        print(f"      T={t} WR={wr:.1f}% P&L=${r*100:+.0f} {t/months:.1f}/mo")
        return dict(strategy="WaveTrend", n1=n1, n2=n2, sl_m=sl_m, rr=rr, mtf=mtf,
                    wr=wr, trades=t, tmo=t/months, pnl=r*100)

def grid_momentum(c1h, months, bias4, c4h, ts4map, conf15, c15m, ts15map):
    print("\n[Momentum Score] Grid search...")
    results = []
    for thr, re, sl_m, rr, mtf_mode in itertools.product(
        [2,3,4],            # threshold
        [5,8,10,15,20],     # reentry_bars
        [1.5,2.0,2.5,3.0],  # sl_mult
        [1.0,1.2,1.5,2.0],  # rr
        ["none","loose","strict"],
    ):
        buy_s, sell_s, atr14, hc, hh, hl = sig_momentum(c1h, threshold=thr, reentry=re)
        if mtf_mode == "loose":
            buy_s, sell_s = apply_mtf(buy_s, sell_s, c1h, bias4,c4h,ts4map, conf15,c15m,ts15map)
        elif mtf_mode == "strict":
            buy_s, sell_s = apply_mtf_strict(buy_s, sell_s, c1h, bias4,c4h,ts4map, conf15,c15m,ts15map)
        w,l,r,tr = simulate(c1h, buy_s, sell_s, atr14, sl_m, rr, ha_c=hc, ha_h=hh, ha_l=hl)
        t=w+l; wr=w/t*100 if t else 0
        results.append((wr,t,r,w,l,tr,thr,re,sl_m,rr,mtf_mode))
    results.sort(key=lambda x: (-x[0],-x[1]))
    print(f"  {'Params':<42} {'T':>3} {'WR%':>6} {'P&L$':>8} {'t/mo':>6}")
    for wr,t,r,w,l,tr,thr,re,sl_m,rr,mtf in results[:20]:
        if t < 3: continue
        tag=f"thr={thr} re={re} sl={sl_m} rr={rr} [{mtf}]"
        star=" ***" if wr>=67 else ""
        print(f"  {tag:<42} {t:>3} {wr:>5.1f}% ${r*100:>+7.0f} {t/months:>5.1f}/mo{star}")
    target=[x for x in results if x[0]>=67 and x[1]>=3]
    if target:
        wr,t,r,w,l,tr,thr,re,sl_m,rr,mtf=target[0]
        print(f"\n  >>> BEST WR≥67%: thr={thr} re={re} sl={sl_m} rr={rr} MTF={mtf}")
        print(f"      T={t} WR={wr:.1f}% P&L=${r*100:+.0f} {t/months:.1f}/mo")
        return dict(strategy="Momentum",thr=thr,re=re,sl_m=sl_m,rr=rr,mtf=mtf,wr=wr,trades=t,tmo=t/months,pnl=r*100)
    wr,t,r,w,l,tr,thr,re,sl_m,rr,mtf=results[0]
    print(f"\n  >>> BEST: thr={thr} re={re} sl={sl_m} rr={rr} MTF={mtf} T={t} WR={wr:.1f}%")
    return dict(strategy="Momentum",thr=thr,re=re,sl_m=sl_m,rr=rr,mtf=mtf,wr=wr,trades=t,tmo=t/months,pnl=r*100)

def grid_utbot(c1h, months, bias4, c4h, ts4map, conf15, c15m, ts15map):
    print("\n[UT Bot] Grid search...")
    results=[]
    for um, ul, sl_m, rr, mtf_mode in itertools.product(
        [0.20,0.30,0.40,0.60],  # ut_mult
        [10,14,20],              # ut_len
        [1.5,2.0,2.5,3.0],
        [1.0,1.2,1.5,2.0],
        ["none","loose","strict"],
    ):
        buy_s,sell_s,atr14,hc,hh,hl=sig_utbot(c1h,ut_mult=um,ut_len=ul)
        if mtf_mode=="loose":  buy_s,sell_s=apply_mtf(buy_s,sell_s,c1h,bias4,c4h,ts4map,conf15,c15m,ts15map)
        elif mtf_mode=="strict": buy_s,sell_s=apply_mtf_strict(buy_s,sell_s,c1h,bias4,c4h,ts4map,conf15,c15m,ts15map)
        w,l,r,tr=simulate(c1h,buy_s,sell_s,atr14,sl_m,rr,ha_c=hc,ha_h=hh,ha_l=hl)
        t=w+l; wr=w/t*100 if t else 0
        results.append((wr,t,r,w,l,tr,um,ul,sl_m,rr,mtf_mode))
    results.sort(key=lambda x:(-x[0],-x[1]))
    print(f"  {'Params':<42} {'T':>3} {'WR%':>6} {'P&L$':>8} {'t/mo':>6}")
    for wr,t,r,w,l,tr,um,ul,sl_m,rr,mtf in results[:20]:
        if t<3: continue
        tag=f"mult={um} len={ul} sl={sl_m} rr={rr} [{mtf}]"
        star=" ***" if wr>=67 else ""
        print(f"  {tag:<42} {t:>3} {wr:>5.1f}% ${r*100:>+7.0f} {t/months:>5.1f}/mo{star}")
    target=[x for x in results if x[0]>=67 and x[1]>=3]
    if target:
        wr,t,r,w,l,tr,um,ul,sl_m,rr,mtf=target[0]
        print(f"\n  >>> BEST WR≥67%: mult={um} len={ul} sl={sl_m} rr={rr} MTF={mtf}")
        print(f"      T={t} WR={wr:.1f}% P&L=${r*100:+.0f} {t/months:.1f}/mo")
        return dict(strategy="UTBot",mult=um,length=ul,sl_m=sl_m,rr=rr,mtf=mtf,wr=wr,trades=t,tmo=t/months,pnl=r*100)
    wr,t,r,w,l,tr,um,ul,sl_m,rr,mtf=results[0]
    print(f"\n  >>> BEST: mult={um} len={ul} sl={sl_m} rr={rr} MTF={mtf} T={t} WR={wr:.1f}%")
    return dict(strategy="UTBot",mult=um,length=ul,sl_m=sl_m,rr=rr,mtf=mtf,wr=wr,trades=t,tmo=t/months,pnl=r*100)

def grid_macd(c1h, months, bias4, c4h, ts4map, conf15, c15m, ts15map):
    print("\n[MACD/EMA] Grid search...")
    results=[]
    for sl_m, rr, mtf_mode in itertools.product(
        [1.5,2.0,2.5,3.0],
        [1.0,1.2,1.5,2.0],
        ["none","loose","strict"],
    ):
        buy_s,sell_s,atr14,hc,hh,hl=sig_macd_ema(c1h)
        if mtf_mode=="loose":   buy_s,sell_s=apply_mtf(buy_s,sell_s,c1h,bias4,c4h,ts4map,conf15,c15m,ts15map)
        elif mtf_mode=="strict": buy_s,sell_s=apply_mtf_strict(buy_s,sell_s,c1h,bias4,c4h,ts4map,conf15,c15m,ts15map)
        w,l,r,tr=simulate(c1h,buy_s,sell_s,atr14,sl_m,rr,ha_c=hc,ha_h=hh,ha_l=hl)
        t=w+l; wr=w/t*100 if t else 0
        results.append((wr,t,r,w,l,tr,sl_m,rr,mtf_mode))
    results.sort(key=lambda x:(-x[0],-x[1]))
    print(f"  {'Params':<38} {'T':>3} {'WR%':>6} {'P&L$':>8} {'t/mo':>6}")
    for wr,t,r,w,l,tr,sl_m,rr,mtf in results:
        if t<3: continue
        tag=f"sl={sl_m} rr={rr} [{mtf}]"
        star=" ***" if wr>=67 else ""
        print(f"  {tag:<38} {t:>3} {wr:>5.1f}% ${r*100:>+7.0f} {t/months:>5.1f}/mo{star}")
    target=[x for x in results if x[0]>=67 and x[1]>=3]
    if target:
        wr,t,r,w,l,tr,sl_m,rr,mtf=target[0]
        print(f"\n  >>> BEST WR≥67%: sl={sl_m} rr={rr} MTF={mtf} T={t} WR={wr:.1f}%")
        return dict(strategy="MACD/EMA",sl_m=sl_m,rr=rr,mtf=mtf,wr=wr,trades=t,tmo=t/months,pnl=r*100)
    wr,t,r,w,l,tr,sl_m,rr,mtf=results[0]
    print(f"\n  >>> BEST: sl={sl_m} rr={rr} MTF={mtf} T={t} WR={wr:.1f}%")
    return dict(strategy="MACD/EMA",sl_m=sl_m,rr=rr,mtf=mtf,wr=wr,trades=t,tmo=t/months,pnl=r*100)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading candles...")
    c1h  = load_tf("1h")
    c4h  = load_tf("4h")
    c15m = load_tf("15m")

    n1h = len(c1h); n4h = len(c4h); n15 = len(c15m)
    t0 = ts_str(c1h[0].ts); t1 = ts_str(c1h[-1].ts)
    days = (c1h[-1].ts - c1h[0].ts) / (1000*3600*24)
    months = days / 30.44

    print(f"\n{'='*65}")
    print(f"  Real BTC/USDT — MTF Backtest Optimizer")
    print(f"  Period : {t0} → {t1}  ({days:.0f} days = {months:.1f} months)")
    print(f"  Bars   : 1H={n1h}  4H={n4h}  15M={n15}")
    p_lo=min(c.low for c in c1h); p_hi=max(c.high for c in c1h)
    print(f"  Price  : ${p_lo:,.0f} – ${p_hi:,.0f}")
    print(f"{'='*65}")

    # Pre-compute MTF indicators
    print("\nBuilding MTF indicators...")
    bias_4h  = build_4h_bias(c4h)
    conf_15m = build_15m_confirm(c15m)
    ts4h_map  = build_ts_index(c4h)
    ts15m_map = build_ts_index(c15m)

    bull4 = int(np.sum(bias_4h==1)); bear4 = int(np.sum(bias_4h==-1)); neut4 = int(np.sum(bias_4h==0))
    bull15=int(np.sum(conf_15m==1)); bear15=int(np.sum(conf_15m==-1)); neut15=int(np.sum(conf_15m==0))
    print(f"  4H bias  : bull={bull4} bear={bear4} neutral={neut4}  ({bull4/n4h*100:.0f}% bull)")
    print(f"  15M conf : bull={bull15} bear={bear15} neutral={neut15}  ({bull15/n15*100:.0f}% bull)")

    # ── Run grids ───────────────────────────────────────────────────
    summaries = []
    summaries.append(grid_wt(c1h, months, bias_4h, c4h, ts4h_map, conf_15m, c15m, ts15m_map))
    summaries.append(grid_momentum(c1h, months, bias_4h, c4h, ts4h_map, conf_15m, c15m, ts15m_map))
    summaries.append(grid_utbot(c1h, months, bias_4h, c4h, ts4h_map, conf_15m, c15m, ts15m_map))
    summaries.append(grid_macd(c1h, months, bias_4h, c4h, ts4h_map, conf_15m, c15m, ts15m_map))

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY  (risk $100/trade)")
    print(f"{'='*65}")
    print(f"  {'Strategy':<14} {'WR%':>6} {'Trades':>7} {'t/mo':>6} {'P&L $':>9} {'MTF':>8}")
    print(f"  {'-'*60}")
    total_pnl=0
    for s in summaries:
        star=" *" if s['wr']>=67 else ""
        print(f"  {s['strategy']:<14} {s['wr']:>5.1f}% {s['trades']:>7} {s['tmo']:>5.1f}/mo {s['pnl']:>+8.0f}$ {s.get('mtf','none'):>8}{star}")
        total_pnl+=s['pnl']
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<14} {'':>6} {'':>7} {'':>6} {total_pnl:>+8.0f}$")
    print(f"{'='*65}")
    print(f"  * = WR ≥ 67%  (target met)")
    print(f"\n  MTF modes: none=no filter | loose=4H+15M allow neutral | strict=must be bull/bear")

if __name__ == "__main__":
    main()
