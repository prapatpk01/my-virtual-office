"""
MTF Long-Only Backtest — Real Binance Data (Jan–May 2026)
Focus: BUY-only signals with strict MTF + EMA200 macro filter
Goal: WR ≥ 67%
"""
import csv, glob, math, itertools, os
import numpy as np
from datetime import datetime, timezone

BASE = "/tmp/chart_data"

class C:
    __slots__ = ("ts","open","high","low","close","volume")
    def __init__(self, ts, o, h, l, c, v):
        self.ts=int(ts); self.open=float(o); self.high=float(h)
        self.low=float(l); self.close=float(c); self.volume=float(v)

def load_tf(tf):
    files = sorted(glob.glob(os.path.join(BASE, tf, tf, "*.csv")))
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                p = line.strip().split(",")
                if len(p) < 6: continue
                rows.append(C(int(p[0])//1000, p[1], p[2], p[3], p[4], p[5]))
    rows.sort(key=lambda x: x.ts)
    return rows

def ts_str(ts_ms): return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d")

# ── Indicators ─────────────────────────────────────────────────────────────

def ema(v, p):
    a=np.array(v, dtype=float); r=np.full(len(a), np.nan)
    if len(a)<p: return r
    k=2/(p+1); r[p-1]=a[:p].mean()
    for i in range(p,len(a)): r[i]=a[i]*k+r[i-1]*(1-k)
    return r

def sma(v, p):
    a=np.array(v, dtype=float); r=np.full(len(a), np.nan)
    for i in range(p-1, len(a)): r[i]=a[i-p+1:i+1].mean()
    return r

def wma(v, p):
    a=np.array(v,dtype=float); r=np.full(len(a),np.nan)
    w=np.arange(1,p+1,dtype=float); d=w.sum()
    for i in range(p-1,len(a)): r[i]=np.dot(a[i-p+1:i+1],w)/d
    return r

def hma(v, p):
    half=max(2,p//2); sq=max(2,int(round(math.sqrt(p))))
    raw=2*wma(v,half)-wma(v,p)
    val=[float(x) for x in raw if not np.isnan(x)]
    if len(val)<sq: return np.full(len(v),np.nan)
    h=wma(val,sq); pad=len(v)-len(h)
    return np.concatenate([np.full(pad,np.nan),h])

def atr_calc(candles, p=14):
    n=len(candles); tr=np.full(n,np.nan)
    for i in range(1,n):
        h=candles[i].high; l=candles[i].low; pc=candles[i-1].close
        tr[i]=max(h-l,abs(h-pc),abs(l-pc))
    r=np.full(n,np.nan)
    if n>p:
        r[p]=float(np.nanmean(tr[1:p+1]))
        for i in range(p+1,n): r[i]=(r[i-1]*(p-1)+tr[i])/p
    return r

def macd_calc(v, fast=12, slow=26, sig=9):
    fe=ema(v,fast); se=ema(v,slow); ml=fe-se
    vals=[x for x in ml if not np.isnan(x)]
    if len(vals)<sig: sl=np.full(len(ml),np.nan)
    else:
        s=ema(vals,sig); pad=len(ml)-len(s)
        sl=np.concatenate([np.full(pad,np.nan),s])
    return ml,sl,ml-sl

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
    n=len(candles); ho=np.zeros(n); hc=np.zeros(n); hh=np.zeros(n); hl=np.zeros(n)
    for i,c in enumerate(candles):
        hc[i]=(c.open+c.high+c.low+c.close)/4
        ho[i]=((c.open+c.close)/2 if i==0 else (ho[i-1]+hc[i-1])/2)
        hh[i]=max(c.high,ho[i],hc[i]); hl[i]=min(c.low,ho[i],hc[i])
    class _C:
        __slots__=("ts","open","high","low","close","volume")
        def __init__(s,i):
            s.ts=candles[i].ts; s.open=ho[i]; s.high=hh[i]
            s.low=hl[i]; s.close=hc[i]; s.volume=candles[i].volume
    return [_C(i) for i in range(n)],ho,hc,hh,hl

# ── 4H bias: EMA20 + EMA200 macro ─────────────────────────────────────────

def build_4h_bias(c4h):
    _, _, hc4, hh4, hl4 = ha(c4h)
    em20=ema(hc4.tolist(),20); em200=ema(hc4.tolist(),200)
    n=len(c4h); bias=np.zeros(n,dtype=int)
    for i in range(n):
        if np.isnan(em20[i]): continue
        macro_bull = True if np.isnan(em200[i]) else float(hc4[i])>float(em200[i])
        bull = float(hc4[i])>float(em20[i]) and macro_bull
        bear = float(hc4[i])<float(em20[i]) and not macro_bull
        if bull: bias[i]=1
        elif bear: bias[i]=-1
    return bias

def build_15m_confirm(c15):
    _,ho15,hc15,_,_=ha(c15)
    rs=rsi_w(hc15,14)
    valid=np.where(~np.isnan(rs))[0]
    rs_ema=np.full(len(rs),np.nan)
    if len(valid)>=9:
        s=int(valid[0])
        re=ema([float(rs[i]) for i in range(s,len(rs))],9)
        rs_ema[s:s+len(re)]=re[:len(rs)-s]
    n=len(c15); conf=np.zeros(n,dtype=int)
    for i in range(1,n):
        if np.isnan(rs[i]) or np.isnan(rs_ema[i]): continue
        bull_rsi=float(rs[i])>50 and float(rs[i])>float(rs_ema[i])
        bull_ha=float(hc15[i])>float(ho15[i])
        bear_rsi=float(rs[i])<50 and float(rs[i])<float(rs_ema[i])
        bear_ha=float(hc15[i])<float(ho15[i])
        if bull_rsi and bull_ha: conf[i]=1
        elif bear_rsi and bear_ha: conf[i]=-1
    return conf

def build_ts_index(candles): return {c.ts:i for i,c in enumerate(candles)}

def find_tf_idx(ts_1h, tf_candles, ts_map, tf_min):
    tf_ms=tf_min*60*1000; al=(ts_1h//tf_ms)*tf_ms
    if al in ts_map: return ts_map[al]
    for d in range(0,tf_min*3+1,tf_min):
        k=al-d*60*1000
        if k in ts_map: return ts_map[k]
    return None

# ── Simulator (long-only) ──────────────────────────────────────────────────

def sim_long(c1h, buy_s, atr14, sl_m, rr, ha_c=None, ha_h=None, ha_l=None):
    LF=60
    cls=np.array([c.close for c in c1h])
    hig=np.array([c.high for c in c1h])
    low=np.array([c.low for c in c1h])
    ec=ha_c if ha_c is not None else cls
    eh=ha_h if ha_h is not None else hig
    el=ha_l if ha_l is not None else low
    n=len(c1h); wins=losses=0; total_r=0.0; trades=[]
    for i in range(n):
        if not buy_s[i]: continue
        av=float(atr14[i]) if not np.isnan(atr14[i]) else 0
        if av<=0: continue
        ep=float(ec[i])
        sl_p=ep-sl_m*av; tp_p=ep+sl_m*rr*av
        out=0; xp=ep
        for j in range(i+1,min(i+LF,n)):
            if el[j]<=sl_p: out=-1; xp=sl_p; break
            if eh[j]>=tp_p: out=1;  xp=tp_p; break
        if out==1:  wins+=1;   total_r+=rr
        elif out==-1: losses+=1; total_r-=1.0
        if out!=0: trades.append((i,1,out,ep,xp,c1h[i].ts))
    return wins,losses,total_r,trades

# ── MTF gate (strict: 4H bull AND 15M bull) ──────────────────────────────

def mtf_buy(buy_s, c1h, bias4, c4h, ts4map, conf15, c15m, ts15map, mode="strict"):
    n=len(c1h); new_buy=np.zeros(n,bool)
    for i in range(n):
        if not buy_s[i]: continue
        ts=c1h[i].ts
        idx4=find_tf_idx(ts,c4h,ts4map,240)
        b4=int(bias4[idx4]) if idx4 is not None and idx4<len(bias4) else 0
        idx15=find_tf_idx(ts,c15m,ts15map,15)
        c15=int(conf15[idx15]) if idx15 is not None and idx15<len(conf15) else 0
        if mode=="strict":
            if b4==1 and c15==1: new_buy[i]=True
        elif mode=="loose":
            if b4>=0 and c15>=0: new_buy[i]=True
        elif mode=="4h_only":
            if b4==1: new_buy[i]=True
        elif mode=="15m_only":
            if c15==1: new_buy[i]=True
    return new_buy

# ── Strategy signal builders (BUY only) ───────────────────────────────────

def sig_wt_buy(c1h, n1=8, n2=21):
    hac,_,hc,hh,hl=ha(c1h)
    hh_a=np.array([c.high for c in hac]); hl_a=np.array([c.low for c in hac])
    ap=(hh_a+hl_a+hc)/3; esa=ema(ap.tolist(),n1)
    dabs=np.abs(ap-esa); fo=np.where(~np.isnan(dabs))[0]
    df=dabs.copy()
    if len(fo): df[:fo[0]]=dabs[fo[0]]
    d_=ema(df.tolist(),n1)
    ci=np.where(d_>1e-10,(ap-esa)/(0.015*d_),0.)
    wt1=ema(ci.tolist(),n2); wt2=sma(wt1.tolist(),4)
    n=len(c1h); buy_s=np.zeros(n,bool); pd=0
    for i in range(1,n):
        if np.isnan(wt1[i]) or np.isnan(wt2[i]): continue
        cu=float(wt1[i-1])<=float(wt2[i-1]) and float(wt1[i])>float(wt2[i])
        if cu and pd!=1: buy_s[i]=True; pd=1
        elif not cu: pd=0
    return buy_s,atr_calc(hac,14),hc,hh_a,hl_a

def sig_mom_buy(c1h, threshold=3, reentry=20):
    hac,ho,hc,hh,hl=ha(c1h)
    cl=hc.tolist(); rs=rsi_w(hc,14)
    re=np.full(len(hc),np.nan); valid=np.where(~np.isnan(rs))[0]
    if len(valid)>=9:
        s=int(valid[0])
        rv=ema([float(rs[i]) for i in range(s,len(rs))],9)
        re[s:s+len(rv)]=rv[:len(rs)-s]
    em14=ema(cl,14); ml,sl,_=macd_calc(cl,12,26,9)
    n=len(c1h); buy_s=np.zeros(n,bool); pb=0; lb=-999
    for i in range(1,n):
        if any(np.isnan(x) for x in [rs[i],re[i],em14[i],ml[i],sl[i]]): continue
        rv_=float(rs[i]); re_=float(re[i]); ho_=float(ho[i]); hc_=float(hc[i])
        em_=float(em14[i]); ml_=float(ml[i]); sl_=float(sl[i])
        bull=int(rv_>50)+int(rv_>re_)+int(hc_>ho_)+int(hc_>em_)+int(ml_>sl_)
        if bull>=threshold:
            if pb<threshold or (i-lb)>=reentry: buy_s[i]=True; lb=i
        pb=bull
    return buy_s,atr_calc(hac,14),hc,np.array([c.high for c in hac]),np.array([c.low for c in hac])

def sig_utbot_buy(c1h, ut_mult=0.4, ut_len=20):
    hac,_,hc,hh,hl=ha(c1h)
    ut_atr=atr_calc(hac,ut_len); atr14=atr_calc(hac,14)
    n=len(c1h); tsl=np.full(n,np.nan)
    for i in range(n):
        sv=float(ut_atr[i])*ut_mult if not np.isnan(ut_atr[i]) else 0
        sc=float(hc[i])
        if i==0 or np.isnan(tsl[i-1]): tsl[i]=sc+sv; continue
        tp=float(tsl[i-1]); sp=float(hc[i-1])
        if sc>tp and sp>tp:   tsl[i]=max(tp,sc-sv)
        elif sc<tp and sp<tp: tsl[i]=min(tp,sc+sv)
        elif sc>tp:           tsl[i]=sc-sv
        else:                 tsl[i]=sc+sv
    buy_s=np.zeros(n,bool); pd=0
    for i in range(1,n):
        if np.isnan(tsl[i]) or np.isnan(tsl[i-1]): continue
        bu=float(hc[i-1])<float(tsl[i-1]) and float(hc[i])>float(tsl[i])
        if bu and pd!=1: buy_s[i]=True; pd=1
        elif not bu: pd=0
    return buy_s,atr14,hc,np.array([c.high for c in hac]),np.array([c.low for c in hac])

def sig_macd_buy(c1h):
    hac,ho,hc,hh,hl=ha(c1h)
    cl=hc.tolist(); hm9=hma(cl,9)
    ml,sl,_=macd_calc(cl,5,13,4); atr14=atr_calc(hac,14)
    n=len(c1h); buy_s=np.zeros(n,bool); pd=0
    for i in range(1,n):
        if any(np.isnan(x) for x in [hm9[i],ml[i],ml[i-1],sl[i],sl[i-1]]): continue
        hc_=float(ml[i])-float(sl[i]); hp_=float(ml[i-1])-float(sl[i-1])
        d=1 if hc_>0 and hp_<=0 and float(hc[i])>float(hm9[i]) else 0
        if d==1 and pd!=1: buy_s[i]=True; pd=1
        elif d==0: pd=0
    return buy_s,atr14,hc,np.array([c.high for c in hac]),np.array([c.low for c in hac])

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    c1h=load_tf("1h"); c4h=load_tf("4h"); c15m=load_tf("15m")
    days=(c1h[-1].ts-c1h[0].ts)/(1000*3600*24); months=days/30.44
    t0=ts_str(c1h[0].ts); t1=ts_str(c1h[-1].ts)

    print(f"\n{'='*68}")
    print(f"  BTC/USDT Long-Only MTF Backtest — {t0} → {t1} ({months:.1f} mo)")
    print(f"  Goal: WR ≥ 67% | Min trades: 8/mo per strategy")
    print(f"  Filter: 4H EMA20+EMA200 bull + 15M RSI bull")
    print(f"{'='*68}")

    print("Building MTF filters...")
    bias4   = build_4h_bias(c4h)
    conf15  = build_15m_confirm(c15m)
    ts4map  = build_ts_index(c4h)
    ts15map = build_ts_index(c15m)

    n4=len(c4h); n15=len(c15m)
    bull4=int(np.sum(bias4==1))
    bull15=int(np.sum(conf15==1))
    print(f"  4H bull bars : {bull4}/{n4} = {bull4/n4*100:.0f}%  (EMA20 + EMA200)")
    print(f"  15M bull bars: {bull15}/{n15} = {bull15/n15*100:.0f}%  (RSI+HA)")

    RISK = 100   # $ per 1R

    # ── Per-strategy grid + MTF mode comparison ────────────────────────────
    all_best = []

    strategies = [
        ("WaveTrend",  "wt"),
        ("Momentum",   "mom"),
        ("UT Bot",     "ut"),
        ("MACD/EMA",   "macd"),
    ]

    mtf_modes = ["none","4h_only","15m_only","loose","strict"]

    for name, key in strategies:
        print(f"\n{'─'*68}")
        print(f"  {name} — Long-Only Grid Search")
        print(f"{'─'*68}")
        print(f"  {'MTF Mode':<12} {'Params':<30} {'T':>4} {'WR%':>6} {'P&L':>8} {'t/mo':>6}")

        best_wr = 0; best_row = None

        # Param grids per strategy
        if key == "wt":
            param_sets = list(itertools.product([2,4,8,10],[4,8,12,18,21],[1.5,2.0,2.5],[1.0,1.2,1.5]))
            def get_sigs(ps): return sig_wt_buy(c1h, n1=ps[0], n2=ps[1])
            def label(ps): return f"n1={ps[0]} n2={ps[1]} sl={ps[2]} rr={ps[3]}"
        elif key == "mom":
            param_sets = list(itertools.product([2,3,4],[5,10,15,20],[1.5,2.0,2.5],[1.0,1.2,1.5]))
            def get_sigs(ps): return sig_mom_buy(c1h, threshold=ps[0], reentry=ps[1])
            def label(ps): return f"thr={ps[0]} re={ps[1]} sl={ps[2]} rr={ps[3]}"
        elif key == "ut":
            param_sets = list(itertools.product([0.2,0.3,0.4,0.6],[10,14,20],[1.5,2.0,2.5],[1.0,1.2,1.5]))
            def get_sigs(ps): return sig_utbot_buy(c1h, ut_mult=ps[0], ut_len=ps[1])
            def label(ps): return f"mult={ps[0]} len={ps[1]} sl={ps[2]} rr={ps[3]}"
        else:  # macd
            param_sets = [(None, None, sl, rr) for sl,rr in itertools.product([1.5,2.0,2.5],[1.0,1.2,1.5])]
            def get_sigs(ps): return sig_macd_buy(c1h)
            def label(ps): return f"sl={ps[2]} rr={ps[3]}"

        results = []
        for ps in param_sets:
            sigs_out = get_sigs(ps)
            buy_raw = sigs_out[0]; atr14=sigs_out[1]; hc=sigs_out[2]; hh=sigs_out[3]; hl=sigs_out[4]
            sl_m=ps[2]; rr=ps[3]
            for mode in mtf_modes:
                if mode == "none":
                    buy_f = buy_raw
                else:
                    buy_f = mtf_buy(buy_raw, c1h, bias4, c4h, ts4map, conf15, c15m, ts15map, mode=mode)
                w,l,r,tr = sim_long(c1h, buy_f, atr14, sl_m, rr, ha_c=hc, ha_h=hh, ha_l=hl)
                t=w+l; wr=w/t*100 if t else 0; tmo=t/months
                results.append((wr,t,r,w,l,ps,mode))

        results.sort(key=lambda x:(-x[0],-x[1]))
        printed=0
        for wr,t,r,w,l,ps,mode in results:
            if t<5 or printed>=25: continue
            star=" ***" if wr>=67 else ""
            print(f"  {mode:<12} {label(ps):<30} {t:>4} {wr:>5.1f}% ${r*RISK:>+7.0f} {t/months:>5.1f}/mo{star}")
            printed+=1
            if wr>=67 and best_row is None:
                best_row=(wr,t,r,w,l,ps,mode)
            if wr>best_wr: best_wr=wr

        if best_row:
            wr,t,r,w,l,ps,mode=best_row
        else:
            wr,t,r,w,l,ps,mode=results[0]
        print(f"\n  >>> BEST for {name}: {label(ps)} MTF={mode}  WR={wr:.1f}%  T={t}  P&L=${r*RISK:+.0f}  {t/months:.1f}/mo")
        all_best.append((name, wr, t, r, ps, mode))

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  FINAL SUMMARY — Long-Only + MTF  (Risk ${RISK}/trade)")
    print(f"{'='*68}")
    print(f"  {'Strategy':<14} {'WR%':>6} {'T':>5} {'t/mo':>6} {'P&L$':>9} {'MTF':<10} {'Target'}")
    print(f"  {'-'*65}")
    total=0.0
    for name,wr,t,r,ps,mode in all_best:
        tmo=t/months; pnl=r*RISK
        met="WR ✓" if wr>=67 else f"WR {wr:.1f}%"
        vol="T ✓" if tmo>=8 else f"T {tmo:.1f}/mo"
        total+=pnl
        print(f"  {name:<14} {wr:>5.1f}% {t:>5} {tmo:>5.1f}/mo {pnl:>+8.0f}$ {mode:<10} {met} | {vol}")
    print(f"  {'-'*65}")
    print(f"  {'TOTAL':<14} {'':>6} {'':>5} {'':>6} {total:>+8.0f}$")
    print(f"{'='*68}")

    # ── Recommended config for live bot ────────────────────────────────────
    print(f"""
RECOMMENDED .env SETTINGS (Long-Only MTF):
─────────────────────────────────────────────────
  STRATEGY_WT_ADX=true
  STRATEGY_MACD_EMA=true
  STRATEGY_MOMENTUM_SCORE=true
  STRATEGY_UT_BOT=true
  STRATEGY_KNN=false          # needs more data

  # MTF filters applied inside bot.py (4H+15M)
  STOP_LOSS_PCT=0.015         # 1.5× ATR tight SL
  TAKE_PROFIT_PCT=0.015       # R:R 1:1 (count on high WR)
  MAX_POSITIONS=3
─────────────────────────────────────────────────
""")

if __name__ == "__main__":
    main()
