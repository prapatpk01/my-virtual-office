"""
Sentinel Strategy v1.0 — 750×3 GBM Backtest
Port of Pine Script v6 "Sentinel Strategy v1.0 — 1H"

Rules (single-strategy comparison):
  - $300 capital, $50/trade, long-only
  - TP=4.8%  SL=4.6%  MinConf=60%
  - Fee 0.25% RT
  - NOTE: MTF gate excluded (single-TF synthetic data)
    → mtf_sc fixed = 0 (neutral), removes mtf>0 condition from base_long
    → confidence contribution from MTF = 0 (no +15/+8 bonus)
"""
import math, random, sys
import numpy as np

POSITION   = 50.0
FEE_RT     = 0.0025
START_BAL  = 300.0

# ── Candle ───────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self,ts,o,h,l,c,v):
        self.timestamp=ts;self.open=o;self.high=h;self.low=l;self.close=c;self.volume=v

def gbm(n, seed=99, start=65_000.0, mu=0.0002, sigma=0.005):
    rng=random.Random(seed); candles=[]; price=start
    for i in range(n):
        ret=mu+sigma*rng.gauss(0,1); o=price; c=price*math.exp(ret)
        h=max(o,c)*(1+abs(rng.gauss(0,sigma*0.5)))
        l=min(o,c)*(1-abs(rng.gauss(0,sigma*0.5)))
        candles.append(C(i*3600,o,h,l,c,rng.uniform(1,10))); price=c
    return candles

# ── Indicator helpers ────────────────────────────────────────

def _ema(arr, p):
    n=len(arr); out=np.full(n,np.nan); k=2.0/(p+1)
    out[p-1]=np.mean(arr[:p])
    for i in range(p, n): out[i]=arr[i]*k+out[i-1]*(1-k)
    return out

def _ema_skipnan(arr, p):
    """EMA that skips leading NaN — same as Pine Script ta.ema on sparse series."""
    n=len(arr); out=np.full(n,np.nan); k=2.0/(p+1)
    valid=np.where(~np.isnan(arr))[0]
    if len(valid)==0 or n-valid[0]<p: return out
    s=int(valid[0])
    out[s+p-1]=np.nanmean(arr[s:s+p])
    for i in range(s+p,n):
        if not np.isnan(arr[i]): out[i]=arr[i]*k+out[i-1]*(1-k)
        else: out[i]=out[i-1]
    return out

def _sma(arr, p):
    n=len(arr); out=np.full(n,np.nan)
    for i in range(p-1,n): out[i]=np.mean(arr[i-p+1:i+1])
    return out

def _wma(arr, p):
    n=len(arr); out=np.full(n,np.nan); w=np.arange(1,p+1,dtype=float)
    for i in range(p-1,n): out[i]=np.dot(arr[i-p+1:i+1],w)/w.sum()
    return out

def _hma(arr, p):
    half=round(p/2); sp=round(math.sqrt(p))
    w1=_wma(arr,half); w2=_wma(arr,p)
    diff=np.full(len(arr),np.nan)
    ok=~np.isnan(w1)&~np.isnan(w2)
    diff[ok]=2*w1[ok]-w2[ok]
    return _wma(diff,sp)

def _rsi(arr, p=14):
    n=len(arr); out=np.full(n,np.nan)
    if n<p+1: return out
    d=np.diff(arr); g=np.where(d>0,d,0.); lo=np.where(d<0,-d,0.)
    ag=np.mean(g[:p]); al=np.mean(lo[:p])
    out[p]=100. if al<1e-10 else 100-100/(1+ag/al)
    for i in range(p+1,n):
        ag=(ag*(p-1)+g[i-1])/p; al=(al*(p-1)+lo[i-1])/p
        out[i]=100. if al<1e-10 else 100-100/(1+ag/al)
    return out

def _macd(arr, fast=12, slow=26, sig=9):
    ef=_ema(arr,fast); es=_ema(arr,slow)
    ml=ef-es; sl=_ema_skipnan(ml,sig); return ml,sl,ml-sl

def _dmi(highs, lows, closes, p=14):
    n=len(closes); tr=np.full(n,np.nan)
    dmp=np.full(n,np.nan); dmm=np.full(n,np.nan)
    for i in range(1,n):
        tr[i]=max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        hd=highs[i]-highs[i-1]; ld=lows[i-1]-lows[i]
        dmp[i]=hd if hd>ld and hd>0 else 0.
        dmm[i]=ld if ld>hd and ld>0 else 0.
    atr=np.full(n,np.nan); dip=np.full(n,np.nan); dim=np.full(n,np.nan); adx=np.full(n,np.nan)
    if n<=p: return adx,dip,dim
    atr[p]=np.nanmean(tr[1:p+1])
    sdp=np.nanmean(dmp[1:p+1]); sdm=np.nanmean(dmm[1:p+1])
    for i in range(p+1,n):
        atr[i]=(atr[i-1]*(p-1)+tr[i])/p
        sdp=(sdp*(p-1)+dmp[i])/p; sdm=(sdm*(p-1)+dmm[i])/p
        if atr[i]>0: dip[i]=sdp/atr[i]*100; dim[i]=sdm/atr[i]*100
    dx=np.full(n,np.nan)
    for i in range(p,n):
        if not (np.isnan(dip[i]) or np.isnan(dim[i])):
            s=dip[i]+dim[i]; dx[i]=abs(dip[i]-dim[i])/s*100 if s>0 else 0.
    s2=p*2
    if s2>=n: return adx,dip,dim
    adx[s2]=np.nanmean(dx[p:s2+1])
    for i in range(s2+1,n):
        if not (np.isnan(dx[i]) or np.isnan(adx[i-1])):
            adx[i]=(adx[i-1]*(p-1)+dx[i])/p
    return adx,dip,dim

def _clamp(v,lo,hi): return max(lo,min(hi,v))

# ── Sentinel signal computation ──────────────────────────────

def compute_signals(candles):
    n   = len(candles)
    cls = np.array([c.close for c in candles],dtype=float)
    hig = np.array([c.high  for c in candles],dtype=float)
    low = np.array([c.low   for c in candles],dtype=float)

    ma1  = _ema(cls,12)
    ma2  = _sma(cls,26)
    ma3  = _ema(cls,125)
    ma4  = _sma(cls,200)
    hma_ = _hma(cls,20)
    adx_,dip,dim = _dmi(hig,low,cls,14)
    rsi_ = _rsi(cls,14)
    _,_,macd_h = _macd(cls,12,26,9)

    # Momentum arrays
    mom_sc = np.full(n,np.nan)
    for i in range(n):
        if not np.isnan(rsi_[i]):
            mom_sc[i] = _clamp(rsi_[i]*2.-100.,-100.,100.)

    diff3 = np.full(n,np.nan)
    for i in range(3,n):
        if not (np.isnan(mom_sc[i]) or np.isnan(mom_sc[i-3])):
            diff3[i] = mom_sc[i]-mom_sc[i-3]
    mom_ac = _ema_skipnan(diff3,5)

    diff2 = np.full(n,np.nan)
    for i in range(2,n):
        if not (np.isnan(mom_sc[i]) or np.isnan(mom_sc[i-2])):
            diff2[i] = mom_sc[i]-mom_sc[i-2]
    mom_sl = _ema_skipnan(diff2,3)

    buy_sig = np.zeros(n,dtype=bool)
    conf_arr= np.zeros(n,dtype=float)
    base_arr= np.zeros(n,dtype=bool)

    MIN_BAR = 200 + 30   # SMA200 warm-up + some buffer

    prev_base = False
    for i in range(MIN_BAR, n):
        if any(np.isnan(v) for v in [
            ma1[i],ma2[i],ma3[i],ma4[i],hma_[i],
            adx_[i],dip[i],dim[i],rsi_[i],macd_h[i],
            mom_sc[i],mom_ac[i],mom_sl[i]
        ]):
            prev_base = False
            continue

        c_i  = cls[i]
        hma_up = hma_[i] > hma_[i-1]

        # MA alignment score
        bull = sum([c_i>ma1[i],c_i>ma2[i],c_i>ma3[i],c_i>ma4[i],
                    ma1[i]>ma2[i],ma2[i]>ma3[i],ma3[i]>ma4[i]])
        bear = sum([c_i<ma1[i],c_i<ma2[i],c_i<ma3[i],c_i<ma4[i],
                    ma1[i]<ma2[i],ma2[i]<ma3[i],ma3[i]<ma4[i]])
        ma_al   = bull - bear
        adx_dir = (1. if dip[i]>dim[i] else -1.) * _clamp(adx_[i]/30.,0.,1.) * 40.
        dist_sc = _clamp((c_i-ma4[i])/ma4[i]*1000.,-40.,40.)
        tr_sc   = _clamp(ma_al/7.*40.+adx_dir+dist_sc,-100.,100.)
        ms_trend = tr_sc > 20.
        ms_bear  = tr_sc < -20.

        # Momentum engine
        msc = float(mom_sc[i]); mac = float(mom_ac[i]); msl = float(mom_sl[i])
        mst = abs(msc)
        mom_rising  = msl > 0.5
        mom_falling = msl < -0.5
        sgn = 1. if msc > 0 else -1.

        mom_build  = (25<=mst<55 and
                      ((msc>0 and mom_rising) or (msc<0 and mom_falling)) and
                      mac*sgn > 0.)
        mom_accel  = (mst>=55 and
                      ((msc>0 and mom_rising) or (msc<0 and mom_falling)) and
                      mac*sgn >= 0.)
        mom_peak   = mst>55 and mac*sgn < -0.3
        mom_fade   = (msc>40 and mom_falling and
                      not np.isnan(macd_h[i-1]) and float(macd_h[i])<float(macd_h[i-1]))

        # Confidence (MTF=0 so +15/+8 never trigger; no MTF gate in base_long)
        ec = 30.
        if ms_trend:             ec += 12.
        if hma_up:               ec += 8.
        if adx_[i] > 25:         ec += 10.
        elif adx_[i] > 18:       ec += 5.
        if mom_build or mom_accel: ec += 10.
        if mom_peak or mom_fade: ec -= 12.
        if ms_bear:              ec -= 10.
        if 50 <= rsi_[i] < 70:  ec += 5.
        econf = _clamp(ec, 10., 96.)

        # Base long (MTF gate excluded)
        base = (hma_up and ms_trend and
                not mom_peak and not mom_fade and
                c_i > hma_[i] and econf >= 60. and rsi_[i] < 75.)

        # Edge-trigger: fires only on transition False→True
        if base and not prev_base:
            buy_sig[i] = True
            conf_arr[i] = econf

        base_arr[i] = base
        prev_base = base

    return buy_sig, conf_arr, cls, hig, low

# ── Single-window simulation ──────────────────────────────────

def simulate_window(candles, tp=0.048, sl=0.046, seed=0):
    buy_sig, conf, cls, hig, low = compute_signals(candles)
    n = len(candles)
    LOOKFWD = 120

    trades = []
    in_pos  = False
    entry   = 0.

    for i in range(n):
        if in_pos:
            tp_p = entry*(1+tp); sl_p = entry*(1-sl)
            if low[i] <= sl_p:
                pnl = (sl_p-entry)/entry * POSITION * (1 - FEE_RT)
                trades.append({"pnl": pnl, "reason": "sl", "conf": 0.})
                in_pos = False
            elif hig[i] >= tp_p:
                pnl = (tp_p-entry)/entry * POSITION * (1 - FEE_RT)
                trades.append({"pnl": pnl, "reason": "tp", "conf": 0.})
                in_pos = False
            continue

        if buy_sig[i] and not in_pos:
            # Check lookforward timeout
            tp_p = cls[i]*(1+tp); sl_p = cls[i]*(1-sl)
            outcome = None
            for j in range(i+1, min(i+LOOKFWD+1, n)):
                if low[j] <= sl_p:  outcome=("sl", sl_p); break
                if hig[j] >= tp_p:  outcome=("tp", tp_p); break
            if outcome:
                reason, ep = outcome
                pnl = (ep-cls[i])/cls[i] * POSITION * (1-FEE_RT)
                trades.append({"pnl": pnl, "reason": reason, "conf": conf[i]})

    return trades

# ── Main ─────────────────────────────────────────────────────

def main():
    N = 750; WINDOWS = 3
    print("="*60)
    print("SENTINEL v1.0  —  750×3 GBM BACKTEST")
    print(f"Capital ${START_BAL} | $50/trade | Long-only | TF=1H")
    print("TP=4.8%  SL=4.6%  MinConf=60%  (MTF excluded)")
    print("="*60)

    all_trades = []
    window_rows = []

    for w in range(WINDOWS):
        candles = gbm(N, seed=42+w*100)
        trades  = simulate_window(candles)

        wins    = [t for t in trades if t["pnl"]>0]
        losses  = [t for t in trades if t["pnl"]<=0]
        tot_pnl = sum(t["pnl"] for t in trades)
        wr      = len(wins)/len(trades)*100 if trades else 0

        tp_hits = sum(1 for t in trades if t["reason"]=="tp")
        sl_hits = sum(1 for t in trades if t["reason"]=="sl")

        print(f"\nWindow {w+1}/3  ({N} bars ≈ 1 month 1H):")
        print(f"  Trades: {len(trades):3d}  WR: {wr:5.1f}%  "
              f"TP:{tp_hits}  SL:{sl_hits}  PnL: ${tot_pnl:+.2f}")
        window_rows.append((len(trades), wr, tot_pnl))
        all_trades.extend(trades)

    # ── Combined summary ─────────────────────────────────────
    total  = len(all_trades)
    wins   = [t for t in all_trades if t["pnl"]>0]
    losses = [t for t in all_trades if t["pnl"]<=0]
    pnl    = sum(t["pnl"] for t in all_trades)
    wr     = len(wins)/total*100 if total else 0
    roi    = pnl/START_BAL*100

    avg_w  = np.mean([t["pnl"] for t in wins])   if wins   else 0.
    avg_l  = np.mean([t["pnl"] for t in losses]) if losses else 0.
    pf     = (len(wins)*avg_w)/(len(losses)*abs(avg_l)) if losses and avg_l else float("inf")

    avg_conf = np.mean([t["conf"] for t in all_trades if t["conf"]>0])

    print(f"\n{'='*60}")
    print(f"COMBINED 750×3  ({total} trades):")
    print(f"  Win Rate  : {wr:5.1f}%")
    print(f"  Total PnL : ${pnl:+.2f}  (ROI {roi:+.1f}%  on ${START_BAL:.0f})")
    print(f"  Avg Win   : ${avg_w:+.2f}  |  Avg Loss: ${avg_l:.2f}")
    print(f"  Prof.Factor: {pf:.2f}")
    print(f"  Avg Conf  : {avg_conf:.1f}%")
    tp_all = sum(1 for t in all_trades if t["reason"]=="tp")
    sl_all = sum(1 for t in all_trades if t["reason"]=="sl")
    print(f"  TP hits: {tp_all}  SL hits: {sl_all}")

    bep = sl_all/(tp_all+sl_all)*100 if (tp_all+sl_all) else 0
    print(f"\n  Break-even WR: {bep:.1f}%  (need >{bep:.1f}% to profit)")

    print(f"\n{'─'*60}")
    print(f"COMPARISON — same GBM, $50/trade, 750×3:")
    print(f"  Bot 5-strats (prev backtest)  WR≈77%  PnL≈+$39")
    print(f"  Sentinel v1.0 (this run)      WR {wr:.0f}%  PnL {pnl:+.0f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
