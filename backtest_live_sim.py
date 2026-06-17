"""
Live Bot Simulation — Real Binance Data Jan–May 2026 (5 months)
Replicates exact bot logic:
  - MAX_POSITIONS=3 global cap
  - WaveTrend: 2-slot stacking (WT#0, WT#1)
  - Other strategies: 1 position each, locked until TP/SL hit
  - SELL signal → block BUY for 10 bars on that strategy
  - MTF 4H gate: block BUY when 4H EMA20+EMA200 bearish
  - $50 USDT per trade, starting capital $300

Shows: trades executed, blocked (with reason), WR, P&L, balance curve
"""
import csv, glob, math, os
import numpy as np
from datetime import datetime, timezone

BASE = "/tmp/chart_data"
TRADE_USDT    = 50.0
START_CAPITAL = 300.0
MAX_POSITIONS = 3
SELL_BLOCK    = 10    # bars to block BUY after SELL signal
LOOKFORWARD   = 60    # max bars to wait for TP/SL

# ── Candle ────────────────────────────────────────────────────────────────────

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

def ts_str(ts_ms, fmt="%Y-%m-%d %H:%M"):
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime(fmt)

# ── Indicators ────────────────────────────────────────────────────────────────

def ema(v, p):
    a=np.array(v,dtype=float); r=np.full(len(a),np.nan)
    if len(a)<p: return r
    k=2/(p+1); r[p-1]=a[:p].mean()
    for i in range(p,len(a)): r[i]=a[i]*k+r[i-1]*(1-k)
    return r

def sma(v, p):
    a=np.array(v,dtype=float); r=np.full(len(a),np.nan)
    for i in range(p-1,len(a)): r[i]=a[i-p+1:i+1].mean()
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
    return [_C(i) for i in range(n)],ho,hc,hh,hl

# ── 4H bias ───────────────────────────────────────────────────────────────────

def build_4h_bias(c4h):
    _,_,hc4,_,_=ha(c4h)
    em20=ema(hc4.tolist(),20); em200=ema(hc4.tolist(),200)
    n=len(c4h); bias=np.zeros(n,dtype=int)
    for i in range(n):
        if np.isnan(em20[i]): continue
        macro=True if np.isnan(em200[i]) else float(hc4[i])>float(em200[i])
        bull=float(hc4[i])>float(em20[i]) and macro
        bear=float(hc4[i])<float(em20[i]) and not macro
        if bull: bias[i]=1
        elif bear: bias[i]=-1
    return bias

def build_ts_index(candles): return {c.ts:i for i,c in enumerate(candles)}

def get_4h_bias_at(ts_1h, c4h, ts4map, bias4h):
    tf_ms=240*60*1000; al=(ts_1h//tf_ms)*tf_ms
    for d in range(0,5):
        k=al-d*tf_ms
        if k in ts4map:
            return int(bias4h[ts4map[k]])
    return 0

# ── Pre-compute all strategy signals ─────────────────────────────────────────

def compute_wt_signals(c1h, n1=8, n2=12):
    hac,_,hc,hh,hl=ha(c1h)
    hh_a=np.array([c.high for c in hac]); hl_a=np.array([c.low for c in hac])
    ap=(hh_a+hl_a+hc)/3; esa=ema(ap.tolist(),n1)
    dabs=np.abs(ap-esa); fo=np.where(~np.isnan(dabs))[0]
    df=dabs.copy()
    if len(fo): df[:fo[0]]=dabs[fo[0]]
    d_=ema(df.tolist(),n1)
    ci=np.where(d_>1e-10,(ap-esa)/(0.015*d_),0.)
    wt1=ema(ci.tolist(),n2); wt2=sma(wt1.tolist(),4)
    n=len(c1h); buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool)
    for i in range(1,n):
        if np.isnan(wt1[i]) or np.isnan(wt2[i]): continue
        cu=float(wt1[i-1])<=float(wt2[i-1]) and float(wt1[i])>float(wt2[i])
        cd=float(wt1[i-1])>=float(wt2[i-1]) and float(wt1[i])<float(wt2[i])
        buy_s[i]=cu; sell_s[i]=cd
    return buy_s, sell_s, atr_calc(hac,14), hc, hh_a, hl_a

def compute_macd_signals(c1h):
    hac,ho,hc,hh,hl=ha(c1h)
    cl=hc.tolist(); hm9=hma(cl,9)
    ml,sl,_=macd_calc(cl,5,13,4); atr14=atr_calc(hac,14)
    n=len(c1h); buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool)
    for i in range(1,n):
        if any(np.isnan(x) for x in [hm9[i],ml[i],ml[i-1],sl[i],sl[i-1]]): continue
        hc_=float(ml[i])-float(sl[i]); hp_=float(ml[i-1])-float(sl[i-1])
        if hc_>0 and hp_<=0 and float(hc[i])>float(hm9[i]): buy_s[i]=True
        elif hc_<0 and hp_>=0: sell_s[i]=True
    return buy_s, sell_s, atr14, hc, np.array([c.high for c in hac]), np.array([c.low for c in hac])

def compute_mom_signals(c1h, thr=4, re=5):
    hac,ho,hc,hh,hl=ha(c1h)
    cl=hc.tolist(); rs=rsi_w(hc,14)
    rev=np.full(len(hc),np.nan); valid=np.where(~np.isnan(rs))[0]
    if len(valid)>=9:
        s=int(valid[0])
        rv=ema([float(rs[i]) for i in range(s,len(rs))],9)
        rev[s:s+len(rv)]=rv[:len(rs)-s]
    em14=ema(cl,14); ml,sl,_=macd_calc(cl,12,26,9)
    n=len(c1h); buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool)
    pb=0; ps=0; lb=-999; ls=-999
    for i in range(1,n):
        if any(np.isnan(x) for x in [rs[i],rev[i],em14[i],ml[i],sl[i]]): continue
        rv_=float(rs[i]); re_=float(rev[i]); ho_=float(ho[i]); hc_=float(hc[i])
        em_=float(em14[i]); ml_=float(ml[i]); sl_=float(sl[i])
        bull=int(rv_>50)+int(rv_>re_)+int(hc_>ho_)+int(hc_>em_)+int(ml_>sl_)
        bear=int(rv_<50)+int(rv_<re_)+int(hc_<ho_)+int(hc_<em_)+int(ml_<sl_)
        if bull>=thr:
            if pb<thr or (i-lb)>=re: buy_s[i]=True; lb=i
        if bear>=thr:
            if ps<thr or (i-ls)>=re: sell_s[i]=True; ls=i
        pb=bull; ps=bear
    return buy_s, sell_s, atr_calc(hac,14), hc, np.array([c.high for c in hac]), np.array([c.low for c in hac])

def compute_ut_signals(c1h, mult=0.3, ln=14):
    hac,_,hc,hh,hl=ha(c1h)
    ut_atr=atr_calc(hac,ln); atr14=atr_calc(hac,14)
    n=len(c1h); tsl=np.full(n,np.nan)
    for i in range(n):
        sv=float(ut_atr[i])*mult if not np.isnan(ut_atr[i]) else 0
        sc=float(hc[i])
        if i==0 or np.isnan(tsl[i-1]): tsl[i]=sc+sv; continue
        tp=float(tsl[i-1]); sp=float(hc[i-1])
        if sc>tp and sp>tp:   tsl[i]=max(tp,sc-sv)
        elif sc<tp and sp<tp: tsl[i]=min(tp,sc+sv)
        elif sc>tp:           tsl[i]=sc-sv
        else:                 tsl[i]=sc+sv
    buy_s=np.zeros(n,bool); sell_s=np.zeros(n,bool)
    for i in range(1,n):
        if np.isnan(tsl[i]) or np.isnan(tsl[i-1]): continue
        bu=float(hc[i-1])<float(tsl[i-1]) and float(hc[i])>float(tsl[i])
        se=float(hc[i-1])>float(tsl[i-1]) and float(hc[i])<float(tsl[i])
        buy_s[i]=bu; sell_s[i]=se
    return buy_s, sell_s, atr14, hc, np.array([c.high for c in hac]), np.array([c.low for c in hac])

# ── Main simulation ───────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    c1h = load_tf("1h"); c4h = load_tf("4h")
    n = len(c1h)
    days=(c1h[-1].ts-c1h[0].ts)/(1000*3600*24); months=days/30.44
    t0=ts_str(c1h[0].ts,"%Y-%m-%d"); t1=ts_str(c1h[-1].ts,"%Y-%m-%d")

    print("Computing signals...")
    wt_buy,  wt_sell,  wt_atr,  wt_hc,  wt_hh,  wt_hl  = compute_wt_signals(c1h)
    md_buy,  md_sell,  md_atr,  md_hc,  md_hh,  md_hl  = compute_macd_signals(c1h)
    mo_buy,  mo_sell,  mo_atr,  mo_hc,  mo_hh,  mo_hl  = compute_mom_signals(c1h)
    ut_buy,  ut_sell,  ut_atr,  ut_hc,  ut_hh,  ut_hl  = compute_ut_signals(c1h)

    print("Computing 4H bias...")
    bias4h = build_4h_bias(c4h)
    ts4map  = build_ts_index(c4h)

    # Per-bar 4H bias lookup (pre-compute for speed)
    bias_at = np.zeros(n, dtype=int)
    for i in range(n):
        bias_at[i] = get_4h_bias_at(c1h[i].ts, c4h, ts4map, bias4h)

    # ── Simulation state ──────────────────────────────────────────────────────
    capital   = START_CAPITAL
    open_pos  = {}    # key → {entry, sl, tp, amount_btc, strat, bar}
    blocked   = {}    # strategy_key → bar to unblock (SELL block)
    bar_ctr   = 0

    # Stats per strategy
    strategies = ["WT#0","WT#1","MACD","Momentum","UTBot"]
    stat = {s: {"trades":0,"wins":0,"losses":0,"pending":0,
                "blk_mtf":0,"blk_lock":0,"blk_maxpos":0,"blk_sell":0,
                "pnl":0.0} for s in strategies}

    trade_log  = []   # full trade log
    balance_curve = []  # (bar, balance)
    monthly_pnl   = {}  # "2026-MM" → pnl

    def count_open(): return len(open_pos)

    def get_signals_at(i):
        # Returns list of (strategy_key, is_buy, is_sell, atr, hc, hh, hl)
        return [
            ("WT#0",     bool(wt_buy[i]),  bool(wt_sell[i]),  wt_atr[i], wt_hc[i], wt_hh[i], wt_hl[i]),
            ("WT#1",     bool(wt_buy[i]),  bool(wt_sell[i]),  wt_atr[i], wt_hc[i], wt_hh[i], wt_hl[i]),
            ("MACD",     bool(md_buy[i]),  bool(md_sell[i]),  md_atr[i], md_hc[i], md_hh[i], md_hl[i]),
            ("Momentum", bool(mo_buy[i]),  bool(mo_sell[i]),  mo_atr[i], mo_hc[i], mo_hh[i], mo_hl[i]),
            ("UTBot",    bool(ut_buy[i]),  bool(ut_sell[i]),  ut_atr[i], ut_hc[i], ut_hh[i], ut_hl[i]),
        ]

    print(f"\nSimulating {n} bars ({months:.1f} months)...\n")

    for i in range(n):
        bar_ctr += 1
        ts = c1h[i].ts
        month_key = ts_str(ts, "%Y-%m")
        if month_key not in monthly_pnl: monthly_pnl[month_key] = 0.0

        # ── Step 1: Check TP/SL on open positions ─────────────────────────────
        to_close = []
        for key, pos in open_pos.items():
            bar_h = float(c1h[i].high); bar_l = float(c1h[i].low)
            sl_hit = bar_l <= pos["sl"]
            tp_hit = bar_h >= pos["tp"]
            if sl_hit or tp_hit:
                reason = "tp" if tp_hit else "sl"
                exit_p = pos["tp"] if tp_hit else pos["sl"]
                pnl = pos["amount_btc"] * (exit_p - pos["entry"])
                capital += TRADE_USDT + pnl
                s = pos["strat"]
                if tp_hit: stat[s]["wins"]+=1
                else:      stat[s]["losses"]+=1
                stat[s]["pnl"]  += pnl
                monthly_pnl[month_key] += pnl
                pnl_pct = pnl/TRADE_USDT*100
                trade_log.append({
                    "date":    ts_str(c1h[pos["bar"]].ts),
                    "close":   ts_str(ts),
                    "strat":   s,
                    "entry":   pos["entry"],
                    "exit":    exit_p,
                    "result":  reason.upper(),
                    "pnl":     round(pnl,3),
                    "pnl_pct": round(pnl_pct,2),
                    "capital": round(capital,2),
                })
                to_close.append(key)
        for key in to_close:
            del open_pos[key]

        balance_curve.append((i, round(capital,2)))

        # ── Step 2: Process signals ────────────────────────────────────────────
        bias = int(bias_at[i])
        sigs = get_signals_at(i)

        wt_slot_used_this_bar = False   # only 1 WT slot per bar

        for strat_key, is_buy, is_sell, atr_v, hc_v, hh_v, hl_v in sigs:
            # SELL signal: ignored — TP/SL handles all exits (no SELL block)
            if not is_buy:
                continue

            # WT: only fire one slot per bar
            if strat_key in ("WT#0","WT#1") and wt_slot_used_this_bar:
                continue

            # ── Check MTF 4H gate (strict: require bullish, not just block bearish) ──
            if bias != 1:
                stat[strat_key]["blk_mtf"] += 1
                continue

            # ── Check strategy lock (already in position) ──────────────────
            if strat_key in open_pos:
                stat[strat_key]["blk_lock"] += 1
                continue

            # ── Check MAX_POSITIONS ────────────────────────────────────────
            if count_open() >= MAX_POSITIONS:
                stat[strat_key]["blk_maxpos"] += 1
                continue

            # ── Check sufficient ATR ───────────────────────────────────────
            if np.isnan(atr_v) or atr_v <= 0:
                continue

            # ── Check capital ──────────────────────────────────────────────
            if capital < TRADE_USDT:
                stat[strat_key]["blk_maxpos"] += 1   # count as maxpos/capital block
                continue

            # ── Execute BUY ────────────────────────────────────────────────
            entry = float(hc_v)
            sl    = entry - 1.5 * float(atr_v)
            tp    = entry + 1.5 * float(atr_v)
            btc   = TRADE_USDT / entry
            capital -= TRADE_USDT

            open_pos[strat_key] = {
                "entry": entry, "sl": sl, "tp": tp,
                "amount_btc": btc, "strat": strat_key, "bar": i,
            }
            stat[strat_key]["trades"] += 1
            if strat_key.startswith("WT"):
                wt_slot_used_this_bar = True

    # ── Close any remaining open positions at last price ──────────────────────
    last_price = float(c1h[-1].close)
    for key, pos in open_pos.items():
        pnl = pos["amount_btc"] * (last_price - pos["entry"])
        capital += TRADE_USDT + pnl
        stat[pos["strat"]]["pending"] += 1
        stat[pos["strat"]]["pnl"] += pnl

    # ── Print results ──────────────────────────────────────────────────────────
    total_trades = sum(s["trades"] for s in stat.values())
    total_wins   = sum(s["wins"]   for s in stat.values())
    total_losses = sum(s["losses"] for s in stat.values())
    closed = total_wins + total_losses
    total_pnl    = sum(s["pnl"]    for s in stat.values())
    total_blk_mtf  = sum(s["blk_mtf"]    for s in stat.values())
    total_blk_lock = sum(s["blk_lock"]   for s in stat.values())
    total_blk_maxp = sum(s["blk_maxpos"] for s in stat.values())
    total_blk_sell = sum(s["blk_sell"]   for s in stat.values())
    total_blk = total_blk_mtf + total_blk_lock + total_blk_maxp + total_blk_sell
    wr = total_wins/closed*100 if closed else 0

    print(f"{'='*65}")
    print(f"  BOT LIVE SIMULATION — BTC/USDT 1H  {t0} → {t1}")
    print(f"  Capital: ${START_CAPITAL} | Trade: ${TRADE_USDT} | MAX_POS: {MAX_POSITIONS}")
    print(f"{'='*65}")

    print(f"\n  EXECUTION SUMMARY")
    print(f"  {'─'*55}")
    print(f"  Trades opened          : {total_trades}")
    print(f"  Closed (TP+SL)         : {closed}   (still open: {total_trades-closed})")
    print(f"  Wins / Losses          : {total_wins}W / {total_losses}L")
    print(f"  Win Rate               : {wr:.1f}%")
    print(f"  Net P&L                : ${total_pnl:+.2f}")
    print(f"  Final capital          : ${capital:.2f}  (started ${START_CAPITAL:.2f})")
    print(f"  Return on capital      : {(capital-START_CAPITAL)/START_CAPITAL*100:+.1f}%")
    print(f"  Trades/month           : {total_trades/months:.1f}")

    print(f"\n  BLOCKED SIGNALS  (total: {total_blk})")
    print(f"  {'─'*55}")
    print(f"  4H MTF gate (bearish)  : {total_blk_mtf:>5}  ({total_blk_mtf/(total_blk+0.001)*100:.0f}%)")
    print(f"  Strategy lock (in pos) : {total_blk_lock:>5}  ({total_blk_lock/(total_blk+0.001)*100:.0f}%)")
    print(f"  SELL-block (10 bars)   : {total_blk_sell:>5}  ({total_blk_sell/(total_blk+0.001)*100:.0f}%)")
    print(f"  MAX_POSITIONS=3        : {total_blk_maxp:>5}  ({total_blk_maxp/(total_blk+0.001)*100:.0f}%)")

    print(f"\n  PER-STRATEGY BREAKDOWN")
    print(f"  {'─'*65}")
    print(f"  {'Strategy':<12} {'T':>4} {'W':>4} {'L':>4} {'WR%':>6} {'P&L$':>8} {'t/mo':>6}  {'BLK:MTF':>8} {'LOCK':>5} {'SELL':>5} {'MAX':>5}")
    print(f"  {'─'*65}")
    for s in strategies:
        st = stat[s]; t=st["trades"]; w=st["wins"]; l=st["losses"]
        cl=w+l; wr_s=w/cl*100 if cl else 0; tmo=t/months
        pend=f"(+{st['pending']} open)" if st["pending"] else ""
        blk_tot=st["blk_mtf"]+st["blk_lock"]+st["blk_sell"]+st["blk_maxpos"]
        print(f"  {s:<12} {t:>4} {w:>4} {l:>4} {wr_s:>5.1f}% {st['pnl']:>+7.2f}$ {tmo:>5.1f}/mo  {st['blk_mtf']:>8} {st['blk_lock']:>5} {st['blk_sell']:>5} {st['blk_maxpos']:>5}  {pend}")
    print(f"  {'─'*65}")
    print(f"  {'TOTAL':<12} {total_trades:>4} {total_wins:>4} {total_losses:>4} {wr:>5.1f}% {total_pnl:>+7.2f}$ {total_trades/months:>5.1f}/mo  {total_blk_mtf:>8} {total_blk_lock:>5} {total_blk_sell:>5} {total_blk_maxp:>5}")

    print(f"\n  MONTHLY BREAKDOWN")
    print(f"  {'─'*40}")
    print(f"  {'Month':<10} {'P&L $':>8} {'Cumulative':>12}")
    cumulative = 0.0
    for month in sorted(monthly_pnl):
        cumulative += monthly_pnl[month]
        bar_ = "+" * int(monthly_pnl[month]/0.1) if monthly_pnl[month]>0 else "-"*int(-monthly_pnl[month]/0.1)
        print(f"  {month:<10} {monthly_pnl[month]:>+7.2f}$  {cumulative:>+10.2f}$  {bar_[:30]}")
    print(f"{'='*65}")

    # Last 20 trades
    if trade_log:
        print(f"\n  RECENT TRADE LOG (last 20)")
        print(f"  {'Date':>10} {'Strat':>10} {'Entry':>8} {'Exit':>8} {'Res':>4} {'P&L':>7} {'Cap':>8}")
        for t in trade_log[-20:]:
            star=" *" if t["result"]=="TP" else "  "
            print(f"  {t['date'][:10]:>10} {t['strat']:>10} {t['entry']:>8,.0f} {t['exit']:>8,.0f} {t['result']:>4}{star} {t['pnl']:>+6.2f}$ {t['capital']:>8.2f}")

if __name__ == "__main__":
    main()
