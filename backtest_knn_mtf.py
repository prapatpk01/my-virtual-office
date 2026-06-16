"""
kNN MTF Backtest — compare baseline vs. filtered (EMA50 + MTF bias gate)
750 bars × 3 windows, multi-position context (max 4 global positions).

MTF simulation:
  15m proxy → 1H candles (same data, shorter-term weight)
  1H         → base GBM candles
  4H         → every 4 bars aggregated from 1H

Gate: kNN BUY only when MTF composite bias > 0 (bullish)
"""
import sys, math, random
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.knn_strategy import KNNStrategy
from trading.strategies.base import BaseStrategy

# ── constants ─────────────────────────────────────────────────────────────────
POSITION   = 50.0
FEE_RT     = 0.0025
START_BAL  = 500.0
LOOKFWD    = 120
TP_PCT     = 0.048
SL_PCT     = 0.046

TP_NET = round(POSITION * TP_PCT - POSITION * FEE_RT, 4)
SL_NET = round(POSITION * SL_PCT + POSITION * FEE_RT, 4)

class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self,ts,o,h,l,c,v):
        self.timestamp=ts;self.open=o;self.high=h;self.low=l;self.close=c;self.volume=v

# ── GBM data ──────────────────────────────────────────────────────────────────
def gbm(n, seed=99, start=100_000.0, mu=0.0002, sigma=0.005):
    rng=random.Random(seed); candles=[]; price=start
    for i in range(n):
        ret=mu+sigma*rng.gauss(0,1); o=price; c=price*math.exp(ret)
        h=max(o,c)*(1+abs(rng.gauss(0,sigma*0.5)))
        l=min(o,c)*(1-abs(rng.gauss(0,sigma*0.5)))
        candles.append(C(i*3600,o,h,l,c,rng.uniform(1,10))); price=c
    return candles

def agg_4h(candles_1h):
    """Aggregate 1H bars into 4H OHLCV bars."""
    result = []
    for i in range(0, len(candles_1h), 4):
        chunk = candles_1h[i:i+4]
        if not chunk: continue
        result.append(C(
            chunk[0].timestamp,
            chunk[0].open,
            max(c.high for c in chunk),
            min(c.low  for c in chunk),
            chunk[-1].close,
            sum(c.volume for c in chunk),
        ))
    return result

# ── pre-compute MTF bias array for each 1H bar ───────────────────────────────
def build_mtf_bias(candles_1h, candles_4h, min_bars=60):
    """Returns array of MTF composite scores, one per 1H bar."""
    n = len(candles_1h)
    bias = np.zeros(n)
    for i in range(min_bars, n):
        c1h = candles_1h[:i+1]
        c4h = candles_4h[:i//4+1]
        if len(c4h) < 15:
            continue
        comp_pct, _ = BaseStrategy.compute_mtf_bias(
            c1h,                                     # "15m" proxy = 1H data
            {"1h": c1h, "4h": c4h},
        )
        bias[i] = comp_pct
    return bias

# ── simulate ──────────────────────────────────────────────────────────────────
def simulate(strat, candles, mtf_bias, balance):
    buy_sig, sell_sig, votes = strat._build_signals(candles, mtf_bias=mtf_bias)
    hig = np.array([c.high  for c in candles])
    low = np.array([c.low   for c in candles])
    cls = np.array([c.close for c in candles])
    n   = len(candles)

    trades = []; exit_bar = -1; block = -1
    for bi in range(n):
        if not buy_sig[bi]: continue
        if bi <= exit_bar or bi < block: continue
        entry = float(cls[bi])
        tp_p  = entry * (1 + TP_PCT)
        sl_p  = entry * (1 - SL_PCT)
        outcome = "timeout"; eb = min(bi + LOOKFWD, n-1); pnl = 0.0
        for j in range(bi+1, min(bi+LOOKFWD, n)):
            if low[j] <= sl_p:  outcome="loss"; pnl=-SL_NET; eb=j; break
            if hig[j] >= tp_p:  outcome="win";  pnl=TP_NET;  eb=j; break
        else:
            ep = float(cls[min(bi+LOOKFWD, n)-1])
            pnl = round((ep-entry)/entry*POSITION - POSITION*FEE_RT, 4)
        balance += pnl
        trades.append({"o": outcome, "p": round(pnl,4), "bi": bi})
        exit_bar = eb
    return trades, balance

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  kNN MTF Backtest — 750 bars × 3 windows | Fee 0.25% RT")
    print(f"  TP=2.25% (+${TP_NET:.3f}) | SL=1.5% (-${SL_NET:.3f}) | R:R 1:1.5")
    print("="*72)

    candles_all = gbm(2400, seed=99)
    tail = candles_all[-2250:]
    candles_4h_all = agg_4h(candles_all)   # 4H for full history

    windows = [tail[i*750:(i+1)*750] for i in range(3)]
    wlabels = ["W1 (~month1)", "W2 (~month2)", "W3 (~month3)"]

    configs = [
        ("Baseline (N=3 CD=3)",
         KNNStrategy("BTC/USDT", {"n_bars":3,"k":3,"lagz":20,"cooldown":3,
                                   "trend_ema":0,"min_vote":0.0,"mtf_min_bias":0.0,
                                   "sl_pct":SL_PCT,"tp_pct":TP_PCT,"long_only":True}),
         False),
        ("EMA50 filter (CD=10)",
         KNNStrategy("BTC/USDT", {"n_bars":3,"k":3,"lagz":20,"cooldown":10,
                                   "trend_ema":50,"min_vote":0.05,"mtf_min_bias":0.0,
                                   "sl_pct":SL_PCT,"tp_pct":TP_PCT,"long_only":True}),
         False),
        ("EMA50 + MTF bias>0",
         KNNStrategy("BTC/USDT", {"n_bars":3,"k":3,"lagz":20,"cooldown":10,
                                   "trend_ema":50,"min_vote":0.05,"mtf_min_bias":0.01,
                                   "sl_pct":SL_PCT,"tp_pct":TP_PCT,"long_only":True}),
         True),
        ("MTF only (no EMA filter)",
         KNNStrategy("BTC/USDT", {"n_bars":3,"k":3,"lagz":20,"cooldown":8,
                                   "trend_ema":0,"min_vote":0.03,"mtf_min_bias":0.01,
                                   "sl_pct":SL_PCT,"tp_pct":TP_PCT,"long_only":True}),
         True),
    ]

    be_wr = SL_NET / (TP_NET + SL_NET) * 100

    for cfg_name, strat, use_mtf in configs:
        print(f"\n{'─'*72}")
        print(f"  Config: {cfg_name}")
        print(f"  {'Window':<14} {'T':>4} {'W':>4} {'L':>4} {'TO':>4} {'WR%':>6} {'Net$':>8} {'Bal':>9}")
        print(f"  {'─'*14}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*9}")

        balance = START_BAL; all_trades = []

        # absolute offsets into candles_all for each window
        base_offset = len(candles_all) - 2250  # = 150

        for wi, win_candles in enumerate(windows):
            start_abs = base_offset + wi * 750     # first bar of this window in candles_all
            end_abs   = start_abs + 750
            warm_start = max(0, start_abs - 200)   # 200-bar warmup for indicators

            full_window = candles_all[warm_start:end_abs]  # warmup + 750 bars

            if use_mtf:
                c4h_win  = agg_4h(candles_all[:end_abs])
                bias_all = build_mtf_bias(full_window, c4h_win)
                # align: bias_all covers len(full_window) bars; we want the last 750
                bias_win = bias_all[-750:]
            else:
                bias_win = None

            trades, balance = simulate(strat, full_window[-750:], bias_win, balance)

            w   = [t for t in trades if t["o"]=="win"]
            l   = [t for t in trades if t["o"]=="loss"]
            to  = [t for t in trades if t["o"]=="timeout"]
            nn  = len(trades)
            wr  = len(w)/max(len(w)+len(l),1)*100 if (len(w)+len(l))>0 else 0.0
            net = sum(t["p"] for t in trades)
            all_trades.extend(trades)
            s   = "+" if net >= 0 else ""
            print(f"  {wlabels[wi]:<14} {nn:>4} {len(w):>4} {len(l):>4} {len(to):>4} "
                  f"{wr:>5.1f}% {s}{net:>7.2f} ${balance:>8.2f}")

        aw = [t for t in all_trades if t["o"]=="win"]
        al = [t for t in all_trades if t["o"]=="loss"]
        at = [t for t in all_trades if t["o"]=="timeout"]
        an = len(all_trades)
        awr = len(aw)/max(len(aw)+len(al),1)*100 if (len(aw)+len(al))>0 else 0.0
        anet = sum(t["p"] for t in all_trades)
        s = "+" if anet >= 0 else ""
        t_day = an / (2250/24)
        print(f"  {'─'*14}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*9}")
        print(f"  {'TOTAL':<14} {an:>4} {len(aw):>4} {len(al):>4} {len(at):>4} "
              f"{awr:>5.1f}% {s}{anet:>7.2f} ${balance:>8.2f}")
        print(f"  → {t_day:.1f} trades/day  |  $500 → ${balance:.2f} "
              f"({'+'if balance>=START_BAL else ''}{balance-START_BAL:.2f})")

    print(f"\n  Break-even WR (R:R 1:1.5, fee 0.25%): {be_wr:.1f}%")
    print("="*72)

main()
