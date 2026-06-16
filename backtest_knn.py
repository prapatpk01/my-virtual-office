"""
kNN Strategy — Full Backtest  250 bars × 3 windows (750 bars total)
GBM synthetic BTC/USDT 1H candles (Binance blocked in cloud)

Rules:
  - $50/trade, $500 starting capital
  - TP = +2.25% of entry
  - SL = -1.50% of entry  (R:R 1:1.5)
  - Fee 0.10% each way = 0.20% round-trip
  - Pure TP/SL exits (no sell-signal exit)
  - Cooldown handled internally by kNN signal generator
  - Lookforward: 60 bars max before timeout

Best params from sweep:  N=15, K=3, LAGZ=20, CD=8, long_only=True
"""
import asyncio, sys, math, random
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.knn_strategy import KNNStrategy

# ── constants ────────────────────────────────────────────────────────────────
POSITION   = 50.0     # USD per trade
TP_PCT     = 0.048    # 4.8%
SL_PCT     = 0.046    # 4.6%
FEE_RT     = 0.0020   # 0.10%×2
START_BAL  = 500.0
LOOKFWD    = 60       # max bars to wait for TP/SL

TP_NET = round(POSITION * TP_PCT - POSITION * FEE_RT, 4)   # +$1.025
SL_NET = round(POSITION * SL_PCT + POSITION * FEE_RT, 4)   # -$0.85

# ── candle dataclass ─────────────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp=ts; self.open=o; self.high=h
        self.low=l; self.close=c; self.volume=v

# ── GBM candle generator ─────────────────────────────────────────────────────
def gbm_candles(n: int, seed: int = 99, start: float = 100_000.0,
                mu: float = 0.0002, sigma: float = 0.005) -> list:
    rng = random.Random(seed)
    candles, price = [], start
    for i in range(n):
        ret   = mu + sigma * rng.gauss(0, 1)
        o     = price
        c     = price * math.exp(ret)
        h     = max(o, c) * (1 + abs(rng.gauss(0, sigma * 0.5)))
        l     = min(o, c) * (1 - abs(rng.gauss(0, sigma * 0.5)))
        candles.append(C(i * 3600, o, h, l, c, rng.uniform(1, 10)))
        price = c
    return candles

# ── Simulate one window ───────────────────────────────────────────────────────
def simulate_window(strategy: KNNStrategy, candles: list, balance: float):
    buy_sig, sell_sig, _ = strategy._build_signals(candles)
    cls = np.array([c.close for c in candles], dtype=float)
    hig = np.array([c.high  for c in candles], dtype=float)
    low = np.array([c.low   for c in candles], dtype=float)
    n   = len(candles)

    buy_bars  = [i for i in range(n) if buy_sig[i]]
    sell_bars = set(i for i in range(n) if sell_sig[i])

    trades = []
    trade_exit_bar = -1

    for bi in buy_bars:
        if bi <= trade_exit_bar:
            continue

        entry = float(cls[bi])
        tp_p  = entry * (1 + TP_PCT)
        sl_p  = entry * (1 - SL_PCT)

        outcome  = "timeout"
        exit_bar = min(bi + LOOKFWD, n - 1)
        pnl      = 0.0

        for j in range(bi + 1, min(bi + LOOKFWD, n)):
            if low[j] <= sl_p:
                outcome  = "loss"
                pnl      = -SL_NET
                exit_bar = j
                break
            if hig[j] >= tp_p:
                outcome  = "win"
                pnl      = TP_NET
                exit_bar = j
                break
        else:
            exit_p = float(cls[min(bi + LOOKFWD, n) - 1])
            pnl    = (exit_p - entry) / entry * POSITION - POSITION * FEE_RT
            pnl    = round(pnl, 4)

        balance += pnl
        trades.append({
            "entry_bar": bi, "exit_bar": exit_bar,
            "outcome":   outcome, "pnl": round(pnl, 4),
            "balance":   round(balance, 2),
        })
        trade_exit_bar = exit_bar

    return trades, balance

# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    print("\n" + "="*72)
    print("  kNN Strategy — 250 bars × 3 windows  |  GBM 1H synthetic")
    print(f"  $50/trade | $500 capital")
    print(f"  TP=+2.25% (+${TP_NET:.2f} net) | SL=-1.50% (-${SL_NET:.2f} net)")
    print(f"  R:R 1:1.5 | Fee 0.20% RT | Lookfwd {LOOKFWD} bars")
    print("="*72)

    # params from sweep
    configs = [
        ("N=15 K=3 LAGZ=20 CD=8",  dict(n_bars=15, k=3, lagz=20, cooldown=8,  long_only=True)),
        ("N=3  K=3 LAGZ=20 CD=3",  dict(n_bars=3,  k=3, lagz=20, cooldown=3,  long_only=True)),
        ("N=10 K=3 LAGZ=20 CD=5",  dict(n_bars=10, k=3, lagz=20, cooldown=5,  long_only=True)),
    ]

    all_candles = gbm_candles(760, seed=99)
    tail     = all_candles[-750:]
    windows  = [tail[i*250:(i+1)*250] for i in range(3)]
    labels   = ["W1 (oldest)", "W2 (mid)", "W3 (newest)"]

    print(f"\n  Data: GBM synthetic 1H (seed=99, μ=0.02%/bar, σ=0.5%/bar)")
    print(f"  Bars 1-250 | 251-500 | 501-750\n")

    best_overall = None
    best_net = -9999.0

    for cfg_name, cfg_params in configs:
        strat = KNNStrategy("BTC/USDT", {
            **cfg_params,
            "sl_pct": SL_PCT, "tp_pct": TP_PCT,
        })

        print(f"{'─'*72}")
        print(f"  Config: {cfg_name}")
        print(f"  {'Window':<14} │ {'Trades':>6} │ {'W':>4} │ {'L':>4} │ "
              f"{'TO':>4} │ {'WR%':>6} │ {'Net $':>8} │ {'Balance':>9}")
        print(f"  {'─'*14}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*4}─┼─"
              f"{'─'*4}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*9}")

        balance    = START_BAL
        all_trades: list = []

        for wi, win_candles in enumerate(windows):
            trades, balance = simulate_window(strat, win_candles, balance)
            wins    = [t for t in trades if t["outcome"] == "win"]
            losses  = [t for t in trades if t["outcome"] == "loss"]
            timeout = [t for t in trades if t["outcome"] == "timeout"]
            total   = len(trades)
            wr      = len(wins) / total * 100 if total else 0.0
            net     = sum(t["pnl"] for t in trades)
            all_trades.extend(trades)

            sign = "+" if net >= 0 else ""
            print(f"  {labels[wi]:<14} │ {total:>6} │ {len(wins):>4} │ {len(losses):>4} │ "
                  f"{len(timeout):>4} │ {wr:>5.1f}% │ {sign}{net:>7.2f} │ ${balance:>8.2f}")

        # totals
        all_wins    = [t for t in all_trades if t["outcome"] == "win"]
        all_losses  = [t for t in all_trades if t["outcome"] == "loss"]
        all_timeout = [t for t in all_trades if t["outcome"] == "timeout"]
        all_total   = len(all_trades)
        all_wr      = len(all_wins) / all_total * 100 if all_total else 0.0
        all_net     = sum(t["pnl"] for t in all_trades)
        sign        = "+" if all_net >= 0 else ""

        print(f"  {'─'*14}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*4}─┼─"
              f"{'─'*4}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*9}")
        print(f"  {'TOTAL':14} │ {all_total:>6} │ {len(all_wins):>4} │ {len(all_losses):>4} │ "
              f"{len(all_timeout):>4} │ {all_wr:>5.1f}% │ {sign}{all_net:>7.2f} │ ${balance:>8.2f}")
        t_per_day = all_total / (750 / 24)
        print(f"  → {t_per_day:.1f} trades/day | Capital: $500 → ${balance:.2f} "
              f"({'+'if balance>=START_BAL else ''}{balance-START_BAL:.2f})")

        if all_net > best_net and all_total >= 10:
            best_net     = all_net
            best_overall = cfg_name

    print(f"\n{'='*72}")
    print(f"  Break-even WR (R:R 1:1.5): {SL_NET/(TP_NET+SL_NET)*100:.1f}%")
    print(f"    TP net=${TP_NET:.3f}  SL net=${SL_NET:.3f}  (fee={POSITION*FEE_RT:.2f}/trade)")
    print(f"  Best config: {best_overall}  (net ${best_net:.2f})")
    print("="*72)

asyncio.run(main())
