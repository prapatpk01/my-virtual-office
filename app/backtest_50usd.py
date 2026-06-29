"""
Full backtest — fixed $50 position size, OKX fees 0.20% round trip.
BTC/USDT synthetic data (GBM).  Bars: 250 / 500 / 1000.
Usage: python backtest_50usd.py [15m|1h]
"""
import asyncio, sys, math, random
import numpy as np
sys.path.insert(0, "")

TF = sys.argv[1] if len(sys.argv) > 1 else "15m"
assert TF in ("15m", "1h"), "TF must be 15m or 1h"

# 15m: sigma=0.002, bar_sec=900, bars/day=96
# 1h:  sigma=0.005, bar_sec=3600, bars/day=24
TF_CFG = {
    "15m": dict(sigma=0.002, bar_sec=900,  bars_per_day=96,  lookfwd=60),
    "1h":  dict(sigma=0.005, bar_sec=3600, bars_per_day=24,  lookfwd=120),
}
cfg = TF_CFG[TF]

from dataclasses import dataclass
from trading.strategies.mcdx_strategy import MCDXStrategy
from trading.strategies.base import SignalType

# ── synthetic BTC OHLCV (GBM) ──────────────────────────────────────────────

@dataclass
class C:
    timestamp: int
    open: float; high: float; low: float; close: float; volume: float

def gbm_candles(n: int, seed: int = 42, start: float = 100_000.0,
                mu: float = 0.0001, sigma: float = None) -> list:
    if sigma is None:
        sigma = cfg["sigma"]
    rng = random.Random(seed)
    candles = []
    price = start
    for i in range(n):
        ret   = mu + sigma * (sum(rng.gauss(0,1) for _ in range(1)))
        open_ = price
        close = price * math.exp(ret)
        high  = max(open_, close) * (1 + abs(rng.gauss(0, sigma * 0.5)))
        low   = min(open_, close) * (1 - abs(rng.gauss(0, sigma * 0.5)))
        candles.append(C(i * 900, open_, high, low, close, rng.uniform(1, 10)))
        price = close
    return candles

# ── backtest engine ────────────────────────────────────────────────────────

FEE_RT    = 0.0020   # 0.10% entry + 0.10% exit
POSITION  = 50.0     # USD per trade
LOOKFWD   = cfg["lookfwd"]
TP_USD    = 7.0      # fixed take-profit in USD
SL_USD    = 5.0      # fixed stop-loss in USD

async def run_strategy_backtest(strategy, candles: list, bars: int) -> dict:
    subset = candles[-bars:]
    n = len(subset)
    warmup = min(120, n // 4)

    closes_all = np.array([c.close for c in subset], dtype=float)
    highs_all  = np.array([c.high  for c in subset], dtype=float)
    lows_all   = np.array([c.low   for c in subset], dtype=float)

    results = []
    locked_until = -1

    for i in range(warmup, n - 1):
        if i <= locked_until:
            continue
        window = subset[:i + 1]
        sig = await strategy.analyze(window, window[-1].close)

        if sig.type == SignalType.BUY:
            entry = float(closes_all[i])
            amount = POSITION / entry
            tp_p   = entry + TP_USD / amount
            sl_p   = entry - SL_USD / amount
            win_usd  =  TP_USD - POSITION * FEE_RT
            loss_usd = -SL_USD - POSITION * FEE_RT

            for j in range(i + 1, min(i + LOOKFWD, n)):
                if lows_all[j] <= sl_p:
                    results.append(loss_usd)
                    locked_until = j
                    break
                if highs_all[j] >= tp_p:
                    results.append(win_usd)
                    locked_until = j
                    break
            else:
                exit_price = float(closes_all[min(i + LOOKFWD, n) - 1])
                pnl = amount * (exit_price - entry) - POSITION * FEE_RT
                results.append(pnl)
                locked_until = i + LOOKFWD

    if not results:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0, "pf": 0,
                "total$": 0, "avg$": 0, "t/day": 0}

    wins   = [p for p in results if p > 0]
    losses = [p for p in results if p <= 0]
    total  = len(results)
    wr     = len(wins)/total*100 if total else 0
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    pf   = gross_win/gross_loss if gross_loss > 0 else float("inf")
    total_usd  = sum(results)
    avg_usd    = total_usd/total if total else 0
    bars_per_day = cfg["bars_per_day"]
    trades_per_day = total / (bars / bars_per_day)

    return {
        "trades": total,
        "wins":   len(wins),
        "losses": len(losses),
        "wr":     round(wr, 1),
        "pf":     round(pf, 2),
        "total$": round(total_usd, 2),
        "avg$":   round(avg_usd, 3),
        "t/day":  round(trades_per_day, 1),
    }


async def main():
    BAR_SETS = [250, 500, 1000]
    candles_max = gbm_candles(1100, seed=99)

    strategies = [
        ("MCDX (Adaptive)", MCDXStrategy("BTC/USDT")),
    ]

    print(f"\n{'='*80}")
    print(f"  Full Backtest — ${POSITION}/trade  TP=+${TP_USD}  SL=-${SL_USD}  OKX 0.20% RT  BTC/USDT {TF}")
    print(f"{'='*80}")

    all_rows = []

    for sname, strat in strategies:
        print(f"\n── {sname} ──────────────────────────────────────────────")
        print(f"  {'Bars':>6} │ {'Trades':>6} │ {'WR%':>6} │ {'PF':>6} │ {'Total $':>9} │ {'Avg$/T':>8} │ {'T/day':>6}")
        print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*6}")
        for bars in BAR_SETS:
            r = await run_strategy_backtest(strat, candles_max, bars)
            sign = "+" if r["total$"] >= 0 else ""
            print(f"  {bars:>6} │ {r['trades']:>6} │ {r['wr']:>5.1f}% │ {r['pf']:>6.2f} │ "
                  f"{sign}{r['total$']:>8.2f} │ {r['avg$']:>+8.3f} │ {r['t/day']:>6.1f}")
            all_rows.append({"strategy": sname, "bars": bars, **r})

    print(f"\n{'='*80}")
    print("  SUMMARY — avg across 250/500/1000 bars")
    print(f"  {'Strategy':18} │ {'Avg WR%':>8} │ {'Avg PF':>7} │ {'Total$ 1k':>10} │ {'Avg$/T':>8}")
    print(f"  {'─'*18}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*8}")
    for sname, _ in strategies:
        rows = [r for r in all_rows if r["strategy"] == sname]
        avg_wr  = sum(r["wr"]     for r in rows) / len(rows)
        avg_pf  = sum(r["pf"]     for r in rows) / len(rows)
        tot_1k  = next((r["total$"] for r in rows if r["bars"]==1000), 0)
        avg_per = sum(r["avg$"]   for r in rows) / len(rows)
        print(f"  {sname:18} │ {avg_wr:>7.1f}% │ {avg_pf:>7.2f} │ "
              f"{tot_1k:>+10.2f} │ {avg_per:>+8.3f}")

    print(f"\n  Note: fee = $0.10/trade ($50 × 0.20%). Fixed TP/SL.")

asyncio.run(main())
