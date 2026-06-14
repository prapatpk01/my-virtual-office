"""
Backtest comparison: MACDEMAStrategy vs MomentumScoreStrategy vs WTADXStrategy
Uses synthetic BTC-like candle data (GBM + trend regimes) — 5 760 bars ≈ 60 days 15m.
"""
import asyncio
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.macd_ema_strategy import MACDEMAStrategy
from app.trading.strategies.momentum_score_strategy import MomentumScoreStrategy
from app.trading.strategies.wt_adx_strategy import WTADXStrategy


def generate_candles(n: int = 5760, start: float = 65_000.0, seed: int = 42) -> list[OHLCV]:
    """
    Geometric Brownian Motion with alternating trend / range regimes.
    Each bar ≈ 15 minutes of BTC-like price action.
    """
    rng = np.random.default_rng(seed)
    closes = [start]
    regime_len = 96          # ~1 day per regime
    vol_trend  = 0.0015      # trend regime volatility per bar
    vol_range  = 0.0008      # range regime volatility per bar

    for i in range(1, n):
        phase = (i // regime_len) % 6
        if phase < 2:         # uptrend
            drift = +0.0003; vol = vol_trend
        elif phase < 4:       # downtrend
            drift = -0.0003; vol = vol_trend
        else:                 # sideways
            drift = 0.0;     vol = vol_range
        ret = rng.normal(drift, vol)
        closes.append(closes[-1] * (1 + ret))

    candles = []
    bar_ms = 15 * 60 * 1000
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        spread = abs(rng.normal(0, c * 0.0005))
        h = max(o, c) + spread
        l = min(o, c) - spread
        candles.append(OHLCV(
            timestamp=i * bar_ms,
            open=round(o, 2), high=round(h, 2),
            low=round(l, 2),  close=round(c, 2),
            volume=float(rng.uniform(5, 50)),
        ))
    return candles


def print_stats(name: str, stats: dict, best: tuple):
    W = 60
    print(f"{'─'*W}")
    print(f"  {name}")
    print(f"{'─'*W}")
    if not stats:
        print("  (not enough data)\n")
        return
    for key, s in stats.items():
        is_best = best and best[0] == float(key.split("=")[1].split("x")[0])
        marker = "  ◀ best" if is_best else ""
        print(f"  {key}{marker}")
        print(f"    trades={s['trades']:3d}  {s['wins']}W/{s['losses']}L  "
              f"WR={s['win_rate']:5.1f}%  PF={s['profit_factor']:.2f}  "
              f"totalR={s['total_r']:+.1f}R")
    if best:
        print(f"\n  Best: SL={best[0]}×ATR  RR=1:{best[1]}")
    print()


async def main():
    candles = generate_candles(5760)
    print(f"\nSynthetic BTC-like data: {len(candles)} bars (≈60 days 15m)")
    print(f"Price range: ${min(c.low for c in candles):,.0f} – ${max(c.high for c in candles):,.0f}\n")

    strategies = [
        ("MACDEMAStrategy",       MACDEMAStrategy("BTCUSD")),
        ("MomentumScore",         MomentumScoreStrategy("BTCUSD")),
        ("WaveTrend (new rules)", WTADXStrategy("BTCUSD")),
    ]

    print("═"*60)
    print("  BACKTEST COMPARISON — synthetic BTC/USD 15m")
    print("═"*60 + "\n")

    for name, strat in strategies:
        stats, best = await strat.backtest(candles)
        print_stats(name, stats, best)

    print("═"*60)
    print("\nหมายเหตุ: ใช้ synthetic data (GBM + trend regimes) ไม่ใช่ราคาจริง")
    print("ผลเปรียบเทียบ relative ระหว่าง strategy ยังสะท้อนคุณภาพ logic ได้")


if __name__ == "__main__":
    asyncio.run(main())
