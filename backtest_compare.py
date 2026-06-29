"""
Backtest: MCDXStrategy (Adaptive Trading Bot)
Uses synthetic BTC-like candle data (GBM + trend regimes) — 5 760 bars ≈ 60 days 15m.
"""
import asyncio
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy


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


async def run_mcdx_backtest(candles: list[OHLCV], warmup: int = 120) -> dict:
    """Run a simple signal-count backtest on MCDXStrategy."""
    strat = MCDXStrategy("BTCUSD")
    buys = sells = holds = 0
    for i in range(warmup, len(candles)):
        window = candles[:i + 1]
        sig = await strat.analyze(window, window[-1].close)
        if sig.type.value == "buy":
            buys += 1
        elif sig.type.value == "sell":
            sells += 1
        else:
            holds += 1
    total = buys + sells + holds
    return {"buys": buys, "sells": sells, "holds": holds, "total": total}


async def main():
    candles = generate_candles(5760)
    print(f"\nSynthetic BTC-like data: {len(candles)} bars (≈60 days 15m)")
    print(f"Price range: ${min(c.low for c in candles):,.0f} – ${max(c.high for c in candles):,.0f}\n")

    print("═" * 60)
    print("  BACKTEST — MCDXStrategy (Adaptive Trading Bot) — synthetic BTC/USD 15m")
    print("═" * 60 + "\n")

    stats = await run_mcdx_backtest(candles)
    total = stats["total"] or 1
    print(f"  Bars analysed : {total}")
    print(f"  BUY  signals  : {stats['buys']}  ({stats['buys']/total*100:.1f}%)")
    print(f"  SELL signals  : {stats['sells']}  ({stats['sells']/total*100:.1f}%)")
    print(f"  HOLD signals  : {stats['holds']}  ({stats['holds']/total*100:.1f}%)")

    print("\n" + "═" * 60)
    print("\nหมายเหตุ: ใช้ synthetic data (GBM + trend regimes) ไม่ใช่ราคาจริง")


if __name__ == "__main__":
    asyncio.run(main())
