"""Moving Average Crossover Strategy.

BUY:  fast EMA crosses above slow EMA
SELL: fast EMA crosses below slow EMA
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MACrossoverStrategy(BaseStrategy):
    """Classic dual-EMA crossover."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.fast = self.params.get("fast", 9)
        self.slow = self.params.get("slow", 21)
        self.position_pct = self.params.get("position_pct", 0.05)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        if len(candles) < self.slow + 2:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]
        fast_ema = self.ema(closes, self.fast)
        slow_ema = self.ema(closes, self.slow)

        curr_fast = float(fast_ema[-1])
        prev_fast = float(fast_ema[-2])
        curr_slow = float(slow_ema[-1])
        prev_slow = float(slow_ema[-2])

        if np.isnan(curr_fast) or np.isnan(curr_slow):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Insufficient data")

        cross_up   = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

        gap_pct = abs(curr_fast - curr_slow) / max(curr_slow, 1e-8) * 100

        if cross_up:
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[MA Cross] EMA{self.fast} crossed above EMA{self.slow} gap={gap_pct:.2f}%",
                confidence=min(1.0, 0.5 + gap_pct * 0.05),
                metadata={"fast": curr_fast, "slow": curr_slow},
            )
        if cross_down:
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[MA Cross] EMA{self.fast} crossed below EMA{self.slow} gap={gap_pct:.2f}%",
                confidence=min(1.0, 0.5 + gap_pct * 0.05),
                metadata={"fast": curr_fast, "slow": curr_slow},
            )

        direction = "above" if curr_fast > curr_slow else "below"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MA Cross] EMA{self.fast} {direction} EMA{self.slow}",
            metadata={"fast": curr_fast, "slow": curr_slow},
        )
