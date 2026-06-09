"""MA Crossover strategy — Golden Cross / Death Cross."""
from .base import BaseStrategy, Signal, SignalType


class MACrossoverStrategy(BaseStrategy):
    """
    Buy when fast EMA crosses above slow EMA (Golden Cross).
    Sell when fast EMA crosses below slow EMA (Death Cross).
    Default: EMA 9 / EMA 21.
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.fast = self.params.get("fast_period", 9)
        self.slow = self.params.get("slow_period", 21)
        self.position_pct = self.params.get("position_pct", 0.1)  # 10% of balance

    async def analyze(self, candles: list, current_price: float) -> Signal:
        closes = [c.close for c in candles]
        if len(closes) < self.slow + 2:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        fast_ema = self.ema(closes, self.fast)
        slow_ema = self.ema(closes, self.slow)

        prev_fast, prev_slow = fast_ema[-2], slow_ema[-2]
        curr_fast, curr_slow = fast_ema[-1], slow_ema[-1]

        # Golden Cross
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            diff_pct = abs(curr_fast - curr_slow) / curr_slow * 100
            confidence = min(1.0, diff_pct / 0.5)
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"Golden Cross EMA{self.fast}/EMA{self.slow} | fast={curr_fast:.4f} slow={curr_slow:.4f}",
                confidence=confidence,
                metadata={"fast_ema": curr_fast, "slow_ema": curr_slow},
            )

        # Death Cross
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            diff_pct = abs(curr_fast - curr_slow) / curr_slow * 100
            confidence = min(1.0, diff_pct / 0.5)
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"Death Cross EMA{self.fast}/EMA{self.slow} | fast={curr_fast:.4f} slow={curr_slow:.4f}",
                confidence=confidence,
                metadata={"fast_ema": curr_fast, "slow_ema": curr_slow},
            )

        trend = "bullish" if curr_fast > curr_slow else "bearish"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"No crossover — trend {trend} | EMA{self.fast}={curr_fast:.4f} EMA{self.slow}={curr_slow:.4f}",
            metadata={"fast_ema": curr_fast, "slow_ema": curr_slow},
        )
