"""Basic Momentum Strategy (legacy).

BUY:  RSI rising from oversold + price above EMA
SELL: RSI falling from overbought + price below EMA
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MomentumStrategy(BaseStrategy):
    """Simple momentum based on RSI direction and EMA position."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period   = int(self.params.get("rsi_period", 14))
        self.ema_period   = int(self.params.get("ema_period", 21))
        self.oversold     = float(self.params.get("oversold", 30))
        self.overbought   = float(self.params.get("overbought", 70))
        self.position_pct = float(self.params.get("position_pct", 0.05))

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = max(self.rsi_period, self.ema_period) + 3
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes  = [c.close for c in candles]
        rsi_arr = self.rsi(closes, self.rsi_period)
        ema_arr = self.ema(closes, self.ema_period)

        curr_rsi = float(rsi_arr[-1])
        prev_rsi = float(rsi_arr[-2])
        curr_ema = float(ema_arr[-1])

        if np.isnan(curr_rsi) or np.isnan(curr_ema):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Insufficient data")

        rsi_rising  = curr_rsi > prev_rsi
        rsi_falling = curr_rsi < prev_rsi
        above_ema   = current_price > curr_ema

        if curr_rsi < self.oversold and rsi_rising and above_ema:
            conf = min(1.0, (self.oversold - curr_rsi) / self.oversold * 2)
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[Momentum] RSI={curr_rsi:.1f} rising from oversold, price>EMA{self.ema_period}",
                confidence=conf,
                metadata={"rsi": curr_rsi, "ema": curr_ema},
            )

        if curr_rsi > self.overbought and rsi_falling and not above_ema:
            conf = min(1.0, (curr_rsi - self.overbought) / (100 - self.overbought) * 2)
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[Momentum] RSI={curr_rsi:.1f} falling from overbought, price<EMA{self.ema_period}",
                confidence=conf,
                metadata={"rsi": curr_rsi, "ema": curr_ema},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Momentum] RSI={curr_rsi:.1f} ema={curr_ema:.2f}",
            metadata={"rsi": curr_rsi, "ema": curr_ema},
        )
