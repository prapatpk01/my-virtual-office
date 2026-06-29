"""RSI + MACD Combination Strategy.

BUY:  RSI < oversold AND MACD histogram turns positive
SELL: RSI > overbought AND MACD histogram turns negative
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class RSIMACDStrategy(BaseStrategy):
    """Combines RSI oversold/overbought with MACD histogram direction change."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period   = self.params.get("rsi_period", 14)
        self.rsi_oversold = self.params.get("rsi_oversold", 35)
        self.rsi_overbought = self.params.get("rsi_overbought", 65)
        self.macd_fast    = self.params.get("macd_fast", 12)
        self.macd_slow    = self.params.get("macd_slow", 26)
        self.macd_signal  = self.params.get("macd_signal", 9)
        self.position_pct = self.params.get("position_pct", 0.05)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_signal + self.rsi_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]
        rsi_arr = self.rsi(closes, self.rsi_period)
        _, _, hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)

        curr_rsi  = float(rsi_arr[-1])
        curr_hist = float(hist[-1])
        prev_hist = float(hist[-2])

        if np.isnan(curr_rsi) or np.isnan(curr_hist) or np.isnan(prev_hist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Insufficient data")

        hist_cross_up   = prev_hist <= 0 and curr_hist > 0
        hist_cross_down = prev_hist >= 0 and curr_hist < 0

        if curr_rsi < self.rsi_oversold and hist_cross_up:
            conf = min(1.0, (self.rsi_oversold - curr_rsi) / self.rsi_oversold * 2)
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[RSI+MACD] RSI={curr_rsi:.1f} oversold + MACD hist turned positive",
                confidence=conf,
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist},
            )

        if curr_rsi > self.rsi_overbought and hist_cross_down:
            conf = min(1.0, (curr_rsi - self.rsi_overbought) / (100 - self.rsi_overbought) * 2)
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[RSI+MACD] RSI={curr_rsi:.1f} overbought + MACD hist turned negative",
                confidence=conf,
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist},
            )

        zone = ("oversold" if curr_rsi < self.rsi_oversold
                else "overbought" if curr_rsi > self.rsi_overbought
                else "neutral")
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[RSI+MACD] RSI={curr_rsi:.1f} ({zone}) hist={curr_hist:.4f}",
            metadata={"rsi": curr_rsi, "macd_hist": curr_hist},
        )
