"""RSI + MACD combined strategy."""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class RSIMACDStrategy(BaseStrategy):
    """
    Buy:  RSI < oversold threshold AND MACD histogram turns positive.
    Sell: RSI > overbought threshold AND MACD histogram turns negative.
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.macd_fast = self.params.get("macd_fast", 12)
        self.macd_slow = self.params.get("macd_slow", 26)
        self.macd_signal = self.params.get("macd_signal", 9)
        self.position_pct = self.params.get("position_pct", 0.1)

    async def analyze(self, candles: list, current_price: float) -> Signal:
        closes = [c.close for c in candles]
        if len(closes) < self.macd_slow + self.macd_signal + 5:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        rsi_arr = self.rsi(closes, self.rsi_period)
        macd_line, signal_line, histogram = self.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )

        curr_rsi = rsi_arr[-1]
        prev_hist = histogram[-2]
        curr_hist = histogram[-1]

        if np.isnan(curr_rsi) or np.isnan(curr_hist) or np.isnan(prev_hist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        # Buy: oversold + MACD histogram crosses above zero
        if curr_rsi < self.rsi_oversold and prev_hist <= 0 and curr_hist > 0:
            confidence = (self.rsi_oversold - curr_rsi) / self.rsi_oversold
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"RSI oversold ({curr_rsi:.1f}) + MACD bullish crossover",
                confidence=min(1.0, confidence),
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist},
            )

        # Sell: overbought + MACD histogram crosses below zero
        if curr_rsi > self.rsi_overbought and prev_hist >= 0 and curr_hist < 0:
            confidence = (curr_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"RSI overbought ({curr_rsi:.1f}) + MACD bearish crossover",
                confidence=min(1.0, confidence),
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist},
            )

        zone = "oversold" if curr_rsi < 40 else "overbought" if curr_rsi > 60 else "neutral"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"RSI={curr_rsi:.1f} ({zone}), MACD hist={curr_hist:.4f}",
            metadata={"rsi": curr_rsi, "macd_hist": curr_hist},
        )
