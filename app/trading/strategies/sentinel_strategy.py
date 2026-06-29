"""Sentinel Strategy — trend-following with multi-indicator confirmation.

BUY:  price > EMA, RSI > 50, MACD positive, ADX > threshold
SELL: price < EMA, RSI < 50, MACD negative, ADX > threshold
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class SentinelStrategy(BaseStrategy):
    """Multi-indicator sentinel: requires consensus from trend, momentum, and strength."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ema_period   = int(self.params.get("ema_period", 50))
        self.rsi_period   = int(self.params.get("rsi_period", 14))
        self.adx_period   = int(self.params.get("adx_period", 14))
        self.adx_min      = float(self.params.get("adx_min", 20.0))
        self.macd_fast    = int(self.params.get("macd_fast", 12))
        self.macd_slow    = int(self.params.get("macd_slow", 26))
        self.macd_signal  = int(self.params.get("macd_signal", 9))
        self.position_pct = float(self.params.get("position_pct", 0.05))

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = max(self.ema_period, self.macd_slow + self.macd_signal,
                      self.adx_period * 2) + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes  = [c.close for c in candles]
        ema_arr = self.ema(closes, self.ema_period)
        rsi_arr = self.rsi(closes, self.rsi_period)
        _, _, hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        adx_arr, plus_di, minus_di = self.adx(candles, self.adx_period)

        curr_ema  = float(ema_arr[-1])
        curr_rsi  = float(rsi_arr[-1])
        curr_hist = float(hist[-1])
        curr_adx  = float(adx_arr[-1])

        if any(np.isnan(v) for v in [curr_ema, curr_rsi, curr_hist, curr_adx]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Insufficient data")

        trend_strong = curr_adx > self.adx_min
        bullish = (current_price > curr_ema and curr_rsi > 50 and curr_hist > 0)
        bearish = (current_price < curr_ema and curr_rsi < 50 and curr_hist < 0)

        # Count how many conditions are met (0-3)
        bull_score = (
            int(current_price > curr_ema) +
            int(curr_rsi > 50) +
            int(curr_hist > 0)
        )
        bear_score = (
            int(current_price < curr_ema) +
            int(curr_rsi < 50) +
            int(curr_hist < 0)
        )

        conf = min(1.0, curr_adx / 50)

        if bullish and trend_strong:
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[Sentinel] Bullish consensus (ADX={curr_adx:.1f} RSI={curr_rsi:.1f})",
                confidence=conf,
                metadata={"adx": curr_adx, "rsi": curr_rsi, "ema": curr_ema, "macd_hist": curr_hist},
            )
        if bearish and trend_strong:
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[Sentinel] Bearish consensus (ADX={curr_adx:.1f} RSI={curr_rsi:.1f})",
                confidence=conf,
                metadata={"adx": curr_adx, "rsi": curr_rsi, "ema": curr_ema, "macd_hist": curr_hist},
            )

        bias = "bull" if bull_score > bear_score else "bear" if bear_score > bull_score else "neutral"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Sentinel] {bias} bias score={max(bull_score, bear_score)}/3 ADX={curr_adx:.1f}",
            metadata={"adx": curr_adx, "rsi": curr_rsi, "ema": curr_ema},
        )
