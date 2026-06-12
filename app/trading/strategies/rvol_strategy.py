"""
RVOL (Relative Volume) + MACD + RSI composite strategy.

Fires on volume spikes that confirm directional momentum.
More sensitive than pure RSI+MACD — designed to catch breakouts early.

BUY:  RVOL >= threshold AND MACD > 0 AND RSI 35-65 (rising momentum)
SELL: RVOL >= threshold AND MACD < 0 AND RSI 35-65 (falling momentum)
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class RVolStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rvol_threshold = self.params.get("rvol_threshold", 1.5)  # volume spike multiplier
        self.rvol_period    = self.params.get("rvol_period",    20)    # avg volume period
        self.rsi_period     = self.params.get("rsi_period",     14)
        self.rsi_min        = self.params.get("rsi_min",        30)    # ignore extreme oversold
        self.rsi_max        = self.params.get("rsi_max",        70)    # ignore extreme overbought
        self.position_pct   = self.params.get("position_pct",   0.08)

    async def analyze(self, candles: list, current_price: float) -> Signal:
        if len(candles) < self.rvol_period + 30:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes  = [c.close  for c in candles]
        volumes = [c.volume for c in candles]

        # RVOL — current volume vs rolling average
        vol_arr = np.array(volumes)
        vol_ma  = float(np.mean(vol_arr[-(self.rvol_period + 1):-1]))
        rvol    = float(vol_arr[-1]) / max(vol_ma, 1e-8)

        # RSI
        rsi_arr  = self.rsi(closes, self.rsi_period)
        curr_rsi = float(rsi_arr[-1])
        prev_rsi = float(rsi_arr[-2])

        # MACD histogram
        _, _, hist = self.macd(closes, 12, 26, 9)
        curr_hist = float(hist[-1])
        prev_hist = float(hist[-2])

        if np.isnan(curr_rsi) or np.isnan(curr_hist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        volume_spike = rvol >= self.rvol_threshold
        rsi_in_range = self.rsi_min < curr_rsi < self.rsi_max
        rsi_rising   = curr_rsi > prev_rsi
        rsi_falling  = curr_rsi < prev_rsi
        macd_bull    = curr_hist > 0 or (curr_hist > prev_hist and curr_hist > -0.001)
        macd_bear    = curr_hist < 0 or (curr_hist < prev_hist and curr_hist < 0.001)

        # BUY: volume spike + bullish momentum
        if volume_spike and rsi_in_range and rsi_rising and macd_bull:
            conf = min(1.0, 0.4 + (rvol - self.rvol_threshold) * 0.15 + (curr_rsi - 50) / 100)
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[RVOL] Volume spike {rvol:.1f}x | RSI={curr_rsi:.1f}↑ MACD={'bull'}",
                confidence=max(0.4, conf),
                metadata={"rvol": rvol, "rsi": curr_rsi, "macd_hist": curr_hist},
            )

        # SELL: volume spike + bearish momentum
        if volume_spike and rsi_in_range and rsi_falling and macd_bear:
            conf = min(1.0, 0.4 + (rvol - self.rvol_threshold) * 0.15 + (50 - curr_rsi) / 100)
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[RVOL] Volume spike {rvol:.1f}x | RSI={curr_rsi:.1f}↓ MACD={'bear'}",
                confidence=max(0.4, conf),
                metadata={"rvol": rvol, "rsi": curr_rsi, "macd_hist": curr_hist},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[RVOL] {rvol:.1f}x | RSI={curr_rsi:.1f} | MACD hist={curr_hist:.5f}",
            metadata={"rvol": rvol, "rsi": curr_rsi, "macd_hist": curr_hist},
        )
