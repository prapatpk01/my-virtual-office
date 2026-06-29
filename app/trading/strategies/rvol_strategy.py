"""Relative Volume (RVOL) Strategy.

BUY:  RVOL spike UP with price rising (bullish volume expansion)
SELL: RVOL spike UP with price falling (bearish volume expansion)

RVOL = current_volume / SMA(volume, period)
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class RVolStrategy(BaseStrategy):
    """Signals on unusual relative volume combined with price direction."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.vol_period   = int(self.params.get("vol_period", 20))
        self.rvol_thresh  = float(self.params.get("rvol_thresh", 1.5))  # 150% of average
        self.ema_period   = int(self.params.get("ema_period", 20))
        self.position_pct = float(self.params.get("position_pct", 0.05))

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        if len(candles) < self.vol_period + 2:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes  = np.array([c.close  for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        vol_sma = self.sma(volumes.tolist(), self.vol_period)
        ema_arr = self.ema(closes.tolist(), self.ema_period)

        curr_vol     = float(volumes[-1])
        avg_vol      = float(vol_sma[-1])
        curr_ema     = float(ema_arr[-1])
        prev_close   = float(closes[-2])

        if np.isnan(avg_vol) or avg_vol <= 0 or np.isnan(curr_ema):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Insufficient data")

        rvol = curr_vol / avg_vol
        price_up   = current_price > prev_close
        above_ema  = current_price > curr_ema
        conf = min(1.0, (rvol - 1.0) / max(self.rvol_thresh - 1.0, 0.1))

        if rvol >= self.rvol_thresh and price_up and above_ema:
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[RVol] Bullish volume spike RVOL={rvol:.2f}x price above EMA{self.ema_period}",
                confidence=conf,
                metadata={"rvol": rvol, "vol": curr_vol, "avg_vol": avg_vol},
            )

        if rvol >= self.rvol_thresh and not price_up and not above_ema:
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[RVol] Bearish volume spike RVOL={rvol:.2f}x price below EMA{self.ema_period}",
                confidence=conf,
                metadata={"rvol": rvol, "vol": curr_vol, "avg_vol": avg_vol},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[RVol] RVOL={rvol:.2f}x (threshold={self.rvol_thresh}x)",
            metadata={"rvol": rvol},
        )
