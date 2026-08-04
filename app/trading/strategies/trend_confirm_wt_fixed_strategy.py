"""Corrected WaveTrend calculation for the unified Trend Confirm strategy.

The previous implementation fed an array containing leading NaN values into
BaseStrategy.ema(). That helper seeds from the first period, so a NaN seed
propagated through the complete deviation series. The division step then
replaced unavailable values with zero, producing WT1=0 and WT2=0 indefinitely.

This class keeps all Trend Confirm gates, position management and diagnostics
unchanged, and only replaces WaveTrend with a finite-value-aware EMA.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .trend_confirm_wt_strategy import TrendConfirmWTStrategy


class TrendConfirmWTFixedStrategy(TrendConfirmWTStrategy):
    """Trend Confirm with a real, NaN-safe 15M WaveTrend oscillator."""

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        wt_overbought: float = 48.0,
        **kwargs,
    ):
        # Long remains oversold <= -45. Short is now overbought >= +48.
        super().__init__(
            symbol=symbol,
            params=params,
            wt_overbought=wt_overbought,
            **kwargs,
        )

    @staticmethod
    def _ema_finite(values, period: int) -> np.ndarray:
        """EMA that starts after `period` finite observations.

        Leading NaNs are preserved rather than poisoning the whole series.
        Once seeded, later non-finite samples keep the prior EMA value.
        """
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        period = max(1, int(period))
        finite_idx = np.flatnonzero(np.isfinite(arr))
        if finite_idx.size < period:
            return out

        seed_idx = int(finite_idx[period - 1])
        seed_values = arr[finite_idx[:period]]
        out[seed_idx] = float(np.mean(seed_values))
        alpha = 2.0 / (period + 1.0)

        prev = out[seed_idx]
        for i in range(seed_idx + 1, len(arr)):
            if np.isfinite(arr[i]):
                prev = alpha * float(arr[i]) + (1.0 - alpha) * prev
            out[i] = prev
        return out

    @staticmethod
    def _sma_finite(values, period: int) -> np.ndarray:
        """SMA requiring a complete finite rolling window."""
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        period = max(1, int(period))
        for i in range(period - 1, len(arr)):
            window = arr[i - period + 1:i + 1]
            if np.all(np.isfinite(window)):
                out[i] = float(np.mean(window))
        return out

    def _wave_trend(self, candles: list) -> Optional[dict]:
        need = (
            self.wt_channel_length * 2
            + self.wt_average_length
            + self.wt_signal_length
            + 8
        )
        if len(candles) < need:
            return None

        _ha, _ha_open, ha_close = self._heikin_ashi(candles)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        source = (highs + lows + np.asarray(ha_close, dtype=float)) / 3.0

        esa = self._ema_finite(source, self.wt_channel_length)
        abs_deviation = np.abs(source - esa)
        deviation = self._ema_finite(abs_deviation, self.wt_channel_length)

        ci = np.full(source.shape, np.nan, dtype=float)
        valid = (
            np.isfinite(source)
            & np.isfinite(esa)
            & np.isfinite(deviation)
            & (deviation > 1e-12)
        )
        ci[valid] = (source[valid] - esa[valid]) / (0.015 * deviation[valid])

        wt1 = self._ema_finite(ci, self.wt_average_length)
        wt2 = self._sma_finite(wt1, self.wt_signal_length)

        required = (wt1[-2], wt1[-1], wt2[-2], wt2[-1])
        if not all(np.isfinite(v) for v in required):
            return None

        prev_wt1, curr_wt1 = float(wt1[-2]), float(wt1[-1])
        prev_wt2, curr_wt2 = float(wt2[-2]), float(wt2[-1])
        cross_up = prev_wt1 <= prev_wt2 and curr_wt1 > curr_wt2
        cross_down = prev_wt1 >= prev_wt2 and curr_wt1 < curr_wt2
        long_extreme = min(prev_wt1, curr_wt1) <= self.wt_oversold
        short_extreme = max(prev_wt1, curr_wt1) >= self.wt_overbought

        return {
            "wt1": curr_wt1,
            "wt2": curr_wt2,
            "wt1_prev": prev_wt1,
            "wt2_prev": prev_wt2,
            "cross_up": bool(cross_up),
            "cross_down": bool(cross_down),
            "long_extreme": bool(long_extreme),
            "short_extreme": bool(short_extreme),
            "long_trigger": bool(cross_up and long_extreme),
            "short_trigger": bool(cross_down and short_extreme),
            "oversold_level": float(self.wt_oversold),
            "overbought_level": float(self.wt_overbought),
        }
