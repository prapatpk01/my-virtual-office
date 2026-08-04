"""Corrected WaveTrend and T1 management for unified Trend Confirm.

Architecture remains one strategy family:
- Layer 1: 4H trend direction.
- Layer 2: 1H context with ADX/CHOP quality gate.
- Layer 3: 15M EMA8/13 cross OR WaveTrend extreme cross, with price on the
  correct side of EMA20.

WaveTrend is finite-value aware, so leading NaNs cannot poison WT1/WT2.
Position management is shared by EMA and WT entries:
- T1 at +0.6% from entry.
- Close 40% at T1.
- Move SL on the remaining 60% to +0.3% from entry.
- Keep the remaining size toward the +1.3% final TP or EMA cross-back exit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .trend_confirm_wt_strategy import TrendConfirmWTStrategy


class TrendConfirmWTFixedStrategy(TrendConfirmWTStrategy):
    """Trend Confirm with NaN-safe WT and a 40% T1 profit trim."""

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        wt_overbought: float = 48.0,
        **kwargs,
    ):
        # Long remains oversold <= -45. Short is overbought >= +48.
        super().__init__(
            symbol=symbol,
            params=params,
            wt_overbought=wt_overbought,
            **kwargs,
        )
        self.t1_trigger_pct = 0.006
        self.t1_trim_pct = 0.40
        self.t1_lock_pct = 0.003

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        """Attach the live T1 plan to every EMA- or WT-generated signal."""
        signal = await super().analyze(candles, current_price, mtf_candles)
        if isinstance(getattr(signal, "metadata", None), dict):
            signal.metadata.update({
                "t1_trigger_pct": self.t1_trigger_pct * 100.0,
                "t1_trim_pct": self.t1_trim_pct * 100.0,
                "t1_lock_pct": self.t1_lock_pct * 100.0,
                "runner_pct_after_t1": (1.0 - self.t1_trim_pct) * 100.0,
                "partial_tp_enabled": True,
                "partial_tp_pct": self.t1_trim_pct * 100.0,
                "tp1_close_pct": self.t1_trim_pct,
            })
        return signal

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Convert Trend Confirm's T1 SL-only update into trim + SL lock.

        The inherited manager validates the +0.6% trigger, consumes T1 and
        calculates the correct +0.3% stop for long/short. The bot's existing
        ``partial_tp`` path closes 40%, updates accounting and re-places the
        SL/TP on the remaining 60%.
        """
        update = super().tick_open_position(current_price, position_key)
        if update is None:
            return None

        # The inherited Trend Confirm emits move_sl only for its +0.6% T1.
        if update.action == "move_sl" and update.new_sl is not None:
            self._tp1_done = True
            update.action = "partial_tp"
            update.close_pct = self.t1_trim_pct
            update.reason = (
                "+0.6% T1 reached — take profit on 40%, move SL to +0.3%; "
                "keep the remaining 60% toward the +1.3% final TP or EMA cross-back"
            )
        return update

    def attach_existing_position(
        self,
        direction: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Recover a live position without repeating an already completed T1.

        After T1, the exchange SL is moved to roughly +0.3% from entry. If that
        protected SL is present during restart reconciliation, mark the T1 trim
        as completed so the remaining 60% cannot be trimmed again.
        """
        super().attach_existing_position(direction, entry_price, stop_loss, take_profit)
        if stop_loss is None or entry_price <= 0:
            return

        lock_tolerance = 0.0002  # tolerate exchange tick-size rounding
        if direction == "long":
            protected = stop_loss >= entry_price * (
                1.0 + self.t1_lock_pct - lock_tolerance
            )
        else:
            protected = stop_loss <= entry_price * (
                1.0 - self.t1_lock_pct + lock_tolerance
            )
        if protected:
            self._tp1_done = True
            self._be_trailed = True

    @staticmethod
    def _ema_finite(values, period: int) -> np.ndarray:
        """EMA that starts after ``period`` finite observations.

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
