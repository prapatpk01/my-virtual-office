"""Trend Confirm with dual 15M entry trigger.

Architecture is unchanged from TrendConfirmStrategy:
- Layer 1: 4H trend direction.
- Layer 2: 1H aligned context with ADX/CHOP quality gate.
- Layer 3: price must be on the correct side of 15M EMA20, then entry may be
  triggered by either a fresh EMA8/13 cross or a fresh WaveTrend extreme cross.

The class deliberately preserves the public strategy name ``TrendConfirm(...)``
so existing position ownership, quotas, Telegram, stats and reconciliation keep
working as one strategy family.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import Signal, SignalType
from .trend_confirm_strategy import TrendConfirmStrategy


class TrendConfirmWTStrategy(TrendConfirmStrategy):
    """Trend Confirm core with EMA8/13 OR WaveTrend extreme entry."""

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        wt_channel_length: int = 10,
        wt_average_length: int = 21,
        wt_signal_length: int = 4,
        wt_oversold: float = -45.0,
        wt_overbought: float = 53.0,
        **kwargs,
    ):
        super().__init__(symbol=symbol, params=params, **kwargs)
        # Preserve the existing family key used by the dual risk manager.
        self.name = f"TrendConfirm({symbol})"
        self.wt_channel_length = max(2, int(wt_channel_length))
        self.wt_average_length = max(2, int(wt_average_length))
        self.wt_signal_length = max(2, int(wt_signal_length))
        self.wt_oversold = float(wt_oversold)
        self.wt_overbought = float(wt_overbought)

    def _wave_trend(self, candles: list) -> Optional[dict]:
        need = self.wt_channel_length + self.wt_average_length + self.wt_signal_length + 8
        if len(candles) < need:
            return None

        _ha, _ha_open, ha_close = self._heikin_ashi(candles)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        source = (highs + lows + ha_close) / 3.0

        esa = self.ema(list(source), self.wt_channel_length)
        deviation = self.ema(list(np.abs(source - esa)), self.wt_channel_length)
        denominator = 0.015 * deviation
        ci = np.divide(
            source - esa,
            denominator,
            out=np.zeros_like(source, dtype=float),
            where=np.isfinite(denominator) & (np.abs(denominator) > 1e-12),
        )
        wt1 = self.ema(list(ci), self.wt_average_length)
        wt2 = self.sma(list(wt1), self.wt_signal_length)

        if len(wt1) < 2 or any(np.isnan(v) for v in (wt1[-2], wt1[-1], wt2[-2], wt2[-1])):
            return None

        cross_up = bool(wt1[-2] <= wt2[-2] and wt1[-1] > wt2[-1])
        cross_down = bool(wt1[-2] >= wt2[-2] and wt1[-1] < wt2[-1])
        long_extreme = min(float(wt1[-2]), float(wt1[-1])) <= self.wt_oversold
        short_extreme = max(float(wt1[-2]), float(wt1[-1])) >= self.wt_overbought
        return {
            "wt1": float(wt1[-1]),
            "wt2": float(wt2[-1]),
            "cross_up": cross_up,
            "cross_down": cross_down,
            "long_trigger": cross_up and long_extreme,
            "short_trigger": cross_down and short_extreme,
        }

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        # EMA path and all 4H/1H gates remain exactly as implemented by the
        # production TrendConfirmStrategy.
        base_signal = await super().analyze(candles, current_price, mtf_candles)
        if base_signal.type != SignalType.HOLD:
            if isinstance(base_signal.metadata, dict):
                base_signal.metadata["entry_router"] = "EMA8_13_OR_WT_EXTREME"
                base_signal.metadata["entry_trigger"] = "EMA8_13_CROSS"
            return base_signal

        # WT is only an alternative to the EMA entry trigger. It must never
        # bypass neutral 4H, failed 1H context, warm-up, data quality or an
        # already-open position. The inherited method reaches this reason only
        # after all of those gates have passed.
        reason = str(base_signal.reason or "")
        waiting_prefix = f"15M waiting EMA{self.ema_fast}/{self.ema_slow} cross"
        if not reason.startswith(waiting_prefix):
            return base_signal

        mtf = mtf_candles or {}
        c15 = self._closed_candle_series(candles, 15 * 60_000, self.closed_bar_grace_ms)
        if not c15:
            return base_signal

        metadata = base_signal.metadata if isinstance(base_signal.metadata, dict) else {}
        macro = metadata.get("macro_4h") if isinstance(metadata.get("macro_4h"), dict) else {}
        ctx = metadata.get("context_1h") if isinstance(metadata.get("context_1h"), dict) else {}
        direction = macro.get("direction")
        if direction not in ("long", "short") or not bool(ctx.get("ready")):
            return base_signal

        wt = self._wave_trend(c15)
        if wt is None:
            base_signal.reason = "15M waiting EMA8/13 OR WT extreme cross; WT warming up"
            return base_signal

        wt_ok = wt["long_trigger"] if direction == "long" else wt["short_trigger"]
        closes = [float(c.close) for c in c15]
        ema20_arr = self.ema(closes, 20)
        if len(ema20_arr) == 0 or np.isnan(ema20_arr[-1]):
            base_signal.reason = "15M waiting EMA8/13 OR WT extreme cross; EMA20 warming up"
            return base_signal
        ema20 = float(ema20_arr[-1])
        price_side_ok = current_price > ema20 if direction == "long" else current_price < ema20

        if not wt_ok or not price_side_ok:
            side_text = "above" if direction == "long" else "below"
            base_signal.reason = (
                "15M waiting EMA8/13 OR WT extreme cross; "
                f"WT1={wt['wt1']:.1f} WT2={wt['wt2']:.1f}, "
                f"price must be {side_text} EMA20 (price-side={price_side_ok})"
            )
            metadata.update({
                "entry_router": "EMA8_13_OR_WT_EXTREME",
                "entry_trigger": "WAIT",
                "wt_15m": wt,
                "ema20_15m": round(ema20, 8),
                "price_side_ok": price_side_ok,
            })
            base_signal.metadata = metadata
            return base_signal

        bar_ts = int(c15[-1].timestamp)
        bar_open_ms = bar_ts * 1000 if bar_ts < 10_000_000_000 else bar_ts
        age_after_close_ms = max(0, int(time.time() * 1000) - (bar_open_ms + 15 * 60_000))
        if age_after_close_ms > 7 * 60_000:
            base_signal.reason = (
                f"15M WT extreme cross expired ({age_after_close_ms / 60_000:.1f}m after close) — "
                "wait for a NEW trigger"
            )
            return base_signal
        if self._last_entry_attempt_bar_ts == bar_ts:
            base_signal.reason = "15M entry trigger already processed — waiting for a new EMA or WT cross"
            return base_signal

        atr_arr = self.atr(c15, self.atr_period)
        atr15 = float(atr_arr[-1]) if len(atr_arr) and not np.isnan(atr_arr[-1]) else 0.0
        if atr15 <= 0:
            return self._hold(current_price, "15M ATR unavailable")

        sma30_arr = self.sma(closes, 30)
        sma30 = float(sma30_arr[-1]) if len(sma30_arr) and not np.isnan(sma30_arr[-1]) else float("nan")
        if np.isnan(sma30):
            return self._hold(current_price, "15M SMA30 anti-chase warming up")
        chase_distance_atr = abs(float(current_price) - sma30) / atr15
        if chase_distance_atr > 1.20:
            return self._hold(
                current_price,
                f"15M WT anti-chase: price is {chase_distance_atr:.2f}ATR from SMA30 (>1.20ATR)",
                metadata={**metadata, "wt_15m": wt, "chase_distance_atr": round(chase_distance_atr, 3)},
            )

        # Keep Trend Confirm's existing live position plan unchanged.
        entry_px = float(current_price)
        sl_pct = 0.010
        tp_pct = 0.013
        if direction == "long":
            sl = entry_px * (1.0 - sl_pct)
            tp = entry_px * (1.0 + tp_pct)
        else:
            sl = entry_px * (1.0 + sl_pct)
            tp = entry_px * (1.0 - tp_pct)

        self._last_entry_attempt_bar_ts = bar_ts
        self._open_position = direction
        self._entry_price = entry_px
        self._entry_sl = float(sl)
        self._entry_bar_ts = bar_ts
        self._reverse_cross_arm_after_ts = bar_ts
        self._adopted_after_restart = False
        self._entry_regime = str(macro.get("state", "TREND"))
        self._tp1_done = False
        self._be_trailed = False
        self._last_exit_bar_ts = None

        trigger_side = "UP" if direction == "long" else "DOWN"
        self._diag_update(
            entry_state="ENTRY_READY_WT",
            direction_15m=f"WT_{trigger_side}",
            aligned=True,
            strategy="EMA_OR_WT_15M",
        )
        metadata = {
            **metadata,
            **self._diag_context,
            "strategy": "TREND_CONFIRM_EMA_OR_WT",
            "entry_type": "WT_EXTREME_CROSS_15M",
            "entry_router": "EMA8_13_OR_WT_EXTREME",
            "entry_trigger": "WT_EXTREME_CROSS",
            "entry_tf": "15m",
            "stop_loss": round(float(sl), 8),
            "take_profit": round(float(tp), 8),
            "rr_ratio": 1.3,
            "sl_pct": 1.0,
            "tp_pct": 1.3,
            "trail_trigger_pct": 0.6,
            "trail_lock_pct": 0.3,
            "wt_15m": wt,
            "ema20_15m": round(ema20, 8),
            "price_side_ok": True,
            "sma30_15m": round(sma30, 8),
            "chase_distance_atr": round(chase_distance_atr, 3),
        }
        return Signal(
            type=SignalType.BUY if direction == "long" else SignalType.SELL,
            symbol=self.symbol,
            price=entry_px,
            amount=0.0,
            reason=(
                f"4H/1H trend aligned + 15M WT extreme cross {trigger_side} "
                f"(WT1={wt['wt1']:.1f}, WT2={wt['wt2']:.1f}) + price correct side of EMA20"
            ),
            confidence=0.72,
            metadata=metadata,
        )
