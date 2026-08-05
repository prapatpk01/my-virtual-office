"""Unified Trend Confirm with symmetric trigger-specific exits.

Responsibilities are deliberately separated:
- Layer 1 (4H) chooses the permitted trading direction only.
- Layer 2 (1H ADX/CHOP/context) decides whether trend quality is tradable only.
- Neither Layer 1 nor Layer 2 may create an order.
- Layer 3 (closed 15M) is the only entry authority:
    * EMA8/13 cross, or
    * WaveTrend extreme cross while price is on the correct side of EMA20.

Signal exits are paired with the trigger that opened the position:
- EMA entry -> opposite EMA8/13 cross (with close beyond EMA13).
- WT entry  -> opposite WT1/WT2 cross; no EMA cross may close it.

Risk management remains independent of the signal exit:
- Initial SL 1.0% and final TP 1.3%.
- T1 at +0.6%: close 40%, move SL on the remaining 60% to +0.3%.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

from .base import SignalType
from .trend_confirm_wt_strategy import TrendConfirmWTStrategy


class TrendConfirmWTFixedStrategy(TrendConfirmWTStrategy):
    """Trend Confirm with NaN-safe WT and trigger-matched exits."""

    ENTRY_EMA = "EMA"
    ENTRY_WT = "WT"
    ENTRY_LEGACY = "LEGACY"

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
        self._active_entry_trigger: Optional[str] = None

        # Keep trigger ownership beside the existing persistent signal-state
        # file. This lets a Railway process restart recover whether a live
        # position was opened by EMA or WT instead of assigning the wrong exit.
        signal_state_path = os.getenv("SIGNAL_STATE_FILE", "/app/signal_state.json")
        state_dir = os.path.dirname(signal_state_path) or "/app"
        self._entry_trigger_state_file = os.getenv(
            "ENTRY_TRIGGER_STATE_FILE",
            os.path.join(state_dir, "trend_confirm_entry_triggers.json"),
        )

    # ------------------------------------------------------------------
    # Entry-trigger persistence
    # ------------------------------------------------------------------

    def _state_key(self) -> str:
        return self.symbol

    def _read_trigger_state(self) -> dict:
        try:
            with open(self._entry_trigger_state_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _persist_entry_trigger(self, trigger: str, direction: str) -> None:
        if trigger not in (self.ENTRY_EMA, self.ENTRY_WT):
            return
        try:
            data = self._read_trigger_state()
            data[self._state_key()] = {
                "trigger": trigger,
                "direction": direction,
                "updated_at": int(time.time()),
            }
            os.makedirs(os.path.dirname(self._entry_trigger_state_file) or ".", exist_ok=True)
            temp_path = f"{self._entry_trigger_state_file}.tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(temp_path, self._entry_trigger_state_file)
        except Exception:
            # Persistence is a recovery aid. It must never block a live order.
            pass

    def _load_entry_trigger(self, direction: Optional[str] = None) -> Optional[str]:
        item = self._read_trigger_state().get(self._state_key(), {})
        if not isinstance(item, dict):
            return None
        trigger = str(item.get("trigger", "")).upper()
        stored_direction = str(item.get("direction", "")).lower()
        if trigger not in (self.ENTRY_EMA, self.ENTRY_WT):
            return None
        if direction and stored_direction and stored_direction != direction.lower():
            return None
        return trigger

    def _clear_entry_trigger(self) -> None:
        try:
            data = self._read_trigger_state()
            if self._state_key() not in data:
                return
            del data[self._state_key()]
            temp_path = f"{self._entry_trigger_state_file}.tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(temp_path, self._entry_trigger_state_file)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Entry routing
    # ------------------------------------------------------------------

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        """Use 4H/1H only as gates; only a closed-15M cross may enter."""
        signal = await super().analyze(candles, current_price, mtf_candles)
        metadata = signal.metadata if isinstance(getattr(signal, "metadata", None), dict) else {}

        if signal.type != SignalType.HOLD:
            raw_trigger = str(metadata.get("entry_trigger", "")).upper()
            trigger = self.ENTRY_WT if "WT" in raw_trigger else self.ENTRY_EMA

            # The WT path already enforces the EMA20 price side. Enforce the
            # same rule on the inherited EMA path so both triggers share the
            # exact Layer-3 location gate requested by the strategy design.
            if trigger == self.ENTRY_EMA:
                c15 = self._closed_candle_series(
                    candles, 15 * 60_000, self.closed_bar_grace_ms
                )
                closes = [float(c.close) for c in c15]
                ema20_arr = self.ema(closes, 20) if closes else np.asarray([])
                ema20 = (
                    float(ema20_arr[-1])
                    if len(ema20_arr) and np.isfinite(ema20_arr[-1])
                    else float("nan")
                )
                direction = "long" if signal.type == SignalType.BUY else "short"
                price_side_ok = (
                    current_price > ema20
                    if direction == "long"
                    else current_price < ema20
                ) if np.isfinite(ema20) else False
                if not price_side_ok:
                    # The parent optimistically marks the position open when it
                    # emits an EMA entry. Roll that state back because EMA20 did
                    # not authorize the Layer-3 trigger.
                    self._reset_position_state()
                    metadata.update({
                        "entry_router": "EMA8_13_OR_WT_EXTREME",
                        "entry_trigger": "WAIT",
                        "ema20_15m": round(ema20, 8) if np.isfinite(ema20) else None,
                        "price_side_ok": False,
                    })
                    return self._hold(
                        current_price,
                        "15M trigger blocked: EMA cross occurred on the wrong side of EMA20",
                        metadata=metadata,
                    )

            direction = "long" if signal.type == SignalType.BUY else "short"
            self._active_entry_trigger = trigger
            self._persist_entry_trigger(trigger, direction)
            metadata.update({
                "entry_trigger_owner": trigger,
                "signal_exit_rule": (
                    "WT_OPPOSITE_CROSS_15M"
                    if trigger == self.ENTRY_WT
                    else "EMA8_13_REVERSE_CROSS_15M"
                ),
                "direction_quality_layers_are_entry_gates_only": True,
                "t1_trigger_pct": self.t1_trigger_pct * 100.0,
                "t1_trim_pct": self.t1_trim_pct * 100.0,
                "t1_lock_pct": self.t1_lock_pct * 100.0,
                "runner_pct_after_t1": (1.0 - self.t1_trim_pct) * 100.0,
                "partial_tp_enabled": True,
                "partial_tp_pct": self.t1_trim_pct * 100.0,
                "tp1_close_pct": self.t1_trim_pct,
            })
            signal.metadata = metadata
            signal.reason += (
                " | exit owner="
                + ("WT opposite cross" if trigger == self.ENTRY_WT else "EMA8/13 reverse cross")
            )
            return signal

        # Keep T1 diagnostics available on HOLD messages without changing the
        # current position's trigger ownership.
        metadata.update({
            "t1_trigger_pct": self.t1_trigger_pct * 100.0,
            "t1_trim_pct": self.t1_trim_pct * 100.0,
            "t1_lock_pct": self.t1_lock_pct * 100.0,
            "runner_pct_after_t1": (1.0 - self.t1_trim_pct) * 100.0,
            "partial_tp_enabled": True,
        })
        signal.metadata = metadata
        return signal

    # ------------------------------------------------------------------
    # Position management: enter with X, exit with X
    # ------------------------------------------------------------------

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Run only the exit indicator that owns the open position."""
        if self._open_position is None:
            return None

        from ..engines.position_manager import PositionUpdate

        trigger = self._active_entry_trigger
        if trigger not in (self.ENTRY_EMA, self.ENTRY_WT):
            trigger = self._load_entry_trigger(self._open_position)
            if trigger in (self.ENTRY_EMA, self.ENTRY_WT):
                self._active_entry_trigger = trigger
            else:
                # A position opened before this deployment has no trustworthy
                # trigger marker. Do not guess and close it with the wrong
                # indicator; hard SL/TP and T1 continue to protect it.
                trigger = self.ENTRY_LEGACY

        candles = self._latest_15m or self._latest_candles
        if candles:
            bar_ts = int(candles[-1].timestamp)
            arm_after = self._reverse_cross_arm_after_ts
            fresh_bar = arm_after is None or bar_ts > int(arm_after)

            if fresh_bar and bar_ts != self._last_exit_bar_ts:
                close_signal = False
                exit_reason = ""

                if trigger == self.ENTRY_EMA:
                    l15 = self._layer3_indicators(candles)
                    if l15 is not None:
                        reverse_cross = (
                            l15["ema_cross_down"]
                            if self._open_position == "long"
                            else l15["ema_cross_up"]
                        )
                        close_px = float(candles[-1].close)
                        ema13 = float(l15["ema_slow_val"])
                        close_confirm = (
                            close_px < ema13
                            if self._open_position == "long"
                            else close_px > ema13
                        )
                        close_signal = bool(reverse_cross and close_confirm)
                        exit_reason = (
                            f"EMA entry owner: fresh 15M EMA{self.ema_fast}/{self.ema_slow} "
                            f"reverse cross + close past EMA{self.ema_slow}"
                        )

                elif trigger == self.ENTRY_WT:
                    wt = self._wave_trend(candles)
                    if wt is not None:
                        # WT exits on the opposite WT cross only. The exit does
                        # not require another overbought/oversold extreme.
                        close_signal = bool(
                            wt["cross_down"]
                            if self._open_position == "long"
                            else wt["cross_up"]
                        )
                        exit_reason = (
                            "WT entry owner: fresh 15M opposite WT1/WT2 cross "
                            f"(WT1={wt['wt1']:.1f}, WT2={wt['wt2']:.1f})"
                        )

                self._last_exit_bar_ts = bar_ts
                if close_signal:
                    side = self._open_position
                    self._last_signal_exit_ts = bar_ts
                    self._last_signal_exit_direction = side
                    self._reset_position_state()
                    return PositionUpdate(
                        action="close",
                        close_pct=1.0,
                        reason=f"{exit_reason} — close {side.upper()}",
                    )

        # T1 is risk management, not an indicator exit. It applies equally to
        # EMA- and WT-owned positions.
        if (
            self.use_be_trail
            and not self._be_trailed
            and self._entry_price is not None
            and self._entry_sl is not None
        ):
            entry_px = float(self._entry_price)
            hit = (
                self._open_position == "long"
                and current_price >= entry_px * (1.0 + self.t1_trigger_pct)
            ) or (
                self._open_position == "short"
                and current_price <= entry_px * (1.0 - self.t1_trigger_pct)
            )
            if hit:
                self._be_trailed = True
                self._tp1_done = True
                new_sl = (
                    entry_px * (1.0 + self.t1_lock_pct)
                    if self._open_position == "long"
                    else entry_px * (1.0 - self.t1_lock_pct)
                )
                return PositionUpdate(
                    action="partial_tp",
                    close_pct=self.t1_trim_pct,
                    new_sl=float(new_sl),
                    reason=(
                        "+0.6% T1 reached — take profit on 40%, move SL to +0.3%; "
                        "keep 60% for final TP or the entry-owner cross exit"
                    ),
                )

        exit_wait = {
            self.ENTRY_EMA: "waiting for EMA8/13 reverse cross",
            self.ENTRY_WT: "waiting for opposite WT1/WT2 cross",
            self.ENTRY_LEGACY: "legacy trigger unknown — indicator exit disabled; SL/TP active",
        }.get(trigger, "waiting for owner exit")
        return PositionUpdate(
            action="hold",
            reason=(
                f"Holding {self._open_position.upper()} [{trigger}] — "
                f"SL 1.0% / TP 1.3% active; {exit_wait}"
            ),
        )

    def attach_existing_position(
        self,
        direction: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Recover entry ownership and prevent a repeated T1 after restart."""
        super().attach_existing_position(direction, entry_price, stop_loss, take_profit)
        self._active_entry_trigger = self._load_entry_trigger(direction)

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

    def _reset_position_state(self) -> None:
        super()._reset_position_state()
        self._active_entry_trigger = None
        self._clear_entry_trigger()

    # ------------------------------------------------------------------
    # NaN-safe WaveTrend
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_finite(values, period: int) -> np.ndarray:
        """EMA that starts after ``period`` finite observations."""
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
