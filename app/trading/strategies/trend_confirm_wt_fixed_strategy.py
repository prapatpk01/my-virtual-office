"""Unified Trend Confirm with three trigger-specific entry/exit engines.

Layer responsibilities:
- 4H chooses the permitted direction only.
- 1H ADX/CHOP/context decides whether quality is tradable only.
- Closed 15M is the only entry authority:
    * EMA8/13 cross,
    * WaveTrend extreme cross, or
    * confirmed Structure BOS + retest.

Each position keeps its entry owner across Railway restarts:
- EMA entry -> EMA8/13 reverse cross exit.
- WT entry -> opposite WT1/WT2 cross exit.
- Structure entry -> BOS/retest invalidation or opposite 15M CHOCH.

Risk management remains independent:
- Initial SL 1.0%, final TP 1.3%.
- T1 at +0.6% closes 40% and moves the remaining 60% SL to +0.3%.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

from .base import Signal, SignalType
from .trend_confirm_wt_strategy import TrendConfirmWTStrategy


class TrendConfirmWTFixedStrategy(TrendConfirmWTStrategy):
    """Trend Confirm with EMA, WT and Structure entry-owner exits."""

    ENTRY_EMA = "EMA"
    ENTRY_WT = "WT"
    ENTRY_STRUCTURE = "STRUCTURE"
    ENTRY_LEGACY = "LEGACY"
    VALID_ENTRY_TRIGGERS = (ENTRY_EMA, ENTRY_WT, ENTRY_STRUCTURE)

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        wt_overbought: float = 48.0,
        structure_swing_span: int = 3,
        structure_retest_min_bars: int = 1,
        structure_retest_max_bars: int = 3,
        structure_bos_buffer_atr: float = 0.05,
        structure_touch_tolerance_atr: float = 0.15,
        structure_invalidation_tolerance_atr: float = 0.25,
        structure_max_close_distance_atr: float = 0.50,
        structure_max_fill_slippage_atr: float = 0.35,
        **kwargs,
    ):
        super().__init__(
            symbol=symbol,
            params=params,
            wt_overbought=wt_overbought,
            **kwargs,
        )
        self.t1_trigger_pct = 0.006
        self.t1_trim_pct = 0.40
        self.t1_lock_pct = 0.003

        self.structure_swing_span = max(2, int(structure_swing_span))
        self.structure_retest_min_bars = max(1, int(structure_retest_min_bars))
        self.structure_retest_max_bars = max(
            self.structure_retest_min_bars,
            int(structure_retest_max_bars),
        )
        self.structure_bos_buffer_atr = max(0.0, float(structure_bos_buffer_atr))
        self.structure_touch_tolerance_atr = max(
            0.0, float(structure_touch_tolerance_atr)
        )
        self.structure_invalidation_tolerance_atr = max(
            0.0, float(structure_invalidation_tolerance_atr)
        )
        self.structure_max_close_distance_atr = max(
            0.05, float(structure_max_close_distance_atr)
        )
        self.structure_max_fill_slippage_atr = max(
            0.05, float(structure_max_fill_slippage_atr)
        )

        self._active_entry_trigger: Optional[str] = None
        self._active_structure_level: Optional[float] = None
        self._active_structure_breakout_ts: Optional[int] = None
        self._active_structure_retest_ts: Optional[int] = None
        self._active_structure_entry_ts: Optional[int] = None

        signal_state_path = os.getenv("SIGNAL_STATE_FILE", "/app/signal_state.json")
        state_dir = os.path.dirname(signal_state_path) or "/app"
        self._entry_trigger_state_file = os.getenv(
            "ENTRY_TRIGGER_STATE_FILE",
            os.path.join(state_dir, "trend_confirm_entry_triggers.json"),
        )

    # ------------------------------------------------------------------
    # Entry-owner persistence
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

    def _persist_entry_trigger(
        self,
        trigger: str,
        direction: str,
        structure_context: Optional[dict] = None,
    ) -> None:
        if trigger not in self.VALID_ENTRY_TRIGGERS:
            return
        try:
            data = self._read_trigger_state()
            item = {
                "trigger": trigger,
                "direction": direction,
                "updated_at": int(time.time()),
            }
            if trigger == self.ENTRY_STRUCTURE and structure_context:
                item.update({
                    "structure_level": float(structure_context["level"]),
                    "structure_breakout_ts": int(structure_context["breakout_ts"]),
                    "structure_retest_ts": int(structure_context["retest_ts"]),
                    "structure_entry_ts": int(structure_context["entry_ts"]),
                    "structure_swing_index": int(
                        structure_context.get("swing_index", -1)
                    ),
                })
            data[self._state_key()] = item
            os.makedirs(
                os.path.dirname(self._entry_trigger_state_file) or ".",
                exist_ok=True,
            )
            temp_path = f"{self._entry_trigger_state_file}.tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(temp_path, self._entry_trigger_state_file)
        except Exception:
            pass

    def _load_entry_state(self, direction: Optional[str] = None) -> dict:
        item = self._read_trigger_state().get(self._state_key(), {})
        if not isinstance(item, dict):
            return {}
        trigger = str(item.get("trigger", "")).upper()
        stored_direction = str(item.get("direction", "")).lower()
        if trigger not in self.VALID_ENTRY_TRIGGERS:
            return {}
        if direction and stored_direction and stored_direction != direction.lower():
            return {}
        return item

    def _load_entry_trigger(self, direction: Optional[str] = None) -> Optional[str]:
        item = self._load_entry_state(direction)
        trigger = str(item.get("trigger", "")).upper()
        return trigger if trigger in self.VALID_ENTRY_TRIGGERS else None

    def _restore_structure_context(self, item: dict) -> None:
        if str(item.get("trigger", "")).upper() != self.ENTRY_STRUCTURE:
            self._active_structure_level = None
            self._active_structure_breakout_ts = None
            self._active_structure_retest_ts = None
            self._active_structure_entry_ts = None
            return
        try:
            self._active_structure_level = float(item["structure_level"])
            self._active_structure_breakout_ts = int(item["structure_breakout_ts"])
            self._active_structure_retest_ts = int(item["structure_retest_ts"])
            self._active_structure_entry_ts = int(item["structure_entry_ts"])
        except (KeyError, TypeError, ValueError):
            self._active_structure_level = None
            self._active_structure_breakout_ts = None
            self._active_structure_retest_ts = None
            self._active_structure_entry_ts = None

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
    # Structure engine
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_timestamp_ms(timestamp: int) -> int:
        value = int(timestamp)
        return value * 1000 if value < 10_000_000_000 else value

    def _confirmed_swings(
        self,
        candles: list,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Return confirmed pivots without using future data at BOS time."""
        span = self.structure_swing_span
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []

        for index in range(span, len(candles) - span):
            left_high = highs[index - span:index]
            right_high = highs[index + 1:index + span + 1]
            left_low = lows[index - span:index]
            right_low = lows[index + 1:index + span + 1]

            if (
                highs[index] > float(np.max(left_high))
                and highs[index] >= float(np.max(right_high))
            ):
                swing_highs.append((index, float(highs[index])))
            if (
                lows[index] < float(np.min(left_low))
                and lows[index] <= float(np.min(right_low))
            ):
                swing_lows.append((index, float(lows[index])))

        return swing_highs, swing_lows

    def _structure_setup(
        self,
        candles: list,
        direction: str,
        current_price: float,
    ) -> dict:
        """Find a current-bar BOS retest formed 1-3 bars after a fresh BOS."""
        span = self.structure_swing_span
        minimum = max(40, span * 2 + self.structure_retest_max_bars + 8)
        if len(candles) < minimum:
            return {
                "trigger": False,
                "reason": f"warming up ({len(candles)}/{minimum} bars)",
            }

        opens = np.asarray([float(c.open) for c in candles], dtype=float)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        closes = np.asarray([float(c.close) for c in candles], dtype=float)
        atr_arr = self.atr(candles, self.atr_period)
        ema20_arr = self.ema(list(closes), 20)

        if (
            len(atr_arr) != len(candles)
            or len(ema20_arr) != len(candles)
            or not np.isfinite(atr_arr[-1])
            or float(atr_arr[-1]) <= 0
            or not np.isfinite(ema20_arr[-1])
        ):
            return {"trigger": False, "reason": "ATR/EMA20 warming up"}

        current_atr = float(atr_arr[-1])
        ema20 = float(ema20_arr[-1])
        price_side_ok = (
            current_price > ema20 if direction == "long" else current_price < ema20
        )

        swing_highs, swing_lows = self._confirmed_swings(candles)
        swings = swing_highs if direction == "long" else swing_lows
        retest_index = len(candles) - 1
        best_diagnostic: Optional[dict] = None

        for bars_after_bos in range(
            self.structure_retest_min_bars,
            self.structure_retest_max_bars + 1,
        ):
            breakout_index = retest_index - bars_after_bos
            if breakout_index <= span or breakout_index >= retest_index:
                continue

            eligible = [
                (index, level)
                for index, level in swings
                if index + span < breakout_index
            ]
            if not eligible:
                continue

            swing_index, level = eligible[-1]
            breakout_atr = (
                float(atr_arr[breakout_index])
                if np.isfinite(atr_arr[breakout_index])
                and float(atr_arr[breakout_index]) > 0
                else current_atr
            )
            bos_buffer = self.structure_bos_buffer_atr * breakout_atr

            if direction == "long":
                fresh_bos = (
                    closes[breakout_index] > level + bos_buffer
                    and closes[breakout_index - 1] <= level
                    and closes[breakout_index] > opens[breakout_index]
                )
                touched = lows[retest_index] <= (
                    level + self.structure_touch_tolerance_atr * current_atr
                )
                not_too_deep = lows[retest_index] >= (
                    level - self.structure_invalidation_tolerance_atr * current_atr
                )
                recaptured = closes[retest_index] > level
                distance_atr = max(
                    0.0, (closes[retest_index] - level) / current_atr
                )
            else:
                fresh_bos = (
                    closes[breakout_index] < level - bos_buffer
                    and closes[breakout_index - 1] >= level
                    and closes[breakout_index] < opens[breakout_index]
                )
                touched = highs[retest_index] >= (
                    level - self.structure_touch_tolerance_atr * current_atr
                )
                not_too_deep = highs[retest_index] <= (
                    level + self.structure_invalidation_tolerance_atr * current_atr
                )
                recaptured = closes[retest_index] < level
                distance_atr = max(
                    0.0, (level - closes[retest_index]) / current_atr
                )

            candle_range = max(
                float(highs[retest_index] - lows[retest_index]),
                current_atr * 0.05,
            )
            close_location = (
                (closes[retest_index] - lows[retest_index]) / candle_range
                if direction == "long"
                else (highs[retest_index] - closes[retest_index]) / candle_range
            )
            direction_candle = (
                closes[retest_index] >= opens[retest_index]
                if direction == "long"
                else closes[retest_index] <= opens[retest_index]
            )
            rejection_confirmed = bool(
                direction_candle and close_location >= 0.55
            )
            close_near_level = (
                distance_atr <= self.structure_max_close_distance_atr
            )
            fill_slippage_atr = (
                abs(float(current_price) - float(closes[retest_index])) / current_atr
            )
            fill_is_fresh = (
                fill_slippage_atr <= self.structure_max_fill_slippage_atr
            )

            diagnostic = {
                "trigger": False,
                "direction": direction,
                "level": round(float(level), 8),
                "swing_index": int(swing_index),
                "breakout_index": int(breakout_index),
                "breakout_ts": int(candles[breakout_index].timestamp),
                "retest_ts": int(candles[retest_index].timestamp),
                "bars_after_bos": int(bars_after_bos),
                "fresh_bos": bool(fresh_bos),
                "touched": bool(touched),
                "not_too_deep": bool(not_too_deep),
                "recaptured": bool(recaptured),
                "rejection_confirmed": bool(rejection_confirmed),
                "price_side_ok": bool(price_side_ok),
                "close_distance_atr": round(float(distance_atr), 3),
                "fill_slippage_atr": round(float(fill_slippage_atr), 3),
                "ema20": round(float(ema20), 8),
                "atr": round(float(current_atr), 8),
            }

            if fresh_bos and best_diagnostic is None:
                best_diagnostic = diagnostic

            trigger = all((
                fresh_bos,
                touched,
                not_too_deep,
                recaptured,
                rejection_confirmed,
                close_near_level,
                fill_is_fresh,
                price_side_ok,
            ))
            if trigger:
                diagnostic["trigger"] = True
                diagnostic["reason"] = (
                    f"{direction.upper()} BOS {bars_after_bos} bar(s) ago "
                    f"+ confirmed retest of {level:.6f}"
                )
                return diagnostic

        if best_diagnostic is not None:
            failed = [
                name
                for name in (
                    "touched",
                    "not_too_deep",
                    "recaptured",
                    "rejection_confirmed",
                    "price_side_ok",
                )
                if not best_diagnostic.get(name)
            ]
            if (
                best_diagnostic.get("close_distance_atr", 0)
                > self.structure_max_close_distance_atr
            ):
                failed.append("close_too_far")
            if (
                best_diagnostic.get("fill_slippage_atr", 0)
                > self.structure_max_fill_slippage_atr
            ):
                failed.append("fill_too_late")
            best_diagnostic["reason"] = "retest incomplete: " + ", ".join(failed)
            return best_diagnostic

        return {
            "trigger": False,
            "direction": direction,
            "price_side_ok": bool(price_side_ok),
            "ema20": round(float(ema20), 8),
            "atr": round(float(current_atr), 8),
            "reason": (
                f"no fresh {direction.upper()} BOS in the last "
                f"{self.structure_retest_min_bars}-"
                f"{self.structure_retest_max_bars} bars"
            ),
        }

    def _opposite_structure_choch(
        self,
        candles: list,
        direction: str,
        entry_ts: Optional[int],
    ) -> Optional[dict]:
        """Detect a fresh opposite CHOCH from a swing formed after entry."""
        if len(candles) < max(20, self.structure_swing_span * 2 + 6):
            return None

        swing_highs, swing_lows = self._confirmed_swings(candles)
        swings = swing_lows if direction == "long" else swing_highs
        current_index = len(candles) - 1
        entry_ms = self._bar_timestamp_ms(entry_ts) if entry_ts is not None else None

        eligible: list[tuple[int, float]] = []
        for index, level in swings:
            if index + self.structure_swing_span >= current_index:
                continue
            pivot_ms = self._bar_timestamp_ms(candles[index].timestamp)
            if entry_ms is not None and pivot_ms <= entry_ms:
                continue
            eligible.append((index, level))

        if not eligible:
            return None

        swing_index, level = eligible[-1]
        previous_close = float(candles[-2].close)
        current_close = float(candles[-1].close)
        if direction == "long":
            crossed = previous_close >= level and current_close < level
            choch_side = "BEARISH"
        else:
            crossed = previous_close <= level and current_close > level
            choch_side = "BULLISH"

        if not crossed:
            return None
        return {
            "side": choch_side,
            "level": float(level),
            "swing_index": int(swing_index),
            "swing_ts": int(candles[swing_index].timestamp),
        }

    # ------------------------------------------------------------------
    # Entry routing
    # ------------------------------------------------------------------

    def _finalize_indicator_entry(
        self,
        signal: Signal,
        candles: list,
        current_price: float,
    ) -> Signal:
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        raw_trigger = str(metadata.get("entry_trigger", "")).upper()
        trigger = self.ENTRY_WT if "WT" in raw_trigger else self.ENTRY_EMA
        direction = "long" if signal.type == SignalType.BUY else "short"

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
            price_side_ok = (
                current_price > ema20
                if direction == "long"
                else current_price < ema20
            ) if np.isfinite(ema20) else False
            if not price_side_ok:
                self._reset_position_state()
                metadata.update({
                    "entry_router": (
                        "EMA8_13_OR_WT_EXTREME_OR_STRUCTURE_BOS_RETEST"
                    ),
                    "entry_trigger": "WAIT",
                    "ema20_15m": (
                        round(ema20, 8) if np.isfinite(ema20) else None
                    ),
                    "price_side_ok": False,
                })
                return self._hold(
                    current_price,
                    "15M EMA trigger blocked: cross occurred on wrong EMA20 side",
                    metadata=metadata,
                )

        self._active_entry_trigger = trigger
        self._active_structure_level = None
        self._active_structure_breakout_ts = None
        self._active_structure_retest_ts = None
        self._active_structure_entry_ts = None
        self._persist_entry_trigger(trigger, direction)

        exit_rule = (
            "WT_OPPOSITE_CROSS_15M"
            if trigger == self.ENTRY_WT
            else "EMA8_13_REVERSE_CROSS_15M"
        )
        exit_text = (
            "WT opposite cross"
            if trigger == self.ENTRY_WT
            else "EMA8/13 reverse cross"
        )
        metadata.update({
            "strategy": "TREND_CONFIRM_EMA_WT_STRUCTURE",
            "entry_router": "EMA8_13_OR_WT_EXTREME_OR_STRUCTURE_BOS_RETEST",
            "entry_trigger_owner": trigger,
            "signal_exit_rule": exit_rule,
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
        signal.reason += f" | exit owner={exit_text}"
        return signal

    def _build_structure_entry(
        self,
        current_price: float,
        metadata: dict,
        macro: dict,
        direction: str,
        candles: list,
        setup: dict,
    ) -> Signal:
        bar_ts = int(candles[-1].timestamp)
        bar_open_ms = self._bar_timestamp_ms(bar_ts)
        age_after_close_ms = max(
            0,
            int(time.time() * 1000) - (bar_open_ms + 15 * 60_000),
        )
        if age_after_close_ms > 7 * 60_000:
            return self._hold(
                current_price,
                (
                    "15M Structure BOS+retest expired "
                    f"({age_after_close_ms / 60_000:.1f}m after close)"
                ),
                metadata={**metadata, "structure_15m": setup},
            )
        if self._last_entry_attempt_bar_ts == bar_ts:
            return self._hold(
                current_price,
                "15M Structure trigger already processed — wait for a new retest",
                metadata={**metadata, "structure_15m": setup},
            )

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

        self._active_entry_trigger = self.ENTRY_STRUCTURE
        self._active_structure_level = float(setup["level"])
        self._active_structure_breakout_ts = int(setup["breakout_ts"])
        self._active_structure_retest_ts = int(setup["retest_ts"])
        self._active_structure_entry_ts = bar_ts
        structure_context = {
            "level": self._active_structure_level,
            "breakout_ts": self._active_structure_breakout_ts,
            "retest_ts": self._active_structure_retest_ts,
            "entry_ts": self._active_structure_entry_ts,
            "swing_index": int(setup.get("swing_index", -1)),
        }
        self._persist_entry_trigger(
            self.ENTRY_STRUCTURE,
            direction,
            structure_context=structure_context,
        )

        self._diag_update(
            entry_state="ENTRY_READY_STRUCTURE",
            direction_15m=(
                "BOS_RETEST_LONG" if direction == "long" else "BOS_RETEST_SHORT"
            ),
            aligned=True,
            strategy="EMA_WT_STRUCTURE_15M",
        )
        metadata = {
            **metadata,
            **self._diag_context,
            "strategy": "TREND_CONFIRM_EMA_WT_STRUCTURE",
            "entry_type": "STRUCTURE_BOS_RETEST_15M",
            "entry_router": "EMA8_13_OR_WT_EXTREME_OR_STRUCTURE_BOS_RETEST",
            "entry_trigger": "STRUCTURE_BOS_RETEST",
            "entry_trigger_owner": self.ENTRY_STRUCTURE,
            "signal_exit_rule": "STRUCTURE_INVALIDATION_OR_OPPOSITE_CHOCH_15M",
            "entry_tf": "15m",
            "stop_loss": round(float(sl), 8),
            "take_profit": round(float(tp), 8),
            "rr_ratio": 1.3,
            "sl_pct": 1.0,
            "tp_pct": 1.3,
            "trail_trigger_pct": 0.6,
            "trail_lock_pct": 0.3,
            "t1_trigger_pct": 0.6,
            "t1_trim_pct": 40.0,
            "t1_lock_pct": 0.3,
            "runner_pct_after_t1": 60.0,
            "partial_tp_enabled": True,
            "partial_tp_pct": 40.0,
            "tp1_close_pct": 0.40,
            "direction_quality_layers_are_entry_gates_only": True,
            "structure_15m": setup,
            "structure_level": round(float(setup["level"]), 8),
            "structure_breakout_ts": int(setup["breakout_ts"]),
            "structure_retest_ts": int(setup["retest_ts"]),
            "price_side_ok": True,
        }
        return Signal(
            type=SignalType.BUY if direction == "long" else SignalType.SELL,
            symbol=self.symbol,
            price=entry_px,
            amount=0.0,
            reason=(
                f"4H/1H gates passed + 15M {direction.upper()} Structure BOS "
                f"+ retest at {float(setup['level']):.6f} "
                "| exit owner=Structure invalidation/opposite CHOCH"
            ),
            confidence=0.74,
            metadata=metadata,
        )

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        """Use 4H/1H only as gates; only a closed-15M trigger may enter."""
        signal = await super().analyze(candles, current_price, mtf_candles)
        metadata = (
            signal.metadata
            if isinstance(getattr(signal, "metadata", None), dict)
            else {}
        )

        if signal.type != SignalType.HOLD:
            return self._finalize_indicator_entry(signal, candles, current_price)

        macro = (
            metadata.get("macro_4h")
            if isinstance(metadata.get("macro_4h"), dict)
            else {}
        )
        context = (
            metadata.get("context_1h")
            if isinstance(metadata.get("context_1h"), dict)
            else {}
        )
        direction = macro.get("direction")
        gates_ready = (
            direction in ("long", "short")
            and bool(context.get("ready"))
            and self._open_position is None
        )

        if gates_ready:
            c15 = self._closed_candle_series(
                candles, 15 * 60_000, self.closed_bar_grace_ms
            )
            if c15:
                setup = self._structure_setup(c15, direction, current_price)
                metadata["structure_15m"] = setup
                metadata["entry_router"] = (
                    "EMA8_13_OR_WT_EXTREME_OR_STRUCTURE_BOS_RETEST"
                )
                if setup.get("trigger"):
                    return self._build_structure_entry(
                        current_price,
                        metadata,
                        macro,
                        direction,
                        c15,
                        setup,
                    )

                reason = str(signal.reason or "")
                if "EMA8/13 OR WT extreme cross" in reason:
                    reason = reason.replace(
                        "EMA8/13 OR WT extreme cross",
                        "EMA8/13 OR WT extreme cross OR Structure BOS+retest",
                    )
                elif reason.startswith("15M"):
                    reason += f"; Structure={setup.get('reason', 'waiting')}"
                signal.reason = reason

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

    def tick_open_position(
        self,
        current_price: float,
        position_key: Optional[str] = None,
    ):
        """Run only the exit engine that owns the open position."""
        if self._open_position is None:
            return None

        from ..engines.position_manager import PositionUpdate

        trigger = self._active_entry_trigger
        if trigger not in self.VALID_ENTRY_TRIGGERS:
            item = self._load_entry_state(self._open_position)
            trigger = str(item.get("trigger", "")).upper()
            if trigger in self.VALID_ENTRY_TRIGGERS:
                self._active_entry_trigger = trigger
                self._restore_structure_context(item)
            else:
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
                    layer3 = self._layer3_indicators(candles)
                    if layer3 is not None:
                        reverse_cross = (
                            layer3["ema_cross_down"]
                            if self._open_position == "long"
                            else layer3["ema_cross_up"]
                        )
                        close_px = float(candles[-1].close)
                        ema13 = float(layer3["ema_slow_val"])
                        close_confirm = (
                            close_px < ema13
                            if self._open_position == "long"
                            else close_px > ema13
                        )
                        close_signal = bool(reverse_cross and close_confirm)
                        exit_reason = (
                            f"EMA entry owner: fresh 15M "
                            f"EMA{self.ema_fast}/{self.ema_slow} reverse cross "
                            f"+ close past EMA{self.ema_slow}"
                        )

                elif trigger == self.ENTRY_WT:
                    wt = self._wave_trend(candles)
                    if wt is not None:
                        close_signal = bool(
                            wt["cross_down"]
                            if self._open_position == "long"
                            else wt["cross_up"]
                        )
                        exit_reason = (
                            "WT entry owner: fresh 15M opposite WT1/WT2 cross "
                            f"(WT1={wt['wt1']:.1f}, WT2={wt['wt2']:.1f})"
                        )

                elif trigger == self.ENTRY_STRUCTURE:
                    close_px = float(candles[-1].close)
                    structure_level = self._active_structure_level
                    invalidated = False
                    if structure_level is not None:
                        invalidated = (
                            close_px < structure_level
                            if self._open_position == "long"
                            else close_px > structure_level
                        )
                    choch = self._opposite_structure_choch(
                        candles,
                        self._open_position,
                        self._active_structure_entry_ts or self._entry_bar_ts,
                    )
                    close_signal = bool(invalidated or choch is not None)
                    if invalidated:
                        exit_reason = (
                            "Structure entry owner: 15M close invalidated "
                            f"BOS/retest level {structure_level:.6f}"
                        )
                    elif choch is not None:
                        exit_reason = (
                            "Structure entry owner: opposite "
                            f"{choch['side']} CHOCH through {choch['level']:.6f}"
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
                        "+0.6% T1 reached — take profit on 40%, move SL to "
                        "+0.3%; keep 60% for final TP or entry-owner exit"
                    ),
                )

        exit_wait = {
            self.ENTRY_EMA: "waiting for EMA8/13 reverse cross",
            self.ENTRY_WT: "waiting for opposite WT1/WT2 cross",
            self.ENTRY_STRUCTURE: (
                "waiting for BOS-level invalidation or opposite CHOCH"
            ),
            self.ENTRY_LEGACY: (
                "legacy trigger unknown — indicator exit disabled; SL/TP active"
            ),
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
        """Recover entry ownership and prevent repeated T1 after restart."""
        super().attach_existing_position(
            direction,
            entry_price,
            stop_loss,
            take_profit,
        )
        item = self._load_entry_state(direction)
        trigger = str(item.get("trigger", "")).upper()
        self._active_entry_trigger = (
            trigger if trigger in self.VALID_ENTRY_TRIGGERS else None
        )
        self._restore_structure_context(item)

        if stop_loss is None or entry_price <= 0:
            return
        lock_tolerance = 0.0002
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
        self._active_structure_level = None
        self._active_structure_breakout_ts = None
        self._active_structure_retest_ts = None
        self._active_structure_entry_ts = None
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
        for index in range(seed_idx + 1, len(arr)):
            if np.isfinite(arr[index]):
                prev = alpha * float(arr[index]) + (1.0 - alpha) * prev
            out[index] = prev
        return out

    @staticmethod
    def _sma_finite(values, period: int) -> np.ndarray:
        """SMA requiring a complete finite rolling window."""
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        period = max(1, int(period))
        for index in range(period - 1, len(arr)):
            window = arr[index - period + 1:index + 1]
            if np.all(np.isfinite(window)):
                out[index] = float(np.mean(window))
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
        source = (
            highs + lows + np.asarray(ha_close, dtype=float)
        ) / 3.0

        esa = self._ema_finite(source, self.wt_channel_length)
        abs_deviation = np.abs(source - esa)
        deviation = self._ema_finite(
            abs_deviation,
            self.wt_channel_length,
        )

        ci = np.full(source.shape, np.nan, dtype=float)
        valid = (
            np.isfinite(source)
            & np.isfinite(esa)
            & np.isfinite(deviation)
            & (deviation > 1e-12)
        )
        ci[valid] = (
            source[valid] - esa[valid]
        ) / (0.015 * deviation[valid])

        wt1 = self._ema_finite(ci, self.wt_average_length)
        wt2 = self._sma_finite(wt1, self.wt_signal_length)

        required = (wt1[-2], wt1[-1], wt2[-2], wt2[-1])
        if not all(np.isfinite(value) for value in required):
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
