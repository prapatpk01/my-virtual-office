"""Sentinel V3.1 — hold winners longer and prevent same-side churn.

Extends Sentinel V3 without changing its 15M quality or entry engines.
Changes are limited to position management:
- two completed 15M bars of entry grace for ordinary EMA noise
- EMA8/13 cross-back is a warning, not an exit by itself
- EMA exit requires cross-back + close beyond EMA20, or two closes beyond EMA20
- strong opposite CHOCH/pivot break can still exit immediately
- technical exits lock same-direction re-entry until a setup reset is observed
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import SignalType
from .sentinel_v3_strategy import SentinelV3Strategy
from ..engines.position_manager import PositionUpdate


class SentinelV31Strategy(SentinelV3Strategy):
    """Sentinel V3 with structure-aware holding and re-entry re-arming."""

    VERSION = "3.1"
    ENTRY_GRACE_BARS = 2

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV3.1({symbol})"
        self._entry_bar_ts: Optional[int] = None
        self._reentry_lock_direction: Optional[str] = None
        self._reentry_exit_bar_ts: Optional[int] = None
        self._reentry_lock_reason: str = ""

    @classmethod
    def _pivot_points(cls, candles: list, span: int = 2) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        highs: list[tuple[int, float]] = []
        lows: list[tuple[int, float]] = []
        for i in range(span, len(candles) - span):
            before = candles[i - span:i]
            after = candles[i + 1:i + span + 1]
            high = float(candles[i].high)
            low = float(candles[i].low)
            if high >= max(float(c.high) for c in before + after):
                highs.append((cls._bar_ts(candles[i]), high))
            if low <= min(float(c.low) for c in before + after):
                lows.append((cls._bar_ts(candles[i]), low))
        return highs, lows

    def _arm_reentry_lock(self, direction: str, bar_ts: int, reason: str) -> None:
        self._reentry_lock_direction = direction
        self._reentry_exit_bar_ts = int(bar_ts)
        self._reentry_lock_reason = reason

    def _clear_reentry_lock(self) -> None:
        self._reentry_lock_direction = None
        self._reentry_exit_bar_ts = None
        self._reentry_lock_reason = ""

    def _same_side_reentry_ready(self, candles: list, direction: str, atr_now: float) -> tuple[bool, str]:
        """Require a reset before re-entering the same side after technical exit.

        Reset can be either:
        1) a completed pullback bar touching EMA20 or HMA16; or
        2) a newly confirmed lower-high (for short) / higher-low (for long).

        The current trigger bar is excluded from the pullback test so the bot
        sees reset first, then a fresh trigger on a later bar.
        """
        if self._reentry_lock_direction != direction or self._reentry_exit_bar_ts is None:
            return True, "not locked"

        exit_ts = int(self._reentry_exit_bar_ts)
        current_ts = self._bar_ts(candles[-1])
        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        hma16 = self.hma(closes, 16)
        tolerance = max(float(atr_now), 1e-12) * 0.10

        pullback_reset = False
        for i in range(max(0, len(candles) - 40), len(candles) - 1):
            ts = self._bar_ts(candles[i])
            if ts <= exit_ts or ts >= current_ts:
                continue
            if not self._finite(ema20[i], hma16[i]):
                continue
            if direction == "short":
                reset_line = min(float(ema20[i]), float(hma16[i]))
                if float(candles[i].high) >= reset_line - tolerance:
                    pullback_reset = True
                    break
            else:
                reset_line = max(float(ema20[i]), float(hma16[i]))
                if float(candles[i].low) <= reset_line + tolerance:
                    pullback_reset = True
                    break

        highs, lows = self._pivot_points(candles[-80:], span=2)
        structure_reset = False
        if direction == "short":
            for i in range(1, len(highs)):
                ts, price = highs[i]
                if ts > exit_ts and price < highs[i - 1][1]:
                    structure_reset = True
                    break
        else:
            for i in range(1, len(lows)):
                ts, price = lows[i]
                if ts > exit_ts and price > lows[i - 1][1]:
                    structure_reset = True
                    break

        if structure_reset:
            return True, "new lower-high" if direction == "short" else "new higher-low"
        if pullback_reset:
            return True, "EMA20/HMA16 pullback reset"
        return False, f"same-side re-entry locked after {self._reentry_lock_reason or 'technical exit'}"

    def _build_entry(self, candles: list, current_price: float, market: dict, structure: dict) -> dict:
        entry = super()._build_entry(candles, current_price, market, structure)
        direction = entry.get("direction")
        if direction not in {"long", "short"}:
            return entry

        atr_now = max(float(self.atr(candles, 14)[-1]), 1e-12)
        ready, reset_reason = self._same_side_reentry_ready(candles, str(direction), atr_now)
        entry["reentry_reset"] = reset_reason
        entry["reentry_locked"] = not ready

        if not ready:
            blocks = list(entry.get("hard_blocks", []))
            if "REENTRY_RESET" not in blocks:
                blocks.append("REENTRY_RESET")
            entry["hard_blocks"] = blocks
            entry["trigger"] = None
            entry["reason"] = f"blocked: {','.join(blocks)}"
        return entry

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await super().analyze(candles, current_price, mtf_candles)
        meta = signal.metadata or {}
        meta["version"] = self.VERSION
        meta["position_management"] = "2BAR_GRACE__EMA20_CONFIRM__STRUCTURE_EXIT__SAME_SIDE_REARM"
        signal.metadata = meta

        if signal.type in {SignalType.BUY, SignalType.SELL}:
            self._entry_bar_ts = self._bar_ts(self._latest_15m[-1]) if self._latest_15m else None
            # A valid new trade proves the old re-entry lock has been re-armed;
            # an opposite-side trade also invalidates the previous same-side lock.
            self._clear_reentry_lock()
        return signal

    def _technical_exit(self, side: str, bar_ts: int, reason: str) -> PositionUpdate:
        self._last_exit_bar_ts = int(bar_ts)
        self._arm_reentry_lock(side, bar_ts, reason)
        self._entry_bar_ts = None
        self._reset_position(keep_exit_ts=True)
        return PositionUpdate(action="close", close_pct=1.0, reason=reason)

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        if self._open_position is None:
            return None

        candles = self._latest_15m
        if len(candles) >= 35:
            bar_ts = self._bar_ts(candles[-1])
            atr_now = float(self.atr(candles, 14)[-1])
            if self._finite(atr_now):
                side = self._open_position
                long = side == "long"
                structure = self._structure_snapshot(candles, max(atr_now, 1e-12))

                # Strong structure invalidation is allowed to bypass entry grace.
                strong_invalid = structure["choch_down"] if long else structure["choch_up"]
                if strong_invalid:
                    direction_word = "bearish" if long else "bullish"
                    return self._technical_exit(
                        side,
                        bar_ts,
                        f"15M strong {direction_word} CHOCH + pivot break — close {side.upper()}",
                    )

                bars_after_entry = 0
                if self._entry_bar_ts is not None:
                    bars_after_entry = sum(1 for candle in candles if self._bar_ts(candle) > self._entry_bar_ts)
                grace_active = self._entry_bar_ts is not None and bars_after_entry < self.ENTRY_GRACE_BARS

                closes = [float(c.close) for c in candles]
                ema8 = self.ema(closes, 8)
                ema13 = self.ema(closes, 13)
                ema20 = self.ema(closes, 20)
                if not grace_active and self._finite(
                    ema8[-1], ema8[-2], ema13[-1], ema13[-2], ema20[-1], ema20[-2]
                ):
                    reverse_cross = (
                        ema8[-2] >= ema13[-2] and ema8[-1] < ema13[-1]
                    ) if long else (
                        ema8[-2] <= ema13[-2] and ema8[-1] > ema13[-1]
                    )
                    close_beyond_ema20 = closes[-1] < ema20[-1] if long else closes[-1] > ema20[-1]
                    two_close_beyond_ema20 = (
                        closes[-1] < ema20[-1] and closes[-2] < ema20[-2]
                    ) if long else (
                        closes[-1] > ema20[-1] and closes[-2] > ema20[-2]
                    )

                    if (reverse_cross and close_beyond_ema20) or two_close_beyond_ema20:
                        cause = "EMA8/13 cross-back + close beyond EMA20" if reverse_cross and close_beyond_ema20 else "2 closes beyond EMA20"
                        return self._technical_exit(
                            side,
                            bar_ts,
                            f"15M trend failure ({cause}) — close {side.upper()}",
                        )

        # Keep the proven T1 partial + breakeven lifecycle, but deliberately do
        # not call SimplePrecision.tick_open_position because its EMA13 exit is
        # too sensitive for Sentinel V3's 15M-only architecture.
        if (
            not self._tp1_done
            and self._entry_price is not None
            and self._initial_risk is not None
            and self._initial_risk > 0
        ):
            profit = (
                float(current_price) - self._entry_price
                if self._open_position == "long"
                else self._entry_price - float(current_price)
            )
            current_r = profit / self._initial_risk
            if current_r >= self.tp1_r:
                self._tp1_done = True
                return PositionUpdate(
                    action="partial_tp",
                    close_pct=self.tp1_trim_pct,
                    new_sl=self._entry_price,
                    reason=f"T1 {current_r:.2f}R — trim {self.tp1_trim_pct:.0%}, move SL to breakeven",
                )

        grace_note = ""
        if self._entry_bar_ts is not None and self._latest_15m:
            bars_after_entry = sum(1 for candle in self._latest_15m if self._bar_ts(candle) > self._entry_bar_ts)
            if bars_after_entry < self.ENTRY_GRACE_BARS:
                grace_note = f" | entry grace {bars_after_entry}/{self.ENTRY_GRACE_BARS}"

        return PositionUpdate(
            action="hold",
            reason=(
                f"Holding {self._open_position.upper()} — EMA8/13 cross-back is warning only; "
                f"exit needs EMA20/structure confirmation{grace_note}"
            ),
        )
