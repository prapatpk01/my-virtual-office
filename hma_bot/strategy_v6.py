"""HMA Expert MTF V4.0 — Sentinel X v2.3 location-aware entries.

4H HMA trend and 1H quality stay intact.  The old generic 15M location layer is
replaced by Sentinel adaptive S1/S2/R1/R2 context:

- Strong trend: LONG may arm at S1 or S2; SHORT may arm at R1 or R2.
- Moderate trend: LONG only at S2; SHORT only at R2.
- Weak trend: no entry.
- Location score must be >= 60.
- The actual order still requires a recent closed-5M HMA execution trigger.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

import strategy as legacy
import strategy_v5 as v5
from sentinel_context import SentinelContext, build_context

Side = legacy.Side
Trend = legacy.Trend
SetupType = legacy.SetupType
EntrySignal = legacy.EntrySignal


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PrecisionTrendStructureV6(v5.PrecisionTrendStructureV5):
    """V3.5 execution/risk plus Sentinel X v2.3 15M location."""

    def __init__(self, config: Optional[legacy.StrategyConfig] = None) -> None:
        super().__init__(config)
        self.sentinel_location_min = _env_float("SENTINEL_LOCATION_MIN", 60.0)
        self.sentinel_strong_trend_min = _env_float("SENTINEL_STRONG_TREND_MIN", 85.0)
        self.sentinel_moderate_trend_min = _env_float("SENTINEL_MODERATE_TREND_MIN", 70.0)
        self.sentinel_min_room_atr = _env_float("SENTINEL_MIN_ROOM_ATR", 0.70)

    def _sentinel_context(
        self,
        df15: pd.DataFrame,
        df1h: pd.DataFrame,
        df4h: pd.DataFrame,
        side: Side,
    ) -> SentinelContext:
        return build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="long" if side == Side.LONG else "short",
        )

    def _location_allowed(self, context: SentinelContext, side: Side) -> bool:
        loc = context.location
        if context.trend_score < self.sentinel_moderate_trend_min:
            return False
        if loc.score < self.sentinel_location_min:
            return False
        if loc.room_atr < self.sentinel_min_room_atr:
            return False

        if side == Side.LONG:
            if context.trend_score >= self.sentinel_strong_trend_min:
                return loc.near_s1 or loc.near_s2
            return loc.near_s2

        if context.trend_score >= self.sentinel_strong_trend_min:
            return loc.near_r1 or loc.near_r2
        return loc.near_r2

    @staticmethod
    def _setup_from_context(context: SentinelContext) -> SetupType:
        return SetupType.SWEEP if context.location.sweep else SetupType.PULLBACK

    @staticmethod
    def _structure_level(context: SentinelContext, side: Side, entry: float) -> float:
        loc = context.location
        if side == Side.LONG:
            candidates = [value for value in (loc.s2, loc.s1) if value is not None and value < entry]
            return max(candidates) if candidates else entry
        candidates = [value for value in (loc.r2, loc.r1) if value is not None and value > entry]
        return min(candidates) if candidates else entry

    def entry_status(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        df5: pd.DataFrame,
    ) -> str:
        if len(df15) < 90 or len(df5) < 70:
            return "WARMUP"

        trend = self.trend_state_4h(df4h)
        if trend.trend == Trend.NEUTRAL:
            return "WAIT 4H direction"
        side = Side.LONG if trend.trend == Trend.BULL else Side.SHORT

        quality = self.quality_state_1h(df1h)
        if quality.q < self.cfg.min_trend_quality:
            return f"WAIT Q {quality.q:.0f}<{self.cfg.min_trend_quality:.0f}"
        edge = self._dmi_edge(quality, side)
        if not self._dmi_aligned(quality, side):
            return f"WAIT DMI edge {edge:+.1f}"

        context = self._sentinel_context(df15, df1h, df4h, side)
        loc = context.location
        if context.trend_score < self.sentinel_moderate_trend_min:
            return f"WAIT Sentinel trend {context.trend_score:.0f}<70"
        if not any((loc.near_s1, loc.near_s2, loc.near_r1, loc.near_r2)):
            expected = "S1/S2" if side == Side.LONG else "R1/R2"
            return f"WAIT 15M {expected} location"
        if loc.score < self.sentinel_location_min:
            return f"ARMED {loc.zone} | location {loc.score:.0f}<60 | {loc.reason}"
        if loc.room_atr < self.sentinel_min_room_atr:
            return f"BLOCK room {loc.room_atr:.2f} ATR<0.70"
        if not self._location_allowed(context, side):
            required = "S2" if side == Side.LONG else "R2"
            return (
                f"WAIT {required} (moderate trend {context.trend_score:.0f}) | "
                f"at {loc.zone}"
            )

        setup_type = self._setup_from_context(context)
        execution = self._execution_trigger(df5, side, setup_type)
        if execution is None:
            return (
                f"ARMED {loc.zone} {loc.score:.0f}/100 | "
                f"WAIT 5M trigger | {loc.reason}"
            )
        return (
            f"READY {loc.zone} {loc.score:.0f}/100 | {execution[0]} | "
            f"Trend {context.trend_score:.0f}"
        )

    def generate_entry(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        df5: pd.DataFrame,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        if has_open_position or len(df15) < 90 or len(df5) < 70:
            return None

        trend = self.trend_state_4h(df4h)
        if trend.trend == Trend.NEUTRAL:
            return None
        side = Side.LONG if trend.trend == Trend.BULL else Side.SHORT

        quality = self.quality_state_1h(df1h)
        if quality.q < self.cfg.min_trend_quality or not self._dmi_aligned(quality, side):
            return None

        context = self._sentinel_context(df15, df1h, df4h, side)
        if not self._location_allowed(context, side):
            return None

        setup_type = self._setup_from_context(context)
        execution = self._execution_trigger(df5, side, setup_type)
        if execution is None:
            return None
        trigger_name, _ = execution

        d15 = df15.copy()
        d15["atr"] = self._atr(d15, self.cfg.atr_len)
        atr15 = float(d15["atr"].iloc[-1])
        if not np.isfinite(atr15) or atr15 <= 0:
            return None

        entry = float(df5["close"].iloc[-1])
        structure_level = self._structure_level(context, side, entry)
        ctx15 = self._structure(d15)
        sl = self._structure_stop(entry, side, atr15, ctx15, structure_level)
        tp = (
            entry * (1.0 + self.cfg.final_take_profit_pct)
            if side == Side.LONG
            else entry * (1.0 - self.cfg.final_take_profit_pct)
        )

        risk = abs(entry - sl)
        rr = abs(tp - entry) / max(risk, 1e-12)
        if rr < self.min_actual_rr:
            return None

        loc = context.location
        trade_score = (
            context.trend_score * 0.30
            + quality.q * 0.25
            + loc.score * 0.30
            + 85.0 * 0.15
        )
        reason = (
            f"Sentinel {loc.zone} | Location {loc.score:.0f}/100 ({loc.reason}) | "
            f"Trend {context.trend_score:.0f} {context.trend_class} | "
            f"Q {quality.q:.0f} | Execution {trigger_name} | "
            f"TradeScore {trade_score:.0f} | room {loc.room_atr:.2f} ATR | RR {rr:.2f}"
        )

        return EntrySignal(
            side=side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=trend.trend,
            q_1h=quality.q,
            adx_1h=quality.adx,
            chop_1h=quality.chop,
            setup=setup_type,
            trigger=trigger_name,
            room_pct=max(0.0, loc.room_atr * atr15 / max(entry, 1e-12)),
            atr15=atr15,
            structure_level=structure_level,
            reason=reason,
        )


MTFStructureStrategyV6 = PrecisionTrendStructureV6
