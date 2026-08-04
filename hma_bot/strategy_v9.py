"""HMA Expert MTF V4.3 — Sentinel S/R entry-zone engine.

Uses Sentinel X v2.3 support/resistance as the authoritative 15M setup:
- LONG is armed only at S1/S2.
- SHORT is armed only at R1/R2.
- S/R is a setup zone, never an immediate market-order trigger.
- A recent closed-5M execution trigger is still required.

Trend tiers:
- STRONG: S1/S2 or R1/R2.
- MODERATE: S2/R2 normally; S1/R1 allowed only with high-confluence location.
- EARLY: S2/R2 only with stricter Q/location thresholds.
"""
from __future__ import annotations

import os
from typing import Optional

import strategy_v8 as v8

Side = v8.Side
DecisionState = v8.DecisionState
EntrySignal = v8.EntrySignal


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PrecisionTrendStructureV9(v8.PrecisionTrendStructureV8):
    """Single-pass strategy with explicit Sentinel S/R entry-zone policy."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.moderate_first_level_min = _env_float(
            "SENTINEL_MODERATE_S1_R1_MIN", 75.0
        )
        self.first_level_requires_rejection = os.environ.get(
            "SENTINEL_FIRST_LEVEL_REQUIRE_REJECTION", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _zone_policy(self, direction, loc) -> tuple[bool, str, float]:
        """Return zone eligibility, required-zone text and minimum location score."""
        is_long = direction.side == Side.LONG

        if direction.tier == "EARLY":
            ok = loc.near_s2 if is_long else loc.near_r2
            return ok, ("S2" if is_long else "R2"), self.early_location_min

        if direction.tier == "MODERATE":
            deep = loc.near_s2 if is_long else loc.near_r2
            first = loc.near_s1 if is_long else loc.near_r1
            rejection = (
                (loc.demand or loc.sweep)
                if is_long
                else (loc.supply or loc.sweep)
            )
            first_ok = (
                first
                and loc.score >= self.moderate_first_level_min
                and (rejection or not self.first_level_requires_rejection)
            )
            return (
                deep or first_ok,
                (
                    f"S2 or S1≥{self.moderate_first_level_min:.0f}+rejection"
                    if is_long
                    else f"R2 or R1≥{self.moderate_first_level_min:.0f}+rejection"
                ),
                self.sentinel_location_min,
            )

        ok = (
            (loc.near_s1 or loc.near_s2)
            if is_long
            else (loc.near_r1 or loc.near_r2)
        )
        return ok, ("S1/S2" if is_long else "R1/R2"), self.sentinel_location_min

    def evaluate(self, df4h, df1h, df15, df5) -> DecisionState:
        if len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            return super().evaluate(df4h, df1h, df15, df5)

        direction = self._adaptive_direction(df4h)
        quality = self.quality_state_1h(df1h)

        if direction.side is None:
            why = (
                f"trend {direction.score:.0f}<{self.early_trend_min:.0f}"
                if direction.score < self.early_trend_min
                else f"direction edge {direction.edge:.0f}<{self.direction_edge_min:.0f}"
            )
            return DecisionState(False, "TREND", why, direction, quality, None, None, None, 0.0)

        if quality.q < self.cfg.min_trend_quality:
            return DecisionState(False, "QUALITY", f"Q {quality.q:.0f}<{self.cfg.min_trend_quality:.0f}", direction, quality, None, None, None, 0.0)
        if direction.tier == "EARLY" and quality.q < self.early_quality_min:
            return DecisionState(False, "QUALITY", f"EARLY Q {quality.q:.0f}<{self.early_quality_min:.0f}", direction, quality, None, None, None, 0.0)
        if not self._dmi_aligned(quality, direction.side):
            edge = self._dmi_edge(quality, direction.side)
            return DecisionState(False, "QUALITY", f"DMI edge {edge:+.1f}", direction, quality, None, None, None, 0.0)

        context = self._context_for_direction(df15, df1h, df4h, direction)
        setup_type = self._setup_from_context(context)
        loc = context.location

        if loc.room_atr < self.sentinel_min_room_atr:
            score = self._trade_score(direction.score, quality.q, loc.score, False, loc.room_atr)
            return DecisionState(False, "LOCATION", f"room {loc.room_atr:.2f}<{self.sentinel_min_room_atr:.2f} ATR", direction, quality, context, setup_type, None, score)

        zone_ok, required, min_loc = self._zone_policy(direction, loc)
        if not zone_ok:
            score = self._trade_score(direction.score, quality.q, loc.score, False, loc.room_atr)
            return DecisionState(False, "LOCATION", f"need {required}; now {loc.zone}", direction, quality, context, setup_type, None, score)
        if loc.score < min_loc:
            score = self._trade_score(direction.score, quality.q, loc.score, False, loc.room_atr)
            return DecisionState(False, "LOCATION", f"location {loc.score:.0f}<{min_loc:.0f}", direction, quality, context, setup_type, None, score)

        execution = self._execution_trigger(df5, direction.side, setup_type)
        score = self._trade_score(direction.score, quality.q, loc.score, execution is not None, loc.room_atr)
        if execution is None:
            return DecisionState(False, "EXECUTION", f"{loc.zone} armed; waiting recent 5M trigger", direction, quality, context, setup_type, None, score)
        if score < self.trade_score_min:
            return DecisionState(False, "SCORE", f"trade {score:.0f}<{self.trade_score_min:.0f}", direction, quality, context, setup_type, execution, score)

        return DecisionState(True, "READY", "", direction, quality, context, setup_type, execution, score)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        d = self.evaluate(df4h, df1h, df15, df5)
        q = d.quality
        side = d.side.value if d.side else "NONE"
        text = (
            f"V4.3 {d.stage} | {side} Trend={d.direction.score:.0f}/"
            f"{d.direction.tier} edge={d.direction.edge:+.0f}"
        )
        if q is not None:
            text += f" | Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f}"
        if d.context is not None:
            loc = d.context.location
            trigger = d.execution[0] if d.execution else "WAIT"
            text += (
                f" | SR={loc.zone} Loc={loc.score:.0f}"
                f" Room={loc.room_atr:.2f}ATR"
                f" | Exec={trigger} | Trade={d.trade_score:.0f}"
            )
        if d.blocker:
            text += f" | BLOCK {d.blocker}"
        return text

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False) -> Optional[EntrySignal]:
        signal = super().generate_entry(
            df4h, df1h, df15, df5, has_open_position=has_open_position
        )
        if signal is None:
            return None

        decision = self.evaluate(df4h, df1h, df15, df5)
        if not decision.ready or decision.context is None:
            return None

        loc = decision.context.location
        signal_reason = (
            f"Sentinel S/R entry zone {loc.zone} | "
            f"{signal.reason}"
        )
        return EntrySignal(
            side=signal.side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trend_4h=signal.trend_4h,
            q_1h=signal.q_1h,
            adx_1h=signal.adx_1h,
            chop_1h=signal.chop_1h,
            setup=signal.setup,
            trigger=signal.trigger,
            room_pct=signal.room_pct,
            atr15=signal.atr15,
            structure_level=signal.structure_level,
            reason=signal_reason,
        )


MTFStructureStrategyV9 = PrecisionTrendStructureV9
