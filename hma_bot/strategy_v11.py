"""HMA Expert MTF V5.1 — smoother 1H-led Sentinel scoring.

Changes from V5.0:
- weights: 1H 40%, 15M location 30%, 5M execution 20%, 4H soft bias 10%
- graduated S/R location value instead of collapsing every non-zone state to zero
- S1/R1 base 65, corridor S1-S2/R1-R2 base 75, S2/R2 base 90
- exact S/R or the first-to-second-level corridor remains mandatory; this does
  not turn proximity alone into an entry trigger
- 5M execution is still compulsory
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

import strategy_v10 as v10
from sentinel_context import build_context

Side = v10.Side
DecisionState = v10.DecisionState
EntrySignal = v10.EntrySignal


class PrecisionTrendStructureV11(v10.PrecisionTrendStructureV10):
    """V5.0 pipeline with smoother, transparent location contribution."""

    def _effective_location(self, loc, side: Side, price: float):
        """Return location with graduated S/R score and a clearer zone label."""
        score = float(loc.score)
        zone = loc.zone
        corridor = False

        if side == Side.LONG:
            if loc.near_s2:
                score, zone = max(score, 90.0), "S2"
            elif loc.near_s1:
                score, zone = max(score, 65.0), "S1"
            elif loc.s1 is not None and loc.s2 is not None:
                low, high = sorted((float(loc.s2), float(loc.s1)))
                corridor = low <= price <= high
                if corridor:
                    score, zone = max(score, 75.0), "S1-S2"
        else:
            if loc.near_r2:
                score, zone = max(score, 90.0), "R2"
            elif loc.near_r1:
                score, zone = max(score, 65.0), "R1"
            elif loc.r1 is not None and loc.r2 is not None:
                low, high = sorted((float(loc.r1), float(loc.r2)))
                corridor = low <= price <= high
                if corridor:
                    score, zone = max(score, 75.0), "R1-R2"

        reason = loc.reason
        if corridor and zone not in reason:
            reason = f"{zone} CORRIDOR + {reason}"
        return replace(loc, score=float(np.clip(score, 0.0, 100.0)), zone=zone, reason=reason), corridor

    @staticmethod
    def _v51_trade_score(direction_score, location_score, execution_ready, macro_score):
        execution_score = 100.0 if execution_ready else 0.0
        return float(np.clip(
            direction_score * 0.40
            + location_score * 0.30
            + execution_score * 0.20
            + macro_score * 0.10,
            0.0,
            100.0,
        ))

    def evaluate(self, df4h, df1h, df15, df5) -> DecisionState:
        if len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            return super().evaluate(df4h, df1h, df15, df5)

        direction, quality = self._one_h_direction(df1h)
        if direction.side is None:
            why = (
                f"1H trend {direction.score:.0f}<{self.one_h_early_min:.0f}"
                if direction.score < self.one_h_early_min
                else f"1H edge {direction.edge:.0f}<{self.one_h_direction_edge_min:.0f}"
            )
            return DecisionState(False, "1H_TREND", why, direction, quality, None, None, None, 0.0)

        if quality.q < self.cfg.min_trend_quality:
            return DecisionState(False, "1H_QUALITY", f"Q {quality.q:.0f}<{self.cfg.min_trend_quality:.0f}", direction, quality, None, None, None, 0.0)
        if direction.tier == "EARLY" and quality.q < self.early_quality_min:
            return DecisionState(False, "1H_QUALITY", f"EARLY Q {quality.q:.0f}<{self.early_quality_min:.0f}", direction, quality, None, None, None, 0.0)
        if not self._dmi_aligned(quality, direction.side):
            dmi_edge = self._dmi_edge(quality, direction.side)
            return DecisionState(False, "1H_QUALITY", f"DMI edge {dmi_edge:+.1f}", direction, quality, None, None, None, 0.0)

        context = build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="long" if direction.side == Side.LONG else "short",
        )
        price15 = float(df15["close"].iloc[-1])
        effective_loc, corridor = self._effective_location(context.location, direction.side, price15)
        context = replace(context, location=effective_loc)
        setup_type = self._setup_from_context(context)
        loc = context.location
        macro_score, macro_label, macro_edge = self._macro_bias(df4h, direction.side)

        if loc.room_atr < self.sentinel_min_room_atr:
            score = self._v51_trade_score(direction.score, loc.score, False, macro_score)
            blocker = f"room {loc.room_atr:.2f}<{self.sentinel_min_room_atr:.2f}ATR | 4H {macro_label} {macro_edge:+.0f}"
            return DecisionState(False, "15M_LOCATION", blocker, direction, quality, context, setup_type, None, score)

        is_long = direction.side == Side.LONG
        if direction.tier == "EARLY":
            zone_ok = loc.near_s2 if is_long else loc.near_r2
            required = "S2" if is_long else "R2"
            min_loc = self.early_location_min
        else:
            zone_ok = (loc.near_s1 or loc.near_s2 or corridor) if is_long else (loc.near_r1 or loc.near_r2 or corridor)
            required = "S1/S2 corridor" if is_long else "R1/R2 corridor"
            min_loc = self.sentinel_location_min

        if not zone_ok:
            score = self._v51_trade_score(direction.score, loc.score, False, macro_score)
            blocker = f"need {required}; now {loc.zone} | 4H {macro_label} {macro_edge:+.0f}"
            return DecisionState(False, "15M_LOCATION", blocker, direction, quality, context, setup_type, None, score)
        if loc.score < min_loc:
            score = self._v51_trade_score(direction.score, loc.score, False, macro_score)
            return DecisionState(False, "15M_LOCATION", f"location {loc.score:.0f}<{min_loc:.0f} | 4H {macro_label}", direction, quality, context, setup_type, None, score)

        execution = self._execution_trigger(df5, direction.side, setup_type)
        score = self._v51_trade_score(direction.score, loc.score, execution is not None, macro_score)
        if execution is None:
            return DecisionState(False, "5M_EXECUTION", f"{loc.zone} armed; waiting trigger | 4H {macro_label}", direction, quality, context, setup_type, None, score)

        deep_zone = loc.near_s2 if is_long else loc.near_r2
        conditional_ok = (
            score >= self.v5_conditional_score
            and loc.score >= self.v5_conditional_location
            and (deep_zone or loc.sweep or corridor)
        )
        if score < self.v5_entry_score and not conditional_ok:
            blocker = (
                f"score {score:.0f}<{self.v5_entry_score:.0f}; "
                f"70-79 needs S/R corridor, deep S2/R2 or sweep + Loc≥{self.v5_conditional_location:.0f} | "
                f"4H {macro_label}"
            )
            return DecisionState(False, "SCORE", blocker, direction, quality, context, setup_type, execution, score)

        return DecisionState(True, "READY", f"4H {macro_label} {macro_edge:+.0f}", direction, quality, context, setup_type, execution, score)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        d = self.evaluate(df4h, df1h, df15, df5)
        q = d.quality
        side = d.side.value if d.side else "NONE"
        text = f"V5.1 {d.stage} | {side} 1H={d.direction.score:.0f}/{d.direction.tier} edge={d.direction.edge:+.0f}"
        if q is not None:
            text += f" Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f}"
        if d.context is not None:
            loc = d.context.location
            trigger = d.execution[0] if d.execution else "WAIT"
            _, macro_label, macro_edge = self._macro_bias(df4h, d.side)
            text += (
                f" | 15M {loc.zone} Loc={loc.score:.0f} Room={loc.room_atr:.2f}ATR"
                f" | 5M={trigger} | 4H={macro_label}({macro_edge:+.0f})"
                f" | Trade={d.trade_score:.0f}"
            )
        if d.blocker:
            text += f" | {'INFO' if d.ready else 'BLOCK'} {d.blocker}"
        return text


MTFStructureStrategyV11 = PrecisionTrendStructureV11
