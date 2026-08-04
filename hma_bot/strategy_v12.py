"""HMA Expert MTF V5.2 — gate-based Sentinel entry engine.

The weighted score remains visible as diagnostic confidence only. It no longer
has authority to approve an entry. A trade must pass every ordered gate:

    1H direction -> 1H quality -> 15M S/R location -> room -> 5M trigger

4H remains informational soft macro context and never blocks an entry.
"""
from __future__ import annotations

from dataclasses import replace

import strategy_v11 as v11
from sentinel_context import build_context

Side = v11.Side
DecisionState = v11.DecisionState
EntrySignal = v11.EntrySignal


class PrecisionTrendStructureV12(v11.PrecisionTrendStructureV11):
    """Single-pass gate logic with transparent non-authoritative confidence."""

    @staticmethod
    def _confirmation_ok(loc, side: Side) -> bool:
        return (loc.demand or loc.sweep) if side == Side.LONG else (loc.supply or loc.sweep)

    def evaluate(self, df4h, df1h, df15, df5) -> DecisionState:
        if len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            return super().evaluate(df4h, df1h, df15, df5)

        # Gate 1: 1H selects the trading side. 4H is not consulted here.
        direction, quality = self._one_h_direction(df1h)
        if direction.side is None:
            why = (
                f"1H trend {direction.score:.0f}<{self.one_h_early_min:.0f}"
                if direction.score < self.one_h_early_min
                else f"1H edge {direction.edge:.0f}<{self.one_h_direction_edge_min:.0f}"
            )
            return DecisionState(False, "G1_1H_DIRECTION", why, direction, quality, None, None, None, 0.0)

        # Gate 2: quality must independently pass; a high trend score cannot
        # compensate for weak Q or materially opposing DMI.
        if quality.q < self.cfg.min_trend_quality:
            return DecisionState(False, "G2_1H_QUALITY", f"Q {quality.q:.0f}<{self.cfg.min_trend_quality:.0f}", direction, quality, None, None, None, 0.0)
        if direction.tier == "EARLY" and quality.q < self.early_quality_min:
            return DecisionState(False, "G2_1H_QUALITY", f"EARLY Q {quality.q:.0f}<{self.early_quality_min:.0f}", direction, quality, None, None, None, 0.0)
        if not self._dmi_aligned(quality, direction.side):
            dmi_edge = self._dmi_edge(quality, direction.side)
            return DecisionState(False, "G2_1H_QUALITY", f"DMI edge {dmi_edge:+.1f}", direction, quality, None, None, None, 0.0)

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
        is_long = direction.side == Side.LONG

        # Diagnostic confidence only. It is shown in logs but never bypasses a gate.
        confidence = self._v51_trade_score(direction.score, loc.score, False, macro_score)

        # Gate 3: authoritative Sentinel S/R location.
        if direction.tier == "EARLY":
            zone_ok = loc.near_s2 if is_long else loc.near_r2
            required = "S2" if is_long else "R2"
            min_loc = max(self.early_location_min, 75.0)
            confirmation_required = True
        else:
            zone_ok = (
                (loc.near_s1 or loc.near_s2 or corridor)
                if is_long else
                (loc.near_r1 or loc.near_r2 or corridor)
            )
            required = "S1/S2 corridor" if is_long else "R1/R2 corridor"
            min_loc = 65.0 if (loc.near_s1 if is_long else loc.near_r1) else 75.0
            confirmation_required = (loc.near_s1 if is_long else loc.near_r1) or corridor

        if not zone_ok:
            blocker = f"need {required}; now {loc.zone} | 4H {macro_label} {macro_edge:+.0f}"
            return DecisionState(False, "G3_15M_LOCATION", blocker, direction, quality, context, setup_type, None, confidence)
        if loc.score < min_loc:
            blocker = f"location {loc.score:.0f}<{min_loc:.0f} at {loc.zone} | 4H {macro_label}"
            return DecisionState(False, "G3_15M_LOCATION", blocker, direction, quality, context, setup_type, None, confidence)
        if confirmation_required and not self._confirmation_ok(loc, direction.side):
            needed = "demand/rejection or sweep" if is_long else "supply/rejection or sweep"
            return DecisionState(False, "G3_15M_LOCATION", f"{loc.zone} needs {needed}", direction, quality, context, setup_type, None, confidence)

        # Gate 4: there must be enough structural room before the opposing level.
        if loc.room_atr < self.sentinel_min_room_atr:
            blocker = f"room {loc.room_atr:.2f}<{self.sentinel_min_room_atr:.2f} ATR"
            return DecisionState(False, "G4_ROOM", blocker, direction, quality, context, setup_type, None, confidence)

        # Gate 5: a recent closed 5M trigger is compulsory.
        execution = self._execution_trigger(df5, direction.side, setup_type)
        confidence = self._v51_trade_score(direction.score, loc.score, execution is not None, macro_score)
        if execution is None:
            return DecisionState(False, "G5_5M_TRIGGER", f"{loc.zone} armed; waiting recent closed-5M trigger", direction, quality, context, setup_type, None, confidence)

        # No final score gate: every required market condition has passed.
        return DecisionState(True, "READY", f"all gates passed | 4H {macro_label} {macro_edge:+.0f}", direction, quality, context, setup_type, execution, confidence)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        d = self.evaluate(df4h, df1h, df15, df5)
        side = d.side.value if d.side else "NONE"
        text = (
            f"V5.2 {d.stage} | {side} 1H={d.direction.score:.0f}/{d.direction.tier} "
            f"edge={d.direction.edge:+.0f}"
        )
        if d.quality is not None:
            q = d.quality
            text += f" Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f}"
        if d.context is not None:
            loc = d.context.location
            trigger = d.execution[0] if d.execution else "WAIT"
            _, macro_label, macro_edge = self._macro_bias(df4h, d.side)
            text += (
                f" | 15M={loc.zone} Loc={loc.score:.0f} Room={loc.room_atr:.2f}ATR"
                f" | 5M={trigger} | 4H={macro_label}({macro_edge:+.0f})"
                f" | Confidence={d.trade_score:.0f}"
            )
        if d.blocker:
            text += f" | {'INFO' if d.ready else 'BLOCK'} {d.blocker}"
        return text


MTFStructureStrategyV12 = PrecisionTrendStructureV12
