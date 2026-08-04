"""HMA Gate Sentinel — authoritative production entry engine.

A single decision path now drives both Railway status and real order creation:

    1H direction -> 1H quality -> 15M S/R location -> room
    -> recent closed-5M trigger -> risk validation -> order

4H remains informational soft macro context.  No legacy super().generate_entry
chain is allowed to re-apply old 4H, score, or location gates behind the log.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import numpy as np

import strategy_v11 as v11
from sentinel_context import build_context

Side = v11.Side
DecisionState = v11.DecisionState
EntrySignal = v11.EntrySignal
Trend = v11.v10.v9.v8.v7.Trend


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PrecisionTrendStructureV12(v11.PrecisionTrendStructureV11):
    """Balanced gate logic with one authoritative entry decision."""

    def __init__(self, config=None) -> None:
        super().__init__(config)

        # Balanced production defaults. Existing Railway env values still take
        # precedence, so every threshold remains auditable and configurable.
        self.one_h_early_min = _env_float("ONE_H_EARLY_TREND_MIN", 55.0)
        self.one_h_strong_min = _env_float("ONE_H_STRONG_TREND_MIN", 68.0)
        self.one_h_direction_edge_min = _env_float("ONE_H_DIRECTION_EDGE_MIN", 5.0)
        self.quality_min = _env_float("MIN_TREND_QUALITY", 52.0)
        self.early_quality_min = _env_float("SENTINEL_EARLY_Q_MIN", 55.0)
        self.early_location_min = _env_float("SENTINEL_EARLY_LOCATION_MIN", 70.0)
        self.sentinel_min_room_atr = _env_float("SENTINEL_MIN_ROOM_ATR", 0.55)
        self.dmi_opposite_tolerance = _env_float("DMI_OPPOSITE_TOLERANCE", 6.0)
        self.exec_trigger_lookback = max(
            1, _env_int("EXEC_TRIGGER_LOOKBACK_BARS", 4)
        )
        self.min_actual_rr = _env_float("MIN_ACTUAL_RR", 1.00)

    @staticmethod
    def _confirmation_ok(loc, side: Side) -> bool:
        return (loc.demand or loc.sweep) if side == Side.LONG else (loc.supply or loc.sweep)

    def _risk_plan(self, decision, df15, df5):
        """Return entry, SL, TP, ATR, structure level and actual R:R."""
        if decision.context is None or decision.side is None:
            return None

        d15 = df15.copy()
        d15["atr"] = self._atr(d15, self.cfg.atr_len)
        atr15 = float(d15["atr"].iloc[-1])
        if not np.isfinite(atr15) or atr15 <= 0.0:
            return None

        entry = float(df5["close"].iloc[-1])
        side = decision.side
        structure_level = self._structure_level(decision.context, side, entry)
        ctx15 = self._structure(d15)
        sl = self._structure_stop(entry, side, atr15, ctx15, structure_level)
        tp = (
            entry * (1.0 + self.cfg.final_take_profit_pct)
            if side == Side.LONG
            else entry * (1.0 - self.cfg.final_take_profit_pct)
        )
        risk = abs(entry - sl)
        rr = abs(tp - entry) / max(risk, 1e-12)
        return entry, sl, tp, atr15, structure_level, rr

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
            return DecisionState(False, "G1_1H_DIRECTION", why, direction, quality, None, None, None, 0.0)

        if quality.q < self.quality_min:
            return DecisionState(False, "G2_1H_QUALITY", f"Q {quality.q:.1f}<{self.quality_min:.0f}", direction, quality, None, None, None, 0.0)
        if direction.tier == "EARLY" and quality.q < self.early_quality_min:
            return DecisionState(False, "G2_1H_QUALITY", f"EARLY Q {quality.q:.1f}<{self.early_quality_min:.0f}", direction, quality, None, None, None, 0.0)
        if not self._dmi_aligned(quality, direction.side):
            dmi_edge = self._dmi_edge(quality, direction.side)
            return DecisionState(False, "G2_1H_QUALITY", f"DMI materially opposed {dmi_edge:+.1f}", direction, quality, None, None, None, 0.0)

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
        confidence = self._v51_trade_score(direction.score, loc.score, False, macro_score)

        # S1/R1 is now available to EARLY trends only when the zone is high
        # quality and confirmed. This preserves the requested S1/R1 entry style
        # without buying/selling a level blindly.
        deep_zone = loc.near_s2 if is_long else loc.near_r2
        first_zone = loc.near_s1 if is_long else loc.near_r1
        confirmation = self._confirmation_ok(loc, direction.side)

        if direction.tier == "EARLY":
            zone_ok = deep_zone or (first_zone and confirmation and loc.score >= 70.0)
            required = (
                "S2 or confirmed S1>=70" if is_long
                else "R2 or confirmed R1>=70"
            )
            min_loc = 70.0
            confirmation_required = first_zone
        else:
            zone_ok = (
                (loc.near_s1 or loc.near_s2 or corridor)
                if is_long else
                (loc.near_r1 or loc.near_r2 or corridor)
            )
            required = "S1/S2 corridor" if is_long else "R1/R2 corridor"
            min_loc = 65.0 if first_zone else 72.0
            confirmation_required = first_zone or corridor

        if not zone_ok:
            blocker = f"need {required}; now {loc.zone} | 4H {macro_label} {macro_edge:+.0f}"
            return DecisionState(False, "G3_15M_LOCATION", blocker, direction, quality, context, setup_type, None, confidence)
        if loc.score < min_loc:
            blocker = f"location {loc.score:.0f}<{min_loc:.0f} at {loc.zone} | 4H {macro_label}"
            return DecisionState(False, "G3_15M_LOCATION", blocker, direction, quality, context, setup_type, None, confidence)
        if confirmation_required and not confirmation:
            needed = "demand/rejection or sweep" if is_long else "supply/rejection or sweep"
            return DecisionState(False, "G3_15M_LOCATION", f"{loc.zone} needs {needed}", direction, quality, context, setup_type, None, confidence)

        if loc.room_atr < self.sentinel_min_room_atr:
            blocker = f"room {loc.room_atr:.2f}<{self.sentinel_min_room_atr:.2f} ATR"
            return DecisionState(False, "G4_ROOM", blocker, direction, quality, context, setup_type, None, confidence)

        execution = self._execution_trigger(df5, direction.side, setup_type)
        confidence = self._v51_trade_score(direction.score, loc.score, execution is not None, macro_score)
        if execution is None:
            return DecisionState(False, "G5_5M_TRIGGER", f"{loc.zone} armed; waiting recent closed-5M trigger", direction, quality, context, setup_type, None, confidence)

        provisional = DecisionState(True, "READY", "", direction, quality, context, setup_type, execution, confidence)
        risk_plan = self._risk_plan(provisional, df15, df5)
        if risk_plan is None:
            return DecisionState(False, "G6_RISK", "15M ATR/structure risk plan unavailable", direction, quality, context, setup_type, execution, confidence)
        rr = risk_plan[-1]
        if rr < self.min_actual_rr:
            return DecisionState(False, "G6_RISK", f"actual RR {rr:.2f}<{self.min_actual_rr:.2f}", direction, quality, context, setup_type, execution, confidence)

        return DecisionState(True, "READY", f"all gates passed | RR {rr:.2f} | 4H {macro_label} {macro_edge:+.0f}", direction, quality, context, setup_type, execution, confidence)

    def generate_entry(
        self,
        df4h,
        df1h,
        df15,
        df5,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        """Create the order only from the same V5.2 decision shown in status.

        Deliberately does not call super().generate_entry(): older strategy
        layers contained hidden 4H/score/location gates that contradicted V5.2.
        """
        if has_open_position:
            return None

        decision = self.evaluate(df4h, df1h, df15, df5)
        if (
            not decision.ready
            or decision.context is None
            or decision.setup_type is None
            or decision.execution is None
            or decision.side is None
        ):
            return None

        risk_plan = self._risk_plan(decision, df15, df5)
        if risk_plan is None:
            return None
        entry, sl, tp, atr15, structure_level, rr = risk_plan
        if rr < self.min_actual_rr:
            return None

        trigger_name, _ = decision.execution
        loc = decision.context.location
        side = decision.side
        macro_score, macro_label, macro_edge = self._macro_bias(df4h, side)
        compat_trend = Trend.BULL if side == Side.LONG else Trend.BEAR
        room_pct = max(0.0, loc.room_atr * atr15 / max(entry, 1e-12))
        reason = (
            f"Gate Sentinel {side.value} | 1H {decision.direction.score:.0f}/"
            f"{decision.direction.tier} edge {decision.direction.edge:+.0f} "
            f"Q {decision.quality.q:.0f} | 15M {loc.zone} Location {loc.score:.0f} "
            f"({loc.reason}) | 5M {trigger_name} | Room {loc.room_atr:.2f}ATR | "
            f"RR {rr:.2f} | 4H soft {macro_label} {macro_edge:+.0f} | "
            f"Confidence {decision.trade_score:.0f}"
        )

        return EntrySignal(
            side=side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=compat_trend,
            q_1h=decision.quality.q,
            adx_1h=decision.quality.adx,
            chop_1h=decision.quality.chop,
            setup=decision.setup_type,
            trigger=trigger_name,
            room_pct=room_pct,
            atr15=atr15,
            structure_level=structure_level,
            reason=reason,
        )

    @staticmethod
    def _stage_label(stage: str) -> str:
        return {
            "G1_1H_DIRECTION": "DIRECTION",
            "G2_1H_QUALITY": "QUALITY",
            "G3_15M_LOCATION": "LOCATION",
            "G4_ROOM": "ROOM",
            "G5_5M_TRIGGER": "TRIGGER",
            "G6_RISK": "RISK",
            "READY": "READY",
        }.get(stage, stage)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        """Compact production log: one readable status line per symbol."""
        d = self.evaluate(df4h, df1h, df15, df5)
        side = d.side.value if d.side else "NONE"
        stage = self._stage_label(d.stage)
        status = "READY" if d.ready else "WAIT"

        parts = [
            f"V5.2 {status}",
            side,
            f"Trend={d.direction.score:.0f}/{d.direction.tier}",
            f"Edge={d.direction.edge:+.0f}",
        ]

        if d.quality is not None:
            parts.append(f"Q={d.quality.q:.1f}")

        if d.context is not None:
            loc = d.context.location
            trigger = d.execution[0] if d.execution else "WAIT"
            parts.extend([
                f"Zone={loc.zone}",
                f"Loc={loc.score:.0f}",
                f"Room={loc.room_atr:.2f}ATR",
                f"Trigger={trigger}",
                f"Conf={d.trade_score:.0f}",
            ])

        reason = d.blocker or "all gates passed"
        parts.extend([f"Stage={stage}", f"Reason={reason}"])
        return " | ".join(parts)


MTFStructureStrategyV12 = PrecisionTrendStructureV12
