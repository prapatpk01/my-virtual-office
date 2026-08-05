"""HMA Simple Sentinel — S1/S2 and R1/R2 entry strategy.

Authoritative decisions:

    Layer 1: 1H direction
    Layer 2: 15M adaptive S1/S2 or R1/R2
    Layer 3: closed-5M hold confirmation or reclaim
    Risk:    existing structure SL, room and actual R:R

LONG:
- Touch S1/S2 and close without losing the level -> enter after the next
  closed 5M candle still holds above it.
- Close below S1/S2 -> wait for the first closed 5M candle to reclaim above
  the level, then enter immediately.

SHORT mirrors the same rules at R1/R2.

4H and confidence remain diagnostic only. Status and live order creation use
this same evaluation result; no legacy entry gate is re-applied.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import numpy as np

import strategy_v11 as v11
from sentinel_context import build_context, trend_score_4h

Side = v11.Side
DecisionState = v11.DecisionState
EntrySignal = v11.EntrySignal
AdaptiveDirection = v11.v10.v9.v8.v7.AdaptiveDirection
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
    """1H direction -> S/R reaction -> one risk check."""

    def __init__(self, config=None) -> None:
        super().__init__(config)

        # Layer 1: choose one side from 1H. Q is checked once and DMI only
        # blocks when it is materially opposed.
        self.one_h_early_min = _env_float("ONE_H_EARLY_TREND_MIN", 55.0)
        self.one_h_strong_min = _env_float("ONE_H_STRONG_TREND_MIN", 68.0)
        self.one_h_direction_edge_min = _env_float(
            "ONE_H_DIRECTION_EDGE_MIN", 4.0
        )
        self.quality_min = _env_float("MIN_TREND_QUALITY", 52.0)
        self.dmi_hard_opposition = _env_float("DMI_HARD_OPPOSITION", 10.0)

        # S/R execution uses closed 5M candles around adaptive 15M levels.
        self.sr_touch_zone_atr5 = _env_float("SR_TOUCH_ZONE_ATR5", 0.25)
        self.sr_break_buffer_atr5 = _env_float("SR_BREAK_BUFFER_ATR5", 0.04)
        self.sr_reclaim_buffer_atr5 = _env_float(
            "SR_RECLAIM_BUFFER_ATR5", 0.03
        )
        self.sr_reclaim_lookback = max(
            2, _env_int("SR_RECLAIM_LOOKBACK_BARS", 8)
        )

        # One combined risk check after a valid S/R trigger.
        self.min_room_atr = _env_float("MIN_ENTRY_ROOM_ATR", 0.30)
        self.min_actual_rr = _env_float("MIN_ACTUAL_RR", 0.90)

    def _simple_direction(self, df1h):
        """Select the stronger 1H side without using 4H as a gate."""
        quality = self.quality_state_1h(df1h)
        long_score = float(trend_score_4h(df1h, "long"))
        short_score = float(trend_score_4h(df1h, "short"))

        if long_score >= short_score:
            side = Side.LONG
            score, opposite = long_score, short_score
        else:
            side = Side.SHORT
            score, opposite = short_score, long_score

        edge = score - opposite
        if score < self.one_h_early_min or edge < self.one_h_direction_edge_min:
            return (
                AdaptiveDirection(None, score, opposite, edge, "NEUTRAL"),
                quality,
            )

        tier = "STRONG" if score >= self.one_h_strong_min else "TREND"
        return AdaptiveDirection(side, score, opposite, edge, tier), quality

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
        sl = self._structure_stop(
            entry, side, atr15, ctx15, structure_level
        )
        tp = (
            entry * (1.0 + self.cfg.final_take_profit_pct)
            if side == Side.LONG
            else entry * (1.0 - self.cfg.final_take_profit_pct)
        )
        risk = abs(entry - sl)
        rr = abs(tp - entry) / max(risk, 1e-12)
        return entry, sl, tp, atr15, structure_level, rr

    @staticmethod
    def _side_levels(loc, side: Side):
        raw = (
            (("S1", loc.s1), ("S2", loc.s2))
            if side == Side.LONG
            else (("R1", loc.r1), ("R2", loc.r2))
        )
        return [
            (name, float(value))
            for name, value in raw
            if value is not None and np.isfinite(float(value))
        ]

    def _level_reaction(self, df5, side: Side, name: str, level: float):
        """Evaluate one S/R level using only closed 5M candles.

        Returns ``(trigger, atr5, armed, reason, distance)``.
        """
        if len(df5) < max(20, self.sr_reclaim_lookback + 2):
            return None, 0.0, False, "5M candles unavailable", float("inf")

        d5 = df5.copy()
        d5["atr"] = self._atr(d5, self.cfg.atr_len)
        current = d5.iloc[-1]
        previous = d5.iloc[-2]
        atr5 = float(current["atr"])
        if not np.isfinite(atr5) or atr5 <= 0.0:
            return None, 0.0, False, "5M ATR unavailable", float("inf")

        zone = max(self.sr_touch_zone_atr5 * atr5, abs(level) * 1e-6)
        break_buffer = self.sr_break_buffer_atr5 * atr5
        reclaim_buffer = self.sr_reclaim_buffer_atr5 * atr5
        distance = abs(float(current["close"]) - level)

        history = d5.iloc[-(self.sr_reclaim_lookback + 1) : -1]
        if side == Side.LONG:
            previous_touch_hold = (
                float(previous["low"]) <= level + zone
                and float(previous["close"]) >= level - break_buffer
            )
            next_candle_holds = (
                float(current["close"]) > level + reclaim_buffer
                and float(current["low"]) >= level - zone
            )
            hold_trigger = previous_touch_hold and next_candle_holds

            was_below = bool(
                (history["close"].astype(float) < level - break_buffer).any()
            )
            reclaim_trigger = (
                was_below
                and float(previous["close"]) <= level + break_buffer
                and float(current["close"]) > level + reclaim_buffer
            )

            near_now = (
                float(current["low"]) <= level + zone
                or float(previous["low"]) <= level + zone
                or distance <= zone
            )
            below_now = float(current["close"]) < level - break_buffer

            if reclaim_trigger:
                return (
                    f"{name}_RECLAIM_LONG",
                    atr5,
                    True,
                    f"{name} reclaimed above {level:.6g}",
                    distance,
                )
            if hold_trigger:
                return (
                    f"{name}_HOLD_CONFIRM_LONG",
                    atr5,
                    True,
                    f"{name} held; next 5M candle confirmed above {level:.6g}",
                    distance,
                )
            if below_now or was_below:
                return (
                    None,
                    atr5,
                    True,
                    f"{name} broken; waiting closed-5M reclaim above {level:.6g}",
                    distance,
                )
            if near_now:
                return (
                    None,
                    atr5,
                    True,
                    f"{name} touched/near; waiting next closed 5M candle to hold",
                    distance,
                )
            return (
                None,
                atr5,
                False,
                f"waiting price to reach {name} {level:.6g}",
                distance,
            )

        previous_touch_hold = (
            float(previous["high"]) >= level - zone
            and float(previous["close"]) <= level + break_buffer
        )
        next_candle_holds = (
            float(current["close"]) < level - reclaim_buffer
            and float(current["high"]) <= level + zone
        )
        hold_trigger = previous_touch_hold and next_candle_holds

        was_above = bool(
            (history["close"].astype(float) > level + break_buffer).any()
        )
        reclaim_trigger = (
            was_above
            and float(previous["close"]) >= level - break_buffer
            and float(current["close"]) < level - reclaim_buffer
        )

        near_now = (
            float(current["high"]) >= level - zone
            or float(previous["high"]) >= level - zone
            or distance <= zone
        )
        above_now = float(current["close"]) > level + break_buffer

        if reclaim_trigger:
            return (
                f"{name}_RECLAIM_SHORT",
                atr5,
                True,
                f"{name} reclaimed below {level:.6g}",
                distance,
            )
        if hold_trigger:
            return (
                f"{name}_HOLD_CONFIRM_SHORT",
                atr5,
                True,
                f"{name} held; next 5M candle confirmed below {level:.6g}",
                distance,
            )
        if above_now or was_above:
            return (
                None,
                atr5,
                True,
                f"{name} broken; waiting closed-5M reclaim below {level:.6g}",
                distance,
            )
        if near_now:
            return (
                None,
                atr5,
                True,
                f"{name} touched/near; waiting next closed 5M candle to hold",
                distance,
            )
        return (
            None,
            atr5,
            False,
            f"waiting price to reach {name} {level:.6g}",
            distance,
        )

    def _sr_entry_state(self, df5, loc, side: Side):
        """Return the best active S/R state.

        Result: ``execution, level_name, level_price, armed, reason``.
        """
        levels = self._side_levels(loc, side)
        if not levels:
            expected = "S1/S2" if side == Side.LONG else "R1/R2"
            return None, expected, None, False, f"{expected} unavailable"

        states = []
        for name, level in levels:
            trigger, atr5, armed, reason, distance = self._level_reaction(
                df5, side, name, level
            )
            states.append(
                {
                    "execution": (trigger, atr5) if trigger else None,
                    "name": name,
                    "level": level,
                    "armed": armed,
                    "reason": reason,
                    "distance": distance,
                }
            )

        triggered = [state for state in states if state["execution"] is not None]
        if triggered:
            best = min(triggered, key=lambda state: state["distance"])
        else:
            armed_states = [state for state in states if state["armed"]]
            pool = armed_states or states
            best = min(pool, key=lambda state: state["distance"])

        return (
            best["execution"],
            best["name"],
            best["level"],
            best["armed"],
            best["reason"],
        )

    def evaluate(self, df4h, df1h, df15, df5) -> DecisionState:
        if len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            direction = AdaptiveDirection(None, 0.0, 0.0, 0.0, "WARMUP")
            quality = (
                self.quality_state_1h(df1h) if len(df1h) >= 60 else None
            )
            return DecisionState(
                False,
                "L0_WARMUP",
                "insufficient candles",
                direction,
                quality,
                None,
                None,
                None,
                0.0,
            )

        direction, quality = self._simple_direction(df1h)
        if direction.side is None:
            why = (
                f"1H trend {direction.score:.0f}<{self.one_h_early_min:.0f}"
                if direction.score < self.one_h_early_min
                else (
                    f"1H edge {direction.edge:.0f}"
                    f"<{self.one_h_direction_edge_min:.0f}"
                )
            )
            return DecisionState(
                False,
                "L1_DIRECTION",
                why,
                direction,
                quality,
                None,
                None,
                None,
                0.0,
            )

        if quality.q < self.quality_min:
            return DecisionState(
                False,
                "L1_DIRECTION",
                f"Q {quality.q:.1f}<{self.quality_min:.0f}",
                direction,
                quality,
                None,
                None,
                None,
                0.0,
            )

        dmi_edge = self._dmi_edge(quality, direction.side)
        if dmi_edge < -self.dmi_hard_opposition:
            return DecisionState(
                False,
                "L1_DIRECTION",
                f"DMI strongly opposed {dmi_edge:+.1f}",
                direction,
                quality,
                None,
                None,
                None,
                0.0,
            )

        context = build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="long" if direction.side == Side.LONG else "short",
        )
        loc = context.location
        execution, level_name, level_price, armed, reaction_reason = (
            self._sr_entry_state(df5, loc, direction.side)
        )

        level_label = (
            f"{level_name}@{level_price:.6g}"
            if level_price is not None
            else level_name
        )
        updated_loc = replace(
            loc,
            zone=level_name,
            score=max(float(loc.score), 60.0) if armed else float(loc.score),
            reason=reaction_reason,
        )
        context = replace(context, location=updated_loc)
        setup_type = self._setup_from_context(context)

        macro_score, macro_label, macro_edge = self._macro_bias(
            df4h, direction.side
        )
        confidence = self._v51_trade_score(
            direction.score,
            updated_loc.score,
            execution is not None,
            macro_score,
        )

        if not armed:
            return DecisionState(
                False,
                "L2_SETUP",
                f"{reaction_reason} | 4H {macro_label} {macro_edge:+.0f}",
                direction,
                quality,
                context,
                setup_type,
                None,
                confidence,
            )

        if execution is None:
            return DecisionState(
                False,
                "L3_TRIGGER",
                f"{level_label} armed; {reaction_reason}",
                direction,
                quality,
                context,
                setup_type,
                None,
                confidence,
            )

        provisional = DecisionState(
            True,
            "READY",
            "",
            direction,
            quality,
            context,
            setup_type,
            execution,
            confidence,
        )
        risk_plan = self._risk_plan(provisional, df15, df5)
        if risk_plan is None:
            return DecisionState(
                False,
                "RISK",
                "15M structure risk plan unavailable",
                direction,
                quality,
                context,
                setup_type,
                execution,
                confidence,
            )

        rr = risk_plan[-1]
        if updated_loc.room_atr < self.min_room_atr:
            return DecisionState(
                False,
                "RISK",
                (
                    f"room {updated_loc.room_atr:.2f}"
                    f"<{self.min_room_atr:.2f} ATR"
                ),
                direction,
                quality,
                context,
                setup_type,
                execution,
                confidence,
            )
        if rr < self.min_actual_rr:
            return DecisionState(
                False,
                "RISK",
                f"actual RR {rr:.2f}<{self.min_actual_rr:.2f}",
                direction,
                quality,
                context,
                setup_type,
                execution,
                confidence,
            )

        return DecisionState(
            True,
            "READY",
            (
                f"{execution[0]} | RR {rr:.2f} | "
                f"4H {macro_label} {macro_edge:+.0f}"
            ),
            direction,
            quality,
            context,
            setup_type,
            execution,
            confidence,
        )

    def generate_entry(
        self,
        df4h,
        df1h,
        df15,
        df5,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        """Create an order only from the same S/R decision shown in status."""
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

        loc = decision.context.location
        if loc.room_atr < self.min_room_atr or rr < self.min_actual_rr:
            return None

        trigger_name, _ = decision.execution
        side = decision.side
        _, macro_label, macro_edge = self._macro_bias(df4h, side)
        compat_trend = Trend.BULL if side == Side.LONG else Trend.BEAR
        room_pct = max(
            0.0, loc.room_atr * atr15 / max(entry, 1e-12)
        )
        reason = (
            f"Sentinel S/R {side.value} | 1H Trend "
            f"{decision.direction.score:.0f} edge "
            f"{decision.direction.edge:+.0f} Q "
            f"{decision.quality.q:.0f} | 15M {loc.zone} | "
            f"5M {trigger_name} ({loc.reason}) | "
            f"Room {loc.room_atr:.2f}ATR | RR {rr:.2f} | "
            f"4H soft {macro_label} {macro_edge:+.0f}"
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
            "L0_WARMUP": "WARMUP",
            "L1_DIRECTION": "DIRECTION",
            "L2_SETUP": "S/R",
            "L3_TRIGGER": "REACTION",
            "RISK": "RISK",
            "READY": "READY",
        }.get(stage, stage)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        """Compact production status showing the active S/R rule."""
        d = self.evaluate(df4h, df1h, df15, df5)
        side = d.side.value if d.side else "NONE"
        stage = self._stage_label(d.stage)
        status = "READY" if d.ready else "WAIT"

        parts = [
            f"SR {status}",
            side,
            f"Trend={d.direction.score:.0f}/{d.direction.tier}",
            f"Edge={d.direction.edge:+.0f}",
        ]
        if d.quality is not None:
            parts.append(f"Q={d.quality.q:.1f}")
        if d.context is not None:
            loc = d.context.location
            trigger = d.execution[0] if d.execution else "WAIT"
            parts.extend(
                [
                    f"Level={loc.zone}",
                    f"Room={loc.room_atr:.2f}ATR",
                    f"Trigger={trigger}",
                    f"Conf={d.trade_score:.0f}",
                ]
            )

        reason = d.blocker or "S/R reaction passed"
        parts.extend([f"Stage={stage}", f"Reason={reason}"])
        return " | ".join(parts)


MTFStructureStrategyV12 = PrecisionTrendStructureV12
