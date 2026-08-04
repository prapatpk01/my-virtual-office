"""HMA Gate Sentinel — simplified three-layer production strategy.

Only four decisions are authoritative:

    Layer 1: 1H direction
    Layer 2: 15M setup area
    Layer 3: recent closed-5M trigger
    Risk:    structure SL, room and actual R:R

4H and confidence remain diagnostic only.  Status and real order creation use
this same evaluation result; no legacy ``super().generate_entry()`` chain is
allowed to re-apply hidden gates.
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
    """Fast 1H -> 15M -> 5M strategy with one final risk check."""

    def __init__(self, config=None) -> None:
        super().__init__(config)

        # Layer 1: direction only. Quality is not blended into direction and is
        # checked once below, preventing the same information being gated twice.
        self.one_h_early_min = _env_float("ONE_H_EARLY_TREND_MIN", 55.0)
        self.one_h_strong_min = _env_float("ONE_H_STRONG_TREND_MIN", 68.0)
        self.one_h_direction_edge_min = _env_float("ONE_H_DIRECTION_EDGE_MIN", 4.0)
        self.quality_min = _env_float("MIN_TREND_QUALITY", 52.0)
        self.dmi_hard_opposition = _env_float("DMI_HARD_OPPOSITION", 10.0)

        # Layer 2: a broad setup area. Exact S/R contact is useful but no longer
        # mandatory; an orderly 15M EMA20 pullback in the selected direction is
        # also a valid location.
        self.setup_proximity_atr = _env_float("SETUP_PROXIMITY_ATR", 0.45)
        self.pullback_touch_atr = _env_float("PULLBACK_TOUCH_ATR", 0.18)
        self.pullback_max_extension_atr = _env_float(
            "PULLBACK_MAX_EXTENSION_ATR", 0.90
        )
        self.pullback_lookback = max(2, _env_int("PULLBACK_LOOKBACK_BARS", 3))

        # Layer 3: recent 5M confirmation. Alignment must still be valid, so a
        # six-bar window does not permit stale or reversed triggers.
        self.exec_trigger_lookback = max(
            1, _env_int("EXEC_TRIGGER_LOOKBACK_BARS", 6)
        )
        self.exec_max_chase_atr = _env_float("EXEC_MAX_CHASE_ATR", 1.35)
        self.exec_min_body_atr = _env_float("EXEC_MIN_BODY_ATR", 0.12)
        self.exec_min_close_location = _env_float(
            "EXEC_MIN_CLOSE_LOCATION", 0.52
        )
        self.exec_break_body_atr = _env_float("EXEC_BREAK_BODY_ATR", 0.08)

        # One combined risk check after a valid trigger.
        self.min_room_atr = _env_float("MIN_ENTRY_ROOM_ATR", 0.30)
        self.min_actual_rr = _env_float("MIN_ACTUAL_RR", 0.90)

    def _simple_direction(self, df1h):
        """Select the stronger 1H side without double-counting Q or 4H."""
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
            return AdaptiveDirection(None, score, opposite, edge, "NEUTRAL"), quality

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
        sl = self._structure_stop(entry, side, atr15, ctx15, structure_level)
        tp = (
            entry * (1.0 + self.cfg.final_take_profit_pct)
            if side == Side.LONG
            else entry * (1.0 - self.cfg.final_take_profit_pct)
        )
        risk = abs(entry - sl)
        rr = abs(tp - entry) / max(risk, 1e-12)
        return entry, sl, tp, atr15, structure_level, rr

    def _setup_area(self, context, direction, df15):
        """Accept Sentinel S/R proximity or a clean 15M EMA20 pullback."""
        d15 = df15.copy()
        d15["atr"] = self._atr(d15, self.cfg.atr_len)
        d15["ema20"] = self._ema(d15["close"], 20)

        atr15 = float(d15["atr"].iloc[-1])
        if not np.isfinite(atr15) or atr15 <= 0.0:
            return context, False, "15M ATR unavailable"

        price = float(d15["close"].iloc[-1])
        ema20 = float(d15["ema20"].iloc[-1])
        ema20_prev = float(d15["ema20"].iloc[-3])
        recent = d15.iloc[-self.pullback_lookback :]

        effective_loc, corridor = self._effective_location(
            context.location, direction.side, price
        )
        loc = effective_loc
        is_long = direction.side == Side.LONG

        if is_long:
            levels = [
                float(value)
                for value in (loc.s1, loc.s2)
                if value is not None and np.isfinite(float(value))
            ]
            near_level = any(
                abs(price - level) <= self.setup_proximity_atr * atr15
                for level in levels
            )
            exact_zone = loc.near_s1 or loc.near_s2 or corridor or near_level
            pullback = (
                price > ema20
                and ema20 > ema20_prev
                and float(recent["low"].min())
                <= ema20 + self.pullback_touch_atr * atr15
                and price - ema20 <= self.pullback_max_extension_atr * atr15
            )
        else:
            levels = [
                float(value)
                for value in (loc.r1, loc.r2)
                if value is not None and np.isfinite(float(value))
            ]
            near_level = any(
                abs(price - level) <= self.setup_proximity_atr * atr15
                for level in levels
            )
            exact_zone = loc.near_r1 or loc.near_r2 or corridor or near_level
            pullback = (
                price < ema20
                and ema20 < ema20_prev
                and float(recent["high"].max())
                >= ema20 - self.pullback_touch_atr * atr15
                and ema20 - price <= self.pullback_max_extension_atr * atr15
            )

        if exact_zone:
            label = loc.zone if loc.zone != "NONE" else "S/R PROXIMITY"
            reason = f"{label} setup area"
            score = max(float(loc.score), 60.0)
        elif pullback:
            label = "EMA20_PULLBACK"
            reason = "15M EMA20 pullback aligned with 1H direction"
            score = max(float(loc.score), 58.0)
        else:
            expected = "support or EMA20 pullback" if is_long else "resistance or EMA20 pullback"
            return replace(context, location=loc), False, f"waiting {expected}"

        updated_loc = replace(
            loc,
            zone=label,
            score=float(np.clip(score, 0.0, 100.0)),
            reason=reason,
        )
        return replace(context, location=updated_loc), True, reason

    def evaluate(self, df4h, df1h, df15, df5) -> DecisionState:
        if len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            direction = AdaptiveDirection(None, 0.0, 0.0, 0.0, "WARMUP")
            quality = self.quality_state_1h(df1h) if len(df1h) >= 60 else None
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

        # Layer 1 — 1H direction and one quality check.
        direction, quality = self._simple_direction(df1h)
        if direction.side is None:
            why = (
                f"1H trend {direction.score:.0f}<{self.one_h_early_min:.0f}"
                if direction.score < self.one_h_early_min
                else f"1H edge {direction.edge:.0f}<{self.one_h_direction_edge_min:.0f}"
            )
            return DecisionState(
                False, "L1_DIRECTION", why, direction, quality, None, None, None, 0.0
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

        # Layer 2 — broad 15M setup area.
        context = build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="long" if direction.side == Side.LONG else "short",
        )
        context, setup_ok, setup_reason = self._setup_area(context, direction, df15)
        setup_type = self._setup_from_context(context)
        loc = context.location
        macro_score, macro_label, macro_edge = self._macro_bias(df4h, direction.side)
        confidence = self._v51_trade_score(
            direction.score, loc.score, False, macro_score
        )

        if not setup_ok:
            return DecisionState(
                False,
                "L2_SETUP",
                f"{setup_reason} | 4H {macro_label} {macro_edge:+.0f}",
                direction,
                quality,
                context,
                setup_type,
                None,
                confidence,
            )

        # Layer 3 — recent closed-5M trigger.
        execution = self._execution_trigger(df5, direction.side, setup_type)
        confidence = self._v51_trade_score(
            direction.score, loc.score, execution is not None, macro_score
        )
        if execution is None:
            return DecisionState(
                False,
                "L3_TRIGGER",
                f"{loc.zone} armed; waiting recent 5M trigger",
                direction,
                quality,
                context,
                setup_type,
                None,
                confidence,
            )

        # One final risk check — room and actual R:R are evaluated together.
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
        if loc.room_atr < self.min_room_atr:
            return DecisionState(
                False,
                "RISK",
                f"room {loc.room_atr:.2f}<{self.min_room_atr:.2f} ATR",
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
            f"3 layers passed | RR {rr:.2f} | 4H {macro_label} {macro_edge:+.0f}",
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
        """Create an order only from the same simplified decision shown in status."""
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
        room_pct = max(0.0, loc.room_atr * atr15 / max(entry, 1e-12))
        reason = (
            f"Simple Sentinel {side.value} | 1H Trend {decision.direction.score:.0f} "
            f"edge {decision.direction.edge:+.0f} Q {decision.quality.q:.0f} | "
            f"15M {loc.zone} ({loc.reason}) | 5M {trigger_name} | "
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
            "L2_SETUP": "SETUP",
            "L3_TRIGGER": "TRIGGER",
            "RISK": "RISK",
            "READY": "READY",
        }.get(stage, stage)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        """Compact production status showing the first real blocker."""
        d = self.evaluate(df4h, df1h, df15, df5)
        side = d.side.value if d.side else "NONE"
        stage = self._stage_label(d.stage)
        status = "READY" if d.ready else "WAIT"

        parts = [
            f"SIMPLE {status}",
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
                    f"Setup={loc.zone}",
                    f"Room={loc.room_atr:.2f}ATR",
                    f"Trigger={trigger}",
                    f"Conf={d.trade_score:.0f}",
                ]
            )

        reason = d.blocker or "3 layers passed"
        parts.extend([f"Stage={stage}", f"Reason={reason}"])
        return " | ".join(parts)


MTFStructureStrategyV12 = PrecisionTrendStructureV12
