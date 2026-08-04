"""HMA Expert MTF V5.0 — 1H-led Sentinel S/R with 4H soft bias.

Authoritative decision path:
    1H direction + quality -> 15M Sentinel S/R -> 5M execution -> trade score

The 4H timeframe is no longer a hard gate.  It contributes only a 0/50/100
macro-bias component to the final score, so a slow 4H neutral/flip cannot block
a valid early 1H move.

Weights:
    1H trend/quality 35%
    15M location     35%
    5M execution     20%
    4H macro bias    10%

Entry policy:
    score >= 80: normal entry
    score 70..79: only deep S2/R2 or a confirmed liquidity sweep, location >=75
    score < 70: no entry
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

import strategy_v9 as v9
from sentinel_context import build_context, trend_score_4h

Side = v9.Side
DecisionState = v9.DecisionState
EntrySignal = v9.EntrySignal


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PrecisionTrendStructureV10(v9.PrecisionTrendStructureV9):
    """Fast 1H/15M/5M core with non-blocking 4H macro context."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.one_h_early_min = _env_float("ONE_H_EARLY_TREND_MIN", 60.0)
        self.one_h_strong_min = _env_float("ONE_H_STRONG_TREND_MIN", 70.0)
        self.one_h_direction_edge_min = _env_float("ONE_H_DIRECTION_EDGE_MIN", 7.0)
        self.v5_entry_score = _env_float("V5_ENTRY_SCORE", 80.0)
        self.v5_conditional_score = _env_float("V5_CONDITIONAL_SCORE", 70.0)
        self.v5_conditional_location = _env_float("V5_CONDITIONAL_LOCATION", 75.0)

    def _one_h_direction(self, df1h):
        """Choose direction from 1H; blend directional trend with market quality."""
        quality = self.quality_state_1h(df1h)
        long_raw = float(trend_score_4h(df1h, "long"))
        short_raw = float(trend_score_4h(df1h, "short"))

        if long_raw >= short_raw:
            side = Side.LONG
            raw, opposite = long_raw, short_raw
        else:
            side = Side.SHORT
            raw, opposite = short_raw, long_raw

        # Q is direction-neutral, so it improves conviction but never chooses side.
        score = float(np.clip(raw * 0.72 + quality.q * 0.28, 0.0, 100.0))
        edge = raw - opposite

        if score < self.one_h_early_min or edge < self.one_h_direction_edge_min:
            return v9.v8.v7.AdaptiveDirection(None, score, opposite, edge, "NEUTRAL"), quality
        tier = "STRONG" if score >= self.one_h_strong_min else "EARLY"
        return v9.v8.v7.AdaptiveDirection(side, score, opposite, edge, tier), quality

    @staticmethod
    def _macro_bias(df4h, side: Side) -> tuple[float, str, float]:
        """Return macro score (0/50/100), label, and directional edge."""
        long_score = float(trend_score_4h(df4h, "long"))
        short_score = float(trend_score_4h(df4h, "short"))
        aligned = long_score if side == Side.LONG else short_score
        opposed = short_score if side == Side.LONG else long_score
        edge = aligned - opposed
        if edge >= 8.0:
            return 100.0, "ALIGNED", edge
        if edge <= -8.0:
            return 0.0, "OPPOSED", edge
        return 50.0, "NEUTRAL", edge

    def _v5_trade_score(
        self,
        direction_score: float,
        location_score: float,
        execution_ready: bool,
        macro_score: float,
    ) -> float:
        execution_score = 100.0 if execution_ready else 0.0
        return float(np.clip(
            direction_score * 0.35
            + location_score * 0.35
            + execution_score * 0.20
            + macro_score * 0.10,
            0.0,
            100.0,
        ))

    def _v5_zone_policy(self, direction, loc) -> tuple[bool, str, float]:
        is_long = direction.side == Side.LONG
        if direction.tier == "EARLY":
            ok = loc.near_s2 if is_long else loc.near_r2
            return ok, ("S2" if is_long else "R2"), self.early_location_min

        ok = (
            (loc.near_s1 or loc.near_s2)
            if is_long
            else (loc.near_r1 or loc.near_r2)
        )
        return ok, ("S1/S2" if is_long else "R1/R2"), self.sentinel_location_min

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

        # 4H is supplied for HTF S/R confluence inside Sentinel context, but it
        # cannot veto direction. Direction is exclusively chosen by 1H above.
        context = build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="long" if direction.side == Side.LONG else "short",
        )
        setup_type = self._setup_from_context(context)
        loc = context.location
        macro_score, macro_label, macro_edge = self._macro_bias(df4h, direction.side)

        if loc.room_atr < self.sentinel_min_room_atr:
            score = self._v5_trade_score(direction.score, loc.score, False, macro_score)
            blocker = f"room {loc.room_atr:.2f}<{self.sentinel_min_room_atr:.2f}ATR | 4H {macro_label} {macro_edge:+.0f}"
            return DecisionState(False, "15M_LOCATION", blocker, direction, quality, context, setup_type, None, score)

        zone_ok, required, min_loc = self._v5_zone_policy(direction, loc)
        if not zone_ok:
            score = self._v5_trade_score(direction.score, loc.score, False, macro_score)
            blocker = f"need {required}; now {loc.zone} | 4H {macro_label} {macro_edge:+.0f}"
            return DecisionState(False, "15M_LOCATION", blocker, direction, quality, context, setup_type, None, score)
        if loc.score < min_loc:
            score = self._v5_trade_score(direction.score, loc.score, False, macro_score)
            return DecisionState(False, "15M_LOCATION", f"location {loc.score:.0f}<{min_loc:.0f} | 4H {macro_label}", direction, quality, context, setup_type, None, score)

        execution = self._execution_trigger(df5, direction.side, setup_type)
        score = self._v5_trade_score(direction.score, loc.score, execution is not None, macro_score)
        if execution is None:
            return DecisionState(False, "5M_EXECUTION", f"{loc.zone} armed; waiting trigger | 4H {macro_label}", direction, quality, context, setup_type, None, score)

        deep_zone = loc.near_s2 if direction.side == Side.LONG else loc.near_r2
        conditional_ok = (
            score >= self.v5_conditional_score
            and loc.score >= self.v5_conditional_location
            and (deep_zone or loc.sweep)
        )
        if score < self.v5_entry_score and not conditional_ok:
            blocker = (
                f"score {score:.0f}<{self.v5_entry_score:.0f}; "
                f"70-79 needs deep S2/R2 or sweep + Loc≥{self.v5_conditional_location:.0f} | "
                f"4H {macro_label}"
            )
            return DecisionState(False, "SCORE", blocker, direction, quality, context, setup_type, execution, score)

        return DecisionState(True, "READY", f"4H {macro_label} {macro_edge:+.0f}", direction, quality, context, setup_type, execution, score)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        d = self.evaluate(df4h, df1h, df15, df5)
        q = d.quality
        side = d.side.value if d.side else "NONE"
        text = (
            f"V5 {d.stage} | {side} 1H={d.direction.score:.0f}/{d.direction.tier} "
            f"edge={d.direction.edge:+.0f}"
        )
        if q is not None:
            text += f" Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f}"
        if d.context is not None:
            loc = d.context.location
            trigger = d.execution[0] if d.execution else "WAIT"
            macro_score, macro_label, macro_edge = self._macro_bias(df4h, d.side)
            text += (
                f" | 15M {loc.zone} Loc={loc.score:.0f} Room={loc.room_atr:.2f}ATR"
                f" | 5M={trigger} | 4H={macro_label}({macro_edge:+.0f})"
                f" | Trade={d.trade_score:.0f}"
            )
        if d.blocker:
            text += f" | {'INFO' if d.ready else 'BLOCK'} {d.blocker}"
        return text

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False) -> Optional[EntrySignal]:
        signal = super().generate_entry(df4h, df1h, df15, df5, has_open_position)
        if signal is None:
            return None
        decision = self.evaluate(df4h, df1h, df15, df5)
        if not decision.ready or decision.context is None:
            return None

        loc = decision.context.location
        macro_score, macro_label, macro_edge = self._macro_bias(df4h, decision.side)
        reason = (
            f"HMA V5 1H-led {decision.side.value} | 1H {decision.direction.score:.0f}/"
            f"{decision.direction.tier} Q {decision.quality.q:.0f} | "
            f"15M {loc.zone} Location {loc.score:.0f} ({loc.reason}) | "
            f"5M {decision.execution[0]} | 4H soft {macro_label} {macro_edge:+.0f} | "
            f"Trade {decision.trade_score:.0f} | {signal.reason}"
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
            reason=reason,
        )


MTFStructureStrategyV10 = PrecisionTrendStructureV10
