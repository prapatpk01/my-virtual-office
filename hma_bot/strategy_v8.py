"""HMA Expert MTF V4.2 — single-pass Sentinel decision pipeline.

V4.2 removes duplicated decision logic. Trend, quality, location, execution,
room and trade score are evaluated once into DecisionState. Both /status and
order generation consume that same state, so the log can never disagree with
entry eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import strategy as legacy
import strategy_v7 as v7

Side = legacy.Side
Trend = legacy.Trend
SetupType = legacy.SetupType
EntrySignal = legacy.EntrySignal


@dataclass(frozen=True)
class DecisionState:
    ready: bool
    stage: str
    blocker: str
    direction: v7.AdaptiveDirection
    quality: object
    context: object | None
    setup_type: SetupType | None
    execution: tuple | None
    trade_score: float

    @property
    def side(self) -> Optional[Side]:
        return self.direction.side


class PrecisionTrendStructureV8(v7.PrecisionTrendStructureV7):
    """One authoritative evaluation path for status and entries."""

    def evaluate(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        df5: pd.DataFrame,
    ) -> DecisionState:
        if len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            direction = self._adaptive_direction(df4h) if len(df4h) >= 60 else v7.AdaptiveDirection(None, 0, 0, 0, "NEUTRAL")
            quality = self.quality_state_1h(df1h) if len(df1h) >= 60 else None
            return DecisionState(False, "WARMUP", "insufficient candles", direction, quality, None, None, None, 0.0)

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

        if direction.tier == "EARLY":
            min_loc = self.early_location_min
            zone_ok = loc.near_s2 if direction.side == Side.LONG else loc.near_r2
            required = "S2" if direction.side == Side.LONG else "R2"
        elif direction.tier == "MODERATE":
            min_loc = self.sentinel_location_min
            zone_ok = loc.near_s2 if direction.side == Side.LONG else loc.near_r2
            required = "S2" if direction.side == Side.LONG else "R2"
        else:
            min_loc = self.sentinel_location_min
            zone_ok = (loc.near_s1 or loc.near_s2) if direction.side == Side.LONG else (loc.near_r1 or loc.near_r2)
            required = "S1/S2" if direction.side == Side.LONG else "R1/R2"

        if not zone_ok:
            score = self._trade_score(direction.score, quality.q, loc.score, False, loc.room_atr)
            return DecisionState(False, "LOCATION", f"need {required}; now {loc.zone}", direction, quality, context, setup_type, None, score)
        if loc.score < min_loc:
            score = self._trade_score(direction.score, quality.q, loc.score, False, loc.room_atr)
            return DecisionState(False, "LOCATION", f"location {loc.score:.0f}<{min_loc:.0f}", direction, quality, context, setup_type, None, score)

        execution = self._execution_trigger(df5, direction.side, setup_type)
        score = self._trade_score(direction.score, quality.q, loc.score, execution is not None, loc.room_atr)
        if execution is None:
            return DecisionState(False, "EXECUTION", "waiting recent 5M trigger", direction, quality, context, setup_type, None, score)
        if score < self.trade_score_min:
            return DecisionState(False, "SCORE", f"trade {score:.0f}<{self.trade_score_min:.0f}", direction, quality, context, setup_type, execution, score)

        return DecisionState(True, "READY", "", direction, quality, context, setup_type, execution, score)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        d = self.evaluate(df4h, df1h, df15, df5)
        q = d.quality
        side = d.side.value if d.side else "NONE"
        head = f"V4.2 {d.stage} | {side} Trend={d.direction.score:.0f}/{d.direction.tier} edge={d.direction.edge:+.0f}"
        if q is not None:
            head += f" | Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f}"
        if d.context is not None:
            loc = d.context.location
            trigger = d.execution[0] if d.execution else "WAIT"
            head += f" | Loc={loc.zone} {loc.score:.0f} Room={loc.room_atr:.2f}ATR | Exec={trigger} | Trade={d.trade_score:.0f}"
        if d.blocker:
            head += f" | BLOCK {d.blocker}"
        return head

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False) -> Optional[EntrySignal]:
        if has_open_position:
            return None
        d = self.evaluate(df4h, df1h, df15, df5)
        if not d.ready or d.side is None or d.context is None or d.setup_type is None or d.execution is None:
            return None

        side = d.side
        quality = d.quality
        loc = d.context.location
        trigger_name, _ = d.execution

        d15 = df15.copy()
        d15["atr"] = self._atr(d15, self.cfg.atr_len)
        atr15 = float(d15["atr"].iloc[-1])
        if not np.isfinite(atr15) or atr15 <= 0:
            return None

        entry = float(df5["close"].iloc[-1])
        structure_level = self._structure_level(d.context, side, entry)
        sl = self._structure_stop(entry, side, atr15, self._structure(d15), structure_level)
        tp = entry * (1.0 + self.cfg.final_take_profit_pct) if side == Side.LONG else entry * (1.0 - self.cfg.final_take_profit_pct)
        risk = abs(entry - sl)
        rr = abs(tp - entry) / max(risk, 1e-12)
        if rr < self.min_actual_rr:
            return None

        reason = (
            f"Sentinel V4.2 {d.direction.tier} {side.value} | Trend {d.direction.score:.0f} "
            f"edge {d.direction.edge:+.0f} | Q {quality.q:.0f} | {loc.zone} "
            f"Location {loc.score:.0f} ({loc.reason}) | {trigger_name} | "
            f"Trade {d.trade_score:.0f} | Room {loc.room_atr:.2f}ATR | RR {rr:.2f}"
        )
        return EntrySignal(
            side=side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=Trend.BULL if side == Side.LONG else Trend.BEAR,
            q_1h=quality.q,
            adx_1h=quality.adx,
            chop_1h=quality.chop,
            setup=d.setup_type,
            trigger=trigger_name,
            room_pct=max(0.0, loc.room_atr * atr15 / max(entry, 1e-12)),
            atr15=atr15,
            structure_level=structure_level,
            reason=reason,
        )


MTFStructureStrategyV8 = PrecisionTrendStructureV8
