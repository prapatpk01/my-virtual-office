"""HMA Expert MTF V4.1 — adaptive Sentinel trend + location decisions.

V4.0 still depended on the legacy discrete 4H BULL/BEAR/NEUTRAL gate before
Sentinel location was evaluated.  V4.1 removes that bottleneck for entries:

- Compute independent 4H Sentinel LONG and SHORT trend scores (0..100).
- Select the stronger side only when score >= 60 and the directional edge is
  meaningful; otherwise remain neutral.
- STRONG >=85: S1/S2 for long, R1/R2 for short.
- MODERATE 70..84: S2 for long, R2 for short.
- EARLY 60..69: S2/R2 only, location >=70, Q >=60.
- Every status string exposes Trend, Q, Location, Execution, Room and the
  estimated Trade Score so Railway logs show the real V4.1 decision path.

Risk, two-stage locks, OKX recovery and FX 24/5 are inherited unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import strategy as legacy
import strategy_v6 as v6
from sentinel_context import SentinelContext, build_context, trend_score_4h

Side = legacy.Side
Trend = legacy.Trend
SetupType = legacy.SetupType
EntrySignal = legacy.EntrySignal


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AdaptiveDirection:
    side: Optional[Side]
    score: float
    opposite_score: float
    edge: float
    tier: str


class PrecisionTrendStructureV7(v6.PrecisionTrendStructureV6):
    """Sentinel V4 location with adaptive score-based 4H direction."""

    def __init__(self, config: Optional[legacy.StrategyConfig] = None) -> None:
        super().__init__(config)
        self.early_trend_min = _env_float("SENTINEL_EARLY_TREND_MIN", 60.0)
        self.direction_edge_min = _env_float("SENTINEL_DIRECTION_EDGE_MIN", 8.0)
        self.early_location_min = _env_float("SENTINEL_EARLY_LOCATION_MIN", 70.0)
        self.early_quality_min = _env_float("SENTINEL_EARLY_Q_MIN", 60.0)
        self.trade_score_min = _env_float("SENTINEL_TRADE_SCORE_MIN", 75.0)

    def _adaptive_direction(self, df4h: pd.DataFrame) -> AdaptiveDirection:
        long_score = float(trend_score_4h(df4h, "long"))
        short_score = float(trend_score_4h(df4h, "short"))
        if long_score >= short_score:
            side = Side.LONG
            score, opposite = long_score, short_score
        else:
            side = Side.SHORT
            score, opposite = short_score, long_score
        edge = score - opposite

        if score < self.early_trend_min or edge < self.direction_edge_min:
            return AdaptiveDirection(None, score, opposite, edge, "NEUTRAL")
        if score >= self.sentinel_strong_trend_min:
            tier = "STRONG"
        elif score >= self.sentinel_moderate_trend_min:
            tier = "MODERATE"
        else:
            tier = "EARLY"
        return AdaptiveDirection(side, score, opposite, edge, tier)

    def _context_for_direction(
        self,
        df15: pd.DataFrame,
        df1h: pd.DataFrame,
        df4h: pd.DataFrame,
        direction: AdaptiveDirection,
    ) -> SentinelContext:
        assert direction.side is not None
        return build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="long" if direction.side == Side.LONG else "short",
        )

    def _adaptive_location_allowed(
        self,
        direction: AdaptiveDirection,
        context: SentinelContext,
        quality_q: float,
    ) -> bool:
        if direction.side is None:
            return False
        loc = context.location
        if loc.room_atr < self.sentinel_min_room_atr:
            return False

        if direction.tier == "EARLY":
            if quality_q < self.early_quality_min or loc.score < self.early_location_min:
                return False
            return loc.near_s2 if direction.side == Side.LONG else loc.near_r2

        if loc.score < self.sentinel_location_min:
            return False
        if direction.tier == "MODERATE":
            return loc.near_s2 if direction.side == Side.LONG else loc.near_r2
        return (
            (loc.near_s1 or loc.near_s2)
            if direction.side == Side.LONG
            else (loc.near_r1 or loc.near_r2)
        )

    @staticmethod
    def _trade_score(
        trend_score: float,
        quality_score: float,
        location_score: float,
        execution_ready: bool,
        room_atr: float,
    ) -> float:
        execution = 85.0 if execution_ready else 45.0
        room_score = float(np.clip(room_atr / 1.5 * 100.0, 0.0, 100.0))
        return float(np.clip(
            trend_score * 0.28
            + quality_score * 0.22
            + location_score * 0.28
            + execution * 0.15
            + room_score * 0.07,
            0.0,
            100.0,
        ))

    def _decision_snapshot(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        df5: pd.DataFrame,
    ):
        direction = self._adaptive_direction(df4h)
        quality = self.quality_state_1h(df1h)
        if direction.side is None:
            return direction, quality, None, None, None, 0.0

        context = self._context_for_direction(df15, df1h, df4h, direction)
        setup_type = self._setup_from_context(context)
        execution = self._execution_trigger(df5, direction.side, setup_type)
        trade_score = self._trade_score(
            direction.score,
            quality.q,
            context.location.score,
            execution is not None,
            context.location.room_atr,
        )
        return direction, quality, context, setup_type, execution, trade_score

    def entry_status(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        df5: pd.DataFrame,
    ) -> str:
        if len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            return "V4.1 WARMUP"

        direction, quality, context, setup_type, execution, trade_score = (
            self._decision_snapshot(df4h, df1h, df15, df5)
        )
        side_text = direction.side.value if direction.side else "NONE"
        base = (
            f"V4.1 Trend={direction.score:.0f}/{direction.tier} {side_text} "
            f"edge={direction.edge:+.0f} | Q={quality.q:.0f} "
            f"ADX={quality.adx:.1f} CHOP={quality.chop:.1f}"
        )

        if direction.side is None:
            return base + " | BLOCK adaptive direction"
        if quality.q < self.cfg.min_trend_quality:
            return base + f" | BLOCK Q<{self.cfg.min_trend_quality:.0f}"
        if direction.tier == "EARLY" and quality.q < self.early_quality_min:
            return base + f" | BLOCK EARLY Q<{self.early_quality_min:.0f}"
        if not self._dmi_aligned(quality, direction.side):
            edge = self._dmi_edge(quality, direction.side)
            return base + f" | BLOCK DMI {edge:+.1f}"

        assert context is not None
        loc = context.location
        exec_text = execution[0] if execution else "WAIT"
        detail = (
            f" | Loc={loc.zone} {loc.score:.0f} ({loc.reason})"
            f" | Room={loc.room_atr:.2f}ATR"
            f" | Exec={exec_text}"
            f" | Trade={trade_score:.0f}"
        )

        if not self._adaptive_location_allowed(direction, context, quality.q):
            required = (
                "S2" if direction.side == Side.LONG else "R2"
            ) if direction.tier in ("EARLY", "MODERATE") else (
                "S1/S2" if direction.side == Side.LONG else "R1/R2"
            )
            return base + detail + f" | BLOCK need {required}"
        if execution is None:
            return base + detail + " | ARMED wait 5M"
        if trade_score < self.trade_score_min:
            return base + detail + f" | BLOCK Trade<{self.trade_score_min:.0f}"
        return base + detail + " | READY"

    def generate_entry(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        df5: pd.DataFrame,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        if has_open_position or len(df4h) < 60 or len(df1h) < 60 or len(df15) < 90 or len(df5) < 70:
            return None

        direction, quality, context, setup_type, execution, trade_score = (
            self._decision_snapshot(df4h, df1h, df15, df5)
        )
        if direction.side is None or context is None or setup_type is None:
            return None
        side = direction.side

        if quality.q < self.cfg.min_trend_quality:
            return None
        if direction.tier == "EARLY" and quality.q < self.early_quality_min:
            return None
        if not self._dmi_aligned(quality, side):
            return None
        if not self._adaptive_location_allowed(direction, context, quality.q):
            return None
        if execution is None or trade_score < self.trade_score_min:
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
        reason = (
            f"Sentinel V4.1 {direction.tier} {side.value} | "
            f"Trend {direction.score:.0f} edge {direction.edge:+.0f} | "
            f"Q {quality.q:.0f} | Location {loc.zone} {loc.score:.0f}/100 "
            f"({loc.reason}) | Execution {trigger_name} | "
            f"TradeScore {trade_score:.0f}/100 | Room {loc.room_atr:.2f}ATR | RR {rr:.2f}"
        )

        # Keep the legacy Trend enum only for downstream display compatibility;
        # entry direction itself comes from the adaptive Sentinel score above.
        compat_trend = Trend.BULL if side == Side.LONG else Trend.BEAR
        return EntrySignal(
            side=side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=compat_trend,
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


MTFStructureStrategyV7 = PrecisionTrendStructureV7
