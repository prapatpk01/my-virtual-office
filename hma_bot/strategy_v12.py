"""TPC Dynamic Zone V6.1 — early, non-repainting trend-pullback entries.

One authoritative path:
    1H direction/Q -> 15M dynamic location -> 15M execution

Supply/demand zones use confirmed 15M pivots plus displacement.  They are
invalidated only by closed candles and never move after confirmation.  A
lightweight EMA13 pullback fallback keeps the strategy active when no clean
structural zone exists. Counter-trend entries are never generated. HMA16 flips
are the early trigger; EMA13 reclaims additionally require 4H alignment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import strategy_v11 as v11
from sentinel_context import trend_score_4h

Side = v11.Side
DecisionState = v11.DecisionState
EntrySignal = v11.EntrySignal
AdaptiveDirection = v11.v10.v9.v8.v7.AdaptiveDirection
Trend = v11.v10.v9.v8.v7.Trend
SetupType = v11.v10.v9.v8.SetupType


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


@dataclass(frozen=True)
class DynamicZone:
    kind: str
    lower: float
    upper: float
    pivot_index: int
    strength: float

    @property
    def label(self) -> str:
        return f"{self.kind.upper()}[{self.lower:.6g}-{self.upper:.6g}]"


@dataclass(frozen=True)
class LocationView:
    zone: str
    room_atr: float
    reason: str
    score: float = 0.0


@dataclass(frozen=True)
class DynamicContext:
    location: LocationView
    active_zone: Optional[DynamicZone]
    opposing_zone: Optional[DynamicZone]
    mode: str
    reaction: str


class PrecisionTrendStructureV12(v11.PrecisionTrendStructureV11):
    """1H trend with confirmed location and closed-15M execution."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.direction_min = _env_float("TPC_DIRECTION_MIN", 52.0)
        self.direction_edge_min = _env_float("TPC_DIRECTION_EDGE_MIN", 3.0)
        self.quality_min = _env_float("TPC_QUALITY_MIN", 55.0)
        self.hard_chop = _env_float("TPC_HARD_CHOP", 68.0)
        self.hard_adx = _env_float("TPC_HARD_ADX", 11.0)
        self.dmi_opposition = _env_float("TPC_DMI_OPPOSITION", 8.0)

        self.zone_lookback = max(30, _env_int("TPC_ZONE_LOOKBACK", 100))
        self.zone_pivot_span = max(2, _env_int("TPC_ZONE_PIVOT_SPAN", 2))
        self.zone_displacement_atr = _env_float("TPC_ZONE_DISPLACEMENT_ATR", 0.25)
        self.zone_body_atr = _env_float("TPC_ZONE_BODY_ATR", 0.50)
        self.zone_width_atr = _env_float("TPC_ZONE_WIDTH_ATR", 0.65)
        self.zone_invalidation_atr = _env_float("TPC_ZONE_INVALIDATION_ATR", 0.15)
        self.zone_touch_atr = _env_float("TPC_ZONE_TOUCH_ATR", 0.20)
        self.zone_reaction_bars = max(1, _env_int("TPC_ZONE_REACTION_BARS", 3))
        self.fallback_touch_atr = _env_float("TPC_FALLBACK_TOUCH_ATR", 0.30)
        self.min_room_atr = _env_float("TPC_MIN_ROOM_ATR", 0.90)

        self.cross_lookback = max(1, _env_int("TPC_5M_CROSS_LOOKBACK", 4))
        self.flip_lookback = max(1, _env_int("TPC_5M_HMA_FLIP_LOOKBACK", 3))
        self.min_body_atr = _env_float("TPC_5M_MIN_BODY_ATR", 0.08)
        self.max_chase_atr = _env_float("TPC_MAX_CHASE_ATR", 1.10)

        self.sl_buffer_atr = _env_float("TPC_SL_BUFFER_ATR", 0.20)
        self.sl_atr = _env_float("TPC_SL_ATR", 1.35)
        self.sl_min_pct = _env_float("TPC_SL_MIN_PCT", 0.006)
        self.sl_max_pct = _env_float("TPC_SL_MAX_PCT", 0.010)
        self.target_buffer_atr = _env_float("TPC_TARGET_BUFFER_ATR", 0.10)
        self.macro_min = _env_float("TPC_4H_MIN", 52.0)
        self.macro_edge_min = _env_float("TPC_4H_EDGE_MIN", 3.0)
        self.stage_locks_enabled = os.environ.get(
            "TPC_STAGE_LOCKS_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}

        # The previous +0.7%/+1.1% locks converted too many valid runners into
        # small wins while full initial stops remained. V6.2 keeps native SL/TP
        # authoritative unless stage locks are explicitly re-enabled.
        self.tp_pct = _env_float("TPC_TP_PCT", 0.020)
        self.min_rr = _env_float("TPC_MIN_RR", 1.80)

    def locked_stop(self, side: Side, entry: float, best_price: float):
        if self.stage_locks_enabled:
            return super().locked_stop(side, entry, best_price)
        default_stop = (
            entry * (1.0 - self.cfg.stop_loss_pct)
            if side == Side.LONG else entry * (1.0 + self.cfg.stop_loss_pct)
        )
        return default_stop, 0

    def _macro_aligned(self, df4h: pd.DataFrame, side: Side) -> tuple[bool, float, float]:
        long_score = float(trend_score_4h(df4h, "long"))
        short_score = float(trend_score_4h(df4h, "short"))
        chosen = long_score if side == Side.LONG else short_score
        opposing = short_score if side == Side.LONG else long_score
        return (
            chosen >= self.macro_min
            and chosen - opposing >= self.macro_edge_min,
            chosen,
            opposing,
        )

    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    def _prepared(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = frame.copy()
        close = d["close"].astype(float)
        d["ema8"] = self._ema(close, 8)
        d["ema13"] = self._ema(close, 13)
        d["ema20"] = self._ema(close, 20)
        d["hma16"] = self._hma(close, 16)
        d["atr"] = self._atr(d, self.cfg.atr_len)
        return d

    def _direction_1h(self, df1h):
        quality = self.quality_state_1h(df1h)
        long_score = float(trend_score_4h(df1h, "long"))
        short_score = float(trend_score_4h(df1h, "short"))
        side = Side.LONG if long_score >= short_score else Side.SHORT
        score, opposite = max(long_score, short_score), min(long_score, short_score)
        edge = score - opposite
        if score < self.direction_min or edge < self.direction_edge_min:
            side = None
        tier = "STRONG" if score >= 68.0 else "TREND"
        return AdaptiveDirection(side, score, opposite, edge, tier), quality

    # Rolling-deploy compatibility with the previous runtime.
    def _simple_direction(self, df1h):
        return self._direction_1h(df1h)

    @staticmethod
    def _recent_cross(fast, slow, side: Side, lookback: int) -> bool:
        start = max(1, len(fast) - lookback)
        for i in range(start, len(fast)):
            if side == Side.LONG and fast.iloc[i] > slow.iloc[i] and fast.iloc[i - 1] <= slow.iloc[i - 1]:
                return True
            if side == Side.SHORT and fast.iloc[i] < slow.iloc[i] and fast.iloc[i - 1] >= slow.iloc[i - 1]:
                return True
        return False

    @staticmethod
    def _recent_hma_flip(hma, side: Side, lookback: int) -> bool:
        slope = hma.diff()
        start = max(2, len(hma) - lookback)
        for i in range(start, len(hma)):
            if side == Side.LONG and slope.iloc[i] > 0 and slope.iloc[i - 1] <= 0:
                return True
            if side == Side.SHORT and slope.iloc[i] < 0 and slope.iloc[i - 1] >= 0:
                return True
        return False

    def _dynamic_zones(self, d15: pd.DataFrame):
        """Build active zones from confirmed pivots; no current-bar pivots."""
        zones = []
        n = len(d15)
        span = self.zone_pivot_span
        start = max(span, n - self.zone_lookback)
        end = n - span - 3
        if end <= start:
            return [], []

        for i in range(start, end + 1):
            row = d15.iloc[i]
            atr = float(row["atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            window = d15.iloc[i - span : i + span + 1]
            forward = d15.iloc[i + span + 1 : i + span + 4]
            if forward.empty:
                continue

            pivot_low = float(row["low"]) <= float(window["low"].min())
            pivot_high = float(row["high"]) >= float(window["high"].max())
            bull_body = ((forward["close"] - forward["open"]).clip(lower=0)).max()
            bear_body = ((forward["open"] - forward["close"]).clip(lower=0)).max()
            bull_displacement = (
                float(forward["close"].max()) >= float(row["high"]) + self.zone_displacement_atr * atr
                or float(bull_body) >= self.zone_body_atr * atr
            )
            bear_displacement = (
                float(forward["close"].min()) <= float(row["low"]) - self.zone_displacement_atr * atr
                or float(bear_body) >= self.zone_body_atr * atr
            )
            later = d15.iloc[i + span + 1 :]

            if pivot_low and bull_displacement:
                lower = float(row["low"])
                upper = min(max(float(row["open"]), float(row["close"])), lower + self.zone_width_atr * atr)
                invalid = bool((later["close"] < lower - self.zone_invalidation_atr * atr).any())
                if not invalid and upper > lower:
                    strength = max(0.0, (float(forward["close"].max()) - float(row["high"])) / atr)
                    zones.append(DynamicZone("demand", lower, upper, i, strength))

            if pivot_high and bear_displacement:
                upper = float(row["high"])
                lower = max(min(float(row["open"]), float(row["close"])), upper - self.zone_width_atr * atr)
                invalid = bool((later["close"] > upper + self.zone_invalidation_atr * atr).any())
                if not invalid and upper > lower:
                    strength = max(0.0, (float(row["low"]) - float(forward["close"].min())) / atr)
                    zones.append(DynamicZone("supply", lower, upper, i, strength))

        price = float(d15["close"].iloc[-1])
        atr_now = float(d15["atr"].iloc[-1])
        demand = [z for z in zones if z.kind == "demand" and z.lower <= price + self.zone_touch_atr * atr_now]
        supply = [z for z in zones if z.kind == "supply" and z.upper >= price - self.zone_touch_atr * atr_now]
        demand.sort(key=lambda z: (abs(price - z.upper), -z.pivot_index))
        supply.sort(key=lambda z: (abs(z.lower - price), -z.pivot_index))
        return demand, supply

    def _zone_context(self, d15: pd.DataFrame, side: Side) -> DynamicContext:
        demand, supply = self._dynamic_zones(d15)
        active = demand[0] if side == Side.LONG and demand else supply[0] if side == Side.SHORT and supply else None
        opposing = supply[0] if side == Side.LONG and supply else demand[0] if side == Side.SHORT and demand else None
        atr = float(d15["atr"].iloc[-1])
        price = float(d15["close"].iloc[-1])
        recent = d15.iloc[-self.zone_reaction_bars :]

        reaction = "WAIT_ZONE"
        mode = "STRUCTURAL_ZONE"
        ready = False
        if active is not None and side == Side.LONG:
            touched = float(recent["low"].min()) <= active.upper + self.zone_touch_atr * atr
            swept = float(recent["low"].min()) < active.lower and price > active.upper
            held = touched and price >= active.upper
            ready = swept or held
            reaction = "DEMAND_SWEEP_RECLAIM" if swept else "DEMAND_HOLD" if held else "WAIT_DEMAND"
        elif active is not None:
            touched = float(recent["high"].max()) >= active.lower - self.zone_touch_atr * atr
            swept = float(recent["high"].max()) > active.upper and price < active.lower
            held = touched and price <= active.lower
            ready = swept or held
            reaction = "SUPPLY_SWEEP_RECLAIM" if swept else "SUPPLY_HOLD" if held else "WAIT_SUPPLY"

        if not ready:
            # Frequency-preserving trend-pullback fallback. It is disabled when
            # price is far from EMA13 or an opposing zone leaves no room.
            r = d15.iloc[-1]
            if side == Side.LONG:
                touched_ema = float(recent["low"].min()) <= float(r["ema13"]) + self.fallback_touch_atr * atr
                aligned = price > float(r["ema13"]) and float(r["ema13"]) >= float(d15["ema13"].iloc[-2])
            else:
                touched_ema = float(recent["high"].max()) >= float(r["ema13"]) - self.fallback_touch_atr * atr
                aligned = price < float(r["ema13"]) and float(r["ema13"]) <= float(d15["ema13"].iloc[-2])
            if touched_ema and aligned:
                active = None
                ready = True
                mode = "EMA13_FALLBACK"
                reaction = "EMA13_TREND_PULLBACK"

        if opposing is None:
            room_atr = 99.0
        elif side == Side.LONG:
            room_atr = max(0.0, (opposing.lower - price) / max(atr, 1e-12))
        else:
            room_atr = max(0.0, (price - opposing.upper) / max(atr, 1e-12))

        label = active.label if active else "EMA13" if ready else "NO_REACTION"
        location = LocationView(label, room_atr, reaction, 100.0 if ready else 0.0)
        return DynamicContext(location, active, opposing, mode, reaction)

    def _execution_trigger(self, d5: pd.DataFrame, side: Side, zone: Optional[DynamicZone]):
        if len(d5) < 30:
            return None
        r, p = d5.iloc[-1], d5.iloc[-2]
        atr = float(r["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None
        close, open_ = float(r["close"]), float(r["open"])
        body = abs(close - open_) / atr
        aligned = close > float(r["ema13"]) if side == Side.LONG else close < float(r["ema13"])
        momentum = close > open_ if side == Side.LONG else close < open_
        if not aligned or not momentum or body < self.min_body_atr:
            return None

        if zone is not None:
            if side == Side.LONG and float(p["low"]) <= zone.upper + self.zone_touch_atr * atr and close > zone.upper:
                return "5M_DEMAND_RECLAIM"
            if side == Side.SHORT and float(p["high"]) >= zone.lower - self.zone_touch_atr * atr and close < zone.lower:
                return "5M_SUPPLY_RECLAIM"
        if self._recent_cross(d5["ema8"], d5["ema13"], side, self.cross_lookback):
            return "5M_EMA8_13_CROSS"
        if self._recent_hma_flip(d5["hma16"], side, self.flip_lookback):
            return "5M_HMA16_FLIP"
        reclaim = (
            float(p["low"]) <= float(p["ema13"]) and close > float(r["ema13"])
            if side == Side.LONG else
            float(p["high"]) >= float(p["ema13"]) and close < float(r["ema13"])
        )
        return "5M_EMA13_RECLAIM" if reclaim else None

    def _risk_plan(self, decision, df15, df5):
        d15, d5 = self._prepared(df15), self._prepared(df5)
        entry = float(d5["close"].iloc[-1])
        atr15 = float(d15["atr"].iloc[-1])
        context = decision.context
        zone = context.active_zone if isinstance(context, DynamicContext) else None
        structure = None
        if zone is not None:
            structure = zone.lower if decision.side == Side.LONG else zone.upper
            raw_dist = (
                entry - structure + self.sl_buffer_atr * atr15
                if decision.side == Side.LONG else
                structure - entry + self.sl_buffer_atr * atr15
            )
        else:
            ctx = self._structure(d15)
            if decision.side == Side.LONG:
                structure = max([v for _, v in ctx["mic_l"][-3:] if v < entry] or [entry - atr15])
                raw_dist = entry - structure + 0.10 * atr15
            else:
                structure = min([v for _, v in ctx["mic_h"][-3:] if v > entry] or [entry + atr15])
                raw_dist = structure - entry + 0.10 * atr15

        raw_dist = max(raw_dist, self.sl_atr * atr15, entry * self.sl_min_pct)
        if raw_dist > entry * self.sl_max_pct:
            return None
        stop_dist = raw_dist
        sl = entry - stop_dist if decision.side == Side.LONG else entry + stop_dist
        fixed_tp = entry * (1 + self.tp_pct) if decision.side == Side.LONG else entry * (1 - self.tp_pct)
        opposing = context.opposing_zone if isinstance(context, DynamicContext) else None
        if opposing is not None:
            zone_tp = (
                opposing.lower - self.target_buffer_atr * atr15
                if decision.side == Side.LONG else
                opposing.upper + self.target_buffer_atr * atr15
            )
            tp = min(fixed_tp, zone_tp) if decision.side == Side.LONG else max(fixed_tp, zone_tp)
        else:
            tp = fixed_tp
        reward = (tp - entry) if decision.side == Side.LONG else (entry - tp)
        if reward <= 0:
            return None
        rr = reward / max(stop_dist, 1e-12)
        return entry, sl, tp, atr15, float(structure), rr

    def evaluate(self, df4h, df1h, df15, df5) -> DecisionState:
        if len(df1h) < 60 or len(df15) < 70 or len(df5) < 40 or len(df4h) < 40:
            direction = AdaptiveDirection(None, 0, 0, 0, "NEUTRAL")
            return DecisionState(False, "WARMUP", "insufficient candles", direction, None, None, None, None, 0.0)

        direction, quality = self._direction_1h(df1h)
        if direction.side is None:
            return DecisionState(False, "1H_DIRECTION", f"score {direction.score:.0f} edge {direction.edge:+.0f}", direction, quality, None, None, None, direction.score)

        dmi_edge = quality.plus_di - quality.minus_di if direction.side == Side.LONG else quality.minus_di - quality.plus_di
        hard_bad_market = quality.chop >= self.hard_chop and quality.adx < self.hard_adx
        if quality.q < self.quality_min or hard_bad_market or dmi_edge < -self.dmi_opposition:
            why = f"Q {quality.q:.0f} ADX {quality.adx:.1f} CHOP {quality.chop:.1f} DMI {dmi_edge:+.1f}"
            return DecisionState(False, "1H_QUALITY", why, direction, quality, None, None, None, direction.score)

        d15, d5 = self._prepared(df15), self._prepared(df5)
        context = self._zone_context(d15, direction.side)
        setup = SetupType.SWEEP if "SWEEP" in context.reaction else SetupType.PULLBACK
        if context.location.score <= 0:
            return DecisionState(False, "15M_LOCATION", context.reaction, direction, quality, context, setup, None, direction.score)
        if context.location.room_atr < self.min_room_atr:
            return DecisionState(False, "ROOM", f"opposing zone {context.location.room_atr:.2f}<{self.min_room_atr:.2f} ATR", direction, quality, context, setup, None, direction.score)

        raw_trigger = self._execution_trigger(d15, direction.side, context.active_zone)
        allowed = {"5M_HMA16_FLIP", "5M_EMA13_RECLAIM"}
        if raw_trigger not in allowed:
            return DecisionState(False, "15M_TRIGGER", f"{context.reaction}; waiting HMA16 flip or EMA13 reclaim", direction, quality, context, setup, None, direction.score)

        macro_ok, macro_score, macro_opposing = self._macro_aligned(df4h, direction.side)
        if raw_trigger == "5M_EMA13_RECLAIM" and not macro_ok:
            return DecisionState(
                False, "4H_ALIGNMENT",
                f"EMA13 reclaim needs 4H {direction.side.value} {macro_score:.0f} edge {macro_score - macro_opposing:+.0f}",
                direction, quality, context, setup, None, direction.score,
            )

        trigger = raw_trigger.replace("5M_", "15M_", 1)
        atr_exec = float(d15["atr"].iloc[-1])
        chase = abs(float(d15["close"].iloc[-1]) - float(d15["ema13"].iloc[-1])) / max(atr_exec, 1e-12)
        if chase > self.max_chase_atr:
            return DecisionState(False, "CHASE", f"distance {chase:.2f}>{self.max_chase_atr:.2f} ATR15", direction, quality, context, setup, (trigger, atr_exec), direction.score)

        provisional = DecisionState(True, "READY", context.reaction, direction, quality, context, setup, (trigger, atr_exec), direction.score)
        risk = self._risk_plan(provisional, df15, df5)
        if risk is None:
            return DecisionState(False, "RISK", "zone stop exceeds 1.00% or target has no room", direction, quality, context, setup, (trigger, atr_exec), direction.score)
        if risk[-1] < self.min_rr:
            return DecisionState(False, "RISK", f"actual RR {risk[-1]:.2f}<{self.min_rr:.2f}", direction, quality, context, setup, (trigger, atr_exec), direction.score)
        return DecisionState(True, "READY", f"{context.reaction} RR {risk[-1]:.2f}", direction, quality, context, setup, (trigger, atr_exec), direction.score)

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position: bool = False) -> Optional[EntrySignal]:
        if has_open_position:
            return None
        decision = self.evaluate(df4h, df1h, df15, df5)
        if not decision.ready or decision.side is None or decision.execution is None:
            return None
        risk = self._risk_plan(decision, df15, df5)
        if risk is None or risk[-1] < self.min_rr:
            return None
        entry, sl, tp, atr, structure, rr = risk
        macro = float(trend_score_4h(df4h, "long")) - float(trend_score_4h(df4h, "short"))
        trigger = decision.execution[0]
        trend = Trend.BULL if decision.side == Side.LONG else Trend.BEAR
        context = decision.context
        reason = (
            f"TPC Zone {decision.side.value} | 1H {decision.direction.score:.0f} "
            f"edge {decision.direction.edge:+.0f} Q {decision.quality.q:.0f} | "
            f"15M {context.location.zone} {context.reaction} → {trigger} | "
            f"Room {context.location.room_atr:.2f}ATR | 4H soft {macro:+.0f} | RR {rr:.2f}"
        )
        room_pct = abs(tp - entry) / max(entry, 1e-12)
        return EntrySignal(
            decision.side, entry, sl, tp, trend, decision.quality.q,
            decision.quality.adx, decision.quality.chop,
            decision.setup_type, trigger, room_pct, atr, structure, reason,
        )

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        d = self.evaluate(df4h, df1h, df15, df5)
        side = d.side.value if d.side else "NONE"
        parts = [
            f"TPC-ZONE {'READY' if d.ready else 'WAIT'}", side,
            f"Stage={d.stage}", f"Trend={d.direction.score:.0f}",
            f"Edge={d.direction.edge:+.0f}",
        ]
        if d.quality is not None:
            parts += [f"Q={d.quality.q:.0f}", f"ADX={d.quality.adx:.1f}", f"CHOP={d.quality.chop:.1f}"]
        if isinstance(d.context, DynamicContext):
            parts += [f"Zone={d.context.location.zone}", f"Room={d.context.location.room_atr:.2f}ATR", f"Mode={d.context.mode}"]
        parts.append(f"Trigger={d.execution[0] if d.execution else 'WAIT'}")
        if d.blocker:
            parts.append(f"Reason={d.blocker}")
        return " | ".join(parts)


MTFStructureStrategyV12 = PrecisionTrendStructureV12
