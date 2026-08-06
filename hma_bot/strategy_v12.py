"""HMA Fast Structure V6 — one-path, early 15M continuation strategy.

The old S/R pipeline waited for a 15M Sentinel level and another closed 5M
confirmation.  That produced late entries, while the counter-trend fallback
could take the opposite side.  V6 has one authoritative path:

    1H direction -> relaxed quality guard -> closed-15M trigger -> risk plan

4H is context only.  There is no weighted score, counter-trend engine, or
mandatory S1/S2/R1/R2 touch.  All signals use closed candles.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

import strategy_v11 as v11
from sentinel_context import build_context, trend_score_4h

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


class PrecisionTrendStructureV12(v11.PrecisionTrendStructureV11):
    """Fast 1H bias with three interchangeable 15M entry triggers."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.direction_min = _env_float("FAST_DIRECTION_MIN", 52.0)
        self.direction_edge_min = _env_float("FAST_DIRECTION_EDGE_MIN", 3.0)
        self.quality_min = _env_float("FAST_QUALITY_MIN", 42.0)
        self.hard_chop = _env_float("FAST_HARD_CHOP", 68.0)
        self.hard_adx = _env_float("FAST_HARD_ADX", 11.0)
        self.dmi_opposition = _env_float("FAST_DMI_OPPOSITION", 8.0)
        self.cross_lookback = max(1, _env_int("FAST_CROSS_LOOKBACK", 3))
        self.flip_lookback = max(1, _env_int("FAST_HMA_FLIP_LOOKBACK", 2))
        self.max_chase_atr = _env_float("FAST_MAX_CHASE_ATR", 1.10)
        self.min_body_atr = _env_float("FAST_MIN_BODY_ATR", 0.08)
        self.sl_atr = _env_float("FAST_SL_ATR", 1.05)
        self.sl_max_pct = _env_float("FAST_SL_MAX_PCT", 0.010)
        self.tp_pct = _env_float("FAST_TP_PCT", 0.012)
        self.min_rr = _env_float("FAST_MIN_RR", 1.05)

    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    def _hma16(self, series: pd.Series) -> pd.Series:
        return self._hma(series.astype(float), 16)

    def _direction_1h(self, df1h):
        quality = self.quality_state_1h(df1h)
        long_score = float(trend_score_4h(df1h, "long"))
        short_score = float(trend_score_4h(df1h, "short"))
        side = Side.LONG if long_score >= short_score else Side.SHORT
        score = max(long_score, short_score)
        opposite = min(long_score, short_score)
        edge = score - opposite
        if score < self.direction_min or edge < self.direction_edge_min:
            side = None
        tier = "TREND" if score < 68.0 else "STRONG"
        return AdaptiveDirection(side, score, opposite, edge, tier), quality

    # Compatibility for a rolling Railway deployment where the previous
    # runtime may briefly import the new strategy before main_v16 is replaced.
    def _simple_direction(self, df1h):
        return self._direction_1h(df1h)

    def _prepared_15m(self, df15):
        d = df15.copy()
        close = d["close"].astype(float)
        d["ema8"] = self._ema(close, 8)
        d["ema13"] = self._ema(close, 13)
        d["ema20"] = self._ema(close, 20)
        d["hma16"] = self._hma16(close)
        d["atr"] = self._atr(d, self.cfg.atr_len)
        return d

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

    def _trigger_15m(self, d, side: Side):
        r, p = d.iloc[-1], d.iloc[-2]
        atr = float(r["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None
        close, open_ = float(r["close"]), float(r["open"])
        body = abs(close - open_) / atr
        aligned = close > float(r["ema13"]) if side == Side.LONG else close < float(r["ema13"])
        momentum = close > open_ if side == Side.LONG else close < open_
        if not aligned or not momentum or body < self.min_body_atr:
            return None

        cross = self._recent_cross(d["ema8"], d["ema13"], side, self.cross_lookback)
        flip = self._recent_hma_flip(d["hma16"], side, self.flip_lookback)
        reclaim = (
            float(p["low"]) <= float(p["ema13"]) and close > float(r["ema13"]) and float(r["ema13"]) >= float(p["ema13"])
            if side == Side.LONG else
            float(p["high"]) >= float(p["ema13"]) and close < float(r["ema13"]) and float(r["ema13"]) <= float(p["ema13"])
        )
        if cross:
            return "EMA8_13_CROSS"
        if flip:
            return "HMA16_FLIP"
        if reclaim:
            return "EMA13_PULLBACK_RECLAIM"
        return None

    def evaluate(self, df4h, df1h, df15, df5) -> DecisionState:
        if len(df1h) < 60 or len(df15) < 60 or len(df4h) < 40:
            direction = AdaptiveDirection(None, 0, 0, 0, "NEUTRAL")
            return DecisionState(False, "WARMUP", "insufficient candles", direction, None, None, None, None, 0.0)

        direction, quality = self._direction_1h(df1h)
        if direction.side is None:
            return DecisionState(False, "1H_DIRECTION", f"score {direction.score:.0f} edge {direction.edge:+.0f}", direction, quality, None, None, None, direction.score)

        dmi_edge = (quality.plus_di - quality.minus_di) if direction.side == Side.LONG else (quality.minus_di - quality.plus_di)
        hard_bad_market = quality.chop >= self.hard_chop and quality.adx < self.hard_adx
        if quality.q < self.quality_min or hard_bad_market or dmi_edge < -self.dmi_opposition:
            why = f"Q {quality.q:.0f} ADX {quality.adx:.1f} CHOP {quality.chop:.1f} DMI {dmi_edge:+.1f}"
            return DecisionState(False, "1H_QUALITY", why, direction, quality, None, None, None, direction.score)

        d15 = self._prepared_15m(df15)
        trigger = self._trigger_15m(d15, direction.side)
        context = build_context(df15=df15, df1h=df1h, df4h=df4h, side="long" if direction.side == Side.LONG else "short")
        setup = SetupType.PULLBACK if trigger == "EMA13_PULLBACK_RECLAIM" else SetupType.BREAKOUT_RETEST
        if trigger is None:
            return DecisionState(False, "15M_TRIGGER", "waiting EMA cross, HMA flip or EMA13 reclaim", direction, quality, context, setup, None, direction.score)

        r = d15.iloc[-1]
        atr = float(r["atr"])
        chase = abs(float(r["close"]) - float(r["ema13"])) / max(atr, 1e-12)
        if chase > self.max_chase_atr:
            return DecisionState(False, "CHASE", f"distance {chase:.2f}>{self.max_chase_atr:.2f} ATR", direction, quality, context, setup, (trigger, atr), direction.score)

        return DecisionState(True, "READY", "fast closed-15M trigger", direction, quality, context, setup, (trigger, atr), direction.score)

    def _risk_plan(self, decision, df15):
        d = self._prepared_15m(df15)
        entry = float(d["close"].iloc[-1])
        atr = float(d["atr"].iloc[-1])
        ctx = self._structure(d)
        if decision.side == Side.LONG:
            swing = max([v for _, v in ctx["mic_l"][-3:] if v < entry] or [entry - atr])
            raw_dist = max(entry - swing + 0.10 * atr, self.sl_atr * atr)
        else:
            swing = min([v for _, v in ctx["mic_h"][-3:] if v > entry] or [entry + atr])
            raw_dist = max(swing - entry + 0.10 * atr, self.sl_atr * atr)
        stop_dist = min(raw_dist, entry * self.sl_max_pct)
        sl = entry - stop_dist if decision.side == Side.LONG else entry + stop_dist
        tp = entry * (1 + self.tp_pct) if decision.side == Side.LONG else entry * (1 - self.tp_pct)
        rr = abs(tp - entry) / max(stop_dist, 1e-12)
        return entry, sl, tp, atr, swing, rr

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position: bool = False) -> Optional[EntrySignal]:
        if has_open_position:
            return None
        decision = self.evaluate(df4h, df1h, df15, df5)
        if not decision.ready or decision.side is None or decision.execution is None:
            return None
        entry, sl, tp, atr, structure, rr = self._risk_plan(decision, df15)
        if rr < self.min_rr:
            return None
        macro_long = float(trend_score_4h(df4h, "long"))
        macro_short = float(trend_score_4h(df4h, "short"))
        macro = macro_long - macro_short
        trigger = decision.execution[0]
        trend = Trend.BULL if decision.side == Side.LONG else Trend.BEAR
        reason = (f"Fast V6 {decision.side.value} | 1H {decision.direction.score:.0f} edge {decision.direction.edge:+.0f} "
                  f"Q {decision.quality.q:.0f} | 15M {trigger} | 4H soft {macro:+.0f} | RR {rr:.2f}")
        return EntrySignal(decision.side, entry, sl, tp, trend, decision.quality.q,
                           decision.quality.adx, decision.quality.chop,
                           decision.setup_type, trigger, self.tp_pct, atr,
                           structure, reason)

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        d = self.evaluate(df4h, df1h, df15, df5)
        side = d.side.value if d.side else "NONE"
        parts = [f"FAST-V6 {'READY' if d.ready else 'WAIT'}", side,
                 f"Stage={d.stage}", f"Trend={d.direction.score:.0f}", f"Edge={d.direction.edge:+.0f}"]
        if d.quality is not None:
            parts += [f"Q={d.quality.q:.0f}", f"ADX={d.quality.adx:.1f}", f"CHOP={d.quality.chop:.1f}"]
        parts.append(f"Trigger={d.execution[0] if d.execution else 'WAIT'}")
        if d.blocker:
            parts.append(f"Reason={d.blocker}")
        return " | ".join(parts)


MTFStructureStrategyV12 = PrecisionTrendStructureV12
