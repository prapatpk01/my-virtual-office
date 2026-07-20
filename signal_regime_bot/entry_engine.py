"""DUALCORE V1.6 — 5M EMA dual-entry execution engine.

Architecture:
    4H macro -> 1H bias -> 15M context/structure -> 5M execution.

The 15M frame is context only.  Actual entries are generated from closed 5M
candles using EMA8/EMA13 timing because it is smoother and less whipsaw-prone
than HMA10/HMA16 on a 5-minute chart while still reacting quickly enough for a
behavioural target of roughly 3–4 trades/day across seven active symbols.

Entry engines:
* FAST_PULLBACK: EMA zone / confirmed support touch -> reclaim, EMA cross,
  previous-bar break, micro BOS or sweep-reclaim.
* MOMENTUM: 5M structure breakout -> direct entry only when displacement is
  strong, otherwise a 1–2 bar breakout-retest.

All calculations use closed candles only.  Setups and duplicate keys persist
across restarts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

LONG = "LONG"
SHORT = "SHORT"
NONE = "NONE"

EMA_CROSS_REVERSAL = "EMA_CROSS_REVERSAL"
PRICE_OPEN_BEYOND_EMA = "PRICE_OPEN_BEYOND_EMA"

FAST_PULLBACK = "FAST_PULLBACK"
MOMENTUM = "MOMENTUM"
MOMENTUM_RETEST = "MOMENTUM_RETEST"


@dataclass
class EntryResult:
    direction: str
    allow_entry: bool
    reason: str = ""
    price: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    macd_hist: float = 0.0  # compatibility: candidate edge
    cross_id: object = None
    entry_score: float = 0.0
    score_evaluated: bool = False
    score_threshold: Optional[float] = None
    setup_type: str = ""
    trigger: str = ""
    planned_stop: Optional[float] = None
    planned_target: Optional[float] = None
    planned_rr: float = 0.0
    structure_room_r: float = 0.0
    invalidation_level: Optional[float] = None
    score_components: dict = field(default_factory=dict)


@dataclass
class ExitCheckResult:
    should_exit: bool
    reason: str = ""
    detail: str = ""


@dataclass
class _PullbackSetup:
    direction: str
    started_bar: int
    started_ts: pd.Timestamp
    setup_low: float
    setup_high: float
    trigger_level: float
    invalidation: float
    location_score: float


@dataclass
class _BreakoutSetup:
    direction: str
    started_bar: int
    started_ts: pd.Timestamp
    breakout_level: float
    breakout_low: float
    breakout_high: float


@dataclass
class _SetupState:
    last_processed_bar: Optional[pd.Timestamp] = None
    pullback: Optional[_PullbackSetup] = None
    breakout: Optional[_BreakoutSetup] = None
    last_entry_key: object = None
    last_candidate_key: object = None


@dataclass
class _Candidate:
    direction: str
    setup_type: str
    score: float
    threshold: float
    trigger: str
    price: float
    stop: float
    target: float
    rr: float
    room_r: float
    invalidation: float
    components: dict
    signal_key: object

    @property
    def edge(self) -> float:
        return self.score - self.threshold


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: dict[str, _SetupState] = {}
        self._state_path = os.path.join(
            getattr(cfg, "state_dir", "state"), "entry_engine_state.json"
        )
        self._load_state()

    @staticmethod
    def _json_key(value):
        if value is None:
            return None
        if isinstance(value, tuple):
            return [EntryEngine._json_key(x) for x in value]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    @staticmethod
    def _restore_key(value):
        if isinstance(value, list):
            return tuple(EntryEngine._restore_key(x) for x in value)
        return value

    @staticmethod
    def _restore_ts(value) -> Optional[pd.Timestamp]:
        if value in (None, ""):
            return None
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return None

    def _load_state(self) -> None:
        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            return
        for symbol, raw in payload.get("symbols", {}).items():
            try:
                pb_raw = raw.get("pullback")
                bo_raw = raw.get("breakout")
                pullback = None
                breakout = None
                if pb_raw:
                    started_ts = self._restore_ts(pb_raw.get("started_ts"))
                    if started_ts is not None:
                        pullback = _PullbackSetup(
                            direction=str(pb_raw["direction"]),
                            started_bar=int(pb_raw.get("started_bar", 0)),
                            started_ts=started_ts,
                            setup_low=float(pb_raw["setup_low"]),
                            setup_high=float(pb_raw["setup_high"]),
                            trigger_level=float(pb_raw["trigger_level"]),
                            invalidation=float(pb_raw["invalidation"]),
                            location_score=float(pb_raw["location_score"]),
                        )
                if bo_raw:
                    started_ts = self._restore_ts(bo_raw.get("started_ts"))
                    if started_ts is not None:
                        breakout = _BreakoutSetup(
                            direction=str(bo_raw["direction"]),
                            started_bar=int(bo_raw.get("started_bar", 0)),
                            started_ts=started_ts,
                            breakout_level=float(bo_raw["breakout_level"]),
                            breakout_low=float(bo_raw["breakout_low"]),
                            breakout_high=float(bo_raw["breakout_high"]),
                        )
                self._state[str(symbol)] = _SetupState(
                    last_processed_bar=self._restore_ts(raw.get("last_processed_bar")),
                    pullback=pullback,
                    breakout=breakout,
                    last_entry_key=self._restore_key(raw.get("last_entry_key")),
                    last_candidate_key=self._restore_key(raw.get("last_candidate_key")),
                )
            except (KeyError, TypeError, ValueError):
                continue

    def _persist_state(self) -> None:
        directory = os.path.dirname(self._state_path) or "."
        os.makedirs(directory, exist_ok=True)
        symbols = {}
        for symbol, state in self._state.items():
            pullback = asdict(state.pullback) if state.pullback is not None else None
            breakout = asdict(state.breakout) if state.breakout is not None else None
            if pullback is not None:
                pullback["started_ts"] = state.pullback.started_ts.isoformat()
            if breakout is not None:
                breakout["started_ts"] = state.breakout.started_ts.isoformat()
            symbols[symbol] = {
                "last_processed_bar": (
                    state.last_processed_bar.isoformat()
                    if state.last_processed_bar is not None else None
                ),
                "pullback": pullback,
                "breakout": breakout,
                "last_entry_key": self._json_key(state.last_entry_key),
                "last_candidate_key": self._json_key(state.last_candidate_key),
            }
        tmp = f"{self._state_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"version": 2, "symbols": symbols},
                    fh,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            os.replace(tmp, self._state_path)
        except OSError:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def _get_state(self, symbol: str) -> _SetupState:
        return self._state.setdefault(symbol, _SetupState())

    def reset_symbol(self, symbol: str) -> None:
        self._state.pop(symbol, None)
        self._persist_state()

    def on_position_closed(self, symbol: str) -> None:
        state = self._get_state(symbol)
        state.pullback = None
        state.breakout = None
        state.last_candidate_key = None
        self._persist_state()

    def observe(
        self,
        df_30m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        symbol: str,
    ) -> None:
        self._get_state(symbol)

    @staticmethod
    def _age_bars(started_ts: pd.Timestamp, current_ts: pd.Timestamp, minutes: int = 5) -> int:
        try:
            delta = current_ts - started_ts
            return max(0, int(delta.total_seconds() // (minutes * 60)))
        except (TypeError, AttributeError):
            return 999

    def _nearest_levels(
        self,
        price: float,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame],
        df_4h: Optional[pd.DataFrame],
    ) -> tuple[Optional[float], Optional[float]]:
        supports: list[float] = []
        resistances: list[float] = []
        # Opposing-room validation intentionally starts at 15M. Tiny 5M
        # pivots are execution noise and otherwise reject too many valid fast
        # entries; 5M structure is still used for triggers and stop placement.
        for frame in (df_15m, df_1h, df_4h):
            support, resistance = ind.nearest_confirmed_levels(
                frame,
                price,
                getattr(self.cfg, "entry_swing_left", 3),
                getattr(self.cfg, "entry_swing_right", 3),
            )
            if support is not None:
                supports.append(support)
            if resistance is not None:
                resistances.append(resistance)
        return (max(supports) if supports else None, min(resistances) if resistances else None)

    def _context_snapshot(self, df: pd.DataFrame, direction: str) -> dict:
        """15M trend/structure context; never an entry trigger."""
        c = self.cfg
        close = df["close"]
        ema_fast = ind.ema(close, getattr(c, "dual_context_ema_fast", 20))
        ema_slow = ind.ema(close, getattr(c, "dual_context_ema_slow", 50))
        atr_s = ind.atr(df, 14)
        atr_value = ind.safe_float(atr_s.iloc[-1])
        adx_s, plus_s, minus_s = ind.adx(df, 12)
        adx_value = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_s.iloc[-1])
        minus_di = ind.safe_float(minus_s.iloc[-1])
        chop = ind.safe_float(ind.choppiness_index(df, 14).iloc[-1], 100.0)
        roc9_s = ind.roc(close, 9)
        roc9 = ind.safe_float(roc9_s.iloc[-1])
        structure = ind.market_structure(
            df["high"], df["low"],
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
        )
        price = ind.safe_float(close.iloc[-1])
        ef = ind.safe_float(ema_fast.iloc[-1])
        es = ind.safe_float(ema_slow.iloc[-1])
        ef_prev = ind.safe_float(ema_fast.iloc[-2])
        long = direction == LONG

        aligned = ef > es if long else ef < es
        slope_ok = ef >= ef_prev if long else ef <= ef_prev
        dmi_ok = plus_di > minus_di if long else minus_di > plus_di
        roc_ok = roc9 > 0 if long else roc9 < 0
        structure_aligned = structure == ("HH_HL" if long else "LH_LL")
        opposite_structure = structure == ("LH_LL" if long else "HH_HL")
        price_side = price > es if long else price < es

        obvious_chop = adx_value < 9.0 and chop > 65.0
        strong = (
            aligned and slope_ok and price_side and dmi_ok
            and adx_value >= 15.0 and chop <= 58.0
        )
        transition = (
            not obvious_chop
            and not (opposite_structure and not aligned and not price_side)
            and (aligned or structure_aligned or price_side or roc_ok)
            and chop <= 66.0
        )
        allowed = strong or transition
        return {
            "price": price,
            "ema_fast": ef,
            "ema_slow": es,
            "atr": atr_value,
            "adx": adx_value,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "chop": chop,
            "roc": roc9,
            "structure": structure,
            "aligned": aligned,
            "slope_ok": slope_ok,
            "dmi_ok": dmi_ok,
            "roc_ok": roc_ok,
            "structure_aligned": structure_aligned,
            "opposite_structure": opposite_structure,
            "strong": strong,
            "transition": transition,
            "allowed": allowed,
            "obvious_chop": obvious_chop,
        }

    def _entry_snapshot(self, df: pd.DataFrame, direction: str) -> dict:
        """5M execution snapshot using EMA8/EMA13 timing."""
        c = self.cfg
        close = df["close"]
        fast_len = getattr(c, "dual_entry_ema_fast", 8)
        slow_len = getattr(c, "dual_entry_ema_slow", 13)
        ema_fast = ind.ema(close, fast_len)
        ema_slow = ind.ema(close, slow_len)
        ema20 = ind.ema(close, getattr(c, "dual_entry_trend_ema", 20))
        ema50 = ind.ema(close, getattr(c, "dual_entry_filter_ema", 50))
        atr_s = ind.atr(df, 14)
        atr_value = ind.safe_float(atr_s.iloc[-1])
        adx_s, plus_s, minus_s = ind.adx(df, 12)
        adx_value = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_s.iloc[-1])
        minus_di = ind.safe_float(minus_s.iloc[-1])
        chop = ind.safe_float(ind.choppiness_index(df, 14).iloc[-1], 100.0)
        roc_s = ind.roc(close, 5)
        roc5 = ind.safe_float(roc_s.iloc[-1])
        roc_prev = ind.safe_float(roc_s.iloc[-2])
        structure = ind.market_structure(
            df["high"], df["low"],
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
        )
        candle = ind.candle_metrics(df, atr_value)
        price = ind.safe_float(close.iloc[-1])
        ef = ind.safe_float(ema_fast.iloc[-1])
        es = ind.safe_float(ema_slow.iloc[-1])
        e20 = ind.safe_float(ema20.iloc[-1])
        e50 = ind.safe_float(ema50.iloc[-1])
        ef_prev = ind.safe_float(ema_fast.iloc[-2])
        es_prev = ind.safe_float(ema_slow.iloc[-2])
        adx_prev = ind.safe_float(adx_s.iloc[-2])
        spread_atr = abs(ef - es) / max(atr_value, ind.EPSILON)
        flip_count = ind.cross_count(ema_fast, ema_slow, 10)
        long = direction == LONG

        aligned = ef > es if long else ef < es
        fresh_cross = (
            ef > es and ef_prev <= es_prev if long
            else ef < es and ef_prev >= es_prev
        )
        fast_slope = ef > ef_prev if long else ef < ef_prev
        slow_slope = es >= es_prev if long else es <= es_prev
        dmi_ok = plus_di > minus_di if long else minus_di > plus_di
        roc_ok = roc5 > 0 if long else roc5 < 0
        roc_accel = roc5 > roc_prev if long else roc5 < roc_prev
        adx_support = adx_value >= getattr(c, "dual_momentum_min_adx", 11.0) or adx_value > adx_prev
        structure_ok = structure != ("LH_LL" if long else "HH_HL")
        ema20_ok = price >= e20 - 0.20 * atr_value if long else price <= e20 + 0.20 * atr_value
        ema50_ok = price >= e50 - 0.35 * atr_value if long else price <= e50 + 0.35 * atr_value
        support_count = sum((aligned, roc_ok, dmi_ok, adx_support))
        obvious_chop = (
            (adx_value < getattr(c, "dual_min_adx", 10.0) and chop > getattr(c, "dual_max_chop", 64.0))
            or (flip_count >= 4 and spread_atr < 0.08)
        )
        strong = (
            aligned and fast_slope and slow_slope and dmi_ok
            and adx_value >= getattr(c, "dual_strong_adx", 18.0)
            and chop <= getattr(c, "dual_strong_chop", 55.0)
        )
        return {
            "price": price,
            "ema_fast_series": ema_fast,
            "ema_slow_series": ema_slow,
            "ema_fast": ef,
            "ema_slow": es,
            "ema20": e20,
            "ema50": e50,
            "atr": atr_value,
            "adx": adx_value,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "chop": chop,
            "roc": roc5,
            "roc_accel": roc_accel,
            "structure": structure,
            "candle": candle,
            "aligned": aligned,
            "fresh_cross": fresh_cross,
            "fast_slope": fast_slope,
            "slow_slope": slow_slope,
            "dmi_ok": dmi_ok,
            "roc_ok": roc_ok,
            "adx_support": adx_support,
            "support_count": support_count,
            "structure_ok": structure_ok,
            "ema20_ok": ema20_ok,
            "ema50_ok": ema50_ok,
            "obvious_chop": obvious_chop,
            "strong": strong,
            "spread_atr": spread_atr,
            "flip_count": flip_count,
        }

    def _finalize_candidate(
        self,
        *,
        direction: str,
        setup_type: str,
        score: float,
        threshold: float,
        trigger: str,
        price: float,
        raw_stop: float,
        atr_value: float,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        bar_ts: pd.Timestamp,
        components: dict,
    ) -> Optional[_Candidate]:
        c = self.cfg
        long = direction == LONG
        min_stop = atr_value * getattr(c, "dual_min_stop_atr", 0.45)
        max_stop = atr_value * getattr(c, "dual_max_stop_atr", 1.40)
        if atr_value <= 0:
            return None

        if long:
            if raw_stop >= price:
                return None
            stop = min(raw_stop, price - min_stop)
            risk = price - stop
            opposing = nearest_resistance if nearest_resistance and nearest_resistance > price else None
        else:
            if raw_stop <= price:
                return None
            stop = max(raw_stop, price + min_stop)
            risk = stop - price
            opposing = nearest_support if nearest_support and nearest_support < price else None
        if risk <= 0 or risk > max_stop:
            return None

        base_rr = (
            getattr(c, "dual_pullback_tp2_r", 2.0)
            if setup_type == FAST_PULLBACK
            else getattr(c, "dual_momentum_tp2_r", 2.0)
        )
        base_target = price + risk * base_rr if long else price - risk * base_rr
        target_buffer = atr_value * getattr(c, "dual_target_buffer_atr", 0.08)
        if opposing is not None:
            structure_target = opposing - target_buffer if long else opposing + target_buffer
            target = min(base_target, structure_target) if long else max(base_target, structure_target)
            room_r = ((opposing - price) / risk) if long else ((price - opposing) / risk)
        else:
            target = base_target
            room_r = 99.0
        rr = ((target - price) / risk) if long else ((price - target) / risk)
        min_room = (
            getattr(c, "dual_pullback_min_room_r", 1.00)
            if setup_type == FAST_PULLBACK
            else getattr(c, "dual_momentum_min_room_r", 1.10)
        )
        if room_r < min_room or rr < getattr(c, "minimum_actual_rr", 1.10):
            return None
        signal_key = (str(bar_ts), direction, setup_type, round(price, 10))
        return _Candidate(
            direction=direction,
            setup_type=setup_type,
            score=round(score, 1),
            threshold=threshold,
            trigger=trigger,
            price=price,
            stop=stop,
            target=target,
            rr=rr,
            room_r=room_r,
            invalidation=raw_stop,
            components=components,
            signal_key=signal_key,
        )

    def _pullback_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        context: dict,
        snapshot: dict,
        state: _SetupState,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        regime=None,
        bias=None,
    ) -> Optional[_Candidate]:
        c = self.cfg
        long = direction == LONG
        atr_value = snapshot["atr"]
        if (
            snapshot["obvious_chop"]
            or not snapshot["structure_ok"]
            or not context["allowed"]
            or not snapshot["ema50_ok"]
        ):
            state.pullback = None
            return None

        price = snapshot["price"]
        row = df.iloc[-1]
        bar_ts = df.index[-1]
        ema_fast, ema_slow = snapshot["ema_fast"], snapshot["ema_slow"]
        zone_upper = max(ema_fast, ema_slow) + atr_value * getattr(c, "dual_pullback_zone_atr", 0.12)
        zone_lower = min(ema_fast, ema_slow) - atr_value * getattr(c, "dual_pullback_depth_atr", 0.20)

        swing_high, swing_low = ind.recent_swing_levels(
            df["high"], df["low"],
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
        )
        level = nearest_support if long else nearest_resistance
        sweep = ind.sweep_reclaim(df, direction, level)
        if long:
            ema_touch = (
                float(row["low"]) <= zone_upper
                and float(row["low"]) >= snapshot["ema20"] - 0.50 * atr_value
                and price >= snapshot["ema20"] - 0.20 * atr_value
            )
            htf_touch = level is not None and float(row["low"]) <= level + 0.15 * atr_value and price >= level
            location_touch = ema_touch or htf_touch or sweep
            reclaim = price > ema_fast and float(row["low"]) <= ema_fast
            previous_break = price > float(df["high"].iloc[-2])
            micro_bos = price > float(df["high"].iloc[-3:-1].max())
            invalidation = float(row["low"])
            if level is not None and 0 <= price - level <= 0.60 * atr_value:
                invalidation = min(invalidation, level)
        else:
            ema_touch = (
                float(row["high"]) >= zone_lower
                and float(row["high"]) <= snapshot["ema20"] + 0.50 * atr_value
                and price <= snapshot["ema20"] + 0.20 * atr_value
            )
            htf_touch = level is not None and float(row["high"]) >= level - 0.15 * atr_value and price <= level
            location_touch = ema_touch or htf_touch or sweep
            reclaim = price < ema_fast and float(row["high"]) >= ema_fast
            previous_break = price < float(df["low"].iloc[-2])
            micro_bos = price < float(df["low"].iloc[-3:-1].min())
            invalidation = float(row["high"])
            if level is not None and 0 <= level - price <= 0.60 * atr_value:
                invalidation = max(invalidation, level)

        location_score = 20.0 if htf_touch or sweep else 14.0 if ema_touch else 0.0
        candle_ok = (
            ind.bullish_trigger_candle(
                snapshot["candle"],
                getattr(c, "dual_pullback_min_body_atr", 0.12),
                getattr(c, "dual_pullback_close_quality", 0.58),
            )
            if long
            else ind.bearish_trigger_candle(
                snapshot["candle"],
                getattr(c, "dual_pullback_min_body_atr", 0.12),
                getattr(c, "dual_pullback_close_quality", 0.58),
            )
        )
        trigger_event = reclaim or snapshot["fresh_cross"] or previous_break or micro_bos or sweep
        same_bar_trigger = location_touch and candle_ok and trigger_event

        if location_touch and (state.pullback is None or state.pullback.direction != direction):
            state.pullback = _PullbackSetup(
                direction=direction,
                started_bar=0,
                started_ts=bar_ts,
                setup_low=float(row["low"]),
                setup_high=float(row["high"]),
                trigger_level=float(row["high"] if long else row["low"]),
                invalidation=invalidation,
                location_score=location_score,
            )

        setup = state.pullback
        if setup is None or setup.direction != direction:
            return None
        age = self._age_bars(setup.started_ts, bar_ts, 5)
        if age > getattr(c, "dual_pullback_window_bars", 4):
            state.pullback = None
            return None

        if long:
            setup.setup_low = min(setup.setup_low, float(row["low"]))
            invalidated = price < setup.invalidation - 0.10 * atr_value
            armed_break = price > setup.trigger_level
        else:
            setup.setup_high = max(setup.setup_high, float(row["high"]))
            invalidated = price > setup.invalidation + 0.10 * atr_value
            armed_break = price < setup.trigger_level
        if invalidated or not context["allowed"]:
            state.pullback = None
            return None

        trigger = ""
        if same_bar_trigger and age == 0:
            if sweep:
                trigger = "SWEEP_RECLAIM"
            elif snapshot["fresh_cross"]:
                trigger = "EMA_CROSS_RECLAIM"
            elif micro_bos:
                trigger = "MICRO_BOS"
            else:
                trigger = "SAME_BAR_RECLAIM"
        elif age >= 1 and candle_ok and (armed_break or reclaim or snapshot["fresh_cross"] or micro_bos or sweep):
            if sweep:
                trigger = "SWEEP_RECLAIM"
            elif micro_bos:
                trigger = "MICRO_BOS"
            elif snapshot["fresh_cross"]:
                trigger = "EMA_CROSS_RECLAIM"
            elif armed_break:
                trigger = "TRIGGER_BREAK"
            else:
                trigger = "EMA_RECLAIM"
        if not trigger:
            return None

        extension = (
            (price - ema_slow) / atr_value if long else (ema_slow - price) / atr_value
        )
        if extension < -0.25 or extension > getattr(c, "dual_pullback_max_extension_atr", 0.95):
            return None

        candle = snapshot["candle"]
        strong_candle = (
            candle.body_atr >= 0.18
            and (candle.bull_close_quality >= 0.65 if long else candle.bear_close_quality >= 0.65)
        )
        bias_edge = abs(getattr(bias, "directional_edge", 8.0))
        components = {
            "macro": 8.0 if getattr(regime, "label", "").startswith("STRONG") else 6.0,
            "bias": 14.0 if bias_edge >= 15 else 10.0 if bias_edge >= 8 else 7.0,
            "context": 15.0 if context["strong"] else 11.0,
            "structure": 14.0 if context["structure_aligned"] else 9.0,
            "location": setup.location_score,
            "trigger": 15.0 if trigger in ("SWEEP_RECLAIM", "MICRO_BOS") else 13.0 if trigger in ("EMA_CROSS_RECLAIM", "TRIGGER_BREAK") else 11.0,
            "candle": 10.0 if strong_candle else 7.0,
            "momentum": 7.0 if snapshot["support_count"] >= 3 else 4.0,
            "trend": 5.0 if snapshot["aligned"] or snapshot["fresh_cross"] else 2.0,
        }
        score = min(100.0, sum(components.values()))
        threshold = (
            getattr(c, "dual_same_bar_pullback_threshold", 66.0)
            if age == 0
            else getattr(c, "dual_pullback_threshold", 60.0)
        )
        if score < threshold:
            return None

        buffer = atr_value * getattr(c, "dual_stop_buffer_atr", 0.08)
        raw_stop = setup.setup_low - buffer if long else setup.setup_high + buffer
        candidate = self._finalize_candidate(
            direction=direction,
            setup_type=FAST_PULLBACK,
            score=score,
            threshold=threshold,
            trigger=trigger,
            price=price,
            raw_stop=raw_stop,
            atr_value=atr_value,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            bar_ts=bar_ts,
            components=components,
        )
        if candidate is not None:
            state.pullback = None
        return candidate

    def _momentum_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        context: dict,
        snapshot: dict,
        state: _SetupState,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        regime=None,
        bias=None,
    ) -> Optional[_Candidate]:
        c = self.cfg
        long = direction == LONG
        atr_value = snapshot["atr"]
        if snapshot["obvious_chop"] or not snapshot["structure_ok"] or not context["allowed"]:
            state.breakout = None
            return None

        price = snapshot["price"]
        row = df.iloc[-1]
        bar_ts = df.index[-1]
        lookback = getattr(c, "dual_breakout_lookback", 5)
        if len(df) < lookback + 5:
            return None
        break_level = (
            float(df["high"].iloc[-lookback - 1:-1].max())
            if long
            else float(df["low"].iloc[-lookback - 1:-1].min())
        )
        crossed = (
            price > break_level and float(df["close"].iloc[-2]) <= break_level
            if long
            else price < break_level and float(df["close"].iloc[-2]) >= break_level
        )
        candle = snapshot["candle"]
        close_quality = candle.bull_close_quality if long else candle.bear_close_quality
        major_displacement = candle.body_atr >= 0.30 and close_quality >= 0.78
        if snapshot["support_count"] < 2 and not (major_displacement and snapshot["support_count"] >= 1):
            return None

        candle_ok = (
            ind.bullish_trigger_candle(
                candle,
                getattr(c, "dual_momentum_min_body_atr", 0.15),
                getattr(c, "dual_momentum_close_quality", 0.65),
            )
            if long
            else ind.bearish_trigger_candle(
                candle,
                getattr(c, "dual_momentum_min_body_atr", 0.15),
                getattr(c, "dual_momentum_close_quality", 0.65),
            )
        )
        strong_breakout = (
            crossed
            and candle.body_atr >= 0.22
            and close_quality >= 0.70
            and (candle.volume_ratio >= getattr(c, "dual_momentum_volume_ratio", 1.05) or candle.body_atr >= 0.30)
        )

        if crossed:
            state.breakout = _BreakoutSetup(
                direction=direction,
                started_bar=0,
                started_ts=bar_ts,
                breakout_level=break_level,
                breakout_low=float(row["low"]),
                breakout_high=float(row["high"]),
            )

        setup = state.breakout
        if setup is None or setup.direction != direction:
            return None
        age = self._age_bars(setup.started_ts, bar_ts, 5)
        expiry = getattr(c, "dual_momentum_expiry_bars", 2)
        if age > expiry:
            state.breakout = None
            return None

        direct = age == 0 and strong_breakout
        if long:
            retest = (
                1 <= age <= expiry
                and float(row["low"]) <= setup.breakout_level + 0.12 * atr_value
                and float(row["low"]) >= setup.breakout_level - 0.35 * atr_value
                and price > setup.breakout_level
                and candle_ok
            )
            invalidated = price < setup.breakout_level - 0.35 * atr_value
            setup.breakout_low = min(setup.breakout_low, float(row["low"]))
        else:
            retest = (
                1 <= age <= expiry
                and float(row["high"]) >= setup.breakout_level - 0.12 * atr_value
                and float(row["high"]) <= setup.breakout_level + 0.35 * atr_value
                and price < setup.breakout_level
                and candle_ok
            )
            invalidated = price > setup.breakout_level + 0.35 * atr_value
            setup.breakout_high = max(setup.breakout_high, float(row["high"]))
        if invalidated:
            state.breakout = None
            return None
        if not (direct or retest):
            return None

        extension = abs(price - setup.breakout_level) / max(atr_value, ind.EPSILON)
        ema_extension = (
            (price - snapshot["ema_slow"]) / atr_value
            if long else (snapshot["ema_slow"] - price) / atr_value
        )
        max_ext = (
            getattr(c, "dual_strong_momentum_extension_atr", 1.10)
            if direct else getattr(c, "dual_momentum_max_extension_atr", 1.00)
        )
        if extension > (0.45 if direct else 0.35) or ema_extension > max_ext:
            return None

        bias_edge = abs(getattr(bias, "directional_edge", 8.0))
        components = {
            "macro": 8.0 if getattr(regime, "label", "").startswith("STRONG") else 6.0,
            "bias": 14.0 if bias_edge >= 15 else 10.0 if bias_edge >= 8 else 7.0,
            "context": 15.0 if context["strong"] else 11.0,
            "structure_break": 20.0,
            "breakout_close": 14.0 if close_quality >= 0.75 else 10.0,
            "displacement": 10.0 if candle.body_atr >= 0.25 else 7.0,
            "volume": 8.0 if candle.volume_ratio >= getattr(c, "dual_momentum_volume_ratio", 1.05) else 5.0,
            "trend": 10.0 if snapshot["strong"] else 7.0,
            "momentum": 8.0 if snapshot["support_count"] >= 3 else 5.0,
            "retest": 5.0 if retest else 2.0,
        }
        score = min(100.0, sum(components.values()))
        threshold = (
            getattr(c, "dual_strong_breakout_threshold", 62.0)
            if direct else getattr(c, "dual_momentum_threshold", 64.0)
        )
        if score < threshold:
            return None

        buffer = atr_value * getattr(c, "dual_stop_buffer_atr", 0.08)
        if long:
            raw_stop = min(setup.breakout_low, setup.breakout_level - buffer)
        else:
            raw_stop = max(setup.breakout_high, setup.breakout_level + buffer)
        candidate = self._finalize_candidate(
            direction=direction,
            setup_type=MOMENTUM if direct else MOMENTUM_RETEST,
            score=score,
            threshold=threshold,
            trigger="DIRECT_BREAKOUT" if direct else "BREAKOUT_RETEST",
            price=price,
            raw_stop=raw_stop,
            atr_value=atr_value,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            bar_ts=bar_ts,
            components=components,
        )
        if candidate is not None:
            state.breakout = None
        return candidate

    def analyze(
        self,
        df_30m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        direction: str,
        symbol: str,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        regime=None,
        bias=None,
    ) -> EntryResult:
        if direction not in (LONG, SHORT):
            return EntryResult(NONE, False, "no fixed direction from Bias")
        if df_15m is None or len(df_15m) < 100:
            return EntryResult(NONE, False, "insufficient 15M context history")
        if df_5m is None or len(df_5m) < 120:
            return EntryResult(NONE, False, "insufficient 5M entry history")

        state = self._get_state(symbol)
        bar_ts = df_5m.index[-1]
        if state.last_processed_bar == bar_ts:
            snap = self._entry_snapshot(df_5m, direction)
            return EntryResult(
                NONE,
                False,
                "5M bar already processed",
                price=snap["price"],
                ema_fast=snap["ema_fast"],
                ema_slow=snap["ema_slow"],
            )
        state.last_processed_bar = bar_ts

        context = self._context_snapshot(df_15m, direction)
        snapshot = self._entry_snapshot(df_5m, direction)
        base = dict(
            price=snapshot["price"],
            ema_fast=snapshot["ema_fast"],
            ema_slow=snapshot["ema_slow"],
        )
        if snapshot["atr"] <= 0 or context["atr"] <= 0:
            self._persist_state()
            return EntryResult(NONE, False, "ATR invalid", **base)
        if not context["allowed"]:
            state.pullback = None
            state.breakout = None
            self._persist_state()
            return EntryResult(
                NONE,
                False,
                f"15M context blocked: structure={context['structure']} EMA20/50={context['ema_fast']:.6f}/{context['ema_slow']:.6f} ADX={context['adx']:.1f} CHOP={context['chop']:.1f}",
                **base,
            )
        if snapshot["obvious_chop"]:
            state.pullback = None
            state.breakout = None
            self._persist_state()
            return EntryResult(
                NONE,
                False,
                f"5M CHOP veto: ADX={snapshot['adx']:.1f} CHOP={snapshot['chop']:.1f} flips={snapshot['flip_count']}",
                **base,
            )

        support, resistance = self._nearest_levels(
            snapshot["price"], df_5m, df_15m, df_1h, df_4h
        )
        pullback = self._pullback_candidate(
            df_5m, direction, context, snapshot, state,
            support, resistance, regime, bias,
        )
        momentum = self._momentum_candidate(
            df_5m, direction, context, snapshot, state,
            support, resistance, regime, bias,
        )
        candidates = [x for x in (pullback, momentum) if x is not None]
        if not candidates:
            armed = []
            if state.pullback is not None:
                armed.append("PB")
            if state.breakout is not None:
                armed.append("MOM")
            armed_text = "+".join(armed) if armed else "none"
            self._persist_state()
            return EntryResult(
                NONE,
                False,
                f"no valid 5M EMA trigger (armed={armed_text}) | 15M={context['structure']} 5M={snapshot['structure']} ADX={snapshot['adx']:.1f} CHOP={snapshot['chop']:.1f}",
                **base,
            )

        candidates.sort(key=lambda x: x.edge, reverse=True)
        selected = candidates[0]
        if len(candidates) > 1 and abs(candidates[0].edge - candidates[1].edge) <= 2:
            selected = next((x for x in candidates if x.setup_type == FAST_PULLBACK), selected)
        if state.last_entry_key is not None and selected.signal_key == state.last_entry_key:
            self._persist_state()
            return EntryResult(NONE, False, "duplicate signal key", **base)

        state.last_candidate_key = selected.signal_key
        self._persist_state()
        return EntryResult(
            direction=selected.direction,
            allow_entry=True,
            reason=(
                f"{selected.setup_type} {selected.trigger}: score={selected.score:.1f}/"
                f"{selected.threshold:.1f} edge={selected.edge:+.1f} "
                f"room={selected.room_r:.2f}R actualRR={selected.rr:.2f}"
            ),
            price=selected.price,
            ema_fast=snapshot["ema_fast"],
            ema_slow=snapshot["ema_slow"],
            macd_hist=selected.edge,
            cross_id=selected.signal_key,
            entry_score=selected.score,
            score_evaluated=True,
            score_threshold=selected.threshold,
            setup_type=selected.setup_type,
            trigger=selected.trigger,
            planned_stop=selected.stop,
            planned_target=selected.target,
            planned_rr=selected.rr,
            structure_room_r=selected.room_r,
            invalidation_level=selected.invalidation,
            score_components=selected.components,
        )

    def confirm_entry(self, symbol: str, cross_ts) -> None:
        if cross_ts is not None:
            state = self._get_state(symbol)
            state.last_entry_key = cross_ts
            state.last_candidate_key = None
            state.pullback = None
            state.breakout = None
            self._persist_state()

    def check_exit(
        self,
        df_5m: pd.DataFrame,
        position_side: str,
        bars_since_entry: Optional[int] = None,
    ) -> ExitCheckResult:
        """Noise-resistant 5M exit using EMA10/EMA20, not the faster entry pair."""
        c = self.cfg
        if bars_since_entry is not None and bars_since_entry < c.exit_grace_bars:
            return ExitCheckResult(False)
        if df_5m is None or len(df_5m) < 50:
            return ExitCheckResult(False)

        close = df_5m["close"]
        fast = ind.ema(close, c.l3c_ema_fast)
        slow = ind.ema(close, c.l3c_ema_slow)
        roc5 = ind.roc(close, 5)
        _, plus_di, minus_di = ind.adx(df_5m, 12)
        is_long = position_side == "long"

        def weakness(i: int) -> tuple[bool, int]:
            if is_long:
                price_wrong = close.iloc[i] < slow.iloc[i]
                score = int(roc5.iloc[i] < 0) + int(minus_di.iloc[i] > plus_di.iloc[i]) + int(close.iloc[i] < fast.iloc[i])
            else:
                price_wrong = close.iloc[i] > slow.iloc[i]
                score = int(roc5.iloc[i] > 0) + int(plus_di.iloc[i] > minus_di.iloc[i]) + int(close.iloc[i] > fast.iloc[i])
            return bool(price_wrong), int(score)

        if is_long:
            cross_back = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
            momentum_wrong = roc5.iloc[-1] < 0 or minus_di.iloc[-1] > plus_di.iloc[-1]
        else:
            cross_back = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
            momentum_wrong = roc5.iloc[-1] > 0 or plus_di.iloc[-1] > minus_di.iloc[-1]
        wrong_now, score_now = weakness(-1)
        wrong_prev, score_prev = weakness(-2)

        if cross_back and wrong_now and momentum_wrong:
            return ExitCheckResult(
                True,
                EMA_CROSS_REVERSAL,
                f"confirmed EMA reversal: weak={score_now}/3 close={close.iloc[-1]:.6f} slow={slow.iloc[-1]:.6f}",
            )
        required = getattr(c, "exit_weak_signals", 2)
        if wrong_now and wrong_prev and score_now >= required and score_prev >= required:
            return ExitCheckResult(
                True,
                PRICE_OPEN_BEYOND_EMA,
                f"2-bar weakness: now={score_now}/3 prev={score_prev}/3",
            )
        return ExitCheckResult(False)
