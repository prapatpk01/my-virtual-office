"""DUALCORE V2.0 — active-frequency 15M structure + 5M execution engine.

Architecture:
    4H macro -> 1H bias -> 15M context/structure -> 5M execution.

Five independent entry engines share the same higher-timeframe permission:
* FAST_PULLBACK: HTF/EMA-zone pullback with reclaim or micro structure break.
* MICRO_PULLBACK: shallow 5M higher-low/lower-high continuation.
* EMA_RECLAIM: EMA13/EMA20 touch followed by EMA8 reclaim; no new cross needed.
* TREND_CONTINUATION: EMA5/9 recross after a real pullback in strong context.
* MOMENTUM: significant swing/base breakout, direct or breakout-retest.

Hard structure, room, fee-drag, stop-distance and actual-R:R gates remain
non-compensable. All decisions use closed candles and persisted duplicate state.
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
TREND_CONTINUATION = "TREND_CONTINUATION"
MICRO_PULLBACK = "MICRO_PULLBACK"
EMA_RECLAIM = "EMA_RECLAIM"


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
    source: str = "BASE"
    compressed: bool = False


@dataclass
class _ReentryLock:
    direction: str
    set_ts: pd.Timestamp
    exit_reason: str


@dataclass
class _SetupState:
    last_processed_bar: Optional[pd.Timestamp] = None
    pullback: Optional[_PullbackSetup] = None
    breakout: Optional[_BreakoutSetup] = None
    reentry_lock: Optional[_ReentryLock] = None
    last_entry_key: object = None
    last_candidate_key: object = None
    sl_streak_direction: str = ""
    sl_streak_count: int = 0
    last_sl_ts: Optional[pd.Timestamp] = None


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
                lock_raw = raw.get("reentry_lock")
                pullback = None
                breakout = None
                reentry_lock = None
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
                            source=str(bo_raw.get("source", "BASE")),
                            compressed=bool(bo_raw.get("compressed", False)),
                        )
                if lock_raw:
                    set_ts = self._restore_ts(lock_raw.get("set_ts"))
                    if set_ts is not None:
                        reentry_lock = _ReentryLock(
                            direction=str(lock_raw.get("direction", "")),
                            set_ts=set_ts,
                            exit_reason=str(lock_raw.get("exit_reason", "")),
                        )
                self._state[str(symbol)] = _SetupState(
                    last_processed_bar=self._restore_ts(raw.get("last_processed_bar")),
                    pullback=pullback,
                    breakout=breakout,
                    reentry_lock=reentry_lock,
                    last_entry_key=self._restore_key(raw.get("last_entry_key")),
                    last_candidate_key=self._restore_key(raw.get("last_candidate_key")),
                    sl_streak_direction=str(raw.get("sl_streak_direction", "")),
                    sl_streak_count=int(raw.get("sl_streak_count", 0)),
                    last_sl_ts=self._restore_ts(raw.get("last_sl_ts")),
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
            reentry_lock = asdict(state.reentry_lock) if state.reentry_lock is not None else None
            if pullback is not None:
                pullback["started_ts"] = state.pullback.started_ts.isoformat()
            if breakout is not None:
                breakout["started_ts"] = state.breakout.started_ts.isoformat()
            if reentry_lock is not None:
                reentry_lock["set_ts"] = state.reentry_lock.set_ts.isoformat()
            symbols[symbol] = {
                "last_processed_bar": (
                    state.last_processed_bar.isoformat()
                    if state.last_processed_bar is not None else None
                ),
                "pullback": pullback,
                "breakout": breakout,
                "reentry_lock": reentry_lock,
                "last_entry_key": self._json_key(state.last_entry_key),
                "last_candidate_key": self._json_key(state.last_candidate_key),
                "sl_streak_direction": state.sl_streak_direction,
                "sl_streak_count": state.sl_streak_count,
                "last_sl_ts": state.last_sl_ts.isoformat() if state.last_sl_ts is not None else None,
            }
        tmp = f"{self._state_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"version": 4, "symbols": symbols},
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

    def _is_precision_symbol(self, symbol: str) -> bool:
        upper = str(symbol).upper()
        return any(k in upper for k in getattr(
            self.cfg, "dual_precision_symbol_keywords", ("BTC", "ETH", "XAU", "XAG")
        ))

    def _is_high_beta_symbol(self, symbol: str) -> bool:
        upper = str(symbol).upper()
        return any(k in upper for k in getattr(
            self.cfg, "dual_high_beta_symbol_keywords", ("SOL", "XRP", "HYPE")
        ))

    def reset_symbol(self, symbol: str) -> None:
        self._state.pop(symbol, None)
        self._persist_state()

    def on_position_closed(
        self,
        symbol: str,
        direction: Optional[str] = None,
        exit_reason: str = "",
        trade_pnl: float = 0.0,
        closed_at: Optional[pd.Timestamp] = None,
    ) -> None:
        """Clear used setups and apply a two-strike same-side structure lock.

        The first full SL uses only the normal symbol cooldown. A fresh 15M
        structure requirement is activated only after repeated same-direction
        full SLs inside a rolling window. This keeps frequency active without
        allowing unlimited retries on a failed trend thesis.
        """
        state = self._get_state(symbol)
        state.pullback = None
        state.breakout = None
        state.last_candidate_key = None
        normalized = str(direction or "").upper()
        now = pd.Timestamp(closed_at) if closed_at is not None else pd.Timestamp.utcnow()
        is_full_loss = (
            normalized in (LONG, SHORT)
            and trade_pnl < 0
            and str(exit_reason).upper() in {"SL_HIT", "FALSE_BREAKOUT"}
        )
        if is_full_loss:
            window_h = max(1, int(getattr(self.cfg, "dual_reentry_sl_window_hours", 12)))
            inside_window = bool(
                state.last_sl_ts is not None
                and now - state.last_sl_ts <= pd.Timedelta(hours=window_h)
                and state.sl_streak_direction == normalized
            )
            state.sl_streak_count = state.sl_streak_count + 1 if inside_window else 1
            state.sl_streak_direction = normalized
            state.last_sl_ts = now
            lock_after = max(1, int(getattr(self.cfg, "dual_reentry_lock_after_sl_count", 2)))
            if (
                getattr(self.cfg, "dual_reentry_requires_new_structure", True)
                and state.sl_streak_count >= lock_after
            ):
                state.reentry_lock = _ReentryLock(
                    direction=normalized,
                    set_ts=now,
                    exit_reason=f"{exit_reason}_X{state.sl_streak_count}",
                )
        elif trade_pnl > 0:
            state.reentry_lock = None
            state.sl_streak_direction = ""
            state.sl_streak_count = 0
            state.last_sl_ts = None
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
        """15M structure/context gate. At least two independent groups must agree."""
        c = self.cfg
        close = df["close"]
        ema_fast_s = ind.ema(close, getattr(c, "dual_context_ema_fast", 20))
        ema_slow_s = ind.ema(close, getattr(c, "dual_context_ema_slow", 50))
        atr_s = ind.atr(df, 14)
        atr_value = ind.safe_float(atr_s.iloc[-1])
        adx_s, plus_s, minus_s = ind.adx(df, 12)
        adx_value = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_s.iloc[-1])
        minus_di = ind.safe_float(minus_s.iloc[-1])
        chop = ind.safe_float(ind.choppiness_index(df, 14).iloc[-1], 100.0)
        roc_s = ind.roc(close, 9)
        roc9 = ind.safe_float(roc_s.iloc[-1])
        price = ind.safe_float(close.iloc[-1])
        ef = ind.safe_float(ema_fast_s.iloc[-1])
        es = ind.safe_float(ema_slow_s.iloc[-1])
        ef_prev = ind.safe_float(ema_fast_s.iloc[-2])
        structure = ind.market_structure(
            df["high"], df["low"],
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
        )
        bull_bos, bull_level = ind.latest_bos(
            df, LONG,
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
            0.18,
        )
        bear_bos, bear_level = ind.latest_bos(
            df, SHORT,
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
            0.18,
        )
        swing_high, swing_low = ind.recent_swing_levels(
            df["high"], df["low"],
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
        )
        long = direction == LONG
        aligned = ef > es if long else ef < es
        slope_ok = ef > ef_prev if long else ef < ef_prev
        di_tolerance = getattr(c, "dual_di_tolerance", 2.0)
        dmi_dominant = plus_di > minus_di if long else minus_di > plus_di
        dmi_ok = plus_di >= minus_di - di_tolerance if long else minus_di >= plus_di - di_tolerance
        roc_ok = roc9 > 0 if long else roc9 < 0
        structure_aligned = structure == ("HH_HL" if long else "LH_LL")
        opposite_structure = structure == ("LH_LL" if long else "HH_HL")
        directional_bos = bull_bos if long else bear_bos
        opposite_bos = bear_bos if long else bull_bos
        price_hold = price > ef if long else price < ef
        slow_hold = price > es if long else price < es

        groups = {
            "trend": bool(aligned and slope_ok),
            "structure": bool(structure_aligned or directional_bos),
            "momentum": bool(roc_ok and dmi_ok),
            "hold": bool(price_hold and slow_hold),
        }
        group_count = sum(int(v) for v in groups.values())
        obvious_chop = (
            (adx_value < 10.0 and chop > 64.0)
            or (structure == "MIXED" and adx_value < 11.0 and chop > 62.0 and not directional_bos)
        )
        opposite_shift = bool(
            opposite_bos
            or (opposite_structure and not aligned and not slow_hold)
        )
        allowed = (
            not obvious_chop
            and not opposite_shift
            and group_count >= getattr(c, "dual_context_min_groups", 2)
        )
        strong = bool(
            allowed
            and groups["trend"]
            and group_count >= 3
            and adx_value >= 15.0
            and chop <= 60.0
        )
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
            "dmi_dominant": dmi_dominant,
            "roc_ok": roc_ok,
            "structure_aligned": structure_aligned,
            "opposite_structure": opposite_structure,
            "directional_bos": directional_bos,
            "opposite_bos": opposite_bos,
            "bull_bos_level": bull_level,
            "bear_bos_level": bear_level,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "groups": groups,
            "group_count": group_count,
            "strong": strong,
            "transition": allowed and not strong,
            "allowed": allowed,
            "obvious_chop": obvious_chop,
            "opposite_shift": opposite_shift,
        }

    def _entry_snapshot(self, df: pd.DataFrame, direction: str) -> dict:
        """5M EMA timing plus independent local bull/bear pressure scores."""
        c = self.cfg
        close = df["close"]
        fast_len = getattr(c, "dual_entry_ema_fast", 8)
        slow_len = getattr(c, "dual_entry_ema_slow", 13)
        fast_s = ind.ema(close, fast_len)
        slow_s = ind.ema(close, slow_len)
        ema20_s = ind.ema(close, getattr(c, "dual_entry_trend_ema", 20))
        ema50_s = ind.ema(close, getattr(c, "dual_entry_filter_ema", 50))
        atr_s = ind.atr(df, 14)
        atr_value = ind.safe_float(atr_s.iloc[-1])
        adx_s, plus_s, minus_s = ind.adx(df, 12)
        adx_value = ind.safe_float(adx_s.iloc[-1])
        adx_prev = ind.safe_float(adx_s.iloc[-2])
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
        bull_bos, _ = ind.latest_bos(
            df, LONG,
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
            0.15,
        )
        bear_bos, _ = ind.latest_bos(
            df, SHORT,
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
            0.15,
        )
        candle = ind.candle_metrics(df, atr_value)
        price = ind.safe_float(close.iloc[-1])
        ef = ind.safe_float(fast_s.iloc[-1])
        es = ind.safe_float(slow_s.iloc[-1])
        e20 = ind.safe_float(ema20_s.iloc[-1])
        e50 = ind.safe_float(ema50_s.iloc[-1])
        ef_prev = ind.safe_float(fast_s.iloc[-2])
        es_prev = ind.safe_float(slow_s.iloc[-2])
        e20_prev = ind.safe_float(ema20_s.iloc[-2])
        spread_atr = abs(ef - es) / max(atr_value, ind.EPSILON)
        flip_count = ind.cross_count(fast_s, slow_s, 10)

        bull_components = {
            "ema_alignment": 15.0 if ef > es else 0.0,
            "fast_slope": 8.0 if ef > ef_prev else 0.0,
            "slow_slope": 5.0 if es >= es_prev else 0.0,
            "price_ema20": 10.0 if price > e20 else 0.0,
            "ema20_slope": 5.0 if e20 > e20_prev else 0.0,
            "price_ema50": 5.0 if price > e50 else 0.0,
            "structure": 15.0 if structure == "HH_HL" else 9.0 if bull_bos else 0.0,
            "roc": 10.0 if roc5 > 0 else 0.0,
            "roc_accel": 5.0 if roc5 > roc_prev else 0.0,
            "dmi": 10.0 if plus_di > minus_di else 0.0,
            "adx": 5.0 if adx_value >= 15 or adx_value > adx_prev else 0.0,
            "candle": 7.0 if candle.bullish and candle.bull_close_quality >= 0.60 else 0.0,
        }
        bear_components = {
            "ema_alignment": 15.0 if ef < es else 0.0,
            "fast_slope": 8.0 if ef < ef_prev else 0.0,
            "slow_slope": 5.0 if es <= es_prev else 0.0,
            "price_ema20": 10.0 if price < e20 else 0.0,
            "ema20_slope": 5.0 if e20 < e20_prev else 0.0,
            "price_ema50": 5.0 if price < e50 else 0.0,
            "structure": 15.0 if structure == "LH_LL" else 9.0 if bear_bos else 0.0,
            "roc": 10.0 if roc5 < 0 else 0.0,
            "roc_accel": 5.0 if roc5 < roc_prev else 0.0,
            "dmi": 10.0 if minus_di > plus_di else 0.0,
            "adx": 5.0 if adx_value >= 15 or adx_value > adx_prev else 0.0,
            "candle": 7.0 if candle.bearish and candle.bear_close_quality >= 0.60 else 0.0,
        }
        bull_local = min(100.0, sum(bull_components.values()))
        bear_local = min(100.0, sum(bear_components.values()))
        local_edge = bull_local - bear_local
        long = direction == LONG
        aligned = ef > es if long else ef < es
        fresh_cross = (
            ef > es and ef_prev <= es_prev
            if long else ef < es and ef_prev >= es_prev
        )
        fast_slope = ef > ef_prev if long else ef < ef_prev
        slow_slope = es >= es_prev if long else es <= es_prev
        di_tolerance = getattr(c, "dual_di_tolerance", 2.0)
        dmi_dominant = plus_di > minus_di if long else minus_di > plus_di
        dmi_ok = plus_di >= minus_di - di_tolerance if long else minus_di >= plus_di - di_tolerance
        roc_ok = roc5 > 0 if long else roc5 < 0
        roc_accel = roc5 > roc_prev if long else roc5 < roc_prev
        adx_support = adx_value >= getattr(c, "dual_momentum_min_adx", 11.0) or adx_value > adx_prev
        structure_ok = structure != ("LH_LL" if long else "HH_HL")
        ema20_ok = price >= e20 - 0.20 * atr_value if long else price <= e20 + 0.20 * atr_value
        ema50_ok = price >= e50 - 0.35 * atr_value if long else price <= e50 + 0.35 * atr_value
        support_count = sum((aligned, roc_ok, dmi_ok, adx_support))
        obvious_chop = (
            (adx_value < getattr(c, "dual_min_adx", 10.0) and chop > getattr(c, "dual_max_chop", 64.0))
            or (flip_count >= 5 and spread_atr < 0.07)
        )
        strong = (
            aligned and fast_slope and slow_slope and dmi_ok
            and adx_value >= getattr(c, "dual_strong_adx", 18.0)
            and chop <= getattr(c, "dual_strong_chop", 55.0)
        )
        direction_score = bull_local if long else bear_local
        opposite_score = bear_local if long else bull_local
        direction_edge = direction_score - opposite_score
        return {
            "price": price,
            "ema_fast_series": fast_s,
            "ema_slow_series": slow_s,
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
            "dmi_dominant": dmi_dominant,
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
            "bull_local_score": round(bull_local, 1),
            "bear_local_score": round(bear_local, 1),
            "local_edge": round(local_edge, 1),
            "direction_score": round(direction_score, 1),
            "opposite_score": round(opposite_score, 1),
            "direction_edge": round(direction_edge, 1),
            "bull_components": bull_components,
            "bear_components": bear_components,
        }

    def _dynamic_threshold(
        self,
        base: float,
        context: dict,
        snapshot: dict,
        regime=None,
        floor: Optional[float] = None,
    ) -> float:
        """Adapt the trigger bar without relaxing hard structure/risk gates."""
        c = self.cfg
        if not getattr(c, "dual_dynamic_threshold_enabled", True):
            return float(base)
        name = str(getattr(regime, "name", getattr(regime, "label", "")))
        threshold = float(base)
        strong = bool(
            name.startswith("STRONG_")
            or (
                context.get("strong", False)
                and snapshot.get("adx", 0.0) >= getattr(c, "dual_strong_adx", 18.0)
                and snapshot.get("chop", 100.0) <= getattr(c, "dual_strong_chop", 57.0)
            )
        )
        if strong:
            threshold -= getattr(c, "dual_strong_threshold_discount", 6.0)
        elif context.get("group_count", 0) >= 3:
            threshold -= getattr(c, "dual_normal_threshold_discount", 3.0)
        elif context.get("transition", False):
            threshold += getattr(c, "dual_transition_threshold_add", 2.0)
        edge = float(snapshot.get("direction_edge", 0.0))
        if edge >= 35.0:
            threshold -= 2.0
        elif edge < 10.0:
            threshold += 2.0
        hard_floor = getattr(c, "dual_threshold_floor", 58.0) if floor is None else floor
        return round(max(float(hard_floor), threshold), 1)

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
        minimum_room_override: Optional[float] = None,
        max_fee_drag_override: Optional[float] = None,
    ) -> Optional[_Candidate]:
        c = self.cfg
        long = direction == LONG
        if atr_value <= 0 or price <= 0:
            return None
        cost_distance = price * (
            2.0 * max(getattr(c, "fee_rate", 0.0), 0.0)
            + max(getattr(c, "expected_slippage_pct", 0.0), 0.0)
        )
        min_stop = max(
            atr_value * getattr(c, "dual_min_stop_atr", 0.80),
            price * max(getattr(c, "sl_min_pct", 0.0075), 0.0),
            cost_distance * getattr(c, "stop_fee_floor_mult", 3.0),
        )
        max_stop = max(atr_value * getattr(c, "dual_max_stop_atr", 2.20), min_stop)
        pct_cap = price * max(getattr(c, "sl_max_pct", 0.020), 0.0)
        if pct_cap > 0:
            max_stop = max(min_stop, min(max_stop, pct_cap))

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
        if risk <= 0 or risk > max_stop * (1.0 + 1e-9):
            return None
        fee_drag_r = cost_distance / max(risk, ind.EPSILON)
        max_fee_drag = (
            getattr(c, "max_fee_drag_r", 0.35)
            if max_fee_drag_override is None
            else max_fee_drag_override
        )
        if fee_drag_r > max_fee_drag:
            return None

        pullback_like = setup_type in {FAST_PULLBACK, MICRO_PULLBACK, EMA_RECLAIM, TREND_CONTINUATION}
        base_rr = (
            getattr(c, "dual_pullback_tp2_r", 2.20)
            if pullback_like
            else getattr(c, "dual_momentum_tp2_r", 2.40)
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
        default_room = (
            getattr(c, "dual_pullback_min_room_r", 1.00)
            if pullback_like
            else getattr(c, "dual_momentum_min_room_r", 1.10)
        )
        min_room = default_room if minimum_room_override is None else minimum_room_override
        if room_r < min_room or rr < getattr(c, "minimum_actual_rr", 1.50):
            return None
        signal_key = (str(bar_ts), direction, setup_type, trigger, round(price, 10))
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
            components={
                **components,
                "fee_drag_r": round(fee_drag_r, 4),
                "stop_pct": round(risk / price, 6),
                "required_room_r": round(min_room, 2),
            },
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
        symbol: str = "",
    ) -> Optional[_Candidate]:
        c = self.cfg
        long = direction == LONG
        precision_symbol = self._is_precision_symbol(symbol)
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
        bar_ts = pd.Timestamp(df.index[-1])
        ema_fast, ema_slow = snapshot["ema_fast"], snapshot["ema_slow"]
        zone_upper = max(ema_fast, ema_slow) + atr_value * getattr(c, "dual_pullback_zone_atr", 0.15)
        zone_lower = min(ema_fast, ema_slow) - atr_value * getattr(c, "dual_pullback_depth_atr", 0.30)

        level = nearest_support if long else nearest_resistance
        sweep = ind.sweep_reclaim(df, direction, level)
        if long:
            ema_touch = (
                float(row["low"]) <= zone_upper
                and float(row["low"]) >= snapshot["ema20"] - 0.55 * atr_value
                and price >= snapshot["ema20"] - 0.20 * atr_value
            )
            htf_touch = level is not None and float(row["low"]) <= level + 0.15 * atr_value and price >= level
            confluence = ema_touch and level is not None and abs(level - ema_slow) <= 0.35 * atr_value
            location_touch = ema_touch or htf_touch or sweep
            reclaim = price > ema_fast and float(row["low"]) <= ema_fast
            previous_break = price > float(df["high"].iloc[-2])
            micro_bos = price > float(df["high"].iloc[-4:-1].max())
            invalidation = min(float(row["low"]), level) if level is not None and price >= level else float(row["low"])
        else:
            ema_touch = (
                float(row["high"]) >= zone_lower
                and float(row["high"]) <= snapshot["ema20"] + 0.55 * atr_value
                and price <= snapshot["ema20"] + 0.20 * atr_value
            )
            htf_touch = level is not None and float(row["high"]) >= level - 0.15 * atr_value and price <= level
            confluence = ema_touch and level is not None and abs(level - ema_slow) <= 0.35 * atr_value
            location_touch = ema_touch or htf_touch or sweep
            reclaim = price < ema_fast and float(row["high"]) >= ema_fast
            previous_break = price < float(df["low"].iloc[-2])
            micro_bos = price < float(df["low"].iloc[-4:-1].min())
            invalidation = max(float(row["high"]), level) if level is not None and price <= level else float(row["high"])

        if not location_touch and state.pullback is None:
            return None
        location_score = (
            20.0 if sweep and htf_touch
            else 18.0 if sweep
            else 17.0 if htf_touch
            else 14.0 if confluence
            else 8.0 if ema_touch
            else 0.0
        )
        candle = snapshot["candle"]
        candle_ok = (
            ind.bullish_trigger_candle(candle, c.dual_pullback_min_body_atr, c.dual_pullback_close_quality)
            if long
            else ind.bearish_trigger_candle(candle, c.dual_pullback_min_body_atr, c.dual_pullback_close_quality)
        )
        strong_candle = bool(
            candle.body_atr >= 0.22
            and (candle.bull_close_quality >= 0.70 if long else candle.bear_close_quality >= 0.70)
        )

        # EMA-only touches may arm a setup, but same-bar entry requires an HTF
        # location/sweep plus a real structure trigger.
        same_bar_trigger = bool(
            location_touch
            and (htf_touch or sweep)
            and candle_ok
            and strong_candle
            and (sweep or micro_bos)
            and snapshot["direction_edge"] >= getattr(c, "dual_direct_directional_edge", 10.0)
        )

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
        elif location_touch and state.pullback is not None:
            state.pullback.location_score = max(state.pullback.location_score, location_score)

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
            trigger = "SWEEP_RECLAIM" if sweep else "HTF_MICRO_BOS"
        elif age >= 1 and candle_ok:
            if sweep:
                trigger = "SWEEP_RECLAIM"
            elif micro_bos:
                trigger = "MICRO_BOS"
            elif armed_break:
                trigger = "TRIGGER_BREAK"
            elif reclaim and (snapshot["fresh_cross"] or previous_break or setup.location_score >= 17.0):
                trigger = "EMA_RECLAIM_CONFIRM"
        if precision_symbol and setup.location_score < 14.0:
            # Precision assets may use a delayed EMA-zone trigger break only
            # when the 15M context is strong and the 5M directional edge is
            # decisive. A naked EMA reclaim remains insufficient.
            if trigger == "EMA_RECLAIM_CONFIRM":
                trigger = ""
            elif trigger == "TRIGGER_BREAK":
                regime_name = str(getattr(regime, "name", getattr(regime, "label", "")))
                if not (
                    regime_name.startswith("STRONG_")
                    and context["strong"]
                    and strong_candle
                    and snapshot["direction_edge"] >= 32.0
                    and age >= 1
                ):
                    trigger = ""
                else:
                    trigger = "EMA_ZONE_TRIGGER_BREAK"

        # EMA reclaim has no structure break of its own. In an EARLY regime it
        # must show near-unanimous 5M directional agreement; in a STRONG regime
        # a lower, but still meaningful, edge is sufficient.
        if trigger == "EMA_RECLAIM_CONFIRM":
            regime_name = str(getattr(regime, "name", getattr(regime, "label", "")))
            required_edge = (
                getattr(c, "dual_ema_reclaim_early_min_edge", 85.0)
                if regime_name.startswith("EARLY_")
                else getattr(c, "dual_ema_reclaim_strong_min_edge", 55.0)
            )
            if snapshot["direction_edge"] < required_edge:
                trigger = ""

        if not trigger:
            return None
        if (
            snapshot["direction_score"] < getattr(c, "dual_local_score_floor", 52.0)
            or snapshot["direction_edge"] < getattr(c, "dual_local_directional_edge", 8.0)
        ):
            return None

        extension = (price - ema_slow) / atr_value if long else (ema_slow - price) / atr_value
        if precision_symbol:
            max_pullback_extension = getattr(c, "dual_precision_pullback_max_extension_atr", 0.65)
        elif self._is_high_beta_symbol(symbol):
            max_pullback_extension = getattr(c, "dual_high_beta_pullback_max_extension_atr", 0.70)
        else:
            max_pullback_extension = getattr(c, "dual_pullback_max_extension_atr", 0.80)
        if extension < -0.20 or extension > max_pullback_extension:
            return None

        recent = df.iloc[-6:]
        if long:
            prior_move = (float(recent["high"].max()) - float(recent["low"].min())) / max(atr_value, ind.EPSILON)
            prior_displacement = any(
                ind.candle_metrics(df.iloc[:i], atr_value).bullish
                and ind.candle_metrics(df.iloc[:i], atr_value).body_atr >= 0.25
                for i in range(max(2, len(df) - 5), len(df))
            )
        else:
            prior_move = (float(recent["high"].max()) - float(recent["low"].min())) / max(atr_value, ind.EPSILON)
            prior_displacement = any(
                ind.candle_metrics(df.iloc[:i], atr_value).bearish
                and ind.candle_metrics(df.iloc[:i], atr_value).body_atr >= 0.25
                for i in range(max(2, len(df) - 5), len(df))
            )
        prior_context = context["structure_aligned"] or context["directional_bos"] or prior_displacement or prior_move >= 0.65
        if not prior_context:
            return None

        bias_edge = abs(getattr(bias, "directional_edge", 0.0))
        components = {
            "bias": 15.0 if bias_edge >= 15 else 12.0 if bias_edge >= 10 else 8.0,
            "context_structure": (
                20.0 if context["structure_aligned"]
                else 17.0 if context["directional_bos"]
                else 14.0 if context["group_count"] >= 3
                else 10.0
            ),
            "location": setup.location_score,
            "prior_context": 10.0 if prior_displacement or context["directional_bos"] else 8.0 if context["structure_aligned"] else 5.0,
            "trigger": 15.0 if trigger in ("SWEEP_RECLAIM", "MICRO_BOS", "HTF_MICRO_BOS") else 13.0 if trigger in ("TRIGGER_BREAK", "EMA_ZONE_TRIGGER_BREAK") else 10.0,
            "candle": 10.0 if strong_candle else 7.0 if candle_ok else 0.0,
            "local_edge": 10.0 if snapshot["direction_edge"] >= 18 else 7.0 if snapshot["direction_edge"] >= 12 else 4.0,
        }
        score = min(100.0, sum(components.values()))
        base_threshold = (
            getattr(c, "dual_same_bar_pullback_threshold", 72.0)
            if age == 0
            else getattr(c, "dual_pullback_threshold", 66.0)
        )
        threshold = self._dynamic_threshold(
            base_threshold, context, snapshot, regime, floor=66.0 if age == 0 else 62.0
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
            components={
                **components,
                "local_direction_score": snapshot["direction_score"],
                "local_opposite_score": snapshot["opposite_score"],
                "local_direction_edge": snapshot["direction_edge"],
                "context_groups": context["group_count"],
                "extension_atr": round(extension, 3),
            },
        )
        if candidate is not None:
            state.pullback = None
        return candidate

    def _micro_pullback_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        context: dict,
        snapshot: dict,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        regime=None,
        bias=None,
    ) -> Optional[_Candidate]:
        """Shallow 5M higher-low/lower-high continuation inside HTF bias."""
        c = self.cfg
        if not getattr(c, "dual_micro_pullback_enabled", True) or len(df) < 12:
            return None
        long = direction == LONG
        if (
            snapshot["obvious_chop"]
            or not context["allowed"]
            or context["group_count"] < 2
            or not snapshot["structure_ok"]
            or not snapshot["ema50_ok"]
            or not snapshot["aligned"]
            or snapshot["direction_score"] < getattr(c, "dual_micro_pullback_min_score", 54.0)
            or snapshot["direction_edge"] < getattr(c, "dual_micro_pullback_min_edge", 10.0)
        ):
            return None
        atr_value = snapshot["atr"]
        if atr_value <= 0:
            return None
        row = df.iloc[-1]
        prev = df.iloc[-2]
        recent = df.iloc[-7:-1]
        candle = snapshot["candle"]
        candle_ok = (
            ind.bullish_trigger_candle(
                candle,
                getattr(c, "dual_micro_pullback_min_body_atr", 0.10),
                getattr(c, "dual_micro_pullback_close_quality", 0.58),
            )
            if long
            else ind.bearish_trigger_candle(
                candle,
                getattr(c, "dual_micro_pullback_min_body_atr", 0.10),
                getattr(c, "dual_micro_pullback_close_quality", 0.58),
            )
        )
        if not candle_ok:
            return None
        price = snapshot["price"]
        if long:
            impulse_high = float(recent["high"].max())
            pullback_low = float(df["low"].iloc[-4:].min())
            depth = (impulse_high - pullback_low) / max(atr_value, ind.EPSILON)
            trigger = price > float(prev["high"])
            hold = pullback_low >= snapshot["ema20"] - 0.18 * atr_value and price > snapshot["ema20"]
            raw_stop = pullback_low - getattr(c, "dual_stop_buffer_atr", 0.08) * atr_value
            extension = (price - snapshot["ema_slow"]) / atr_value
        else:
            impulse_low = float(recent["low"].min())
            pullback_high = float(df["high"].iloc[-4:].max())
            depth = (pullback_high - impulse_low) / max(atr_value, ind.EPSILON)
            trigger = price < float(prev["low"])
            hold = pullback_high <= snapshot["ema20"] + 0.18 * atr_value and price < snapshot["ema20"]
            raw_stop = pullback_high + getattr(c, "dual_stop_buffer_atr", 0.08) * atr_value
            extension = (snapshot["ema_slow"] - price) / atr_value
        if (
            not trigger
            or not hold
            or depth < 0.12
            or depth > getattr(c, "dual_micro_pullback_max_depth_atr", 0.85)
            or extension < -0.10
            or extension > getattr(c, "dual_micro_pullback_max_extension_atr", 0.75)
            or not (snapshot["roc_ok"] or snapshot["roc_accel"])
        ):
            return None
        bias_edge = abs(getattr(bias, "directional_edge", 0.0))
        components = {
            "bias": 14.0 if bias_edge >= 12 else 11.0,
            "context": 18.0 if context["strong"] else 14.0,
            "ema_alignment": 14.0,
            "shallow_pullback": 15.0 if depth <= 0.55 else 11.0,
            "bar_break": 14.0,
            "candle": 10.0,
            "local_edge": 10.0 if snapshot["direction_edge"] >= 22 else 7.0,
            "momentum": 8.0 if snapshot["dmi_dominant"] and snapshot["roc_ok"] else 5.0,
        }
        score = min(100.0, sum(components.values()))
        threshold = self._dynamic_threshold(
            getattr(c, "dual_micro_pullback_threshold", 65.0),
            context,
            snapshot,
            regime,
            floor=64.0,
        )
        if score < threshold:
            return None
        return self._finalize_candidate(
            direction=direction,
            setup_type=MICRO_PULLBACK,
            score=score,
            threshold=threshold,
            trigger="MICRO_HL_BREAK" if long else "MICRO_LH_BREAK",
            price=price,
            raw_stop=raw_stop,
            atr_value=atr_value,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            bar_ts=pd.Timestamp(df.index[-1]),
            components={
                **components,
                "depth_atr": round(depth, 3),
                "extension_atr": round(extension, 3),
                "local_direction_edge": snapshot["direction_edge"],
                "context_groups": context["group_count"],
            },
            minimum_room_override=getattr(c, "dual_micro_pullback_min_room_r", 1.00),
        )

    def _ema_reclaim_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        context: dict,
        snapshot: dict,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        regime=None,
        bias=None,
        symbol: str = "",
    ) -> Optional[_Candidate]:
        """EMA13/EMA20 touch followed by EMA8 reclaim; no fresh cross required."""
        c = self.cfg
        if not getattr(c, "dual_ema_reclaim_engine_enabled", True) or len(df) < 20:
            return None
        long = direction == LONG
        precision = self._is_precision_symbol(symbol)
        if (
            snapshot["obvious_chop"]
            or not context["allowed"]
            or context["group_count"] < (3 if precision else 2)
            or not snapshot["structure_ok"]
            or not snapshot["ema50_ok"]
            or not snapshot["aligned"]
            or snapshot["direction_score"] < getattr(c, "dual_ema_reclaim_min_score", 56.0)
            or snapshot["direction_edge"] < getattr(c, "dual_ema_reclaim_min_edge", 12.0)
        ):
            return None
        atr_value = snapshot["atr"]
        if atr_value <= 0:
            return None
        n = max(2, int(getattr(c, "dual_ema_reclaim_touch_bars", 4)))
        recent = df.iloc[-n-1:-1]
        fast = snapshot["ema_fast_series"]
        slow = snapshot["ema_slow_series"]
        ema20 = ind.ema(df["close"], getattr(c, "dual_entry_trend_ema", 20))
        recent_slow = slow.iloc[-n-1:-1]
        recent_e20 = ema20.iloc[-n-1:-1]
        price = snapshot["price"]
        row = df.iloc[-1]
        prev_close = float(df["close"].iloc[-2])
        candle = snapshot["candle"]
        candle_ok = (
            ind.bullish_trigger_candle(
                candle,
                getattr(c, "dual_ema_reclaim_min_body_atr", 0.10),
                getattr(c, "dual_ema_reclaim_close_quality", 0.57),
            )
            if long
            else ind.bearish_trigger_candle(
                candle,
                getattr(c, "dual_ema_reclaim_min_body_atr", 0.10),
                getattr(c, "dual_ema_reclaim_close_quality", 0.57),
            )
        )
        if long:
            touched_slow = bool((recent["low"].to_numpy() <= recent_slow.to_numpy() + 0.10 * atr_value).any())
            touched_e20 = bool((recent["low"].to_numpy() <= recent_e20.to_numpy() + 0.08 * atr_value).any())
            touched = touched_slow or touched_e20
            reclaimed = price > snapshot["ema_fast"] and prev_close <= float(fast.iloc[-2]) + 0.05 * atr_value
            hold = price > snapshot["ema20"] and price > snapshot["ema50"]
            raw_stop = min(float(df["low"].iloc[-n-1:].min()), snapshot["ema20"] - 0.08 * atr_value)
            extension = (price - snapshot["ema_slow"]) / atr_value
        else:
            touched_slow = bool((recent["high"].to_numpy() >= recent_slow.to_numpy() - 0.10 * atr_value).any())
            touched_e20 = bool((recent["high"].to_numpy() >= recent_e20.to_numpy() - 0.08 * atr_value).any())
            touched = touched_slow or touched_e20
            reclaimed = price < snapshot["ema_fast"] and prev_close >= float(fast.iloc[-2]) - 0.05 * atr_value
            hold = price < snapshot["ema20"] and price < snapshot["ema50"]
            raw_stop = max(float(df["high"].iloc[-n-1:].max()), snapshot["ema20"] + 0.08 * atr_value)
            extension = (snapshot["ema_slow"] - price) / atr_value
        if (
            not touched
            or not reclaimed
            or not hold
            or not candle_ok
            or extension < 0
            or extension > getattr(c, "dual_ema_reclaim_max_extension_atr", 0.70)
            or not (snapshot["roc_ok"] or snapshot["roc_accel"])
            or not snapshot["dmi_ok"]
        ):
            return None
        if precision and not (
            touched_e20
            and snapshot["dmi_dominant"]
            and snapshot["roc_ok"]
            and snapshot["direction_edge"] >= 45.0
            and (
                context["strong"]
                or context["structure_aligned"]
                or context["directional_bos"]
            )
        ):
            return None
        bias_edge = abs(getattr(bias, "directional_edge", 0.0))
        components = {
            "bias": 14.0 if bias_edge >= 12 else 11.0,
            "context": 18.0 if context["strong"] else 14.0,
            "ema_touch": 14.0,
            "ema_reclaim": 15.0,
            "trend_hold": 12.0,
            "candle": 10.0,
            "local_edge": 10.0 if snapshot["direction_edge"] >= 25 else 7.0,
            "momentum": 8.0 if snapshot["dmi_dominant"] and snapshot["roc_ok"] else 5.0,
        }
        score = min(100.0, sum(components.values()))
        threshold = self._dynamic_threshold(
            getattr(c, "dual_ema_reclaim_threshold", 64.0),
            context,
            snapshot,
            regime,
            floor=64.0 if not precision else 68.0,
        )
        if score < threshold:
            return None
        return self._finalize_candidate(
            direction=direction,
            setup_type=EMA_RECLAIM,
            score=score,
            threshold=threshold,
            trigger="EMA8_RECLAIM_AFTER_TOUCH",
            price=price,
            raw_stop=raw_stop,
            atr_value=atr_value,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            bar_ts=pd.Timestamp(df.index[-1]),
            components={
                **components,
                "precision_profile": precision,
                "extension_atr": round(extension, 3),
                "local_direction_edge": snapshot["direction_edge"],
                "context_groups": context["group_count"],
            },
            minimum_room_override=getattr(c, "dual_ema_reclaim_min_room_r", 1.00),
        )

    def _continuation_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        context: dict,
        snapshot: dict,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        regime=None,
        bias=None,
    ) -> Optional[_Candidate]:
        """Strong-context 5M EMA recross after a real pullback.

        This is the frequency engine. EMA8/13 is only the timing event; 15M
        trend/structure, a prior touch of EMA13/EMA20, local directional edge,
        candle quality, room and fee-aware risk remain mandatory.
        """
        c = self.cfg
        if not getattr(c, "dual_continuation_enabled", True):
            return None
        long = direction == LONG
        regime_name = str(getattr(regime, "name", getattr(regime, "label", "")))
        strong_regime = regime_name.startswith("STRONG_")
        early_allowed = bool(
            getattr(c, "dual_continuation_allow_early_regime", False)
            and regime_name.startswith("EARLY_")
            and context.get("group_count", 0) >= 3
        )
        if not (strong_regime or early_allowed):
            return None
        if (
            snapshot["obvious_chop"]
            or not snapshot["structure_ok"]
            or not snapshot["ema50_ok"]
            or not context["allowed"]
            or context["group_count"] < 2
        ):
            return None
        if (
            snapshot["direction_score"] < getattr(c, "dual_continuation_min_direction_score", 68.0)
            or snapshot["direction_edge"] < getattr(c, "dual_continuation_min_edge", 25.0)
        ):
            return None
        if len(df) < 30:
            return None

        close = df["close"]
        fast = ind.ema(close, getattr(c, "dual_continuation_ema_fast", 5))
        slow = ind.ema(close, getattr(c, "dual_continuation_ema_slow", 9))
        ema20 = ind.ema(close, getattr(c, "dual_entry_trend_ema", 20))
        max_age = max(0, int(getattr(c, "dual_continuation_max_cross_age_bars", 1)))
        cross_age = None
        for age in range(0, max_age + 1):
            i = -1 - age
            j = i - 1
            if abs(j) > len(df):
                break
            crossed = (
                fast.iloc[i] > slow.iloc[i] and fast.iloc[j] <= slow.iloc[j]
                if long
                else fast.iloc[i] < slow.iloc[i] and fast.iloc[j] >= slow.iloc[j]
            )
            if crossed:
                cross_age = age
                break
        if cross_age is None:
            return None
        # Alignment must still be valid at the actual decision candle.
        if not (fast.iloc[-1] > slow.iloc[-1] if long else fast.iloc[-1] < slow.iloc[-1]):
            return None

        atr_value = snapshot["atr"]
        price = snapshot["price"]
        recent = df.iloc[-5:-1]
        recent_slow = slow.iloc[-5:-1]
        recent_e20 = ema20.iloc[-5:-1]
        if long:
            pullback_touch = bool(
                (recent["low"].to_numpy() <= (recent_slow.to_numpy() + 0.12 * atr_value)).any()
                or (recent["low"].to_numpy() <= (recent_e20.to_numpy() + 0.10 * atr_value)).any()
            )
            hold_ok = price > snapshot["ema20"] and price > snapshot["ema50"]
            raw_stop = min(float(df["low"].iloc[-5:].min()), float(snapshot["ema20"] - 0.08 * atr_value))
        else:
            pullback_touch = bool(
                (recent["high"].to_numpy() >= (recent_slow.to_numpy() - 0.12 * atr_value)).any()
                or (recent["high"].to_numpy() >= (recent_e20.to_numpy() - 0.10 * atr_value)).any()
            )
            hold_ok = price < snapshot["ema20"] and price < snapshot["ema50"]
            raw_stop = max(float(df["high"].iloc[-5:].max()), float(snapshot["ema20"] + 0.08 * atr_value))
        if not pullback_touch or not hold_ok:
            return None

        candle = snapshot["candle"]
        candle_ok = (
            ind.bullish_trigger_candle(
                candle,
                getattr(c, "dual_continuation_min_body_atr", 0.14),
                getattr(c, "dual_continuation_close_quality", 0.62),
            )
            if long
            else ind.bearish_trigger_candle(
                candle,
                getattr(c, "dual_continuation_min_body_atr", 0.14),
                getattr(c, "dual_continuation_close_quality", 0.62),
            )
        )
        if not candle_ok or not snapshot["dmi_ok"] or not (snapshot["roc_ok"] or snapshot["roc_accel"]):
            return None
        extension = (price - snapshot["ema_slow"]) / atr_value if long else (snapshot["ema_slow"] - price) / atr_value
        if extension < 0 or extension > getattr(c, "dual_continuation_max_extension_atr", 0.60):
            return None

        bias_edge = abs(getattr(bias, "directional_edge", 0.0))
        components = {
            "bias": 15.0 if bias_edge >= 15 else 12.0,
            "context": 20.0 if context["structure_aligned"] or context["directional_bos"] else 16.0,
            "pullback_touch": 15.0,
            "ema_recross": 15.0 if cross_age == 0 else 12.0,
            "candle": 10.0,
            "local_edge": 15.0 if snapshot["direction_edge"] >= 40 else 12.0,
            "momentum": 10.0 if snapshot["adx_support"] and snapshot["roc_accel"] else 7.0,
        }
        score = min(100.0, sum(components.values()))
        threshold = self._dynamic_threshold(
            getattr(c, "dual_continuation_threshold", 70.0),
            context,
            snapshot,
            regime,
            floor=64.0,
        )
        if score < threshold:
            return None
        return self._finalize_candidate(
            direction=direction,
            setup_type=TREND_CONTINUATION,
            score=score,
            threshold=threshold,
            trigger="EMA_CROSS_CONTINUATION",
            price=price,
            raw_stop=raw_stop,
            atr_value=atr_value,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            bar_ts=pd.Timestamp(df.index[-1]),
            components={
                **components,
                "engine": TREND_CONTINUATION,
                "cross_age": cross_age,
                "local_direction_score": snapshot["direction_score"],
                "local_opposite_score": snapshot["opposite_score"],
                "local_direction_edge": snapshot["direction_edge"],
                "extension_atr": round(extension, 3),
            },
            minimum_room_override=getattr(c, "dual_continuation_min_room_r", 1.15),
        )

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
        symbol: str = "",
    ) -> Optional[_Candidate]:
        c = self.cfg
        long = direction == LONG
        precision_symbol = self._is_precision_symbol(symbol)
        atr_value = snapshot["atr"]
        if snapshot["obvious_chop"] or not snapshot["structure_ok"] or not context["allowed"]:
            state.breakout = None
            return None
        if (
            snapshot["direction_score"] < getattr(c, "dual_local_score_floor", 52.0)
            or snapshot["direction_edge"] < getattr(c, "dual_local_directional_edge", 8.0)
        ):
            return None

        price = snapshot["price"]
        row = df.iloc[-1]
        bar_ts = pd.Timestamp(df.index[-1])
        lookback = getattr(c, "dual_breakout_lookback", 10)
        if len(df) < lookback + 8:
            return None
        base_level = (
            float(df["high"].iloc[-lookback - 1:-1].max())
            if long else float(df["low"].iloc[-lookback - 1:-1].min())
        )
        major_level = context["swing_high"] if long else context["swing_low"]
        major_valid = bool(major_level is not None and pd.notna(major_level))
        prev_close = float(df["close"].iloc[-2])
        major_crossed = bool(
            major_valid
            and (price > major_level and prev_close <= major_level if long else price < major_level and prev_close >= major_level)
        )
        base_crossed = bool(
            price > base_level and prev_close <= base_level
            if long else price < base_level and prev_close >= base_level
        )
        compressed = ind.compression_ratio(df, 4, 20) <= getattr(c, "dual_base_compression_ratio", 0.88)
        crossed = major_crossed or base_crossed
        break_level = float(major_level) if major_crossed else base_level
        source = "15M_SWING" if major_crossed else "5M_BASE"

        candle = snapshot["candle"]
        close_quality = candle.bull_close_quality if long else candle.bear_close_quality
        candle_ok = (
            ind.bullish_trigger_candle(candle, c.dual_momentum_min_body_atr, c.dual_momentum_close_quality)
            if long else ind.bearish_trigger_candle(candle, c.dual_momentum_min_body_atr, c.dual_momentum_close_quality)
        )
        strong_displacement = bool(
            candle.body_atr >= getattr(c, "dual_direct_min_body_atr", 0.30)
            and close_quality >= getattr(c, "dual_direct_close_quality", 0.75)
            and (candle.volume_ratio >= getattr(c, "dual_momentum_volume_ratio", 1.05) or candle.body_atr >= 0.38)
        )
        direct_quality = bool(
            crossed
            and strong_displacement
            and snapshot["direction_edge"] >= getattr(c, "dual_direct_directional_edge", 10.0)
            and (major_crossed or compressed)
            and (context["strong"] or major_crossed)
            and (
                compressed
                or candle.volume_ratio >= getattr(c, "dual_direct_min_volume_ratio", 1.10)
            )
        )

        if crossed and (state.breakout is None or state.breakout.direction != direction):
            state.breakout = _BreakoutSetup(
                direction=direction,
                started_bar=0,
                started_ts=bar_ts,
                breakout_level=break_level,
                breakout_low=float(row["low"]),
                breakout_high=float(row["high"]),
                source=source,
                compressed=compressed,
            )

        setup = state.breakout
        if setup is None or setup.direction != direction:
            return None
        age = self._age_bars(setup.started_ts, bar_ts, 5)
        expiry = getattr(c, "dual_momentum_expiry_bars", 2)
        if age > expiry:
            state.breakout = None
            return None

        regime_name = str(getattr(regime, "name", getattr(regime, "label", "")))
        direct_regime_ok = not (
            getattr(c, "dual_block_direct_breakout_in_early_trend", True)
            and regime_name.startswith("EARLY_")
            and not (setup.source == "15M_SWING" or (setup.compressed and candle.volume_ratio >= 1.05))
        )
        direct = age == 0 and direct_quality and direct_regime_ok
        early_retest_quality = bool(
            not regime_name.startswith("EARLY_")
            or setup.compressed
            or candle.volume_ratio >= getattr(c, "dual_early_retest_min_volume_ratio", 1.10)
        )
        if long:
            retest = (
                1 <= age <= expiry
                and early_retest_quality
                and float(row["low"]) <= setup.breakout_level + 0.15 * atr_value
                and float(row["low"]) >= setup.breakout_level - 0.35 * atr_value
                and price > setup.breakout_level
                and candle_ok
                and snapshot["direction_edge"] >= getattr(c, "dual_local_directional_edge", 8.0)
            )
            invalidated = price < setup.breakout_level - 0.35 * atr_value
            setup.breakout_low = min(setup.breakout_low, float(row["low"]))
        else:
            retest = (
                1 <= age <= expiry
                and early_retest_quality
                and float(row["high"]) >= setup.breakout_level - 0.15 * atr_value
                and float(row["high"]) <= setup.breakout_level + 0.35 * atr_value
                and price < setup.breakout_level
                and candle_ok
                and snapshot["direction_edge"] >= getattr(c, "dual_local_directional_edge", 8.0)
            )
            invalidated = price > setup.breakout_level + 0.35 * atr_value
            setup.breakout_high = max(setup.breakout_high, float(row["high"]))
        if invalidated:
            state.breakout = None
            return None
        if not (direct or retest):
            return None

        level_extension = abs(price - setup.breakout_level) / max(atr_value, ind.EPSILON)
        ema_extension = (
            (price - snapshot["ema_slow"]) / atr_value
            if long else (snapshot["ema_slow"] - price) / atr_value
        )
        level_limit = (
            getattr(c, "dual_direct_max_level_extension_atr", 0.60)
            if direct else getattr(c, "dual_retest_max_level_extension_atr", 0.50)
        )
        max_ema_ext = (
            getattr(c, "dual_direct_max_ema_extension_atr", 0.85)
            if direct else getattr(c, "dual_retest_max_ema_extension_atr", 1.35)
        )
        if level_extension > level_limit:
            return None
        # A valid retest does not justify chasing when price is already far
        # from EMA13; this was a recurring losing pattern in the exact-data test.
        if ema_extension > max_ema_ext:
            return None

        bias_edge = abs(getattr(bias, "directional_edge", 0.0))
        significance = 20.0 if setup.source == "15M_SWING" else 13.0
        candle_points = 15.0 if strong_displacement else 11.0 if candle.body_atr >= 0.22 and close_quality >= 0.70 else 7.0
        components = {
            "bias": 15.0 if bias_edge >= 15 else 12.0 if bias_edge >= 10 else 8.0,
            "context": 20.0 if context["strong"] else 15.0 if context["group_count"] >= 3 else 10.0,
            "break_significance": significance,
            "candle_quality": candle_points,
            "hold_retest": 10.0 if retest else 6.0,
            "local_edge": 10.0 if snapshot["direction_edge"] >= 18 else 7.0 if snapshot["direction_edge"] >= 12 else 4.0,
            "volume_compression": 10.0 if candle.volume_ratio >= 1.10 and setup.compressed else 7.0 if candle.volume_ratio >= 1.05 or setup.compressed else 3.0,
            "extension": 5.0 if level_extension <= 0.25 else 3.0,
        }
        score = min(100.0, sum(components.values()))
        base_threshold = (
            getattr(c, "dual_strong_breakout_threshold", 76.0)
            if direct else getattr(c, "dual_momentum_threshold", 70.0)
        )
        threshold = self._dynamic_threshold(
            base_threshold,
            context,
            snapshot,
            regime,
            floor=70.0 if direct else 64.0,
        )
        if score < threshold:
            return None

        buffer = atr_value * getattr(c, "dual_stop_buffer_atr", 0.08)
        raw_stop = (
            min(setup.breakout_low, setup.breakout_level - buffer)
            if long else max(setup.breakout_high, setup.breakout_level + buffer)
        )
        trigger = (
            "DIRECT_MAJOR_BREAKOUT" if direct and setup.source == "15M_SWING"
            else "DIRECT_BASE_BREAKOUT" if direct
            else "MAJOR_BREAKOUT_RETEST" if setup.source == "15M_SWING"
            else "BASE_BREAKOUT_RETEST"
        )
        min_room = (
            getattr(c, "dual_direct_breakout_min_room_r", 1.30)
            if direct else getattr(c, "dual_momentum_min_room_r", 1.20)
        )
        candidate = self._finalize_candidate(
            direction=direction,
            setup_type=MOMENTUM if direct else MOMENTUM_RETEST,
            score=score,
            threshold=threshold,
            trigger=trigger,
            price=price,
            raw_stop=raw_stop,
            atr_value=atr_value,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            bar_ts=bar_ts,
            components={
                **components,
                "breakout_source": setup.source,
                "compressed": setup.compressed,
                "local_direction_score": snapshot["direction_score"],
                "local_opposite_score": snapshot["opposite_score"],
                "local_direction_edge": snapshot["direction_edge"],
                "level_extension_atr": round(level_extension, 3),
                "ema_extension_atr": round(ema_extension, 3),
            },
            minimum_room_override=min_room,
            max_fee_drag_override=(
                getattr(c, "dual_direct_max_fee_drag_r", 0.28)
                if direct else None
            ),
        )
        if candidate is not None:
            state.breakout = None
        return candidate

    def _reentry_allowed(
        self,
        state: _SetupState,
        direction: str,
        df_15m: pd.DataFrame,
        context: dict,
    ) -> tuple[bool, str]:
        lock = state.reentry_lock
        if (
            lock is None
            or lock.direction != direction
            or not getattr(self.cfg, "dual_reentry_requires_new_structure", True)
        ):
            return True, ""

        # Evaluate only structure created AFTER the lock. This avoids reusing an
        # old BOS and is materially faster than rescanning every historical
        # swing on each 5M loop.
        lock_ts = pd.Timestamp(lock.set_ts)
        post = df_15m.loc[df_15m.index > lock_ts]
        if len(post) < 4:
            return False, f"same-side re-entry locked after {lock.exit_reason}; waiting for post-SL 15M structure"
        atr_value = ind.safe_float(ind.atr(df_15m, 14).iloc[-1])
        row = post.iloc[-1]
        prior = post.iloc[:-1].tail(max(3, min(12, len(post) - 1)))
        if prior.empty or atr_value <= 0:
            return False, f"same-side re-entry locked after {lock.exit_reason}; waiting for post-SL 15M structure"
        body = abs(float(row["close"]) - float(row["open"]))
        if direction == LONG:
            range_break = float(row["close"]) > float(prior["high"].max())
            hold = float(row["close"]) > context["ema_fast"]
        else:
            range_break = float(row["close"]) < float(prior["low"].min())
            hold = float(row["close"]) < context["ema_fast"]
        fresh_structure = bool(
            range_break
            and hold
            and body >= 0.15 * atr_value
            and context["group_count"] >= 3
            and not context["opposite_shift"]
        )
        if fresh_structure:
            state.reentry_lock = None
            state.sl_streak_direction = ""
            state.sl_streak_count = 0
            state.last_sl_ts = None
            return True, "fresh post-SL 15M range break confirmed"
        return False, f"same-side re-entry locked after {lock.exit_reason}; waiting for post-SL 15M range break"

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

        reentry_ok, reentry_reason = self._reentry_allowed(
            state, direction, df_15m, context
        )
        if not reentry_ok:
            state.pullback = None
            state.breakout = None
            self._persist_state()
            return EntryResult(
                NONE,
                False,
                reentry_reason,
                **base,
            )

        support, resistance = self._nearest_levels(
            snapshot["price"], df_5m, df_15m, df_1h, df_4h
        )
        pullback = self._pullback_candidate(
            df_5m, direction, context, snapshot, state,
            support, resistance, regime, bias, symbol,
        )
        micro_pullback = self._micro_pullback_candidate(
            df_5m, direction, context, snapshot,
            support, resistance, regime, bias,
        )
        ema_reclaim = self._ema_reclaim_candidate(
            df_5m, direction, context, snapshot,
            support, resistance, regime, bias, symbol,
        )
        continuation = self._continuation_candidate(
            df_5m, direction, context, snapshot,
            support, resistance, regime, bias,
        )
        momentum = self._momentum_candidate(
            df_5m, direction, context, snapshot, state,
            support, resistance, regime, bias, symbol,
        )
        candidates = [
            x for x in (pullback, micro_pullback, ema_reclaim, continuation, momentum)
            if x is not None
        ]
        # Prevent one permissive trigger family from repeatedly firing inside
        # the same short trend leg. Other engines remain eligible, so frequency
        # is preserved through setup diversity rather than same-engine churn.
        if candidates and isinstance(state.last_entry_key, tuple) and len(state.last_entry_key) >= 3:
            try:
                last_ts = pd.Timestamp(state.last_entry_key[0])
                last_setup = str(state.last_entry_key[2])
                cooldown_bars = (
                    getattr(self.cfg, "dual_same_engine_cooldown_bars_precision", 36)
                    if self._is_precision_symbol(symbol)
                    else getattr(self.cfg, "dual_same_engine_cooldown_bars_high_beta", 12)
                )
                age_bars = self._age_bars(last_ts, pd.Timestamp(df_5m.index[-1]), 5)
                if age_bars < cooldown_bars:
                    candidates = [x for x in candidates if x.setup_type != last_setup]
            except (TypeError, ValueError, IndexError):
                pass
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
                f"no valid 5M structure trigger (armed={armed_text}) | "
                f"15M={context['structure']} groups={context['group_count']}/4 "
                f"5M={snapshot['structure']} local={snapshot['direction_score']:.0f}/"
                f"{snapshot['opposite_score']:.0f} edge={snapshot['direction_edge']:+.0f} "
                f"ADX={snapshot['adx']:.1f} CHOP={snapshot['chop']:.1f}",
                **base,
            )

        candidates.sort(key=lambda x: x.edge, reverse=True)
        selected = candidates[0]
        if len(candidates) > 1 and abs(candidates[0].edge - candidates[1].edge) <= 5:
            priority = {
                FAST_PULLBACK: 0,
                MICRO_PULLBACK: 1,
                EMA_RECLAIM: 2,
                TREND_CONTINUATION: 3,
                MOMENTUM_RETEST: 4,
                MOMENTUM: 5,
            }
            near = [x for x in candidates if candidates[0].edge - x.edge <= 5]
            selected = min(near, key=lambda x: priority.get(x.setup_type, 99))
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
