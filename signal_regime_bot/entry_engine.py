"""DUALCORE V3.0 Expert Multi-Mode entry engine.

The engine may execute from either a closed 15M bar or a closed 5M bar. It
routes setup families by regime instead of forcing every market through the
same trigger:

Trend / early trend
    15M EMA cross, 5M EMA cross, structure pullback, SMC zone rejection,
    breakout-retest/direct breakout, liquidity sweep, momentum continuation.
Range
    SMC zone rejection, liquidity sweep and range-edge reversal only.
Compression
    breakout/retest plus qualified SMC edge reactions.

EMA crosses are timing signals, not standalone permission. Every candidate is
checked against HTF bias, local directional edge, SMC location, extension,
opposing structure room, fee drag, stop distance and actual R:R.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from regime_engine import (
    STRONG_BULL, EARLY_BULL, STRONG_BEAR, EARLY_BEAR,
    RANGE, COMPRESSION, BULL_LABELS, BEAR_LABELS,
)

LONG = "LONG"
SHORT = "SHORT"
BOTH = "BOTH"
NONE = "NONE"

EMA_CROSS_REVERSAL = "EMA_CROSS_REVERSAL"
PRICE_OPEN_BEYOND_EMA = "PRICE_OPEN_BEYOND_EMA"

EMA_CROSS_15M = "EMA_CROSS_15M"
EMA_CROSS_5M = "EMA_CROSS_5M"
STRUCTURE_PULLBACK = "STRUCTURE_PULLBACK"
SMC_ZONE_REJECTION = "SMC_ZONE_REJECTION"
BREAKOUT_RETEST = "BREAKOUT_RETEST"
DIRECT_BREAKOUT = "DIRECT_BREAKOUT"
LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
MOMENTUM_CONTINUATION = "MOMENTUM_CONTINUATION"
RANGE_REVERSAL = "RANGE_REVERSAL"

# Legacy aliases retained for stats/Telegram compatibility.
FAST_PULLBACK = STRUCTURE_PULLBACK
MICRO_PULLBACK = STRUCTURE_PULLBACK
EMA_RECLAIM = EMA_CROSS_5M
TREND_CONTINUATION = MOMENTUM_CONTINUATION
MOMENTUM = DIRECT_BREAKOUT
MOMENTUM_RETEST = BREAKOUT_RETEST


@dataclass
class EntryResult:
    direction: str
    allow_entry: bool
    reason: str = ""
    price: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    macd_hist: float = 0.0
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
class _Zone:
    kind: str  # DEMAND | SUPPLY
    low: float
    high: float
    timeframe: str
    created_ts: pd.Timestamp
    strength: float
    source: str

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass
class _State:
    last_processed_5m: Optional[pd.Timestamp] = None
    last_entry_key: object = None
    last_entry_ts: Optional[pd.Timestamp] = None
    last_setup: str = ""
    sl_direction: str = ""
    sl_count: int = 0
    last_sl_ts: Optional[pd.Timestamp] = None
    reentry_lock_direction: str = ""
    reentry_lock_ts: Optional[pd.Timestamp] = None
    leg_direction: str = ""
    leg_entries: int = 0
    leg_anchor_ts: Optional[pd.Timestamp] = None


@dataclass
class _Candidate:
    direction: str
    setup_type: str
    timeframe: str
    trigger: str
    score: float
    threshold: float
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
        self._state: dict[str, _State] = {}
        self._state_path = os.path.join(getattr(cfg, "state_dir", "state"), "entry_engine_state.json")
        self._load_state()

    @staticmethod
    def _ts(value) -> Optional[pd.Timestamp]:
        if value in (None, ""):
            return None
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _restore_key(value):
        """JSON stores tuples as lists; restore immutable signal keys after restart."""
        if isinstance(value, list):
            return tuple(EntryEngine._restore_key(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((k, EntryEngine._restore_key(v)) for k, v in value.items()))
        return value

    def _load_state(self) -> None:
        try:
            raw = json.loads(open(self._state_path, "r", encoding="utf-8").read())
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return
        for symbol, d in raw.get("symbols", {}).items():
            self._state[symbol] = _State(
                last_processed_5m=self._ts(d.get("last_processed_5m")),
                last_entry_key=self._restore_key(d.get("last_entry_key")),
                last_entry_ts=self._ts(d.get("last_entry_ts")),
                last_setup=str(d.get("last_setup", "")),
                sl_direction=str(d.get("sl_direction", "")),
                sl_count=int(d.get("sl_count", 0)),
                last_sl_ts=self._ts(d.get("last_sl_ts")),
                reentry_lock_direction=str(d.get("reentry_lock_direction", "")),
                reentry_lock_ts=self._ts(d.get("reentry_lock_ts")),
                leg_direction=str(d.get("leg_direction", "")),
                leg_entries=int(d.get("leg_entries", 0)),
                leg_anchor_ts=self._ts(d.get("leg_anchor_ts")),
            )

    def _persist_state(self) -> None:
        directory = os.path.dirname(self._state_path) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {"version": 6, "symbols": {}}
        for symbol, s in self._state.items():
            payload["symbols"][symbol] = {
                "last_processed_5m": s.last_processed_5m.isoformat() if s.last_processed_5m is not None else None,
                "last_entry_key": s.last_entry_key,
                "last_entry_ts": s.last_entry_ts.isoformat() if s.last_entry_ts is not None else None,
                "last_setup": s.last_setup,
                "sl_direction": s.sl_direction,
                "sl_count": s.sl_count,
                "last_sl_ts": s.last_sl_ts.isoformat() if s.last_sl_ts is not None else None,
                "reentry_lock_direction": s.reentry_lock_direction,
                "reentry_lock_ts": s.reentry_lock_ts.isoformat() if s.reentry_lock_ts is not None else None,
                "leg_direction": s.leg_direction,
                "leg_entries": s.leg_entries,
                "leg_anchor_ts": s.leg_anchor_ts.isoformat() if s.leg_anchor_ts is not None else None,
            }
        tmp = self._state_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), default=str)
            os.replace(tmp, self._state_path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _get_state(self, symbol: str) -> _State:
        return self._state.setdefault(symbol, _State())

    def reset_symbol(self, symbol: str) -> None:
        self._state.pop(symbol, None)
        self._persist_state()

    def observe(self, df_30m, df_15m, df_5m, symbol: str) -> None:
        self._get_state(symbol)

    def on_position_closed(
        self,
        symbol: str,
        direction: Optional[str] = None,
        exit_reason: str = "",
        trade_pnl: float = 0.0,
        closed_at: Optional[pd.Timestamp] = None,
    ) -> None:
        s = self._get_state(symbol)
        side = str(direction or "").upper()
        now = pd.Timestamp(closed_at) if closed_at is not None else pd.Timestamp.utcnow()
        is_full_sl = side in (LONG, SHORT) and trade_pnl < 0 and str(exit_reason).upper() in {"SL_HIT", "FALSE_BREAKOUT"}
        if is_full_sl:
            window = pd.Timedelta(hours=max(1, int(self.cfg.expert_reentry_lock_hours)))
            same_window = s.last_sl_ts is not None and now - s.last_sl_ts <= window and s.sl_direction == side
            s.sl_count = s.sl_count + 1 if same_window else 1
            s.sl_direction = side
            s.last_sl_ts = now
            if s.sl_count >= max(2, int(getattr(self.cfg, "dual_reentry_lock_after_sl_count", 3))):
                s.reentry_lock_direction = side
                s.reentry_lock_ts = now
        elif trade_pnl > 0:
            s.sl_direction = ""
            s.sl_count = 0
            s.last_sl_ts = None
            s.reentry_lock_direction = ""
            s.reentry_lock_ts = None
        self._persist_state()

    @staticmethod
    def _fresh_cross(fast: pd.Series, slow: pd.Series, direction: str, max_age: int = 0) -> tuple[bool, int]:
        if len(fast) < max_age + 3:
            return False, 999
        for age in range(max_age + 1):
            i = -1 - age
            p = i - 1
            if direction == LONG and fast.iloc[i] > slow.iloc[i] and fast.iloc[p] <= slow.iloc[p]:
                return True, age
            if direction == SHORT and fast.iloc[i] < slow.iloc[i] and fast.iloc[p] >= slow.iloc[p]:
                return True, age
        return False, 999

    def _snapshot(self, df: pd.DataFrame, direction: str) -> dict:
        close = df["close"].astype(float)
        price = ind.safe_float(close.iloc[-1])
        ema8_s = ind.ema(close, 8)
        ema13_s = ind.ema(close, 13)
        ema20_s = ind.ema(close, 20)
        ema50_s = ind.ema(close, 50)
        atr_s = ind.atr(df, 14)
        atr_v = max(ind.safe_float(atr_s.iloc[-1]), 1e-12)
        adx_s, plus_s, minus_s = ind.adx(df, 14)
        adx_v = ind.safe_float(adx_s.iloc[-1])
        plus = ind.safe_float(plus_s.iloc[-1])
        minus = ind.safe_float(minus_s.iloc[-1])
        chop = ind.safe_float(ind.choppiness_index(df, 14).iloc[-1], 100.0)
        rsi14 = ind.safe_float(ind.rsi(close, 14).iloc[-1], 50.0)
        roc5 = ind.safe_float(ind.roc(close, 5).iloc[-1])
        _, _, hist_s = ind.macd(close)
        hist = ind.safe_float(hist_s.iloc[-1])
        structure = ind.market_structure(df["high"], df["low"], 3, 3)
        bull_bos, bull_level = ind.latest_bos(df, LONG, 3, 3, 0.12)
        bear_bos, bear_level = ind.latest_bos(df, SHORT, 3, 3, 0.12)
        swing_high, swing_low = ind.recent_swing_levels(df["high"], df["low"], 3, 3)
        candle = ind.candle_metrics(df, atr_v)
        vwap_v = ind.safe_float(ind.vwap(df, min(48, max(2, len(df)-1))).iloc[-1], price)
        aligned = (
            price > ema20_s.iloc[-1] and ema8_s.iloc[-1] > ema13_s.iloc[-1]
            if direction == LONG else
            price < ema20_s.iloc[-1] and ema8_s.iloc[-1] < ema13_s.iloc[-1]
        )
        strong_aligned = (
            ema8_s.iloc[-1] > ema13_s.iloc[-1] > ema20_s.iloc[-1] and price > ema20_s.iloc[-1]
            if direction == LONG else
            ema8_s.iloc[-1] < ema13_s.iloc[-1] < ema20_s.iloc[-1] and price < ema20_s.iloc[-1]
        )
        structure_aligned = structure == ("HH_HL" if direction == LONG else "LH_LL")
        bos = bull_bos if direction == LONG else bear_bos
        opposite_bos = bear_bos if direction == LONG else bull_bos
        directional_candle = candle.bullish if direction == LONG else candle.bearish
        close_quality = candle.bull_close_quality if direction == LONG else candle.bear_close_quality
        momentum_good = (hist > 0 and roc5 > 0) if direction == LONG else (hist < 0 and roc5 < 0)
        di_good = plus >= minus - self.cfg.dual_di_tolerance if direction == LONG else minus >= plus - self.cfg.dual_di_tolerance

        own = 0.0
        opp = 0.0
        own += 20 if structure_aligned else 10 if bos else 0
        opp += 20 if structure == ("LH_LL" if direction == LONG else "HH_HL") else 10 if opposite_bos else 0
        own += 18 if strong_aligned else 12 if aligned else 0
        opposite_aligned = (
            price < ema20_s.iloc[-1] and ema8_s.iloc[-1] < ema13_s.iloc[-1]
            if direction == LONG else
            price > ema20_s.iloc[-1] and ema8_s.iloc[-1] > ema13_s.iloc[-1]
        )
        opp += 18 if opposite_aligned else 0
        own += 12 if momentum_good else 5 if (hist > 0 if direction == LONG else hist < 0) else 0
        opp += 12 if ((hist < 0 and roc5 < 0) if direction == LONG else (hist > 0 and roc5 > 0)) else 0
        own += 10 if di_good else 0
        opp += 10 if (minus > plus + 2 if direction == LONG else plus > minus + 2) else 0
        own += 8 if directional_candle and close_quality >= 0.60 else 3 if directional_candle else 0
        opp += 8 if ((candle.bearish and candle.bear_close_quality >= 0.60) if direction == LONG else (candle.bullish and candle.bull_close_quality >= 0.60)) else 0
        own += 5 if (price >= vwap_v if direction == LONG else price <= vwap_v) else 0
        own += 5 if adx_v >= 14 else 2 if adx_v >= 10 else 0

        return {
            "price": price,
            "ema8": ind.safe_float(ema8_s.iloc[-1]),
            "ema13": ind.safe_float(ema13_s.iloc[-1]),
            "ema20": ind.safe_float(ema20_s.iloc[-1]),
            "ema50": ind.safe_float(ema50_s.iloc[-1]),
            "ema8_s": ema8_s,
            "ema13_s": ema13_s,
            "ema20_s": ema20_s,
            "atr": atr_v,
            "adx": adx_v,
            "plus_di": plus,
            "minus_di": minus,
            "di_spread": (plus - minus) if direction == LONG else (minus - plus),
            "chop": chop,
            "rsi": rsi14,
            "roc": roc5,
            "hist": hist,
            "structure": structure,
            "bos": bool(bos),
            "opposite_bos": bool(opposite_bos),
            "bos_level": bull_level if direction == LONG else bear_level,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "candle": candle,
            "aligned": bool(aligned),
            "strong_aligned": bool(strong_aligned),
            "structure_aligned": bool(structure_aligned),
            "momentum_good": bool(momentum_good),
            "directional_candle": bool(directional_candle),
            "close_quality": close_quality,
            "direction_score": round(own, 1),
            "opposite_score": round(opp, 1),
            "edge": round(own - opp, 1),
        }

    def _detect_zones(self, df: Optional[pd.DataFrame], timeframe: str, lookback: int) -> list[_Zone]:
        if df is None or len(df) < 25:
            return []
        work = df.iloc[-max(30, lookback):].copy()
        atr_s = ind.atr(work, 14)
        zones: list[_Zone] = []
        start = max(5, len(work) - lookback)
        for i in range(start, len(work) - 1):
            j = i + 1
            atr_v = max(ind.safe_float(atr_s.iloc[j]), 1e-12)
            o, h, l, cl = map(float, (work["open"].iloc[i], work["high"].iloc[i], work["low"].iloc[i], work["close"].iloc[i]))
            no, nh, nl, nc = map(float, (work["open"].iloc[j], work["high"].iloc[j], work["low"].iloc[j], work["close"].iloc[j]))
            prev_high = float(work["high"].iloc[max(0, i-5):i+1].max())
            prev_low = float(work["low"].iloc[max(0, i-5):i+1].min())
            next_body_atr = abs(nc-no) / atr_v
            if nc > prev_high and nc > no and next_body_atr >= 0.38:
                low, high = l, max(o, cl)
                if high > low:
                    later = work.iloc[j+1:]
                    invalid = len(later) and float(later["close"].min()) < low
                    if not invalid:
                        zones.append(_Zone("DEMAND", low, high, timeframe, pd.Timestamp(work.index[i]), min(2.0, 0.7 + next_body_atr), "ORDER_BLOCK"))
            if nc < prev_low and nc < no and next_body_atr >= 0.38:
                low, high = min(o, cl), h
                if high > low:
                    later = work.iloc[j+1:]
                    invalid = len(later) and float(later["close"].max()) > high
                    if not invalid:
                        zones.append(_Zone("SUPPLY", low, high, timeframe, pd.Timestamp(work.index[i]), min(2.0, 0.7 + next_body_atr), "ORDER_BLOCK"))
            # Three-candle fair value gap, retained while not fully closed through.
            if i >= 2:
                h2 = float(work["high"].iloc[i-2]); l2 = float(work["low"].iloc[i-2])
                if l > h2:
                    later = work.iloc[i+1:]
                    invalid = len(later) and float(later["close"].min()) < h2
                    if not invalid:
                        zones.append(_Zone("DEMAND", h2, l, timeframe, pd.Timestamp(work.index[i]), 0.8, "FVG"))
                if h < l2:
                    later = work.iloc[i+1:]
                    invalid = len(later) and float(later["close"].max()) > l2
                    if not invalid:
                        zones.append(_Zone("SUPPLY", h, l2, timeframe, pd.Timestamp(work.index[i]), 0.8, "FVG"))
        # newest and strongest first; deduplicate near-identical zones
        zones.sort(key=lambda z: (z.created_ts, z.strength), reverse=True)
        out: list[_Zone] = []
        for z in zones:
            if not any(abs(z.midpoint - x.midpoint) <= max(z.high-z.low, x.high-x.low)*0.35 for x in out):
                out.append(z)
            if len(out) >= 16:
                break
        return out

    def _zone_context(self, direction: str, price: float, atr_v: float, frames: list[tuple[Optional[pd.DataFrame], str, int]]) -> dict:
        zones: list[_Zone] = []
        for frame, tf, lookback in frames:
            zones.extend(self._detect_zones(frame, tf, lookback))
        own_kind = "DEMAND" if direction == LONG else "SUPPLY"
        opp_kind = "SUPPLY" if direction == LONG else "DEMAND"
        own = [z for z in zones if z.kind == own_kind]
        opp = [z for z in zones if z.kind == opp_kind]

        def dist(z: _Zone) -> float:
            if z.low <= price <= z.high:
                return 0.0
            return min(abs(price-z.low), abs(price-z.high))

        own_zone = min(own, key=dist) if own else None
        # Opposing zone must be in trade direction to define room.
        if direction == LONG:
            forward = [z for z in opp if z.low > price]
        else:
            forward = [z for z in opp if z.high < price]
        opp_zone = min(forward, key=dist) if forward else None
        proximity = self.cfg.expert_zone_touch_atr * atr_v
        near_own = bool(own_zone and dist(own_zone) <= proximity)
        in_own = bool(own_zone and own_zone.low - proximity <= price <= own_zone.high + proximity)
        return {"zones": zones, "own": own_zone, "opposing": opp_zone, "near_own": near_own, "in_own": in_own, "own_distance_atr": dist(own_zone)/atr_v if own_zone else 99.0}

    def _nearest_structure(self, price: float, frames: list[Optional[pd.DataFrame]]) -> tuple[Optional[float], Optional[float]]:
        supports: list[float] = []
        resistances: list[float] = []
        for frame in frames:
            if frame is None or len(frame) < 12:
                continue
            sup, res = ind.nearest_confirmed_levels(frame, price, 3, 3)
            if sup is not None:
                supports.append(float(sup))
            if res is not None:
                resistances.append(float(res))
        return (max(supports) if supports else None, min(resistances) if resistances else None)

    def _threshold(self, base: float, regime, bias, direction: str) -> float:
        threshold = float(base)
        label = getattr(regime, "label", "")
        if label in (STRONG_BULL, STRONG_BEAR):
            threshold -= self.cfg.expert_strong_trend_discount
        relevant = getattr(bias, "bull_score", 50.0) if direction == LONG else getattr(bias, "bear_score", 50.0)
        edge = getattr(bias, "directional_edge", 0.0) * (1 if direction == LONG else -1)
        if relevant < self.cfg.expert_bias_score_min or edge < self.cfg.expert_bias_edge_min:
            threshold += self.cfg.expert_weak_context_add
        return max(58.0, min(78.0, threshold))

    def _htf_points(self, direction: str, regime, bias) -> tuple[float, dict]:
        label = getattr(regime, "label", "")
        trend_match = label in BULL_LABELS if direction == LONG else label in BEAR_LABELS
        if label in (STRONG_BULL, STRONG_BEAR) and trend_match:
            regime_pts = 12.0
        elif label in (EARLY_BULL, EARLY_BEAR) and trend_match:
            regime_pts = 9.0
        elif label in (RANGE, COMPRESSION):
            regime_pts = 6.0
        else:
            regime_pts = 2.0
        relevant = getattr(bias, "bull_score", 50.0) if direction == LONG else getattr(bias, "bear_score", 50.0)
        edge = getattr(bias, "directional_edge", 0.0) * (1 if direction == LONG else -1)
        bias_pts = 8.0 if relevant >= 60 and edge >= 8 else 6.0 if relevant >= 52 and edge >= 3 else 3.0
        return regime_pts + bias_pts, {"regime_points": regime_pts, "bias_points": bias_pts, "bias_relevant": relevant, "bias_edge_side": edge}

    def _finalize(
        self,
        *, direction: str, setup_type: str, timeframe: str, trigger: str,
        score: float, threshold: float, price: float, atr_v: float,
        invalidation: float, desired_r: float, opposing_level: Optional[float],
        bar_ts: pd.Timestamp, components: dict, min_room_r: Optional[float] = None,
    ) -> Optional[_Candidate]:
        c = self.cfg
        cost_distance = price * (2*c.fee_rate + c.expected_slippage_pct)
        min_distance = max(atr_v * 0.65, price * c.sl_min_pct, cost_distance * c.stop_fee_floor_mult)
        max_distance = max(atr_v * c.dual_max_stop_atr, min_distance)
        max_distance = min(max_distance, price * c.sl_max_pct) if price*c.sl_max_pct >= min_distance else min_distance
        raw_distance = price - invalidation if direction == LONG else invalidation - price
        if raw_distance <= 0:
            return None
        # A distant structural invalidation does not suppress the signal. The
        # executable stop is capped at the configured ATR/percent maximum and
        # position size is reduced from that final distance.
        distance = min(max(raw_distance, min_distance), max_distance)
        stop = price - distance if direction == LONG else price + distance
        fee_drag_r = cost_distance / max(distance, 1e-12)
        if fee_drag_r > c.max_fee_drag_r + 1e-9:
            return None

        room_distance = math.inf
        if opposing_level is not None:
            room_distance = opposing_level - price if direction == LONG else price - opposing_level
            if room_distance <= 0:
                return None
        room_r = room_distance / distance if math.isfinite(room_distance) else 99.0
        required_room = c.expert_min_room_r if min_room_r is None else min_room_r
        if room_r < required_room:
            return None

        target_distance = desired_r * distance
        if math.isfinite(room_distance):
            target_distance = min(target_distance, max(0.0, room_distance - c.dual_target_buffer_atr*atr_v))
        rr = target_distance / max(distance, 1e-12)
        if rr < c.minimum_actual_rr:
            return None
        target = price + target_distance if direction == LONG else price - target_distance
        components = dict(components)
        components.update({
            "timeframe": timeframe,
            "fee_drag_r": round(fee_drag_r, 3),
            "stop_pct": round(distance/price*100, 3),
            "room_r": round(room_r, 2),
            "desired_r": desired_r,
        })
        key = (str(bar_ts), direction, setup_type, timeframe, trigger)
        return _Candidate(direction, setup_type, timeframe, trigger, round(score,1), round(threshold,1), price, stop, target, rr, room_r, invalidation, components, key)

    @staticmethod
    def _directional_candle(snap: dict, min_body: float = 0.12, min_close: float = 0.58) -> bool:
        c = snap["candle"]
        return snap["directional_candle"] and c.body_atr >= min_body and snap["close_quality"] >= min_close

    def _cross_candidate(self, frame: pd.DataFrame, tf: str, direction: str, snap: dict, context: dict, zone: dict, regime, bias, opposing: Optional[float]) -> Optional[_Candidate]:
        fresh, age = self._fresh_cross(snap["ema8_s"], snap["ema13_s"], direction, 0)
        if not fresh:
            return None
        label = getattr(regime, "label", "")
        if label == RANGE:
            return None
        if snap["edge"] < self.cfg.expert_min_local_edge:
            return None
        # A cross in low-ADX or opposing DMI conditions is usually chop rather
        # than a new impulse. HTF bias cannot compensate for this execution gate.
        if snap["adx"] < self.cfg.expert_ema_cross_adx_min:
            return None
        if snap["di_spread"] < self.cfg.expert_ema_cross_di_spread_min:
            return None
        max_ext = self.cfg.expert_max_extension_atr_15m if tf == "15M" else self.cfg.expert_max_extension_atr_5m
        extension = abs(snap["price"] - snap["ema20"]) / snap["atr"]
        if extension > max_ext:
            return None
        if tf == "5M" and not (context["aligned"] or context["structure_aligned"] or context["bos"]):
            return None
        if not (snap["price"] > snap["ema20"] if direction == LONG else snap["price"] < snap["ema20"]):
            if not snap["bos"]:
                return None

        htf, htf_detail = self._htf_points(direction, regime, bias)
        structure_pts = 17 if snap["structure_aligned"] else 12 if snap["bos"] else 6 if context["structure_aligned"] else 2
        location_pts = 15 if zone["near_own"] else 11 if abs(snap["price"]-snap["ema20"])/snap["atr"] <= 0.45 else 6
        trigger_pts = 18 if self._directional_candle(snap, 0.13, 0.60) else 13 if snap["directional_candle"] else 9
        momentum_pts = 10 if snap["momentum_good"] else 6 if snap["aligned"] else 2
        quality_pts = 8 if snap["candle"].volume_ratio >= 1.0 else 5
        score = htf + structure_pts + location_pts + trigger_pts + momentum_pts + quality_pts + 8
        setup = EMA_CROSS_15M if tf == "15M" else EMA_CROSS_5M
        base_thr = self.cfg.expert_thr_15m_ema_cross if tf == "15M" else self.cfg.expert_thr_5m_ema_cross
        threshold = self._threshold(base_thr, regime, bias, direction)
        if score < threshold:
            return None
        lows = frame["low"].iloc[-6:]
        highs = frame["high"].iloc[-6:]
        invalidation = float(lows.min()) if direction == LONG else float(highs.max())
        if zone["own"] is not None and zone["near_own"]:
            invalidation = min(invalidation, zone["own"].low) if direction == LONG else max(invalidation, zone["own"].high)
        desired = self.cfg.expert_tp2_ema_cross_r
        return self._finalize(
            direction=direction, setup_type=setup, timeframe=tf, trigger=f"EMA8_13_CROSS_{tf}",
            score=score, threshold=threshold, price=snap["price"], atr_v=snap["atr"], invalidation=invalidation,
            desired_r=desired, opposing_level=opposing, bar_ts=pd.Timestamp(frame.index[-1]),
            components={"htf": htf_detail, "structure": structure_pts, "location": location_pts, "trigger": trigger_pts, "momentum": momentum_pts, "quality": quality_pts, "local_edge": snap["edge"], "extension_atr": round(extension,2)},
        )

    def _structure_pullback(self, df5: pd.DataFrame, direction: str, s5: dict, s15: dict, zone: dict, regime, bias, opposing: Optional[float]) -> Optional[_Candidate]:
        if getattr(regime, "label", "") in (RANGE, COMPRESSION):
            return None
        if not (s15["structure_aligned"] or s15["bos"] or s15["aligned"]):
            return None
        recent = df5.iloc[-6:]
        touched_ema = (float(recent["low"].min()) <= s5["ema20"] + 0.15*s5["atr"]) if direction == LONG else (float(recent["high"].max()) >= s5["ema20"] - 0.15*s5["atr"])
        touched_location = touched_ema or zone["near_own"]
        if not touched_location:
            return None
        prior_level = float(df5["high"].iloc[-2]) if direction == LONG else float(df5["low"].iloc[-2])
        micro_break = s5["price"] > prior_level if direction == LONG else s5["price"] < prior_level
        reclaim = (df5["close"].iloc[-2] <= s5["ema8_s"].iloc[-2] and s5["price"] > s5["ema8"]) if direction == LONG else (df5["close"].iloc[-2] >= s5["ema8_s"].iloc[-2] and s5["price"] < s5["ema8"])
        if not (micro_break or reclaim or s5["bos"]):
            return None
        if s5["edge"] < self.cfg.expert_min_local_edge:
            return None
        # Pullbacks need enough trend pressure to resume, but an extremely
        # one-sided 5M score often means the reclaim is already late/chased.
        if s5["adx"] < self.cfg.expert_pullback_adx_min:
            return None
        if s5["di_spread"] < self.cfg.expert_pullback_di_spread_min:
            return None
        if not (self.cfg.expert_pullback_edge_min <= s5["edge"] <= self.cfg.expert_pullback_edge_max):
            return None
        extension = abs(s5["price"]-s5["ema20"])/s5["atr"]
        if extension > 0.95:
            return None
        htf, htf_detail = self._htf_points(direction, regime, bias)
        structure_pts = 19 if s15["structure_aligned"] and (micro_break or s5["bos"]) else 15
        location_pts = 19 if zone["near_own"] else 14
        trigger_pts = 18 if self._directional_candle(s5,0.12,0.58) and micro_break else 14
        momentum_pts = 9 if s5["momentum_good"] else 5
        quality_pts = 8 if s5["candle"].volume_ratio >= 0.9 else 5
        score = htf + structure_pts + location_pts + trigger_pts + momentum_pts + quality_pts + 8
        threshold = self._threshold(self.cfg.expert_thr_structure_pullback, regime, bias, direction)
        if score < threshold:
            return None
        invalidation = float(recent["low"].min()) if direction == LONG else float(recent["high"].max())
        if zone["own"] is not None and zone["near_own"]:
            invalidation = min(invalidation, zone["own"].low) if direction == LONG else max(invalidation, zone["own"].high)
        return self._finalize(
            direction=direction, setup_type=STRUCTURE_PULLBACK, timeframe="5M", trigger="PULLBACK_RECLAIM_MICRO_BOS",
            score=score, threshold=threshold, price=s5["price"], atr_v=s5["atr"], invalidation=invalidation,
            desired_r=self.cfg.expert_tp2_pullback_r, opposing_level=opposing, bar_ts=pd.Timestamp(df5.index[-1]),
            components={"htf":htf_detail,"structure":structure_pts,"location":location_pts,"trigger":trigger_pts,"momentum":momentum_pts,"quality":quality_pts,"local_edge":s5["edge"],"extension_atr":round(extension,2)},
        )

    def _smc_rejection(self, df5: pd.DataFrame, direction: str, s5: dict, s15: dict, zone: dict, regime, bias, opposing: Optional[float]) -> Optional[_Candidate]:
        z = zone["own"]
        if z is None:
            return None
        recent = df5.iloc[-4:]
        touched = float(recent["low"].min()) <= z.high + self.cfg.expert_zone_touch_atr*s5["atr"] and float(recent["high"].max()) >= z.low - self.cfg.expert_zone_touch_atr*s5["atr"]
        if not touched:
            return None
        if z.source == "FVG" and (z.high - z.low) / max(s5["price"], 1e-12) > self.cfg.expert_fvg_max_width_pct:
            return None
        row = df5.iloc[-1]
        wick_reject = (
            float(row["low"]) <= z.high and float(row["close"]) > z.high and s5["candle"].lower_wick >= s5["candle"].body*0.8
            if direction == LONG else
            float(row["high"]) >= z.low and float(row["close"]) < z.low and s5["candle"].upper_wick >= s5["candle"].body*0.8
        )
        micro_break = s5["price"] > float(df5["high"].iloc[-2]) if direction == LONG else s5["price"] < float(df5["low"].iloc[-2])
        if not (wick_reject or micro_break or s5["bos"]):
            return None
        label = getattr(regime,"label","")
        # Standalone zone reactions are disabled in range/compression by
        # default. Those regimes must use the stricter sweep-at-zone route.
        if label in (RANGE, COMPRESSION) and not self.cfg.expert_allow_range_trades:
            return None
        edge_min = (self.cfg.expert_smc_edge_min_early
                    if label in (EARLY_BULL, EARLY_BEAR)
                    else self.cfg.expert_smc_edge_min_strong)
        if label not in (RANGE, COMPRESSION) and s5["edge"] < edge_min:
            return None
        if label not in (RANGE, COMPRESSION) and s5["edge"] > self.cfg.expert_smc_edge_max:
            return None
        if label in (RANGE, COMPRESSION) and s5["edge"] < 12:
            return None
        if s5["di_spread"] < self.cfg.expert_smc_di_spread_min:
            return None
        ema20_extension = abs(s5["price"] - s5["ema20"]) / s5["atr"]
        if ema20_extension > self.cfg.expert_smc_max_ema20_extension_atr:
            return None
        if direction == LONG and s5["rsi"] > self.cfg.expert_smc_rsi_long_max:
            return None
        if direction == SHORT and s5["rsi"] < self.cfg.expert_smc_rsi_short_min:
            return None
        htf, htf_detail = self._htf_points(direction, regime, bias)
        structure_pts = 18 if s15["structure_aligned"] or s15["bos"] else 10 if label in (RANGE,COMPRESSION) else 6
        if (self.cfg.expert_1h_zone_requires_15m_structure and z.timeframe == "1H"
                and not (s15["structure_aligned"] or s15["bos"])):
            return None
        location_pts = min(20.0, 15.0 + 2.5*z.strength)
        trigger_pts = 19 if wick_reject and micro_break else 16 if wick_reject or micro_break else 12
        momentum_pts = 8 if s5["momentum_good"] else 5 if label in (RANGE,COMPRESSION) else 2
        quality_pts = 8 if s5["close_quality"] >= 0.62 else 5
        # A zone touch without a decisive close is only a reaction, not an entry.
        # Keep the zone for the next candle rather than paying fees on weak closes.
        if quality_pts < 8:
            return None
        score = htf + structure_pts + location_pts + trigger_pts + momentum_pts + quality_pts + 8
        threshold = self._threshold(self.cfg.expert_thr_smc_zone_rejection, regime, bias, direction)
        if score < threshold:
            return None
        invalidation = z.low - 0.08*s5["atr"] if direction == LONG else z.high + 0.08*s5["atr"]
        return self._finalize(
            direction=direction, setup_type=SMC_ZONE_REJECTION, timeframe="5M", trigger=f"{z.timeframe}_{z.source}_REJECTION",
            score=score, threshold=threshold, price=s5["price"], atr_v=s5["atr"], invalidation=invalidation,
            desired_r=self.cfg.expert_tp2_smc_r, opposing_level=opposing, bar_ts=pd.Timestamp(df5.index[-1]),
            components={"htf":htf_detail,"structure":structure_pts,"location":location_pts,"trigger":trigger_pts,"momentum":momentum_pts,"quality":quality_pts,"zone_tf":z.timeframe,"zone_source":z.source,"zone_low":z.low,"zone_high":z.high,"local_edge":s5["edge"],"di_spread":round(s5["di_spread"],1),"rsi":round(s5["rsi"],1),"ema20_extension_atr":round(ema20_extension,2)},
            min_room_r=self.cfg.expert_range_min_room_r if label in (RANGE,COMPRESSION) else None,
        )

    def _breakout_candidate(self, frame: pd.DataFrame, tf: str, direction: str, snap: dict, context: dict, regime, bias, opposing: Optional[float]) -> Optional[_Candidate]:
        label = getattr(regime,"label","")
        if label == RANGE:
            return None
        lookback = self.cfg.expert_breakout_lookback_15m if tf == "15M" else self.cfg.expert_breakout_lookback_5m
        if len(frame) < lookback + self.cfg.expert_retest_window_bars + 3:
            return None
        prev_high = float(frame["high"].iloc[-lookback-1:-1].max())
        prev_low = float(frame["low"].iloc[-lookback-1:-1].min())
        level = prev_high if direction == LONG else prev_low
        direct = snap["price"] > level if direction == LONG else snap["price"] < level
        retest = False
        break_age = None
        for age in range(1, self.cfg.expert_retest_window_bars+1):
            idx = -1-age
            before = frame.iloc[:idx]
            if len(before) < lookback:
                continue
            old_level = float(before["high"].iloc[-lookback:].max()) if direction == LONG else float(before["low"].iloc[-lookback:].min())
            broke = float(frame["close"].iloc[idx]) > old_level if direction == LONG else float(frame["close"].iloc[idx]) < old_level
            if not broke:
                continue
            row = frame.iloc[-1]
            touched = float(row["low"]) <= old_level + 0.22*snap["atr"] if direction == LONG else float(row["high"]) >= old_level - 0.22*snap["atr"]
            reclaimed = snap["price"] > old_level if direction == LONG else snap["price"] < old_level
            if touched and reclaimed:
                retest, level, break_age = True, old_level, age
                break
        if not direct and not retest:
            return None
        if retest and (break_age is None or break_age > self.cfg.expert_retest_max_age_bars):
            return None
        # Prefer retests by default. Direct displacement entries can be enabled
        # explicitly, while the same break is still remembered for a later retest.
        if direct and not retest and not self.cfg.expert_direct_breakout_enabled:
            return None
        if tf == "5M" and not (context["aligned"] or context["bos"] or label == COMPRESSION):
            return None
        if snap["edge"] < (self.cfg.expert_min_local_edge if label != COMPRESSION else 0):
            return None
        extension = abs(snap["price"]-level)/snap["atr"]
        if retest and extension > 0.50:
            return None
        if direct:
            if tf == "5M" and label in (EARLY_BULL, EARLY_BEAR):
                return None
            body_ok = snap["candle"].body_atr >= self.cfg.expert_direct_breakout_body_atr
            volume_ok = snap["candle"].volume_ratio >= self.cfg.expert_direct_breakout_volume_ratio
            close_ok = snap["close_quality"] >= 0.68
            compression = ind.compression_ratio(frame, 4, 20) <= 0.90
            if not (body_ok and close_ok and (volume_ok or compression or label in (STRONG_BULL,STRONG_BEAR))):
                return None
            if extension > 0.55:
                return None
        htf, htf_detail = self._htf_points(direction, regime, bias)
        structure_pts = 19 if tf=="15M" or context["bos"] else 15
        location_pts = 15 if retest else 10
        trigger_pts = 19 if retest and self._directional_candle(snap,0.12,0.58) else 18 if direct else 14
        momentum_pts = 10 if snap["momentum_good"] else 6
        quality_pts = 9 if snap["candle"].volume_ratio >= 1.1 else 6
        score = htf+structure_pts+location_pts+trigger_pts+momentum_pts+quality_pts+8
        setup = BREAKOUT_RETEST if retest else DIRECT_BREAKOUT
        base = self.cfg.expert_thr_breakout_retest if retest else self.cfg.expert_thr_direct_breakout
        threshold = self._threshold(base, regime, bias, direction)
        if score < threshold:
            return None
        if retest:
            recent = frame.iloc[-(self.cfg.expert_retest_window_bars+2):]
            invalidation = float(recent["low"].min()) if direction==LONG else float(recent["high"].max())
        else:
            invalidation = float(frame["low"].iloc[-1]) if direction==LONG else float(frame["high"].iloc[-1])
        return self._finalize(
            direction=direction, setup_type=setup, timeframe=tf, trigger=("BREAKOUT_RETEST" if retest else "DISPLACEMENT_BREAKOUT"),
            score=score, threshold=threshold, price=snap["price"], atr_v=snap["atr"], invalidation=invalidation,
            desired_r=self.cfg.expert_tp2_breakout_r, opposing_level=opposing, bar_ts=pd.Timestamp(frame.index[-1]),
            components={"htf":htf_detail,"structure":structure_pts,"location":location_pts,"trigger":trigger_pts,"momentum":momentum_pts,"quality":quality_pts,"break_level":level,"break_age":break_age,"extension_atr":round(extension,2),"local_edge":snap["edge"]},
        )

    def _liquidity_sweep(self, frame: pd.DataFrame, tf: str, direction: str, snap: dict, zone: dict, regime, bias, opposing: Optional[float]) -> Optional[_Candidate]:
        level = snap["swing_low"] if direction == LONG else snap["swing_high"]
        if level is None or (isinstance(level,float) and math.isnan(level)):
            level = float(frame["low"].iloc[-12:-1].min()) if direction==LONG else float(frame["high"].iloc[-12:-1].max())
        row = frame.iloc[-1]
        swept = (
            float(row["low"]) < level and snap["price"] > level and snap["candle"].lower_wick >= max(snap["candle"].body*0.7,1e-12)
            if direction == LONG else
            float(row["high"]) > level and snap["price"] < level and snap["candle"].upper_wick >= max(snap["candle"].body*0.7,1e-12)
        )
        if not swept:
            return None
        label = getattr(regime,"label","")
        # A sweep is meaningful only at mapped liquidity/location. This keeps
        # range trading active without treating every wick as institutional SMC.
        if not zone["near_own"]:
            return None
        if label not in (RANGE,COMPRESSION):
            if snap["edge"] < self.cfg.expert_min_local_edge:
                return None
            if snap["di_spread"] < 0:
                return None
        htf, htf_detail = self._htf_points(direction, regime, bias)
        structure_pts = 18 if snap["bos"] or snap["structure_aligned"] else 12
        location_pts = 18 if zone["near_own"] else 13 if label in (RANGE,COMPRESSION) else 8
        trigger_pts = 20
        momentum_pts = 8 if snap["directional_candle"] else 5
        quality_pts = 8 if snap["close_quality"] >= 0.60 else 5
        # Sweep entries must still be early. Weak closes and already-extreme
        # local scores indicate a late continuation wick, not a fresh reclaim.
        if quality_pts < 8 or not snap["directional_candle"]:
            return None
        if snap["edge"] > self.cfg.expert_sweep_edge_max:
            return None
        score=htf+structure_pts+location_pts+trigger_pts+momentum_pts+quality_pts+8
        threshold=self._threshold(self.cfg.expert_thr_liquidity_sweep,regime,bias,direction)
        if score<threshold:
            return None
        invalidation=float(row["low"])-0.05*snap["atr"] if direction==LONG else float(row["high"])+0.05*snap["atr"]
        return self._finalize(
            direction=direction,setup_type=LIQUIDITY_SWEEP,timeframe=tf,trigger="SWEEP_AND_RECLAIM",
            score=score,threshold=threshold,price=snap["price"],atr_v=snap["atr"],invalidation=invalidation,
            desired_r=self.cfg.expert_tp2_smc_r,opposing_level=opposing,bar_ts=pd.Timestamp(frame.index[-1]),
            components={"htf":htf_detail,"structure":structure_pts,"location":location_pts,"trigger":trigger_pts,"momentum":momentum_pts,"quality":quality_pts,"swept_level":level,"local_edge":snap["edge"]},
            min_room_r=self.cfg.expert_range_min_room_r if label in (RANGE,COMPRESSION) else None,
        )

    def _continuation(self, df5: pd.DataFrame, direction: str, s5: dict, s15: dict, regime, bias, opposing: Optional[float]) -> Optional[_Candidate]:
        label=getattr(regime,"label","")
        if label not in BULL_LABELS+BEAR_LABELS:
            return None
        if not s5["strong_aligned"] or not s15["structure_aligned"]:
            return None
        fresh_cross,_=self._fresh_cross(s5["ema8_s"],s5["ema13_s"],direction,1)
        if not fresh_cross:
            return None
        recent=df5.iloc[-5:-1]
        pulled=(float(recent["low"].min())<=s5["ema13"]+0.12*s5["atr"]) if direction==LONG else (float(recent["high"].max())>=s5["ema13"]-0.12*s5["atr"])
        break_prev=s5["price"]>float(df5["high"].iloc[-2]) if direction==LONG else s5["price"]<float(df5["low"].iloc[-2])
        if not (pulled and break_prev and self._directional_candle(s5,0.14,0.60)):
            return None
        if s5["edge"]<self.cfg.expert_min_local_edge:
            return None
        extension=abs(s5["price"]-s5["ema13"])/s5["atr"]
        if extension>0.80:
            return None
        htf,htf_detail=self._htf_points(direction,regime,bias)
        structure_pts=17 if s15["structure_aligned"] else 13
        location_pts=14
        trigger_pts=18
        momentum_pts=10 if s5["momentum_good"] else 6
        quality_pts=8 if s5["candle"].volume_ratio>=0.9 else 5
        score=htf+structure_pts+location_pts+trigger_pts+momentum_pts+quality_pts+8
        threshold=self._threshold(self.cfg.expert_thr_momentum_continuation,regime,bias,direction)
        if score<threshold:
            return None
        invalidation=float(recent["low"].min()) if direction==LONG else float(recent["high"].max())
        return self._finalize(
            direction=direction,setup_type=MOMENTUM_CONTINUATION,timeframe="5M",trigger="PULLBACK_THEN_PREVIOUS_BAR_BREAK",
            score=score,threshold=threshold,price=s5["price"],atr_v=s5["atr"],invalidation=invalidation,
            desired_r=self.cfg.expert_tp2_continuation_r,opposing_level=opposing,bar_ts=pd.Timestamp(df5.index[-1]),
            components={"htf":htf_detail,"structure":structure_pts,"location":location_pts,"trigger":trigger_pts,"momentum":momentum_pts,"quality":quality_pts,"extension_atr":round(extension,2),"local_edge":s5["edge"]},
        )

    def _range_reversal(self, df15: pd.DataFrame, df5: pd.DataFrame, direction: str, s5: dict, zone: dict, regime, bias, opposing: Optional[float]) -> Optional[_Candidate]:
        if getattr(regime,"label","") != RANGE or not self.cfg.expert_allow_range_trades:
            return None
        if len(df15)<24:
            return None
        range_low=float(df15["low"].iloc[-21:-1].min())
        range_high=float(df15["high"].iloc[-21:-1].max())
        width=max(range_high-range_low,1e-12)
        position=(s5["price"]-range_low)/width
        row=df5.iloc[-1]
        if direction==LONG:
            at_edge=position<=0.22
            rsi_ok=s5["rsi"]<=self.cfg.expert_range_rsi_long
            rejection=float(row["low"])<=range_low+0.12*width and s5["price"]>range_low
            invalidation=min(float(row["low"]),range_low)-0.05*s5["atr"]
        else:
            at_edge=position>=0.78
            rsi_ok=s5["rsi"]>=self.cfg.expert_range_rsi_short
            rejection=float(row["high"])>=range_high-0.12*width and s5["price"]<range_high
            invalidation=max(float(row["high"]),range_high)+0.05*s5["atr"]
        if not at_edge or not (rsi_ok or zone["near_own"]) or not (rejection or self._directional_candle(s5,0.10,0.56)):
            return None
        htf,htf_detail=self._htf_points(direction,regime,bias)
        structure_pts=12
        location_pts=20 if zone["near_own"] else 17
        trigger_pts=18 if rejection else 14
        momentum_pts=8 if rsi_ok else 5
        quality_pts=8 if s5["directional_candle"] else 5
        score=htf+structure_pts+location_pts+trigger_pts+momentum_pts+quality_pts+8
        threshold=self._threshold(self.cfg.expert_thr_range_reversal,regime,bias,direction)
        if score<threshold:
            return None
        # Mid/other edge is a natural opposing target; target R is capped there.
        natural=range_high if direction==LONG else range_low
        if opposing is None or (natural<opposing if direction==LONG else natural>opposing):
            opposing=natural
        return self._finalize(
            direction=direction,setup_type=RANGE_REVERSAL,timeframe="5M",trigger="RANGE_EDGE_REJECTION",
            score=score,threshold=threshold,price=s5["price"],atr_v=s5["atr"],invalidation=invalidation,
            desired_r=self.cfg.expert_tp2_range_r,opposing_level=opposing,bar_ts=pd.Timestamp(df5.index[-1]),
            components={"htf":htf_detail,"structure":structure_pts,"location":location_pts,"trigger":trigger_pts,"momentum":momentum_pts,"quality":quality_pts,"range_position":round(position,3),"range_low":range_low,"range_high":range_high,"rsi":round(s5["rsi"],1)},
            min_room_r=self.cfg.expert_range_min_room_r,
        )

    def _trend_lifecycle(self, df5: pd.DataFrame, direction: str) -> tuple[str, float]:
        """Approximate the current directional leg age + extension.

        This is intentionally execution-frame local. It distinguishes a fresh
        trend from a mature/extended leg so a 100 regime score cannot by itself
        justify chasing the third continuation entry.
        """
        if df5 is None or len(df5) < 40:
            return "DEVELOPING", 0.0
        close = df5["close"].astype(float)
        ema20 = ind.ema(close, 20)
        atrs = ind.atr(df5, 14)
        atrv = max(float(atrs.iloc[-1]), 1e-12)
        aligned = (close > ema20) if direction == LONG else (close < ema20)
        age = 0
        for ok in reversed(aligned.iloc[-48:].tolist()):
            if not bool(ok):
                break
            age += 1
        ext = abs(float(close.iloc[-1] - ema20.iloc[-1])) / atrv
        mature_bars = int(getattr(self.cfg, "expert_lifecycle_mature_bars", 18))
        extended_bars = int(getattr(self.cfg, "expert_lifecycle_extended_bars", 30))
        exhaust_ext = float(getattr(self.cfg, "expert_lifecycle_exhaustion_extension_atr", 1.35))
        # A fresh impulse can be far from EMA20 without being an exhausted trend.
        # Only mature legs can be labelled EXHAUSTING.
        if age >= mature_bars and ext >= exhaust_ext:
            return "EXHAUSTING", ext
        if age >= extended_bars or (age >= mature_bars and ext >= 1.00):
            return "EXTENDED", ext
        if age >= mature_bars:
            return "MATURE", ext
        if age <= 6:
            return "EARLY", ext
        return "DEVELOPING", ext

    def _adaptive_candidate_threshold(self, candidate: _Candidate, lifecycle: str, state: _State, symbol: str) -> float:
        add = 0.0
        if bool(getattr(self.cfg, "expert_trend_lifecycle_enabled", True)):
            add += {
                "MATURE": float(getattr(self.cfg, "expert_lifecycle_mature_threshold_add", 2.0)),
                "EXTENDED": float(getattr(self.cfg, "expert_lifecycle_extended_threshold_add", 5.0)),
                "EXHAUSTING": float(getattr(self.cfg, "expert_lifecycle_exhausting_threshold_add", 8.0)),
            }.get(lifecycle, 0.0)
        if bool(getattr(self.cfg, "expert_leg_budget_enabled", True)) and state.leg_direction == candidate.direction:
            if state.leg_entries == 1:
                add += float(getattr(self.cfg, "expert_leg_second_entry_add", 4.0))
            elif state.leg_entries >= 2:
                add += float(getattr(self.cfg, "expert_leg_third_entry_add", 8.0))
        if bool(getattr(self.cfg, "expert_xau_probation_enabled", True)) and "XAU" in str(symbol).upper():
            add += float(getattr(self.cfg, "expert_xau_probation_threshold_add", 5.0))
        return candidate.threshold + add

    def _reentry_allowed(self, state: _State, direction: str, df15: pd.DataFrame) -> tuple[bool,str]:
        if state.reentry_lock_direction != direction or state.reentry_lock_ts is None:
            return True,""
        age=pd.Timestamp(df15.index[-1])-state.reentry_lock_ts
        if age>pd.Timedelta(hours=self.cfg.expert_reentry_lock_hours):
            state.reentry_lock_direction=""; state.reentry_lock_ts=None
            return True,"re-entry time lock expired"
        # A new 15M BOS after the lock resets the failed thesis.
        bos,_=ind.latest_bos(df15,direction,3,3,0.12)
        if bos and pd.Timestamp(df15.index[-1])>state.reentry_lock_ts:
            state.reentry_lock_direction=""; state.reentry_lock_ts=None
            return True,"fresh 15M BOS reset re-entry lock"
        return False,"same-side locked after 3 full SLs; waiting for fresh 15M BOS or lock expiry"

    def _same_setup_cooldown(self,state:_State,candidate:_Candidate,current_ts:pd.Timestamp)->bool:
        if not state.last_setup or state.last_entry_ts is None or candidate.setup_type!=state.last_setup:
            return False
        bars= self.cfg.expert_same_setup_cooldown_15m_bars if candidate.timeframe=="15M" else self.cfg.expert_same_setup_cooldown_5m_bars
        minutes=15 if candidate.timeframe=="15M" else 5
        return current_ts-state.last_entry_ts < pd.Timedelta(minutes=bars*minutes)

    def analyze(
        self,
        df_30m: Optional[pd.DataFrame],
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        direction: str,
        symbol: str,
        df_1h: Optional[pd.DataFrame]=None,
        df_4h: Optional[pd.DataFrame]=None,
        regime=None,
        bias=None,
    )->EntryResult:
        if direction not in (LONG,SHORT,BOTH):
            return EntryResult(NONE,False,"no usable direction from Bias")
        if df_15m is None or len(df_15m)<80 or df_5m is None or len(df_5m)<100:
            return EntryResult(NONE,False,"insufficient 15M/5M history")
        state=self._get_state(symbol)
        current_ts=pd.Timestamp(df_5m.index[-1])
        price=float(df_5m["close"].iloc[-1])
        if state.last_processed_5m==current_ts:
            return EntryResult(NONE,False,"5M bar already processed",price=price)
        state.last_processed_5m=current_ts

        label=getattr(regime,"label","")
        sides=[LONG,SHORT] if direction==BOTH else [direction]
        candidates:list[_Candidate]=[]
        diagnostics=[]
        for side in sides:
            allowed,reason=self._reentry_allowed(state,side,df_15m)
            if not allowed:
                diagnostics.append(f"{side}:{reason}")
                continue
            s15=self._snapshot(df_15m,side)
            s5=self._snapshot(df_5m,side)
            # Strong opposite 15M cluster remains a hard veto in directional regimes.
            if label not in (RANGE,COMPRESSION) and s15["opposite_score"]>=72 and s15["edge"]<=-18:
                diagnostics.append(f"{side}:strong opposite 15M cluster")
                continue
            zone=self._zone_context(side,s5["price"],s5["atr"],[
                (df_15m,"15M",self.cfg.expert_zone_lookback_15m),
                (df_1h,"1H",self.cfg.expert_zone_lookback_1h),
            ])
            # Hard room is based on active opposing Supply/Demand, not every
            # minor swing. Minor pivots are frequently crossed during trend
            # continuation and treating all of them as hard ceilings caused
            # the previous bot to reject virtually every setup.
            # Major pivots are retained for diagnostics/context. Active
            # opposing SMC zones are the non-compensable hard room gate; minor
            # pivots are crossed frequently during continuation.
            support,resistance=self._nearest_structure(s5["price"],[df_15m,df_1h,df_4h])
            opposing=None
            if zone["opposing"] is not None:
                opposing=zone["opposing"].low if side==LONG else zone["opposing"].high

            if self.cfg.expert_allow_15m_entry:
                for cand in (
                    self._cross_candidate(df_15m,"15M",side,s15,s15,zone,regime,bias,opposing),
                    self._breakout_candidate(df_15m,"15M",side,s15,s15,regime,bias,opposing),
                    self._liquidity_sweep(df_15m,"15M",side,s15,zone,regime,bias,opposing),
                ):
                    if cand is not None: candidates.append(cand)
            if self.cfg.expert_allow_5m_entry:
                for cand in (
                    self._cross_candidate(df_5m,"5M",side,s5,s15,zone,regime,bias,opposing),
                    (self._structure_pullback(df_5m,side,s5,s15,zone,regime,bias,opposing)
                     if self.cfg.expert_structure_pullback_enabled else None),
                    self._smc_rejection(df_5m,side,s5,s15,zone,regime,bias,opposing),
                    self._breakout_candidate(df_5m,"5M",side,s5,s15,regime,bias,opposing),
                    self._liquidity_sweep(df_5m,"5M",side,s5,zone,regime,bias,opposing),
                    self._continuation(df_5m,side,s5,s15,regime,bias,opposing),
                    self._range_reversal(df_15m,df_5m,side,s5,zone,regime,bias,opposing),
                ):
                    if cand is not None: candidates.append(cand)
            diagnostics.append(f"{side}:15Medge={s15['edge']:+.0f} 5Medge={s5['edge']:+.0f} zone={zone['own'].timeframe if zone['own'] else '-'}")

        candidates=[c for c in candidates if not self._same_setup_cooldown(state,c,current_ts)]
        candidates=[c for c in candidates if c.signal_key!=state.last_entry_key]

        # V3.3.1 local adaptation: lifecycle, trend-leg budget and XAU probation.
        adapted=[]
        for cand in candidates:
            lifecycle, lifecycle_ext = self._trend_lifecycle(df_5m, cand.direction)
            # XAU must have meaningful 15M confirmation while on probation.
            if (
                bool(getattr(self.cfg, "expert_xau_probation_enabled", True))
                and "XAU" in str(symbol).upper()
                and bool(getattr(self.cfg, "expert_xau_require_15m_confirm", True))
            ):
                x15=self._snapshot(df_15m,cand.direction)
                if x15["edge"] < float(getattr(self.cfg, "expert_xau_15m_min_edge", 6.0)):
                    diagnostics.append(f"{cand.direction}:{cand.setup_type} XAU probation 15M edge {x15['edge']:+.0f}")
                    continue

            effective_thr=self._adaptive_candidate_threshold(cand,lifecycle,state,symbol)
            if cand.score < effective_thr:
                diagnostics.append(
                    f"{cand.direction}:{cand.setup_type} lifecycle={lifecycle} "
                    f"score {cand.score:.0f}<{effective_thr:.0f}"
                )
                continue
            cand.threshold=round(effective_thr,1)
            cand.components["trend_lifecycle"]=lifecycle
            cand.components["lifecycle_extension_atr"]=round(lifecycle_ext,2)
            cand.components["leg_entry_no"]=(state.leg_entries+1 if state.leg_direction==cand.direction else 1)
            adapted.append(cand)
        candidates=adapted
        if not candidates:
            self._persist_state()
            s5=self._snapshot(df_5m,sides[0])
            diag_text = "; ".join(diagnostics[-8:]) if diagnostics else "no setup trigger formed"
            return EntryResult(NONE,False,f"no valid expert setup | regime={label} | {diag_text}",price=price,ema_fast=s5["ema8"],ema_slow=s5["ema13"],entry_score=0.0)

        # Candidate edge is primary; score and higher timeframe are tie-breakers.
        candidates.sort(key=lambda x:(x.edge,x.score,1 if x.timeframe=="15M" else 0),reverse=True)
        selected=candidates[0]
        # In BOTH mode require the winner to exceed an opposite-side candidate by
        # a small margin unless it is an SMC/range-edge setup at a valid zone.
        if direction==BOTH:
            opposite=[x for x in candidates if x.direction!=selected.direction]
            if opposite and selected.edge-opposite[0].edge<2 and selected.setup_type not in (SMC_ZONE_REJECTION,LIQUIDITY_SWEEP,RANGE_REVERSAL):
                self._persist_state()
                return EntryResult(NONE,False,"range/compression candidates too balanced; waiting for clearer edge",price=price)

        self._persist_state()
        snap=self._snapshot(df_5m,selected.direction)
        return EntryResult(
            direction=selected.direction,
            allow_entry=True,
            reason=(f"{selected.setup_type} [{selected.timeframe}] {selected.trigger}: "
                    f"score={selected.score:.1f}/{selected.threshold:.1f} edge={selected.edge:+.1f} "
                    f"room={selected.room_r:.2f}R actualRR={selected.rr:.2f}"),
            price=selected.price,
            ema_fast=snap["ema8"],
            ema_slow=snap["ema13"],
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

    def confirm_entry(self,symbol:str,cross_ts)->None:
        if cross_ts is None:
            return
        s=self._get_state(symbol)
        s.last_entry_key=cross_ts
        try:
            s.last_entry_ts=pd.Timestamp(cross_ts[0])
            s.last_setup=str(cross_ts[2])
            direction=str(cross_ts[1])
            if s.leg_direction == direction:
                s.leg_entries += 1
            else:
                s.leg_direction = direction
                s.leg_entries = 1
            if s.leg_anchor_ts is None:
                s.leg_anchor_ts = s.last_entry_ts
        except (TypeError,ValueError,IndexError):
            s.last_entry_ts=pd.Timestamp.utcnow()
        self._persist_state()

    def check_exit(self,df_5m:pd.DataFrame,position_side:str,bars_since_entry:Optional[int]=None)->ExitCheckResult:
        c=self.cfg
        if bars_since_entry is not None and bars_since_entry<c.exit_grace_bars:
            return ExitCheckResult(False)
        if df_5m is None or len(df_5m)<50:
            return ExitCheckResult(False)
        close=df_5m["close"].astype(float)
        fast=ind.ema(close,c.l3c_ema_fast)
        slow=ind.ema(close,c.l3c_ema_slow)
        roc5=ind.roc(close,5)
        _,plus,minus=ind.adx(df_5m,12)
        long=position_side.lower()=="long"
        cross_back=(fast.iloc[-1]<slow.iloc[-1] and fast.iloc[-2]>=slow.iloc[-2]) if long else (fast.iloc[-1]>slow.iloc[-1] and fast.iloc[-2]<=slow.iloc[-2])
        wrong_close=close.iloc[-1]<slow.iloc[-1] if long else close.iloc[-1]>slow.iloc[-1]
        wrong_prev=close.iloc[-2]<slow.iloc[-2] if long else close.iloc[-2]>slow.iloc[-2]
        momentum_wrong=roc5.iloc[-1]<0 if long else roc5.iloc[-1]>0
        di_wrong=minus.iloc[-1]>plus.iloc[-1] if long else plus.iloc[-1]>minus.iloc[-1]
        if cross_back and (momentum_wrong or di_wrong):
            return ExitCheckResult(True,EMA_CROSS_REVERSAL,"5M EMA10/20 crossed back with momentum/DI confirmation")
        if wrong_close and wrong_prev and momentum_wrong and di_wrong:
            return ExitCheckResult(True,PRICE_OPEN_BEYOND_EMA,"two closed 5M bars beyond EMA20 with ROC and DMI reversal")
        return ExitCheckResult(False)
