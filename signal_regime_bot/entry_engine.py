"""Layer 3: balanced dual-entry engine for 15-minute execution.

Two setup engines share the same fixed direction from Bias:

* FAST_PULLBACK: valid trend/location -> touch/arm -> reclaim or micro break.
* MOMENTUM: confirmed structure breakout on the breakout candle when quality is
  exceptional, otherwise a one-bar breakout-retest continuation.

The implementation is stateful, symmetric and closed-candle only.  It avoids
indicator-voting confluence and does not require a new moving-average cross for
every trade, which preserves trade frequency while Structure Room, extension,
chop and candle-quality gates reduce false entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    macd_hist: float = 0.0  # kept for existing status views; now candidate edge
    cross_id: object = None  # deterministic signal key, kept for caller compatibility
    entry_score: float = 0.0
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

    def _get_state(self, symbol: str) -> _SetupState:
        return self._state.setdefault(symbol, _SetupState())

    def reset_symbol(self, symbol: str) -> None:
        self._state.pop(symbol, None)

    def on_position_closed(self, symbol: str) -> None:
        # Clear any setup that existed before/during the position.  last_entry_key
        # remains the anti-duplicate guard, so re-entry needs a new timestamp.
        state = self._get_state(symbol)
        state.pullback = None
        state.breakout = None
        state.last_candidate_key = None

    def observe(
        self,
        df_30m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        symbol: str,
    ) -> None:
        """Compatibility hook retained for Pipeline; setup logic runs in analyze."""
        self._get_state(symbol)

    @staticmethod
    def _is_long(direction: str) -> bool:
        return direction == LONG

    def _nearest_levels(
        self,
        price: float,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame],
        df_4h: Optional[pd.DataFrame],
    ) -> tuple[Optional[float], Optional[float]]:
        supports: list[float] = []
        resistances: list[float] = []
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

    def _local_snapshot(self, df: pd.DataFrame, direction: str) -> dict:
        c = self.cfg
        close = df["close"]
        hma_fast = ind.hma(close, getattr(c, "dual_hma_fast", 10))
        hma_slow = ind.hma(close, getattr(c, "dual_hma_slow", 16))
        ema20 = ind.ema(close, 20)
        ema50 = ind.ema(close, 50)
        atr_s = ind.atr(df, 14)
        atr_value = ind.safe_float(atr_s.iloc[-1])
        adx_s, plus_s, minus_s = ind.adx(df, 12)
        adx_value = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_s.iloc[-1])
        minus_di = ind.safe_float(minus_s.iloc[-1])
        chop = ind.safe_float(ind.choppiness_index(df, 14).iloc[-1], 100.0)
        roc5 = ind.safe_float(ind.roc(close, 5).iloc[-1])
        roc_prev = ind.safe_float(ind.roc(close, 5).iloc[-2])
        structure = ind.market_structure(
            df["high"],
            df["low"],
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
        )
        candle = ind.candle_metrics(df, atr_value)
        price = ind.safe_float(close.iloc[-1])
        hfast = ind.safe_float(hma_fast.iloc[-1])
        hslow = ind.safe_float(hma_slow.iloc[-1])
        hfast_prev = ind.safe_float(hma_fast.iloc[-2])
        hslow_prev = ind.safe_float(hma_slow.iloc[-2])
        spread_atr = abs(hfast - hslow) / max(atr_value, ind.EPSILON)
        flip_count = ind.cross_count(hma_fast, hma_slow, 8)

        long = direction == LONG
        aligned = hfast > hslow if long else hfast < hslow
        fast_slope = hfast > hfast_prev if long else hfast < hfast_prev
        slow_slope = hslow >= hslow_prev if long else hslow <= hslow_prev
        dmi_ok = plus_di > minus_di if long else minus_di > plus_di
        roc_ok = roc5 > 0 if long else roc5 < 0
        roc_accel = roc5 > roc_prev if long else roc5 < roc_prev
        structure_ok = structure != ("LH_LL" if long else "HH_HL")
        close_above_ema50 = price > ema50.iloc[-1] if long else price < ema50.iloc[-1]

        obvious_chop = (
            (adx_value < getattr(c, "dual_min_adx", 11.0) and chop > getattr(c, "dual_max_chop", 62.0))
            or (flip_count >= 3 and spread_atr < 0.12)
        )
        strong = (
            aligned
            and fast_slope
            and slow_slope
            and dmi_ok
            and adx_value >= getattr(c, "dual_strong_adx", 20.0)
            and chop <= getattr(c, "dual_strong_chop", 52.0)
        )
        transition = (
            structure_ok
            and not obvious_chop
            and adx_value >= 9.0
            and chop <= 64.0
            and (aligned or abs(hfast - hslow) / max(atr_value, ind.EPSILON) <= 0.18)
        )
        return {
            "price": price,
            "hma_fast_series": hma_fast,
            "hma_slow_series": hma_slow,
            "hma_fast": hfast,
            "hma_slow": hslow,
            "ema20": ind.safe_float(ema20.iloc[-1]),
            "ema50": ind.safe_float(ema50.iloc[-1]),
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
            "fast_slope": fast_slope,
            "slow_slope": slow_slope,
            "dmi_ok": dmi_ok,
            "roc_ok": roc_ok,
            "structure_ok": structure_ok,
            "ema50_ok": close_above_ema50,
            "obvious_chop": obvious_chop,
            "strong": strong,
            "transition": transition,
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
        min_stop = atr_value * getattr(c, "dual_min_stop_atr", 0.55)
        max_stop = atr_value * getattr(c, "dual_max_stop_atr", 1.50)
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

        base_rr = getattr(c, "dual_pullback_tp2_r", 2.0) if setup_type == FAST_PULLBACK else getattr(c, "dual_momentum_tp2_r", 2.0)
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
            getattr(c, "dual_pullback_min_room_r", 1.05)
            if setup_type == FAST_PULLBACK
            else getattr(c, "dual_momentum_min_room_r", 1.15)
        )
        if room_r < min_room or rr < getattr(c, "minimum_actual_rr", 1.20):
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
        if snapshot["obvious_chop"] or not snapshot["structure_ok"] or not snapshot["transition"]:
            state.pullback = None
            return None

        price = snapshot["price"]
        row = df.iloc[-1]
        bar_ts = df.index[-1]
        bar_no = len(df) - 1
        hfast, hslow = snapshot["hma_fast"], snapshot["hma_slow"]
        zone_upper = max(hfast, hslow) + atr_value * getattr(c, "dual_pullback_zone_atr", 0.20)
        zone_lower = min(hfast, hslow) - atr_value * getattr(c, "dual_pullback_depth_atr", 0.35)

        swing_high, swing_low = ind.recent_swing_levels(
            df["high"],
            df["low"],
            getattr(c, "entry_swing_left", 3),
            getattr(c, "entry_swing_right", 3),
        )
        level = nearest_support if long else nearest_resistance
        sweep = ind.sweep_reclaim(df, direction, level)
        if long:
            hma_touch = float(row["low"]) <= zone_upper and float(row["low"]) >= zone_lower
            htf_touch = level is not None and float(row["low"]) <= level + 0.20 * atr_value and price > level
            location_touch = hma_touch or htf_touch or sweep
            invalidation = min(float(row["low"]), swing_low if np.isfinite(swing_low) else float(row["low"]))
            reclaim = price > hfast and float(row["low"]) <= max(hfast, hslow)
            previous_break = price > float(df["high"].iloc[-2])
            micro_bos = price > float(df["high"].iloc[-3:-1].max())
        else:
            hma_touch = float(row["high"]) >= zone_lower and float(row["high"]) <= zone_upper
            htf_touch = level is not None and float(row["high"]) >= level - 0.20 * atr_value and price < level
            location_touch = hma_touch or htf_touch or sweep
            invalidation = max(float(row["high"]), swing_high if np.isfinite(swing_high) else float(row["high"]))
            reclaim = price < hfast and float(row["high"]) >= min(hfast, hslow)
            previous_break = price < float(df["low"].iloc[-2])
            micro_bos = price < float(df["low"].iloc[-3:-1].min())

        location_score = 20.0 if htf_touch or sweep else 14.0 if hma_touch else 0.0
        candle_ok = (
            ind.bullish_trigger_candle(
                snapshot["candle"],
                getattr(c, "dual_pullback_min_body_atr", 0.15),
                getattr(c, "dual_pullback_close_quality", 0.62),
            )
            if long
            else ind.bearish_trigger_candle(
                snapshot["candle"],
                getattr(c, "dual_pullback_min_body_atr", 0.15),
                getattr(c, "dual_pullback_close_quality", 0.62),
            )
        )
        same_bar_trigger = location_touch and candle_ok and (reclaim or sweep or previous_break)

        # Arm a location touch.  The same-bar path is evaluated before waiting,
        # allowing fast entries only when location and candle quality are strong.
        if location_touch and state.pullback is None:
            state.pullback = _PullbackSetup(
                direction=direction,
                started_bar=bar_no,
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
        age = bar_no - setup.started_bar
        if age > getattr(c, "dual_pullback_window_bars", 3):
            state.pullback = None
            return None
        if long:
            setup.setup_low = min(setup.setup_low, float(row["low"]))
            invalidated = price < setup.invalidation
            armed_break = price > setup.trigger_level
        else:
            setup.setup_high = max(setup.setup_high, float(row["high"]))
            invalidated = price > setup.invalidation
            armed_break = price < setup.trigger_level
        if invalidated:
            state.pullback = None
            return None

        trigger = ""
        if same_bar_trigger and age == 0:
            trigger = "SAME_BAR_RECLAIM" if not sweep else "SWEEP_RECLAIM"
        elif age >= 1 and candle_ok and (armed_break or reclaim or micro_bos or sweep):
            if sweep:
                trigger = "SWEEP_RECLAIM"
            elif micro_bos:
                trigger = "MICRO_BOS"
            elif armed_break:
                trigger = "TRIGGER_BREAK"
            else:
                trigger = "HMA_RECLAIM"
        if not trigger:
            return None

        extension = (
            (price - hslow) / atr_value if long else (hslow - price) / atr_value
        )
        if extension < -0.20 or extension > getattr(c, "dual_pullback_max_extension_atr", 0.75):
            return None

        components = {
            "macro": 10.0 if getattr(regime, "label", "").startswith("STRONG") else 7.0,
            "bias": min(15.0, max(6.0, abs(getattr(bias, "directional_edge", 8.0)) * 0.6)),
            "structure": 15.0 if snapshot["structure"] == ("HH_HL" if long else "LH_LL") else 9.0,
            "location": setup.location_score,
            "prior_context": 10.0 if snapshot["aligned"] else 6.0,
            "trigger": 15.0 if trigger in ("SWEEP_RECLAIM", "MICRO_BOS") else 12.0,
            "candle": 10.0 if candle_ok else 0.0,
            "momentum": 5.0 if snapshot["roc_ok"] or snapshot["dmi_ok"] else 2.0,
        }
        score = min(100.0, sum(components.values()))
        threshold = (
            getattr(c, "dual_same_bar_pullback_threshold", 70.0)
            if age == 0
            else getattr(c, "dual_pullback_threshold", 64.0)
        )
        if score < threshold:
            return None

        buffer = atr_value * getattr(c, "dual_stop_buffer_atr", 0.10)
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
        if snapshot["obvious_chop"] or not snapshot["structure_ok"]:
            state.breakout = None
            return None
        if not (snapshot["strong"] or (snapshot["aligned"] and snapshot["adx"] >= getattr(c, "dual_momentum_min_adx", 13.0))):
            return None

        price = snapshot["price"]
        row = df.iloc[-1]
        bar_ts = df.index[-1]
        bar_no = len(df) - 1
        lookback = getattr(c, "dual_breakout_lookback", 4)
        if len(df) < lookback + 3:
            return None
        break_level = (
            float(df["high"].iloc[-lookback - 1 : -1].max())
            if long
            else float(df["low"].iloc[-lookback - 1 : -1].min())
        )
        crossed = (
            price > break_level and float(df["close"].iloc[-2]) <= break_level
            if long
            else price < break_level and float(df["close"].iloc[-2]) >= break_level
        )
        candle = snapshot["candle"]
        candle_ok = (
            ind.bullish_trigger_candle(
                candle,
                getattr(c, "dual_momentum_min_body_atr", 0.18),
                getattr(c, "dual_momentum_close_quality", 0.68),
            )
            if long
            else ind.bearish_trigger_candle(
                candle,
                getattr(c, "dual_momentum_min_body_atr", 0.18),
                getattr(c, "dual_momentum_close_quality", 0.68),
            )
        )
        strong_breakout = (
            crossed
            and candle.body_atr >= 0.25
            and (candle.bull_close_quality >= 0.75 if long else candle.bear_close_quality >= 0.75)
            and (candle.volume_ratio >= getattr(c, "dual_momentum_volume_ratio", 1.05) or candle.body_atr >= 0.32)
        )

        if crossed:
            state.breakout = _BreakoutSetup(
                direction=direction,
                started_bar=bar_no,
                started_ts=bar_ts,
                breakout_level=break_level,
                breakout_low=float(row["low"]),
                breakout_high=float(row["high"]),
            )

        setup = state.breakout
        if setup is None or setup.direction != direction:
            return None
        age = bar_no - setup.started_bar
        if age > 1:
            state.breakout = None
            return None

        direct = age == 0 and strong_breakout
        if long:
            retest = (
                age == 1
                and float(row["low"]) <= setup.breakout_level + 0.15 * atr_value
                and float(row["low"]) >= setup.breakout_level - 0.35 * atr_value
                and price > setup.breakout_level
                and candle_ok
            )
            invalidated = price < setup.breakout_level - 0.35 * atr_value
        else:
            retest = (
                age == 1
                and float(row["high"]) >= setup.breakout_level - 0.15 * atr_value
                and float(row["high"]) <= setup.breakout_level + 0.35 * atr_value
                and price < setup.breakout_level
                and candle_ok
            )
            invalidated = price > setup.breakout_level + 0.35 * atr_value
        if invalidated:
            state.breakout = None
            return None
        if not (direct or retest):
            return None

        reference = setup.breakout_level
        extension = abs(price - reference) / max(atr_value, ind.EPSILON)
        hma_extension = (
            (price - snapshot["hma_slow"]) / atr_value
            if long
            else (snapshot["hma_slow"] - price) / atr_value
        )
        max_ext = (
            getattr(c, "dual_strong_momentum_extension_atr", 1.10)
            if strong_breakout
            else getattr(c, "dual_momentum_max_extension_atr", 0.90)
        )
        if extension > 0.55 or hma_extension > max_ext:
            return None

        components = {
            "macro": 10.0 if getattr(regime, "label", "").startswith("STRONG") else 7.0,
            "bias": min(15.0, max(6.0, abs(getattr(bias, "directional_edge", 8.0)) * 0.6)),
            "structure_break": 22.0 if crossed else 18.0,
            "breakout_close": 8.0 if candle_ok else 4.0,
            "displacement": 10.0 if candle.body_atr >= 0.25 else 6.0,
            "volume": 8.0 if candle.volume_ratio >= getattr(c, "dual_momentum_volume_ratio", 1.05) else 4.0,
            "trend": 12.0 if snapshot["strong"] else 8.0,
            "momentum": 10.0 if snapshot["roc_ok"] and snapshot["dmi_ok"] else 6.0,
            "retest": 5.0 if retest else 2.0,
        }
        score = min(100.0, sum(components.values()))
        threshold = (
            getattr(c, "dual_strong_breakout_threshold", 66.0)
            if direct
            else getattr(c, "dual_momentum_threshold", 70.0)
        )
        if score < threshold:
            return None

        buffer = atr_value * getattr(c, "dual_stop_buffer_atr", 0.10)
        if long:
            base_low = float(df["low"].iloc[-lookback - 1 : -1].min())
            raw_stop = min(setup.breakout_low, setup.breakout_level - buffer, base_low)
        else:
            base_high = float(df["high"].iloc[-lookback - 1 : -1].max())
            raw_stop = max(setup.breakout_high, setup.breakout_level + buffer, base_high)
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
        c = self.cfg
        if direction not in (LONG, SHORT):
            return EntryResult(NONE, False, "no fixed direction from Bias")
        if df_15m is None or len(df_15m) < 100:
            return EntryResult(NONE, False, "insufficient 15M history")

        state = self._get_state(symbol)
        bar_ts = df_15m.index[-1]
        # The strategy is 15M-entry. A repeated poll of the same closed bar may
        # display the existing candidate but must never create a new signal.
        if state.last_processed_bar == bar_ts:
            snap = self._local_snapshot(df_15m, direction)
            return EntryResult(
                NONE,
                False,
                "15M bar already processed",
                price=snap["price"],
                ema_fast=snap["hma_fast"],
                ema_slow=snap["hma_slow"],
            )
        state.last_processed_bar = bar_ts

        snapshot = self._local_snapshot(df_15m, direction)
        base = dict(
            price=snapshot["price"],
            ema_fast=snapshot["hma_fast"],
            ema_slow=snapshot["hma_slow"],
        )
        if snapshot["atr"] <= 0:
            return EntryResult(NONE, False, "ATR invalid", **base)
        if snapshot["obvious_chop"]:
            state.pullback = None
            state.breakout = None
            return EntryResult(
                NONE,
                False,
                f"CHOP veto: ADX={snapshot['adx']:.1f} CHOP={snapshot['chop']:.1f} flips={snapshot['flip_count']}",
                **base,
            )
        if getattr(regime, "volatility_shock", False):
            # A pullback can still be evaluated after the shock bar, but direct
            # momentum on the shock itself is blocked unless exceptionally clean.
            candle = snapshot["candle"]
            if candle.body_atr > 2.5 and candle.volume_ratio < 1.5:
                return EntryResult(NONE, False, "volatility shock without quality confirmation", **base)

        support, resistance = self._nearest_levels(
            snapshot["price"], df_15m, df_1h, df_4h
        )
        pullback = self._pullback_candidate(
            df_15m,
            direction,
            snapshot,
            state,
            support,
            resistance,
            regime,
            bias,
        )
        momentum = self._momentum_candidate(
            df_15m,
            direction,
            snapshot,
            state,
            support,
            resistance,
            regime,
            bias,
        )
        candidates = [x for x in (pullback, momentum) if x is not None]
        if not candidates:
            armed = []
            if state.pullback is not None:
                armed.append("PB")
            if state.breakout is not None:
                armed.append("MOM")
            armed_text = "+".join(armed) if armed else "none"
            return EntryResult(
                NONE,
                False,
                f"no valid 15M trigger (armed={armed_text}) | structure={snapshot['structure']} "
                f"ADX={snapshot['adx']:.1f} CHOP={snapshot['chop']:.1f}",
                macd_hist=0.0,
                **base,
            )

        # Prefer the higher edge. If quality is nearly equal, Pullback gets
        # priority because it normally offers better location and less slippage.
        candidates.sort(key=lambda x: x.edge, reverse=True)
        selected = candidates[0]
        if len(candidates) > 1 and abs(candidates[0].edge - candidates[1].edge) <= 2:
            selected = next((x for x in candidates if x.setup_type == FAST_PULLBACK), selected)
        if state.last_entry_key is not None and selected.signal_key == state.last_entry_key:
            return EntryResult(NONE, False, "duplicate signal key", **base)

        state.last_candidate_key = selected.signal_key
        return EntryResult(
            direction=selected.direction,
            allow_entry=True,
            reason=(
                f"{selected.setup_type} {selected.trigger}: score={selected.score:.1f}/"
                f"{selected.threshold:.1f} edge={selected.edge:+.1f} "
                f"room={selected.room_r:.2f}R actualRR={selected.rr:.2f}"
            ),
            price=selected.price,
            ema_fast=snapshot["hma_fast"],
            ema_slow=snapshot["hma_slow"],
            macd_hist=selected.edge,
            cross_id=selected.signal_key,
            entry_score=selected.score,
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

    def check_exit(
        self,
        df_5m: pd.DataFrame,
        position_side: str,
        bars_since_entry: Optional[int] = None,
    ) -> ExitCheckResult:
        """Noise-resistant 5M early exit.

        A single EMA flip no longer closes a position.  Hard reversal requires a
        cross plus price confirmation and either ROC or DMI confirmation.  The
        slower weakness path requires two consecutive closed bars.
        """
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
