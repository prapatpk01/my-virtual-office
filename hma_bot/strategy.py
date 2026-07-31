"""
Precision Trend Structure V2
============================

Architecture
------------
Layer 1 — 4H Direction
    EMA20 / EMA50 + EMA20 slope + HMA16 state
    -> LONG ONLY / SHORT ONLY / NEUTRAL

Layer 2 — 1H Market Quality
    Q = ADX score + CHOP score
    -> Q >= 55 = tradable

Layer 3 — 15M Location / Setup
    Only three setup families:
      1) Pullback continuation
      2) Liquidity sweep back into trend
      3) Breakout retest

Layer 4 — 15M Execution Trigger
    Location must exist BEFORE the trigger.
    Trigger = local reclaim / micro CHOCH / displacement close.
    No weighted entry score.

Risk / management
-----------------
Initial hard SL: 1.5%
T1 +0.6% -> lock +0.3%
T2 +1.0% -> lock +0.7%
Runner -> final TP +1.5%

Early exit:
    opposite micro break = warning
    opposite micro break + major structure loss = exit
    opposite 4H trend = exit

All structure decisions use closed candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Trend(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class SetupType(str, Enum):
    PULLBACK = "PULLBACK"
    SWEEP = "LIQUIDITY_SWEEP"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"


class ExitReason(str, Enum):
    STRUCTURE_INVALIDATION = "STRUCTURE_INVALIDATION"
    HTF_TREND_INVALIDATION = "HTF_TREND_INVALIDATION"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"


@dataclass(frozen=True)
class StrategyConfig:
    trend_tf: str = "4h"
    quality_tf: str = "1h"
    entry_tf: str = "15m"

    ema_fast_len: int = 20
    ema_slow_len: int = 50
    hma_len: int = 16

    atr_len: int = 14
    dmi_len: int = 14
    adx_len: int = 14
    chop_len: int = 14
    min_trend_quality: float = 55.0

    # 15M structure
    major_left: int = 3
    major_right: int = 3
    micro_left: int = 2
    micro_right: int = 2
    setup_lookback_bars: int = 8

    # Location / trigger
    location_atr: float = 0.65
    retest_atr: float = 0.35
    min_trigger_body_atr: float = 0.20
    max_trigger_body_atr: float = 1.20
    min_close_location: float = 0.60
    max_chase_atr: float = 0.75

    # Need enough room to at least reach T1 + buffer.
    min_room_pct: float = 0.008

    # Exit hysteresis
    invalidation_confirm_bars: int = 2

    # Risk
    stop_loss_pct: float = 0.015
    final_take_profit_pct: float = 0.015
    target1_trigger_pct: float = 0.006
    target1_lock_pct: float = 0.003
    target2_trigger_pct: float = 0.010
    target2_lock_pct: float = 0.007


@dataclass
class TrendState:
    trend: Trend
    ema20: float
    ema50: float
    ema20_slope: float
    hma16: float
    hma_state: int
    warning: bool = False


@dataclass
class QualityState:
    q: float
    adx: float
    chop: float
    plus_di: float
    minus_di: float


@dataclass
class EntrySignal:
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    trend_4h: Trend
    q_1h: float
    adx_1h: float
    chop_1h: float
    setup: SetupType
    trigger: str
    room_pct: float
    atr15: float
    structure_level: float
    reason: str


@dataclass
class StructureExit:
    should_exit: bool
    reason: Optional[ExitReason] = None
    level: Optional[float] = None


class PrecisionTrendStructureV2:
    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.cfg = config or StrategyConfig()

    # ---------- Indicators ----------

    @staticmethod
    def _wma(s: pd.Series, length: int) -> pd.Series:
        w = np.arange(1, length + 1, dtype=float)
        return s.rolling(length).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

    @classmethod
    def _hma(cls, s: pd.Series, length: int) -> pd.Series:
        half = max(1, length // 2)
        root = max(1, int(round(sqrt(length))))
        raw = 2.0 * cls._wma(s, half) - cls._wma(s, length)
        return cls._wma(raw, root)

    @staticmethod
    def _tr(df: pd.DataFrame) -> pd.Series:
        pc = df["close"].shift(1)
        return pd.concat(
            [(df["high"] - df["low"]).abs(),
             (df["high"] - pc).abs(),
             (df["low"] - pc).abs()],
            axis=1,
        ).max(axis=1)

    @staticmethod
    def _rma(s: pd.Series, length: int) -> pd.Series:
        return s.ewm(alpha=1.0 / length, adjust=False).mean()

    @classmethod
    def _atr(cls, df: pd.DataFrame, length: int) -> pd.Series:
        return cls._rma(cls._tr(df), length)

    @classmethod
    def _dmi_adx(cls, df: pd.DataFrame, dmi_len: int, adx_len: int):
        up = df["high"].diff()
        down = -df["low"].diff()
        plus_dm = pd.Series(
            np.where((up > down) & (up > 0), up, 0.0),
            index=df.index, dtype=float)
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0),
            index=df.index, dtype=float)
        atr = cls._rma(cls._tr(df), dmi_len).replace(0, np.nan)
        pdi = 100.0 * cls._rma(plus_dm, dmi_len) / atr
        mdi = 100.0 * cls._rma(minus_dm, dmi_len) / atr
        denom = (pdi + mdi).replace(0, np.nan)
        dx = 100.0 * (pdi - mdi).abs() / denom
        adx = cls._rma(dx.fillna(0.0), adx_len)
        return pdi, mdi, adx

    @classmethod
    def _chop(cls, df: pd.DataFrame, length: int) -> pd.Series:
        tr_sum = cls._tr(df).rolling(length).sum()
        hh = df["high"].rolling(length).max()
        ll = df["low"].rolling(length).min()
        rng = (hh - ll).replace(0, np.nan)
        ratio = (tr_sum / rng).clip(lower=1.0)
        return (100.0 * np.log10(ratio) / np.log10(float(length))).clip(0, 100)

    @staticmethod
    def _adx_score(adx: float) -> float:
        # Continuous, no threshold cliff.
        return float(np.clip((adx - 10.0) / 20.0 * 50.0, 0.0, 50.0))

    @staticmethod
    def _chop_score(chop: float) -> float:
        return float(np.clip((62.0 - chop) / 17.0 * 50.0, 0.0, 50.0))

    # ---------- 4H / 1H ----------

    def trend_state_4h(self, df4h: pd.DataFrame) -> TrendState:
        if len(df4h) < 70:
            return TrendState(Trend.NEUTRAL, np.nan, np.nan, np.nan, np.nan, 0)

        d = df4h.copy()
        d["ema20"] = d["close"].ewm(span=self.cfg.ema_fast_len, adjust=False).mean()
        d["ema50"] = d["close"].ewm(span=self.cfg.ema_slow_len, adjust=False).mean()
        d["ema20_slope"] = d["ema20"] - d["ema20"].shift(1)
        d["hma16"] = self._hma(d["close"], self.cfg.hma_len)
        d["hma_slope"] = d["hma16"] - d["hma16"].shift(1)
        d["hma_state"] = np.select([d["hma_slope"] > 0, d["hma_slope"] < 0], [1, -1], 0)

        r = d.iloc[-1]
        prev = d.iloc[-2]
        bull_ema = r["ema20"] > r["ema50"] and r["ema20_slope"] > 0
        bear_ema = r["ema20"] < r["ema50"] and r["ema20_slope"] < 0
        hma_up = int(r["hma_state"]) > 0
        hma_down = int(r["hma_state"]) < 0

        # HMA flip against the EMA trend is a warning first, not an instant opposite bias.
        warning = False
        if bull_ema and hma_up:
            trend = Trend.BULL
        elif bear_ema and hma_down:
            trend = Trend.BEAR
        elif bull_ema and hma_down and int(prev["hma_state"]) > 0:
            trend = Trend.NEUTRAL
            warning = True
        elif bear_ema and hma_up and int(prev["hma_state"]) < 0:
            trend = Trend.NEUTRAL
            warning = True
        else:
            trend = Trend.NEUTRAL

        return TrendState(
            trend, float(r["ema20"]), float(r["ema50"]),
            float(r["ema20_slope"]), float(r["hma16"]), int(r["hma_state"]), warning
        )

    def quality_state_1h(self, df1h: pd.DataFrame) -> QualityState:
        if len(df1h) < 60:
            return QualityState(0.0, 0.0, 100.0, 0.0, 0.0)
        pdi, mdi, adx = self._dmi_adx(df1h, self.cfg.dmi_len, self.cfg.adx_len)
        chop = self._chop(df1h, self.cfg.chop_len)
        a, c = float(adx.iloc[-1]), float(chop.iloc[-1])
        q = self._adx_score(a) + self._chop_score(c)
        return QualityState(float(q), a, c, float(pdi.iloc[-1]), float(mdi.iloc[-1]))

    # ---------- Structure ----------

    @staticmethod
    def _pivots(series: pd.Series, left: int, right: int, high_mode: bool):
        vals = series.to_numpy(dtype=float)
        out = []
        for i in range(left, len(vals) - right):
            w = vals[i-left:i+right+1]
            v = vals[i]
            if not np.isfinite(v):
                continue
            if high_mode and v >= np.nanmax(w):
                out.append((i, float(v)))
            elif not high_mode and v <= np.nanmin(w):
                out.append((i, float(v)))
        return out

    def _structure(self, d: pd.DataFrame) -> dict:
        maj_h = self._pivots(d["high"], self.cfg.major_left, self.cfg.major_right, True)
        maj_l = self._pivots(d["low"], self.cfg.major_left, self.cfg.major_right, False)
        mic_h = self._pivots(d["high"], self.cfg.micro_left, self.cfg.micro_right, True)
        mic_l = self._pivots(d["low"], self.cfg.micro_left, self.cfg.micro_right, False)

        bull = (len(maj_h) >= 2 and len(maj_l) >= 2
                and maj_h[-1][1] > maj_h[-2][1]
                and maj_l[-1][1] > maj_l[-2][1])
        bear = (len(maj_h) >= 2 and len(maj_l) >= 2
                and maj_h[-1][1] < maj_h[-2][1]
                and maj_l[-1][1] < maj_l[-2][1])

        return {"maj_h": maj_h, "maj_l": maj_l, "mic_h": mic_h, "mic_l": mic_l,
                "bull": bull, "bear": bear}

    @staticmethod
    def _close_location(row: pd.Series, bull: bool) -> float:
        rng = max(float(row["high"] - row["low"]), 1e-12)
        if bull:
            return float((row["close"] - row["low"]) / rng)
        return float((row["high"] - row["close"]) / rng)

    def _find_setup(self, d: pd.DataFrame, side: Side, atr: float, ctx: dict):
        """
        Search the bars BEFORE the current trigger for a valid location/setup.
        This enforces Location -> Armed -> Trigger instead of BOS-first logic.
        """
        end = len(d) - 1
        start = max(5, end - self.cfg.setup_lookback_bars)

        maj_h, maj_l = ctx["maj_h"], ctx["maj_l"]
        mic_h, mic_l = ctx["mic_h"], ctx["mic_l"]

        if side == Side.LONG:
            last_major_low = maj_l[-1][1] if maj_l else None
            last_major_high = maj_h[-1][1] if maj_h else None

            for i in range(end - 1, start - 1, -1):
                r = d.iloc[i]

                # 1) Pullback continuation near a confirmed HL / EMA20 value area.
                near_hl = last_major_low is not None and abs(float(r["low"]) - last_major_low) <= self.cfg.location_atr * atr
                near_ema = abs(float(r["close"]) - float(r["ema20"])) <= self.cfg.location_atr * atr
                if ctx["bull"] and (near_hl or near_ema):
                    return SetupType.PULLBACK, float(last_major_low or r["low"]), i

                # 2) Liquidity sweep below recent micro/major low, then reclaim close.
                ref_lows = [x[1] for x in mic_l if x[0] < i]
                if last_major_low is not None:
                    ref_lows.append(last_major_low)
                if ref_lows:
                    ref = ref_lows[-1]
                    if float(r["low"]) < ref and float(r["close"]) > ref:
                        return SetupType.SWEEP, float(ref), i

                # 3) Breakout retest of prior resistance after a close above it.
                if last_major_high is not None:
                    recent = d.iloc[max(0, i-4):i+1]
                    broke_before = (recent["close"] > last_major_high).any()
                    retest = float(r["low"]) <= last_major_high + self.cfg.retest_atr * atr and float(r["close"]) >= last_major_high
                    if broke_before and retest:
                        return SetupType.BREAKOUT_RETEST, float(last_major_high), i

        else:
            last_major_high = maj_h[-1][1] if maj_h else None
            last_major_low = maj_l[-1][1] if maj_l else None

            for i in range(end - 1, start - 1, -1):
                r = d.iloc[i]
                near_lh = last_major_high is not None and abs(float(r["high"]) - last_major_high) <= self.cfg.location_atr * atr
                near_ema = abs(float(r["close"]) - float(r["ema20"])) <= self.cfg.location_atr * atr
                if ctx["bear"] and (near_lh or near_ema):
                    return SetupType.PULLBACK, float(last_major_high or r["high"]), i

                ref_highs = [x[1] for x in mic_h if x[0] < i]
                if last_major_high is not None:
                    ref_highs.append(last_major_high)
                if ref_highs:
                    ref = ref_highs[-1]
                    if float(r["high"]) > ref and float(r["close"]) < ref:
                        return SetupType.SWEEP, float(ref), i

                if last_major_low is not None:
                    recent = d.iloc[max(0, i-4):i+1]
                    broke_before = (recent["close"] < last_major_low).any()
                    retest = float(r["high"]) >= last_major_low - self.cfg.retest_atr * atr and float(r["close"]) <= last_major_low
                    if broke_before and retest:
                        return SetupType.BREAKOUT_RETEST, float(last_major_low), i

        return None

    def _trigger(self, d: pd.DataFrame, side: Side, setup_index: int, atr: float, ctx: dict):
        row = d.iloc[-1]
        prev = d.iloc[-2]
        body_atr = abs(float(row["close"] - row["open"])) / atr
        if body_atr < self.cfg.min_trigger_body_atr or body_atr > self.cfg.max_trigger_body_atr:
            return None

        close_loc = self._close_location(row, side == Side.LONG)
        if close_loc < self.cfg.min_close_location:
            return None

        if side == Side.LONG:
            candidates = [p for p in ctx["mic_h"] if setup_index < p[0] < len(d)-1]
            if not candidates:
                # local trigger high from bars after setup
                local = d.iloc[setup_index+1:-1]["high"]
                if local.empty:
                    return None
                trigger_level = float(local.max())
            else:
                trigger_level = float(candidates[-1][1])

            crossed = float(row["close"]) > trigger_level and float(prev["close"]) <= trigger_level
            if not crossed:
                return None
            chase = (float(row["close"]) - trigger_level) / atr
            if chase > self.cfg.max_chase_atr:
                return None
            return "MICRO_CHOCH_RECLAIM", trigger_level

        candidates = [p for p in ctx["mic_l"] if setup_index < p[0] < len(d)-1]
        if not candidates:
            local = d.iloc[setup_index+1:-1]["low"]
            if local.empty:
                return None
            trigger_level = float(local.min())
        else:
            trigger_level = float(candidates[-1][1])

        crossed = float(row["close"]) < trigger_level and float(prev["close"]) >= trigger_level
        if not crossed:
            return None
        chase = (trigger_level - float(row["close"])) / atr
        if chase > self.cfg.max_chase_atr:
            return None
        return "MICRO_CHOCH_RECLAIM", trigger_level

    def _room_pct(self, price: float, side: Side, ctx: dict) -> float:
        if price <= 0:
            return 0.0
        if side == Side.LONG:
            resistance = [v for _, v in ctx["maj_h"] if v > price]
            if not resistance:
                return 9.99
            return (min(resistance) - price) / price
        support = [v for _, v in ctx["maj_l"] if v < price]
        if not support:
            return 9.99
        return (price - max(support)) / price

    # ---------- Entry ----------

    def generate_entry(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        if has_open_position or len(df15) < 90:
            return None

        trend = self.trend_state_4h(df4h)
        if trend.trend == Trend.NEUTRAL:
            return None

        quality = self.quality_state_1h(df1h)
        if quality.q < self.cfg.min_trend_quality:
            return None

        side = Side.LONG if trend.trend == Trend.BULL else Side.SHORT

        d = df15.copy()
        d["atr"] = self._atr(d, self.cfg.atr_len)
        d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
        row = d.iloc[-1]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None

        ctx = self._structure(d)
        setup = self._find_setup(d, side, atr, ctx)
        if setup is None:
            return None
        setup_type, structure_level, setup_index = setup

        trigger = self._trigger(d, side, setup_index, atr, ctx)
        if trigger is None:
            return None
        trigger_name, trigger_level = trigger

        entry = float(row["close"])
        room = self._room_pct(entry, side, ctx)
        if room < self.cfg.min_room_pct:
            return None

        if side == Side.LONG:
            sl = entry * (1.0 - self.cfg.stop_loss_pct)
            tp = entry * (1.0 + self.cfg.final_take_profit_pct)
        else:
            sl = entry * (1.0 + self.cfg.stop_loss_pct)
            tp = entry * (1.0 - self.cfg.final_take_profit_pct)

        return EntrySignal(
            side=side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=trend.trend,
            q_1h=quality.q,
            adx_1h=quality.adx,
            chop_1h=quality.chop,
            setup=setup_type,
            trigger=trigger_name,
            room_pct=room,
            atr15=atr,
            structure_level=structure_level,
            reason=f"{setup_type.value} -> {trigger_name} | room={room*100:.2f}%",
        )

    # ---------- Management ----------

    def locked_stop(self, side: Side, entry: float, best_price: float):
        if side == Side.LONG:
            favorable = best_price / entry - 1.0
            if favorable >= self.cfg.target2_trigger_pct:
                return entry * (1.0 + self.cfg.target2_lock_pct), 2
            if favorable >= self.cfg.target1_trigger_pct:
                return entry * (1.0 + self.cfg.target1_lock_pct), 1
            return entry * (1.0 - self.cfg.stop_loss_pct), 0

        favorable = entry / best_price - 1.0
        if favorable >= self.cfg.target2_trigger_pct:
            return entry * (1.0 - self.cfg.target2_lock_pct), 2
        if favorable >= self.cfg.target1_trigger_pct:
            return entry * (1.0 - self.cfg.target1_lock_pct), 1
        return entry * (1.0 + self.cfg.stop_loss_pct), 0

    def evaluate_structure_exit(
        self,
        side: Side,
        df4h: pd.DataFrame,
        df15: pd.DataFrame,
    ) -> StructureExit:
        t = self.trend_state_4h(df4h)
        if side == Side.LONG and t.trend == Trend.BEAR:
            return StructureExit(True, ExitReason.HTF_TREND_INVALIDATION)
        if side == Side.SHORT and t.trend == Trend.BULL:
            return StructureExit(True, ExitReason.HTF_TREND_INVALIDATION)

        if len(df15) < 60:
            return StructureExit(False)

        d = df15.copy()
        d["atr"] = self._atr(d, self.cfg.atr_len)
        atr = float(d["atr"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return StructureExit(False)

        ctx = self._structure(d)
        n = max(1, self.cfg.invalidation_confirm_bars)

        if side == Side.LONG and ctx["mic_l"] and ctx["maj_l"]:
            micro = ctx["mic_l"][-1][1]
            major = ctx["maj_l"][-1][1]
            recent = d.iloc[-n:]
            weak_break = (recent["close"] < micro).all()
            major_loss = float(d["close"].iloc[-1]) < major
            if weak_break and major_loss:
                return StructureExit(True, ExitReason.STRUCTURE_INVALIDATION, major)

        if side == Side.SHORT and ctx["mic_h"] and ctx["maj_h"]:
            micro = ctx["mic_h"][-1][1]
            major = ctx["maj_h"][-1][1]
            recent = d.iloc[-n:]
            weak_break = (recent["close"] > micro).all()
            major_loss = float(d["close"].iloc[-1]) > major
            if weak_break and major_loss:
                return StructureExit(True, ExitReason.STRUCTURE_INVALIDATION, major)

        return StructureExit(False)


# Backward-compatible alias for main.py / other imports.
MTFStructureStrategy = PrecisionTrendStructureV2
