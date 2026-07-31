"""
MTF Structure Trend Bot Strategy
Architecture:
  Layer 1 — 4H trend direction: EMA20/EMA50 + EMA20 slope + HMA16 state
  Layer 2 — 1H quality: Q = ADX score + CHOP score, Q >= 55
  Layer 3 — 15M structure entry: HH/HL/LH/LL, sweep/rejection, micro BOS, chase gate

Position management:
  Initial SL 1.5%
  Final TP 1.5%
  T1 at +0.6% => lock +0.3%
  T2 at +1.0% => lock +0.7%
  Runner continues to final TP
  15M opposite micro-BOS/structure invalidation can exit early
  4H opposite trend invalidation can exit early

All indicator-driven decisions use closed candles.
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

    major_swing_left: int = 3
    major_swing_right: int = 3
    micro_swing_left: int = 2
    micro_swing_right: int = 2

    min_bos_body_atr: float = 0.15
    max_bos_body_atr: float = 1.20
    max_chase_atr: float = 0.75
    location_atr: float = 0.60
    min_entry_score: float = 60.0

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


@dataclass
class QualityState:
    q: float
    adx: float
    chop: float
    plus_di: float
    minus_di: float

    def dmi_aligned(self, side: Side) -> bool:
        return self.plus_di > self.minus_di if side == Side.LONG else self.minus_di > self.plus_di


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
    dmi_aligned: bool
    setup: str
    entry_score: float
    micro_bos_level: float
    atr15: float
    reason: str


@dataclass
class StructureExit:
    should_exit: bool
    reason: Optional[ExitReason] = None
    level: Optional[float] = None


class MTFStructureStrategy:
    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.cfg = config or StrategyConfig()

    @staticmethod
    def _wma(series: pd.Series, length: int) -> pd.Series:
        weights = np.arange(1, length + 1, dtype=float)
        return series.rolling(length).apply(
            lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
        )

    @classmethod
    def _hma(cls, series: pd.Series, length: int) -> pd.Series:
        half = max(1, length // 2)
        root = max(1, int(round(sqrt(length))))
        raw = 2.0 * cls._wma(series, half) - cls._wma(series, length)
        return cls._wma(raw, root)

    @staticmethod
    def _tr(df: pd.DataFrame) -> pd.Series:
        pc = df["close"].shift(1)
        return pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - pc).abs(),
                (df["low"] - pc).abs(),
            ],
            axis=1,
        ).max(axis=1)

    @staticmethod
    def _rma(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(alpha=1.0 / length, adjust=False).mean()

    @classmethod
    def _atr(cls, df: pd.DataFrame, length: int) -> pd.Series:
        return cls._rma(cls._tr(df), length)

    @classmethod
    def _dmi_adx(cls, df: pd.DataFrame, dmi_len: int, adx_len: int):
        up = df["high"].diff()
        down = -df["low"].diff()

        plus_dm = pd.Series(
            np.where((up > down) & (up > 0), up, 0.0), index=df.index, dtype=float
        )
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0), index=df.index, dtype=float
        )

        atr = cls._rma(cls._tr(df), dmi_len).replace(0, np.nan)
        plus_di = 100.0 * cls._rma(plus_dm, dmi_len) / atr
        minus_di = 100.0 * cls._rma(minus_dm, dmi_len) / atr
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / denom
        adx = cls._rma(dx.fillna(0.0), adx_len)
        return plus_di, minus_di, adx

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
        # Continuous, deliberately soft: ADX 10 -> 0, ADX 30 -> 50.
        return float(np.clip((adx - 10.0) / 20.0 * 50.0, 0.0, 50.0))

    @staticmethod
    def _chop_score(chop: float) -> float:
        # Continuous: CHOP 62 -> 0, CHOP 45 -> 50.
        return float(np.clip((62.0 - chop) / 17.0 * 50.0, 0.0, 50.0))

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema20"] = out["close"].ewm(span=self.cfg.ema_fast_len, adjust=False).mean()
        out["ema50"] = out["close"].ewm(span=self.cfg.ema_slow_len, adjust=False).mean()
        out["ema20_slope"] = out["ema20"] - out["ema20"].shift(1)
        out["hma16"] = self._hma(out["close"], self.cfg.hma_len)
        out["hma_slope"] = out["hma16"] - out["hma16"].shift(1)
        out["hma_state"] = np.select(
            [out["hma_slope"] > 0, out["hma_slope"] < 0], [1, -1], default=0
        )
        return out

    def trend_state_4h(self, df4h: pd.DataFrame) -> TrendState:
        if len(df4h) < max(self.cfg.ema_slow_len + 5, self.cfg.hma_len * 2):
            return TrendState(Trend.NEUTRAL, np.nan, np.nan, np.nan, np.nan, 0)

        d = self.add_trend_indicators(df4h)
        r = d.iloc[-1]

        if r["ema20"] > r["ema50"] and r["ema20_slope"] > 0 and int(r["hma_state"]) > 0:
            trend = Trend.BULL
        elif r["ema20"] < r["ema50"] and r["ema20_slope"] < 0 and int(r["hma_state"]) < 0:
            trend = Trend.BEAR
        else:
            trend = Trend.NEUTRAL

        return TrendState(
            trend=trend,
            ema20=float(r["ema20"]),
            ema50=float(r["ema50"]),
            ema20_slope=float(r["ema20_slope"]),
            hma16=float(r["hma16"]),
            hma_state=int(r["hma_state"]),
        )

    def quality_state_1h(self, df1h: pd.DataFrame) -> QualityState:
        if len(df1h) < 60:
            return QualityState(0.0, 0.0, 100.0, 0.0, 0.0)

        plus_di, minus_di, adx = self._dmi_adx(df1h, self.cfg.dmi_len, self.cfg.adx_len)
        chop = self._chop(df1h, self.cfg.chop_len)

        a = float(adx.iloc[-1])
        c = float(chop.iloc[-1])
        p = float(plus_di.iloc[-1])
        m = float(minus_di.iloc[-1])
        q = self._adx_score(a) + self._chop_score(c)

        return QualityState(q=float(q), adx=a, chop=c, plus_di=p, minus_di=m)

    @staticmethod
    def _confirmed_pivots(series: pd.Series, left: int, right: int, mode: str) -> list[tuple[int, float]]:
        vals = series.to_numpy(dtype=float)
        pivots: list[tuple[int, float]] = []
        if len(vals) < left + right + 1:
            return pivots
        for i in range(left, len(vals) - right):
            window = vals[i - left : i + right + 1]
            v = vals[i]
            if mode == "high":
                if np.isfinite(v) and v >= np.nanmax(window):
                    pivots.append((i, float(v)))
            else:
                if np.isfinite(v) and v <= np.nanmin(window):
                    pivots.append((i, float(v)))
        return pivots

    @staticmethod
    def _bull_rejection(row: pd.Series) -> bool:
        body = abs(float(row["close"] - row["open"]))
        lower = min(float(row["open"]), float(row["close"])) - float(row["low"])
        return row["close"] > row["open"] and lower >= max(body * 0.5, 1e-12)

    @staticmethod
    def _bear_rejection(row: pd.Series) -> bool:
        body = abs(float(row["close"] - row["open"]))
        upper = float(row["high"]) - max(float(row["open"]), float(row["close"]))
        return row["close"] < row["open"] and upper >= max(body * 0.5, 1e-12)

    def _structure_context(self, df15: pd.DataFrame) -> dict:
        major_highs = self._confirmed_pivots(
            df15["high"], self.cfg.major_swing_left, self.cfg.major_swing_right, "high"
        )
        major_lows = self._confirmed_pivots(
            df15["low"], self.cfg.major_swing_left, self.cfg.major_swing_right, "low"
        )
        micro_highs = self._confirmed_pivots(
            df15["high"], self.cfg.micro_swing_left, self.cfg.micro_swing_right, "high"
        )
        micro_lows = self._confirmed_pivots(
            df15["low"], self.cfg.micro_swing_left, self.cfg.micro_swing_right, "low"
        )

        bull_structure = (
            len(major_highs) >= 2 and len(major_lows) >= 2
            and major_highs[-1][1] > major_highs[-2][1]
            and major_lows[-1][1] > major_lows[-2][1]
        )
        bear_structure = (
            len(major_highs) >= 2 and len(major_lows) >= 2
            and major_highs[-1][1] < major_highs[-2][1]
            and major_lows[-1][1] < major_lows[-2][1]
        )

        return {
            "major_highs": major_highs,
            "major_lows": major_lows,
            "micro_highs": micro_highs,
            "micro_lows": micro_lows,
            "bull_structure": bull_structure,
            "bear_structure": bear_structure,
        }

    def generate_entry(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        if has_open_position or len(df15) < 80:
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
        prev = d.iloc[-2]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None

        ctx = self._structure_context(d)
        micro_highs = ctx["micro_highs"]
        micro_lows = ctx["micro_lows"]
        major_highs = ctx["major_highs"]
        major_lows = ctx["major_lows"]
        body_atr = abs(float(row["close"] - row["open"])) / atr

        if side == Side.LONG:
            if not micro_highs:
                return None
            bos_level = micro_highs[-1][1]
            micro_bos = float(row["close"]) > bos_level and float(prev["close"]) <= bos_level
            if not micro_bos or body_atr < self.cfg.min_bos_body_atr:
                return None

            chase = (float(row["close"]) - bos_level) / atr
            if chase > self.cfg.max_chase_atr or body_atr > self.cfg.max_bos_body_atr:
                return None

            score = 30.0  # mandatory micro BOS
            tags = ["MICRO_BOS"]

            if ctx["bull_structure"]:
                score += 20.0
                tags.append("HH_HL")

            last_low = major_lows[-1][1] if major_lows else None
            near_hl = last_low is not None and abs(float(row["low"]) - last_low) <= self.cfg.location_atr * atr
            if near_hl:
                score += 20.0
                tags.append("HL_LOCATION")

            ref_low = micro_lows[-1][1] if micro_lows else last_low
            sweep = (
                ref_low is not None
                and float(row["low"]) < ref_low
                and float(row["close"]) > ref_low
            )
            if sweep:
                score += 20.0
                tags.append("SWEEP")

            if self._bull_rejection(row):
                score += 15.0
                tags.append("REJECTION")

            if abs(float(row["close"]) - float(row["ema20"])) <= self.cfg.location_atr * atr:
                score += 10.0
                tags.append("EMA20_LOCATION")

            dmi_ok = quality.dmi_aligned(side)
            if dmi_ok:
                score += 5.0
                tags.append("DMI")

            if score < self.cfg.min_entry_score:
                return None

            entry = float(row["close"])
            setup = "SWEEP_RECLAIM" if sweep else ("HL_RECLAIM" if near_hl else "STRUCTURE_BOS")
            return EntrySignal(
                side=side,
                entry_price=entry,
                stop_loss=entry * (1.0 - self.cfg.stop_loss_pct),
                take_profit=entry * (1.0 + self.cfg.final_take_profit_pct),
                trend_4h=trend.trend,
                q_1h=quality.q,
                adx_1h=quality.adx,
                chop_1h=quality.chop,
                dmi_aligned=dmi_ok,
                setup=setup,
                entry_score=score,
                micro_bos_level=bos_level,
                atr15=atr,
                reason=" + ".join(tags),
            )

        # SHORT
        if not micro_lows:
            return None
        bos_level = micro_lows[-1][1]
        micro_bos = float(row["close"]) < bos_level and float(prev["close"]) >= bos_level
        if not micro_bos or body_atr < self.cfg.min_bos_body_atr:
            return None

        chase = (bos_level - float(row["close"])) / atr
        if chase > self.cfg.max_chase_atr or body_atr > self.cfg.max_bos_body_atr:
            return None

        score = 30.0
        tags = ["MICRO_BOS"]

        if ctx["bear_structure"]:
            score += 20.0
            tags.append("LH_LL")

        last_high = major_highs[-1][1] if major_highs else None
        near_lh = last_high is not None and abs(float(row["high"]) - last_high) <= self.cfg.location_atr * atr
        if near_lh:
            score += 20.0
            tags.append("LH_LOCATION")

        ref_high = micro_highs[-1][1] if micro_highs else last_high
        sweep = (
            ref_high is not None
            and float(row["high"]) > ref_high
            and float(row["close"]) < ref_high
        )
        if sweep:
            score += 20.0
            tags.append("SWEEP")

        if self._bear_rejection(row):
            score += 15.0
            tags.append("REJECTION")

        if abs(float(row["close"]) - float(row["ema20"])) <= self.cfg.location_atr * atr:
            score += 10.0
            tags.append("EMA20_LOCATION")

        dmi_ok = quality.dmi_aligned(side)
        if dmi_ok:
            score += 5.0
            tags.append("DMI")

        if score < self.cfg.min_entry_score:
            return None

        entry = float(row["close"])
        setup = "SWEEP_REJECT" if sweep else ("LH_REJECT" if near_lh else "STRUCTURE_BOS")
        return EntrySignal(
            side=side,
            entry_price=entry,
            stop_loss=entry * (1.0 + self.cfg.stop_loss_pct),
            take_profit=entry * (1.0 - self.cfg.final_take_profit_pct),
            trend_4h=trend.trend,
            q_1h=quality.q,
            adx_1h=quality.adx,
            chop_1h=quality.chop,
            dmi_aligned=dmi_ok,
            setup=setup,
            entry_score=score,
            micro_bos_level=bos_level,
            atr15=atr,
            reason=" + ".join(tags),
        )

    def evaluate_structure_exit(
        self,
        side: Side,
        df4h: pd.DataFrame,
        df15: pd.DataFrame,
    ) -> StructureExit:
        trend = self.trend_state_4h(df4h)

        if side == Side.LONG and trend.trend == Trend.BEAR:
            return StructureExit(True, ExitReason.HTF_TREND_INVALIDATION)
        if side == Side.SHORT and trend.trend == Trend.BULL:
            return StructureExit(True, ExitReason.HTF_TREND_INVALIDATION)

        if len(df15) < 40:
            return StructureExit(False)

        d = df15.copy()
        d["atr"] = self._atr(d, self.cfg.atr_len)
        row, prev = d.iloc[-1], d.iloc[-2]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return StructureExit(False)

        ctx = self._structure_context(d)
        body_atr = abs(float(row["close"] - row["open"])) / atr

        if side == Side.LONG and ctx["micro_lows"]:
            level = ctx["micro_lows"][-1][1]
            broke = float(row["close"]) < level and float(prev["close"]) >= level
            if broke and body_atr >= self.cfg.min_bos_body_atr:
                return StructureExit(True, ExitReason.STRUCTURE_INVALIDATION, level)

        if side == Side.SHORT and ctx["micro_highs"]:
            level = ctx["micro_highs"][-1][1]
            broke = float(row["close"]) > level and float(prev["close"]) <= level
            if broke and body_atr >= self.cfg.min_bos_body_atr:
                return StructureExit(True, ExitReason.STRUCTURE_INVALIDATION, level)

        return StructureExit(False)

    def locked_stop(self, side: Side, entry: float, best_price: float) -> tuple[float, int]:
        """
        Returns (stop_price, stage)
          stage 0 = initial -1.5%
          stage 1 = T1 hit (+0.6%), lock +0.3%
          stage 2 = T2 hit (+1.0%), lock +0.7%
        """
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
