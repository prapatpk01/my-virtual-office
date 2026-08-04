"""Sentinel X v2.3 context engine adapted for the HMA bot.

This module ports the non-visual algorithms that matter for automated entries:
confirmed pivots, adaptive S1/S2/R1/R2, structure, sweeps, location quality,
and a compact trend score.  It deliberately excludes Pine plots, labels,
tables and alerts.

All pivots are confirmed (right-hand bars have already closed), so the engine
never uses future-looking structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SentinelLocation:
    side: str
    score: float
    zone: str
    s1: Optional[float]
    s2: Optional[float]
    r1: Optional[float]
    r2: Optional[float]
    near_s1: bool
    near_s2: bool
    near_r1: bool
    near_r2: bool
    demand: bool
    supply: bool
    sweep: bool
    structure: str
    ema_context: bool
    room_atr: float
    level_strength: float
    reason: str


@dataclass(frozen=True)
class SentinelContext:
    trend_score: float
    trend_class: str
    location: SentinelLocation


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(
        lambda values: float(np.dot(values, weights) / weights.sum()), raw=True
    )


def _hma(series: pd.Series, length: int) -> pd.Series:
    half = max(1, length // 2)
    root = max(1, int(round(length ** 0.5)))
    return _wma(2.0 * _wma(series, half) - _wma(series, length), root)


def _true_range(df: pd.DataFrame) -> pd.Series:
    previous = df["close"].shift(1)
    return pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - previous).abs(),
            (df["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1.0 / length, adjust=False).mean()


def _confirmed_pivots(
    df: pd.DataFrame, left: int = 4, right: int = 4
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return confirmed pivot highs/lows as (position, price)."""
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    pivot_highs: list[tuple[int, float]] = []
    pivot_lows: list[tuple[int, float]] = []
    if len(df) < left + right + 3:
        return pivot_highs, pivot_lows

    for index in range(left, len(df) - right):
        high_window = highs[index - left : index + right + 1]
        low_window = lows[index - left : index + right + 1]
        if np.isfinite(highs[index]) and highs[index] >= np.nanmax(high_window):
            pivot_highs.append((index, float(highs[index])))
        if np.isfinite(lows[index]) and lows[index] <= np.nanmin(low_window):
            pivot_lows.append((index, float(lows[index])))
    return pivot_highs, pivot_lows


def _last_two(values: list[tuple[int, float]]) -> tuple[Optional[float], Optional[float]]:
    latest = values[-1][1] if values else None
    previous = values[-2][1] if len(values) >= 2 else None
    return latest, previous


def _nearest_above(price: float, candidates: list[float]) -> list[float]:
    return sorted({float(value) for value in candidates if np.isfinite(value) and value > price})


def _nearest_below(price: float, candidates: list[float]) -> list[float]:
    return sorted(
        {float(value) for value in candidates if np.isfinite(value) and value < price},
        reverse=True,
    )


def _merge_levels(first: Optional[float], second: Optional[float], atr: float, merge_atr: float):
    if first is None or second is None:
        return first, second, False
    if abs(second - first) <= atr * merge_atr:
        return (first + second) / 2.0, None, True
    return first, second, False


def _level_strength(
    level: Optional[float], local: list[float], htf1: list[float], htf2: list[float], atr: float
) -> float:
    if level is None or atr <= 0:
        return 0.0
    tolerance = atr * 0.30
    score = 35.0
    for value, weight in [
        *((value, 18.0 if index == 0 else 14.0) for index, value in enumerate(local[:2])),
        *((value, 18.0 if index == 0 else 15.0) for index, value in enumerate(htf1[:1] + htf2[:1])),
    ]:
        if abs(level - value) <= tolerance:
            score += weight
    return _clamp(score, 0.0, 100.0)


def trend_score_4h(df4h: pd.DataFrame, side: str) -> float:
    if df4h is None or len(df4h) < 60:
        return 0.0
    close = df4h["close"].astype(float)
    atr = float(_atr(df4h, 14).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return 0.0
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    hma16 = _hma(close, 16)

    current = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    h16 = float(hma16.iloc[-1])
    e20_slope = (e20 - float(ema20.iloc[-5])) / atr
    hma_slope = (h16 - float(hma16.iloc[-4])) / atr
    separation = (e20 - e50) / atr

    sign = 1.0 if side == "long" else -1.0
    score = 0.0
    score += 25.0 if sign * (e20 - e50) > 0 else 0.0
    score += 20.0 if sign * (current - e20) > 0 else 0.0
    score += 15.0 if sign * (current - h16) > 0 else 0.0
    score += _clamp(sign * e20_slope * 18.0, 0.0, 15.0)
    score += _clamp(sign * hma_slope * 14.0, 0.0, 12.0)
    score += _clamp(sign * separation * 8.0, 0.0, 13.0)
    return _clamp(score, 0.0, 100.0)


def _structure_state(df15: pd.DataFrame):
    highs, lows = _confirmed_pivots(df15, 4, 4)
    last_high, previous_high = _last_two(highs)
    last_low, previous_low = _last_two(lows)
    hh = last_high is not None and previous_high is not None and last_high > previous_high
    lh = last_high is not None and previous_high is not None and last_high < previous_high
    hl = last_low is not None and previous_low is not None and last_low > previous_low
    ll = last_low is not None and previous_low is not None and last_low < previous_low
    if hh and hl:
        label = "HH/HL"
    elif lh and ll:
        label = "LH/LL"
    elif hh or hl:
        label = "BULLISH"
    elif lh or ll:
        label = "BEARISH"
    else:
        label = "MIXED"
    return label, highs, lows, last_high, last_low


def build_context(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    side: str,
    zone_atr: float = 0.45,
    merge_atr: float = 0.65,
) -> SentinelContext:
    """Build the Sentinel location score used before HMA 5M execution."""
    side = side.lower()
    d15 = df15.copy()
    atr_series = _atr(d15, 14)
    atr = float(atr_series.iloc[-1])
    price = float(d15["close"].iloc[-1])
    ema20 = _ema(d15["close"].astype(float), 20)
    current_ema20 = float(ema20.iloc[-1])

    structure, local_highs, local_lows, last_high, last_low = _structure_state(d15)
    h1_highs, h1_lows = _confirmed_pivots(df1h, 3, 3)
    h4_highs, h4_lows = _confirmed_pivots(df4h, 3, 3)

    local_high_values = [value for _, value in local_highs[-2:]]
    local_low_values = [value for _, value in local_lows[-2:]]
    h1_high_values = [value for _, value in h1_highs[-1:]]
    h1_low_values = [value for _, value in h1_lows[-1:]]
    h4_high_values = [value for _, value in h4_highs[-1:]]
    h4_low_values = [value for _, value in h4_lows[-1:]]

    above = _nearest_above(price, local_high_values + h1_high_values + h4_high_values)
    below = _nearest_below(price, local_low_values + h1_low_values + h4_low_values)
    r1_raw = above[0] if above else None
    r2_raw = above[1] if len(above) > 1 else None
    s1_raw = below[0] if below else None
    s2_raw = below[1] if len(below) > 1 else None
    r1, r2, merged_r = _merge_levels(r1_raw, r2_raw, atr, merge_atr)
    s1, s2, merged_s = _merge_levels(s1_raw, s2_raw, atr, merge_atr)

    near_s1 = s1 is not None and abs(price - s1) <= atr * zone_atr
    near_s2 = s2 is not None and abs(price - s2) <= atr * zone_atr
    near_r1 = r1 is not None and abs(price - r1) <= atr * zone_atr
    near_r2 = r2 is not None and abs(price - r2) <= atr * zone_atr

    latest = d15.iloc[-1]
    previous = d15.iloc[-2]
    candle_range = max(float(latest["high"] - latest["low"]), 1e-12)
    close_location = (float(latest["close"] - latest["low"])) / candle_range
    body_atr = abs(float(latest["close"] - latest["open"])) / max(atr, 1e-12)

    sweep_long = (
        last_low is not None
        and float(latest["low"]) < last_low - atr * 0.04
        and float(latest["close"]) > last_low
        and close_location > 0.62
        and float(latest["close"]) > float(latest["open"])
    )
    sweep_short = (
        last_high is not None
        and float(latest["high"]) > last_high + atr * 0.04
        and float(latest["close"]) < last_high
        and close_location < 0.38
        and float(latest["close"]) < float(latest["open"])
    )

    demand = (
        (near_s1 or near_s2)
        and float(latest["close"]) > float(latest["open"])
        and close_location >= 0.58
        and body_atr >= 0.10
    )
    supply = (
        (near_r1 or near_r2)
        and float(latest["close"]) < float(latest["open"])
        and close_location <= 0.42
        and body_atr >= 0.10
    )

    ema_long = (
        float(latest["low"]) <= current_ema20 + atr * 0.15
        and float(latest["close"]) >= current_ema20
    )
    ema_short = (
        float(latest["high"]) >= current_ema20 - atr * 0.15
        and float(latest["close"]) <= current_ema20
    )

    trend_score = trend_score_4h(df4h, side)
    trend_class = "STRONG" if trend_score >= 85.0 else "MODERATE" if trend_score >= 70.0 else "WEAK"

    if side == "long":
        level = s2 if near_s2 else s1 if near_s1 else None
        level_name = "S2" if near_s2 else "S1" if near_s1 else "NONE"
        level_raw_strength = _level_strength(
            level, local_low_values, h1_low_values, h4_low_values, atr
        )
        score = 25.0 if near_s2 else 15.0 if near_s1 else 0.0
        score += 20.0 if demand else 0.0
        score += 20.0 if sweep_long else 0.0
        score += 10.0 if structure in ("HH/HL", "BULLISH") else 0.0
        score += 10.0 if ema_long else 0.0
        score += min(10.0, level_raw_strength / 10.0) if level is not None else 0.0
        room = (r1 - price) / atr if r1 is not None else 9.0
        zone_allowed = near_s2 or (trend_class == "STRONG" and near_s1)
        reason_bits = [level_name]
        if demand:
            reason_bits.append("DEMAND")
        if sweep_long:
            reason_bits.append("SWEEP")
        if structure in ("HH/HL", "BULLISH"):
            reason_bits.append(structure)
        if ema_long:
            reason_bits.append("EMA20_PULLBACK")
        location = SentinelLocation(
            side=side, score=_clamp(score, 0.0, 100.0), zone=level_name,
            s1=s1, s2=s2, r1=r1, r2=r2,
            near_s1=near_s1, near_s2=near_s2, near_r1=near_r1, near_r2=near_r2,
            demand=demand, supply=False, sweep=sweep_long, structure=structure,
            ema_context=ema_long, room_atr=room, level_strength=level_raw_strength,
            reason=" + ".join(reason_bits) + ("" if zone_allowed else " | ZONE_NOT_ALLOWED"),
        )
    else:
        level = r2 if near_r2 else r1 if near_r1 else None
        level_name = "R2" if near_r2 else "R1" if near_r1 else "NONE"
        level_raw_strength = _level_strength(
            level, local_high_values, h1_high_values, h4_high_values, atr
        )
        score = 25.0 if near_r2 else 15.0 if near_r1 else 0.0
        score += 20.0 if supply else 0.0
        score += 20.0 if sweep_short else 0.0
        score += 10.0 if structure in ("LH/LL", "BEARISH") else 0.0
        score += 10.0 if ema_short else 0.0
        score += min(10.0, level_raw_strength / 10.0) if level is not None else 0.0
        room = (price - s1) / atr if s1 is not None else 9.0
        zone_allowed = near_r2 or (trend_class == "STRONG" and near_r1)
        reason_bits = [level_name]
        if supply:
            reason_bits.append("SUPPLY")
        if sweep_short:
            reason_bits.append("SWEEP")
        if structure in ("LH/LL", "BEARISH"):
            reason_bits.append(structure)
        if ema_short:
            reason_bits.append("EMA20_REJECT")
        location = SentinelLocation(
            side=side, score=_clamp(score, 0.0, 100.0), zone=level_name,
            s1=s1, s2=s2, r1=r1, r2=r2,
            near_s1=near_s1, near_s2=near_s2, near_r1=near_r1, near_r2=near_r2,
            demand=False, supply=supply, sweep=sweep_short, structure=structure,
            ema_context=ema_short, room_atr=room, level_strength=level_raw_strength,
            reason=" + ".join(reason_bits) + ("" if zone_allowed else " | ZONE_NOT_ALLOWED"),
        )

    return SentinelContext(
        trend_score=trend_score,
        trend_class=trend_class,
        location=location,
    )
