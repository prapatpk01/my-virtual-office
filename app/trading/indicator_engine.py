"""Adaptive SMC v14 indicator engine.

Computes only the evidence required by the v14 state machine:
4H EMA trend, 1H liquidity/structure, and 15M OB/FVG/price-action entry data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numpy as np

ENGINE_SCHEMA = "adaptive-smc-v14-structure-v1"


def _v(candle: Any, name: str, index: int) -> float:
    value = getattr(candle, name, None)
    if value is None and isinstance(candle, dict):
        value = candle.get(name)
    if value is None and isinstance(candle, (list, tuple)) and len(candle) > index:
        value = candle[index]
    return float(value or 0.0)


def _series(candles: List[Any], name: str, index: int) -> List[float]:
    return [_v(c, name, index) for c in candles]


def ema(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    output = [float(values[0])]
    for value in values[1:]:
        output.append(alpha * float(value) + (1.0 - alpha) * output[-1])
    return output


def atr(candles: List[Any], length: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    tr = [highs[0] - lows[0]]
    for index in range(1, len(candles)):
        tr.append(max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        ))
    return float(np.mean(tr[-length:]))


def _pivots(values: List[float], high: bool, left: int = 2, right: int = 2) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for index in range(left, len(values) - right):
        window = values[index - left:index + right + 1]
        value = values[index]
        if high and value == max(window) and window.count(value) == 1:
            points.append((index, value))
        if not high and value == min(window) and window.count(value) == 1:
            points.append((index, value))
    return points


def _last_two(points: List[Tuple[int, float]], fallback_a: float, fallback_b: float) -> Tuple[Tuple[int, float], Tuple[int, float]]:
    if len(points) >= 2:
        return points[-2], points[-1]
    if len(points) == 1:
        return (-1, fallback_a), points[-1]
    return (-2, fallback_a), (-1, fallback_b)


def _detect_fvg(highs: List[float], lows: List[float]) -> Dict[str, Any]:
    bullish = False
    bearish = False
    low = high = 0.0
    age = 999
    for i in range(max(2, len(highs) - 12), len(highs)):
        if lows[i] > highs[i - 2]:
            bullish, bearish = True, False
            low, high = highs[i - 2], lows[i]
            age = len(highs) - 1 - i
        if highs[i] < lows[i - 2]:
            bearish, bullish = True, False
            low, high = highs[i], lows[i - 2]
            age = len(highs) - 1 - i
    return {"bullish": bullish, "bearish": bearish, "low": low, "high": high, "age": age}


def _detect_order_block(opens: List[float], highs: List[float], lows: List[float], closes: List[float], atr_value: float) -> Dict[str, Any]:
    bullish = bearish = False
    low = high = 0.0
    age = 999
    start = max(2, len(closes) - 16)
    for i in range(start, len(closes)):
        displacement_up = closes[i] > highs[i - 1] and (closes[i] - opens[i]) >= 0.45 * atr_value
        displacement_down = closes[i] < lows[i - 1] and (opens[i] - closes[i]) >= 0.45 * atr_value
        if displacement_up:
            for j in range(i - 1, max(-1, i - 5), -1):
                if closes[j] < opens[j]:
                    bullish, bearish = True, False
                    low, high = lows[j], highs[j]
                    age = len(closes) - 1 - j
                    break
        if displacement_down:
            for j in range(i - 1, max(-1, i - 5), -1):
                if closes[j] > opens[j]:
                    bearish, bullish = True, False
                    low, high = lows[j], highs[j]
                    age = len(closes) - 1 - j
                    break
    return {"bullish": bullish, "bearish": bearish, "low": low, "high": high, "age": age}


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 80:
        return {}

    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    atr_value = max(atr(candles), closes[-1] * 0.0005)

    high_points = _pivots(highs[-80:], True)
    low_points = _pivots(lows[-80:], False)
    (ph_i, previous_high), (lh_i, last_high) = _last_two(high_points, max(highs[-40:-20]), max(highs[-20:-1]))
    (pl_i, previous_low), (ll_i, last_low) = _last_two(low_points, min(lows[-40:-20]), min(lows[-20:-1]))

    higher_high = last_high > previous_high
    higher_low = last_low > previous_low
    lower_high = last_high < previous_high
    lower_low = last_low < previous_low
    structure = "BULL" if higher_high and higher_low else "BEAR" if lower_high and lower_low else "MIXED"

    sell_side_sweep = lows[-1] < last_low and closes[-1] > last_low
    buy_side_sweep = highs[-1] > last_high and closes[-1] < last_high
    recent_sell_sweep = any(lows[i] < last_low and closes[i] > last_low for i in range(max(0, len(closes) - 5), len(closes)))
    recent_buy_sweep = any(highs[i] > last_high and closes[i] < last_high for i in range(max(0, len(closes) - 5), len(closes)))

    bullish_choch = closes[-1] > last_high and (structure in {"BEAR", "MIXED"} or recent_sell_sweep)
    bearish_choch = closes[-1] < last_low and (structure in {"BULL", "MIXED"} or recent_buy_sweep)
    bullish_bos = closes[-1] > last_high and closes[-2] <= last_high
    bearish_bos = closes[-1] < last_low and closes[-2] >= last_low

    fvg = _detect_fvg(highs, lows)
    order_block = _detect_order_block(opens, highs, lows, closes, atr_value)

    body = abs(closes[-1] - opens[-1])
    lower_wick = min(opens[-1], closes[-1]) - lows[-1]
    upper_wick = highs[-1] - max(opens[-1], closes[-1])
    bull_engulf = closes[-1] > opens[-1] and closes[-2] < opens[-2] and closes[-1] >= opens[-2] and opens[-1] <= closes[-2]
    bear_engulf = closes[-1] < opens[-1] and closes[-2] > opens[-2] and closes[-1] <= opens[-2] and opens[-1] >= closes[-2]
    bull_pin = closes[-1] > opens[-1] and lower_wick >= max(body * 1.8, atr_value * 0.15)
    bear_pin = closes[-1] < opens[-1] and upper_wick >= max(body * 1.8, atr_value * 0.15)
    break_high = closes[-1] > highs[-2]
    break_low = closes[-1] < lows[-2]
    volume_ratio = volumes[-1] / max(float(np.mean(volumes[-20:])), 1e-12)
    bull_volume = closes[-1] > opens[-1] and volume_ratio >= 1.2
    bear_volume = closes[-1] < opens[-1] and volume_ratio >= 1.2

    zone_low = max(order_block["low"] if order_block["bullish"] else 0.0, fvg["low"] if fvg["bullish"] else 0.0)
    zone_high_candidates = [value for value in (
        order_block["high"] if order_block["bullish"] else 0.0,
        fvg["high"] if fvg["bullish"] else 0.0,
    ) if value > 0]
    bull_overlap_high = min(zone_high_candidates) if zone_high_candidates else 0.0
    bull_overlap = zone_low > 0 and bull_overlap_high > zone_low

    bear_zone_low_candidates = [value for value in (
        order_block["low"] if order_block["bearish"] else 0.0,
        fvg["low"] if fvg["bearish"] else 0.0,
    ) if value > 0]
    bear_overlap_low = max(bear_zone_low_candidates) if bear_zone_low_candidates else 0.0
    bear_zone_high_candidates = [value for value in (
        order_block["high"] if order_block["bearish"] else 0.0,
        fvg["high"] if fvg["bearish"] else 0.0,
    ) if value > 0]
    bear_overlap_high = min(bear_zone_high_candidates) if bear_zone_high_candidates else 0.0
    bear_overlap = bear_overlap_low > 0 and bear_overlap_high > bear_overlap_low

    return {
        "schema": ENGINE_SCHEMA,
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_open": opens[-2], "prev_high": highs[-2], "prev_low": lows[-2], "prev_close": closes[-2],
        "ema20": e20[-1], "ema50": e50[-1], "ema20_series": e20[-80:], "ema50_series": e50[-80:],
        "ema20_slope_atr": (e20[-1] - e20[-4]) / atr_value,
        "atr": atr_value, "volume": volumes[-1], "volume_ratio": volume_ratio,
        "last_swing_high": last_high, "previous_swing_high": previous_high,
        "last_swing_low": last_low, "previous_swing_low": previous_low,
        "higher_high": higher_high, "higher_low": higher_low, "lower_high": lower_high, "lower_low": lower_low,
        "structure": structure,
        "sell_side_sweep": sell_side_sweep, "buy_side_sweep": buy_side_sweep,
        "recent_sell_sweep": recent_sell_sweep, "recent_buy_sweep": recent_buy_sweep,
        "bullish_choch": bullish_choch, "bearish_choch": bearish_choch,
        "bullish_bos": bullish_bos, "bearish_bos": bearish_bos,
        "ob_bull": order_block["bullish"], "ob_bear": order_block["bearish"],
        "ob_low": order_block["low"], "ob_high": order_block["high"], "ob_age": order_block["age"],
        "fvg_bull": fvg["bullish"], "fvg_bear": fvg["bearish"],
        "fvg_low": fvg["low"], "fvg_high": fvg["high"], "fvg_age": fvg["age"],
        "bull_zone_low": zone_low if bull_overlap else (order_block["low"] if order_block["bullish"] else fvg["low"]),
        "bull_zone_high": bull_overlap_high if bull_overlap else (order_block["high"] if order_block["bullish"] else fvg["high"]),
        "bear_zone_low": bear_overlap_low if bear_overlap else (order_block["low"] if order_block["bearish"] else fvg["low"]),
        "bear_zone_high": bear_overlap_high if bear_overlap else (order_block["high"] if order_block["bearish"] else fvg["high"]),
        "bull_zone_overlap": bull_overlap, "bear_zone_overlap": bear_overlap,
        "bull_engulf": bull_engulf, "bear_engulf": bear_engulf,
        "bull_pin": bull_pin, "bear_pin": bear_pin,
        "break_high": break_high, "break_low": break_low,
        "bull_volume": bull_volume, "bear_volume": bear_volume,
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), compute(c1h), compute(c4h)
