"""Indicators and confirmed market structure for Adaptive Bot v13."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numpy as np


def _value(candle: Any, name: str, index: int) -> float:
    value = getattr(candle, name, None)
    if value is None and isinstance(candle, dict):
        value = candle.get(name)
    if value is None and isinstance(candle, (list, tuple)) and len(candle) > index:
        value = candle[index]
    return float(value or 0.0)


def _series(candles: List[Any], name: str, index: int) -> List[float]:
    return [_value(c, name, index) for c in candles]


def ema(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def atr(candles: List[Any], length: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    true_ranges = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return float(np.mean(true_ranges[-length:]))


def adx(candles: List[Any], length: int = 14) -> float:
    if len(candles) < length + 2:
        return 0.0
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    true_ranges: List[float] = []
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    tr_sum = max(sum(true_ranges[-length:]), 1e-12)
    plus_di = 100.0 * sum(plus_dm[-length:]) / tr_sum
    minus_di = 100.0 * sum(minus_dm[-length:]) / tr_sum
    return float(100.0 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-12))


def choppiness(candles: List[Any], length: int = 14) -> float:
    if len(candles) < length + 1:
        return 100.0
    window = candles[-length:]
    highs = _series(window, "high", 2)
    lows = _series(window, "low", 3)
    tr_sum = 0.0
    previous_close = _value(candles[-length - 1], "close", 4)
    for candle in window:
        high = _value(candle, "high", 2)
        low = _value(candle, "low", 3)
        tr_sum += max(high - low, abs(high - previous_close), abs(low - previous_close))
        previous_close = _value(candle, "close", 4)
    price_range = max(max(highs) - min(lows), 1e-12)
    return float(100.0 * math.log10(max(tr_sum / price_range, 1e-12)) / math.log10(length))


def bollinger(values: List[float], length: int = 20, multiplier: float = 2.0) -> Dict[str, float]:
    window = np.asarray(values[-length:], dtype=float)
    mid = float(np.mean(window))
    deviation = float(np.std(window))
    return {
        "mid": mid,
        "upper": mid + multiplier * deviation,
        "lower": mid - multiplier * deviation,
    }


def _confirmed_pivots(values: List[float], mode: str, left: int = 2, right: int = 2) -> List[Tuple[int, float]]:
    pivots: List[Tuple[int, float]] = []
    for i in range(left, len(values) - right):
        window = values[i - left:i + right + 1]
        value = values[i]
        if mode == "high" and value == max(window) and window.count(value) == 1:
            pivots.append((i, value))
        elif mode == "low" and value == min(window) and window.count(value) == 1:
            pivots.append((i, value))
    return pivots


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 80:
        return {}

    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)

    ema8 = ema(closes, 8)
    ema13 = ema(closes, 13)
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    current_atr = max(atr(candles), closes[-1] * 0.0005)
    bands = bollinger(closes)

    pivot_highs = _confirmed_pivots(highs[-60:], "high")
    pivot_lows = _confirmed_pivots(lows[-60:], "low")
    last_highs = [value for _, value in pivot_highs[-2:]]
    last_lows = [value for _, value in pivot_lows[-2:]]

    last_swing_high = last_highs[-1] if last_highs else max(highs[-12:-1])
    previous_swing_high = last_highs[-2] if len(last_highs) >= 2 else max(highs[-24:-12])
    last_swing_low = last_lows[-1] if last_lows else min(lows[-12:-1])
    previous_swing_low = last_lows[-2] if len(last_lows) >= 2 else min(lows[-24:-12])

    higher_high = last_swing_high > previous_swing_high
    higher_low = last_swing_low > previous_swing_low
    lower_high = last_swing_high < previous_swing_high
    lower_low = last_swing_low < previous_swing_low

    if higher_high and higher_low:
        structure = "BULL"
    elif lower_high and lower_low:
        structure = "BEAR"
    else:
        structure = "MIXED"

    ema20_slope_atr = (ema20[-1] - ema20[-4]) / current_atr
    cross_up = ema8[-2] <= ema13[-2] and ema8[-1] > ema13[-1]
    cross_down = ema8[-2] >= ema13[-2] and ema8[-1] < ema13[-1]

    return {
        "open": opens[-1],
        "high": highs[-1],
        "low": lows[-1],
        "close": closes[-1],
        "prev_close": closes[-2],
        "prev_high": highs[-2],
        "prev_low": lows[-2],
        "ema8": ema8[-1],
        "ema13": ema13[-1],
        "ema20": ema20[-1],
        "ema50": ema50[-1],
        "ema8_prev": ema8[-2],
        "ema13_prev": ema13[-2],
        "ema20_slope_atr": ema20_slope_atr,
        "cross_up": cross_up,
        "cross_down": cross_down,
        "atr": current_atr,
        "adx": adx(candles),
        "chop": choppiness(candles),
        "bb_mid": bands["mid"],
        "bb_upper": bands["upper"],
        "bb_lower": bands["lower"],
        "volume": volumes[-1],
        "vol_avg": float(np.mean(volumes[-20:])),
        "body_atr": abs(closes[-1] - opens[-1]) / current_atr,
        "extension_atr": abs(closes[-1] - ema20[-1]) / current_atr,
        "last_swing_high": last_swing_high,
        "previous_swing_high": previous_swing_high,
        "last_swing_low": last_swing_low,
        "previous_swing_low": previous_swing_low,
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low,
        "structure": structure,
    }


class IndicatorEngine:
    @staticmethod
    def _candle(candles: List[Any]) -> Dict[str, Any]:
        if not candles:
            return {}
        candle = candles[-1]
        return {
            "open": _value(candle, "open", 1),
            "high": _value(candle, "high", 2),
            "low": _value(candle, "low", 3),
            "close": _value(candle, "close", 4),
            "volume": _value(candle, "volume", 5),
        }

    def compute(self, c15m, c1h, c4h) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
        return (
            self._candle(c15m),
            self._candle(c1h),
            self._candle(c4h),
            compute(c15m),
            compute(c1h),
            compute(c4h),
        )
