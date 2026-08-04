"""Indicators and confirmed market structure for Adaptive Bot v13.

This module intentionally exposes no CDC fields.  The schema guard makes
mixed v12/v13 deployments fail clearly instead of raising a legacy KeyError.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numpy as np

ENGINE_SCHEMA = "adaptive-v13-structure-v1"
REQUIRED_OUTPUT_KEYS = frozenset({
    "open", "high", "low", "close",
    "ema8", "ema13", "ema20", "ema50", "ema20_slope_atr",
    "cross_up", "cross_down", "atr", "adx", "chop",
    "bb_mid", "bb_upper", "bb_lower",
    "body_atr", "extension_atr",
    "last_swing_high", "last_swing_low",
    "higher_low", "lower_high", "structure",
})


def validate_indicator_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the v13 indicator contract before strategy evaluation."""
    if not result:
        return result
    missing = sorted(REQUIRED_OUTPUT_KEYS.difference(result))
    if missing:
        raise ValueError(
            f"V13_SCHEMA_MISMATCH schema={ENGINE_SCHEMA} missing={','.join(missing)}"
        )
    return result


def _value(candle: Any, name: str, index: int) -> float:
    value = getattr(candle, name, None)
    if value is None and isinstance(candle, dict):
        value = candle.get(name)
    if value is None and isinstance(candle, (list, tuple)) and len(candle) > index:
        value = candle[index]
    return float(value or 0.0)


def _series(candles: List[Any], name: str, index: int) -> List[float]:
    return [_value(candle, name, index) for candle in candles]


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
    true_ranges = [highs[0] - lows[0]]
    for index in range(1, len(closes)):
        true_ranges.append(max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
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
    for index in range(1, len(closes)):
        up = highs[index] - highs[index - 1]
        down = lows[index - 1] - lows[index]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        true_ranges.append(max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        ))
    true_range_sum = max(sum(true_ranges[-length:]), 1e-12)
    plus_di = 100.0 * sum(plus_dm[-length:]) / true_range_sum
    minus_di = 100.0 * sum(minus_dm[-length:]) / true_range_sum
    return float(100.0 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-12))


def choppiness(candles: List[Any], length: int = 14) -> float:
    if len(candles) < length + 1:
        return 100.0
    window = candles[-length:]
    highs = _series(window, "high", 2)
    lows = _series(window, "low", 3)
    true_range_sum = 0.0
    previous_close = _value(candles[-length - 1], "close", 4)
    for candle in window:
        high = _value(candle, "high", 2)
        low = _value(candle, "low", 3)
        true_range_sum += max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )
        previous_close = _value(candle, "close", 4)
    price_range = max(max(highs) - min(lows), 1e-12)
    return float(
        100.0
        * math.log10(max(true_range_sum / price_range, 1e-12))
        / math.log10(length)
    )


def bollinger(
    values: List[float],
    length: int = 20,
    multiplier: float = 2.0,
) -> Dict[str, float]:
    window = np.asarray(values[-length:], dtype=float)
    middle = float(np.mean(window))
    deviation = float(np.std(window))
    return {
        "mid": middle,
        "upper": middle + multiplier * deviation,
        "lower": middle - multiplier * deviation,
    }


def _confirmed_pivots(
    values: List[float],
    mode: str,
    left: int = 2,
    right: int = 2,
) -> List[Tuple[int, float]]:
    pivots: List[Tuple[int, float]] = []
    for index in range(left, len(values) - right):
        window = values[index - left:index + right + 1]
        value = values[index]
        if mode == "high" and value == max(window) and window.count(value) == 1:
            pivots.append((index, value))
        elif mode == "low" and value == min(window) and window.count(value) == 1:
            pivots.append((index, value))
    return pivots


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 80:
        return {}

    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)

    ema8_values = ema(closes, 8)
    ema13_values = ema(closes, 13)
    ema20_values = ema(closes, 20)
    ema50_values = ema(closes, 50)
    current_atr = max(atr(candles), closes[-1] * 0.0005)
    bands = bollinger(closes)

    pivot_highs = _confirmed_pivots(highs[-60:], "high")
    pivot_lows = _confirmed_pivots(lows[-60:], "low")
    last_highs = [value for _, value in pivot_highs[-2:]]
    last_lows = [value for _, value in pivot_lows[-2:]]

    last_swing_high = last_highs[-1] if last_highs else max(highs[-12:-1])
    previous_swing_high = (
        last_highs[-2] if len(last_highs) >= 2 else max(highs[-24:-12])
    )
    last_swing_low = last_lows[-1] if last_lows else min(lows[-12:-1])
    previous_swing_low = (
        last_lows[-2] if len(last_lows) >= 2 else min(lows[-24:-12])
    )

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

    result: Dict[str, Any] = {
        "schema": ENGINE_SCHEMA,
        "open": opens[-1],
        "high": highs[-1],
        "low": lows[-1],
        "close": closes[-1],
        "prev_close": closes[-2],
        "prev_high": highs[-2],
        "prev_low": lows[-2],
        "ema8": ema8_values[-1],
        "ema13": ema13_values[-1],
        "ema20": ema20_values[-1],
        "ema50": ema50_values[-1],
        "ema8_prev": ema8_values[-2],
        "ema13_prev": ema13_values[-2],
        "ema20_slope_atr": (ema20_values[-1] - ema20_values[-4]) / current_atr,
        "cross_up": (
            ema8_values[-2] <= ema13_values[-2]
            and ema8_values[-1] > ema13_values[-1]
        ),
        "cross_down": (
            ema8_values[-2] >= ema13_values[-2]
            and ema8_values[-1] < ema13_values[-1]
        ),
        "atr": current_atr,
        "adx": adx(candles),
        "chop": choppiness(candles),
        "bb_mid": bands["mid"],
        "bb_upper": bands["upper"],
        "bb_lower": bands["lower"],
        "volume": volumes[-1],
        "vol_avg": float(np.mean(volumes[-20:])),
        "body_atr": abs(closes[-1] - opens[-1]) / current_atr,
        "extension_atr": abs(closes[-1] - ema20_values[-1]) / current_atr,
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
    return validate_indicator_output(result)


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

    def compute(
        self,
        candles_15m,
        candles_1h,
        candles_4h,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
        return (
            self._candle(candles_15m),
            self._candle(candles_1h),
            self._candle(candles_4h),
            compute(candles_15m),
            compute(candles_1h),
            compute(candles_4h),
        )
