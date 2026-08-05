"""Adaptive Momentum v2.1 indicator engine for closed 15-minute candles."""
from __future__ import annotations

from typing import Any, Dict, List
import math

ENGINE_SCHEMA = "adaptive-momentum-v2.1-15m"


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


def _rma(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 1.0 / max(length, 1)
    output = [float(values[0])]
    for value in values[1:]:
        output.append(alpha * float(value) + (1.0 - alpha) * output[-1])
    return output


def _true_ranges(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    output = [max(highs[0] - lows[0], 0.0)]
    for index in range(1, len(closes)):
        output.append(max(highs[index] - lows[index], abs(highs[index] - closes[index - 1]), abs(lows[index] - closes[index - 1])))
    return output


def _adx(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> List[float]:
    tr = _true_ranges(highs, lows, closes)
    plus_dm, minus_dm = [0.0], [0.0]
    for index in range(1, len(closes)):
        up = highs[index] - highs[index - 1]
        down = lows[index - 1] - lows[index]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    atr_rma, plus_rma, minus_rma = _rma(tr, length), _rma(plus_dm, length), _rma(minus_dm, length)
    dx: List[float] = []
    for atr_value, plus_value, minus_value in zip(atr_rma, plus_rma, minus_rma):
        if atr_value <= 1e-12:
            dx.append(0.0)
            continue
        plus_di = 100.0 * plus_value / atr_value
        minus_di = 100.0 * minus_value / atr_value
        total = plus_di + minus_di
        dx.append(100.0 * abs(plus_di - minus_di) / total if total > 1e-12 else 0.0)
    return _rma(dx, length)


def _chop(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> float:
    tr_sum = sum(_true_ranges(highs, lows, closes)[-length:])
    span = max(highs[-length:]) - min(lows[-length:])
    if tr_sum <= 0 or span <= 1e-12:
        return 100.0
    return 100.0 * math.log10(tr_sum / span) / math.log10(length)


def _recent_flip_up(hist: List[float], bars: int = 3) -> bool:
    start = max(1, len(hist) - bars)
    return any(hist[i] > 0 and hist[i - 1] <= 0 for i in range(start, len(hist)))


def _recent_flip_down(hist: List[float], bars: int = 3) -> bool:
    start = max(1, len(hist) - bars)
    return any(hist[i] < 0 and hist[i - 1] >= 0 for i in range(start, len(hist)))


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 80:
        return {}
    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)

    e8, e13, e20, e50 = ema(closes, 8), ema(closes, 13), ema(closes, 20), ema(closes, 50)
    macd_line = [fast - slow for fast, slow in zip(ema(closes, 12), ema(closes, 26))]
    macd_signal = ema(macd_line, 9)
    macd_hist = [line - signal for line, signal in zip(macd_line, macd_signal)]
    atr_series = _rma(_true_ranges(highs, lows, closes), 14)
    atr_value = max(atr_series[-1], closes[-1] * 0.0005)
    adx_series = _adx(highs, lows, closes, 14)
    chop_value = _chop(highs, lows, closes, 14)
    distance_atr = abs(closes[-1] - e13[-1]) / atr_value

    hist_flip_up_recent = _recent_flip_up(macd_hist, 3)
    hist_flip_down_recent = _recent_flip_down(macd_hist, 3)
    hist_expanding_up = macd_hist[-1] > macd_hist[-2]
    hist_expanding_down = macd_hist[-1] < macd_hist[-2]
    macd_state_bull = hist_flip_up_recent or macd_line[-1] > macd_signal[-1]
    macd_state_bear = hist_flip_down_recent or macd_line[-1] < macd_signal[-1]

    return {
        "schema": ENGINE_SCHEMA,
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_open": opens[-2], "prev_high": highs[-2], "prev_low": lows[-2], "prev_close": closes[-2],
        "ema8": e8[-1], "ema13": e13[-1], "ema20": e20[-1], "ema50": e50[-1],
        "ema8_prev": e8[-2], "ema13_prev": e13[-2],
        "ema8_series": e8[-100:], "ema13_series": e13[-100:], "ema20_series": e20[-100:], "ema50_series": e50[-100:],
        "trend_bull": e20[-1] > e50[-1], "trend_bear": e20[-1] < e50[-1],
        "entry_bull": e8[-1] > e13[-1], "entry_bear": e8[-1] < e13[-1],
        "ema_cross_up": e8[-1] > e13[-1] and e8[-2] <= e13[-2],
        "ema_cross_down": e8[-1] < e13[-1] and e8[-2] >= e13[-2],
        "macd": macd_line[-1], "macd_signal": macd_signal[-1],
        "macd_hist": macd_hist[-1], "macd_hist_prev": macd_hist[-2],
        "macd_hist_flip_up_recent": hist_flip_up_recent,
        "macd_hist_flip_down_recent": hist_flip_down_recent,
        "macd_hist_expanding_up": hist_expanding_up,
        "macd_hist_expanding_down": hist_expanding_down,
        "macd_state_bull": macd_state_bull,
        "macd_state_bear": macd_state_bear,
        "adx": adx_series[-1], "adx_prev": adx_series[-2], "adx_rising": adx_series[-1] > adx_series[-2],
        "chop": chop_value, "atr": atr_value,
        "distance_ema13_atr": distance_atr,
        "location_long": closes[-1] >= e13[-1] and distance_atr <= 1.0,
        "location_short": closes[-1] <= e13[-1] and distance_atr <= 1.0,
        "recent_low": min(lows[-6:-1]), "recent_high": max(highs[-6:-1]),
        "volume": volumes[-1],
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
