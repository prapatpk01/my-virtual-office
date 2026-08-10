"""Adaptive Momentum v3.2 fast-entry indicator engine.

Direction comes from closed 1H EMA20/50 built from closed 15M candles.
All alignment, momentum, quality, location and entry triggers remain on 15M.
"""
from __future__ import annotations

from typing import Any, Dict, List
import math

ENGINE_SCHEMA = "adaptive-momentum-v3.2-15m"


def _v(candle: Any, name: str, index: int) -> float:
    value = getattr(candle, name, None)
    if value is None and isinstance(candle, dict):
        value = candle.get(name)
    if value is None and isinstance(candle, (list, tuple)) and len(candle) > index:
        value = candle[index]
    return float(value or 0.0)


def _timestamp(candle: Any) -> float:
    value = getattr(candle, "timestamp", None)
    if value is None and isinstance(candle, dict):
        value = candle.get("timestamp")
    if value is None and isinstance(candle, (list, tuple)) and candle:
        value = candle[0]
    return float(value or 0.0)


def _series(candles: List[Any], name: str, index: int) -> List[float]:
    return [_v(candle, name, index) for candle in candles]


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
        output.append(max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        ))
    return output


def _adx(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> List[float]:
    tr = _true_ranges(highs, lows, closes)
    plus_dm, minus_dm = [0.0], [0.0]
    for index in range(1, len(closes)):
        up = highs[index] - highs[index - 1]
        down = lows[index - 1] - lows[index]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    atr_rma = _rma(tr, length)
    plus_rma = _rma(plus_dm, length)
    minus_rma = _rma(minus_dm, length)
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


def _cross_up_recent(fast: List[float], slow: List[float], bars: int = 3) -> bool:
    start = max(1, len(fast) - bars)
    return any(fast[i] > slow[i] and fast[i - 1] <= slow[i - 1] for i in range(start, len(fast)))


def _cross_down_recent(fast: List[float], slow: List[float], bars: int = 3) -> bool:
    start = max(1, len(fast) - bars)
    return any(fast[i] < slow[i] and fast[i - 1] >= slow[i - 1] for i in range(start, len(fast)))


def _closed_1h_from_15m(candles: List[Any]) -> List[List[float]]:
    """Aggregate only complete four-candle 1H bars from closed 15M candles."""
    groups: Dict[int, List[Any]] = {}
    for candle in candles:
        ts = int(_timestamp(candle))
        if ts <= 0:
            continue
        # CCXT timestamps are milliseconds. Keep support for seconds for tests/local data.
        hour_ms = 3_600_000
        if ts < 10_000_000_000:
            ts *= 1000
        bucket = ts // hour_ms
        groups.setdefault(bucket, []).append(candle)

    output: List[List[float]] = []
    for bucket in sorted(groups):
        rows = sorted(groups[bucket], key=_timestamp)
        if len(rows) != 4:
            continue
        output.append([
            bucket * 3_600_000,
            _v(rows[0], "open", 1),
            max(_v(row, "high", 2) for row in rows),
            min(_v(row, "low", 3) for row in rows),
            _v(rows[-1], "close", 4),
            sum(_v(row, "volume", 5) for row in rows),
        ])
    return output


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 240:
        return {}

    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)

    # 15M execution indicators.
    e8, e13, e20_15m, e50_15m = ema(closes, 8), ema(closes, 13), ema(closes, 20), ema(closes, 50)
    macd_line = [fast - slow for fast, slow in zip(ema(closes, 12), ema(closes, 26))]
    macd_signal = ema(macd_line, 9)
    macd_hist = [line - signal for line, signal in zip(macd_line, macd_signal)]
    atr_series = _rma(_true_ranges(highs, lows, closes), 14)
    atr_value = max(atr_series[-1], closes[-1] * 0.0005)
    adx_series = _adx(highs, lows, closes, 14)
    chop_value = _chop(highs, lows, closes, 14)

    # 1H direction filter. Never use an in-progress 1H candle.
    candles_1h = _closed_1h_from_15m(candles)
    if len(candles_1h) < 50:
        return {}
    closes_1h = [row[4] for row in candles_1h]
    e20_1h = ema(closes_1h, 20)
    e50_1h = ema(closes_1h, 50)
    trend_bull = e20_1h[-1] > e50_1h[-1]
    trend_bear = e20_1h[-1] < e50_1h[-1]

    recent_high = max(highs[-6:-1])
    recent_low = min(lows[-6:-1])
    distance_atr = abs(closes[-1] - e13[-1]) / atr_value

    fresh_up = _cross_up_recent(e8, e13, 3)
    fresh_down = _cross_down_recent(e8, e13, 3)
    reclaim_long = lows[-1] <= e13[-1] and closes[-1] > e13[-1] and closes[-1] > opens[-1]
    reclaim_short = highs[-1] >= e13[-1] and closes[-1] < e13[-1] and closes[-1] < opens[-1]
    prev_break_long = closes[-1] > highs[-2]
    prev_break_short = closes[-1] < lows[-2]

    return {
        "schema": ENGINE_SCHEMA,
        "trend_tf": "1H",
        "trend_ema20": e20_1h[-1],
        "trend_ema50": e50_1h[-1],
        "trend_bull": trend_bull,
        "trend_bear": trend_bear,
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_open": opens[-2], "prev_high": highs[-2], "prev_low": lows[-2], "prev_close": closes[-2],
        "ema8": e8[-1], "ema13": e13[-1], "ema20": e20_15m[-1], "ema50": e50_15m[-1],
        "ema8_prev": e8[-2], "ema13_prev": e13[-2],
        "ema8_series": e8[-100:], "ema13_series": e13[-100:],
        "ema20_series": e20_15m[-100:], "ema50_series": e50_15m[-100:],
        "entry_bull": e8[-1] > e13[-1], "entry_bear": e8[-1] < e13[-1],
        "ema_cross_up": e8[-1] > e13[-1] and e8[-2] <= e13[-2],
        "ema_cross_down": e8[-1] < e13[-1] and e8[-2] >= e13[-2],
        "ema_cross_up_recent": fresh_up, "ema_cross_down_recent": fresh_down,
        "ema13_reclaim_long": reclaim_long, "ema13_reclaim_short": reclaim_short,
        "prev_bar_break_long": prev_break_long, "prev_bar_break_short": prev_break_short,
        "trigger_long": fresh_up or reclaim_long or prev_break_long,
        "trigger_short": fresh_down or reclaim_short or prev_break_short,
        "macd": macd_line[-1], "macd_signal": macd_signal[-1],
        "macd_hist": macd_hist[-1], "macd_hist_prev": macd_hist[-2],
        "macd_hist_improving_long": macd_hist[-1] > macd_hist[-2],
        "macd_hist_improving_short": macd_hist[-1] < macd_hist[-2],
        "macd_hist_weaken_long_3": macd_hist[-1] < macd_hist[-2] < macd_hist[-3] < macd_hist[-4],
        "macd_hist_weaken_short_3": macd_hist[-1] > macd_hist[-2] > macd_hist[-3] > macd_hist[-4],
        "adx": adx_series[-1], "adx_prev": adx_series[-2],
        "adx_rising": adx_series[-1] > adx_series[-2],
        "chop": chop_value, "atr": atr_value,
        "distance_ema13_atr": distance_atr,
        "location_long": closes[-1] >= e13[-1] and distance_atr <= 1.0,
        "location_short": closes[-1] <= e13[-1] and distance_atr <= 1.0,
        "recent_low": recent_low, "recent_high": recent_high,
        "volume": volumes[-1],
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
