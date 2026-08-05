"""Momentum v1 indicator engine: EMA5/9, MACD, ADX, CHOP and location on 15M."""
from __future__ import annotations
from typing import Any, Dict, List
import math

ENGINE_SCHEMA = "adaptive-momentum-v1-15m"

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
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out

def _rma(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    out = [float(values[0])]
    alpha = 1.0 / max(length, 1)
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out

def _true_ranges(highs, lows, closes):
    tr = [max(highs[0] - lows[0], 0.0)]
    for i in range(1, len(closes)):
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
    return tr

def _adx(highs, lows, closes, length=14):
    tr = _true_ranges(highs, lows, closes)
    plus_dm, minus_dm = [0.0], [0.0]
    for i in range(1, len(closes)):
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    atr_rma, plus_rma, minus_rma = _rma(tr, length), _rma(plus_dm, length), _rma(minus_dm, length)
    dx = []
    for atr_value, p, m in zip(atr_rma, plus_rma, minus_rma):
        if atr_value <= 1e-12:
            dx.append(0.0)
            continue
        pdi, mdi = 100.0*p/atr_value, 100.0*m/atr_value
        total = pdi + mdi
        dx.append(100.0*abs(pdi-mdi)/total if total > 1e-12 else 0.0)
    return _rma(dx, length)

def _chop(highs, lows, closes, length=14):
    if len(closes) < length + 1:
        return 50.0
    tr_sum = sum(_true_ranges(highs, lows, closes)[-length:])
    span = max(highs[-length:]) - min(lows[-length:])
    if tr_sum <= 0 or span <= 1e-12:
        return 100.0
    return 100.0 * math.log10(tr_sum/span) / math.log10(length)

def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 80:
        return {}
    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)
    e5, e9, e20, e50 = ema(closes,5), ema(closes,9), ema(closes,20), ema(closes,50)
    macd_line = [a-b for a,b in zip(ema(closes,12), ema(closes,26))]
    macd_signal = ema(macd_line,9)
    macd_hist = [a-b for a,b in zip(macd_line, macd_signal)]
    tr = _true_ranges(highs,lows,closes)
    atr_series = _rma(tr,14)
    atr_value = max(atr_series[-1], closes[-1]*0.0005)
    adx_series = _adx(highs,lows,closes,14)
    chop_value = _chop(highs,lows,closes,14)
    cross_up = e5[-1] > e9[-1] and e5[-2] <= e9[-2]
    cross_down = e5[-1] < e9[-1] and e5[-2] >= e9[-2]
    cross_up_prev = e5[-2] > e9[-2] and e5[-3] <= e9[-3]
    cross_down_prev = e5[-2] < e9[-2] and e5[-3] >= e9[-3]
    distance_atr = abs(closes[-1]-e9[-1]) / atr_value
    return {
        "schema": ENGINE_SCHEMA,
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_open": opens[-2], "prev_high": highs[-2], "prev_low": lows[-2], "prev_close": closes[-2],
        "ema5": e5[-1], "ema9": e9[-1], "ema20": e20[-1], "ema50": e50[-1],
        "ema5_prev": e5[-2], "ema9_prev": e9[-2],
        "ema5_series": e5[-80:], "ema9_series": e9[-80:], "ema20_series": e20[-80:], "ema50_series": e50[-80:],
        "ema_cross_up": cross_up, "ema_cross_down": cross_down,
        "ema_cross_up_recent": cross_up or cross_up_prev,
        "ema_cross_down_recent": cross_down or cross_down_prev,
        "macd": macd_line[-1], "macd_signal": macd_signal[-1], "macd_hist": macd_hist[-1],
        "macd_prev": macd_line[-2], "macd_signal_prev": macd_signal[-2],
        "macd_bull": macd_line[-1] > macd_signal[-1] and macd_hist[-1] > 0,
        "macd_bear": macd_line[-1] < macd_signal[-1] and macd_hist[-1] < 0,
        "macd_cross_up": macd_line[-1] > macd_signal[-1] and macd_line[-2] <= macd_signal[-2],
        "macd_cross_down": macd_line[-1] < macd_signal[-1] and macd_line[-2] >= macd_signal[-2],
        "adx": adx_series[-1], "adx_prev": adx_series[-2], "adx_rising": adx_series[-1] > adx_series[-2],
        "chop": chop_value, "atr": atr_value, "distance_ema9_atr": distance_atr,
        "location_long": closes[-1] >= e9[-1] and distance_atr <= 1.0,
        "location_short": closes[-1] <= e9[-1] and distance_atr <= 1.0,
        "recent_low": min(lows[-6:-1]), "recent_high": max(highs[-6:-1]), "volume": volumes[-1],
    }

class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
