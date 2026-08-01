"""Compact indicators for Adaptive Bot v12."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


def _values(candles, name: str):
    return [float(getattr(c, name, c[name] if isinstance(c, dict) else 0.0)) for c in candles]


def ema(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def atr(candles, length: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    highs = _values(candles, "high")
    lows = _values(candles, "low")
    closes = _values(candles, "close")
    true_ranges = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return float(np.mean(true_ranges[-length:]))


def rsi(values: List[float], length: int = 14) -> float:
    if len(values) <= length:
        return 50.0
    delta = np.diff(np.asarray(values[-(length + 1):], dtype=float))
    gain = np.mean(np.clip(delta, 0, None))
    loss = np.mean(np.clip(-delta, 0, None))
    if loss <= 1e-12:
        return 100.0
    rs = gain / loss
    return float(100.0 - 100.0 / (1.0 + rs))


def adx(candles, length: int = 14) -> float:
    if len(candles) < length + 2:
        return 20.0
    highs = _values(candles, "high")
    lows = _values(candles, "low")
    closes = _values(candles, "close")
    plus_dm, minus_dm, true_ranges = [], [], []
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


def bollinger(values: List[float], length: int = 20, multiplier: float = 2.0) -> Dict[str, float]:
    window = np.asarray(values[-length:], dtype=float)
    mid = float(np.mean(window))
    standard_deviation = float(np.std(window))
    return {
        "mid": mid,
        "upper": mid + multiplier * standard_deviation,
        "lower": mid - multiplier * standard_deviation,
        "width": (2.0 * multiplier * standard_deviation) / max(abs(mid), 1e-12),
    }


def compute(candles) -> Dict[str, Any]:
    if len(candles) < 60:
        return {}
    closes = _values(candles, "close")
    opens = _values(candles, "open")
    highs = _values(candles, "high")
    lows = _values(candles, "low")
    volumes = _values(candles, "volume")
    ema8, ema13, ema20, ema50 = (ema(closes, n) for n in (8, 13, 20, 50))
    ema12, ema26 = ema(closes, 12), ema(closes, 26)
    bands = bollinger(closes)
    current_atr = atr(candles)
    volume_average = float(np.mean(volumes[-20:]))
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
        "ema12": ema12[-1],
        "ema26": ema26[-1],
        "ema12_prev": ema12[-2],
        "ema26_prev": ema26[-2],
        "cdc_bull": ema12[-1] > ema26[-1],
        "cdc_bear": ema12[-1] < ema26[-1],
        "cdc_cross_up": ema12[-2] <= ema26[-2] and ema12[-1] > ema26[-1],
        "cdc_cross_down": ema12[-2] >= ema26[-2] and ema12[-1] < ema26[-1],
        "bb_mid": bands["mid"],
        "bb_upper": bands["upper"],
        "bb_lower": bands["lower"],
        "bb_width": bands["width"],
        "atr": current_atr,
        "adx": adx(candles),
        "rsi": rsi(closes),
        "volume": volumes[-1],
        "vol_avg": volume_average,
        "body_atr": abs(closes[-1] - opens[-1]) / max(current_atr, 1e-12),
        "extension_atr": abs(closes[-1] - ema20[-1]) / max(current_atr, 1e-12),
        "swing_high": max(highs[-10:-1]),
        "swing_low": min(lows[-10:-1]),
    }
