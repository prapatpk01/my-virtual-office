"""Adaptive Trading Bot V5 indicator engine — 15M only.

The engine does one job: classify the current 15M market into a lightweight regime
and expose the trigger for that regime.

REGIME ROUTER
- TREND:    ADX >= 20 and CHOP < 50
- MEAN:     ADX < 20 and CHOP > 55
- BREAKOUT: ADX rising + CHOP 45..55 + BB width expanding + ATR expanding
- WAIT:     anything ambiguous

STYLE ENGINES
- TREND: EMA8/13 + MACD12/26/9 + RSI14 + EMA13 reclaim/reject
- MEAN: Bollinger Bands 20,2 + RSI14 + re-entry into the band
- BREAKOUT: swing break + Bollinger expansion + ATR14 expansion + ROC9

No 1H/4H dependency and no multi-indicator score.
"""
from __future__ import annotations

from typing import Any, Dict, List
import math

ENGINE_SCHEMA = "adaptive-v5-three-style-15m"


def _v(c: Any, name: str, idx: int) -> float:
    value = getattr(c, name, None)
    if value is None and isinstance(c, dict):
        value = c.get(name)
    if value is None and isinstance(c, (list, tuple)) and len(c) > idx:
        value = c[idx]
    return float(value or 0.0)


def _series(candles, name, idx):
    return [_v(c, name, idx) for c in candles]


def ema(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def _sma(values: List[float], length: int) -> List[float]:
    out = []
    for index in range(len(values)):
        window = values[max(0, index - length + 1):index + 1]
        out.append(sum(window) / len(window))
    return out


def _rma(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 1.0 / max(length, 1)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def _true_range(highs, lows, closes):
    out = [max(highs[0] - lows[0], 0.0)]
    for index in range(1, len(closes)):
        out.append(max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        ))
    return out


def _adx(highs, lows, closes, length=14):
    tr = _true_range(highs, lows, closes)
    plus_dm, minus_dm = [0.0], [0.0]
    for index in range(1, len(closes)):
        up = highs[index] - highs[index - 1]
        down = lows[index - 1] - lows[index]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    atr = _rma(tr, length)
    plus = _rma(plus_dm, length)
    minus = _rma(minus_dm, length)
    dx = []
    for current_atr, p, m in zip(atr, plus, minus):
        if current_atr <= 1e-12:
            dx.append(0.0)
            continue
        pdi = 100.0 * p / current_atr
        mdi = 100.0 * m / current_atr
        total = pdi + mdi
        dx.append(100.0 * abs(pdi - mdi) / total if total > 1e-12 else 0.0)
    return _rma(dx, length)


def _chop(highs, lows, closes, length=14):
    tr_sum = sum(_true_range(highs, lows, closes)[-length:])
    span = max(highs[-length:]) - min(lows[-length:])
    if tr_sum <= 0 or span <= 1e-12:
        return 100.0
    return 100.0 * math.log10(tr_sum / span) / math.log10(length)


def _bb(closes, length=20, mult=2.0):
    middle = _sma(closes, length)
    upper, lower = [], []
    for index in range(len(closes)):
        window = closes[max(0, index - length + 1):index + 1]
        mean = middle[index]
        variance = sum((value - mean) ** 2 for value in window) / len(window)
        std = math.sqrt(variance)
        upper.append(mean + mult * std)
        lower.append(mean - mult * std)
    return middle, upper, lower


def _rsi(closes, length=14):
    gains, losses = [0.0], [0.0]
    for index in range(1, len(closes)):
        change = closes[index] - closes[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = _rma(gains, length)
    avg_loss = _rma(losses, length)
    out = []
    for gain, loss in zip(avg_gain, avg_loss):
        if loss <= 1e-12:
            out.append(100.0 if gain > 0 else 50.0)
        else:
            rs = gain / loss
            out.append(100.0 - 100.0 / (1.0 + rs))
    return out


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 100:
        return {}

    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)

    e8 = ema(closes, 8)
    e13 = ema(closes, 13)
    macd_line = [fast - slow for fast, slow in zip(ema(closes, 12), ema(closes, 26))]
    macd_signal = ema(macd_line, 9)
    macd_hist = [line - signal for line, signal in zip(macd_line, macd_signal)]
    bb_mid, bb_upper, bb_lower = _bb(closes, 20, 2.0)
    tr = _true_range(highs, lows, closes)
    atrs = _rma(tr, 14)
    atr = max(atrs[-1], closes[-1] * 0.0005)
    adxs = _adx(highs, lows, closes, 14)
    chop = _chop(highs, lows, closes, 14)
    rsi14 = _rsi(closes, 14)
    roc9 = ((closes[-1] / closes[-10]) - 1.0) * 100.0 if closes[-10] else 0.0

    adx = adxs[-1]
    adx_rising = adxs[-1] > adxs[-2]
    atr_rising = atrs[-1] > atrs[-2]
    bb_width = (bb_upper[-1] - bb_lower[-1]) / max(abs(bb_mid[-1]), 1e-12)
    bb_width_prev = (bb_upper[-2] - bb_lower[-2]) / max(abs(bb_mid[-2]), 1e-12)
    bb_expanding = bb_width > bb_width_prev

    # Regime router. BREAKOUT gets first priority because it is a transition state.
    if adx_rising and 45.0 <= chop <= 55.0 and bb_expanding and atr_rising:
        regime = "BREAKOUT"
    elif adx >= 20.0 and chop < 50.0:
        regime = "TREND"
    elif adx < 20.0 and chop > 55.0:
        regime = "MEAN"
    else:
        regime = "WAIT"

    ema_bull = e8[-1] > e13[-1]
    ema_bear = e8[-1] < e13[-1]
    ema_cross_up = ema_bull and e8[-2] <= e13[-2]
    ema_cross_down = ema_bear and e8[-2] >= e13[-2]

    # TREND style: pullback into EMA13 then reclaim/reject.
    touch_tolerance = 0.10 * atr
    trend_long = (
        regime == "TREND"
        and ema_bull
        and macd_line[-1] > macd_signal[-1]
        and rsi14[-1] >= 45.0
        and lows[-1] <= e13[-1] + touch_tolerance
        and closes[-1] > e13[-1]
        and closes[-1] > opens[-1]
    )
    trend_short = (
        regime == "TREND"
        and ema_bear
        and macd_line[-1] < macd_signal[-1]
        and rsi14[-1] <= 55.0
        and highs[-1] >= e13[-1] - touch_tolerance
        and closes[-1] < e13[-1]
        and closes[-1] < opens[-1]
    )

    # MEAN style: pierce/touch an outer band and close back inside it.
    mean_long = (
        regime == "MEAN"
        and lows[-1] <= bb_lower[-1]
        and closes[-1] > bb_lower[-1]
        and rsi14[-1] < 35.0
    )
    mean_short = (
        regime == "MEAN"
        and highs[-1] >= bb_upper[-1]
        and closes[-1] < bb_upper[-1]
        and rsi14[-1] > 67.0
    )

    # BREAKOUT style: local swing break + outer-band close + volatility expansion + ROC9.
    swing_high = max(highs[-6:-1])
    swing_low = min(lows[-6:-1])
    breakout_long = (
        regime == "BREAKOUT"
        and closes[-1] > swing_high
        and closes[-1] > bb_upper[-1]
        and bb_expanding
        and atr_rising
        and roc9 > 0.0
    )
    breakout_short = (
        regime == "BREAKOUT"
        and closes[-1] < swing_low
        and closes[-1] < bb_lower[-1]
        and bb_expanding
        and atr_rising
        and roc9 < 0.0
    )

    recent_low = min(lows[-6:-1])
    recent_high = max(highs[-6:-1])

    return {
        "schema": ENGINE_SCHEMA,
        "timeframe": "15M",
        "regime": regime,
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_close": closes[-2], "volume": volumes[-1],
        "ema8": e8[-1], "ema13": e13[-1],
        "ema8_prev": e8[-2], "ema13_prev": e13[-2],
        "ema_bull": ema_bull, "ema_bear": ema_bear,
        "ema_cross_up": ema_cross_up, "ema_cross_down": ema_cross_down,
        "macd": macd_line[-1], "macd_signal": macd_signal[-1], "macd_hist": macd_hist[-1],
        "rsi14": rsi14[-1], "roc9": roc9,
        "bb_mid": bb_mid[-1], "bb_upper": bb_upper[-1], "bb_lower": bb_lower[-1],
        "bb_width": bb_width, "bb_expanding": bb_expanding,
        "atr": atr, "atr_prev": atrs[-2], "atr_rising": atr_rising,
        "adx": adx, "adx_prev": adxs[-2], "adx_rising": adx_rising,
        "chop": chop,
        "swing_high": swing_high, "swing_low": swing_low,
        "recent_low": recent_low, "recent_high": recent_high,
        "trend_long": trend_long, "trend_short": trend_short,
        "mean_long": mean_long, "mean_short": mean_short,
        "breakout_long": breakout_long, "breakout_short": breakout_short,
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
