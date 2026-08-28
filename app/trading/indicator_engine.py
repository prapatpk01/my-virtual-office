"""Simple Structure Trading Bot V6 indicator engine — CLOSED 15M bars only.

One strategy. Four tools only:
- EMA20: direction/bias
- Market structure: BOS or pullback continuation trigger
- RSI14: lightweight momentum confirmation
- ATR14: volatility/risk reference

No regime router, ADX, CHOP, MACD, Bollinger Bands, ROC, scoring, or higher TF.
"""
from __future__ import annotations

from typing import Any, Dict, List

ENGINE_SCHEMA = "simple-v6-structure-15m"


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
    for i in range(1, len(closes)):
        out.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return out


def _rsi(closes, length=14):
    gains, losses = [0.0], [0.0]
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
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
    if len(candles) < 60:
        return {}

    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    volumes = _series(candles, "volume", 5)

    e20 = ema(closes, 20)
    tr = _true_range(highs, lows, closes)
    atrs = _rma(tr, 14)
    atr = max(atrs[-1], closes[-1] * 0.0005)
    rsi = _rsi(closes, 14)

    # Bias: price must be on the correct side of EMA20 and EMA20 must slope.
    bias_long = closes[-1] > e20[-1] and e20[-1] > e20[-4]
    bias_short = closes[-1] < e20[-1] and e20[-1] < e20[-4]

    # Structure: closed-bar break of the previous five-bar range.
    swing_high = max(highs[-6:-1])
    swing_low = min(lows[-6:-1])
    bos_long = closes[-1] > swing_high
    bos_short = closes[-1] < swing_low

    # Pullback continuation: previous candle interacted with EMA20, then the
    # current closed candle breaks that candle's extreme in the bias direction.
    touch = 0.15 * atr
    max_depth = 0.40 * atr
    pullback_long = (
        lows[-2] <= e20[-2] + touch
        and closes[-2] >= e20[-2] - max_depth
        and closes[-1] > highs[-2]
        and closes[-1] > e20[-1]
        and closes[-1] > opens[-1]
    )
    pullback_short = (
        highs[-2] >= e20[-2] - touch
        and closes[-2] <= e20[-2] + max_depth
        and closes[-1] < lows[-2]
        and closes[-1] < e20[-1]
        and closes[-1] < opens[-1]
    )

    # RSI is deliberately lightweight: only require momentum on the correct
    # side of 50, while avoiding extremely stretched entries.
    rsi_long_ok = 50.0 < rsi[-1] < 75.0
    rsi_short_ok = 25.0 < rsi[-1] < 50.0

    distance_atr = abs(closes[-1] - e20[-1]) / max(atr, 1e-12)
    not_chasing = distance_atr <= 1.50

    long_signal = bias_long and rsi_long_ok and not_chasing and (bos_long or pullback_long)
    short_signal = bias_short and rsi_short_ok and not_chasing and (bos_short or pullback_short)

    trigger = ""
    if long_signal:
        trigger = "Bullish BOS" if bos_long else "EMA20 pullback continuation"
    elif short_signal:
        trigger = "Bearish BOS" if bos_short else "EMA20 pullback continuation"

    recent_low = min(lows[-6:-1])
    recent_high = max(highs[-6:-1])

    return {
        "schema": ENGINE_SCHEMA,
        "timeframe": "15M",
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_close": closes[-2], "volume": volumes[-1],
        "ema20": e20[-1], "ema20_prev": e20[-2],
        "atr": atr, "atr_prev": atrs[-2],
        "rsi14": rsi[-1], "rsi14_prev": rsi[-2],
        "bias_long": bias_long, "bias_short": bias_short,
        "swing_high": swing_high, "swing_low": swing_low,
        "recent_low": recent_low, "recent_high": recent_high,
        "bos_long": bos_long, "bos_short": bos_short,
        "pullback_long": pullback_long, "pullback_short": pullback_short,
        "distance_atr": distance_atr,
        "long_signal": long_signal, "short_signal": short_signal,
        "trigger": trigger,
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
