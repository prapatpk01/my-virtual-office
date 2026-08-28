"""Simple Structure Trading Bot V6.1 indicator engine — CLOSED 15M bars only.

One strategy. Four tools only:
- EMA20: directional bias
- Market structure: confirmed HL/LH setup then structure break
- RSI14: lightweight momentum confirmation
- ATR14: volatility/risk reference

The old raw 5-bar BOS entry is removed. A single rebound candle is not enough.
Entry requires a real pullback structure first, then continuation through the
setup extreme. Signals use CLOSED 15M candles only.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Keep schema stable so rolling Railway commits do not create a temporary
# indicator/bot mismatch. BUILD_ID identifies V6.1 behavior.
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

    # Bias is intentionally simple. EMA20 is direction, not an entry trigger.
    bias_long = closes[-1] > e20[-1] and e20[-1] > e20[-4]
    bias_short = closes[-1] < e20[-1] and e20[-1] < e20[-4]

    # Meaningful structure is split into two completed zones:
    # - previous structure: bars -10..-5
    # - pullback/setup:     bars -4..-2
    # The current closed bar (-1) is only the trigger bar.
    previous_high = max(highs[-10:-4])
    previous_low = min(lows[-10:-4])
    setup_high = max(highs[-4:-1])
    setup_low = min(lows[-4:-1])

    # A valid continuation must first create a Higher Low / Lower High. This
    # rejects a one-candle reversal being mislabeled as a BOS.
    higher_low = setup_low > previous_low
    lower_high = setup_high < previous_high

    # Pullback must actually return toward EMA20, but cannot close deeply
    # through it. This keeps entries near the trend reference without forcing
    # an exact EMA touch.
    setup_lows = lows[-4:-1]
    setup_highs = highs[-4:-1]
    setup_closes = closes[-4:-1]
    setup_ema20 = e20[-4:-1]
    near = 0.35 * atr
    max_close_break = 0.30 * atr

    pullback_long = (
        min(low - ema_value for low, ema_value in zip(setup_lows, setup_ema20)) <= near
        and min(close - ema_value for close, ema_value in zip(setup_closes, setup_ema20)) >= -max_close_break
    )
    pullback_short = (
        max(high - ema_value for high, ema_value in zip(setup_highs, setup_ema20)) >= -near
        and max(close - ema_value for close, ema_value in zip(setup_closes, setup_ema20)) <= max_close_break
    )

    # Trigger only AFTER structure is formed: break the completed setup extreme.
    # This is deliberately not a raw 5-bar BOS.
    structure_break_long = closes[-1] > setup_high and closes[-1] > opens[-1]
    structure_break_short = closes[-1] < setup_low and closes[-1] < opens[-1]

    rsi_long_ok = 50.0 < rsi[-1] < 72.0
    rsi_short_ok = 28.0 < rsi[-1] < 50.0

    # Main fix for late entries: do not chase more than 0.70 ATR from EMA20.
    distance_atr = abs(closes[-1] - e20[-1]) / max(atr, 1e-12)
    not_chasing = distance_atr <= 0.70

    long_signal = (
        bias_long
        and higher_low
        and pullback_long
        and structure_break_long
        and rsi_long_ok
        and not_chasing
    )
    short_signal = (
        bias_short
        and lower_high
        and pullback_short
        and structure_break_short
        and rsi_short_ok
        and not_chasing
    )

    trigger = ""
    if long_signal:
        trigger = "HL + structure break"
    elif short_signal:
        trigger = "LH + structure break"

    # Stop reference comes from the actual setup structure, not an arbitrary
    # five-bar range. The bot adds ATR padding/caps around this level.
    recent_low = setup_low
    recent_high = setup_high

    return {
        "schema": ENGINE_SCHEMA,
        "timeframe": "15M",
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_close": closes[-2], "volume": volumes[-1],
        "ema20": e20[-1], "ema20_prev": e20[-2],
        "atr": atr, "atr_prev": atrs[-2],
        "rsi14": rsi[-1], "rsi14_prev": rsi[-2],
        "bias_long": bias_long, "bias_short": bias_short,
        "previous_high": previous_high, "previous_low": previous_low,
        "setup_high": setup_high, "setup_low": setup_low,
        "higher_low": higher_low, "lower_high": lower_high,
        "pullback_long": pullback_long, "pullback_short": pullback_short,
        "structure_break_long": structure_break_long,
        "structure_break_short": structure_break_short,
        "recent_low": recent_low, "recent_high": recent_high,
        "distance_atr": distance_atr,
        "long_signal": long_signal, "short_signal": short_signal,
        "trigger": trigger,
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
