"""Adaptive Trading Bot V5.2 indicator engine — 15M Dual Engine.

Only two entry engines are active:
- LEGACY_MOMENTUM: continuation structure break (BOS/CHOCH) with EMA8/13 bias,
  ADX/CHOP quality, and 2-of-3 confirmation from MACD, ROC9 and structure trend.
- BREAKOUT: volatility expansion + local swing break + ROC9. BREAKOUT has priority.

TREND and MEAN entries are intentionally disabled. The engine still computes the
same core indicators so existing position management and charts remain compatible.
Signals are designed for CLOSED 15M candles only.
"""
from __future__ import annotations

from typing import Any, Dict, List
import math

# Keep schema stable across Railway rolling commits; the deployed bot explicitly
# identifies the new behaviour through BUILD_ID / strategy / style.
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

    ema_bull = e8[-1] > e13[-1]
    ema_bear = e8[-1] < e13[-1]
    ema_cross_up = ema_bull and e8[-2] <= e13[-2]
    ema_cross_down = ema_bear and e8[-2] >= e13[-2]

    # Local structure. The five-bar swing is used for full BOS; the three-bar
    # micro swing allows a CHOCH/re-acceleration trigger without waiting too late.
    swing_high = max(highs[-6:-1])
    swing_low = min(lows[-6:-1])
    micro_high = max(highs[-4:-1])
    micro_low = min(lows[-4:-1])
    bullish_bos = closes[-1] > swing_high
    bearish_bos = closes[-1] < swing_low
    bullish_choch = (
        ema_bull
        and closes[-1] > micro_high
        and closes[-2] <= micro_high
        and closes[-1] > opens[-1]
    )
    bearish_choch = (
        ema_bear
        and closes[-1] < micro_low
        and closes[-2] >= micro_low
        and closes[-1] < opens[-1]
    )

    # Structure trend confirmation: compare the most recent three completed
    # swings with the preceding three. This approximates HH/HL and LL/LH.
    new_high = max(highs[-4:-1])
    old_high = max(highs[-7:-4])
    new_low = min(lows[-4:-1])
    old_low = min(lows[-7:-4])
    structure_bull = new_high > old_high and new_low > old_low
    structure_bear = new_high < old_high and new_low < old_low

    macd_bull = macd_line[-1] > macd_signal[-1]
    macd_bear = macd_line[-1] < macd_signal[-1]
    roc_bull = roc9 > 0.0
    roc_bear = roc9 < 0.0
    legacy_long_score = int(macd_bull) + int(roc_bull) + int(structure_bull)
    legacy_short_score = int(macd_bear) + int(roc_bear) + int(structure_bear)

    legacy_quality = adx >= 15.0 and chop <= 55.0
    legacy_long = (
        legacy_quality
        and ema_bull
        and (bullish_bos or bullish_choch)
        and legacy_long_score >= 2
    )
    legacy_short = (
        legacy_quality
        and ema_bear
        and (bearish_bos or bearish_choch)
        and legacy_short_score >= 2
    )

    # BREAKOUT: volatility transition has priority over Legacy Momentum. The
    # old hard requirement for a close outside the Bollinger band is removed;
    # BB width expansion is the volatility evidence, which reduces late entries.
    breakout_regime = adx_rising and 45.0 <= chop <= 55.0 and bb_expanding and atr_rising
    breakout_long = breakout_regime and bullish_bos and roc_bull
    breakout_short = breakout_regime and bearish_bos and roc_bear

    # Router priority: BREAKOUT -> LEGACY_MOMENTUM -> WAIT.
    if breakout_long or breakout_short:
        regime = "BREAKOUT"
    elif legacy_long or legacy_short:
        regime = "LEGACY_MOMENTUM"
    else:
        regime = "WAIT"

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
        "micro_high": micro_high, "micro_low": micro_low,
        "recent_low": recent_low, "recent_high": recent_high,
        "bullish_bos": bullish_bos, "bearish_bos": bearish_bos,
        "bullish_choch": bullish_choch, "bearish_choch": bearish_choch,
        "structure_bull": structure_bull, "structure_bear": structure_bear,
        "legacy_long_score": legacy_long_score, "legacy_short_score": legacy_short_score,
        "legacy_long": legacy_long, "legacy_short": legacy_short,
        "breakout_regime": breakout_regime,
        "breakout_long": breakout_long, "breakout_short": breakout_short,
        # Explicitly disabled styles retained as compatibility fields.
        "trend_long": False, "trend_short": False,
        "mean_long": False, "mean_short": False,
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
