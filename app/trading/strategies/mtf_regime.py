"""
Multi-timeframe regime detection and signal-scoring helpers.

Used by AISignalStrategy but importable by any strategy that needs:
  - Market regime classification (4h candles)
  - Directional bias scoring   (1h candles)
  - Volume validity filter     (15m candles)
  - Entry trigger scoring      (15m candles)
  - TP/SL ladder trade plan    (price + ATR → metadata dict)
"""
from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Optional

from .base import BaseStrategy


class RegimeType(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_CONVICTION = "low_conviction"


_VOLATILE_ATR_PCT = 0.030
_LOW_CONVICTION_ADX = 15
_STRONG_TREND_ADX = 25

_RSI_BULL_BIAS = 55.0
_RSI_BEAR_BIAS = 45.0
_RSI_BULL_BIAS_STRONG = 60.0
_RSI_BEAR_BIAS_STRONG = 40.0
_BIAS_NEUTRAL_RSI_LOW = 48.0
_BIAS_NEUTRAL_RSI_HIGH = 52.0

_RSI_TREND_LONG_MIN = 40
_RSI_TREND_LONG_MAX = 65
_RSI_TREND_SHORT_MIN = 35
_RSI_TREND_SHORT_MAX = 60

_RSI_OVERSOLD_RANGE = 32
_RSI_OVERBOUGHT_RANGE = 68
_RSI_SOFT_LOW_RANGE = 42
_RSI_SOFT_HIGH_RANGE = 58

_RSI_EXTREME_OVERSOLD = 28
_RSI_EXTREME_OVERBOUGHT = 72

_VOL_SPIKE_RATIO = 1.5
_VOL_ABOVE_AVG_RATIO = 1.2

_HA_STRONG_STREAK = 3
_HA_MODERATE_STREAK = 2

_THRESHOLD_TRENDING = 0.45
_THRESHOLD_RANGING = 0.42
_THRESHOLD_VOLATILE = 0.65
_THRESHOLD_DISABLED = 1.01

BIAS_MISALIGN_LONG_MIN = -1.0
BIAS_MISALIGN_SHORT_MAX = 1.0


def detect_regime(candles_4h: list, min_candles: int = 40) -> tuple[RegimeType, dict]:
    n = len(candles_4h)
    if n < min_candles:
        return RegimeType.LOW_CONVICTION, {"reason": "insufficient_4h_data", "n": n}

    closes = [float(c.close) for c in candles_4h]
    price = closes[-1]

    adx_arr, plus_di_arr, minus_di_arr = BaseStrategy.adx(candles_4h, 14)
    adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
    pdi = float(plus_di_arr[-1])
    mdi = float(minus_di_arr[-1])

    atr_arr = BaseStrategy.atr(candles_4h, 14)
    atr4h = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
    atr_pct = atr4h / price if price > 0 else 0.0

    ema20_arr = BaseStrategy.ema(closes, 20)
    ema50_arr = BaseStrategy.ema(closes, 50)
    ema20 = float(ema20_arr[-1]) if not np.isnan(ema20_arr[-1]) else price
    ema50 = float(ema50_arr[-1]) if not np.isnan(ema50_arr[-1]) else price

    obv_arr = BaseStrategy.obv(candles_4h)
    lookback = min(10, n - 1)
    obv_slope = (obv_arr[-1] - obv_arr[-lookback - 1]) if lookback > 0 else 0.0

    debug = {
        "adx_4h": round(adx, 2),
        "pdi_4h": round(pdi, 2),
        "mdi_4h": round(mdi, 2),
        "atr_pct_4h": round(atr_pct * 100, 3),
        "ema20_4h": round(ema20, 4),
        "ema50_4h": round(ema50, 4),
        "obv_slope": round(float(obv_slope), 2),
    }

    if atr_pct > _VOLATILE_ATR_PCT:
        debug["regime_reason"] = "atr_pct_high"
        return RegimeType.VOLATILE, debug

    if adx < _LOW_CONVICTION_ADX:
        debug["regime_reason"] = "adx_low"
        return RegimeType.LOW_CONVICTION, debug

    if adx >= _STRONG_TREND_ADX:
        if pdi > mdi and ema20 >= ema50:
            debug["regime_reason"] = "adx_trending_up"
            return RegimeType.TRENDING_UP, debug
        if mdi > pdi and ema20 <= ema50:
            debug["regime_reason"] = "adx_trending_down"
            return RegimeType.TRENDING_DOWN, debug

    debug["regime_reason"] = "ranging"
    return RegimeType.RANGING, debug


def directional_bias(candles_1h: list, min_candles: int = 55) -> tuple[float, dict]:
    n = len(candles_1h)
    if n < min_candles:
        return 0.0, {"reason": "insufficient_1h_data", "n": n}

    closes = [float(c.close) for c in candles_1h]
    price = closes[-1]

    ema20_arr = BaseStrategy.ema(closes, 20)
    ema50_arr = BaseStrategy.ema(closes, 50)
    ema20 = float(ema20_arr[-1]) if not np.isnan(ema20_arr[-1]) else price
    ema50 = float(ema50_arr[-1]) if not np.isnan(ema50_arr[-1]) else price
    ema20_prev = float(ema20_arr[-2]) if len(ema20_arr) > 1 and not np.isnan(ema20_arr[-2]) else ema20
    ema50_prev = float(ema50_arr[-2]) if len(ema50_arr) > 1 and not np.isnan(ema50_arr[-2]) else ema50

    rsi14 = float(BaseStrategy.rsi(closes, 14)[-1])
    macd_line, signal_line, histogram = BaseStrategy.macd(closes, 12, 26, 9)
    hist = float(histogram[-1]) if not np.isnan(histogram[-1]) else 0.0
    hist_prev = float(histogram[-2]) if len(histogram) > 1 and not np.isnan(histogram[-2]) else hist

    ema_gap_pct = ((ema20 - ema50) / price) if price > 0 else 0.0
    price_vs_ema20_pct = ((price - ema20) / ema20) if ema20 not in (0, np.nan) else 0.0
    ema20_slope_pct = ((ema20 - ema20_prev) / ema20_prev) if ema20_prev not in (0, np.nan) else 0.0
    ema50_slope_pct = ((ema50 - ema50_prev) / ema50_prev) if ema50_prev not in (0, np.nan) else 0.0
    macd_hist_strength = (hist / price) if price > 0 else 0.0
    hist_delta = hist - hist_prev

    score = 0.0
    components = {}

    ema_component = 0.0
    if not np.isnan(ema20) and not np.isnan(ema50):
        if ema_gap_pct > 0.006:
            ema_component = 1.0
        elif ema_gap_pct > 0.002:
            ema_component = 0.75
        elif ema_gap_pct > 0.0005:
            ema_component = 0.40
        elif ema_gap_pct < -0.006:
            ema_component = -1.0
        elif ema_gap_pct < -0.002:
            ema_component = -0.75
        elif ema_gap_pct < -0.0005:
            ema_component = -0.40
        score += ema_component
    components["ema_structure"] = round(ema_component, 4)

    rsi_component = 0.0
    if not np.isnan(rsi14):
        if rsi14 >= _RSI_BULL_BIAS_STRONG:
            rsi_component = 0.60
        elif rsi14 > _RSI_BULL_BIAS:
            rsi_component = 0.30
        elif rsi14 <= _RSI_BEAR_BIAS_STRONG:
            rsi_component = -0.60
        elif rsi14 < _RSI_BEAR_BIAS:
            rsi_component = -0.30
        elif _BIAS_NEUTRAL_RSI_LOW <= rsi14 <= _BIAS_NEUTRAL_RSI_HIGH:
            rsi_component = 0.0
        elif rsi14 > _BIAS_NEUTRAL_RSI_HIGH:
            rsi_component = 0.10
        elif rsi14 < _BIAS_NEUTRAL_RSI_LOW:
            rsi_component = -0.10
        score += rsi_component
    components["rsi_momentum"] = round(rsi_component, 4)

    macd_component = 0.0
    if hist > 0:
        macd_component = 0.50
        if hist_delta > 0:
            macd_component += 0.25
        if macd_hist_strength > 0.0008:
            macd_component += 0.25
    elif hist < 0:
        macd_component = -0.50
        if hist_delta < 0:
            macd_component -= 0.25
        if macd_hist_strength < -0.0008:
            macd_component -= 0.25
    score += macd_component
    components["macd_momentum"] = round(macd_component, 4)

    price_component = 0.0
    if not np.isnan(ema20) and ema20 > 0:
        if price_vs_ema20_pct > 0.004:
            price_component = 0.40
        elif price_vs_ema20_pct > 0.001:
            price_component = 0.20
        elif price_vs_ema20_pct < -0.004:
            price_component = -0.40
        elif price_vs_ema20_pct < -0.001:
            price_component = -0.20
        score += price_component
    components["price_vs_ema20"] = round(price_component, 4)

    slope_component = 0.0
    slope_sum = ema20_slope_pct + ema50_slope_pct
    if slope_sum > 0.0025:
        slope_component = 0.20
    elif slope_sum < -0.0025:
        slope_component = -0.20
    score += slope_component
    components["ema_slope"] = round(slope_component, 4)

    score = max(-3.0, min(score, 3.0))

    debug = {
        "ema20_1h": round(ema20, 4) if not np.isnan(ema20) else None,
        "ema50_1h": round(ema50, 4) if not np.isnan(ema50) else None,
        "rsi14_1h": round(rsi14, 2) if not np.isnan(rsi14) else None,
        "macd_hist": round(hist, 6),
        "macd_hist_prev": round(hist_prev, 6),
        "hist_delta": round(hist_delta, 6),
        "ema_gap_pct": round(ema_gap_pct * 100, 4),
        "price_vs_ema20_pct": round(price_vs_ema20_pct * 100, 4),
        "ema20_slope_pct": round(ema20_slope_pct * 100, 4),
        "ema50_slope_pct": round(ema50_slope_pct * 100, 4),
        "macd_hist_strength": round(macd_hist_strength * 100, 6),
        "bias_components": components,
        "bias_score": round(score, 3),
    }
    return score, debug


def volume_ok(candles_15m: list, period: int = 20, threshold: float = 0.70) -> tuple[bool, float]:
    n = len(candles_15m)
    if n < period + 1:
        return True, 1.0

    vols = [float(c.volume) for c in candles_15m]
    recent_avg = float(np.mean(vols[-(period + 1):-1]))
    current = vols[-1]

    ratio = current / recent_avg if recent_avg > 0 else 1.0
    return ratio >= threshold, round(ratio, 4)


def _score_factors(
    candles_15m: list,
    side: str,
    regime: RegimeType,
    bias: float,
    min_candles: int = 40,
) -> tuple[float, list[str], dict]:
    """
    Score a potential entry trigger on 15m candles.

    Step 2 hardening:
      - Factor weights adapt more clearly by regime.
      - Trend regime favors continuation + aligned pullbacks.
      - Range regime favors mean-reversion + exhaustion.
      - Volatile regime demands stronger confirmation and volume support.
      - Bias alignment now affects factor quality instead of only downstream gating.
    """
    n = len(candles_15m)
    if n < min_candles:
        return 0.0, [], {"reason": "insufficient_15m_data"}

    closes = [float(c.close) for c in candles_15m]
    volumes = [float(c.volume) for c in candles_15m]
    price = closes[-1]
    is_long = side == "long"

    rsi14 = float(BaseStrategy.rsi(closes, 14)[-1])
    macd_line, signal_line, histogram = BaseStrategy.macd(closes, 12, 26, 9)
    hist = float(histogram[-1]) if not np.isnan(histogram[-1]) else 0.0
    hist_p = float(histogram[-2]) if len(histogram) > 1 and not np.isnan(histogram[-2]) else hist
    macd_val = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0.0
    sig_val = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else 0.0

    ema9_arr = BaseStrategy.ema(closes, 9)
    ema21_arr = BaseStrategy.ema(closes, 21)
    ema9 = float(ema9_arr[-1]) if not np.isnan(ema9_arr[-1]) else price
    ema21 = float(ema21_arr[-1]) if not np.isnan(ema21_arr[-1]) else price
    ema9_prev = float(ema9_arr[-2]) if len(ema9_arr) > 1 and not np.isnan(ema9_arr[-2]) else ema9
    ema21_prev = float(ema21_arr[-2]) if len(ema21_arr) > 1 and not np.isnan(ema21_arr[-2]) else ema21

    st_line, st_dir = BaseStrategy.supertrend(candles_15m, period=7, multiplier=3.0)
    st_now = int(st_dir[-1]) if len(st_dir) > 0 else 0
    st_prev = int(st_dir[-2]) if len(st_dir) > 1 else st_now

    ha_candles, _, ha_closes = BaseStrategy._heikin_ashi(candles_15m)
    ha_opens = [float(ha_candles[i].open) for i in range(n)]
    streak = 0
    for i in range(n - 1, max(n - 6, 0), -1):
        ha_bull = ha_closes[i] > ha_opens[i]
        if (is_long and ha_bull) or (not is_long and not ha_bull):
            streak += 1
        else:
            break

    vol_avg10 = float(np.mean(volumes[-11:-1])) if len(volumes) >= 11 else float(np.mean(volumes[:-1]) or 1)
    vol_ratio = volumes[-1] / vol_avg10 if vol_avg10 > 0 else 1.0

    ema_gap_pct = ((ema9 - ema21) / price) if price > 0 else 0.0
    ema_slope_pct = ((ema9 - ema9_prev) / ema9_prev) if ema9_prev not in (0, np.nan) else 0.0
    bias_aligned = (is_long and bias > 0) or ((not is_long) and bias < 0)
    bias_strength = abs(float(bias))

    score = 0.0
    factors: list[str] = []
    components = {
        "rsi": 0.0,
        "macd": 0.0,
        "supertrend": 0.0,
        "volume": 0.0,
        "ema": 0.0,
        "heikin_ashi": 0.0,
        "bias_alignment": 0.0,
    }

    def add_component(name: str, value: float, label: Optional[str] = None):
        nonlocal score
        score += value
        components[name] += value
        if label:
            factors.append(label)

    if regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN):
        if not np.isnan(rsi14):
            if is_long and _RSI_TREND_LONG_MIN <= rsi14 <= _RSI_TREND_LONG_MAX:
                add_component("rsi", 0.24, "rsi_trend_zone")
            elif not is_long and _RSI_TREND_SHORT_MIN <= rsi14 <= _RSI_TREND_SHORT_MAX:
                add_component("rsi", 0.24, "rsi_trend_zone")
            elif is_long and (_RSI_TREND_LONG_MIN - 8) <= rsi14 < _RSI_TREND_LONG_MIN:
                add_component("rsi", 0.16, "rsi_pullback")
            elif not is_long and _RSI_TREND_SHORT_MAX < rsi14 <= (_RSI_TREND_SHORT_MAX + 8):
                add_component("rsi", 0.16, "rsi_pullback")

        if is_long:
            if hist > 0 and hist >= hist_p:
                add_component("macd", 0.22, "macd_bull_momentum")
            elif hist > 0:
                add_component("macd", 0.12, "macd_positive")
            elif hist_p < 0 and hist > hist_p:
                add_component("macd", 0.10, "macd_turning_bull")
            if macd_val > sig_val:
                add_component("macd", 0.04)
        else:
            if hist < 0 and hist <= hist_p:
                add_component("macd", 0.22, "macd_bear_momentum")
            elif hist < 0:
                add_component("macd", 0.12, "macd_negative")
            elif hist_p > 0 and hist < hist_p:
                add_component("macd", 0.10, "macd_turning_bear")
            if macd_val < sig_val:
                add_component("macd", 0.04)

        if is_long and st_now == 1:
            add_component("supertrend", 0.18, "supertrend_up")
            if st_prev == -1:
                add_component("supertrend", 0.04, "supertrend_flip_bull")
        elif (not is_long) and st_now == -1:
            add_component("supertrend", 0.18, "supertrend_down")
            if st_prev == 1:
                add_component("supertrend", 0.04, "supertrend_flip_bear")

        if vol_ratio >= _VOL_SPIKE_RATIO:
            add_component("volume", 0.12, "volume_spike")
        elif vol_ratio >= _VOL_ABOVE_AVG_RATIO:
            add_component("volume", 0.07, "volume_above_avg")

        if is_long:
            if ema9 > ema21 and price > ema9:
                add_component("ema", 0.10, "ema_stack_bull")
            elif ema_gap_pct > 0 and price > ema21:
                add_component("ema", 0.06, "price_above_ema21")
        else:
            if ema9 < ema21 and price < ema9:
                add_component("ema", 0.10, "ema_stack_bear")
            elif ema_gap_pct < 0 and price < ema21:
                add_component("ema", 0.06, "price_below_ema21")

        if streak >= _HA_STRONG_STREAK:
            add_component("heikin_ashi", 0.08, f"ha_streak_{streak}")
        elif streak >= _HA_MODERATE_STREAK:
            add_component("heikin_ashi", 0.05, f"ha_streak_{streak}")

    elif regime == RegimeType.RANGING:
        if not np.isnan(rsi14):
            if is_long and rsi14 < _RSI_OVERSOLD_RANGE:
                add_component("rsi", 0.28, "rsi_oversold")
            elif not is_long and rsi14 > _RSI_OVERBOUGHT_RANGE:
                add_component("rsi", 0.28, "rsi_overbought")
            elif is_long and rsi14 < _RSI_SOFT_LOW_RANGE:
                add_component("rsi", 0.14, "rsi_low_range")
            elif not is_long and rsi14 > _RSI_SOFT_HIGH_RANGE:
                add_component("rsi", 0.14, "rsi_high_range")

        if is_long:
            if hist_p < 0 and hist > hist_p:
                add_component("macd", 0.18, "macd_turning_bull")
            elif hist > 0:
                add_component("macd", 0.08, "macd_positive")
        else:
            if hist_p > 0 and hist < hist_p:
                add_component("macd", 0.18, "macd_turning_bear")
            elif hist < 0:
                add_component("macd", 0.08, "macd_negative")

        if is_long and st_prev == -1 and st_now == 1:
            add_component("supertrend", 0.14, "supertrend_flip_bull")
        elif (not is_long) and st_prev == 1 and st_now == -1:
            add_component("supertrend", 0.14, "supertrend_flip_bear")
        elif (is_long and st_now == 1) or ((not is_long) and st_now == -1):
            add_component("supertrend", 0.08, "supertrend_support")

        if vol_ratio >= _VOL_SPIKE_RATIO:
            add_component("volume", 0.10, "volume_spike")
        elif vol_ratio >= _VOL_ABOVE_AVG_RATIO:
            add_component("volume", 0.05, "volume_above_avg")

        if is_long and ema_slope_pct >= 0:
            add_component("ema", 0.05, "ema_slope_support")
        elif (not is_long) and ema_slope_pct <= 0:
            add_component("ema", 0.05, "ema_slope_support")

        if streak >= _HA_STRONG_STREAK:
            add_component("heikin_ashi", 0.08, f"ha_streak_{streak}")
        elif streak >= _HA_MODERATE_STREAK:
            add_component("heikin_ashi", 0.05, f"ha_streak_{streak}")

    else:
        if not np.isnan(rsi14):
            if is_long and rsi14 < _RSI_EXTREME_OVERSOLD:
                add_component("rsi", 0.26, "rsi_extreme_oversold")
            elif not is_long and rsi14 > _RSI_EXTREME_OVERBOUGHT:
                add_component("rsi", 0.26, "rsi_extreme_overbought")

        if is_long:
            if hist_p < 0 and hist > hist_p:
                add_component("macd", 0.14, "macd_turning_bull")
            if macd_val > sig_val and hist > hist_p:
                add_component("macd", 0.08, "macd_cross_support")
        else:
            if hist_p > 0 and hist < hist_p:
                add_component("macd", 0.14, "macd_turning_bear")
            if macd_val < sig_val and hist < hist_p:
                add_component("macd", 0.08, "macd_cross_support")

        if vol_ratio >= _VOL_SPIKE_RATIO:
            add_component("volume", 0.18, "volume_spike")
        elif vol_ratio >= _VOL_ABOVE_AVG_RATIO:
            add_component("volume", 0.08, "volume_above_avg")

        if (is_long and st_prev == -1 and st_now == 1) or ((not is_long) and st_prev == 1 and st_now == -1):
            add_component("supertrend", 0.16, "supertrend_flip")
        elif (is_long and st_now == 1) or ((not is_long) and st_now == -1):
            add_component("supertrend", 0.08, "supertrend_hold")

        if is_long and ema9 > ema21 and price > ema21:
            add_component("ema", 0.06, "ema_reclaim")
        elif (not is_long) and ema9 < ema21 and price < ema21:
            add_component("ema", 0.06, "ema_reclaim")

        if streak >= _HA_STRONG_STREAK:
            add_component("heikin_ashi", 0.10, f"ha_streak_{streak}")
        elif streak >= _HA_MODERATE_STREAK:
            add_component("heikin_ashi", 0.06, f"ha_streak_{streak}")

    if bias_aligned:
        if bias_strength >= 2.0:
            add_component("bias_alignment", 0.08, "bias_aligned_strong")
        elif bias_strength >= 1.0:
            add_component("bias_alignment", 0.05, "bias_aligned")
        elif bias_strength >= 0.35:
            add_component("bias_alignment", 0.02, "bias_aligned_soft")
    else:
        if regime == RegimeType.VOLATILE and bias_strength >= 1.0:
            add_component("bias_alignment", -0.04, "bias_counter_volatile")
        elif regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN) and bias_strength >= 1.0:
            add_component("bias_alignment", -0.03, "bias_counter_trend")

    raw_score = min(score, 1.0)
    debug = {
        "rsi14_15m": round(rsi14, 2) if not np.isnan(rsi14) else None,
        "macd_hist": round(hist, 6),
        "macd_hist_prev": round(hist_p, 6),
        "st_dir": st_now,
        "st_prev": st_prev,
        "vol_ratio10": round(vol_ratio, 4),
        "ema9_15m": round(ema9, 4),
        "ema21_15m": round(ema21, 4),
        "ema_gap_pct": round(ema_gap_pct * 100, 4),
        "ema_slope_pct": round(ema_slope_pct * 100, 4),
        "ha_streak": streak,
        "bias_aligned": bias_aligned,
        "bias_strength": round(bias_strength, 4),
        "score_components": {k: round(v, 4) for k, v in components.items()},
        "raw_score": round(score, 4),
        "capped_score": round(raw_score, 4),
    }
    return raw_score, factors, debug


def entry_threshold(regime: RegimeType) -> float:
    """Minimum score required to fire an entry signal, per regime."""
    return {
        RegimeType.TRENDING_UP: _THRESHOLD_TRENDING,
        RegimeType.TRENDING_DOWN: _THRESHOLD_TRENDING,
        RegimeType.RANGING: _THRESHOLD_RANGING,
        RegimeType.VOLATILE: _THRESHOLD_VOLATILE,
        RegimeType.LOW_CONVICTION: _THRESHOLD_DISABLED,
    }.get(regime, 0.50)


_TP_TARGETS_R = [0.5, 0.7, 1.0, 1.2]
_SL_AFTER_R = [0.3, 0.5, 0.8]
_SL_DIST_MIN_PCT = 0.005
_SL_DIST_MAX_PCT = 0.035


def build_trade_plan(price: float, atr_15m: float, side: str) -> dict:
    if price <= 0 or atr_15m <= 0:
        return {"sl_ladder_enabled": False}

    raw_1r = atr_15m * 1.0
    min_1r = price * _SL_DIST_MIN_PCT
    max_1r = price * _SL_DIST_MAX_PCT
    one_r = max(min_1r, min(raw_1r, max_1r))

    is_long = side == "long"
    sign = 1 if is_long else -1

    sl_price = round(price - sign * one_r, 8)
    tp_ladder = {}
    for i, r in enumerate(_TP_TARGETS_R):
        label = f"T{i + 1}"
        tp_ladder[label] = round(price + sign * r * one_r, 8)

    sl_ladder = {}
    for i, (trig_r, new_sl_r) in enumerate(zip(_TP_TARGETS_R[:3], _SL_AFTER_R)):
        trig_label = f"T{i + 1}_hit_sl_moves_to"
        sl_ladder[trig_label] = round(price + sign * new_sl_r * one_r, 8)

    return {
        "one_r": round(one_r, 8),
        "rr_tp1": _TP_TARGETS_R[0],
        "rr_tp2": _TP_TARGETS_R[-1],
        "stop_loss": sl_price,
        "take_profit": tp_ladder["T4"],
        "tp1": tp_ladder["T1"],
        "tp2": tp_ladder["T4"],
        "sl_dist_pct": round(one_r / price, 8),
        "risk_pct": 0.02,
        "sl_ladder_enabled": True,
        "tp_ladder": tp_ladder,
        "sl_ladder": sl_ladder,
        "atr_1r_raw": round(atr_15m, 8),
    }
