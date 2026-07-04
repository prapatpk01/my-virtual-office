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


# ─────────────────────────────────────────────────────────────────────────────
# Regime enum
# ─────────────────────────────────────────────────────────────────────────────

class RegimeType(str, Enum):
    TRENDING_UP    = "trending_up"
    TRENDING_DOWN  = "trending_down"
    RANGING        = "ranging"
    VOLATILE       = "volatile"
    LOW_CONVICTION = "low_conviction"


# ─────────────────────────────────────────────────────────────────────────────
# 4h  ──  Market Regime Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(candles_4h: list, min_candles: int = 40) -> tuple[RegimeType, dict]:
    """
    Classify the current market regime from 4h candles.

    Indicators used:
      - ADX(14) + ±DI: trend strength and direction
      - ATR%:          relative volatility (flags VOLATILE regime)
      - EMA(20/50):    structural trend alignment
      - OBV 10-bar slope: volume-confirmed direction

    Returns (RegimeType, debug_dict).
    """
    n = len(candles_4h)
    if n < min_candles:
        return RegimeType.LOW_CONVICTION, {"reason": "insufficient_4h_data", "n": n}

    closes = [float(c.close) for c in candles_4h]
    price  = closes[-1]

    # ── ADX ──────────────────────────────────────────────────────────────────
    adx_arr, plus_di_arr, minus_di_arr = BaseStrategy.adx(candles_4h, 14)
    adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
    pdi = float(plus_di_arr[-1])
    mdi = float(minus_di_arr[-1])

    # ── ATR % ─────────────────────────────────────────────────────────────────
    atr_arr = BaseStrategy.atr(candles_4h, 14)
    atr4h   = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
    atr_pct = atr4h / price if price > 0 else 0.0

    # ── EMA structural alignment ──────────────────────────────────────────────
    ema20_arr = BaseStrategy.ema(closes, 20)
    ema50_arr = BaseStrategy.ema(closes, 50)
    ema20 = float(ema20_arr[-1]) if not np.isnan(ema20_arr[-1]) else price
    ema50 = float(ema50_arr[-1]) if not np.isnan(ema50_arr[-1]) else price

    # ── OBV momentum slope (10-bar) ───────────────────────────────────────────
    obv_arr = BaseStrategy.obv(candles_4h)
    lookback = min(10, n - 1)
    obv_slope = (obv_arr[-1] - obv_arr[-lookback - 1]) if lookback > 0 else 0.0

    debug = {
        "adx_4h":    round(adx,    2),
        "pdi_4h":    round(pdi,    2),
        "mdi_4h":    round(mdi,    2),
        "atr_pct_4h": round(atr_pct * 100, 3),
        "ema20_4h":  round(ema20,  4),
        "ema50_4h":  round(ema50,  4),
        "obv_slope": round(float(obv_slope), 2),
    }

    # ── Classify ──────────────────────────────────────────────────────────────
    # 1. Very choppy / wide swings → VOLATILE (skip or require heavy confirmation)
    if atr_pct > 0.030:
        debug["regime_reason"] = "atr_pct_high"
        return RegimeType.VOLATILE, debug

    # 2. Weak momentum → LOW_CONVICTION
    if adx < 15:
        debug["regime_reason"] = "adx_low"
        return RegimeType.LOW_CONVICTION, debug

    # 3. Strong trend
    if adx >= 25:
        if pdi > mdi and ema20 >= ema50:
            debug["regime_reason"] = "adx_trending_up"
            return RegimeType.TRENDING_UP, debug
        if mdi > pdi and ema20 <= ema50:
            debug["regime_reason"] = "adx_trending_down"
            return RegimeType.TRENDING_DOWN, debug

    # 4. Moderate ADX or mixed DI → RANGING (mean-reversion opportunities)
    debug["regime_reason"] = "ranging"
    return RegimeType.RANGING, debug


# ─────────────────────────────────────────────────────────────────────────────
# 1h  ──  Directional Bias
# ─────────────────────────────────────────────────────────────────────────────

def directional_bias(candles_1h: list, min_candles: int = 55) -> tuple[float, dict]:
    """
    Compute a directional bias score from 1h candles.

    Score range: -3.0 (strong bear) → +3.0 (strong bull)
      +1.0 / -1.0  : EMA20 vs EMA50 structural alignment
      +0.5 / -0.5  : RSI > 55 (bull) / RSI < 45 (bear)
      +1.0 / -1.0  : MACD histogram sign (momentum confirmation)
      +0.5 / -0.5  : price position above/below EMA20

    Returns (score, debug_dict).
    """
    n = len(candles_1h)
    if n < min_candles:
        return 0.0, {"reason": "insufficient_1h_data", "n": n}

    closes = [float(c.close) for c in candles_1h]
    price  = closes[-1]

    ema20 = float(BaseStrategy.ema(closes, 20)[-1])
    ema50 = float(BaseStrategy.ema(closes, 50)[-1])
    rsi14 = float(BaseStrategy.rsi(closes, 14)[-1])
    macd_line, signal_line, histogram = BaseStrategy.macd(closes, 12, 26, 9)
    hist  = float(histogram[-1]) if not np.isnan(histogram[-1]) else 0.0

    score = 0.0

    # EMA structure
    if not np.isnan(ema20) and not np.isnan(ema50):
        score += 1.0 if ema20 > ema50 else -1.0

    # RSI momentum zone
    if not np.isnan(rsi14):
        if rsi14 > 55:
            score += 0.5
        elif rsi14 < 45:
            score -= 0.5

    # MACD histogram (momentum)
    score += 1.0 if hist > 0 else -1.0

    # Price vs EMA20 (short-term health)
    if not np.isnan(ema20) and ema20 > 0:
        score += 0.5 if price > ema20 else -0.5

    debug = {
        "ema20_1h":   round(ema20,  4) if not np.isnan(ema20) else None,
        "ema50_1h":   round(ema50,  4) if not np.isnan(ema50) else None,
        "rsi14_1h":   round(rsi14,  2) if not np.isnan(rsi14) else None,
        "macd_hist":  round(hist,   6),
        "bias_score": round(score,  3),
    }
    return score, debug


# ─────────────────────────────────────────────────────────────────────────────
# 15m  ──  Volume Filter
# ─────────────────────────────────────────────────────────────────────────────

def volume_ok(candles_15m: list, period: int = 20,
              threshold: float = 0.70) -> tuple[bool, float]:
    """
    Return (valid, vol_ratio).

    vol_ratio = current_volume / mean(last `period` volumes excluding current).
    Entry is skipped when vol_ratio < threshold (default 0.70 of the 20-bar avg).
    """
    n = len(candles_15m)
    if n < period + 1:
        return True, 1.0   # insufficient history → allow (be conservative)

    vols = [float(c.volume) for c in candles_15m]
    recent_avg = float(np.mean(vols[-(period + 1):-1]))
    current    = vols[-1]

    ratio = current / recent_avg if recent_avg > 0 else 1.0
    return ratio >= threshold, round(ratio, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 15m  ──  Entry Trigger Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_factors(
    candles_15m: list,
    side: str,              # "long" | "short"
    regime: RegimeType,
    bias: float,
    min_candles: int = 40,
) -> tuple[float, list[str], dict]:
    """
    Score a potential entry trigger on 15m candles.

    Returns (raw_score 0-1, confirmed_factors, debug_dict).

    Scoring weights:
      RSI condition     0.25
      MACD cross/hist   0.20
      Supertrend dir    0.20
      Volume impulse    0.15
      EMA9/21 alignment 0.10
      HA candle streak  0.10
    """
    n = len(candles_15m)
    if n < min_candles:
        return 0.0, [], {"reason": "insufficient_15m_data"}

    closes  = [float(c.close)  for c in candles_15m]
    volumes = [float(c.volume) for c in candles_15m]
    price   = closes[-1]

    is_long = side == "long"

    # ── Indicators ─────────────────────────────────────────────────────────────
    rsi14 = float(BaseStrategy.rsi(closes, 14)[-1])
    macd_line, signal_line, histogram = BaseStrategy.macd(closes, 12, 26, 9)
    hist     = float(histogram[-1])  if not np.isnan(histogram[-1])  else 0.0
    hist_p   = float(histogram[-2])  if len(histogram) > 1 and not np.isnan(histogram[-2]) else hist
    macd_val = float(macd_line[-1])  if not np.isnan(macd_line[-1])  else 0.0
    sig_val  = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else 0.0

    ema9_arr  = BaseStrategy.ema(closes, 9)
    ema21_arr = BaseStrategy.ema(closes, 21)
    ema9  = float(ema9_arr[-1])  if not np.isnan(ema9_arr[-1])  else price
    ema21 = float(ema21_arr[-1]) if not np.isnan(ema21_arr[-1]) else price

    st_line, st_dir = BaseStrategy.supertrend(candles_15m, period=7, multiplier=3.0)
    st_now = int(st_dir[-1]) if len(st_dir) > 0 else 0

    # HA streak
    ha_candles, _, ha_closes = BaseStrategy._heikin_ashi(candles_15m)
    ha_opens = [float(ha_candles[i].open) for i in range(n)]
    # Count how many consecutive HA candles are in our direction
    streak = 0
    for i in range(n - 1, max(n - 6, 0), -1):
        ha_bull = ha_closes[i] > ha_opens[i]
        if (is_long and ha_bull) or (not is_long and not ha_bull):
            streak += 1
        else:
            break

    # Volume impulse (vs 10-bar lookback to detect current bar's volume spike)
    vol_avg10 = float(np.mean(volumes[-11:-1])) if len(volumes) >= 11 else float(np.mean(volumes[:-1]) or 1)
    vol_ratio = volumes[-1] / vol_avg10 if vol_avg10 > 0 else 1.0

    # ── Score by factor ─────────────────────────────────────────────────────
    score = 0.0
    factors: list[str] = []

    # 1. RSI condition (regime-aware thresholds)
    if not np.isnan(rsi14):
        if regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN):
            # Trend-following: healthy momentum zone, not over-extended
            if is_long and 40 <= rsi14 <= 65:
                score += 0.25; factors.append("rsi_trend_zone")
            elif not is_long and 35 <= rsi14 <= 60:
                score += 0.25; factors.append("rsi_trend_zone")
            elif is_long and rsi14 < 40:
                # pullback / dip buy in uptrend
                score += 0.18; factors.append("rsi_dip")
            elif not is_long and rsi14 > 60:
                score += 0.18; factors.append("rsi_dip")
        elif regime == RegimeType.RANGING:
            # Counter-trend: RSI extremes
            if is_long and rsi14 < 32:
                score += 0.25; factors.append("rsi_oversold")
            elif not is_long and rsi14 > 68:
                score += 0.25; factors.append("rsi_overbought")
            elif is_long and rsi14 < 42:
                score += 0.12; factors.append("rsi_low_range")
            elif not is_long and rsi14 > 58:
                score += 0.12; factors.append("rsi_high_range")
        else:
            # VOLATILE: require strong RSI extreme
            if is_long and rsi14 < 28:
                score += 0.25; factors.append("rsi_extreme_oversold")
            elif not is_long and rsi14 > 72:
                score += 0.25; factors.append("rsi_extreme_overbought")

    # 2. MACD histogram momentum
    if is_long:
        if hist > 0 and hist >= hist_p:   # positive and rising (momentum)
            score += 0.20; factors.append("macd_bull_momentum")
        elif hist > 0:                     # positive but not accelerating
            score += 0.10; factors.append("macd_positive")
        elif hist_p < 0 and hist > hist_p: # turning up from negative
            score += 0.15; factors.append("macd_turning_bull")
        # MACD line crossing above signal
        if macd_val > sig_val:
            score += 0.05
    else:
        if hist < 0 and hist <= hist_p:
            score += 0.20; factors.append("macd_bear_momentum")
        elif hist < 0:
            score += 0.10; factors.append("macd_negative")
        elif hist_p > 0 and hist < hist_p:
            score += 0.15; factors.append("macd_turning_bear")
        if macd_val < sig_val:
            score += 0.05

    # 3. Supertrend direction
    if is_long:
        if st_now == 1:
            score += 0.20; factors.append("supertrend_up")
            # Extra for fresh flip (previous bar was -1)
            if len(st_dir) > 1 and st_dir[-2] == -1:
                score += 0.05; factors.append("supertrend_flip_bull")
    else:
        if st_now == -1:
            score += 0.20; factors.append("supertrend_down")
            if len(st_dir) > 1 and st_dir[-2] == 1:
                score += 0.05; factors.append("supertrend_flip_bear")

    # 4. Volume impulse
    if vol_ratio >= 1.5:
        score += 0.15; factors.append("volume_spike")
    elif vol_ratio >= 1.2:
        score += 0.08; factors.append("volume_above_avg")

    # 5. EMA9/EMA21 short-term alignment
    if is_long:
        if ema9 > ema21 and price > ema9:
            score += 0.10; factors.append("ema_stack_bull")
        elif price > ema21:
            score += 0.05; factors.append("price_above_ema21")
    else:
        if ema9 < ema21 and price < ema9:
            score += 0.10; factors.append("ema_stack_bear")
        elif price < ema21:
            score += 0.05; factors.append("price_below_ema21")

    # 6. Heikin-Ashi candle streak
    if streak >= 3:
        score += 0.10; factors.append(f"ha_streak_{streak}")
    elif streak >= 2:
        score += 0.06; factors.append(f"ha_streak_{streak}")

    debug = {
        "rsi14_15m":   round(rsi14, 2) if not np.isnan(rsi14) else None,
        "macd_hist":   round(hist, 6),
        "st_dir":      st_now,
        "vol_ratio10": round(vol_ratio, 4),
        "ema9_15m":    round(ema9, 4),
        "ema21_15m":   round(ema21, 4),
        "ha_streak":   streak,
        "raw_score":   round(score, 4),
    }
    return min(score, 1.0), factors, debug


def entry_threshold(regime: RegimeType) -> float:
    """Minimum score required to fire an entry signal, per regime."""
    return {
        RegimeType.TRENDING_UP:   0.45,
        RegimeType.TRENDING_DOWN: 0.45,
        RegimeType.RANGING:       0.42,
        RegimeType.VOLATILE:      0.65,
        RegimeType.LOW_CONVICTION: 1.01,  # never fires
    }.get(regime, 0.50)


# ─────────────────────────────────────────────────────────────────────────────
# Trade Plan — TP/SL ladder
# ─────────────────────────────────────────────────────────────────────────────

# SL = 1.0R, T1=0.5R / T2=0.7R / T3=1.0R / T4=1.2R (matches Position.LADDER_LEVELS)
_TP_TARGETS_R = [0.5, 0.7, 1.0, 1.2]
_SL_AFTER_R   = [0.3, 0.5, 0.8]       # SL move after T1/T2/T3 hit
_SL_DIST_MIN_PCT = 0.005               # 0.5% of price floor
_SL_DIST_MAX_PCT = 0.035               # 3.5% of price ceiling


def build_trade_plan(price: float, atr_15m: float, side: str) -> dict:
    """
    Build the TP/SL ladder metadata dict consumed by TradingBot._open_position().

    SL distance = ATR(15m) × 1.0, clamped to [0.5%, 3.5%] of price.
    T1-T4 prices are multiples of one_r (the clamped SL distance).

    Returned dict is merged into Signal.metadata.  Key fields consumed by the bot:
      one_r            — absolute 1R distance (re-anchored at fill price by bot)
      rr_tp1           — R-ratio of T1 target  (0.5)
      rr_tp2           — R-ratio of final target (1.2 = T4)
      stop_loss        — indicative initial SL price  (re-anchored by bot)
      take_profit      — indicative T4 price          (re-anchored by bot)
      tp1              — indicative T1 price           (re-anchored by bot)
      tp2              — same as take_profit for ladder mode
      sl_dist_pct      — SL distance / price (for RiskManager.size_by_risk)
      risk_pct         — portfolio risk budget (2%)
      sl_ladder_enabled — True → bot uses SL-ratchet mode (no partial closes)
      tp_ladder        — human-readable T1..T4 prices for logging / Telegram
    """
    if price <= 0 or atr_15m <= 0:
        return {"sl_ladder_enabled": False}

    # Clamp 1R distance
    raw_1r  = atr_15m * 1.0
    min_1r  = price * _SL_DIST_MIN_PCT
    max_1r  = price * _SL_DIST_MAX_PCT
    one_r   = max(min_1r, min(raw_1r, max_1r))

    is_long = side == "long"
    sign    = 1 if is_long else -1

    sl_price = round(price - sign * one_r, 8)
    tp_ladder = {}
    for i, r in enumerate(_TP_TARGETS_R):
        label = f"T{i + 1}"
        tp_ladder[label] = round(price + sign * r * one_r, 8)

    # Indicative SL move levels for metadata (human-readable; bot recalculates from one_r)
    sl_ladder = {}
    for i, (trig_r, new_sl_r) in enumerate(zip(_TP_TARGETS_R[:3], _SL_AFTER_R)):
        trig_label = f"T{i + 1}_hit_sl_moves_to"
        sl_ladder[trig_label] = round(price + sign * new_sl_r * one_r, 8)

    return {
        # ── Bot-critical fields ───────────────────────────────────────────
        "one_r":             round(one_r, 8),
        "rr_tp1":            _TP_TARGETS_R[0],      # 0.5
        "rr_tp2":            _TP_TARGETS_R[-1],     # 1.2
        "stop_loss":         sl_price,
        "take_profit":       tp_ladder["T4"],
        "tp1":               tp_ladder["T1"],
        "tp2":               tp_ladder["T4"],
        "sl_dist_pct":       round(one_r / price, 8),
        "risk_pct":          0.02,
        "sl_ladder_enabled": True,
        # ── Informational ─────────────────────────────────────────────────
        "tp_ladder":         tp_ladder,
        "sl_ladder":         sl_ladder,
        "atr_1r_raw":        round(atr_15m, 8),
    }
