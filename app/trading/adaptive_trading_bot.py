"""
Adaptive Trading Bot — v8.0  SwingReversalPro + MeanReversion
==============================================================
7 major improvements from V7:

[V8-1] Market State Engine — 8 states redesigned:
       STRONG_TREND / TRENDING / SIDEWAY / LOW_VOL / HIGH_VOL
       EXHAUSTION / BREAKOUT / REVERSAL
       Using: ADX, ATR%, EMA Slope, BB Width, Volume, RSI
       Tradeable: STRONG_TREND, TRENDING, BREAKOUT, REVERSAL, HIGH_VOL, EXHAUSTION, SIDEWAY
       Skip: LOW_VOL immediately

[V8-2] Adaptive Thresholds — RSI/MACD/ADX/ATR gates adapt per market state:
       TRENDING: RSI<45 (dip-buy)  |  SIDEWAY: RSI<35  |  EXHAUSTION: RSI<50

[V8-3] Entry Health Score (0-100) — composite quality gate before entry:
       EMA Trend:20  MACD:15  ADX:15  Volume:15  ATR:10  RSI:10  Pattern:15  State:10
       ≥75→full position  65-74→reduced  <65→skip

[V8-4] Confidence Score (0-100) — position conviction level:
       Trend:20  Momentum:20  Volume:15  Volatility:15  Structure:15  Pattern:15
       95→Strong, 85→Buy, 75→Normal, 65→Weak, <65→Skip

[V8-5] Regime Bias Engine — 5-level trend bias from 4H+1H:
       STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR
       Long only: BULL, STRONG_BULL
       Short only: BEAR, STRONG_BEAR
       NEUTRAL: mean-revert states only (SIDEWAY, REVERSAL)

[V8-6] Cooldown (tighter):
       2 consecutive SL hits → 90-min cooldown
       3 losses in session  → 4-hour cooldown

[V8-7] Pattern Learning Engine (simplified, anti-overfit):
       Track per entry_type: Wins, Losses, Avg R, Avg Hold, Expectancy
       WR<45% → reduce weight ×0.85 (min 0.5)
       WR>65% → increase weight ×1.15 (max 1.5)

All V7 infrastructure preserved:
- State machine (SCANNING→FILTERING→WAIT_CONFIRM→PENDING_ORDER→IN_POSITION→EXITING)
- Partial TP (TP1 50% + TP2 full), Break-even, ATR Trailing Stop
- Position Health Calculator, Reversal Spike/Trend Fade protection
- save_state/load_state/reconcile_with_exchange
- Daily PnL limits, win-streak risk reduction
"""

import numpy as np
import datetime
import json
import os
import logging
from typing import Optional, Callable, Dict, List, Any

from .strategies.mean_reversion import MeanReversionStrategy


logger = logging.getLogger("adaptive_trading_bot")


# ──────────────────────────────────────────────────────────────────────────────
# [V8-2] ADAPTIVE THRESHOLDS  — per 8-state market
# RSI, MACD mode, ADX min, health min  (direction gates applied separately by Regime Bias)
# ──────────────────────────────────────────────────────────────────────────────

ADAPTIVE_THRESHOLDS: Dict[str, Dict] = {
    # Strong directional trend — dip-buy / dip-sell entry, lower RSI bar
    "STRONG_TREND": {
        "rsi_long":  (38, 65), "rsi_short": (35, 62),
        "macd_long": "above_signal", "macd_short": "below_signal",
        "adx_min": 22, "health_min": 68, "confidence_min": 65,
    },
    # Normal trend — wait for pullback to RSI<45 (dip-buy zone)
    "TRENDING": {
        "rsi_long":  (30, 55), "rsi_short": (45, 65),
        "macd_long": "above_signal", "macd_short": "below_signal",
        "adx_min": 18, "health_min": 72, "confidence_min": 65,
    },
    # Breakout from compression — momentum entry, RSI expansive
    "BREAKOUT": {
        "rsi_long":  (42, 72), "rsi_short": (28, 58),
        "macd_long": "above_zero", "macd_short": "below_zero",
        "adx_min": 15, "health_min": 72, "confidence_min": 68,
    },
    # Counter-trend reversal — deep OS/OB required
    "REVERSAL": {
        "rsi_long":  (25, 45), "rsi_short": (55, 75),
        "macd_long": "turning_up", "macd_short": "turning_down",
        "adx_min": 12, "health_min": 75, "confidence_min": 68,
    },
    # Ranging market — mean-revert gates
    "SIDEWAY": {
        "rsi_long":  (28, 48), "rsi_short": (52, 72),
        "macd_long": "any", "macd_short": "any",
        "adx_min": 10, "health_min": 72, "confidence_min": 62,
    },
    # Volatility expansion — very strict entry
    "HIGH_VOL": {
        "rsi_long":  (35, 55), "rsi_short": (45, 65),
        "macd_long": "above_signal", "macd_short": "below_signal",
        "adx_min": 22, "health_min": 80, "confidence_min": 72,
    },
    # Trend losing momentum — deeper OS/OB, MACD must be turning
    "EXHAUSTION": {
        "rsi_long":  (25, 44), "rsi_short": (56, 75),
        "macd_long": "turning_up", "macd_short": "turning_down",
        "adx_min": 18, "health_min": 78, "confidence_min": 70,
    },
    # Squeeze phase — skip immediately
    "LOW_VOL": {
        "rsi_long": (50, 50), "rsi_short": (50, 50),
        "macd_long": "any", "macd_short": "any",
        "adx_min": 999, "health_min": 999, "confidence_min": 999,
    },
}

# States the bot will trade in — LOW_VOL skipped entirely
_TRADEABLE_STATES: frozenset = frozenset({
    "STRONG_TREND", "TRENDING", "BREAKOUT", "REVERSAL",
    "SIDEWAY", "HIGH_VOL", "EXHAUSTION",
})

# Which states trade counter-direction of Regime Bias
# (reversal states want OPPOSITE of macro trend)
_COUNTER_TREND_STATES: frozenset = frozenset({"REVERSAL", "EXHAUSTION"})

# Entry type label per state — for learning engine
_STATE_ENTRY_TYPE: Dict[str, str] = {
    "STRONG_TREND": "trend_follow",
    "TRENDING":     "trend_follow",
    "BREAKOUT":     "breakout",
    "REVERSAL":     "reversal",
    "SIDEWAY":      "mean_revert",
    "HIGH_VOL":     "momentum",
    "EXHAUSTION":   "counter_trend",
    "LOW_VOL":      "none",
}

# Invert-keys for SHORT scoring (same as V7)
_DIR_INVERT_KEYS: tuple = ("momentum", "structure", "ema", "vwap", "macd")


# ──────────────────────────────────────────────────────────────────────────────
# [V8-3] ENTRY HEALTH SCORER
# 100-pt composite: EMA(20) + MACD(15) + ADX(15) + Volume(15) + ATR(10)
#                    + RSI(10) + Pattern(15) + Market State(10)
# ──────────────────────────────────────────────────────────────────────────────

class EntryHealthScorer:
    """Compute a 0-100 entry quality score before opening a position."""

    _STATE_QUALITY = {
        "STRONG_TREND": 95, "TRENDING": 80, "BREAKOUT": 88,
        "REVERSAL": 72, "HIGH_VOL": 65, "EXHAUSTION": 68,
        "SIDEWAY": 60, "LOW_VOL": 5,
    }

    def compute(self, ind: Dict, direction: str, market_state: str) -> float:
        score = 0.0

        # — EMA Trend (20 pts) —
        ema5  = ind.get("ema5",  ind.get("close", 0))
        ema20 = ind.get("ema20", ind.get("close", 1))
        price = ind.get("close", ema5)
        if direction == "LONG":
            if price > ema20 and ema5 > ema20:  ema_s = 100.0
            elif price > ema20:                  ema_s = 60.0
            else:                                ema_s = 15.0
        else:
            if price < ema20 and ema5 < ema20:  ema_s = 100.0
            elif price < ema20:                  ema_s = 60.0
            else:                                ema_s = 15.0
        score += ema_s * 0.20

        # — MACD (15 pts) —
        macd        = ind.get("macd", 0.0)
        macd_signal = ind.get("macd_signal", 0.0)
        macd_hist   = ind.get("macd_hist", macd - macd_signal)
        if direction == "LONG":
            above_sig = macd > macd_signal
            hist_pos  = macd_hist > 0
            macd_s = 100.0 if (above_sig and hist_pos) else 65.0 if above_sig else 20.0
        else:
            below_sig = macd < macd_signal
            hist_neg  = macd_hist < 0
            macd_s = 100.0 if (below_sig and hist_neg) else 65.0 if below_sig else 20.0
        score += macd_s * 0.15

        # — ADX (15 pts) —
        adx = ind.get("adx", 0.0)
        adx_s = float(np.clip((adx - 10.0) / 30.0 * 100, 0, 100))
        score += adx_s * 0.15

        # — Volume (15 pts) —
        volume  = ind.get("volume", 0)
        vol_avg = ind.get("vol_avg", volume if volume > 0 else 1)
        vol_ratio = volume / max(vol_avg, 1e-9)
        vol_s = float(np.clip(vol_ratio / 2.0 * 100, 0, 100))
        score += vol_s * 0.15

        # — ATR expansion (10 pts) — expanding ATR = momentum
        atr     = ind.get("atr", 0.0)
        atr_avg = ind.get("atr_avg", atr if atr > 0 else 1e-9)
        atr_ratio = atr / max(atr_avg, 1e-9)
        atr_s = float(np.clip((atr_ratio - 0.5) / 1.5 * 100, 0, 100))
        score += atr_s * 0.10

        # — RSI pullback zone (10 pts) —
        rsi = ind.get("rsi", 50.0)
        thrs = ADAPTIVE_THRESHOLDS.get(market_state, ADAPTIVE_THRESHOLDS["TRENDING"])
        if direction == "LONG":
            lo, hi = thrs["rsi_long"]
            rsi_s = 100.0 if lo <= rsi <= hi else 40.0 if rsi < 50 else 10.0
        else:
            lo, hi = thrs["rsi_short"]
            rsi_s = 100.0 if lo <= rsi <= hi else 40.0 if rsi > 50 else 10.0
        score += rsi_s * 0.10

        # — Pattern score (15 pts) — from indicator engine
        pat_s = float(np.clip(ind.get("pattern_score", 50.0), 0, 100))
        score += pat_s * 0.15

        # — Market State quality (10 pts) —
        st_s = self._STATE_QUALITY.get(market_state, 50.0)
        score += st_s * 0.10

        return float(np.clip(score, 0, 100))

    def breakdown(self, ind: Dict, direction: str, market_state: str) -> Dict:
        """Return per-component breakdown for logging."""
        return {
            "health": round(self.compute(ind, direction, market_state), 1),
            "ema":  round(ind.get("ema5", 0), 2),
            "adx":  round(ind.get("adx", 0), 1),
            "rsi":  round(ind.get("rsi", 50), 1),
            "macd": round(ind.get("macd_hist", 0), 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# [V8-4] CONFIDENCE SCORER
# 100-pt conviction: Trend(20) + Momentum(20) + Volume(15) + Volatility(15)
#                    + Structure(15) + Pattern(15)
# ──────────────────────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """Compute a 0-100 directional confidence score."""

    LEVELS = [
        (95, "STRONG_BUY"),
        (85, "BUY"),
        (75, "NORMAL"),
        (65, "WEAK"),
        (0,  "SKIP"),
    ]

    def compute(self, ind_15m: Dict, ind_1h: Dict, direction: str) -> float:
        score = 0.0
        long = direction == "LONG"

        # — Trend alignment 4H (20 pts) —
        ema5_1h  = ind_1h.get("ema5",  0.0)
        ema20_1h = ind_1h.get("ema20", 1.0)
        slope_1h = ind_1h.get("ema20_slope_score", 50.0)
        if long:
            trend_s = 100.0 if ema5_1h > ema20_1h and slope_1h > 52 else \
                      65.0 if ema5_1h > ema20_1h else 20.0
        else:
            trend_s = 100.0 if ema5_1h < ema20_1h and slope_1h < 48 else \
                      65.0 if ema5_1h < ema20_1h else 20.0
        score += trend_s * 0.20

        # — Momentum 15M (20 pts) —
        rsi      = ind_15m.get("rsi", 50.0)
        mom_s_raw = ind_15m.get("momentum_score", 50.0)
        if long:
            rsi_ok   = rsi < 60
            mom_ok   = mom_s_raw > 50
        else:
            rsi_ok   = rsi > 40
            mom_ok   = mom_s_raw < 50
        mom_s = 100.0 if (rsi_ok and mom_ok) else 55.0 if (rsi_ok or mom_ok) else 15.0
        score += mom_s * 0.20

        # — Volume (15 pts) —
        vol_score = ind_15m.get("volume_score", 50.0)
        score += float(np.clip(vol_score, 0, 100)) * 0.15

        # — Volatility (15 pts) — ATR in healthy range (not too low, not extreme)
        atr_score = ind_15m.get("atr_score", 50.0)
        score += float(np.clip(atr_score, 0, 100)) * 0.15

        # — Structure (15 pts) —
        struct_score = ind_15m.get("structure_score", 50.0)
        score += float(np.clip(struct_score, 0, 100)) * 0.15

        # — Pattern (15 pts) —
        pattern_score = ind_15m.get("pattern_score", 50.0)
        score += float(np.clip(pattern_score, 0, 100)) * 0.15

        return float(np.clip(score, 0, 100))

    def get_level(self, score: float) -> str:
        for threshold, label in self.LEVELS:
            if score >= threshold:
                return label
        return "SKIP"

    def get_size_multiplier(self, score: float) -> float:
        """Translate confidence level to position size multiplier."""
        if score >= 85:   return 1.0
        if score >= 75:   return 1.0
        if score >= 65:   return 0.65
        return 0.0  # skip


# ──────────────────────────────────────────────────────────────────────────────
# [V8-5] REGIME BIAS ENGINE — 5-level macro direction from 4H + 1H
# STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR
# ──────────────────────────────────────────────────────────────────────────────

class RegimeBiasEngine:
    """5-level directional bias from 4H and 1H trend structure."""

    def compute(self, ind_4h: Dict, ind_1h: Dict) -> str:
        score = 0

        # 4H signals — weight 3
        ema5_4h  = ind_4h.get("ema5",  0.0)
        ema20_4h = ind_4h.get("ema20", 1.0)
        slope_4h = ind_4h.get("ema20_slope_score", 50.0)
        rsi_4h   = ind_4h.get("rsi", 50.0)
        if ema5_4h > ema20_4h: score += 2
        elif ema5_4h < ema20_4h: score -= 2
        if slope_4h > 58:   score += 1
        elif slope_4h < 42: score -= 1
        if rsi_4h > 58:     score += 1
        elif rsi_4h < 42:   score -= 1

        # 1H signals — weight 2
        ema5_1h  = ind_1h.get("ema5",  0.0)
        ema20_1h = ind_1h.get("ema20", 1.0)
        rsi_1h   = ind_1h.get("rsi", 50.0)
        if ema5_1h > ema20_1h: score += 2
        elif ema5_1h < ema20_1h: score -= 2
        if rsi_1h > 55:    score += 1
        elif rsi_1h < 45:  score -= 1

        if score >= 5:    return "STRONG_BULL"
        if score >= 2:    return "BULL"
        if score >= -1:   return "NEUTRAL"
        if score >= -4:   return "BEAR"
        return "STRONG_BEAR"

    def allows_long(self, bias: str, market_state: str) -> bool:
        """
        Long allowed when:
        - Trend/Breakout/Momentum state  → bias BULL or STRONG_BULL
        - Counter-trend (REVERSAL/EXHAUSTION) → bias BEAR or STRONG_BEAR (reverting up)
        - Mean-revert (SIDEWAY) → always both directions
        """
        if market_state == "SIDEWAY":
            return True
        if market_state in _COUNTER_TREND_STATES:
            return bias in ("BEAR", "STRONG_BEAR", "NEUTRAL")
        return bias in ("BULL", "STRONG_BULL")

    def allows_short(self, bias: str, market_state: str) -> bool:
        if market_state == "SIDEWAY":
            return True
        if market_state in _COUNTER_TREND_STATES:
            return bias in ("BULL", "STRONG_BULL", "NEUTRAL")
        return bias in ("BEAR", "STRONG_BEAR")


# ──────────────────────────────────────────────────────────────────────────────
# [V8-7] PATTERN LEARNING ENGINE — simplified, anti-overfit
# Tracks per entry_type: WR, Avg R, Avg Hold, Expectancy
# WR<45% → weight ×0.85 (floor 0.5)   WR>65% → weight ×1.15 (ceil 1.5)
# ──────────────────────────────────────────────────────────────────────────────

class PatternLearningEngine:
    """Simple per-entry-type win-rate learning.  No AI, no overfitting."""

    ENTRY_TYPES = ["trend_follow", "breakout", "reversal", "mean_revert",
                   "counter_trend", "momentum"]
    MIN_TRADES = 10          # minimum trades before adjusting weight
    MIN_WEIGHT = 0.50
    MAX_WEIGHT = 1.50

    def __init__(self):
        self.stats: Dict[str, Dict] = {
            t: {"wins": 0, "losses": 0, "total_r": 0.0, "total_hold": 0}
            for t in self.ENTRY_TYPES
        }
        self.weights: Dict[str, float] = {t: 1.0 for t in self.ENTRY_TYPES}

    def record(self, entry_type: str, win: bool, r_multiple: float, hold_bars: int):
        if entry_type not in self.stats:
            return
        s = self.stats[entry_type]
        if win:
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["total_r"]    += r_multiple
        s["total_hold"] += hold_bars

    def update_weights(self):
        """Recalculate weights from accumulated stats."""
        for t, s in self.stats.items():
            total = s["wins"] + s["losses"]
            if total < self.MIN_TRADES:
                continue
            wr = s["wins"] / total
            if wr < 0.45:
                self.weights[t] = max(self.weights[t] * 0.85, self.MIN_WEIGHT)
            elif wr > 0.65:
                self.weights[t] = min(self.weights[t] * 1.15, self.MAX_WEIGHT)

    def get_weight(self, entry_type: str) -> float:
        return self.weights.get(entry_type, 1.0)

    def get_summary(self) -> Dict:
        out = {}
        for t, s in self.stats.items():
            total = s["wins"] + s["losses"]
            if total == 0:
                continue
            out[t] = {
                "trades":     total,
                "win_rate":   round(s["wins"] / total, 3),
                "avg_r":      round(s["total_r"] / total, 3),
                "avg_hold":   round(s["total_hold"] / total, 1),
                "expectancy": round((s["wins"] / total * s["total_r"] / max(s["wins"], 1))
                                    - (s["losses"] / total * abs(s["total_r"] - s["wins"] / total
                                                                  * s["total_r"] / max(s["wins"], 1))), 3),
                "weight":     round(self.weights.get(t, 1.0), 3),
            }
        return out

    def to_dict(self) -> Dict:
        return {"stats": self.stats, "weights": self.weights}

    def from_dict(self, data: Dict):
        self.stats   = data.get("stats",   self.stats)
        self.weights = data.get("weights", self.weights)


# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE ENGINE (V7 base weights — used as fallback scoring only)
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveEngine:
    """Legacy per-state score weights (used as context multiplier in FILTERING)."""

    def __init__(self):
        self.atr_percentile_points = [10, 25, 50, 75, 90]
        self.atr_vol_labels = ["Very Low", "Low", "Normal", "High", "Extreme"]

        # Base weights per state — sum=100; used to weight indicator sub-scores
        self.base_weights: Dict[str, Dict[str, float]] = {
            "STRONG_TREND": {
                "trend": 25, "momentum": 20, "volume": 10, "pattern": 8,
                "structure": 12, "atr": 5, "rsi": 5, "sweep": 0,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 0,
            },
            "TRENDING": {
                "trend": 20, "momentum": 15, "volume": 10, "pattern": 10,
                "structure": 10, "atr": 5, "rsi": 5, "sweep": 5,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 5,
            },
            "BREAKOUT": {
                "trend": 15, "momentum": 20, "volume": 20, "pattern": 10,
                "structure": 5, "atr": 15, "rsi": 5, "sweep": 5,
                "divergence": 0, "ema": 5, "vwap": 0, "macd": 0,
            },
            "REVERSAL": {
                "trend": 5, "momentum": 10, "volume": 15, "pattern": 10,
                "structure": 10, "atr": 5, "rsi": 10, "sweep": 5,
                "divergence": 20, "ema": 5, "vwap": 5, "macd": 0,
            },
            "SIDEWAY": {
                "trend": 5, "momentum": 10, "volume": 10, "pattern": 15,
                "structure": 10, "atr": 5, "rsi": 20, "sweep": 10,
                "divergence": 10, "ema": 0, "vwap": 5, "macd": 0,
            },
            "HIGH_VOL": {
                "trend": 10, "momentum": 15, "volume": 20, "pattern": 10,
                "structure": 5, "atr": 15, "rsi": 5, "sweep": 5,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 0,
            },
            "EXHAUSTION": {
                "trend": 5, "momentum": 10, "volume": 15, "pattern": 10,
                "structure": 10, "atr": 5, "rsi": 10, "sweep": 5,
                "divergence": 20, "ema": 5, "vwap": 0, "macd": 5,
            },
            "LOW_VOL": {
                "trend": 10, "momentum": 5, "volume": 5, "pattern": 20,
                "structure": 10, "atr": 20, "rsi": 5, "sweep": 10,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 0,
            },
        }

    def get_atr_volatility_state(self, current_atr: float, atr_history: List[float]) -> str:
        if len(atr_history) < 20:
            return "Normal"
        history = atr_history[-100:] if len(atr_history) >= 100 else atr_history
        percentiles = np.percentile(history, self.atr_percentile_points)
        for i in range(len(percentiles) - 1, -1, -1):
            if current_atr >= percentiles[i]:
                return self.atr_vol_labels[min(i, len(self.atr_vol_labels) - 1)]
        return "Very Low"

    def get_dynamic_weights(self, market_state: str) -> Dict[str, float]:
        return self.base_weights.get(market_state, self.base_weights["TRENDING"])


# ──────────────────────────────────────────────────────────────────────────────
# POSITION HEALTH CALCULATOR (unchanged from V7)
# ──────────────────────────────────────────────────────────────────────────────

class PositionHealthCalculator:
    """คำนวณ position health score 0–100 จาก 5 ตัวชี้วัด"""

    def calculate(self, ind: Dict[str, Any], trade: Dict[str, Any],
                  current_price: float) -> float:
        scores = []

        rsi = ind.get("rsi", 50)
        direction = trade.get("direction", "LONG")
        if direction == "LONG":
            rsi_score = np.clip((rsi - 30) / 40 * 100, 0, 100)
        else:
            rsi_score = np.clip((70 - rsi) / 40 * 100, 0, 100)

        macd = ind.get("macd", 0)
        macd_signal = ind.get("macd_signal", 0)
        if direction == "LONG":
            macd_score = 80.0 if macd > macd_signal else 30.0
        else:
            macd_score = 80.0 if macd < macd_signal else 30.0
        momentum_score = rsi_score * 0.6 + macd_score * 0.4
        scores.append(("momentum", momentum_score, 0.30))

        volume = ind.get("volume", 0)
        vol_avg = ind.get("vol_avg", volume if volume > 0 else 1)
        vol_ratio = volume / max(vol_avg, 1e-9)
        vol_score = float(np.clip(vol_ratio * 60, 0, 100))
        scores.append(("volume", vol_score, 0.20))

        ema5 = ind.get("ema5", current_price)
        ema20 = ind.get("ema20", current_price)
        if direction == "LONG":
            struct_score = 100.0 if (current_price > ema5 and current_price > ema20) else \
                           60.0 if current_price > ema20 else 20.0
        else:
            struct_score = 100.0 if (current_price < ema5 and current_price < ema20) else \
                           60.0 if current_price < ema20 else 20.0
        scores.append(("structure", struct_score, 0.20))

        entry_atr = trade.get("atr_at_entry", ind.get("atr", 1))
        current_atr = ind.get("atr", entry_atr)
        atr_ratio = current_atr / max(entry_atr, 1e-9)
        atr_score = float(np.clip((2.0 - atr_ratio) / 1.0 * 100, 0, 100))
        scores.append(("atr", atr_score, 0.15))

        adx = ind.get("adx", 20)
        adx_score = float(np.clip((adx - 15) / 25 * 100, 0, 100))
        scores.append(("trend", adx_score, 0.15))

        total = sum(score * weight for _, score, weight in scores)
        return float(np.clip(total, 0, 100))


# ──────────────────────────────────────────────────────────────────────────────
# TRADING BOT — V8
# ──────────────────────────────────────────────────────────────────────────────

class TradingBot:
    """
    V8 state-machine trading bot.
    SwingReversalPro with all 7 V8 improvements.
    """

    STATES = {
        "SCANNING", "FILTERING", "WAIT_CONFIRM", "PENDING_ORDER",
        "IN_POSITION", "PARTIAL_EXIT", "TRAILING", "EXITING",
        "COOLDOWN", "BLOCKED", "RECOVERY", "ERROR"
    }

    def __init__(self,
                 account_balance: float = 10_000.0,
                 base_risk_pct: float = 0.01,
                 daily_loss_limit_pct: float = -3.0,
                 daily_profit_limit_pct: float = 8.0,
                 cooldown_minutes: int = 20,
                 max_loss_streak: int = 4,
                 tp1_close_pct: float = 0.50,
                 state_file: Optional[str] = None,
                 execution_callback: Optional[Callable] = None,
                 startup_warmup_minutes: int = 45,
                 enable_swing_reversal: bool = True,
                 enable_mean_reversion: bool = False):
        self.state: str = "SCANNING"
        self.adaptive_engine   = AdaptiveEngine()
        self.health_calc       = PositionHealthCalculator()
        self.entry_scorer      = EntryHealthScorer()
        self.conf_scorer       = ConfidenceScorer()
        self.bias_engine       = RegimeBiasEngine()
        self.learning_engine   = PatternLearningEngine()
        self.tp1_close_pct     = tp1_close_pct
        self._state_file: str  = state_file or self.DEFAULT_STATE_FILE

        self.account_balance       = account_balance
        self.base_risk_pct         = base_risk_pct
        self.daily_loss_limit_pct  = daily_loss_limit_pct
        self.daily_profit_limit_pct = daily_profit_limit_pct
        self.cooldown_minutes      = cooldown_minutes
        self.max_loss_streak       = max_loss_streak

        self.position_open: bool     = False
        self.order_status: str       = "CLOSED"
        self.current_trade: Dict     = {}

        self.atr_history: List[float]               = []
        self.current_market_state: str              = "SIDEWAY"
        self.current_regime_bias: str               = "NEUTRAL"
        self.regime_score: float                    = 0.0
        self.direction_focus: Optional[str]         = None
        self.bars_since_trigger: int                = 0

        # [V8-6] Cooldown trackers
        self.loss_streak: int              = 0
        self.win_streak: int               = 0
        self.consecutive_sl_hits: int      = 0    # reset on win or non-SL close
        self.session_losses: int           = 0    # total losses in current trading day
        self.daily_pnl_pct: float          = 0.0
        self.cooldown_until: Optional[datetime.datetime] = None
        self.trading_date: Optional[datetime.date]       = None
        self._bar_now: Optional[datetime.datetime]       = None

        self.trade_journal: List[Dict] = []
        self.execution_callback        = execution_callback
        self._tick_depth: int          = 0
        self._max_tick_depth: int      = 6
        self._bar_count: int           = 0
        self._position_entry_bar: int  = 0
        self.startup_warmup_minutes    = startup_warmup_minutes
        self._startup_unblock_at: Optional[datetime.datetime] = None
        self._last_candle_15m: Dict    = {}
        self._log: List[str]           = []

        # Strategy instances
        self.enable_swing_reversal  = enable_swing_reversal
        self.enable_mean_reversion  = enable_mean_reversion
        self._mr_strategy           = MeanReversionStrategy() if enable_mean_reversion else None
        self._pending_signal: Optional[Dict] = None   # signal from MR strategy pending order

        # Last computed scores (for health report / logging)
        self._last_entry_health: float    = 0.0
        self._last_confidence: float      = 0.0
        self._last_confidence_level: str  = "SKIP"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log_event(self, msg: str, level: str = "info"):
        log_fn = getattr(logger, level, logger.info)
        log_fn(msg)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.append(f"[{ts}] {msg}")

    def _send_order(self, order_type: str, trade_info: Dict):
        if self.execution_callback:
            try:
                result = self.execution_callback(order_type, trade_info)
                self._log_event(f"[EXEC OK] {order_type} confirmed → {result}")
                return result
            except Exception as e:
                self._log_event(
                    f"[ERROR] execution_callback failed for {order_type}: {e}",
                    level="error",
                )
                self.state = "ERROR"
                return None
        else:
            self._log_event(f"[PAPER] {order_type}: {trade_info}")
            return None

    @staticmethod
    def _scale_score(val: float, max_weight: float) -> float:
        return min((min(float(val), 100.0) / 100.0) * max_weight, max_weight)

    @staticmethod
    def _health_level(score: float) -> str:
        if score >= 80: return "STRONG"
        if score >= 60: return "GOOD"
        if score >= 40: return "WARN"
        if score >= 20: return "POOR"
        return "CRITICAL"

    def _detect_reversal_signals(self, ind_15m: Dict) -> Dict:
        signals: Dict = {}
        if not self.position_open or not self.current_trade:
            return signals

        direction = self.current_trade.get("direction", "LONG")
        c = self._last_candle_15m

        if c:
            high       = c.get("high",  0.0)
            low        = c.get("low",   0.0)
            close      = c.get("close", 0.0)
            open_price = c.get("open",  close)
            total_range = max(high - low, 1e-9)
            body_top   = max(close, open_price)
            body_bot   = min(close, open_price)
            upper_wick = high - body_top
            lower_wick = body_bot - low

            if direction == "LONG":
                if (upper_wick / total_range > 0.60
                        and close < open_price
                        and upper_wick > lower_wick * 1.5):
                    signals["reversal_spike"] = {
                        "severity": "HIGH",
                        "wick_ratio": round(upper_wick / total_range, 2),
                    }
            else:
                if (lower_wick / total_range > 0.60
                        and close > open_price
                        and lower_wick > upper_wick * 1.5):
                    signals["reversal_spike"] = {
                        "severity": "HIGH",
                        "wick_ratio": round(lower_wick / total_range, 2),
                    }

        holding_bars = self._bar_count - self._position_entry_bar
        if holding_bars >= 4:
            adx       = ind_15m.get("adx", 20)
            ema5      = ind_15m.get("ema5",  0.0)
            ema20     = ind_15m.get("ema20", 1.0)
            macd_hist = ind_15m.get("macd_hist", 0.0)
            ema_gap_pct = abs(ema5 - ema20) / max(ema20, 1e-9) * 100

            fade_count = sum([
                adx < 14,
                ema_gap_pct < 0.25,
                (direction == "LONG"  and macd_hist < 0),
                (direction == "SHORT" and macd_hist > 0),
            ])
            if fade_count >= 2:
                signals["trend_fade"] = {
                    "severity":    "MEDIUM",
                    "adx":         round(adx, 1),
                    "ema_gap_pct": round(ema_gap_pct, 3),
                }

        return signals

    # ── [V8-1] MODULE 1: Market State Engine (8 states) ─────────────────────

    def _step1_market_state_engine(self, ind: Dict) -> str:
        """
        8-state regime from 4H indicators.
        Priority: HIGH_VOL → LOW_VOL → BREAKOUT → STRONG_TREND
                → TRENDING → REVERSAL → EXHAUSTION → SIDEWAY
        """
        adx      = ind.get("adx", 0)
        eff      = ind.get("eff_ratio", 0)
        atr_exp  = ind.get("atr_exp", 1.0)
        bb_w     = ind.get("bb_width", 0.5)
        pdi      = ind.get("pdi", 20)
        mdi      = ind.get("mdi", 20)
        rsi      = ind.get("rsi", 50)
        volume   = ind.get("volume", 0)
        vol_avg  = ind.get("vol_avg", max(volume, 1))
        vol_ratio = volume / max(vol_avg, 1e-9)

        # 1. HIGH_VOL — dangerous volatility expansion
        if atr_exp > 1.8 and bb_w > 0.7:
            return "HIGH_VOL"

        # 2. LOW_VOL — squeeze, skip
        if bb_w < 0.2 and atr_exp < 0.8:
            return "LOW_VOL"

        # 3. BREAKOUT — expanding from compression with volume confirmation
        if bb_w > 0.3 and atr_exp > 1.2 and vol_ratio > 1.4 and adx < 24:
            return "BREAKOUT"

        # 4. STRONG_TREND — efficient trend, large DI gap
        if adx > 25 and eff > 0.55 and abs(pdi - mdi) > 8:
            return "STRONG_TREND"

        # 5. TRENDING — moderate trend with directional dominance
        if adx > 18 and eff > 0.35 and abs(pdi - mdi) > 3:
            return "TRENDING"

        # 6. REVERSAL — extreme RSI with fading efficiency
        if (rsi > 68 or rsi < 32) and adx > 15 and eff < 0.42:
            return "REVERSAL"

        # 7. EXHAUSTION — elevated ADX but directional efficiency collapsed
        if adx > 20 and eff < 0.28:
            return "EXHAUSTION"

        # 8. SIDEWAY — default (ranging/no-trend)
        return "SIDEWAY"

    # ── Step 2: 4H Regime score ───────────────────────────────────────────────

    def _step2_4h_regime(self, ind: Dict) -> float:
        score  = self._scale_score(ind.get("ema20_slope_score", 50), 30)
        score += self._scale_score(ind.get("adx_score",         50), 20)
        score += self._scale_score(ind.get("eff_score",         50), 20)
        score += self._scale_score(ind.get("atr_score",         50), 15)
        score += self._scale_score(ind.get("vol_score",         50), 15)
        return float(np.clip(score, 0, 100))

    # ── Step 3: Global gates ──────────────────────────────────────────────────

    def _check_global_gates(self) -> bool:
        if self.position_open:
            return False

        _now = self._bar_now or datetime.datetime.now()

        if self._startup_unblock_at and _now < self._startup_unblock_at:
            remaining = int((self._startup_unblock_at - _now).total_seconds() / 60)
            self._log_event(f"WARMUP: {remaining}m remaining", level="debug")
            return False

        if self.cooldown_until and _now < self.cooldown_until:
            self.state = "COOLDOWN"
            return False

        if self.daily_pnl_pct <= self.daily_loss_limit_pct:
            self.state = "BLOCKED"
            self._log_event(
                f"BLOCKED: daily PnL {self.daily_pnl_pct:.2f}% hit loss limit",
                level="warning",
            )
            return False

        if self.daily_pnl_pct >= self.daily_profit_limit_pct:
            self.state = "BLOCKED"
            self._log_event(
                f"BLOCKED: daily PnL {self.daily_pnl_pct:.2f}% hit profit limit",
                level="warning",
            )
            return False

        # [V8-1] LOW_VOL: skip immediately
        if self.current_market_state == "LOW_VOL":
            self._log_event("SKIP: LOW_VOL — awaiting breakout", level="debug")
            return False

        if self.current_market_state not in _TRADEABLE_STATES:
            self._log_event(
                f"SKIP: unrecognised state={self.current_market_state}", level="debug"
            )
            return False

        return True

    # ── Step 4: FILTERING — [V8-3/V8-4/V8-5] Entry Health + Confidence + Bias ─

    def _layer_filtering(self, direction: str, ind_15m: Dict,
                         market_state: str, ind_1h: Dict, ind_4h: Dict) -> bool:
        """
        Pass all three gates:
        1. Regime Bias allows this direction
        2. Entry Health Score ≥ health_min for this state
        3. Confidence Score ≥ confidence_min for this state
        """
        bias = self.current_regime_bias

        # [V8-5] Bias gate
        if direction == "LONG" and not self.bias_engine.allows_long(bias, market_state):
            return False
        if direction == "SHORT" and not self.bias_engine.allows_short(bias, market_state):
            return False

        # [V8-3] Entry Health
        health = self.entry_scorer.compute(ind_15m, direction, market_state)
        thrs   = ADAPTIVE_THRESHOLDS.get(market_state, ADAPTIVE_THRESHOLDS["TRENDING"])
        health_min = thrs["health_min"]
        if health < health_min:
            return False

        # [V8-4] Confidence
        confidence     = self.conf_scorer.compute(ind_15m, ind_1h, direction)
        conf_min       = thrs["confidence_min"]
        conf_level     = self.conf_scorer.get_level(confidence)
        if confidence < conf_min:
            return False

        # Store for use in sizing and logging
        self._last_entry_health    = health
        self._last_confidence      = confidence
        self._last_confidence_level = conf_level
        return True

    # ── Step 3.5: 1H trend alignment check ───────────────────────────────────

    def _check_htf_trend_filter(self, direction: str, ind_1h: Dict) -> bool:
        """Quick 1H EMA alignment — called from FILTERING as secondary gate."""
        adx_1h   = ind_1h.get("adx", 0)
        ema5_1h  = ind_1h.get("ema5")
        ema20_1h = ind_1h.get("ema20")

        thrs = ADAPTIVE_THRESHOLDS.get(
            self.current_market_state, ADAPTIVE_THRESHOLDS["TRENDING"]
        )
        adx_min = thrs["adx_min"]

        if adx_1h < adx_min or ema5_1h is None or ema20_1h is None:
            return False

        tolerance = ema20_1h * 0.003
        if direction == "LONG":
            return ema5_1h >= ema20_1h - tolerance
        else:
            return ema5_1h <= ema20_1h + tolerance

    # ── Step 5: Risk engine + [V8-3/V8-4] adaptive sizing ───────────────────

    def _step5_risk_engine(self, candle: Dict, direction: str, ind: Dict,
                           mr_signal: Optional[Dict] = None):
        """
        Compute SL/TP/size and open position.
        mr_signal: if provided (from MeanReversionStrategy), use its SL price and
                   tp_config directly instead of the swing reversal defaults.
        """
        entry_price = float(candle.get("close", 0))

        # ── SL calculation ──────────────────────────────────────────────────
        if mr_signal is not None:
            # MeanReversion: use sweep low/high SL from strategy
            pattern_sl = mr_signal["sl_price"]
        elif direction == "LONG":
            pattern_sl = candle.get("pattern_low", entry_price * 0.99)
        else:
            pattern_sl = candle.get("pattern_high", entry_price * 1.01)

        sl_dist = abs(entry_price - float(pattern_sl))
        if sl_dist < 1e-8:
            sl_dist = entry_price * 0.01

        min_sl_dist = entry_price * 0.020
        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
            pattern_sl = (entry_price - sl_dist if direction == "LONG"
                          else entry_price + sl_dist)

        # ── Size multiplier ──────────────────────────────────────────────────
        risk_pct = self.base_risk_pct
        if self.win_streak >= 5:
            risk_pct *= 0.80
            self._log_event(f"Win streak {self.win_streak} → risk {risk_pct:.2%}")

        if mr_signal is not None:
            # MeanReversion: size from health score (Step 13)
            size_mult  = mr_signal.get("size_mult", 1.0)
            health     = mr_signal.get("health_score", 0.0)
            confidence = mr_signal.get("health_score", 0.0)
            entry_type = "mean_revert"
        else:
            health     = self._last_entry_health
            confidence = self._last_confidence
            conf_mult  = self.conf_scorer.get_size_multiplier(confidence)
            if conf_mult == 0.0:
                self._log_event("Confidence too low → skip order", level="warning")
                return
            health_mult = 1.0 if health >= 75 else 0.65
            entry_type  = _STATE_ENTRY_TYPE.get(self.current_market_state, "trend_follow")
            learning_mult = self.learning_engine.get_weight(entry_type)
            size_mult   = conf_mult * health_mult * learning_mult

        risk_amount   = self.account_balance * risk_pct * size_mult
        position_size = risk_amount / max(sl_dist, 1e-9)

        # ── TP levels ────────────────────────────────────────────────────────
        mult = 1 if direction == "LONG" else -1

        if mr_signal is not None:
            # MeanReversion TP: 1R/2R/3R with configurable close fractions
            tc    = mr_signal["tp_config"]
            tp1   = float(tc["tp1"])
            tp2   = float(tc["tp2"])
            tp3   = float(tc["tp3"])
            tp1_pct = float(tc.get("tp1_pct", 0.30))
            tp2_pct = float(tc.get("tp2_pct", 0.40))
            trail_atr = float(tc.get("trail_atr_mult", 2.0))
        else:
            # SwingReversalPro TP: 0.7R / 2.0R (no TP3)
            tp1 = entry_price + sl_dist * 0.7 * mult
            tp2 = entry_price + sl_dist * 2.0 * mult
            tp3 = None
            tp1_pct = self.tp1_close_pct   # default 0.50
            tp2_pct = 1.0
            trail_atr = 2.0

        strategy_tag = "MeanReversion" if mr_signal else "SwingReversal"
        self._log_event(
            f"[{strategy_tag}] OPEN {direction} entry={entry_price:.4f} "
            f"sl={pattern_sl:.4f} tp1={tp1:.4f} tp2={tp2:.4f} "
            f"size×{size_mult:.2f} health={health:.0f}"
        )

        self.current_trade = {
            "direction":            direction,
            "entry":                entry_price,
            "sl":                   float(pattern_sl),
            "sl_dist":              sl_dist,
            "tp1":                  tp1,
            "tp2":                  tp2,
            "tp3":                  tp3,           # None for SwingReversal
            "tp1_pct":              tp1_pct,
            "tp2_pct":              tp2_pct,
            "trail_atr_mult":       trail_atr,
            "tp1_hit":              False,
            "tp2_hit":              False,
            "tp3_hit":              False,
            "break_even_triggered": False,
            "status":               "OPEN",
            "entry_time":           datetime.datetime.now(),
            "atr_at_entry":         ind.get("atr", sl_dist),
            "realized_pnl":         0.0,
            "remaining_size":       position_size,
            "exit_price":           None,
            "final_rr":             None,
            "mae":                  0.0,
            "mfe":                  0.0,
            "entry_health":         health,
            "entry_confidence":     confidence,
            "entry_type":           entry_type,
            "strategy":             strategy_tag,
        }

        self._send_order("OPEN_LONG" if direction == "LONG" else "OPEN_SHORT", {
            "entry": entry_price,
            "sl":    float(pattern_sl),
            "tp1":   tp1, "tp2": tp2,
            "tp3":   tp3,
            "size":  position_size,
        })

        self.position_open         = True
        self._position_entry_bar   = self._bar_count
        self.order_status          = "OPEN"

    # ── Step 6: Position management (V7 logic unchanged) ─────────────────────

    def _manage_open_position(self, current_price: float, ind: Dict) -> str:
        t         = self.current_trade
        direction = t["direction"]
        sl_dist   = t["sl_dist"]
        dir_mult  = 1 if direction == "LONG" else -1
        current_r = ((current_price - t["entry"]) * dir_mult) / max(sl_dist, 1e-9)

        t["mae"] = min(t["mae"], current_r)
        t["mfe"] = max(t["mfe"], current_r)

        reversal = self._detect_reversal_signals(ind)
        if reversal.get("reversal_spike"):
            sig = reversal["reversal_spike"]
            self._log_event(
                f"[PROTECT] REVERSAL_SPIKE wick={sig['wick_ratio']:.0%} → exit",
                level="warning",
            )
            self._close_position("REVERSAL_SPIKE", current_price, 1.0, ind)
            return "EXITING"

        if reversal.get("trend_fade"):
            sig  = reversal["trend_fade"]
            atr  = ind.get("atr", t["sl_dist"] * 0.5)
            mult = 1 if direction == "LONG" else -1
            tight_sl = current_price - atr * 0.8 * mult
            if direction == "LONG":
                t["sl"] = max(t["sl"], tight_sl)
            else:
                t["sl"] = min(t["sl"], tight_sl)
            self._log_event(
                f"[PROTECT] TREND_FADE ADX={sig['adx']} → SL → {t['sl']:.2f}",
                level="warning",
            )

        emergency_signals = [
            ind.get("opposite_choch"),
            ind.get("atr_collapse"),
            ind.get("momentum_collapse"),
            ind.get("volume_collapse"),
            ind.get("invalid_structure"),
        ]
        if any(emergency_signals):
            self._close_position("EMERGENCY_EXIT", current_price, 1.0, ind)
            return "EXITING"

        if direction == "LONG":
            tp1_hit_cond = current_price >= t["tp1"]
            tp2_hit_cond = current_price >= t["tp2"]
        else:
            tp1_hit_cond = current_price <= t["tp1"]
            tp2_hit_cond = current_price <= t["tp2"]

        # TP2 first (gap candle)
        if tp2_hit_cond and not t["tp2_hit"]:
            tp2_close_pct = t.get("tp2_pct", 1.0)
            if tp2_close_pct < 1.0 and t.get("tp3"):
                # MeanReversion: partial close — keep runner for TP3
                self._close_position("PARTIAL_TP2", t["tp2"], tp2_close_pct, ind)
                t["tp1_hit"] = True
                t["tp2_hit"] = True
                if not t["break_even_triggered"]:
                    t["sl"] = t["entry"]
                    t["break_even_triggered"] = True
                    self._log_event(
                        f"TP2 @ {t['tp2']:.4f} → {tp2_close_pct:.0%} partial + SL → BE"
                    )
                self.state = "TRAILING"
                # continue to TP3 runner check below
            else:
                self._close_position("FULL_TP2", t["tp2"], 1.0, ind)
                t["tp1_hit"] = True
                t["tp2_hit"] = True
                return "EXITING"

        # TP1 — close tp1_pct + break-even
        if tp1_hit_cond and not t["tp1_hit"]:
            tp1_close_pct = t.get("tp1_pct", self.tp1_close_pct)
            self._close_position("PARTIAL_TP1", t["tp1"], tp1_close_pct, ind)
            t["tp1_hit"] = True
            if not t["break_even_triggered"]:
                if direction == "LONG":
                    t["sl"] = max(t["sl"], t["entry"])
                else:
                    t["sl"] = min(t["sl"], t["entry"])
                t["break_even_triggered"] = True
                self._log_event(
                    f"TP1 @ {t['tp1']:.4f} → {tp1_close_pct:.0%} close "
                    f"+ SL → breakeven ({t['entry']:.4f})"
                )
            self.state = "PARTIAL_EXIT"

        # TP3 runner — MeanReversion only (active once TP2 partial close done)
        if t.get("tp3") and t.get("tp2_hit") and not t.get("tp3_hit"):
            atr        = ind.get("atr", t["sl_dist"])
            trail_mult = t.get("trail_atr_mult", 2.0)
            if direction == "LONG":
                t["sl"] = max(t["sl"], current_price - atr * trail_mult)
                if current_price >= t["tp3"]:
                    self._close_position("TP3_RUNNER", t["tp3"], 1.0, ind)
                    t["tp3_hit"] = True
                    return "EXITING"
                if current_price <= t["sl"]:
                    self._close_position("RUNNER_SL", t["sl"], 1.0, ind)
                    return "EXITING"
            else:
                t["sl"] = min(t["sl"], current_price + atr * trail_mult)
                if current_price <= t["tp3"]:
                    self._close_position("TP3_RUNNER", t["tp3"], 1.0, ind)
                    t["tp3_hit"] = True
                    return "EXITING"
                if current_price >= t["sl"]:
                    self._close_position("RUNNER_SL", t["sl"], 1.0, ind)
                    return "EXITING"
            return "TRAILING"

        # Health-based position management
        tp1_hit = t.get("tp1_hit", False)
        health  = self.health_calc.calculate(ind, t, current_price)

        if health >= 80:
            self._log_event(
                f"Health {health:.0f} (≥80) → HOLD (r={current_r:.2f})", level="debug"
            )
        elif health >= 60 or t["break_even_triggered"]:
            atr      = ind.get("atr", sl_dist)
            atr_trail = atr * 2
            if direction == "LONG":
                t["sl"] = max(t["sl"], current_price - atr_trail)
            else:
                t["sl"] = min(t["sl"], current_price + atr_trail)
        elif health >= 40:
            if not t["break_even_triggered"] and tp1_hit:
                t["sl"] = t["entry"]
                t["break_even_triggered"] = True
                self._log_event(f"Health {health:.0f} → forced breakeven (post-TP1)")
        elif health >= 20:
            if t["remaining_size"] > 0 and tp1_hit:
                self._close_position("HEALTH_REDUCE", current_price, 0.50, ind)
                self.state = "PARTIAL_EXIT"
        else:
            self._close_position("POOR_HEALTH_EXIT", current_price, 1.0, ind)
            return "EXITING"

        if direction == "LONG" and current_price <= t["sl"]:
            self._close_position("SL_HIT", t["sl"], 1.0, ind)
            return "EXITING"
        if direction == "SHORT" and current_price >= t["sl"]:
            self._close_position("SL_HIT", t["sl"], 1.0, ind)
            return "EXITING"

        return self.state

    def _close_position(self, reason: str, price: float, portion: float, ind: Dict):
        t = self.current_trade
        if not t or t.get("status") != "OPEN":
            return

        direction   = t["direction"]
        entry_price = t["entry"]
        remaining   = t.get("remaining_size", t.get("size", 1.0))

        close_size = remaining * min(max(float(portion), 0.0), 1.0)
        if close_size <= 0:
            return

        if direction == "LONG":
            pnl_per_unit = float(price) - entry_price
        else:
            pnl_per_unit = entry_price - float(price)

        pnl = pnl_per_unit * close_size

        t["realized_pnl"]   = t.get("realized_pnl", 0.0) + pnl
        t["remaining_size"] = remaining - close_size
        t["exit_price"]     = float(price)

        if t["sl_dist"] > 0:
            t["final_rr"] = pnl_per_unit / t["sl_dist"]

        self._send_order("CLOSE_PARTIAL" if portion < 1.0 else "CLOSE_FULL", {
            "reason":    reason,
            "price":     float(price),
            "size":      close_size,
            "pnl":       pnl,
            "direction": direction,
        })

        self._log_event(
            f"CLOSE {reason} | {direction} | price={price:.2f} "
            f"| size={close_size:.4f} | pnl={pnl:+.2f}"
        )

        if portion >= 1.0 or t["remaining_size"] <= 1e-9:
            t["status"]        = "CLOSED"
            self.position_open = False
            self.order_status  = "CLOSED"

    # ── Step 7: Journal + [V8-6/V8-7] learning ───────────────────────────────

    def _log_trade(self, result: str, close_reason: str, ind: Dict, extras: Dict):
        t   = self.current_trade
        pnl = t.get("realized_pnl", 0.0)

        _entry      = t.get("entry", 0.0)
        _sl         = t.get("sl", _entry)
        _exit       = t.get("exit_price", _entry)
        _sl_d       = max(abs(_entry - _sl), 1e-8)
        _d_mult     = 1 if t.get("direction") == "LONG" else -1
        _realized_r = _d_mult * (_exit - _entry) / _sl_d
        _win_r      = max(_realized_r, 0.0)
        _loss_r     = max(-_realized_r, 0.0)

        _holding_bars = self._bar_count - self._position_entry_bar
        _vol_state    = self.adaptive_engine.get_atr_volatility_state(
            ind.get("atr", 0), self.atr_history)

        entry_type = t.get("entry_type", _STATE_ENTRY_TYPE.get(self.current_market_state, "trend_follow"))

        entry = {
            "symbol":              extras.get("symbol", "BTCUSDT"),
            "timeframe":           "15M",
            "direction":           t.get("direction"),
            "entry":               t.get("entry"),
            "exit":                t.get("exit_price"),
            "sl":                  t.get("sl"),
            "tp1":                 t.get("tp1"),
            "tp2":                 t.get("tp2"),
            "rr":                  t.get("final_rr"),
            "win_loss":            result,
            "pnl":                 pnl,
            "win_r":               round(_win_r, 3),
            "loss_r":              round(_loss_r, 3),
            "mae":                 t.get("mae"),
            "mfe":                 t.get("mfe"),
            "holding_bars":        _holding_bars,
            "hold_time_sec":       (datetime.datetime.now() -
                                    t.get("entry_time", datetime.datetime.now())).total_seconds(),
            "atr":                 ind.get("atr"),
            "adx":                 ind.get("adx"),
            "rsi":                 ind.get("rsi"),
            "ema":                 ind.get("ema5"),
            "macd":                ind.get("macd"),
            "market_state":        self.current_market_state,
            "regime_bias":         self.current_regime_bias,
            "entry_type":          entry_type,
            "entry_health":        t.get("entry_health"),
            "entry_confidence":    t.get("entry_confidence"),
            "atr_percentile":      _vol_state,
            "hour_utc":            (self._bar_now or datetime.datetime.now()).hour,
            "volatility_state":    _vol_state,
            "regime_score":        self.regime_score,
            "session":             extras.get("session"),
            "funding":             extras.get("funding_rate"),
            "open_interest":       extras.get("oi"),
            "exit_reason":         close_reason,
        }
        self.trade_journal.append(entry)

        # [V8-7] Learning engine record
        self.learning_engine.record(
            entry_type,
            win=(result == "WIN"),
            r_multiple=_realized_r,
            hold_bars=_holding_bars,
        )
        # Update weights every 20 trades
        if len(self.trade_journal) % 20 == 0:
            self.learning_engine.update_weights()
            self._log_event(
                f"[LEARN] weights updated: {self.learning_engine.get_summary()}"
            )

        # Update streaks + daily PnL
        self.daily_pnl_pct     += (pnl / max(self.account_balance, 1)) * 100
        self.account_balance   += pnl

        win = (result == "WIN")
        if win:
            self.win_streak             += 1
            self.loss_streak             = 0
            self.consecutive_sl_hits     = 0
        else:
            self.loss_streak            += 1
            self.session_losses         += 1
            self.win_streak              = 0
            if close_reason == "SL_HIT":
                self.consecutive_sl_hits += 1
            else:
                self.consecutive_sl_hits  = 0

        # [V8-6] Tiered cooldown
        _now = self._bar_now or datetime.datetime.now()
        cooldown_mins = None

        if self.consecutive_sl_hits >= 2:
            cooldown_mins = 90
            self._log_event(
                f"[V8-6] {self.consecutive_sl_hits} consecutive SL hits → 90-min cooldown",
                level="warning",
            )
            self.consecutive_sl_hits = 0
        elif self.session_losses >= 3:
            cooldown_mins = 240
            self._log_event(
                f"[V8-6] {self.session_losses} session losses → 4-hour cooldown",
                level="warning",
            )
        elif self.loss_streak >= self.max_loss_streak:
            cooldown_mins = self.cooldown_minutes
            self._log_event(
                f"Loss streak {self.loss_streak} → {cooldown_mins}m cooldown",
                level="warning",
            )

        if cooldown_mins is not None:
            self.cooldown_until = _now + datetime.timedelta(minutes=cooldown_mins)
            self.state = "COOLDOWN"

        self._log_event(
            f"TRADE CLOSED | {result} | PnL={pnl:+.2f} "
            f"| Balance={self.account_balance:.2f} "
            f"| WS={self.win_streak} LS={self.loss_streak} SL_hits={self.consecutive_sl_hits}"
        )

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _check_daily_reset(self, bar_dt: Optional[datetime.datetime] = None):
        today = (bar_dt.date() if bar_dt else datetime.date.today())
        if self.trading_date != today:
            self.trading_date    = today
            self.daily_pnl_pct   = 0.0
            self.session_losses  = 0    # reset daily loss counter
            if self.state == "BLOCKED":
                self.state = "SCANNING"
                self._log_event("New trading day — BLOCKED state reset")

    # ── [V8-2] Entry trigger check (adaptive thresholds) ─────────────────────

    def _check_entry_trigger(self, ind: Dict) -> bool:
        """
        Confirm entry from WAIT_CONFIRM using Adaptive Thresholds.
        RSI gate + MACD mode gate per market state.
        """
        direction = self.direction_focus
        if direction is None:
            return False

        rsi       = ind.get("rsi", 50)
        macd      = ind.get("macd", 0.0)
        macd_sig  = ind.get("macd_signal", 0.0)
        macd_hist = ind.get("macd_hist", macd - macd_sig)
        momentum  = ind.get("momentum_score", 50)

        thrs = ADAPTIVE_THRESHOLDS.get(
            self.current_market_state, ADAPTIVE_THRESHOLDS["TRENDING"]
        )

        if direction == "LONG":
            lo, hi = thrs["rsi_long"]
            if not (lo < rsi < hi):
                return False
            if momentum <= 50:
                return False
            macd_mode = thrs.get("macd_long", "any")
            if macd_mode == "above_signal"  and macd <= macd_sig:  return False
            if macd_mode == "above_zero"    and macd <= 0:          return False
            if macd_mode == "turning_up"    and macd_hist <= 0:     return False
        else:
            lo, hi = thrs["rsi_short"]
            if not (lo < rsi < hi):
                return False
            if momentum >= 50:
                return False
            macd_mode = thrs.get("macd_short", "any")
            if macd_mode == "below_signal"  and macd >= macd_sig:  return False
            if macd_mode == "below_zero"    and macd >= 0:          return False
            if macd_mode == "turning_down"  and macd_hist >= 0:     return False

        return True

    # ── Core tick engine ──────────────────────────────────────────────────────

    def on_tick(self, candle_15m: Dict, candle_1h: Dict, candle_4h: Dict,
                ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                extras: Dict, current_price: float,
                bar_dt: Optional[datetime.datetime] = None):
        """
        Called once per closed 15M candle.
        Signature unchanged from V7 — backtest engine compatible.
        """
        now = bar_dt or datetime.datetime.now()
        self._bar_now = now
        self._bar_count += 1
        self._last_candle_15m = candle_15m

        if self._startup_unblock_at is None and self.startup_warmup_minutes > 0:
            self._startup_unblock_at = now + datetime.timedelta(
                minutes=self.startup_warmup_minutes)
            self._log_event(
                f"WARMUP: blocking entries until "
                f"{self._startup_unblock_at.strftime('%H:%M UTC')} "
                f"({self.startup_warmup_minutes}m)",
                level="warning",
            )

        self._check_daily_reset(bar_dt)

        atr_4h = ind_4h.get("atr", 0)
        if atr_4h > 0:
            self.atr_history.append(atr_4h)
            if len(self.atr_history) > 200:
                self.atr_history.pop(0)

        # [V8-1] Market state from 4H
        self.current_market_state = self._step1_market_state_engine(ind_4h)
        self.regime_score          = self._step2_4h_regime(ind_4h)
        # [V8-5] Regime bias from 4H + 1H
        self.current_regime_bias   = self.bias_engine.compute(ind_4h, ind_1h)

        self._tick_depth = 0
        state_changed = True

        while state_changed and self._tick_depth < self._max_tick_depth:
            state_changed = False
            self._tick_depth += 1

            if self.state in ("BLOCKED", "COOLDOWN"):
                if (self.state == "COOLDOWN" and
                        (self.cooldown_until is None or now >= self.cooldown_until)):
                    self.state       = "SCANNING"
                    self.loss_streak = 0
                    self._log_event("Cooldown expired → SCANNING")
                    state_changed    = True

            elif self.state == "SCANNING":
                self.direction_focus = None
                if self._check_global_gates():
                    self.state    = "FILTERING"
                    state_changed = True

            elif self.state == "FILTERING":
                best_dir: Optional[str]  = None
                best_health: float       = -1.0
                vol_state = self.adaptive_engine.get_atr_volatility_state(
                    atr_4h, self.atr_history)
                mr_candidate = None

                # [MR] Try MeanReversion first if enabled
                if self.enable_mean_reversion and self._mr_strategy:
                    mr_candidate = self._mr_strategy.generate_signal(
                        candle_15m, ind_15m, ind_1h, ind_4h,
                        self.current_market_state, extras, bar_dt,
                    )
                    if mr_candidate:
                        self._log_event(
                            f"[MR] {mr_candidate['direction']} health={mr_candidate['health_score']:.0f} "
                            f"| {' '.join(mr_candidate['step_notes'][:4])}"
                        )

                # [SR] Try SwingReversal if enabled
                if self.enable_swing_reversal:
                    for direction in ("LONG", "SHORT"):
                        # Quick 1H structural filter
                        if not self._check_htf_trend_filter(direction, ind_1h):
                            continue

                        # [V8-3/4/5] Full health + confidence + bias gate
                        if not self._layer_filtering(
                                direction, ind_15m, self.current_market_state,
                                ind_1h, ind_4h):
                            continue

                        if self._last_entry_health > best_health:
                            best_health = self._last_entry_health
                            best_dir    = direction
                            best_health_stored   = self._last_entry_health
                            best_conf_stored     = self._last_confidence
                            best_conf_lvl_stored = self._last_confidence_level

                if mr_candidate is not None:
                    # MeanReversion: all 13 steps already validated — skip WAIT_CONFIRM
                    self._pending_signal = mr_candidate
                    self.direction_focus = mr_candidate["direction"]
                    self.state           = "PENDING_ORDER"
                    state_changed        = True
                elif best_dir is not None:
                    self._pending_signal       = None
                    self._last_entry_health    = best_health_stored
                    self._last_confidence      = best_conf_stored
                    self._last_confidence_level = best_conf_lvl_stored
                    self.direction_focus        = best_dir
                    self.state                  = "WAIT_CONFIRM"
                    self.bars_since_trigger     = 0
                    self._log_event(
                        f"Signal: {best_dir} | {self.current_market_state} "
                        f"| bias={self.current_regime_bias} "
                        f"| health={best_health:.0f} conf={best_conf_stored:.0f}({best_conf_lvl_stored})"
                    )
                    state_changed = True
                else:
                    self._pending_signal = None
                    self.state = "SCANNING"

            elif self.state == "WAIT_CONFIRM":
                if self.bars_since_trigger > 3:
                    self._log_event("WAIT_CONFIRM timeout → SCANNING")
                    self.state    = "SCANNING"
                    state_changed = True
                    continue

                if self._check_entry_trigger(ind_15m):
                    self.state    = "PENDING_ORDER"
                    state_changed = True
                else:
                    self.bars_since_trigger += 1

            elif self.state == "PENDING_ORDER":
                self._step5_risk_engine(
                    candle_15m, self.direction_focus, ind_15m,
                    mr_signal=self._pending_signal,
                )
                self._pending_signal = None  # consumed
                self.state    = "IN_POSITION"
                state_changed = False

            elif self.state in ("IN_POSITION", "PARTIAL_EXIT", "TRAILING"):
                new_state = self._manage_open_position(current_price, ind_15m)
                if new_state == "EXITING":
                    self.state    = "EXITING"
                    state_changed = True
                elif new_state != self.state:
                    self.state = new_state

            elif self.state == "EXITING":
                t           = self.current_trade
                pnl         = t.get("realized_pnl", 0.0)
                result      = "WIN" if pnl > 0 else "LOSS"
                close_reason = t.get("exit_price") and "SL_HIT"  # will be overridden below
                # Derive close reason from last CLOSE_FULL send
                close_reason = (
                    "SL_HIT"        if t.get("exit_price") is not None and pnl < 0 else
                    "TP3_RUNNER"    if t.get("tp3_hit") else
                    "FULL_TP2"      if t.get("tp2_hit") else
                    "PARTIAL_TP1"   if t.get("tp1_hit") else
                    "EXIT"
                )
                self._log_trade(result, close_reason, ind_15m, extras)
                if self.state not in ("COOLDOWN", "BLOCKED"):
                    self.state = "SCANNING"

            elif self.state == "RECOVERY":
                if not self.position_open:
                    self._log_event("RECOVERY: waiting for half-risk setup")
                    if self._check_global_gates():
                        self._enter_recovery_trade(candle_15m, ind_15m, ind_1h,
                                                   ind_4h, atr_4h)

            elif self.state == "ERROR":
                self._log_event("Bot in ERROR state — manual check required", level="error")
                break

        self.save_state(self._state_file)

    def _enter_recovery_trade(self, candle: Dict, ind_15m: Dict, ind_1h: Dict,
                               ind_4h: Dict, atr_4h: float):
        original_risk = self.base_risk_pct
        self.base_risk_pct *= 0.5
        for direction in ("LONG", "SHORT"):
            if not self._check_htf_trend_filter(direction, ind_1h):
                continue
            if self._layer_filtering(direction, ind_15m,
                                     self.current_market_state, ind_1h, ind_4h):
                self._step5_risk_engine(candle, direction, ind_15m)
                self.state = "IN_POSITION"
                self._log_event(f"RECOVERY trade: {direction} at half-risk")
                break
        self.base_risk_pct = original_risk

    # ── Public API ────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        _now = self._bar_now or datetime.datetime.now()
        warmup_remaining = 0
        if self._startup_unblock_at and _now < self._startup_unblock_at:
            warmup_remaining = int(
                (_now - self._startup_unblock_at).total_seconds() / 60)
        return {
            "state":              self.state,
            "position_open":      self.position_open,
            "direction":          self.current_trade.get("direction") if self.position_open else None,
            "entry":              self.current_trade.get("entry") if self.position_open else None,
            "realized_pnl":       self.current_trade.get("realized_pnl", 0),
            "account_balance":    self.account_balance,
            "daily_pnl_pct":      self.daily_pnl_pct,
            "loss_streak":        self.loss_streak,
            "win_streak":         self.win_streak,
            "consecutive_sl":     self.consecutive_sl_hits,
            "session_losses":     self.session_losses,
            "total_trades":       len(self.trade_journal),
            "market_state":       self.current_market_state,
            "regime_bias":        self.current_regime_bias,
            "regime_score":       self.regime_score,
            "cooldown_until":     self.cooldown_until.isoformat() if self.cooldown_until else None,
            "warmup_remaining_m": warmup_remaining,
            "recent_log":         list(self._log[-20:]),
        }

    def get_position_health_report(self, current_price: float, ind_15m: Dict) -> Dict:
        if not self.position_open or not self.current_trade:
            return {"in_position": False}

        t         = self.current_trade
        direction = t["direction"]
        entry     = t["entry"]
        sl_dist   = max(t["sl_dist"], 1e-9)
        dir_mult  = 1 if direction == "LONG" else -1
        current_r = ((current_price - entry) * dir_mult) / sl_dist
        pnl       = (current_price - entry) * dir_mult * t.get("remaining_size", t["size"])

        health = self.health_calc.calculate(ind_15m, t, current_price)
        level  = self._health_level(health)
        reversal = self._detect_reversal_signals(ind_15m)

        return {
            "in_position":       True,
            "direction":         direction,
            "entry":             entry,
            "current_price":     current_price,
            "sl":                t["sl"],
            "tp1":               t["tp1"],
            "tp2":               t["tp2"],
            "tp1_hit":           t.get("tp1_hit", False),
            "current_r":         round(current_r, 3),
            "pnl":               round(pnl, 2),
            "health_score":      round(health, 1),
            "health_level":      level,
            "entry_health":      t.get("entry_health", 0),
            "entry_confidence":  t.get("entry_confidence", 0),
            "regime_bias":       self.current_regime_bias,
            "reversal_signals":  reversal,
            "adx":               round(ind_15m.get("adx",  0), 1),
            "rsi":               round(ind_15m.get("rsi", 50), 1),
            "macd_hist":         round(ind_15m.get("macd_hist", 0), 4),
            "holding_bars":      self._bar_count - self._position_entry_bar,
        }

    def get_performance_summary(self) -> Dict:
        if not self.trade_journal:
            return {"message": "No trades yet"}

        wins   = [t for t in self.trade_journal if t["win_loss"] == "WIN"]
        losses = [t for t in self.trade_journal if t["win_loss"] == "LOSS"]
        pnls   = [t["pnl"] for t in self.trade_journal if t.get("pnl") is not None]

        by_state: Dict[str, Dict] = {}
        for t in self.trade_journal:
            s = t.get("market_state", "Unknown")
            by_state.setdefault(s, {"wins": 0, "total": 0, "pnl": 0.0})
            by_state[s]["total"] += 1
            by_state[s]["pnl"]   += t.get("pnl", 0)
            if t["win_loss"] == "WIN":
                by_state[s]["wins"] += 1

        return {
            "total_trades":  len(self.trade_journal),
            "win_rate":      len(wins) / len(self.trade_journal),
            "net_pnl":       sum(pnls),
            "avg_win":       np.mean([t["pnl"] for t in wins])   if wins   else 0,
            "avg_loss":      np.mean([t["pnl"] for t in losses]) if losses else 0,
            "profit_factor": (sum(t["pnl"] for t in wins) /
                              abs(sum(t["pnl"] for t in losses)))
                             if losses else float("inf"),
            "by_market_state": {
                s: {**v, "win_rate": v["wins"] / max(v["total"], 1)}
                for s, v in by_state.items()
            },
            "pattern_learning": self.learning_engine.get_summary(),
        }

    def force_state(self, new_state: str):
        if new_state in self.STATES:
            self._log_event(f"Force state: {self.state} → {new_state}")
            self.state = new_state
        else:
            raise ValueError(f"Unknown state: {new_state}")

    # ── State persistence ─────────────────────────────────────────────────────

    DEFAULT_STATE_FILE = "bot_state.json"

    def save_state(self, path: str = DEFAULT_STATE_FILE):
        import os as _os
        if not path or path == _os.devnull:
            return
        snapshot = {
            "state":                self.state,
            "position_open":        self.position_open,
            "order_status":         self.order_status,
            "current_trade":        self.current_trade,
            "account_balance":      self.account_balance,
            "daily_pnl_pct":        self.daily_pnl_pct,
            "loss_streak":          self.loss_streak,
            "win_streak":           self.win_streak,
            "consecutive_sl_hits":  self.consecutive_sl_hits,
            "session_losses":       self.session_losses,
            "cooldown_until":       self.cooldown_until.isoformat() if self.cooldown_until else None,
            "trading_date":         self.trading_date.isoformat() if self.trading_date else None,
            "current_market_state": self.current_market_state,
            "current_regime_bias":  self.current_regime_bias,
            "regime_score":         self.regime_score,
            "direction_focus":      self.direction_focus,
            "bars_since_trigger":   self.bars_since_trigger,
            "atr_history":          self.atr_history[-200:],
            "adaptive_weights":     self.adaptive_engine.base_weights,
            "pattern_learning":     self.learning_engine.to_dict(),
            "saved_at":             datetime.datetime.now().isoformat(),
        }
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str, ensure_ascii=False)
            os.replace(tmp_path, path)
        except OSError as e:
            self._log_event(f"[ERROR] save_state failed: {e}", level="error")

    def load_state(self, path: str = DEFAULT_STATE_FILE) -> bool:
        if not os.path.exists(path):
            self._log_event(f"No saved state at '{path}' — starting fresh")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._log_event(
                f"[ERROR] load_state: '{path}': {e} — starting fresh",
                level="error",
            )
            return False

        self.state           = data.get("state", "SCANNING")
        self.position_open    = data.get("position_open", False)
        self.order_status     = data.get("order_status", "CLOSED")
        self.current_trade    = data.get("current_trade") or {}

        entry_time = self.current_trade.get("entry_time")
        if isinstance(entry_time, str):
            try:
                self.current_trade["entry_time"] = datetime.datetime.fromisoformat(entry_time)
            except ValueError:
                self.current_trade["entry_time"] = datetime.datetime.now()

        self.account_balance      = data.get("account_balance", self.account_balance)
        self.daily_pnl_pct         = data.get("daily_pnl_pct", 0.0)
        self.loss_streak           = data.get("loss_streak", 0)
        self.win_streak            = data.get("win_streak", 0)
        self.consecutive_sl_hits   = data.get("consecutive_sl_hits", 0)
        self.session_losses        = data.get("session_losses", 0)

        cooldown = data.get("cooldown_until")
        self.cooldown_until = (datetime.datetime.fromisoformat(cooldown)
                               if cooldown else None)

        trading_date = data.get("trading_date")
        self.trading_date = (datetime.date.fromisoformat(trading_date)
                             if trading_date else None)

        self.current_market_state  = data.get("current_market_state", "SIDEWAY")
        self.current_regime_bias   = data.get("current_regime_bias", "NEUTRAL")
        self.regime_score          = data.get("regime_score", 0.0)
        self.direction_focus       = data.get("direction_focus")
        self.bars_since_trigger    = data.get("bars_since_trigger", 0)
        self.atr_history           = data.get("atr_history", [])

        weights = data.get("adaptive_weights")
        if weights:
            self.adaptive_engine.base_weights = weights

        pl = data.get("pattern_learning")
        if pl:
            self.learning_engine.from_dict(pl)

        self._log_event(
            f"State loaded | state={self.state} pos={self.position_open} "
            f"balance={self.account_balance:.2f} saved_at={data.get('saved_at')}"
        )
        return True

    def reconcile_with_exchange(self, symbol: str, exchange_adapter) -> None:
        try:
            live_position = exchange_adapter.fetch_open_position(symbol)
        except Exception as e:
            self._log_event(
                f"[ERROR] reconcile: fetch_open_position failed: {e} → ERROR state",
                level="error",
            )
            self.state = "ERROR"
            return

        live_size = float((live_position or {}).get("contracts", 0) or 0)

        if self.position_open and live_size == 0:
            self._log_event(
                "[RECONCILE] local=open but exchange=no position → closed offline; "
                "clearing local state → SCANNING",
                level="warning",
            )
            self.position_open = False
            self.current_trade = {}
            if self.state not in ("BLOCKED", "COOLDOWN", "ERROR"):
                self.state = "SCANNING"

        elif not self.position_open and live_size != 0:
            direction   = "LONG" if live_size > 0 else "SHORT"
            entry_price = float(
                live_position.get("entryPrice") or live_position.get("entryPx") or 0
            )
            self._log_event(
                f"[RECONCILE] exchange has unknown {direction} pos (size={abs(live_size)}) "
                "→ creating trade record from exchange data",
                level="warning",
            )
            sl_dist_r = max(entry_price * 0.01, 1e-9)
            mult_r    = 1 if direction == "LONG" else -1
            self.current_trade = {
                "direction": direction, "entry": entry_price,
                "sl": entry_price * (0.99 if direction == "LONG" else 1.01),
                "sl_dist": sl_dist_r,
                "tp1": entry_price + sl_dist_r * 0.7 * mult_r,
                "tp2": entry_price + sl_dist_r * 2.0 * mult_r,
                "size": abs(live_size), "remaining_size": abs(live_size),
                "tp1_hit": False, "tp2_hit": False,
                "break_even_triggered": False, "status": "OPEN",
                "entry_time": datetime.datetime.now(),
                "atr_at_entry": sl_dist_r,
                "realized_pnl": 0.0, "exit_price": None, "final_rr": None,
                "mae": 0.0, "mfe": 0.0,
            }
            self.position_open = True
            self.state         = "IN_POSITION"
