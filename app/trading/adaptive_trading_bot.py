"""
Adaptive Trading Bot — v9.3  Balanced Active Multi-Layer Engine
============================================================
3-layer pipeline replacing the old 8-state flat classifier:

  L1  4H MacroTrendEngine
      EMA20/50 cross · EMA slope · ADX+DI direction · efficiency ratio
      ATR regime · structure score
      → Trend Score 0-100 · 5-level label (STRONG_BULL … STRONG_BEAR)

  L2  1H ContextBiasEngine
      7 weighted components: structure(20%) · pattern(20%) · liquidity(15%)
      · EMA pullback(15%) · RSI(10%) · MACD(10%) · volume(10%)
      → separate Bull score and Bear score 0-100

  L3  RegimeClassifier
      Combines L1 + L2 + 15M ADX/ATR/BB/RSI/efficiency
      → 5 regimes: Trend · Range · Breakout · Reversal · Exhaustion

  Dynamic Strategy Selection (4 strategies per regime):
      Trend     → EMA_Pullback · ADX_Trend · MACD_Trend · HMA_Trend
      Range     → RSI_Bounce · BB_Revert · VWAP_Rev · Mean_Rev
      Breakout  → Volume_Break · ATR_Expand · BOS_Break · BB_Squeeze
      Reversal  → RSI_Diverge · QM_Pattern · CHOCH_Rev · Exhaust_Rev
      Exhaustion→ RSI_Diverge · CHOCH_Rev · BB_Revert · QM_Pattern

  Dynamic Weighting (weights shift per regime):
      Trend: EMA25% · ADX20% · Momentum20% · Volume15% · Liquidity10% · Pattern10%
      Range: RSI25% · BB20% · VWAP20% · Volume15% · Liquidity10% · Pattern10%
      Breakout: Volume25% · ATR20% · Momentum20% · EMA15% · Pattern10% · Liquidity10%
      Reversal/Exhaustion: RSI25% · Pattern20% · Structure20% · Momentum15% …

  Confidence Engine selects single highest-scoring strategy per signal.

V8 execution infrastructure preserved:
  - State machine SCANNING→FILTERING→PENDING_ORDER→IN_POSITION→EXITING
  - 2-target TP: T1=0.5R close 50% + SL→breakeven, T2=1.2R full close
  - Position Health Calculator, reversal spike / trend-fade protection
  - save_state / load_state / reconcile_with_exchange
  - Daily PnL limits · cooldown · win-streak risk reduction
  - ConditionLearningEngine (Level 0/2/3 adaptive learning)
  - PatternLearningEngine (per-entry-type WR tracking)
"""

import numpy as np
import datetime
import json
import os
import logging
from typing import Optional, Callable, Dict, List, Any

from .strategies.mean_reversion import MeanReversionStrategy
from .mtf_confluence_engine import MTFConfluenceEngine, hma, roc
from .strategies.base import BaseStrategy


logger = logging.getLogger("adaptive_trading_bot")


# ══════════════════════════════════════════════════════════════════════════════
# REGIME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Regime → candidate strategy pool (scored by StrategyScorer, best selected by ConfidenceEngine)
REGIME_STRATEGIES: Dict[str, List[str]] = {
    "Trend":      ["EMA_Pullback", "ADX_Trend",    "MACD_Trend",  "HMA_Trend"],
    "Range":      ["RSI_Bounce",   "BB_Revert",    "VWAP_Rev",    "Mean_Rev"],
    "Breakout":   ["Volume_Break", "ATR_Expand",   "BOS_Break",   "BB_Squeeze"],
    "Reversal":   ["RSI_Diverge",  "QM_Pattern",   "CHOCH_Rev",   "Exhaust_Rev"],
    "Exhaustion": ["RSI_Diverge",  "CHOCH_Rev",    "BB_Revert",   "QM_Pattern"],
}

# Per-regime indicator weights (values in each dict must sum to 1.0)
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Trend": {
        "ema": 0.25, "adx": 0.20, "momentum": 0.20,
        "volume": 0.15, "liquidity": 0.10, "pattern": 0.10,
    },
    "Range": {
        "rsi": 0.25, "bb": 0.20, "vwap": 0.20,
        "volume": 0.15, "liquidity": 0.10, "pattern": 0.10,
    },
    "Breakout": {
        "volume": 0.25, "atr": 0.20, "momentum": 0.20,
        "ema": 0.15, "pattern": 0.10, "liquidity": 0.10,
    },
    "Reversal": {
        "rsi": 0.25, "pattern": 0.20, "structure": 0.20,
        "momentum": 0.15, "volume": 0.10, "liquidity": 0.10,
    },
    "Exhaustion": {
        "rsi": 0.25, "pattern": 0.20, "structure": 0.20,
        "momentum": 0.15, "volume": 0.10, "atr": 0.10,
    },
}

# Minimum composite score (strategy*0.40 + L2ctx*0.30 + L1fit*0.30 - penalty) to generate a signal
REGIME_THRESHOLDS: Dict[str, int] = {
    # [V9.3 ACTIVE] Small relaxation only after all hard quality gates pass.
    # This increases usable setups without turning the score into a loose vote.
    "Trend":      57,
    # Conservative Range Edge-Fade: hard location/sweep/reclaim gates run
    # before this score, so 70 is a final quality bar rather than a loose vote.
    "Range":      70,
    # [V9.4 BREAKOUT] Raised 56→64: the dedicated Breakout engine
    # (_generate_breakout_signal) adds a hard macro gate + 4/6 confirmations +
    # retest logic, so this is a final quality bar on top of those, not a
    # loose vote. Only real close-through breaks with macro support survive.
    "Breakout":   64,
    "Reversal":   65,
    "Exhaustion": 68,
}

# [V9.4 RANGE ADD-ON] Entry regime policy.
# Trend and Breakout keep their existing signal paths unchanged. Range is
# admitted only through the dedicated Conservative Range Edge-Fade engine
# (_generate_range_signal), which requires a mature range, edge location,
# liquidity sweep + reclaim, 3/5 confirmation, acceptable structure R:R, and
# half-risk sizing. Reversal/Exhaustion remain diagnostics-only.
_TRADEABLE_REGIMES: frozenset = frozenset({"Trend", "Breakout", "Range"})

# Regimes where entries FADE the macro trend (want opposite of L1 direction)
_COUNTER_REGIMES: frozenset = frozenset({"Reversal", "Exhaustion"})

# Regimes where we use mean-reversion SL calculation (MR strategy's _step14_sl)
_MR_REGIMES: frozenset = frozenset({"Range", "Reversal", "Exhaustion"})

# Entry-type label per regime (fed into PatternLearningEngine + journal)
_REGIME_ENTRY_TYPE: Dict[str, str] = {
    "Trend":      "trend_follow",
    "Range":      "mean_revert",
    "Breakout":   "breakout",
    "Reversal":   "reversal",
    "Exhaustion": "counter_trend",
}


# ══════════════════════════════════════════════════════════════════════════════
# L1 — 4H MACRO TREND ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _fast_lean_score(candles: List) -> float:
    """
    [EARLY TREND] Continuous 0-100 directional lean from the SAME
    low-lag indicator family as mtf_confluence_engine (HMA10/20, ROC9,
    MACD histogram) — evaluated on the current bar's state, not on a
    cross event, so it moves every bar rather than only at the moment
    of a cross. This is the "fast" read MacroTrendEngine.compute()
    blends in as its early-trend component: EMA20/50 needs the price to
    travel a long way before crossing, while HMA/MACD/ROC already lean
    a clear direction well before that (see the reference chart that
    prompted this — HMA/MACD/ROC all turned up several bars before any
    EMA20/50-based signal would have reacted).
    50.0 (neutral) when there isn't enough history to compute yet.
    """
    if len(candles) < 40:
        return 50.0
    closes = [float(c.close) for c in candles]
    hma_f = hma(closes, 10)
    hma_s = hma(closes, 20)
    r     = roc(closes, 9)
    _, macd_sig, macd_hist = BaseStrategy.macd(closes, 12, 26, 9)
    vals = (hma_f[-1], hma_s[-1], r[-1], macd_hist[-1], macd_sig[-1])
    if any(np.isnan(v) for v in vals):
        return 50.0

    price = max(closes[-1], 1e-9)
    hma_lean  = float(np.clip(50.0 + (hma_f[-1] - hma_s[-1]) / price * 3000.0, 0, 100))
    roc_lean  = float(np.clip(50.0 + r[-1] * 10.0, 0, 100))
    hist_lean = float(np.clip(
        50.0 + macd_hist[-1] / max(abs(macd_sig[-1]), 1e-9) * 50.0, 0, 100))

    return hma_lean * 0.4 + roc_lean * 0.3 + hist_lean * 0.3


def compute_early_trend(candles_4h: List, candles_1h: List) -> Dict:
    """
    [EARLY TREND] Dual-timeframe confirmation gate for the fast lean above.
    Both 4H and 1H must lean the SAME direction (both >55 or both <45) —
    matching the user's own explicit requirement to confirm across 2 TFs
    before accelerating the regime call, so a single timeframe's noise
    can't jerk the macro trend around on its own.
    """
    tf4h = _fast_lean_score(candles_4h)
    tf1h = _fast_lean_score(candles_1h)
    confirmed = (tf4h > 55 and tf1h > 55) or (tf4h < 45 and tf1h < 45)
    return {
        "tf4h": tf4h, "tf1h": tf1h,
        "score": (tf4h + tf1h) / 2.0,
        "confirmed": confirmed,
    }


class MacroTrendEngine:
    """
    L1: 4H Macro Trend Engine
    Inputs: 4H indicators (ema20, ema50, slope, adx, pdi, mdi, eff_ratio,
            atr_exp, structure_score)
    Outputs: score 0-100 (100=max bull, 0=max bear), 5-level label, direction

    Component weights (early_trend=None, the default — identical to every
    backtest run so far, no regression):
      EMA20 vs EMA50 cross & distance : 25%
      EMA20 slope                      : 20%
      ADX strength + DI direction      : 20%
      Efficiency ratio (HH/HL proxy)   : 15%
      ATR regime (expansion=trending)  : 10%
      Market structure score           : 10%

    [EARLY TREND] When called with a CONFIRMED early_trend dict (see
    compute_early_trend — both 4H and 1H fast indicators agree), weights
    rebalance to fold in a 7th, low-lag component so a fresh reversal
    shows up in the score before EMA20/50 would otherwise catch up:
      EMA20 vs EMA50 : 20%   EMA20 slope : 15%   ADX+DI : 15%
      Efficiency     : 10%   ATR regime  : 10%   Structure : 10%
      Early Trend (confirmed 4H+1H HMA/MACD/ROC) : 20%
    Unconfirmed (only one TF agrees, or none) -> behaves exactly like
    early_trend=None; this only ever adds signal, never dilutes toward
    neutral, so it's a pure accelerant as requested, not a new veto.
    """

    LEVELS = [
        (80, "STRONG_BULL"),
        (60, "BULL"),
        (40, "NEUTRAL"),
        (20, "BEAR"),
        ( 0, "STRONG_BEAR"),
    ]

    def compute(self, ind_4h: Dict, early_trend: Optional[Dict] = None,
               ema_fast_override: Optional[float] = None,
               ema_slow_override: Optional[float] = None) -> Dict:
        # [FAST MACRO EMA] Component 1 defaults to IndicatorEngine's EMA20/50
        # (shared with every other consumer — chop-zone veto, ContextBiasEngine,
        # etc., untouched). ema_fast_override/ema_slow_override (e.g. EMA12/26,
        # computed by the caller from raw 4H candles — see on_tick) let this ONE
        # component react faster without recomputing IndicatorEngine's shared
        # EMA20/50 globally. slope_score below intentionally keeps reading
        # ind_4h's real EMA20 slope either way — this only affects the cross
        # /distance read, a narrower, independently-testable change.
        ema20  = ema_fast_override if ema_fast_override is not None else ind_4h.get("ema20", 0.0)
        ema50  = ema_slow_override if ema_slow_override is not None else ind_4h.get("ema50", ema20)
        slope  = ind_4h.get("ema20_slope_score", 50.0)
        adx    = ind_4h.get("adx", 20.0)
        pdi    = ind_4h.get("pdi", 20.0)
        mdi    = ind_4h.get("mdi", 20.0)
        atr_e  = ind_4h.get("atr_exp", 1.0)
        struct = ind_4h.get("structure_score", 50.0)  # price vs EMA20 vs EMA50

        # 1. EMA fast vs slow cross / distance (25%) — EMA20/50 unless overridden
        ref = max(ema50, 1e-9)
        ema_spread  = (ema20 - ref) / ref
        ema_score   = float(np.clip(50.0 + ema_spread * 1500.0, 0, 100))

        # 2. EMA20 slope (20%)
        slope_score = float(np.clip(slope, 0, 100))

        # 3. ADX strength + DI direction (20%)
        adx_str   = float(np.clip(adx / 45.0 * 100, 0, 100))
        di_bull   = pdi > mdi
        adx_score = float(np.clip(50.0 + (1 if di_bull else -1) * adx_str * 0.5, 0, 100)) \
                    if adx > 12 else 50.0

        # 4. Efficiency ratio — structure_score carries directionality (15%)
        eff_dir_score = float(np.clip(struct, 0, 100))

        # 5. ATR regime (10%) — atr_exp > 1 = expanding/trending, < 1 = contracting
        atr_score = float(np.clip(50.0 + (atr_e - 1.0) * 50.0, 0, 100))

        # 6. Structure score (10%) — 15/35/50/65/85 from indicator engine
        struct_score = float(np.clip(struct, 0, 100))

        if early_trend and early_trend.get("confirmed"):
            # [EARLY TREND] 7th component, rebalanced weights — see class
            # docstring. Only ever engages when compute_early_trend() found
            # BOTH 4H and 1H fast indicators agreeing; otherwise identical
            # to the branch below.
            score = (
                ema_score     * 0.20 +
                slope_score   * 0.15 +
                adx_score     * 0.15 +
                eff_dir_score * 0.10 +
                atr_score     * 0.10 +
                struct_score  * 0.10 +
                float(early_trend["score"]) * 0.20
            )
        else:
            score = (
                ema_score    * 0.25 +
                slope_score  * 0.20 +
                adx_score    * 0.20 +
                eff_dir_score * 0.15 +
                atr_score    * 0.10 +
                struct_score * 0.10
            )
        score = float(np.clip(score, 0, 100))

        level = "STRONG_BEAR"
        for threshold, lbl in self.LEVELS:
            if score >= threshold:
                level = lbl
                break

        return {
            "score":       score,
            "level":       level,
            "direction":   1 if score >= 55 else (-1 if score <= 45 else 0),
            "ema_score":   ema_score,
            "adx_score":   adx_score,
            "slope_score": slope_score,
        }


# ══════════════════════════════════════════════════════════════════════════════
# L2 — 1H CONTEXT BIAS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ContextBiasEngine:
    """
    L2: 1H Context & Bias Engine
    Returns separate bull_score and bear_score (0-100 each).
    Caller uses bull_score for LONG candidates, bear_score for SHORT.

    Component weights:
      Structure (EMA20/50 alignment)  : 20%
      Pattern (candle clarity)         : 20%
      Liquidity (momentum proxy)       : 15%
      EMA pullback (EMA5 vs EMA20)     : 15%
      RSI (mean-reversion lean)        : 10%
      MACD (histogram direction)       : 10%
      Volume (directional confirmation): 10%
    """

    W = {
        "structure":    0.20,
        "pattern":      0.20,
        "liquidity":    0.15,
        "ema_pullback": 0.15,
        "rsi":          0.10,
        "macd":         0.10,
        "volume":       0.10,
    }

    def compute(self, ind_1h: Dict) -> Dict:
        ema5   = ind_1h.get("ema5",  0.0)
        ema20  = ind_1h.get("ema20", max(ema5, 1.0))
        rsi    = ind_1h.get("rsi",   50.0)
        macd   = ind_1h.get("macd",  0.0)
        msig   = ind_1h.get("macd_signal", 0.0)
        vol    = ind_1h.get("volume", 0.0)
        vavg   = max(ind_1h.get("vol_avg", max(vol, 1e-9)), 1e-9)
        mom    = ind_1h.get("momentum_score", 50.0)
        struct = ind_1h.get("structure_score", 50.0)
        pat    = float(np.clip(ind_1h.get("pattern_score", 50.0), 0, 100))

        # Structure (directional 15/35/50/65/85 from indicator engine)
        sb = float(np.clip(struct, 0, 100));       sB = 100.0 - sb

        # Pattern
        pb = pat;                                  pB = 100.0 - pat

        # Liquidity (momentum_score as sweep/liquidity proxy)
        lb = float(np.clip(mom, 0, 100));          lB = 100.0 - lb

        # EMA pullback: EMA5 vs EMA20 lean
        el   = (ema5 - ema20) / max(ema20, 1e-9)
        eb   = float(np.clip(50.0 + el * 2000.0, 0, 100))
        eB   = 100.0 - eb

        # RSI (oversold=bullish lean, overbought=bearish)
        rb   = float(np.clip(50.0 + (50.0 - rsi) * 1.5, 0, 100))
        rB   = 100.0 - rb

        # MACD histogram direction
        mhist = macd - msig
        mstr  = float(np.clip(abs(mhist / max(abs(msig), 1e-9)) * 50.0, 0, 50))
        mb    = float(np.clip(50.0 + (mstr if mhist > 0 else -mstr), 0, 100))
        mB    = 100.0 - mb

        # Volume (directionless — slight amplifier, clipped to [30,70])
        vr  = vol / vavg
        vb  = float(np.clip(50.0 + (vr - 1.0) * 10.0, 30, 70))
        vB  = vb

        W = self.W
        bull = (sb * W["structure"] + pb * W["pattern"] + lb * W["liquidity"] +
                eb * W["ema_pullback"] + rb * W["rsi"] + mb * W["macd"] + vb * W["volume"])
        bear = (sB * W["structure"] + pB * W["pattern"] + lB * W["liquidity"] +
                eB * W["ema_pullback"] + rB * W["rsi"] + mB * W["macd"] + vB * W["volume"])

        return {
            "bull_score": float(np.clip(bull, 0, 100)),
            "bear_score": float(np.clip(bear, 0, 100)),
            "components": {
                "struct_bull": sb, "pat_bull": pb,
                "liq_bull":   lb, "ema_bull": eb,
                "rsi_bull":   rb, "macd_bull": mb,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# L3 — MARKET REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class RegimeClassifier:
    """
    L3: Combines L1 macro trend + L2 context + 15M micro indicators
    to classify into one of 5 regimes.

    Priority order (most specific wins):
      Exhaustion → Reversal → Breakout → Trend → Range
    """

    def classify(self, l1: Dict, l2: Dict, ind_15m: Dict) -> Dict:
        ts    = l1["score"]           # 0-100, 50=neutral
        bull  = l2["bull_score"]
        bear  = l2["bear_score"]
        bias_diff = abs(bull - bear)

        adx   = ind_15m.get("adx",      20.0)
        atr_e = ind_15m.get("atr_exp",   1.0)
        bb_w  = ind_15m.get("bb_width",  0.5)
        rsi   = ind_15m.get("rsi",      50.0)
        eff   = ind_15m.get("eff_ratio", 0.5)

        # 1. Exhaustion: trend extended + RSI extreme + efficiency collapsing
        if (ts >= 72 or ts <= 28) and (rsi >= 70 or rsi <= 30) and eff < 0.30 and adx > 18:
            return {"regime": "Exhaustion", "confidence": 0.85}

        # 2. Reversal: RSI extreme + context losing momentum direction
        if (rsi >= 67 or rsi <= 33) and bias_diff < 25 and adx > 15 and eff < 0.40:
            return {"regime": "Reversal", "confidence": 0.78}

        # 3. Breakout: BB compression expanding with ATR
        if bb_w < 0.25 and atr_e > 1.15 and adx < 22:
            return {"regime": "Breakout", "confidence": 0.80}

        # 4. Trend: L1 directional + 15M efficiency + L2 context agreement
        if adx > 20 and eff > 0.38:
            if (ts >= 58 and bull >= 53) or (ts <= 42 and bear >= 53):
                return {"regime": "Trend", "confidence": 0.85}
            if (ts >= 62 or ts <= 38) and adx > 24:
                return {"regime": "Trend", "confidence": 0.72}

        # 5. Range: balanced / low-energy / choppy
        if adx < 22 and eff < 0.38 and atr_e < 1.2:
            return {"regime": "Range", "confidence": 0.75}

        # Fallback
        if adx > 20 and eff > 0.30:
            return {"regime": "Trend", "confidence": 0.60}
        return {"regime": "Range", "confidence": 0.60}


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY SCORER
# ══════════════════════════════════════════════════════════════════════════════

class StrategyScorer:
    """Score each strategy candidate (0-100) for a direction in a given regime."""

    def score(self, strategy: str, direction: str, ind_15m: Dict,
              l1: Dict, l2: Dict, regime: str) -> float:
        W   = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["Trend"])
        dm  = 1 if direction == "LONG" else -1

        ema5   = ind_15m.get("ema5",  0.0)
        ema20  = ind_15m.get("ema20", max(ema5, 1.0))
        price  = ind_15m.get("close", ema20)
        rsi    = ind_15m.get("rsi",   50.0)
        adx    = ind_15m.get("adx",   20.0)
        pdi    = ind_15m.get("pdi",   20.0)
        mdi    = ind_15m.get("mdi",   20.0)
        macd   = ind_15m.get("macd",  0.0)
        msig   = ind_15m.get("macd_signal", 0.0)
        vol    = ind_15m.get("volume",  0.0)
        vavg   = max(ind_15m.get("vol_avg", max(vol, 1e-9)), 1e-9)
        atr_e  = ind_15m.get("atr_exp", 1.0)
        bb_w   = ind_15m.get("bb_width", 0.5)
        mom    = ind_15m.get("momentum_score", 50.0)
        struct = ind_15m.get("structure_score", 50.0)
        pat    = ind_15m.get("pattern_score",   50.0)

        # — EMA lean —
        ema_lean  = (price - ema20) / max(ema20, 1e-9)
        ema_raw   = float(np.clip(50.0 + ema_lean * 2000.0 * dm, 0, 100))

        # — ADX + DI direction —
        adx_str  = float(np.clip(adx / 40.0 * 100, 0, 100))
        di_ok    = (pdi > mdi) if direction == "LONG" else (mdi > pdi)
        adx_raw  = float(np.clip(adx_str * (1.0 if di_ok else 0.25), 0, 100))

        # — Momentum (MACD-derived) —
        mom_raw  = float(np.clip(mom if direction == "LONG" else 100.0 - mom, 0, 100))

        # — RSI (MR regimes want extremes; trend wants healthy mid-zone) —
        if regime in _MR_REGIMES:
            rsi_raw = float(np.clip((50.0 - rsi) / 25.0 * 50.0 + 50.0, 0, 100)) if direction == "LONG" \
                      else float(np.clip((rsi - 50.0) / 25.0 * 50.0 + 50.0, 0, 100))
        else:
            rsi_raw = float(np.clip(100.0 - max(0.0, rsi - 60.0) * 5.0 - max(0.0, 38.0 - rsi) * 3.0, 0, 100)) \
                      if direction == "LONG" \
                      else float(np.clip(100.0 - max(0.0, 62.0 - rsi) * 5.0 - max(0.0, rsi - 40.0) * 3.0, 0, 100))

        # — BB (tight = good for Range/Reversal; wide = good for Breakout/Trend) —
        bb_raw   = float(np.clip((0.5 - bb_w) / 0.5 * 100, 0, 100)) \
                   if regime in ("Range", "Reversal") \
                   else float(np.clip(bb_w / 0.6 * 100, 0, 100))

        # — Volume —
        vol_raw  = float(np.clip((vol / vavg) / 2.0 * 100, 0, 100))

        # — ATR expansion —
        atr_raw  = float(np.clip((atr_e - 0.8) / 1.0 * 100, 0, 100)) \
                   if regime in ("Breakout", "Trend") \
                   else float(np.clip((2.0 - atr_e) / 1.5 * 100, 0, 100))

        # — Liquidity (L2 context bias for this direction) —
        liq_raw  = (l2["bull_score"] if direction == "LONG" else l2["bear_score"])

        # — Structure —
        struct_raw = float(np.clip(struct, 0, 100)) if direction == "LONG" \
                     else float(np.clip(100.0 - struct, 0, 100))

        # — Pattern —
        pat_raw  = float(np.clip(pat, 0, 100)) if direction == "LONG" \
                   else float(np.clip(100.0 - pat, 0, 100))

        base = (
            ema_raw    * W.get("ema",       0) +
            adx_raw    * W.get("adx",       0) +
            mom_raw    * W.get("momentum",  0) +
            rsi_raw    * W.get("rsi",       0) +
            bb_raw     * W.get("bb",        0) +
            vol_raw    * W.get("volume",    0) +
            atr_raw    * W.get("atr",       0) +
            liq_raw    * W.get("liquidity", 0) +
            struct_raw * W.get("structure", 0) +
            pat_raw    * W.get("pattern",   0) +
            ema_raw    * W.get("vwap",      0)   # EMA20 as VWAP proxy
        )

        bonus = self._strategy_bonus(strategy, direction, ind_15m)
        return float(np.clip(base + bonus, 0, 100))

    def _strategy_bonus(self, strategy: str, direction: str, ind_15m: Dict) -> float:
        rsi   = ind_15m.get("rsi",   50.0)
        adx   = ind_15m.get("adx",   20.0)
        ema20 = ind_15m.get("ema20",  0.0)
        price = ind_15m.get("close", ema20)
        atr   = max(ind_15m.get("atr",  1e-9), 1e-9)
        macd  = ind_15m.get("macd",   0.0)
        msig  = ind_15m.get("macd_signal", 0.0)
        bb_w  = ind_15m.get("bb_width", 0.5)
        vol   = ind_15m.get("volume",  0.0)
        vavg  = max(ind_15m.get("vol_avg", max(vol, 1e-9)), 1e-9)
        pat   = ind_15m.get("pattern_score", 50.0)
        atr_e = ind_15m.get("atr_exp", 1.0)
        struct = ind_15m.get("structure_score", 50.0)

        if strategy == "EMA_Pullback":
            dist = (price - ema20) / atr if direction == "LONG" else (ema20 - price) / atr
            if 0 < dist < 1.5:   return 12.0
            if dist <= 0:        return -5.0

        elif strategy == "ADX_Trend":
            if adx > 25:    return 12.0
            if adx < 15:    return -8.0

        elif strategy == "MACD_Trend":
            return 10.0 if ((macd > msig) if direction == "LONG" else (macd < msig)) else -5.0

        elif strategy == "HMA_Trend":
            mom = ind_15m.get("momentum_score", 50.0)
            lean = mom if direction == "LONG" else 100.0 - mom
            return float(np.clip((lean - 50.0) * 0.2, -8, 10))

        elif strategy == "RSI_Bounce":
            if direction == "LONG":
                if rsi < 35: return 15.0
                if rsi < 45: return  7.0
            else:
                if rsi > 65: return 15.0
                if rsi > 55: return  7.0
            return -5.0

        elif strategy == "BB_Revert":
            if bb_w < 0.2:   return 10.0
            if bb_w > 0.5:   return -5.0

        elif strategy in ("VWAP_Rev", "Mean_Rev"):
            return float(np.clip(abs(price - ema20) / atr * 5.0, -5, 12))

        elif strategy in ("RSI_Diverge", "Exhaust_Rev", "CHOCH_Rev"):
            if direction == "LONG":
                if rsi < 35: return 15.0
                if rsi < 42: return  8.0
            else:
                if rsi > 65: return 15.0
                if rsi > 58: return  8.0
            return -8.0

        elif strategy == "QM_Pattern":
            return float(np.clip((pat - 50.0) * 0.2, -8, 12))

        elif strategy == "Volume_Break":
            r = vol / vavg
            if r > 1.8: return 12.0
            if r > 1.3: return  6.0
            return -5.0

        elif strategy in ("ATR_Expand", "BB_Squeeze"):
            if atr_e > 1.3: return 12.0
            if atr_e > 1.1: return  5.0

        elif strategy == "BOS_Break":
            return float(np.clip((struct - 50.0) * 0.3, -8, 12))

        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceEngine:
    """Select the single highest-scoring strategy for a direction."""

    def select_best(self, scores: Dict[str, float]) -> Optional[tuple]:
        """Returns (strategy_name, score) or None if scores is empty."""
        if not scores:
            return None
        best = max(scores, key=scores.__getitem__)
        return (best, scores[best])


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTANCY ENGINE — per (regime, strategy) self-pruning
# ══════════════════════════════════════════════════════════════════════════════

class ExpectancyEngine:
    """
    Tracks realized win-rate per (regime, strategy) combo and BLOCKS combos
    whose live track record proves negative expectancy. Same rule-based,
    no-ML design as PatternLearningEngine/ConditionLearningEngine: a combo
    is only judged after MIN_TRADES samples, and a blocked combo is
    re-admitted automatically if its rolling window recovers (stats use a
    rolling window, not lifetime totals, so one bad month doesn't ban a
    strategy forever).
    """

    MIN_TRADES = 12          # samples before a combo can be blocked
    MIN_WR     = 0.45        # rolling WR below this → blocked
    WINDOW     = 40          # rolling window per combo (recent outcomes only)

    def __init__(self):
        # {"Trend|EMA_Pullback": [1,0,1,...]}  (1=win, most recent last)
        self.outcomes: Dict[str, List[int]] = {}

    @staticmethod
    def _key(regime: str, strategy: str) -> str:
        return f"{regime}|{strategy}"

    def record(self, regime: str, strategy: str, win: bool) -> None:
        if not regime or not strategy:
            return
        k = self._key(regime, strategy)
        lst = self.outcomes.setdefault(k, [])
        lst.append(1 if win else 0)
        if len(lst) > self.WINDOW:
            del lst[:-self.WINDOW]

    def is_blocked(self, regime: str, strategy: str) -> bool:
        lst = self.outcomes.get(self._key(regime, strategy))
        # Range is a lower-frequency, regime-transition-sensitive engine, so
        # demand more evidence and a higher live WR before continuing to risk
        # capital. This does not alter Trend/Breakout expectancy rules.
        min_trades = 15 if regime == "Range" else self.MIN_TRADES
        min_wr = 0.55 if regime == "Range" else self.MIN_WR
        if not lst or len(lst) < min_trades:
            return False
        return (sum(lst) / len(lst)) < min_wr

    def get_summary(self) -> Dict:
        out = {}
        for k, lst in self.outcomes.items():
            if not lst:
                continue
            regime, strategy = k.split("|", 1)
            out[k] = {
                "trades":  len(lst),
                "wr":      round(sum(lst) / len(lst), 3),
                "blocked": self.is_blocked(regime, strategy),
            }
        return out

    def to_dict(self) -> Dict:
        return {"outcomes": self.outcomes}

    def from_dict(self, data: Dict):
        """
        Merge (not overwrite): this engine instance is shared across every
        symbol's bot, and each bot calls load_state() -> from_dict()
        independently at startup with ITS OWN symbol's saved snapshot. A
        plain overwrite would let whichever symbol loads last discard the
        history every other symbol had already restored. Keep the longer
        list per key (a reasonable proxy for "more complete rolling
        history") and merge in any keys unique to the incoming snapshot.
        """
        incoming = data.get("outcomes", {})
        for k, v in incoming.items():
            if k not in self.outcomes or len(v) > len(self.outcomes[k]):
                self.outcomes[k] = v


# ══════════════════════════════════════════════════════════════════════════════
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
# [LEVEL 2 — ADAPTIVE SCORING] Condition-tag learning
# PatternLearningEngine (above) tracks win-rate per ENTRY_TYPE (a handful of
# broad buckets); this tracks win-rate per DIAGNOSTIC TAG — the specific
# thing that was off about a losing entry (overextended, low volume, ...).
# A candidate resembling a historically bad pattern gets a score penalty
# even the first time its entry_type is tried, because the tag itself
# already has a track record. Same "no ML, deterministic threshold" design
# as PatternLearningEngine, on purpose — auditable and hard to overfit.
# ──────────────────────────────────────────────────────────────────────────────

class ConditionLearningEngine:
    """Per-diagnostic-tag win-rate tracking, feeding a scoring penalty."""

    TAGS = ["overextended", "rsi_extreme", "momentum_climax", "low_volume",
            "choppy", "trend_to_range",
            # [SESSION EXPERT] learned, not assumed — see _session_label
            "session_asia", "session_london", "session_overlap",
            "session_ny", "session_offhours"]
    MIN_SAMPLES = 8       # minimum tagged trades before a tag affects scoring
    MAX_PENALTY = 20.0    # score points subtracted from `total`, at worst

    def __init__(self):
        self.stats: Dict[str, Dict] = {
            tag: {"wins": 0, "losses": 0} for tag in self.TAGS
        }

    def record(self, tags: List[str], win: bool) -> None:
        for tag in tags:
            s = self.stats.setdefault(tag, {"wins": 0, "losses": 0})
            if win:
                s["wins"] += 1
            else:
                s["losses"] += 1

    def get_penalty(self, tags: List[str]) -> float:
        """Score penalty for a candidate carrying these tags. Untested or
        low-sample tags contribute 0 — no penalty until there's evidence."""
        penalty = 0.0
        for tag in tags:
            s = self.stats.get(tag)
            if not s:
                continue
            total = s["wins"] + s["losses"]
            if total < self.MIN_SAMPLES:
                continue
            wr = s["wins"] / total
            if wr < 0.45:
                penalty += self.MAX_PENALTY * (0.45 - wr) / 0.45
        return min(penalty, self.MAX_PENALTY)

    def get_summary(self) -> Dict:
        out = {}
        for tag, s in self.stats.items():
            total = s["wins"] + s["losses"]
            if total == 0:
                continue
            out[tag] = {
                "trades":    total,
                "win_rate":  round(s["wins"] / total, 3),
                "penalty":   round(self.get_penalty([tag]), 1),
            }
        return out

    def to_dict(self) -> Dict:
        return {"stats": self.stats}

    def from_dict(self, data: Dict):
        self.stats = data.get("stats", self.stats)


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
                 # [V9.2] 75% banked at T1 (was 50%): the majority of the
                 # position locks in at the high-probability first target,
                 # leaving a 25% runner for T2.
                 tp1_close_pct: float = 0.75,
                 tp1_r: Optional[float] = None,
                 tp2_r: Optional[float] = None,
                 breakeven_lock_r: Optional[float] = None,
                 min_ema_dist_atr: Optional[float] = None,
                 entry_spacing_min: int = 60,
                 margin_usdt: float = 0.0,
                 margin_pct_min: float = 0.0,
                 margin_pct_max: float = 0.0,
                 sizing_leverage: int = 10,
                 state_file: Optional[str] = None,
                 execution_callback: Optional[Callable] = None,
                 # 45 minutes could repeatedly suppress entries after Railway
                 # redeploys/restarts. 15 minutes is enough to stabilize live feeds
                 # because indicator history is already loaded before on_tick.
                 startup_warmup_minutes: int = 15,
                 enable_swing_reversal: bool = True,
                 enable_mean_reversion: bool = False,
                 expectancy_engine: Optional["ExpectancyEngine"] = None,
                 # [MTF-CONFLUENCE] Alternate, fully deterministic entry engine
                 # (see mtf_confluence_engine.py) selectable alongside the
                 # default V9.2 L1/L2/L3/StrategyScorer pipeline — swap via
                 # ADAPTIVE_ENTRY_ENGINE=mtf_confluence in run_bot.py. Exit
                 # management (T1/T2/SL/post-T1 protection) is unchanged
                 # either way; this only replaces how direction+timing is
                 # decided. "adaptive" = existing V9.2 pipeline (default).
                 entry_engine: str = "adaptive",
                 # [EARLY TREND] see compute_early_trend/MacroTrendEngine —
                 # UNVALIDATED pending backtest; default off so nothing
                 # changes for existing deployments until proven.
                 enable_early_trend: bool = False,
                 # [FAST MACRO EMA] Override L1's EMA20/50 cross/distance
                 # component with a faster pair — see MacroTrendEngine.compute()
                 # 's ema_fast_override/ema_slow_override. 12/26 (the classic
                 # MACD periods) is the DEFAULT as of this writing: backtested
                 # on real Jan-Jun 2026 data at +$209 combined vs EMA20/50,
                 # improving both the in-sample AND out-of-sample splits
                 # independently. Pass None/None to restore the old EMA20/50
                 # behavior (IndicatorEngine's shared values, untouched).
                 macro_ema_fast: Optional[int] = 12,
                 macro_ema_slow: Optional[int] = 26,
                 # [MIN-SL FLOOR] SL distance is never allowed to be tighter
                 # than this fraction of entry price, regardless of what the
                 # ATR-based calc in _compute_sl_price returned — guards
                 # against an artificially-tiny ATR reading producing an
                 # oversized position. 0.020 (2%) was a single global value
                 # for every symbol; low-volatility instruments (gold/silver/
                 # oil) rarely move 2% intra-trade, so it was routinely
                 # clamping their SL far wider than their own ATR called for.
                 # Set per-symbol in run_bot.py. Tried crypto=0.012/
                 # commodity=0.008 (gold/silver/oil rarely move 2% intra-
                 # trade) but 0.8% backtested WORSE for XAU/XAG than the old
                 # uniform 2% (-$470 combined, every loss a full -1R stop) —
                 # too tight for gold's actual ATR. Both buckets now share
                 # 0.012 (the value that DID backtest better, +$343 on crypto).
                 min_sl_pct: float = 0.020,
                 # [PULLBACK FILTER] Max 15m ADX for a Trend entry — above
                 # this, the leg is treated as already-extended (chasing) and
                 # vetoed. None = use the class default (MAX_15M_ADX_TREND).
                 # This is the single biggest entry blocker (41-55% of all
                 # Trend-regime checks in a 14-day live-config replay), so it's
                 # the main knob for trade frequency vs entry quality.
                 max_15m_adx_trend: Optional[float] = None,
                 # [RANGE ADD-ON] Range is intentionally half-risk and has a
                 # dedicated cooldown that never pauses Trend/Breakout.
                 range_risk_multiplier: float = 0.50,
                 range_cooldown_minutes: int = 90):
        self.state: str = "SCANNING"
        self.entry_engine       = entry_engine
        self.enable_early_trend = enable_early_trend
        self.macro_ema_fast     = macro_ema_fast
        self.macro_ema_slow     = macro_ema_slow
        self.min_sl_pct         = min_sl_pct
        # Shadow the class constant on this instance when an override is
        # given — every `self.MAX_15M_ADX_TREND` read (the veto comparison
        # and its log string) then picks it up automatically.
        if max_15m_adx_trend is not None:
            self.MAX_15M_ADX_TREND = float(max_15m_adx_trend)
        self.range_risk_multiplier = float(np.clip(range_risk_multiplier, 0.10, 1.00))
        self.range_cooldown_minutes = max(int(range_cooldown_minutes), 0)
        # Runner updates this before each tick to enforce a portfolio-wide
        # MAX_RANGE_POSITIONS without touching Trend/Breakout slots.
        self.range_entry_allowed: bool = True
        self.range_cooldown_until: Optional[datetime.datetime] = None
        self.range_loss_streak: int = 0
        self.mtf_engine         = MTFConfluenceEngine()
        self.adaptive_engine    = AdaptiveEngine()
        self.health_calc        = PositionHealthCalculator()
        self.macro_engine       = MacroTrendEngine()
        self.context_engine     = ContextBiasEngine()
        self.regime_clf         = RegimeClassifier()
        self.strategy_scorer    = StrategyScorer()
        self.confidence_engine  = ConfidenceEngine()
        # [SHARED-LEARNING] A single symbol rarely sees 12+ occurrences of one
        # narrow (regime, strategy) combo on its own — pass one ExpectancyEngine
        # instance shared across every symbol's bot (see run_bot.py /
        # backtest_engine.py) so the MIN_TRADES threshold is reached from
        # pooled cross-symbol history instead of each bot learning in
        # isolation. Defaults to a private instance when not wired up (e.g.
        # ad-hoc scripts/tests).
        self.expectancy_engine  = expectancy_engine if expectancy_engine is not None else ExpectancyEngine()
        self.learning_engine    = PatternLearningEngine()
        # [LEVEL 2/3] Diagnostic-tag learning + temporary strategy tightening
        # (see ConditionLearningEngine, _diagnose_conditions, _check_strategy_dominance)
        self.condition_engine  = ConditionLearningEngine()
        self._active_strategy_adjustments: Dict[str, datetime.datetime] = {}
        self.tp1_close_pct     = tp1_close_pct
        # TP geometry — instance overrides of the class defaults (env-tunable
        # WR↔profit dial: lower TP1_R = higher win-rate, smaller avg win).
        if tp1_r is not None:
            self.TP1_R = tp1_r
        if tp2_r is not None:
            self.TP2_R = tp2_r
        if breakeven_lock_r is not None:
            self.BREAKEVEN_LOCK_R = breakeven_lock_r
        # Fake-signal chop-zone filter (env-tunable): higher = stricter, more
        # WR, fewer trades. 0.8 default; ~1.2 pushes WR toward 56%.
        if min_ema_dist_atr is not None:
            self.MIN_EMA_DIST_ATR = min_ema_dist_atr

        # [WHIPSAW GUARD] Per-symbol entry spacing: after CLOSING a position,
        # no NEW entry on this symbol until entry_spacing_min minutes have
        # passed since that close — prevents immediate re-entry chasing right
        # after getting stopped/taken out while price whips up/down (each bot
        # instance == one symbol). Stamped in _close_position, not on open.
        self.entry_spacing_min = entry_spacing_min
        self._last_close_at: Optional[datetime.datetime] = None

        # [SIZING] Three modes, checked in this precedence order in
        # _step5_risk_engine: (1) margin_pct_min/max > 0 → [LEVEL 1] dynamic
        # %-of-balance sizing, the bot's own conviction (score headroom above
        # this state's bar, penalized by any historically-bad condition tags)
        # decides where in [min, max] this trade's size falls; (2) margin_usdt
        # > 0 → legacy fixed-$ notional (kept for explicit override); (3)
        # neither set → classic risk-% sizing (backtest default).
        self.margin_usdt     = margin_usdt
        self.margin_pct_min  = margin_pct_min
        self.margin_pct_max  = margin_pct_max
        self.sizing_leverage = sizing_leverage

        # [SCAN-INFO] last per-direction signal evaluation, for the runner's
        # 5-min scan log (why we are / aren't trading right now)
        self._scan_info: Dict[str, str] = {}

        # [MTF-CONFLUENCE] raw OHLCV lists from the most recent on_tick call
        # (see on_tick's raw_candles param) — None until the first tick.
        self._raw_candles: Optional[Dict[str, List]] = None

        # [BREAKOUT ADD-ON] Pending retest-breakout: when a Normal (4/6) or
        # macro-transition breakout fires, we don't enter on the break bar —
        # we remember it here and wait 1-3 bars for a retest+reclaim of the
        # broken level. In-memory only (a restart safely abandons any pending
        # retest — the window is just 1-3 bars). None when nothing is pending.
        self._pending_breakout: Optional[Dict] = None

        # [SESSION CONTROL] set by run_bot.py from the shared TradingSessionEngine
        # BEFORE each on_tick call (see session_engine.py) — gates only the
        # SCANNING->FILTERING transition (_check_global_gates), never position
        # management. Defaults open so ad-hoc scripts/backtests without a
        # session engine wired up behave exactly as before this feature.
        self.session_gate_open: bool = True
        self.session_state: str = "ACTIVE"

        # [TARGET ALERTS] queued Telegram-ready dicts for each target hit /
        # SL ratchet move, popped by the runner after on_tick / intrabar checks
        self._pending_target_alerts: List[Dict] = []

        # [PROTECT-LAYER SWITCH] see _manage_open_position. Env-tunable so a
        # backtest can measure pure T1/T2/SL outcomes (ADAPTIVE_PROTECT_EXITS=0)
        # without touching live behavior (default on).
        self.enable_protect_exits: bool = \
            os.environ.get("ADAPTIVE_PROTECT_EXITS", "1") != "0"

        # [MIN-LOT TP1] Smallest close size (in coins) the exchange can fill —
        # set by the runner from the symbol's contract size. When a TP1 partial
        # would be below this (e.g. a 1-contract position can't close 50%),
        # TP1 becomes a breakeven-move only: SL -> entry, full size rides to
        # TP2. 0.0 = disabled (backtest keeps fractional fills).
        self.min_close_size: float = 0.0

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
        self.current_market_state: str              = "Range"
        self.current_regime_bias: str               = "NEUTRAL"
        self.regime_score: float                    = 50.0
        self._l1_cache: Dict                        = {}
        self._l2_cache: Dict                        = {}
        self._l3_cache: Dict                        = {}
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

        # Strategy instances — unified pipeline uses _mr_strategy's step methods
        # for mean-revert-state entry scoring regardless of either flag; the
        # flags now only gate whether entries happen at all (legacy on/off knobs).
        self.enable_swing_reversal  = enable_swing_reversal
        self.enable_mean_reversion  = enable_mean_reversion
        self._entries_enabled       = enable_swing_reversal or enable_mean_reversion
        self._mr_strategy           = MeanReversionStrategy()
        self._pending_signal: Optional[Dict] = None   # signal from unified engine pending order

        # Last computed scores (for health report / logging)
        self._last_entry_health: float    = 0.0
        self._last_confidence: float      = 0.0
        self._last_confidence_level: str  = "SKIP"

        # Rejection-reason tally for _generate_signal (INFO-level periodic summary,
        # so the dominant blocking component is visible without needing LOG_LEVEL=DEBUG)
        self._filter_stats: Dict[str, int] = {
            "checked": 0, "passed": 0,
            "veto_chop": 0, "veto_climax": 0, "veto_1h_chop": 0,
            "veto_chase": 0, "veto_macro": 0,
            "strategy_fail": 0, "threshold_fail": 0,
        }

        # [LESSON] loss-cluster alerting state
        self._pending_lesson: Optional[str] = None
        self._lesson_alerted_at: int = 0   # journal length when last alert fired

        # [LEVEL 3] activation/expiry notifications — separate from the
        # lesson alert (which has its own anti-spam gate that could suppress
        # a lesson alert on the exact loss that also activates a temporary
        # tightening) so this is always visible regardless of that timing.
        self._pending_strategy_alerts: List[str] = []

    def get_filter_stats(self) -> Dict[str, int]:
        """Return and reset the rejection-reason tally since last call."""
        stats = dict(self._filter_stats)
        for k in self._filter_stats:
            self._filter_stats[k] = 0
        return stats

    # ── [V9] New helper methods ───────────────────────────────────────────────

    def _compute_l1_fit(self, l1: Dict, direction: str, regime: str) -> float:
        """0-100 fit of this direction vs L1 macro score.
        Counter-trend regimes (Reversal/Exhaustion) invert the lean."""
        lean = (l1["score"] - 50.0) * (1 if direction == "LONG" else -1)
        if regime in _COUNTER_REGIMES:
            lean = -lean
        return float(np.clip(50.0 + lean, 0, 100))

    def _compute_sl_price(self, ind_15m: Dict, candle_15m: Dict,
                          direction: str, regime: str) -> float:
        """SL price: MR regimes use MR strategy method; Trend/Breakout use ATR-based."""
        if regime in _MR_REGIMES:
            sl_price, _ = self._mr_strategy._step14_sl(ind_15m, candle_15m, direction)
            return sl_price
        entry = float(candle_15m.get("close", ind_15m.get("close", 0.0)))
        atr   = max(ind_15m.get("atr", entry * 0.01), 1e-9)
        mult  = 1 if direction == "LONG" else -1
        return entry - atr * 1.5 * mult

    # ── [LESSON] loss-cluster detection & post-mortem ────────────────────────

    LESSON_WINDOW = 5          # look-back window (trades)
    LESSON_MIN_LOSSES = 3      # losses within window that trigger an alert

    def _check_lessons(self) -> None:
        """
        After every closed trade: if the last 3 trades were all losses, or
        >=LESSON_MIN_LOSSES of the last LESSON_WINDOW were losses, build a
        post-mortem (which logic entered, which logic exited, scores, R) and
        queue it for the runner to push to Telegram. The learning engine's
        per-entry-type weights (the bot's own self-adjustment) are included
        so the alert shows what the bot is already doing about it.
        """
        j = self.trade_journal
        if len(j) < 3 or len(j) <= self._lesson_alerted_at:
            return
        # Anti-spam: only re-evaluate on a fresh LOSS, and require at least
        # 2 new trades since the previous alert (otherwise a lingering
        # 3-losses-in-window condition would alert on every close).
        if j[-1].get("win_loss") != "LOSS":
            return
        if self._lesson_alerted_at and len(j) - self._lesson_alerted_at < 2:
            return

        streak3 = all(t.get("win_loss") == "LOSS" for t in j[-3:])
        window  = j[-self.LESSON_WINDOW:]
        losses  = [t for t in window if t.get("win_loss") == "LOSS"]
        if not (streak3 or len(losses) >= self.LESSON_MIN_LOSSES):
            return

        trigger = "3 losses in a row" if streak3 else \
                  f"{len(losses)} losses in last {len(window)} trades"
        lines = [f"📚 LESSON ALERT — {trigger}", ""]
        for t in losses[-3:]:
            # journal fields exist but may hold None (adopted/legacy trades) —
            # `t.get(k, 0)` does NOT default those, so coerce with `or 0`.
            _n = lambda k: float(t.get(k) or 0)
            targets_hit = t.get("targets_hit") or []
            reached = ",".join(targets_hit) if targets_hit else "none"
            tags = t.get("loss_tags") or []
            lines.append(
                f"• {t.get('direction','?')} {t.get('e_state') or t.get('market_state','?')}"
                f"/{t.get('entry_type','?')} → exit {t.get('exit_reason','?')} "
                f"({_n('realized_r'):+.2f}R) | targets reached: {reached}\n"
                f"  entry scores: sig={_n('e_total'):.0f} "
                f"e={_n('e_entry'):.0f} ctx={_n('e_context'):.0f} "
                f"fit={_n('e_fit'):.0f} | rsi={_n('e_rsi'):.0f} "
                f"adx={_n('e_adx'):.0f}"
                + (f"\n  likely cause: {', '.join(tags)}" if tags else "")
            )
        # exit-reason tally over the window (what's killing us)
        from collections import Counter as _Counter
        reasons = _Counter(t.get("exit_reason", "?") for t in losses)
        lines.append("")
        lines.append("exit reasons: " + ", ".join(f"{k}×{v}" for k, v in reasons.most_common()))
        # [LEVEL 0] which diagnostic tags dominate this window's losses
        tag_counts = _Counter(tag for t in losses for tag in (t.get("loss_tags") or []))
        if tag_counts:
            lines.append("likely causes: " + ", ".join(f"{k}×{v}" for k, v in tag_counts.most_common()))
        # bot's own self-adjustment (learning engine + condition engine)
        try:
            lines.append(f"auto-adjust weights: {self.learning_engine.get_summary()}")
        except Exception:
            pass
        try:
            cond_summary = self.condition_engine.get_summary()
            if cond_summary:
                lines.append(f"condition-tag stats: {cond_summary}")
        except Exception:
            pass
        if self._active_strategy_adjustments:
            lines.append(
                "⚠ active temporary tightening: " +
                ", ".join(self._active_strategy_adjustments.keys())
            )
        lines.append("(weights auto-reduce sizing on entry types with WR<45%; "
                     "condition tags with WR<45% get an entry-score penalty)")

        self._pending_lesson    = "\n".join(lines)
        self._lesson_alerted_at = len(j)

    def pop_lesson_alert(self) -> Optional[str]:
        """Return and clear the queued lesson alert (runner forwards to Telegram)."""
        msg = self._pending_lesson
        self._pending_lesson = None
        return msg

    def pop_strategy_alerts(self) -> List[str]:
        """Return and clear queued [LEVEL 3] activation/expiry notifications."""
        alerts = self._pending_strategy_alerts
        self._pending_strategy_alerts = []
        return alerts

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_hold_secs(entry_time) -> float:
        """Hold time in seconds, tolerant of naive/aware/None entry_time —
        a mixed naive/aware subtraction raises TypeError, which previously
        crash-looped the EXITING state every bar."""
        if not isinstance(entry_time, datetime.datetime):
            return 0.0
        try:
            now = (datetime.datetime.now(datetime.timezone.utc)
                   if entry_time.tzinfo else datetime.datetime.now())
            return (now - entry_time).total_seconds()
        except Exception:
            return 0.0

    def _log_event(self, msg: str, level: str = "info"):
        log_fn = getattr(logger, level, logger.info)
        log_fn(msg)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.append(f"[{ts}] {msg}")
        if len(self._log) > 500:          # cap memory: only last 20 are ever shown
            del self._log[:-250]

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

    def _send_amend_sl(self, new_sl: float) -> None:
        """
        Push a ladder SL-ratchet (T1-T3) to the real exchange-attached stop
        order (OKX AMEND_SL), so the live SL tracks the bot's local one
        instead of staying at the original wider level for the whole trade.
        Deliberately NOT routed through _send_order: a failed amend is
        non-fatal here (the position stays protected by whichever SL is
        currently live on the exchange either way, and local
        check_price_protection/_manage_open_position remain the fallback) —
        unlike a failed OPEN/CLOSE, it must never flip the bot to ERROR.
        No-ops silently when there's no known algo_id (paper/backtest mode,
        or the id lookup failed at entry).
        """
        if not self.execution_callback or not self.current_trade:
            return
        algo_id = self.current_trade.get("sl_algo_id")
        if not algo_id:
            # Visible-but-non-fatal: the Telegram "SL moved" alert still fires
            # from _queue_target_alert regardless of this — without this log
            # there is NO signal anywhere that the real exchange SL was never
            # touched (e.g. after a reconcile-adopted position without a
            # recovered algo_id).
            self._log_event(
                f"[WARN] SL amend SKIPPED (no algo_id known) — local SL moved "
                f"to {new_sl:.4f} but the exchange stop was NOT amended",
                level="warning",
            )
            return
        try:
            self.execution_callback("AMEND_SL", {
                "sl_algo_id": algo_id,
                "new_sl":     new_sl,
            })
            self._log_event(f"[OKX] SL amend requested → {new_sl:.4f}")
        except Exception as e:
            self._log_event(
                f"[WARN] SL amend failed (exchange keeps its prior SL; "
                f"local price-protection remains the fallback): {e}",
                level="warning",
            )

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

    # ══════════════════════════════════════════════════════════════════════════
    # [V9] MULTI-LAYER SIGNAL PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    # TP geometry in R-multiples — TP1_R/TP2_R remain the env-tunable
    # endpoints. Unified 2-target structure (user-designed): T1 takes a
    # partial (self.tp1_close_pct, constructor-configurable — see __init__)
    # and moves the stop to BREAKEVEN_LOCK_R (locks in a small win instead of
    # a flat scratch at exactly entry), T2 closes what's left (matches the
    # exchange-attached TP2, unchanged).
    BREAKEVEN_LOCK_R: float = 0.15
    # TP geometry — widened from 0.5R/1.0R to 0.6R/1.2R by request. Bar-close
    # A/B backtest (8 symbols, Jan–Jul 2026, same tp1_close_pct 60% /
    # breakeven-lock 0.15 / min-SL 1.2%): aggregate PnL improved by the
    # equivalent of +$4.3k (mainly XAU +$3.5k, XRP +$1.7k; SOL/XAG/CL a bit
    # worse) at the cost of ~3pp win-rate (66.8% → 63.4%) — targets sit
    # further out so fewer trades reach T1/T2, but each banked leg is bigger.
    TP1_R: float = 0.6
    TP2_R: float = 1.2

    def _target_ladder(self) -> List[tuple]:
        """
        (trigger_R, close_pct, new_SL_R) triples, walked in order by
        _check_targets. close_pct is the fraction of the CURRENT remaining
        size closed at that level (1.0 on the final entry — always closes
        everything left). new_SL_R=None means "don't move the SL here" (the
        final level has nothing left to protect).
        T1: TP1_R  -> close self.tp1_close_pct (default 50%), SL -> breakeven (0R).
        T2: TP2_R  -> close 100% of what's left (exchange-attached TP2).
        """
        tp1_r, tp2_r = self.TP1_R, self.TP2_R
        # [SAFETY] TP2_R must stay strictly above TP1_R — otherwise the
        # ladder's sequential trigger order breaks, and the exchange
        # -attached TP2 order (fires at TP2_R independent of local polling)
        # could close the real position before the bot's local T1 partial
        # is ever reached, leaving the bot thinking the position is still
        # fully open (and still holding a stale breakeven-move pending).
        if tp2_r <= tp1_r:
            clamped = round(tp1_r * 1.5, 4)
            self._log_event(
                f"[LADDER] TP2_R {tp2_r}R <= TP1_R {tp1_r}R — clamped to "
                f"{clamped}R (check ADAPTIVE_TP1_R/ADAPTIVE_TP2_R)",
                level="warning",
            )
            tp2_r = clamped
        close_pct = float(np.clip(self.tp1_close_pct, 0.0, 0.99))
        return [
            (tp1_r, close_pct, self.BREAKEVEN_LOCK_R),  # T1: partial close, SL -> BE+lock
            (tp2_r, 1.0,       None),  # T2: full close of what's left
        ]

    # [FAKE-FILTER] Minimum price-to-EMA20 distance (in ATR) for a trend-state
    # entry. Below this, price is in the chop-zone with near-zero edge. Swept.
    # [V9.3 ACTIVE] 0.60 ATR rejected many valid pullback/reclaim entries
    # near EMA20. Keep a small anti-chop buffer instead of requiring extension.
    MIN_EMA_DIST_ATR: float = 0.25

    # ── [V9.1 QUALITY GATES] Backtest-proven filters (2,972 trades) ──────────
    # Trend entries with 15m ADX already elevated are LATE (chasing an
    # extended leg): ADX≤22 quartile ran 68.9% WR vs 47-49% above 30.
    # Enter the pullback/quiet phase of a 4H trend, not the climax.
    # [V9.3 ACTIVE] RegimeClassifier labels Trend at ADX >20, therefore
    # the old <=22 gate left an impractically narrow 20-22 entry window.
    # <=32 still avoids late climax legs but permits normal developing trends.
    MAX_15M_ADX_TREND: float = 32.0

    # Trend-direction RSI chase guard: LONG into overbought / SHORT into
    # oversold on the 15m entry bar = buying the top of the leg.
    TREND_RSI_CHASE_HI: float = 70.0
    TREND_RSI_CHASE_LO: float = 30.0

    # Asia session (00-05 UTC) ran 37-45% WR across every regime — thin
    # liquidity whipsaw. Hard-gated rather than left to the session-tag
    # learner (which needs 8+ samples per tag to react).
    # [V9.3 ACTIVE] Do not globally block Asia hours: this class is used
    # by both crypto and commodities, and crypto remains liquid 24/7. Commodity
    # market-hours protection stays in session_engine via session_gate_open.
    BLOCKED_ENTRY_HOURS_UTC: frozenset = frozenset()

    # [CLIMAX-VETO] Skip trend entries on bars with range > this × ATR
    # (vertical blow-off spikes). 99 = disabled.
    CLIMAX_BAR_ATR: float = 2.5

    # [1H CHOP-FILTER] Minimum Kaufman efficiency ratio (0=pure noise,
    # 1=perfectly smooth trend) the 1H timeframe itself must show. A choppy
    # 1H means the "confirming" timeframe isn't actually trending, no matter
    # how the EMA/RSI/momentum snapshot reads at this instant. MR states are
    # exempt (they deliberately trade mean-reversion in choppy/exhausted
    # markets). 0 = disabled.
    MIN_1H_EFFICIENCY: float = 0.15

    # [DIAGNOSTIC TAGS] Rule-based thresholds for "what was off about this
    # entry" — deliberately simple/auditable (no ML), built entirely from
    # features already scored at entry time (e_ema_dist_atr, e_rsi,
    # e_atr_exp, e_vol_ratio, e_adx). Feeds the lesson alert (Level 0),
    # ConditionLearningEngine's scoring penalty (Level 2), and the temporary
    # strategy tightening when one tag dominates recent losses (Level 3).
    TAG_OVEREXTENDED_ATR    = 1.0    # e_ema_dist_atr >= this = entered far from EMA20
    TAG_RSI_HIGH            = 68.0   # LONG entries at/above this RSI = already extended
    TAG_RSI_LOW             = 32.0   # SHORT entries at/below this RSI = already extended
    TAG_MOMENTUM_CLIMAX_EXP = 1.4    # e_atr_exp >= this = entered on an expansion/climax bar
    TAG_LOW_VOLUME          = 0.85   # e_vol_ratio <= this = below-average volume at entry
    TAG_CHOPPY_ADX          = 20.0   # e_adx <= this = weak/no trend at entry

    # [LEVEL 3 — ADAPTIVE STRATEGY] When a single tag dominates recent losses,
    # temporarily add extra scoring penalty for candidates carrying it — a
    # bounded, self-expiring, ONE-DIRECTION-ONLY tightening (never loosens
    # anything). Reuses ConditionLearningEngine's penalty mechanism rather
    # than a separate rule path, so there's one scoring-penalty code path
    # to reason about, not two.
    STRATEGY_DOMINANCE_COUNT = 2      # tag must appear in >= this many of...
    STRATEGY_LOOKBACK_LOSSES = 3      # ...the last N losses to trigger
    STRATEGY_EXTRA_PENALTY   = 15.0   # additional score penalty while active
    STRATEGY_DURATION_MIN    = 360    # auto-expire after 6 hours

    @staticmethod
    def _session_label(now: datetime.datetime) -> str:
        """
        [SESSION EXPERT] UTC-hour session bucket. Deliberately NOT scored
        with assumed session quality (crypto liquidity/session behavior
        varies by symbol and isn't ours to guess) — instead fed as a tag
        into ConditionLearningEngine so it's LEARNED from this bot's own
        trade history which sessions actually work (or don't) per Level 9:
        "BTC London Trend Score>87 -> Win 78%, but Asia Range -> Win 45%".
        """
        h = now.hour
        if 0 <= h < 8:    return "session_asia"
        if 8 <= h < 13:   return "session_london"
        if 13 <= h < 16:  return "session_overlap"
        if 16 <= h < 21:  return "session_ny"
        return "session_offhours"

    def _diagnose_conditions(self, direction: str, ema_dist_atr: float, rsi: float,
                             atr_exp: float, vol_ratio: float, adx: float,
                             now: Optional[datetime.datetime] = None) -> List[str]:
        """
        Rule-based tags for "what was off about this entry", from features
        already computed at entry-scoring time. Used two ways: (1) post-hoc
        on a closed losing trade (via its stored e_* fields) to explain why
        in the lesson alert and train ConditionLearningEngine; (2) pre-entry
        on a live candidate (via current indicator values) to check whether
        it resembles a historically bad pattern before it's ever taken.
        """
        tags: List[str] = []
        if ema_dist_atr >= self.TAG_OVEREXTENDED_ATR:
            tags.append("overextended")
        if direction == "LONG" and rsi >= self.TAG_RSI_HIGH:
            tags.append("rsi_extreme")
        elif direction == "SHORT" and rsi <= self.TAG_RSI_LOW:
            tags.append("rsi_extreme")
        if atr_exp >= self.TAG_MOMENTUM_CLIMAX_EXP:
            tags.append("momentum_climax")
        if vol_ratio <= self.TAG_LOW_VOLUME:
            tags.append("low_volume")
        if adx <= self.TAG_CHOPPY_ADX:
            tags.append("choppy")
        if now is not None:
            tags.append(self._session_label(now))
        return tags

    def _diagnose_loss(self, t: Dict, close_reason: str) -> List[str]:
        """Post-hoc: tag a closed trade's condition profile from its stored
        entry-time features — called for wins too (ConditionLearningEngine's
        win-rate per tag needs both outcomes), despite the name. STATE_DRIFT_EXIT
        (the market regime changed mid-trade) maps directly to the
        'trend_to_range' tag — that exit reason already means exactly this,
        no threshold needed."""
        entry_time = t.get("entry_time")
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.datetime.fromisoformat(entry_time)
            except ValueError:
                entry_time = None
        tags = self._diagnose_conditions(
            direction    = t.get("direction", "LONG"),
            ema_dist_atr = t.get("e_ema_dist_atr") or 0.0,
            rsi          = t.get("e_rsi") or 50.0,
            atr_exp      = t.get("e_atr_exp") or 1.0,
            vol_ratio    = t.get("e_vol_ratio") or 1.0,
            adx          = t.get("e_adx") or 25.0,
            now          = entry_time if isinstance(entry_time, datetime.datetime) else None,
        )
        if close_reason == "STATE_DRIFT_EXIT" and "trend_to_range" not in tags:
            tags.append("trend_to_range")
        return tags

    def _active_strategy_penalty(self, tags: List[str], now: datetime.datetime) -> float:
        """[LEVEL 3] Extra penalty for tags currently under temporary
        tightening (dominated recent losses). Opportunistically drops
        expired entries so the dict doesn't grow unbounded."""
        expired = [tag for tag, until in self._active_strategy_adjustments.items() if now >= until]
        for tag in expired:
            del self._active_strategy_adjustments[tag]
            msg = f"[STRATEGY] temporary tightening on '{tag}' expired — reverted to normal"
            self._log_event(msg)
            self._pending_strategy_alerts.append(f"🔄 Adaptive Strategy\n{msg}")
        return sum(self.STRATEGY_EXTRA_PENALTY for tag in tags
                   if tag in self._active_strategy_adjustments)

    def _check_strategy_dominance(self) -> None:
        """
        [LEVEL 3] After a loss: if one diagnostic tag dominates the last
        STRATEGY_LOOKBACK_LOSSES losses, activate (or refresh) a temporary
        extra scoring penalty for that tag. Self-expires after
        STRATEGY_DURATION_MIN — this only ever makes entry MORE cautious for
        a bounded window, never loosens anything, and always reverts.
        """
        losses = [t for t in self.trade_journal if t.get("win_loss") == "LOSS"]
        recent = losses[-self.STRATEGY_LOOKBACK_LOSSES:]
        if len(recent) < self.STRATEGY_LOOKBACK_LOSSES:
            return
        from collections import Counter as _Counter
        # Session tags are excluded here on purpose: consecutive journal
        # entries are often temporally close (same trading day), so
        # "3 losses in the same session" is common by mere clustering, not
        # necessarily a real causal pattern. Session still feeds Level 2's
        # steady-state penalty (MIN_SAMPLES=8 is a far more robust bar)
        # — just not this fast-reacting 3-loss trigger.
        tag_counts = _Counter(
            tag for t in recent for tag in (t.get("loss_tags") or [])
            if not tag.startswith("session_")
        )
        if not tag_counts:
            return
        dominant_tag, count = tag_counts.most_common(1)[0]
        if count < self.STRATEGY_DOMINANCE_COUNT:
            return
        now = self._bar_now or datetime.datetime.now(datetime.timezone.utc)
        was_active = dominant_tag in self._active_strategy_adjustments
        self._active_strategy_adjustments[dominant_tag] = now + datetime.timedelta(
            minutes=self.STRATEGY_DURATION_MIN)
        if not was_active:
            msg = (
                f"'{dominant_tag}' caused {count}/{len(recent)} recent losses "
                f"→ temporary +{self.STRATEGY_EXTRA_PENALTY:.0f} entry penalty for "
                f"{self.STRATEGY_DURATION_MIN}min"
            )
            self._log_event(f"[STRATEGY] {msg}", level="warning")
            self._pending_strategy_alerts.append(f"🧠 Adaptive Strategy activated\n{msg}")

    @staticmethod
    def _bar_value(bar: Any, key: str, default: float = 0.0) -> float:
        """Read OHLCV fields from either connector objects or plain dicts."""
        try:
            if isinstance(bar, dict):
                return float(bar.get(key, default))
            return float(getattr(bar, key, default))
        except (TypeError, ValueError):
            return float(default)

    def _generate_range_signal(self, direction: str, candle_15m: Dict,
                               ind_15m: Dict, l1: Dict, l2: Dict) -> Optional[Dict]:
        """
        Conservative Range Edge-Fade engine. It is deliberately isolated from
        the Trend and Breakout paths: those continue through the original
        StrategyScorer/composite pipeline unchanged.

        Mandatory sequence:
          mature 24-bar range -> edge 25% -> liquidity sweep -> close reclaim
          -> 3/5 confirmations -> structure R:R -> score >= 70.
        """
        threshold = REGIME_THRESHOLDS["Range"]
        now = self._bar_now or datetime.datetime.now(datetime.timezone.utc)

        if not self.range_entry_allowed:
            self._scan_info[direction] = "veto:range portfolio slot full"
            return None

        if self.range_cooldown_until:
            try:
                if now < self.range_cooldown_until:
                    remain = max(0, int((self.range_cooldown_until - now).total_seconds() / 60))
                    self._scan_info[direction] = f"veto:range cooldown ({remain}m left)"
                    return None
            except TypeError:
                # State files from old deployments may mix naive/aware times.
                self.range_cooldown_until = None

        raw = self._raw_candles or {}
        bars = list(raw.get("15m") or [])
        lookback = 24
        if len(bars) < lookback + 2:
            self._scan_info[direction] = (
                f"veto:range history {len(bars)}/{lookback + 2} bars")
            return None

        history = bars[-(lookback + 1):-1]  # define the range before trigger bar
        prev = bars[-2]
        atr = max(float(ind_15m.get("atr", 0.0) or 0.0), 1e-9)
        if atr <= 1e-8:
            self._scan_info[direction] = "veto:range ATR unavailable"
            return None

        o = float(candle_15m.get("open", self._bar_value(bars[-1], "open")))
        h = float(candle_15m.get("high", self._bar_value(bars[-1], "high")))
        lo_cur = float(candle_15m.get("low", self._bar_value(bars[-1], "low")))
        close = float(candle_15m.get("close", self._bar_value(bars[-1], "close")))
        prev_h = self._bar_value(prev, "high")
        prev_l = self._bar_value(prev, "low")

        range_high = max(self._bar_value(b, "high") for b in history)
        range_low = min(self._bar_value(b, "low") for b in history)
        width = range_high - range_low
        width_atr = width / atr
        if width <= 0 or not (1.5 <= width_atr <= 7.0):
            self._scan_info[direction] = (
                f"veto:range width {width_atr:.2f} ATR (need 1.5-7.0)")
            return None

        touch_tol = 0.22 * atr
        upper_touches = sum(
            1 for b in history
            if self._bar_value(b, "high") >= range_high - touch_tol)
        lower_touches = sum(
            1 for b in history
            if self._bar_value(b, "low") <= range_low + touch_tol)
        # [RANGE GATE — direction-aware] Require the edge we're actually
        # FADING to be well-tested (>=2 touches); the opposite edge only needs
        # to exist (>=1). The original symmetric 2/2 gate rejected ~85% of all
        # Range-regime bars in backtest (most had 1 touch on one side) — for
        # an edge-fade, only the faded edge's integrity truly matters.
        faded_touches, other_touches = (
            (lower_touches, upper_touches) if direction == "LONG"
            else (upper_touches, lower_touches))
        if faded_touches < 2 or other_touches < 1:
            self._scan_info[direction] = (
                f"veto:range touches U{upper_touches}/L{lower_touches} "
                f"(need faded>=2, other>=1)")
            return None

        adx = float(ind_15m.get("adx", 20.0))
        eff = float(ind_15m.get("eff_ratio", 0.5))
        atr_exp = float(ind_15m.get("atr_exp", 0.0) or 0.0)
        if atr_exp <= 0:
            atr_exp = atr / max(float(ind_15m.get("atr_avg", atr) or atr), 1e-9)
        volume = float(ind_15m.get("volume", 0.0) or 0.0)
        vol_avg = max(float(ind_15m.get("vol_avg", volume or 1.0) or 1.0), 1e-9)
        vol_ratio = volume / vol_avg

        if not (8.0 <= adx <= 23.0):
            self._scan_info[direction] = f"veto:range ADX {adx:.1f} (need 8-23)"
            return None
        if eff > 0.40:
            self._scan_info[direction] = f"veto:range efficiency {eff:.2f} > 0.40"
            return None
        if not (0.65 <= atr_exp <= 1.20):
            self._scan_info[direction] = (
                f"veto:range ATR expansion {atr_exp:.2f} (need 0.65-1.20)")
            return None

        candle_range = max(h - lo_cur, 1e-9)
        body = abs(close - o)
        # Transition veto: strong displacement + participation often means the
        # range is breaking, not rejecting. Never fade that condition.
        if vol_ratio > 1.40 or (body / atr > 0.90 and vol_ratio > 1.25):
            self._scan_info[direction] = (
                f"veto:breakout-transition vol={vol_ratio:.2f} body={body/atr:.2f}ATR")
            return None

        location = float(np.clip((close - range_low) / max(width, 1e-9), 0.0, 1.0))
        macro_level = str(l1.get("level", "NEUTRAL"))
        if direction == "LONG" and macro_level == "STRONG_BEAR":
            self._scan_info[direction] = "veto:range LONG against STRONG_BEAR 4H"
            return None
        if direction == "SHORT" and macro_level == "STRONG_BULL":
            self._scan_info[direction] = "veto:range SHORT against STRONG_BULL 4H"
            return None

        if direction == "LONG":
            sweep_depth = (range_low - lo_cur) / atr
            swept = 0.01 <= sweep_depth <= 0.60
            reclaimed = close >= range_low + 0.02 * atr
            at_edge = location <= 0.25
            sl_price = lo_cur - 0.15 * atr
            tp1_price = (range_low + range_high) / 2.0
            tp2_price = range_high - 0.10 * atr
        else:
            sweep_depth = (h - range_high) / atr
            swept = 0.01 <= sweep_depth <= 0.60
            reclaimed = close <= range_high - 0.02 * atr
            at_edge = location >= 0.75
            sl_price = h + 0.15 * atr
            tp1_price = (range_low + range_high) / 2.0
            tp2_price = range_low + 0.10 * atr

        if not swept:
            self._scan_info[direction] = (
                f"veto:no liquidity sweep (depth {sweep_depth:.2f} ATR; need 0.02-0.45)")
            return None
        if not reclaimed:
            self._scan_info[direction] = "veto:sweep did not close back inside range"
            return None
        if not at_edge:
            self._scan_info[direction] = (
                f"veto:range location {location:.0%} (must be outer 25%)")
            return None

        # 3 of 5 confirmation filter.
        rsi = float(ind_15m.get("rsi", 50.0))
        rsi_confirm = (20.0 <= rsi <= 43.0) if direction == "LONG" else (57.0 <= rsi <= 80.0)

        closes = [self._bar_value(b, "close") for b in bars[-50:]]
        macd_turn = False
        if len(closes) >= 35:
            try:
                _m, _sig, hist = BaseStrategy.macd(closes, 12, 26, 9)
                if len(hist) >= 2 and not (np.isnan(hist[-1]) or np.isnan(hist[-2])):
                    macd_turn = hist[-1] > hist[-2] if direction == "LONG" else hist[-1] < hist[-2]
            except Exception:
                macd_turn = False

        micro_choch = close > prev_h if direction == "LONG" else close < prev_l
        close_loc = (close - lo_cur) / candle_range
        lower_wick = min(o, close) - lo_cur
        upper_wick = h - max(o, close)
        if direction == "LONG":
            rejection = lower_wick >= max(body * 1.20, atr * 0.08) and close_loc >= 0.62
        else:
            rejection = upper_wick >= max(body * 1.20, atr * 0.08) and close_loc <= 0.38
        volume_confirm = vol_ratio >= 1.10

        confirmations = {
            "RSI": rsi_confirm,
            "MACD": macd_turn,
            "CHOCH": micro_choch,
            "REJECTION": rejection,
            "VOLUME": volume_confirm,
        }
        confirm_count = sum(1 for ok in confirmations.values() if ok)
        if confirm_count < 3:
            passed = ",".join(k for k, ok in confirmations.items() if ok) or "none"
            self._scan_info[direction] = (
                f"veto:range confirm {confirm_count}/5 (need 3/5; passed={passed})")
            return None

        risk_dist = abs(close - sl_price)
        if risk_dist <= 1e-9:
            self._scan_info[direction] = "veto:range invalid stop distance"
            return None
        dir_mult = 1.0 if direction == "LONG" else -1.0
        rr1 = (tp1_price - close) * dir_mult / risk_dist
        rr2 = (tp2_price - close) * dir_mult / risk_dist
        if rr1 < 0.80 or rr2 < 1.10 or rr2 <= rr1:
            self._scan_info[direction] = (
                f"veto:range structure room T1={rr1:.2f}R T2={rr2:.2f}R")
            return None

        strategy_scores = {
            name: self.strategy_scorer.score(name, direction, ind_15m, l1, l2, "Range")
            for name in REGIME_STRATEGIES["Range"]
            if not self.expectancy_engine.is_blocked("Range", name)
        }
        best = self.confidence_engine.select_best(strategy_scores)
        if best is None:
            self._scan_info[direction] = "strategy_fail: all Range strategies blocked"
            return None
        best_strategy, best_score = best

        width_quality = float(np.clip(100.0 - abs(width_atr - 3.5) / 2.5 * 100.0, 0, 100))
        touch_quality = float(np.clip((upper_touches + lower_touches) / 6.0 * 100.0, 0, 100))
        adx_quality = float(np.clip(100.0 - abs(adx - 15.0) / 5.0 * 100.0, 0, 100))
        eff_quality = float(np.clip((0.30 - eff) / 0.22 * 100.0, 0, 100))
        atr_quality = float(np.clip(100.0 - abs(atr_exp - 0.90) / 0.20 * 100.0, 0, 100))
        range_quality = float(np.mean([
            width_quality, touch_quality, adx_quality, eff_quality, atr_quality]))

        sweep_quality = float(np.clip(100.0 - abs(sweep_depth - 0.18) / 0.27 * 100.0, 0, 100))
        location_quality = ((0.25 - location) / 0.25 * 100.0
                            if direction == "LONG"
                            else (location - 0.75) / 0.25 * 100.0)
        edge_quality = float(np.clip((sweep_quality + location_quality) / 2.0, 0, 100))
        confirm_quality = confirm_count / 5.0 * 100.0
        rr1_q = float(np.clip((rr1 - 0.80) / 0.70 * 100.0, 0, 100))
        rr2_q = float(np.clip((rr2 - 1.10) / 1.20 * 100.0, 0, 100))
        rr_quality = (rr1_q + rr2_q) / 2.0

        aligned = ((direction == "LONG" and macro_level in ("BULL", "STRONG_BULL")) or
                   (direction == "SHORT" and macro_level in ("BEAR", "STRONG_BEAR")))
        if aligned:
            macro_quality = 100.0 if macro_level.startswith("STRONG") else 85.0
        elif macro_level == "NEUTRAL":
            macro_quality = 70.0
        else:
            macro_quality = 55.0

        total = (
            range_quality * 0.20 + edge_quality * 0.20 +
            confirm_quality * 0.25 + rr_quality * 0.20 +
            macro_quality * 0.10 + best_score * 0.05
        )
        passed_names = ",".join(k for k, ok in confirmations.items() if ok)
        self._filter_stats["checked"] += 1
        self._scan_info[direction] = (
            f"total {total:.0f}/{threshold} regime=Range strat={best_strategy}({best_score:.0f}) "
            f"edge={edge_quality:.0f} confirm={confirm_count}/5[{passed_names}] "
            f"T1={rr1:.2f}R T2={rr2:.2f}R"
            + (" → SIGNAL" if total >= threshold else "")
        )
        if total < threshold:
            self._filter_stats["threshold_fail"] += 1
            return None

        self._filter_stats["passed"] += 1
        health = range_quality * 0.40 + confirm_quality * 0.40 + best_score * 0.20
        confidence = (edge_quality + rr_quality + macro_quality) / 3.0
        return {
            "direction": direction,
            "sl_price": float(sl_price),
            "tp1_price": float(tp1_price),
            "tp2_price": float(tp2_price),
            "tp1_close_pct": 0.60,
            "risk_multiplier": self.range_risk_multiplier,
            "health_score": float(health),
            "confidence_score": float(confidence),
            "total_score": float(total),
            "entry_score": float(best_score),
            "context_score": float(confirm_quality),
            "direction_fit": float(macro_quality),
            "entry_type": "mean_revert",
            "strategy": best_strategy,
            "regime": "Range",
            "l1_score": l1.get("score", 50.0),
            "l1_level": macro_level,
            "l2_bull": l2.get("bull_score", 50.0),
            "l2_bear": l2.get("bear_score", 50.0),
            "all_strategies": strategy_scores,
            "condition_penalty": 0.0,
            "entry_tags": ["range_edge_fade", f"range_confirm_{confirm_count}of5"],
            "range_low": float(range_low),
            "range_high": float(range_high),
            "range_confirmations": confirmations,
        }

    # ── [BREAKOUT ADD-ON] Dedicated Breakout engine ──────────────────────────
    BREAKOUT_BOS_LOOKBACK   = 20      # bars defining the broken structure level
    BREAKOUT_MIN_BEYOND_ATR = 0.12    # close must clear BOS by >= this × ATR
    BREAKOUT_MIN_BODY_ATR   = 0.28    # candle body >= this × ATR
    BREAKOUT_MIN_VOL_RATIO  = 1.20    # volume >= this × its average
    BREAKOUT_RETEST_BARS    = 3       # retest must complete within this many bars
    BREAKOUT_TRANSITION_RISK = 0.60   # half-ish risk in a macro-transition zone

    def _breakout_macro_zone(self, direction: str, macro_score: float) -> str:
        """User-defined hard macro-conflict gate on the 4H macro score (0-100).
        Returns 'veto' (trade against a decisive macro — forbidden),
        'transition' (mild conflict — allowed only with retest + reduced risk),
        or 'normal' (macro supports or is neutral-favourable)."""
        if direction == "LONG":
            if macro_score <= 35:  return "veto"
            if macro_score <= 44:  return "transition"
            return "normal"
        else:  # SHORT
            if macro_score >= 65:  return "veto"
            if macro_score >= 56:  return "transition"
            return "normal"

    def _breakout_signal_dict(self, direction, entry_price, sl_price, confirms,
                              confirm_count, score, best_strategy, best_score,
                              risk_mult, macro_quality, l1, l2, entry_mode):
        """Build the unified signal dict for a confirmed breakout. TP is left
        to the standard R-ladder in _step5 (no custom tp1/tp2) so Breakout
        keeps the same 0.6R/1.2R geometry as Trend."""
        health     = float(np.clip(confirm_count / 6.0 * 100.0 * 0.6 + best_score * 0.4, 0, 100))
        confidence = float(np.clip((score + macro_quality) / 2.0, 0, 100))
        return {
            "direction": direction,
            "sl_price": float(sl_price),
            "risk_multiplier": float(risk_mult),
            "health_score": health,
            "confidence_score": confidence,
            "total_score": float(score),
            "entry_score": float(best_score),
            "context_score": float(confirm_count / 6.0 * 100.0),
            "direction_fit": float(macro_quality),
            "entry_type": "breakout",
            "strategy": best_strategy,
            "regime": "Breakout",
            "l1_score": l1.get("score", 50.0),
            "l1_level": str(l1.get("level", "NEUTRAL")),
            "l2_bull": l2.get("bull_score", 50.0),
            "l2_bear": l2.get("bear_score", 50.0),
            "condition_penalty": 0.0,
            "entry_tags": ["breakout", f"breakout_{entry_mode}",
                           f"breakout_confirm_{confirm_count}of6"],
            "breakout_mode": entry_mode,
            "breakout_confirmations": confirms,
        }

    def _generate_breakout_signal(self, direction: str, candle_15m: Dict,
                                  ind_15m: Dict, ind_1h: Dict,
                                  l1: Dict, l2: Dict) -> Optional[Dict]:
        """
        Dedicated Breakout engine — isolated from Trend/Range. Enforces:
          1) hard 4H-macro conflict gate (never break INTO a decisive opposing
             macro; a mild conflict is a 'transition' → retest + reduced risk),
          2) >=4/6 breakout confirmations (5/6 when macro is a transition),
          3) immediate entry only for a Strong (5-6/6, macro normal) break;
             a Normal (4/6) or transition break must wait for a retest+reclaim
             of the broken level within 1-3 bars,
          4) composite score >= REGIME_THRESHOLDS['Breakout'] (64).
        """
        threshold = REGIME_THRESHOLDS["Breakout"]
        atr = max(float(ind_15m.get("atr", 0.0) or 0.0), 1e-9)
        o  = float(candle_15m.get("open", 0.0))
        h  = float(candle_15m.get("high", 0.0))
        lo_cur = float(candle_15m.get("low", 0.0))
        close = float(candle_15m.get("close", 0.0))
        dir_mult = 1.0 if direction == "LONG" else -1.0

        # ── (A) process a pending retest first ───────────────────────────────
        pb = self._pending_breakout
        if pb is not None:
            if pb["direction"] != direction:
                return None  # the matching direction handles it
            waited = self._bar_count - pb["created_bar"]
            if waited > self.BREAKOUT_RETEST_BARS:
                self._pending_breakout = None
                self._scan_info[direction] = "veto:breakout retest expired (>3 bars, no reclaim)"
                return None
            lvl = pb["bos_level"]; a = pb["atr"]
            if direction == "LONG":
                retested  = lo_cur <= lvl + 0.15 * a      # dipped back to old resistance
                reclaimed = close >= lvl                  # closed back above it
            else:
                retested  = h >= lvl - 0.15 * a
                reclaimed = close <= lvl
            if retested and reclaimed:
                self._pending_breakout = None
                sl_price = (lo_cur - 0.15 * a) if direction == "LONG" else (h + 0.15 * a)
                self._filter_stats["checked"] += 1
                self._filter_stats["passed"] += 1
                self._scan_info[direction] = (
                    f"total {pb['score']:.0f}/{threshold} regime=Breakout RETEST-reclaim "
                    f"strat={pb['best_strategy']}({pb['best_score']:.0f}) "
                    f"confirm={pb['confirm_count']}/6 → SIGNAL")
                return self._breakout_signal_dict(
                    direction, close, sl_price, pb["confirms"], pb["confirm_count"],
                    pb["score"], pb["best_strategy"], pb["best_score"], pb["risk_mult"],
                    pb["macro_quality"], l1, l2, "retest")
            self._scan_info[direction] = (
                f"breakout retest wait {waited}/{self.BREAKOUT_RETEST_BARS} "
                f"(retested={retested} reclaimed={reclaimed})")
            return None

        # ── (B) hard macro-conflict gate ─────────────────────────────────────
        macro_score = float(l1.get("score", 50.0))
        macro_level = str(l1.get("level", "NEUTRAL"))
        zone = self._breakout_macro_zone(direction, macro_score)
        if zone == "veto":
            self._scan_info[direction] = (
                f"veto:breakout vs decisive 4H macro ({direction} macro={macro_score:.0f})")
            return None
        is_transition = (zone == "transition")

        # ── (C) locate the broken structure (BOS) level ──────────────────────
        raw = self._raw_candles or {}
        bars = list(raw.get("15m") or [])
        lb = self.BREAKOUT_BOS_LOOKBACK
        if len(bars) < lb + 2:
            self._scan_info[direction] = f"veto:breakout history {len(bars)}/{lb + 2} bars"
            return None
        history = bars[-(lb + 1):-1]  # structure BEFORE the current (break) bar
        if direction == "LONG":
            bos_level = max(self._bar_value(b, "high") for b in history)
            beyond = (close - bos_level)          # >0 means closed above resistance
        else:
            bos_level = min(self._bar_value(b, "low") for b in history)
            beyond = (bos_level - close)          # >0 means closed below support

        # ── (D) six confirmations ────────────────────────────────────────────
        body = abs(close - o)
        vol_ratio = (float(ind_15m.get("volume", 0.0) or 0.0)
                     / max(float(ind_15m.get("vol_avg", 1.0) or 1.0), 1e-9))
        pdi = float(ind_15m.get("pdi", 0.0)); mdi = float(ind_15m.get("mdi", 0.0))
        mh  = float(ind_15m.get("macd_hist", 0.0) or 0.0)
        e20_1h = float(ind_1h.get("ema20", 0.0) or 0.0)
        e50_1h = float(ind_1h.get("ema50", 0.0) or 0.0)
        mh_1h  = float(ind_1h.get("macd_hist", 0.0) or 0.0)
        if direction == "LONG":
            c_macd_di = (mh > 0) and (pdi > mdi)
            c_1h      = (e20_1h > e50_1h) or (mh_1h > 0)
        else:
            c_macd_di = (mh < 0) and (mdi > pdi)
            c_1h      = (e20_1h < e50_1h) or (mh_1h < 0)
        confirms = {
            "CLOSE_BREAK": beyond > 0,                              # real close break, not a wick
            "BEYOND_ATR":  beyond >= self.BREAKOUT_MIN_BEYOND_ATR * atr,
            "BODY":        body   >= self.BREAKOUT_MIN_BODY_ATR * atr,
            "VOLUME":      vol_ratio >= self.BREAKOUT_MIN_VOL_RATIO,
            "MACD_DI":     c_macd_di,
            "1H_ALIGN":    c_1h,
        }
        # CLOSE_BREAK is a prerequisite: without a real close beyond the level
        # there is no breakout to trade at all.
        if not confirms["CLOSE_BREAK"]:
            self._scan_info[direction] = "veto:no close beyond structure (wick only / inside range)"
            return None
        confirm_count = sum(1 for ok in confirms.values() if ok)
        required = 5 if is_transition else 4
        if confirm_count < required:
            passed = ",".join(k for k, ok in confirms.items() if ok)
            self._scan_info[direction] = (
                f"veto:breakout confirm {confirm_count}/6 (need {required}; "
                f"passed={passed}{'; macro-transition' if is_transition else ''})")
            return None

        # ── (E) strategy score (expectancy-blocked combos excluded) ──────────
        strategy_scores = {
            name: self.strategy_scorer.score(name, direction, ind_15m, l1, l2, "Breakout")
            for name in REGIME_STRATEGIES["Breakout"]
            if not self.expectancy_engine.is_blocked("Breakout", name)
        }
        best = self.confidence_engine.select_best(strategy_scores)
        if best is None:
            self._scan_info[direction] = "strategy_fail: all Breakout strategies blocked"
            return None
        best_strategy, best_score = best

        # ── (F) composite quality score → threshold 64 ───────────────────────
        confirm_q = confirm_count / 6.0 * 100.0
        break_q   = float(np.clip((beyond / atr - 0.10) / 0.30 * 100.0, 0, 100))
        body_q    = float(np.clip((body / atr - 0.25) / 0.35 * 100.0, 0, 100))
        vol_q     = float(np.clip((vol_ratio - 1.0) / 0.80 * 100.0, 0, 100))
        aligned = ((direction == "LONG" and macro_level in ("BULL", "STRONG_BULL")) or
                   (direction == "SHORT" and macro_level in ("BEAR", "STRONG_BEAR")))
        if aligned:
            macro_quality = 100.0 if macro_level.startswith("STRONG") else 85.0
        elif is_transition:
            macro_quality = 50.0
        else:
            macro_quality = 70.0
        total = (confirm_q * 0.35 + break_q * 0.12 + body_q * 0.08 +
                 vol_q * 0.15 + macro_quality * 0.20 + best_score * 0.10)

        strong = (not is_transition) and (confirm_count >= 5)
        mode = "immediate" if strong else "retest"
        self._filter_stats["checked"] += 1
        passed_names = ",".join(k for k, ok in confirms.items() if ok)
        self._scan_info[direction] = (
            f"total {total:.0f}/{threshold} regime=Breakout strat={best_strategy}({best_score:.0f}) "
            f"confirm={confirm_count}/6[{passed_names}] macro={macro_score:.0f}({zone}) mode={mode}"
            + (" → SIGNAL" if (total >= threshold and strong) else ""))
        if total < threshold:
            self._filter_stats["threshold_fail"] += 1
            return None

        risk_mult = self.BREAKOUT_TRANSITION_RISK if is_transition else 1.0

        # ── (G) immediate vs retest ──────────────────────────────────────────
        if strong:
            self._filter_stats["passed"] += 1
            sl_price = (lo_cur - 0.15 * atr) if direction == "LONG" else (h + 0.15 * atr)
            return self._breakout_signal_dict(
                direction, close, sl_price, confirms, confirm_count, total,
                best_strategy, best_score, risk_mult, macro_quality, l1, l2, "immediate")

        # Normal (4/6) or transition → arm a retest; enter only on reclaim.
        self._pending_breakout = {
            "direction": direction, "bos_level": float(bos_level), "atr": atr,
            "created_bar": self._bar_count, "confirm_count": confirm_count,
            "confirms": confirms, "score": total, "best_strategy": best_strategy,
            "best_score": best_score, "risk_mult": risk_mult,
            "macro_quality": macro_quality,
        }
        self._scan_info[direction] += " | armed retest (await reclaim 1-3 bars)"
        return None

    def _generate_signal(self, direction: str, candle_15m: Dict, ind_15m: Dict,
                         ind_1h: Dict, ind_4h: Dict,
                         l1: Dict, l2: Dict, l3: Dict) -> Optional[Dict]:
        """
        V9 3-layer signal pipeline.
        Scores all regime strategies, selects the best, combines with L2 context
        and L1 macro fit, gates on REGIME_THRESHOLDS.
        """
        regime     = l3["regime"]
        threshold  = REGIME_THRESHOLDS.get(regime, 62)

        # Range and Breakout each have a dedicated isolated engine — keeping
        # these branches first guarantees the add-ons cannot modify Trend's
        # veto/scoring path (Trend still falls through to the pipeline below).
        if regime == "Range":
            return self._generate_range_signal(direction, candle_15m, ind_15m, l1, l2)
        if regime == "Breakout":
            return self._generate_breakout_signal(direction, candle_15m, ind_15m, ind_1h, l1, l2)

        # ── Veto filters (non-MR regimes only) ──────────────────────────────
        if regime not in _MR_REGIMES:
            _px  = float(candle_15m.get("close", ind_15m.get("close", 0.0)))
            _e20 = ind_15m.get("ema20", _px)
            _atr = max(ind_15m.get("atr", 1e-9), 1e-9)
            if abs(_px - _e20) / _atr < self.MIN_EMA_DIST_ATR:
                self._filter_stats["checked"] += 1
                self._filter_stats["veto_chop"] += 1
                self._scan_info[direction] = (
                    f"veto:chop-zone (dist {abs(_px-_e20)/_atr:.2f} "
                    f"< {self.MIN_EMA_DIST_ATR} ATR)")
                return None

            _rng = float(candle_15m.get("high", _px)) - float(candle_15m.get("low", _px))
            if _rng > self.CLIMAX_BAR_ATR * _atr:
                self._filter_stats["checked"] += 1
                self._filter_stats["veto_climax"] += 1
                self._scan_info[direction] = (
                    f"veto:climax-bar (range {_rng/_atr:.1f} > {self.CLIMAX_BAR_ATR} ATR)")
                return None

            # 1H efficiency confirms a Trend pullback. It must NOT veto a
            # Breakout, which naturally begins from compression/chop before expansion.
            if regime == "Trend":
                _eff_1h = ind_1h.get("eff_ratio", 0.5)
                if self.MIN_1H_EFFICIENCY > 0 and _eff_1h < self.MIN_1H_EFFICIENCY:
                    self._filter_stats["checked"] += 1
                    self._filter_stats["veto_1h_chop"] += 1
                    self._scan_info[direction] = (
                        f"veto:1h-chop (eff {_eff_1h:.2f} < {self.MIN_1H_EFFICIENCY})")
                    return None

        # ── [V9.2 QUALITY] Macro-conviction veto ─────────────────────────────
        # A 15m "Trend" classification with a NEUTRAL 4H macro behind it is a
        # trend with no higher-timeframe fuel. Clean-run evidence: L1 score in
        # the neutral middle third → 44% WR (-$1,217); decisive L1 either
        # direction → 79-82% WR (+$945).
        # [V9.3 ACTIVE] A broad NEUTRAL label covers L1 scores 40-59. The old
        # hard veto rejected even 58/42, which already carries useful directional
        # lean. Only reject a truly flat macro when 1H context is also weak;
        # otherwise allow it with a bounded score penalty below.
        _macro_soft_penalty = 0.0
        if regime == "Trend" and l1.get("level") == "NEUTRAL":
            _l1_score = float(l1.get("score", 50.0))
            _dir_edge = ((_l1_score - 50.0) if direction == "LONG"
                         else (50.0 - _l1_score))
            _l2_preview = (l2.get("bull_score", 50.0) if direction == "LONG"
                           else l2.get("bear_score", 50.0))
            if _dir_edge < 4.0 and _l2_preview < 60.0:
                self._filter_stats["checked"] += 1
                self._filter_stats["veto_macro"] += 1
                self._scan_info[direction] = (
                    f"veto:flat-macro (L1 {_l1_score:.0f}, edge {_dir_edge:.1f}, "
                    f"L2 {_l2_preview:.0f})")
                return None
            _macro_soft_penalty = float(np.clip((8.0 - max(_dir_edge, 0.0)) * 0.75, 0.0, 6.0))

        # ── [V9.1 QUALITY] Trend pullback + RSI-chase vetoes ─────────────────
        # Enter the QUIET phase of a 4H trend (15m ADX still low = pullback),
        # never the extended leg. Backtest: ADX≤22 → 68.9% WR, ADX>30 → 47%.
        if regime == "Trend":
            _adx15 = ind_15m.get("adx", 20.0)
            if _adx15 > self.MAX_15M_ADX_TREND:
                self._filter_stats["checked"] += 1
                self._filter_stats["veto_chase"] += 1
                self._scan_info[direction] = (
                    f"veto:late-trend (15m ADX {_adx15:.0f} > "
                    f"{self.MAX_15M_ADX_TREND:.0f} — leg already extended)")
                return None

            _rsi15 = ind_15m.get("rsi", 50.0)
            if ((direction == "LONG" and _rsi15 > self.TREND_RSI_CHASE_HI) or
                    (direction == "SHORT" and _rsi15 < self.TREND_RSI_CHASE_LO)):
                self._filter_stats["checked"] += 1
                self._filter_stats["veto_chase"] += 1
                self._scan_info[direction] = (
                    f"veto:rsi-chase (rsi {_rsi15:.0f} — entering into extreme)")
                return None

        # ── Strategy scoring (expectancy-blocked combos excluded) ────────────
        strategy_scores = {
            s: self.strategy_scorer.score(s, direction, ind_15m, l1, l2, regime)
            for s in REGIME_STRATEGIES.get(regime, REGIME_STRATEGIES["Trend"])
            if not self.expectancy_engine.is_blocked(regime, s)
        }
        best = self.confidence_engine.select_best(strategy_scores)
        if best is None:
            self._filter_stats["checked"] += 1
            self._filter_stats["strategy_fail"] += 1
            return None
        best_strategy, best_score = best

        if best_score < 30.0:
            self._filter_stats["checked"] += 1
            self._filter_stats["strategy_fail"] += 1
            self._scan_info[direction] = f"strategy_fail: best={best_strategy} {best_score:.0f}"
            return None

        # ── L1 fit + L2 context ──────────────────────────────────────────────
        l1_fit  = self._compute_l1_fit(l1, direction, regime)
        l2_ctx  = l2["bull_score"] if direction == "LONG" else l2["bear_score"]

        # ── Condition penalty (Level 2/3 adaptive learning) ──────────────────
        _px_now      = float(candle_15m.get("close", ind_15m.get("close", 0.0)))
        _atr_now     = max(ind_15m.get("atr", 1e-9), 1e-9)
        _ema_dist    = abs(_px_now - ind_15m.get("ema20", _px_now)) / _atr_now
        _atr_exp     = ind_15m.get("atr", 0.0) / max(ind_15m.get("atr_avg", ind_15m.get("atr", 1.0)), 1e-9)
        _vol_ratio   = ind_15m.get("volume", 0.0) / max(ind_15m.get("vol_avg", ind_15m.get("volume", 1.0)), 1e-9)
        _now_ts      = self._bar_now or datetime.datetime.now(datetime.timezone.utc)
        _current_tags = self._diagnose_conditions(
            direction=direction, ema_dist_atr=_ema_dist,
            rsi=ind_15m.get("rsi", 50.0), atr_exp=_atr_exp,
            vol_ratio=_vol_ratio, adx=ind_15m.get("adx", 25.0), now=_now_ts,
        )
        _condition_penalty = (self.condition_engine.get_penalty(_current_tags)
                              + self._active_strategy_penalty(_current_tags, _now_ts)
                              + _macro_soft_penalty)

        # ── Composite score ──────────────────────────────────────────────────
        total = best_score * 0.40 + l2_ctx * 0.30 + l1_fit * 0.30 - _condition_penalty

        self._filter_stats["checked"] += 1
        self._scan_info[direction] = (
            f"total {total:.0f}/{threshold} regime={regime} "
            f"strat={best_strategy}({best_score:.0f}) l2={l2_ctx:.0f} l1fit={l1_fit:.0f}"
            + (f" pen=-{_condition_penalty:.0f}" if _condition_penalty > 0 else "")
            + (" → SIGNAL" if total >= threshold else "")
        )

        if total < threshold:
            self._filter_stats["threshold_fail"] += 1
            self._log_event(
                f"\n{'='*36}\n"
                f"  ENTRY CHECK ({direction} | {regime})\n"
                f"  Strategy   : {best_strategy} {best_score:.0f}/100\n"
                f"  L2 Context : {l2_ctx:.0f}/100\n"
                f"  L1 Fit     : {l1_fit:.0f}/100\n"
                f"  Total      : {total:.0f} / {threshold}\n"
                f"  Result     : NO TRADE\n"
                f"{'='*36}",
                level="debug",
            )
            return None

        self._filter_stats["passed"] += 1
        sl_price   = self._compute_sl_price(ind_15m, candle_15m, direction, regime)
        entry_type = _REGIME_ENTRY_TYPE.get(regime, "trend_follow")

        return {
            "direction":         direction,
            "sl_price":          sl_price,
            "health_score":      best_score,
            "confidence_score":  (l2_ctx + l1_fit) / 2.0,
            "total_score":       total,
            "entry_score":       best_score,
            "context_score":     l2_ctx,
            "direction_fit":     l1_fit,
            "entry_type":        entry_type,
            "strategy":          best_strategy,
            "regime":            regime,
            "l1_score":          l1["score"],
            "l1_level":          l1["level"],
            "l2_bull":           l2["bull_score"],
            "l2_bear":           l2["bear_score"],
            "all_strategies":    strategy_scores,
            "condition_penalty": _condition_penalty,
            "entry_tags":        _current_tags,
        }

    # ── [MTF-CONFLUENCE] Alternate entry engine ───────────────────────────────

    def _generate_signal_mtf(self, candle_15m: Dict, ind_15m: Dict) -> Optional[Dict]:
        """
        Deterministic MTF alignment + 3-signal confluence entry (see
        mtf_confluence_engine.py). Direction/timing come entirely from that
        engine — SL/sizing/exits below reuse the existing V9.2 machinery
        unchanged so this only swaps *when and which way* a trade opens.
        The confluence check is boolean (all-or-nothing), so unlike the
        scored V9.2 signal there's no graded confidence to report — a fired
        signal gets full conviction (health/confidence/total = 100).
        """
        raw = self._raw_candles or {}
        c15 = raw.get("15m")
        if not c15:
            self._scan_info["MTF"] = "no raw 15m candles supplied to on_tick"
            return None

        direction = self.mtf_engine.process_15m(
            c15, self._bar_count, self.position_open, self.state == "COOLDOWN",
        )
        diag = self.mtf_engine.last_diag
        self._scan_info["MTF"] = (
            f"macro={diag.get('macro')} mid={diag.get('mid')} "
            f"dir={diag.get('trade_direction')} "
            f"long={diag.get('long_status')} short={diag.get('short_status')}"
        )
        if direction is None:
            return None

        sl_price = self._compute_sl_price(ind_15m, candle_15m, direction, "Trend")
        return {
            "direction":         direction,
            "sl_price":          sl_price,
            "health_score":      100.0,
            "confidence_score":  100.0,
            "total_score":       100.0,
            "entry_score":       100.0,
            "context_score":     100.0,
            "direction_fit":     100.0,
            "entry_type":        "mtf_confluence",
            "strategy":          f"MTF_{direction}",
            "regime":            self.current_market_state,
            "l1_score":          self.regime_score,
            "l1_level":          self.current_regime_bias,
            "l2_bull":           0.0,
            "l2_bear":           0.0,
            "all_strategies":    {},
            "condition_penalty": 0.0,
            "entry_tags":        [],
        }

    # ── Lightweight cooldown check — independent of new-candle ticks ─────────

    def check_cooldown_expiry(self, now: Optional[datetime.datetime] = None) -> bool:
        """
        Check if COOLDOWN has expired without waiting for the next 15m candle
        close (on_tick only runs per-bar, so cooldown could sit expired for
        up to 15 minutes otherwise). Called every 5 min by the runner.
        Returns True if the state transitioned to SCANNING.
        """
        if self.state != "COOLDOWN":
            return False
        _now = now or datetime.datetime.now(datetime.timezone.utc)
        if self.cooldown_until is None or _now >= self.cooldown_until:
            self.state       = "SCANNING"
            self.loss_streak = 0
            self._log_event("Cooldown expired → SCANNING (5-min check)")
            return True
        return False

    # ── Unified target ladder — single shared TP/SL-ratchet for all states ───

    def _queue_target_alert(self, label: str, price: float,
                            old_sl: Optional[float], new_sl: Optional[float],
                            final: bool = False, close_pct: float = 0.0,
                            fill: Optional[Dict] = None) -> None:
        """Queue a Telegram-ready dict for the runner to format/send. `fill`
        is current_trade["last_fill"] (see _close_position) — the OKX-
        verified avgPx/fee/realizedPnl/net_pnl for THIS close, or None on
        paper/backtest executors that don't report it."""
        self._pending_target_alerts.append({
            "label": label, "price": price,
            "old_sl": old_sl, "new_sl": new_sl, "final": final,
            "close_pct": close_pct, "fill": fill,
        })

    def pop_target_alerts(self) -> List[Dict]:
        """Return and clear queued target-hit alerts (runner forwards to Telegram)."""
        alerts = self._pending_target_alerts
        self._pending_target_alerts = []
        return alerts

    def _check_targets(self, t: Dict, direction: str, current_price: float,
                       ind: Dict, now: Optional[datetime.datetime] = None) -> Optional[str]:
        """
        Walk the 2-level target structure (T1/T2, see _target_ladder): T1
        closes a partial and moves the SL to breakeven, T2 closes whatever's
        left (matches the exchange-attached TP2). A single gap candle can
        cross both levels — loop so neither is skipped and each crossed
        level still gets its Telegram alert. Used by both
        check_price_protection (intrabar) and _manage_open_position (bar
        close) so behavior is identical either way.
        Returns a short action description, or None if nothing fired.
        """
        targets = t.get("targets")
        if not targets:
            return None
        dir_mult = 1 if direction == "LONG" else -1
        actions: List[str] = []

        while t.get("next_target_idx", 0) < len(targets):
            idx = t["next_target_idx"]
            r, close_pct, new_sl_r = targets[idx]
            level_price = t["entry"] + t["sl_dist"] * r * dir_mult
            hit = (current_price >= level_price) if direction == "LONG" \
                  else (current_price <= level_price)
            if not hit:
                break
            label = f"T{idx + 1}"
            is_final = (idx == len(targets) - 1)

            # [MIN-LOT] A partial close below the smallest fillable size
            # would wipe the WHOLE remaining position instead of the
            # intended fraction — degrade to a breakeven-move-only (skip
            # the close, still tighten SL) on non-final levels, the same
            # fallback the pre-ladder split-TP system used.
            effective_close_pct = close_pct
            if not is_final and self.min_close_size > 0:
                close_amount = t.get("remaining_size", 0.0) * close_pct
                if 0 < close_amount < self.min_close_size:
                    effective_close_pct = 0.0
                    self._log_event(
                        f"{label}: partial close {close_amount:.6f} below "
                        f"min-lot {self.min_close_size:.6f} — SL-move only, no close",
                        level="warning",
                    )

            if effective_close_pct > 0:
                self._close_position(f"{label}_HIT", level_price, effective_close_pct, ind, now=now)
                if self.state == "ERROR":
                    # Flush any earlier progress made this same pass even
                    # though this level's close attempt itself failed.
                    if actions:
                        self.save_state(self._state_file)
                    return " | ".join(actions) if actions else None

            old_sl = t["sl"]
            if new_sl_r is not None:
                new_sl = t["entry"] + t["sl_dist"] * new_sl_r * dir_mult
                t["sl"] = max(t["sl"], new_sl) if direction == "LONG" else min(t["sl"], new_sl)

            t["next_target_idx"] = idx + 1
            t.setdefault("targets_hit", []).append(label)

            if is_final or t.get("remaining_size", 0.0) <= 1e-9:
                # Final level — closes whatever remains (matches the
                # exchange-attached TP2), or the position happened to be
                # fully closed already (min-lot floor consumed 100%).
                t["tp1_hit"] = True   # legacy flags some downstream logic reads
                t["tp2_hit"] = True
                self.state = "EXITING"
                self._queue_target_alert(label, level_price, None, None, final=True,
                                         fill=t.get("last_fill"))
                actions.append(f"{label}_HIT(close) @ {level_price:.4f}")
                self.save_state(self._state_file)
                return " | ".join(actions)

            t["tp1_hit"] = True
            t["break_even_triggered"] = True
            self._log_event(
                f"{label} hit @ {level_price:.4f} ({r}R) → closed "
                f"{effective_close_pct*100:.0f}% | SL {old_sl:.4f} → {t['sl']:.4f}"
            )
            self._send_amend_sl(t["sl"])
            self._queue_target_alert(label, level_price, old_sl, t["sl"], close_pct=effective_close_pct,
                                     fill=t.get("last_fill"))
            actions.append(
                f"{label} @ {level_price:.4f} close={effective_close_pct*100:.0f}% SL→{t['sl']:.4f}"
            )

        if actions:
            self.save_state(self._state_file)
            return " | ".join(actions)
        return None

    # ── Intrabar price protection — runs every runner poll between bar closes ─

    def check_price_protection(self, current_price: float,
                               now: Optional[datetime.datetime] = None) -> Optional[str]:
        """
        Lightweight price-level protection (SL / target-ladder crossings)
        checked every poll (~30-60s) instead of only on 15m bar close. Full
        indicator-based management (health tiers, reversal spike, trend fade,
        state drift) still runs per closed bar in _manage_open_position —
        those need fresh indicators; price levels don't. Closes/partials use
        the same _close_position path (real orders + accounting).
        Returns a short action description, or None if nothing fired.

        `now` defaults to real wall-clock time — correct for live polling,
        where intrabar closes happen in real elapsed time and self._bar_now
        (only advanced on bar close) would understate the gap. A backtest
        replaying historical intrabar candles must override this with the
        SIMULATED candle time instead, or the whipsaw-spacing gate would get
        stamped with today's real date.
        """
        if not self.position_open or not self.current_trade:
            return None
        t = self.current_trade
        if t.get("status") != "OPEN":
            return None
        direction = t["direction"]
        _now = now or datetime.datetime.now(datetime.timezone.utc)

        sl_hit = (current_price <= t["sl"]) if direction == "LONG" else (current_price >= t["sl"])
        if sl_hit:
            self._close_position("SL_HIT", t["sl"], 1.0, {}, now=_now)
            if self.state != "ERROR":
                self.state = "EXITING"
                self.save_state(self._state_file)
                return f"SL_HIT @ {t['sl']:.4f}"
            return None

        return self._check_targets(t, direction, current_price, {}, now=_now)

    # ── Step 3: Global gates ──────────────────────────────────────────────────

    def _check_global_gates(self) -> bool:
        if self.position_open:
            return False

        # [SESSION CONTROL] Extended commodity-market session gate (see
        # session_engine.py) — blocks only new entries. Existing positions,
        # protective orders, Telegram alerts, and exchange heartbeat all run
        # through code paths that never call _check_global_gates(), so
        # nothing else in the bot is affected while this is closed.
        if not self.session_gate_open:
            self._log_event(f"SKIP: session {self.session_state} — new entries disabled", level="debug")
            return False

        _now = self._bar_now or datetime.datetime.now()

        if self._startup_unblock_at and _now < self._startup_unblock_at:
            remaining = int((self._startup_unblock_at - _now).total_seconds() / 60)
            self._log_event(f"WARMUP: {remaining}m remaining", level="debug")
            return False

        if self.cooldown_until and _now < self.cooldown_until:
            self.state = "COOLDOWN"
            return False

        # [WHIPSAW GUARD] entry spacing — no new entry within N minutes of the
        # previous CLOSE on this symbol (blocks re-entry chasing right after
        # getting stopped out while price whips up/down).
        if self._last_close_at and self.entry_spacing_min > 0:
            try:
                elapsed = (_now - self._last_close_at).total_seconds() / 60.0
                if 0 <= elapsed < self.entry_spacing_min:
                    self._log_event(
                        f"ENTRY-SPACING: {elapsed:.0f}m since last close "
                        f"(< {self.entry_spacing_min}m) — waiting",
                        level="debug",
                    )
                    return False
            except TypeError:
                # naive/aware mismatch after a state reload — reset rather than block forever
                self._last_close_at = None

        # [DAILY LIMITS] Disabled sentinel: loss limit off when >= 0, profit
        # limit off when <= 0 (so 0/0 = trade 24/7, never day-halt). On a
        # small balance the old -3%/+8% tripped almost immediately (one 5%-
        # risk SL loss alone is -5% < -3%, and ~4 small wins clear +8%),
        # which stranded most symbols in BLOCKED for the rest of the day.
        if self.daily_loss_limit_pct < 0 and self.daily_pnl_pct <= self.daily_loss_limit_pct:
            self.state = "BLOCKED"
            self._log_event(
                f"BLOCKED: daily PnL {self.daily_pnl_pct:.2f}% hit loss limit",
                level="warning",
            )
            return False

        if self.daily_profit_limit_pct > 0 and self.daily_pnl_pct >= self.daily_profit_limit_pct:
            self.state = "BLOCKED"
            self._log_event(
                f"BLOCKED: daily PnL {self.daily_pnl_pct:.2f}% hit profit limit",
                level="warning",
            )
            return False

        # [MTF-CONFLUENCE] These two gates are V9.2-specific entry-quality
        # rules learned from the L1/L2/L3 pipeline's own backtests (the
        # 5-regime taxonomy and the Asia-hours WR finding). The MTF engine
        # has its own direction gate (4H+1H trend alignment, checked inside
        # process_15m/_generate_signal_mtf) that plays the equivalent role,
        # so neither V9.2 rule applies when it's the active engine.
        if self.entry_engine == "adaptive":
            if self.current_market_state not in _TRADEABLE_REGIMES:
                self._log_event(
                    f"SKIP: untradeable regime={self.current_market_state}", level="debug"
                )
                return False

            # [V9.1 QUALITY] Session gate — Asia hours (00-05 UTC) proved 37-45% WR
            if _now.hour in self.BLOCKED_ENTRY_HOURS_UTC:
                self._log_event(
                    f"SKIP: blocked session hour {_now.hour:02d} UTC", level="debug"
                )
                return False

        return True

    # ── Step 5: Risk engine + adaptive sizing ───────────────────────────────

    def _step5_risk_engine(self, candle: Dict, direction: str, ind: Dict,
                           mr_signal: Optional[Dict] = None):
        """
        Compute SL/TP/size and open position from the unified signal dict
        produced by _generate_signal (sl_price, health_score, confidence_score,
        entry_type, strategy).
        """
        signal = mr_signal
        if signal is None:
            self._log_event("_step5_risk_engine called with no signal — abort", level="error")
            return
        entry_price = float(candle.get("close", 0))

        # ── SL calculation ──────────────────────────────────────────────────
        pattern_sl = signal["sl_price"]

        sl_dist = abs(entry_price - float(pattern_sl))
        if sl_dist < 1e-8:
            sl_dist = entry_price * 0.01

        # [RANGE FLOOR] The default min-SL floor (self.min_sl_pct, 1.2% of
        # price) is calibrated for Trend's ATR×1.5 stops. Range uses a much
        # tighter structure stop (edge ± 0.15 ATR) whose TP levels are also
        # structure-based, so forcing the 1.2% floor blew the stop past the
        # range midpoint and collapsed the R:R (every passing Range signal
        # got skipped at the ladder R:R re-check). Range instead floors on a
        # fraction of ATR — proportional to the same volatility its targets
        # are, so the structure R:R survives — with a hard 0.25% absolute
        # minimum so a tiny-ATR reading can't produce a runaway position size.
        _is_range = signal.get("regime") == "Range"
        if _is_range:
            _atr_now = max(float(ind.get("atr", 0.0) or 0.0), 0.0)
            min_sl_dist = max(0.30 * _atr_now, entry_price * 0.0025)
        else:
            min_sl_dist = entry_price * self.min_sl_pct
        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
            pattern_sl = (entry_price - sl_dist if direction == "LONG"
                          else entry_price + sl_dist)

        # ── Size multiplier ──────────────────────────────────────────────────
        risk_pct = self.base_risk_pct
        if self.win_streak >= 5:
            risk_pct *= 0.80
            self._log_event(f"Win streak {self.win_streak} → risk {risk_pct:.2%}")

        health        = signal.get("health_score", 0.0)
        confidence    = signal.get("confidence_score", health)
        health_mult   = 1.0 if health >= 75 else 0.65
        entry_type    = signal.get("entry_type") or _REGIME_ENTRY_TYPE.get(
            self.current_market_state, "trend_follow")
        learning_mult = self.learning_engine.get_weight(entry_type)
        regime_risk_mult = float(np.clip(signal.get("risk_multiplier", 1.0), 0.10, 1.00))
        size_mult     = health_mult * learning_mult * regime_risk_mult

        # [SIZING] Three modes, in this precedence order:
        # - margin_pct_min/max > 0 (Level 1 default): dynamic %-of-balance,
        #   scaled between the two bounds by the bot's own conviction (score
        #   headroom above this state's bar, penalized by any historically
        #   -bad condition tags present) — a strong, clean signal gets a
        #   bigger position than one that barely cleared the bar.
        # - margin_usdt > 0 (legacy override): FIXED notional = margin × leverage,
        #   the same size regardless of signal quality.
        # - neither set (backtest / opt-out): classic risk-% of balance.
        if self.margin_pct_min > 0 or self.margin_pct_max > 0:
            lo = max(self.margin_pct_min, 0.0)
            hi = max(self.margin_pct_max, lo)
            _thrs_now = REGIME_THRESHOLDS.get(self.current_market_state, 62)
            headroom  = signal.get("total_score", 0.0) - _thrs_now
            conf_norm = float(np.clip(headroom / 25.0, 0.0, 1.0))
            tag_penalty_norm = float(np.clip(
                signal.get("condition_penalty", 0.0) / max(self.condition_engine.MAX_PENALTY, 1e-9), 0.0, 1.0))
            conviction    = conf_norm * (1.0 - tag_penalty_norm)
            margin_pct    = (lo + (hi - lo) * conviction) * regime_risk_mult
            notional      = self.account_balance * margin_pct * max(self.sizing_leverage, 1)
            position_size = notional / max(entry_price, 1e-9)

            # [MIN-LOT FLOOR] same guard as the fixed-margin branch below —
            # the exchange fills whole contracts, so the real required
            # margin can silently exceed the nominal one at small sizes.
            real_size   = max(position_size, self.min_close_size) \
                          if self.min_close_size > 0 else position_size
            real_margin = (real_size * entry_price) / max(self.sizing_leverage, 1)
            if real_size > position_size:
                position_size = real_size

            if real_margin > self.account_balance:
                self._log_event(
                    f"[SIZING] required margin ${real_margin:.2f} "
                    f"(min-lot floored) > balance ${self.account_balance:.2f} "
                    f"— skip order (would be rejected for insufficient margin)",
                    level="warning",
                )
                return

            _risk_now = position_size * entry_price * (sl_dist / max(entry_price, 1e-9))
            self._log_event(
                f"[SIZING] adaptive-risk {margin_pct:.1%} of balance "
                f"(conviction={conviction:.2f}, headroom={headroom:.1f}, "
                f"tag_penalty={signal.get('condition_penalty', 0.0):.1f}) "
                f"= ${notional:.0f} notional (real margin≈${real_margin:.2f}, "
                f"risk≈${_risk_now:.2f} = "
                f"{_risk_now / max(self.account_balance, 1e-9) * 100:.1f}% of balance)"
            )
        elif self.margin_usdt and self.margin_usdt > 0:
            notional      = self.margin_usdt * max(self.sizing_leverage, 1) * regime_risk_mult
            position_size = notional / max(entry_price, 1e-9)

            # [MIN-LOT FLOOR] The exchange fills whole contracts — if the
            # intended notional is below one contract's real notional
            # (ct_val × price, known via min_close_size), the actual fill (and
            # therefore the real required margin) is silently larger than
            # margin_usdt. Recompute the REAL margin against the floored size
            # before gating, not the nominal margin_usdt.
            real_size   = max(position_size, self.min_close_size) \
                          if self.min_close_size > 0 else position_size
            real_margin = (real_size * entry_price) / max(self.sizing_leverage, 1)
            if real_size > position_size:
                position_size = real_size

            # [MARGIN CHECK] If the (floor-adjusted) required margin exceeds
            # balance, the exchange will reject with insufficient-margin —
            # abort here with a clear reason instead of an opaque OKX error.
            if real_margin > self.account_balance:
                self._log_event(
                    f"[SIZING] required margin ${real_margin:.2f} "
                    f"(min-lot floored) > balance ${self.account_balance:.2f} "
                    f"— skip order (would be rejected for insufficient margin)",
                    level="warning",
                )
                return

            _risk_now = position_size * entry_price * (sl_dist / max(entry_price, 1e-9))
            self._log_event(
                f"[SIZING] fixed-margin ${self.margin_usdt:.0f}×{self.sizing_leverage}x "
                f"= ${notional:.0f} notional (real margin≈${real_margin:.2f}, "
                f"risk≈${_risk_now:.2f} = "
                f"{_risk_now / max(self.account_balance, 1e-9) * 100:.1f}% of balance)"
            )
        else:
            risk_amount   = self.account_balance * risk_pct * size_mult
            position_size = risk_amount / max(sl_dist, 1e-9)

        # ── TP levels ────────────────────────────────────────────────────────
        mult = 1 if direction == "LONG" else -1

        # Trend/Breakout keep the original fixed-R target ladder. Range may
        # supply structure targets (midpoint / opposite edge); after the
        # min-SL floor above, re-check their actual R:R and reject only that
        # Range setup if the widened stop leaves insufficient room.
        custom_tp1 = signal.get("tp1_price")
        custom_tp2 = signal.get("tp2_price")
        if custom_tp1 is not None and custom_tp2 is not None:
            tp1 = float(custom_tp1)
            tp2 = float(custom_tp2)
            tp1_r_actual = (tp1 - entry_price) * mult / sl_dist
            tp2_r_actual = (tp2 - entry_price) * mult / sl_dist
            if tp1_r_actual < 0.80 or tp2_r_actual < 1.10 or tp2_r_actual <= tp1_r_actual:
                self._log_event(
                    f"[Range] SKIP after min-SL floor: T1={tp1_r_actual:.2f}R "
                    f"T2={tp2_r_actual:.2f}R", level="debug")
                return
            range_close_pct = float(np.clip(
                signal.get("tp1_close_pct", 0.60), 0.0, 0.99))
            ladder = [
                (tp1_r_actual, range_close_pct, self.BREAKEVEN_LOCK_R),
                (tp2_r_actual, 1.0, None),
            ]
        else:
            ladder = self._target_ladder()
            tp1 = entry_price + sl_dist * ladder[0][0] * mult
            tp2 = entry_price + sl_dist * ladder[-1][0] * mult
        tp3 = None
        tp1_pct   = ladder[0][1]
        tp2_pct   = 1.0
        trail_atr = 2.0

        strategy_tag = signal.get("strategy", "Adaptive")

        self._log_event(
            f"[{strategy_tag}] OPEN {direction} entry={entry_price:.4f} "
            f"sl={pattern_sl:.4f} tp1={tp1:.4f}({ladder[0][0]}R, close {tp1_pct:.0%}) "
            f"tp2={tp2:.4f}({ladder[-1][0]:.2f}R, close rest) "
            f"size×{size_mult:.2f} risk×{regime_risk_mult:.2f} health={health:.0f}"
        )

        self.current_trade = {
            "direction":            direction,
            "entry":                entry_price,
            "sl":                   float(pattern_sl),
            "sl_dist":              sl_dist,
            "tp1":                  tp1,
            "tp2":                  tp2,
            "tp3":                  tp3,           # None for SwingReversal
            "targets":              ladder,        # (trigger_R, close_pct, new_SL_R) list
            "next_target_idx":      0,
            "targets_hit":          [],            # ["T1","T2",...] for stats/lessons
            "sl_algo_id":           None,          # OKX algoId of the attached SL/TP order (for amends)
            "tp1_pct":              tp1_pct,
            "tp2_pct":              tp2_pct,
            "trail_atr_mult":       trail_atr,
            "tp1_hit":              False,
            "tp2_hit":              False,
            "tp3_hit":              False,
            "break_even_triggered": False,
            "status":               "OPEN",
            "entry_time":           datetime.datetime.now(datetime.timezone.utc),
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
            # Entry-time score breakdown (for fake-signal analysis / tuning)
            "e_entry":              signal.get("entry_score", health),
            "e_context":            signal.get("context_score", 0.0),
            "e_fit":                signal.get("direction_fit", 0.0),
            "e_total":              signal.get("total_score", 0.0),
            "e_state":              self.current_market_state,
            # Entry-time raw features the score ignores (fake-signal hunting)
            "e_adx":                ind.get("adx", 0.0),
            "e_atr_exp":            ind.get("atr", 0.0) / max(ind.get("atr_avg", ind.get("atr", 1.0)), 1e-9),
            "e_vol_ratio":          ind.get("volume", 0.0) / max(ind.get("vol_avg", ind.get("volume", 1.0)), 1e-9),
            "e_ema_dist_atr":       abs(entry_price - ind.get("ema20", entry_price)) / max(ind.get("atr", 1.0), 1e-9),
            "e_rsi":                ind.get("rsi", 50.0),
            "risk_multiplier":       regime_risk_mult,
            "range_low":             signal.get("range_low"),
            "range_high":            signal.get("range_high"),
            "range_confirmations":   signal.get("range_confirmations"),
        }

        _open_result = self._send_order(
            "OPEN_LONG" if direction == "LONG" else "OPEN_SHORT", {
                "entry": entry_price,
                "sl":    float(pattern_sl),
                "tp1":   tp1, "tp2": tp2,
                "tp3":   tp3,
                "size":  position_size,
                # Full T1-T2 ladder prices for the runner's OPEN notification
                # (label, price, trigger_R); extra key is ignored by the
                # exchange adapters and the backtest executor.
                "ladder": [
                    (f"T{i + 1}", entry_price + sl_dist * r * mult, r)
                    for i, (r, _close_pct, _sl_r) in enumerate(ladder)
                ],
            })

        # FIX-#1: if execution_callback failed, _send_order sets state=ERROR — abort
        if self.state == "ERROR":
            self.current_trade = {}
            return

        # [FILL SYNC] Exchanges fill whole contracts — the actually-filled size
        # can differ from the requested coin amount (int/round conversion).
        # Track the REAL size so partial closes and PnL match the exchange.
        if isinstance(_open_result, dict):
            _fc = float(_open_result.get("_filled_coins") or 0)
            if _fc > 0 and abs(_fc - position_size) / max(position_size, 1e-9) > 0.01:
                self._log_event(
                    f"[FILL SYNC] requested {position_size:.6f} → filled {_fc:.6f} coins"
                )
                self.current_trade["remaining_size"] = _fc
                # "size" is the ORIGINAL full-position denominator later used
                # to allocate the entry fee across each partial close — keep
                # it matched to what actually filled, not what was requested.
                self.current_trade["size"] = _fc
            self.current_trade["sl_algo_id"] = _open_result.get("_sl_algo_id")
            # [OKX FILL DATA] entry-side avgPx/fee, ground truth from the
            # exchange — used by _close_position to compute each close's Net
            # PnL (see the formula there). None on paper/backtest executors,
            # which don't return these fields — Telegram then falls back to
            # the bot's own pre-fill estimate.
            self.current_trade["entry_fee"]    = float(_open_result.get("_entry_fee") or 0.0)
            self.current_trade["entry_avg_px"] = float(_open_result.get("_entry_avg_px") or entry_price)
            # [REAL SIZE/MARGIN] ground truth from the exchange fill, for
            # display (Telegram OPEN alert, /stats open-position line) —
            # never re-derived from the pre-fill request, which can be far
            # off when the exchange's min-lot floor rounds the size up.
            self.current_trade["entry_order_value"] = float(_open_result.get("_entry_order_value") or 0.0)
            self.current_trade["entry_margin"]      = float(_open_result.get("_entry_margin") or 0.0)

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

        # [PROTECT-LAYER SWITCH] Master gate for every discretionary mid-trade
        # exit below (reversal spike, trend fade, emergency, state drift,
        # health tiers). Backtest evidence: ~77% of trades were closed by this
        # layer instead of TP/SL (61/71 losses cut early at -0.1..-0.9R,
        # 36/55 wins truncated to <0.3R), so the designed T1/T2 geometry was
        # almost never allowed to play out — WR 43.7% with the layer on vs
        # 63.4% with it off, same entries.
        #
        # [V9.2] The layer is therefore active only AFTER T1: pre-T1 the trade
        # has exactly two exits (T1 or the hard SL) and is allowed to breathe;
        # post-T1 75% is banked and the SL is at breakeven, so the protection
        # exits are locking in profit on the runner instead of strangling the
        # trade before it reaches its first target.
        # ADAPTIVE_PROTECT_EXITS=0 disables the layer entirely (experiments).
        _protect = self.enable_protect_exits and t.get("tp1_hit", False)
        reversal = self._detect_reversal_signals(ind) if _protect else {}
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
        ] if _protect else []
        if any(emergency_signals):
            self._close_position("EMERGENCY_EXIT", current_price, 1.0, ind)
            return "EXITING"

        # Unified 2-level target structure (T1/T2) — same for every state now.
        # T1 closes a partial + moves SL to breakeven; T2 (matches the
        # exchange-attached TP2) closes what's left. See _check_targets /
        # _target_ladder.
        target_action = self._check_targets(t, direction, current_price, ind)
        if target_action:
            if self.state == "EXITING":
                return "EXITING"
            self.state = "TRAILING"

        # Health-based position management
        tp1_hit = t.get("tp1_hit", False)
        health  = self.health_calc.calculate(ind, t, current_price)

        if not _protect:
            pass   # pre-T1 (or experiment mode): pure T1/T2/SL — no health-tier actions
        elif health >= 80:
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
                dir_mult_be = 1 if direction == "LONG" else -1
                t["sl"] = t["entry"] + t["sl_dist"] * self.BREAKEVEN_LOCK_R * dir_mult_be
                t["break_even_triggered"] = True
                self._log_event(f"Health {health:.0f} → forced breakeven+lock (post-TP1)")
        elif health >= 20:
            if t["remaining_size"] > 0 and tp1_hit:
                # [MIN-LOT] a 50% reduce below one contract would floor up and
                # close the WHOLE position while local accounting thinks half
                # remains — skip the reduce (SL is already at BE/+lock here).
                if (self.min_close_size > 0
                        and t["remaining_size"] * 0.50 < self.min_close_size):
                    self._log_event(
                        f"Health {health:.0f} → reduce skipped (min lot); "
                        f"SL protection already active"
                    )
                else:
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

    def _close_position(self, reason: str, price: float, portion: float, ind: Dict,
                        now: Optional[datetime.datetime] = None):
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
        t["exit_reason"]    = reason   # stored so EXITING state can read it

        if t["sl_dist"] > 0:
            t["final_rr"] = pnl_per_unit / t["sl_dist"]

        _close_result = self._send_order("CLOSE_PARTIAL" if portion < 1.0 else "CLOSE_FULL", {
            "reason":    reason,
            "price":     float(price),
            "size":      close_size,
            "pnl":       pnl,
            "direction": direction,
        })

        # [OKX FILL RECONCILIATION] Telegram should report what the exchange
        # actually filled/charged, not this pre-fill estimate (computed above
        # from the target/SL trigger price, before the market order actually
        # executed). Only populated on a live OKX close (paper/backtest
        # executors don't return these fields) — callers fall back to the
        # local pnl/price estimate above when this is None.
        #   Net PnL = Realized Trading PnL (OKX) - Entry Fee Allocation - Exit Fee
        # Entry Fee Allocation = entry_fee * (this close's size / the
        # original full position size) — this close only "used up" its
        # proportional share of the one fee paid when the position opened.
        t["last_fill"] = None
        if isinstance(_close_result, dict) and _close_result.get("_exit_avg_px"):
            _exit_fee     = float(_close_result.get("_exit_fee") or 0.0)
            _exit_avg_px  = float(_close_result.get("_exit_avg_px") or price)
            _exit_fill_sz = float(_close_result.get("_exit_fill_sz") or close_size)
            _realized_okx = float(_close_result.get("_realized_pnl") or 0.0)
            _entry_fee    = float(t.get("entry_fee") or 0.0)
            _orig_size    = float(t.get("size") or close_size)
            _entry_fee_alloc = _entry_fee * (_exit_fill_sz / max(_orig_size, 1e-9))
            t["last_fill"] = {
                "avg_px":          _exit_avg_px,
                "fill_sz":         _exit_fill_sz,
                "fee":             _exit_fee,
                "realized_pnl":    _realized_okx,
                "entry_fee_alloc": _entry_fee_alloc,
                "net_pnl":         _realized_okx - _entry_fee_alloc - _exit_fee,
            }

        # FIX-#4: if order failed (state=ERROR), don't mark local position as closed —
        # exchange still holds the real position.
        if self.state == "ERROR":
            t["remaining_size"] = remaining  # undo local accounting
            t["realized_pnl"]   = t.get("realized_pnl", 0.0) - pnl
            return

        self._log_event(
            f"CLOSE {reason} | {direction} | price={price:.2f} "
            f"| size={close_size:.4f} | pnl={pnl:+.2f}"
        )

        if portion >= 1.0 or t["remaining_size"] <= 1e-9:
            t["status"]        = "CLOSED"
            self.position_open = False
            self.order_status  = "CLOSED"
            # [WHIPSAW GUARD] stamp CLOSE time (not open) for the spacing gate.
            # Intrabar closes pass real wall-clock `now`; bar-close callers
            # leave it None so this falls back to simulated bar time (needed
            # for backtest determinism).
            self._last_close_at = now or self._bar_now or datetime.datetime.now(datetime.timezone.utc)

    # ── Step 7: Journal + [V8-6/V8-7] learning ───────────────────────────────

    def _log_trade(self, result: str, close_reason: str, ind: Dict, extras: Dict):
        t   = self.current_trade
        pnl = t.get("realized_pnl", 0.0)

        _entry      = t.get("entry", 0.0)
        _sl         = t.get("sl", _entry)
        _exit       = t.get("exit_price", _entry)
        # [FIX] Use the ORIGINAL entry-to-SL distance (t["sl_dist"], fixed at
        # position open) to normalize R, not the CURRENT t["sl"] — that price
        # ratchets to breakeven at T1 (t["sl"] = t["entry"]), which made
        # abs(_entry - _sl) collapse to ~0 for any trade closed after a T1
        # partial and exploded realized_r into billions (a fixed but
        # unrelated PnL divided by a near-zero denominator).
        _sl_d       = max(float(t.get("sl_dist") or 0.0), 1e-8)
        _d_mult     = 1 if t.get("direction") == "LONG" else -1
        _realized_r = _d_mult * (_exit - _entry) / _sl_d
        _win_r      = max(_realized_r, 0.0)
        _loss_r     = max(-_realized_r, 0.0)

        _holding_bars = self._bar_count - self._position_entry_bar
        _vol_state    = self.adaptive_engine.get_atr_volatility_state(
            ind.get("atr", 0), self.atr_history)

        entry_type = t.get("entry_type", _REGIME_ENTRY_TYPE.get(self.current_market_state, "trend_follow"))

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
            "hold_time_sec":       self._safe_hold_secs(t.get("entry_time")),
            "atr":                 ind.get("atr"),
            "adx":                 ind.get("adx"),
            "rsi":                 ind.get("rsi"),
            "ema":                 ind.get("ema5"),
            "macd":                ind.get("macd"),
            "market_state":        self.current_market_state,
            "regime_bias":         self.current_regime_bias,
            "strategy":            t.get("strategy", "SwingReversal"),
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
            # Entry-time score breakdown (fake-signal analysis)
            "e_entry":             t.get("e_entry"),
            "e_context":           t.get("e_context"),
            "e_fit":               t.get("e_fit"),
            "e_total":             t.get("e_total"),
            "e_state":             t.get("e_state"),
            "e_adx":               t.get("e_adx"),
            "e_atr_exp":           t.get("e_atr_exp"),
            "e_vol_ratio":         t.get("e_vol_ratio"),
            "e_ema_dist_atr":      t.get("e_ema_dist_atr"),
            "e_rsi":               t.get("e_rsi"),
            "realized_r":          round(_realized_r, 3),
            # [TARGET LADDER] which levels this trade reached before its
            # final close — direct evidence for fake-vs-real diagnosis
            # (e.g. "no targets hit -> SL" vs "T1,T2 hit then reversed").
            "targets_hit":         list(t.get("targets_hit", [])),
            # [STATS] closed-trade wall-clock time — /stats "last N trades"
            # sorts/labels by this (ISO strings sort correctly as text).
            "closed_at":           (self._bar_now or datetime.datetime.now(datetime.timezone.utc)).isoformat(),
            # [LEVEL 0/2/3] why this loss likely happened — populated below
            "loss_tags":           [],
        }

        # [LEVEL 0 — DIAGNOSIS] Tag every closed trade (win or loss) with its
        # entry-time condition profile and feed ConditionLearningEngine
        # (Level 2) — win-rate per tag needs both outcomes to be meaningful,
        # not just losses. "loss_tags" in the journal (used by the lesson
        # alert / STRATEGY_DOMINANCE) is only populated for actual losses.
        _tags = self._diagnose_loss(t, close_reason)
        if result == "LOSS":
            entry["loss_tags"] = _tags
        if _tags:
            self.condition_engine.record(_tags, win=(result == "WIN"))
        self.trade_journal.append(entry)

        # [V8-7] Learning engine record
        self.learning_engine.record(
            entry_type,
            win=(result == "WIN"),
            r_multiple=_realized_r,
            hold_bars=_holding_bars,
        )
        # [V9.1] Expectancy engine — per (regime, strategy) self-pruning
        self.expectancy_engine.record(
            t.get("e_state") or self.current_market_state,
            t.get("strategy") or "",
            win=(result == "WIN"),
        )
        # Update weights every 20 trades
        if len(self.trade_journal) % 20 == 0:
            self.learning_engine.update_weights()
            self._log_event(
                f"[LEARN] weights updated: {self.learning_engine.get_summary()}"
            )

        # [LESSON] loss-cluster detection → alert for the runner to forward.
        # Never let an alert-formatting problem break trade accounting below.
        try:
            self._check_lessons()
        except Exception as _le:
            self._log_event(f"[LESSON] alert build failed (non-fatal): {_le}", level="warning")

        # [LEVEL 3] Check AFTER the lesson-cluster logic above so a fresh
        # dominance-triggering loss is already in trade_journal.
        if result == "LOSS":
            try:
                self._check_strategy_dominance()
            except Exception as _se:
                self._log_event(f"[STRATEGY] dominance check failed (non-fatal): {_se}", level="warning")

        # Update streaks + daily PnL
        self.daily_pnl_pct     += (pnl / max(self.account_balance, 1)) * 100
        self.account_balance   += pnl

        win = (result == "WIN")
        _now = self._bar_now or datetime.datetime.now(datetime.timezone.utc)
        is_range_trade = (t.get("e_state") == "Range")
        cooldown_mins = None

        if is_range_trade:
            # Range has its own streak/cooldown. A failed edge fade must never
            # pause a valid Trend/Breakout setup; only the account-wide daily
            # PnL limits above remain shared.
            if win:
                self.range_loss_streak = 0
            else:
                self.range_loss_streak += 1
                if close_reason == "SL_HIT" and self.range_cooldown_minutes > 0:
                    self.range_cooldown_until = _now + datetime.timedelta(
                        minutes=self.range_cooldown_minutes)
                    self._log_event(
                        f"[Range] SL → Range-only cooldown "
                        f"{self.range_cooldown_minutes}m (Trend/Breakout unaffected)",
                        level="warning",
                    )
        else:
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

            # [V8-6] Tiered cooldown — Trend/Breakout only.
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

    # ── Core tick engine ──────────────────────────────────────────────────────

    def on_tick(self, candle_15m: Dict, candle_1h: Dict, candle_4h: Dict,
                ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                extras: Dict, current_price: float,
                bar_dt: Optional[datetime.datetime] = None,
                raw_candles: Optional[Dict[str, List]] = None):
        """
        Called once per closed 15M candle.
        Signature backward compatible with V7/V9.2 callers — `raw_candles`
        is new and optional (default None), only read when entry_engine is
        "mtf_confluence" (that engine needs whole OHLCV series, current +
        previous bar, for HMA/ROC/MACD cross detection — the ind_* dicts
        here carry only the latest bar's scalar values). Expected shape:
        {"15m": [...], "1h": [...], "4h": [...]} (connectors.base.OHLCV lists).
        """
        now = bar_dt or datetime.datetime.now()
        self._bar_now = now
        self._bar_count += 1
        self._last_candle_15m = candle_15m
        self._raw_candles = raw_candles
        self._scan_info = {}  # never show a stale verdict from the prior bar/regime

        # [BREAKOUT] Expire an armed retest by age, independent of regime — if
        # the regime flips away from Breakout and never returns within the
        # window, the pending would otherwise linger and fire a stale retest
        # on an old level much later.
        if (self._pending_breakout and
                (self._bar_count - self._pending_breakout["created_bar"]) > self.BREAKOUT_RETEST_BARS):
            self._pending_breakout = None

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

        # [V9.3 RECOVERY] A persisted BLOCKED state should not strand the bot
        # when its daily PnL is currently inside both limits (for example after
        # a stale state-file restore or manual balance correction).
        if (self.state == "BLOCKED"
                and self.daily_loss_limit_pct < self.daily_pnl_pct < self.daily_profit_limit_pct):
            self.state = "SCANNING"
            self._log_event("Recovered stale BLOCKED state → SCANNING", level="warning")

        atr_4h = ind_4h.get("atr", 0)
        if atr_4h > 0:
            self.atr_history.append(atr_4h)
            if len(self.atr_history) > 200:
                self.atr_history.pop(0)

        # [EARLY TREND] Fast dual-TF (4H+1H) HMA/MACD/ROC lean — see
        # compute_early_trend. Needs raw candle series (current+previous bar),
        # not just the latest-bar ind_4h/ind_1h scalars, so it's computed here
        # from on_tick's raw_candles rather than inside MacroTrendEngine.
        _early_trend = None
        if self.enable_early_trend and raw_candles and raw_candles.get("4h") and raw_candles.get("1h"):
            _early_trend = compute_early_trend(raw_candles["4h"], raw_candles["1h"])

        # [FAST MACRO EMA] EMA_fast/EMA_slow (e.g. 12/26) computed fresh from
        # raw 4H candles — IndicatorEngine's shared ema20/ema50 are untouched.
        _ema_fast_ovr = _ema_slow_ovr = None
        if self.macro_ema_fast and self.macro_ema_slow and raw_candles and raw_candles.get("4h"):
            _closes_4h = [float(c.close) for c in raw_candles["4h"]]
            if len(_closes_4h) >= self.macro_ema_slow:
                _fast_arr = BaseStrategy.ema(_closes_4h, self.macro_ema_fast)
                _slow_arr = BaseStrategy.ema(_closes_4h, self.macro_ema_slow)
                if not (np.isnan(_fast_arr[-1]) or np.isnan(_slow_arr[-1])):
                    _ema_fast_ovr, _ema_slow_ovr = float(_fast_arr[-1]), float(_slow_arr[-1])

        # [V9] 3-layer macro → context → regime classification
        _l1 = self.macro_engine.compute(
            ind_4h, early_trend=_early_trend,
            ema_fast_override=_ema_fast_ovr, ema_slow_override=_ema_slow_ovr,
        )
        _l2 = self.context_engine.compute(ind_1h)
        _l3 = self.regime_clf.classify(_l1, _l2, ind_15m)
        self.current_market_state = _l3["regime"]
        self.current_regime_bias  = _l1["level"]
        self.regime_score         = _l1["score"]
        self._l1_cache, self._l2_cache, self._l3_cache = _l1, _l2, _l3

        # [MTF-CONFLUENCE] Runs alongside the V9.2 classification above
        # (cheap, and keeps current_market_state/regime_score meaningful for
        # logging/chart-alert display even in this mode) — only the FILTERING
        # branch below actually acts on it when entry_engine is set to it.
        if raw_candles and raw_candles.get("4h") and raw_candles.get("1h"):
            self.mtf_engine.update_macro(raw_candles["4h"])
            self.mtf_engine.update_mid(raw_candles["1h"])

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
                # Log current state distribution so user can see which state dominates
                self._log_event(
                    f"[STATE] {self.current_market_state} | bias={self.current_regime_bias}",
                    level="debug",
                )

                best_signal: Optional[Dict] = None
                if self._entries_enabled and self.entry_engine == "mtf_confluence":
                    best_signal = self._generate_signal_mtf(candle_15m, ind_15m)
                elif self._entries_enabled:
                    for direction in ("LONG", "SHORT"):
                        sig = self._generate_signal(
                            direction, candle_15m, ind_15m, ind_1h, ind_4h,
                            self._l1_cache, self._l2_cache, self._l3_cache,
                        )
                        if sig and (best_signal is None
                                    or sig["total_score"] > best_signal["total_score"]):
                            best_signal = sig

                if best_signal is not None:
                    # Unified signal already validated across all 3 tiers — skip
                    # WAIT_CONFIRM's redundant re-check and go straight to order.
                    self._pending_signal        = best_signal
                    self.direction_focus         = best_signal["direction"]
                    self._last_entry_health      = best_signal["health_score"]
                    self._last_confidence        = best_signal["confidence_score"]
                    self._last_confidence_level  = "PASS"
                    self.state                   = "PENDING_ORDER"
                    state_changed                = True
                    self._log_event(
                        f"Signal: {best_signal['direction']} | {self.current_market_state} "
                        f"| strat={best_signal['strategy']} "
                        f"| total={best_signal['total_score']:.0f} "
                        f"(strat={best_signal['entry_score']:.0f} "
                        f"l2ctx={best_signal['context_score']:.0f} "
                        f"l1fit={best_signal['direction_fit']:.0f})"
                    )
                else:
                    self._pending_signal = None
                    self.state = "SCANNING"

            elif self.state == "WAIT_CONFIRM":
                # Legacy state — the unified signal pipeline always produces a
                # pre-validated signal in FILTERING and skips this state. Kept
                # only so a state file saved before this rewrite resumes safely
                # instead of crashing on the old confirmation logic.
                self.state    = "PENDING_ORDER"
                state_changed = True

            elif self.state == "PENDING_ORDER":
                if self._pending_signal is None:
                    # Defensive: only reachable if a pre-rewrite state file
                    # resumed into WAIT_CONFIRM/PENDING_ORDER with no signal.
                    self._log_event(
                        "PENDING_ORDER with no signal (stale resumed state) → SCANNING",
                        level="warning",
                    )
                    self.state = "SCANNING"
                else:
                    self._step5_risk_engine(
                        candle_15m, self.direction_focus, ind_15m,
                        mr_signal=self._pending_signal,
                    )
                    self._pending_signal = None  # consumed
                    if self.position_open:
                        self.state = "IN_POSITION"
                    else:
                        self.state = "SCANNING"   # risk engine skipped (confidence too low etc.)
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
                # FIX-#4 (close_reason): use the explicit exit_reason stored by _close_position
                # via the 'reason' field if available; fall back to heuristic.
                # Also treat pnl==0 (break-even SL) as SL_HIT so consecutive_sl_hits counts correctly.
                #
                # [LADDER FIX] tp1_hit is now set True by the FIRST ladder level
                # (a cheap +0.5R SL move, not a real partial close) — any LATER
                # close via reversal-spike/emergency/state-drift/poor-health was
                # falling through to the tp-flag heuristic below and getting
                # mislabeled "PARTIAL_TP1", corrupting the lesson system's
                # exit-reason diagnosis. Recognize the real reasons explicitly
                # before ever consulting the (now much less reliable) flags.
                _exit_reason = t.get("exit_reason", "")
                if _exit_reason in ("SL_HIT", "RUNNER_SL"):
                    close_reason = "SL_HIT"
                elif _exit_reason == "T4_HIT":
                    close_reason = "FULL_TP2"   # keep legacy label for stats continuity
                elif _exit_reason in ("REVERSAL_SPIKE", "EMERGENCY_EXIT",
                                      "STATE_DRIFT_EXIT", "POOR_HEALTH_EXIT",
                                      "HEALTH_REDUCE"):
                    close_reason = _exit_reason
                elif t.get("tp3_hit"):
                    close_reason = "TP3_RUNNER"
                elif t.get("tp2_hit"):
                    close_reason = "FULL_TP2"
                elif t.get("tp1_hit"):
                    close_reason = "PARTIAL_TP1"
                elif t.get("exit_price") is not None and pnl <= 0:
                    # break-even or loss exit with no TP hit → SL or emergency exit
                    close_reason = "SL_HIT"
                else:
                    close_reason = "EXIT"
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
            sig = self._generate_signal(direction, candle, ind_15m, ind_1h, ind_4h,
                                        self._l1_cache, self._l2_cache, self._l3_cache)
            if sig:
                self._step5_risk_engine(candle, direction, ind_15m, mr_signal=sig)
                # mirror PENDING_ORDER: the risk engine may skip (low confidence)
                # or fail (ERROR) — only enter IN_POSITION on a real open, and
                # never mask an ERROR state.
                if self.position_open:
                    self.state = "IN_POSITION"
                    self._log_event(f"RECOVERY trade: {direction} at half-risk")
                elif self.state != "ERROR":
                    self.state = "SCANNING"
                break
        self.base_risk_pct = original_risk

    # ── Public API ────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        _now = self._bar_now or datetime.datetime.now()
        warmup_remaining = 0
        if self._startup_unblock_at and _now < self._startup_unblock_at:
            warmup_remaining = int(
                (self._startup_unblock_at - _now).total_seconds() / 60)
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
            "entry_engine":       self.entry_engine,
            "session_state":      self.session_state,
            "session_gate_open":  self.session_gate_open,
            "cooldown_until":     self.cooldown_until.isoformat() if self.cooldown_until else None,
            "range_cooldown_until": self.range_cooldown_until.isoformat() if self.range_cooldown_until else None,
            "range_loss_streak":  self.range_loss_streak,
            "range_entry_allowed": self.range_entry_allowed,
            "warmup_remaining_m": warmup_remaining,
            "scan_info":          dict(self._scan_info),
            "filter_stats":       dict(self._filter_stats),
            "active_entry_config": {
                "tradeable_regimes": sorted(_TRADEABLE_REGIMES),
                "trend_threshold": REGIME_THRESHOLDS["Trend"],
                "breakout_threshold": REGIME_THRESHOLDS["Breakout"],
                "range_threshold": REGIME_THRESHOLDS["Range"],
                "range_risk_multiplier": self.range_risk_multiplier,
                "range_cooldown_minutes": self.range_cooldown_minutes,
                "min_ema_dist_atr": self.MIN_EMA_DIST_ATR,
                "max_15m_adx_trend": self.MAX_15M_ADX_TREND,
                "min_1h_efficiency": self.MIN_1H_EFFICIENCY,
                "blocked_hours_utc": sorted(self.BLOCKED_ENTRY_HOURS_UTC),
            },
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
        pnl       = (current_price - entry) * dir_mult * t.get("remaining_size", t.get("size", 0.0))

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
            "wins":          len(wins),
            "losses":        len(losses),
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
            "range_cooldown_until": self.range_cooldown_until.isoformat() if self.range_cooldown_until else None,
            "range_loss_streak":    self.range_loss_streak,
            "last_close_at":        self._last_close_at.isoformat() if self._last_close_at else None,
            "trading_date":         self.trading_date.isoformat() if self.trading_date else None,
            "current_market_state": self.current_market_state,
            "current_regime_bias":  self.current_regime_bias,
            "regime_score":         self.regime_score,
            "direction_focus":      self.direction_focus,
            "bars_since_trigger":   self.bars_since_trigger,
            "atr_history":          self.atr_history[-200:],
            "adaptive_weights":     self.adaptive_engine.base_weights,
            "pattern_learning":     self.learning_engine.to_dict(),
            # [LEVEL 2/3] condition-tag stats + any active temporary tightening
            "condition_learning":   self.condition_engine.to_dict(),
            # [V9.1] per (regime, strategy) expectancy tracking
            "expectancy":           self.expectancy_engine.to_dict(),
            "active_strategy_adjustments": {
                tag: until.isoformat() for tag, until in self._active_strategy_adjustments.items()
            },
            # [TARGET ALERTS] persist queued-but-not-yet-sent alerts so a
            # crash/restart between this save and the runner popping them
            # (run_bot.py's _send_target_alerts) doesn't silently drop a
            # Telegram notification for a ladder level that already fired.
            "pending_target_alerts": self._pending_target_alerts,
            # [MTF-CONFLUENCE] in-flight setup state (bar-window based, so a
            # restart mid-collection would otherwise silently drop it)
            "mtf_engine":           self.mtf_engine.to_dict(),
            # [STATS] Closed-trade history — was never persisted, so every
            # restart (redeploy, crash, manual restart) silently wiped
            # /stats back to 0 trades even though trades had actually
            # closed. Capped at the most recent 200 to keep the state file
            # bounded; /stats only ever needs recent history anyway.
            "trade_journal":        self.trade_journal[-200:],
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
        range_cooldown = data.get("range_cooldown_until")
        self.range_cooldown_until = (datetime.datetime.fromisoformat(range_cooldown)
                                     if range_cooldown else None)
        self.range_loss_streak = int(data.get("range_loss_streak", 0) or 0)

        last_close = data.get("last_close_at", data.get("last_entry_at"))  # tolerate old field name
        self._last_close_at = (datetime.datetime.fromisoformat(last_close)
                               if last_close else None)

        trading_date = data.get("trading_date")
        self.trading_date = (datetime.date.fromisoformat(trading_date)
                             if trading_date else None)

        self.current_market_state  = data.get("current_market_state", "Range")
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

        cl = data.get("condition_learning")
        if cl:
            self.condition_engine.from_dict(cl)

        ex = data.get("expectancy")
        if ex:
            self.expectancy_engine.from_dict(ex)

        self._active_strategy_adjustments = {
            tag: datetime.datetime.fromisoformat(until)
            for tag, until in (data.get("active_strategy_adjustments") or {}).items()
        }

        self._pending_target_alerts = data.get("pending_target_alerts", [])

        self.mtf_engine.load_dict(data.get("mtf_engine"))

        self.trade_journal = data.get("trade_journal", [])

        self._log_event(
            f"State loaded | state={self.state} pos={self.position_open} "
            f"balance={self.account_balance:.2f} saved_at={data.get('saved_at')}"
        )
        return True

    def _backfill_reconciled_trade(self, symbol: str, exchange_adapter) -> None:
        """A position that closes via an exchange-side order (the attached
        TP2/SL) while the bot is offline or mid-restart is caught by
        reconcile_with_exchange AFTER the fact — _log_trade() only ever runs
        from the bot's own on_tick EXITING state, so it never fires here,
        and the trade would otherwise vanish from trade_journal entirely
        (this is why /stats' TP1/TP2/SL-hit breakdown can undercount OKX's
        real trade total — e.g. 3/9 instead of 9/9, every restart drops
        whatever closed while the process was down).
        Backfills a best-effort entry using OKX's own closed-position history
        for the real exit price/pnl, and infers 'targets_hit' from where
        that close price landed relative to this trade's own tp1/tp2 —
        purely additive to whatever the bot already saw locally before going
        offline (never removes a hit it already recorded)."""
        t = self.current_trade
        if not t or not hasattr(exchange_adapter, "fetch_closed_positions_history"):
            return
        entry_dt = t.get("entry_time")
        if not isinstance(entry_dt, datetime.datetime):
            entry_dt = datetime.datetime.now(datetime.timezone.utc)
        elif entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=datetime.timezone.utc)
        try:
            since_ms = int(entry_dt.timestamp() * 1000) - 60_000
            rows = exchange_adapter.fetch_closed_positions_history(
                since_ms=since_ms, symbols=[symbol])
        except Exception as e:
            self._log_event(f"[RECONCILE] OKX history backfill failed (non-fatal): {e}",
                            level="warning")
            return
        if not rows:
            return

        row         = rows[-1]  # most recent close for this symbol since entry
        direction   = t.get("direction", "LONG")
        dir_mult    = 1 if direction == "LONG" else -1
        close_price = float(row.get("close_price") or 0.0)
        tp1         = float(t.get("tp1") or 0.0)
        tp2         = float(t.get("tp2") or 0.0)
        targets_hit = list(t.get("targets_hit", []))
        pnl = float(row.get("pnl") or 0.0)
        close_ts = row.get("close_ts")

        # [T2 DETECTION] The position's AVERAGE close price (row["close_price"]
        # = OKX closeAvgPx) can't tell whether the runner hit T2: on a
        # T1(60%)+T2(40%) split the big T1 partial drags the average below T2
        # even when the runner closed exactly at T2 — which is why /stats
        # showed TP2 hit 0/N despite trades clearly reaching their exchange-
        # attached TP2. Use the most-FAVORABLE individual close FILL instead
        # (max sell for LONG / min buy for SHORT); if any close reached T2,
        # that fill did. Falls back to the average when the per-fill lookup
        # isn't available (paper/backtest, fetch failure).
        ref_price = close_price
        try:
            entry_ms = int(entry_dt.timestamp() * 1000)
            best = None
            if hasattr(exchange_adapter, "fetch_best_close_price"):
                best = exchange_adapter.fetch_best_close_price(
                    symbol, entry_ms, int(close_ts) if close_ts else None, direction)
            if best is not None and best > 0:
                # Most-favorable = furthest in the trade's direction, so take
                # whichever of avg/best is further along for T-level detection.
                ref_price = max(ref_price, best) if direction == "LONG" else min(ref_price, best)
        except Exception as e:
            self._log_event(f"[RECONCILE] best-close lookup failed (non-fatal): {e}",
                            level="warning")

        if tp2 and dir_mult * (ref_price - tp2) >= 0 and "T2" not in targets_hit:
            targets_hit.append("T2")
        if tp1 and dir_mult * (ref_price - tp1) >= 0 and "T1" not in targets_hit:
            targets_hit.append("T1")
        closed_at = (
            datetime.datetime.fromtimestamp(close_ts / 1000, tz=datetime.timezone.utc)
            if close_ts else datetime.datetime.now(datetime.timezone.utc)
        ).isoformat()

        self.trade_journal.append({
            "symbol":      symbol,
            "direction":   direction,
            "entry":       t.get("entry"),
            "exit":        close_price,
            "sl":          t.get("sl"),
            "tp1":         tp1,
            "tp2":         tp2,
            "win_loss":    "WIN" if pnl > 0 else "LOSS",
            "pnl":         pnl,
            "targets_hit": targets_hit,
            "exit_reason": "RECONCILE_EXTERNAL_CLOSE",
            "closed_at":   closed_at,
        })
        self.trade_journal = self.trade_journal[-200:]
        self._log_event(
            f"[RECONCILE] backfilled trade_journal from OKX history: "
            f"targets_hit={targets_hit} pnl={pnl:+.2f}"
        )

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

        # [ERROR RECOVERY] A FAILED OPEN (order rejected/errored before
        # position_open was ever set True) also lands in ERROR, but neither
        # branch below fires for it (position_open is already False) — so it
        # was never cleared and persisted across restarts forever. Any time
        # the exchange confirms flat, a stale ERROR is safe to clear.
        if self.state == "ERROR" and live_size == 0 and not self.position_open:
            self._log_event(
                "[RECONCILE] ERROR with exchange flat (failed open never "
                "cleared) → SCANNING",
                level="warning",
            )
            self.state = "SCANNING"
            return

        if self.position_open and live_size == 0:
            self._log_event(
                "[RECONCILE] local=open but exchange=no position → closed offline; "
                "clearing local state → SCANNING",
                level="warning",
            )
            self._backfill_reconciled_trade(symbol, exchange_adapter)
            self.position_open = False
            self.current_trade = {}
            # [WHIPSAW GUARD] A close that happens outside _close_position()
            # (manual close on the exchange, or an exchange-side TP/SL that
            # fired while the bot was offline) never stamped _last_close_at —
            # the entry-spacing cooldown was silently bypassed for every
            # externally-closed position. Stamp it here too, same as a
            # bot-initiated close.
            self._last_close_at = datetime.datetime.now(datetime.timezone.utc)
            if self.state == "ERROR":
                # A close that raced an exchange-side TP/SL fill lands here:
                # exchange is flat, local cleared — the error is stale, recover.
                self._log_event("[RECONCILE] exchange flat — clearing stale ERROR → SCANNING",
                                level="warning")
                self.state = "SCANNING"
            elif self.state not in ("BLOCKED", "COOLDOWN"):
                self.state = "SCANNING"

        elif not self.position_open and live_size != 0:
            # BUG FIX: ccxt's unified `contracts` field is UNSIGNED for both
            # long and short positions (OKX hedge mode reports size as a
            # positive magnitude regardless of side) — inferring direction
            # from its sign always resolved to "LONG". Read the actual side
            # instead: ccxt unified `side` ('long'/'short'), falling back to
            # OKX's raw `info.posSide`.
            raw_side = str(
                (live_position or {}).get("side")
                or ((live_position or {}).get("info") or {}).get("posSide")
                or ""
            ).lower()
            direction = "SHORT" if raw_side == "short" else "LONG"
            entry_price = float(
                live_position.get("entryPrice") or live_position.get("entryPx") or 0
            )
            # [UNITS FIX] ccxt's `contracts` is the raw exchange contract
            # count (e.g. OKX's "pos"), NOT base-currency coins — using it
            # directly as size/remaining_size understated or overstated every
            # PnL calc and, worse, any later PARTIAL close (e.g.
            # HEALTH_REDUCE's 50%) computes its order size from
            # remaining_size and would send the wrong real quantity to the
            # exchange. CLOSE_FULL is safe regardless (it re-fetches the live
            # contract count directly) but nothing else is.
            real_size = abs(live_size)
            try:
                get_ct_val = getattr(exchange_adapter, "_get_ct_val", None)
                if get_ct_val is not None:
                    ct_val = float(get_ct_val(symbol))
                    if ct_val > 0:
                        real_size = abs(live_size) * ct_val
            except Exception as e:
                self._log_event(f"[RECONCILE] ct_val lookup failed, using raw contract count as size: {e}", level="warning")
            self._log_event(
                f"[RECONCILE] exchange has unknown {direction} pos (size={abs(live_size)}) "
                "→ creating trade record from exchange data",
                level="warning",
            )
            # Recover the REAL attached SL price from the exchange (the same
            # algo-order lookup used for SL amends) instead of guessing an
            # arbitrary placeholder — a guessed sl_dist corrupts every target
            # level derived from it (T1/T2 both computed as entry + sl_dist*R)
            # for the rest of the trade. Falls back to the old 1%-of-price
            # guess only if the real SL can't be read back (adapter doesn't
            # support the lookup, or the algo order isn't found).
            real_stop = None
            try:
                pos_side_r = "long" if direction == "LONG" else "short"
                fetch_fn = getattr(exchange_adapter, "fetch_attached_sl_tp", None)
                if fetch_fn is not None:
                    real_stop = fetch_fn(symbol, pos_side_r)
            except Exception as e:
                self._log_event(f"[RECONCILE] fetch_attached_sl_tp failed: {e}", level="warning")

            if real_stop and real_stop.get("sl"):
                sl_price  = float(real_stop["sl"])
                sl_dist_r = abs(entry_price - sl_price)
                self._log_event(
                    f"[RECONCILE] recovered real SL={sl_price:.4f} from exchange "
                    f"(sl_dist={sl_dist_r:.4f})"
                )
            else:
                sl_price  = entry_price * (0.99 if direction == "LONG" else 1.01)
                sl_dist_r = max(entry_price * 0.01, 1e-9)
                self._log_event(
                    "[RECONCILE] could not recover real SL — falling back to "
                    f"1% placeholder (sl_dist={sl_dist_r:.4f})",
                    level="warning",
                )
            mult_r    = 1 if direction == "LONG" else -1
            ladder_r  = self._target_ladder()
            # FIX-#9: include all fields _manage_open_position expects
            self.current_trade = {
                "direction": direction, "entry": entry_price,
                "sl": sl_price,
                "sl_dist": sl_dist_r,
                "tp1": entry_price + sl_dist_r * ladder_r[0][0] * mult_r,
                "tp2": entry_price + sl_dist_r * ladder_r[-1][0] * mult_r,
                "tp3": None,
                "targets": ladder_r, "next_target_idx": 0, "targets_hit": [],
                "sl_algo_id": (real_stop or {}).get("algo_id"),
                "tp1_pct": 0.50, "tp2_pct": 1.0,
                "trail_atr_mult": 2.0,
                "size": real_size, "remaining_size": real_size,
                # Same fields the normal OPEN flow gets from the real OKX
                # fill (_step5_risk_engine) — a reconciled position never ran
                # that flow, so approximate from what's known here (Telegram
                # /stats' "Open" line would otherwise show margin=$0).
                "entry_order_value": entry_price * real_size,
                "entry_margin": (entry_price * real_size) / max(self.sizing_leverage, 1),
                "tp1_hit": False, "tp2_hit": False, "tp3_hit": False,
                "break_even_triggered": False, "status": "OPEN",
                "entry_time": datetime.datetime.now(datetime.timezone.utc),
                "atr_at_entry": sl_dist_r,
                "realized_pnl": 0.0, "exit_price": None, "exit_reason": None,
                "final_rr": None, "mae": 0.0, "mfe": 0.0,
                "entry_health": 0.0, "entry_confidence": 0.0,
                "entry_type": "reconcile", "strategy": "Reconcile",
            }
            self.position_open = True
            self.state         = "IN_POSITION"
