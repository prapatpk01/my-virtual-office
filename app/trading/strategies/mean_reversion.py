"""
Mean Reversion Strategy — 18-step signal generator
====================================================
Designed for: SIDEWAY / EXHAUSTION / LOW_VOL (Compression) market states.
Forbidden in: TRENDING / STRONG_TREND / BREAKOUT / HIGH_VOL

TP structure (same as SwingReversalPro):
  TP1 = 0.7R → close 50% → SL to Break-Even
  TP2 = 1.5R → close remaining 50%

SL = max(ATR×1.5, sweep_low/high)

Steps:
  1  Market Regime Filter (4H) — need ≥3/7 non-trending conditions
  2  HTF Bias (1H + 4H) — detect long/short lean
  3  Over-Extension Filter — price stretched from mean ≥1.5×ATR or BB/VWAP extreme
  4  Liquidity Sweep — false breakdown/breakout + re-entry
  5  Reversal Structure — CHOCH / Minor BOS signal
  6  Momentum Reversal — MACD hist + RSI + EMA cross
  7  Candle Quality — hammer / engulf / strong rejection (body ≥ 60%)
  8  Volume Filter — spike ≥1.2× avg, OBV / delta confirmation
  9  Distance Filter — ≥1.5R room to mean, ≥1R room to EMA20
  10 Session Filter — London/NY/Overlap full, Asia -5 pts
  11 Funding Filter — contrarian funding bonus
  12 Open Interest Filter — OI rising with price divergence
  13 Entry Health Score (0-100) — gates and size multiplier
  14-16: SL/TP/Health → handled by TradingBot (via tp_config in signal)
  17 Emergency Exit conditions returned for TradingBot to check
  18 Trade Journal fields returned in signal metadata
"""

import datetime
import math
from typing import Any, Dict, List, Optional

import numpy as np


# ── States where Mean Reversion is applicable ──────────────────────────────────
_MR_ALLOWED_STATES = frozenset({"SIDEWAY", "EXHAUSTION", "LOW_VOL"})
_MR_FORBIDDEN_STATES = frozenset({"TRENDING", "STRONG_TREND", "BREAKOUT", "HIGH_VOL"})


class MeanReversionStrategy:
    """
    18-step Mean Reversion signal generator.

    Call generate_signal() each tick (15M candle close).
    Returns a signal dict or None.

    Signal dict keys:
      direction       : "LONG" | "SHORT"
      health_score    : 0-100 (Step 13)
      size_mult       : 1.0 / 0.80 / 0.50 (Step 13 gate)
      sl_price        : computed stop-loss price
      sl_method       : "sweep" | "atr"
      tp_config       : TP levels and close fractions for TradingBot
      step_notes      : list of passed/failed step notes (for logging)
      metadata        : extra fields for trade journal (Step 18)
    """

    # Health score weight map (Step 13)
    _HEALTH_WEIGHTS = {
        "regime":    15,  # Step 1
        "extension": 20,  # Step 3
        "sweep":     20,  # Step 4
        "structure": 15,  # Step 5
        "momentum":  15,  # Step 6
        "volume":    10,  # Step 8
        "pattern":    5,  # Step 7
    }

    # BB width — compute bounds from IndicatorEngine's bb_width field
    # bb_width = 4*std/mean  →  2σ = bb_width * mean / 2
    @staticmethod
    def _bb_bounds(ind: Dict) -> tuple:
        ema20   = ind.get("ema20", 0.0)
        bb_w    = ind.get("bb_width", 0.04)
        two_sig = bb_w * ema20 / 2.0
        return ema20 + two_sig, ema20 - two_sig  # upper, lower

    # VWAP proxy (ema50) + ±2σ (±2*atr)
    @staticmethod
    def _vwap_bounds(ind: Dict) -> tuple:
        vwap = ind.get("ema50", ind.get("ema20", 0.0))
        atr  = ind.get("atr", 1.0)
        return vwap + 2.0 * atr, vwap - 2.0 * atr  # upper, lower

    # ── STEP 1: Market Regime Filter (4H) ─────────────────────────────────────

    def _step1_regime(self, ind_4h: Dict) -> tuple:
        """Pass if ≥4 of 7 non-trending conditions are met. Returns (ok, score, count)."""
        adx      = ind_4h.get("adx", 25.0)
        slope    = ind_4h.get("ema20_slope_score", 50.0)   # 45-55 = flat
        eff      = ind_4h.get("eff_ratio", 0.5)
        atr_exp  = ind_4h.get("atr_exp", 1.0)
        bb_w     = ind_4h.get("bb_width", 0.04)
        ema20    = ind_4h.get("ema20", 0.0)
        ema50    = ind_4h.get("ema50", ema20)
        price    = ind_4h.get("close", ema20)
        atr      = ind_4h.get("atr", 1.0)

        checks = [
            ("adx<20",         adx < 20),
            ("slope_flat",     44 < slope < 56),
            ("eff<0.45",       eff < 0.45),
            ("atr_stable",     atr_exp < 1.1),          # loosened from 1.0
            ("bb_low",         bb_w < 0.08),            # loosened from 0.06
            ("near_ema50",     abs(price - ema50) < atr * 2.0),
            ("no_breakout",    atr_exp < 1.15),         # no strong expansion
        ]
        passed = [name for name, ok in checks if ok]
        count  = len(passed)
        return count >= 3, count, passed  # loosened from >=4 to >=3

    # ── STEP 2: Higher-Timeframe Bias ──────────────────────────────────────────

    def _step2_htf_bias(self, ind_1h: Dict, ind_4h: Dict) -> Optional[str]:
        """Return 'LONG', 'SHORT', or None (neutral/unclear)."""
        rsi_1h    = ind_1h.get("rsi", 50.0)
        macd_1h   = ind_1h.get("macd_hist", 0.0)
        slope_1h  = ind_1h.get("ema20_slope_score", 50.0)
        adx_4h    = ind_4h.get("adx", 25.0)
        ema20_4h  = ind_4h.get("ema20", 0.0)
        ema50_4h  = ind_4h.get("ema50", ema20_4h)
        atr_4h    = ind_4h.get("atr", 1.0)

        # 4H: not strongly trending
        if adx_4h >= 22:
            return None
        # 4H: EMA20 ≈ EMA50 (not separated by more than 1×ATR)
        if abs(ema20_4h - ema50_4h) > atr_4h * 1.5:
            return None

        # 1H conditions
        slope_flat = 43 < slope_1h < 57
        if rsi_1h < 45 and macd_1h > 0 and slope_flat:
            return "LONG"
        if rsi_1h > 55 and macd_1h < 0 and slope_flat:
            return "SHORT"
        # Weak bias — allow but note
        if rsi_1h < 48:
            return "LONG"
        if rsi_1h > 52:
            return "SHORT"
        return None

    # ── STEP 3: Over-Extension Filter ──────────────────────────────────────────

    def _step3_overextension(self, ind: Dict, candle: Dict, direction: str) -> bool:
        """Price must be stretched ≥1.5×ATR from EMA20, or at BB/VWAP extreme."""
        # FIX-#3: 'close' is not in the indicator dict — read from the candle
        close  = float(candle.get("close", 0.0))
        ema20  = ind.get("ema20", close)
        atr    = ind.get("atr", 1.0)
        bb_up, bb_lo = self._bb_bounds(ind)
        vp_up, vp_lo = self._vwap_bounds(ind)

        if direction == "LONG":
            cond_atr  = close < ema20 - atr * 1.5
            cond_bb   = close <= bb_lo
            cond_vwap = close <= vp_lo
        else:
            cond_atr  = close > ema20 + atr * 1.5
            cond_bb   = close >= bb_up
            cond_vwap = close >= vp_up

        return cond_atr or cond_bb or cond_vwap

    # ── STEP 4: Liquidity Sweep ──────────────────────────────────────────────────

    def _step4_sweep(self, ind: Dict, candle: Dict, direction: str) -> bool:
        """
        Detect false breakdown (LONG) or false breakout (SHORT).
        Approximated from sweep_score (wick ratio) + price returning inside BB.
        """
        sweep_score = ind.get("sweep_score", 0.0)
        close       = ind.get("close", candle.get("close", 0.0))
        bb_up, bb_lo = self._bb_bounds(ind)
        low         = candle.get("low",  close)
        high        = candle.get("high", close)

        if direction == "LONG":
            # Wick swept below BB lower but closed back inside
            swept_below = low < bb_lo
            return_in   = close > bb_lo
            return sweep_score > 40 and (swept_below and return_in)
        else:
            swept_above = high > bb_up
            return_in   = close < bb_up
            return sweep_score > 40 and (swept_above and return_in)

    # ── STEP 5: Reversal Structure (CHOCH / Minor BOS) ─────────────────────────

    def _step5_structure(self, ind: Dict, direction: str) -> bool:
        """
        CHOCH proxy: divergence_score high (price moved one way but RSI diverges)
        + structure_score showing nascent reversal.
        """
        divergence_score = ind.get("divergence_score", 50.0)
        structure_score  = ind.get("structure_score", 50.0)
        sweep_score      = ind.get("sweep_score", 0.0)

        # High divergence = price/RSI disagree → reversal signal
        if direction == "LONG":
            struct_ok = structure_score > 30   # price was depressed, now trying to rise
        else:
            struct_ok = structure_score < 70   # price was elevated, now fading

        return divergence_score > 60 and sweep_score > 30 and struct_ok

    # ── STEP 6: Momentum Reversal ───────────────────────────────────────────────

    def _step6_momentum(self, ind: Dict, direction: str) -> bool:
        """MACD hist turning + RSI confirming + EMA5 crossing EMA20 direction."""
        macd_hist = ind.get("macd_hist", 0.0)
        rsi       = ind.get("rsi", 50.0)
        ema5      = ind.get("ema5", 0.0)
        ema20     = ind.get("ema20", 1.0)
        adx       = ind.get("adx", 25.0)

        if direction == "LONG":
            return (macd_hist > 0 and rsi > 35 and rsi < 55
                    and ema5 >= ema20 * 0.998 and adx < 25)
        else:
            return (macd_hist < 0 and rsi < 65 and rsi > 45
                    and ema5 <= ema20 * 1.002 and adx < 25)

    # ── STEP 7: Candle Quality ────────────────────────────────────────────────

    def _step7_candle(self, candle: Dict, direction: str) -> bool:
        """
        Body ≥ 60% of range, close near appropriate end.
        Also catches hammer / bullish engulf patterns via sweep_score.
        """
        high  = float(candle.get("high",  0.0))
        low   = float(candle.get("low",   0.0))
        open_ = float(candle.get("open",  0.0))
        close = float(candle.get("close", 0.0))

        total_range = max(high - low, 1e-9)
        body        = abs(close - open_)
        body_pct    = body / total_range

        upper_wick = high - max(close, open_)
        lower_wick = min(close, open_) - low

        if direction == "LONG":
            hammer    = lower_wick > body * 2 and close > open_  # hammer
            rejection = close >= low + total_range * 0.65         # close near high
            marubozu  = body_pct >= 0.60 and close > open_
            return hammer or rejection or marubozu
        else:
            inv_hammer = upper_wick > body * 2 and close < open_
            rejection  = close <= high - total_range * 0.65
            marubozu   = body_pct >= 0.60 and close < open_
            return inv_hammer or rejection or marubozu

    # ── STEP 8: Volume Filter ────────────────────────────────────────────────

    def _step8_volume(self, ind: Dict) -> bool:
        """Volume spike ≥1.2× avg, OBV direction (divergence_score proxy)."""
        volume    = ind.get("volume", 0.0)
        vol_avg   = ind.get("vol_avg", volume if volume > 0 else 1.0)
        vol_score = ind.get("volume_score", 0.0)

        vol_spike = volume >= vol_avg * 1.2
        vol_active = vol_score >= 55
        return vol_spike or vol_active

    # ── STEP 9: Distance Filter ────────────────────────────────────────────────

    def _step9_distance(self, ind: Dict, direction: str,
                        sl_price: float, entry_price: float) -> bool:
        """
        ≥1.5R room to mean (EMA20) and ≥1R room to EMA20.
        R = |entry - sl|
        """
        ema20  = ind.get("ema20", entry_price)
        ema50  = ind.get("ema50", ema20)
        sl_r   = max(abs(entry_price - sl_price), 1e-9)

        # Distance to mean (EMA20)
        if direction == "LONG":
            dist_mean = ema20 - entry_price
        else:
            dist_mean = entry_price - ema20

        # Must be at least 1.5R to mean, 1R to EMA50
        if direction == "LONG":
            dist_vwap = ema50 - entry_price
        else:
            dist_vwap = entry_price - ema50

        mean_ok = dist_mean >= sl_r * 1.5
        vwap_ok = dist_vwap >= sl_r * 1.0
        return mean_ok or vwap_ok

    # ── STEP 10: Session Filter ────────────────────────────────────────────────

    @staticmethod
    def _step10_session(bar_dt: Optional[datetime.datetime]) -> tuple:
        """Returns (allowed, pts_adjustment).  Asia = -5 pts."""
        if bar_dt is None:
            return True, 0
        h = bar_dt.hour  # UTC hour
        if 7 <= h < 10:   return True, +5   # London open
        if 12 <= h < 17:  return True, +5   # New York / Overlap
        if 10 <= h < 12:  return True, 0    # London morning
        if 17 <= h < 20:  return True, 0    # NY afternoon
        return True, -5                       # Asia — allowed but -5 pts

    # ── STEP 11: Funding Filter ────────────────────────────────────────────────

    @staticmethod
    def _step11_funding(funding_rate: Optional[float], direction: str) -> tuple:
        """
        Best: LONG when funding negative (shorts pay longs).
              SHORT when funding positive.
        Extreme (|funding| > 0.05%): skip.
        Returns (allowed, pts_bonus).
        """
        if funding_rate is None:
            return True, 0
        f = float(funding_rate)
        if abs(f) > 0.0005:      # 0.05% = extreme → skip
            return False, 0
        if direction == "LONG"  and f < 0:  return True, +5
        if direction == "SHORT" and f > 0:  return True, +5
        return True, 0

    # ── STEP 12: Open Interest Filter ──────────────────────────────────────────

    @staticmethod
    def _step12_oi(oi: Optional[float], oi_prev: Optional[float],
                   direction: str, price_down: bool) -> tuple:
        """
        LONG:  price fell + OI rose → short squeeze potential → bonus.
        SHORT: price rose + OI rose → long trap potential → bonus.
        Returns (allowed, pts_bonus).
        """
        if oi is None or oi_prev is None:
            return True, 0
        oi_rising = float(oi) > float(oi_prev) * 1.01
        if direction == "LONG"  and price_down and oi_rising: return True, +5
        if direction == "SHORT" and not price_down and oi_rising: return True, +5
        return True, 0

    # ── STEP 13: Entry Health Score (0-100) ────────────────────────────────────

    def _step13_health(self, component_scores: Dict[str, float],
                       session_adj: int, funding_adj: int, oi_adj: int) -> float:
        """
        Aggregate Step 1-12 results into 0-100 score.
        Size gate: ≥85→1.0, 75-84→0.80, 70-74→0.50, <70→None (skip)
        """
        score = 0.0
        for key, weight in self._HEALTH_WEIGHTS.items():
            raw = component_scores.get(key, 0.0)
            score += float(np.clip(raw, 0, 100)) * weight / 100.0

        # Session, funding, OI adjustments (±5 pts each)
        score += session_adj + funding_adj + oi_adj
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _size_from_health(health: float) -> Optional[float]:
        if health >= 85: return 1.00
        if health >= 75: return 0.80
        if health >= 70: return 0.50
        return None  # skip

    # ── STEP 14: Stop-Loss Computation ─────────────────────────────────────────

    def _step14_sl(self, ind: Dict, candle: Dict, direction: str) -> tuple:
        """
        SL = max(ATR×1.5,  sweep_low/high from pattern candle).
        Returns (sl_price, sl_method).
        """
        atr   = ind.get("atr", 1.0)
        close = float(candle.get("close", 0.0))
        low   = float(candle.get("low",   close))
        high  = float(candle.get("high",  close))

        swing_lo = float(candle.get("pattern_low",  low))
        swing_hi = float(candle.get("pattern_high", high))

        if direction == "LONG":
            sl_atr   = close - atr * 1.5
            sl_sweep = swing_lo - atr * 0.3   # slight buffer below sweep low
            sl_price = min(sl_atr, sl_sweep)  # wider of the two = more conservative
            method   = "sweep" if sl_sweep < sl_atr else "atr"
        else:
            sl_atr   = close + atr * 1.5
            sl_sweep = swing_hi + atr * 0.3
            sl_price = max(sl_atr, sl_sweep)
            method   = "sweep" if sl_sweep > sl_atr else "atr"

        return sl_price, method

    # ── Public: Generate Signal ─────────────────────────────────────────────────

    def generate_signal(
        self,
        candle_15m: Dict,
        ind_15m: Dict,
        ind_1h: Dict,
        ind_4h: Dict,
        market_state: str,
        extras: Optional[Dict] = None,
        bar_dt: Optional[datetime.datetime] = None,
    ) -> Optional[Dict]:
        """
        Run all 18 steps. Returns signal dict or None.
        """
        extras = extras or {}

        # State gate
        if market_state in _MR_FORBIDDEN_STATES:
            return None

        # Step 1: 4H Regime
        regime_ok, regime_count, regime_passed = self._step1_regime(ind_4h)
        if not regime_ok:
            return None

        # Step 2: HTF Bias
        bias = self._step2_htf_bias(ind_1h, ind_4h)
        directions = [bias] if bias else ["LONG", "SHORT"]

        # Extras
        funding_rate = extras.get("funding_rate")
        oi_now       = extras.get("open_interest")
        oi_prev      = extras.get("oi_prev")

        close        = float(candle_15m.get("close", ind_15m.get("close", 0.0)))
        close_prev   = float(extras.get("close_prev", close))
        price_down   = close < close_prev

        best: Optional[Dict] = None

        for direction in directions:

            # Step 3: Over-extension
            ext_ok = self._step3_overextension(ind_15m, candle_15m, direction)

            # SL estimate for distance filter
            sl_price, sl_method = self._step14_sl(ind_15m, candle_15m, direction)
            entry_price = close

            # Step 4: Sweep
            sweep_ok = self._step4_sweep(ind_15m, candle_15m, direction)

            # Step 5: Structure (CHOCH / BOS)
            struct_ok = self._step5_structure(ind_15m, direction)

            # Step 6: Momentum reversal
            mom_ok = self._step6_momentum(ind_15m, direction)

            # Step 7: Candle quality
            candle_ok = self._step7_candle(candle_15m, direction)

            # Step 8: Volume
            vol_ok = self._step8_volume(ind_15m)

            # Step 9: Distance filter
            dist_ok = self._step9_distance(ind_15m, direction, sl_price, entry_price)

            # Step 10: Session
            session_allowed, session_adj = self._step10_session(bar_dt)
            if not session_allowed:
                continue

            # Step 11: Funding
            funding_allowed, funding_adj = self._step11_funding(funding_rate, direction)
            if not funding_allowed:
                continue

            # Step 12: OI
            oi_allowed, oi_adj = self._step12_oi(oi_now, oi_prev, direction, price_down)
            if not oi_allowed:
                continue

            # Step 13: Entry Health Score
            component_scores = {
                "regime":    min(100.0, regime_count / 7 * 100),
                "extension": 100.0 if ext_ok  else 0.0,
                "sweep":     100.0 if sweep_ok else 0.0,
                "structure": 100.0 if struct_ok else 0.0,
                "momentum":  100.0 if mom_ok  else 0.0,
                "volume":    100.0 if vol_ok  else 0.0,
                "pattern":   100.0 if candle_ok else 0.0,
            }
            health = self._step13_health(
                component_scores, session_adj, funding_adj, oi_adj)
            size_mult = self._size_from_health(health)
            if size_mult is None:
                continue

            # Step 9: Distance must be met for risk-reward
            if not dist_ok:
                continue

            # All hard gates passed — build TP config (Step 15)
            sl_dist = abs(entry_price - sl_price)
            mult    = 1 if direction == "LONG" else -1

            # TP targets — unified with SwingReversal: TP1=0.7R/50%, TP2=1.5R/100%
            tp1 = entry_price + sl_dist * 0.7 * mult
            tp2 = entry_price + sl_dist * 1.5 * mult

            step_notes = [
                f"regime={regime_count}/7({','.join(regime_passed[:3])})",
                f"ext={'✓' if ext_ok else '✗'}",
                f"sweep={'✓' if sweep_ok else '✗'}",
                f"struct={'✓' if struct_ok else '✗'}",
                f"mom={'✓' if mom_ok else '✗'}",
                f"vol={'✓' if vol_ok else '✗'}",
                f"candle={'✓' if candle_ok else '✗'}",
                f"dist={'✓' if dist_ok else '✗'}",
                f"health={health:.0f} size×{size_mult:.2f}",
            ]

            # FIX-#10: separate confidence_score from health_score
            # Confidence = quality of HTF+structure+momentum alignment (conviction)
            confidence_score = (
                (100.0 if struct_ok else 0.0) * 0.40 +
                (100.0 if mom_ok   else 0.0) * 0.35 +
                (75.0  if bias is not None else 50.0) * 0.25
            )

            signal = {
                "strategy":        "MeanReversion",
                "direction":       direction,
                "entry":           entry_price,
                "sl_price":        sl_price,
                "sl_method":       sl_method,
                "health_score":    health,
                "confidence_score": confidence_score,
                "size_mult":       size_mult,
                "tp_config": {
                    "tp1":    tp1, "tp1_pct": 0.50,   # close 50% at 0.7R
                    "tp2":    tp2, "tp2_pct": 1.00,   # close 100% at 1.5R
                    "tp1_r":  0.7, "tp2_r":   1.5,
                    "trail_atr_mult": 2.0,
                },
                "step_notes": step_notes,
                # Step 18: Trade Journal metadata
                "metadata": {
                    "strategy":      "mean_reversion",
                    "regime_count":  regime_count,
                    "ext_ok":        ext_ok,
                    "sweep_ok":      sweep_ok,
                    "struct_ok":     struct_ok,
                    "mom_ok":        mom_ok,
                    "vol_ok":        vol_ok,
                    "candle_ok":     candle_ok,
                    "htf_bias":      bias,
                    "session_adj":   session_adj,
                    "funding_rate":  funding_rate,
                    "oi":            oi_now,
                    "entry_health":  health,
                    "bb_upper":      self._bb_bounds(ind_15m)[0],
                    "bb_lower":      self._bb_bounds(ind_15m)[1],
                    "ema20_dist_atr": abs(entry_price - ind_15m.get("ema20", entry_price)) / max(ind_15m.get("atr", 1), 1e-9),
                    "vwap_dist_atr":  abs(entry_price - ind_15m.get("ema50", entry_price)) / max(ind_15m.get("atr", 1), 1e-9),
                    "rsi":            ind_15m.get("rsi"),
                    "macd_hist":      ind_15m.get("macd_hist"),
                    "adx":            ind_15m.get("adx"),
                },
            }

            # Pick the highest health signal across both directions
            if best is None or health > best["health_score"]:
                best = signal

        return best
