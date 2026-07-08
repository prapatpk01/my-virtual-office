"""
Layer 2: 1H Context & Bias Engine

Analyzes 1H candles to classify the current market context and directional bias.

Score components (100 pts, scored separately for bull and bear):
  EMA Pullback   15 pts  — price pulling back to EMA (continuation setup)
  RSI            10 pts  — oversold/overbought extremes
  MACD           10 pts  — momentum direction and crossovers
  Volume         10 pts  — volume confirms the direction
  Liquidity      15 pts  — liquidity sweeps, equal highs/lows grabbed
  Structure      20 pts  — HH/HL or LL/LH on 1H
  Pattern        20 pts  — engulfing, pin bar, inside bar, consolidation break

Context types:
  PULLBACK       — retracing against trend, continuation setup
  BREAKOUT       — expanding from compression, volume surge
  CONTINUATION   — momentum continuation mid-move
  RANGE          — balanced, oscillating between S/R
  DISTRIBUTION   — topping / supply absorption
  ACCUMULATION   — bottoming / demand absorption
  UNCLEAR        — no dominant context
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ContextType(str, Enum):
    PULLBACK      = "pullback"
    BREAKOUT      = "breakout"
    CONTINUATION  = "continuation"
    RANGE         = "range"
    DISTRIBUTION  = "distribution"
    ACCUMULATION  = "accumulation"
    UNCLEAR       = "unclear"


@dataclass
class ContextBiasResult:
    bull_score: float         # 0-100
    bear_score: float         # 0-100
    context: ContextType
    dominant_bias: str        # "bull" | "bear" | "neutral"
    score_breakdown: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    @property
    def net_bias(self) -> float:
        """Positive = bull, negative = bear, range -100..+100."""
        return self.bull_score - self.bear_score

    def bias_aligned_long(self, macro_score: float) -> bool:
        """True if 1H bias aligns with macro (both bullish)."""
        return self.dominant_bias == "bull" and macro_score >= 50

    def bias_aligned_short(self, macro_score: float) -> bool:
        return self.dominant_bias == "bear" and macro_score <= 50


class ContextBiasEngine:
    """
    Scores 1H candles for directional bias and market context.
    Designed to run after MacroTrendEngine — uses its direction as a hint
    but scores independently.
    """

    def analyze(self, candles_1h: list, macro_score: float = 50.0) -> ContextBiasResult:
        if not candles_1h or len(candles_1h) < 30:
            return ContextBiasResult(
                bull_score=50.0, bear_score=50.0,
                context=ContextType.UNCLEAR, dominant_bias="neutral",
                detail={"reason": "insufficient_candles"},
            )

        closes  = np.array([float(c.close)  for c in candles_1h], dtype=float)
        highs   = np.array([float(c.high)   for c in candles_1h], dtype=float)
        lows    = np.array([float(c.low)    for c in candles_1h], dtype=float)
        opens   = np.array([float(c.open)   for c in candles_1h], dtype=float)
        volumes = np.array([float(c.volume) for c in candles_1h], dtype=float)

        bull_pts = 0.0
        bear_pts = 0.0
        breakdown: dict = {}
        detail:    dict = {}

        # ── 1. EMA Pullback (15 pts) ─────────────────────────────────────────
        bp, brp, d = self._ema_pullback_score(closes, 15)
        bull_pts += bp
        bear_pts += brp
        breakdown["ema_pullback"] = {"bull": bp, "bear": brp}
        detail["ema_pullback"] = d

        # ── 2. RSI (10 pts) ──────────────────────────────────────────────────
        bp, brp, d = self._rsi_score(closes, 10)
        bull_pts += bp
        bear_pts += brp
        breakdown["rsi"] = {"bull": bp, "bear": brp}
        detail["rsi"] = d

        # ── 3. MACD (10 pts) ─────────────────────────────────────────────────
        bp, brp, d = self._macd_score(closes, 10)
        bull_pts += bp
        bear_pts += brp
        breakdown["macd"] = {"bull": bp, "bear": brp}
        detail["macd"] = d

        # ── 4. Volume (10 pts) ───────────────────────────────────────────────
        bp, brp, d = self._volume_score(closes, volumes, 10)
        bull_pts += bp
        bear_pts += brp
        breakdown["volume"] = {"bull": bp, "bear": brp}
        detail["volume"] = d

        # ── 5. Liquidity (15 pts) ────────────────────────────────────────────
        bp, brp, d = self._liquidity_score(closes, highs, lows, 15)
        bull_pts += bp
        bear_pts += brp
        breakdown["liquidity"] = {"bull": bp, "bear": brp}
        detail["liquidity"] = d

        # ── 6. Structure (20 pts) ────────────────────────────────────────────
        bp, brp, d = self._structure_score(closes, highs, lows, 20)
        bull_pts += bp
        bear_pts += brp
        breakdown["structure"] = {"bull": bp, "bear": brp}
        detail["structure"] = d

        # ── 7. Pattern (20 pts) ──────────────────────────────────────────────
        bp, brp, d = self._pattern_score(opens, closes, highs, lows, 20)
        bull_pts += bp
        bear_pts += brp
        breakdown["pattern"] = {"bull": bp, "bear": brp}
        detail["pattern"] = d

        bull_score = round(min(100.0, bull_pts), 1)
        bear_score = round(min(100.0, bear_pts), 1)

        context = self._classify_context(closes, highs, lows, volumes, bull_score, bear_score)

        if bull_score > bear_score + 15:
            dominant_bias = "bull"
        elif bear_score > bull_score + 15:
            dominant_bias = "bear"
        else:
            dominant_bias = "neutral"

        return ContextBiasResult(
            bull_score=bull_score,
            bear_score=bear_score,
            context=context,
            dominant_bias=dominant_bias,
            score_breakdown=breakdown,
            detail=detail,
        )

    # ── Component scorers ────────────────────────────────────────────────────

    def _ema_pullback_score(self, closes: np.ndarray, max_pts: float):
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        price = closes[-1]

        e20 = ema20[-1]
        e50 = ema50[-1]

        # Bull: price pulled back to near EMA20 from above (continuation long setup)
        near_ema20 = abs(price - e20) / (e20 + 1e-9) < 0.005
        near_ema50 = abs(price - e50) / (e50 + 1e-9) < 0.008
        above_both = price > e20 > e50
        below_both = price < e20 < e50

        if above_both and near_ema20:
            bull = max_pts
            bear = 2.0
            d = "bull_pullback_to_ema20"
        elif above_both and near_ema50:
            bull = max_pts * 0.8
            bear = 3.0
            d = "bull_pullback_to_ema50"
        elif above_both:
            bull = max_pts * 0.5
            bear = 2.0
            d = "above_emas_no_pullback"
        elif below_both and near_ema20:
            bull = 2.0
            bear = max_pts
            d = "bear_pullback_to_ema20"
        elif below_both and near_ema50:
            bull = 3.0
            bear = max_pts * 0.8
            d = "bear_pullback_to_ema50"
        elif below_both:
            bull = 2.0
            bear = max_pts * 0.5
            d = "below_emas_no_pullback"
        else:
            bull = max_pts * 0.3
            bear = max_pts * 0.3
            d = "between_emas"

        return bull, bear, d

    def _rsi_score(self, closes: np.ndarray, max_pts: float):
        rsi = self._rsi(closes, 14)
        val = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0

        if val <= 30:
            bull = max_pts
            bear = 0.0
            d = f"oversold RSI={val:.1f}"
        elif val <= 45:
            bull = max_pts * 0.7
            bear = max_pts * 0.1
            d = f"below_mid RSI={val:.1f}"
        elif val >= 70:
            bull = 0.0
            bear = max_pts
            d = f"overbought RSI={val:.1f}"
        elif val >= 55:
            bull = max_pts * 0.1
            bear = max_pts * 0.7
            d = f"above_mid RSI={val:.1f}"
        else:
            bull = max_pts * 0.3
            bear = max_pts * 0.3
            d = f"neutral RSI={val:.1f}"

        return bull, bear, d

    def _macd_score(self, closes: np.ndarray, max_pts: float):
        macd, signal, hist = self._macd(closes)
        if np.isnan(macd[-1]):
            return max_pts * 0.3, max_pts * 0.3, "macd_unavailable"

        m = float(macd[-1])
        s = float(signal[-1])
        h = float(hist[-1])
        h_prev = float(hist[-2]) if len(hist) >= 2 and not np.isnan(hist[-2]) else 0.0

        crossed_up   = h > 0 and h_prev <= 0
        crossed_down = h < 0 and h_prev >= 0

        if crossed_up:
            bull = max_pts
            bear = 0.0
            d = "macd_cross_up"
        elif crossed_down:
            bull = 0.0
            bear = max_pts
            d = "macd_cross_down"
        elif m > s and h > 0 and h > h_prev:
            bull = max_pts * 0.8
            bear = max_pts * 0.1
            d = f"macd_bull_accel h={h:.4f}"
        elif m < s and h < 0 and h < h_prev:
            bull = max_pts * 0.1
            bear = max_pts * 0.8
            d = f"macd_bear_accel h={h:.4f}"
        elif m > s:
            bull = max_pts * 0.5
            bear = max_pts * 0.2
            d = "macd_above_signal"
        else:
            bull = max_pts * 0.2
            bear = max_pts * 0.5
            d = "macd_below_signal"

        return bull, bear, d

    def _volume_score(self, closes: np.ndarray, volumes: np.ndarray, max_pts: float):
        if len(volumes) < 20:
            return max_pts * 0.3, max_pts * 0.3, "insufficient"

        vol_avg = float(np.mean(volumes[-20:]))
        vol_now = float(volumes[-1])
        vol_ratio = vol_now / (vol_avg + 1e-9)

        last_bull = closes[-1] > closes[-2]
        last_bear = closes[-1] < closes[-2]

        if vol_ratio > 1.5 and last_bull:
            bull = max_pts
            bear = 0.0
            d = f"high_vol_bull x{vol_ratio:.1f}"
        elif vol_ratio > 1.5 and last_bear:
            bull = 0.0
            bear = max_pts
            d = f"high_vol_bear x{vol_ratio:.1f}"
        elif vol_ratio > 1.2 and last_bull:
            bull = max_pts * 0.7
            bear = 0.0
            d = f"above_avg_vol_bull x{vol_ratio:.1f}"
        elif vol_ratio > 1.2 and last_bear:
            bull = 0.0
            bear = max_pts * 0.7
            d = f"above_avg_vol_bear x{vol_ratio:.1f}"
        elif vol_ratio < 0.7:
            bull = max_pts * 0.2
            bear = max_pts * 0.2
            d = f"low_vol x{vol_ratio:.1f}"
        else:
            bull = max_pts * 0.4
            bear = max_pts * 0.4
            d = f"avg_vol x{vol_ratio:.1f}"

        return bull, bear, d

    def _liquidity_score(self, closes: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray, max_pts: float):
        if len(closes) < 20:
            return max_pts * 0.3, max_pts * 0.3, "insufficient"

        # Detect liquidity sweep: equal lows grabbed then reclaimed (bull)
        # or equal highs grabbed then fell (bear)
        window = 20
        recent_lows  = lows[-window:-2]
        recent_highs = highs[-window:-2]
        price        = closes[-1]
        low_now      = lows[-1]
        high_now     = highs[-1]

        eq_low_level  = float(np.min(recent_lows))
        eq_high_level = float(np.max(recent_highs))

        bull_sweep = low_now < eq_low_level * 0.999 and price > eq_low_level
        bear_sweep = high_now > eq_high_level * 1.001 and price < eq_high_level

        if bull_sweep:
            bull = max_pts
            bear = 0.0
            d = f"bull_liq_sweep below {eq_low_level:.2f}"
        elif bear_sweep:
            bull = 0.0
            bear = max_pts
            d = f"bear_liq_sweep above {eq_high_level:.2f}"
        else:
            # Proximity to swing levels (closer = more likely sweep coming)
            dist_to_low  = (price - eq_low_level) / (price + 1e-9)
            dist_to_high = (eq_high_level - price) / (price + 1e-9)

            if dist_to_low < 0.005:
                bull = max_pts * 0.6
                bear = max_pts * 0.2
                d = f"near_eq_low {eq_low_level:.2f}"
            elif dist_to_high < 0.005:
                bull = max_pts * 0.2
                bear = max_pts * 0.6
                d = f"near_eq_high {eq_high_level:.2f}"
            else:
                bull = max_pts * 0.3
                bear = max_pts * 0.3
                d = "no_sweep"

        return bull, bear, d

    def _structure_score(self, closes: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray, max_pts: float):
        window = min(30, len(closes) - 1)
        if window < 6:
            return max_pts * 0.4, max_pts * 0.4, "insufficient"

        h = highs[-window:]
        l = lows[-window:]
        q = window // 4

        h_early = float(np.mean(h[:q]))
        h_late  = float(np.mean(h[-q:]))
        l_early = float(np.mean(l[:q]))
        l_late  = float(np.mean(l[-q:]))

        hh = h_late > h_early * 1.001
        hl = l_late > l_early * 1.001
        ll = l_late < l_early * 0.999
        lh = h_late < h_early * 0.999

        if hh and hl:
            bull = max_pts
            bear = 0.0
            d = "HH+HL (bull structure)"
        elif ll and lh:
            bull = 0.0
            bear = max_pts
            d = "LL+LH (bear structure)"
        elif hh:
            bull = max_pts * 0.65
            bear = max_pts * 0.1
            d = "HH only"
        elif ll:
            bull = max_pts * 0.1
            bear = max_pts * 0.65
            d = "LL only"
        else:
            bull = max_pts * 0.35
            bear = max_pts * 0.35
            d = "mixed structure"

        return bull, bear, d

    def _pattern_score(self, opens: np.ndarray, closes: np.ndarray,
                       highs: np.ndarray, lows: np.ndarray, max_pts: float):
        if len(closes) < 5:
            return max_pts * 0.3, max_pts * 0.3, "insufficient"

        bull = 0.0
        bear = 0.0
        patterns = []

        # Bullish engulfing
        if (closes[-1] > opens[-1] and closes[-2] < opens[-2]
                and closes[-1] > opens[-2] and opens[-1] < closes[-2]):
            bull += max_pts * 0.5
            patterns.append("bull_engulfing")

        # Bearish engulfing
        if (closes[-1] < opens[-1] and closes[-2] > opens[-2]
                and closes[-1] < opens[-2] and opens[-1] > closes[-2]):
            bear += max_pts * 0.5
            patterns.append("bear_engulfing")

        # Bullish pin bar (lower wick > 2x body, close in upper half)
        body = abs(closes[-1] - opens[-1])
        lower_wick = min(closes[-1], opens[-1]) - lows[-1]
        upper_wick = highs[-1] - max(closes[-1], opens[-1])
        if lower_wick > 2.5 * body and closes[-1] > opens[-1]:
            bull += max_pts * 0.4
            patterns.append("bull_pin_bar")

        # Bearish pin bar
        if upper_wick > 2.5 * body and closes[-1] < opens[-1]:
            bear += max_pts * 0.4
            patterns.append("bear_pin_bar")

        # Morning star / three white soldiers (simplified: 3 up candles)
        if all(closes[-3+i] > opens[-3+i] for i in range(3)):
            bull += max_pts * 0.3
            patterns.append("3_white_soldiers")

        # Evening star / three black crows
        if all(closes[-3+i] < opens[-3+i] for i in range(3)):
            bear += max_pts * 0.3
            patterns.append("3_black_crows")

        # Inside bar breakout
        inside = (highs[-1] < highs[-2] and lows[-1] > lows[-2])
        if inside:
            # Setup, not direction — add partial to both
            bull += max_pts * 0.15
            bear += max_pts * 0.15
            patterns.append("inside_bar")

        bull = min(bull, max_pts)
        bear = min(bear, max_pts)

        # Floor if no pattern
        if bull + bear == 0:
            bull = max_pts * 0.25
            bear = max_pts * 0.25
            patterns.append("no_pattern")

        return bull, bear, ", ".join(patterns) if patterns else "none"

    def _classify_context(self, closes: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, volumes: np.ndarray,
                          bull_score: float, bear_score: float) -> ContextType:
        if len(closes) < 20:
            return ContextType.UNCLEAR

        # ATR-based compression check
        atr = self._atr_simple(closes, highs, lows, 14)
        atr_now  = float(np.nanmean(atr[-5:]))  if len(atr) >= 5  else 0.0
        atr_prev = float(np.nanmean(atr[-20:]))  if len(atr) >= 20 else 1e-9
        compressed = atr_prev > 0 and atr_now / atr_prev < 0.7

        # Volume spike
        vol_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1e-9
        vol_spike = float(volumes[-1]) / vol_avg > 1.8

        # Breakout
        if vol_spike and atr_now > atr_prev * 1.3:
            return ContextType.BREAKOUT

        # Pullback (trend exists, retracing)
        net = bull_score - bear_score
        if abs(net) > 30 and not vol_spike:
            ema20 = self._ema(closes, 20)
            price = closes[-1]
            near_ema = abs(price - ema20[-1]) / (ema20[-1] + 1e-9) < 0.008
            if near_ema:
                return ContextType.PULLBACK
            return ContextType.CONTINUATION

        # Range
        if compressed and abs(net) < 20:
            return ContextType.RANGE

        # Distribution / Accumulation
        if bull_score < 30 and bear_score > 60:
            return ContextType.DISTRIBUTION
        if bear_score < 30 and bull_score > 60:
            return ContextType.ACCUMULATION

        return ContextType.UNCLEAR

    # ── Math helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        if len(arr) < period:
            return out
        k = 2.0 / (period + 1)
        out[period - 1] = float(np.mean(arr[:period]))
        for i in range(period, len(arr)):
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
        return out

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(closes)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
        delta = np.diff(closes)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - 100 / (1 + rs)
        return rsi

    @staticmethod
    def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        n = len(closes)
        nan_arr = np.full(n, np.nan)
        if n < slow + signal:
            return nan_arr, nan_arr, nan_arr

        def ema(a, p):
            out = np.full(len(a), np.nan)
            k = 2.0 / (p + 1)
            out[p - 1] = float(np.mean(a[:p]))
            for i in range(p, len(a)):
                out[i] = a[i] * k + out[i - 1] * (1 - k)
            return out

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = ema_fast - ema_slow
        valid = ~np.isnan(macd_line)
        sig_line = np.full(n, np.nan)
        idx = np.where(valid)[0]
        if len(idx) >= signal:
            seg = macd_line[idx]
            sig_seg = ema(seg, signal)
            sig_line[idx] = sig_seg
        hist = macd_line - sig_line
        return macd_line, sig_line, hist

    @staticmethod
    def _atr_simple(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                    period: int = 14) -> np.ndarray:
        n = len(closes)
        tr = np.full(n, np.nan)
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
        atr = np.full(n, np.nan)
        if n > period:
            atr[period] = float(np.nanmean(tr[1:period + 1]))
            for i in range(period + 1, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr
