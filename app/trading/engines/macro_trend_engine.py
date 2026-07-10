"""
Layer 1: 4H Macro Trend Engine

Scores the macro trend from 4H candles on a 0-100 scale.
High score = strong bull macro, low score = strong bear macro, ~50 = neutral.

Score components (100 pts total):
  EMA20 / EMA50 alignment   20 pts
  EMA slope                 15 pts
  ADX strength              15 pts
  HH/HL structure           25 pts
  ATR expansion             10 pts
  Market structure (BOS)    15 pts

Thresholds:
  90+   STRONG_BULL  -> allowed_direction = long_only
  70-89 BULL         -> allowed_direction = both
  45-69 NEUTRAL      -> allowed_direction = both
  20-44 BEAR         -> allowed_direction = both
  0-19  STRONG_BEAR  -> allowed_direction = short_only

Macro NEVER picks entries. It only constrains which directions the
lower layers (Regime Classifier / Strategy Selector / Strategy Engine)
are permitted to act on — this is Layer 1's "Direction Gate" role.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class TrendBias(str, Enum):
    STRONG_BULL = "strong_bull"
    BULL        = "bull"
    NEUTRAL     = "neutral"
    BEAR        = "bear"
    STRONG_BEAR = "strong_bear"


@dataclass
class MacroTrendResult:
    score: float          # 0-100
    bias: TrendBias
    counter_trend_block: bool  # True = do NOT trade against this bias
    ema_aligned: bool
    adx: float
    structure: str        # "hh_hl" | "ll_lh" | "mixed"
    detail: dict

    def allows_long(self) -> bool:
        return self.score >= 20

    def allows_short(self) -> bool:
        return self.score <= 80

    def is_counter_trend_long(self) -> bool:
        return self.bias in (TrendBias.BEAR, TrendBias.STRONG_BEAR)

    def is_counter_trend_short(self) -> bool:
        return self.bias in (TrendBias.BULL, TrendBias.STRONG_BULL)

    def allowed_direction(self) -> str:
        """Layer 1 output: 'long_only' | 'short_only' | 'both' | 'no_trade'.
        Macro NEVER picks entries — it only fences which directions Layer 4/5
        are allowed to act on."""
        if self.bias == TrendBias.STRONG_BULL:
            return "long_only"
        if self.bias == TrendBias.STRONG_BEAR:
            return "short_only"
        if self.bias in (TrendBias.BULL, TrendBias.NEUTRAL, TrendBias.BEAR):
            return "both"
        return "no_trade"


class MacroTrendEngine:
    """
    Analyzes 4H candles to classify the macro trend.
    Called once per tick with the 4H candles from the MTF fetch.
    """

    def analyze(self, candles_4h: list) -> MacroTrendResult:
        if not candles_4h or len(candles_4h) < 50:
            return MacroTrendResult(
                score=50.0, bias=TrendBias.NEUTRAL,
                counter_trend_block=False, ema_aligned=False,
                adx=0.0, structure="mixed", detail={"reason": "insufficient_candles"},
            )

        closes = np.array([float(c.close) for c in candles_4h], dtype=float)
        highs  = np.array([float(c.high)  for c in candles_4h], dtype=float)
        lows   = np.array([float(c.low)   for c in candles_4h], dtype=float)

        score      = 0.0
        detail: dict = {}

        # ── EMA alignment (20 pts) ──────────────────────────────────────────
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        price = closes[-1]

        if price > ema20[-1] > ema50[-1]:
            score += 20.0
            detail["ema"] = "price>ema20>ema50 (bull)"
        elif price < ema20[-1] < ema50[-1]:
            score += 0.0
            detail["ema"] = "price<ema20<ema50 (bear)"
        elif price > ema50[-1]:
            score += 12.0
            detail["ema"] = "price>ema50 (weak bull)"
        else:
            score += 5.0
            detail["ema"] = "price<ema50 (weak bear)"

        # ── EMA slope (15 pts) ──────────────────────────────────────────────
        ema20_slope = (ema20[-1] - ema20[-5]) / (ema20[-5] + 1e-9)
        if ema20_slope > 0.005:
            score += 15.0
            detail["ema_slope"] = f"rising {ema20_slope:.4f}"
        elif ema20_slope > 0.0:
            score += 9.0
            detail["ema_slope"] = f"slightly rising {ema20_slope:.4f}"
        elif ema20_slope > -0.005:
            score += 5.0
            detail["ema_slope"] = f"slightly falling {ema20_slope:.4f}"
        else:
            score += 0.0
            detail["ema_slope"] = f"falling {ema20_slope:.4f}"

        # ── ADX (15 pts) ────────────────────────────────────────────────────
        adx_arr, pdi, mdi = self._adx(closes, highs, lows, 14)
        adx_val = float(adx_arr[-1]) if len(adx_arr) > 0 and not np.isnan(adx_arr[-1]) else 0.0
        pdi_val = float(pdi[-1])     if len(pdi) > 0     and not np.isnan(pdi[-1])     else 50.0
        mdi_val = float(mdi[-1])     if len(mdi) > 0     and not np.isnan(mdi[-1])     else 50.0

        if adx_val >= 30:
            adx_pts = 15.0
        elif adx_val >= 20:
            adx_pts = 10.0
        elif adx_val >= 15:
            adx_pts = 5.0
        else:
            adx_pts = 2.0

        # Directional bias from +DI / -DI
        if pdi_val > mdi_val:
            score += adx_pts        # trending up
            detail["adx"] = f"ADX={adx_val:.1f} +DI>{mdi_val:.1f} (bull trend)"
        else:
            score += adx_pts * 0.0  # trending down — no points (bearish)
            detail["adx"] = f"ADX={adx_val:.1f} -DI>{pdi_val:.1f} (bear trend)"

        # ── HH / HL structure (25 pts) ──────────────────────────────────────
        structure, hh_hl_pts = self._hh_hl_score(highs, lows, lookback=20)
        score += hh_hl_pts
        detail["structure"] = f"{structure} ({hh_hl_pts:.0f}pts)"

        # ── ATR expansion (10 pts) ──────────────────────────────────────────
        atr_arr = self._atr(closes, highs, lows, 14)
        if len(atr_arr) >= 20:
            atr_now  = float(np.nanmean(atr_arr[-5:]))
            atr_prev = float(np.nanmean(atr_arr[-20:-5]))
            if atr_prev > 0 and atr_now / atr_prev > 1.2:
                # ATR expanding — slightly bullish in a bull trend, neutral otherwise
                score += 6.0
                detail["atr"] = f"expanding ({atr_now/atr_prev:.2f}x)"
            elif atr_prev > 0 and atr_now / atr_prev < 0.8:
                score += 3.0
                detail["atr"] = f"compressing ({atr_now/atr_prev:.2f}x)"
            else:
                score += 5.0
                detail["atr"] = "neutral"
        else:
            score += 5.0

        # ── Market structure BOS (15 pts) ───────────────────────────────────
        bos_pts, bos_label = self._bos_score(highs, lows, closes, lookback=15)
        score += bos_pts
        detail["bos"] = bos_label

        # Normalize to 0-100
        score = max(0.0, min(100.0, score))

        bias = self._classify(score)
        ema_aligned = price > ema20[-1] > ema50[-1] or price < ema20[-1] < ema50[-1]

        return MacroTrendResult(
            score=round(score, 1),
            bias=bias,
            counter_trend_block=bias in (TrendBias.STRONG_BULL, TrendBias.STRONG_BEAR),
            ema_aligned=ema_aligned,
            adx=round(adx_val, 1),
            structure=structure,
            detail=detail,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> TrendBias:
        if score >= 90:
            return TrendBias.STRONG_BULL
        if score >= 70:
            return TrendBias.BULL
        if score >= 45:
            return TrendBias.NEUTRAL
        if score >= 20:
            return TrendBias.BEAR
        return TrendBias.STRONG_BEAR

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
    def _atr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
             period: int = 14) -> np.ndarray:
        n = len(closes)
        tr = np.full(n, np.nan)
        for i in range(1, n):
            hl  = highs[i] - lows[i]
            hpc = abs(highs[i] - closes[i - 1])
            lpc = abs(lows[i]  - closes[i - 1])
            tr[i] = max(hl, hpc, lpc)
        atr = np.full(n, np.nan)
        if n > period:
            atr[period] = float(np.mean(tr[1:period + 1]))
            for i in range(period + 1, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _adx(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
             period: int = 14):
        n = len(closes)
        pdi = np.full(n, np.nan)
        mdi = np.full(n, np.nan)
        adx = np.full(n, np.nan)
        if n < period * 2:
            return adx, pdi, mdi

        tr_arr = np.full(n, 0.0)
        pdm    = np.full(n, 0.0)
        mdm    = np.full(n, 0.0)
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hpc = abs(highs[i] - closes[i - 1])
            lpc = abs(lows[i]  - closes[i - 1])
            tr_arr[i] = max(hl, hpc, lpc)
            up   = highs[i]  - highs[i - 1]
            down = lows[i - 1] - lows[i]
            pdm[i] = up   if up > down and up > 0   else 0.0
            mdm[i] = down if down > up and down > 0 else 0.0

        atr14  = np.full(n, 0.0)
        pdi14  = np.full(n, 0.0)
        mdi14  = np.full(n, 0.0)
        atr14[period]  = float(np.sum(tr_arr[1:period + 1]))
        pdi14[period]  = float(np.sum(pdm[1:period + 1]))
        mdi14[period]  = float(np.sum(mdm[1:period + 1]))
        for i in range(period + 1, n):
            atr14[i] = atr14[i - 1] - atr14[i - 1] / period + tr_arr[i]
            pdi14[i] = pdi14[i - 1] - pdi14[i - 1] / period + pdm[i]
            mdi14[i] = mdi14[i - 1] - mdi14[i - 1] / period + mdm[i]

        for i in range(period, n):
            if atr14[i] > 0:
                pdi[i] = 100 * pdi14[i] / atr14[i]
                mdi[i] = 100 * mdi14[i] / atr14[i]

        dx = np.full(n, np.nan)
        for i in range(period, n):
            denom = pdi[i] + mdi[i]
            if denom > 0:
                dx[i] = 100 * abs(pdi[i] - mdi[i]) / denom

        adx[period * 2 - 1] = float(np.nanmean(dx[period:period * 2]))
        for i in range(period * 2, n):
            if not np.isnan(adx[i - 1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx, pdi, mdi

    @staticmethod
    def _hh_hl_score(highs: np.ndarray, lows: np.ndarray,
                     lookback: int = 20) -> tuple[str, float]:
        """Score based on higher-highs / higher-lows (bullish) or lower-lows / lower-highs."""
        window = min(lookback, len(highs) - 1)
        if window < 4:
            return "mixed", 12.5

        # Find pivot highs/lows in the window
        h = highs[-window:]
        l = lows[-window:]

        # Count directional moves (simplified: compare quarters)
        q = window // 4
        h_early = float(np.mean(h[:q]))
        h_late  = float(np.mean(h[-q:]))
        l_early = float(np.mean(l[:q]))
        l_late  = float(np.mean(l[-q:]))

        hh = h_late > h_early
        hl = l_late > l_early
        ll = l_late < l_early
        lh = h_late < h_early

        if hh and hl:
            return "hh_hl", 25.0
        if ll and lh:
            return "ll_lh", 0.0
        if hh:
            return "hh_only", 15.0
        if ll:
            return "ll_only", 5.0
        return "mixed", 12.0

    @staticmethod
    def _bos_score(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   lookback: int = 15) -> tuple[float, str]:
        """Break of Structure: bullish BOS = close above recent swing high."""
        if len(closes) < lookback + 5:
            return 7.5, "unknown"

        swing_high = float(np.max(highs[-lookback - 5:-5]))
        swing_low  = float(np.min(lows[-lookback - 5:-5]))
        price      = float(closes[-1])

        if price > swing_high:
            return 15.0, f"BOS_bull (>{swing_high:.2f})"
        if price < swing_low:
            return 0.0,  f"BOS_bear (<{swing_low:.2f})"
        return 7.5, "inside_range"
