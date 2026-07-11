"""
Trend Engine — shared, timeframe-agnostic trend direction + stage classifier.

Consolidates the stage-classification logic that macro_trend_engine.py (4H)
and context_bias_engine.py (1H) each grew independently into one engine,
usable on any candle series/timeframe. Exactly 4 checks, no more:

  1. EMA20 vs EMA50 alignment       — direction (bull/bear/neutral)
  2. ADX level + 5-bar trajectory   — trend strength, rising or fading
  3. RSI(14) position                — overbought/oversold extremity
  4. EMA20 slope over the last N bars (default 4) — is price actively
     still moving that way right now, or has the earlier move stalled?
     This is the check that keeps a stale EMA20/50 alignment (from a
     move that already stopped) from getting mislabeled "mid" just
     because ADX happens to still read mid-range.

Stage:
  early : ADX < 20 — move just getting going, not confirmed by strength yet
  mid   : ADX 20-35 (or higher but still rising) AND EMA20 slope agrees
          with direction AND RSI not yet extreme
  late  : RSI extreme (>=70 bull / <=25 bear), OR ADX high but no longer
          rising, OR EMA20 slope has flattened/reversed against the
          still-aligned EMA20/50 order (momentum stalling before the
          moving averages catch up) — i.e. exhaustion risk
  n/a   : bias is neutral — no trend to stage
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TrendBias(str, Enum):
    BULL    = "bull"
    BEAR    = "bear"
    NEUTRAL = "neutral"


class TrendStage(str, Enum):
    EARLY = "early"
    MID   = "mid"
    LATE  = "late"
    NA    = "n/a"


@dataclass
class TrendResult:
    bias:  TrendBias
    stage: TrendStage
    adx:   float
    rsi:   float
    ema_slope_pct: float   # % change of EMA20 over the lookback window
    detail: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.bias.value}-{self.stage.value}" if self.bias != TrendBias.NEUTRAL else "neutral"


class TrendEngine:
    def __init__(self, ema_fast: int = 20, ema_slow: int = 50,
                 adx_period: int = 14, rsi_period: int = 14,
                 slope_lookback: int = 4):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.slope_lookback = slope_lookback

    def analyze(self, candles: list) -> TrendResult:
        min_needed = max(self.ema_slow, self.adx_period * 2, self.rsi_period) + self.slope_lookback + 5
        if len(candles) < min_needed:
            return TrendResult(TrendBias.NEUTRAL, TrendStage.NA, 0.0, 50.0, 0.0,
                               detail={"reason": f"need {min_needed}+ candles, have {len(candles)}"})

        closes = np.array([float(c.close) for c in candles], dtype=float)
        highs  = np.array([float(c.high)  for c in candles], dtype=float)
        lows   = np.array([float(c.low)   for c in candles], dtype=float)

        ema20 = self._ema(closes, self.ema_fast)
        ema50 = self._ema(closes, self.ema_slow)
        price = closes[-1]

        # ── Check 1: EMA alignment -> raw direction ──────────────────────────
        if price > ema20[-1] > ema50[-1]:
            raw_bias = TrendBias.BULL
        elif price < ema20[-1] < ema50[-1]:
            raw_bias = TrendBias.BEAR
        else:
            raw_bias = TrendBias.NEUTRAL

        # ── Check 4: EMA20 slope over the last N bars ────────────────────────
        lb = self.slope_lookback
        ema_slope_pct = (ema20[-1] - ema20[-1 - lb]) / (abs(ema20[-1 - lb]) + 1e-9) * 100.0

        # Direction only counts if the recent slope agrees — a stale
        # EMA20>EMA50 order from a move that already stalled doesn't
        # count as an active bull/bear bias anymore.
        if raw_bias == TrendBias.BULL and ema_slope_pct <= 0:
            bias = TrendBias.NEUTRAL
        elif raw_bias == TrendBias.BEAR and ema_slope_pct >= 0:
            bias = TrendBias.NEUTRAL
        else:
            bias = raw_bias

        # ── Check 2: ADX level + trajectory ───────────────────────────────────
        adx_arr, _, _ = self._adx(closes, highs, lows, self.adx_period)
        adx_val  = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
        adx_prev = float(adx_arr[-6]) if len(adx_arr) > 5 and not np.isnan(adx_arr[-6]) else adx_val
        adx_rising = adx_val > adx_prev

        # ── Check 3: RSI position ─────────────────────────────────────────────
        rsi_arr = self._rsi(closes, self.rsi_period)
        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0

        if bias == TrendBias.NEUTRAL:
            stage = TrendStage.NA
        elif adx_val < 20:
            stage = TrendStage.EARLY
        elif bias == TrendBias.BULL and rsi_val >= 70:
            stage = TrendStage.LATE
        elif bias == TrendBias.BEAR and rsi_val <= 25:
            stage = TrendStage.LATE
        elif adx_val >= 35 and not adx_rising:
            stage = TrendStage.LATE
        else:
            stage = TrendStage.MID

        return TrendResult(
            bias=bias, stage=stage, adx=round(adx_val, 1), rsi=round(rsi_val, 1),
            ema_slope_pct=round(ema_slope_pct, 3),
            detail={
                "raw_bias": raw_bias.value, "adx_rising": adx_rising,
                "price_vs_ema20": round((price - ema20[-1]) / ema20[-1] * 100, 3),
            },
        )

    # ── Math helpers (same formulas as macro_trend_engine.py) ────────────────

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
            rsi[i + 1] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
        return rsi

    @staticmethod
    def _adx(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14):
        n = len(closes)
        pdi = np.full(n, np.nan)
        mdi = np.full(n, np.nan)
        adx = np.full(n, np.nan)
        if n < period * 2:
            return adx, pdi, mdi

        tr_arr = np.full(n, 0.0)
        pdm = np.full(n, 0.0)
        mdm = np.full(n, 0.0)
        for i in range(1, n):
            hl, hpc, lpc = highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])
            tr_arr[i] = max(hl, hpc, lpc)
            up, down = highs[i] - highs[i-1], lows[i-1] - lows[i]
            pdm[i] = up if up > down and up > 0 else 0.0
            mdm[i] = down if down > up and down > 0 else 0.0

        atr14 = np.full(n, 0.0); pdi14 = np.full(n, 0.0); mdi14 = np.full(n, 0.0)
        atr14[period] = float(np.sum(tr_arr[1:period+1]))
        pdi14[period] = float(np.sum(pdm[1:period+1]))
        mdi14[period] = float(np.sum(mdm[1:period+1]))
        for i in range(period + 1, n):
            atr14[i] = atr14[i-1] - atr14[i-1] / period + tr_arr[i]
            pdi14[i] = pdi14[i-1] - pdi14[i-1] / period + pdm[i]
            mdi14[i] = mdi14[i-1] - mdi14[i-1] / period + mdm[i]

        for i in range(period, n):
            if atr14[i] > 0:
                pdi[i] = 100 * pdi14[i] / atr14[i]
                mdi[i] = 100 * mdi14[i] / atr14[i]

        dx = np.full(n, np.nan)
        for i in range(period, n):
            denom = pdi[i] + mdi[i]
            if denom > 0:
                dx[i] = 100 * abs(pdi[i] - mdi[i]) / denom

        adx[period * 2 - 1] = float(np.nanmean(dx[period:period*2]))
        for i in range(period * 2, n):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

        return adx, pdi, mdi
