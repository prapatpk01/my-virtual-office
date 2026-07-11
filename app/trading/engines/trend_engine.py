"""
Trend Engine — shared, timeframe-agnostic trend direction + stage classifier.

Consolidates the stage-classification logic that macro_trend_engine.py (4H)
and context_bias_engine.py (1H) each grew independently into one engine,
usable on any candle series/timeframe. Exactly 4 checks:

  1. EMA12 vs EMA26 alignment       — direction (bull/bear/neutral)
  2. ADX +DI/-DI dominance          — which side is actually in control
  3. RSI(14) position                — overbought/oversold extremity
  4. EMA12 bar-to-bar slope direction — is price actively still moving
     that way right now?

Each check is evaluated bar-by-bar over the LAST 3 BARS, not just the
current one — a single-bar read is noisy (one wick or one bar's RSI
blip flips it), so every check requires its own read to agree on all
3 of the last 3 bars before it "votes" for a direction. Any check that
flickers (doesn't read the same way on all 3 bars) contributes nothing
that bar rather than forcing a guess — direction is only bull/bear once
all 4 checks vote the same way *and* stay consistent for 3 bars running.

Stage (only meaningful once bias is bull/bear):
  early : ADX < 20 for the last 3 bars — move just getting going
  mid   : trend confirmed (all 4 checks agree, 3-bar consistent),
          RSI not yet extreme
  late  : RSI extreme (>=70 bull / <=25 bear) held for the last 3 bars,
          OR ADX >= 35 but not rising over the last 3 bars — exhaustion risk
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
    votes: dict = field(default_factory=dict)   # which of the 4 checks confirmed, and what
    detail: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.bias.value}-{self.stage.value}" if self.bias != TrendBias.NEUTRAL else "neutral"


class TrendEngine:
    def __init__(self, ema_fast: int = 12, ema_slow: int = 26,
                 adx_period: int = 14, rsi_period: int = 14,
                 confirm_bars: int = 3):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.confirm_bars = confirm_bars

    def analyze(self, candles: list) -> TrendResult:
        n = self.confirm_bars
        min_needed = max(self.ema_slow, self.adx_period * 2, self.rsi_period) + n + 5
        if len(candles) < min_needed:
            return TrendResult(TrendBias.NEUTRAL, TrendStage.NA, 0.0, 50.0,
                               detail={"reason": f"need {min_needed}+ candles, have {len(candles)}"})

        closes = np.array([float(c.close) for c in candles], dtype=float)
        highs  = np.array([float(c.high)  for c in candles], dtype=float)
        lows   = np.array([float(c.low)   for c in candles], dtype=float)

        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        adx_arr, pdi_arr, mdi_arr = self._adx(closes, highs, lows, self.adx_period)
        rsi_arr = self._rsi(closes, self.rsi_period)

        # ── Per-bar reads for each of the 4 checks, over the last n bars ────
        ema_dirs, adx_dirs, rsi_zones, slope_dirs = [], [], [], []
        for k in range(n):
            i = -1 - k  # -1, -2, -3, ...
            c, ef, es = closes[i], ema_f[i], ema_s[i]
            ema_dirs.append("bull" if c > ef > es else "bear" if c < ef < es else "neutral")

            pdi, mdi = pdi_arr[i], mdi_arr[i]
            if np.isnan(pdi) or np.isnan(mdi) or pdi == mdi:
                adx_dirs.append("neutral")
            else:
                adx_dirs.append("bull" if pdi > mdi else "bear")

            r = rsi_arr[i]
            r = 50.0 if np.isnan(r) else r
            rsi_zones.append("bull" if r >= 55 else "bear" if r <= 45 else "neutral")

            ef_prev = ema_f[i - 1]
            slope_dirs.append("bull" if ef > ef_prev else "bear" if ef < ef_prev else "neutral")

        def _confirmed(reads: list) -> str | None:
            """Only 'votes' if it read the SAME non-neutral direction on
            every one of the last n bars — a check that flickers doesn't
            get a say this bar."""
            first = reads[0]
            if first == "neutral":
                return None
            return first if all(r == first for r in reads) else None

        votes = {
            "ema_align":  _confirmed(ema_dirs),
            "adx_dir":    _confirmed(adx_dirs),
            "rsi_zone":   _confirmed(rsi_zones),
            "ema_slope":  _confirmed(slope_dirs),
        }
        confirmed_votes = [v for v in votes.values() if v is not None]

        # Bias only when every check that DID confirm agrees with each
        # other, AND at least 3 of the 4 checks actually confirmed
        # (not flickering) — this is what makes the read "accurate"
        # instead of a single noisy bar deciding it.
        if len(confirmed_votes) >= 3 and len(set(confirmed_votes)) == 1:
            bias = TrendBias.BULL if confirmed_votes[0] == "bull" else TrendBias.BEAR
        else:
            bias = TrendBias.NEUTRAL

        adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0

        adx_below_20_all    = all((adx_arr[-1 - k] if not np.isnan(adx_arr[-1 - k]) else 99) < 20 for k in range(n))
        adx_high_flat_all   = all((adx_arr[-1 - k] if not np.isnan(adx_arr[-1 - k]) else 0) >= 35 for k in range(n)) \
                              and adx_arr[-1] <= adx_arr[-n]
        rsi_extreme_all_bull = all((rsi_arr[-1 - k] if not np.isnan(rsi_arr[-1 - k]) else 50) >= 70 for k in range(n))
        rsi_extreme_all_bear = all((rsi_arr[-1 - k] if not np.isnan(rsi_arr[-1 - k]) else 50) <= 25 for k in range(n))

        if bias == TrendBias.NEUTRAL:
            stage = TrendStage.NA
        elif adx_below_20_all:
            stage = TrendStage.EARLY
        elif (bias == TrendBias.BULL and rsi_extreme_all_bull) or (bias == TrendBias.BEAR and rsi_extreme_all_bear):
            stage = TrendStage.LATE
        elif adx_high_flat_all:
            stage = TrendStage.LATE
        else:
            stage = TrendStage.MID

        return TrendResult(
            bias=bias, stage=stage, adx=round(adx_val, 1), rsi=round(rsi_val, 1),
            votes=votes,
            detail={
                "confirmed_count": len(confirmed_votes),
                "ema_dirs": ema_dirs, "adx_dirs": adx_dirs,
                "rsi_zones": rsi_zones, "slope_dirs": slope_dirs,
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
