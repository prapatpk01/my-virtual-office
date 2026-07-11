"""
Trend Engine — shared, timeframe-agnostic trend scoring engine.

Weighted composite score, 0-100, one number instead of a discrete
bull/bear/neutral vote. 3 categories, 5 checks total (1-2 per
category), each check turned into its own continuous 0-100 sub-score
(50 = neutral, 100 = maximally bullish, 0 = maximally bearish):

  Trend    50% — direction & structure
    EMA12/26 spread   25% — (ema12-ema26)/ATR, tanh-scaled
    ADX + DI direction 25% — strength capped at 50, continuously signed
                             by +DI vs -DI (tanh, not a hard flip)
  Momentum 30% — how hard and how fast it's currently moving
    RSI(14)            15% — used directly, already 0-100/50-center
    EMA12 slope        15% — bar-to-bar change of EMA12/ATR, tanh-scaled
  Volume   20% — does volume confirm the bar's own direction
    Volume vs 20-bar average, signed by that bar's close-vs-open,
    tanh-scaled — a high-volume bar confirms whichever way it closed;
    an average-or-below-volume bar contributes near nothing regardless
    of direction (weak conviction either way).

Each check is computed at the last 3 bars and blended 50/30/20%
weighted toward the most recent bar (not a flat average), so a single
noisy bar can't swing the read without lagging behind a real move.

EMA and slope are normalized by ATR, not raw %-of-price — a flat %
threshold miscalibrates across timeframes (the same real move is a
much smaller % on a 15m bar than a 1H bar), so scaling by the
instrument's own current volatility keeps the score equally responsive
whether it's fed 15m, 30m, 1H, or 4H candles.

Bands (symmetric around 50):
  76-100  strong_bull      56-65  early_bull
  66-75   bull             45-55  sideway
                            35-44  early_bear
                            25-34  bear
                            0-24   strong_bear
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TrendBand(str, Enum):
    STRONG_BULL = "strong_bull"
    BULL        = "bull"
    EARLY_BULL  = "early_bull"
    SIDEWAY     = "sideway"
    EARLY_BEAR  = "early_bear"
    BEAR        = "bear"
    STRONG_BEAR = "strong_bear"


@dataclass
class TrendResult:
    score: float           # 0-100, 50 = perfectly neutral
    band:  TrendBand
    sub_scores: dict = field(default_factory=dict)   # {"ema": .., "adx": .., "rsi": .., "slope": ..} last-bar values
    detail: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.band.value


class TrendEngine:
    # category weights: trend 50% (ema+adx), momentum 30% (rsi+slope), volume 20%
    _WEIGHTS = {"ema": 0.25, "adx": 0.25, "rsi": 0.15, "slope": 0.15, "volume": 0.20}
    _CATEGORY = {"ema": "trend", "adx": "trend", "rsi": "momentum", "slope": "momentum", "volume": "volume"}
    _BAR_WEIGHTS = (0.5, 0.3, 0.2)   # most-recent-first; must match avg_bars length

    def __init__(self, ema_fast: int = 12, ema_slow: int = 26,
                 adx_period: int = 14, rsi_period: int = 14, atr_period: int = 14,
                 volume_lookback: int = 20,
                 avg_bars: int = 3,
                 ema_spread_sensitivity: float = 0.8,
                 slope_sensitivity: float = 1.5,
                 volume_sensitivity: float = 1.2):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.volume_lookback = volume_lookback
        self.avg_bars = avg_bars
        self.ema_spread_sensitivity = ema_spread_sensitivity
        self.slope_sensitivity = slope_sensitivity
        self.volume_sensitivity = volume_sensitivity

    def analyze(self, candles: list) -> TrendResult:
        n = self.avg_bars
        min_needed = max(self.ema_slow, self.adx_period * 2, self.rsi_period,
                         self.atr_period, self.volume_lookback) + n + 5
        if len(candles) < min_needed:
            return TrendResult(50.0, TrendBand.SIDEWAY,
                               detail={"reason": f"need {min_needed}+ candles, have {len(candles)}"})

        closes  = np.array([float(c.close)  for c in candles], dtype=float)
        highs   = np.array([float(c.high)   for c in candles], dtype=float)
        lows    = np.array([float(c.low)    for c in candles], dtype=float)
        opens   = np.array([float(c.open)   for c in candles], dtype=float)
        volumes = np.array([float(c.volume) for c in candles], dtype=float)

        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        adx_arr, pdi_arr, mdi_arr = self._adx(closes, highs, lows, self.adx_period)
        rsi_arr = self._rsi(closes, self.rsi_period)
        atr_arr = self._atr(closes, highs, lows, self.atr_period)

        composites = []
        cat_composites = []
        last_subs = None
        for k in range(n):
            i = -1 - k  # -1, -2, -3, ...
            atr_v = float(atr_arr[i]) if not np.isnan(atr_arr[i]) and atr_arr[i] > 0 else closes[i] * 0.005

            # ── Trend: EMA12/26 spread, ATR-normalized + tanh-scaled ──────
            spread_atr = (ema_f[i] - ema_s[i]) / atr_v
            ema_score = 50.0 + 50.0 * math.tanh(self.ema_spread_sensitivity * spread_atr)

            # ── Trend: ADX strength, signed by +DI/-DI (continuous, not a
            # hard flip — see history: a hard sign(+1/-1) snapped this
            # ~50pts on a hairline +DI/-DI cross) ─────────────────────────
            adx_v = float(adx_arr[i]) if not np.isnan(adx_arr[i]) else 0.0
            pdi_v = float(pdi_arr[i]) if not np.isnan(pdi_arr[i]) else 50.0
            mdi_v = float(mdi_arr[i]) if not np.isnan(mdi_arr[i]) else 50.0
            di_lean = math.tanh(0.08 * (pdi_v - mdi_v))   # -1..1, continuous
            adx_score = 50.0 + di_lean * min(adx_v, 50.0)

            # ── Momentum: RSI, used directly ───────────────────────────────
            rsi_v = float(rsi_arr[i]) if not np.isnan(rsi_arr[i]) else 50.0
            rsi_score = rsi_v

            # ── Momentum: EMA12 bar-to-bar slope, ATR-normalized ──────────
            ef_prev = ema_f[i - 1]
            slope_atr = (ema_f[i] - ef_prev) / atr_v
            slope_score = 50.0 + 50.0 * math.tanh(self.slope_sensitivity * slope_atr)

            # ── Volume: volume vs its 20-bar average, signed by that bar's
            # own close-vs-open — confirms whichever way the bar closed,
            # scaled by how much above/below average the volume was ────
            vlb = self.volume_lookback
            vol_window = volumes[i - vlb:i] if i - vlb >= -len(volumes) else volumes[:i]
            vol_avg = float(np.mean(vol_window)) if len(vol_window) > 0 else volumes[i]
            vol_ratio = volumes[i] / (vol_avg + 1e-9)
            bar_dir = 1.0 if closes[i] > opens[i] else (-1.0 if closes[i] < opens[i] else 0.0)
            volume_score = 50.0 + bar_dir * 50.0 * math.tanh(self.volume_sensitivity * (vol_ratio - 1.0))

            subs = {"ema": ema_score, "adx": adx_score, "rsi": rsi_score,
                    "slope": slope_score, "volume": volume_score}
            composite = sum(subs[k2] * w for k2, w in self._WEIGHTS.items())
            composites.append(composite)
            if k == 0:
                last_subs = subs
                last_cats = {
                    "trend":    (subs["ema"] * 0.25 + subs["adx"] * 0.25) / 0.50,
                    "momentum": (subs["rsi"] * 0.15 + subs["slope"] * 0.15) / 0.30,
                    "volume":   subs["volume"],
                }

        # Recency-weighted blend across the last n bars (0.5/0.3/0.2 for
        # n=3) instead of a flat average — dilutes single-bar noise
        # without lagging a full bar-count behind a real move.
        bar_weights = self._BAR_WEIGHTS if n == 3 else [1.0 / n] * n
        score = round(sum(c * w for c, w in zip(composites, bar_weights)), 1)
        score = max(0.0, min(100.0, score))
        band = self._classify(score)

        return TrendResult(
            score=score, band=band, sub_scores={k: round(v, 1) for k, v in last_subs.items()},
            detail={
                "composites": [round(c, 1) for c in composites],
                "categories": {k: round(v, 1) for k, v in last_cats.items()},
            },
        )

    @staticmethod
    def _classify(score: float) -> TrendBand:
        if score >= 76:
            return TrendBand.STRONG_BULL
        if score >= 66:
            return TrendBand.BULL
        if score >= 56:
            return TrendBand.EARLY_BULL
        if score >= 45:
            return TrendBand.SIDEWAY
        if score >= 35:
            return TrendBand.EARLY_BEAR
        if score >= 25:
            return TrendBand.BEAR
        return TrendBand.STRONG_BEAR

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
    def _atr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(closes)
        tr = np.full(n, np.nan)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr = np.full(n, np.nan)
        if n > period:
            atr[period] = float(np.nanmean(tr[1:period + 1]))
            for i in range(period + 1, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

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
