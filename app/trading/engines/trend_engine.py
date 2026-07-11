"""
Trend Engine — shared, timeframe-agnostic trend/entry scoring engine.

Weighted composite score, 0-100 (50 = neutral), built from 4 checks:

  EMA12/26 spread     30 pts — (ema12-ema26)/ATR, tanh-scaled direction
                                confirmation.
  MACD(12,26) hist     30 pts — histogram/ATR, tanh-scaled direction
                                confirmation, but with a DYNAMIC ATR-based
                                penalty: once the histogram is stretched
                                further than ~1xATR from zero, points get
                                deducted (up to 60%) the further it goes —
                                a MACD signal that's already extended far
                                is chasing a move that mostly happened,
                                not confirming a fresh one.
  ROC(9)               20 pts — 9-bar rate of change, tanh-scaled.
  ADX(14) rising       20 pts — ADX has no direction of its own (it's a
                                strength gauge), so it doesn't vote on
                                bull/bear independently — it AMPLIFIES
                                whichever direction the other 3 checks
                                already lean, scaled by how much ADX has
                                risen over the last 3 bars. A falling ADX
                                contributes nothing either way.

Each check is computed at the last 3 bars and blended 50/30/20% weighted
toward the most recent bar (not a flat average), so a single noisy bar
can't swing the read without lagging behind a real move.

Entry gate: score > 70 -> long entry ready. score < 30 -> short entry
ready (mirrors the same threshold). Between 30-70 -> not ready.

EMA/MACD/ROC are ATR-normalized, not raw %-of-price — a flat % threshold
miscalibrates across timeframes (the same real move is a much smaller %
on a 15m bar than a 1H bar), so scaling by the instrument's own current
volatility keeps the score equally responsive whether it's fed 15m,
30m, 1H, or 4H candles.
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
    entry_ready: str        # "long" | "short" | "none"
    sub_scores: dict = field(default_factory=dict)   # {"ema":.., "macd":.., "roc":.., "adx":..} last-bar values
    detail: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.band.value


class TrendEngine:
    _BAR_WEIGHTS = (0.5, 0.3, 0.2)   # most-recent-first; must match avg_bars length
    ENTRY_LONG_THRESHOLD  = 70.0
    ENTRY_SHORT_THRESHOLD = 30.0

    def __init__(self, ema_fast: int = 12, ema_slow: int = 26,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 roc_period: int = 9, adx_period: int = 14, atr_period: int = 14,
                 avg_bars: int = 3,
                 ema_spread_sensitivity: float = 0.8,
                 macd_hist_sensitivity: float = 1.0,
                 macd_extension_sweet_spot: float = 1.0,   # in ATR units
                 macd_extension_max_penalty: float = 0.6,   # fraction of points lost when fully extended
                 roc_sensitivity: float = 50.0,
                 adx_rising_sensitivity: float = 0.15):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.roc_period = roc_period
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.avg_bars = avg_bars
        self.ema_spread_sensitivity = ema_spread_sensitivity
        self.macd_hist_sensitivity = macd_hist_sensitivity
        self.macd_extension_sweet_spot = macd_extension_sweet_spot
        self.macd_extension_max_penalty = macd_extension_max_penalty
        self.roc_sensitivity = roc_sensitivity
        self.adx_rising_sensitivity = adx_rising_sensitivity

    def analyze(self, candles: list) -> TrendResult:
        n = self.avg_bars
        min_needed = max(self.ema_slow, self.macd_slow + self.macd_signal,
                         self.roc_period, self.adx_period * 2, self.atr_period) + n + 5
        if len(candles) < min_needed:
            return TrendResult(50.0, TrendBand.SIDEWAY, "none",
                               detail={"reason": f"need {min_needed}+ candles, have {len(candles)}"})

        closes = np.array([float(c.close) for c in candles], dtype=float)
        highs  = np.array([float(c.high)  for c in candles], dtype=float)
        lows   = np.array([float(c.low)   for c in candles], dtype=float)

        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        _, _, macd_hist = self._macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        roc_arr = self._roc(closes, self.roc_period)
        adx_arr, _, _ = self._adx(closes, highs, lows, self.adx_period)
        atr_arr = self._atr(closes, highs, lows, self.atr_period)

        composites = []
        last_subs = None
        for k in range(n):
            i = -1 - k  # -1, -2, -3, ...
            atr_v = float(atr_arr[i]) if not np.isnan(atr_arr[i]) and atr_arr[i] > 0 else closes[i] * 0.005

            # ── EMA12/26 spread, ATR-normalized, 30pts ─────────────────────
            spread_atr = (ema_f[i] - ema_s[i]) / atr_v
            ema_lean = math.tanh(self.ema_spread_sensitivity * spread_atr)   # -1..1
            ema_score = 50.0 + 30.0 * ema_lean

            # ── MACD histogram, ATR-normalized, 30pts, with dynamic
            # extension penalty — a histogram stretched beyond ~1xATR from
            # zero is chasing a move that already happened, so points get
            # deducted (up to 60%) the further beyond the sweet spot it is
            hist_v = float(macd_hist[i]) if not np.isnan(macd_hist[i]) else 0.0
            hist_atr = hist_v / atr_v
            macd_lean = math.tanh(self.macd_hist_sensitivity * hist_atr)     # -1..1
            sweet = self.macd_extension_sweet_spot
            extension = max(0.0, (abs(hist_atr) - sweet) / sweet)            # 0 within sweet spot, grows beyond
            penalty = min(self.macd_extension_max_penalty, extension * self.macd_extension_max_penalty)
            macd_score = 50.0 + 30.0 * macd_lean * (1.0 - penalty)

            # ── ROC(9), tanh-scaled, 20pts ──────────────────────────────────
            roc_v = float(roc_arr[i]) if not np.isnan(roc_arr[i]) else 0.0
            roc_lean = math.tanh(self.roc_sensitivity * roc_v / 100.0)       # roc_v is a % already
            roc_score = 50.0 + 20.0 * roc_lean

            # ── ADX(14) rising, 20pts, directionless on its own — amplifies
            # whichever way ema+macd+roc already lean ───────────────────
            adx_v      = float(adx_arr[i])     if not np.isnan(adx_arr[i])     else 0.0
            adx_3ago_i = i - 3
            adx_3ago   = float(adx_arr[adx_3ago_i]) if -len(adx_arr) <= adx_3ago_i < len(adx_arr) and not np.isnan(adx_arr[adx_3ago_i]) else adx_v
            adx_rise   = max(0.0, adx_v - adx_3ago)
            rising_factor = math.tanh(self.adx_rising_sensitivity * adx_rise)  # 0..1ish
            prelim_lean = ema_lean + macd_lean + roc_lean   # which way the other 3 already point
            prelim_sign = 1.0 if prelim_lean > 0 else (-1.0 if prelim_lean < 0 else 0.0)
            adx_score = 50.0 + prelim_sign * 20.0 * rising_factor

            subs = {"ema": ema_score, "macd": macd_score, "roc": roc_score, "adx": adx_score}
            composite = sum(subs.values()) - 3 * 50.0   # 4 checks each centered at 50 -> recenter to 50 overall
            composites.append(composite)
            if k == 0:
                last_subs = subs

        bar_weights = self._BAR_WEIGHTS if n == 3 else [1.0 / n] * n
        score = round(sum(c * w for c, w in zip(composites, bar_weights)), 1)
        score = max(0.0, min(100.0, score))
        band = self._classify(score)

        entry_ready = "long" if score > self.ENTRY_LONG_THRESHOLD else \
                     "short" if score < self.ENTRY_SHORT_THRESHOLD else "none"

        return TrendResult(
            score=score, band=band, entry_ready=entry_ready,
            sub_scores={k: round(v, 1) for k, v in last_subs.items()},
            detail={"composites": [round(c, 1) for c in composites]},
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

    # ── Math helpers ────────────────────────────────────────────────────────

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

    @classmethod
    def _macd(cls, closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = cls._ema(closes, fast)
        ema_slow = cls._ema(closes, slow)
        macd_line = ema_fast - ema_slow
        valid = ~np.isnan(macd_line)
        sig_line = np.full(len(closes), np.nan)
        idx = np.where(valid)[0]
        if len(idx) >= signal:
            sig_seg = cls._ema(macd_line[idx], signal)
            sig_line[idx] = sig_seg
        hist = macd_line - sig_line
        return macd_line, sig_line, hist

    @staticmethod
    def _roc(closes: np.ndarray, period: int) -> np.ndarray:
        n = len(closes)
        out = np.full(n, np.nan)
        for i in range(period, n):
            prev = closes[i - period]
            if prev != 0:
                out[i] = (closes[i] - prev) / prev * 100.0
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
