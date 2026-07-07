"""
Layer 1: Market Intelligence Engine

Classifies market regime into scored categories (0-100 each).
The dominant regime drives downstream strategy selection and scoring weights.

Regimes:
  Trend       — EMA alignment, ADX, HH/HL structure, momentum
  Range       — ATR compression, BB squeeze, ADX low, volume decay
  Breakout    — Price compression + expansion, volume surge, liquidity sweep
  Reversal    — Divergence, exhaustion candles, liquidity grab, order flow shift
  HighVol     — ATR expansion, wide BB, realized vol spike
  LowVol      — Opposite of HighVol
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np


class MarketRegime(str, Enum):
    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    regime: MarketRegime
    trend_score: float = 0.0
    range_score: float = 0.0
    breakout_score: float = 0.0
    reversal_score: float = 0.0
    high_vol_score: float = 0.0
    low_vol_score: float = 0.0
    regime_confidence: float = 0.0  # 0-1, how clearly one regime dominates
    detail: dict = field(default_factory=dict)

    def scores(self) -> dict[str, float]:
        return {
            "trend": self.trend_score,
            "range": self.range_score,
            "breakout": self.breakout_score,
            "reversal": self.reversal_score,
            "high_vol": self.high_vol_score,
            "low_vol": self.low_vol_score,
        }


class MarketIntelligenceEngine:
    """Scores market regimes from raw OHLCV candles."""

    def analyze(self, candles: list, mtf_candles: dict = None) -> RegimeResult:
        """Return regime scores and dominant regime. Requires >= 60 candles."""
        if len(candles) < 60:
            return RegimeResult(regime=MarketRegime.UNKNOWN)

        closes = np.array([c.close for c in candles], dtype=float)
        highs  = np.array([c.high  for c in candles], dtype=float)
        lows   = np.array([c.low   for c in candles], dtype=float)
        vols   = np.array([c.volume for c in candles], dtype=float)

        atr_arr  = self._atr(candles, 14)
        atr_val  = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        atr_mean = float(np.nanmean(atr_arr[-50:])) if np.any(~np.isnan(atr_arr[-50:])) else 1e-9
        price    = float(closes[-1])
        atr_pct  = atr_val / price if price > 0 else 0.0

        adx_arr, pdi, mdi = self._adx(candles, 14)
        adx_val  = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0

        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        ema200= self._ema(closes, 200) if len(closes) >= 200 else ema50

        rsi_arr  = self._rsi(closes, 14)
        rsi_val  = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0

        bb_upper, bb_mid, bb_lower = self._bollinger(closes, 20, 2.0)
        bb_width = float((bb_upper[-1] - bb_lower[-1]) / bb_mid[-1]) if bb_mid[-1] != 0 else 0.0
        bb_width_mean = float(np.nanmean(
            [(bb_upper[i] - bb_lower[i]) / bb_mid[i] for i in range(-50, 0)
             if bb_mid[i] != 0]
        )) if len(closes) > 50 else bb_width

        vol_mean20 = float(np.nanmean(vols[-20:])) if len(vols) >= 20 else 1e-9
        vol_mean5  = float(np.nanmean(vols[-5:]))  if len(vols) >= 5  else vol_mean20
        vol_ratio  = vol_mean5 / vol_mean20 if vol_mean20 > 0 else 1.0

        hh_hl = self._detect_hh_hl(highs, lows, lookback=20)
        lh_ll = self._detect_lh_ll(highs, lows, lookback=20)

        obv_slope = self._obv_slope(closes, vols, period=14)

        # ── Trend Score ──────────────────────────────────────────────────
        t_ema_align = self._score_ema_alignment(ema20, ema50, ema200, closes)
        t_adx       = min(100.0, adx_val * 2.5)              # ADX>40 → 100
        t_structure = hh_hl * 80 + lh_ll * (-40)             # HH/HL bullish, LH/LL bearish
        t_structure = max(0.0, min(100.0, 50 + t_structure))
        t_momentum  = (100 - abs(rsi_val - 60)) if rsi_val > 50 else (100 - abs(rsi_val - 40)) * 0.4
        t_momentum  = max(0.0, min(100.0, t_momentum))
        t_vol_obv   = min(100.0, max(0.0, 50 + obv_slope * 50))
        trend_score = (
            t_ema_align * 0.30 + t_adx * 0.25 +
            t_structure * 0.25 + t_momentum * 0.10 + t_vol_obv * 0.10
        )

        # ── Range Score ──────────────────────────────────────────────────
        r_atr_low    = max(0.0, 100 - (atr_val / atr_mean) * 100) if atr_mean > 0 else 50.0
        r_bb_squeeze = max(0.0, 100 - (bb_width / bb_width_mean) * 100) if bb_width_mean > 0 else 50.0
        r_adx_low    = max(0.0, 100 - adx_val * 2.5)
        r_vol_low    = max(0.0, 100 - vol_ratio * 80)
        range_score  = (
            r_atr_low    * 0.30 + r_bb_squeeze * 0.25 +
            r_adx_low   * 0.25 + r_vol_low    * 0.20
        )

        # ── Breakout Score ───────────────────────────────────────────────
        compression_bars = self._compression_bars(highs, lows, 20)
        b_compression = min(100.0, compression_bars * 10)
        b_vol_expand  = min(100.0, max(0.0, (vol_ratio - 1.0) * 60))
        b_atr_expand  = min(100.0, max(0.0, (atr_val / atr_mean - 1.0) * 80))
        b_liq_sweep   = self._liquidity_sweep_score(highs, lows, closes, 20)
        breakout_score = (
            b_compression * 0.30 + b_vol_expand  * 0.25 +
            b_atr_expand  * 0.25 + b_liq_sweep   * 0.20
        )

        # ── Reversal Score ───────────────────────────────────────────────
        rv_divergence = self._rsi_divergence_score(closes, rsi_arr, 14)
        rv_exhaustion = self._exhaustion_score(candles, atr_arr)
        rv_liq_grab   = self._liquidity_grab_score(highs, lows, closes, 10)
        rv_order_flow = self._order_flow_shift_score(closes, vols, 10)
        reversal_score = (
            rv_divergence * 0.30 + rv_exhaustion * 0.25 +
            rv_liq_grab   * 0.25 + rv_order_flow * 0.20
        )

        # ── Volatility Scores ────────────────────────────────────────────
        atr_ratio    = atr_val / atr_mean if atr_mean > 0 else 1.0
        hv_score     = min(100.0, max(0.0, (atr_ratio - 1.0) * 80 + bb_width / max(bb_width_mean, 1e-9) * 40))
        lv_score     = max(0.0, 100.0 - hv_score)

        # ── Dominant Regime ──────────────────────────────────────────────
        scores = {
            MarketRegime.TREND:          trend_score,
            MarketRegime.RANGE:          range_score,
            MarketRegime.BREAKOUT:       breakout_score,
            MarketRegime.REVERSAL:       reversal_score,
            MarketRegime.HIGH_VOLATILITY: hv_score,
            MarketRegime.LOW_VOLATILITY:  lv_score,
        }
        dominant = max(scores, key=scores.__getitem__)
        dom_score = scores[dominant]
        second_best = sorted(scores.values(), reverse=True)[1]
        confidence = (dom_score - second_best) / 100.0

        return RegimeResult(
            regime=dominant,
            trend_score=round(trend_score, 1),
            range_score=round(range_score, 1),
            breakout_score=round(breakout_score, 1),
            reversal_score=round(reversal_score, 1),
            high_vol_score=round(hv_score, 1),
            low_vol_score=round(lv_score, 1),
            regime_confidence=round(max(0.0, min(1.0, confidence)), 3),
            detail={
                "adx": round(adx_val, 1),
                "atr_pct": round(atr_pct * 100, 3),
                "bb_width": round(bb_width * 100, 3),
                "vol_ratio": round(vol_ratio, 2),
                "rsi": round(rsi_val, 1),
                "hh_hl": hh_hl,
                "lh_ll": lh_ll,
            },
        )

    # ── Indicators ──────────────────────────────────────────────────────

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        k = 2.0 / (period + 1)
        result[period - 1] = values[:period].mean()
        for i in range(period, len(values)):
            result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(values)
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = np.full(len(values), np.nan)
        avg_l = np.full(len(values), np.nan)
        if len(gain) < period:
            return avg_g
        avg_g[period] = gain[:period].mean()
        avg_l[period] = loss[:period].mean()
        for i in range(period + 1, len(values)):
            avg_g[i] = (avg_g[i-1] * (period-1) + gain[i-1]) / period
            avg_l[i] = (avg_l[i-1] * (period-1) + loss[i-1]) / period
        rs  = np.where(avg_l == 0, 100.0, avg_g / np.where(avg_l == 0, 1.0, avg_l))
        out = 100 - (100 / (1 + rs))
        out[:period] = np.nan
        return out

    @staticmethod
    def _atr(candles: list, period: int = 14) -> np.ndarray:
        n = len(candles)
        tr = np.full(n, np.nan)
        for i in range(1, n):
            h, l, pc = candles[i].high, candles[i].low, candles[i-1].close
            tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        result = np.full(n, np.nan)
        if n > period:
            result[period] = float(np.nanmean(tr[1:period+1]))
            for i in range(period+1, n):
                result[i] = (result[i-1] * (period-1) + tr[i]) / period
        return result

    @staticmethod
    def _adx(candles: list, period: int = 14):
        n = len(candles)
        pdm = np.zeros(n); mdm = np.zeros(n); tr = np.zeros(n)
        for i in range(1, n):
            h, l = candles[i].high, candles[i].low
            ph, pl, pc = candles[i-1].high, candles[i-1].low, candles[i-1].close
            up, dn = h - ph, pl - l
            pdm[i] = up if up > dn and up > 0 else 0.0
            mdm[i] = dn if dn > up and dn > 0 else 0.0
            tr[i]  = max(h - l, abs(h - pc), abs(l - pc))
        s_tr = np.full(n, np.nan); s_pd = np.full(n, np.nan); s_md = np.full(n, np.nan)
        if n > period:
            s_tr[period] = tr[1:period+1].sum()
            s_pd[period] = pdm[1:period+1].sum()
            s_md[period] = mdm[1:period+1].sum()
            for i in range(period+1, n):
                s_tr[i] = s_tr[i-1] - s_tr[i-1]/period + tr[i]
                s_pd[i] = s_pd[i-1] - s_pd[i-1]/period + pdm[i]
                s_md[i] = s_md[i-1] - s_md[i-1]/period + mdm[i]
        pdi  = np.where(s_tr > 0, 100 * s_pd / s_tr, 0.0)
        mdi  = np.where(s_tr > 0, 100 * s_md / s_tr, 0.0)
        dsum = pdi + mdi
        dx   = np.where(dsum > 0, 100 * np.abs(pdi - mdi) / np.where(dsum > 0, dsum, 1.0), 0.0)
        adx  = np.full(n, np.nan)
        start = 2 * period
        if n > start:
            adx[start] = float(np.nanmean(dx[period:start+1]))
            for i in range(start+1, n):
                if not np.isnan(adx[i-1]):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx, pdi, mdi

    @staticmethod
    def _bollinger(values: np.ndarray, period: int = 20, std_dev: float = 2.0):
        sma = np.full_like(values, np.nan)
        std = np.full_like(values, np.nan)
        for i in range(period - 1, len(values)):
            window = values[i-period+1:i+1]
            sma[i] = window.mean(); std[i] = window.std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower

    @staticmethod
    def _detect_hh_hl(highs: np.ndarray, lows: np.ndarray, lookback: int = 20) -> float:
        """Return 0-1 ratio of how consistently price makes HH and HL."""
        if len(highs) < lookback + 2:
            return 0.0
        h = highs[-lookback:]; l = lows[-lookback:]
        hh = sum(1 for i in range(1, len(h)) if h[i] > h[i-1])
        hl = sum(1 for i in range(1, len(l)) if l[i] > l[i-1])
        return (hh + hl) / (2 * (lookback - 1))

    @staticmethod
    def _detect_lh_ll(highs: np.ndarray, lows: np.ndarray, lookback: int = 20) -> float:
        if len(highs) < lookback + 2:
            return 0.0
        h = highs[-lookback:]; l = lows[-lookback:]
        lh = sum(1 for i in range(1, len(h)) if h[i] < h[i-1])
        ll = sum(1 for i in range(1, len(l)) if l[i] < l[i-1])
        return (lh + ll) / (2 * (lookback - 1))

    @staticmethod
    def _score_ema_alignment(ema20, ema50, ema200, closes) -> float:
        """Score 0-100 based on EMA alignment and price position."""
        c = float(closes[-1])
        e20 = float(ema20[-1]) if not np.isnan(ema20[-1]) else c
        e50 = float(ema50[-1]) if not np.isnan(ema50[-1]) else c
        e200= float(ema200[-1])if not np.isnan(ema200[-1]) else c
        score = 0.0
        # Bullish alignment
        if c > e20:  score += 20
        if c > e50:  score += 15
        if c > e200: score += 15
        if e20 > e50:  score += 20
        if e50 > e200: score += 20
        if e20 > e200: score += 10
        return min(100.0, score)

    @staticmethod
    def _obv_slope(closes: np.ndarray, volumes: np.ndarray, period: int = 14) -> float:
        """OBV slope normalized to -1..+1."""
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            obv[i] = obv[i-1] + (volumes[i] if closes[i] > closes[i-1] else
                                  -volumes[i] if closes[i] < closes[i-1] else 0)
        if len(obv) < period + 1:
            return 0.0
        recent = obv[-period:]
        if recent.std() == 0:
            return 0.0
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        norm  = slope / (recent.std() + 1e-9)
        return float(np.clip(norm, -1.0, 1.0))

    @staticmethod
    def _compression_bars(highs: np.ndarray, lows: np.ndarray, lookback: int = 20) -> float:
        """Count how many bars have contracted range (below 50% of ATR mean)."""
        if len(highs) < lookback:
            return 0.0
        ranges = highs[-lookback:] - lows[-lookback:]
        mean_r = ranges.mean()
        compressed = sum(1 for r in ranges if r < mean_r * 0.5)
        return compressed / lookback * 10   # scale to 0-10

    @staticmethod
    def _liquidity_sweep_score(highs, lows, closes, lookback: int = 20) -> float:
        """Detect if recent bar swept a prior high/low (stop hunt indicator)."""
        if len(highs) < lookback:
            return 0.0
        prev_high = float(highs[-lookback:-1].max())
        prev_low  = float(lows[-lookback:-1].min())
        last_high = float(highs[-1])
        last_low  = float(lows[-1])
        last_close= float(closes[-1])
        swept_high = last_high > prev_high and last_close < prev_high
        swept_low  = last_low  < prev_low  and last_close > prev_low
        if swept_high or swept_low:
            return 80.0
        near_high = (prev_high - last_close) / (prev_high + 1e-9) < 0.002
        near_low  = (last_close - prev_low)  / (prev_low  + 1e-9) < 0.002
        return 40.0 if (near_high or near_low) else 10.0

    @staticmethod
    def _rsi_divergence_score(closes: np.ndarray, rsi_arr: np.ndarray, lookback: int = 14) -> float:
        """Detect bullish/bearish RSI divergence."""
        if len(closes) < lookback * 2:
            return 0.0
        c_recent = closes[-lookback:]
        r_recent = rsi_arr[-lookback:]
        if np.any(np.isnan(r_recent)):
            return 0.0
        price_up = c_recent[-1] > c_recent[0]
        rsi_up   = r_recent[-1] > r_recent[0]
        # Bearish divergence: price up but RSI down (reversal signal)
        if price_up and not rsi_up:
            magnitude = (c_recent[-1] - c_recent[0]) / c_recent[0] * 100
            return min(100.0, 40 + magnitude * 20)
        # Bullish divergence: price down but RSI up
        if not price_up and rsi_up:
            magnitude = (c_recent[0] - c_recent[-1]) / c_recent[0] * 100
            return min(100.0, 40 + magnitude * 20)
        return 10.0

    @staticmethod
    def _exhaustion_score(candles: list, atr_arr: np.ndarray) -> float:
        """Detect exhaustion candles (very large body, volume spike, pin bar)."""
        if len(candles) < 5:
            return 0.0
        c = candles[-1]
        atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        if atr == 0:
            return 0.0
        body   = abs(c.close - c.open)
        upper  = c.high - max(c.open, c.close)
        lower  = min(c.open, c.close) - c.low
        total  = c.high - c.low
        # Pin bar / hammer
        if total > 0:
            pin_ratio = max(upper, lower) / total
            body_ratio= body / total
            if pin_ratio > 0.6 and body_ratio < 0.3:
                return 70.0
        # Large body exhaustion (> 2x ATR)
        if body > 2 * atr:
            return 60.0
        return 10.0

    @staticmethod
    def _liquidity_grab_score(highs, lows, closes, lookback: int = 10) -> float:
        """Score liquidity grab: swept prev high/low then snapped back."""
        if len(highs) < lookback:
            return 0.0
        prev_high = float(highs[-lookback:-1].max())
        prev_low  = float(lows[-lookback:-1].min())
        c = float(closes[-1])
        h = float(highs[-1])
        l = float(lows[-1])
        if h > prev_high and c < prev_high:
            return 85.0   # Grabbed upper liquidity and rejected
        if l < prev_low  and c > prev_low:
            return 85.0   # Grabbed lower liquidity and rejected
        return 10.0

    @staticmethod
    def _order_flow_shift_score(closes: np.ndarray, volumes: np.ndarray, period: int = 10) -> float:
        """Proxy: large-volume candles reversing recent direction."""
        if len(closes) < period + 1:
            return 0.0
        prior_dir = np.sign(closes[-period] - closes[-period-1])
        recent = list(zip(closes[-period:], volumes[-period:]))
        vol_bull = sum(v for c, v in zip(closes[-period:], volumes[-period:])
                       if c > closes[list(closes).index(c) - 1] if list(closes).index(c) > 0 else False)
        # Simplified: large volume candle opposite to prior trend
        last_dir  = np.sign(closes[-1] - closes[-2])
        last_vol  = float(volumes[-1])
        mean_vol  = float(np.mean(volumes[-period:]))
        if last_dir != prior_dir and last_vol > mean_vol * 1.5:
            return 75.0
        if last_dir != prior_dir:
            return 40.0
        return 10.0
