"""
IndicatorEngine — converts raw OHLCV candles → indicator dicts for AdaptiveTradingBot.

All *_score fields are 0–100.
Computes: EMA5/12/20/26/50, CDC Action Zone, RSI14, ADX14, MACD(12,26,9), ATR14, Bollinger, volume avg,
          efficiency ratio, ATR expansion, BB width, and all 12 condition scores.
"""
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .connectors.base import OHLCV
from .strategies.base import BaseStrategy

logger = logging.getLogger("indicator_engine")

SWING_LOOKBACK = 15  # bars for swing high/low (5→15 = 3.75h lookback, wider S/R levels)
EFF_PERIOD     = 10  # Kaufman efficiency ratio lookback
ATR_EXP_PERIOD = 20  # ATR expansion window
BB_PERIOD      = 20  # Bollinger period
BB_STD_MULT    = 2.0 # Bollinger standard deviations


class IndicatorEngine:
    """
    Usage:
        engine = IndicatorEngine()
        candle_15m, candle_1h, candle_4h, ind_15m, ind_1h, ind_4h = \
            engine.compute(c15m, c1h, c4h)
    """

    def compute(
        self,
        c15m: List[OHLCV],
        c1h:  List[OHLCV],
        c4h:  List[OHLCV],
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
        candle_15m = self._make_candle(c15m)
        candle_1h  = self._make_candle(c1h)
        candle_4h  = self._make_candle(c4h)
        ind_15m    = self._compute_ind(c15m)
        ind_1h     = self._compute_ind(c1h)
        ind_4h     = self._compute_ind(c4h)
        return candle_15m, candle_1h, candle_4h, ind_15m, ind_1h, ind_4h

    # ── Candle dict ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_candle(candles: List[OHLCV]) -> Dict[str, Any]:
        if not candles:
            return {}
        c  = candles[-1]
        n  = SWING_LOOKBACK
        ph = max(float(b.high) for b in candles[-n:]) if len(candles) >= n else float(c.high)
        pl = min(float(b.low)  for b in candles[-n:]) if len(candles) >= n else float(c.low)
        return {
            "open":         float(c.open),
            "high":         float(c.high),
            "low":          float(c.low),
            "close":        float(c.close),
            "volume":       float(c.volume),
            "pattern_high": ph,
            "pattern_low":  pl,
        }

    # ── Full indicator dict ──────────────────────────────────────────────────

    def _compute_ind(self, candles: List[OHLCV]) -> Dict[str, Any]:
        if len(candles) < 30:
            return {}

        closes  = [float(c.close)  for c in candles]
        volumes = [float(c.volume) for c in candles]
        n = len(candles) - 1

        # ── Standard indicators ─────────────────────────────────────────────
        ema5_arr  = BaseStrategy.ema(closes, 5)
        ema12_arr = BaseStrategy.ema(closes, 12)
        ema20_arr = BaseStrategy.ema(closes, 20)
        ema26_arr = BaseStrategy.ema(closes, 26)
        ema50_arr = BaseStrategy.ema(closes, 50) if len(closes) >= 50 else ema20_arr

        ema5  = self._safe(ema5_arr,  n)
        ema12 = self._safe(ema12_arr, n)
        ema20 = self._safe(ema20_arr, n)
        ema26 = self._safe(ema26_arr, n)
        ema50 = self._safe(ema50_arr, n)
        price = closes[n]

        rsi_arr = BaseStrategy.rsi(closes, 14)
        rsi     = self._safe(rsi_arr, n, default=50.0)

        adx_arr, pdi_arr, mdi_arr = BaseStrategy.adx(candles, 14)
        adx = self._safe(adx_arr, n, default=20.0)
        pdi = self._safe(pdi_arr, n, default=20.0)
        mdi = self._safe(mdi_arr, n, default=20.0)

        atr_arr = BaseStrategy.atr(candles, 14)
        atr     = self._safe(atr_arr, n, default=price * 0.01)

        macd_arr, sig_arr, hist_arr = BaseStrategy.macd(closes, 12, 26, 9)
        macd      = self._safe(macd_arr,  n, 0.0)
        macd_sig  = self._safe(sig_arr,   n, 0.0)
        macd_hist = self._safe(hist_arr,  n, 0.0)

        vol_cur = volumes[n]
        vol_avg = sum(volumes[max(0, n - 19):n + 1]) / min(20, n + 1)

        # ── Derived metrics ─────────────────────────────────────────────────
        eff_ratio = self._efficiency_ratio(closes, EFF_PERIOD)
        atr_exp   = self._atr_expansion(atr_arr, n, ATR_EXP_PERIOD)
        bb_mid, bb_upper, bb_lower, bb_width = self._bollinger(
            closes, n, BB_PERIOD, BB_STD_MULT
        )
        bb_position = (price - bb_lower) / max(bb_upper - bb_lower, 1e-9)
        bb_position = float(np.clip(bb_position, 0.0, 1.0))

        # CDC Action Zone: fast EMA12 vs slow EMA26.
        ema12_prev = self._safe(ema12_arr, max(0, n - 1), ema12)
        ema26_prev = self._safe(ema26_arr, max(0, n - 1), ema26)
        cdc_bull = ema12 > ema26
        cdc_bear = ema12 < ema26
        cdc_cross_up = ema12_prev <= ema26_prev and ema12 > ema26
        cdc_cross_down = ema12_prev >= ema26_prev and ema12 < ema26
        cdc_spread_atr = (ema12 - ema26) / max(atr, 1e-9)

        ema20_prev  = self._safe(ema20_arr, max(0, n - 5), ema20)
        ema20_slope = (ema20 - ema20_prev) / max(abs(ema20_prev), 1e-9)

        # ── Condition scores (0–100) ────────────────────────────────────────
        # trend_score: EMA alignment + ADX
        ema_bull = price > ema5 > ema20
        ema_bear = price < ema5 < ema20
        trend_score = float(np.clip(
            (50 if (ema_bull or ema_bear) else 20) + min(50.0, (adx - 10) * 2.5), 0, 100))

        # momentum_score: MACD hist direction & strength
        macd_strength = abs(macd_hist) / max(atr * 0.05, 1e-9)
        macd_dir = 1 if macd_hist > 0 else -1
        momentum_score = float(np.clip(50.0 + macd_dir * min(50.0, macd_strength * 15), 0, 100))

        # volume_score
        volume_score = float(np.clip((vol_cur / max(vol_avg, 1e-9)) * 60.0, 0, 100))

        # pattern_score: clarity of most recent candle vs recent range
        recent = candles[-10:]
        hl_range = max(float(b.high) for b in recent) - min(float(b.low) for b in recent)
        c_range  = float(candles[-1].high) - float(candles[-1].low)
        pattern_score = float(np.clip(
            50 + (c_range / max(hl_range / 10, 1e-9) - 1) * 25, 0, 100))

        # structure_score: price vs EMA stack
        if price > ema20 > ema50:
            structure_score = 85.0
        elif price > ema20:
            structure_score = 65.0
        elif price < ema20 < ema50:
            structure_score = 15.0
        elif price < ema20:
            structure_score = 35.0
        else:
            structure_score = 50.0

        # atr_score: prefer non-expanded volatility
        atr_score = float(np.clip((2.0 - atr_exp) * 50.0, 0, 100))

        # rsi_score: how close to neutral (both directions benefit from non-extreme RSI)
        rsi_score = float(np.clip(100 - abs(rsi - 50) * 2, 0, 100))

        # sweep_score: wick ratio (large wick = potential sweep/reversal)
        c_last = candles[-1]
        upper_wick = float(c_last.high) - max(float(c_last.close), float(c_last.open))
        lower_wick = min(float(c_last.close), float(c_last.open)) - float(c_last.low)
        total_range = max(float(c_last.high) - float(c_last.low), 1e-9)
        sweep_score = float(np.clip(max(upper_wick, lower_wick) / total_range * 100, 0, 100))

        # divergence_score: price vs RSI direction agreement
        price_up = closes[n] > closes[max(0, n - 5)]
        rsi_n5   = self._safe(rsi_arr, max(0, n - 5), rsi)
        rsi_up   = rsi > rsi_n5
        divergence_score = 25.0 if price_up != rsi_up else 75.0

        # ema_score: EMA5 vs EMA20 spread
        ema_spread = (ema5 - ema20) / max(ema20, 1e-9)
        ema_score  = float(np.clip(50.0 + ema_spread * 5000, 0, 100))

        # vwap_score: use EMA50 as VWAP proxy (no tick data)
        vwap_diff  = (price - ema50) / max(ema50, 1e-9)
        vwap_score = float(np.clip(50.0 + vwap_diff * 2000, 0, 100))

        # macd_score
        macd_score = 80.0 if macd_hist > 0 else 30.0

        # 4H-regime-style scores (also computed for 15m/1h for completeness)
        ema20_slope_score = float(np.clip(50.0 + ema20_slope * 5000, 0, 100))
        adx_score_r  = float(np.clip(adx * 2.5, 0, 100))
        eff_score_r  = float(np.clip(eff_ratio * 100.0, 0, 100))
        vol_score_r  = float(np.clip((vol_cur / max(vol_avg, 1e-9)) * 60.0, 0, 100))

        return {
            # Raw values
            "adx":         adx,
            "pdi":         pdi,
            "mdi":         mdi,
            "rsi":         rsi,
            "macd":        macd,
            "macd_signal": macd_sig,
            "macd_hist":   macd_hist,
            "ema5":        ema5,
            "ema12":       ema12,
            "ema20":       ema20,
            "ema26":       ema26,
            "ema50":       ema50,
            # CDC Action Zone
            "cdc_bull":       cdc_bull,
            "cdc_bear":       cdc_bear,
            "cdc_cross_up":   cdc_cross_up,
            "cdc_cross_down": cdc_cross_down,
            "cdc_spread_atr": cdc_spread_atr,
            "atr":         atr,
            "volume":      vol_cur,
            "vol_avg":     vol_avg,
            # Derived
            "eff_ratio":   eff_ratio,
            "atr_exp":     atr_exp,
            "bb_mid":      bb_mid,
            "bb_upper":    bb_upper,
            "bb_lower":    bb_lower,
            "bb_width":    bb_width,
            "bb_position": bb_position,
            # Condition scores
            "trend_score":      trend_score,
            "momentum_score":   momentum_score,
            "volume_score":     volume_score,
            "pattern_score":    pattern_score,
            "structure_score":  structure_score,
            "atr_score":        atr_score,
            "rsi_score":        rsi_score,
            "sweep_score":      sweep_score,
            "divergence_score": divergence_score,
            "ema_score":        ema_score,
            "vwap_score":       vwap_score,
            "macd_score":       macd_score,
            # Regime scores
            "ema20_slope_score": ema20_slope_score,
            "adx_score":         adx_score_r,
            "eff_score":         eff_score_r,
            "vol_score":         vol_score_r,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(arr, idx: int, default: float = 0.0) -> float:
        try:
            v = float(arr[idx])
            return default if math.isnan(v) else v
        except (IndexError, TypeError, ValueError):
            return default

    @staticmethod
    def _efficiency_ratio(closes: List[float], period: int) -> float:
        n = len(closes)
        if n < period + 1:
            return 0.5
        net   = abs(closes[-1] - closes[-(period + 1)])
        total = sum(abs(closes[-i] - closes[-i - 1]) for i in range(1, period + 1))
        return float(np.clip(net / max(total, 1e-9), 0, 1))

    @staticmethod
    def _atr_expansion(atr_arr, n: int, period: int) -> float:
        start = max(0, n - period + 1)
        vals  = [float(atr_arr[i]) for i in range(start, n + 1)
                 if not math.isnan(float(atr_arr[i]))]
        if not vals:
            return 1.0
        avg = sum(vals) / len(vals)
        cur = float(atr_arr[n])
        return cur / max(avg, 1e-9)

    @staticmethod
    def _bollinger(
        closes: List[float], n: int, period: int, std_mult: float = 2.0
    ) -> Tuple[float, float, float, float]:
        """Return Bollinger mid, upper, lower and normalized width."""
        window = closes[max(0, n - period + 1):n + 1]
        if not window:
            price = float(closes[n]) if closes else 0.0
            return price, price, price, 0.0
        mean = sum(window) / len(window)
        if len(window) < 2:
            return mean, mean, mean, 0.0
        stdev = math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
        upper = mean + std_mult * stdev
        lower = mean - std_mult * stdev
        width = float(np.clip((upper - lower) / max(abs(mean), 1e-9), 0, 1))
        return float(mean), float(upper), float(lower), width

    @staticmethod
    def _bb_width(closes: List[float], n: int, period: int) -> float:
        # Backward-compatible helper for any older caller.
        return IndicatorEngine._bollinger(closes, n, period, BB_STD_MULT)[3]
