"""
Mean-Reversion entry-quality primitives — used by the unified signal
pipeline in adaptive_trading_bot.py (TradingBot._entry_score) for the
SIDEWAY / EXHAUSTION / REVERSAL market states.

These are pure boolean checks on the 15M candle/indicators; the caller
folds them into a single weighted 0-100 entry score instead of a
sequential AND-gate cascade. SL is computed by _step14_sl.
"""

import datetime
from typing import Dict, Optional

import numpy as np


class MeanReversionStrategy:
    """Mean-reversion entry-quality checks (extension/sweep/structure/momentum/
    candle/volume) plus SL computation, reused by the unified entry scorer."""

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

    # ── Over-Extension Filter ────────────────────────────────────────────────

    def _step3_overextension(self, ind: Dict, candle: Dict, direction: str) -> bool:
        """Price must be stretched ≥1.5×ATR from EMA20, or at BB/VWAP extreme."""
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

    # ── Liquidity Sweep ──────────────────────────────────────────────────────

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
            swept_below = low < bb_lo
            return_in   = close > bb_lo
            return sweep_score > 40 and (swept_below and return_in)
        else:
            swept_above = high > bb_up
            return_in   = close < bb_up
            return sweep_score > 40 and (swept_above and return_in)

    # ── Reversal Structure (CHOCH / Minor BOS) ──────────────────────────────

    def _step5_structure(self, ind: Dict, direction: str) -> bool:
        """
        CHOCH proxy: divergence_score (price moved one way but RSI diverges)
        + structure_score showing nascent reversal.
        """
        divergence_score = ind.get("divergence_score", 50.0)
        structure_score  = ind.get("structure_score", 50.0)
        sweep_score      = ind.get("sweep_score", 0.0)

        if direction == "LONG":
            struct_ok = structure_score > 30
        else:
            struct_ok = structure_score < 70

        return divergence_score > 50 and sweep_score > 20 and struct_ok

    # ── Momentum Reversal ────────────────────────────────────────────────────

    def _step6_momentum(self, ind: Dict, direction: str) -> bool:
        """MACD hist turning + RSI confirming."""
        macd_hist = ind.get("macd_hist", 0.0)
        rsi       = ind.get("rsi", 50.0)
        adx       = ind.get("adx", 25.0)

        if direction == "LONG":
            return macd_hist > 0 and rsi > 30 and rsi < 58 and adx < 28
        else:
            return macd_hist < 0 and rsi < 70 and rsi > 42 and adx < 28

    # ── Candle Quality ───────────────────────────────────────────────────────

    def _step7_candle(self, candle: Dict, direction: str) -> bool:
        """
        Body ≥ 60% of range, close near appropriate end.
        Also catches hammer / bullish engulf patterns via wick ratio.
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

    # ── Volume Filter ────────────────────────────────────────────────────────

    def _step8_volume(self, ind: Dict) -> bool:
        """Volume spike ≥1.2× avg, OBV direction (divergence_score proxy)."""
        volume    = ind.get("volume", 0.0)
        vol_avg   = ind.get("vol_avg", volume if volume > 0 else 1.0)
        vol_score = ind.get("volume_score", 0.0)

        vol_spike = volume >= vol_avg * 1.2
        vol_active = vol_score >= 55
        return vol_spike or vol_active

    # ── Stop-Loss Computation ────────────────────────────────────────────────

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
