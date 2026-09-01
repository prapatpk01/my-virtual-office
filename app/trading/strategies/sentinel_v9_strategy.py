"""Sentinel V9 — Scored Setup Execution.

Hybrid architecture requested for production:
- 15M analysis engine ports the useful ideas from the Pine v6.2 study:
  trend/quality, confirmed structure, S/R + Fib value, RSI/SMA + HMA timing,
  four setup families (PB/LQ/BO/REV), and an interpretable 0..10 score.
- 5M remains Sentinel execution: price-action confirmation, quality close,
  fee-aware structure risk, anti-chase, actual-fill synchronization.
- Scoring is intentionally not an AND-stack of hard filters. Different setup
  families have different score thresholds so PB/LQ stay responsive while
  REV/direct breakout require stronger evidence.
- TP1 stays +1R close 50% -> runner SL +0.15R.
- TP2 is dynamic 1.5..2.5R when visible 15M structure/Fib supports it,
  otherwise falls back to 2R.
"""
from __future__ import annotations

import math
import numpy as np

from .base import Signal, SignalType
from .sentinel_v81_strategy import SentinelV81Strategy


class SentinelV9Strategy(SentinelV81Strategy):
    VERSION = "9.0"

    # Faster confirmed structure than the Pine display default (4/4).
    # 2/2 on 15M confirms in ~30m rather than ~60m while remaining non-lookahead.
    PIVOT_SPAN = 2
    BOS_BUFFER_ATR = 0.05
    SWEEP_BUFFER_ATR = 0.05
    EMA_VALUE_ATR = 0.28
    RETEST_BARS = 5

    # Pine-inspired breakout context.
    BO_MAX_DIST_ATR = 1.50
    BO_ROOM_MIN_ATR = 1.20
    BO_BODY_MIN_ATR = 0.40
    BO_RSI_LONG_MAX = 68.0
    BO_RSI_SHORT_MIN = 32.0

    # Setup-specific minimum score. This is deliberately looser than
    # Pine's global Grade A rule so the bot does not return to no-trade mode.
    SCORE_MIN = {
        "PB": 6.0,
        "LQ": 6.0,
        "BO_DIRECT": 7.0,
        "BO_RETEST": 6.5,
        "REV": 7.5,
    }

    DYNAMIC_TP_MIN_R = 1.50
    DYNAMIC_TP_MAX_R = 2.50
    DYNAMIC_TP_FALLBACK_R = 2.00

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV9({symbol})"
        self._latest_analysis: dict = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _pivot_points(candles: list, span: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        highs: list[tuple[int, float]] = []
        lows: list[tuple[int, float]] = []
        n = len(candles)
        if n < span * 2 + 3:
            return highs, lows
        for i in range(span, n - span):
            h = float(candles[i].high)
            l = float(candles[i].low)
            hwin = [float(c.high) for c in candles[i - span:i + span + 1]]
            lwin = [float(c.low) for c in candles[i - span:i + span + 1]]
            if h >= max(hwin):
                highs.append((i, h))
            if l <= min(lwin):
                lows.append((i, l))
        return highs, lows

    @staticmethod
    def _nearest_above(levels: list[float], price: float) -> float | None:
        xs = [float(x) for x in levels if np.isfinite(x) and float(x) > price]
        return min(xs) if xs else None

    @staticmethod
    def _nearest_below(levels: list[float], price: float) -> float | None:
        xs = [float(x) for x in levels if np.isfinite(x) and float(x) < price]
        return max(xs) if xs else None

    # ------------------------------------------------------------------
    # 15M Pine-inspired analysis engine
    # ------------------------------------------------------------------
    def _analysis_15m(self, candles: list) -> dict:
        if len(candles) < max(self.MIN_15M_BARS, 55):
            return {"ready": False, "direction": None, "reason": "15M analysis warmup"}

        closes = [float(c.close) for c in candles]
        opens = [float(c.open) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        volumes = [float(c.volume or 0.0) for c in candles]

        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)
        ema200 = self.ema(closes, 200) if len(closes) >= 200 else np.full(len(closes), np.nan)
        hma16 = self.hma(closes, 16)
        atr = self.atr(candles, 14)
        adx, di_plus, di_minus = self.adx(candles, 14)
        rsi = self.rsi(closes, 14)
        rsi_sma = self.sma(list(rsi), 14)
        chop = self._choppiness(candles, 14)

        vals = [ema20[-1], ema50[-1], hma16[-1], atr[-1], adx[-1], rsi[-1], rsi_sma[-1]]
        if chop is None or not self._finite(*vals):
            return {"ready": False, "direction": None, "reason": "15M indicators unavailable"}

        close = closes[-1]
        open_ = opens[-1]
        high = highs[-1]
        low = lows[-1]
        atr_now = max(float(atr[-1]), 1e-12)
        e20 = float(ema20[-1])
        e50 = float(ema50[-1])
        e200 = float(ema200[-1]) if np.isfinite(ema200[-1]) else None
        h16 = float(hma16[-1])
        rsi_now = float(rsi[-1])
        rsi_sma_now = float(rsi_sma[-1])
        adx_now = float(adx[-1])
        di_p = float(di_plus[-1]) if np.isfinite(di_plus[-1]) else 0.0
        di_m = float(di_minus[-1]) if np.isfinite(di_minus[-1]) else 0.0
        chop_now = float(chop)

        rng = max(high - low, 1e-12)
        body_eff = abs(close - open_) / rng
        clv = (close - low) / rng
        body_atr = abs(close - open_) / atr_now

        ema20_slope = (e20 - float(ema20[-4])) / atr_now
        ema50_slope = (e50 - float(ema50[-6])) / atr_now
        ema_sep = abs(e20 - e50) / atr_now
        hma_slope = (h16 - float(hma16[-3])) / atr_now

        bull_stack = e20 > e50
        bear_stack = e20 < e50
        macro_bull = (e200 is not None and e50 > e200)
        macro_bear = (e200 is not None and e50 < e200)
        bull_trend = bull_stack and close > e50 and ema20_slope > 0
        bear_trend = bear_stack and close < e50 and ema20_slope < 0
        strong_bull = bull_trend and macro_bull and close > e20 and ema50_slope >= 0
        strong_bear = bear_trend and macro_bear and close < e20 and ema50_slope <= 0
        trend_dir = 2 if strong_bull else 1 if bull_trend else -2 if strong_bear else -1 if bear_trend else 0

        # Interpretable trend quality, 0..10. EMA200 is a bonus, not a hard gate.
        align_bull = 2.0 if bull_stack and macro_bull else 1.0 if bull_stack else 0.0
        align_bear = 2.0 if bear_stack and macro_bear else 1.0 if bear_stack else 0.0
        slope_bull = 2.0 if ema20_slope > 0.10 else 1.0 if ema20_slope > 0 else 0.0
        slope_bear = 2.0 if ema20_slope < -0.10 else 1.0 if ema20_slope < 0 else 0.0
        sep_score = 2.0 if ema_sep >= 0.60 else 1.0 if ema_sep >= 0.20 else 0.0
        adx_score = 2.0 if adx_now >= 25 else 1.0 if adx_now >= 18 else 0.0
        chop_score = 2.0 if chop_now <= 45 else 1.0 if chop_now <= 55 else 0.0
        bull_quality10 = self._clamp(align_bull + slope_bull + sep_score + adx_score + chop_score, 0.0, 10.0)
        bear_quality10 = self._clamp(align_bear + slope_bear + sep_score + adx_score + chop_score, 0.0, 10.0)

        # Confirmed pivots / structure.
        phs, pls = self._pivot_points(candles, self.PIVOT_SPAN)
        last_ph = phs[-1] if phs else None
        prev_ph = phs[-2] if len(phs) >= 2 else None
        last_pl = pls[-1] if pls else None
        prev_pl = pls[-2] if len(pls) >= 2 else None

        hh = bool(last_ph and prev_ph and last_ph[1] > prev_ph[1])
        lh = bool(last_ph and prev_ph and last_ph[1] < prev_ph[1])
        hl = bool(last_pl and prev_pl and last_pl[1] > prev_pl[1])
        ll = bool(last_pl and prev_pl and last_pl[1] < prev_pl[1])
        structure_state = 2 if hh and hl else -2 if lh and ll else 1 if (hh or hl) else -1 if (lh or ll) else 0

        last_ph_level = float(last_ph[1]) if last_ph else None
        last_pl_level = float(last_pl[1]) if last_pl else None
        prev_close = closes[-2]
        bos_up = bool(
            last_ph_level is not None
            and close > last_ph_level + atr_now * self.BOS_BUFFER_ATR
            and prev_close <= last_ph_level
            and body_eff >= 0.35
        )
        bos_dn = bool(
            last_pl_level is not None
            and close < last_pl_level - atr_now * self.BOS_BUFFER_ATR
            and prev_close >= last_pl_level
            and body_eff >= 0.35
        )
        choch_up = structure_state <= 0 and bos_up
        choch_dn = structure_state >= 0 and bos_dn
        sweep_low = bool(
            last_pl_level is not None
            and low < last_pl_level - atr_now * self.SWEEP_BUFFER_ATR
            and close > last_pl_level and close > open_ and clv >= 0.58
        )
        sweep_high = bool(
            last_ph_level is not None
            and high > last_ph_level + atr_now * self.SWEEP_BUFFER_ATR
            and close < last_ph_level and close < open_ and clv <= 0.42
        )

        prev4_high = max(highs[-5:-1])
        prev4_low = min(lows[-5:-1])
        micro_bos_up = close > prev4_high and close > open_ and body_eff >= 0.35
        micro_bos_dn = close < prev4_low and close < open_ and body_eff >= 0.35

        # S/R levels from recent confirmed pivots.
        ph_levels = [p for _, p in phs[-8:]]
        pl_levels = [p for _, p in pls[-8:]]
        r1 = self._nearest_above(ph_levels, close)
        s1 = self._nearest_below(pl_levels, close)
        near_resist = r1 is not None and abs(close - r1) <= atr_now * 0.50
        near_support = s1 is not None and abs(close - s1) <= atr_now * 0.50
        room_long = 5.0 if r1 is None else max(0.0, (r1 - close) / atr_now)
        room_short = 5.0 if s1 is None else max(0.0, (close - s1) / atr_now)

        # Most recent completed impulse leg for Fib value and extensions.
        bull_leg_hi = last_ph
        bull_leg_lo = None
        if bull_leg_hi:
            prior_lows = [p for p in pls if p[0] < bull_leg_hi[0]]
            bull_leg_lo = prior_lows[-1] if prior_lows else None
        bear_leg_lo = last_pl
        bear_leg_hi = None
        if bear_leg_lo:
            prior_highs = [p for p in phs if p[0] < bear_leg_lo[0]]
            bear_leg_hi = prior_highs[-1] if prior_highs else None

        bull_leg_valid = bool(
            bull_leg_hi and bull_leg_lo and bull_leg_hi[1] > bull_leg_lo[1]
            and (bull_leg_hi[1] - bull_leg_lo[1]) >= 1.5 * atr_now
        )
        bear_leg_valid = bool(
            bear_leg_hi and bear_leg_lo and bear_leg_hi[1] > bear_leg_lo[1]
            and (bear_leg_hi[1] - bear_leg_lo[1]) >= 1.5 * atr_now
        )
        fib_bull = bull_leg_valid and (trend_dir > 0 or (trend_dir == 0 and structure_state > 0))
        fib_bear = bear_leg_valid and (trend_dir < 0 or (trend_dir == 0 and structure_state < 0))

        fib38 = fib50 = fib62 = None
        fib127_long = fib162_long = fib127_short = fib162_short = None
        if fib_bull:
            lo0, hi0 = float(bull_leg_lo[1]), float(bull_leg_hi[1])
            leg = hi0 - lo0
            fib38, fib50, fib62 = hi0 - leg * 0.382, hi0 - leg * 0.500, hi0 - leg * 0.618
            fib127_long, fib162_long = lo0 + leg * 1.272, lo0 + leg * 1.618
        elif fib_bear:
            hi0, lo0 = float(bear_leg_hi[1]), float(bear_leg_lo[1])
            leg = hi0 - lo0
            fib38, fib50, fib62 = lo0 + leg * 0.382, lo0 + leg * 0.500, lo0 + leg * 0.618
            fib127_short, fib162_short = hi0 - leg * 1.272, hi0 - leg * 1.618

        fib_value_long = bool(
            fib_bull and fib38 is not None and fib62 is not None
            and close <= fib38 + 0.18 * atr_now and close >= fib62 - 0.18 * atr_now
        )
        fib_value_short = bool(
            fib_bear and fib38 is not None and fib62 is not None
            and close >= fib38 - 0.18 * atr_now and close <= fib62 + 0.18 * atr_now
        )
        fib_deep_long = bool(
            fib_bull and fib50 is not None and fib62 is not None
            and close <= fib50 + 0.12 * atr_now and close >= fib62 - 0.12 * atr_now
        )
        fib_deep_short = bool(
            fib_bear and fib50 is not None and fib62 is not None
            and close >= fib50 - 0.12 * atr_now and close <= fib62 + 0.12 * atr_now
        )

        ema_zone_hi = max(e20, e50) + self.EMA_VALUE_ATR * atr_now
        ema_zone_lo = min(e20, e50) - self.EMA_VALUE_ATR * atr_now
        in_ema_value = low <= ema_zone_hi and high >= ema_zone_lo
        at_ema20_long = low <= e20 + self.EMA_VALUE_ATR * atr_now and close >= e20
        at_ema20_short = high >= e20 - self.EMA_VALUE_ATR * atr_now and close <= e20
        location_long = near_support or fib_value_long or in_ema_value
        location_short = near_resist or fib_value_short or in_ema_value

        # Momentum engine: scoring evidence, not a stack of mandatory gates.
        rsi_prev = float(rsi[-2]) if np.isfinite(rsi[-2]) else rsi_now
        rsi_sma_prev = float(rsi_sma[-2]) if np.isfinite(rsi_sma[-2]) else rsi_sma_now
        rsi_cross_up = rsi_prev <= rsi_sma_prev and rsi_now > rsi_sma_now
        rsi_cross_dn = rsi_prev >= rsi_sma_prev and rsi_now < rsi_sma_now
        rsi_recovery_long = rsi_cross_up and rsi_now <= 58.0
        rsi_recovery_short = rsi_cross_dn and rsi_now >= 42.0
        rsi_bull = rsi_now > rsi_sma_now and 45.0 <= rsi_now < 72.0
        rsi_bear = rsi_now < rsi_sma_now and 28.0 < rsi_now <= 55.0

        hma_prev_slope = (float(hma16[-2]) - float(hma16[-4])) / atr_now if self._finite(hma16[-2], hma16[-4]) else 0.0
        hma_flip_up = hma_slope > 0 and hma_prev_slope <= 0
        hma_flip_dn = hma_slope < 0 and hma_prev_slope >= 0
        hma_reclaim_up = close > h16 and closes[-2] <= float(hma16[-2]) and hma_slope > 0
        hma_reclaim_dn = close < h16 and closes[-2] >= float(hma16[-2]) and hma_slope < 0
        momentum_long = rsi_recovery_long or (rsi_bull and (hma_flip_up or hma_reclaim_up or micro_bos_up))
        momentum_short = rsi_recovery_short or (rsi_bear and (hma_flip_dn or hma_reclaim_dn or micro_bos_dn))

        vol_hist = [v for v in volumes[-21:-1] if v > 0]
        vol_ma = float(np.mean(vol_hist)) if vol_hist else 0.0
        rvol = volumes[-1] / vol_ma if vol_ma > 0 else 1.0
        volume_ok = rvol >= 0.90
        volume_boost = rvol >= 1.15
        bull_reject = close > open_ and clv >= 0.60 and body_eff >= 0.32
        bear_reject = close < open_ and clv <= 0.40 and body_eff >= 0.32
        price_ema20_dist = abs(close - e20) / atr_now
        adx_rising = np.isfinite(adx[-2]) and adx_now > float(adx[-2])

        # Setup families. These are 15M context/permission, not final execution.
        pb_long = (
            bull_trend and location_long and close >= e50
            and not (near_resist and room_long < 0.45)
            and 35 <= rsi_now <= 70
            and (bull_reject or at_ema20_long or momentum_long)
        )
        pb_short = (
            bear_trend and location_short and close <= e50
            and not (near_support and room_short < 0.45)
            and 30 <= rsi_now <= 65
            and (bear_reject or at_ema20_short or momentum_short)
        )

        lq_long = sweep_low and trend_dir >= 0 and (rsi_now >= rsi_sma_now or hma_slope > 0)
        lq_short = sweep_high and trend_dir <= 0 and (rsi_now <= rsi_sma_now or hma_slope < 0)

        bo_core_long = bos_up and bull_stack and close > e20 and di_p >= di_m and rsi_now > rsi_sma_now
        bo_core_short = bos_dn and bear_stack and close < e20 and di_m >= di_p and rsi_now < rsi_sma_now
        bo_direct_long = (
            bo_core_long and price_ema20_dist <= self.BO_MAX_DIST_ATR
            and rsi_now <= self.BO_RSI_LONG_MAX and body_atr >= self.BO_BODY_MIN_ATR
            and room_long >= self.BO_ROOM_MIN_ATR
            and (adx_now >= 18 or adx_rising or volume_boost)
        )
        bo_direct_short = (
            bo_core_short and price_ema20_dist <= self.BO_MAX_DIST_ATR
            and rsi_now >= self.BO_RSI_SHORT_MIN and body_atr >= self.BO_BODY_MIN_ATR
            and room_short >= self.BO_ROOM_MIN_ATR
            and (adx_now >= 18 or adx_rising or volume_boost)
        )

        # Retest detection using the current confirmed pivot level and a recent
        # close-through event in the prior RETEST_BARS.
        bo_retest_long = False
        bo_retest_short = False
        if last_ph_level is not None:
            for i in range(max(1, len(candles) - self.RETEST_BARS - 1), len(candles) - 1):
                if closes[i] > last_ph_level + self.BOS_BUFFER_ATR * atr_now and closes[i - 1] <= last_ph_level:
                    bo_retest_long = (
                        low <= last_ph_level + 0.20 * atr_now and close > last_ph_level
                        and close > open_ and price_ema20_dist <= self.BO_MAX_DIST_ATR * 1.35
                        and rsi_now <= 72 and room_long >= 0.70 and volume_ok
                    )
        if last_pl_level is not None:
            for i in range(max(1, len(candles) - self.RETEST_BARS - 1), len(candles) - 1):
                if closes[i] < last_pl_level - self.BOS_BUFFER_ATR * atr_now and closes[i - 1] >= last_pl_level:
                    bo_retest_short = (
                        high >= last_pl_level - 0.20 * atr_now and close < last_pl_level
                        and close < open_ and price_ema20_dist <= self.BO_MAX_DIST_ATR * 1.35
                        and rsi_now >= 28 and room_short >= 0.70 and volume_ok
                    )

        rev_long = (
            (sweep_low or near_support or fib_value_long)
            and trend_dir <= 0 and ema20_slope >= -0.10
            and (choch_up or micro_bos_up)
            and (rsi_now >= rsi_sma_now or hma_slope > 0)
        )
        rev_short = (
            (sweep_high or near_resist or fib_value_short)
            and trend_dir >= 0 and ema20_slope <= 0.10
            and (choch_dn or micro_bos_dn)
            and (rsi_now <= rsi_sma_now or hma_slope < 0)
        )

        # Pine priority: PB -> LQ -> REV -> BO.
        def choose_setup(side: str) -> str | None:
            if side == "long":
                if pb_long:
                    return "PB"
                if lq_long:
                    return "LQ"
                if rev_long:
                    return "REV"
                if bo_retest_long:
                    return "BO_RETEST"
                if bo_direct_long:
                    return "BO_DIRECT"
            else:
                if pb_short:
                    return "PB"
                if lq_short:
                    return "LQ"
                if rev_short:
                    return "REV"
                if bo_retest_short:
                    return "BO_RETEST"
                if bo_direct_short:
                    return "BO_DIRECT"
            return None

        setup_long = choose_setup("long")
        setup_short = choose_setup("short")

        # 0..10 score: Trend + Quality + Structure + Location + Momentum.
        long_trend_pts = 2.0 if strong_bull else 1.5 if bull_trend else 1.0 if (rev_long or lq_long) else 0.5 if bull_stack else 0.0
        short_trend_pts = 2.0 if strong_bear else 1.5 if bear_trend else 1.0 if (rev_short or lq_short) else 0.5 if bear_stack else 0.0
        long_quality_pts = 2.0 if bull_quality10 >= 8 else 1.5 if bull_quality10 >= 6 else 1.0 if bull_quality10 >= 4 else 0.5
        short_quality_pts = 2.0 if bear_quality10 >= 8 else 1.5 if bear_quality10 >= 6 else 1.0 if bear_quality10 >= 4 else 0.5
        long_structure_pts = 2.0 if (bos_up or choch_up or sweep_low) else 1.5 if structure_state > 0 else 1.0 if micro_bos_up else 0.5
        short_structure_pts = 2.0 if (bos_dn or choch_dn or sweep_high) else 1.5 if structure_state < 0 else 1.0 if micro_bos_dn else 0.5
        long_location_pts = 2.0 if (near_support or fib_deep_long) else 1.5 if (fib_value_long or in_ema_value or at_ema20_long) else 1.0 if room_long >= 1.0 else 0.0
        short_location_pts = 2.0 if (near_resist or fib_deep_short) else 1.5 if (fib_value_short or in_ema_value or at_ema20_short) else 1.0 if room_short >= 1.0 else 0.0
        long_momentum_pts = 2.0 if (rsi_recovery_long and (hma_flip_up or hma_reclaim_up)) else 1.5 if momentum_long else 1.0 if rsi_bull else 0.0
        short_momentum_pts = 2.0 if (rsi_recovery_short and (hma_flip_dn or hma_reclaim_dn)) else 1.5 if momentum_short else 1.0 if rsi_bear else 0.0

        long_components = {
            "trend": long_trend_pts,
            "quality": long_quality_pts,
            "structure": long_structure_pts,
            "location": long_location_pts,
            "momentum": long_momentum_pts,
        }
        short_components = {
            "trend": short_trend_pts,
            "quality": short_quality_pts,
            "structure": short_structure_pts,
            "location": short_location_pts,
            "momentum": short_momentum_pts,
        }
        long_score = self._clamp(sum(long_components.values()) + (0.25 if volume_boost else 0.0), 0.0, 10.0)
        short_score = self._clamp(sum(short_components.values()) + (0.25 if volume_boost else 0.0), 0.0, 10.0)

        long_min = self.SCORE_MIN.get(setup_long, 99.0) if setup_long else 99.0
        short_min = self.SCORE_MIN.get(setup_short, 99.0) if setup_short else 99.0
        long_ok = bool(setup_long and long_score >= long_min)
        short_ok = bool(setup_short and short_score >= short_min)

        # Room sanity from the Pine logic, but not a broad hard S/R engine.
        if long_ok and near_resist and room_long < 0.35:
            long_ok = False
        if short_ok and near_support and room_short < 0.35:
            short_ok = False

        direction = None
        selected_setup = None
        selected_score = 0.0
        selected_components: dict = {}
        threshold = None

        if long_ok and (not short_ok or long_score > short_score):
            direction, selected_setup, selected_score = "long", setup_long, long_score
            selected_components = long_components
            threshold = long_min
        elif short_ok and (not long_ok or short_score > long_score):
            direction, selected_setup, selected_score = "short", setup_short, short_score
            selected_components = short_components
            threshold = short_min

        # V8.1 strict post-cooldown mode reads _bias_strength. Map the scored
        # analysis to the same 0..3 concept without making it a normal hard gate.
        self._bias_strength = (
            3 if direction and selected_score >= 8.0
            else 2 if direction and selected_score >= 7.0
            else 1 if direction
            else 0
        )

        # Forecast/runner target candidates. Used later to convert visible
        # structure/Fib into a dynamic R target after 5M risk is known.
        target_levels_long = [x for x in [r1, fib127_long, fib162_long] if x is not None and x > close]
        target_levels_short = [x for x in [s1, fib127_short, fib162_short] if x is not None and x < close]

        location_txt = (
            "SUPPORT" if near_support else "RESISTANCE" if near_resist
            else "FIB_50_61.8" if (fib_deep_long or fib_deep_short)
            else "FIB_38.2_61.8" if (fib_value_long or fib_value_short)
            else "EMA_VALUE" if in_ema_value else "MID"
        )
        structure_txt = "HH/HL" if structure_state == 2 else "LH/LL" if structure_state == -2 else "BULLISH" if structure_state == 1 else "BEARISH" if structure_state == -1 else "MIXED"

        return {
            "ready": True,
            "direction": direction,
            "selected_setup": selected_setup,
            "selected_score": round(float(selected_score), 2),
            "score_threshold": threshold,
            "score_long": round(float(long_score), 2),
            "score_short": round(float(short_score), 2),
            "setup_long": setup_long or "NONE",
            "setup_short": setup_short or "NONE",
            "components": selected_components,
            "components_long": long_components,
            "components_short": short_components,
            "trend_dir": trend_dir,
            "trend": "STRONG_BULL" if trend_dir == 2 else "BULL" if trend_dir == 1 else "STRONG_BEAR" if trend_dir == -2 else "BEAR" if trend_dir == -1 else "NEUTRAL",
            "trend_quality_long": round(float(bull_quality10), 1),
            "trend_quality_short": round(float(bear_quality10), 1),
            "structure_state": structure_state,
            "structure": structure_txt,
            "bos_up": bos_up,
            "bos_down": bos_dn,
            "choch_up": choch_up,
            "choch_down": choch_dn,
            "sweep_low": sweep_low,
            "sweep_high": sweep_high,
            "location": location_txt,
            "room_long_atr": round(float(room_long), 2),
            "room_short_atr": round(float(room_short), 2),
            "near_support": near_support,
            "near_resistance": near_resist,
            "fib_active": bool(fib_bull or fib_bear),
            "fib38": round(float(fib38), 8) if fib38 is not None else None,
            "fib50": round(float(fib50), 8) if fib50 is not None else None,
            "fib62": round(float(fib62), 8) if fib62 is not None else None,
            "target_levels_long": target_levels_long,
            "target_levels_short": target_levels_short,
            "ema20": round(e20, 8),
            "ema50": round(e50, 8),
            "ema200": round(e200, 8) if e200 is not None else None,
            "ema20_slope_atr": round(float(ema20_slope), 3),
            "adx": round(adx_now, 1),
            "chop": round(chop_now, 1),
            "di_plus": round(di_p, 1),
            "di_minus": round(di_m, 1),
            "rsi": round(rsi_now, 2),
            "rsi_sma": round(rsi_sma_now, 2),
            "hma_slope_atr": round(float(hma_slope), 3),
            "rvol": round(float(rvol), 2),
            "reason": (
                f"15M {selected_setup} {direction.upper()} score {selected_score:.2f}/{threshold:.2f}"
                if direction and selected_setup and threshold is not None
                else f"15M no qualified setup | L {long_score:.2f}({setup_long or 'NONE'}) / S {short_score:.2f}({setup_short or 'NONE'})"
            ),
        }

    # Keep _bias_15m compatible with inherited slow technical-exit logic.
    def _bias_15m(self, candles: list) -> dict:
        a = self._analysis_15m(candles)
        return {
            "ready": bool(a.get("direction")),
            "direction": a.get("direction"),
            "strength": int(round(float(a.get("selected_score") or 0))),
            "ema20": a.get("ema20"),
            "ema20_slope_atr": a.get("ema20_slope_atr"),
            "rsi": a.get("rsi"),
            "reason": a.get("reason", "15M scored analysis"),
        }

    # ------------------------------------------------------------------
    # 5M execution compatibility by setup family
    # ------------------------------------------------------------------
    def _apply_setup_execution_map(self, setup: dict, setup_family: str | None) -> dict:
        if not setup_family or not setup.get("trigger"):
            return setup
        allowed = {
            "PB": {"PULLBACK_RECLAIM", "MICRO_BREAKOUT"},
            "LQ": {"SWEEP_RECLAIM", "MICRO_BREAKOUT"},
            "BO_DIRECT": {"MICRO_BREAKOUT", "PULLBACK_RECLAIM"},
            "BO_RETEST": {"PULLBACK_RECLAIM", "MICRO_BREAKOUT"},
            "REV": {"SWEEP_RECLAIM", "MICRO_BREAKOUT"},
        }.get(setup_family, set())
        if setup.get("trigger") not in allowed:
            out = dict(setup)
            out["trigger_candidate"] = setup.get("trigger")
            out["trigger"] = None
            blocks = list(out.get("blocks", []))
            blocks.append("SETUP_EXEC_MISMATCH")
            out["blocks"] = list(dict.fromkeys(blocks))
            out["reason"] = f"5M trigger not compatible with 15M {setup_family}"
            return out
        return setup

    def _dynamic_target_r(self, analysis: dict, direction: str, entry: float, risk: float) -> tuple[float, str]:
        if risk <= 0:
            return self.DYNAMIC_TP_FALLBACK_R, "FALLBACK_2R"
        levels = analysis.get("target_levels_long", []) if direction == "long" else analysis.get("target_levels_short", [])
        rr_candidates: list[float] = []
        for level in levels:
            try:
                rr = ((float(level) - entry) / risk) if direction == "long" else ((entry - float(level)) / risk)
            except (TypeError, ValueError):
                continue
            if rr >= self.DYNAMIC_TP_MIN_R:
                rr_candidates.append(rr)
        if not rr_candidates:
            return self.DYNAMIC_TP_FALLBACK_R, "FALLBACK_2R"

        rr = min(rr_candidates)
        rr = self._clamp(rr, self.DYNAMIC_TP_MIN_R, self.DYNAMIC_TP_MAX_R)
        return rr, "STRUCTURE_FIB"

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c15 = self._closed_candle_series(candles, self.FIFTEEN_MIN_MS)
        c5 = self._closed_candle_series(mtf.get("5m", []), self.FIVE_MIN_MS)
        self._latest_15m = c15
        self._latest_5m = c5

        meta = {
            "strategy": "SENTINEL_V9",
            "version": self.VERSION,
            "architecture": "15M_PINE_V62_SCORED_ANALYSIS__5M_V81_EXECUTION_RISK",
            "entry_tf": "5m_closed",
            "mtf_used": "15m_scored_setup__5m_execution",
            "risk_plan": "V8.1_FEE_AWARE_STRUCTURE_SL__TP1_1R_CLOSE50_LOCK+0.15R__TP2_DYNAMIC_1.5_2.5R_FALLBACK2R",
        }

        if len(c15) < max(self.MIN_15M_BARS, 55) or len(c5) < self.MIN_5M_BARS:
            return self._hold(float(current_price), "waiting for closed 15M/5M warmup", meta)

        analysis = self._analysis_15m(c15)
        self._latest_analysis = analysis
        direction = analysis.get("direction")
        setup_family = analysis.get("selected_setup")

        setup = self._snapshot_5m(c5, direction, float(current_price))
        setup = self._apply_setup_execution_map(setup, setup_family)
        meta["analysis_15m"] = analysis
        meta["bias_15m"] = {
            "direction": direction,
            "strength": analysis.get("selected_score"),
            "ema20": analysis.get("ema20"),
            "ema20_slope_atr": analysis.get("ema20_slope_atr"),
            "rsi": analysis.get("rsi"),
        }
        meta["setup_5m"] = setup

        if self._open_position is not None:
            return self._hold(float(current_price), f"managing open {self._open_position} position", meta)

        bar5_ts = int(self._bar_ts(c5[-1]))
        if self._last_5m_evaluated_ts == bar5_ts:
            return self._hold(float(current_price), "5M bar already evaluated", meta)
        self._last_5m_evaluated_ts = bar5_ts

        if self._last_exit_5m_ts is not None:
            elapsed = bar5_ts - self._last_exit_5m_ts
            if elapsed < self.EXIT_COOLDOWN_5M_BARS * self.FIVE_MIN_MS:
                return self._hold(float(current_price), "post-exit 5M cooldown", meta)

        if direction not in {"long", "short"} or not setup_family:
            return self._hold(float(current_price), analysis.get("reason", "waiting for qualified 15M setup"), meta)

        # V8.1 one-filled-entry-per-15M protection.
        current_15m_ts = int(self._bar_ts(c15[-1]))
        if self._last_entry_15m_ts == current_15m_ts:
            meta["setup_5m"] = dict(setup)
            meta["setup_5m"]["blocks"] = ["ONE_ENTRY_PER_15M"]
            return self._hold(float(current_price), "one successful Sentinel entry already used this 15M bar", meta)

        if not setup.get("market_ready", False):
            return self._hold(float(current_price), setup.get("reason", "5M market gate blocked"), meta)
        if not setup.get("trigger"):
            return self._hold(float(current_price), setup.get("reason", "waiting for compatible 5M trigger"), meta)

        entry = float(setup["entry"])
        stop = float(setup["stop_loss"])
        risk = float(setup["risk"])
        tp2_r, tp_source = self._dynamic_target_r(analysis, direction, entry, risk)
        target = entry + tp2_r * risk if direction == "long" else entry - tp2_r * risk
        tp1 = entry + self.TP1_R * risk if direction == "long" else entry - self.TP1_R * risk

        self._open_position = direction
        self._pending_entry = True
        self._entry_price = entry
        self._entry_sl = stop
        self._entry_tp = target
        self._initial_risk = risk
        self._tp1_done = False

        meta.update({
            "direction": direction,
            "setup_family": setup_family,
            "setup_score": analysis.get("selected_score"),
            "setup_score_threshold": analysis.get("score_threshold"),
            "score_components": analysis.get("components", {}),
            "entry_trigger": setup["trigger"],
            "stop_loss": round(stop, 8),
            "take_profit": round(target, 8),
            "tp1_price": round(tp1, 8),
            "rr_ratio": round(float(tp2_r), 3),
            "tp2_r_dynamic": round(float(tp2_r), 3),
            "tp2_source": tp_source,
            "tp1_r": self.TP1_R,
            "tp1_close_pct": self.TP1_CLOSE_PCT,
            "tp1_lock_r": self.TP1_LOCK_R,
        })

        score = float(analysis.get("selected_score") or 0.0)
        confidence = self._clamp(0.58 + score * 0.04, 0.65, 0.95)
        signal_type = SignalType.BUY if direction == "long" else SignalType.SELL
        reason = (
            f"{direction.upper()} {setup_family}/{setup['trigger']} | "
            f"15M score={score:.2f}/{float(analysis.get('score_threshold') or 0):.2f} "
            f"T={analysis.get('trend')} S={analysis.get('structure')} L={analysis.get('location')} | "
            f"ADX5={setup.get('adx')} CHOP5={setup.get('chop')} "
            f"SL={setup.get('sl_atr')}ATR TP2={tp2_r:.2f}R({tp_source})"
        )
        return Signal(signal_type, self.symbol, entry, 0.0, reason, float(confidence), meta)
