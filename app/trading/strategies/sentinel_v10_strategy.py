"""Sentinel V10 — Momentum + Location Forecast Core.

15M decides WHY:
- EMA20 trend/value
- RSI14/SMA14 primary momentum
- MACD momentum quality
- KDJ fast acceleration
- ADX/CHOP/ATR-activity regime (2-of-3)
- confirmed S/R location
- setup families: PULLBACK / BREAKOUT_RETEST / SWEEP_REVERSAL
- interpretable score 0..10
- forecast is advisory/targeting, never an entry hard gate

5M decides WHEN:
- closed-bar price action
- anti-chase
- fee-aware local structure SL
- TP1 +1R close 50%, runner SL +0.15R
- TP2 selected from S/R in 1.5..2.5R, fallback 2R
"""
from __future__ import annotations

import numpy as np

from .base import Signal, SignalType
from .sentinel_v81_strategy import SentinelV81Strategy
from ..engines.position_manager import PositionUpdate


class SentinelV10Strategy(SentinelV81Strategy):
    VERSION = "10.0"
    PIVOT_SPAN = 2
    RSI_MEMORY_BARS = 6
    RETEST_BARS = 5

    ADX_FLOOR = 12.0
    CHOP_CEILING = 64.0
    ATR_ACTIVITY_FLOOR = 0.65
    REGIME_MIN_PASS = 2

    MAX_TRIGGER_CHASE_ATR = 0.30
    SL_BUFFER_ATR = 0.20
    MIN_SL_ATR = 0.90
    MAX_SL_ATR = 1.80
    MIN_ECONOMIC_RISK_PCT = 0.0040

    SCORE_MIN = {
        "PULLBACK": 6.0,
        "BREAKOUT_RETEST": 6.5,
        "SWEEP_REVERSAL": 7.0,
    }
    DYNAMIC_TP_MIN_R = 1.50
    DYNAMIC_TP_MAX_R = 2.50
    DYNAMIC_TP_FALLBACK_R = 2.00

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV10({symbol})"
        self._latest_analysis: dict = {}
        self._latest_forecast: dict = {}
        self._tp2_rr_active = self.DYNAMIC_TP_FALLBACK_R

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _cross_up(a: np.ndarray, b: np.ndarray) -> bool:
        return (
            len(a) >= 2 and len(b) >= 2
            and np.isfinite(a[-1]) and np.isfinite(a[-2])
            and np.isfinite(b[-1]) and np.isfinite(b[-2])
            and float(a[-1]) > float(b[-1]) and float(a[-2]) <= float(b[-2])
        )

    @staticmethod
    def _cross_down(a: np.ndarray, b: np.ndarray) -> bool:
        return (
            len(a) >= 2 and len(b) >= 2
            and np.isfinite(a[-1]) and np.isfinite(a[-2])
            and np.isfinite(b[-1]) and np.isfinite(b[-2])
            and float(a[-1]) < float(b[-1]) and float(a[-2]) >= float(b[-2])
        )

    @staticmethod
    def _pivot_points(candles: list, span: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        highs: list[tuple[int, float]] = []
        lows: list[tuple[int, float]] = []
        if len(candles) < span * 2 + 3:
            return highs, lows
        for i in range(span, len(candles) - span):
            h = float(candles[i].high)
            l = float(candles[i].low)
            hwin = [float(c.high) for c in candles[i-span:i+span+1]]
            lwin = [float(c.low) for c in candles[i-span:i+span+1]]
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

    @staticmethod
    def _kdj(candles: list, period: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(candles)
        k = np.full(n, np.nan)
        d = np.full(n, np.nan)
        j = np.full(n, np.nan)
        if n < period:
            return k, d, j
        kp = 50.0
        dp = 50.0
        for i in range(period - 1, n):
            window = candles[i-period+1:i+1]
            hh = max(float(c.high) for c in window)
            ll = min(float(c.low) for c in window)
            close = float(candles[i].close)
            rsv = 50.0 if hh <= ll else (close - ll) / (hh - ll) * 100.0
            kp = (2.0 / 3.0) * kp + (1.0 / 3.0) * rsv
            dp = (2.0 / 3.0) * dp + (1.0 / 3.0) * kp
            k[i] = kp
            d[i] = dp
            j[i] = 3.0 * kp - 2.0 * dp
        return k, d, j

    def _regime_snapshot(self, candles: list) -> dict:
        atr = self.atr(candles, 14)
        adx, _, _ = self.adx(candles, 14)
        chop = self._choppiness(candles, 14)
        if chop is None or not self._finite(atr[-1], adx[-1]):
            return {"ready": False, "pass_count": 0, "market_ready": False}
        atr_now = max(float(atr[-1]), 1e-12)
        hist = [float(x) for x in atr[-21:-1] if np.isfinite(x)]
        atr_med = float(np.median(hist)) if hist else atr_now
        atr_ratio = atr_now / max(atr_med, 1e-12)
        adx_ok = float(adx[-1]) >= self.ADX_FLOOR
        chop_ok = float(chop) < self.CHOP_CEILING
        atr_ok = atr_ratio >= self.ATR_ACTIVITY_FLOOR
        cnt = int(adx_ok) + int(chop_ok) + int(atr_ok)
        return {
            "ready": True,
            "market_ready": cnt >= self.REGIME_MIN_PASS,
            "pass_count": cnt,
            "adx_ok": adx_ok,
            "chop_ok": chop_ok,
            "atr_ok": atr_ok,
            "adx": round(float(adx[-1]), 1),
            "chop": round(float(chop), 1),
            "atr_ratio": round(float(atr_ratio), 2),
            "atr": atr_now,
        }

    def _analysis_15m(self, candles: list) -> dict:
        if len(candles) < max(self.MIN_15M_BARS, 60):
            return {"ready": False, "direction": None, "reason": "15M analysis warmup"}

        closes = [float(c.close) for c in candles]
        opens = [float(c.open) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]

        ema20 = self.ema(closes, 20)
        atr = self.atr(candles, 14)
        rsi = self.rsi(closes, 14)
        rsi_sma = self.sma(list(rsi), 14)
        macd_line, macd_sig, macd_hist = self.macd(closes, 12, 26, 9)
        k, d, j = self._kdj(candles, 9)
        regime = self._regime_snapshot(candles)

        values = [ema20[-1], ema20[-4], atr[-1], rsi[-1], rsi_sma[-1], macd_line[-1], macd_sig[-1], macd_hist[-1], k[-1], d[-1], j[-1]]
        if not regime.get("ready") or not self._finite(*values):
            return {"ready": False, "direction": None, "reason": "15M indicators unavailable"}

        close = closes[-1]
        open_ = opens[-1]
        high = highs[-1]
        low = lows[-1]
        atr_now = max(float(atr[-1]), 1e-12)
        e20 = float(ema20[-1])
        slope = (e20 - float(ema20[-4])) / atr_now
        r = float(rsi[-1])
        rs = float(rsi_sma[-1])
        rsi_cross_up = self._cross_up(rsi, rsi_sma)
        rsi_cross_dn = self._cross_down(rsi, rsi_sma)

        recent_rsi = [float(x) for x in rsi[-self.RSI_MEMORY_BARS:] if np.isfinite(x)]
        recent_min = min(recent_rsi) if recent_rsi else r
        recent_max = max(recent_rsi) if recent_rsi else r
        rsi_rotation_long = rsi_cross_up and recent_min <= 45.0 and 40.0 <= r <= 60.0
        rsi_rotation_short = rsi_cross_dn and recent_max >= 55.0 and 40.0 <= r <= 60.0
        rsi_support_long = r > rs and 45.0 <= r <= 68.0
        rsi_support_short = r < rs and 32.0 <= r <= 55.0

        m = float(macd_line[-1])
        ms = float(macd_sig[-1])
        mh = float(macd_hist[-1])
        mh_prev = float(macd_hist[-2]) if np.isfinite(macd_hist[-2]) else mh
        mh_prev2 = float(macd_hist[-3]) if np.isfinite(macd_hist[-3]) else mh_prev
        macd_long = (m >= ms and mh >= mh_prev) or (mh > mh_prev > mh_prev2)
        macd_short = (m <= ms and mh <= mh_prev) or (mh < mh_prev < mh_prev2)
        macd_accel = mh - mh_prev

        kval = float(k[-1]); dval = float(d[-1]); jval = float(j[-1])
        kprev = float(k[-2]) if np.isfinite(k[-2]) else kval
        dprev = float(d[-2]) if np.isfinite(d[-2]) else dval
        jprev = float(j[-2]) if np.isfinite(j[-2]) else jval
        kdj_cross_up = kval > dval and kprev <= dprev
        kdj_cross_dn = kval < dval and kprev >= dprev
        kdj_long = kdj_cross_up or (jval > jprev and (jprev < 20.0 or jval - jprev >= 8.0))
        kdj_short = kdj_cross_dn or (jval < jprev and (jprev > 80.0 or jprev - jval >= 8.0))

        trend_long = slope > 0.0
        trend_short = slope < 0.0
        price_above = close >= e20
        price_below = close <= e20

        phs, pls = self._pivot_points(candles, self.PIVOT_SPAN)
        ph_levels = [float(p) for _, p in phs[-10:]]
        pl_levels = [float(p) for _, p in pls[-10:]]
        r1 = self._nearest_above(ph_levels, close)
        s1 = self._nearest_below(pl_levels, close)
        last_ph = float(phs[-1][1]) if phs else None
        last_pl = float(pls[-1][1]) if pls else None

        near_support = s1 is not None and abs(close - s1) <= 0.55 * atr_now
        near_resist = r1 is not None and abs(close - r1) <= 0.55 * atr_now
        ema_value = abs(close - e20) <= 0.45 * atr_now or (low <= e20 <= high)

        sweep_long = bool(last_pl is not None and low < last_pl - 0.05 * atr_now and close > last_pl and close > open_)
        sweep_short = bool(last_ph is not None and high > last_ph + 0.05 * atr_now and close < last_ph and close < open_)

        retest_long = False
        retest_short = False
        breakout_level_long = None
        breakout_level_short = None
        if last_ph is not None:
            start = max(1, len(candles) - self.RETEST_BARS - 1)
            for idx in range(start, len(candles) - 1):
                if closes[idx] > last_ph + 0.05 * atr_now and closes[idx - 1] <= last_ph:
                    breakout_level_long = last_ph
            if breakout_level_long is not None:
                retest_long = low <= breakout_level_long + 0.20 * atr_now and close > breakout_level_long and close > open_
        if last_pl is not None:
            start = max(1, len(candles) - self.RETEST_BARS - 1)
            for idx in range(start, len(candles) - 1):
                if closes[idx] < last_pl - 0.05 * atr_now and closes[idx - 1] >= last_pl:
                    breakout_level_short = last_pl
            if breakout_level_short is not None:
                retest_short = high >= breakout_level_short - 0.20 * atr_now and close < breakout_level_short and close < open_

        pb_long = trend_long and (ema_value or near_support) and (rsi_rotation_long or rsi_support_long)
        pb_short = trend_short and (ema_value or near_resist) and (rsi_rotation_short or rsi_support_short)
        bo_long = trend_long and retest_long and (rsi_support_long or macd_long)
        bo_short = trend_short and retest_short and (rsi_support_short or macd_short)
        sw_long = sweep_long and (rsi_rotation_long or kdj_long) and slope >= -0.10
        sw_short = sweep_short and (rsi_rotation_short or kdj_short) and slope <= 0.10

        def score(side: str) -> tuple[float, dict]:
            long = side == "long"
            if long:
                trend_pts = 2.0 if (slope >= 0.10 and price_above) else 1.5 if slope > 0 else 0.5 if price_above else 0.0
                rsi_pts = 2.0 if rsi_rotation_long else 1.5 if rsi_support_long else 0.5 if r >= 45 else 0.0
                macd_pts = 1.5 if (m > ms and mh > 0 and mh >= mh_prev) else 1.0 if macd_long else 0.5 if mh > mh_prev else 0.0
                kdj_pts = 1.0 if kdj_cross_up else 0.75 if kdj_long else 0.25 if jval > jprev else 0.0
                loc_pts = 2.0 if (near_support or sweep_long or retest_long) else 1.5 if ema_value else 0.5 if r1 is None or (r1 - close) >= atr_now else 0.0
            else:
                trend_pts = 2.0 if (slope <= -0.10 and price_below) else 1.5 if slope < 0 else 0.5 if price_below else 0.0
                rsi_pts = 2.0 if rsi_rotation_short else 1.5 if rsi_support_short else 0.5 if r <= 55 else 0.0
                macd_pts = 1.5 if (m < ms and mh < 0 and mh <= mh_prev) else 1.0 if macd_short else 0.5 if mh < mh_prev else 0.0
                kdj_pts = 1.0 if kdj_cross_dn else 0.75 if kdj_short else 0.25 if jval < jprev else 0.0
                loc_pts = 2.0 if (near_resist or sweep_short or retest_short) else 1.5 if ema_value else 0.5 if s1 is None or (close - s1) >= atr_now else 0.0
            cnt = int(regime["pass_count"])
            reg_pts = 1.5 if cnt == 3 else 1.0 if cnt == 2 else 0.0
            comps = {"ema20": trend_pts, "rsi_sma": rsi_pts, "macd": macd_pts, "kdj": kdj_pts, "location": loc_pts, "regime": reg_pts}
            return self._clamp(sum(comps.values()), 0.0, 10.0), comps

        long_score, long_comp = score("long")
        short_score, short_comp = score("short")

        def choose(side: str) -> str | None:
            if side == "long":
                if pb_long: return "PULLBACK"
                if bo_long: return "BREAKOUT_RETEST"
                if sw_long: return "SWEEP_REVERSAL"
            else:
                if pb_short: return "PULLBACK"
                if bo_short: return "BREAKOUT_RETEST"
                if sw_short: return "SWEEP_REVERSAL"
            return None

        setup_long = choose("long")
        setup_short = choose("short")
        long_min = self.SCORE_MIN.get(setup_long, 99.0) if setup_long else 99.0
        short_min = self.SCORE_MIN.get(setup_short, 99.0) if setup_short else 99.0
        long_ok = bool(setup_long and regime["market_ready"] and long_score >= long_min)
        short_ok = bool(setup_short and regime["market_ready"] and short_score >= short_min)

        direction = None; selected_setup = None; selected_score = 0.0; selected_comp = {}; threshold = None
        if long_ok and (not short_ok or long_score > short_score):
            direction, selected_setup, selected_score, selected_comp, threshold = "long", setup_long, long_score, long_comp, long_min
        elif short_ok and (not long_ok or short_score > long_score):
            direction, selected_setup, selected_score, selected_comp, threshold = "short", setup_short, short_score, short_comp, short_min

        self._bias_strength = 3 if direction and selected_score >= 8.0 else 2 if direction and selected_score >= 7.0 else 1 if direction else 0
        fresh_event_long = bool(rsi_cross_up or sweep_long or retest_long)
        fresh_event_short = bool(rsi_cross_dn or sweep_short or retest_short)

        setup_diff = self._clamp((long_score - short_score) / 4.0, -1.0, 1.0)
        ema_dir = self._clamp(slope / 0.25, -1.0, 1.0)
        rsi_dir = self._clamp((r - rs) / 8.0, -1.0, 1.0)
        macd_scale = max(abs(mh), abs(mh_prev), 1e-9)
        macd_dir = self._clamp((mh + macd_accel) / (2.0 * macd_scale), -1.0, 1.0)
        kdj_dir = self._clamp(((kval - dval) / 15.0) + ((jval - jprev) / 30.0), -1.0, 1.0)
        regime_quality = 1.0 if regime["pass_count"] == 3 else 0.65 if regime["pass_count"] == 2 else 0.2
        trend_sign = 1.0 if slope > 0 else -1.0 if slope < 0 else 0.0
        regime_dir = regime_quality * trend_sign
        loc_dir = 1.0 if (near_support or sweep_long or retest_long) else -1.0 if (near_resist or sweep_short or retest_short) else 0.0

        forecast_raw = 100.0 * (0.30*setup_diff + 0.15*ema_dir + 0.15*rsi_dir + 0.15*macd_dir + 0.10*kdj_dir + 0.10*regime_dir + 0.05*loc_dir)
        forecast_raw = self._clamp(forecast_raw, -100.0, 100.0)
        fside = "BULLISH" if forecast_raw >= 20.0 else "BEARISH" if forecast_raw <= -20.0 else "NEUTRAL"
        signs = [setup_diff, ema_dir, rsi_dir, macd_dir, kdj_dir, regime_dir, loc_dir]
        nonzero = [x for x in signs if abs(x) >= 0.10]
        if nonzero:
            dominant = 1 if forecast_raw >= 0 else -1
            coherence = sum(1 for x in nonzero if (x > 0) == (dominant > 0)) / len(nonzero)
        else:
            coherence = 0.0
        confidence = self._clamp(38.0 + abs(forecast_raw) * 0.42 + coherence * 15.0, 5.0, 95.0)
        forecast = {"raw": round(float(forecast_raw), 1), "side": fside, "confidence": round(float(confidence), 1), "coherence": round(float(coherence), 2)}

        target_levels_long = sorted({x for x in ph_levels if x > close})
        target_levels_short = sorted({x for x in pl_levels if x < close}, reverse=True)
        location_txt = "SUPPORT" if near_support else "RESISTANCE" if near_resist else "EMA20_VALUE" if ema_value else "MID"

        return {
            "ready": True, "direction": direction, "selected_setup": selected_setup,
            "selected_score": round(float(selected_score), 2), "score_threshold": threshold,
            "score_long": round(float(long_score), 2), "score_short": round(float(short_score), 2),
            "components": selected_comp, "components_long": long_comp, "components_short": short_comp,
            "setup_long": setup_long or "NONE", "setup_short": setup_short or "NONE",
            "ema20": round(e20, 8), "ema20_slope_atr": round(float(slope), 3),
            "price_vs_ema20": "ABOVE" if price_above else "BELOW",
            "rsi": round(r, 2), "rsi_sma": round(rs, 2), "rsi_cross_up": rsi_cross_up,
            "rsi_cross_down": rsi_cross_dn, "rsi_rotation_long": rsi_rotation_long, "rsi_rotation_short": rsi_rotation_short,
            "macd": round(m, 6), "macd_signal": round(ms, 6), "macd_hist": round(mh, 6), "macd_hist_delta": round(macd_accel, 6),
            "kdj_k": round(kval, 2), "kdj_d": round(dval, 2), "kdj_j": round(jval, 2),
            "regime": regime, "support": round(s1, 8) if s1 is not None else None, "resistance": round(r1, 8) if r1 is not None else None,
            "near_support": near_support, "near_resistance": near_resist, "location": location_txt,
            "sweep_long": sweep_long, "sweep_short": sweep_short, "retest_long": retest_long, "retest_short": retest_short,
            "fresh_event_long": fresh_event_long, "fresh_event_short": fresh_event_short,
            "target_levels_long": target_levels_long, "target_levels_short": target_levels_short,
            "forecast": forecast,
            "reason": f"15M {selected_setup} {direction.upper()} score {selected_score:.2f}/{threshold:.2f}" if direction and selected_setup and threshold is not None else f"15M no qualified setup | L {long_score:.2f}({setup_long or 'NONE'}) / S {short_score:.2f}({setup_short or 'NONE'})",
        }

    def _bias_15m(self, candles: list) -> dict:
        a = self._analysis_15m(candles)
        return {"ready": bool(a.get("direction")), "direction": a.get("direction"), "strength": int(round(float(a.get("selected_score") or 0))), "ema20": a.get("ema20"), "ema20_slope_atr": a.get("ema20_slope_atr"), "rsi": a.get("rsi"), "reason": a.get("reason", "15M V10 analysis")}

    def _snapshot_5m_v10(self, candles: list, direction: str | None, current_price: float) -> dict:
        if len(candles) < self.MIN_5M_BARS:
            return {"ready": False, "market_ready": False, "trigger": None, "blocks": ["5M_WARMUP"]}
        if direction not in {"long", "short"}:
            return {"ready": True, "market_ready": False, "trigger": None, "blocks": ["NO_15M_SETUP"]}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        atr = self.atr(candles, 14)
        regime = self._regime_snapshot(candles)
        k, d, j = self._kdj(candles, 9)
        if not regime.get("ready") or not self._finite(ema20[-1], atr[-1], k[-1], d[-1], j[-1]):
            return {"ready": False, "market_ready": False, "trigger": None, "blocks": ["5M_INDICATORS"]}

        bar = candles[-1]
        close = float(bar.close); open_ = float(bar.open); high = float(bar.high); low = float(bar.low)
        atr5 = max(float(atr[-1]), 1e-12); e20 = float(ema20[-1]); long = direction == "long"
        bullish = close > open_; bearish = close < open_; body_atr = abs(close - open_) / atr5
        rng = max(high - low, 1e-12); close_pos = (close - low) / rng
        slope5 = (e20 - float(ema20[-4])) / atr5; dist_ema = abs(close - e20) / atr5
        vols = [float(c.volume or 0.0) for c in candles[-21:-1]]
        med_vol = float(np.median(vols)) if vols else 0.0
        vol_ratio = float(bar.volume or 0.0) / med_vol if med_vol > 0 else 1.0

        recent3 = candles[-3:]
        if long:
            touched = any(float(c.low) <= e20 + 0.10 * atr5 for c in recent3)
            pullback = touched and bullish and close > float(candles[-2].high) and close >= e20
        else:
            touched = any(float(c.high) >= e20 - 0.10 * atr5 for c in recent3)
            pullback = touched and bearish and close < float(candles[-2].low) and close <= e20
        prev3 = candles[-4:-1]
        p3h = max(float(c.high) for c in prev3); p3l = min(float(c.low) for c in prev3)
        breakout = (bullish and close > p3h and body_atr >= 0.20 and close >= e20) if long else (bearish and close < p3l and body_atr >= 0.20 and close <= e20)
        prev5 = candles[-6:-1]
        p5h = max(float(c.high) for c in prev5); p5l = min(float(c.low) for c in prev5)
        sweep = (low < p5l and close > p5l and bullish) if long else (high > p5h and close < p5h and bearish)
        trigger = "SWEEP_RECLAIM" if sweep else "PULLBACK_RECLAIM" if pullback else "MICRO_BREAKOUT" if breakout else None

        blocks: list[str] = []
        if not regime["market_ready"]: blocks.append("REGIME_2OF3")
        candidate = trigger
        if trigger:
            close_quality = close_pos >= 0.62 if long else close_pos <= 0.38
            if not close_quality: blocks.append("WEAK_CLOSE")
            if body_atr < (0.25 if trigger == "MICRO_BREAKOUT" else 0.15): blocks.append("WEAK_BODY")
            max_dist = 1.25 if trigger == "MICRO_BREAKOUT" else 1.10
            if dist_ema > max_dist: blocks.append("EXTENDED_FROM_EMA20")
            if trigger == "MICRO_BREAKOUT" and vol_ratio < 0.80: blocks.append("BREAKOUT_VOLUME")
            if trigger == "MICRO_BREAKOUT" and ((long and slope5 <= 0) or ((not long) and slope5 >= 0)): blocks.append("5M_SLOPE")

        chase = None
        if trigger and not blocks:
            adverse = max(0.0, float(current_price) - close) if long else max(0.0, close - float(current_price))
            chase = adverse / atr5
            if chase > self.MAX_TRIGGER_CHASE_ATR: blocks.append("ANTI_CHASE")

        structure = raw_sl_atr = raw_risk_pct = stop = tp1 = tp2 = risk = None
        if trigger and not blocks:
            lookback = 3 if trigger == "MICRO_BREAKOUT" else 5
            recent = candles[-lookback:]
            entry = float(current_price)
            if long:
                structure = min(float(c.low) for c in recent); raw_stop = structure - self.SL_BUFFER_ATR * atr5; raw_risk = entry - raw_stop
            else:
                structure = max(float(c.high) for c in recent); raw_stop = structure + self.SL_BUFFER_ATR * atr5; raw_risk = raw_stop - entry
            if raw_risk <= 0:
                blocks.append("SL_STRUCTURE")
            else:
                raw_sl_atr = raw_risk / atr5; raw_risk_pct = raw_risk / max(entry, 1e-12)
                if raw_sl_atr > self.MAX_SL_ATR: blocks.append("SL_TOO_WIDE")
                if raw_risk_pct < self.MIN_ECONOMIC_RISK_PCT: blocks.append("FEE_EDGE_TOO_TIGHT")
            if not blocks:
                risk = max(raw_risk, self.MIN_SL_ATR * atr5)
                stop = entry - risk if long else entry + risk
                tp1 = entry + self.TP1_R * risk if long else entry - self.TP1_R * risk
                tp2 = entry + self.DYNAMIC_TP_FALLBACK_R * risk if long else entry - self.DYNAMIC_TP_FALLBACK_R * risk

        out = {
            "ready": True, "direction": direction, "market_ready": bool(regime["market_ready"]), "regime_pass": regime["pass_count"],
            "adx": regime["adx"], "chop": regime["chop"], "atr_ratio": regime["atr_ratio"], "candidate": candidate,
            "trigger": trigger if trigger and not blocks else None, "trigger_candidate": candidate,
            "pullback": bool(pullback), "breakout": bool(breakout), "sweep": bool(sweep),
            "body_atr": round(float(body_atr), 2), "close_pos": round(float(close_pos), 2), "dist_ema_atr": round(float(dist_ema), 2),
            "ema20_slope_atr": round(float(slope5), 3), "volume_ratio": round(float(vol_ratio), 2),
            "kdj_k": round(float(k[-1]), 2), "kdj_d": round(float(d[-1]), 2), "kdj_j": round(float(j[-1]), 2),
            "chase_atr": round(float(chase), 3) if chase is not None else None,
            "structure": round(float(structure), 8) if structure is not None else None,
            "raw_sl_atr": round(float(raw_sl_atr), 2) if raw_sl_atr is not None else None,
            "raw_risk_pct": round(float(raw_risk_pct * 100.0), 3) if raw_risk_pct is not None else None,
            "blocks": list(dict.fromkeys(blocks)),
        }
        if risk is not None and stop is not None:
            out.update({"entry": float(current_price), "risk": float(risk), "stop_loss": float(stop), "tp1_price": float(tp1), "take_profit": float(tp2), "sl_atr": round(float(risk / atr5), 2)})
        return out

    @staticmethod
    def _apply_setup_execution_map(setup: dict, family: str | None) -> dict:
        if not family or not setup.get("trigger"):
            return setup
        allowed = {
            "PULLBACK": {"PULLBACK_RECLAIM", "MICRO_BREAKOUT"},
            "BREAKOUT_RETEST": {"PULLBACK_RECLAIM", "MICRO_BREAKOUT"},
            "SWEEP_REVERSAL": {"SWEEP_RECLAIM", "MICRO_BREAKOUT"},
        }.get(family, set())
        if setup.get("trigger") in allowed:
            return setup
        out = dict(setup); out["trigger_candidate"] = setup.get("trigger"); out["trigger"] = None
        out["blocks"] = list(dict.fromkeys(list(out.get("blocks", [])) + ["SETUP_EXEC_MISMATCH"]))
        return out

    def _dynamic_target_r(self, analysis: dict, direction: str, entry: float, risk: float) -> tuple[float, str]:
        if risk <= 0: return self.DYNAMIC_TP_FALLBACK_R, "FALLBACK_2R"
        levels = analysis.get("target_levels_long", []) if direction == "long" else analysis.get("target_levels_short", [])
        rr_candidates = []
        for level in levels:
            try: rr = (float(level)-entry)/risk if direction == "long" else (entry-float(level))/risk
            except (TypeError, ValueError): continue
            if self.DYNAMIC_TP_MIN_R <= rr <= self.DYNAMIC_TP_MAX_R: rr_candidates.append(rr)
        forecast = analysis.get("forecast") or {}; conf = float(forecast.get("confidence") or 0.0)
        aligned = (direction == "long" and forecast.get("side") == "BULLISH") or (direction == "short" and forecast.get("side") == "BEARISH")
        if rr_candidates:
            rr_candidates.sort(); rr = rr_candidates[-1] if aligned and conf >= 70.0 else rr_candidates[0]
            return self._clamp(rr, self.DYNAMIC_TP_MIN_R, self.DYNAMIC_TP_MAX_R), "S_R_FORECAST"
        return self.DYNAMIC_TP_FALLBACK_R, "FALLBACK_2R"

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c15 = self._closed_candle_series(candles, self.FIFTEEN_MIN_MS)
        c5 = self._closed_candle_series(mtf.get("5m", []), self.FIVE_MIN_MS)
        self._latest_15m = c15; self._latest_5m = c5
        meta = {"strategy": "SENTINEL_V10", "version": self.VERSION, "architecture": "15M_MOMENTUM_LOCATION_FORECAST__5M_PRICE_ACTION_EXECUTION", "entry_tf": "5m_closed", "risk_plan": "STRUCTURE+0.20ATR_MIN0.90_MAX1.80_FEE0.40PCT__TP1_1R50_LOCK0.15R__TP2_SR_1.5_2.5R"}
        if len(c15) < max(self.MIN_15M_BARS, 60) or len(c5) < self.MIN_5M_BARS:
            return self._hold(float(current_price), "waiting for closed 15M/5M warmup", meta)

        analysis = self._analysis_15m(c15); self._latest_analysis = analysis; self._latest_forecast = analysis.get("forecast") or {}
        direction = analysis.get("direction"); family = analysis.get("selected_setup")
        setup = self._apply_setup_execution_map(self._snapshot_5m_v10(c5, direction, float(current_price)), family)
        meta["analysis_15m"] = analysis; meta["forecast"] = self._latest_forecast; meta["setup_5m"] = setup
        meta["bias_15m"] = {"direction": direction, "strength": analysis.get("selected_score"), "ema20": analysis.get("ema20"), "ema20_slope_atr": analysis.get("ema20_slope_atr"), "rsi": analysis.get("rsi")}

        if self._open_position is not None: return self._hold(float(current_price), f"managing open {self._open_position} position", meta)
        bar5_ts = int(self._bar_ts(c5[-1]))
        if self._last_5m_evaluated_ts == bar5_ts: return self._hold(float(current_price), "5M bar already evaluated", meta)
        self._last_5m_evaluated_ts = bar5_ts
        if self._last_exit_5m_ts is not None and bar5_ts - self._last_exit_5m_ts < self.EXIT_COOLDOWN_5M_BARS * self.FIVE_MIN_MS:
            return self._hold(float(current_price), "post-exit 5M cooldown", meta)
        if direction not in {"long", "short"}: return self._hold(float(current_price), analysis.get("reason", "waiting for 15M setup"), meta)
        if self._last_close_was_hard_sl:
            fresh = analysis.get("fresh_event_long") if direction == "long" else analysis.get("fresh_event_short")
            if not fresh: return self._hold(float(current_price), "hard-SL rearm waiting fresh RSI/SR event", meta)
        current_15m_ts = int(self._bar_ts(c15[-1]))
        if self._last_entry_15m_ts == current_15m_ts: return self._hold(float(current_price), "one successful Sentinel entry already used this 15M bar", meta)
        if not setup.get("market_ready"): return self._hold(float(current_price), "5M regime blocked", meta)
        if not setup.get("trigger"): return self._hold(float(current_price), "waiting compatible 5M price-action trigger", meta)

        direction = str(direction); entry = float(setup["entry"]); stop = float(setup["stop_loss"]); risk = float(setup["risk"])
        rr, tp_source = self._dynamic_target_r(analysis, direction, entry, risk)
        target = entry + rr*risk if direction == "long" else entry - rr*risk
        tp1 = entry + self.TP1_R*risk if direction == "long" else entry - self.TP1_R*risk
        self._open_position = direction; self._pending_entry = True; self._entry_price = entry; self._entry_sl = stop; self._entry_tp = target; self._initial_risk = risk; self._tp1_done = False; self._tp2_rr_active = rr
        meta.update({"direction": direction, "entry_trigger": setup["trigger"], "stop_loss": round(stop,8), "take_profit": round(target,8), "tp1_price": round(tp1,8), "rr_ratio": round(rr,3), "tp2_r_dynamic": round(rr,3), "tp2_source": tp_source, "tp1_r": self.TP1_R, "tp1_close_pct": self.TP1_CLOSE_PCT, "tp1_lock_r": self.TP1_LOCK_R, "setup_score": analysis.get("selected_score"), "setup_family": family, "forecast_side": self._latest_forecast.get("side"), "forecast_confidence": self._latest_forecast.get("confidence")})
        confidence = min(0.94, 0.65 + 0.025*float(analysis.get("selected_score") or 0.0) + 0.0008*float(self._latest_forecast.get("confidence") or 0.0))
        sig_type = SignalType.BUY if direction == "long" else SignalType.SELL
        reason = f"{direction.upper()} {family} score={analysis.get('selected_score')}/{analysis.get('score_threshold')} | 5M={setup['trigger']} | FC={self._latest_forecast.get('side')} {self._latest_forecast.get('confidence')}% | TP2={rr:.2f}R"
        return Signal(sig_type, self.symbol, entry, 0.0, reason, confidence, meta)

    def tick_open_position(self, current_price: float, position_key: str | None = None):
        if self._open_position is None: return None
        candles = self._latest_15m
        if len(candles) >= max(self.MIN_15M_BARS, 60):
            bar_ts = int(self._bar_ts(candles[-1]))
            if bar_ts != self._last_exit_check_ts:
                self._last_exit_check_ts = bar_ts
                a = self._analysis_15m(candles)
                rsi_opp = bool(a.get("rsi_cross_down")) if self._open_position == "long" else bool(a.get("rsi_cross_up"))
                price_opp = a.get("price_vs_ema20") == "BELOW" if self._open_position == "long" else a.get("price_vs_ema20") == "ABOVE"
                mh = float(a.get("macd_hist") or 0.0); dh = float(a.get("macd_hist_delta") or 0.0)
                macd_opp = (mh < 0 and dh <= 0) if self._open_position == "long" else (mh > 0 and dh >= 0)
                exit_votes = int(rsi_opp) + int(price_opp) + int(macd_opp)
                k = float(a.get("kdj_k") or 50.0); d = float(a.get("kdj_d") or 50.0); j = float(a.get("kdj_j") or 50.0)
                kdj_opp = (k < d and j < 35.0) if self._open_position == "long" else (k > d and j > 65.0)
                forecast = a.get("forecast") or {}
                fc_flip = (self._open_position == "long" and forecast.get("side") == "BEARISH" and float(forecast.get("confidence") or 0.0) >= 65.0) or (self._open_position == "short" and forecast.get("side") == "BULLISH" and float(forecast.get("confidence") or 0.0) >= 65.0)
                if not self._tp1_done and exit_votes >= 2:
                    side = self._open_position
                    if self._latest_5m: self._last_exit_5m_ts = int(self._bar_ts(self._latest_5m[-1]))
                    self._reset_position(keep_exit_ts=True)
                    return PositionUpdate(action="close", close_pct=1.0, reason=f"V10_EXIT_2OF3: EMA/RSI/MACD deterioration {exit_votes}/3 — close {side.upper()}")
                if self._tp1_done and ((rsi_opp and kdj_opp) or exit_votes >= 2 or (fc_flip and rsi_opp)):
                    side = self._open_position
                    if self._latest_5m: self._last_exit_5m_ts = int(self._bar_ts(self._latest_5m[-1]))
                    self._reset_position(keep_exit_ts=True)
                    return PositionUpdate(action="close", close_pct=1.0, reason=f"V10_RUNNER_EXIT: momentum/forecast reversal — close {side.upper()}")
        if not self._tp1_done and self._entry_price is not None and self._initial_risk is not None and self._initial_risk > 0:
            profit = float(current_price)-self._entry_price if self._open_position == "long" else self._entry_price-float(current_price)
            current_r = profit/self._initial_risk
            if current_r >= self.TP1_R:
                self._tp1_done = True
                new_sl = self._entry_price + self.TP1_LOCK_R*self._initial_risk if self._open_position == "long" else self._entry_price - self.TP1_LOCK_R*self._initial_risk
                return PositionUpdate(action="partial_tp", close_pct=self.TP1_CLOSE_PCT, new_sl=round(float(new_sl),8), reason=f"TP1 {current_r:.2f}R — close 50%, runner SL +{self.TP1_LOCK_R:.2f}R")
        return PositionUpdate(action="hold", reason=f"V10 hold | TP1=1R/50% lock+0.15R | TP2={self._tp2_rr_active:.2f}R S/R forecast")
