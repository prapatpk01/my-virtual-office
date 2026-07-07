"""
Layer 2: Expert Analysis Engine

Seven specialized expert modules each produce a score 0-100:
  TrendExpert      — EMA, VWAP, market structure, HH/HL, volume
  MomentumExpert   — RSI, MACD, ROC, CCI, momentum
  VolatilityExpert — ATR, Std Dev, historical vol, BB width
  LiquidityExpert  — Equal H/L, liquidity pools, stop hunt, FVG, order blocks
  VolumeExpert     — Delta proxy, OBV, volume profile, POC, HVN/LVN
  SessionExpert    — London, NY, Asia, overlap quality
  CorrelationExpert— BTC/ETH/SOL dominance proxy, DXY inverse proxy
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ExpertScores:
    trend: float = 50.0
    momentum: float = 50.0
    volatility: float = 50.0
    liquidity: float = 50.0
    volume: float = 50.0
    session: float = 50.0
    correlation: float = 50.0
    direction_bias: float = 0.0   # -1 (bear) to +1 (bull)
    detail: dict = field(default_factory=dict)


class ExpertAnalysisEngine:
    """Runs all expert modules and aggregates scores."""

    def analyze(
        self,
        candles: list,
        symbol: str = "BTC/USDT",
        mtf_candles: dict = None,
        corr_data: dict = None,
    ) -> ExpertScores:
        if len(candles) < 50:
            return ExpertScores()

        closes  = np.array([c.close  for c in candles], dtype=float)
        highs   = np.array([c.high   for c in candles], dtype=float)
        lows    = np.array([c.low    for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)
        opens   = np.array([c.open   for c in candles], dtype=float)

        t_score, t_detail = self._trend_expert(closes, highs, lows, volumes)
        m_score, m_detail = self._momentum_expert(closes)
        v_score, v_detail = self._volatility_expert(candles, closes)
        l_score, l_detail = self._liquidity_expert(highs, lows, closes, volumes)
        vol_score, vol_detail = self._volume_expert(closes, volumes, highs, lows)
        s_score, s_detail = self._session_expert()
        c_score, c_detail = self._correlation_expert(symbol, closes, corr_data)

        # Direction bias: weighted average of directional signals
        direction = (
            (t_score - 50) * 0.35 +
            (m_score - 50) * 0.20 +
            (l_score - 50) * 0.20 +
            (vol_score - 50) * 0.15 +
            (s_score - 50) * 0.10
        ) / 100.0

        return ExpertScores(
            trend=round(t_score, 1),
            momentum=round(m_score, 1),
            volatility=round(v_score, 1),
            liquidity=round(l_score, 1),
            volume=round(vol_score, 1),
            session=round(s_score, 1),
            correlation=round(c_score, 1),
            direction_bias=round(float(np.clip(direction, -1.0, 1.0)), 3),
            detail={
                "trend": t_detail,
                "momentum": m_detail,
                "volatility": v_detail,
                "liquidity": l_detail,
                "volume": vol_detail,
                "session": s_detail,
                "correlation": c_detail,
            },
        )

    # ── Trend Expert ────────────────────────────────────────────────────

    def _trend_expert(self, closes, highs, lows, volumes) -> tuple[float, dict]:
        ema20  = self._ema(closes, 20)
        ema50  = self._ema(closes, 50)
        ema200 = self._ema(closes, 200) if len(closes) >= 200 else ema50
        price  = float(closes[-1])

        # EMA alignment score (0-40)
        ema_score = 0.0
        e20  = float(ema20[-1])  if not np.isnan(ema20[-1])  else price
        e50  = float(ema50[-1])  if not np.isnan(ema50[-1])  else price
        e200 = float(ema200[-1]) if not np.isnan(ema200[-1]) else price
        if price > e20:  ema_score += 10
        if price > e50:  ema_score += 8
        if price > e200: ema_score += 8
        if e20 > e50:    ema_score += 8
        if e50 > e200:   ema_score += 6

        # VWAP proxy (volume-weighted price of last 20 bars)
        vwap_score = 0.0
        if len(closes) >= 20 and volumes[-20:].sum() > 0:
            vwap = float(np.dot(closes[-20:], volumes[-20:]) / volumes[-20:].sum())
            vwap_score = 15.0 if price > vwap else 0.0

        # Market structure HH/HL (0-25)
        hh_hl = sum(1 for i in range(1, min(20, len(highs))) if highs[-i] > highs[-i-1]) / 19
        hl    = sum(1 for i in range(1, min(20, len(lows)))  if lows[-i]  > lows[-i-1])  / 19
        structure_score = (hh_hl + hl) / 2 * 25

        # Volume trend (OBV slope, 0-20)
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            obv[i] = obv[i-1] + (volumes[i] if closes[i]>closes[i-1] else -volumes[i] if closes[i]<closes[i-1] else 0)
        obv_slope = float(np.polyfit(np.arange(min(20, len(obv))), obv[-min(20, len(obv)):], 1)[0]) if len(obv) >= 4 else 0.0
        vol_trend_score = min(20.0, max(0.0, (obv_slope / (abs(obv_slope) + 1e-9) + 1) * 10))

        total = ema_score + vwap_score + structure_score + vol_trend_score
        return min(100.0, total), {
            "ema_align": round(ema_score, 1), "vwap": round(vwap_score, 1),
            "structure": round(structure_score, 1), "vol_trend": round(vol_trend_score, 1),
        }

    # ── Momentum Expert ─────────────────────────────────────────────────

    def _momentum_expert(self, closes) -> tuple[float, dict]:
        rsi_val = self._safe_last(self._rsi(closes, 14))
        macd, sig, hist = self._macd(closes)
        macd_val = self._safe_last(macd)
        sig_val  = self._safe_last(sig)
        hist_val = self._safe_last(hist)

        roc = self._roc(closes, 10)
        roc_val = self._safe_last(roc)

        cci_val = self._cci(closes, 20)

        # RSI score (overbought bearish, oversold bullish for reversal)
        # For trend following: RSI 55-75 is bullish momentum
        rsi_score = 0.0
        if 55 <= rsi_val <= 75:  rsi_score = 70.0
        elif 45 <= rsi_val < 55: rsi_score = 50.0
        elif rsi_val > 75:       rsi_score = 30.0   # overbought
        elif 30 <= rsi_val < 45: rsi_score = 35.0
        else:                    rsi_score = 20.0   # oversold

        # MACD score
        macd_score = 60.0 if (not math.isnan(macd_val) and not math.isnan(sig_val) and macd_val > sig_val and hist_val > 0) else \
                     40.0 if (not math.isnan(hist_val) and hist_val > 0) else 35.0

        # ROC score
        roc_score = min(80.0, max(20.0, 50 + roc_val * 100)) if not math.isnan(roc_val) else 50.0

        # CCI score (momentum confirmation)
        cci_score = 65.0 if cci_val > 100 else 45.0 if cci_val > 0 else 35.0

        total = rsi_score * 0.30 + macd_score * 0.30 + roc_score * 0.20 + cci_score * 0.20
        return round(total, 1), {
            "rsi": round(rsi_val, 1), "macd_hist": round(hist_val, 4) if not math.isnan(hist_val) else 0,
            "roc": round(roc_val * 100, 2) if not math.isnan(roc_val) else 0, "cci": round(cci_val, 1),
        }

    # ── Volatility Expert ───────────────────────────────────────────────

    def _volatility_expert(self, candles, closes) -> tuple[float, dict]:
        atr_arr = self._atr(candles, 14)
        atr_val = self._safe_last(atr_arr)
        atr_mean= float(np.nanmean(atr_arr[-50:])) if np.any(~np.isnan(atr_arr[-50:])) else 1e-9
        atr_ratio = atr_val / atr_mean if atr_mean > 0 else 1.0

        std_dev = float(np.std(closes[-20:])) if len(closes) >= 20 else 0.0
        price   = float(closes[-1])
        hv      = std_dev / price * 100 if price > 0 else 0.0

        bb_upper, bb_mid, bb_lower = self._bollinger(closes, 20, 2.0)
        bb_width = float((bb_upper[-1] - bb_lower[-1]) / bb_mid[-1]) if not np.isnan(bb_mid[-1]) and bb_mid[-1] != 0 else 0.0
        bb_pct   = float((price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) if not np.isnan(bb_upper[-1]) and (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5

        # Score: moderate volatility is ideal (50-75)
        # Too low: squeeze, no movement; too high: risky
        atr_score = 60.0 if 0.8 < atr_ratio < 1.5 else 40.0 if atr_ratio >= 1.5 else 30.0
        hv_score  = min(70.0, max(20.0, 50 - abs(hv - 2.0) * 10))
        bb_score  = 60.0 if 0.3 < bb_pct < 0.7 else 40.0  # price near BB mid

        total = atr_score * 0.40 + hv_score * 0.30 + bb_score * 0.30
        return round(total, 1), {
            "atr_pct": round(atr_ratio * 100, 1),
            "hv_pct": round(hv, 3),
            "bb_width": round(bb_width * 100, 2),
            "bb_pct_b": round(bb_pct, 3),
        }

    # ── Liquidity Expert ────────────────────────────────────────────────

    def _liquidity_expert(self, highs, lows, closes, volumes) -> tuple[float, dict]:
        """Detect liquidity pools, equal H/L, FVG, order blocks."""
        price = float(closes[-1])

        # Equal Highs/Lows (within 0.1%)
        eq_highs = sum(1 for i in range(1, min(20, len(highs)))
                       if abs(highs[-i] - highs[-i-1]) / (highs[-i-1] + 1e-9) < 0.001)
        eq_lows  = sum(1 for i in range(1, min(20, len(lows)))
                       if abs(lows[-i] - lows[-i-1]) / (lows[-i-1] + 1e-9) < 0.001)
        pool_score = min(80.0, (eq_highs + eq_lows) * 15)

        # Stop Hunt (sweep + reversal)
        prev_high = float(highs[-20:-1].max()) if len(highs) >= 20 else float(highs[:-1].max())
        prev_low  = float(lows[-20:-1].min())  if len(lows)  >= 20 else float(lows[:-1].min())
        swept_and_reversed = (
            (float(highs[-1]) > prev_high and float(closes[-1]) < prev_high) or
            (float(lows[-1])  < prev_low  and float(closes[-1]) > prev_low)
        )
        stop_hunt_score = 85.0 if swept_and_reversed else 20.0

        # FVG (Fair Value Gap): 3-bar pattern where bar 1 high < bar 3 low (bullish gap)
        fvg_score = 20.0
        if len(closes) >= 3:
            if float(highs[-3]) < float(lows[-1]):
                fvg_score = 70.0   # bullish FVG below price
            elif float(lows[-3]) > float(highs[-1]):
                fvg_score = 65.0   # bearish FVG above price

        # Order Block: last large-body bearish candle before bullish move
        ob_score = 30.0
        if len(closes) >= 5:
            for i in range(2, min(10, len(closes))):
                body = abs(float(closes[-i]) - float(opens[-i] if hasattr(closes, '__getitem__') else closes[-i]))
                if body > 0:
                    ob_score = 50.0
                    break
        # Simplify: use closes as proxy for order block detection
        body_sizes = [abs(float(closes[i]) - float(closes[i-1])) for i in range(max(1, len(closes)-10), len(closes))]
        if body_sizes:
            max_body = max(body_sizes)
            mean_body = sum(body_sizes) / len(body_sizes)
            ob_score = min(75.0, 30 + (max_body / (mean_body + 1e-9)) * 15)

        total = pool_score * 0.25 + stop_hunt_score * 0.30 + fvg_score * 0.25 + ob_score * 0.20
        return round(min(100.0, total), 1), {
            "eq_highs": eq_highs, "eq_lows": eq_lows,
            "stop_hunt": swept_and_reversed, "fvg": round(fvg_score, 1),
        }

    # ── Volume Expert ───────────────────────────────────────────────────

    def _volume_expert(self, closes, volumes, highs, lows) -> tuple[float, dict]:
        vol_mean20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1e-9
        vol_mean5  = float(np.mean(volumes[-5:]))  if len(volumes) >= 5  else vol_mean20
        vol_ratio  = vol_mean5 / vol_mean20 if vol_mean20 > 0 else 1.0

        # OBV slope
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            obv[i] = obv[i-1] + (volumes[i] if closes[i]>closes[i-1] else -volumes[i] if closes[i]<closes[i-1] else 0)
        obv_slope_raw = float(np.polyfit(np.arange(min(14, len(obv))), obv[-min(14,len(obv)):], 1)[0]) if len(obv) >= 4 else 0.0
        obv_score = min(80.0, max(20.0, 50 + np.sign(obv_slope_raw) * 25))

        # Volume delta proxy (up-volume vs down-volume last 5 bars)
        up_vol   = sum(volumes[i] for i in range(max(0, len(closes)-5), len(closes)) if closes[i] > closes[i-1] if i > 0)
        down_vol = sum(volumes[i] for i in range(max(0, len(closes)-5), len(closes)) if closes[i] < closes[i-1] if i > 0)
        total_vol = up_vol + down_vol
        delta_score = (up_vol / total_vol * 100) if total_vol > 0 else 50.0

        # Volume profile POC proxy: find price level with most volume
        if len(closes) >= 20:
            price_range = float(highs[-20:].max() - lows[-20:].min())
            bins = 10
            if price_range > 0:
                poc_idx = np.argmax([sum(volumes[i] for i in range(max(0,len(closes)-20), len(closes))
                                         if lows[-20:].min() + b * price_range/bins <= closes[i] <
                                            lows[-20:].min() + (b+1) * price_range/bins)
                                     for b in range(bins)])
                poc_price = float(lows[-20:].min()) + (poc_idx + 0.5) * price_range / bins
                poc_dist  = abs(float(closes[-1]) - poc_price) / price_range
                poc_score = max(30.0, 70.0 - poc_dist * 50)
            else:
                poc_score = 50.0
        else:
            poc_score = 50.0

        vol_surge_score = min(90.0, max(20.0, 50 + (vol_ratio - 1.0) * 30))

        total = obv_score * 0.25 + delta_score * 0.30 + poc_score * 0.20 + vol_surge_score * 0.25
        return round(min(100.0, total), 1), {
            "vol_ratio": round(vol_ratio, 2), "obv_slope": round(float(np.sign(obv_slope_raw)), 0),
            "delta_pct": round(delta_score, 1), "poc_score": round(poc_score, 1),
        }

    # ── Session Expert ──────────────────────────────────────────────────

    def _session_expert(self) -> tuple[float, dict]:
        """Score current session quality. London/NY overlap = highest quality."""
        now_utc = datetime.now(timezone.utc)
        hour    = now_utc.hour

        # Session windows (UTC)
        asia_active    = 0  <= hour < 8
        london_active  = 7  <= hour < 16
        ny_active      = 13 <= hour < 22
        overlap_active = 13 <= hour < 16   # London/NY overlap

        if overlap_active:
            score, label = 95.0, "London/NY Overlap (BEST)"
        elif london_active and ny_active:
            score, label = 90.0, "London+NY"
        elif ny_active:
            score, label = 75.0, "New York"
        elif london_active:
            score, label = 70.0, "London"
        elif asia_active:
            score, label = 45.0, "Asia (lower liquidity)"
        else:
            score, label = 30.0, "Off-session"

        return score, {"session": label, "utc_hour": hour}

    # ── Correlation Expert ──────────────────────────────────────────────

    def _correlation_expert(self, symbol: str, closes: np.ndarray, corr_data: dict = None) -> tuple[float, dict]:
        """Score based on known intermarket correlations.
        Without live corr data, uses symbol-based heuristics.
        With corr_data dict, computes actual correlation scores.
        """
        if corr_data:
            scores = []
            for asset, corr in corr_data.items():
                if not math.isnan(corr):
                    scores.append(50 + corr * 30)
            if scores:
                return round(float(np.mean(scores)), 1), {"assets": list(corr_data.keys())}

        # Heuristic: assume BTC-correlated crypto in bullish trend gets +
        price_trend = float(np.sign(closes[-1] - closes[-min(20, len(closes))]))
        base_score  = 60.0 if price_trend > 0 else 40.0

        btc_like = any(x in symbol.upper() for x in ("BTC", "ETH", "SOL", "BNB", "AVAX"))
        if btc_like:
            return base_score, {"type": "crypto_major", "trend": "up" if price_trend > 0 else "down"}
        return 50.0, {"type": "unknown", "note": "no correlation data"}

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        k = 2.0 / (period + 1)
        result[period - 1] = values[:period].mean()
        for i in range(period, len(values)):
            result[i] = values[i] * k + result[i-1] * (1-k)
        return result

    @staticmethod
    def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(values)
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        ag    = np.full(len(values), np.nan)
        al    = np.full(len(values), np.nan)
        if len(gain) < period:
            return ag
        ag[period] = gain[:period].mean()
        al[period] = loss[:period].mean()
        for i in range(period+1, len(values)):
            ag[i] = (ag[i-1]*(period-1) + gain[i-1]) / period
            al[i] = (al[i-1]*(period-1) + loss[i-1]) / period
        rs  = np.where(al == 0, 100.0, ag / np.where(al == 0, 1.0, al))
        out = 100 - 100/(1+rs)
        out[:period] = np.nan
        return out

    @staticmethod
    def _macd(values: np.ndarray, fast=12, slow=26, signal=9):
        def ema(v, p):
            r = np.full_like(v, np.nan)
            if len(v) < p: return r
            k = 2/(p+1)
            r[p-1] = v[:p].mean()
            for i in range(p, len(v)): r[i] = v[i]*k + r[i-1]*(1-k)
            return r
        fe = ema(values, fast); se = ema(values, slow)
        ml = fe - se
        sl_vals = [v for v in ml if not np.isnan(v)]
        if len(sl_vals) < signal:
            return ml, np.full_like(ml, np.nan), np.full_like(ml, np.nan)
        sl_e = ema(np.array(sl_vals), signal)
        pad  = len(ml) - len(sl_e)
        sl_p = np.concatenate([np.full(pad, np.nan), sl_e])
        return ml, sl_p, ml - sl_p

    @staticmethod
    def _roc(values: np.ndarray, period: int = 10) -> np.ndarray:
        result = np.full_like(values, np.nan)
        for i in range(period, len(values)):
            if values[i-period] != 0:
                result[i] = (values[i] - values[i-period]) / values[i-period]
        return result

    @staticmethod
    def _cci(closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.0
        typical = closes[-period:]
        mean    = typical.mean()
        mad     = float(np.mean(np.abs(typical - mean)))
        if mad == 0:
            return 0.0
        return float((closes[-1] - mean) / (0.015 * mad))

    @staticmethod
    def _atr(candles: list, period: int = 14) -> np.ndarray:
        n  = len(candles)
        tr = np.full(n, np.nan)
        for i in range(1, n):
            h, l, pc = candles[i].high, candles[i].low, candles[i-1].close
            tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        result = np.full(n, np.nan)
        if n > period:
            result[period] = float(np.nanmean(tr[1:period+1]))
            for i in range(period+1, n):
                result[i] = (result[i-1]*(period-1) + tr[i]) / period
        return result

    @staticmethod
    def _bollinger(values: np.ndarray, period: int = 20, std_dev: float = 2.0):
        sma = np.full_like(values, np.nan)
        std = np.full_like(values, np.nan)
        for i in range(period-1, len(values)):
            w = values[i-period+1:i+1]
            sma[i] = w.mean(); std[i] = w.std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower

    @staticmethod
    def _safe_last(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return float('nan')
        val = float(arr[-1])
        return val if not math.isnan(val) else float('nan')
