"""
Layer 8: Exit AI Engine

Dynamic exit scoring based on:
  - Momentum exhaustion (RSI divergence, MACD cross)
  - Trend change (EMA cross, structure break)
  - Opposing liquidity signal
  - Volume decrease on continuation
  - Price divergence

Output: ExitSignal with score 0-100.
Score >= 70 triggers exit review; >= 85 forces close.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .market_intelligence import MarketRegime


@dataclass
class ExitSignal:
    score:       float = 0.0    # 0-100 (higher = stronger exit reason)
    should_exit: bool  = False
    forced_exit: bool  = False  # True if score >= 85
    reasons:     list  = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class ExitEngine:
    """Scores how strongly the position should be exited."""

    def __init__(
        self,
        soft_threshold:  float = 70.0,
        hard_threshold:  float = 85.0,
    ):
        self.soft_thr = soft_threshold
        self.hard_thr = hard_threshold

    def evaluate(
        self,
        candles: list,
        direction: str,          # "long" | "short"
        entry_price: float,
        current_price: float,
        regime: Optional[MarketRegime] = None,
    ) -> ExitSignal:
        if len(candles) < 30:
            return ExitSignal(score=0.0)

        closes  = np.array([c.close  for c in candles], dtype=float)
        highs   = np.array([c.high   for c in candles], dtype=float)
        lows    = np.array([c.low    for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        reasons = []
        score   = 0.0

        # ── 1. Momentum exhaustion (RSI) ───────────────────────────────────
        rsi_arr = self._rsi(closes, 14)
        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0
        if direction == "long" and rsi_val >= 75:
            s = min(25.0, (rsi_val - 75) * 2)
            score += s; reasons.append(f"RSI overbought {rsi_val:.0f} (+{s:.0f}pt)")
        elif direction == "short" and rsi_val <= 25:
            s = min(25.0, (25 - rsi_val) * 2)
            score += s; reasons.append(f"RSI oversold {rsi_val:.0f} (+{s:.0f}pt)")

        # ── 2. RSI divergence ────────────────────────────────────────────────
        div_score = self._divergence_score(closes, rsi_arr, direction, 14)
        if div_score > 0:
            score += div_score
            reasons.append(f"RSI divergence (+{div_score:.0f}pt)")

        # ── 3. MACD cross ───────────────────────────────────────────────────
        macd_cross = self._macd_cross_score(closes, direction)
        if macd_cross > 0:
            score += macd_cross
            reasons.append(f"MACD cross against position (+{macd_cross:.0f}pt)")

        # ── 4. Trend structure break ────────────────────────────────────────
        structure_break = self._structure_break_score(closes, highs, lows, direction)
        if structure_break > 0:
            score += structure_break
            reasons.append(f"Structure break (+{structure_break:.0f}pt)")

        # ── 5. EMA cross (fast < slow for long) ─────────────────────────────
        ema_cross = self._ema_cross_score(closes, direction)
        if ema_cross > 0:
            score += ema_cross
            reasons.append(f"EMA cross bearish (+{ema_cross:.0f}pt)")

        # ── 6. Volume decay ──────────────────────────────────────────────────
        vol_decay = self._volume_decay_score(volumes, closes, direction)
        if vol_decay > 0:
            score += vol_decay
            reasons.append(f"Volume decaying on move (+{vol_decay:.0f}pt)")

        # ── 7. Opposing liquidity signal ────────────────────────────────────
        liq_opp = self._opposing_liquidity_score(highs, lows, closes, direction)
        if liq_opp > 0:
            score += liq_opp
            reasons.append(f"Opposing liquidity grab (+{liq_opp:.0f}pt)")

        score = min(100.0, round(score, 1))
        return ExitSignal(
            score=score,
            should_exit=score >= self.soft_thr,
            forced_exit=score >= self.hard_thr,
            reasons=reasons,
        )

    # ── Internal scorers ──────────────────────────────────────────────────────

    @staticmethod
    def _rsi(values: np.ndarray, period: int) -> np.ndarray:
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
    def _divergence_score(closes, rsi_arr, direction, lookback=14) -> float:
        if len(closes) < lookback * 2 or np.any(np.isnan(rsi_arr[-lookback:])):
            return 0.0
        c_slice = closes[-lookback:]
        r_slice = rsi_arr[-lookback:]
        price_up = c_slice[-1] > c_slice[0]
        rsi_up   = r_slice[-1] > r_slice[0]
        # Bearish divergence when long: price up, RSI down
        if direction == "long" and price_up and not rsi_up:
            return 20.0
        # Bullish divergence when short: price down, RSI up
        if direction == "short" and not price_up and rsi_up:
            return 20.0
        return 0.0

    @staticmethod
    def _macd_cross_score(closes: np.ndarray, direction: str) -> float:
        def ema(v, p):
            if len(v) < p: return np.full(len(v), np.nan)
            r = np.full(len(v), np.nan)
            k = 2/(p+1)
            r[p-1] = v[:p].mean()
            for i in range(p, len(v)): r[i] = v[i]*k + r[i-1]*(1-k)
            return r
        fe = ema(closes, 12); se = ema(closes, 26)
        ml = fe - se
        valid = [v for v in ml if not np.isnan(v)]
        if len(valid) < 10:
            return 0.0
        sl_vals = ema(np.array(valid), 9)
        hist = np.array(valid) - sl_vals
        if len(hist) < 2:
            return 0.0
        # Bearish cross: histogram went from + to -
        if direction == "long" and hist[-1] < 0 < hist[-2]:
            return 20.0
        if direction == "short" and hist[-1] > 0 > hist[-2]:
            return 20.0
        # Weakening histogram
        if direction == "long" and hist[-1] < hist[-2] < hist[-3] if len(hist) >= 3 else False:
            return 10.0
        return 0.0

    @staticmethod
    def _structure_break_score(closes, highs, lows, direction, lookback=10) -> float:
        if len(closes) < lookback + 2:
            return 0.0
        if direction == "long":
            # Break of structure: price broke below recent swing low
            recent_low = float(lows[-lookback:-1].min())
            if float(closes[-1]) < recent_low:
                return 25.0
        else:
            recent_high = float(highs[-lookback:-1].max())
            if float(closes[-1]) > recent_high:
                return 25.0
        return 0.0

    @staticmethod
    def _ema_cross_score(closes: np.ndarray, direction: str) -> float:
        if len(closes) < 52:
            return 0.0
        def ema(v, p):
            r = np.full(len(v), np.nan)
            k = 2/(p+1)
            r[p-1] = v[:p].mean()
            for i in range(p, len(v)): r[i] = v[i]*k + r[i-1]*(1-k)
            return r
        e20 = ema(closes, 20); e50 = ema(closes, 50)
        if np.isnan(e20[-1]) or np.isnan(e50[-1]):
            return 0.0
        if direction == "long"  and e20[-1] < e50[-1] and e20[-2] >= e50[-2]:
            return 20.0
        if direction == "short" and e20[-1] > e50[-1] and e20[-2] <= e50[-2]:
            return 20.0
        return 0.0

    @staticmethod
    def _volume_decay_score(volumes, closes, direction, lookback=10) -> float:
        if len(volumes) < lookback:
            return 0.0
        recent_vol = float(np.mean(volumes[-5:]))
        prior_vol  = float(np.mean(volumes[-lookback:-5]))
        if prior_vol == 0:
            return 0.0
        vol_ratio = recent_vol / prior_vol
        # If price is still moving in direction but volume is decaying
        price_dir = float(np.sign(closes[-1] - closes[-5]))
        expected  = 1.0 if direction == "long" else -1.0
        if price_dir == expected and vol_ratio < 0.6:
            return 15.0
        return 0.0

    @staticmethod
    def _opposing_liquidity_score(highs, lows, closes, direction, lookback=10) -> float:
        if len(highs) < lookback:
            return 0.0
        prev_high = float(highs[-lookback:-1].max())
        prev_low  = float(lows[-lookback:-1].min())
        c = float(closes[-1])
        h = float(highs[-1])
        l = float(lows[-1])
        # If long and price just swept a high then rejected
        if direction == "long" and h > prev_high and c < prev_high:
            return 20.0
        # If short and price swept a low then rejected
        if direction == "short" and l < prev_low and c > prev_low:
            return 20.0
        return 0.0
