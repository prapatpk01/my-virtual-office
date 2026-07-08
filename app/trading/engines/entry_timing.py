"""
Layer 5 + 6: Entry Timing Engine & Entry Trigger Checklist

Layer 5 — Multi-Timeframe cascade:
  4H  → Bias direction
  1H  → Structure confirmation
  15m → Entry zone
  5m  → Precision trigger

Layer 6 — Entry Trigger Checklist (all must pass for HIGH_CONVICTION entry):
  Trend confirmed on higher TF
  Momentum not overbought
  Liquidity sweep present
  Volume confirmation
  ATR within acceptable range
  Session quality sufficient
  R:R ratio >= 2.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .decision_engine import DecisionResult, ConfidenceLevel
from .market_intelligence import RegimeResult, MarketRegime
from .expert_analysis import ExpertScores


@dataclass
class EntrySignal:
    valid:       bool  = False
    direction:   str   = "hold"   # "long" | "short"
    entry_price: float = 0.0
    stop_loss:   float = 0.0
    take_profit: float = 0.0
    rr_ratio:    float = 0.0
    confidence:  float = 0.0
    checklist:   dict  = field(default_factory=dict)
    mtf_bias:    dict  = field(default_factory=dict)
    reason:      str   = ""


class EntryTimingEngine:
    """Multi-timeframe entry timing with full trigger checklist."""

    def __init__(
        self,
        min_rr: float = 2.0,
        min_session_score: float = 45.0,
        require_all_checks: bool = False,  # True = institutional mode
    ):
        self.min_rr            = min_rr
        self.min_session       = min_session_score
        self.require_all_checks= require_all_checks

    def evaluate(
        self,
        candles_15m: list,
        decision:    DecisionResult,
        regime:      RegimeResult,
        experts:     ExpertScores,
        mtf_candles: dict = None,  # {"1h": [...], "4h": [...], "5m": [...]}
        current_price: float = 0.0,
    ) -> EntrySignal:
        """Evaluate all layers and return EntrySignal."""
        if decision.direction == "hold" or not candles_15m:
            return EntrySignal(valid=False, direction="hold", reason="No signal")

        direction   = decision.direction
        price       = current_price or float(candles_15m[-1].close)
        closes_15m  = np.array([c.close for c in candles_15m], dtype=float)
        highs_15m   = np.array([c.high  for c in candles_15m], dtype=float)
        lows_15m    = np.array([c.low   for c in candles_15m], dtype=float)

        # ATR for SL/TP calculation
        atr_arr = self._atr(candles_15m, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else price * 0.01

        # ── Layer 5: MTF Bias ─────────────────────────────────────────────
        mtf_bias = self._compute_mtf_bias(candles_15m, mtf_candles or {})
        bias_aligned = (
            (direction == "long"  and mtf_bias["composite"] > 0) or
            (direction == "short" and mtf_bias["composite"] < 0)
        )

        # ── Layer 6: Entry Trigger Checklist ──────────────────────────────
        sl_price, tp_price = self._compute_sl_tp(
            direction, price, atr_val, highs_15m, lows_15m,
            regime.regime, experts
        )
        rr = self._compute_rr(direction, price, sl_price, tp_price)

        checklist = {
            "trend_confirmed":      experts.trend >= 55 and bias_aligned,
            "momentum_ok":          40 <= experts.momentum <= 85,  # not extreme
            "liquidity_confirmed":  experts.liquidity >= 50,
            "volume_confirmed":     experts.volume >= 50,
            "atr_ok":               0.5 * price * 0.005 <= atr_val <= 5 * price * 0.01,
            "session_ok":           experts.session >= self.min_session,
            "rr_ok":                rr >= self.min_rr,
            "confidence_ok":        decision.confidence_level in (
                                        ConfidenceLevel.GOOD,
                                        ConfidenceLevel.HIGH_CONVICTION,
                                    ),
            "regime_favorable":     self._regime_favorable(regime.regime, direction),
            "mtf_bias_aligned":     bias_aligned,
        }

        checks_passed  = sum(1 for v in checklist.values() if v)
        checks_total   = len(checklist)
        checks_pct     = checks_passed / checks_total

        # Gate: HIGH_CONVICTION requires 8/10, GOOD requires 6/10
        if decision.confidence_level == ConfidenceLevel.HIGH_CONVICTION:
            valid = checks_passed >= 8
        elif decision.confidence_level == ConfidenceLevel.GOOD:
            valid = checks_passed >= 6
        elif decision.confidence_level == ConfidenceLevel.WEAK:
            valid = checks_passed >= 7  # WEAK needs more checks to compensate
        else:
            valid = False

        if self.require_all_checks:
            valid = checks_passed == checks_total

        if not valid:
            failed = [k for k, v in checklist.items() if not v]
            return EntrySignal(
                valid=False, direction="hold",
                checklist=checklist, mtf_bias=mtf_bias,
                reason=f"Failed checks: {', '.join(failed)}",
            )

        return EntrySignal(
            valid=True,
            direction=direction,
            entry_price=round(price, 8),
            stop_loss=round(sl_price, 8),
            take_profit=round(tp_price, 8),
            rr_ratio=round(rr, 2),
            confidence=round(decision.confidence * checks_pct, 1),
            checklist=checklist,
            mtf_bias=mtf_bias,
            reason=f"{direction.upper()} | {decision.confidence_level.value} | {checks_passed}/{checks_total} checks | R:R {rr:.1f}",
        )

    def _compute_mtf_bias(
        self, candles_15m: list, mtf_candles: dict
    ) -> dict:
        """Score 4H > 1H > 15m bias cascade."""
        def _ema_slope(candles, period=20):
            if len(candles) < period + 2:
                return 0.0
            closes = np.array([c.close for c in candles], dtype=float)
            k = 2.0 / (period + 1)
            e = closes[:period].mean()
            for i in range(period, len(closes)):
                e = closes[i] * k + e * (1 - k)
            prev = closes[-(period//2)]
            return float(np.sign(closes[-1] - prev))

        def _rsi_val(candles, period=14):
            if len(candles) < period + 2:
                return 50.0
            closes = np.array([c.close for c in candles], dtype=float)
            delta  = np.diff(closes)
            gain   = delta[delta > 0].mean() if any(delta > 0) else 0.0
            loss   = (-delta[delta < 0]).mean() if any(delta < 0) else 0.0
            if loss == 0:
                return 100.0
            rs = gain / loss
            return 100 - 100 / (1 + rs)

        bias_4h  = _ema_slope(mtf_candles.get("4h", []), 20)
        bias_1h  = _ema_slope(mtf_candles.get("1h", []), 20)
        bias_15m = _ema_slope(candles_15m, 20)
        rsi_4h   = _rsi_val(mtf_candles.get("4h", []))
        rsi_1h   = _rsi_val(mtf_candles.get("1h", []))

        # Weighted composite: 4H=3x, 1H=2x, 15m=1x
        composite = (bias_4h * 3 + bias_1h * 2 + bias_15m * 1) / 6

        return {
            "4h_bias":   bias_4h,
            "1h_bias":   bias_1h,
            "15m_bias":  bias_15m,
            "rsi_4h":    round(rsi_4h, 1),
            "rsi_1h":    round(rsi_1h, 1),
            "composite": round(composite, 3),
            "label":     "BULL" if composite > 0.1 else "BEAR" if composite < -0.1 else "NEUTRAL",
        }

    def _compute_sl_tp(
        self, direction: str, price: float, atr: float,
        highs: np.ndarray, lows: np.ndarray,
        regime: MarketRegime, experts: ExpertScores,
    ) -> tuple[float, float]:
        """ATR-based SL/TP, regime-adjusted, with structure awareness."""
        # ATR multipliers per regime
        sl_mult = {
            MarketRegime.TREND:          1.5,
            MarketRegime.RANGE:          1.0,
            MarketRegime.BREAKOUT:       2.0,
            MarketRegime.REVERSAL:       1.2,
            MarketRegime.HIGH_VOLATILITY:2.5,
            MarketRegime.LOW_VOLATILITY: 1.0,
        }.get(regime, 1.5)

        # R:R target per regime (minimum 2.0)
        rr_target = {
            MarketRegime.TREND:          3.0,
            MarketRegime.RANGE:          2.0,
            MarketRegime.BREAKOUT:       3.5,
            MarketRegime.REVERSAL:       2.0,
            MarketRegime.HIGH_VOLATILITY:2.5,
            MarketRegime.LOW_VOLATILITY: 2.0,
        }.get(regime, 2.5)

        # Structure-based SL: beyond the recent swing with an ATR buffer.
        # SL must clear both the structure level and a minimum ATR distance,
        # otherwise ordinary noise stops the trade out. Width capped at 3×ATR.
        lookback = min(20, len(highs))
        if direction == "long":
            swing_sl  = float(lows[-lookback:].min()) if lookback > 0 else price * 0.98
            struct_sl = swing_sl - 0.3 * atr
            atr_sl    = price - sl_mult * atr
            sl        = min(struct_sl, atr_sl)          # wider of the two
            sl        = max(sl, price - 3.0 * atr)      # cap max width
            tp        = price + rr_target * (price - sl)
        else:  # short
            swing_sl  = float(highs[-lookback:].max()) if lookback > 0 else price * 1.02
            struct_sl = swing_sl + 0.3 * atr
            atr_sl    = price + sl_mult * atr
            sl        = max(struct_sl, atr_sl)          # wider of the two
            sl        = min(sl, price + 3.0 * atr)      # cap max width
            tp        = price - rr_target * (sl - price)

        return sl, tp

    @staticmethod
    def _compute_rr(direction: str, entry: float, sl: float, tp: float) -> float:
        if direction == "long":
            risk   = entry - sl
            reward = tp - entry
        else:
            risk   = sl - entry
            reward = entry - tp
        return reward / risk if risk > 0 else 0.0

    @staticmethod
    def _regime_favorable(regime: MarketRegime, direction: str) -> bool:
        """Some regimes are inherently more favorable for certain directions."""
        unfavorable = {
            MarketRegime.REVERSAL:       {"long"},   # Reversal regime → prefer short
            MarketRegime.HIGH_VOLATILITY:{},          # Both acceptable with wide SL
            MarketRegime.UNKNOWN:        {"long", "short"},
        }
        return direction not in unfavorable.get(regime, set())

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
