"""Production enhancements for AIExpertStrategy.

Keeps the existing Layer 0-8 decision engine intact, then applies a compact
entry-quality layer before an order is allowed to leave the strategy:

1. One accepted signal per closed 15m candle (prevents repeated/stale entries).
2. Strategy-specific confidence thresholds.
3. Strategy-specific SMA30/ATR anti-chase limits.
4. Recalculate live reward/risk from the actual current price.
5. Reduce risk while expectancy history is still immature.

The wrapper deliberately does not change open-position management, exit logic,
trade journaling, regime classification, or strategy selection.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .ai_expert_strategy import AIExpertStrategy
from .base import Signal, SignalType


_DEFAULT_CONFIDENCE = {
    "trend_continuation": 52.0,
    "momentum_expansion": 58.0,
    "breakout": 60.0,
    "mean_reversion": 56.0,
    "swing_reversal": 62.0,
}

_DEFAULT_CHASE_ATR = {
    "trend_continuation": 1.10,
    "momentum_expansion": 1.40,
    "breakout": 1.50,
    # Mean reversion naturally starts farther from the mean.
    "mean_reversion": 2.00,
    "swing_reversal": 1.20,
}

_DEFAULT_MIN_LIVE_RR = {
    "trend_continuation": 1.00,
    "momentum_expansion": 1.05,
    "breakout": 1.10,
    "mean_reversion": 1.00,
    "swing_reversal": 1.10,
}


class EnhancedAIExpertStrategy(AIExpertStrategy):
    """AI Expert with fresh-entry, anti-chase and low-sample risk controls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"AIExpert({self.symbol})"
        self._last_accepted_entry_bar_ts: Optional[int] = None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _latest_bar_ts(candles: list) -> int:
        if not candles:
            return 0
        ts = getattr(candles[-1], "timestamp", 0)
        try:
            return int(ts)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _sma(values: list[float], period: int) -> float:
        if len(values) < period:
            return float("nan")
        return float(np.mean(np.asarray(values[-period:], dtype=float)))

    @staticmethod
    def _atr(candles: list, period: int = 14) -> float:
        if len(candles) < period + 1:
            return float("nan")
        tr = []
        for i in range(1, len(candles)):
            high = float(candles[i].high)
            low = float(candles[i].low)
            prev_close = float(candles[i - 1].close)
            tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        if len(tr) < period:
            return float("nan")
        return float(np.mean(np.asarray(tr[-period:], dtype=float)))

    def _cancel_generated_entry(self, reason: str) -> None:
        """Undo AIExpertStrategy's optimistic internal open-state assignment."""
        cancel = getattr(self, "cancel_pending_entry", None)
        if callable(cancel):
            try:
                cancel(reason)
                return
            except Exception:
                pass
        self._open_entry = None

    def _blocked_hold(self, signal: Signal, reason: str, extra: dict) -> Signal:
        self._cancel_generated_entry(reason)
        metadata = dict(getattr(signal, "metadata", None) or {})
        metadata.setdefault("entry_quality", {}).update(extra)
        metadata["entry_quality"]["passed"] = False
        metadata["entry_quality"]["block_reason"] = reason
        return self._hold(float(signal.price), reason=reason, metadata=metadata)

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        signal = await super().analyze(candles, current_price, mtf_candles)
        if signal.type not in (SignalType.BUY, SignalType.SELL):
            return signal

        entry = self._open_entry or {}
        strategy_type = str(entry.get("strategy_type", "") or "").lower()
        confidence_score = float(entry.get("decision_score", 0.0) or 0.0)
        bar_ts = self._latest_bar_ts(candles)

        # A valid setup can remain true for several five-minute scans. Accept it
        # only once for the latest 15m bar; a rejected order is reset by the bot
        # via cancel_pending_entry and may be reconsidered only on a new bar.
        if bar_ts and self._last_accepted_entry_bar_ts == bar_ts:
            return self._blocked_hold(
                signal,
                "Fresh-entry gate: this 15M setup bar was already processed",
                {"strategy": strategy_type, "bar_ts": bar_ts},
            )

        threshold_default = _DEFAULT_CONFIDENCE.get(strategy_type, 58.0)
        env_key = f"AI_CONF_{strategy_type.upper()}" if strategy_type else "AI_CONF_DEFAULT"
        confidence_required = self._env_float(env_key, threshold_default)
        if confidence_score < confidence_required:
            return self._blocked_hold(
                signal,
                f"{strategy_type or 'AI'} confidence {confidence_score:.0f} < {confidence_required:.0f}",
                {
                    "strategy": strategy_type,
                    "confidence": confidence_score,
                    "confidence_required": confidence_required,
                },
            )

        closes = [float(c.close) for c in candles]
        sma30 = self._sma(closes, 30)
        atr14 = self._atr(candles, 14)
        chase_atr = 0.0
        chase_limit_default = _DEFAULT_CHASE_ATR.get(strategy_type, 1.25)
        chase_env = f"AI_CHASE_ATR_{strategy_type.upper()}" if strategy_type else "AI_CHASE_ATR_DEFAULT"
        chase_limit = self._env_float(chase_env, chase_limit_default)
        if np.isfinite(sma30) and np.isfinite(atr14) and atr14 > 0:
            chase_atr = abs(float(current_price) - sma30) / atr14
            if chase_atr > chase_limit:
                return self._blocked_hold(
                    signal,
                    f"Anti-chase: price {chase_atr:.2f}ATR from SMA30 > {chase_limit:.2f}ATR",
                    {
                        "strategy": strategy_type,
                        "sma30": round(sma30, 8),
                        "atr14": round(atr14, 8),
                        "chase_atr": round(chase_atr, 3),
                        "chase_limit_atr": chase_limit,
                    },
                )

        metadata = dict(getattr(signal, "metadata", None) or {})
        sl = float(metadata.get("stop_loss", entry.get("stop_loss", 0.0)) or 0.0)
        tp = float(metadata.get("take_profit", entry.get("take_profit", 0.0)) or 0.0)
        risk = abs(float(current_price) - sl)
        reward = abs(tp - float(current_price))
        live_rr = reward / risk if risk > 0 else 0.0
        min_rr_default = _DEFAULT_MIN_LIVE_RR.get(strategy_type, 1.0)
        rr_env = f"AI_MIN_LIVE_RR_{strategy_type.upper()}" if strategy_type else "AI_MIN_LIVE_RR_DEFAULT"
        min_live_rr = self._env_float(rr_env, min_rr_default)
        if sl <= 0 or tp <= 0 or risk <= 0 or live_rr < min_live_rr:
            return self._blocked_hold(
                signal,
                f"Live R:R {live_rr:.2f} below {min_live_rr:.2f} after price movement",
                {
                    "strategy": strategy_type,
                    "live_rr": round(live_rr, 3),
                    "min_live_rr": min_live_rr,
                    "stop_loss": sl,
                    "take_profit": tp,
                },
            )

        expectancy = metadata.get("expectancy", {}) or {}
        sample_size = int(expectancy.get("sample_size", 0) or 0)
        if sample_size < 10:
            sample_risk_cap = self._env_float("AI_LOW_SAMPLE_RISK_MULT", 0.60)
            expectancy_mode = "learning"
        elif sample_size < 20:
            sample_risk_cap = self._env_float("AI_SOFT_SAMPLE_RISK_MULT", 0.80)
            expectancy_mode = "soft"
        else:
            sample_risk_cap = 1.0
            expectancy_mode = "hard"

        dynamic = dict(metadata.get("dynamic_risk", {}) or {})
        original_risk_mult = float(dynamic.get("risk_multiplier", entry.get("risk_multiplier", 1.0)) or 1.0)
        adjusted_risk_mult = min(original_risk_mult, sample_risk_cap)
        dynamic["risk_multiplier"] = adjusted_risk_mult
        dynamic["original_risk_multiplier"] = original_risk_mult
        dynamic["expectancy_mode"] = expectancy_mode
        dynamic["expectancy_sample_size"] = sample_size
        metadata["dynamic_risk"] = dynamic
        metadata["entry_quality"] = {
            "passed": True,
            "strategy": strategy_type,
            "bar_ts": bar_ts,
            "confidence": confidence_score,
            "confidence_required": confidence_required,
            "sma30": round(sma30, 8) if np.isfinite(sma30) else None,
            "atr14": round(atr14, 8) if np.isfinite(atr14) else None,
            "chase_atr": round(chase_atr, 3),
            "chase_limit_atr": chase_limit,
            "live_rr": round(live_rr, 3),
            "min_live_rr": min_live_rr,
            "expectancy_mode": expectancy_mode,
            "sample_size": sample_size,
            "risk_multiplier": adjusted_risk_mult,
        }

        signal.metadata = metadata
        if self._open_entry is not None:
            self._open_entry["risk_multiplier"] = adjusted_risk_mult
            self._open_entry["entry_quality_checked_at"] = time.time()
            self._open_entry["entry_bar_ts"] = bar_ts

        self._last_accepted_entry_bar_ts = bar_ts or self._last_accepted_entry_bar_ts
        return signal
