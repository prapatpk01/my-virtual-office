"""AI Expert with strategy-specific T1 profit-lock management.

T1 never closes part of the position. It only moves SL into profit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Signal, SignalType
from .safe_enhanced_ai_expert_strategy import SafeEnhancedAIExpertStrategy


_T1_RULES = {
    "trend_continuation": {"trigger_rr": 0.60, "lock_rr": 0.30},
    "momentum_expansion": {"trigger_rr": 0.70, "lock_rr": 0.30},
    "breakout": {"trigger_rr": 0.70, "lock_rr": 0.35},
    "mean_reversion": {"trigger_rr": 0.50, "lock_rr": 0.20},
    "swing_reversal": {"trigger_rr": 0.60, "lock_rr": 0.25},
}


class ManagedAIExpertStrategy(SafeEnhancedAIExpertStrategy):
    """Safe Enhanced AI Expert with T1 lock-only position management."""

    def _t1_rule(self, strategy_type: str) -> dict:
        default = {"trigger_rr": 0.60, "lock_rr": 0.30}
        rule = dict(_T1_RULES.get(strategy_type, default))
        key = strategy_type.upper() if strategy_type else "DEFAULT"
        rule["trigger_rr"] = self._env_float(
            f"AI_T1_RR_{key}", rule["trigger_rr"]
        )
        rule["lock_rr"] = self._env_float(
            f"AI_T1_LOCK_RR_{key}", rule["lock_rr"]
        )
        return rule

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        signal = await super().analyze(candles, current_price, mtf_candles)
        if signal.type not in (SignalType.BUY, SignalType.SELL):
            return signal

        metadata = self._normalize_metadata(signal)
        entry = self._open_entry or {}
        strategy_type = str(
            entry.get("strategy_type") or metadata.get("selected_strategy") or ""
        ).lower()
        rule = self._t1_rule(strategy_type)
        final_rr = self._safe_float(metadata.get("rr_ratio"), 1.2)

        metadata["t1_management"] = {
            "strategy": strategy_type,
            "trigger_rr": rule["trigger_rr"],
            "lock_rr": rule["lock_rr"],
            "partial_close_pct": 0.0,
            "final_tp_rr": final_rr,
        }
        signal.metadata = metadata

        if self._open_entry is not None:
            self._open_entry["tp1_rr"] = rule["trigger_rr"]
            self._open_entry["tp1_lock_rr"] = rule["lock_rr"]
            self._open_entry["tp2_rr"] = final_rr
            self._open_entry["partial_close_pct"] = 0.0

        return self._attach_decision_trace(signal, replace_hold_reason=False)

    def tick_open_position(
        self,
        current_price: float,
        position_key: Optional[str] = None,
    ):
        if not self._open_entry or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate
        from ..engines import indicators as ind

        entry = self._open_entry
        candles = self._latest_candles
        pos_id = position_key or self.symbol

        closes = np.array([float(c.close) for c in candles], dtype=float)
        highs = np.array([float(c.high) for c in candles], dtype=float)
        lows = np.array([float(c.low) for c in candles], dtype=float)
        atr_arr = ind.atr(closes, highs, lows, 14)
        valid = atr_arr[~np.isnan(atr_arr)]
        atr = float(valid[-1]) if len(valid) > 0 else 0.0
        if atr <= 0:
            return PositionUpdate(action="hold", reason="ATR unavailable")

        exit_sig = self._exit_engine.evaluate(
            candles,
            direction=entry["direction"],
            entry_price=entry["entry_price"],
            current_price=current_price,
            regime=None,
        )

        if pos_id not in self._position_manager._positions:
            self._position_manager.register_position(
                position_id=pos_id,
                direction=entry["direction"],
                entry_price=entry["entry_price"],
                stop_loss=entry["stop_loss"],
                take_profit=entry["take_profit"],
                atr=atr,
                tp1_rr=entry.get("tp1_rr"),
                tp2_rr=entry.get("tp2_rr"),
                tp1_lock_rr=entry.get("tp1_lock_rr"),
            )

        return self._position_manager.update(
            position_id=pos_id,
            current_price=current_price,
            current_atr=atr,
            exit_score=exit_sig.score,
        )
