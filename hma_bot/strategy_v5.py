"""HMA Expert MTF V3.5 — balanced two-stage profit locks.

Entry logic is inherited unchanged from V3.1 Balanced. Position management:

    Stage 1 +0.7% -> move local SL to +0.4%
    Stage 2 +1.1% -> move local SL to +0.75%
    Runner         -> final TP (TP_PCT, default +1.5%)

The stages move the stop only; they do not partially close the position.
"""
from __future__ import annotations

import os
from typing import Optional

import strategy as legacy
import strategy_v3 as v3

Side = legacy.Side


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PrecisionTrendStructureV5(v3.PrecisionTrendStructureV3):
    """V3 entry model with two balanced profit-lock stages."""

    def __init__(self, config: Optional[legacy.StrategyConfig] = None) -> None:
        super().__init__(config)

        # Dedicated names prevent old T1/T2 or one-target Railway variables from
        # silently changing this production profile.
        self.stage1_trigger_pct = _env_float("STAGE1_TRIGGER_PCT", 0.007)
        self.stage1_lock_pct = _env_float("STAGE1_LOCK_PCT", 0.004)
        self.stage2_trigger_pct = _env_float("STAGE2_TRIGGER_PCT", 0.011)
        self.stage2_lock_pct = _env_float("STAGE2_LOCK_PCT", 0.0075)

        # Defensive normalization: later stages must always be farther from entry
        # and lock more profit than earlier stages.
        self.stage1_trigger_pct = max(0.0, self.stage1_trigger_pct)
        self.stage1_lock_pct = min(
            max(0.0, self.stage1_lock_pct), self.stage1_trigger_pct
        )
        self.stage2_trigger_pct = max(
            self.stage1_trigger_pct, self.stage2_trigger_pct
        )
        self.stage2_lock_pct = min(
            max(self.stage1_lock_pct, self.stage2_lock_pct),
            self.stage2_trigger_pct,
        )

    def locked_stop(self, side: Side, entry: float, best_price: float):
        """Return stage 0, 1, or 2 and the corresponding one-way profit stop."""
        entry = float(entry or 0.0)
        best_price = float(best_price or 0.0)
        if entry <= 0.0 or best_price <= 0.0:
            return 0.0, 0

        if side == Side.LONG:
            favorable = best_price / entry - 1.0
            if favorable >= self.stage2_trigger_pct:
                return entry * (1.0 + self.stage2_lock_pct), 2
            if favorable >= self.stage1_trigger_pct:
                return entry * (1.0 + self.stage1_lock_pct), 1
            return entry * (1.0 - self.cfg.stop_loss_pct), 0

        favorable = entry / best_price - 1.0
        if favorable >= self.stage2_trigger_pct:
            return entry * (1.0 - self.stage2_lock_pct), 2
        if favorable >= self.stage1_trigger_pct:
            return entry * (1.0 - self.stage1_lock_pct), 1
        return entry * (1.0 + self.cfg.stop_loss_pct), 0


MTFStructureStrategyV5 = PrecisionTrendStructureV5
