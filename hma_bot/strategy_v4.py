"""HMA Expert MTF V3.4 — one-target runner management.

Entry logic is inherited unchanged from V3.1 Balanced. Position management is
simplified to one profit milestone:

    Target 1 +0.8% -> move local SL to +0.5%
    Runner          -> final TP (TP_PCT, default +1.5%)

There is no second target or second lock stage.
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


class PrecisionTrendStructureV4(v3.PrecisionTrendStructureV3):
    """V3 entry model with one target-lock stage and a final-TP runner."""

    def __init__(self, config: Optional[legacy.StrategyConfig] = None) -> None:
        super().__init__(config)
        # New names avoid accidentally inheriting obsolete T1/T2 Railway values.
        self.runner_trigger_pct = _env_float("RUNNER_TRIGGER_PCT", 0.008)
        self.runner_lock_pct = _env_float("RUNNER_LOCK_PCT", 0.005)

    def locked_stop(self, side: Side, entry: float, best_price: float):
        """Return only stage 0 or stage 1; stage 2 no longer exists."""
        entry = float(entry or 0.0)
        best_price = float(best_price or 0.0)
        if entry <= 0.0 or best_price <= 0.0:
            return 0.0, 0

        if side == Side.LONG:
            favorable = best_price / entry - 1.0
            if favorable >= self.runner_trigger_pct:
                return entry * (1.0 + self.runner_lock_pct), 1
            return entry * (1.0 - self.cfg.stop_loss_pct), 0

        favorable = entry / best_price - 1.0
        if favorable >= self.runner_trigger_pct:
            return entry * (1.0 - self.runner_lock_pct), 1
        return entry * (1.0 + self.cfg.stop_loss_pct), 0


# Explicit alias for imports/tests.
MTFStructureStrategyV4 = PrecisionTrendStructureV4
