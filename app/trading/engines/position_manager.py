"""AI Expert position lifecycle manager.

Management rules:
- T1 is a protection trigger only: do not partially close the position.
- When T1 is reached, move SL into profit by the configured lock R.
- Final TP closes the full remaining position.
- Exit AI may force a full close before final TP.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionUpdate:
    action: str = "hold"          # "partial_tp" | "close" | "hold"
    new_sl: Optional[float] = None
    close_pct: float = 0.0
    reason: str = ""
    exit_score: float = 0.0


class PositionManager:
    """Manage T1 profit locking, final TP and Exit-AI closures."""

    def __init__(
        self,
        partial_tp_1_rr: float = 0.6,
        partial_tp_2_rr: float = 1.2,
        close_exit_score: float = 70.0,
        tp1_lock_rr: float = 0.3,
        be_atr_mult: float = 1.0,
        trail_atr_mult: float = 2.0,
    ):
        self.tp1_rr = partial_tp_1_rr
        self.tp2_rr = partial_tp_2_rr
        self.tp1_lock_rr = tp1_lock_rr
        self.close_exit_thr = close_exit_score
        self._positions: dict[str, dict] = {}

    def register_position(
        self,
        position_id: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        tp1_rr: Optional[float] = None,
        tp2_rr: Optional[float] = None,
        tp1_lock_rr: Optional[float] = None,
    ) -> None:
        self._positions[position_id] = {
            "direction": direction,
            "entry": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "initial_risk": abs(entry_price - stop_loss),
            "tp1_rr": tp1_rr if tp1_rr is not None else self.tp1_rr,
            "tp2_rr": tp2_rr if tp2_rr is not None else self.tp2_rr,
            "tp1_lock_rr": tp1_lock_rr if tp1_lock_rr is not None else self.tp1_lock_rr,
            "tp1_done": False,
            "tp2_done": False,
            "opened_at": time.time(),
        }

    def update(
        self,
        position_id: str,
        current_price: float,
        current_atr: float,
        exit_score: float = 0.0,
    ) -> PositionUpdate:
        pos = self._positions.get(position_id)
        if not pos:
            return PositionUpdate(action="hold", reason="Unknown position")

        direction = pos["direction"]
        entry = pos["entry"]
        init_risk = pos["initial_risk"]
        tp1_rr = float(pos.get("tp1_rr", self.tp1_rr))
        tp2_rr = float(pos.get("tp2_rr", self.tp2_rr))
        lock_rr = float(pos.get("tp1_lock_rr", self.tp1_lock_rr))

        profit = current_price - entry if direction == "long" else entry - current_price
        current_rr = profit / init_risk if init_risk > 0 else 0.0

        if exit_score >= self.close_exit_thr:
            pos["tp2_done"] = True
            return PositionUpdate(
                action="close", close_pct=1.0,
                reason=f"Exit AI {exit_score:.0f} ≥ {self.close_exit_thr:.0f}",
                exit_score=exit_score,
            )

        if not pos["tp2_done"] and current_rr >= tp2_rr:
            pos["tp2_done"] = True
            return PositionUpdate(
                action="close", close_pct=1.0,
                reason=f"Final TP @ {current_rr:.2f}R (target {tp2_rr:.2f}R)",
                exit_score=exit_score,
            )

        if not pos["tp1_done"] and current_rr >= tp1_rr:
            pos["tp1_done"] = True
            locked_sl = (
                entry + lock_rr * init_risk
                if direction == "long"
                else entry - lock_rr * init_risk
            )
            locked_sl = round(locked_sl, 8)
            pos["sl"] = locked_sl
            # Keep the existing action name so the production bot follows its
            # established SL-modification path. close_pct=0 prevents any sale.
            return PositionUpdate(
                action="partial_tp", close_pct=0.0, new_sl=locked_sl,
                reason=(
                    f"T1 @ {current_rr:.2f}R (≥{tp1_rr:.2f}R) → "
                    f"no partial close, SL locks +{lock_rr:.2f}R"
                ),
                exit_score=exit_score,
            )

        return PositionUpdate(action="hold", reason="Holding", exit_score=exit_score)

    def remove_position(self, position_id: str) -> None:
        self._positions.pop(position_id, None)

    def get_position_state(self, position_id: str) -> Optional[dict]:
        return self._positions.get(position_id)
