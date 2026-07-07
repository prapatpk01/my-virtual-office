"""
Layer 7: Position Manager

Trade management rules:
  TP1 @ 0.5R  → close 50% of position, move SL to break-even immediately
  TP2 @ 1.2R  → close remaining 100%
  Exit AI     → forced close if exit score ≥ threshold
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionUpdate:
    action:     str            = "hold"   # "partial_tp" | "close" | "hold"
    new_sl:     Optional[float] = None    # set on partial_tp (BE) and close
    close_pct:  float           = 0.0    # fraction of current position to close
    reason:     str             = ""
    exit_score: float           = 0.0


class PositionManager:
    """
    Post-entry lifecycle:
      TP1 (0.5R) → take 50%, slide SL to entry (break-even)
      TP2 (1.2R) → close all remaining
      Exit AI    → close if score ≥ close_exit_score
    """

    def __init__(
        self,
        partial_tp_1_rr:  float = 0.5,    # R multiple to trigger TP1
        partial_tp_2_rr:  float = 1.2,    # R multiple to trigger TP2 (full close)
        close_exit_score: float = 70.0,   # exit AI score threshold for forced close
        # legacy params accepted but unused (keeps call-sites compatible)
        be_atr_mult:      float = 1.0,
        trail_atr_mult:   float = 2.0,
    ):
        self.tp1_rr          = partial_tp_1_rr
        self.tp2_rr          = partial_tp_2_rr
        self.close_exit_thr  = close_exit_score
        self._positions: dict[str, dict] = {}

    # ------------------------------------------------------------------

    def register_position(
        self,
        position_id: str,
        direction:   str,
        entry_price: float,
        stop_loss:   float,
        take_profit: float,
        atr:         float,
    ) -> None:
        self._positions[position_id] = {
            "direction":      direction,
            "entry":          entry_price,
            "sl":             stop_loss,
            "tp":             take_profit,
            "initial_risk":   abs(entry_price - stop_loss),
            "tp1_done":       False,
            "tp2_done":       False,
            "opened_at":      time.time(),
        }

    def update(
        self,
        position_id:   str,
        current_price: float,
        current_atr:   float,   # kept for API compatibility; not used in new logic
        exit_score:    float = 0.0,
    ) -> PositionUpdate:
        pos = self._positions.get(position_id)
        if not pos:
            return PositionUpdate(action="hold", reason="Unknown position")

        direction  = pos["direction"]
        entry      = pos["entry"]
        init_risk  = pos["initial_risk"]

        if direction == "long":
            profit = current_price - entry
        else:
            profit = entry - current_price

        current_rr = profit / init_risk if init_risk > 0 else 0.0

        # ── Exit AI override ──────────────────────────────────────────────
        if exit_score >= self.close_exit_thr:
            pos["tp2_done"] = True
            return PositionUpdate(
                action="close", close_pct=1.0,
                reason=f"Exit AI {exit_score:.0f} ≥ {self.close_exit_thr:.0f}",
                exit_score=exit_score,
            )

        # ── TP2 @ 1.2R → close all remaining ─────────────────────────────
        if not pos["tp2_done"] and current_rr >= self.tp2_rr:
            pos["tp2_done"] = True
            return PositionUpdate(
                action="close", close_pct=1.0,
                reason=f"TP2 @ R:R {current_rr:.2f} (target {self.tp2_rr}R)",
                exit_score=exit_score,
            )

        # ── TP1 @ 0.5R → close 50%, move SL to BE ────────────────────────
        if not pos["tp1_done"] and current_rr >= self.tp1_rr:
            pos["tp1_done"] = True
            # break-even = entry price (slight buffer of 0 pips — user can adjust)
            be_sl = round(entry, 8)
            pos["sl"] = be_sl
            return PositionUpdate(
                action="partial_tp", close_pct=0.50,
                new_sl=be_sl,
                reason=f"TP1 @ R:R {current_rr:.2f} (≥{self.tp1_rr}R) → 50% closed, SL→BE",
                exit_score=exit_score,
            )

        return PositionUpdate(action="hold", reason="Holding", exit_score=exit_score)

    def remove_position(self, position_id: str) -> None:
        self._positions.pop(position_id, None)

    def get_position_state(self, position_id: str) -> Optional[dict]:
        return self._positions.get(position_id)
