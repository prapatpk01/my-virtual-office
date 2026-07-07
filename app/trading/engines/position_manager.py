"""
Layer 7: Position Manager

Manages open positions with institutional-grade techniques:
  - Break Even: move SL to entry after price moves 1x ATR in direction
  - Trailing Stop: ATR-based trailing once in profit
  - Partial TP: take 25% / 50% / 100% at RR 1:1, 2:1, final target
  - Exit Score integration: close if exit score threshold hit
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PositionUpdate:
    action:       str   = "hold"     # "break_even" | "trail" | "partial_tp" | "close" | "hold"
    new_sl:       Optional[float] = None
    close_pct:    float = 0.0        # 0-1, fraction of position to close
    reason:       str   = ""
    exit_score:   float = 0.0


class PositionManager:
    """Post-entry position lifecycle management."""

    def __init__(
        self,
        be_atr_mult:      float = 1.0,   # Move to BE after 1x ATR profit
        trail_atr_mult:   float = 2.0,   # Trail at 2x ATR distance
        partial_tp_1_rr:  float = 1.0,   # Partial close at RR 1.0 (take 25%)
        partial_tp_2_rr:  float = 2.0,   # Partial close at RR 2.0 (take 50%)
        close_exit_score: float = 70.0,  # Close if exit AI score exceeds this
    ):
        self.be_mult        = be_atr_mult
        self.trail_mult     = trail_atr_mult
        self.partial_tp_1   = partial_tp_1_rr
        self.partial_tp_2   = partial_tp_2_rr
        self.close_exit_thr = close_exit_score
        self._positions: dict[str, dict] = {}  # key: position_id

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
            "direction":   direction,
            "entry":       entry_price,
            "sl":          stop_loss,
            "tp":          take_profit,
            "atr":         atr,
            "initial_sl":  stop_loss,
            "initial_risk": abs(entry_price - stop_loss),
            "be_triggered": False,
            "partial_1_done": False,
            "partial_2_done": False,
            "highest_price": entry_price,
            "lowest_price":  entry_price,
            "opened_at":    time.time(),
        }

    def update(
        self,
        position_id: str,
        current_price: float,
        current_atr:   float,
        exit_score:    float = 0.0,
    ) -> PositionUpdate:
        """Evaluate one position and return recommended action."""
        pos = self._positions.get(position_id)
        if not pos:
            return PositionUpdate(action="hold", reason="Unknown position")

        direction  = pos["direction"]
        entry      = pos["entry"]
        sl         = pos["sl"]
        tp         = pos["tp"]
        init_risk  = pos["initial_risk"]
        atr        = current_atr or pos["atr"]

        # Track high/low
        if direction == "long":
            pos["highest_price"] = max(pos["highest_price"], current_price)
            in_profit = current_price > entry
            profit    = current_price - entry
            risk_dist = entry - sl
        else:
            pos["lowest_price"] = min(pos["lowest_price"], current_price)
            in_profit = current_price < entry
            profit    = entry - current_price
            risk_dist = sl - entry

        current_rr = profit / init_risk if init_risk > 0 else 0.0

        # ── Exit Score override ────────────────────────────────────────────
        if exit_score >= self.close_exit_thr:
            return PositionUpdate(
                action="close", close_pct=1.0,
                reason=f"Exit AI score {exit_score:.0f} ≥ {self.close_exit_thr:.0f}",
                exit_score=exit_score,
            )

        # ── Partial TP 2 (50% at RR 2.0) ─────────────────────────────────
        if not pos["partial_2_done"] and current_rr >= self.partial_tp_2:
            pos["partial_2_done"] = True
            return PositionUpdate(
                action="partial_tp", close_pct=0.50,
                reason=f"Partial TP2 at R:R {current_rr:.1f} (≥{self.partial_tp_2})",
                exit_score=exit_score,
            )

        # ── Partial TP 1 (25% at RR 1.0) ────────────────────────────────
        if not pos["partial_1_done"] and current_rr >= self.partial_tp_1:
            pos["partial_1_done"] = True
            return PositionUpdate(
                action="partial_tp", close_pct=0.25,
                reason=f"Partial TP1 at R:R {current_rr:.1f} (≥{self.partial_tp_1})",
                exit_score=exit_score,
            )

        # ── Trailing Stop (after BE) ──────────────────────────────────────
        if pos["be_triggered"] and in_profit:
            if direction == "long":
                trail_sl = pos["highest_price"] - self.trail_mult * atr
                if trail_sl > sl:
                    pos["sl"] = trail_sl
                    return PositionUpdate(
                        action="trail", new_sl=round(trail_sl, 8),
                        reason=f"Trailing stop @ {trail_sl:.4f} (ATR x{self.trail_mult})",
                        exit_score=exit_score,
                    )
            else:
                trail_sl = pos["lowest_price"] + self.trail_mult * atr
                if trail_sl < sl:
                    pos["sl"] = trail_sl
                    return PositionUpdate(
                        action="trail", new_sl=round(trail_sl, 8),
                        reason=f"Trailing stop @ {trail_sl:.4f} (ATR x{self.trail_mult})",
                        exit_score=exit_score,
                    )

        # ── Break Even trigger ─────────────────────────────────────────────
        if not pos["be_triggered"]:
            be_trigger = self.be_mult * atr
            if profit >= be_trigger:
                pos["be_triggered"] = True
                new_sl = entry + atr * 0.1 if direction == "long" else entry - atr * 0.1
                pos["sl"] = new_sl
                return PositionUpdate(
                    action="break_even", new_sl=round(new_sl, 8),
                    reason=f"Break even triggered (profit={profit:.4f} ≥ {be_trigger:.4f})",
                    exit_score=exit_score,
                )

        return PositionUpdate(action="hold", reason="Position healthy", exit_score=exit_score)

    def remove_position(self, position_id: str) -> None:
        self._positions.pop(position_id, None)

    def get_position_state(self, position_id: str) -> Optional[dict]:
        return self._positions.get(position_id)
