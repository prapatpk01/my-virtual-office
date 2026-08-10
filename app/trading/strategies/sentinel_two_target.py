"""Sentinel-specific two-target position management.

Keeps Sentinel's existing 1H S/R stop and final-target logic intact, but adds
an intermediate protection target:

- TP1 = +1.00R from the actual Sentinel entry/initial SL distance.
- At TP1, close 60% of the position.
- Move the remaining 40% stop to +0.30R.
- TP2 stays strategy-native:
  * mapped trades keep the existing 1H R1/S1 hard target;
  * OPEN_SKY/OPEN_FLOOR keep the existing dynamic S/R/structure runner exit.

This is installed as a small class patch so the production Sentinel strategy
can keep its S/R, MCDX and entry implementation isolated from generic position
management used by other strategies.
"""
from __future__ import annotations

from typing import Optional

from ..engines.position_manager import PositionUpdate


TP1_RR = 1.00
TP1_CLOSE_PCT = 0.60
TP1_LOCK_RR = 0.30


def install_sentinel_two_target(strategy_cls) -> None:
    """Install Sentinel-only TP1 trim + protected runner management once."""
    if getattr(strategy_cls, "_sentinel_two_target_installed", False):
        return

    original_tick = strategy_cls.tick_open_position
    original_attach = strategy_cls.attach_existing_position
    original_reset = strategy_cls._reset_position_state

    # Public strategy attributes are useful to monitoring/notification code.
    strategy_cls.tp1_rr = TP1_RR
    strategy_cls.tp1_close_pct = TP1_CLOSE_PCT
    strategy_cls.tp1_lock_rr = TP1_LOCK_RR
    strategy_cls._sentinel_two_target_installed = True

    def _clear_tp_state(self) -> None:
        self._sentinel_tp1_done = False
        self._sentinel_initial_risk = None
        self._sentinel_tp1_lock_sl = None

    def _reset_position_state(self) -> None:
        original_reset(self)
        _clear_tp_state(self)

    def attach_existing_position(
        self,
        direction: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        original_attach(self, direction, entry_price, stop_loss, take_profit)
        _clear_tp_state(self)

        # On restart/reconcile, an SL already locked on the profitable side of
        # entry means TP1 was previously completed. Do not trim a second time.
        if stop_loss is None:
            return
        entry = float(entry_price)
        sl = float(stop_loss)
        side = str(direction).lower()
        already_locked = (
            (side == "long" and sl > entry)
            or (side == "short" and sl < entry)
        )
        if already_locked:
            self._sentinel_tp1_done = True
            self._sentinel_tp1_lock_sl = sl
        else:
            risk = abs(entry - sl)
            self._sentinel_initial_risk = risk if risk > 0 else None

    def _ensure_tp_state(self) -> None:
        if not hasattr(self, "_sentinel_tp1_done"):
            _clear_tp_state(self)
        if self._sentinel_tp1_done:
            return
        if self._sentinel_initial_risk is not None:
            return
        entry = getattr(self, "_entry_price", None)
        sl = getattr(self, "_entry_sl", None)
        if entry is None or sl is None:
            return
        risk = abs(float(entry) - float(sl))
        if risk > 0:
            self._sentinel_initial_risk = risk

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        if getattr(self, "_open_position", None) is None:
            return original_tick(self, current_price, position_key)

        _ensure_tp_state(self)

        # OPEN_SKY/FLOOR retains its existing emergency structure/SR exit. If
        # the runner has already produced a full-close signal on this tick,
        # that takes priority over trimming only 60% at TP1.
        base_update = None
        if bool(getattr(self, "_open_ended", False)):
            base_update = original_tick(self, current_price, position_key)
            if base_update is not None and base_update.action == "close":
                return base_update

        if not bool(getattr(self, "_sentinel_tp1_done", False)):
            entry = getattr(self, "_entry_price", None)
            initial_risk = getattr(self, "_sentinel_initial_risk", None)
            side = str(getattr(self, "_open_position", "")).lower()
            if entry is not None and initial_risk is not None and initial_risk > 0 and side in ("long", "short"):
                entry = float(entry)
                current = float(current_price)
                profit = current - entry if side == "long" else entry - current
                current_r = profit / float(initial_risk)

                if current_r >= TP1_RR:
                    lock_sl = (
                        entry + TP1_LOCK_RR * float(initial_risk)
                        if side == "long"
                        else entry - TP1_LOCK_RR * float(initial_risk)
                    )
                    lock_sl = round(lock_sl, 8)
                    self._sentinel_tp1_done = True
                    self._sentinel_tp1_lock_sl = lock_sl
                    # Keep the strategy's internal stop aligned with the risk
                    # manager/exchange stop that bot.py will modify.
                    self._entry_sl = lock_sl
                    runner_mode = (
                        "dynamic S/R/structure runner"
                        if bool(getattr(self, "_open_ended", False))
                        else "1H S/R TP2 runner"
                    )
                    return PositionUpdate(
                        action="partial_tp",
                        close_pct=TP1_CLOSE_PCT,
                        new_sl=lock_sl,
                        reason=(
                            f"Sentinel TP1 @ {current_r:.2f}R: trim 60%; "
                            f"lock remaining 40% SL at +{TP1_LOCK_RR:.2f}R; "
                            f"{runner_mode}"
                        ),
                    )

        # Fixed mapped trades keep their original R1/S1 hard TP. Open-ended
        # trades keep the already-computed dynamic runner HOLD reason/exit.
        if base_update is not None:
            return base_update
        return original_tick(self, current_price, position_key)

    strategy_cls.tick_open_position = tick_open_position
    strategy_cls.attach_existing_position = attach_existing_position
    strategy_cls._reset_position_state = _reset_position_state
