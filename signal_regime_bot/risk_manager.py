"""Portfolio risk and fee-aware position sizing.

The user-requested base risk is 5% of balance per accepted trade.  Because two
positions may be open concurrently, total planned open risk is capped at 10%.
Position size includes estimated round-trip fees and slippage so the intended
5% risk is not exceeded by execution costs.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from config import Config

logger = logging.getLogger("risk_manager")


def _day_key(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


@dataclass
class RiskState:
    day: str = ""
    day_start_balance: float = 0.0
    day_realized_pnl: float = 0.0
    loss_streak: int = 0
    cooldown_until: float = 0.0
    peak_balance: float = 0.0


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = RiskState()

    def _roll_day(self, balance: float, now: float) -> None:
        today = _day_key(now)
        if today != self.state.day or self.state.day_start_balance <= 0:
            self.state.day = today
            self.state.day_start_balance = max(balance, 0.0)
            self.state.day_realized_pnl = 0.0
            if self.state.peak_balance <= 0:
                self.state.peak_balance = max(balance, 0.0)

    def register_trade_result(self, pnl: float, balance_after: float, now: float) -> None:
        """Register one *fully closed trade*, not each partial leg."""
        self._roll_day(balance_after - pnl, now)
        self.state.day_realized_pnl += pnl
        self.state.peak_balance = max(self.state.peak_balance, balance_after)
        epsilon = 1e-8
        if pnl < -epsilon:
            self.state.loss_streak += 1
            if self.state.loss_streak >= self.cfg.loss_streak_limit:
                self.state.cooldown_until = max(
                    self.state.cooldown_until,
                    now + self.cfg.loss_streak_cooldown_min * 60,
                )
                logger.warning(
                    "[RISK] %d consecutive losses — cooldown %d minutes",
                    self.state.loss_streak,
                    self.cfg.loss_streak_cooldown_min,
                )
        elif pnl > epsilon:
            self.state.loss_streak = 0
        # A near-zero trade neither resets nor increases the streak.

    def is_in_cooldown(self, now: float) -> bool:
        return now < self.state.cooldown_until

    def cooldown_remaining_sec(self, now: float) -> float:
        return max(0.0, self.state.cooldown_until - now)

    def check_daily_limits(self, balance: float, now: float) -> tuple[bool, str]:
        self._roll_day(balance, now)
        if self.state.day_start_balance <= 0:
            return True, "ok"
        pnl_pct = self.state.day_realized_pnl / self.state.day_start_balance
        if self.cfg.daily_loss_limit_enabled and pnl_pct <= -self.cfg.daily_loss_limit_pct:
            return (
                False,
                f"daily loss lock: {pnl_pct * 100:.1f}% <= -{self.cfg.daily_loss_limit_pct * 100:.1f}%",
            )
        if self.cfg.daily_profit_lock_enabled and pnl_pct >= self.cfg.daily_profit_lock_pct:
            return (
                False,
                f"daily profit lock: {pnl_pct * 100:.1f}% >= +{self.cfg.daily_profit_lock_pct * 100:.1f}%",
            )
        return True, "ok"

    def size_by_risk(
        self,
        balance: float,
        entry_price: float,
        sl_price: float,
        regime_size_multiplier: float = 1.0,
        fee_rate: float | None = None,
        expected_slippage_pct: float | None = None,
    ) -> float:
        """Return base quantity whose worst-case cash loss is about 5%.

        Per-unit effective risk includes price-to-stop distance, entry+exit fees
        and estimated slippage.  Leverage limits margin usage; it does not alter
        the cash risk budget.
        """
        if balance <= 0 or entry_price <= 0 or sl_price <= 0:
            return 0.0
        stop_distance = abs(entry_price - sl_price)
        if stop_distance <= 0:
            return 0.0
        fee = self.cfg.fee_rate if fee_rate is None else max(0.0, fee_rate)
        slip_pct = (
            getattr(self.cfg, "expected_slippage_pct", 0.0005)
            if expected_slippage_pct is None
            else max(0.0, expected_slippage_pct)
        )
        # One entry fill + one eventual exit fill.
        cost_distance = entry_price * (2.0 * fee + slip_pct)
        effective_risk_per_unit = stop_distance + cost_distance
        multiplier = max(0.0, min(1.0, regime_size_multiplier))
        risk_cash = balance * self.cfg.risk_per_trade * multiplier
        quantity = risk_cash / effective_risk_per_unit

        # Isolated-margin safety cap.  Leave 5% margin free for fees/maintenance.
        max_notional = balance * self.cfg.leverage * 0.95
        quantity = min(quantity, max_notional / entry_price)
        return round(max(quantity, 0.0), 8)

    def estimated_risk_cash(
        self,
        quantity: float,
        entry_price: float,
        sl_price: float,
    ) -> float:
        fee_distance = entry_price * (
            2.0 * self.cfg.fee_rate + getattr(self.cfg, "expected_slippage_pct", 0.0005)
        )
        return quantity * (abs(entry_price - sl_price) + fee_distance)

    def can_open_new(
        self,
        balance: float,
        now: float,
        open_position_count: int,
    ) -> tuple[bool, str]:
        if self.is_in_cooldown(now):
            return False, f"loss-streak cooldown ({self.cooldown_remaining_sec(now) / 60:.0f} min left)"
        allowed, reason = self.check_daily_limits(balance, now)
        if not allowed:
            return False, reason
        if open_position_count >= self.cfg.max_open_positions:
            return False, f"max open positions ({self.cfg.max_open_positions}) reached"
        return True, "ok"
