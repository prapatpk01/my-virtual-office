"""
Risk Manager — position sizing, daily loss/profit limits, loss-streak cooldown.

Time is injected (never read from the wall clock internally by the sizing
math) so the exact same class can be used in the backtest with simulated
timestamps.
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

    # ── Daily accounting ─────────────────────────────────────────────────────

    def _roll_day(self, balance: float, now: float):
        today = _day_key(now)
        if today != self.state.day or self.state.day_start_balance <= 0:
            self.state.day = today
            self.state.day_start_balance = balance
            self.state.day_realized_pnl = 0.0

    def register_trade_result(self, pnl: float, balance_after: float, now: float):
        self._roll_day(balance_after - pnl, now)
        self.state.day_realized_pnl += pnl
        if balance_after > self.state.peak_balance:
            self.state.peak_balance = balance_after

        if pnl < 0:
            self.state.loss_streak += 1
            if self.state.loss_streak >= self.cfg.loss_streak_limit:
                self.state.cooldown_until = now + self.cfg.loss_streak_cooldown_min * 60
                logger.warning("[RISK] Loss streak %d reached — cooldown %d min",
                               self.state.loss_streak, self.cfg.loss_streak_cooldown_min)
        else:
            self.state.loss_streak = 0

    def is_in_cooldown(self, now: float) -> bool:
        return now < self.state.cooldown_until

    def cooldown_remaining_sec(self, now: float) -> float:
        return max(0.0, self.state.cooldown_until - now)

    def check_daily_limits(self, balance: float, now: float) -> tuple[bool, str]:
        """Returns (allowed_to_open_new, reason). Open positions keep running either way."""
        self._roll_day(balance, now)
        if self.state.day_start_balance <= 0:
            return True, "ok"
        pnl_pct = self.state.day_realized_pnl / self.state.day_start_balance
        if pnl_pct <= -self.cfg.daily_loss_limit_pct:
            return False, f"Daily loss limit hit: {pnl_pct*100:.1f}% <= -{self.cfg.daily_loss_limit_pct*100:.0f}%"
        if pnl_pct >= self.cfg.daily_profit_lock_pct:
            return False, f"Daily profit lock hit: {pnl_pct*100:.1f}% >= +{self.cfg.daily_profit_lock_pct*100:.0f}%"
        return True, "ok"

    # ── Position sizing ───────────────────────────────────────────────────────

    def size_by_risk(self, balance: float, entry_price: float, sl_price: float,
                     regime_size_multiplier: float = 1.0) -> float:
        """
        notional such that (notional * sl_dist_pct) == balance * risk_per_trade,
        scaled by regime_size_multiplier (e.g. 0.7 in HIGH_VOLATILITY),
        capped at 95% of balance * leverage.
        """
        if entry_price <= 0 or sl_price <= 0 or balance <= 0:
            return 0.0
        sl_dist_pct = abs(entry_price - sl_price) / entry_price
        if sl_dist_pct <= 0:
            return 0.0
        risk_amount = balance * self.cfg.risk_per_trade * max(0.0, min(1.0, regime_size_multiplier))
        notional = risk_amount / sl_dist_pct
        max_notional = balance * 0.95 * self.cfg.leverage
        notional = min(notional, max_notional)
        return round(notional / entry_price, 8)

    def can_open_new(self, balance: float, now: float, open_position_count: int) -> tuple[bool, str]:
        if self.is_in_cooldown(now):
            return False, f"loss-streak cooldown ({self.cooldown_remaining_sec(now)/60:.0f} min left)"
        allowed, reason = self.check_daily_limits(balance, now)
        if not allowed:
            return False, reason
        if open_position_count >= self.cfg.max_open_positions:
            return False, f"max open positions ({self.cfg.max_open_positions}) reached"
        return True, "ok"
