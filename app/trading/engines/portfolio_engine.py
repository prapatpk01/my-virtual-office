"""
Module 9: Portfolio Engine

Controls risk across multiple assets:
  - Correlation exposure limits (avoid over-concentration)
  - Total portfolio heat (% at risk across all open positions)
  - Max drawdown at portfolio level
  - Sector / asset class exposure caps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PortfolioPosition:
    symbol:       str
    direction:    str
    entry_price:  float
    current_price:float
    amount:       float
    stop_loss:    float
    notional:     float   # amount * entry_price
    risk_usd:     float   # abs(entry - stop) * amount


@dataclass
class PortfolioCheck:
    can_add:          bool
    reason:           str
    portfolio_heat:   float   # % of portfolio at risk
    correlation_risk: float   # 0-1
    open_positions:   int
    suggestions:      list = field(default_factory=list)


# Known correlation groups (heuristic)
_CORRELATION_GROUPS: dict[str, str] = {
    "BTC":  "crypto_major",  "ETH":   "crypto_major",
    "SOL":  "crypto_major",  "BNB":   "crypto_major",
    "AVAX": "crypto_major",  "MATIC": "crypto_alt",
    "LINK": "crypto_alt",    "DOGE":  "crypto_meme",
    "SHIB": "crypto_meme",   "XAU":   "commodities",
    "GOLD": "commodities",   "SILVER":"commodities",
    "EUR":  "forex_major",   "GBP":   "forex_major",
    "JPY":  "forex_major",
}


class PortfolioEngine:
    """Manages portfolio-level risk constraints."""

    def __init__(
        self,
        max_portfolio_heat:    float = 0.06,  # Max 6% of portfolio at risk at once
        max_same_group:        int   = 2,     # Max 2 positions in same correlation group
        max_total_positions:   int   = 5,
        max_drawdown_pct:      float = 0.15,  # 15% portfolio drawdown limit
        peak_balance:          float = 0.0,
    ):
        self.max_heat       = max_portfolio_heat
        self.max_same_group = max_same_group
        self.max_positions  = max_total_positions
        self.max_drawdown   = max_drawdown_pct
        self._peak_balance  = peak_balance
        self._positions: Dict[str, PortfolioPosition] = {}

    def can_add_position(
        self,
        symbol:          str,
        direction:       str,
        entry_price:     float,
        stop_loss:       float,
        amount:          float,
        portfolio_value: float,
    ) -> PortfolioCheck:
        """Check if adding this position violates any portfolio constraint."""
        risk_usd = abs(entry_price - stop_loss) * amount
        notional = entry_price * amount

        # ── Count existing positions ─────────────────────────────────────────
        if len(self._positions) >= self.max_positions:
            return PortfolioCheck(
                can_add=False,
                reason=f"Max positions ({self.max_positions}) reached",
                portfolio_heat=self._current_heat(portfolio_value),
                correlation_risk=self._correlation_risk(symbol),
                open_positions=len(self._positions),
            )

        # ── Portfolio heat check ───────────────────────────────────────────
        current_heat     = self._current_heat(portfolio_value)
        new_risk_pct     = risk_usd / portfolio_value if portfolio_value > 0 else 0
        projected_heat   = current_heat + new_risk_pct
        if projected_heat > self.max_heat:
            return PortfolioCheck(
                can_add=False,
                reason=f"Portfolio heat {projected_heat*100:.1f}% would exceed {self.max_heat*100:.0f}%",
                portfolio_heat=current_heat,
                correlation_risk=self._correlation_risk(symbol),
                open_positions=len(self._positions),
                suggestions=["Reduce position size", "Close a position first"],
            )

        # ── Correlation exposure ────────────────────────────────────────────
        base = symbol.split("/")[0].upper()
        group= _CORRELATION_GROUPS.get(base, "other")
        same_group = sum(
            1 for sym in self._positions
            if _CORRELATION_GROUPS.get(sym.split("/")[0].upper(), "other") == group
        )
        if same_group >= self.max_same_group:
            return PortfolioCheck(
                can_add=False,
                reason=f"Correlation group '{group}' already has {same_group}/{self.max_same_group} positions",
                portfolio_heat=current_heat,
                correlation_risk=self._correlation_risk(symbol),
                open_positions=len(self._positions),
                suggestions=[f"Avoid adding more {group} exposure"],
            )

        return PortfolioCheck(
            can_add=True,
            reason="All portfolio checks passed",
            portfolio_heat=round(projected_heat, 4),
            correlation_risk=round(self._correlation_risk(symbol), 3),
            open_positions=len(self._positions),
        )

    def add_position(self, symbol: str, direction: str,
                     entry_price: float, current_price: float,
                     amount: float, stop_loss: float) -> None:
        self._positions[symbol] = PortfolioPosition(
            symbol=symbol, direction=direction,
            entry_price=entry_price, current_price=current_price,
            amount=amount, stop_loss=stop_loss,
            notional=entry_price * amount,
            risk_usd=abs(entry_price - stop_loss) * amount,
        )
        self._peak_balance = max(self._peak_balance, entry_price * amount)

    def remove_position(self, symbol: str) -> None:
        self._positions.pop(symbol, None)

    def update_prices(self, prices: dict[str, float]) -> None:
        for sym, price in prices.items():
            if sym in self._positions:
                self._positions[sym].current_price = price

    def portfolio_summary(self, portfolio_value: float) -> dict:
        total_pnl = sum(
            (p.current_price - p.entry_price) * p.amount * (1 if p.direction == "long" else -1)
            for p in self._positions.values()
        )
        return {
            "open_positions":  len(self._positions),
            "portfolio_heat":  round(self._current_heat(portfolio_value), 4),
            "total_unrealized_pnl": round(total_pnl, 4),
            "symbols":         list(self._positions.keys()),
            "max_drawdown_hit": self._check_drawdown(portfolio_value),
        }

    def _current_heat(self, portfolio_value: float) -> float:
        if portfolio_value <= 0:
            return 0.0
        total_risk = sum(p.risk_usd for p in self._positions.values())
        return total_risk / portfolio_value

    def _correlation_risk(self, new_symbol: str) -> float:
        base  = new_symbol.split("/")[0].upper()
        group = _CORRELATION_GROUPS.get(base, "other")
        same  = sum(1 for sym in self._positions
                    if _CORRELATION_GROUPS.get(sym.split("/")[0].upper(), "other") == group)
        return same / self.max_positions

    def _check_drawdown(self, portfolio_value: float) -> bool:
        if self._peak_balance <= 0:
            return False
        drawdown = (self._peak_balance - portfolio_value) / self._peak_balance
        return drawdown >= self.max_drawdown
