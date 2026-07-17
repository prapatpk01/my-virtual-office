"""Exchange abstraction (spec §3, §37) — strategy logic never touches OKX
directly; live, paper and backtest all speak this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketRules:
    symbol: str
    contract_size: float = 1.0
    lot_step: float = 1.0            # in contracts
    min_qty: float = 1.0             # in contracts
    tick_size: float = 0.0
    min_notional: float = 0.0

    def as_dict(self) -> dict:
        return {"contract_size": self.contract_size, "lot_step": self.lot_step,
                "min_qty": self.min_qty, "tick_size": self.tick_size,
                "min_notional": self.min_notional}


@dataclass
class OrderResult:
    order_id: str
    client_order_id: str
    symbol: str
    side: str                        # buy | sell
    status: str                      # filled | open | rejected | canceled
    filled_qty: float = 0.0          # base units
    avg_price: float = 0.0
    fee_cost: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class PositionInfo:
    symbol: str
    direction: str                   # LONG | SHORT
    quantity: float                  # base units
    entry_price: float
    unrealized_pnl: float = 0.0
    attached_sl: Optional[float] = None
    attached_tp: Optional[float] = None


@dataclass
class ExchangeStateSnapshot:
    positions: list = field(default_factory=list)       # [PositionInfo]
    open_orders: list = field(default_factory=list)     # [OrderResult]
    equity: float = 0.0
    free_margin: float = 0.0
    last_price: float = 0.0
    spread_pct: Optional[float] = None
    clock_skew_sec: Optional[float] = None

    def position_for(self, symbol: str) -> Optional[PositionInfo]:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None


class ExchangeInterface(ABC):
    """Everything the engines need from a venue. Implementations: OKXExchange
    (live/paper via ccxt) and the backtest's SimulatedExchange."""

    def now_ms(self) -> int:
        """Current time for gating decisions — wall clock live, sim clock in
        backtest (so staleness/expiry checks replay correctly)."""
        import time as _t
        return int(_t.time() * 1000)

    @abstractmethod
    async def get_closed_candles(self, symbol: str, timeframe: str, limit: int) -> list: ...

    @abstractmethod
    async def get_market_rules(self, symbol: str) -> MarketRules: ...

    @abstractmethod
    async def get_state(self, symbol: str) -> ExchangeStateSnapshot: ...

    @abstractmethod
    async def get_all_open_positions(self) -> list: ...

    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, contracts: float,
                                 direction: str, client_order_id: str,
                                 sl_price: Optional[float] = None,
                                 tp_price: Optional[float] = None) -> OrderResult: ...

    @abstractmethod
    async def amend_protection(self, symbol: str, direction: str, quantity: float,
                               sl_price: Optional[float],
                               tp_price: Optional[float]) -> bool: ...

    @abstractmethod
    async def close_position(self, symbol: str, direction: str,
                             quantity: Optional[float] = None) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool: ...

    @abstractmethod
    async def find_order_by_client_id(self, symbol: str,
                                      client_order_id: str) -> Optional[OrderResult]: ...

    @abstractmethod
    async def close(self) -> None: ...
