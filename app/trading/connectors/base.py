"""Base connector interface for all exchanges."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class OHLCV:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str          # 'buy' | 'sell'
    amount: float
    price: float
    filled: float
    status: str        # 'open' | 'closed' | 'canceled'
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class Balance:
    asset: str
    free: float
    used: float
    total: float


class BaseConnector(ABC):
    """Abstract base for exchange connectors."""

    def __init__(self, api_key: str = "", api_secret: str = "", paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper  # paper = simulation mode (no real orders)
        self._paper_balance: dict[str, float] = {}
        self._paper_orders: list[OrderResult] = []

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> list[OHLCV]:
        """Fetch candlestick data."""

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker price."""

    @abstractmethod
    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market", price: Optional[float] = None) -> OrderResult:
        """Place an order (or simulate if paper=True)."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""

    @abstractmethod
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        """Get all open orders."""

    @abstractmethod
    async def fetch_balance(self) -> list[Balance]:
        """Get account balances."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
