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


def to_heikin_ashi(candles: list["OHLCV"]) -> list["OHLCV"]:
    """Convert standard OHLCV candles to Heikin Ashi.

    HA_Close = (O + H + L + C) / 4
    HA_Open  = (prev_HA_Open + prev_HA_Close) / 2
    HA_High  = max(H, HA_Open, HA_Close)
    HA_Low   = min(L, HA_Open, HA_Close)

    Volume and timestamp are preserved unchanged.
    """
    if not candles:
        return candles
    result: list[OHLCV] = []
    ha_open = (candles[0].open + candles[0].close) / 2.0
    for c in candles:
        ha_close = (c.open + c.high + c.low + c.close) / 4.0
        ha_high  = max(c.high, ha_open, ha_close)
        ha_low   = min(c.low,  ha_open, ha_close)
        result.append(OHLCV(
            timestamp=c.timestamp,
            open=ha_open,
            high=ha_high,
            low=ha_low,
            close=ha_close,
            volume=c.volume,
        ))
        ha_open = (ha_open + ha_close) / 2.0
    return result


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
                           order_type: str = "market", price: Optional[float] = None,
                           position_side: str = "LONG") -> OrderResult:
        """Place an order (or simulate if paper=True).
        position_side: 'LONG' | 'SHORT' — for futures hedge mode (OKX posSide).
        """

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
