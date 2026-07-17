"""Market data facade — closed candles only, per-tick cache."""
from __future__ import annotations

from typing import Optional

from .config import Config
from .interfaces import ExchangeInterface


class MarketData:
    def __init__(self, cfg: Config, exchange: ExchangeInterface):
        self.cfg = cfg
        self.x = exchange
        self._cache: dict = {}

    def new_tick(self) -> None:
        self._cache.clear()

    async def get_closed_candles(self, symbol: str, timeframe: str, limit: int) -> list:
        key = (symbol, timeframe)
        if key in self._cache:
            return self._cache[key]
        candles = await self.x.get_closed_candles(symbol, timeframe, limit)
        self._cache[key] = candles
        return candles
