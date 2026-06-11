"""Binance connector via CCXT (supports paper trading simulation)."""
import asyncio
import time
import uuid
from typing import Optional

import ccxt.async_support as ccxt

from .base import OHLCV, Balance, BaseConnector, OrderResult


class BinanceConnector(BaseConnector):
    """
    Crypto connector for Binance (or any CCXT-compatible exchange).
    Set paper=True (default) to simulate trades without real API keys.
    """

    TIMEFRAME_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
    }

    def __init__(self, api_key: str = "", api_secret: str = "",
                 paper: bool = True, exchange_id: str = "binance"):
        super().__init__(api_key, api_secret, paper)
        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self._exchange_id = exchange_id
        self._paper_balance = {"USDT": 10000.0, "BTC": 0.0, "ETH": 0.0}
        self._paper_open_orders: list[OrderResult] = []

    # ------------------------------------------------------------------
    # Market data (always live, no API key needed for public endpoints)
    # ------------------------------------------------------------------

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> list[OHLCV]:
        tf = self.TIMEFRAME_MAP.get(timeframe, "1h")
        raw = await self._exchange.fetch_ohlcv(symbol, tf, limit=limit)
        return [OHLCV(ts, o, h, l, c, v) for ts, o, h, l, c, v in raw]

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(symbol)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market", price: Optional[float] = None) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price)

        kwargs = {}
        if order_type == "limit" and price:
            kwargs["price"] = price

        raw = await self._exchange.create_order(symbol, order_type, side, amount, **kwargs)
        return OrderResult(
            order_id=str(raw.get("id", uuid.uuid4())),
            symbol=symbol,
            side=side,
            amount=amount,
            price=raw.get("price") or price or 0.0,
            filled=raw.get("filled", 0.0),
            status=raw.get("status", "open"),
        )

    async def _paper_order(self, symbol: str, side: str, amount: float,
                           order_type: str, price: Optional[float]) -> OrderResult:
        ticker = await self.fetch_ticker(symbol)
        exec_price = price if (order_type == "limit" and price) else ticker["last"]
        base_asset = symbol.split("/")[0]
        quote_asset = symbol.split("/")[1] if "/" in symbol else "USDT"

        if side == "buy":
            cost = amount * exec_price
            if self._paper_balance.get(quote_asset, 0) < cost:
                raise ValueError(f"[Paper] Insufficient {quote_asset}: need {cost:.2f}, have {self._paper_balance.get(quote_asset, 0):.2f}")
            self._paper_balance[quote_asset] = self._paper_balance.get(quote_asset, 0) - cost
            self._paper_balance[base_asset] = self._paper_balance.get(base_asset, 0) + amount
        else:
            if self._paper_balance.get(base_asset, 0) < amount:
                raise ValueError(f"[Paper] Insufficient {base_asset}: need {amount}, have {self._paper_balance.get(base_asset, 0)}")
            self._paper_balance[base_asset] = self._paper_balance.get(base_asset, 0) - amount
            self._paper_balance[quote_asset] = self._paper_balance.get(quote_asset, 0) + amount * exec_price

        order = OrderResult(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            amount=amount,
            price=exec_price,
            filled=amount,
            status="closed",
        )
        self._paper_orders.append(order)
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self.paper:
            self._paper_open_orders = [o for o in self._paper_open_orders if o.order_id != order_id]
            return True
        await self._exchange.cancel_order(order_id, symbol)
        return True

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        if self.paper:
            return [o for o in self._paper_open_orders
                    if symbol is None or o.symbol == symbol]
        raw_list = await self._exchange.fetch_open_orders(symbol)
        return [OrderResult(
            order_id=str(r["id"]),
            symbol=r["symbol"],
            side=r["side"],
            amount=r["amount"],
            price=r.get("price") or 0.0,
            filled=r.get("filled", 0.0),
            status=r.get("status", "open"),
        ) for r in raw_list]

    async def fetch_balance(self) -> list[Balance]:
        if self.paper:
            return [Balance(asset=k, free=v, used=0.0, total=v)
                    for k, v in self._paper_balance.items() if v > 0]
        raw = await self._exchange.fetch_balance()
        return [Balance(asset=k, free=v["free"], used=v["used"], total=v["total"])
                for k, v in raw["total"].items()
                if isinstance(raw.get(k), dict) and raw[k].get("total", 0) > 0]

    async def close(self):
        await self._exchange.close()
