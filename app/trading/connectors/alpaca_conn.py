"""Alpaca connector for Stocks & Crypto (paper trading supported natively)."""
import time
import uuid
from typing import Optional
import aiohttp

from .base import OHLCV, Balance, BaseConnector, OrderResult


class AlpacaConnector(BaseConnector):
    """
    Stocks / Crypto connector for Alpaca Markets.
    Alpaca has a built-in paper trading environment — set paper=True
    to use paper.alpaca.markets endpoints (default).
    """

    ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
    ALPACA_LIVE_BASE = "https://api.alpaca.markets"
    ALPACA_DATA_BASE = "https://data.alpaca.markets"

    TIMEFRAME_MAP = {
        "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
        "1h": "1Hour", "4h": "4Hour", "1d": "1Day",
    }

    def __init__(self, api_key: str = "", api_secret: str = "", paper: bool = True):
        super().__init__(api_key, api_secret, paper)
        self._base = self.ALPACA_PAPER_BASE if paper else self.ALPACA_LIVE_BASE
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json",
        }

    async def _get(self, url: str, params: Optional[dict] = None) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._headers, params=params) as r:
                r.raise_for_status()
                return await r.json()

    async def _post(self, url: str, data: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._headers, json=data) as r:
                r.raise_for_status()
                return await r.json()

    async def _delete(self, url: str) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=self._headers) as r:
                return r.status in (200, 204)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> list[OHLCV]:
        tf = self.TIMEFRAME_MAP.get(timeframe, "1Hour")
        url = f"{self.ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars"
        params = {"timeframe": tf, "limit": limit, "adjustment": "raw"}
        data = await self._get(url, params)
        bars = data.get("bars", [])
        result = []
        for b in bars:
            ts = int(b["t"][:19].replace("T", " ").replace("-", "").replace(":", "").replace(" ", "")) if isinstance(b["t"], str) else int(b["t"])
            result.append(OHLCV(
                timestamp=ts,
                open=b["o"], high=b["h"], low=b["l"], close=b["c"], volume=b["v"]
            ))
        return result

    async def fetch_ticker(self, symbol: str) -> dict:
        url = f"{self.ALPACA_DATA_BASE}/v2/stocks/{symbol}/quotes/latest"
        data = await self._get(url)
        quote = data.get("quote", {})
        mid = (quote.get("ap", 0) + quote.get("bp", 0)) / 2
        return {"symbol": symbol, "last": mid, "ask": quote.get("ap"), "bid": quote.get("bp")}

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market", price: Optional[float] = None) -> OrderResult:
        payload: dict = {
            "symbol": symbol,
            "qty": str(amount),
            "side": side,
            "type": order_type,
            "time_in_force": "gtc" if order_type == "limit" else "day",
        }
        if order_type == "limit" and price:
            payload["limit_price"] = str(price)

        url = f"{self._base}/v2/orders"
        raw = await self._post(url, payload)
        return OrderResult(
            order_id=raw["id"],
            symbol=symbol,
            side=side,
            amount=float(raw.get("qty", amount)),
            price=float(raw.get("limit_price") or raw.get("filled_avg_price") or price or 0),
            filled=float(raw.get("filled_qty", 0)),
            status=raw.get("status", "open"),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        url = f"{self._base}/v2/orders/{order_id}"
        return await self._delete(url)

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        url = f"{self._base}/v2/orders"
        params = {"status": "open"}
        if symbol:
            params["symbols"] = symbol
        data = await self._get(url, params)
        return [OrderResult(
            order_id=o["id"],
            symbol=o["symbol"],
            side=o["side"],
            amount=float(o.get("qty", 0)),
            price=float(o.get("limit_price") or 0),
            filled=float(o.get("filled_qty", 0)),
            status=o.get("status", "open"),
        ) for o in (data if isinstance(data, list) else [])]

    async def fetch_balance(self) -> list[Balance]:
        url = f"{self._base}/v2/account"
        data = await self._get(url)
        equity = float(data.get("equity", 0))
        cash = float(data.get("cash", 0))
        return [
            Balance(asset="USD", free=cash, used=equity - cash, total=equity),
        ]

    async def close(self):
        pass
