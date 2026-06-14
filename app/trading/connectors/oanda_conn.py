"""
OANDA v20 REST API connector.

Supports: BTC_USD, XAU_USD, EUR_USD and all OANDA instruments.

Environment variables:
  OANDA_API_KEY     — personal access token (from OANDA account settings)
  OANDA_ACCOUNT_ID  — account ID  e.g. 001-001-1234567-001
  OANDA_ENV         — "practice" (default) or "live"

paper=True  → candle/ticker data fetched from OANDA, orders simulated locally
paper=False → full order execution via OANDA API (practice or live)
"""
import logging
import time
import uuid
from typing import Optional

import aiohttp

from .base import OHLCV, Balance, BaseConnector, OrderResult

logger = logging.getLogger("oanda_conn")

_BASE = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}

_TF_MAP = {
    "1m":  "M1",  "5m":  "M5",  "15m": "M15", "30m": "M30",
    "1h":  "H1",  "4h":  "H4",  "1d":  "D",   "1w":  "W",
}


def _to_oanda_symbol(symbol: str) -> str:
    """BTC/USD or BTCUSD or BTC_USD → BTC_USD"""
    return symbol.replace("/", "_").replace("-", "_").upper()


def _from_oanda_symbol(instrument: str) -> str:
    """BTC_USD → BTC/USD"""
    parts = instrument.split("_")
    return "/".join(parts) if len(parts) == 2 else instrument


class OANDAConnector(BaseConnector):

    def __init__(self, api_key: str = "", account_id: str = "",
                 paper: bool = True, env: str = "practice"):
        super().__init__(api_key=api_key, paper=paper)
        self.account_id  = account_id
        self._env        = env if env in ("practice", "live") else "practice"
        self._base_url   = _BASE[self._env]
        self._headers    = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        self._paper_balance: dict[str, float] = {"USD": 10_000.0}

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self._base_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._headers, params=params,
                                   timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    body = await r.text()
                    raise RuntimeError(f"OANDA GET {path} → {r.status}: {body[:200]}")
                return await r.json()

    async def _post(self, path: str, body: dict) -> dict:
        url = f"{self._base_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._headers, json=body,
                                    timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()
                if r.status not in (200, 201):
                    raise RuntimeError(f"OANDA POST {path} → {r.status}: {data}")
                return data

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "15m",
                          limit: int = 300) -> list[OHLCV]:
        instrument = _to_oanda_symbol(symbol)
        gran = _TF_MAP.get(timeframe, "M15")
        data = await self._get(
            f"/v3/instruments/{instrument}/candles",
            params={"count": str(limit), "granularity": gran, "price": "M"},
        )
        candles = []
        for c in data.get("candles", []):
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            ts  = int(time.mktime(
                time.strptime(c["time"][:19], "%Y-%m-%dT%H:%M:%S")
            ) * 1000)
            candles.append(OHLCV(
                timestamp=ts,
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=float(c.get("volume", 0)),
            ))
        return candles

    async def fetch_ticker(self, symbol: str) -> dict:
        instrument = _to_oanda_symbol(symbol)
        data = await self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": instrument},
        )
        prices = data.get("prices", [])
        if not prices:
            raise RuntimeError(f"No pricing data for {instrument}")
        p    = prices[0]
        bid  = float(p["bids"][0]["price"])
        ask  = float(p["asks"][0]["price"])
        mid  = (bid + ask) / 2.0
        return {"last": mid, "bid": bid, "ask": ask, "symbol": symbol}

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market",
                           price: Optional[float] = None) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount)

        instrument = _to_oanda_symbol(symbol)
        # OANDA: positive units = buy, negative = sell
        units = str(int(amount)) if side == "buy" else str(-int(amount))
        body = {"order": {"type": "MARKET", "instrument": instrument, "units": units}}

        data  = await self._post(f"/v3/accounts/{self.account_id}/orders", body)
        fill  = data.get("orderFillTransaction", {})
        exec_price = float(fill.get("price", 0)) or price or 0.0
        filled     = abs(float(fill.get("units", amount)))

        return OrderResult(
            order_id=str(fill.get("id", uuid.uuid4())),
            symbol=symbol, side=side,
            amount=amount, price=exec_price,
            filled=filled, status="closed",
        )

    async def _paper_order(self, symbol: str, side: str, amount: float) -> OrderResult:
        ticker = await self.fetch_ticker(symbol)
        price  = ticker["ask"] if side == "buy" else ticker["bid"]
        cost   = amount * price
        if side == "buy":
            if self._paper_balance.get("USD", 0) < cost:
                raise ValueError(f"[Paper] Insufficient USD: need {cost:.2f}")
            self._paper_balance["USD"] = self._paper_balance.get("USD", 0) - cost
        else:
            self._paper_balance["USD"] = self._paper_balance.get("USD", 0) + cost
        return OrderResult(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side,
            amount=amount, price=price,
            filled=amount, status="closed",
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self.paper:
            return True
        instrument = _to_oanda_symbol(symbol)
        await self._get(f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel")
        return True

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        if self.paper:
            return []
        params = {}
        if symbol:
            params["instrument"] = _to_oanda_symbol(symbol)
        data = await self._get(f"/v3/accounts/{self.account_id}/orders", params)
        return [
            OrderResult(
                order_id=str(o["id"]),
                symbol=_from_oanda_symbol(o["instrument"]),
                side="buy" if float(o.get("units", 0)) > 0 else "sell",
                amount=abs(float(o.get("units", 0))),
                price=float(o.get("price", 0)),
                filled=0.0, status="open",
            )
            for o in data.get("orders", [])
        ]

    async def fetch_balance(self) -> list[Balance]:
        if self.paper:
            return [Balance(asset="USD",
                            free=self._paper_balance.get("USD", 0),
                            used=0.0,
                            total=self._paper_balance.get("USD", 0))]
        data = await self._get(f"/v3/accounts/{self.account_id}/summary")
        acct = data.get("account", {})
        bal  = float(acct.get("balance", 0))
        nav  = float(acct.get("NAV", bal))
        return [Balance(asset="USD", free=bal, used=nav - bal, total=nav)]

    async def close(self):
        pass  # aiohttp sessions are per-request, nothing to close
