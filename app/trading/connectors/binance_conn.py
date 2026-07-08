"""Binance/OKX/Bybit connector via CCXT (supports spot, futures, paper trading)."""
import asyncio
import logging
import time
import uuid
from typing import Optional

import ccxt.async_support as ccxt

from .base import OHLCV, Balance, BaseConnector, OrderResult

logger = logging.getLogger("binance_conn")


class BinanceConnector(BaseConnector):
    """
    Crypto connector for Binance, OKX, Bybit (any CCXT-compatible exchange).

    Futures mode (futures=True):
      - Uses defaultType="swap" for perpetual contracts
      - Symbols must use the perpetual format: BTC/USDT:USDT (OKX), BTC/USDT (Binance perp)
      - Calls setup_futures(symbol) before the first order to set leverage + hedge mode
      - Passes posSide='long'/'short' in hedge mode orders (required by OKX)

    Paper mode: simulates order fills at last market price; no real API calls for orders.
    """

    TIMEFRAME_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
    }

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        paper: bool = True,
        exchange_id: str = "binance",
        passphrase: str = "",
        futures: bool = False,
        leverage: int = 1,
        hedge_mode: bool = False,
    ):
        # Safety guard: never attempt live trading without credentials
        if not paper and (not api_key or not api_secret):
            logger.warning(
                "Live mode requested for %s but API credentials are missing — "
                "switching to paper mode automatically.", exchange_id,
            )
            paper = True
        super().__init__(api_key, api_secret, paper)
        exchange_class = getattr(ccxt, exchange_id)

        if futures:
            options: dict = {"defaultType": "swap"}
        else:
            options: dict = {"defaultType": "spot"}

        if exchange_id == "bybit":
            options["fetchCurrencies"] = False

        cfg: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": options,
        }
        if passphrase:  # OKX requires password (passphrase)
            cfg["password"] = passphrase

        self._exchange = exchange_class(cfg)
        self._exchange_id = exchange_id
        self._futures = futures
        self._leverage = leverage
        self._hedge_mode = hedge_mode
        self._futures_setup_symbols: set[str] = set()  # symbols already configured

        self._paper_balance = {"USDT": 10000.0, "BTC": 0.0, "ETH": 0.0}
        self._paper_open_orders: list[OrderResult] = []

    # ------------------------------------------------------------------
    # Futures setup (called once per symbol before first live order)
    # ------------------------------------------------------------------

    async def setup_futures(self, symbol: str) -> None:
        """Set leverage and hedge mode for a symbol. No-op in paper mode or if already done."""
        if self.paper or symbol in self._futures_setup_symbols:
            return
        try:
            await self._exchange.set_leverage(self._leverage, symbol)
            logger.info("[Futures] Leverage set to %dx for %s", self._leverage, symbol)
        except Exception as e:
            logger.warning("[Futures] set_leverage failed for %s: %s", symbol, e)
        if self._hedge_mode:
            try:
                # OKX / Binance: True = hedge (dual-side), False = net (one-way)
                await self._exchange.set_position_mode(True, symbol)
                logger.info("[Futures] Hedge mode enabled for %s", symbol)
            except Exception as e:
                logger.warning("[Futures] set_position_mode failed for %s: %s", symbol, e)
        self._futures_setup_symbols.add(symbol)

    # ------------------------------------------------------------------
    # Market data (always live; no API key needed for public endpoints)
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

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        pos_side: Optional[str] = None,
    ) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price, pos_side)

        kwargs: dict = {}
        if order_type == "limit" and price:
            kwargs["price"] = price

        # OKX hedge mode requires posSide in every order
        if pos_side and self._hedge_mode:
            kwargs["params"] = {"posSide": pos_side}

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

    async def _paper_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str,
        price: Optional[float],
        pos_side: Optional[str] = None,
    ) -> OrderResult:
        ticker = await self.fetch_ticker(symbol)
        exec_price = price if (order_type == "limit" and price) else ticker["last"]

        if self._futures:
            # Futures paper: deduct margin from USDT (notional / leverage)
            margin = round((amount * exec_price) / max(self._leverage, 1), 4)
            if side == "buy" and pos_side in ("long", None):
                if self._paper_balance.get("USDT", 0) < margin:
                    raise ValueError(
                        f"[Paper/Futures] Insufficient USDT: need {margin:.2f}, "
                        f"have {self._paper_balance.get('USDT', 0):.2f}"
                    )
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) - margin
            elif side == "sell" and pos_side in ("short", None):
                if self._paper_balance.get("USDT", 0) < margin:
                    raise ValueError(
                        f"[Paper/Futures] Insufficient USDT: need {margin:.2f}, "
                        f"have {self._paper_balance.get('USDT', 0):.2f}"
                    )
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) - margin
            else:
                # Closing a position: return margin to balance (approximate)
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + margin
        else:
            # Spot trading
            base_asset  = symbol.split("/")[0]
            quote_asset = symbol.split("/")[1].split(":")[0] if "/" in symbol else "USDT"
            if side == "buy":
                cost = amount * exec_price
                if self._paper_balance.get(quote_asset, 0) < cost:
                    raise ValueError(
                        f"[Paper] Insufficient {quote_asset}: need {cost:.2f}, "
                        f"have {self._paper_balance.get(quote_asset, 0):.2f}"
                    )
                self._paper_balance[quote_asset] = self._paper_balance.get(quote_asset, 0) - cost
                self._paper_balance[base_asset]  = self._paper_balance.get(base_asset, 0) + amount
            else:
                if self._paper_balance.get(base_asset, 0) < amount:
                    raise ValueError(
                        f"[Paper] Insufficient {base_asset}: need {amount}, "
                        f"have {self._paper_balance.get(base_asset, 0)}"
                    )
                self._paper_balance[base_asset]  = self._paper_balance.get(base_asset, 0) - amount
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
        self._paper_open_orders.append(order)
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
        # ccxt unified format: raw["total"]/["free"]/["used"] are {asset: float},
        # while raw[asset] (top-level) is the {"free","used","total"} dict per asset.
        skip = {"info", "free", "used", "total", "timestamp", "datetime"}
        return [
            Balance(asset=k, free=v.get("free", 0.0), used=v.get("used", 0.0), total=v.get("total", 0.0))
            for k, v in raw.items()
            if k not in skip and isinstance(v, dict) and v.get("total", 0) > 0
        ]

    async def close(self):
        await self._exchange.close()
