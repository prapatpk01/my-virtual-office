"""Binance / OKX / Bybit connector via CCXT (supports spot, cross-margin, paper trading)."""
import logging
import uuid
from typing import Optional

import ccxt.async_support as ccxt

from .base import OHLCV, Balance, BaseConnector, OrderResult

logger = logging.getLogger(__name__)


class BinanceConnector(BaseConnector):
    """
    Crypto connector for Binance / OKX / Bybit via CCXT.

    Margin mode (OKX only):
      margin_mode="cross"  → OKX Spot Margin, Cross, auto-borrow enabled
      leverage             → sets leverage on the pair before first order
    """

    TIMEFRAME_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
    }

    def __init__(self, api_key: str = "", api_secret: str = "",
                 paper: bool = True, exchange_id: str = "binance",
                 passphrase: str = "",
                 margin_mode: str = "",   # "cross" | "isolated" | "" (spot)
                 leverage: int = 1):
        super().__init__(api_key, api_secret, paper)
        self._exchange_id  = exchange_id
        self._margin_mode  = margin_mode.lower()  # "cross", "isolated", or ""
        self._leverage     = leverage

        # Choose defaultType based on margin mode
        if self._margin_mode in ("cross", "isolated"):
            default_type = "margin"
        else:
            default_type = "spot"

        options: dict = {"defaultType": default_type}
        if exchange_id == "bybit":
            options["fetchCurrencies"] = False

        cfg: dict = {
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options":         options,
        }
        if passphrase:
            cfg["password"] = passphrase

        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class(cfg)

        self._paper_balance = {"USDT": 10_000.0, "BTC": 0.0, "ETH": 0.0}
        self._paper_open_orders: list[OrderResult] = []
        self._leverage_set: set[str] = set()   # symbols where leverage already set

    # ──────────────────────────────────────────────────────────────────
    # Market data
    # ──────────────────────────────────────────────────────────────────

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h",
                          limit: int = 300) -> list[OHLCV]:
        tf = self.TIMEFRAME_MAP.get(timeframe, "1h")
        raw = await self._exchange.fetch_ohlcv(symbol, tf, limit=limit)
        return [OHLCV(ts, o, h, l, c, v) for ts, o, h, l, c, v in raw]

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(symbol)

    # ──────────────────────────────────────────────────────────────────
    # Leverage / margin setup (called once per symbol on first order)
    # ──────────────────────────────────────────────────────────────────

    async def _ensure_leverage(self, symbol: str):
        """Set cross-margin leverage on OKX (once per symbol per session)."""
        if self._exchange_id != "okx" or not self._margin_mode:
            return
        if symbol in self._leverage_set:
            return
        try:
            # OKX: set_leverage requires marginMode and side for isolated; cross needs no side
            await self._exchange.set_leverage(
                self._leverage, symbol,
                params={"mgnMode": self._margin_mode}
            )
            self._leverage_set.add(symbol)
            logger.info("OKX leverage set: %s × %dx cross-margin", symbol, self._leverage)
        except Exception as e:
            logger.warning("set_leverage failed (%s): %s — continuing", symbol, e)
            self._leverage_set.add(symbol)   # don't retry on every order

    # ──────────────────────────────────────────────────────────────────
    # Orders
    # ──────────────────────────────────────────────────────────────────

    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market",
                           price: Optional[float] = None,
                           tp_price: Optional[float] = None,
                           sl_price: Optional[float] = None) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price)

        await self._ensure_leverage(symbol)

        params: dict = {}
        if order_type == "limit" and price:
            params["price"] = price

        # OKX Spot Margin: tdMode + ccy required per order
        if self._exchange_id == "okx" and self._margin_mode:
            params["tdMode"] = self._margin_mode
            params["ccy"]    = symbol.split("/")[1] if "/" in symbol else "USDT"

            # Attach TP/SL inline (OKX API v5 — works for spot margin buy orders).
            # Uses "last" price as trigger; order executes at market (-1).
            # This keeps TP/SL active on OKX even if the bot goes offline.
            if side == "buy" and tp_price and sl_price:
                params["tpTriggerPx"]     = str(round(tp_price, 2))
                params["tpOrdPx"]         = "-1"       # market order on TP hit
                params["tpTriggerPxType"] = "last"
                params["slTriggerPx"]     = str(round(sl_price, 2))
                params["slOrdPx"]         = "-1"       # market order on SL hit
                params["slTriggerPxType"] = "last"
                logger.info(
                    "OKX order: %s %s %.6f  TP=%.2f  SL=%.2f",
                    side.upper(), symbol, amount, tp_price, sl_price,
                )

        raw = await self._exchange.create_order(
            symbol, order_type, side, amount, price if order_type == "limit" else None,
            params=params,
        )
        return OrderResult(
            order_id=str(raw.get("id", uuid.uuid4())),
            symbol=symbol, side=side, amount=amount,
            price=raw.get("price") or price or 0.0,
            filled=raw.get("filled", 0.0),
            status=raw.get("status", "open"),
        )

    async def _paper_order(self, symbol: str, side: str, amount: float,
                           order_type: str, price: Optional[float]) -> OrderResult:
        ticker     = await self.fetch_ticker(symbol)
        exec_price = price if (order_type == "limit" and price) else ticker["last"]
        base_asset = symbol.split("/")[0]
        quote_asset = symbol.split("/")[1] if "/" in symbol else "USDT"

        if side == "buy":
            cost = amount * exec_price
            if self._paper_balance.get(quote_asset, 0) < cost:
                raise ValueError(
                    f"[Paper] Insufficient {quote_asset}: "
                    f"need {cost:.2f}, have {self._paper_balance.get(quote_asset, 0):.2f}"
                )
            self._paper_balance[quote_asset] = self._paper_balance.get(quote_asset, 0) - cost
            self._paper_balance[base_asset]  = self._paper_balance.get(base_asset,  0) + amount
        else:
            if self._paper_balance.get(base_asset, 0) < amount:
                raise ValueError(
                    f"[Paper] Insufficient {base_asset}: "
                    f"need {amount}, have {self._paper_balance.get(base_asset, 0)}"
                )
            self._paper_balance[base_asset]  = self._paper_balance.get(base_asset,  0) - amount
            self._paper_balance[quote_asset] = self._paper_balance.get(quote_asset, 0) + amount * exec_price

        order = OrderResult(
            order_id=str(uuid.uuid4())[:8], symbol=symbol, side=side,
            amount=amount, price=exec_price, filled=amount, status="closed",
        )
        self._paper_open_orders.append(order)
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self.paper:
            self._paper_open_orders = [
                o for o in self._paper_open_orders if o.order_id != order_id
            ]
            return True
        await self._exchange.cancel_order(order_id, symbol)
        return True

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        if self.paper:
            return [o for o in self._paper_open_orders
                    if symbol is None or o.symbol == symbol]
        raw_list = await self._exchange.fetch_open_orders(symbol)
        return [OrderResult(
            order_id=str(r["id"]), symbol=r["symbol"], side=r["side"],
            amount=r["amount"], price=r.get("price") or 0.0,
            filled=r.get("filled", 0.0), status=r.get("status", "open"),
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
