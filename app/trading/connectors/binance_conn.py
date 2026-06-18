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
                 paper: bool = True, exchange_id: str = "binance",
                 passphrase: str = "", leverage: int = 1,
                 margin_mode: str = "cross"):
        super().__init__(api_key, api_secret, paper)
        self.leverage     = max(1, int(leverage))
        self.margin_mode  = margin_mode  # "cross" | "isolated"
        exchange_class = getattr(ccxt, exchange_id)
        # OKX spot-margin uses defaultType="margin"; others stay "spot"
        default_type = "margin" if (exchange_id == "okx" and self.leverage > 1) else "spot"
        options: dict = {"defaultType": default_type}
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

    async def set_leverage_for(self, symbol: str) -> None:
        """Set OKX cross/isolated margin leverage. No-op for spot or paper."""
        if self.paper or self._exchange_id != "okx" or self.leverage <= 1:
            return
        try:
            await self._exchange.set_leverage(
                self.leverage, symbol,
                params={"mgnMode": self.margin_mode},
            )
        except Exception as e:
            import logging
            logging.getLogger("binance_conn").warning(
                "set_leverage %sx %s failed: %s", self.leverage, symbol, e)

    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market", price: Optional[float] = None,
                           tp: Optional[float] = None, sl: Optional[float] = None) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price)

        if self._exchange_id == "okx":
            return await self._okx_order(symbol, side, amount, order_type, price, tp, sl)

        # Non-OKX: standard CCXT path
        kwargs: dict = {}
        if order_type == "limit" and price:
            kwargs["price"] = price
        raw = await self._exchange.create_order(symbol, order_type, side, amount, **kwargs)
        exec_price = raw.get("average") or raw.get("price") or price or 0.0
        return OrderResult(
            order_id=str(raw.get("id", uuid.uuid4())),
            symbol=symbol, side=side, amount=amount,
            price=float(exec_price),
            filled=raw.get("filled") or raw.get("amount") or amount,
            status=raw.get("status", "closed"),
        )

    async def _okx_order(self, symbol: str, side: str, amount: float,
                         order_type: str, price: Optional[float],
                         tp: Optional[float], sl: Optional[float]) -> OrderResult:
        """OKX order placement using direct API calls to control exactly what is sent.

        Spot (leverage=1): no tdMode — Classic and Simple Unified accounts reject it (51000).
        Margin (leverage>1): tdMode=cross/isolated required.
        TP/SL placed as a separate OCO algo order after the main fill.
        """
        import logging
        _log = logging.getLogger("binance_conn")

        await self._exchange.load_markets()
        mkt  = self._exchange.markets.get(symbol) or {}
        info = mkt.get("info") or {}
        inst_id = info.get("instId") or symbol.replace("/", "-")
        sz_str  = self._exchange.amount_to_precision(symbol, amount)

        # Validate OKX minimum order size
        min_sz = float(info.get("minSz") or 0)
        if min_sz > 0 and float(sz_str or 0) < min_sz:
            px     = price or 60000.0
            needed = int(min_sz * px / max(self.leverage, 1)) + 1
            raise ValueError(
                f"Amount {amount:.6f} below OKX minimum {min_sz} BTC "
                f"(≈${min_sz * px:,.0f} notional). "
                f"Set TRADE_AMOUNT_USDT ≥ {needed} in Railway Variables."
            )

        # Main order — tdMode only for margin mode
        req: dict = {"instId": inst_id, "side": side, "ordType": order_type, "sz": sz_str}
        if self.leverage > 1:
            req["tdMode"] = self.margin_mode
            if side == "buy":
                req["ccy"] = "USDT"
        if order_type == "limit" and price:
            req["px"] = self._exchange.price_to_precision(symbol, price)

        resp     = await self._exchange.privatePostTradeOrder(req)
        data_row = ((resp or {}).get("data") or [{}])[0]
        ord_id   = data_row.get("ordId") or str(uuid.uuid4())

        try:
            ticker     = await self._exchange.fetch_ticker(symbol)
            exec_price = float(ticker.get("last") or price or 0)
        except Exception:
            exec_price = float(price or 0)

        result = OrderResult(
            order_id=ord_id, symbol=symbol, side=side, amount=amount,
            price=exec_price, filled=amount, status="closed",
        )

        # OCO algo order for TP/SL (separate from main order)
        if side == "buy" and tp and sl and tp > 0 and sl > 0:
            algo: dict = {
                "instId":         inst_id,
                "side":           "sell",
                "ordType":        "oco",
                "sz":             sz_str,
                "tpTriggerPx":    f"{tp:.2f}",
                "tpOrdPx":        "-1",
                "tpTriggerPxType":"last",
                "slTriggerPx":    f"{sl:.2f}",
                "slOrdPx":        "-1",
                "slTriggerPxType":"last",
            }
            if self.leverage > 1:
                algo["tdMode"] = self.margin_mode
            try:
                await self._exchange.privatePostTradeOrderAlgo(algo)
                _log.info("OCO algo placed: TP=%.2f SL=%.2f", tp, sl)
            except Exception as e:
                _log.warning("OCO algo failed (position open without TP/SL): %s", e)

        return result

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
        # raw["total"] = {asset: float}; per-asset dict is raw[asset]
        return [Balance(asset=k, free=float(v.get("free") or 0),
                        used=float(v.get("used") or 0), total=float(v.get("total") or 0))
                for k, v in raw.items()
                if isinstance(v, dict) and (v.get("total") or 0) > 0]

    async def get_open_position_symbols(self) -> set[str]:
        """Return set of symbols that have open positions on the exchange.

        Used by the bot to detect when OKX's native OCO has already closed a
        position so the bot can clean up its internal state without placing a
        redundant SELL order (which would open an unwanted short).

        Returns None for paper/non-OKX exchanges OR on API error — the caller
        treats None as "could not determine, skip sync". An empty set is only
        returned when the API call succeeds and genuinely reports zero open
        positions; that legitimately means everything closed on the exchange.

        CRITICAL: never return an empty set on failure. The bot interprets an
        empty set as "all positions closed" and would wipe every tracked
        position — a transient network/rate-limit error must not do that.
        """
        if self.paper or self._exchange_id != "okx" or self.leverage <= 1:
            return None
        try:
            # OKX margin positions require instType=MARGIN (not SWAP/FUTURES)
            raw = await self._exchange.fetch_positions(
                params={"instType": "MARGIN"}
            )
            open_syms: set[str] = set()
            for p in raw:
                # OKX margin: check notional or availSubPos; contracts may be 0 for margin mode
                notional = p.get("notional") or p.get("initialMargin") or 0
                contracts = p.get("contracts") or 0
                if float(notional) > 0 or float(contracts) > 0:
                    open_syms.add(p["symbol"])
            return open_syms
        except Exception as e:
            import logging
            logging.getLogger("binance_conn").warning(
                "fetch_positions failed (skipping sync): %s", e)
            return None

    async def close(self):
        await self._exchange.close()

