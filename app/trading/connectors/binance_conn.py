"""OKX/CCXT connector (named BinanceConnector for backward compatibility)."""
import logging
import uuid
from typing import Optional

import ccxt.async_support as ccxt

from .base import OHLCV, Balance, BaseConnector, OrderResult

_log = logging.getLogger("binance_conn")


class BinanceConnector(BaseConnector):
    """
    Crypto connector supporting OKX and other CCXT exchanges.
    Uses paper trading simulation by default.
    For OKX Unified Account: always sends tdMode=cross for all orders.
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
        self.leverage = max(1, int(leverage))
        self.margin_mode = margin_mode
        self._exchange_id = exchange_id

        exchange_class = getattr(ccxt, exchange_id)
        # OKX: use defaultType="margin" so market data (minSz, lotSz, precision) matches
        # the cross-margin instrument we trade via tdMode=cross. Using "spot" would give
        # the wrong minSz and cause OKX error 51020 on otherwise valid order sizes.
        options: dict = {"defaultType": "margin" if exchange_id == "okx" else "spot"}
        if exchange_id == "bybit":
            options["fetchCurrencies"] = False

        cfg: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": options,
        }
        if passphrase:
            cfg["password"] = passphrase

        self._exchange = exchange_class(cfg)
        self._paper_balance: dict = {"USDT": 10000.0, "BTC": 0.0, "ETH": 0.0}
        self._paper_open_orders: list[OrderResult] = []

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h",
                          limit: int = 200) -> list[OHLCV]:
        tf = self.TIMEFRAME_MAP.get(timeframe, "1h")
        raw = await self._exchange.fetch_ohlcv(symbol, tf, limit=limit)
        return [OHLCV(ts, o, h, l, c, v) for ts, o, h, l, c, v in raw]

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(symbol)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market",
                           price: Optional[float] = None,
                           tp: Optional[float] = None,
                           sl: Optional[float] = None) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price)

        if self._exchange_id == "okx":
            return await self._okx_order(symbol, side, amount, order_type, price, tp, sl)

        # Other exchanges: standard CCXT
        kwargs: dict = {}
        if order_type == "limit" and price:
            kwargs["price"] = price
        raw = await self._exchange.create_order(symbol, order_type, side, amount, **kwargs)
        exec_price = raw.get("average") or raw.get("price") or price or 0.0
        return OrderResult(
            order_id=str(raw.get("id", uuid.uuid4())),
            symbol=symbol, side=side, amount=amount,
            price=float(exec_price),
            filled=raw.get("filled") or amount,
            status=raw.get("status", "closed"),
        )

    async def _okx_order(self, symbol: str, side: str, amount: float,
                         order_type: str, price: Optional[float],
                         tp: Optional[float], sl: Optional[float]) -> OrderResult:
        """OKX order via direct API. Uses tdMode=cross for OKX Unified Account."""
        await self._exchange.load_markets()
        mkt = self._exchange.markets.get(symbol) or {}
        info = mkt.get("info") or {}
        inst_id = info.get("instId") or symbol.replace("/", "-")
        sz_str = self._exchange.amount_to_precision(symbol, amount)

        # Validate minimum order size
        min_sz = float(info.get("minSz") or 0)
        if min_sz > 0 and float(sz_str or 0) < min_sz:
            px = price or 60000.0
            needed = int(min_sz * px) + 1
            raise ValueError(
                f"Amount {amount:.6f} below OKX min {min_sz}. "
                f"Set TRADE_AMOUNT_USDT >= {needed}."
            )

        # Main order — tdMode=cross ALWAYS for OKX Unified Account
        req: dict = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": side,
            "ordType": order_type,
            "sz": sz_str,
        }
        if order_type == "limit" and price:
            req["px"] = self._exchange.price_to_precision(symbol, price)

        resp = await self._exchange.privatePostTradeOrder(req)
        data_row = ((resp or {}).get("data") or [{}])[0]
        ord_id = data_row.get("ordId") or str(uuid.uuid4())

        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            exec_price = float(ticker.get("last") or price or 0)
        except Exception:
            exec_price = float(price or 0)

        result = OrderResult(
            order_id=ord_id, symbol=symbol, side=side,
            amount=amount, price=exec_price, filled=amount, status="closed",
        )

        # OCO algo order for TP/SL after BUY fills
        if side == "buy" and tp and sl and tp > 0 and sl > 0:
            try:
                await self._exchange.privatePostTradeOrderAlgo({
                    "instId": inst_id,
                    "tdMode": "cross",
                    "side": "sell",
                    "ordType": "oco",
                    "sz": sz_str,
                    "tpTriggerPx": f"{tp:.2f}",
                    "tpOrdPx": "-1",
                    "tpTriggerPxType": "last",
                    "slTriggerPx": f"{sl:.2f}",
                    "slOrdPx": "-1",
                    "slTriggerPxType": "last",
                })
                _log.info("OCO placed: TP=%.2f SL=%.2f", tp, sl)
            except Exception as e:
                _log.warning("OCO failed: %s", e)

        return result

    async def _paper_order(self, symbol: str, side: str, amount: float,
                           order_type: str, price: Optional[float]) -> OrderResult:
        ticker = await self.fetch_ticker(symbol)
        exec_price = price if (order_type == "limit" and price) else ticker["last"]
        base = symbol.split("/")[0]
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"

        if side == "buy":
            cost = amount * exec_price
            if self._paper_balance.get(quote, 0) < cost:
                raise ValueError(
                    f"[Paper] Insufficient {quote}: need {cost:.2f}, "
                    f"have {self._paper_balance.get(quote, 0):.2f}"
                )
            self._paper_balance[quote] = self._paper_balance.get(quote, 0) - cost
            self._paper_balance[base] = self._paper_balance.get(base, 0) + amount
        else:
            if self._paper_balance.get(base, 0) < amount:
                raise ValueError(
                    f"[Paper] Insufficient {base}: need {amount}, "
                    f"have {self._paper_balance.get(base, 0)}"
                )
            self._paper_balance[base] = self._paper_balance.get(base, 0) - amount
            self._paper_balance[quote] = self._paper_balance.get(quote, 0) + amount * exec_price

        order = OrderResult(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side, amount=amount,
            price=exec_price, filled=amount, status="closed",
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
        return [
            OrderResult(
                order_id=str(r["id"]),
                symbol=r["symbol"], side=r["side"],
                amount=r["amount"], price=r.get("price") or 0.0,
                filled=r.get("filled", 0.0), status=r.get("status", "open"),
            )
            for r in raw_list
        ]

    async def fetch_balance(self) -> list[Balance]:
        if self.paper:
            return [
                Balance(asset=k, free=v, used=0.0, total=v)
                for k, v in self._paper_balance.items() if v > 0
            ]
        raw = await self._exchange.fetch_balance()
        return [
            Balance(
                asset=k, free=float(v.get("free") or 0),
                used=float(v.get("used") or 0), total=float(v.get("total") or 0),
            )
            for k, v in raw.items()
            if isinstance(v, dict) and (v.get("total") or 0) > 0
        ]

    async def get_open_position_symbols(self) -> Optional[set]:
        """Return set of symbols with open cross-margin positions on OKX, or None on error.

        Returns None for paper mode, non-OKX, or on API error.
        An empty set means OKX confirmed zero open positions.
        Never return empty set on failure — caller treats empty set as "all closed".
        """
        if self.paper or self._exchange_id != "okx":
            return None
        try:
            raw = await self._exchange.fetch_positions(params={"instType": "MARGIN"})
            syms: set = set()
            for p in raw:
                notional = p.get("notional") or p.get("initialMargin") or 0
                if float(notional) > 0:
                    syms.add(p["symbol"])
            return syms
        except Exception as e:
            _log.warning("fetch_positions failed: %s", e)
            return None

    async def close(self):
        await self._exchange.close()
