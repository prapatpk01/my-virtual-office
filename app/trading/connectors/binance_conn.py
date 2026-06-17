"""Binance / OKX / Bybit connector via CCXT (supports spot, cross-margin, futures/swap)."""
import logging
import os
import uuid
from typing import Optional

import ccxt.async_support as ccxt

from .base import OHLCV, Balance, BaseConnector, OrderResult

logger = logging.getLogger(__name__)


class BinanceConnector(BaseConnector):
    """
    Crypto connector for Binance / OKX / Bybit via CCXT.

    Margin / market modes (OKX):
      margin_mode="cross"  → OKX Spot Cross Margin, auto-borrow
      market_type="swap"   → OKX Perpetual Futures (long & short enabled)
      leverage             → leverage ratio set before first order
    """

    TIMEFRAME_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
    }

    def __init__(self, api_key: str = "", api_secret: str = "",
                 paper: bool = True, exchange_id: str = "binance",
                 passphrase: str = "",
                 margin_mode: str = "",    # "cross" | "isolated" | "" (spot)
                 market_type: str = "",    # "swap" | "futures" | "" (spot/margin)
                 leverage: int = 1):
        super().__init__(api_key, api_secret, paper)
        self._exchange_id  = exchange_id
        self._margin_mode  = margin_mode.lower()
        self._market_type  = market_type.lower()
        self._leverage     = leverage
        self._futures      = self._market_type in ("swap", "futures")

        # defaultType drives which OKX endpoint CCXT hits
        if self._futures:
            default_type = "swap"
        elif self._margin_mode in ("cross", "isolated"):
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

        _bal = float(os.environ.get("PAPER_BALANCE", "10000"))
        if _bal <= 0:
            _bal = 10000.0
        self._paper_balance = {"USDT": _bal, "BTC": 0.0, "ETH": 0.0}
        self._paper_open_orders: list[OrderResult] = []
        self._leverage_set: set[str] = set()
        # Futures paper: track open contracts {symbol: {"side": "long"|"short", "amount": float, "entry": float}}
        self._paper_futures: dict[str, dict] = {}

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
        """Set leverage on OKX (once per symbol per session)."""
        if self._exchange_id != "okx":
            return
        if symbol in self._leverage_set:
            return
        try:
            if self._futures:
                # OKX hedge mode: set leverage for both sides separately
                for ps in ("long", "short"):
                    try:
                        await self._exchange.set_leverage(
                            self._leverage, symbol,
                            params={"mgnMode": "cross", "posSide": ps},
                        )
                    except Exception:
                        pass
                logger.info("OKX futures leverage: %s × %dx (long+short)", symbol, self._leverage)
            elif self._margin_mode:
                await self._exchange.set_leverage(
                    self._leverage, symbol,
                    params={"mgnMode": self._margin_mode},
                )
                logger.info("OKX margin leverage: %s × %dx %s", symbol, self._leverage, self._margin_mode)
        except Exception as e:
            logger.warning("set_leverage failed (%s): %s — continuing", symbol, e)
        self._leverage_set.add(symbol)

    # ──────────────────────────────────────────────────────────────────
    # Orders
    # ──────────────────────────────────────────────────────────────────

    async def create_order(self, symbol: str, side: str, amount: float,
                           order_type: str = "market",
                           price: Optional[float] = None,
                           tp_price: Optional[float] = None,
                           sl_price: Optional[float] = None,
                           pos_side: str = "") -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price, pos_side)

        await self._ensure_leverage(symbol)

        params: dict = {}
        if order_type == "limit" and price:
            params["price"] = price

        if self._exchange_id == "okx":
            if self._futures:
                # OKX Perpetual Futures: cross margin, one-way mode (posSide=net)
                # pos_side="" means one-way (OKX default after account setup).
                # pos_side="long"|"short" enables hedge mode.
                params["tdMode"] = "cross"
                if pos_side:
                    params["posSide"] = pos_side
                if tp_price and sl_price:
                    params["tpTriggerPx"]     = str(round(tp_price, 2))
                    params["tpOrdPx"]         = "-1"
                    params["tpTriggerPxType"] = "last"
                    params["slTriggerPx"]     = str(round(sl_price, 2))
                    params["slOrdPx"]         = "-1"
                    params["slTriggerPxType"] = "last"
                logger.info("OKX futures: %s %s %s %.6f  TP=%s  SL=%s  pos=%s",
                            side.upper(), symbol, order_type, amount,
                            tp_price or "—", sl_price or "—", pos_side or "net")
            elif self._margin_mode:
                # OKX Spot Cross Margin (original behaviour)
                params["tdMode"] = self._margin_mode
                params["ccy"]    = symbol.split("/")[1] if "/" in symbol else "USDT"
                if side == "buy" and tp_price and sl_price:
                    params["tpTriggerPx"]     = str(round(tp_price, 2))
                    params["tpOrdPx"]         = "-1"
                    params["tpTriggerPxType"] = "last"
                    params["slTriggerPx"]     = str(round(sl_price, 2))
                    params["slOrdPx"]         = "-1"
                    params["slTriggerPxType"] = "last"
                    logger.info("OKX margin: %s %s %.6f  TP=%.2f  SL=%.2f",
                                side.upper(), symbol, amount, tp_price, sl_price)

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
                           order_type: str, price: Optional[float],
                           pos_side: str = "") -> OrderResult:
        ticker      = await self.fetch_ticker(symbol)
        exec_price  = price if (order_type == "limit" and price) else ticker["last"]
        base_asset  = symbol.split("/")[0]
        quote_asset = symbol.split("/")[1].split(":")[0] if "/" in symbol else "USDT"

        if self._futures:
            # Futures paper — hedge mode: key = "symbol||side" so long+short can coexist
            margin   = (amount * exec_price) / max(self._leverage, 1)
            fut_side = pos_side if pos_side else ("long" if side == "buy" else "short")
            fut_key  = f"{symbol}||{fut_side}"
            is_close = (fut_side == "long" and side == "sell") or \
                       (fut_side == "short" and side == "buy")
            pos = self._paper_futures.get(fut_key)
            if pos and is_close:
                # Close this leg → realise PnL
                pnl_mult = 1 if fut_side == "long" else -1
                pnl = pnl_mult * (exec_price - pos["entry"]) * pos["amount"]
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + pos["margin"] + pnl
                logger.info("[Paper Futures] Close %s %s @ %.2f  PnL=%.2f USDT",
                            fut_side.upper(), symbol, exec_price, pnl)
                del self._paper_futures[fut_key]
            elif not is_close:
                # Open new leg
                if pos:
                    logger.warning("[Paper Futures] %s already open, ignoring duplicate", fut_key)
                elif self._paper_balance.get("USDT", 0) < margin:
                    raise ValueError(
                        f"[Paper Futures] Insufficient USDT margin: "
                        f"need {margin:.2f}, have {self._paper_balance.get('USDT', 0):.2f}"
                    )
                else:
                    self._paper_balance["USDT"] -= margin
                    self._paper_futures[fut_key] = {
                        "side": fut_side, "amount": amount,
                        "entry": exec_price, "margin": margin,
                    }
                    logger.info("[Paper Futures] Open %s %s %.6f @ %.2f  margin=%.2f USDT",
                                fut_side.upper(), symbol, amount, exec_price, margin)
        else:
            # Spot / margin paper
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
                self._paper_balance[base_asset]  -= amount
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
