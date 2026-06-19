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
        if self._futures:
            # OKX hedge mode: must set leverage for BOTH sides; fail hard if both fail
            any_ok = False
            for ps in ("long", "short"):
                try:
                    await self._exchange.set_leverage(
                        self._leverage, symbol,
                        params={"mgnMode": "isolated", "posSide": ps},
                    )
                    any_ok = True
                except Exception as e:
                    logger.error("set_leverage FAILED %s posSide=%s: %s", symbol, ps, e)
            if not any_ok:
                raise RuntimeError(
                    f"Could not set leverage for {symbol} — cannot place orders safely"
                )
            logger.info("OKX futures leverage: %s × %dx (long+short)", symbol, self._leverage)
        elif self._margin_mode:
            try:
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
                           pos_side: str = "",
                           reduce_only: bool = False) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price, pos_side)

        await self._ensure_leverage(symbol)

        params: dict = {}
        if order_type == "limit" and price:
            params["price"] = price

        if self._exchange_id == "okx":
            if self._futures:
                # ── OKX Perpetual Futures (hedge mode) ────────────────────────
                # Convert BTC quantity → integer contract count (1 contract = ctVal BTC)
                try:
                    market = self._exchange.market(symbol)
                    ct_val = float(market.get("contractSize", 0.01))
                except Exception:
                    ct_val = 0.01  # OKX BTC/USDT:USDT default
                contracts = int(amount / ct_val)
                if contracts < 1:
                    raise ValueError(
                        f"Order too small: {amount:.6f} = {amount/ct_val:.3f} contracts. "
                        f"Minimum 1 contract = {ct_val}. "
                        f"Increase FIXED_TRADE_USDT or LEVERAGE."
                    )
                raw_amount = amount
                amount = float(contracts)
                logger.info("OKX contract conversion: %.6f → %d contracts (ctVal=%.4f)",
                            raw_amount, contracts, ct_val)

                params["tdMode"] = "isolated"
                if pos_side:
                    params["posSide"] = pos_side
                # OKX hedge mode: posSide already specifies which leg to close;
                # reduceOnly is incompatible with hedge mode and causes API error 51000
                if reduce_only and not pos_side:
                    params["reduceOnly"] = True
                # Attach algo TP/SL inline (OKX attachAlgoOrds structure)
                # attachAlgoClOrdId omitted — OKX auto-generates; manual UUID with hyphens → error 51000
                if tp_price and sl_price:
                    params["attachAlgoOrds"] = [{
                        "tpTriggerPx":     str(round(tp_price, 2)),
                        "tpOrdPx":         "-1",    # market execution at trigger
                        "tpTriggerPxType": "last",
                        "slTriggerPx":     str(round(sl_price, 2)),
                        "slOrdPx":         "-1",
                        "slTriggerPxType": "last",
                    }]
                logger.info("OKX futures: %s %s %s %d contracts  TP=%s  SL=%s  pos=%s  reduceOnly=%s",
                            side.upper(), symbol, order_type, int(amount),
                            tp_price or "—", sl_price or "—",
                            pos_side or "net", reduce_only)
            elif self._margin_mode:
                # OKX Spot Cross Margin
                params["tdMode"] = self._margin_mode
                params["ccy"]    = symbol.split("/")[1] if "/" in symbol else "USDT"
                if side == "buy" and tp_price and sl_price:
                    params["attachAlgoOrds"] = [{
                        "tpTriggerPx":     str(round(tp_price, 2)),
                        "tpOrdPx":         "-1",
                        "tpTriggerPxType": "last",
                        "slTriggerPx":     str(round(sl_price, 2)),
                        "slOrdPx":         "-1",
                        "slTriggerPxType": "last",
                    }]
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
            # Futures paper — hedge mode: key = "symbol||side", value = list of open legs
            # Multiple strategies can hold the same direction (e.g. v2 short + v3 short)
            margin   = (amount * exec_price) / max(self._leverage, 1)
            fut_side = pos_side if pos_side else ("long" if side == "buy" else "short")
            fut_key  = f"{symbol}||{fut_side}"
            is_close = (fut_side == "long" and side == "sell") or \
                       (fut_side == "short" and side == "buy")
            legs: list = self._paper_futures.get(fut_key, [])
            if is_close and legs:
                # Close oldest open leg (FIFO)
                pos = legs.pop(0)
                pnl_mult = 1 if fut_side == "long" else -1
                pnl = pnl_mult * (exec_price - pos["entry"]) * pos["amount"]
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + pos["margin"] + pnl
                logger.info("[Paper Futures] Close %s %s @ %.2f  PnL=%.2f USDT",
                            fut_side.upper(), symbol, exec_price, pnl)
                if legs:
                    self._paper_futures[fut_key] = legs
                else:
                    self._paper_futures.pop(fut_key, None)
            elif not is_close:
                # Open new leg — allow multiple per direction (different strategies)
                if self._paper_balance.get("USDT", 0) < margin:
                    raise ValueError(
                        f"[Paper Futures] Insufficient USDT margin: "
                        f"need {margin:.2f}, have {self._paper_balance.get('USDT', 0):.2f}"
                    )
                self._paper_balance["USDT"] -= margin
                legs.append({"side": fut_side, "amount": amount,
                             "entry": exec_price, "margin": margin})
                self._paper_futures[fut_key] = legs
                logger.info("[Paper Futures] Open %s %s %.6f @ %.2f  margin=%.2f USDT  legs=%d",
                            fut_side.upper(), symbol, amount, exec_price, margin, len(legs))
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
        # CCXT: top-level keys are asset names; each value is {free, used, total}
        # "total"/"free"/"used"/"info"/"debt" are summary/meta keys — skip them
        _skip = {"info", "free", "used", "total", "debt", "datetime", "timestamp"}
        return [
            Balance(asset=k, free=float(raw[k].get("free") or 0),
                    used=float(raw[k].get("used") or 0),
                    total=float(raw[k].get("total") or 0))
            for k in raw
            if k not in _skip and isinstance(raw.get(k), dict)
            and float(raw[k].get("total") or 0) > 0
        ]

    async def close(self):
        await self._exchange.close()
