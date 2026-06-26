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
        # OKX: defaultType="swap" loads perpetual futures markets so symbols like
        # BTC/USDT:USDT resolve correctly and minSz/contractSize match the SWAP instrument.
        options: dict = {"defaultType": "swap" if exchange_id == "okx" else "spot"}
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
        self._leverage_set: set = set()   # symbols already configured (leverage + hedge mode)
        self._pos_mode_set: bool = False  # account-wide hedge mode set flag

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
                           sl: Optional[float] = None,
                           position_side: str = "LONG") -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, order_type, price)

        if self._exchange_id == "okx":
            return await self._okx_order(symbol, side, amount, order_type, price, tp, sl,
                                         position_side)

        if self._exchange_id == "binance":
            return await self._binance_order(symbol, side, amount, order_type, price, tp, sl)

        # Other exchanges: standard CCXT (no exchange-side SL/TP)
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

    async def _ensure_swap_ready(self, symbol: str) -> None:
        """Set OKX account to hedge mode and configure leverage (once per symbol)."""
        if symbol in self._leverage_set:
            return
        if not self._pos_mode_set:
            try:
                await self._exchange.set_position_mode(hedged=True, symbol=symbol)
                self._pos_mode_set = True
                _log.info("OKX position mode → hedge (long_short_mode)")
            except Exception as e:
                _log.debug("set_position_mode (non-fatal, may already be set): %s", e)
                self._pos_mode_set = True  # don't retry
        for pos_side in ("long", "short"):
            try:
                await self._exchange.set_leverage(
                    self.leverage, symbol,
                    params={"mgnMode": "cross", "posSide": pos_side},
                )
            except Exception as e:
                _log.warning("set_leverage %dx %s %s (non-fatal): %s",
                             self.leverage, symbol, pos_side, e)
        self._leverage_set.add(symbol)
        _log.info("OKX leverage=%dx set for %s (both sides)", self.leverage, symbol)

    async def _okx_order(self, symbol: str, side: str, amount: float,
                         order_type: str, price: Optional[float],
                         tp: Optional[float], sl: Optional[float],
                         position_side: str = "LONG") -> OrderResult:
        """OKX perpetual swap order in hedge mode.

        Uses posSide=long/short so both directions can coexist on the same symbol.
        TP/SL attached atomically via attachAlgoOrds — OKX creates algo orders on fill.
        """
        await self._exchange.load_markets()
        await self._ensure_swap_ready(symbol)

        mkt  = self._exchange.markets.get(symbol) or {}
        info = mkt.get("info") or {}

        # instId: prefer the exchange's own field; fallback constructs SWAP id from symbol
        inst_id = info.get("instId") or (
            symbol.split(":")[0].replace("/", "-") + "-SWAP"
            if ":" in symbol else symbol.replace("/", "-")
        )

        # Convert base-currency amount → contracts (ctVal = base per contract)
        ct_val    = float(info.get("ctVal") or mkt.get("contractSize") or 1)
        contracts = max(1, int(amount / ct_val))
        sz_str    = str(contracts)

        # Minimum contract check
        min_sz = int(float(info.get("minSz") or 1))
        if contracts < min_sz:
            px = price
            if not px:
                try:
                    px = float((await self._exchange.fetch_ticker(symbol)).get("last") or 0)
                except Exception:
                    px = 0.0
            needed = int(min_sz * ct_val * (px or 1)) + 1
            raise ValueError(
                f"Contracts {contracts} < OKX min {min_sz} for {symbol}. "
                f"Set TRADE_AMOUNT_USDT >= {needed}."
            )

        # SL/TP direction validation
        if sl and tp and price:
            if position_side == "LONG":
                if sl >= price:
                    raise ValueError(f"LONG SL {sl:.2f} must be below entry {price:.2f}")
                if tp <= price:
                    raise ValueError(f"LONG TP {tp:.2f} must be above entry {price:.2f}")
            else:
                if sl <= price:
                    raise ValueError(f"SHORT SL {sl:.2f} must be above entry {price:.2f}")
                if tp >= price:
                    raise ValueError(f"SHORT TP {tp:.2f} must be below entry {price:.2f}")

        pos_side = "long" if position_side == "LONG" else "short"

        req: dict = {
            "instId":  inst_id,
            "tdMode":  "cross",
            "side":    side,
            "posSide": pos_side,   # hedge mode — required for both open and close
            "ordType": order_type,
            "sz":      sz_str,
        }
        if order_type == "limit" and price:
            req["px"] = self._exchange.price_to_precision(symbol, price)

        # Attach TP/SL atomically on entry orders (opening a new position)
        is_entry = (side == "buy" and position_side == "LONG") or \
                   (side == "sell" and position_side == "SHORT")
        if is_entry and tp and sl and tp > 0 and sl > 0:
            req["attachAlgoOrds"] = [{
                "attachAlgoClOrdId": str(uuid.uuid4()).replace("-", "")[:32],
                "tpTriggerPx":     self._exchange.price_to_precision(symbol, tp),
                "tpOrdPx":         "-1",   # market order when TP triggers
                "tpTriggerPxType": "last",
                "slTriggerPx":     self._exchange.price_to_precision(symbol, sl),
                "slOrdPx":         "-1",   # market order when SL triggers
                "slTriggerPxType": "last",
            }]
            _log.info("OKX %s %s contracts=%d TP=%.4f SL=%.4f",
                      pos_side, side, contracts, tp, sl)

        resp = await self._exchange.privatePostTradeOrder(req)
        data_row = ((resp or {}).get("data") or [{}])[0]
        ord_id = data_row.get("ordId") or str(uuid.uuid4())

        exec_price = await self._resolve_fill_price(ord_id, symbol, price)

        return OrderResult(
            order_id=ord_id, symbol=symbol, side=side,
            amount=amount, price=exec_price, filled=amount, status="closed",
        )

    async def _binance_order(self, symbol: str, side: str, amount: float,
                              order_type: str, price: Optional[float],
                              tp: Optional[float], sl: Optional[float]) -> OrderResult:
        """Binance spot: market/limit order + OCO sell order for SL/TP protection.

        Two-step process:
          1. Execute the main entry order (market or limit).
          2. Immediately place an OCO sell order: TP as limit + SL as stop-limit.
             The OCO is atomic — only one of the two legs can fill.

        If OCO placement fails the position is already open and unprotected, so we
        raise immediately so the bot can log/alert and the user can intervene.
        """
        await self._exchange.load_markets()
        mkt     = self._exchange.markets.get(symbol) or {}
        amt_str = self._exchange.amount_to_precision(symbol, amount)
        famt    = float(amt_str)

        # Validate direction
        if side == "buy" and sl and tp and price:
            if sl >= price:
                raise ValueError(f"SL {sl:.4f} must be < entry {price:.4f}")
            if tp <= price:
                raise ValueError(f"TP {tp:.4f} must be > entry {price:.4f}")

        # Step 1 — main order
        kwargs: dict = {}
        if order_type == "limit" and price:
            kwargs["price"] = self._exchange.price_to_precision(symbol, price)
        raw = await self._exchange.create_order(symbol, order_type, side, famt, **kwargs)
        ord_id     = str(raw.get("id", uuid.uuid4()))
        exec_price = await self._resolve_fill_price(ord_id, symbol, price)

        # Step 2 — OCO sell order (only after a successful buy with valid SL/TP)
        if side == "buy" and tp and sl and tp > 0 and sl > 0:
            # sl_limit: slightly below the stop trigger to ensure fill in fast markets
            prec       = int(mkt.get("precision", {}).get("price") or 2)
            sl_limit   = round(sl * 0.995, prec)
            tp_precise = self._exchange.price_to_precision(symbol, tp)
            sl_precise = self._exchange.price_to_precision(symbol, sl)
            sl_lim_str = self._exchange.price_to_precision(symbol, sl_limit)
            try:
                await self._exchange.create_order(
                    symbol, "oco", "sell", famt, price=float(tp_precise),
                    params={
                        "stopPrice":            float(sl_precise),
                        "stopLimitPrice":       float(sl_lim_str),
                        "stopLimitTimeInForce": "GTC",
                    },
                )
                _log.info("[Binance] OCO placed: TP=%.4f  SL=%.4f  SL-limit=%.4f",
                          tp, sl, sl_limit)
            except Exception as e:
                _log.error(
                    "[Binance] OCO FAILED — position OPEN WITHOUT stop protection! %s", e
                )
                raise RuntimeError(f"Binance OCO failed after entry fill: {e}") from e

        return OrderResult(
            order_id=ord_id, symbol=symbol, side=side,
            amount=famt, price=exec_price, filled=famt, status="closed",
        )

    async def _resolve_fill_price(self, ord_id: str, symbol: str,
                                  fallback: Optional[float]) -> float:
        """Poll the order for its average fill price; fall back to ticker, then `fallback`.

        Returns a strictly positive price. Raises ValueError if no positive price can be
        found anywhere (so the bot never stores entry_price=0 and corrupts PnL/exit math).
        """
        import asyncio
        for attempt in range(3):
            try:
                od = await self._exchange.fetch_order(ord_id, symbol)
                px = float(od.get("average") or od.get("price") or 0)
                if px > 0:
                    return px
            except Exception:
                pass
            await asyncio.sleep(0.5 * (attempt + 1))

        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            px = float(ticker.get("last") or 0)
            if px > 0:
                return px
        except Exception:
            pass

        if fallback and float(fallback) > 0:
            return float(fallback)
        raise ValueError(f"Could not resolve fill price for {symbol} order {ord_id}")

    async def _paper_order(self, symbol: str, side: str, amount: float,
                           order_type: str, price: Optional[float]) -> OrderResult:
        ticker = await self.fetch_ticker(symbol)
        exec_price = price if (order_type == "limit" and price) else ticker["last"]
        base = symbol.split("/")[0]
        # Strip settlement suffix: "USDT:USDT" → "USDT"
        quote = (symbol.split("/")[1] if "/" in symbol else "USDT").split(":")[0]

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
        """Return set of symbols with open leveraged positions on OKX, or None.

        Returns None for paper mode, non-OKX, leverage=1 (spot), or on API error.
        The OKX positions API only tracks LEVERAGED margin positions (borrowed funds).
        For leverage=1 spot cross-margin, no "position" entry is created — the asset
        simply moves in the balance. Returning None tells the bot to rely on in-memory
        tracking and price-based SL/TP checks instead.
        """
        if self.paper or self._exchange_id != "okx":
            return None
        try:
            raw = await self._exchange.fetch_positions(params={"instType": "SWAP"})
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
        try:
            await self._exchange.close()
        except Exception as e:
            _log.warning("Exchange close error (non-fatal): %s", e)
        # CCXT __del__ warns when session or socks_proxy_sessions is not None.
        # Force-null both so GC never emits warnings if close() above was partial.
        try:
            self._exchange.session = None
            self._exchange.socks_proxy_sessions = None
        except Exception:
            pass
