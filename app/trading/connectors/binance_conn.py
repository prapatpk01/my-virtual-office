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

    PAPER_TAKER_FEE = 0.0005  # 0.05% per side, simulated in paper mode

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
        self._paper_used: dict[str, float] = {}     # margin locked in open futures positions
        self._paper_positions: dict[str, dict] = {} # key: f"{symbol}|{pos_side}" -> {amount, entry_price, margin}
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

    def _ct_size(self, symbol: str) -> float:
        """Contract size (ctVal) for a market — how many base-ccy units one
        contract is worth. 1.0 when unknown (base==contracts)."""
        try:
            return float(self._exchange.market(symbol).get("contractSize") or 1) or 1.0
        except Exception:
            return 1.0

    def _to_contracts(self, symbol: str, base_amount: float) -> float:
        """Convert a base-currency size into exchange contracts, rounded to lot
        precision (OKX order/algo sizes are in contracts)."""
        ct = self._ct_size(symbol)
        contracts = base_amount / ct if ct else base_amount
        try:
            return float(self._exchange.amount_to_precision(symbol, contracts))
        except Exception:
            return contracts

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

        # OKX swap/futures orders take the size in CONTRACTS, not base currency.
        # Convert base -> contracts with the market's contractSize (ctVal): SOL
        # has ctVal=1 (base==contracts, no-op) but e.g. HYPE has ctVal=0.1, so a
        # 6-HYPE order is 60 contracts — sending the base amount directly filled
        # 10x too small. Fills come back in contracts and are converted back to
        # base below so the rest of the bot keeps accounting in base units.
        ct_size = self._ct_size(symbol)
        send_amount = self._to_contracts(symbol, amount)

        raw = await self._exchange.create_order(symbol, order_type, side, send_amount, **kwargs)
        order_id = str(raw.get("id", uuid.uuid4()))

        # Post-fill data — the create_order response on OKX carries little
        # more than the id, so fetch the order back to get the ACTUAL fill:
        # avgPx (average fill price), accFillSz (filled size), fee, and pnl
        # (realized trading PnL on reduce/close orders, fees excluded).
        # Notifications must report these real numbers, not our estimates.
        avg_px = raw.get("average") or 0.0
        fill_sz = raw.get("filled") or 0.0
        fee_cost = 0.0
        realized_pnl = None
        status = raw.get("status", "open")
        for attempt in range(4):
            try:
                o = await self._exchange.fetch_order(order_id, symbol)
            except Exception as e:
                logger.debug("fetch_order retry %d for %s: %s", attempt, order_id, e)
                await asyncio.sleep(0.4)
                continue
            info = o.get("info") or {}
            avg_px = o.get("average") or float(info.get("avgPx") or 0) or avg_px
            fill_sz = o.get("filled") or float(info.get("accFillSz") or info.get("fillSz") or 0) or fill_sz
            status = o.get("status") or status
            fee_obj = o.get("fee") or {}
            if fee_obj.get("cost") is not None:
                fee_cost = abs(float(fee_obj["cost"]))
            elif o.get("fees"):
                fee_cost = sum(abs(float(f.get("cost") or 0)) for f in o["fees"])
            elif info.get("fee") is not None:
                fee_cost = abs(float(info["fee"]))  # OKX reports fees as negative
            if info.get("pnl") not in (None, ""):
                try:
                    realized_pnl = float(info["pnl"])
                except (TypeError, ValueError):
                    pass
            if status == "closed" and avg_px and fill_sz:
                break
            await asyncio.sleep(0.4)

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=avg_px or raw.get("price") or price or 0.0,
            filled=(fill_sz or 0.0) * ct_size,   # contracts -> base units
            status=status,
            fee=fee_cost,
            realized_pnl=realized_pnl,
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
        # Simulate the taker fee so paper notifications carry the same
        # post-fill fields (fee, realized_pnl) a live OKX order reports.
        fee_cost = round(amount * exec_price * self.PAPER_TAKER_FEE, 6)
        realized_pnl = None

        if self._futures:
            # side='buy'+pos_side='long' or side='sell'+pos_side='short' OPENS
            # a position; the opposite side CLOSES it. Non-hedge mode never
            # passes pos_side and only ever tracks longs (BUY opens, SELL
            # closes), so pos_side=None always means "long".
            direction = pos_side or "long"
            pos_key   = f"{symbol}|{direction}"
            is_open   = (side == "buy" and direction == "long") or \
                        (side == "sell" and direction == "short")

            if is_open:
                margin = round((amount * exec_price) / max(self._leverage, 1), 4)
                if self._paper_balance.get("USDT", 0) < margin + fee_cost:
                    raise ValueError(
                        f"[Paper/Futures] Insufficient USDT: need {margin + fee_cost:.2f}, "
                        f"have {self._paper_balance.get('USDT', 0):.2f}"
                    )
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) - margin - fee_cost
                self._paper_used["USDT"] = self._paper_used.get("USDT", 0) + margin
                existing = self._paper_positions.get(pos_key)
                if existing:
                    # Adding to an existing position — blend entry price
                    total_amt = existing["amount"] + amount
                    existing["entry_price"] = (
                        existing["entry_price"] * existing["amount"] + exec_price * amount
                    ) / total_amt
                    existing["amount"] = total_amt
                    existing["margin"] += margin
                else:
                    self._paper_positions[pos_key] = {
                        "amount": amount, "entry_price": exec_price, "margin": margin,
                    }
            else:
                # Closing (all or part of) an existing position — realize P&L,
                # return the ORIGINAL margin for the closed portion (not
                # recomputed at exit price), and release the used-margin lock.
                pos = self._paper_positions.get(pos_key)
                if not pos or pos["amount"] <= 0:
                    logger.warning(
                        "[Paper/Futures] Close order for %s with no tracked position "
                        "— crediting notional/leverage only (no P&L realized)", pos_key,
                    )
                    margin = round((amount * exec_price) / max(self._leverage, 1), 4)
                    self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + margin
                else:
                    close_amt = min(amount, pos["amount"])
                    entry     = pos["entry_price"]
                    pnl = ((exec_price - entry) if direction == "long" else (entry - exec_price)) * close_amt
                    realized_pnl = round(pnl, 6)  # trading PnL, fees excluded (mirrors OKX `pnl`)
                    margin_released = pos["margin"] * (close_amt / pos["amount"])

                    self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + margin_released + pnl - fee_cost
                    self._paper_used["USDT"] = max(0.0, self._paper_used.get("USDT", 0) - margin_released)

                    pos["amount"] -= close_amt
                    pos["margin"] -= margin_released
                    if pos["amount"] <= 1e-12:
                        self._paper_positions.pop(pos_key, None)
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
            fee=fee_cost,
            realized_pnl=realized_pnl,
        )
        self._paper_open_orders.append(order)
        return order

    async def set_position_tpsl(
        self,
        symbol: str,
        pos_side: str,          # 'long' | 'short'
        amount: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Place (or replace) reduce-only TP/SL algo orders on the exchange so
        the stop/target are visible and enforced by OKX itself — not only by
        the bot loop. Cancels any existing trigger orders for the symbol first,
        then places a stop-market (SL) and a take-profit-market (TP), each sized
        to `amount` and closing the position side. Used on entry (full size) and
        again after TP1 (remaining size + break-even SL).

        No-op in paper mode. Best-effort: a failure here is logged and swallowed
        so it can never take down the actual position management."""
        if self.paper or not self._futures:
            return False
        close_side = "sell" if pos_side == "long" else "buy"
        params_base: dict = {"reduceOnly": True}
        if self._hedge_mode:
            params_base["posSide"] = pos_side
        ok = True
        # 1) clear existing trigger/algo orders for this symbol
        try:
            await self._exchange.cancel_all_orders(symbol, params={"trigger": True})
        except Exception as e:
            logger.debug("[TPSL] cancel existing trigger orders failed for %s: %s", symbol, e)
        # 2) (re)place SL and TP as reduce-only trigger orders
        if sl:
            try:
                await self._exchange.create_order(
                    symbol, "market", close_side, self._to_contracts(symbol, amount),
                    params={**params_base, "stopLossPrice": sl},
                )
                logger.info("[TPSL] SL set on %s %s @ %.6g (sz %.6g)", symbol, pos_side, sl, amount)
            except Exception as e:
                ok = False
                logger.warning("[TPSL] failed to set SL on %s: %s", symbol, e)
        if tp:
            try:
                await self._exchange.create_order(
                    symbol, "market", close_side, self._to_contracts(symbol, amount),
                    params={**params_base, "takeProfitPrice": tp},
                )
                logger.info("[TPSL] TP set on %s %s @ %.6g (sz %.6g)", symbol, pos_side, tp, amount)
            except Exception as e:
                ok = False
                logger.warning("[TPSL] failed to set TP on %s: %s", symbol, e)
        return ok

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

    async def fetch_positions(self, symbols: Optional[list[str]] = None) -> list[dict]:
        """Live open positions from the exchange (nonzero contracts only) —
        used on startup to re-derive positions that were opened before a
        bot restart, since nothing in-process persists them. Paper mode has
        no exchange-side state to reconcile against (a fresh process starts
        with an empty paper position book anyway), so it always returns [].
        Returns a plain dict per position: symbol/side/amount/entry_price."""
        if self.paper:
            return []
        raw_list = await self._exchange.fetch_positions(symbols)
        out = []
        for p in raw_list:
            contracts = p.get("contracts") or 0
            if not contracts:
                continue
            side = p.get("side") or ("long" if contracts > 0 else "short")
            entry_price = p.get("entryPrice")
            if not entry_price:
                continue
            info = p.get("info") or {}
            def _f(*keys):
                for src in (p, info):
                    for k in keys:
                        v = src.get(k)
                        if v not in (None, ""):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
                return None
            out.append({
                "symbol": p.get("symbol"),
                "side": side,
                "amount": abs(float(contracts)) * self._ct_size(p.get("symbol")),  # contracts -> base
                "entry_price": float(entry_price),
                "mark_price": _f("markPrice", "markPx"),
                "unrealized_pnl": _f("unrealizedPnl", "upl"),
                "notional": _f("notional", "notionalUsd"),
                "margin": _f("initialMargin", "margin", "imr", "mmr"),
            })
        return out

    async def fetch_positions_history(self, since: Optional[int] = None,
                                      limit: int = 100) -> list[dict]:
        """Closed round-trip positions from OKX positions-history. ONE row per
        position (open -> fully closed; T1/T2 partials already collapsed by OKX).
        `realized_pnl` is OKX's own realizedPnl = pnl - fees - funding, so it
        matches the OKX app exactly — never recompute it. Each dict:
        symbol/side/realized_pnl/open_ts/close_ts/entry_px/close_px/size/lever.
        Live only; [] on paper or if the endpoint isn't available."""
        if self.paper:
            return []
        try:
            raw = await self._exchange.fetch_positions_history(
                symbols=None, since=since, limit=limit)
        except Exception as e:
            logger.warning("[Stats] fetch_positions_history failed: %s", e)
            return []
        out = []
        for p in raw:
            info = p.get("info") or {}
            def _f(*keys, src_first=True):
                for src in ((info, p) if not src_first else (p, info)):
                    for k in keys:
                        v = src.get(k)
                        if v not in (None, ""):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
                return None
            rpnl = _f("realizedPnl", src_first=False)
            if rpnl is None:
                rpnl = _f("realized_pnl") or (float(info["pnl"]) if info.get("pnl") not in (None, "") else 0.0)
            # OKX: direction 'long'/'short'; posSide too. ccxt normalizes 'side'.
            side = (p.get("side") or info.get("direction") or info.get("posSide") or "").lower()
            open_ts = int(_f("timestamp", src_first=True) or float(info.get("cTime") or 0) or 0)
            close_ts = int(float(info.get("uTime") or 0) or p.get("lastUpdateTimestamp") or open_ts)
            out.append({
                "symbol": p.get("symbol") or info.get("instId"),
                "side": "long" if side.startswith("l") else "short" if side.startswith("s") else side,
                "realized_pnl": round(float(rpnl or 0.0), 6),
                "open_ts": open_ts,
                "close_ts": close_ts,
                "entry_px": _f("entryPrice") or (float(info["openAvgPx"]) if info.get("openAvgPx") else None),
                "close_px": float(info["closeAvgPx"]) if info.get("closeAvgPx") else None,
                "size": round(abs(float(info.get("closeTotalPos") or 0)) * self._ct_size(p.get("symbol") or info.get("instId") or ""), 8),
                "lever": _f("leverage") or (float(info["lever"]) if info.get("lever") else None),
            })
        return out

    async def fetch_closed_orders_raw(self, symbol: str, since: Optional[int] = None,
                                      limit: int = 100) -> list[dict]:
        """Normalized FILLED orders (opens + closes) from OKX order history for
        one symbol. Each dict: symbol/side/pos_side/price(avgPx)/amount(fillSz)/
        fee/pnl(realized, closes only)/reduce_only/ts/ord_id. Live only."""
        if self.paper:
            return []
        raw = await self._exchange.fetch_closed_orders(symbol, since=since, limit=limit)
        out = []
        for o in raw:
            info = o.get("info") or {}
            filled = float(o.get("filled") or info.get("accFillSz") or 0)
            if filled <= 0:
                continue
            fee = 0.0
            if o.get("fee") and o["fee"].get("cost") is not None:
                fee = abs(float(o["fee"]["cost"]))
            elif info.get("fee") not in (None, ""):
                try:
                    fee = abs(float(info["fee"]))
                except (TypeError, ValueError):
                    pass
            pnl = None
            if info.get("pnl") not in (None, ""):
                try:
                    pnl = float(info["pnl"])
                except (TypeError, ValueError):
                    pass
            reduce_only = str(info.get("reduceOnly")).lower() in ("true", "1")
            out.append({
                "symbol":   o.get("symbol") or symbol,
                "side":     o.get("side"),
                "pos_side": info.get("posSide"),
                "price":    float(o.get("average") or info.get("avgPx") or o.get("price") or 0),
                "amount":   filled,
                "fee":      fee,
                "pnl":      pnl,
                "reduce_only": reduce_only,
                "ts":       int(o.get("timestamp") or int(info.get("uTime") or 0) or 0),
                "ord_id":   o.get("id"),
            })
        out.sort(key=lambda x: x["ts"])
        return out

    async def fetch_recent_closes(self, symbol: str, limit: int = 5) -> list[dict]:
        """Most-recent reduce-only (close) fills for a symbol, newest first."""
        orders = await self.fetch_closed_orders_raw(symbol, limit=max(limit * 4, 20))
        closes = [o for o in orders if o["reduce_only"] or (o["pnl"] not in (None, 0.0))]
        return list(reversed(closes))[:limit]

    async def fetch_balance(self) -> list[Balance]:
        if self.paper:
            # total = free + used (margin locked in open positions) so that
            # opening a leveraged position never looks like a loss of equity —
            # only realized P&L on close changes total.
            assets = set(self._paper_balance) | set(self._paper_used)
            out = []
            for k in assets:
                free  = self._paper_balance.get(k, 0.0)
                used  = self._paper_used.get(k, 0.0)
                total = free + used
                if total > 0:
                    out.append(Balance(asset=k, free=free, used=used, total=total))
            return out
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
