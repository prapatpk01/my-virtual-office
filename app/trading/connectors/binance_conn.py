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
                 leverage: int = 1,
                 symbol_leverage: dict | None = None):  # per-symbol override e.g. {"XAU/USDT:USDT": 10}
        super().__init__(api_key, api_secret, paper)
        self._exchange_id  = exchange_id
        self._margin_mode  = margin_mode.lower()
        self._market_type  = market_type.lower()
        self._leverage     = leverage
        self._symbol_leverage: dict[str, int] = symbol_leverage or {}
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

    async def ensure_hedge_mode(self) -> None:
        """
        Verify OKX account is in long_short_mode (hedge mode). Raises RuntimeError
        if not — every posSide order will be rejected by OKX error 51000 otherwise.
        No-op for paper trading or non-OKX exchanges.
        """
        if self.paper or self._exchange_id != "okx" or not self._futures:
            return
        try:
            await self._exchange.load_markets()
            resp = await self._exchange.private_get_account_config()
            pos_mode = resp.get("data", [{}])[0].get("posMode", "unknown")
        except Exception as e:
            logger.warning("[Connector] Could not verify OKX posMode: %s — assuming hedge mode is set", e)
            return
        if pos_mode != "long_short_mode":
            raise RuntimeError(
                f"OKX account is in '{pos_mode}' mode — must be 'long_short_mode' (hedge mode). "
                "Go to OKX → Account → Settings → Position Mode → Hedge Mode, then restart the bot."
            )
        logger.info("[Connector] OKX posMode=long_short_mode ✓")

    async def _ensure_leverage(self, symbol: str, override_leverage: int = 0):
        """Set leverage on OKX (once per symbol per session)."""
        if self._exchange_id != "okx":
            return
        if symbol in self._leverage_set:
            return
        lev = override_leverage if override_leverage > 0 else self._leverage
        if self._futures:
            # OKX hedge mode: must set leverage for BOTH sides; fail hard if either fails.
            failed = []
            for ps in ("long", "short"):
                try:
                    await self._exchange.set_leverage(
                        lev, symbol,
                        params={"mgnMode": "isolated", "posSide": ps},
                    )
                except Exception as e:
                    logger.error("set_leverage FAILED %s posSide=%s: %s", symbol, ps, e)
                    failed.append(ps)
            if failed:
                raise RuntimeError(
                    f"Could not set leverage for {symbol} side(s) {failed} — cannot place orders safely"
                )
            logger.info("OKX futures leverage: %s × %dx (long+short)", symbol, lev)
        elif self._margin_mode:
            try:
                await self._exchange.set_leverage(
                    lev, symbol,
                    params={"mgnMode": self._margin_mode},
                )
                logger.info("OKX margin leverage: %s × %dx %s", symbol, self._leverage, self._margin_mode)
            except Exception as e:
                logger.warning("set_leverage failed (%s): %s — continuing", symbol, e)
        self._leverage_set.add(symbol)

    # OKX ctVal per contract when markets haven't loaded yet (fallback only).
    _OKX_CONTRACT_SIZE: dict[str, float] = {
        "BTC/USDT:USDT": 0.01,
        "XAU/USDT:USDT": 1.0,
        "ETH/USDT:USDT": 0.1,
    }

    async def contract_size(self, symbol: str) -> float:
        """Base-asset units per contract (OKX futures: ctVal)."""
        if self._exchange_id == "okx" and self._futures:
            try:
                sz = self._exchange.market(symbol).get("contractSize")
                if sz is not None:
                    return float(sz)
            except Exception:
                pass
            return self._OKX_CONTRACT_SIZE.get(symbol, 0.01)
        return 1.0

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

        await self._ensure_leverage(symbol, override_leverage=self._symbol_leverage.get(symbol, 0))

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
                # Close oldest open leg (FIFO) — supports PARTIAL closes (TP1).
                pos = legs[0]
                close_amt = min(amount, pos["amount"]) if amount > 0 else pos["amount"]
                frac      = close_amt / pos["amount"] if pos["amount"] > 0 else 1.0
                refund    = pos["margin"] * frac
                pnl_mult  = 1 if fut_side == "long" else -1
                pnl       = pnl_mult * (exec_price - pos["entry"]) * close_amt
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + refund + pnl
                pos["amount"] -= close_amt
                pos["margin"] -= refund
                if pos["amount"] <= 1e-9:
                    legs.pop(0)
                logger.info("[Paper Futures] Close %s %s %.6f @ %.2f  PnL=%.2f USDT  (leg left=%.6f)",
                            fut_side.upper(), symbol, close_amt, exec_price, pnl, pos["amount"])
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

    async def move_sl_to_breakeven(
        self, symbol: str, pos_side: str, entry_price: float, remaining_amount: float
    ) -> bool:
        """
        After TP1, cancel the exchange algo SL and replace it at breakeven price.
        OKX futures only; no-op for paper/non-OKX (returns True so caller treats it as success).
        Returns True on success or no-op, False if the attempt fails.
        """
        if self.paper or self._exchange_id != "okx" or not self._futures:
            return True

        try:
            market  = self._exchange.market(symbol)
            inst_id = market["id"]   # e.g. "BTC-USDT-SWAP"
        except Exception as e:
            logger.warning("[SL→BE] market lookup failed for %s: %s", symbol, e)
            return False

        try:
            # 1. Fetch all pending conditional (SL/TP) algo orders for this symbol
            resp = await self._exchange.privateGetTradeAlgosPending({
                "instId": inst_id,
                "ordType": "conditional",
            })
            algo_orders = (resp or {}).get("data", [])

            # Identify SL algo orders belonging to our position leg
            sl_orders = [
                o for o in algo_orders
                if (o.get("posSide") == pos_side
                    and o.get("slTriggerPx") not in (None, "", "0", "0.0"))
            ]

            if sl_orders:
                # 2. Cancel each SL algo — failure here is non-fatal; we still place the new one
                cancel_payload = [{"algoId": o["algoId"], "instId": inst_id} for o in sl_orders]
                try:
                    await self._exchange.privatePostTradeCancelAlgos(cancel_payload)
                    logger.info("[SL→BE] Cancelled %d algo SL(s) for %s %s",
                                len(sl_orders), symbol, pos_side)
                except Exception as ce:
                    logger.warning("[SL→BE] Cancel algo SL failed (proceeding anyway): %s", ce)
            else:
                logger.info("[SL→BE] No pending algo SL for %s %s — attaching new one at BE",
                            symbol, pos_side)

            # 3. Place new SL at breakeven for the remaining runner contracts
            ct_val    = await self.contract_size(symbol)
            contracts = max(1, round(remaining_amount / ct_val))
            close_side = "sell" if pos_side == "long" else "buy"

            await self._exchange.privatePostTradeOrderAlgo({
                "instId":          inst_id,
                "tdMode":          "isolated",
                "side":            close_side,
                "posSide":         pos_side,
                "ordType":         "conditional",
                "sz":              str(contracts),
                "slTriggerPx":     str(round(entry_price, 2)),
                "slOrdPx":         "-1",
                "slTriggerPxType": "last",
            })
            logger.info("[SL→BE] Exchange SL moved to BE %.4f for %s %s runner (%d contracts)",
                        entry_price, symbol, pos_side, contracts)
            return True

        except Exception as e:
            logger.warning("[SL→BE] move_sl_to_breakeven failed for %s %s: %s",
                           symbol, pos_side, e)
            return False

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
