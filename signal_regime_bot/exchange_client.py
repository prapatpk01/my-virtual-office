"""
OKX Perpetual Swap client via ccxt — hedge mode, isolated margin.

Every fix learned from the TrendContV2 connector review is applied here
from the start:
  - contract-size resolution refuses to trade rather than guessing
  - TP/SL attached independently (either leg alone still protects)
  - move_sl_to_breakeven queries oco+conditional+move_order_stop (the
    original TP+SL is stored as `oco`, not `conditional`) and re-arms
    the stop as an OCO so the take-profit backstop survives
  - leverage set once per symbol before the first order, isolated margin
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import ccxt.async_support as ccxt

logger = logging.getLogger("exchange_client")


# Fallback contract sizes (base units per contract) when ccxt market metadata
# is unavailable. NEVER silently default beyond this map — refuse to trade
# instead (a wrong guess can 100x an order at 20x leverage).
_OKX_CONTRACT_SIZE: dict[str, float] = {
    "BTC/USDT:USDT": 0.01,
    "ETH/USDT:USDT": 0.1,
    "XAU/USDT:USDT": 1.0,
    "XAG/USDT:USDT": 1.0,
    "SOL/USDT:USDT": 1.0,
    "XRP/USDT:USDT": 100.0,
}


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str


@dataclass
class Balance:
    asset: str
    free: float
    used: float
    total: float


class ExchangeClient:
    def __init__(self, api_key: str, api_secret: str, passphrase: str,
                paper: bool, leverage: int = 20, margin_mode: str = "isolated"):
        self.paper = paper
        self._leverage = leverage
        self._margin_mode = margin_mode
        self._leverage_set: set[str] = set()
        self._hedge_confirmed = False
        self._paper_balance: dict[str, float] = {"USDT": 10_000.0}
        self._paper_positions: dict[str, dict] = {}

        self._exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": api_secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    async def close(self):
        try:
            await self._exchange.close()
        except Exception:
            pass

    # ── Setup ─────────────────────────────────────────────────────────────────

    async def ensure_hedge_mode(self) -> bool:
        if self.paper:
            return True
        try:
            resp = await self._exchange.privateGetAccountConfig()
            data = (resp or {}).get("data", [])
            pos_mode = data[0].get("posMode") if data else None
            if pos_mode == "long_short_mode":
                self._hedge_confirmed = True
                return True
            logger.warning("[SETUP] Account posMode=%s (expected long_short_mode) — "
                           "attempting to set hedge mode", pos_mode)
            await self._exchange.privatePostAccountSetPositionMode({"posMode": "long_short_mode"})
            self._hedge_confirmed = True
            return True
        except Exception as e:
            logger.error("[SETUP] Could not confirm/set hedge mode: %s — refusing to trade live", e)
            self._hedge_confirmed = False
            return False

    async def _ensure_leverage(self, symbol: str):
        if self.paper or symbol in self._leverage_set:
            return
        try:
            for pos_side in ("long", "short"):
                await self._exchange.set_leverage(
                    self._leverage, symbol,
                    params={"mgnMode": self._margin_mode, "posSide": pos_side},
                )
            self._leverage_set.add(symbol)
            logger.info("[SETUP] Leverage set %dx isolated for %s", self._leverage, symbol)
        except Exception as e:
            logger.error("[SETUP] set_leverage failed for %s: %s", symbol, e)
            raise

    async def contract_size(self, symbol: str) -> float:
        if self.paper:
            return _OKX_CONTRACT_SIZE.get(symbol, 0.0) or 1.0
        try:
            market = self._exchange.market(symbol)
            sz = market.get("contractSize")
            if sz:
                return float(sz)
        except Exception:
            pass
        return _OKX_CONTRACT_SIZE.get(symbol, 0.0)

    # ── Market data ───────────────────────────────────────────────────────────

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> list:
        # One retry on transient network/timeout errors — OKX candle requests
        # occasionally time out under load, and without a retry that single
        # blip skips the whole symbol for this poll cycle instead of just
        # costing ~1s.
        last_err = None
        for attempt in range(2):
            try:
                return await self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            except Exception as e:
                last_err = e
                if attempt == 0:
                    logger.warning("[DATA] fetch_ohlcv %s %s attempt 1 failed, retrying: %s",
                                  symbol, timeframe, e)
                    await asyncio.sleep(1.0)
        logger.error("[DATA] fetch_ohlcv failed %s %s after retry: %s", symbol, timeframe, last_err)
        raise last_err

    async def fetch_ticker(self, symbol: str) -> dict:
        if self.paper:
            # Paper mode still needs a live price reference — always hit the real ticker.
            pass
        return await self._exchange.fetch_ticker(symbol)

    async def fetch_balance_usdt(self) -> float:
        if self.paper:
            return self._paper_balance.get("USDT", 0.0)
        try:
            raw = await self._exchange.fetch_balance()
            usdt = raw.get("USDT", {})
            return float(usdt.get("free") or usdt.get("total") or 0.0)
        except Exception as e:
            logger.error("[DATA] fetch_balance failed: %s", e)
            raise

    async def fetch_position_amount(self, symbol: str, side: str) -> float:
        """Actual base-asset size of an open OKX position (0.0 if none). Ground truth."""
        if self.paper:
            key = f"{symbol}||{side}"
            pos = self._paper_positions.get(key)
            return float(pos["amount"]) if pos else 0.0
        try:
            positions = await self._exchange.fetch_positions([symbol])
            for p in positions:
                if p.get("side") == side and float(p.get("contracts", 0)) > 0:
                    ct_val = float(p.get("contractSize") or 1.0)
                    return float(p["contracts"]) * ct_val
        except Exception as e:
            logger.warning("[DATA] fetch_position_amount failed %s %s: %s", symbol, side, e)
        return 0.0

    # ── Orders ───────────────────────────────────────────────────────────────

    async def create_order(self, symbol: str, side: str, amount: float,
                           pos_side: str, tp_price: Optional[float] = None,
                           sl_price: Optional[float] = None,
                           reduce_only: bool = False) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, pos_side, tp_price, sl_price)

        await self._ensure_leverage(symbol)

        ct_val = await self.contract_size(symbol)
        if not ct_val or ct_val <= 0:
            raise ValueError(
                f"Cannot resolve contract size for {symbol} — refusing order "
                f"(guessing risks a massively oversized position at {self._leverage}x)."
            )
        contracts = int(round(amount / ct_val))
        if contracts < 1:
            raise ValueError(
                f"Order too small: {amount:.6f} = {amount/ct_val:.3f} contracts "
                f"(min 1 = {ct_val}). Increase risk_per_trade or balance."
            )

        params: dict = {"tdMode": self._margin_mode, "posSide": pos_side}
        if reduce_only:
            # Hedge mode: posSide already scopes the close to that leg;
            # reduceOnly is incompatible with hedge mode (OKX error 51000).
            pass
        if tp_price or sl_price:
            algo: dict = {}
            if tp_price:
                algo["tpTriggerPx"] = str(round(tp_price, 6))
                algo["tpOrdPx"] = "-1"
                algo["tpTriggerPxType"] = "last"
            if sl_price:
                algo["slTriggerPx"] = str(round(sl_price, 6))
                algo["slOrdPx"] = "-1"
                algo["slTriggerPxType"] = "last"
            params["attachAlgoOrds"] = [algo]

        try:
            raw = await self._exchange.create_order(symbol, "market", side, float(contracts),
                                                     None, params=params)
        except Exception as e:
            logger.error("[ORDER] create_order failed %s %s %s: %s", symbol, side, pos_side, e)
            raise

        return OrderResult(
            order_id=str(raw.get("id", uuid.uuid4())), symbol=symbol, side=side,
            amount=float(contracts) * ct_val, price=raw.get("price") or 0.0,
            status=raw.get("status", "open"),
        )

    async def move_sl_to_breakeven(self, symbol: str, pos_side: str, entry_price: float,
                                   remaining_amount: float,
                                   tp_price: Optional[float] = None) -> bool:
        if self.paper:
            return True
        try:
            market = self._exchange.market(symbol)
            inst_id = market["id"]
        except Exception as e:
            logger.warning("[SL->BE] market lookup failed for %s: %s", symbol, e)
            return False

        try:
            algo_orders: list = []
            for ord_type in ("oco", "conditional", "move_order_stop"):
                try:
                    resp = await self._exchange.privateGetTradeAlgosPending({
                        "instId": inst_id, "ordType": ord_type,
                    })
                    algo_orders.extend((resp or {}).get("data", []))
                except Exception as qe:
                    logger.warning("[SL->BE] pending-algos query (%s) failed: %s", ord_type, qe)

            stale = [o for o in algo_orders
                    if o.get("posSide") == pos_side
                    and o.get("slTriggerPx") not in (None, "", "0", "0.0")]
            for algo in stale:
                try:
                    await self._exchange.privatePostTradeCancelAlgos(
                        [{"algoId": algo["algoId"], "instId": inst_id}])
                    logger.info("[SL->BE] Cancelled stale algo %s (%s) for %s %s",
                                algo.get("algoId"), algo.get("ordType"), symbol, pos_side)
                except Exception as ce:
                    logger.warning("[SL->BE] Cancel algo %s failed: %s", algo.get("algoId"), ce)

            ct_val = await self.contract_size(symbol)
            if not ct_val or ct_val <= 0:
                logger.error("[SL->BE] Cannot resolve contract size for %s — cannot re-arm stop", symbol)
                return False
            contracts = max(1, round(remaining_amount / ct_val))
            close_side = "sell" if pos_side == "long" else "buy"

            algo_req: dict = {
                "instId": inst_id, "tdMode": self._margin_mode, "side": close_side,
                "posSide": pos_side, "sz": str(contracts),
                "slTriggerPx": str(round(entry_price, 6)), "slOrdPx": "-1",
                "slTriggerPxType": "last",
            }
            if tp_price:
                algo_req["ordType"] = "oco"
                algo_req["tpTriggerPx"] = str(round(tp_price, 6))
                algo_req["tpOrdPx"] = "-1"
                algo_req["tpTriggerPxType"] = "last"
            else:
                algo_req["ordType"] = "conditional"

            await self._exchange.privatePostTradeOrderAlgo(algo_req)
            logger.info("[SL->BE] Exchange stop moved to BE %.6f for %s %s (%d contracts, tp=%s)",
                       entry_price, symbol, pos_side, contracts, tp_price or "-")
            return True
        except Exception as e:
            logger.warning("[SL->BE] move_sl_to_breakeven failed %s %s: %s", symbol, pos_side, e)
            return False

    # ── Paper mode ───────────────────────────────────────────────────────────

    async def _paper_order(self, symbol: str, side: str, amount: float, pos_side: str,
                           tp_price, sl_price) -> OrderResult:
        ticker = await self.fetch_ticker(symbol)
        price = float(ticker["last"])
        key = f"{symbol}||{pos_side}"
        is_close = (pos_side == "long" and side == "sell") or (pos_side == "short" and side == "buy")

        if is_close:
            pos = self._paper_positions.get(key)
            if pos:
                closed_amt = min(amount, pos["amount"])
                pnl_mult = 1 if pos_side == "long" else -1
                pnl = pnl_mult * (price - pos["entry"]) * closed_amt
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0.0) + pnl
                pos["amount"] -= closed_amt
                if pos["amount"] <= 1e-9:
                    del self._paper_positions[key]
            return OrderResult(str(uuid.uuid4()), symbol, side, amount, price, "closed")

        existing = self._paper_positions.get(key)
        if existing:
            existing["amount"] += amount
        else:
            self._paper_positions[key] = {"entry": price, "amount": amount,
                                          "tp": tp_price, "sl": sl_price}
        return OrderResult(str(uuid.uuid4()), symbol, side, amount, price, "open")
