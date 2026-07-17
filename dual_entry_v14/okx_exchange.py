"""OKX Perpetual Swap implementation of ExchangeInterface (ccxt).

Carries every hard-won lesson from the previous bot:
  - contract-size resolution refuses to trade rather than guessing
  - amounts FLOOR to lot steps (never round up at leverage)
  - fill actuals (avgPx / fee / pnl) read back after placement; resolution
    is best-effort and NEVER raises once an order is live
  - hedge-mode posSide on every order; attached native SL/TP
  - paper mode mirrors live quantization + fees exactly
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from typing import Optional

import ccxt.async_support as ccxt

from .config import Config
from .interfaces import (ExchangeInterface, ExchangeStateSnapshot, MarketRules,
                         OrderResult, PositionInfo)
from .models import Candle

logger = logging.getLogger("dual_entry.okx")

_FALLBACK_CT = {
    "BTC/USDT:USDT": 0.01, "ETH/USDT:USDT": 0.1, "XAU/USDT:USDT": 1.0,
    "XAG/USDT:USDT": 1.0, "SOL/USDT:USDT": 1.0, "XRP/USDT:USDT": 100.0,
}


class OKXExchange(ExchangeInterface):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paper = cfg.paper
        self._leverage_set: set = set()
        self._rules_cache: dict = {}
        self._paper_balance = 10_000.0
        self._paper_positions: dict = {}      # symbol -> dict
        self._paper_orders: dict = {}         # client_order_id -> OrderResult
        self._x = ccxt.okx({
            "apiKey": cfg.okx_api_key, "secret": cfg.okx_secret,
            "password": cfg.okx_passphrase, "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    async def close(self) -> None:
        try:
            await self._x.close()
        except Exception:
            pass

    # ── data ─────────────────────────────────────────────────────────────────

    async def get_closed_candles(self, symbol: str, timeframe: str, limit: int) -> list:
        from .config import TF_MS
        last_err = None
        for attempt in range(2):
            try:
                raw = await self._x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit + 1)
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.0)
        else:
            raise last_err
        now_ms = int(time.time() * 1000)
        out = [Candle(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                      float(r[5] or 0.0)) for r in raw]
        # drop the still-forming bar
        step = TF_MS[timeframe]
        return [c for c in out if c.timestamp + step <= now_ms]

    async def get_market_rules(self, symbol: str) -> MarketRules:
        if symbol in self._rules_cache:
            return self._rules_cache[symbol]
        ct, lot, min_qty, tick, min_notional = 0.0, 1.0, 1.0, 0.0, 0.0
        try:
            await self._x.load_markets()
            m = self._x.market(symbol)
            ct = float(m.get("contractSize") or 0.0)
            prec = (m.get("precision") or {}).get("amount")
            if prec:
                lot = float(prec)
            lim = (m.get("limits") or {})
            min_qty = float(((lim.get("amount") or {}).get("min")) or lot)
            tick = float(((m.get("precision") or {}).get("price")) or 0.0)
            min_notional = float(((lim.get("cost") or {}).get("min")) or 0.0)
        except Exception as e:
            logger.warning("[RULES] market load failed for %s: %s", symbol, e)
        if ct <= 0:
            ct = _FALLBACK_CT.get(symbol, 0.0)
        if ct <= 0:
            raise ValueError(f"Cannot resolve contract size for {symbol} — refusing to trade")
        rules = MarketRules(symbol, contract_size=ct, lot_step=lot, min_qty=min_qty,
                            tick_size=tick, min_notional=min_notional)
        self._rules_cache[symbol] = rules
        return rules

    async def get_state(self, symbol: str) -> ExchangeStateSnapshot:
        snap = ExchangeStateSnapshot()
        try:
            t = await self._x.fetch_ticker(symbol)
            snap.last_price = float(t.get("last") or 0.0)
            bid, ask = t.get("bid"), t.get("ask")
            if bid and ask and snap.last_price > 0:
                snap.spread_pct = (float(ask) - float(bid)) / snap.last_price
        except Exception as e:
            logger.warning("[STATE] ticker failed %s: %s", symbol, e)
        if self.paper:
            snap.equity = self._paper_balance
            snap.free_margin = self._paper_balance
            p = self._paper_positions.get(symbol)
            if p:
                snap.positions = [PositionInfo(symbol, p["direction"], p["qty"],
                                               p["entry"], attached_sl=p.get("sl"),
                                               attached_tp=p.get("tp"))]
            return snap
        try:
            bal = await self._x.fetch_balance()
            usdt = bal.get("USDT", {})
            snap.equity = float(usdt.get("total") or 0.0)
            snap.free_margin = float(usdt.get("free") or snap.equity)
        except Exception as e:
            logger.warning("[STATE] balance failed: %s", e)
        try:
            poss = await self._x.fetch_positions([symbol])
            for p in poss:
                qty_ct = float(p.get("contracts") or 0.0)
                if qty_ct <= 0:
                    continue
                ct = float(p.get("contractSize") or 1.0)
                direction = "LONG" if p.get("side") == "long" else "SHORT"
                snap.positions.append(PositionInfo(
                    symbol, direction, qty_ct * ct, float(p.get("entryPrice") or 0.0),
                    unrealized_pnl=float(p.get("unrealizedPnl") or 0.0)))
        except Exception as e:
            logger.warning("[STATE] positions failed %s: %s", symbol, e)
        try:
            oo = await self._x.fetch_open_orders(symbol)
            snap.open_orders = [OrderResult(str(o.get("id")), str(o.get("clientOrderId") or ""),
                                            symbol, str(o.get("side")), "open")
                                for o in oo]
        except Exception as e:
            logger.warning("[STATE] open orders failed %s: %s", symbol, e)
        return snap

    async def get_all_open_positions(self) -> list:
        if self.paper:
            return [PositionInfo(s, p["direction"], p["qty"], p["entry"])
                    for s, p in self._paper_positions.items()]
        out = []
        try:
            poss = await self._x.fetch_positions()
            for p in poss:
                qty_ct = float(p.get("contracts") or 0.0)
                if qty_ct <= 0:
                    continue
                ct = float(p.get("contractSize") or 1.0)
                out.append(PositionInfo(str(p.get("symbol")),
                                        "LONG" if p.get("side") == "long" else "SHORT",
                                        qty_ct * ct, float(p.get("entryPrice") or 0.0)))
        except Exception as e:
            logger.warning("[STATE] fetch all positions failed: %s", e)
        return out

    # ── orders ───────────────────────────────────────────────────────────────

    async def _ensure_leverage(self, symbol: str):
        if self.paper or symbol in self._leverage_set:
            return
        try:
            for side in ("long", "short"):
                await self._x.set_leverage(self.cfg.leverage, symbol,
                                           params={"mgnMode": self.cfg.margin_mode,
                                                   "posSide": side})
            self._leverage_set.add(symbol)
        except Exception as e:
            logger.error("[SETUP] set_leverage failed %s: %s", symbol, e)
            raise

    async def place_market_order(self, symbol: str, side: str, contracts: float,
                                 direction: str, client_order_id: str,
                                 sl_price: Optional[float] = None,
                                 tp_price: Optional[float] = None) -> OrderResult:
        rules = await self.get_market_rules(symbol)
        # floor to lot step, never up
        lot = rules.lot_step or 1.0
        contracts = math.floor(contracts / lot + 1e-9) * lot
        if contracts < max(rules.min_qty, lot) - 1e-9:
            return OrderResult("", client_order_id, symbol, side, "rejected")

        if self.paper:
            return await self._paper_fill(symbol, side, contracts, direction,
                                          client_order_id, rules, sl_price, tp_price)

        await self._ensure_leverage(symbol)
        params: dict = {"tdMode": self.cfg.margin_mode,
                        "posSide": "long" if direction == "LONG" else "short",
                        "clOrdId": client_order_id}
        if sl_price or tp_price:
            algo: dict = {}
            if tp_price:
                algo.update(tpTriggerPx=str(round(tp_price, 6)), tpOrdPx="-1",
                            tpTriggerPxType="last")
            if sl_price:
                algo.update(slTriggerPx=str(round(sl_price, 6)), slOrdPx="-1",
                            slTriggerPxType="last")
            params["attachAlgoOrds"] = [algo]
        raw = await self._x.create_order(symbol, "market", side, float(contracts),
                                         None, params=params)
        oid = str(raw.get("id") or uuid.uuid4())
        avg, fee, pnl, filled_ct = 0.0, 0.0, 0.0, contracts
        try:
            avg, fee, pnl, filled_ct = await self._resolve_fill(symbol, oid, raw, rules)
        except Exception as e:
            logger.warning("[ORDER] fill resolution failed %s %s (order live): %s",
                           symbol, oid, e)
        return OrderResult(oid, client_order_id, symbol, side, "filled",
                           filled_qty=(filled_ct or contracts) * rules.contract_size,
                           avg_price=avg, fee_cost=fee, realized_pnl=pnl)

    async def _resolve_fill(self, symbol: str, oid: str, raw: dict, rules: MarketRules):
        def _ext(o: dict):
            info = o.get("info") or {}
            avg = float(o.get("average") or info.get("avgPx") or 0.0) or 0.0
            filled = float(o.get("filled") or info.get("accFillSz") or 0.0) or 0.0
            fee_obj = o.get("fee") or {}
            fee = fee_obj.get("cost", info.get("fee"))
            fee = abs(float(fee)) if fee not in (None, "") else 0.0
            pnl = info.get("pnl")
            pnl = float(pnl) if pnl not in (None, "") else 0.0
            return avg, fee, pnl, filled
        avg, fee, pnl, filled = _ext(raw)
        if avg > 0 and fee > 0:
            return avg, fee, pnl, filled
        o = await self._x.fetch_order(oid, symbol)
        a2, f2, p2, fl2 = _ext(o)
        return (a2 or avg), (f2 or fee), (p2 or pnl), (fl2 or filled)

    async def amend_protection(self, symbol: str, direction: str, quantity: float,
                               sl_price: Optional[float],
                               tp_price: Optional[float]) -> bool:
        if self.paper:
            p = self._paper_positions.get(symbol)
            if p:
                if sl_price is not None:
                    p["sl"] = sl_price
                if tp_price is not None:
                    p["tp"] = tp_price
            return True
        try:
            m = self._x.market(symbol)
            inst = m["id"]
            pos_side = "long" if direction == "LONG" else "short"
            # cancel stale algos on this leg, then re-arm as OCO
            for ord_type in ("oco", "conditional", "move_order_stop"):
                try:
                    resp = await self._x.privateGetTradeAlgosPending(
                        {"instId": inst, "ordType": ord_type})
                    for algo in (resp or {}).get("data", []):
                        if algo.get("posSide") == pos_side:
                            await self._x.privatePostTradeCancelAlgos(
                                [{"algoId": algo["algoId"], "instId": inst}])
                except Exception:
                    pass
            rules = await self.get_market_rules(symbol)
            contracts = max(rules.min_qty, round(quantity / rules.contract_size))
            req = {"instId": inst, "tdMode": self.cfg.margin_mode,
                   "side": "sell" if direction == "LONG" else "buy",
                   "posSide": pos_side, "sz": str(contracts)}
            if sl_price is not None:
                req.update(slTriggerPx=str(round(sl_price, 6)), slOrdPx="-1",
                           slTriggerPxType="last")
            if tp_price is not None:
                req.update(ordType="oco", tpTriggerPx=str(round(tp_price, 6)),
                           tpOrdPx="-1", tpTriggerPxType="last")
            else:
                req["ordType"] = "conditional"
            await self._x.privatePostTradeOrderAlgo(req)
            return True
        except Exception as e:
            logger.warning("[PROTECT] amend failed %s: %s", symbol, e)
            return False

    async def close_position(self, symbol: str, direction: str,
                             quantity: Optional[float] = None) -> OrderResult:
        side = "sell" if direction == "LONG" else "buy"
        cl_id = f"cl{uuid.uuid4().hex[:20]}"
        if self.paper:
            p = self._paper_positions.get(symbol)
            if not p:
                return OrderResult("", cl_id, symbol, side, "rejected")
            snap = await self.get_state(symbol)
            px = snap.last_price or p["entry"]
            qty = min(quantity or p["qty"], p["qty"])
            mult = 1 if direction == "LONG" else -1
            pnl = mult * (px - p["entry"]) * qty
            fee = qty * px * self.cfg.fee_rate
            self._paper_balance += pnl - fee
            p["qty"] -= qty
            if p["qty"] <= 1e-9:
                del self._paper_positions[symbol]
            return OrderResult(uuid.uuid4().hex, cl_id, symbol, side, "filled",
                               filled_qty=qty, avg_price=px, fee_cost=fee,
                               realized_pnl=pnl)
        rules = await self.get_market_rules(symbol)
        snap = await self.get_state(symbol)
        pos = snap.position_for(symbol)
        if pos is None:
            return OrderResult("", cl_id, symbol, side, "rejected")
        qty = min(quantity or pos.quantity, pos.quantity)
        contracts = math.floor(qty / rules.contract_size / (rules.lot_step or 1.0) + 1e-9) \
            * (rules.lot_step or 1.0)
        params = {"tdMode": self.cfg.margin_mode,
                  "posSide": "long" if direction == "LONG" else "short",
                  "clOrdId": cl_id}
        raw = await self._x.create_order(symbol, "market", side, float(contracts),
                                         None, params=params)
        oid = str(raw.get("id") or uuid.uuid4())
        avg, fee, pnl, filled = 0.0, 0.0, 0.0, contracts
        try:
            avg, fee, pnl, filled = await self._resolve_fill(symbol, oid, raw, rules)
        except Exception as e:
            logger.warning("[CLOSE] fill resolution failed %s: %s", symbol, e)
        return OrderResult(oid, cl_id, symbol, side, "filled",
                           filled_qty=(filled or contracts) * rules.contract_size,
                           avg_price=avg, fee_cost=fee, realized_pnl=pnl)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.paper:
            return True
        try:
            await self._x.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning("[CANCEL] failed %s %s: %s", symbol, order_id, e)
            return False

    async def find_order_by_client_id(self, symbol: str,
                                      client_order_id: str) -> Optional[OrderResult]:
        if self.paper:
            return self._paper_orders.get(client_order_id)
        try:
            o = await self._x.fetch_order(None, symbol,
                                          params={"clOrdId": client_order_id})
            if o:
                return OrderResult(str(o.get("id")), client_order_id, symbol,
                                   str(o.get("side")), str(o.get("status") or "open"),
                                   filled_qty=float(o.get("filled") or 0.0),
                                   avg_price=float(o.get("average") or 0.0))
        except Exception:
            return None
        return None

    # ── paper fills ──────────────────────────────────────────────────────────

    async def _paper_fill(self, symbol, side, contracts, direction, cl_id, rules,
                          sl, tp) -> OrderResult:
        if cl_id in self._paper_orders:            # idempotency
            return self._paper_orders[cl_id]
        snap = await self.get_state(symbol)
        px = snap.last_price
        if px <= 0:
            return OrderResult("", cl_id, symbol, side, "rejected")
        qty = contracts * rules.contract_size
        fee = qty * px * self.cfg.fee_rate
        self._paper_balance -= fee
        self._paper_positions[symbol] = {"direction": direction, "qty": qty,
                                         "entry": px, "sl": sl, "tp": tp}
        res = OrderResult(uuid.uuid4().hex, cl_id, symbol, side, "filled",
                          filled_qty=qty, avg_price=px, fee_cost=fee)
        self._paper_orders[cl_id] = res
        return res
