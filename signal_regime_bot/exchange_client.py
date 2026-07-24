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
import math
import random
import uuid
from dataclasses import dataclass
from typing import Optional

import aiohttp
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
    amount: float          # base-asset size actually filled
    price: float           # legacy: same as avg_price (kept for callers)
    status: str
    avg_price: float = 0.0     # OKX avgPx — the ACTUAL average fill price
    fee_cost: float = 0.0      # OKX fee for THIS fill, as a positive USDT cost
    realized_pnl: float = 0.0  # OKX realized trading PnL for THIS order (pre-fee); 0 on opens


@dataclass
class Balance:
    asset: str
    free: float
    used: float
    total: float


class ExchangeClient:
    def __init__(self, api_key: str, api_secret: str, passphrase: str,
                paper: bool, leverage: int = 20, margin_mode: str = "isolated",
                fee_rate: float = 0.001):
        self.paper = paper
        self._leverage = leverage
        self._margin_mode = margin_mode
        self._fee_rate = fee_rate          # 0.10% per fill (open/close/TP/SL)
        self._leverage_set: set[str] = set()
        self._hedge_confirmed = False
        self._paper_balance: dict[str, float] = {"USDT": 10_000.0}
        self._paper_positions: dict[str, dict] = {}
        self._public_request_lock = asyncio.Lock()

        # CCXT's default HTTP timeout is too short for occasional OKX/Railway
        # latency spikes.  A slow public-data response must not crash the whole
        # symbol cycle or generate a Telegram error every 30 seconds.
        self._exchange_config = {
            "apiKey": api_key,
            "secret": api_secret,
            "password": passphrase,
            "enableRateLimit": True,
            "timeout": 30_000,
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True,
            },
        }
        self._exchange = ccxt.okx(self._exchange_config)

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
        # Try real market metadata FIRST even in paper mode — paper still
        # talks to the live public API for tickers, so metadata is usually
        # loadable and paper sizing then matches live exactly.
        try:
            market = self._exchange.market(symbol)
            sz = market.get("contractSize")
            if sz:
                return float(sz)
        except Exception:
            pass
        fallback = _OKX_CONTRACT_SIZE.get(symbol, 0.0)
        if self.paper:
            return fallback or 1.0
        return fallback

    async def quantize_amount(self, symbol: str, base_amount: float) -> tuple[float, float]:
        """
        Convert a desired BASE-asset amount into (contracts, effective_base)
        exactly as OKX will fill it: floored to whole lot steps of the
        contract, NEVER rounded up (rounding up silently oversizes risk at
        20x). Returns (0.0, 0.0) if the amount is below one tradeable lot.

        This is the single source of truth for order sizing — live orders,
        paper fills, and the Position bookkeeping all use the value returned
        here, so the size the bot reports is byte-for-byte the size OKX
        actually trades (the old int(round(...)) path could differ from the
        recorded amount by up to half a contract).
        """
        ct = await self.contract_size(symbol)
        if not ct or ct <= 0:
            raise ValueError(f"Cannot resolve contract size for {symbol} — refusing to size")
        lot = 1.0      # lot step, in contracts (OKX lotSz; often 1, sometimes 0.1/0.01)
        min_ct = 1.0   # minimum order, in contracts (OKX minSz)
        try:
            market = self._exchange.market(symbol)
            prec = (market.get("precision") or {}).get("amount")
            if prec:
                lot = float(prec)
            mn = ((market.get("limits") or {}).get("amount") or {}).get("min")
            if mn:
                min_ct = float(mn)
        except Exception:
            pass
        steps = math.floor(base_amount / ct / lot + 1e-9)
        contracts = steps * lot
        if contracts < max(min_ct, lot) - 1e-9:
            return 0.0, 0.0
        return contracts, contracts * ct

    # ── Market data ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_retryable_network_error(exc: Exception) -> bool:
        """Only retry transport/rate-limit availability failures.

        Authentication, invalid-symbol and bad-request errors must surface
        immediately; retrying them only creates noise and delays diagnosis.
        Using class names as a final fallback keeps this compatible across
        CCXT minor releases where the async module may expose subclasses from
        a slightly different import path.
        """
        retryable_types = tuple(
            t for t in (
                getattr(ccxt, "RequestTimeout", None),
                getattr(ccxt, "NetworkError", None),
                getattr(ccxt, "ExchangeNotAvailable", None),
                getattr(ccxt, "DDoSProtection", None),
                getattr(ccxt, "RateLimitExceeded", None),
            ) if isinstance(t, type)
        )
        if retryable_types and isinstance(exc, retryable_types):
            return True
        return exc.__class__.__name__ in {
            "RequestTimeout", "NetworkError", "ExchangeNotAvailable",
            "DDoSProtection", "RateLimitExceeded", "TimeoutError",
        }

    @staticmethod
    def _okx_bar(timeframe: str) -> str:
        mapping = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "1H", "2h": "2H", "4h": "4H",
            "6h": "6H", "12h": "12H", "1d": "1Dutc",
        }
        return mapping.get(timeframe, timeframe)

    async def _fetch_ohlcv_rest_fallback(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list:
        """Independent public REST fallback when CCXT's session times out.

        The endpoint and payload are OKX's native candle API.  Returning the
        same six-column, oldest-to-newest shape as CCXT means the rest of the
        bot does not need a separate code path.
        """
        try:
            market = self._exchange.market(symbol)
            inst_id = market.get("id") or symbol.replace("/", "-").replace(":USDT", "-SWAP")
        except Exception:
            base = symbol.split("/")[0]
            quote = symbol.split("/")[1].split(":")[0] if "/" in symbol else "USDT"
            inst_id = f"{base}-{quote}-SWAP"

        params = {
            "instId": inst_id,
            "bar": self._okx_bar(timeframe),
            "limit": str(max(1, min(int(limit), 300))),
        }
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=25)
        headers = {"User-Agent": "RegimeBiasBot/3.0"}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(
                "https://www.okx.com/api/v5/market/candles", params=params,
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json(content_type=None)

        if str(payload.get("code", "0")) != "0":
            raise RuntimeError(
                f"OKX candle fallback error code={payload.get('code')} msg={payload.get('msg')}"
            )
        rows = payload.get("data") or []
        parsed = []
        for row in rows:
            if len(row) < 6:
                continue
            parsed.append([
                int(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5]),
            ])
        parsed.sort(key=lambda x: x[0])
        return parsed

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> list:
        """Fetch candles with bounded retry, backoff and an independent fallback.

        A timeout is a data-transport issue, not a strategy error.  The caller
        can then use its last-known-good cache rather than stopping analysis for
        every timeframe and flooding Telegram with stack traces.
        """
        last_err: Exception | None = None
        attempts = 4
        async with self._public_request_lock:
            for attempt in range(1, attempts + 1):
                try:
                    return await self._exchange.fetch_ohlcv(
                        symbol, timeframe=timeframe, limit=limit,
                    )
                except Exception as exc:
                    last_err = exc
                    if not self._is_retryable_network_error(exc):
                        raise
                    if attempt < attempts:
                        delay = min(8.0, 0.8 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.35)
                        logger.warning(
                            "[DATA] fetch_ohlcv %s %s attempt %d/%d failed (%s); retry in %.1fs",
                            symbol, timeframe, attempt, attempts,
                            exc.__class__.__name__, delay,
                        )
                        await asyncio.sleep(delay)

            logger.warning(
                "[DATA] CCXT candle fetch exhausted for %s %s; trying native OKX REST fallback",
                symbol, timeframe,
            )
            try:
                rows = await self._fetch_ohlcv_rest_fallback(symbol, timeframe, limit)
                if rows:
                    logger.info("[DATA] native OKX fallback recovered %s %s", symbol, timeframe)
                    return rows
            except Exception as fallback_err:
                logger.error(
                    "[DATA] native fallback failed %s %s: %s",
                    symbol, timeframe, fallback_err,
                )
                if last_err is None:
                    last_err = fallback_err

        assert last_err is not None
        raise last_err

    async def fetch_ticker(self, symbol: str) -> dict:
        # Paper mode still needs a live price reference — always hit the real ticker.
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                async with self._public_request_lock:
                    return await self._exchange.fetch_ticker(symbol)
            except Exception as exc:
                last_err = exc
                if not self._is_retryable_network_error(exc) or attempt == 2:
                    raise
                await asyncio.sleep(0.8 * (2 ** attempt))
        assert last_err is not None
        raise last_err

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

    async def fetch_trade_history(self, since_ms: int, known_symbols: list[str]) -> list[dict]:
        """Authoritative closed-trade history straight from OKX's own ledger
        (GET /api/v5/account/positions-history) — one row per FULLY closed
        position. `realizedPnl` there is OKX's own net figure = pnl + fee +
        fundingFee (+liqPenalty), i.e. exactly what the OKX app shows as that
        position's PnL. /stats must never compute this number itself — this
        is the one place fee-accurate numbers can't drift from OKX, including
        across a redeploy that wipes this process's in-memory trade log.
        Returns [] in paper mode (no real OKX account to query) or on any
        API failure — callers fall back to their own in-memory record."""
        if self.paper:
            return []
        id_to_symbol: dict[str, str] = {}
        for sym in known_symbols:
            try:
                id_to_symbol[self._exchange.market(sym)["id"]] = sym
            except Exception:
                continue
        out: list[dict] = []
        after: Optional[str] = None
        for _ in range(20):   # 20 * 100 rows = far beyond any realistic backfill window
            params: dict = {"limit": "100", "instType": "SWAP"}
            if after:
                params["after"] = after
            try:
                raw = await self._exchange.privateGetAccountPositionsHistory(params)
            except Exception as e:
                logger.warning("[STATS] positions-history fetch failed: %s", e)
                break
            rows = (raw or {}).get("data") or []
            if not rows:
                break
            stop = False
            for r in rows:
                try:
                    u_time = int(r.get("uTime") or 0)
                except (TypeError, ValueError):
                    continue
                if u_time < since_ms:
                    stop = True
                    continue
                symbol = id_to_symbol.get(r.get("instId", ""))
                if symbol is None:
                    continue   # a symbol this bot doesn't currently track — skip
                try:
                    pnl = float(r.get("pnl") or 0.0)
                    fee = float(r.get("fee") or 0.0)            # negative = cost
                    funding = float(r.get("fundingFee") or 0.0)
                    liq = float(r.get("liqPenalty") or 0.0)
                    realized = r.get("realizedPnl")
                    realized = (float(realized) if realized not in (None, "")
                               else (pnl + fee + funding + liq))
                    out.append({
                        "symbol": symbol, "side": str(r.get("direction", "")),
                        "open_avg_px": float(r.get("openAvgPx") or 0.0),
                        "close_avg_px": float(r.get("closeAvgPx") or 0.0),
                        "pnl": realized, "close_time_ms": u_time,
                        "open_time_ms": int(r.get("cTime") or 0),
                        "pos_id": str(r.get("posId", "")),
                    })
                except (TypeError, ValueError) as e:
                    logger.warning("[STATS] skipping malformed positions-history row: %s", e)
            after = rows[-1].get("uTime")
            if stop or len(rows) < 100 or not after:
                break
        out.sort(key=lambda x: x["close_time_ms"])
        return out

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

    async def fetch_position_details(self, symbol: str, side: str) -> Optional[dict]:
        """Amount + entry price of an open OKX position, for restart reconciliation."""
        if self.paper:
            key = f"{symbol}||{side}"
            pos = self._paper_positions.get(key)
            if pos:
                return {"amount": float(pos["amount"]), "entry_price": float(pos["entry"])}
            return None
        try:
            positions = await self._exchange.fetch_positions([symbol])
            for p in positions:
                if p.get("side") == side and float(p.get("contracts", 0)) > 0:
                    ct_val = float(p.get("contractSize") or 1.0)
                    entry = p.get("entryPrice")
                    if entry is None:
                        return None
                    return {"amount": float(p["contracts"]) * ct_val, "entry_price": float(entry)}
        except Exception as e:
            logger.warning("[DATA] fetch_position_details failed %s %s: %s", symbol, side, e)
        return None

    async def fetch_attached_stops(self, symbol: str, pos_side: str) -> tuple:
        """
        Current SL/TP prices from OKX's pending algo orders for this leg — used
        to reconstruct a Position after a restart (we don't know the strategy's
        original TP1, so an adopted position resumes as single-TP: whatever
        SL/TP the exchange currently has attached is treated as (SL, TP2)).
        Returns (sl_price, tp_price), either may be None if not found.
        """
        if self.paper:
            return None, None
        try:
            market = self._exchange.market(symbol)
            inst_id = market["id"]
        except Exception as e:
            logger.warning("[RECONCILE] market lookup failed for %s: %s", symbol, e)
            return None, None

        sl_price, tp_price = None, None
        for ord_type in ("oco", "conditional", "move_order_stop"):
            try:
                resp = await self._exchange.privateGetTradeAlgosPending({
                    "instId": inst_id, "ordType": ord_type,
                })
                for algo in (resp or {}).get("data", []):
                    if algo.get("posSide") != pos_side:
                        continue
                    sl_raw = algo.get("slTriggerPx")
                    tp_raw = algo.get("tpTriggerPx")
                    if sl_raw not in (None, "", "0", "0.0") and sl_price is None:
                        sl_price = float(sl_raw)
                    if tp_raw not in (None, "", "0", "0.0") and tp_price is None:
                        tp_price = float(tp_raw)
            except Exception as e:
                logger.warning("[RECONCILE] pending-algos query (%s) failed: %s", ord_type, e)
        return sl_price, tp_price

    # ── Orders ───────────────────────────────────────────────────────────────

    async def create_order(self, symbol: str, side: str, amount: float,
                           pos_side: str, tp_price: Optional[float] = None,
                           sl_price: Optional[float] = None,
                           reduce_only: bool = False) -> OrderResult:
        if self.paper:
            return await self._paper_order(symbol, side, amount, pos_side, tp_price, sl_price)

        await self._ensure_leverage(symbol)

        contracts, effective_base = await self.quantize_amount(symbol, amount)
        if contracts <= 0:
            raise ValueError(
                f"Order too small for {symbol}: {amount:.6f} base is below one "
                f"tradeable lot. Increase risk_per_trade or balance."
            )
        ct_val = effective_base / contracts   # base units per contract (for fill parsing)
        if abs(effective_base - amount) / max(amount, 1e-12) > 0.005:
            logger.info("[ORDER] %s quantized %.8f -> %.8f base (%.4g contracts)",
                       symbol, amount, effective_base, contracts)

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

        # The order is now LIVE on OKX. Resolving the fill details (avgPx/fee/
        # pnl) is best-effort ONLY — it must NEVER raise, or the caller would
        # treat a real, placed order as a failure and never track it (leaving a
        # position on OKX the bot flies blind on). Any failure -> fall back to
        # the placement response / estimates.
        order_id = str(raw.get("id") or uuid.uuid4())
        try:
            avg, fee_cost, realized, filled_base = await self._resolve_fill(symbol, order_id, raw, ct_val)
        except Exception as e:
            logger.warning("[ORDER] fill resolution failed for %s %s (order is placed) — "
                          "using fallback: %s", symbol, order_id, e)
            avg, fee_cost, realized, filled_base = 0.0, 0.0, 0.0, 0.0
        return OrderResult(
            order_id=order_id, symbol=symbol, side=side,
            amount=(filled_base or effective_base),
            price=avg or (raw.get("price") or 0.0),
            status=raw.get("status", "open"),
            avg_price=avg, fee_cost=fee_cost, realized_pnl=realized,
        )

    async def _resolve_fill(self, symbol: str, order_id: str, raw: dict,
                            ct_val: float) -> tuple[float, float, float, float]:
        """Read the ACTUAL fill back from OKX: (avg_price, fee_cost>=0,
        realized_pnl, filled_base). A market order's create response often
        lacks avgPx/fee/pnl, so re-fetch the settled order when they're
        missing. All values fall back to 0.0 on any failure — the caller
        then uses its own estimate rather than crashing."""
        def _extract(o: dict) -> tuple[float, float, float, float]:
            info = o.get("info") or {}
            avg = float(o.get("average") or info.get("avgPx") or 0.0) or 0.0
            filled_ct = float(o.get("filled") or info.get("accFillSz") or 0.0) or 0.0
            fee_obj = o.get("fee") or {}
            fee_cost = fee_obj.get("cost")
            if fee_cost is None:
                fee_cost = info.get("fee")
            fee_cost = abs(float(fee_cost)) if fee_cost not in (None, "") else 0.0
            realized = info.get("pnl")
            realized = float(realized) if realized not in (None, "") else 0.0
            return avg, fee_cost, realized, filled_ct * ct_val

        avg, fee_cost, realized, filled_base = _extract(raw)
        if avg > 0 and fee_cost > 0:
            return avg, fee_cost, realized, filled_base
        # settle: re-fetch the order once (market fills are near-instant)
        try:
            o = await self._exchange.fetch_order(order_id, symbol)
            a2, f2, r2, b2 = _extract(o)
            return (a2 or avg), (f2 or fee_cost), (r2 or realized), (b2 or filled_base)
        except Exception as e:
            logger.warning("[ORDER] fill re-fetch failed for %s %s: %s", symbol, order_id, e)
            return avg, fee_cost, realized, filled_base

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
        """Paper fills mirror live exactly: the amount is quantized to OKX
        contract lots (same quantize_amount as live orders) and every fill
        pays fee_rate on its notional — open, close, TP and SL alike."""
        ticker = await self.fetch_ticker(symbol)
        price = float(ticker["last"])
        key = f"{symbol}||{pos_side}"
        is_close = (pos_side == "long" and side == "sell") or (pos_side == "short" and side == "buy")

        contracts, eff_amount = await self.quantize_amount(symbol, amount)

        if is_close:
            pos = self._paper_positions.get(key)
            closed_amt = 0.0
            pnl = fee = 0.0
            if pos:
                closed_amt = min(eff_amount if eff_amount > 0 else pos["amount"], pos["amount"])
                pnl_mult = 1 if pos_side == "long" else -1
                pnl = pnl_mult * (price - pos["entry"]) * closed_amt   # realized (pre-fee)
                fee = closed_amt * price * self._fee_rate
                self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0.0) + pnl - fee
                pos["amount"] -= closed_amt
                if pos["amount"] <= 1e-9:
                    del self._paper_positions[key]
            return OrderResult(str(uuid.uuid4()), symbol, side, closed_amt, price, "closed",
                               avg_price=price, fee_cost=fee, realized_pnl=pnl)

        if eff_amount <= 0:
            raise ValueError(
                f"Order too small for {symbol}: {amount:.6f} base is below one tradeable lot.")
        open_fee = eff_amount * price * self._fee_rate
        self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0.0) - open_fee
        existing = self._paper_positions.get(key)
        if existing:
            existing["amount"] += eff_amount
        else:
            self._paper_positions[key] = {"entry": price, "amount": eff_amount,
                                          "tp": tp_price, "sl": sl_price}
        return OrderResult(str(uuid.uuid4()), symbol, side, eff_amount, price, "open",
                           avg_price=price, fee_cost=open_fee, realized_pnl=0.0)
