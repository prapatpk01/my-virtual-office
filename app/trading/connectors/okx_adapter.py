"""
OKX Adapter — production connector for OKX perpetual futures.

Implements both interfaces:
  1. BaseConnector (async, used by bot.py / run_bot.py)
  2. Synchronous execute() / fetch_open_position() (used by AdaptiveTradingBot)

Supports:
  - Hedge mode (posSide = long/short) required by OKX SWAP
  - Exchange-attached stop-loss on open orders (SL survives bot restart)
  - Retry + exponential backoff with jitter (network/rate-limit errors only)
  - Fatal-error fast-fail (auth, insufficient funds, invalid order)
  - Fill confirmation after order submission
  - Market data: OHLCV, ticker, balance
  - Async wrappers for bot.py compatibility

Environment variables (fallbacks when not passed directly):
    OKX_API_KEY
    OKX_API_SECRET
    OKX_API_PASSPHRASE
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import ccxt.async_support as ccxt_async

from .base import OHLCV, Balance, BaseConnector, OrderResult

logger = logging.getLogger("okx_adapter")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ── Error classification ──────────────────────────────────────────────────────

_FATAL_SYNC = (
    ccxt.AuthenticationError,
    ccxt.PermissionDenied,
    ccxt.InvalidOrder,
    ccxt.InsufficientFunds,
    ccxt.BadRequest,
    ccxt.AccountSuspended,
)

_RETRYABLE_SYNC = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.DDoSProtection,
)

_FATAL_ASYNC = (
    ccxt_async.AuthenticationError,
    ccxt_async.PermissionDenied,
    ccxt_async.InvalidOrder,
    ccxt_async.InsufficientFunds,
    ccxt_async.BadRequest,
    ccxt_async.AccountSuspended,
)

_RETRYABLE_ASYNC = (
    ccxt_async.NetworkError,
    ccxt_async.ExchangeNotAvailable,
    ccxt_async.RateLimitExceeded,
    ccxt_async.DDoSProtection,
)


# ── OKX Adapter ───────────────────────────────────────────────────────────────

class OKXAdapter(BaseConnector):
    """
    Full OKX perpetual futures adapter.

    Async path (bot.py):
        adapter = OKXAdapter(api_key=..., api_secret=..., api_passphrase=..., paper=False)
        candles = await adapter.fetch_ohlcv("BTC/USDT:USDT", "15m", 300)
        result  = await adapter.create_order("BTC/USDT:USDT", "buy", 1.0, sl=..., position_side="LONG")

    Sync path (AdaptiveTradingBot):
        result = adapter.execute("OPEN_LONG", {"entry": ..., "sl": ..., "size": ...})
        pos    = adapter.fetch_open_position("BTC/USDT:USDT")
    """

    DEFAULT_TD_MODE    = "cross"
    MAX_RETRIES        = 5
    BASE_DELAY         = 1.0     # seconds
    MAX_DELAY          = 30.0    # seconds
    FILL_POLL_INTERVAL = 0.5     # seconds between fill-check polls
    FILL_POLL_TIMEOUT  = 15.0    # seconds before giving up on fill confirmation

    def __init__(
        self,
        api_key:        Optional[str] = None,
        api_secret:     Optional[str] = None,
        api_passphrase: Optional[str] = None,
        paper:          bool           = True,   # True = demo/sandbox
        td_mode:        str            = DEFAULT_TD_MODE,
        symbol:         Optional[str]  = None,   # default symbol for sync path
        max_retries:    int            = MAX_RETRIES,
        base_delay:     float          = BASE_DELAY,
        max_delay:      float          = MAX_DELAY,
        leverage:       int            = 10,
    ):
        super().__init__(
            api_key    = api_key    or os.environ.get("OKX_API_KEY",        ""),
            api_secret = api_secret or os.environ.get("OKX_API_SECRET",     ""),
            paper      = paper,
        )
        self._passphrase = api_passphrase or os.environ.get("OKX_API_PASSPHRASE", "")
        self.td_mode     = td_mode
        self.symbol      = symbol
        self.max_retries = max_retries
        self.base_delay  = base_delay
        self.max_delay   = max_delay
        self.leverage    = leverage
        self._swap_ready_symbols: set = set()  # symbols that have had leverage/mode set

        cfg = {
            "apiKey":          self.api_key,
            "secret":          self.api_secret,
            "password":        self._passphrase,
            "enableRateLimit": True,
            "options":         {"defaultType": "swap"},
        }

        # Sync exchange (for AdaptiveTradingBot / synchronous path)
        self._ex: ccxt.okx = ccxt.okx(cfg)

        # Async exchange (for bot.py / BaseConnector path)
        self._aex: ccxt_async.okx = ccxt_async.okx(cfg)

        if paper:
            self._ex.set_sandbox_mode(True)
            self._aex.set_sandbox_mode(True)
            logger.warning("[OKX] Sandbox/demo mode — orders go to OKX demo trading")

    # ── Sync retry wrapper ───────────────────────────────────────────────────

    def _call(self, fn, *args, label: str = "", **kwargs):
        """Synchronous call with retry/backoff."""
        name = label or getattr(fn, "__name__", str(fn))
        for attempt in range(1, self.max_retries + 2):
            try:
                return fn(*args, **kwargs)
            except _FATAL_SYNC as e:
                logger.error("[OKX][FATAL] %s: %s", name, e)
                raise
            except _RETRYABLE_SYNC as e:
                if attempt > self.max_retries:
                    logger.error("[OKX][EXHAUSTED] %s after %d retries: %s",
                                 name, self.max_retries, e)
                    raise
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                delay += random.uniform(0, delay * 0.1)
                logger.warning("[OKX][RETRY %d/%d] %s: %s — wait %.1fs",
                               attempt, self.max_retries, name, e, delay)
                time.sleep(delay)
            except ccxt.ExchangeError as e:
                logger.error("[OKX][ERROR] %s unexpected: %s", name, e)
                raise

    # ── Async retry wrapper ──────────────────────────────────────────────────

    async def _acall(self, fn, *args, label: str = "", **kwargs):
        """Asynchronous call with retry/backoff."""
        name = label or getattr(fn, "__name__", str(fn))
        for attempt in range(1, self.max_retries + 2):
            try:
                return await fn(*args, **kwargs)
            except _FATAL_ASYNC as e:
                logger.error("[OKX][FATAL] %s: %s", name, e)
                raise
            except _RETRYABLE_ASYNC as e:
                if attempt > self.max_retries:
                    logger.error("[OKX][EXHAUSTED] %s after %d retries: %s",
                                 name, self.max_retries, e)
                    raise
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                delay += random.uniform(0, delay * 0.1)
                logger.warning("[OKX][RETRY %d/%d] %s: %s — wait %.1fs",
                               attempt, self.max_retries, name, e, delay)
                await asyncio.sleep(delay)
            except ccxt_async.ExchangeError as e:
                logger.error("[OKX][ERROR] %s unexpected: %s", name, e)
                raise

    # ────────────────────────────────────────────────────────────────────────
    # BaseConnector  (async interface — used by bot.py)
    # ────────────────────────────────────────────────────────────────────────

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h",
                          limit: int = 200) -> List[OHLCV]:
        if self.paper:
            return []
        raw = await self._acall(
            self._aex.fetch_ohlcv, symbol, timeframe, None, limit,
            label=f"fetch_ohlcv({symbol},{timeframe})",
        )
        return [
            OHLCV(
                timestamp=int(bar[0]),
                open=float(bar[1]),
                high=float(bar[2]),
                low=float(bar[3]),
                close=float(bar[4]),
                volume=float(bar[5]),
            )
            for bar in raw
        ]

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        if self.paper:
            return {}
        return await self._acall(
            self._aex.fetch_ticker, symbol,
            label=f"fetch_ticker({symbol})",
        )

    async def create_order(
        self,
        symbol:        str,
        side:          str,          # "buy" | "sell"
        amount:        float,
        order_type:    str           = "market",
        price:         Optional[float] = None,
        tp:            Optional[float] = None,
        sl:            Optional[float] = None,
        position_side: str            = "LONG",   # "LONG" | "SHORT"
    ) -> OrderResult:
        """
        Async order placement with optional exchange-attached SL.
        position_side: OKX hedge-mode posSide ("long" | "short").
        """
        if self.paper:
            return self._paper_order(symbol, side, amount, price or 0.0)

        pos_side = position_side.lower()
        params: Dict[str, Any] = {
            "tdMode":  self.td_mode,
            "posSide": pos_side,
        }
        if sl is not None:
            params["stopLoss"] = {"triggerPrice": str(sl), "type": "market"}
        if tp is not None:
            params["takeProfit"] = {"triggerPrice": str(tp), "type": "market"}

        amount_str = float(await self._acall(
            self._aex.amount_to_precision, symbol, amount,
            label="amount_to_precision",
        ) or amount)

        raw = await self._acall(
            self._aex.create_order,
            symbol, order_type, side, amount_str, price, params,
            label=f"create_order({symbol},{side},{pos_side})",
        )
        order_id = raw.get("id", "")
        logger.info("[OKX] %s %s %s qty=%.4f order_id=%s",
                    "OPEN" if "open" not in side else "CLOSE",
                    pos_side.upper(), symbol, amount_str, order_id)

        # Confirm fill
        filled, fill_price = await self._await_fill_async(symbol, order_id)
        return OrderResult(
            order_id  = order_id,
            symbol    = symbol,
            side      = side,
            amount    = amount_str,
            price     = fill_price,
            filled    = filled,
            status    = "closed" if filled > 0 else "open",
            timestamp = int(raw.get("timestamp") or time.time() * 1000),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self.paper:
            return True
        try:
            await self._acall(
                self._aex.cancel_order, order_id, symbol,
                label=f"cancel_order({order_id})",
            )
            return True
        except Exception as e:
            logger.error("[OKX] cancel_order %s failed: %s", order_id, e)
            return False

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        if self.paper:
            return self._paper_orders[:]
        raw = await self._acall(
            self._aex.fetch_open_orders, symbol,
            label="fetch_open_orders",
        )
        return [self._to_order_result(o) for o in raw]

    async def fetch_balance(self) -> List[Balance]:
        if self.paper:
            return [Balance("USDT", 10_000, 0, 10_000)]
        raw = await self._acall(self._aex.fetch_balance, label="fetch_balance")
        out = []
        for asset, info in (raw or {}).items():
            if isinstance(info, dict) and "free" in info:
                out.append(Balance(
                    asset=asset,
                    free=float(info.get("free")  or 0),
                    used=float(info.get("used")  or 0),
                    total=float(info.get("total") or 0),
                ))
        return out

    async def close(self):
        """Close the async ccxt session. Call on shutdown."""
        try:
            await self._aex.close()
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────────────
    # Synchronous interface — used by AdaptiveTradingBot
    # ────────────────────────────────────────────────────────────────────────

    def execute(self, order_type: str, trade_info: Dict[str, Any]) -> Optional[Dict]:
        """
        Synchronous execution entry point for AdaptiveTradingBot.
        order_type: "OPEN_LONG" | "OPEN_SHORT" | "CLOSE_PARTIAL" | "CLOSE_FULL"
        """
        if order_type in ("OPEN_LONG", "OPEN_SHORT"):
            return self._sync_open(order_type, trade_info)
        if order_type in ("CLOSE_PARTIAL", "CLOSE_FULL"):
            return self._sync_close(order_type, trade_info)
        logger.error("[OKX] Unknown order_type: %s", order_type)
        return None

    def fetch_open_position(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """
        Return the first non-zero position for reconciliation.
        Returns the raw ccxt position dict, or None if flat.
        """
        sym = symbol or self.symbol
        if not sym:
            logger.warning("[OKX] fetch_open_position: no symbol provided")
            return None
        positions = self._call(
            self._ex.fetch_positions, [sym],
            label=f"fetch_positions({sym})",
        )
        for p in (positions or []):
            if float(p.get("contracts") or 0) != 0:
                return p
        return None

    def fetch_balance_usdt(self) -> float:
        """Return free USDT balance (sync)."""
        raw = self._call(self._ex.fetch_balance, label="fetch_balance_sync")
        return float((raw.get("USDT") or {}).get("free", 0) or 0)

    def cancel_all_open_orders_sync(self, symbol: Optional[str] = None) -> None:
        sym = symbol or self.symbol
        orders = self._call(self._ex.fetch_open_orders, sym, label="fetch_open_orders_sync")
        for o in (orders or []):
            try:
                self._call(self._ex.cancel_order, o["id"], sym,
                           label=f"cancel_order({o['id']})")
            except Exception as e:
                logger.warning("[OKX] cancel order %s failed: %s", o["id"], e)

    # ── Sync internals ───────────────────────────────────────────────────────

    def _ensure_swap_ready(self, sym: str) -> None:
        """Set hedge mode + leverage for sym before first order. Cached per symbol.
        Mirrors BinanceConnector._ensure_swap_ready for the synchronous path."""
        if sym in self._swap_ready_symbols:
            return
        try:
            self._call(self._ex.set_position_mode, True,
                       label=f"set_position_mode(hedge)")
        except ccxt.ExchangeError as e:
            # OKX returns an error if already in hedge mode — not fatal
            logger.debug("[OKX] set_position_mode (already set?): %s", e)
        try:
            lev = self.leverage
            self._call(
                self._ex.set_leverage, lev, sym,
                {"mgnMode": self.td_mode, "posSide": "long"},
                label=f"set_leverage({lev}, {sym}, long)",
            )
            self._call(
                self._ex.set_leverage, lev, sym,
                {"mgnMode": self.td_mode, "posSide": "short"},
                label=f"set_leverage({lev}, {sym}, short)",
            )
        except Exception as e:
            logger.warning("[OKX] set_leverage warning for %s: %s", sym, e)
        self._swap_ready_symbols.add(sym)
        logger.info("[OKX] Swap ready: %s (hedge mode + %dx leverage set)", sym, self.leverage)

    def _get_ct_val(self, sym: str) -> float:
        """Return contract size (ctVal) for a SWAP symbol — 1 contract = ctVal base coin.
        E.g. BTC/USDT:USDT → ctVal=0.01, so 1 contract = 0.01 BTC."""
        try:
            if not self._ex.markets:
                self._ex.load_markets()
            mkt  = self._ex.markets.get(sym, {})
            info = mkt.get("info", {})
            val  = float(info.get("ctVal") or mkt.get("contractSize") or 1)
            return val if val > 0 else 1.0
        except Exception as e:
            logger.warning("[OKX] _get_ct_val(%s) failed, defaulting to 1: %s", sym, e)
            return 1.0

    def _sync_open(self, order_type: str, trade_info: Dict[str, Any]) -> Dict:
        sym      = self.symbol or trade_info.get("symbol", "")
        side     = "buy"  if order_type == "OPEN_LONG" else "sell"
        pos_side = "long" if order_type == "OPEN_LONG" else "short"

        # Ensure hedge mode and leverage are configured before first order per symbol
        self._ensure_swap_ready(sym)

        # Convert base-currency coin quantity → OKX contracts (bot computes size in coins)
        ct_val         = self._get_ct_val(sym)
        raw_contracts  = float(trade_info["size"]) / ct_val
        contracts      = max(1, int(raw_contracts))
        amount         = float(self._ex.amount_to_precision(sym, contracts))

        # [RISK-FLOOR VISIBILITY] The exchange's 1-contract minimum can force a
        # position larger than the risk-based calc intended (e.g. small balance
        # + high-priced symbol → calculated size < 1 contract). When that
        # happens, actual risk taken exceeds the configured ADAPTIVE_RISK_PCT —
        # log it so this is visible instead of silently distorting risk.
        if raw_contracts < 1.0 and raw_contracts > 0:
            overshoot = (1.0 / raw_contracts - 1.0) * 100
            logger.warning(
                "[OKX] %s: risk-calc wanted %.3f contracts, floored to 1 "
                "(actual risk ≈ +%.0f%% over target — balance may be too small "
                "for this symbol's minimum lot at the configured risk %%)",
                sym, raw_contracts, overshoot,
            )

        params: Dict[str, Any] = {
            "tdMode":  self.td_mode,
            "posSide": pos_side,
        }
        sl = trade_info.get("sl")
        if sl is not None:
            params["stopLoss"] = {"triggerPrice": str(sl), "type": "market"}

        # Attach TP2 (the final/full-size target) as a real exchange-side order
        # so the position is protected even if the bot goes offline. TP1's 50%
        # partial close stays bot-managed — OKX's attach mechanism only supports
        # one take-profit leg per order, tied to the full position size, so it
        # can't express a partial-size leg. reduceOnly semantics mean this TP2
        # order safely closes whatever remains even after the bot's own TP1
        # partial close has already reduced the position.
        tp2 = trade_info.get("tp2")
        if tp2 is not None:
            params["takeProfit"] = {"triggerPrice": str(tp2), "type": "market"}

        raw = self._call(
            self._ex.create_order,
            sym, "market", side, amount, None, params,
            label=f"OPEN_{pos_side.upper()}({sym})",
        )
        order_id = raw.get("id", "")
        logger.info("[OKX] OPEN %s %s qty=%.4f SL=%s TP2=%s → order_id=%s",
                    pos_side.upper(), sym, amount, sl, tp2, order_id)

        filled, fill_price = self._await_fill_sync(sym, order_id)
        raw["_filled"]       = filled
        raw["_fill_price"]   = fill_price
        raw["_filled_coins"] = filled * ct_val   # actual size in coins for bot accounting
        return raw

    def _sync_close(self, order_type: str, trade_info: Dict[str, Any]) -> Dict:
        sym       = self.symbol or trade_info.get("symbol", "")
        direction = trade_info.get("direction", "LONG")
        side      = "sell" if direction == "LONG" else "buy"
        pos_side  = "long" if direction == "LONG" else "short"

        # Ensure hedge mode/leverage set (idempotent) before close
        self._ensure_swap_ready(sym)

        # Convert coin quantity → OKX contracts.
        # CLOSE_FULL uses the exchange's LIVE contract count, not the local
        # coin figure — int() truncation at open + partial-close truncation
        # otherwise drifts (e.g. open int(3.5)=3, TP1 closes int(1.75)=1,
        # final close int(1.75)=1 → 1 contract orphaned live on the exchange).
        # Partials use round() (min 1) instead of truncation for the same reason.
        ct_val        = self._get_ct_val(sym)
        raw_contracts = float(trade_info["size"]) / ct_val
        if order_type == "CLOSE_FULL":
            live = None
            try:
                live = self.fetch_open_position(sym)
            except Exception as e:
                logger.warning("[OKX] %s: live-position lookup for CLOSE_FULL failed: %s", sym, e)
            live_contracts = float((live or {}).get("contracts") or 0)
            if live_contracts > 0:
                contracts = live_contracts
            elif live is not None:
                # Position already flat — an exchange-side TP/SL beat us to it.
                logger.warning(
                    "[OKX] %s: CLOSE_FULL requested but exchange is already flat "
                    "(exchange-side TP/SL fired first) — skipping duplicate close.",
                    sym,
                )
                return {"id": "", "_filled": 0.0, "_fill_price": 0.0, "_already_flat": True}
            else:
                contracts = max(1, round(raw_contracts))
        else:
            contracts = max(1, round(raw_contracts))
        amount        = float(self._ex.amount_to_precision(sym, contracts))

        # [RISK-FLOOR VISIBILITY] A PARTIAL close (e.g. TP1's 50%) on a
        # minimum-lot position (1 contract) can't be split at the exchange's
        # integer-contract resolution — flooring back up to 1 contract closes
        # the ENTIRE remaining position instead of the intended fraction. The
        # bot's local accounting (remaining_size, realized_pnl) still reflects
        # the intended partial until the next periodic reconciliation (every
        # 5 min) corrects it against the real (now-flat) exchange position.
        if order_type == "CLOSE_PARTIAL" and raw_contracts < 1.0 and raw_contracts > 0:
            logger.warning(
                "[OKX] %s: PARTIAL close wanted %.3f contracts, floored to 1 "
                "— this closes the FULL remaining position, not the intended "
                "partial. Local PnL/remaining_size will be corrected by the "
                "next reconciliation pass.",
                sym, raw_contracts,
            )

        params = {"tdMode": self.td_mode, "posSide": pos_side, "reduceOnly": True}

        try:
            raw = self._call(
                self._ex.create_order,
                sym, "market", side, amount, None, params,
                label=f"CLOSE_{pos_side.upper()}({sym})",
            )
        except ccxt.InvalidOrder as e:
            # Reduce-only on a flat position → the exchange-attached TP/SL
            # already closed it between our poll and this order. Not an error:
            # report already-flat so the bot finalizes its local close instead
            # of locking itself into ERROR.
            msg = str(e)
            if any(code in msg for code in ("51169", "51023", "51000")) \
                    or "position" in msg.lower():
                logger.warning(
                    "[OKX] %s: close rejected — position already flat "
                    "(exchange-side TP/SL fired first): %s", sym, msg[:180],
                )
                return {"id": "", "_filled": 0.0, "_fill_price": 0.0, "_already_flat": True}
            raise
        order_id = raw.get("id", "")
        logger.info("[OKX] CLOSE(%s) %s %s qty=%.4f → order_id=%s",
                    trade_info.get("reason", ""), pos_side.upper(), sym, amount, order_id)

        filled, fill_price = self._await_fill_sync(sym, order_id)
        raw["_filled"]     = filled
        raw["_fill_price"] = fill_price
        return raw

    # ── Fill confirmation ────────────────────────────────────────────────────

    def _await_fill_sync(self, symbol: str,
                         order_id: str) -> Tuple[float, float]:
        """Poll order status until filled or timeout. Returns (filled_qty, avg_price)."""
        if not order_id:
            return 0.0, 0.0
        deadline = time.monotonic() + self.FILL_POLL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                o = self._call(
                    self._ex.fetch_order, order_id, symbol,
                    label=f"fetch_order({order_id})",
                )
                status = o.get("status", "")
                filled = float(o.get("filled") or 0)
                price  = float(o.get("average") or o.get("price") or 0)
                if status == "closed" or filled > 0:
                    return filled, price
                if status == "canceled":
                    logger.warning("[OKX] order %s was canceled", order_id)
                    return 0.0, 0.0
            except Exception as e:
                logger.debug("[OKX] fill poll error (order=%s): %s", order_id, e)
            time.sleep(self.FILL_POLL_INTERVAL)
        logger.warning("[OKX] fill confirmation timeout for order %s", order_id)
        return 0.0, 0.0

    async def _await_fill_async(self, symbol: str,
                                 order_id: str) -> Tuple[float, float]:
        """Async version of fill polling."""
        if not order_id:
            return 0.0, 0.0
        deadline = time.monotonic() + self.FILL_POLL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                o = await self._acall(
                    self._aex.fetch_order, order_id, symbol,
                    label=f"fetch_order({order_id})",
                )
                status = o.get("status", "")
                filled = float(o.get("filled") or 0)
                price  = float(o.get("average") or o.get("price") or 0)
                if status == "closed" or filled > 0:
                    return filled, price
                if status == "canceled":
                    logger.warning("[OKX] order %s was canceled", order_id)
                    return 0.0, 0.0
            except Exception as e:
                logger.debug("[OKX] fill poll error (order=%s): %s", order_id, e)
            await asyncio.sleep(self.FILL_POLL_INTERVAL)
        logger.warning("[OKX] fill confirmation timeout for order %s", order_id)
        return 0.0, 0.0

    # ── Paper mode helpers ───────────────────────────────────────────────────

    def _paper_order(self, symbol: str, side: str,
                     amount: float, price: float) -> OrderResult:
        oid = f"paper-{int(time.time()*1000)}"
        r = OrderResult(
            order_id=oid, symbol=symbol, side=side,
            amount=amount, price=price, filled=amount,
            status="closed",
        )
        logger.info("[OKX][PAPER] %s %s qty=%.4f @ %.4f → %s", side, symbol, amount, price, oid)
        return r

    @staticmethod
    def _to_order_result(raw: Dict) -> OrderResult:
        return OrderResult(
            order_id  = raw.get("id", ""),
            symbol    = raw.get("symbol", ""),
            side      = raw.get("side", ""),
            amount    = float(raw.get("amount") or 0),
            price     = float(raw.get("price")  or 0),
            filled    = float(raw.get("filled") or 0),
            status    = raw.get("status", "open"),
            timestamp = int(raw.get("timestamp") or time.time() * 1000),
        )
