"""Runtime ownership layer for Trend Confirm + UT Bot sharing XAU.

OKX hedge mode keeps LONG and SHORT separately, but multiple entries on the
same side are aggregated into one exchange position.  This module lets the bot
keep independent strategy-owned slices inside that aggregate position.

Key guarantees:
- Trend Confirm and UT Bot may coexist on XAU in the same OR opposite direction.
- Every strategy closes/reduces only the amount recorded under its own key.
- UT reversal never cancels Trend Confirm's exchange-side TP/SL orders.
- PortfolioEngine stores XAU slices by strategy key instead of overwriting one
  XAU position with another.
- A persistent ledger restores strategy ownership after Railway restart instead
  of assigning the aggregate exchange position to every XAU strategy.
"""
from __future__ import annotations

import logging
import math
import time
from contextvars import ContextVar
from typing import Optional

from .bot import TradingBot
from .connectors.binance_conn import BinanceConnector
from .engines.portfolio_engine import PortfolioEngine
from .strategy_position_ledger import StrategyPositionLedger


logger = logging.getLogger("xau_multi_strategy_runtime")
XAU_SYMBOL = "XAU/USDT:USDT"
UT_PREFIX = "UTBotXAU("
TC_PREFIX = "TrendConfirm("

_ACTIVE_STRATEGY: ContextVar[str | None] = ContextVar(
    "xau_multi_strategy_active_owner",
    default=None,
)
_LEDGER = StrategyPositionLedger()


def strip_side_suffix(strategy: str) -> str:
    value = str(strategy or "")
    return value[:-2] if value.endswith((":L", ":S")) else value


def family(strategy: str) -> str:
    base = strip_side_suffix(strategy)
    if base.startswith(UT_PREFIX):
        return "utbot_xau"
    if base.startswith(TC_PREFIX):
        return "trend_confirm"
    return "other"


def _is_owned_xau(strategy: str, symbol: str = XAU_SYMBOL) -> bool:
    return symbol == XAU_SYMBOL and family(strategy) in {"utbot_xau", "trend_confirm"}


def _portfolio_key(symbol: str) -> str:
    strategy = _ACTIVE_STRATEGY.get()
    if _is_owned_xau(strategy, symbol):
        return f"{symbol}||{strategy}"
    return symbol


def _near(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(1e-8, scale * 0.005)  # 0.5% handles lot rounding


def _install_portfolio_strategy_keys() -> None:
    if getattr(PortfolioEngine, "_xau_strategy_keys_installed", False):
        return

    original_can_add = PortfolioEngine.can_add_position
    original_add = PortfolioEngine.add_position
    original_remove = PortfolioEngine.remove_position
    original_update = PortfolioEngine.update_prices

    def _can_add(self, symbol, direction, entry_price, stop_loss, amount, portfolio_value):
        return original_can_add(
            self,
            _portfolio_key(symbol),
            direction,
            entry_price,
            stop_loss,
            amount,
            portfolio_value,
        )

    def _add(self, symbol, direction, entry_price, current_price, amount, stop_loss):
        return original_add(
            self,
            _portfolio_key(symbol),
            direction,
            entry_price,
            current_price,
            amount,
            stop_loss,
        )

    def _remove(self, symbol):
        return original_remove(self, _portfolio_key(symbol))

    def _update(self, prices):
        # Propagate a plain XAU market price into each strategy-specific XAU key.
        expanded = dict(prices or {})
        if XAU_SYMBOL in expanded:
            xau_price = expanded[XAU_SYMBOL]
            for key in list(self._positions):
                if str(key).startswith(f"{XAU_SYMBOL}||"):
                    expanded[key] = xau_price
        return original_update(self, expanded)

    PortfolioEngine.can_add_position = _can_add
    PortfolioEngine.add_position = _add
    PortfolioEngine.remove_position = _remove
    PortfolioEngine.update_prices = _update
    PortfolioEngine._xau_strategy_keys_installed = True


def _install_ut_tpsl_isolation() -> None:
    """UT has no fixed TPSL; its reversal must not cancel TC trigger orders."""
    if getattr(BinanceConnector, "_ut_tpsl_isolation_installed", False):
        return

    original = BinanceConnector.set_position_tpsl

    async def _set_tpsl(self, symbol, pos_side, amount, sl=None, tp=None):
        strategy = _ACTIVE_STRATEGY.get()
        if (
            symbol == XAU_SYMBOL
            and family(strategy) == "utbot_xau"
            and float(amount or 0.0) <= 0.0
            and not sl
            and not tp
        ):
            logger.info(
                "[%s] UT reversal close: preserve Trend Confirm XAU TPSL orders",
                strategy,
            )
            return True
        return await original(self, symbol, pos_side, amount, sl=sl, tp=tp)

    BinanceConnector.set_position_tpsl = _set_tpsl
    BinanceConnector._ut_tpsl_isolation_installed = True


def _register_from_risk(bot: TradingBot, symbol: str, strategy: str) -> None:
    if not _is_owned_xau(strategy, symbol):
        return
    pos = bot.risk._positions.get(f"{symbol}||{strategy}")
    if pos is None:
        return
    _LEDGER.set(
        symbol=symbol,
        strategy=strategy,
        side=pos.side,
        entry=pos.entry_price,
        amount=pos.amount,
        stop_loss=pos.stop_loss,
        take_profit=pos.take_profit,
    )


def _install_bot_ownership_hooks() -> None:
    if getattr(TradingBot, "_xau_multi_owner_installed", False):
        return

    original_execute = TradingBot._execute_signal
    original_handle_update = TradingBot._handle_pos_update
    original_on_closed = TradingBot._on_position_closed
    original_reconcile = TradingBot._reconcile_positions

    async def _execute(self, signal, strategy_name, *args, **kwargs):
        token = _ACTIVE_STRATEGY.set(strategy_name)
        try:
            result = await original_execute(self, signal, strategy_name, *args, **kwargs)
            _register_from_risk(self, signal.symbol, strategy_name)
            return result
        finally:
            _ACTIVE_STRATEGY.reset(token)

    async def _handle_update(self, pos_info, update, price, strategy_inst=None):
        strategy_name = str(pos_info.get("strategy", "") or "")
        token = _ACTIVE_STRATEGY.set(strategy_name)
        try:
            result = await original_handle_update(
                self,
                pos_info,
                update,
                price,
                strategy_inst,
            )
            # Partial TP / SL move changes the strategy-owned slice in RiskManager.
            if _is_owned_xau(strategy_name, pos_info.get("symbol")):
                current = self.risk._positions.get(
                    f"{pos_info['symbol']}||{strategy_name}"
                )
                if current is None:
                    _LEDGER.remove(pos_info["symbol"], strategy_name)
                else:
                    _register_from_risk(self, pos_info["symbol"], strategy_name)
            return result
        finally:
            _ACTIVE_STRATEGY.reset(token)

    def _on_closed(self, symbol, strategy_name, exit_price, reason, strategy_inst=None):
        token = _ACTIVE_STRATEGY.set(strategy_name)
        try:
            result = original_on_closed(
                self,
                symbol,
                strategy_name,
                exit_price,
                reason,
                strategy_inst,
            )
            if _is_owned_xau(strategy_name, symbol):
                _LEDGER.remove(symbol, strategy_name)
            return result
        finally:
            _ACTIVE_STRATEGY.reset(token)

    async def _reconcile(self):
        """Reconcile non-XAU normally; restore XAU from the ownership ledger."""
        saved = list(self.strategies)
        non_xau = [s for s in saved if s.symbol != XAU_SYMBOL]
        try:
            self.strategies = non_xau
            if non_xau:
                await original_reconcile(self)
        finally:
            self.strategies = saved

        xau_strategies = [s for s in saved if s.symbol == XAU_SYMBOL]
        if not xau_strategies or not hasattr(self.connector, "fetch_positions"):
            return

        try:
            live_rows = await self.connector.fetch_positions([XAU_SYMBOL])
        except Exception as exc:
            logger.warning("[XAU-Reconcile] fetch_positions failed: %s", exc)
            return

        live_by_side = {
            str(row.get("side", "")).lower(): row
            for row in (live_rows or [])
            if row.get("amount")
        }
        ledger_rows = [
            row for row in _LEDGER.all_for_symbol(XAU_SYMBOL)
            if family(row.get("strategy", "")) in {"trend_confirm", "utbot_xau"}
        ]

        # First deployment of this ledger: an already-open XAU position most
        # likely belongs to the pre-existing Trend Confirm strategy. Never make
        # the newly-added UT Bot claim it unless UT is the only XAU strategy.
        if not ledger_rows and live_by_side:
            tc_candidates = [s for s in xau_strategies if s.name.startswith(TC_PREFIX)]
            ut_candidates = [s for s in xau_strategies if s.name.startswith(UT_PREFIX)]
            target_strategy = tc_candidates[0] if tc_candidates else (
                ut_candidates[0] if len(xau_strategies) == 1 and ut_candidates else None
            )
            if target_strategy is not None:
                for side, live in live_by_side.items():
                    strategy_name = (
                        f"{target_strategy.name}:{'L' if side == 'long' else 'S'}"
                        if self._hedge_mode else target_strategy.name
                    )
                    _LEDGER.set(
                        XAU_SYMBOL,
                        strategy_name,
                        side,
                        float(live.get("entry_price") or 0.0),
                        float(live.get("amount") or 0.0),
                    )
                ledger_rows = _LEDGER.all_for_symbol(XAU_SYMBOL)

        # Resolve stale ledger rows by comparing each side's aggregate exchange
        # amount. With at most TC+UT there are only 0/1/2 owned slices per side.
        active_rows: list[dict] = []
        for side in ("long", "short"):
            rows = [r for r in ledger_rows if r.get("side") == side]
            live = live_by_side.get(side)
            live_amount = float((live or {}).get("amount") or 0.0)
            if live_amount <= 0:
                for row in rows:
                    _LEDGER.remove(XAU_SYMBOL, row.get("strategy", ""))
                continue
            if not rows:
                continue

            total = sum(float(r.get("amount") or 0.0) for r in rows)
            if _near(total, live_amount):
                active_rows.extend(rows)
                continue

            # If one exchange-side TP/SL fired while the bot was offline, the
            # remaining aggregate commonly equals exactly the other slice.
            if len(rows) == 2:
                first, second = rows
                a = float(first.get("amount") or 0.0)
                b = float(second.get("amount") or 0.0)
                if _near(live_amount, a):
                    _LEDGER.remove(XAU_SYMBOL, second.get("strategy", ""))
                    active_rows.append(first)
                    logger.warning(
                        "[XAU-Reconcile] inferred %s closed while offline; %s remains",
                        second.get("strategy"), first.get("strategy"),
                    )
                    continue
                if _near(live_amount, b):
                    _LEDGER.remove(XAU_SYMBOL, first.get("strategy", ""))
                    active_rows.append(second)
                    logger.warning(
                        "[XAU-Reconcile] inferred %s closed while offline; %s remains",
                        first.get("strategy"), second.get("strategy"),
                    )
                    continue

            # Ambiguous aggregate mismatch: preserve ownership proportions but
            # clamp their sum to the exchange truth instead of duplicating size.
            if total > 0:
                ratio = live_amount / total
                for row in rows:
                    scaled = dict(row)
                    scaled["amount"] = float(row.get("amount") or 0.0) * ratio
                    _LEDGER.set(
                        XAU_SYMBOL,
                        scaled.get("strategy", ""),
                        side,
                        float(scaled.get("entry") or (live or {}).get("entry_price") or 0.0),
                        scaled["amount"],
                        scaled.get("stop_loss"),
                        scaled.get("take_profit"),
                    )
                    active_rows.append(scaled)
                logger.warning(
                    "[XAU-Reconcile] aggregate amount %.8f differs from ledger %.8f; "
                    "scaled strategy slices proportionally",
                    live_amount, total,
                )

        strategy_map = {s.name: s for s in xau_strategies}
        for row in active_rows:
            strategy_name = str(row.get("strategy") or "")
            base_name = strip_side_suffix(strategy_name)
            strategy_inst = strategy_map.get(base_name)
            if strategy_inst is None:
                continue

            side = str(row.get("side") or "").lower()
            entry = float(row.get("entry") or 0.0)
            amount = float(row.get("amount") or 0.0)
            if side not in {"long", "short"} or entry <= 0 or amount <= 0:
                continue

            risk_key = f"{XAU_SYMBOL}||{strategy_name}"
            if risk_key in self.risk._positions:
                continue
            sl = row.get("stop_loss")
            tp = row.get("take_profit")
            pos = self.risk.open_position(
                XAU_SYMBOL,
                side,
                entry,
                amount,
                strategy=strategy_name,
                stop_loss=sl,
                take_profit=tp,
            )
            # Avoid RiskManager's fallback SL/TP for UT or any ledger-null level.
            pos.stop_loss = float(sl) if sl else None
            pos.take_profit = float(tp) if tp else None

            token = _ACTIVE_STRATEGY.set(strategy_name)
            try:
                portfolio_sl = pos.stop_loss if pos.stop_loss is not None else entry
                self._portfolio.add_position(
                    XAU_SYMBOL,
                    side,
                    entry,
                    entry,
                    amount,
                    portfolio_sl,
                )
            finally:
                _ACTIVE_STRATEGY.reset(token)

            self._position_open_times[risk_key] = time.time()
            if hasattr(strategy_inst, "attach_existing_position"):
                try:
                    strategy_inst.attach_existing_position(
                        side,
                        entry,
                        pos.stop_loss,
                        pos.take_profit,
                    )
                except Exception as exc:
                    logger.warning(
                        "[XAU-Reconcile] attach failed [%s]: %s",
                        strategy_name,
                        exc,
                    )
            logger.warning(
                "[XAU-Reconcile] restored owner=%s side=%s entry=%.4f amount=%.8f",
                strategy_name,
                side.upper(),
                entry,
                amount,
            )

    TradingBot._execute_signal = _execute
    TradingBot._handle_pos_update = _handle_update
    TradingBot._on_position_closed = _on_closed
    TradingBot._reconcile_positions = _reconcile
    TradingBot._xau_multi_owner_installed = True


def install_xau_multi_strategy_runtime() -> None:
    _install_portfolio_strategy_keys()
    _install_ut_tpsl_isolation()
    _install_bot_ownership_hooks()
    logger.info(
        "XAU multi-strategy ownership runtime installed — TC+UT may stack or hedge"
    )
