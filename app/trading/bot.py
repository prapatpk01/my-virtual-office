"""
Trading Bot main engine — Adaptive AI Edition.

Integrates the full 9-layer AI Expert pipeline into the live trading loop:
  - Layer 7: Position Manager (break-even, trailing stop, partial TP) runs every tick
  - Layer 8: Exit AI scores feed the Position Manager each tick
  - Layer 9: Learning Engine is fed after every position close
  - Portfolio Engine gates every new position for heat / correlation limits
  - Drift Detector alerts are broadcast to Telegram when thresholds are crossed
"""
import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .connectors.base import BaseConnector
from .strategies.base import BaseStrategy, Signal, SignalType
from .strategies.wt_adx_strategy import WTADXStrategy
from .risk_manager import RiskManager
from .telegram_notifier import TelegramNotifier
from .signal_state import SignalState
from .engines.portfolio_engine import PortfolioEngine
from .engines.drift_detector import DriftAction

logger = logging.getLogger("trading_bot")

# Confidence level → fraction of balance to allocate per trade (fallback
# sizing only, used when a signal has no usable SL price).
_CONFIDENCE_SIZE: dict[str, float] = {
    "WEAK":            float(os.getenv("SIZE_WEAK",  "0.08")),
    "GOOD":            float(os.getenv("SIZE_GOOD",  "0.10")),
    "HIGH_CONVICTION": float(os.getenv("SIZE_HIGH",  "0.12")),
}
_DEFAULT_SIZE = float(os.getenv("SIZE_WEAK", "0.08"))
# Layer 6 Confidence Engine level names -> legacy tier names above
_LEVEL_ALIAS = {"GOOD": "GOOD", "HIGH_CONFIDENCE": "HIGH_CONVICTION", "SKIP": "WEAK"}


def _confidence_size_pct(metadata: dict) -> float:
    """Return allocation fraction (0.08–0.12) based on AI confidence level."""
    level = (metadata or {}).get("confidence_level", "").upper()
    level = _LEVEL_ALIAS.get(level, level)
    return _CONFIDENCE_SIZE.get(level, _DEFAULT_SIZE)


@dataclass
class TradeRecord:
    timestamp: int
    symbol: str
    side: str
    price: float
    amount: float
    pnl: float
    strategy: str
    reason: str
    paper: bool


@dataclass
class BotState:
    running: bool = False
    paper: bool = True
    total_balance: float = 0.0
    equity: float = 0.0
    pnl_today: float = 0.0
    pnl_total: float = 0.0
    open_positions: list = field(default_factory=list)
    recent_trades: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    strategy_states: dict = field(default_factory=dict)
    last_updated: int = 0
    error: str = ""


class TradingBot:
    """
    Orchestrates multiple strategies across multiple connectors.
    Emits state updates via a broadcast callback so the Virtual Office
    server can push them to connected WebSocket clients.

    AI Expert strategies get full lifecycle integration:
      tick_open_position() → handles BE / trailing / partial TP / AI exit
      record_closed_trade() → feeds learning engine after every close
      Portfolio engine gates new positions for heat and correlation risk
    """

    def __init__(
        self,
        connector: BaseConnector,
        strategies: list[BaseStrategy],
        risk_manager: Optional[RiskManager] = None,
        interval_seconds: int = 60,
        broadcast_fn: Optional[Callable[[dict], Any]] = None,
        telegram: Optional[TelegramNotifier] = None,
        state_file: Optional[str] = None,
    ):
        self.connector = connector
        self.strategies = strategies
        self.risk = risk_manager or RiskManager()
        self.interval = interval_seconds
        self._broadcast = broadcast_fn or (lambda x: None)
        self.telegram = telegram
        self.state = BotState(paper=connector.paper)
        self._task: Optional[asyncio.Task] = None
        self._start_balance = 0.0
        self._trade_history: list[TradeRecord] = []
        kwargs = {"path": state_file} if state_file else {}
        self._sig = SignalState(**kwargs)
        self.wt_verify: bool = os.getenv("WTV", "false").lower() == "true"
        if self.wt_verify:
            logger.info("[WTV] WaveTrend verify ENABLED (WT1 gate ±10 active)")

        # ── Adaptive extensions ─────────────────────────────────────────────
        # Fast lookup: strategy_name → strategy instance (for lifecycle hooks)
        self._strategy_map: dict[str, BaseStrategy] = {s.name: s for s in strategies}

        # Track when each position was opened to calculate trade duration
        self._position_open_times: dict[str, float] = {}  # key = "symbol||strategy_name"
        # Actual ENTRY fill per open position (avg_px/size/fee from the
        # exchange order, post-fill). Used to allocate the entry fee across
        # partial/final closes:  Net PnL = realized_pnl - entry_fee_alloc -
        # exit_fee.  fee_frac_left tracks how much of the entry fee is still
        # unallocated after partial TPs. key = "symbol||strategy_name"
        self._entry_fills: dict[str, dict] = {}

        # Portfolio-level risk engine (shared across all strategies)
        self._portfolio = PortfolioEngine(
            max_portfolio_heat=float(os.getenv("MAX_PORTFOLIO_HEAT", "0.06")),
            max_same_group=int(os.getenv("MAX_SAME_GROUP", "2")),
            max_total_positions=self.risk.max_open_positions,
        )

        # Tick counter — used for periodic drift alerts
        self._tick_count = 0
        self._last_drift_alert_tick = 0

        # Warm-up: skip opening new positions for the first N ticks after
        # (re)start, so a restart doesn't fire an entry off the very first
        # scan before the strategy has re-observed live market conditions.
        self._warmup_ticks_remaining = int(os.getenv("WARMUP_TICKS", "1"))

        # Futures hedge mode — read from connector if available, else env
        self._hedge_mode: bool = (
            getattr(connector, "_hedge_mode", False)
            or os.getenv("HEDGE_MODE", "false").lower() in ("1", "true", "yes")
        )
        # Track symbols that have had futures setup (leverage + hedge mode)
        self._futures_setup_done: set[str] = set()

    def _resolve_strategy_inst(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Look up a strategy instance by its position-tracking key, which in
        hedge mode carries a ':L'/':S' suffix (e.g. "AIExpert(BTC/USDT:USDT):L")
        that _strategy_map's keys (plain strategy.name) don't have."""
        inst = self._strategy_map.get(strategy_name)
        if inst is None and strategy_name.endswith((":L", ":S")):
            inst = self._strategy_map.get(strategy_name[:-2])
        return inst

    def _cancel_pending_entry(self, strategy_name: str, reason: str = "") -> None:
        """Call after a signal is rejected/fails between analyze() and a
        confirmed order. Several strategies (ema_sma, ema_macd, hma_macd_roc)
        optimistically flip an internal 'position open' flag as soon as
        analyze() emits an entry signal — if bot.py never actually opens
        the position (risk/portfolio gate, low balance, order error), that
        flag would otherwise stay stuck forever and silently block every
        future entry on the symbol."""
        strategy_inst = self._resolve_strategy_inst(strategy_name)
        if hasattr(strategy_inst, "cancel_pending_entry"):
            try:
                strategy_inst.cancel_pending_entry(reason)
            except Exception as e:
                logger.warning("cancel_pending_entry failed [%s]: %s", strategy_name, e)

    def _close_fill_info(self, pos_key: str, order, fallback_price: float,
                         close_amt: float, close_frac: float, final: bool) -> dict:
        """Build the post-fill accounting for a close/reduce order from the
        exchange's OWN numbers (avgPx, fillSz, fee, realized pnl) — never our
        estimates.  Net PnL = realized_pnl - entry_fee_allocation - exit_fee,
        where the entry fee is allocated by the fraction of the position this
        order closed. Consumes that fraction from the tracked entry fill; on a
        final close the remaining allocation is used and tracking is dropped.
        Falls back gracefully (alloc 0, pnl None) for reconciled positions
        whose entry fill was never seen."""
        entry = self._entry_fills.get(pos_key)
        if entry is not None:
            frac = min(close_frac, entry.get("fee_frac_left", 1.0))
            if final:
                frac = entry.get("fee_frac_left", 1.0)
            entry_fee_alloc = entry["fee"] * frac
            entry["fee_frac_left"] = max(0.0, entry.get("fee_frac_left", 1.0) - frac)
            if final:
                self._entry_fills.pop(pos_key, None)
        else:
            entry_fee_alloc = 0.0
        exit_fee = getattr(order, "fee", 0.0) or 0.0
        realized = getattr(order, "realized_pnl", None)
        net = (realized - entry_fee_alloc - exit_fee) if realized is not None else None
        return {
            "exit_avg_px": getattr(order, "price", 0.0) or fallback_price,
            "exit_sz": getattr(order, "filled", 0.0) or close_amt,
            "exit_fee": round(exit_fee, 6),
            "entry_fee_alloc": round(entry_fee_alloc, 6),
            "entry_avg_px": entry.get("avg_px") if entry else None,
            "realized_pnl": realized,
            "net_pnl": round(net, 6) if net is not None else None,
        }

    @staticmethod
    def _chart_ma_kwargs(strategy_inst: Optional[BaseStrategy]) -> dict:
        """Pulls the actual MA/MACD periods a strategy trades on so the
        Telegram entry chart draws the SAME lines the strategy used to
        decide the trade, instead of a fixed generic EMA20/EMA50 that may
        not match at all. Falls back to ai_expert's EMA20/EMA50 defaults
        for strategies that don't expose these attributes (ai_expert
        itself). A strategy exposing ema_fast/ema_slow draws those (e.g.
        trend_confirm, whose Layer3 entry triggers on EMA10/20); one exposing
        only hma_fast/hma_slow draws the HMA pair instead."""
        if strategy_inst is None:
            return {}
        kwargs: dict = {}
        if hasattr(strategy_inst, "ema_fast") and hasattr(strategy_inst, "ema_slow"):
            kwargs["ema_fast"] = strategy_inst.ema_fast
            kwargs["ema_slow"] = strategy_inst.ema_slow
        elif hasattr(strategy_inst, "hma_fast") and hasattr(strategy_inst, "hma_slow"):
            kwargs["ma_type"] = "hma"
            kwargs["ema_fast"] = strategy_inst.hma_fast
            kwargs["ema_slow"] = strategy_inst.hma_slow
        # trend_confirm's entry runs on 5m — draw its EMA50 stop reference and
        # label the timeframe, and skip the 30m SMA30 (meaningless on a 5m
        # chart). Everything else keeps SMA30 as before.
        if hasattr(strategy_inst, "entry_tf") and hasattr(strategy_inst, "sl_ema_ref"):
            kwargs["extra_ema"] = strategy_inst.sl_ema_ref
            kwargs["tf_label"] = strategy_inst.entry_tf
        elif hasattr(strategy_inst, "sma_trend"):
            kwargs["sma_period"] = strategy_inst.sma_trend
        if hasattr(strategy_inst, "macd_fast"):
            kwargs["macd_fast"] = strategy_inst.macd_fast
        if hasattr(strategy_inst, "macd_slow"):
            kwargs["macd_slow"] = strategy_inst.macd_slow
        if hasattr(strategy_inst, "macd_signal"):
            kwargs["macd_signal_period"] = strategy_inst.macd_signal
        return kwargs

    async def _reconcile_closed_positions(self) -> None:
        """Runs every tick. If a position the bot still tracks is no longer
        open on the exchange, the exchange closed it itself — its OKX TP/SL
        algo order fired between ticks. Clear the bot's state for it (risk,
        portfolio, strategy, locks), pull the actual close from OKX order
        history to report the real post-fee PnL, and notify. Without this the
        bot keeps 'managing' a position that's already gone (the log/positions
        show it open long after OKX closed it)."""
        tracked = list(self.risk.get_positions())
        if not tracked or self.connector.paper or not hasattr(self.connector, "fetch_positions"):
            return
        try:
            live = await self.connector.fetch_positions(list({p["symbol"] for p in tracked}))
        except Exception as e:
            logger.debug("[Reconcile-closed] fetch_positions failed: %s", e)
            return
        live_keys = {(p["symbol"], p["side"]) for p in live if p.get("amount")}
        for pos_info in tracked:
            sym = pos_info["symbol"]; side = pos_info["side"]; strat = pos_info.get("strategy", "")
            if (sym, side) in live_keys:
                continue  # still open — normal management handles it
            # Gone on the exchange → closed by OKX (algo SL/TP).
            logger.info("[Reconcile-closed] %s %s no longer open on exchange — "
                        "closed by OKX (algo SL/TP). Clearing bot state.", sym, side)
            exit_px = pos_info.get("entry", 0.0)
            reason = "exchange_sl_tp"
            fill = None
            try:
                # Pull the real close (avgPx/fee/realized pnl) from OKX history.
                fill = await self._okx_last_close_fill(sym, side)
                if fill:
                    exit_px = fill.get("exit_avg_px", exit_px)
                    reason = fill.get("reason", reason)
            except Exception as e:
                logger.debug("[Reconcile-closed] history lookup failed for %s: %s", sym, e)
            strategy_inst = self._resolve_strategy_inst(strat)
            _outcome = self._sig.record_outcome(
                symbol=sym, side=side, entry=pos_info.get("entry", 0.0), exit_price=exit_px,
                sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"),
                reason=reason, strategy=strat, fill=fill,
            )
            self._entry_fills.pop(f"{sym}||{strat}", None)
            self._sig.unlock_strategy(sym, strat)
            self.risk.close_position(sym, strategy=strat)
            self._on_position_closed(sym, strat, exit_px, reason, strategy_inst)
            # Clear any leftover exchange algo orders for the symbol.
            try:
                await self.connector.set_position_tpsl(sym, side, 0.0)
            except Exception:
                pass
            if self.telegram:
                self.telegram.notify_trade_closed(sym, _outcome, self._sig.summary())

    async def _okx_last_close_fill(self, symbol: str, side: str) -> Optional[dict]:
        """Best-effort: fetch the most recent reduce-only (close) fill for a
        symbol from OKX order history and shape it like _close_fill_info so the
        outcome/notification carry real avgPx/fee/realized-PnL. Returns None if
        unavailable."""
        if not hasattr(self.connector, "fetch_recent_closes"):
            return None
        closes = await self.connector.fetch_recent_closes(symbol, limit=5)
        if not closes:
            return None
        c = closes[0]  # most recent
        entry_fee_alloc = 0.0
        # allocate whatever entry fee we still have tracked for this symbol
        for key, ef in list(self._entry_fills.items()):
            if key.startswith(f"{symbol}||"):
                entry_fee_alloc = ef.get("fee", 0.0) * ef.get("fee_frac_left", 1.0)
                break
        realized = c.get("pnl")
        exit_fee = c.get("fee", 0.0)
        net = (realized - entry_fee_alloc - exit_fee) if realized is not None else None
        return {
            "exit_avg_px": c.get("price", 0.0),
            "exit_sz": c.get("amount", 0.0),
            "exit_fee": round(exit_fee, 6),
            "entry_fee_alloc": round(entry_fee_alloc, 6),
            "entry_avg_px": None,
            "realized_pnl": realized,
            "net_pnl": round(net, 6) if net is not None else None,
            "reason": "exchange_sl_tp",
        }

    async def _reconcile_positions(self) -> None:
        """Runs once, right before the first tick of every (re)start.

        Nothing in this process persists open positions across a restart —
        RiskManager._positions, PortfolioEngine._positions, and each
        strategy's _open_position/_open_entry are all plain in-memory state
        that starts empty every time. If the bot restarts while a position
        is genuinely still open on the exchange, it would otherwise be
        invisible to the bot forever: no hard SL/TP fallback, no portfolio
        heat accounting, and the strategy would try to open a duplicate on
        the next signal instead of managing the existing one.

        This re-derives those positions directly from the exchange (ccxt
        fetch_positions) and re-registers them everywhere they're normally
        tracked. The original SL/TP levels aren't recoverable (never
        persisted), so RiskManager.open_position() falls back to its
        default percentage-based stops — better than no protection at all,
        but a human should sanity-check them against the original plan.
        Connectors that don't support fetch_positions (or paper mode, which
        has no exchange-side state to reconcile against) are a no-op."""
        if not hasattr(self.connector, "fetch_positions"):
            return
        symbols = list({s.symbol for s in self.strategies})
        try:
            live_positions = await self.connector.fetch_positions(symbols)
        except Exception as e:
            logger.warning("[Reconcile] fetch_positions failed: %s", e)
            return
        if not live_positions:
            return

        by_symbol: dict[str, list[dict]] = {}
        for p in live_positions:
            by_symbol.setdefault(p["symbol"], []).append(p)

        for strategy in self.strategies:
            for p in by_symbol.get(strategy.symbol, []):
                side = p["side"]
                strategy_name = (
                    f"{strategy.name}:{'L' if side == 'long' else 'S'}"
                    if self._hedge_mode else strategy.name
                )
                risk_key = f"{strategy.symbol}||{strategy_name}"
                if risk_key in self.risk._positions:
                    continue

                entry_price = p["entry_price"]
                amount = p["amount"]
                pos = self.risk.open_position(
                    strategy.symbol, side, entry_price, amount, strategy=strategy_name,
                )
                self._portfolio.add_position(
                    symbol=strategy.symbol, direction=side,
                    entry_price=entry_price, current_price=entry_price,
                    amount=amount, stop_loss=pos.stop_loss,
                )
                self._position_open_times[f"{strategy.symbol}||{strategy_name}"] = time.time()

                strategy_inst = self._resolve_strategy_inst(strategy_name)
                if hasattr(strategy_inst, "attach_existing_position"):
                    try:
                        strategy_inst.attach_existing_position(
                            side, entry_price, pos.stop_loss, pos.take_profit,
                        )
                    except Exception as e:
                        logger.warning("[Reconcile] attach_existing_position failed [%s]: %s",
                                       strategy_name, e)

                logger.warning(
                    "[Reconcile] Found existing %s position on %s (%s) — entry=%.4f amount=%.6f "
                    "SL=%.4f TP=%.4f (default stops — original unknown) — resuming management",
                    side.upper(), strategy.symbol, strategy_name, entry_price, amount,
                    pos.stop_loss or 0, pos.take_profit or 0,
                )
                if self.telegram:
                    try:
                        self.telegram.notify_reconciled_position(
                            strategy.symbol, strategy_name, side, entry_price, amount,
                            pos.stop_loss, pos.take_profit,
                        )
                    except Exception as e:
                        logger.warning("[Reconcile] notify_reconciled_position failed [%s]: %s",
                                       strategy_name, e)

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self):
        if self._task and not self._task.done():
            logger.warning("Bot already running")
            return
        self.state.running = True
        loop = asyncio.get_event_loop()
        skip_polling = getattr(self, "_skip_telegram_polling", False)
        if self.telegram and not skip_polling:
            self.telegram.start_polling(loop)
            strategy_names = [s.name for s in self.strategies]
            symbols = list({s.symbol for s in self.strategies})
            self.telegram.notify_bot_started(self.connector.paper, strategy_names, symbols)
        self._task = asyncio.create_task(self._run_loop())
        logger.info("TradingBot started (paper=%s, interval=%ds)", self.connector.paper, self.interval)

    async def stop(self):
        self.state.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.telegram:
            self.telegram.stop_polling()
            self.telegram.notify_bot_stopped()
        logger.info("TradingBot stopped")

    async def manual_signal(self, symbol: str, side: str, amount: float, reason: str = "manual"):
        """Execute a manual trade bypassing strategy analysis."""
        try:
            order = await self.connector.create_order(symbol, side, amount)
            trade = TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=symbol, side=side,
                price=order.price, amount=amount,
                pnl=0.0, strategy="manual", reason=reason,
                paper=self.connector.paper,
            )
            self._record_trade(trade)
            await self._refresh_balance()
            self._broadcast_state()
        except Exception as e:
            logger.error("Manual order failed: %s", e)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _run_loop(self):
        await self._refresh_balance()
        self._start_balance = self.state.total_balance
        self.risk.update_peak(self._start_balance)
        await self._reconcile_positions()

        while self.state.running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bot tick error: %s", e, exc_info=True)
                self.state.error = str(e)
            await asyncio.sleep(self.interval)

    async def _tick(self):
        self.state.error = ""
        self._tick_count += 1
        await self._refresh_balance()

        if not self.risk.check_drawdown(self.state.total_balance):
            logger.warning("Max drawdown hit — bot paused")
            self.state.error = "Max drawdown reached. Trading paused."
            if self.telegram:
                self.telegram.notify_drawdown_halt(
                    self.state.total_balance, self.risk._peak_balance
                )
            self._broadcast_state()
            return

        # Per-symbol cooldown (3 consecutive losing closes) blocks NEW entries
        # for that symbol only — existing positions are still managed below.
        _cooling = [(s, self.risk.in_cooldown(s)[1]) for s in {st.symbol for st in self.strategies}
                    if self.risk.in_cooldown(s)[0]]
        if _cooling:
            _cooling.sort(key=lambda x: x[1])
            self.state.error = (f"{len(_cooling)} symbol(s) cooling — "
                                f"{_cooling[0][0].split('/')[0]} resumes in {_cooling[0][1]/60:.0f} min")

        # ── Detect positions the EXCHANGE closed on its own (OKX TP/SL algo
        # firing) so the bot doesn't keep managing a ghost position. ─────────
        await self._reconcile_closed_positions()

        # ── Position management: SL/TP, AI Layer 7, and learning callbacks ──
        for pos_info in list(self.risk.get_positions()):
            sym           = pos_info["symbol"]
            strategy_name = pos_info.get("strategy", "")
            ticker        = await self.connector.fetch_ticker(sym)
            price         = ticker["last"]

            strategy_inst = self._resolve_strategy_inst(strategy_name)

            # Layer 7 + 8: AI Expert position management (break-even, trailing, partial TP, exit AI)
            if hasattr(strategy_inst, "tick_open_position"):
                try:
                    pos_update = strategy_inst.tick_open_position(
                        current_price=price,
                        position_key=f"{sym}||{strategy_name}",
                    )
                    if pos_update and pos_update.action != "hold":
                        closed = await self._handle_pos_update(pos_info, pos_update, price, strategy_inst)
                        if closed:
                            continue  # position fully closed; skip SL/TP check below
                except Exception as e:
                    logger.error("tick_open_position error [%s %s]: %s", strategy_name, sym, e)

            # Fallback: risk-manager hard SL/TP stop check
            trigger = self.risk.check_stops(sym, price, strategy=strategy_name)
            if trigger:
                side = "sell" if pos_info["side"] == "long" else "buy"
                pos_side = pos_info["side"] if self._hedge_mode else None
                close_order = await self.connector.create_order(sym, side, pos_info["amount"], pos_side=pos_side)
                fill = self._close_fill_info(f"{sym}||{strategy_name}", close_order, price,
                                             pos_info["amount"], 1.0, final=True)
                exit_px = fill["exit_avg_px"]
                pnl = (fill["net_pnl"] if fill["net_pnl"] is not None else
                       ((exit_px - pos_info["entry"]) * pos_info["amount"]
                        if pos_info["side"] == "long"
                        else (pos_info["entry"] - exit_px) * pos_info["amount"]))
                trade = TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym, side=side,
                    price=exit_px, amount=fill["exit_sz"],
                    pnl=round(pnl, 4),
                    strategy=strategy_name or "risk_manager", reason=trigger,
                    paper=self.connector.paper,
                )
                self._record_trade(trade)
                _outcome = self._sig.record_outcome(
                    symbol=sym, side=pos_info["side"],
                    entry=pos_info["entry"], exit_price=exit_px,
                    sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"),
                    reason=trigger, strategy=strategy_name, fill=fill,
                )
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                self._on_position_closed(sym, strategy_name, price, trigger, strategy_inst)
                logger.info("Position closed by %s: %s [%s]", trigger, sym, strategy_name)
                if self.telegram:
                    self.telegram.notify_trade_closed(sym, _outcome, self._sig.summary())
                self._check_cooldown_trigger(pnl, sym)

        # ── Periodic drift alert (every 50 ticks) ────────────────────────────
        if self._tick_count - self._last_drift_alert_tick >= 50:
            self._last_drift_alert_tick = self._tick_count
            self._broadcast_drift_alerts()

        # ── Run each strategy ────────────────────────────────────────────────
        new_signals = []
        _resolved_symbols: set[str] = set()
        for strategy in self.strategies:
            try:
                _tf    = os.getenv("CANDLE_TF", "15m")
                _limit = int(os.getenv("CANDLE_LIMIT", "300"))
                candles = await self.connector.fetch_ohlcv(strategy.symbol, timeframe=_tf, limit=_limit)
                ticker  = await self.connector.fetch_ticker(strategy.symbol)
                current_price = ticker["last"]

                if strategy.symbol not in _resolved_symbols and candles:
                    _resolved_symbols.add(strategy.symbol)
                    last_c = candles[-1]
                    v_high = max(float(last_c.high), current_price)
                    v_low  = min(float(last_c.low),  current_price)
                    resolved = self._sig.check_and_resolve_pending(strategy.symbol, v_high, v_low)
                    for v_reason, v_price, v_outcome in resolved:
                        if self.telegram:
                            self.telegram.notify_virtual_closed(
                                strategy.symbol, v_reason, v_price,
                                self._sig.summary(), outcome=v_outcome,
                            )

                mtf_candles = {}
                _base_tf = os.getenv("CANDLE_TF", "15m")
                _mtf_specs = [(t, 100) for t in ("1h", "4h") if t != _base_tf]
                # trend_confirm runs its Layer3 entry/SL/TP/exit on 5m — fetch
                # enough 5m bars for EMA50 + cross history. Harmless extra data
                # for strategies that don't read it.
                if _base_tf != "5m":
                    _mtf_specs.append(("5m", 200))
                for tf, _lim in _mtf_specs:
                    try:
                        mtf_candles[tf] = await self.connector.fetch_ohlcv(
                            strategy.symbol, timeframe=tf, limit=_lim
                        )
                    except Exception:
                        pass

                # Post-cooldown tightening: raise this symbol's quality gate for
                # its first few re-entries after resuming from a loss-streak pause.
                if hasattr(strategy, "_entry_threshold_bonus"):
                    strategy._entry_threshold_bonus = self.risk.entry_threshold_bonus(strategy.symbol)
                signal = await strategy.analyze(candles, current_price, mtf_candles=mtf_candles)
                self._log_scan(strategy.symbol, strategy.name, current_price, signal)

                sig_dict = {
                    "strategy":   strategy.name,
                    "symbol":     signal.symbol,
                    "type":       signal.type.value,
                    "price":      signal.price,
                    "confidence": signal.confidence,
                    "reason":     signal.reason,
                    "ts":         int(time.time() * 1000),
                    "metadata":   signal.metadata,
                }
                new_signals.append(sig_dict)

                if signal.type != SignalType.HOLD:
                    if self._warmup_ticks_remaining > 0:
                        logger.info(
                            "[WARMUP] %s %s signal on %s suppressed — %d scan(s) left "
                            "before entries are allowed post-restart",
                            strategy.name, signal.type.value.upper(), strategy.symbol,
                            self._warmup_ticks_remaining,
                        )
                    else:
                        await self._maybe_notify(signal, sig_dict, strategy.name, candles)

            except Exception as e:
                logger.error("Strategy %s error: %s", strategy.name, e)
                new_signals.append({
                    "strategy": strategy.name,
                    "symbol":   strategy.symbol,
                    "type":     "error",
                    "reason":   str(e)[:80],
                    "ts":       int(time.time() * 1000),
                })

        if self._warmup_ticks_remaining > 0:
            self._warmup_ticks_remaining -= 1
            if self._warmup_ticks_remaining == 0:
                logger.info("[WARMUP] Complete — entries allowed from the next scan onward")

        self.state.signals         = (new_signals + self.state.signals)[:20]
        self.state.open_positions  = self.risk.get_positions()
        self.state.last_updated    = int(time.time() * 1000)
        self._broadcast_state()

    def _log_scan(self, symbol: str, strategy_name: str, price: float, signal: "Signal") -> None:
        """INFO-level one-line summary printed every tick for every symbol,
        so Railway logs show live scan activity even when no trade fires."""
        meta = signal.metadata or {}

        tc = meta.get("trend_confirm")
        if tc is not None:
            self._log_scan_trend_confirm(symbol, strategy_name, price, signal, tc)
            return

        macro      = meta.get("macro_trend", {})
        context1h  = meta.get("context_1h", {})
        mtf        = meta.get("mtf_combined", {})
        regime     = meta.get("regime", "?")
        strat_sel  = meta.get("selected_strategy", "?")
        strat_conf = meta.get("strategy_confidence")
        conf_str   = f"{strat_conf:.0f}" if isinstance(strat_conf, (int, float)) else "?"
        macro_str  = f"{macro.get('bias', '?')}/{macro.get('stage', '?')}({macro.get('score', 0):.0f})" if macro else "?"
        ctx_str    = f"{context1h.get('dominant_bias', '?')}/{context1h.get('stage', '?')}" if context1h else "?"
        aligned    = mtf.get("aligned_1h_4h")
        align_str  = "✓" if aligned else ("✗" if aligned is not None else "?")
        mtf_str    = f"{mtf.get('pct', 0):+.0f}%" if mtf else "?"
        reason     = (signal.reason or "")[:90]

        logger.info(
            "[SCAN] %-16s %-22s px=%-12.4f sig=%-4s regime=%-10s 4H=%-14s 1H=%-10s aligned=%s mtf=%-6s strat=%s(%s) | %s",
            strategy_name, symbol, price, signal.type.value.upper(),
            regime, macro_str, ctx_str, align_str, mtf_str, strat_sel, conf_str, reason,
        )

    def _log_scan_trend_confirm(self, symbol: str, strategy_name: str, price: float,
                                 signal: "Signal", tc: dict) -> None:
        """TrendConfirmStrategy-specific scan line for the 3-layer design:
        Layer1 (30m: SMA30/EMA10-20/EMA20 slope/MACD, all must agree on
        up or down), Layer2 (trend-quality score — per-TF Align/ADX/Chop/
        Volume, weighted 15m 65% + 1H 35%, must clear layer2_threshold),
        Layer3 (5m EMA10/20 cross with-trend + location/structure-room filter +
        price above/below EMA20 + within 1.5xATR of EMA50) — instead of the
        ai_expert-only fields (macro/context/mtf) that don't apply to this
        strategy."""
        sma_trend  = tc.get("sma_trend", "?")
        ema1020    = tc.get("ema10_20_trend", "?")
        slope      = tc.get("ema20_slope", "?")
        macd_trend = tc.get("macd_trend", "?")
        confirmed  = tc.get("confirmed") or "none"
        q15        = tc.get("q15")
        q1h        = tc.get("q1h")
        l2_score   = tc.get("layer2_score")
        l2_thr     = tc.get("layer2_threshold")
        open_pos   = tc.get("open_position") or "-"
        status     = tc.get("entry_status", "?")
        fb         = tc.get("fresh_trend_bars")
        is_early   = tc.get("is_early_trend")

        _STATUS_LABEL = {
            "position_open":            "holding",
            "no_trend":                 "n/a (Layer1 not confirmed)",
            "sideways_veto":            "n/a (Layer2 SIDEWAYS veto)",
            "quality_fail":             "n/a (Layer2a quality too low)",
            "early_quality_fail":       "FAIL (early trend, cross spent)",
            "location_reject":          "n/a (Layer2b location reject)",
            "location_quality_fail":    "n/a (Layer2b location-adjusted quality)",
            "waiting_cross":            "wait_ema_cross (Layer3 5m)",
            "ema_ref_fail":             "cross_ok/ema_ref_fail (Layer3 5m)",
            "cross_pass_distance_fail": "cross_ok/dist_fail (Layer3 5m)",
            "entered":                  "entered",
        }
        entry_str = _STATUS_LABEL.get(status, status)

        loc = tc.get("location") or {}
        loc_str = ""
        if loc.get("location_type") not in (None, "UNKNOWN"):
            room_r = loc.get("structure_room_r")
            room_s = f"{room_r:.2f}R" if room_r is not None else "n/a"
            loc_str = f" loc={loc.get('location_type')} room={room_s}"
            if not loc.get("valid"):
                loc_str += f" REJECT({loc.get('reason')})"
            elif loc.get("penalize"):
                loc_str += " penalized"

        l1_str = f"SMA={sma_trend} EMA10/20={ema1020} slope={slope} MACD={macd_trend}"
        if l2_score is not None:
            early_tag = "early" if is_early else "estab"
            score_str = f"{l2_score:.0f}/{l2_thr:.0f}[{early_tag}] (15m={q15:.0f} 1h={q1h:.0f})"
        else:
            score_str = "n/a"

        def _ago_str(ago: Optional[int]) -> str:
            return f"{ago}b" if ago is not None else "-"

        fresh_str = f"early<{fb}b" if is_early else "steady"
        if confirmed == "up":
            cross_str = f"EMA↑{_ago_str(tc.get('ema_cross_up_ago'))} ({fresh_str})"
        elif confirmed == "down":
            cross_str = f"EMA↓{_ago_str(tc.get('ema_cross_down_ago'))} ({fresh_str})"
        else:
            cross_str = "n/a"

        above = tc.get("above_ema_ref")
        ema_str = ("above" if above else "below") if above is not None else "?"
        dist_atr = tc.get("dist_atr")
        max_dist = tc.get("max_dist_atr")
        dist_str = f"{dist_atr:.2f}/{max_dist:.1f}xATR" if dist_atr is not None else "n/a"

        reason = (signal.reason or "")[:90]
        logger.info(
            "[SCAN] %-16s %-22s px=%-12.4f sig=%-4s L1[%s]=%-5s "
            "L2[quality=%s] pos=%-5s | "
            "L3[5m %s ema_ref=%s dist=%s]%s=%s | %s",
            strategy_name, symbol, price, signal.type.value.upper(),
            l1_str, confirmed, score_str,
            open_pos, cross_str, ema_str, dist_str, loc_str, entry_str, reason,
        )

    # ------------------------------------------------------------------
    # Adaptive position management helpers
    # ------------------------------------------------------------------

    async def _handle_pos_update(
        self,
        pos_info: dict,
        update: Any,   # PositionUpdate from position_manager
        price: float,
        strategy_inst: Optional[BaseStrategy] = None,
    ) -> bool:
        """
        Execute a Layer-7 PositionManager action.
        Returns True if the position was fully closed (caller should skip stop-check).
        """
        sym           = pos_info["symbol"]
        strategy_name = pos_info.get("strategy", "")

        if update.action in ("break_even", "trail"):
            if update.new_sl:
                self.risk.update_stop_loss(sym, update.new_sl, strategy=strategy_name)
                logger.info(
                    "[%s] %s %s → SL=%.4f | %s",
                    strategy_name, update.action.upper(), sym, update.new_sl, update.reason,
                )
            return False

        if update.action == "move_sl":
            # Ratchet the stop without closing any of the position, and push the
            # new stop to the exchange so OKX enforces the locked profit even if
            # the bot is offline. Position stays open (rides to the cross-back).
            if update.new_sl:
                self.risk.update_stop_loss(sym, update.new_sl, strategy=strategy_name)
                logger.info("[%s] MOVE_SL %s → SL=%.4f | %s",
                            strategy_name, sym, update.new_sl, update.reason)
                try:
                    await self.connector.set_position_tpsl(
                        sym, pos_info["side"], pos_info["amount"],
                        sl=update.new_sl, tp=pos_info.get("take_profit"))
                except Exception as e:
                    logger.warning("[TPSL] move_sl exchange update failed for %s: %s", sym, e)
                if self.telegram:
                    try:
                        self.telegram.notify(
                            f"🔒 *SL moved to lock profit* `{sym}` [{strategy_name}]\n"
                            f"New SL: `{update.new_sl:,.4f}`\n"
                            f"_{update.reason}_")
                    except Exception as e:
                        logger.warning("[%s] move_sl notify failed: %s", strategy_name, e)
            return False

        if update.action == "partial_tp":
            close_amt = round(pos_info["amount"] * update.close_pct, 8)
            if close_amt > 0:
                try:
                    close_side = "sell" if pos_info["side"] == "long" else "buy"
                    pos_side = pos_info["side"] if self._hedge_mode else None
                    close_order = await self.connector.create_order(sym, close_side, close_amt, pos_side=pos_side)
                    pos_key = f"{sym}||{strategy_name}"
                    entry_fill = self._entry_fills.get(pos_key)
                    close_frac = (close_amt / entry_fill["size"]) if (entry_fill and entry_fill.get("size")) \
                                 else update.close_pct
                    fill = self._close_fill_info(pos_key, close_order, price, close_amt,
                                                 close_frac, final=False)
                    exit_px = fill["exit_avg_px"]
                    pnl = (fill["net_pnl"] if fill["net_pnl"] is not None else
                           ((exit_px - pos_info["entry"]) * close_amt
                            if pos_info["side"] == "long"
                            else (pos_info["entry"] - exit_px) * close_amt))
                    self.risk.reduce_position(sym, fill["exit_sz"], strategy=strategy_name)
                    # Move SL to break-even when TP1 fires (new_sl is entry price)
                    if update.new_sl is not None:
                        self.risk.update_stop_loss(sym, update.new_sl, strategy=strategy_name)
                        logger.info("[%s] SL moved to BE=%.4f after TP1", strategy_name, update.new_sl)
                        # Re-place the exchange SL/TP on the REMAINING size at BE,
                        # so OKX enforces break-even + the original target.
                        remaining = max(0.0, pos_info["amount"] - fill["exit_sz"])
                        if remaining > 0:
                            try:
                                await self.connector.set_position_tpsl(
                                    sym, pos_info["side"], remaining,
                                    sl=update.new_sl, tp=pos_info.get("take_profit"))
                            except Exception as e:
                                logger.warning("[TPSL] BE update failed for %s: %s", sym, e)
                    self._record_trade(TradeRecord(
                        timestamp=int(time.time() * 1000),
                        symbol=sym, side=close_side,
                        price=exit_px, amount=fill["exit_sz"],
                        pnl=round(pnl, 4),
                        strategy=strategy_name, reason=update.reason,
                        paper=self.connector.paper,
                    ))
                    logger.info(
                        "[%s] Partial TP %.0f%% %s @ %.4f NetPnL=%.4f (fees in=%.4f out=%.4f) | %s",
                        strategy_name, update.close_pct * 100, sym, exit_px, pnl,
                        fill["entry_fee_alloc"], fill["exit_fee"], update.reason,
                    )
                    self._check_cooldown_trigger(pnl, sym)
                    # Book the partial into the paper account (fixed entry-size).
                    self._sig.record_paper_partial(
                        sym, exit_px, update.close_pct, strategy=strategy_name,
                    )
                    if self.telegram:
                        try:
                            sign = "+" if pnl >= 0 else "-"
                            fees_total = fill["entry_fee_alloc"] + fill["exit_fee"]
                            self.telegram.notify(
                                f"💰 *Partial Take-Profit* `{sym}` [{strategy_name}]\n"
                                f"Fill: `{fill['exit_sz']:.6g}` @ `{exit_px:,.4f}`\n"
                                f"💵 Net P&L: `{sign}${abs(pnl):,.4f}`  "
                                f"(fees: `${fees_total:,.4f}`)\n"
                                f"_{update.reason}_"
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error("Partial TP failed [%s %s]: %s", strategy_name, sym, e)
            return False

        if update.action == "close":
            close_side = "sell" if pos_info["side"] == "long" else "buy"
            pos_side = pos_info["side"] if self._hedge_mode else None
            try:
                # Clear the exchange SL/TP first so no reduce-only algo lingers
                # after we close the position ourselves.
                try:
                    await self.connector.set_position_tpsl(sym, pos_info["side"], 0.0)
                except Exception:
                    pass
                close_order = await self.connector.create_order(sym, close_side, pos_info["amount"], pos_side=pos_side)
                fill = self._close_fill_info(f"{sym}||{strategy_name}", close_order, price,
                                             pos_info["amount"], 1.0, final=True)
                exit_px = fill["exit_avg_px"]
                pnl = (fill["net_pnl"] if fill["net_pnl"] is not None else
                       ((exit_px - pos_info["entry"]) * pos_info["amount"]
                        if pos_info["side"] == "long"
                        else (pos_info["entry"] - exit_px) * pos_info["amount"]))
                self._record_trade(TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym, side=close_side,
                    price=exit_px, amount=fill["exit_sz"],
                    pnl=round(pnl, 4),
                    strategy=strategy_name, reason=update.reason,
                    paper=self.connector.paper,
                ))
                _outcome = self._sig.record_outcome(
                    symbol=sym, side=pos_info["side"],
                    entry=pos_info["entry"], exit_price=exit_px,
                    sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"),
                    reason=update.reason, strategy=strategy_name, fill=fill,
                )
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                self._on_position_closed(sym, strategy_name, exit_px, update.reason, strategy_inst)
                logger.info(
                    "[%s] AI-driven CLOSE %s @ %.4f NetPnL=%.4f | %s",
                    strategy_name, sym, exit_px, pnl, update.reason,
                )
                if self.telegram:
                    self.telegram.notify_trade_closed(sym, _outcome, self._sig.summary())
                self._check_cooldown_trigger(pnl, sym)
            except Exception as e:
                logger.error("AI-driven close failed [%s %s]: %s", strategy_name, sym, e)
            return True

        return False

    def _check_cooldown_trigger(self, pnl: float, symbol: str = "") -> None:
        """Feed a closed trade's PnL into that SYMBOL's consecutive-loss streak.
        Notifies Telegram the moment a per-symbol cooldown gets triggered."""
        triggered = self.risk.record_trade_result(pnl, symbol)
        if triggered:
            hours = self.risk.cooldown_seconds / 3600
            strict_n = self.risk.post_cooldown_strict_trades
            logger.warning(
                "[%s] Cooldown triggered: %d consecutive losing closes — %s paused for %.1fh "
                "(next %d entries tightened after resume)",
                symbol, self.risk.max_consecutive_sl, symbol, hours, strict_n,
            )
            if self.telegram:
                self.telegram.notify_cooldown_halt(self.risk.max_consecutive_sl, hours,
                                                   symbol=symbol, strict_trades=strict_n)

    def _on_position_closed(
        self,
        symbol: str,
        strategy_name: str,
        exit_price: float,
        reason: str,
        strategy_inst: Optional[BaseStrategy] = None,
    ) -> None:
        """
        Lifecycle callback called after any position close (SL/TP, signal, or AI exit).
        Feeds the learning engine and releases portfolio engine slot.
        """
        pos_key   = f"{symbol}||{strategy_name}"
        open_time = self._position_open_times.pop(pos_key, time.time())
        duration_min = (time.time() - open_time) / 60.0

        if strategy_inst is None:
            strategy_inst = self._resolve_strategy_inst(strategy_name)

        # Layer 9: feed learning engine
        if hasattr(strategy_inst, "record_closed_trade"):
            try:
                strategy_inst.record_closed_trade(exit_price, reason, duration_min)
                logger.debug(
                    "[%s] Learning engine updated: exit=%.4f reason=%s dur=%.1fmin",
                    strategy_name, exit_price, reason, duration_min,
                )
            except Exception as e:
                logger.warning("record_closed_trade failed [%s]: %s", strategy_name, e)

        # Release portfolio engine slot
        try:
            self._portfolio.remove_position(symbol)
        except Exception:
            pass

    def _broadcast_drift_alerts(self) -> None:
        """Check all AI Expert strategies for drift and alert if action is needed."""
        for strategy in self.strategies:
            if not hasattr(strategy, "_drift_detector"):
                continue
            try:
                action = strategy._drift_detector.highest_severity_action()
                if action in (DriftAction.RETRAIN, DriftAction.PAUSE):
                    logger.warning(
                        "[%s] Drift detected: action=%s", strategy.name, action.value
                    )
                    if self.telegram:
                        try:
                            self.telegram.notify(
                                f"⚠️ *Drift Alert* [{strategy.name}]\n"
                                f"Action: *{action.value.upper()}*\n"
                                f"Model performance has degraded — review trading conditions."
                            )
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("Drift check failed [%s]: %s", strategy.name, e)

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    async def _maybe_notify(self, signal: Signal, sig_dict: dict,
                             strategy_name: str, candles: list = None):
        """Execution engine.

        Normal mode  — BUY → open long; SELL → close existing long only.
        Hedge mode   — BUY → open long; SELL → open short (independent direction).
        Strategy isolation: a SELL from strategy A cannot close strategy B's position.
        """
        sym = signal.symbol

        # ── WTV gate ────────────────────────────────────────────────────────
        if self.wt_verify and candles:
            wt1 = WTADXStrategy.compute_wt1(candles)
            if not math.isnan(wt1):
                if signal.type == SignalType.BUY and wt1 >= 10:
                    logger.debug("[WTV] %s %s BUY blocked — WT1=%.1f ≥+10", strategy_name, sym, wt1)
                    return
                if signal.type == SignalType.SELL and wt1 <= -10:
                    logger.debug("[WTV] %s %s SELL blocked — WT1=%.1f ≤-10", strategy_name, sym, wt1)
                    return
                logger.debug("[WTV] %s %s passed — WT1=%.1f", strategy_name, sym, wt1)

        signal_only = self.risk.max_open_positions == 0

        if signal_only:
            last_dir, last_ts = self._sig.last_direction(sym)
            direction_changed = last_dir != signal.type.value
            stale = (int(time.time() * 1000) - last_ts) > 4 * 3600 * 1000
            if direction_changed or stale:
                self._sig.lock(sym, signal.type.value)
                self._sig.record_signal(sym, signal.type.value, signal.price,
                                        signal.confidence, strategy=strategy_name)
                meta = signal.metadata or {}
                sl_p = meta.get("stop_loss"); tp_p = meta.get("take_profit")
                if sl_p and tp_p:
                    vkey = f"forex||{sym}||{int(time.time() * 1000)}"
                    self._sig.add_pending(vkey, sym, signal.type.value, signal.price,
                                          sl_p, tp_p, strategy=strategy_name)
                if self.telegram:
                    self.telegram.notify_signal(sig_dict)
            else:
                logger.debug("Forex %s suppressed — same direction", sym)
            return

        # ── WaveTrend: 2-slot long stacking ─────────────────────────────────
        _WT = "WTADXStrategy"
        _WT_MAX = 2
        if strategy_name == _WT:
            positions = self.risk.get_positions()
            wt_pos   = [p for p in positions if p["symbol"] == sym and p["strategy"].startswith(_WT)]
            wt_longs = [p for p in wt_pos if p["side"] == "long"]

            if signal.type == SignalType.SELL:
                if not wt_longs:
                    return
                logger.info("[WT] SELL signal on %s — exiting %d long(s)", sym, len(wt_longs))
                ticker     = await self.connector.fetch_ticker(sym)
                exit_price = ticker["last"]
                for pos in list(wt_longs):
                    slot_name = pos["strategy"]
                    try:
                        close_order = await self.connector.create_order(sym, "sell", pos["amount"])
                        fill = self._close_fill_info(f"{sym}||{slot_name}", close_order, exit_price,
                                                     pos["amount"], 1.0, final=True)
                        _outcome = self._sig.record_outcome(
                            symbol=sym, side="long",
                            entry=pos["entry"], exit_price=fill["exit_avg_px"],
                            sl=pos.get("stop_loss"), tp=pos.get("take_profit"),
                            reason="sell_signal", strategy=slot_name, fill=fill,
                        )
                        self._sig.unlock_strategy(sym, slot_name)
                        self.risk.close_position(sym, strategy=slot_name)
                        self._on_position_closed(sym, slot_name, exit_price, "sell_signal")
                        if self.telegram:
                            self.telegram.notify_trade_closed(sym, _outcome, self._sig.summary())
                    except Exception as e:
                        logger.error("WT exit on SELL failed [%s]: %s", slot_name, e)
                return

            if len(wt_longs) >= _WT_MAX:
                logger.debug("[WT] %s max stack (%d) reached, suppressing", sym, _WT_MAX)
                return
            slot_key = None
            for s in range(_WT_MAX):
                candidate = f"{_WT}#{s}"
                if not self._sig.is_locked_for_strategy(sym, candidate):
                    slot_key = candidate
                    break
            if slot_key is None:
                return
            can, reason = self.risk.can_open(sym, strategy=slot_key)
            if not can:
                logger.debug("[WT] %s can_open blocked: %s", sym, reason)
                return
            self._sig.lock_strategy(sym, slot_key, signal.type.value)
            self._sig.record_signal(sym, signal.type.value, signal.price,
                                    signal.confidence, strategy=slot_key)
            # Notification (with chart) is sent from _execute_signal ONLY after
            # the position actually opens — no pre-open signal spam.
            await self._execute_signal(signal, slot_key, candles=candles)
            return

        # ── Normal / AI Expert strategies ────────────────────────────────────
        if signal.type == SignalType.SELL:
            if self._hedge_mode:
                # Hedge mode: SELL signal opens a SHORT position (independent of any LONG)
                short_key = f"{strategy_name}:S"
                if self._sig.is_locked_for_strategy(sym, short_key):
                    logger.debug("%s [%s] already short — suppressing SELL", sym, short_key)
                    return
                can, reason = self.risk.can_open(sym, strategy=short_key)
                if not can:
                    logger.debug("Crypto %s short suppressed — %s", sym, reason)
                    self._cancel_pending_entry(short_key, reason=reason)
                    return
                self._sig.lock_strategy(sym, short_key, signal.type.value)
                self._sig.record_signal(sym, signal.type.value, signal.price,
                                        signal.confidence, strategy=short_key)
                # Notification (with chart) is sent post-open from _execute_signal.
                await self._execute_signal(signal, short_key, direction="short", candles=candles)
            else:
                # Normal mode: SELL closes existing long only
                if self._sig.is_locked_for_strategy(sym, strategy_name):
                    positions = self.risk.get_positions()
                    existing  = next(
                        (p for p in positions if p["symbol"] == sym and p["strategy"] == strategy_name),
                        None,
                    )
                    if existing and existing["side"] == "long":
                        logger.info("[%s] SELL signal — exiting long on %s", strategy_name, sym)
                        try:
                            ticker     = await self.connector.fetch_ticker(sym)
                            exit_price = ticker["last"]
                            close_order = await self.connector.create_order(sym, "sell", existing["amount"])
                            fill = self._close_fill_info(f"{sym}||{strategy_name}", close_order, exit_price,
                                                         existing["amount"], 1.0, final=True)
                            _outcome = self._sig.record_outcome(
                                symbol=sym, side="long",
                                entry=existing["entry"], exit_price=fill["exit_avg_px"],
                                sl=existing.get("stop_loss"), tp=existing.get("take_profit"),
                                reason="sell_signal", strategy=strategy_name, fill=fill,
                            )
                            self._sig.unlock_strategy(sym, strategy_name)
                            self.risk.close_position(sym, strategy=strategy_name)
                            self._on_position_closed(
                                sym, strategy_name, exit_price, "sell_signal",
                                self._resolve_strategy_inst(strategy_name),
                            )
                            if self.telegram:
                                self.telegram.notify_trade_closed(sym, _outcome, self._sig.summary())
                        except Exception as e:
                            logger.error("Exit on SELL signal failed [%s]: %s", strategy_name, e)
            return

        # BUY signal → open LONG
        long_key = f"{strategy_name}:L" if self._hedge_mode else strategy_name
        if self._sig.is_locked_for_strategy(sym, long_key):
            logger.debug("%s [%s] already long — suppressing BUY", sym, long_key)
            return
        can, reason = self.risk.can_open(sym, strategy=long_key)
        if not can:
            logger.debug("Crypto %s suppressed — %s", sym, reason)
            self._cancel_pending_entry(long_key, reason=reason)
            return
        self._sig.lock_strategy(sym, long_key, signal.type.value)
        self._sig.record_signal(sym, signal.type.value, signal.price,
                                signal.confidence, strategy=long_key)
        # Notification (with chart) is sent post-open from _execute_signal.
        await self._execute_signal(signal, long_key, direction="long", candles=candles)

    async def _execute_signal(self, signal: Signal, strategy_name: str,
                              direction: str = "long", candles: list = None):
        """Open a position. direction='long' for BUY, 'short' for SELL (hedge mode)."""
        sym       = signal.symbol
        order_side = "buy" if direction == "long" else "sell"
        pos_side   = direction if self._hedge_mode else None

        can, reason = self.risk.can_open(sym, strategy=strategy_name)
        if not can:
            logger.info("Skipping %s for %s: %s", direction.upper(), sym, reason)
            self._sig.unlock_strategy(sym, strategy_name)
            self._cancel_pending_entry(strategy_name, reason=reason)
            return

        # Futures setup (leverage + hedge mode) — once per symbol per session
        if self._hedge_mode and sym not in self._futures_setup_done:
            if hasattr(self.connector, "setup_futures"):
                try:
                    await self.connector.setup_futures(sym)
                except Exception as e:
                    logger.warning("[Futures] setup_futures failed for %s: %s", sym, e)
            self._futures_setup_done.add(sym)

        # ── Balance check (always fresh, right before sizing) ─────────────────
        balances      = await self.connector.fetch_balance()
        _quote = [b for b in balances if b.asset in ("USDT", "USD", "BUSD")]
        quote_balance  = next((b.free for b in _quote), 0)          # spendable now
        equity_balance = next((b.total for b in _quote), quote_balance)  # total account value
        ticker        = await self.connector.fetch_ticker(sym)
        price         = ticker["last"]
        meta          = signal.metadata or {}
        sl_p          = meta.get("stop_loss")
        tp_p          = meta.get("take_profit")

        min_balance = float(os.getenv("MIN_BALANCE_USD", "10"))
        logger.info(
            "[%s] Balance check: equity=$%.2f  free=$%.2f  min_required=$%.2f  (paper=%s)",
            strategy_name, equity_balance, quote_balance, min_balance, self.connector.paper,
        )
        if quote_balance < min_balance:
            logger.warning(
                "[%s] Balance too low ($%.2f < $%.2f) — skipping %s on %s",
                strategy_name, quote_balance, min_balance, direction.upper(), sym,
            )
            self._sig.unlock_strategy(sym, strategy_name)
            self._cancel_pending_entry(strategy_name, reason="balance too low")
            return

        # Flip SL/TP for short positions
        if direction == "short" and sl_p and tp_p:
            if sl_p < price:  # AI gave LONG SL/TP; invert for short
                sl_p = price + (price - sl_p)
                tp_p = price - (tp_p - price)

        # Leverage used for sizing MUST match what the exchange has set, or the
        # notional comes out N× wrong. Prefer the connector's configured value;
        # fall back to the LEVERAGE env so a connector that didn't receive it
        # still sizes correctly.
        # Take the MAX of the connector's configured leverage and the LEVERAGE
        # env. This defends against the recurring "size came out ~10x too small"
        # bug: if the connector's _leverage was left stale/low (e.g. 2) while the
        # OKX position is actually 20x, using the low value would size the
        # notional ~10x under. env LEVERAGE=20 is the source of truth, so never
        # size with anything below it.
        conn_lev      = getattr(self.connector, "_leverage", None) or 0
        env_lev       = int(os.getenv("LEVERAGE", "20"))
        leverage      = max(conn_lev, env_lev, 1)
        is_futures    = getattr(self.connector, "_futures", False)
        # Default to margin-based sizing (5% of equity × leverage) — the mode
        # the account is meant to run. Strategy metadata or SIZING_MODE env can
        # override. (Previously defaulted to "risk", so any strategy that didn't
        # explicitly set sizing_mode got risk-based/tiny sizes.)
        sizing_mode   = meta.get("sizing_mode") or os.getenv("SIZING_MODE", "margin")
        if conn_lev and conn_lev != env_lev:
            logger.warning(
                "[%s] leverage mismatch: connector=%s but LEVERAGE env=%s — using %s. "
                "If the OKX position shows a different x than this, sizing will be off.",
                strategy_name, conn_lev, env_lev, leverage,
            )

        # ── Fixed-MARGIN sizing (highest precedence) ─────────────────────────
        # Lock a FIXED $ margin (collateral) per trade, then notional = margin ×
        # leverage. e.g. $35 margin × 20x = $700 position. This is what shows in
        # the OKX "Margin" column. Aggressive on a small account — one trade's
        # margin is a big share of the balance — so the free-balance clamp below
        # still applies (can't lock more margin than the account can cover), but
        # the % concentration cap is skipped (this IS the intended big size).
        # FIXED_MARGIN_USD=0 disables it.
        fixed_margin = float(os.getenv("FIXED_MARGIN_USD", "20"))
        fixed_notional = float(os.getenv("FIXED_NOTIONAL_USD", "0"))
        if fixed_margin > 0:
            sizing_mode = "fixed_margin"
            notional_target = fixed_margin * (leverage if is_futures else 1)
            amount = round(notional_target / price, 6) if price > 0 else 0
            risk_per_unit = abs(price - sl_p) if sl_p else 0
            sizing_label = (f"fixed-margin ${fixed_margin:.2f} × {leverage}x = notional "
                            f"${notional_target:.2f} → amount {amount:g} @ ${price:,.4f}")
        elif fixed_notional > 0:
            sizing_mode = "fixed"
            notional_target = fixed_notional
            amount = round(notional_target / price, 6) if price > 0 else 0
            risk_per_unit = abs(price - sl_p) if sl_p else 0
            sizing_label = (f"fixed-notional ${fixed_notional:.2f} → amount {amount:g} "
                            f"@ ${price:,.4f} (margin ${fixed_notional/max(leverage,1):.2f} at {leverage}x)")
        elif sizing_mode == "margin":
            # ── Margin-based sizing (opt-in via signal.metadata) ─────────────
            # Position margin = margin_pct of TOTAL account equity (so "5% of
            # balance" stays consistent whether or not another position is
            # already open), notional = margin × leverage. e.g. $100 equity ×
            # 5% = $5 margin × 20x = $100 notional. The free-balance
            # availability clamp below still stops us exceeding spendable cash.
            margin_pct = float(meta.get("margin_pct", 0.05))
            margin = equity_balance * margin_pct
            notional_target = margin * (leverage if is_futures else 1)
            amount = round(notional_target / price, 6) if price > 0 else 0
            risk_per_unit = abs(price - sl_p) if sl_p else 0
            sizing_label = (f"margin-based {margin_pct*100:.1f}% of equity ${equity_balance:.2f} "
                            f"→ margin ${margin:.2f} × {leverage}x = notional ${notional_target:.2f}")
        else:
            # ── Risk-based position sizing ───────────────────────────────────
            # amount is sized so that a full SL hit loses exactly RISK_PER_TRADE_PCT
            # of balance — not a fixed % of balance as notional (that conflates
            # position size with risk, which vary a lot with how far SL sits).
            # Layer 8 (Dynamic Risk Engine) scales this base risk% up/down per
            # trade based on market quality / confidence / expectancy / volatility.
            base_risk_pct    = float(os.getenv("RISK_PER_TRADE_PCT", "0.05"))
            risk_multiplier  = meta.get("dynamic_risk", {}).get("risk_multiplier", 1.0)
            risk_pct         = base_risk_pct * risk_multiplier
            risk_per_unit    = abs(price - sl_p) if sl_p else 0

            if risk_per_unit > 0:
                risk_dollars = quote_balance * risk_pct
                amount = round(risk_dollars / risk_per_unit, 6)
                sizing_label = (
                    f"risk-based {risk_pct*100:.2f}% (base {base_risk_pct*100:.1f}% x "
                    f"{risk_multiplier:.2f} Layer8) — SL {risk_per_unit/price*100:.2f}% away"
                )
            else:
                # No usable SL from the signal — fall back to confidence-tiered notional sizing
                size_pct = _confidence_size_pct(meta)
                amount = self.risk.size_position(quote_balance, price, size_pct=size_pct)
                sizing_label = f"confidence-based {size_pct*100:.1f}% (no SL available)"

        # ── Margin concentration cap — independent of the risk target ─────────
        # A very tight SL (e.g. low-volatility regime) blows up amount = risk$/SL-distance
        # to a huge notional under risk-based sizing. Cap it so one trade can
        # never lock up more than MAX_MARGIN_PCT_PER_TRADE (margin-mode already
        # targets a fixed, typically-smaller share, so this rarely engages there).
        max_margin_pct = float(os.getenv("MAX_MARGIN_PCT_PER_TRADE", "0.20"))
        notional        = amount * price
        required_margin = (notional / leverage) if is_futures else notional
        max_margin      = quote_balance * max_margin_pct

        if sizing_mode not in ("margin", "fixed", "fixed_margin") and required_margin > max_margin:
            max_notional_by_risk_cap = max_margin * (leverage if is_futures else 1)
            capped_amount = round(max_notional_by_risk_cap / price, 6) if price > 0 else 0
            logger.warning(
                "[%s] SL too tight (%.2f%% away) — risk-sized margin $%.2f would exceed the "
                "%.0f%% per-trade cap ($%.2f). Capping size %.6f → %.6f",
                strategy_name, risk_per_unit / price * 100, required_margin,
                max_margin_pct * 100, max_margin, amount, capped_amount,
            )
            amount = capped_amount
            notional = amount * price
            required_margin = (notional / leverage) if is_futures else notional

        # ── Margin availability check: clamp size to what the free balance can
        # actually cover, instead of attempting an order that the exchange
        # would reject.
        safety_buffer = 0.95  # leave headroom for fees/slippage
        if required_margin > quote_balance * safety_buffer:
            if sizing_mode in ("fixed", "fixed_margin"):
                # Fixed sizing means "this exact size or nothing" — clamping it
                # down produces a meaningless micro-position that only pays fees
                # (e.g. $0.33 margin instead of $35 when the balance is tied up
                # in other positions). Skip the trade instead; it can re-fire
                # once capital frees up.
                logger.warning(
                    "[%s] Fixed sizing needs $%.2f margin but only $%.2f free — SKIPPING "
                    "this entry (won't open a clamped micro-position). Frees up when other "
                    "positions close.",
                    strategy_name, required_margin, quote_balance,
                )
                amount = 0
            else:
                max_notional = quote_balance * safety_buffer * (leverage if is_futures else 1)
                clamped_amount = round(max_notional / price, 6) if price > 0 else 0
                logger.warning(
                    "[%s] Margin required $%.2f exceeds available $%.2f — clamping size %.6f → %.6f",
                    strategy_name, required_margin, quote_balance, amount, clamped_amount,
                )
                amount = clamped_amount

        actual_risk_dollars = amount * risk_per_unit if risk_per_unit > 0 else 0
        logger.info(
            "[%s] Position size: %s → %.6f  "
            "(notional=$%.2f, margin=$%.2f, leverage=%dx, risk=$%.2f)",
            strategy_name, sizing_label, amount, amount * price,
            (amount * price / leverage) if is_futures else amount * price,
            leverage, actual_risk_dollars,
        )
        if amount <= 0:
            logger.info("Position size 0 for %s — tracking virtually", sym)
            if sl_p and tp_p:
                vkey = f"virtual||{sym}||{strategy_name}||{int(time.time() * 1000)}"
                self._sig.add_pending(vkey, sym, signal.type.value, signal.price,
                                      sl_p, tp_p, strategy=strategy_name)
            self._sig.unlock_strategy(sym, strategy_name)
            self._cancel_pending_entry(strategy_name, reason="position size rounded to 0")
            return

        # ── Portfolio engine pre-trade check ────────────────────────────────
        if direction == "long":
            effective_sl = sl_p or (price * (1 - self.risk.stop_loss_pct))
        else:
            effective_sl = sl_p or (price * (1 + self.risk.stop_loss_pct))
        port_check = self._portfolio.can_add_position(
            symbol=sym, direction=direction,
            entry_price=price, stop_loss=effective_sl,
            amount=amount, portfolio_value=quote_balance,
        )
        if not port_check.can_add:
            logger.info(
                "[%s] Portfolio gate blocked %s %s: %s (heat=%.1f%%)",
                strategy_name, direction.upper(), sym, port_check.reason,
                port_check.portfolio_heat * 100,
            )
            self._sig.unlock_strategy(sym, strategy_name)
            self._cancel_pending_entry(strategy_name, reason=port_check.reason)
            return

        try:
            order = await self.connector.create_order(
                sym, order_side, amount, pos_side=pos_side
            )
            # From here on, sizes/prices come from the exchange's post-fill
            # numbers (avgPx / fillSz / fee), not our request.
            entry_px = order.price or price
            entry_sz = order.filled or amount
            self.risk.open_position(sym, direction, entry_px, entry_sz, strategy=strategy_name,
                                    stop_loss=sl_p, take_profit=tp_p)
            # Count down the post-cooldown strict window once a real entry opens.
            self.risk.consume_strict_entry(sym)

            # Snapshot the paper-account position at entry so partial TPs and
            # the final close book against one fixed size (keeps $ coherent).
            self._sig.open_paper_position(sym, direction, entry_px, sl_p, strategy=strategy_name)

            # Track open time for learning engine duration calculation
            pos_key = f"{sym}||{strategy_name}"
            self._position_open_times[pos_key] = time.time()
            self._entry_fills[pos_key] = {
                "avg_px": entry_px, "size": entry_sz,
                "fee": getattr(order, "fee", 0.0) or 0.0, "fee_frac_left": 1.0,
            }

            # Attach the hard SL/TP on the exchange itself (OKX algo orders), so
            # the stop/target are enforced even if the bot loop stalls. No-op in
            # paper/spot; failure is logged and never blocks the open position.
            if sl_p or tp_p:
                try:
                    await self.connector.set_position_tpsl(
                        sym, direction, entry_sz, sl=sl_p, tp=tp_p)
                except Exception as e:
                    logger.warning("[TPSL] set on open failed for %s: %s", sym, e)

            # Register position in portfolio engine
            self._portfolio.add_position(
                symbol=sym, direction=direction,
                entry_price=entry_px, current_price=entry_px,
                amount=entry_sz, stop_loss=effective_sl,
            )

            trade = TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=sym, side=order_side,
                price=entry_px, amount=entry_sz,
                pnl=0.0, strategy=strategy_name, reason=signal.reason,
                paper=self.connector.paper,
            )
            self._record_trade(trade)
            logger.info(
                "[%s] %s %s @ %.4f  SL=%.4f  TP=%.4f (paper=%s)",
                strategy_name, direction.upper(), sym, price,
                sl_p or 0, tp_p or 0, self.connector.paper,
            )
        except Exception as e:
            # Order genuinely never went through — no real position exists,
            # so it's safe (and necessary) to unlock and let the strategy
            # know its optimistic entry latch didn't pan out.
            logger.error("Order failed for %s %s: %s", direction, sym, e)
            self._sig.unlock_strategy(sym, strategy_name)
            self._cancel_pending_entry(strategy_name, reason=f"order failed: {e}")
            return

        # ── Telegram notify — deliberately OUTSIDE the order try/except: a
        # notify failure here must never be mistaken for an order failure.
        # The position is already open at this point regardless of whether
        # the notification succeeds. ──────────────────────────────────────
        if self.telegram:
            try:
                meta = signal.metadata or {}
                macro_info = meta.get("macro_trend", {})
                early_trend = bool(meta.get("trend_confirm", {}).get("is_early_trend"))
                dir_label = None
                if early_trend:
                    # plain-text tag for the chart title — matplotlib's font
                    # has no emoji glyph, so keep the 🌱 to the TG caption only.
                    dir_label = ("LONG (+) [EARLY]" if direction == "long"
                                 else "SHORT (-) [EARLY]")
                chart_path = None
                strategy_inst = self._resolve_strategy_inst(strategy_name)
                # Render on the timeframe the strategy actually entered on: if
                # it exposes a finer entry series (trend_confirm's 5m), chart
                # THAT so the EMA lines match the trade; else the base candles.
                # render_entry_chart needs >=20 bars — pick whichever series has
                # enough. A short 5m series must NOT shadow a full 15m one
                # (that silently killed the chart for some symbols, e.g. HYPE).
                _l5 = getattr(strategy_inst, "_latest_5m", None)
                if _l5 and len(_l5) >= 20:
                    chart_candles = _l5
                elif candles and len(candles) >= 20:
                    chart_candles = candles
                else:
                    chart_candles = _l5 or candles  # best effort; may still be too short
                if chart_candles:
                    try:
                        from .chart_renderer import render_entry_chart
                        chart_kwargs = self._chart_ma_kwargs(strategy_inst)
                        chart_path = render_entry_chart(
                            chart_candles, sym, direction, order.price,
                            sl=sl_p, tp=tp_p, strategy=strategy_name,
                            macro_bias=macro_info.get("bias", ""),
                            dir_label=dir_label,
                            **chart_kwargs,
                        )
                    except Exception as e:
                        logger.warning("Chart render failed for %s: %s", sym, e)
                if chart_path is None:
                    logger.warning(
                        "[%s] no entry chart sent — 5m bars=%s, base bars=%s (need >=20). "
                        "Notification goes out text-only.",
                        sym, len(_l5) if _l5 else 0, len(candles) if candles else 0,
                    )
                _fill_px = order.price or entry_px or price
                _fill_sz = order.filled or amount
                _notional = _fill_sz * _fill_px
                _margin = (_notional / leverage) if (is_futures and leverage) else _notional
                _use_trail = getattr(strategy_inst, "use_be_trail", False)
                _trig_r = getattr(strategy_inst, "be_trail_trigger_r", None) if _use_trail else None
                _sl_r = getattr(strategy_inst, "be_trail_sl_r", None) if _use_trail else None
                self.telegram.notify_order(
                    sym, order_side, _fill_sz, _fill_px,
                    strategy_name, self.connector.paper,
                    fee=getattr(order, "fee", 0.0),
                    sl=sl_p, tp=tp_p,
                    notional=_notional, margin=_margin,
                    trail_trigger_r=_trig_r, trail_sl_r=_sl_r,
                    macro_score=macro_info.get("score"),
                    macro_bias=macro_info.get("bias"),
                    selected_strategy=meta.get("selected_strategy"),
                    strategy_confidence=meta.get("strategy_confidence"),
                    regime=meta.get("regime"),
                    direction=direction,
                    chart_path=chart_path,
                    early_trend=early_trend,
                    reason=signal.reason,
                )
            except Exception as e:
                logger.warning("Telegram notify_order failed for %s %s (position is still open): %s", sym, direction, e)
                # Never leave an opened position unannounced — send a minimal
                # text alert as a fallback if the rich (chart) notification blew up.
                try:
                    _sl = f"{sl_p:.4f}" if sl_p else "—"
                    self.telegram.notify(
                        f"🟢 *Order Executed* {'📄 PAPER' if self.connector.paper else '💰 LIVE'}\n"
                        f"`{sym}` — *{direction.upper()}*\n"
                        f"Fill: `{(order.filled or amount):.6g}` @ "
                        f"`{(order.price or entry_px or price):,.4f}`  |  SL: `{_sl}`\n"
                        f"_{signal.reason}_"
                    )
                except Exception as e2:
                    logger.error("Fallback order notify also failed for %s: %s", sym, e2)

    # ------------------------------------------------------------------
    # Balance, state, and stats helpers
    # ------------------------------------------------------------------

    async def _refresh_balance(self):
        try:
            balances = await self.connector.fetch_balance()
            self.state.total_balance = sum(
                b.total for b in balances if b.asset in ("USDT", "USD", "BUSD", "BTC", "ETH")
            )
            self.state.equity = self.state.total_balance
            if self._start_balance:
                self.state.pnl_total = self.state.total_balance - self._start_balance
                self.risk.update_peak(self.state.total_balance)
            self._portfolio.update_prices({})   # price updates handled per-tick above
        except Exception as e:
            logger.warning("Balance refresh failed: %s", e)

    def _record_trade(self, trade: TradeRecord):
        self._trade_history.append(trade)
        self.state.recent_trades = [
            {
                "ts": t.timestamp, "symbol": t.symbol, "side": t.side,
                "price": t.price, "amount": t.amount, "pnl": t.pnl,
                "strategy": t.strategy, "reason": t.reason, "paper": t.paper,
            }
            for t in self._trade_history[-50:]
        ]

    def _broadcast_state(self):
        try:
            self._broadcast({
                "type": "trading_update",
                "state": {
                    "running":       self.state.running,
                    "paper":         self.state.paper,
                    "balance":       round(self.state.total_balance, 2),
                    "equity":        round(self.state.equity, 2),
                    "pnl_today":     round(self.state.pnl_today, 2),
                    "pnl_total":     round(self.state.pnl_total, 2),
                    "positions":     self.state.open_positions,
                    "recent_trades": self.state.recent_trades,
                    "signals":       self.state.signals,
                    "error":         self.state.error,
                    "last_updated":  self.state.last_updated,
                },
            })
        except Exception as e:
            logger.warning("Broadcast failed: %s", e)

    def get_stats(self) -> dict:
        return self._sig.summary()

    async def get_okx_stats(self) -> dict:
        """Real /stats sourced from OKX order history (post-fee PnL), grouped
        into round-trip trades since STATS_SINCE_DATE (default 2026-07-16).
        Falls back to the internal summary (+ real balance) in paper mode or if
        the exchange history isn't available."""
        from .okx_stats import build_stats, since_ts_for
        since = since_ts_for(os.getenv("STATS_SINCE_DATE", "2026-07-16"))
        symbols = list({s.symbol for s in self.strategies})

        # real balance (total equity)
        balance = None
        try:
            bals = await self.connector.fetch_balance()
            balance = next((b.total for b in bals if b.asset in ("USDT", "USD", "BUSD")), None)
        except Exception as e:
            logger.debug("[Stats] balance fetch failed: %s", e)
        # Live open positions from OKX (symbol/side/size/entry/mark/uPnL) so
        # /stats can LIST them, not just count. Falls back to the in-memory
        # count if the exchange call isn't available.
        open_positions_detail: list = []
        if not self.connector.paper and hasattr(self.connector, "fetch_positions"):
            try:
                open_positions_detail = await self.connector.fetch_positions(symbols)
            except Exception as e:
                logger.debug("[Stats] fetch_positions failed: %s", e)
        open_pos = len(open_positions_detail) if open_positions_detail else len(self.risk.get_positions())

        orders_by_symbol: dict = {}
        if not self.connector.paper and hasattr(self.connector, "fetch_closed_orders_raw"):
            for sym in symbols:
                try:
                    orders_by_symbol[sym] = await self.connector.fetch_closed_orders_raw(
                        sym, since=since, limit=100)
                except Exception as e:
                    logger.warning("[Stats] history fetch failed for %s: %s", sym, e)

        if orders_by_symbol and any(orders_by_symbol.values()):
            return build_stats(orders_by_symbol, balance, open_pos, since,
                               open_positions_detail=open_positions_detail)

        # Fallback — internal record + real balance, tagged so the renderer knows.
        s = self._sig.summary()
        s["source"] = "internal"
        s["balance"] = balance
        s["open_positions"] = open_pos
        return s

    def get_learning_insights(self, days: int = 30) -> dict:
        """Deep-dive analytics: win-rate by strategy/symbol/confidence/hour, trend, recommendations."""
        return self._sig.deep_analysis(days=days)

    def get_portfolio_state(self, portfolio_value: float = None) -> dict:
        """Current portfolio-level risk summary from the portfolio engine."""
        val = portfolio_value or self.state.total_balance or 1.0
        return self._portfolio.portfolio_summary(val)

    def get_ai_strategy_states(self) -> dict:
        """Dashboard state for all AI Expert strategies."""
        result = {}
        for name, strat in self._strategy_map.items():
            if hasattr(strat, "get_analysis_state"):
                try:
                    result[name] = strat.get_analysis_state()
                except Exception:
                    pass
        return result

    def get_state(self) -> dict:
        return {
            "running":       self.state.running,
            "paper":         self.state.paper,
            "balance":       round(self.state.total_balance, 2),
            "equity":        round(self.state.equity, 2),
            "pnl_today":     round(self.state.pnl_today, 2),
            "pnl_total":     round(self.state.pnl_total, 2),
            "positions":     self.state.open_positions,
            "recent_trades": self.state.recent_trades,
            "signals":       self.state.signals,
            "error":         self.state.error,
            "last_updated":  self.state.last_updated,
        }
