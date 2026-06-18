"""
Trading Bot main engine.
Runs strategy loops, manages orders, broadcasts state via WebSocket.
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

logger = logging.getLogger("trading_bot")


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
        fixed_sl_pct: float = 0.0,
        fixed_tp_pct: float = 0.0,
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
        self.fixed_sl_pct = fixed_sl_pct
        self.fixed_tp_pct = fixed_tp_pct
        # Persistent signal state — survives restarts
        kwargs = {"path": state_file} if state_file else {}
        self._sig = SignalState(**kwargs)
        # WTV=true → gate all signals through WaveTrend WT1 before notifying
        self.wt_verify: bool = os.getenv("WTV", "false").lower() == "true"
        if self.wt_verify:
            logger.info("[WTV] WaveTrend verify ENABLED (WT1 gate ±10 active)")
        # FUTURES_MODE: auto-enable when MARKET_TYPE=swap; override with explicit env var
        _mkt = os.getenv("MARKET_TYPE", "").lower()
        _fm_default = "true" if _mkt == "swap" else "false"
        self.futures_mode: bool = os.getenv("FUTURES_MODE", _fm_default).lower() == "true"
        if self.futures_mode:
            logger.info("[FUTURES] Futures mode ENABLED — SELL opens short, BUY closes short")
        # REVERSAL_MODE=true → BUY first closes any open SHORT, SELL first closes any open LONG
        # (sequential / always-in-market; requires FUTURES_MODE=true)
        self.reversal_mode: bool = os.getenv("REVERSAL_MODE", "false").lower() == "true"
        if self.reversal_mode:
            logger.info("[REVERSAL] Reversal mode ENABLED — signals flip position")

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

        # Check stop-loss / take-profit on open positions
        for pos_info in list(self.risk.get_positions()):
            sym = pos_info["symbol"]
            strategy_name = pos_info.get("strategy", "")
            ticker = await self.connector.fetch_ticker(sym)
            price = ticker["last"]
            trigger = self.risk.check_stops(sym, price, strategy=strategy_name)
            if trigger:
                is_long = pos_info["side"] == "long"
                side = "sell" if is_long else "buy"
                ps   = ("long" if is_long else "short") if self.futures_mode else ""
                await self.connector.create_order(
                    sym, side, pos_info["amount"], pos_side=ps, reduce_only=True
                )
                pnl_mult = 1 if is_long else -1
                pnl = pnl_mult * (price - pos_info["entry"]) * pos_info["amount"]
                trade = TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym, side=side,
                    price=price, amount=pos_info["amount"],
                    pnl=pnl, strategy=strategy_name or "risk_manager", reason=trigger,
                    paper=self.connector.paper,
                )
                self._record_trade(trade)
                # Record outcome + unlock so next signal can fire
                self._sig.record_outcome(
                    symbol=sym, side=pos_info["side"],
                    entry=pos_info["entry"], exit_price=price,
                    sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"),
                    reason=trigger, strategy=strategy_name,
                )
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                logger.info("Position closed by %s: %s [%s] → signal lock released", trigger, sym, strategy_name)
                if self.telegram:
                    self.telegram.notify_trade_closed(
                        sym, trigger, price,
                        pos_info["entry"],
                        pos_info.get("stop_loss"),
                        pos_info.get("take_profit"),
                        self._sig.summary(),
                    )

        # Run each strategy
        new_signals = []
        _resolved_symbols: set[str] = set()  # check virtual SL/TP once per symbol per tick
        for strategy in self.strategies:
            try:
                _tf    = os.getenv("CANDLE_TF", "15m")
                _limit = int(os.getenv("CANDLE_LIMIT", "300"))
                candles = await self.connector.fetch_ohlcv(strategy.symbol, timeframe=_tf, limit=_limit)
                ticker = await self.connector.fetch_ticker(strategy.symbol)
                current_price = ticker["last"]

                # Check virtual SL/TP + update WT1 cache (once per symbol per tick)
                if strategy.symbol not in _resolved_symbols and candles:
                    _resolved_symbols.add(strategy.symbol)
                    last_c = candles[-1]
                    v_high = max(float(last_c.high), current_price)
                    v_low  = min(float(last_c.low),  current_price)
                    resolved = self._sig.check_and_resolve_pending(strategy.symbol, v_high, v_low)
                    for v_reason, v_price in resolved:
                        if self.telegram:
                            self.telegram.notify_virtual_closed(
                                strategy.symbol, v_reason, v_price, self._sig.summary()
                            )
                # Fetch MTF candles — use strategy's declared timeframes if available
                mtf_candles = {}
                _base_tf = os.getenv("CANDLE_TF", "15m")
                if hasattr(strategy, "MTF_TIMEFRAMES"):
                    _mtf_tfs = [t for t in strategy.MTF_TIMEFRAMES if t != _base_tf]
                else:
                    _mtf_tfs = [t for t in ("1h", "4h") if t != _base_tf]
                for tf in _mtf_tfs:
                    try:
                        mtf_candles[tf] = await self.connector.fetch_ohlcv(
                            strategy.symbol, timeframe=tf, limit=150
                        )
                    except Exception as e:
                        logger.warning("MTF fetch failed [%s %s]: %s", strategy.symbol, tf, e)

                signal = await strategy.analyze(candles, current_price, mtf_candles=mtf_candles)

                sig_dict = {
                    "strategy": strategy.name,
                    "symbol": signal.symbol,
                    "type": signal.type.value,
                    "price": signal.price,
                    "confidence": signal.confidence,
                    "reason": signal.reason,
                    "ts": int(time.time() * 1000),
                    "metadata": signal.metadata,
                }
                new_signals.append(sig_dict)

                if signal.type != SignalType.HOLD:
                    await self._maybe_notify(signal, sig_dict, strategy.name, candles)

            except Exception as e:
                logger.error("Strategy %s error: %s", strategy.name, e)
                new_signals.append({
                    "strategy": strategy.name,
                    "symbol": strategy.symbol,
                    "type": "error",
                    "reason": str(e)[:80],
                    "ts": int(time.time() * 1000),
                })

        self.state.signals = (new_signals + self.state.signals)[:20]
        self.state.open_positions = self.risk.get_positions()
        self.state.last_updated = int(time.time() * 1000)
        self._broadcast_state()

    async def _maybe_notify(self, signal: Signal, sig_dict: dict,
                             strategy_name: str, candles: list = None):
        """BUY-only execution engine.

        BUY  → open long (if no position open for this strategy).
        SELL → close existing long from THIS strategy only; never opens a short.
        Strategy isolation: a SELL from strategy A cannot close strategy B's position.
        """
        sym = signal.symbol

        # ── WTV gate (env WTV=true) ──────────────────────────────────────
        if self.wt_verify and candles:
            wt1 = WTADXStrategy.compute_wt1(candles)
            if not math.isnan(wt1):
                if signal.type == SignalType.BUY and wt1 >= 10:
                    logger.debug("[WTV] %s %s BUY blocked — WT1=%.1f ≥+10",
                                 strategy_name, sym, wt1)
                    return
                if signal.type == SignalType.SELL and wt1 <= -10:
                    logger.debug("[WTV] %s %s SELL blocked — WT1=%.1f ≤-10",
                                 strategy_name, sym, wt1)
                    return
                logger.debug("[WTV] %s %s passed — WT1=%.1f", strategy_name, sym, wt1)

        signal_only = self.risk.max_open_positions == 0

        if signal_only:
            # Forex / signal-only mode: alert on direction change or after 4h stale
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

        # ── WaveTrend: 2-slot long stacking; SELL exits all WT longs ────
        _WT = "WTADXStrategy"
        _WT_MAX = 2
        if strategy_name == _WT:
            positions = self.risk.get_positions()
            wt_pos    = [p for p in positions
                         if p["symbol"] == sym and p["strategy"].startswith(_WT)]
            wt_longs  = [p for p in wt_pos if p["side"] == "long"]

            if signal.type == SignalType.SELL:
                if not wt_longs:
                    return
                logger.info("[WT] SELL signal on %s — exiting %d long(s)", sym, len(wt_longs))
                ticker = await self.connector.fetch_ticker(sym)
                exit_price = ticker["last"]
                for pos in list(wt_longs):
                    slot_name = pos["strategy"]
                    try:
                        await self.connector.create_order(sym, "sell", pos["amount"])
                        self._sig.record_outcome(
                            symbol=sym, side="long",
                            entry=pos["entry"], exit_price=exit_price,
                            sl=pos.get("stop_loss"), tp=pos.get("take_profit"),
                            reason="sell_signal", strategy=slot_name,
                        )
                        self._sig.unlock_strategy(sym, slot_name)
                        self.risk.close_position(sym, strategy=slot_name)
                        if self.telegram:
                            self.telegram.notify_trade_closed(
                                sym, "sell_signal", exit_price,
                                pos["entry"], pos.get("stop_loss"),
                                pos.get("take_profit"), self._sig.summary(),
                            )
                    except Exception as e:
                        logger.error("WT exit on SELL failed [%s]: %s", slot_name, e)
                return  # Never open short

            # BUY: stack longs up to _WT_MAX
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
            if self.telegram:
                self.telegram.notify_signal(sig_dict)
            await self._execute_signal(signal, slot_key)
            return

        # ── Normal strategies (spot / futures hedge) ─────────────────────
        # In futures mode: BUY opens LONG, SELL opens SHORT independently.
        # Both can coexist simultaneously (OKX hedge mode).
        # Each position carries its own TP/SL — no forced opposite-close.
        short_name = strategy_name + "_short"

        if signal.type == SignalType.SELL:
            if self.futures_mode:
                # Reversal: close existing LONG before opening SHORT
                if self.reversal_mode and self._sig.is_locked_for_strategy(sym, strategy_name):
                    await self._close_leg(sym, strategy_name, "long", "sell_signal → flip to short")

                # Futures: open SHORT (don't close long — hedge mode)
                if self._sig.is_locked_for_strategy(sym, short_name):
                    logger.debug("%s [%s] already short — suppressing SELL", sym, short_name)
                    return
                can, reason = self.risk.can_open(sym, strategy=short_name)
                if not can:
                    logger.debug("[%s] SHORT blocked — %s", short_name, reason)
                    return
                self._sig.lock_strategy(sym, short_name, "sell")
                self._sig.record_signal(sym, "sell", signal.price,
                                        signal.confidence, strategy=short_name)
                if self.telegram:
                    self.telegram.notify_signal({**sig_dict, "strategy": short_name,
                                                 "side": "short"})
                await self._execute_short(signal, short_name)
            else:
                # Spot: close long belonging to THIS strategy
                if self._sig.is_locked_for_strategy(sym, strategy_name):
                    positions = self.risk.get_positions()
                    existing  = next((p for p in positions
                                      if p["symbol"] == sym and p["strategy"] == strategy_name), None)
                    if existing and existing["side"] == "long":
                        logger.info("[%s] SELL — exiting long on %s", strategy_name, sym)
                        try:
                            ticker     = await self.connector.fetch_ticker(sym)
                            exit_price = ticker["last"]
                            await self.connector.create_order(sym, "sell", existing["amount"])
                            self._sig.record_outcome(
                                symbol=sym, side="long",
                                entry=existing["entry"], exit_price=exit_price,
                                sl=existing.get("stop_loss"), tp=existing.get("take_profit"),
                                reason="sell_signal", strategy=strategy_name,
                            )
                            self._sig.unlock_strategy(sym, strategy_name)
                            self.risk.close_position(sym, strategy=strategy_name)
                            if self.telegram:
                                self.telegram.notify_trade_closed(
                                    sym, "sell_signal", exit_price,
                                    existing["entry"], existing.get("stop_loss"),
                                    existing.get("take_profit"), self._sig.summary(),
                                )
                        except Exception as e:
                            logger.error("Exit long on SELL failed [%s]: %s", strategy_name, e)
            return

        # Reversal: close existing SHORT before opening LONG
        if self.reversal_mode and self.futures_mode \
                and self._sig.is_locked_for_strategy(sym, short_name):
            await self._close_leg(sym, short_name, "short", "buy_signal → flip to long")

        # BUY signal — open LONG (hedge mode: short stays open independently)
        if self._sig.is_locked_for_strategy(sym, strategy_name):
            logger.debug("%s [%s] already long — suppressing BUY", sym, strategy_name)
            return
        can, reason = self.risk.can_open(sym, strategy=strategy_name)
        if not can:
            logger.debug("Crypto %s suppressed — %s", sym, reason)
            return
        self._sig.lock_strategy(sym, strategy_name, signal.type.value)
        self._sig.record_signal(sym, signal.type.value, signal.price,
                                signal.confidence, strategy=strategy_name)
        if self.telegram:
            self.telegram.notify_signal(sig_dict)
        await self._execute_signal(signal, strategy_name)

    async def _close_leg(self, sym: str, strategy_name: str, leg_side: str, reason: str):
        """Close one leg (long or short) for reversal mode."""
        positions = self.risk.get_positions()
        pos = next((p for p in positions if p["symbol"] == sym and p["strategy"] == strategy_name), None)
        if not pos:
            self._sig.unlock_strategy(sym, strategy_name)
            return
        close_side = "sell" if leg_side == "long" else "buy"
        try:
            ticker     = await self.connector.fetch_ticker(sym)
            exit_price = ticker["last"]
            ps = leg_side if self.futures_mode else ""
            await self.connector.create_order(sym, close_side, pos["amount"], pos_side=ps)
            self._sig.record_outcome(
                symbol=sym, side=leg_side,
                entry=pos["entry"], exit_price=exit_price,
                sl=pos.get("stop_loss"), tp=pos.get("take_profit"),
                reason=reason, strategy=strategy_name,
            )
            self._sig.unlock_strategy(sym, strategy_name)
            self.risk.close_position(sym, strategy=strategy_name)
            logger.info("[REV] Closed %s %s @ %.2f (%s)", leg_side.upper(), sym, exit_price, reason)
            if self.telegram:
                self.telegram.notify_trade_closed(
                    sym, reason, exit_price,
                    pos["entry"], pos.get("stop_loss"), pos.get("take_profit"),
                    self._sig.summary(),
                )
        except Exception as e:
            logger.error("_close_leg failed [%s %s]: %s", strategy_name, leg_side, e)

    async def _execute_signal(self, signal: Signal, strategy_name: str):
        """Open a LONG position. Only called for BUY signals."""
        if signal.type != SignalType.BUY:
            logger.warning("[%s] _execute_signal called with non-BUY — ignored", strategy_name)
            self._sig.unlock_strategy(signal.symbol, strategy_name)
            return

        sym = signal.symbol
        can, reason = self.risk.can_open(sym, strategy=strategy_name)
        if not can:
            logger.info("Skipping BUY for %s: %s", sym, reason)
            self._sig.unlock_strategy(sym, strategy_name)
            return

        try:
            balances = await self.connector.fetch_balance()
            quote_balance = next((b.free for b in balances if b.asset in ("USDT", "USD", "BUSD")), 0)
            ticker = await self.connector.fetch_ticker(sym)
            price = ticker["last"]

            # Fixed SL/TP (% from config) takes priority; fallback to signal metadata
            if self.fixed_sl_pct > 0 and self.fixed_tp_pct > 0:
                sl_p = round(price * (1 - self.fixed_sl_pct), 8)
                tp_p = round(price * (1 + self.fixed_tp_pct), 8)
            else:
                meta = signal.metadata or {}
                sl_p = meta.get("stop_loss") or meta.get("sl")
                tp_p = meta.get("take_profit") or meta.get("tp1")
                if not sl_p or not tp_p:
                    logger.error("[%s] No SL/TP for %s — refusing order without stops", strategy_name, sym)
                    self._sig.unlock_strategy(sym, strategy_name)
                    return

            amount = self.risk.size_position(quote_balance, price)
            if amount <= 0:
                logger.info("Position size 0 for %s — tracking virtually", sym)
                if sl_p and tp_p:
                    vkey = f"virtual||{sym}||{strategy_name}||{int(time.time() * 1000)}"
                    self._sig.add_pending(vkey, sym, signal.type.value, signal.price,
                                          sl_p, tp_p, strategy=strategy_name)
                self._sig.unlock_strategy(sym, strategy_name)
                return

            order = await self.connector.create_order(
                sym, "buy", amount, tp_price=tp_p, sl_price=sl_p,
                pos_side="long" if self.futures_mode else "",
            )
            self.risk.open_position(sym, "long", price, amount, strategy=strategy_name,
                                    stop_loss=sl_p, take_profit=tp_p)
            trade = TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=sym, side="buy",
                price=order.price, amount=amount,
                pnl=0.0, strategy=strategy_name, reason=signal.reason,
                paper=self.connector.paper,
            )
            self._record_trade(trade)
            logger.info("[%s] LONG %s @ %.4f  SL=%.4f  TP=%.4f (paper=%s)",
                        strategy_name, sym, price, sl_p or 0, tp_p or 0, self.connector.paper)
            if self.telegram:
                self.telegram.notify_order(sym, "buy", amount, order.price,
                                           strategy_name, self.connector.paper)
        except Exception as e:
            logger.error("Order failed for %s: %s", sym, e)
            self._sig.unlock_strategy(sym, strategy_name)

    async def _execute_short(self, signal: Signal, strategy_name: str):
        """Open a SHORT position (futures only). Called for SELL signals in futures_mode."""
        sym = signal.symbol

        try:
            balances = await self.connector.fetch_balance()
            quote_balance = next((b.free for b in balances if b.asset in ("USDT", "USD", "BUSD")), 0)
            ticker = await self.connector.fetch_ticker(sym)
            price  = ticker["last"]

            # SL above entry, TP below entry (opposite of long)
            if self.fixed_sl_pct > 0 and self.fixed_tp_pct > 0:
                sl_p = round(price * (1 + self.fixed_sl_pct), 8)
                tp_p = round(price * (1 - self.fixed_tp_pct), 8)
            else:
                meta = signal.metadata or {}
                sl_p = meta.get("stop_loss") or meta.get("sl")
                tp_p = meta.get("take_profit") or meta.get("tp1")
                # Invert if strategy returned long-side levels
                if sl_p and sl_p < price:
                    sl_p = round(price * (1 + (self.fixed_sl_pct or self.risk.stop_loss_pct)), 8)
                if tp_p and tp_p > price:
                    tp_p = round(price * (1 - (self.fixed_tp_pct or self.risk.take_profit_pct)), 8)
                if not sl_p or not tp_p:
                    logger.error("[%s] No SL/TP for %s short — refusing order without stops", strategy_name, sym)
                    self._sig.unlock_strategy(sym, strategy_name)
                    return

            amount = self.risk.size_position(quote_balance, price)
            if amount <= 0:
                logger.info("Position size 0 for %s short — skipping", sym)
                self._sig.unlock_strategy(sym, strategy_name)
                return

            order = await self.connector.create_order(
                sym, "sell", amount,
                tp_price=tp_p, sl_price=sl_p,
                pos_side="short",
            )
            self.risk.open_position(sym, "short", price, amount, strategy=strategy_name,
                                    stop_loss=sl_p, take_profit=tp_p)
            trade = TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=sym, side="sell",
                price=order.price, amount=amount,
                pnl=0.0, strategy=strategy_name, reason=signal.reason,
                paper=self.connector.paper,
            )
            self._record_trade(trade)
            logger.info("[%s] SHORT %s @ %.4f  SL=%.4f  TP=%.4f (paper=%s)",
                        strategy_name, sym, price, sl_p or 0, tp_p or 0, self.connector.paper)
            if self.telegram:
                self.telegram.notify_order(sym, "sell_short", amount, order.price,
                                           strategy_name, self.connector.paper)
        except Exception as e:
            logger.error("Short order failed for %s: %s", sym, e)
            self._sig.unlock_strategy(sym, strategy_name)

    async def _refresh_balance(self):
        try:
            balances = await self.connector.fetch_balance()
            # Only count stable/USDT balances — BTC/ETH values are not in USDT terms
            self.state.total_balance = sum(
                b.total for b in balances if b.asset in ("USDT", "USD", "BUSD")
            )
            self.state.equity = self.state.total_balance
            if self._start_balance:
                self.state.pnl_total = self.state.total_balance - self._start_balance
                self.risk.update_peak(self.state.total_balance)
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
                    "running": self.state.running,
                    "paper": self.state.paper,
                    "balance": round(self.state.total_balance, 2),
                    "equity": round(self.state.equity, 2),
                    "pnl_today": round(self.state.pnl_today, 2),
                    "pnl_total": round(self.state.pnl_total, 2),
                    "positions": self.state.open_positions,
                    "recent_trades": self.state.recent_trades,
                    "signals": self.state.signals,
                    "error": self.state.error,
                    "last_updated": self.state.last_updated,
                },
            })
        except Exception as e:
            logger.warning("Broadcast failed: %s", e)

    def get_stats(self) -> dict:
        return self._sig.summary()

    def get_state(self) -> dict:
        return {
            "running": self.state.running,
            "paper": self.state.paper,
            "balance": round(self.state.total_balance, 2),
            "equity": round(self.state.equity, 2),
            "pnl_today": round(self.state.pnl_today, 2),
            "pnl_total": round(self.state.pnl_total, 2),
            "positions": self.state.open_positions,
            "recent_trades": self.state.recent_trades,
            "signals": self.state.signals,
            "error": self.state.error,
            "last_updated": self.state.last_updated,
        }
