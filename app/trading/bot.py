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
        trade_amount_usdt: float = 0.0,
    ):
        self.connector = connector
        self.strategies = strategies
        self.risk = risk_manager or RiskManager()
        self.interval = interval_seconds
        self._broadcast = broadcast_fn or (lambda x: None)
        self.telegram = telegram
        # Fixed USDT per trade (overrides risk_per_trade when > 0)
        self.trade_amount_usdt = float(trade_amount_usdt)
        if self.trade_amount_usdt > 0:
            logger.info("Fixed trade size: $%.2f USDT per trade", self.trade_amount_usdt)
        self.state = BotState(paper=connector.paper)
        self._task: Optional[asyncio.Task] = None
        self._start_balance = 0.0
        self._trade_history: list[TradeRecord] = []
        # Persistent signal state — survives restarts
        kwargs = {"path": state_file} if state_file else {}
        self._sig = SignalState(**kwargs)
        # WTV=true → gate all signals through WaveTrend WT1 before notifying
        self.wt_verify: bool = os.getenv("WTV", "false").lower() == "true"
        if self.wt_verify:
            logger.info("[WTV] WaveTrend verify ENABLED (WT1 gate ±10 active)")
        # Bar-level blocking: SELL signal blocks new BUYs for 10 bars per sym+strategy
        self._blocked_until: dict = {}  # key="sym||strategy" → bar_index to unblock
        self._bar_counter: int = 0  # incremented each tick

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

    async def close_all_positions(self, reason: str = "shutdown") -> int:
        """Sell every open position immediately. Returns number of positions closed."""
        positions = self.risk.get_positions()
        if not positions:
            logger.info("No open positions to close on shutdown")
            return 0

        n = len(positions)
        logger.info("Graceful shutdown: closing %d open position(s)...", n)
        if self.telegram:
            self.telegram.notify(
                f"🔄 *Bot restarting* — closing {n} open position(s) before shutdown..."
            )

        closed = 0
        for pos in list(positions):
            sym            = pos["symbol"]
            strategy_name  = pos.get("strategy", "")
            amount         = pos.get("amount", 0.0)
            entry          = pos.get("entry", 0.0)
            if amount <= 0:
                continue
            try:
                ticker = await self.connector.fetch_ticker(sym)
                price  = ticker["last"]
                side   = "sell" if pos["side"] == "long" else "buy"
                await self.connector.create_order(sym, side, amount)
                pnl = round((price - entry) * amount, 4) if side == "sell" else 0.0
                self._record_trade(TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym, side=side,
                    price=price, amount=amount,
                    pnl=pnl, strategy=strategy_name,
                    reason=f"shutdown:{reason}",
                    paper=self.connector.paper,
                ))
                self.risk.close_position(sym, strategy=strategy_name)
                self._sig.unlock_strategy(sym, strategy_name)
                self._sig.remove_position(f"{sym}||{strategy_name}")
                closed += 1
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"−${abs(pnl):.2f}"
                logger.info("Closed %s [%s] @ %.4f  entry=%.4f  pnl=%s",
                            sym, strategy_name, price, entry, pnl_str)
                if self.telegram:
                    self.telegram.notify(
                        f"✅ *Position closed* (shutdown)\n"
                        f"`{sym}` [{strategy_name}]\n"
                        f"SELL @ ${price:,.2f}  Entry ${entry:,.2f}  PnL {pnl_str}"
                    )
            except Exception as e:
                logger.error("Failed to close %s [%s] on shutdown: %s", sym, strategy_name, e)
                if self.telegram:
                    self.telegram.notify(
                        f"⚠️ *Could not close position*\n"
                        f"`{sym}` [{strategy_name}]: `{str(e)[:120]}`\n"
                        f"_Please close manually on OKX_"
                    )

        logger.info("Graceful shutdown: %d/%d positions closed", closed, n)
        return closed

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

    async def _restore_positions(self):
        """Re-register persisted open positions into RiskManager after a restart."""
        saved = self._sig.get_saved_positions()
        if not saved:
            return
        logger.info("Restoring %d open position(s) from saved state...", len(saved))
        restored = []
        for p in saved:
            sym      = p["symbol"]
            strategy = p["strategy"]
            try:
                self.risk.open_position(
                    sym, p["side"], p["entry"], p["amount"],
                    strategy=strategy, stop_loss=p.get("sl"), take_profit=p.get("tp"),
                )
                restored.append(p)
                logger.info("Restored: %s [%s] entry=%.4f  SL=%.4f  TP=%.4f",
                            sym, strategy, p["entry"], p.get("sl") or 0, p.get("tp") or 0)
            except Exception as e:
                logger.warning("Could not restore %s [%s]: %s", sym, strategy, e)
        if restored and self.telegram:
            lines = "\n".join(
                f"`{p['symbol']}` [{p['strategy']}] entry=${p['entry']:,.2f} ×{p['amount']:.6f}"
                for p in restored
            )
            self.telegram.notify(
                f"🔄 *Bot restarted* — {len(restored)} position(s) restored:\n{lines}\n"
                f"_Monitoring TP/SL continues_"
            )

    async def _run_loop(self):
        await self._restore_positions()
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
        self._bar_counter += 1
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
                side = "sell" if pos_info["side"] == "long" else "buy"
                await self.connector.create_order(sym, side, pos_info["amount"])
                pnl = (price - pos_info["entry"]) * pos_info["amount"] if side == "sell" else 0
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
                self._sig.remove_position(f"{sym}||{strategy_name}")
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
                # Fetch MTF candles for higher-TF bias (skip if already on 1h/4h)
                mtf_candles = {}
                _base_tf = os.getenv("CANDLE_TF", "15m")
                _mtf_tfs = [t for t in ("1h", "4h") if t != _base_tf]
                for tf in _mtf_tfs:
                    try:
                        mtf_candles[tf] = await self.connector.fetch_ohlcv(
                            strategy.symbol, timeframe=tf, limit=100
                        )
                    except Exception:
                        pass

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
                # Do NOT close WT positions on SELL signal — let TP/SL handle exits.
                # Block new BUY slots for 10 bars.
                for s in range(_WT_MAX):
                    block_key = f"{sym}||{_WT}#{s}"
                    self._blocked_until[block_key] = self._bar_counter + 10
                logger.debug("[WT] SELL on %s — blocking all WT slots for 10 bars", sym)
                return  # Never open short

            # BUY: stack longs up to _WT_MAX
            if len(wt_longs) >= _WT_MAX:
                logger.debug("[WT] %s max stack (%d) reached, suppressing", sym, _WT_MAX)
                return
            slot_key = None
            for s in range(_WT_MAX):
                candidate = f"{_WT}#{s}"
                block_key = f"{sym}||{candidate}"
                if (not self._sig.is_locked_for_strategy(sym, candidate)
                        and self._bar_counter >= self._blocked_until.get(block_key, 0)):
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

        # ── Normal strategies: BUY-only, pure TP/SL exits ───────────────
        if signal.type == SignalType.SELL:
            # Do NOT close position on SELL signal — let TP/SL handle exits.
            # Instead, block new BUY signals for this sym+strategy for 10 bars.
            block_key = f"{sym}||{strategy_name}"
            self._blocked_until[block_key] = self._bar_counter + 10
            logger.debug("[%s] SELL signal on %s — blocking new BUYs for 10 bars (bar %d→%d)",
                         strategy_name, sym, self._bar_counter, self._bar_counter + 10)
            return  # Never open short, never close on SELL

        # BUY signal — open long if not already holding one for this strategy
        # and not blocked from a recent SELL signal
        block_key = f"{sym}||{strategy_name}"
        if self._bar_counter < self._blocked_until.get(block_key, 0):
            logger.debug("[%s] BUY on %s blocked until bar %d (current %d)",
                         strategy_name, sym, self._blocked_until[block_key], self._bar_counter)
            return

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

        balances = await self.connector.fetch_balance()
        quote_balance = next((b.free for b in balances if b.asset in ("USDT", "USD", "BUSD")), 0)
        ticker = await self.connector.fetch_ticker(sym)
        price = ticker["last"]
        meta  = signal.metadata or {}
        sl_p  = meta.get("stop_loss")
        tp_p  = meta.get("take_profit")

        leverage = getattr(self.connector, "leverage", 1)
        if self.trade_amount_usdt > 0:
            # For OKX margin: order the full leveraged BTC amount;
            # OKX deducts (amount × price / leverage) as collateral automatically.
            amount = round((self.trade_amount_usdt * leverage) / price, 6)
        else:
            amount = self.risk.size_position(quote_balance, price) * leverage
        logger.info("[%s] Order attempt: %s BUY %.6f BTC @ %.2f  notional=$%.1f (usdt=%.1f × %dx)",
                    strategy_name, sym, amount, price, amount * price, self.trade_amount_usdt, leverage)
        if amount <= 0:
            logger.info("Position size 0 for %s — tracking virtually", sym)
            if sl_p and tp_p:
                vkey = f"virtual||{sym}||{strategy_name}||{int(time.time() * 1000)}"
                self._sig.add_pending(vkey, sym, signal.type.value, signal.price,
                                      sl_p, tp_p, strategy=strategy_name)
            self._sig.unlock_strategy(sym, strategy_name)
            return

        # Ensure leverage is set on exchange before first order
        if hasattr(self.connector, "set_leverage_for"):
            await self.connector.set_leverage_for(sym)

        try:
            order = await self.connector.create_order(sym, "buy", amount, tp=tp_p, sl=sl_p)
            # Use ATR-based SL/TP from signal; fall back to % if not provided
            self.risk.open_position(sym, "long", price, amount, strategy=strategy_name,
                                    stop_loss=sl_p, take_profit=tp_p)
            # Persist position so it survives a bot restart
            self._sig.save_position(
                f"{sym}||{strategy_name}", sym, "long", price, amount,
                sl_p, tp_p, strategy_name,
            )
            trade = TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=sym, side="buy",
                price=order.price, amount=amount,
                pnl=0.0, strategy=strategy_name, reason=signal.reason,
                paper=self.connector.paper,
            )
            self._record_trade(trade)
            logger.info("[%s] BUY %s @ %.4f  SL=%.4f  TP=%.4f (paper=%s)",
                        strategy_name, sym, price, sl_p or 0, tp_p or 0, self.connector.paper)
            if self.telegram:
                self.telegram.notify_order(sym, "buy", amount, order.price,
                                           strategy_name, self.connector.paper)
        except Exception as e:
            logger.error("Order failed for %s: %s", sym, e)
            if self.telegram:
                err_detail = f"{e}\n\nAttempted: {amount:.6f} BTC @ ${price:,.0f} = ${amount*price:,.1f} USDT"
                self.telegram.notify_order_error(sym, strategy_name, err_detail)
            self._sig.unlock_strategy(sym, strategy_name)

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
