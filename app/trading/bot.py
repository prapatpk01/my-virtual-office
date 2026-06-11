"""
Trading Bot main engine.
Runs strategy loops, manages orders, broadcasts state via WebSocket.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .connectors.base import BaseConnector
from .strategies.base import BaseStrategy, Signal, SignalType
from .risk_manager import RiskManager
from .telegram_notifier import TelegramNotifier

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
    recent_trades: list = field(default_factory=list)  # last 50
    signals: list = field(default_factory=list)        # last 20
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

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self):
        if self._task and not self._task.done():
            logger.warning("Bot already running")
            return
        self.state.running = True
        loop = asyncio.get_event_loop()
        if self.telegram:
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
            ticker = await self.connector.fetch_ticker(sym)
            price = ticker["last"]
            trigger = self.risk.check_stops(sym, price)
            if trigger:
                side = "sell" if pos_info["side"] == "long" else "buy"
                order = await self.connector.create_order(sym, side, pos_info["amount"])
                pnl = (price - pos_info["entry"]) * pos_info["amount"] if side == "sell" else 0
                trade = TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym, side=side,
                    price=price, amount=pos_info["amount"],
                    pnl=pnl, strategy="risk_manager", reason=trigger,
                    paper=self.connector.paper,
                )
                self._record_trade(trade)
                self.risk.close_position(sym)
                logger.info("Position closed by %s: %s", trigger, sym)
                if self.telegram:
                    self.telegram.notify_stop_event(sym, trigger, price, pnl)

        # Run each strategy
        new_signals = []
        for strategy in self.strategies:
            try:
                candles = await self.connector.fetch_ohlcv(strategy.symbol, timeframe="1h", limit=250)
                ticker = await self.connector.fetch_ticker(strategy.symbol)
                current_price = ticker["last"]
                signal = await strategy.analyze(candles, current_price)

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
                    if self.telegram:
                        self.telegram.notify_signal(sig_dict)
                    await self._execute_signal(signal, strategy.name)

            except Exception as e:
                logger.error("Strategy %s error: %s", strategy.name, e)
                new_signals.append({
                    "strategy": strategy.name,
                    "symbol": strategy.symbol,
                    "type": "error",
                    "reason": str(e)[:80],
                    "ts": int(time.time() * 1000),
                })

        # Keep last 20 signals
        self.state.signals = (new_signals + self.state.signals)[:20]
        self.state.open_positions = self.risk.get_positions()
        self.state.last_updated = int(time.time() * 1000)
        self._broadcast_state()

    async def _execute_signal(self, signal: Signal, strategy_name: str):
        sym = signal.symbol
        can, reason = self.risk.can_open(sym)
        if not can:
            logger.info("Skipping %s signal for %s: %s", signal.type, sym, reason)
            return

        balances = await self.connector.fetch_balance()
        quote_balance = next((b.free for b in balances if b.asset in ("USDT", "USD", "BUSD")), 0)
        ticker = await self.connector.fetch_ticker(sym)
        price = ticker["last"]

        amount = self.risk.size_position(quote_balance, price)
        if amount <= 0:
            logger.info("Position size 0, skipping %s", sym)
            return

        side = "buy" if signal.type == SignalType.BUY else "sell"
        try:
            order = await self.connector.create_order(sym, side, amount)
            pos = self.risk.open_position(sym, "long" if side == "buy" else "short", price, amount)
            trade = TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=sym, side=side,
                price=order.price, amount=amount,
                pnl=0.0, strategy=strategy_name, reason=signal.reason,
                paper=self.connector.paper,
            )
            self._record_trade(trade)
            logger.info("[%s] %s %s @ %.4f (paper=%s)", strategy_name, side.upper(), sym, price, self.connector.paper)
            if self.telegram:
                self.telegram.notify_order(sym, side, amount, order.price,
                                           strategy_name, self.connector.paper)
        except Exception as e:
            logger.error("Order failed for %s: %s", sym, e)

    async def _refresh_balance(self):
        try:
            balances = await self.connector.fetch_balance()
            self.state.total_balance = sum(b.total for b in balances if b.asset in ("USDT", "USD", "BUSD", "BTC", "ETH"))
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

    def get_state(self) -> dict:
        self._broadcast_state.__func__  # just reference
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
