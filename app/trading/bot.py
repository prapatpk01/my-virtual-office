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

        # Portfolio-level risk engine (shared across all strategies)
        self._portfolio = PortfolioEngine(
            max_portfolio_heat=float(os.getenv("MAX_PORTFOLIO_HEAT", "0.06")),
            max_same_group=int(os.getenv("MAX_SAME_GROUP", "2")),
            max_total_positions=self.risk.max_open_positions,
        )

        # Tick counter — used for periodic drift alerts
        self._tick_count = 0
        self._last_drift_alert_tick = 0

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

        # ── Position management: SL/TP, AI Layer 7, and learning callbacks ──
        for pos_info in list(self.risk.get_positions()):
            sym           = pos_info["symbol"]
            strategy_name = pos_info.get("strategy", "")
            ticker        = await self.connector.fetch_ticker(sym)
            price         = ticker["last"]

            strategy_inst = self._strategy_map.get(strategy_name)

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
                await self.connector.create_order(sym, side, pos_info["amount"])
                pnl = ((price - pos_info["entry"]) * pos_info["amount"]
                       if pos_info["side"] == "long"
                       else (pos_info["entry"] - price) * pos_info["amount"])
                trade = TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym, side=side,
                    price=price, amount=pos_info["amount"],
                    pnl=round(pnl, 4),
                    strategy=strategy_name or "risk_manager", reason=trigger,
                    paper=self.connector.paper,
                )
                self._record_trade(trade)
                self._sig.record_outcome(
                    symbol=sym, side=pos_info["side"],
                    entry=pos_info["entry"], exit_price=price,
                    sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"),
                    reason=trigger, strategy=strategy_name,
                )
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                self._on_position_closed(sym, strategy_name, price, trigger, strategy_inst)
                logger.info("Position closed by %s: %s [%s]", trigger, sym, strategy_name)
                if self.telegram:
                    self.telegram.notify_trade_closed(
                        sym, trigger, price,
                        pos_info["entry"],
                        pos_info.get("stop_loss"),
                        pos_info.get("take_profit"),
                        self._sig.summary(),
                    )

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
                    for v_reason, v_price in resolved:
                        if self.telegram:
                            self.telegram.notify_virtual_closed(
                                strategy.symbol, v_reason, v_price, self._sig.summary()
                            )

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

        self.state.signals         = (new_signals + self.state.signals)[:20]
        self.state.open_positions  = self.risk.get_positions()
        self.state.last_updated    = int(time.time() * 1000)
        self._broadcast_state()

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

        if update.action == "partial_tp":
            close_amt = round(pos_info["amount"] * update.close_pct, 8)
            if close_amt > 0:
                try:
                    close_side = "sell" if pos_info["side"] == "long" else "buy"
                    await self.connector.create_order(sym, close_side, close_amt)
                    pnl = ((price - pos_info["entry"]) * close_amt
                           if pos_info["side"] == "long"
                           else (pos_info["entry"] - price) * close_amt)
                    self.risk.reduce_position(sym, close_amt, strategy=strategy_name)
                    self._record_trade(TradeRecord(
                        timestamp=int(time.time() * 1000),
                        symbol=sym, side=close_side,
                        price=price, amount=close_amt,
                        pnl=round(pnl, 4),
                        strategy=strategy_name, reason=update.reason,
                        paper=self.connector.paper,
                    ))
                    logger.info(
                        "[%s] Partial TP %.0f%% %s @ %.4f PnL=%.4f | %s",
                        strategy_name, update.close_pct * 100, sym, price, pnl, update.reason,
                    )
                    if self.telegram:
                        try:
                            self.telegram.send(
                                f"💰 *Partial TP* `{sym}` [{strategy_name}]\n"
                                f"Closed *{update.close_pct*100:.0f}%* @ `{price:.4f}`\n"
                                f"PnL: `{pnl:+.4f}` USDT\n_{update.reason}_"
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error("Partial TP failed [%s %s]: %s", strategy_name, sym, e)
            return False

        if update.action == "close":
            close_side = "sell" if pos_info["side"] == "long" else "buy"
            try:
                await self.connector.create_order(sym, close_side, pos_info["amount"])
                pnl = ((price - pos_info["entry"]) * pos_info["amount"]
                       if pos_info["side"] == "long"
                       else (pos_info["entry"] - price) * pos_info["amount"])
                self._record_trade(TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym, side=close_side,
                    price=price, amount=pos_info["amount"],
                    pnl=round(pnl, 4),
                    strategy=strategy_name, reason=update.reason,
                    paper=self.connector.paper,
                ))
                self._sig.record_outcome(
                    symbol=sym, side=pos_info["side"],
                    entry=pos_info["entry"], exit_price=price,
                    sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"),
                    reason=update.reason, strategy=strategy_name,
                )
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                self._on_position_closed(sym, strategy_name, price, update.reason, strategy_inst)
                logger.info(
                    "[%s] AI-driven CLOSE %s @ %.4f PnL=%.4f | %s",
                    strategy_name, sym, price, pnl, update.reason,
                )
                if self.telegram:
                    self.telegram.notify_trade_closed(
                        sym, update.reason, price,
                        pos_info["entry"],
                        pos_info.get("stop_loss"),
                        pos_info.get("take_profit"),
                        self._sig.summary(),
                    )
            except Exception as e:
                logger.error("AI-driven close failed [%s %s]: %s", strategy_name, sym, e)
            return True

        return False

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
            strategy_inst = self._strategy_map.get(strategy_name)

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
                            self.telegram.send(
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
        """BUY-only execution engine.

        BUY  → open long (if no position open for this strategy).
        SELL → close existing long from THIS strategy only; never opens a short.
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
                        await self.connector.create_order(sym, "sell", pos["amount"])
                        self._sig.record_outcome(
                            symbol=sym, side="long",
                            entry=pos["entry"], exit_price=exit_price,
                            sl=pos.get("stop_loss"), tp=pos.get("take_profit"),
                            reason="sell_signal", strategy=slot_name,
                        )
                        self._sig.unlock_strategy(sym, slot_name)
                        self.risk.close_position(sym, strategy=slot_name)
                        self._on_position_closed(sym, slot_name, exit_price, "sell_signal")
                        if self.telegram:
                            self.telegram.notify_trade_closed(
                                sym, "sell_signal", exit_price,
                                pos["entry"], pos.get("stop_loss"),
                                pos.get("take_profit"), self._sig.summary(),
                            )
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
            if self.telegram:
                self.telegram.notify_signal(sig_dict)
            await self._execute_signal(signal, slot_key)
            return

        # ── Normal / AI Expert strategies: BUY-only ─────────────────────────
        if signal.type == SignalType.SELL:
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
                        await self.connector.create_order(sym, "sell", existing["amount"])
                        self._sig.record_outcome(
                            symbol=sym, side="long",
                            entry=existing["entry"], exit_price=exit_price,
                            sl=existing.get("stop_loss"), tp=existing.get("take_profit"),
                            reason="sell_signal", strategy=strategy_name,
                        )
                        self._sig.unlock_strategy(sym, strategy_name)
                        self.risk.close_position(sym, strategy=strategy_name)
                        self._on_position_closed(
                            sym, strategy_name, exit_price, "sell_signal",
                            self._strategy_map.get(strategy_name),
                        )
                        if self.telegram:
                            self.telegram.notify_trade_closed(
                                sym, "sell_signal", exit_price,
                                existing["entry"], existing.get("stop_loss"),
                                existing.get("take_profit"), self._sig.summary(),
                            )
                    except Exception as e:
                        logger.error("Exit on SELL signal failed [%s]: %s", strategy_name, e)
            return

        # BUY signal
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

        balances      = await self.connector.fetch_balance()
        quote_balance = next((b.free for b in balances if b.asset in ("USDT", "USD", "BUSD")), 0)
        ticker        = await self.connector.fetch_ticker(sym)
        price         = ticker["last"]
        meta          = signal.metadata or {}
        sl_p          = meta.get("stop_loss")
        tp_p          = meta.get("take_profit")

        amount = self.risk.size_position(quote_balance, price)
        if amount <= 0:
            logger.info("Position size 0 for %s — tracking virtually", sym)
            if sl_p and tp_p:
                vkey = f"virtual||{sym}||{strategy_name}||{int(time.time() * 1000)}"
                self._sig.add_pending(vkey, sym, signal.type.value, signal.price,
                                      sl_p, tp_p, strategy=strategy_name)
            self._sig.unlock_strategy(sym, strategy_name)
            return

        # ── Portfolio engine pre-trade check ────────────────────────────────
        effective_sl = sl_p or (price * (1 - self.risk.stop_loss_pct))
        port_check   = self._portfolio.can_add_position(
            symbol=sym, direction="long",
            entry_price=price, stop_loss=effective_sl,
            amount=amount, portfolio_value=quote_balance,
        )
        if not port_check.can_add:
            logger.info(
                "[%s] Portfolio gate blocked %s: %s (heat=%.1f%%)",
                strategy_name, sym, port_check.reason,
                port_check.portfolio_heat * 100,
            )
            self._sig.unlock_strategy(sym, strategy_name)
            return

        try:
            order = await self.connector.create_order(sym, "buy", amount)
            self.risk.open_position(sym, "long", price, amount, strategy=strategy_name,
                                    stop_loss=sl_p, take_profit=tp_p)

            # Track open time for learning engine duration calculation
            pos_key = f"{sym}||{strategy_name}"
            self._position_open_times[pos_key] = time.time()

            # Register position in portfolio engine
            self._portfolio.add_position(
                symbol=sym, direction="long",
                entry_price=price, current_price=price,
                amount=amount, stop_loss=effective_sl,
            )

            trade = TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=sym, side="buy",
                price=order.price, amount=amount,
                pnl=0.0, strategy=strategy_name, reason=signal.reason,
                paper=self.connector.paper,
            )
            self._record_trade(trade)
            logger.info(
                "[%s] BUY %s @ %.4f  SL=%.4f  TP=%.4f (paper=%s)",
                strategy_name, sym, price, sl_p or 0, tp_p or 0, self.connector.paper,
            )
            if self.telegram:
                self.telegram.notify_order(sym, "buy", amount, order.price,
                                           strategy_name, self.connector.paper)
        except Exception as e:
            logger.error("Order failed for %s: %s", sym, e)
            self._sig.unlock_strategy(sym, strategy_name)

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
