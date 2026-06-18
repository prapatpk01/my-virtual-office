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
        # 4H bias cache: sym → +1 bull | 0 neutral | -1 bear (updated each tick)
        self._4h_bias: dict = {}
        self._mtf_gate: bool = os.getenv("MTF_GATE", "true").lower() in ("1","true","yes")
        # 15M entry refinement: queue 1H BUY signals, enter on 15M EMA20 confirmation
        self._pending_entries: dict = {}   # "sym||strategy" → {symbol, strategy_name, signal, expires_at}
        self._last_1h_tick: float = 0.0
        self._entry_tf: int = int(os.getenv("ENTRY_TF_SECONDS", "0"))
        if self._entry_tf > 0:
            logger.info("15M entry refinement ENABLED (entry check every %ds, expire 4h)", self._entry_tf)

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

    async def export_candles(self, symbol: str, timeframe: str, limit: int) -> list:
        """Fetch raw OHLCV from the exchange — used by Telegram /export_candles."""
        exchange = getattr(self.connector, "_exchange", None)
        if exchange is None:
            raise ValueError("Connector has no underlying exchange object")
        return await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def manual_buy(self, symbol: str) -> str:
        """Force a BUY position — bypasses strategy/MTF gate, uses configured ATR SL/TP.
        Used by /buy Telegram command for manual testing."""
        from .strategies.base import BaseStrategy, Signal, SignalType
        try:
            candles = await self.connector.fetch_ohlcv(symbol, timeframe="1h", limit=50)
            ticker  = await self.connector.fetch_ticker(symbol)
            price   = ticker["last"]
            if not candles:
                return "Error: could not fetch candles"
            atr_arr = BaseStrategy.atr(candles, 14)
            atr     = float(atr_arr[-1]) if not math.isnan(float(atr_arr[-1])) else price * 0.015
            sl_mult = float(os.getenv("UT_SL", "2.5"))
            rr      = float(os.getenv("UT_RR", "1.2"))
            sl_p    = round(price - sl_mult * atr, 4)
            tp_p    = round(price + sl_mult * rr * atr, 4)
            signal  = Signal(
                type=SignalType.BUY, symbol=symbol,
                price=price, amount=0.0, confidence=1.0,
                reason="manual /buy command",
                metadata={"stop_loss": sl_p, "take_profit": tp_p, "atr": round(atr, 4)},
            )
            strategy_name = "manual"
            can, reason_str = self.risk.can_open(symbol, strategy=strategy_name)
            if not can:
                return f"Cannot open: {reason_str}"
            self._sig.lock_strategy(symbol, strategy_name, "buy")
            await self._execute_signal(signal, strategy_name)
            return (f"BUY `{symbol}` @ ${price:,.2f}\n"
                    f"SL: ${sl_p:,.2f}  TP: ${tp_p:,.2f}  ATR: ${atr:,.2f}")
        except Exception as e:
            return f"Error: {e}"

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
        self._last_1h_tick = 0.0  # force 1H analysis on first tick

        while self.state.running:
            try:
                run_strategy = (time.time() - self._last_1h_tick) >= self.interval
                await self._tick(run_strategy=run_strategy)
                if run_strategy:
                    self._last_1h_tick = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bot tick error: %s", e, exc_info=True)
                self.state.error = str(e)
            _sleep = self._entry_tf if self._entry_tf > 0 else self.interval
            await asyncio.sleep(_sleep)

    # ------------------------------------------------------------------
    # MTF 4H bias gate
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_simple(values: list, period: int) -> list:
        result = [float("nan")] * len(values)
        if len(values) < period:
            return result
        k = 2.0 / (period + 1)
        result[period - 1] = sum(values[:period]) / period
        for i in range(period, len(values)):
            result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    async def _update_4h_bias(self):
        """Fetch 4H candles and compute EMA20+EMA200 trend bias per symbol.
        Yahoo connector has no native 4H → set neutral (allow all BUYs in paper mode).
        """
        # YahooConnector maps '4h'→'1h' so data would be wrong; skip gate in paper/Yahoo mode
        if type(self.connector).__name__ == "YahooConnector":
            for s in self.strategies:
                self._4h_bias[s.symbol] = 0
            return
        symbols = list({s.symbol for s in self.strategies})
        for sym in symbols:
            try:
                candles4h = await self.connector.fetch_ohlcv(sym, timeframe="4h", limit=220)
                if not candles4h or len(candles4h) < 25:  # need at least EMA20 worth of bars
                    self._4h_bias[sym] = 0
                    continue
                closes = [float(c.close) for c in candles4h]
                em20  = self._ema_simple(closes, 20)
                em200 = self._ema_simple(closes, 200)
                last  = closes[-1]
                e20   = em20[-1]
                e200  = em200[-1]
                if math.isnan(e20):
                    # EMA20 not yet valid — not enough bars; treat as neutral
                    self._4h_bias[sym] = 0
                    continue
                macro_bull = True if math.isnan(e200) else last > e200
                bull  = last > e20 and macro_bull
                bear  = last < e20 and not macro_bull
                self._4h_bias[sym] = 1 if bull else (-1 if bear else 0)
                logger.debug("[MTF4H] %s close=%.0f EMA20=%.0f EMA200=%s bias=%+d",
                             sym, last, e20,
                             f"{e200:.0f}" if not math.isnan(e200) else "—",
                             self._4h_bias[sym])
            except Exception as e:
                logger.debug("4H bias update failed for %s: %s", sym, e)
                self._4h_bias[sym] = 0

    async def _check_pending_entries(self):
        """Try to enter queued 1H signals when 15M candle closes above EMA20."""
        now = time.time()
        for key in list(self._pending_entries.keys()):
            pend  = self._pending_entries[key]
            sym   = pend["symbol"]
            strat = pend["strategy_name"]

            if now > pend["expires_at"]:
                logger.info("[15M] %s [%s] expired — no 15M confirm within 4h, cancelled", sym, strat)
                del self._pending_entries[key]
                if self.telegram:
                    self.telegram.notify(
                        f"⏰ *Signal expired* — no 15M EMA20 confirm in 4h\n"
                        f"`{sym}` [{strat}] — skipped"
                    )
                continue

            try:
                candles_15m = await self.connector.fetch_ohlcv(sym, timeframe="15m", limit=30)
                if not candles_15m or len(candles_15m) < 21:
                    logger.debug("[15M] %s: not enough 15M candles, retrying next tick", sym)
                    continue
                closes = [float(c.close) for c in candles_15m]
                ema20  = self._ema_simple(closes, 20)
                close  = closes[-1]
                e20    = ema20[-1]
                if math.isnan(e20):
                    continue
                logger.debug("[15M] %s [%s] close=%.4f EMA20=%.4f", sym, strat, close, e20)
                if close > e20:
                    logger.info("[15M] %s [%s] confirmed: close %.4f > EMA20 %.4f → entering",
                                sym, strat, close, e20)
                    del self._pending_entries[key]
                    self._sig.lock_strategy(sym, strat, "buy")
                    await self._execute_signal(pend["signal"], strat)
                    if self.telegram:
                        self.telegram.notify(
                            f"✅ *15M entry confirmed*\n"
                            f"`{sym}` [{strat}]  close={close:,.2f} > EMA20={e20:,.2f}"
                        )
            except Exception as e:
                logger.warning("[15M] Pending entry check error for %s [%s]: %s", sym, strat, e)

    async def _tick(self, run_strategy: bool = True):
        self.state.error = ""
        if run_strategy:
            self._bar_counter += 1
        await self._refresh_balance()
        if self._mtf_gate and run_strategy:
            await self._update_4h_bias()

        if not self.risk.check_drawdown(self.state.total_balance):
            logger.warning("Max drawdown hit — bot paused")
            self.state.error = "Max drawdown reached. Trading paused."
            if self.telegram:
                self.telegram.notify_drawdown_halt(
                    self.state.total_balance, self.risk._peak_balance
                )
            self._broadcast_state()
            return

        # ── OKX margin: sync positions that were closed by native OCO ──────────
        # When OKX's attached TP/SL algo fires, the exchange closes the position
        # automatically.  If the bot doesn't learn about this before calling
        # create_order("sell"), it would accidentally open a short.  We detect
        # these by fetching actual open positions from OKX and cleaning up any
        # that are no longer there.
        #
        # IMPORTANT: get_open_position_symbols() returns set() (empty) for paper
        # mode and non-OKX exchanges — that must NOT trigger a sync wipe.
        # We only sync when the connector is OKX live (leverage > 1, not paper).
        _okx_live = (
            not getattr(self.connector, "paper", True)
            and getattr(self.connector, "_exchange_id", "") == "okx"
            and getattr(self.connector, "leverage", 1) > 1
        )
        if _okx_live and hasattr(self.connector, "get_open_position_symbols"):
            try:
                live_syms = await self.connector.get_open_position_symbols()
                if live_syms is not None:  # empty set means all closed on exchange
                    for pos_info in list(self.risk.get_positions()):
                        sym = pos_info["symbol"]
                        strategy_name = pos_info.get("strategy", "")
                        if sym not in live_syms:
                            # OKX already closed this — reconcile internal state
                            ticker = await self.connector.fetch_ticker(sym)
                            price  = ticker["last"]
                            pnl = (price - pos_info["entry"]) * pos_info["amount"]
                            # Infer whether TP or SL was hit by comparing price to levels
                            _tp = pos_info.get("take_profit")
                            _sl = pos_info.get("stop_loss")
                            if _tp and abs(price - _tp) < abs(price - (_sl or price)):
                                _outcome_reason = "take_profit"
                            elif _sl and abs(price - _sl) < abs(price - (_tp or price)):
                                _outcome_reason = "stop_loss"
                            else:
                                _outcome_reason = "take_profit" if pnl >= 0 else "stop_loss"
                            self._sig.record_outcome(
                                symbol=sym, side=pos_info["side"],
                                entry=pos_info["entry"], exit_price=price,
                                sl=_sl, tp=_tp,
                                reason=_outcome_reason, strategy=strategy_name,
                            )
                            self._sig.unlock_strategy(sym, strategy_name)
                            self._sig.remove_position(f"{sym}||{strategy_name}")
                            self.risk.close_position(sym, strategy=strategy_name)
                            logger.info(
                                "[OKX-sync] %s [%s] closed by native OCO — internal state synced (pnl≈%.2f)",
                                sym, strategy_name, pnl)
                            if self.telegram:
                                self.telegram.notify_trade_closed(
                                    sym, _outcome_reason, price,
                                    pos_info["entry"],
                                    pos_info.get("stop_loss"),
                                    pos_info.get("take_profit"),
                                    self._sig.summary(),
                                )
            except Exception as e:
                logger.warning("OKX position sync failed: %s", e)

        # Check stop-loss / take-profit on open positions
        for pos_info in list(self.risk.get_positions()):
            sym = pos_info["symbol"]
            strategy_name = pos_info.get("strategy", "")
            ticker = await self.connector.fetch_ticker(sym)
            price = ticker["last"]
            trigger = self.risk.check_stops(sym, price, strategy=strategy_name)
            if trigger:
                side = "sell" if pos_info["side"] == "long" else "buy"
                pnl = (price - pos_info["entry"]) * pos_info["amount"] if side == "sell" else 0
                try:
                    await self.connector.create_order(sym, side, pos_info["amount"])
                except Exception as e:
                    # Position may have already been closed by OKX's OCO algo
                    # between the sync above and now — safe to ignore and clean up.
                    logger.warning(
                        "[%s] Close order failed for %s (likely OCO already filled): %s",
                        strategy_name, sym, e)
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

        # Check 15M pending entries (runs on every 15M tick, not just 1H)
        if self._pending_entries:
            await self._check_pending_entries()

        if not run_strategy:
            self.state.open_positions = self.risk.get_positions()
            self.state.last_updated = int(time.time() * 1000)
            self._broadcast_state()
            return

        # Run each strategy (1H only)
        new_signals = []
        _resolved_symbols: set[str] = set()  # check virtual SL/TP once per symbol per tick
        for strategy in self.strategies:
            try:
                _tf    = os.getenv("CANDLE_TF", "1h")
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
                _base_tf = os.getenv("CANDLE_TF", "1h")
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
                # In a bull market SELL signals are often just pullbacks; no block.
                logger.debug("[WT] SELL on %s — ignored (TP/SL handles exits)", sym)
                return  # Never open short

            # BUY: MTF 4H gate — strict mode: only trade when 4H is clearly bullish
            if self._mtf_gate:
                bias4h = self._4h_bias.get(sym, 0)
                if bias4h != 1:
                    logger.debug("[MTF4H] %s [WT] BUY blocked — 4H bias not bullish (%d)", sym, bias4h)
                    return
            # BUY: stack longs up to _WT_MAX
            if len(wt_longs) >= _WT_MAX:
                logger.debug("[WT] %s max stack (%d) reached, suppressing", sym, _WT_MAX)
                return
            slot_key = None
            for s in range(_WT_MAX):
                candidate = f"{_WT}#{s}"
                block_key = f"{sym}||{candidate}"
                if (not self._sig.is_locked_for_strategy(sym, candidate)
                        and f"{sym}||{candidate}" not in self._pending_entries
                        and self._bar_counter >= self._blocked_until.get(block_key, 0)):
                    slot_key = candidate
                    break
            if slot_key is None:
                return
            can, reason = self.risk.can_open(sym, strategy=slot_key)
            if not can:
                logger.debug("[WT] %s can_open blocked: %s", sym, reason)
                return
            self._sig.record_signal(sym, signal.type.value, signal.price,
                                    signal.confidence, strategy=slot_key)
            if self.telegram:
                self.telegram.notify_signal(sig_dict)
            if self._entry_tf > 0:
                pending_key = f"{sym}||{slot_key}"
                self._pending_entries[pending_key] = {
                    "symbol": sym, "strategy_name": slot_key,
                    "signal": signal, "expires_at": time.time() + 4 * 3600,
                }
                logger.info("[15M] %s [%s] queued — waiting for 15M EMA20 confirm", sym, slot_key)
                if self.telegram:
                    self.telegram.notify(
                        f"⏳ *Waiting for 15M entry*\n"
                        f"`{sym}` [{slot_key}]  1H signal @ ${signal.price:,.2f}\n"
                        f"_Will enter when 15M candle closes above EMA20_"
                    )
            else:
                self._sig.lock_strategy(sym, slot_key, signal.type.value)
                await self._execute_signal(signal, slot_key)
            return

        # ── Normal strategies: BUY-only, pure TP/SL exits ───────────────
        if signal.type == SignalType.SELL:
            # Do NOT close position on SELL signal — let TP/SL handle exits.
            # In a bull market SELL signals are often just pullbacks; no block.
            logger.debug("[%s] SELL signal on %s — ignored (TP/SL handles exits)",
                         strategy_name, sym)
            return  # Never open short, never close on SELL

        # BUY signal — MTF 4H gate strict: only trade when 4H is clearly bullish
        if self._mtf_gate and signal.type == SignalType.BUY:
            bias4h = self._4h_bias.get(sym, 0)
            if bias4h != 1:
                logger.debug("[MTF4H] %s [%s] BUY blocked — 4H bias not bullish (%d)",
                             sym, strategy_name, bias4h)
                return

        _pending_key = f"{sym}||{strategy_name}"
        if self._sig.is_locked_for_strategy(sym, strategy_name) or _pending_key in self._pending_entries:
            logger.debug("%s [%s] already long or pending — suppressing BUY", sym, strategy_name)
            return
        can, reason = self.risk.can_open(sym, strategy=strategy_name)
        if not can:
            logger.debug("Crypto %s suppressed — %s", sym, reason)
            return
        self._sig.record_signal(sym, signal.type.value, signal.price,
                                signal.confidence, strategy=strategy_name)
        if self.telegram:
            self.telegram.notify_signal(sig_dict)
        if self._entry_tf > 0:
            self._pending_entries[_pending_key] = {
                "symbol": sym, "strategy_name": strategy_name,
                "signal": signal, "expires_at": time.time() + 4 * 3600,
            }
            logger.info("[15M] %s [%s] queued — waiting for 15M EMA20 confirm", sym, strategy_name)
            if self.telegram:
                self.telegram.notify(
                    f"⏳ *Waiting for 15M entry*\n"
                    f"`{sym}` [{strategy_name}]  1H signal @ ${signal.price:,.2f}\n"
                    f"_Will enter when 15M candle closes above EMA20_"
                )
        else:
            self._sig.lock_strategy(sym, strategy_name, signal.type.value)
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
