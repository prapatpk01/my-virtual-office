"""Trading bot: fetches candles, runs strategies, executes BUY orders with OCO TP/SL."""
import asyncio
import logging
import math
import os
import time
from typing import Optional

from .claude_analyzer import ClaudeAnalyzer
from .connectors.base import BaseConnector
from .risk_manager import RiskManager, Position
from .strategies.base import BaseStrategy, Signal, SignalType
from .telegram_notifier import TelegramNotifier

logger = logging.getLogger("trading_bot")


class TradingBot:
    """
    Simple trading bot for BTC/USDT and ETH/USDT on OKX spot.

    Each tick (INTERVAL seconds):
      1. Fetch 1H candles for each symbol
      2. Run enabled strategies
      3. Execute BUY if signaled and position slot is free
      4. Sync OKX positions to detect OCO-closed trades
    """

    def __init__(
        self,
        connector: BaseConnector,
        strategies: list,
        risk_manager: Optional[RiskManager] = None,
        interval_seconds: int = 3600,
        telegram: Optional[TelegramNotifier] = None,
        trade_amount_usdt: float = 100.0,
        max_positions: int = 2,
        candle_tf: str = "1h",
        candle_limit: int = 200,
        mtf_gate: bool = False,
        use_ai_gate: Optional[bool] = None,
    ):
        self.connector = connector
        self.strategies = strategies
        self.risk = risk_manager or RiskManager()
        self.interval = interval_seconds
        self.telegram = telegram
        self.trade_amount_usdt = float(trade_amount_usdt)
        self.max_positions = max_positions
        self.candle_tf = candle_tf
        self.candle_limit = candle_limit
        self.mtf_gate = mtf_gate

        # Claude AI gate: confirm BUY signals before executing
        if use_ai_gate is None:
            use_ai_gate = os.getenv("USE_AI_GATE", "").lower() in ("1", "true", "yes")
        self._ai_gate: Optional[ClaudeAnalyzer] = ClaudeAnalyzer() if use_ai_gate else None
        if self._ai_gate:
            logger.info("Claude AI gate ENABLED (model=%s)", self._ai_gate.model)

        # In-memory position tracking: symbol -> Position
        self._positions: dict[str, Position] = {}

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_balance: float = 0.0
        self._current_balance: float = 0.0
        self._paper = connector.paper

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self):
        if self._task and not self._task.done():
            logger.warning("Bot already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("TradingBot started (paper=%s, interval=%ds, symbols=%s)",
                    self._paper, self.interval,
                    list({s.symbol for s in self.strategies}))
        if self.telegram:
            loop = asyncio.get_event_loop()
            self.telegram.start_polling(loop)
            self.telegram.notify_bot_started(
                self._paper,
                list({s.name for s in self.strategies}),
                list({s.symbol for s in self.strategies}),
            )

    async def stop(self):
        self._running = False
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

    def get_state(self) -> dict:
        positions = []
        for sym, pos in self._positions.items():
            positions.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "entry": pos.entry_price,
                "amount": pos.amount,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "strategy": "",
            })
        pnl_total = self._current_balance - self._start_balance if self._start_balance else 0.0
        return {
            "running": self._running,
            "paper": self._paper,
            "balance": round(self._current_balance, 2),
            "equity": round(self._current_balance, 2),
            "pnl_today": 0.0,
            "pnl_total": round(pnl_total, 2),
            "positions": positions,
            "recent_trades": [],
            "signals": [],
            "error": "",
            "last_updated": int(time.time() * 1000),
        }

    async def manual_buy(self, symbol: str) -> str:
        """Force a BUY — bypasses strategy, uses ATR-based SL/TP."""
        try:
            candles = await self.connector.fetch_ohlcv(symbol, timeframe="1h", limit=50)
            ticker = await self.connector.fetch_ticker(symbol)
            price = float(ticker["last"])
            if not candles:
                return "Error: could not fetch candles"
            atr_arr = BaseStrategy.atr(candles, 14)
            atr_val = float(atr_arr[-1])
            if math.isnan(atr_val):
                atr_val = price * 0.015
            sl_mult = float(os.getenv("UT_SL", "2.5"))
            rr = float(os.getenv("UT_RR", "1.2"))
            sl_p = round(price - sl_mult * atr_val, 4)
            tp_p = round(price + sl_mult * rr * atr_val, 4)
            signal = Signal(
                type=SignalType.BUY, symbol=symbol,
                price=price, amount=0.0, confidence=1.0,
                reason="manual /buy command",
                metadata={"stop_loss": sl_p, "take_profit": tp_p, "atr": round(atr_val, 4)},
            )
            if symbol in self._positions:
                return f"Already in position for {symbol}"
            if len(self._positions) >= self.max_positions:
                return f"Max positions ({self.max_positions}) reached"
            await self._execute_buy(signal, strategy_name="manual")
            return (
                f"BUY {symbol} @ ${price:,.2f}\n"
                f"SL: ${sl_p:,.2f}  TP: ${tp_p:,.2f}  ATR: ${atr_val:,.2f}"
            )
        except Exception as e:
            logger.error("manual_buy error: %s", e)
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _run_loop(self):
        await self._refresh_balance()
        self._start_balance = self._current_balance

        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bot tick error: %s", e, exc_info=True)
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    async def _tick(self):
        await self._refresh_balance()
        await self._sync_okx_positions()   # leveraged OKX positions only
        await self._check_spot_exits()     # spot SL/TP price check

        symbols = list({s.symbol for s in self.strategies})
        for sym in symbols:
            try:
                candles = await self.connector.fetch_ohlcv(
                    sym, timeframe=self.candle_tf, limit=self.candle_limit
                )
                ticker = await self.connector.fetch_ticker(sym)
                price = float(ticker["last"])
            except Exception as e:
                logger.warning("Failed to fetch data for %s: %s", sym, e)
                continue

            # MTF gate: check 4H EMA20 > EMA200 for bullish bias
            if self.mtf_gate and not await self._is_4h_bullish(sym):
                logger.debug("[MTF] %s 4H not bullish — skipping", sym)
                continue

            # Skip if already in position
            if sym in self._positions:
                continue

            # Skip if max positions reached
            if len(self._positions) >= self.max_positions:
                continue

            # Run strategies for this symbol
            sym_strategies = [s for s in self.strategies if s.symbol == sym]
            for strategy in sym_strategies:
                try:
                    signal = await strategy.analyze(candles, price)
                except Exception as e:
                    logger.error("Strategy %s error on %s: %s", strategy.name, sym, e)
                    continue

                if signal.type != SignalType.BUY:
                    continue

                meta = signal.metadata or {}
                sl_p = meta.get("stop_loss")
                tp_p = meta.get("take_profit")
                if not sl_p or not tp_p:
                    logger.warning("[%s] BUY signal missing SL/TP metadata — skipping", strategy.name)
                    continue

                logger.info("[%s] BUY signal: %s @ %.2f  SL=%.2f  TP=%.2f",
                            strategy.name, sym, price, sl_p, tp_p)

                # Claude AI confirmation gate
                if self._ai_gate is not None:
                    indicators = self._build_indicators_snapshot(candles, price, meta)
                    recent_candles = [
                        {"open": float(c.open), "high": float(c.high),
                         "low": float(c.low), "close": float(c.close),
                         "volume": float(c.volume)}
                        for c in candles[-10:]
                    ]
                    confirmed, ai_reason = await self._ai_gate.confirm_buy(
                        symbol=sym,
                        price=price,
                        strategy_name=strategy.name,
                        signal_reason=signal.reason,
                        indicators=indicators,
                        recent_candles=recent_candles,
                    )
                    if not confirmed:
                        logger.info("[AI-Gate] REJECTED %s %s: %s", strategy.name, sym, ai_reason)
                        continue
                    logger.info("[AI-Gate] CONFIRMED %s %s: %s", strategy.name, sym, ai_reason)

                await self._execute_buy(signal, strategy_name=strategy.name)
                # One BUY per symbol per tick
                break

    async def _execute_buy(self, signal: Signal, strategy_name: str):
        sym = signal.symbol
        meta = signal.metadata or {}
        sl_p = meta.get("stop_loss")
        tp_p = meta.get("take_profit")

        # Refresh price
        try:
            ticker = await self.connector.fetch_ticker(sym)
            price = float(ticker["last"])
        except Exception as e:
            logger.error("fetch_ticker failed for %s: %s", sym, e)
            if self.telegram:
                self.telegram.notify_error(sym, str(e))
            return

        if self.trade_amount_usdt > 0:
            amount = round(self.trade_amount_usdt / price, 6)
        else:
            # Fall back to 1% of balance
            amount = round(self._current_balance * 0.01 / price, 6)

        if amount <= 0:
            logger.warning("Computed amount=0 for %s — skipping", sym)
            return

        try:
            order = await self.connector.create_order(
                sym, "buy", amount, tp=tp_p, sl=sl_p
            )
            fill_price = order.price or price
            pos = Position(
                symbol=sym, side="long",
                entry_price=fill_price, amount=amount,
                stop_loss=sl_p, take_profit=tp_p,
            )
            self._positions[sym] = pos
            logger.info("[%s] BUY filled: %s %.6f @ %.2f  SL=%.2f  TP=%.2f",
                        strategy_name, sym, amount, fill_price,
                        sl_p or 0, tp_p or 0)
            if self.telegram:
                self.telegram.notify_buy(sym, fill_price, amount,
                                         sl_p or 0.0, tp_p or 0.0, strategy_name)
        except Exception as e:
            logger.error("[%s] Order failed for %s: %s", strategy_name, sym, e)
            if self.telegram:
                self.telegram.notify_error(sym, str(e))

    # ------------------------------------------------------------------
    # OKX position sync
    # ------------------------------------------------------------------

    async def _sync_okx_positions(self):
        """Detect positions closed by OCO on OKX and clean up local state."""
        if not hasattr(self.connector, "get_open_position_symbols"):
            return
        try:
            live_syms = await self.connector.get_open_position_symbols()
        except Exception as e:
            logger.warning("get_open_position_symbols error: %s", e)
            return

        if live_syms is None:
            # Paper mode or non-OKX — no sync
            return

        for sym in list(self._positions.keys()):
            if sym not in live_syms:
                pos = self._positions.pop(sym)
                try:
                    ticker = await self.connector.fetch_ticker(sym)
                    exit_price = float(ticker["last"])
                except Exception:
                    exit_price = pos.entry_price

                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
                reason = "take_profit" if exit_price >= pos.entry_price else "stop_loss"
                logger.info("[OKX-sync] %s closed by OCO: exit=%.2f entry=%.2f pnl=%.2f%%",
                            sym, exit_price, pos.entry_price, pnl_pct)
                if self.telegram:
                    self.telegram.notify_close(
                        sym, pos.entry_price, exit_price, pnl_pct, reason
                    )

    # ------------------------------------------------------------------
    # Spot SL/TP check (leverage=1, non-paper)
    # ------------------------------------------------------------------

    async def _check_spot_exits(self):
        """For spot (leverage=1) live positions: check price vs SL/TP each tick.

        OKX's OCO algo order handles the actual exchange-side close, but the bot
        needs to detect the closure and update its in-memory state. We check the
        current price on each tick; if SL or TP is hit, we attempt a market SELL
        (which will fail gracefully if OCO already closed it) and clear the position.
        """
        if self._paper or getattr(self.connector, "leverage", 1) > 1:
            return
        for sym in list(self._positions.keys()):
            pos = self._positions[sym]
            if not pos.stop_loss and not pos.take_profit:
                continue
            try:
                ticker = await self.connector.fetch_ticker(sym)
                price = float(ticker["last"])
            except Exception:
                continue

            hit_tp = bool(pos.take_profit and price >= pos.take_profit)
            hit_sl = bool(pos.stop_loss  and price <= pos.stop_loss)
            if not hit_tp and not hit_sl:
                continue

            reason = "take_profit" if hit_tp else "stop_loss"
            await self._close_spot_position(sym, price, reason)

    async def _close_spot_position(self, sym: str, exit_price: float, reason: str):
        """Market sell to close spot position. If OCO already sold, the sell will
        fail (no balance) — we catch that and clear the in-memory position anyway."""
        pos = self._positions.pop(sym, None)
        if not pos:
            return
        try:
            await self.connector.create_order(sym, "sell", pos.amount)
        except Exception as e:
            logger.info("[Spot-Exit] Sell skipped (OCO already closed?): %s — %s", sym, e)

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        logger.info("[Spot-Exit] %s %s: exit=%.4f entry=%.4f pnl=%.2f%%",
                    sym, reason, exit_price, pos.entry_price, pnl_pct)
        if self.telegram:
            self.telegram.notify_close(sym, pos.entry_price, exit_price, pnl_pct, reason)

    # ------------------------------------------------------------------
    # 4H MTF gate
    # ------------------------------------------------------------------

    async def _is_4h_bullish(self, symbol: str) -> bool:
        """Returns True if 4H EMA20 > EMA200 (bullish bias)."""
        try:
            candles = await self.connector.fetch_ohlcv(symbol, timeframe="4h", limit=220)
            if not candles or len(candles) < 25:
                return True  # Not enough data — allow trading
            closes = [float(c.close) for c in candles]
            ema20 = BaseStrategy.ema(closes, 20)
            ema200 = BaseStrategy.ema(closes, 200)
            e20 = float(ema20[-1])
            e200 = float(ema200[-1])
            if math.isnan(e20):
                return True
            if math.isnan(e200):
                return closes[-1] > e20
            return e20 > e200
        except Exception as e:
            logger.debug("4H MTF check failed for %s: %s", symbol, e)
            return True  # On error, allow trading

    # ------------------------------------------------------------------
    # Claude AI helpers
    # ------------------------------------------------------------------

    def _build_indicators_snapshot(self, candles: list, price: float, meta: dict) -> dict:
        """Compute indicator values from candles for the Claude AI gate prompt."""
        import numpy as np
        closes = [float(c.close) for c in candles]

        rsi_arr = BaseStrategy.rsi(closes)
        rsi_val = float(rsi_arr[-1])

        hma_arr = BaseStrategy.hma(closes, 50)
        hma_val = float(hma_arr[-1])

        ema5_arr = BaseStrategy.ema(closes, 5)
        sma9_arr = BaseStrategy.sma(closes, 9)
        ema5_val = float(ema5_arr[-1])
        sma9_val = float(sma9_arr[-1])

        def _safe(v):
            return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

        return {
            "rsi": _safe(rsi_val),
            "hma_period": 50,
            "price_above_hma": None if math.isnan(hma_val) else price > hma_val,
            "ema5": _safe(ema5_val),
            "sma9": _safe(sma9_val),
            "ema5_above_sma9": None if (math.isnan(ema5_val) or math.isnan(sma9_val))
                               else ema5_val > sma9_val,
            "atr": meta.get("atr"),
            "sl_price": meta.get("stop_loss"),
            "tp_price": meta.get("take_profit"),
            "rr": meta.get("rr", 1.2),
        }

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    async def _refresh_balance(self):
        try:
            balances = await self.connector.fetch_balance()
            usdt = next(
                (b.total for b in balances if b.asset in ("USDT", "USD")), 0.0
            )
            self._current_balance = float(usdt)
        except Exception as e:
            logger.warning("Balance refresh failed: %s", e)
