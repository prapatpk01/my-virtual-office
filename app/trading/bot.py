"""Trading bot: fetches candles, runs strategies, executes BUY orders with OCO TP/SL."""
import asyncio
import json
import logging
import math
import os
import time
from typing import Optional

from .claude_analyzer import ClaudeAnalyzer
from .connectors.base import BaseConnector, to_heikin_ashi
from .risk_manager import RiskManager, Position
from .strategies.base import BaseStrategy, Signal, SignalType
from .telegram_notifier import TelegramNotifier

logger = logging.getLogger("trading_bot")

_POSITIONS_FILE = os.environ.get("POSITIONS_FILE", "/data/positions.json")


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

        # Claude Chief: independent decision-maker; strategies become advisors
        use_ai_chief = os.getenv("USE_AI_CHIEF", "").lower() in ("1", "true", "yes")
        self._ai_chief: Optional[ClaudeAnalyzer] = (
            ClaudeAnalyzer(chief_mode=True) if use_ai_chief else None
        )
        self._chief_min_conf = float(os.getenv("CHIEF_MIN_CONFIDENCE", "65"))
        if self._ai_chief:
            logger.info("Claude CHIEF MODE ENABLED (model=%s, min_conf=%.0f%%)",
                        self._ai_chief.model, self._chief_min_conf)

        # In-memory position tracking: "{symbol}::{strategy_name}" -> Position
        # Using compound key allows each strategy to hold its own position on the same symbol.
        self._positions: dict[str, Position] = {}
        self._position_lock = asyncio.Lock()

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_balance: float = 0.0
        self._current_balance: float = 0.0
        self._paper = connector.paper
        self._dd_halted_notified = False

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self):
        if self._task and not self._task.done():
            logger.warning("Bot already running")
            return
        # Pre-flight check: warn loudly if live mode but credentials missing
        if not self._paper:
            cfg_key = getattr(self.connector, "_exchange", None)
            api_key = getattr(self.connector._exchange if cfg_key else self.connector,
                              "apiKey", "") or ""
            if not api_key.strip():
                logger.critical(
                    "LIVE MODE enabled but EXCHANGE_API_KEY is not set — stopping."
                )
                raise RuntimeError("LIVE MODE: EXCHANGE_API_KEY missing")
            logger.warning("=" * 60)
            logger.warning("  ⚠  LIVE TRADING MODE — real money at risk")
            logger.warning("=" * 60)
        await self._load_positions()
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("TradingBot started (paper=%s, interval=%ds, symbols=%s)",
                    self._paper, self.interval,
                    list({s.symbol for s in self.strategies}))
        if self.telegram:
            loop = asyncio.get_running_loop()
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
        for pos_key, pos in self._positions.items():
            positions.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "entry": pos.entry_price,
                "amount": pos.amount,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "strategy": pos_key.split("::")[-1] if "::" in pos_key else pos_key,
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
            candles = to_heikin_ashi(candles)
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
            open_syms = {k.split("::")[0] for k in self._positions}
            if symbol in open_syms:
                return f"Already in position for {symbol}"
            if len(self._positions) >= self.max_positions:
                return f"Max positions ({self.max_positions}) reached"
            await self._execute_buy(signal, strategy_name="manual",
                                    pos_key=f"{symbol}::manual")
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
        await self._sync_okx_positions()
        await self._check_spot_exits()

        # Drawdown guard: halt new entries if equity fell past the max-drawdown limit.
        self.risk.update_peak(self._current_balance)
        if not self.risk.check_drawdown(self._current_balance):
            if not self._dd_halted_notified:
                logger.error("MAX DRAWDOWN reached — halting new entries (balance=%.2f peak=%.2f)",
                             self._current_balance, self.risk._peak_balance)
                if self.telegram:
                    self.telegram.notify_drawdown_halt(
                        self._current_balance, self.risk._peak_balance)
                self._dd_halted_notified = True
            return
        # Balance recovered — reset notification flag so next halt sends a fresh alert
        self._dd_halted_notified = False

        symbols = list({s.symbol for s in self.strategies})
        for sym in symbols:
            try:
                candles = await self.connector.fetch_ohlcv(
                    sym, timeframe=self.candle_tf, limit=self.candle_limit
                )
                candles = to_heikin_ashi(candles)
                ticker = await self.connector.fetch_ticker(sym)
                price  = float(ticker["last"])
            except Exception as e:
                logger.warning("Failed to fetch data for %s: %s", sym, e)
                continue

            # Fetch 4H and 1D candles for SpotMaster multi-TF layers
            mtf: dict = {}
            try:
                c4h = await self.connector.fetch_ohlcv(sym, timeframe="4h", limit=250)
                if c4h and len(c4h) >= 50:
                    mtf["4h"] = to_heikin_ashi(c4h)
            except Exception:
                pass  # 4H optional — strategy degrades gracefully without it
            try:
                c1d = await self.connector.fetch_ohlcv(sym, timeframe="1d", limit=250)
                if c1d and len(c1d) >= 55:
                    mtf["1d"] = to_heikin_ashi(c1d)
            except Exception:
                pass  # 1D optional — strategy skips regime layer without it

            if self._ai_chief:
                # ── Chief mode: strategy exits first, then Chief decides ───
                open_syms = {k.split("::")[0] for k in self._positions}
                if sym in open_syms:
                    await self._check_strategy_exits(sym, candles, price)
                await self._tick_chief(sym, candles, price)
            else:
                # ── Strategy mode (+ optional gate) ───────────────────────
                await self._tick_strategies(sym, candles, price, mtf)

    async def _tick_chief(self, sym: str, candles_1h: list, price: float):
        """Chief mode: collect strategy opinions, ask Claude, execute if BUY."""
        open_syms = {k.split("::")[0] for k in self._positions}
        if sym in open_syms or len(self._positions) >= self.max_positions:
            return  # exits already handled by _check_strategy_exits before this call

        # Fetch 4h and 1d for multi-TF context (HA already applied to 1h)
        candles_by_tf: dict = {"1h": candles_1h}
        for tf in ("4h", "1d"):
            try:
                c = await self.connector.fetch_ohlcv(sym, timeframe=tf, limit=120)
                if c and len(c) >= 30:
                    candles_by_tf[tf] = to_heikin_ashi(c)
            except Exception as e:
                logger.debug("Failed to fetch %s for %s: %s", tf, sym, e)

        # Gather strategy opinions (they advise, they don't gate)
        opinions = []
        for strat in [s for s in self.strategies if s.symbol == sym]:
            try:
                sig = await strat.analyze(candles_1h, price)
                opinions.append({
                    "name":   strat.name,
                    "signal": sig.type.value.upper(),
                    "reason": (sig.reason or "")[:100],
                })
            except Exception as e:
                logger.debug("Strategy opinion error %s: %s", strat.name, e)

        from .claude_analyzer import build_snapshot, fetch_news
        snapshot = build_snapshot(sym, candles_by_tf, opinions)
        news     = await fetch_news(sym)

        decision = await self._ai_chief.chief_decide(snapshot, news)

        if decision.action != "BUY" or decision.confidence < self._chief_min_conf:
            logger.debug("[Chief] %s HOLD (conf=%.0f) | %s",
                         sym, decision.confidence, decision.reasoning[:60])
            return

        # Compute ATR-based SL/TP
        atr_arr = BaseStrategy.atr(candles_1h, 14)
        atr_val = float(atr_arr[-1]) if not math.isnan(float(atr_arr[-1])) else price * 0.015
        sl_atr  = float(os.getenv("CHIEF_SL_ATR", "2.5"))
        tp_atr  = float(os.getenv("CHIEF_TP_ATR", "1.5"))
        sl_p    = round(price - sl_atr * atr_val, 4)
        tp_p    = round(price + tp_atr * atr_val, 4)

        # Telegram: send Chief analysis before the order
        if self.telegram:
            lines = [f"Chief Analysis — {sym}",
                     f"BUY (มั่นใจ {decision.confidence:.0f}%)",
                     decision.reasoning]
            if decision.confluence:
                lines += ["สนับสนุน:"] + [f"  + {c}" for c in decision.confluence[:3]]
            if decision.risk_flags:
                lines += ["ความเสี่ยง:"] + [f"  ! {r}" for r in decision.risk_flags[:2]]
            self.telegram.send("\n".join(lines))

        signal = Signal(
            type=SignalType.BUY, symbol=sym, price=price, amount=0.0,
            confidence=decision.confidence / 100,
            reason=f"[Chief] {decision.reasoning[:80]}",
            metadata={
                "stop_loss":   sl_p,
                "take_profit": tp_p,
                "atr":         round(atr_val, 4),
                "chief_confluence": decision.confluence,
                "chief_risk_flags": decision.risk_flags,
            },
        )
        await self._execute_buy(signal, strategy_name="Chief")

    async def _tick_strategies(self, sym: str, candles: list, price: float,
                               mtf_candles: dict = None):
        """Strategy mode: run each strategy; optionally confirm with gate."""
        if self.mtf_gate and not await self._is_4h_bullish(sym):
            logger.debug("[MTF] %s 4H not bullish — skipping", sym)
            return

        for strategy in [s for s in self.strategies if s.symbol == sym]:
            pos_key = f"{sym}::{strategy.name}"
            try:
                signal = await strategy.analyze(candles, price, mtf_candles=mtf_candles)
            except Exception as e:
                logger.error("Strategy %s error on %s: %s", strategy.name, sym, e)
                continue

            # SELL: close this strategy's own position
            if signal.type == SignalType.SELL:
                if pos_key in self._positions:
                    await self._execute_sell_early(pos_key, price, signal.reason, strategy.name)
                continue

            if signal.type != SignalType.BUY:
                continue

            # Block if this strategy already has an open position, or max total reached
            if pos_key in self._positions or len(self._positions) >= self.max_positions:
                continue

            meta = signal.metadata or {}
            sl_p = meta.get("stop_loss")
            tp_p = meta.get("take_profit")
            if sl_p is None or tp_p is None or sl_p <= 0 or tp_p <= 0:
                logger.warning("[%s] BUY missing/invalid SL/TP — skipping", strategy.name)
                continue

            logger.info("[%s] BUY: %s @ %.2f  SL=%.2f  TP=%.2f",
                        strategy.name, sym, price, sl_p, tp_p)

            if self._ai_gate is not None:
                indicators    = self._build_indicators_snapshot(candles, price, meta)
                recent_candles = [
                    {"open": float(c.open), "high": float(c.high),
                     "low": float(c.low), "close": float(c.close),
                     "volume": float(c.volume)}
                    for c in candles[-10:]
                ]
                confirmed, ai_reason = await self._ai_gate.confirm_buy(
                    symbol=sym, price=price, strategy_name=strategy.name,
                    signal_reason=signal.reason, indicators=indicators,
                    recent_candles=recent_candles,
                )
                if not confirmed:
                    logger.info("[Gate] REJECTED %s %s: %s", strategy.name, sym, ai_reason)
                    continue
                logger.info("[Gate] CONFIRMED %s %s: %s", strategy.name, sym, ai_reason)

            await self._execute_buy(signal, strategy_name=strategy.name, pos_key=pos_key)

    async def _execute_buy(self, signal: Signal, strategy_name: str,
                           pos_key: str = None):
        sym = signal.symbol
        if pos_key is None:
            pos_key = f"{sym}::{strategy_name}"
        meta = signal.metadata or {}
        sl_p = meta.get("stop_loss")
        tp_p = meta.get("take_profit")

        async with self._position_lock:
            # Re-check inside lock — another strategy may have entered while we awaited
            if pos_key in self._positions:
                logger.debug("[%s] Position %s already open — skipping", strategy_name, pos_key)
                return
            if len(self._positions) >= self.max_positions:
                logger.debug("[%s] Max positions reached — skipping", strategy_name)
                return

            # Refresh price
            try:
                ticker = await self.connector.fetch_ticker(sym)
                price = float(ticker["last"])
            except Exception as e:
                logger.error("fetch_ticker failed for %s: %s", sym, e)
                if self.telegram:
                    self.telegram.notify_error(sym, str(e))
                return

            # Validate SL/TP direction
            if sl_p and tp_p:
                if sl_p >= price:
                    logger.warning("[%s] SL %.2f >= price %.2f — skipping", strategy_name, sl_p, price)
                    return
                if tp_p <= price:
                    logger.warning("[%s] TP %.2f <= price %.2f — skipping", strategy_name, tp_p, price)
                    return

            if self.trade_amount_usdt > 0:
                amount = round(self.trade_amount_usdt / price, 6)
            else:
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
                self._positions[pos_key] = pos
                self._save_positions()
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
    # Strategy-driven early exit
    # ------------------------------------------------------------------

    async def _check_strategy_exits(self, sym: str, candles: list, price: float):
        """Poll strategies for a SELL signal; execute early exit if one fires.

        Called in Chief mode before _tick_chief so exit signals are honored
        independently of the Chief's BUY/HOLD decision.
        """
        for strat in [s for s in self.strategies if s.symbol == sym]:
            try:
                sig = await strat.analyze(candles, price)
            except Exception:
                continue
            if sig.type == SignalType.SELL:
                await self._execute_sell_early(sym, price, sig.reason, strat.name)
                return  # position closed — stop checking

    async def _execute_sell_early(self, pos_key: str, price: float,
                                   reason: str, strategy_name: str):
        """Market-sell to close a long position on a strategy SELL signal."""
        pos = self._positions.get(pos_key)
        if not pos:
            return
        sym = pos_key.split("::")[0]
        try:
            order = await self.connector.create_order(sym, "sell", pos.amount)
            exit_price = order.price or price
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
            self._positions.pop(pos_key, None)
            self._save_positions()
            logger.info(
                "[%s] Early exit %s @ %.2f  entry=%.2f  pnl=%.2f%%  | %s",
                strategy_name, sym, exit_price, pos.entry_price, pnl_pct, reason,
            )
            if self.telegram:
                self.telegram.notify_close(
                    sym, pos.entry_price, exit_price, pnl_pct, "strategy_exit"
                )
        except Exception as e:
            logger.error("[%s] Early exit failed for %s: %s", strategy_name, sym, e)
            if self.telegram:
                self.telegram.notify_error(sym, f"Early exit failed: {e}")

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

        for pos_key in list(self._positions.keys()):
            sym = pos_key.split("::")[0]
            if sym not in live_syms:
                pos = self._positions.pop(pos_key)
                self._save_positions()
                try:
                    ticker = await self.connector.fetch_ticker(sym)
                    exit_price = float(ticker["last"])
                except Exception:
                    exit_price = pos.entry_price

                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
                if pos.take_profit and pos.stop_loss:
                    d_tp = abs(exit_price - pos.take_profit)
                    d_sl = abs(exit_price - pos.stop_loss)
                    reason = "take_profit" if d_tp <= d_sl else "stop_loss"
                else:
                    reason = "take_profit" if exit_price >= pos.entry_price else "stop_loss"
                logger.info("[OKX-sync] %s closed by OCO: exit=%.2f entry=%.2f pnl=%.2f%%",
                            pos_key, exit_price, pos.entry_price, pnl_pct)
                if self.telegram:
                    self.telegram.notify_close(
                        sym, pos.entry_price, exit_price, pnl_pct, reason
                    )

    # ------------------------------------------------------------------
    # Spot SL/TP check (leverage=1, non-paper)
    # ------------------------------------------------------------------

    async def _check_spot_exits(self):
        """Detect exchange-side position closures (OKX algo orders / Binance OCO).

        Two detection methods run in order:
          1. Balance check: if base asset dropped below 10% of expected, the exchange
             already closed it (most reliable — catches bounced-price scenarios).
          2. Price check: fallback for when balance hasn't settled yet.
        """
        if self._paper or getattr(self.connector, "leverage", 1) > 1:
            return
        if not self._positions:
            return

        # Fetch balances and prices once
        try:
            balances = await self.connector.fetch_balance()
            bal_map  = {b.asset: b.free for b in balances}
        except Exception:
            bal_map = {}

        # Pre-fetch prices per unique symbol
        price_map: dict[str, float] = {}
        for pos_key in list(self._positions.keys()):
            sym = pos_key.split("::")[0]
            if sym not in price_map:
                try:
                    ticker = await self.connector.fetch_ticker(sym)
                    price_map[sym] = float(ticker["last"])
                except Exception:
                    pass

        # Total expected base asset per symbol (sum across all strategies)
        base_total: dict[str, float] = {}
        for pos_key, pos in self._positions.items():
            sym  = pos_key.split("::")[0]
            base = sym.split("/")[0]
            base_total[base] = base_total.get(base, 0.0) + pos.amount

        for pos_key in list(self._positions.keys()):
            pos  = self._positions.get(pos_key)
            if not pos:
                continue
            sym   = pos_key.split("::")[0]
            base  = sym.split("/")[0]
            price = price_map.get(sym)
            if price is None:
                continue

            actual_base    = bal_map.get(base, 0.0)
            total_expected = base_total.get(base, pos.amount)

            # Method 1: total base gone → exchange algo closed all positions for this asset
            if actual_base < total_expected * 0.1:
                reason = "take_profit" if price > pos.entry_price else "stop_loss"
                logger.info("[Spot-Exit] %s detected closed by exchange algo (balance=%.6f < %.6f)",
                            pos_key, actual_base, total_expected)
                await self._close_spot_position(pos_key, price, reason)
                continue

            # Method 2: this position's SL/TP hit — sell its specific amount
            if not pos.stop_loss and not pos.take_profit:
                continue
            hit_tp = bool(pos.take_profit and price >= pos.take_profit)
            hit_sl = bool(pos.stop_loss  and price <= pos.stop_loss)
            if hit_tp or hit_sl:
                reason = "take_profit" if hit_tp else "stop_loss"
                await self._close_spot_position(pos_key, price, reason)

    async def _close_spot_position(self, pos_key: str, exit_price: float, reason: str):
        """Market sell to close spot position. If OCO already sold, the sell will
        fail (no balance) — we catch that and clear the in-memory position anyway."""
        pos = self._positions.pop(pos_key, None)
        if not pos:
            return
        sym = pos_key.split("::")[0]
        self._save_positions()
        try:
            await self.connector.create_order(sym, "sell", pos.amount)
        except Exception as e:
            logger.info("[Spot-Exit] Sell skipped (OCO already closed?): %s — %s", pos_key, e)

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        logger.info("[Spot-Exit] %s %s: exit=%.4f entry=%.4f pnl=%.2f%%",
                    pos_key, reason, exit_price, pos.entry_price, pnl_pct)
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

    # ------------------------------------------------------------------
    # Position persistence (survive container restarts)
    # ------------------------------------------------------------------

    def _save_positions(self):
        """Write open positions to disk so they survive a restart."""
        data = {
            pos_key: {
                "symbol":        pos.symbol,
                "strategy_name": pos_key.split("::")[-1] if "::" in pos_key else "",
                "side":          pos.side,
                "entry_price":   pos.entry_price,
                "amount":        pos.amount,
                "stop_loss":     pos.stop_loss,
                "take_profit":   pos.take_profit,
                "opened_at":     pos.opened_at,
            }
            for pos_key, pos in self._positions.items()
        }
        try:
            os.makedirs(os.path.dirname(_POSITIONS_FILE), exist_ok=True)
            with open(_POSITIONS_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Could not save positions: %s", e)

    async def _load_positions(self):
        """Reload persisted positions on startup; reconcile with exchange balance."""
        if self._paper:
            return
        try:
            if not os.path.exists(_POSITIONS_FILE):
                return
            with open(_POSITIONS_FILE) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Could not load positions file: %s", e)
            return

        if not data:
            return

        try:
            balances = await self.connector.fetch_balance()
            bal_map  = {b.asset: b.total for b in balances}
        except Exception as e:
            logger.warning("Balance fetch failed during position reload: %s", e)
            return

        for pos_key, pd in data.items():
            sym  = pos_key.split("::")[0]
            base = sym.split("/")[0]
            on_exch = bal_map.get(base, 0.0)
            if on_exch >= pd["amount"] * 0.9:
                pos = Position(
                    symbol=pd["symbol"], side=pd["side"],
                    entry_price=pd["entry_price"], amount=pd["amount"],
                    stop_loss=pd.get("stop_loss"),
                    take_profit=pd.get("take_profit"),
                    opened_at=pd.get("opened_at", int(time.time())),
                )
                self._positions[pos_key] = pos
                logger.info("[Restart] Restored %s: %.6f @ %.2f  SL=%.2f  TP=%.2f",
                            pos_key, pos.amount, pos.entry_price,
                            pos.stop_loss or 0, pos.take_profit or 0)
            else:
                logger.info("[Restart] %s was closed while offline (balance=%.6f < %.6f)",
                            pos_key, on_exch, pd["amount"])
