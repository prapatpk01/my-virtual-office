"""
OKX Perpetual Futures trading bot — hedge mode.

BUY  signal → open LONG  (independent per strategy per symbol)
SELL signal → open SHORT (independent per strategy per symbol)

Each strategy gets 1 long slot + 1 short slot per symbol.
Long and short can coexist simultaneously (OKX hedge mode).
SL/TP: uses strategy metadata first, falls back to fixed percentages.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .connectors.base import BaseConnector
from .strategies.base import BaseStrategy, Signal, SignalType
from .risk_manager import RiskManager
from .telegram_notifier import TelegramNotifier
from .signal_state import SignalState
from .position_health import PositionHealthMonitor, HealthResult
from .lesson_learned import LessonTracker, TERMINAL_REASONS
from .patterns import pattern_gate_passes, pattern_best_tier, TP1_BY_TIER

logger = logging.getLogger("trading_bot")


def _fmt_details(d: dict) -> str:
    """Format health indicator dict into a compact Telegram string.
    Bool keys (no leading _) are shown with ✓/✗.
    Companion numeric keys (leading _) are matched by exact name pattern.
    """
    parts = []
    for k, v in d.items():
        if k.startswith("_"):
            continue
        icon = "✓" if v else "✗"
        # companion key convention: f"_{k.lower()}_val"
        # e.g. "MTF_bias" → "_mtf_bias_val", "ADX>20" → "_adx>20_val"
        vkey = f"_{k.lower()}_val"
        val_str = f"={d[vkey]:.1f}" if vkey in d else ""
        parts.append(f"{icon}{k}{val_str}")
    return "  ".join(parts)


def _l1_wait_hint(reason: str) -> str:
    """Parse a strategy HOLD reason string and return what's blocking L1.

    Finds the direction (LONG/SHORT) closest to passing, lists its failing
    sub-conditions, and returns a compact 'LONG needs: bias+adx' string.
    """
    best_dir: str = ""
    best_fails: list = []
    for part in reason.split(" | "):
        if not (part.startswith("L:") or part.startswith("S:")):
            continue
        direction = "LONG" if part.startswith("L:") else "SHORT"
        tokens    = part[2:].split()
        fails     = [t.replace("✗", "").replace("×", "") for t in tokens if "✗" in t or "×" in t]
        if not best_dir or len(fails) < len(best_fails):
            best_dir, best_fails = direction, fails
    if best_dir and best_fails:
        return f"{best_dir} needs: {'+'.join(best_fails)}"
    # Fallback: show the score/bias/ADX summary (first segment before ' | ')
    return reason.split(" | ")[0] if " | " in reason else reason[:80]


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
    detail: str = ""   # exit forensics (e.g. which guard fired) for the lesson tracker


class TradingBot:
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
        dynamic_sizing: bool = True,
        monitor_interval: int = 180,   # health-check every 3 min
    ):
        self.connector = connector
        self.strategies = strategies
        self.risk = risk_manager or RiskManager()
        self.interval = interval_seconds
        self._broadcast = broadcast_fn or (lambda x: None)
        self.telegram = telegram
        self.paper = connector.paper
        self.fixed_sl_pct = fixed_sl_pct
        self.fixed_tp_pct = fixed_tp_pct
        self._dynamic_sizing = dynamic_sizing
        self._monitor_interval = monitor_interval

        self._task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._closing_positions: set[str] = set()  # keys being closed — prevents double-close
        self._maxpos_cooldown: dict[str, float] = {}  # slot → unix ts when 60m cooldown expires
        # Lesson-learned tracker: per-trade entry/exit forensics, 5-trade round
        # reviews, loss-pattern alerts to Telegram, optional adaptive caution.
        self.lessons = LessonTracker(telegram=self.telegram)
        self._lesson_bump: float = 0.0            # currently applied min_score bump
        self._lesson_base: dict[int, float] = {}  # id(strategy) → base min_score
        self._balance = 0.0
        self._start_balance = 0.0
        self._pnl_total = 0.0
        self._error = ""
        self._entries_paused = False   # set by daily circuit breaker
        self._trade_history: list[TradeRecord] = []
        self._signals_cache: list[dict] = []

        kwargs = {"path": state_file} if state_file else {}
        self._sig = SignalState(**kwargs)

        # Derive futures_mode from the connector's own flag so bot and connector
        # always agree on hedge-mode posSide logic. Fall back to MARKET_TYPE env
        # only when connector doesn't expose _futures (e.g. mock/paper connector).
        _mkt = os.getenv("MARKET_TYPE", "").lower()
        self.futures_mode: bool = getattr(connector, '_futures', _mkt in ("swap", "futures"))

        # Optional health-score entry gate: require PositionHealthMonitor score >
        # TCI_HEALTH_ENTRY_MIN at entry (0 = off). Filters low-quality setups.
        # TCI_HEALTH_ENTRY_SYMBOLS (comma list) restricts the gate to those symbols;
        # empty/unset = gate all symbols. Backtest Jan-May 2026 with new health score:
        # gate>65 helps BOTH BTC and XAU → leave TCI_HEALTH_ENTRY_SYMBOLS unset.
        self._health_entry_min = float(os.getenv("TCI_HEALTH_ENTRY_MIN", "0") or 0)
        self._health_entry_syms = {
            s.strip() for s in os.getenv("TCI_HEALTH_ENTRY_SYMBOLS", "").split(",") if s.strip()
        }
        self._entry_health = PositionHealthMonitor(self.connector)
        self._health_weak_confirm = int(os.getenv("TCI_HEALTH_WEAK_CONFIRM", "3"))

        # Layer 3: pattern scanner feeds TP1 optimizer (no longer an entry gate).
        # TCI_PATTERN_GATE env var is no longer used — L3 always scans patterns
        # at entry time and stores the best tier on the position for TP1 upgrade.

        logger.info("[BOT] futures=%s paper=%s interval=%ds SL=%.1f%% TP=%.1f%%",
                    self.futures_mode, self.paper, self.interval,
                    fixed_sl_pct * 100, fixed_tp_pct * 100)

    # ── Control ──────────────────────────────────────────────────────────

    async def start(self):
        if self._task and not self._task.done():
            return
        # Verify OKX account is in hedge mode before starting — raises RuntimeError
        # if posMode != long_short_mode (every posSide order would fail with 51000).
        if hasattr(self.connector, "ensure_hedge_mode"):
            await self.connector.ensure_hedge_mode()
        self._running = True
        loop = asyncio.get_event_loop()
        if self.telegram:
            self.telegram.start_polling(loop)
            names   = [s.name for s in self.strategies]
            symbols = sorted({s.symbol for s in self.strategies})
            self.telegram.notify_bot_started(self.paper, names, symbols)
        self._task = asyncio.create_task(self._run_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[BOT] Started — strategies: %s  health-monitor: every %ds",
                    [s.name for s in self.strategies], self._monitor_interval)

    async def stop(self):
        self._running = False
        for t in (self._task, self._monitor_task):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        if self.telegram:
            self.telegram.stop_polling()
            self.telegram.notify_bot_stopped()
        logger.info("[BOT] Stopped")

    # ── Core loop ─────────────────────────────────────────────────────────

    async def _run_loop(self):
        # Clear locks older than 2h so a crash mid-trade doesn't permanently block slots.
        # OKX algo SL/TP orders remain active on exchange even when bot is offline.
        stale = self._sig.clear_stale_strategy_locks(max_age_hours=2)
        if stale:
            logger.warning("[BOT] Cleared %d stale strategy lock(s) from previous session: %s",
                           len(stale), stale)
        await self._refresh_balance()
        self._start_balance = self._balance
        self.risk.update_peak(self._balance)
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[BOT] Tick error: %s", e, exc_info=True)
                self._error = str(e)
            await asyncio.sleep(self.interval)

    async def _tick(self):
        self._error = ""
        await self._refresh_balance()

        if not self.risk.check_drawdown(self._balance):
            logger.warning("[BOT] Max drawdown reached — paused (balance=%.2f)", self._balance)
            self._error = "Max drawdown — trading paused"
            if self.telegram:
                self.telegram.notify_drawdown_halt(self._balance, self.risk._peak_balance)
            self._broadcast_state()
            return

        # Check SL/TP (staged: TP1/breakeven/TP2) on every open position every tick
        for pos in list(self.risk.get_positions()):
            await self._check_stop(pos)

        # Daily circuit breaker — block NEW entries (open positions keep running)
        allowed_daily, dreason = self.risk.check_daily_circuit(self._balance)
        self._entries_paused = not allowed_daily
        if self._entries_paused:
            logger.warning("[BOT] %s", dreason)
            self._error = dreason

        # Run strategies — Phase 1: analyze all symbols, collect signals
        candle_cache: dict[tuple, list] = {}
        new_signals: list[dict] = []
        action_queue: list[tuple] = []   # (Signal, sig_dict, strategy)

        for strategy in self.strategies:
            result = await self._analyze_only(strategy, candle_cache)
            if result is None:
                continue
            signal, sig_dict = result
            new_signals.append(sig_dict)
            if signal.type != SignalType.HOLD:
                action_queue.append((signal, sig_dict, strategy))

        # Phase 2: score-rank then execute — highest SJ score enters first
        # Prevents first-in-list bias when multiple symbols signal simultaneously
        if len(action_queue) > 1:
            action_queue.sort(
                key=lambda t: (t[0].metadata or {}).get("sj_score", 0),
                reverse=True,
            )
            scores = [(t[1]["symbol"], (t[0].metadata or {}).get("sj_score", 0))
                      for t in action_queue]
            logger.info("[BOT] Score-rank: %s", scores)

        for signal, sig_dict, strategy in action_queue:
            await self._handle_signal(signal, sig_dict, strategy)

        self._signals_cache = (new_signals + self._signals_cache)[:20]
        self._broadcast_state()

    async def _analyze_only(self, strategy: BaseStrategy,
                            cache: dict) -> Optional[tuple]:
        """Analyze one strategy and return (Signal, sig_dict) without executing.
        Used by score-ranked two-phase dispatch in _tick()."""
        sym   = strategy.symbol
        tf    = strategy.params.get("tf", "15m")
        limit = strategy.params.get("limit", 300)

        key = (sym, tf)
        if key not in cache:
            cache[key] = await self.connector.fetch_ohlcv(sym, timeframe=tf, limit=limit)
        candles = cache[key]

        ticker = await self.connector.fetch_ticker(sym)
        price  = float(ticker["last"])

        mtf_candles: dict = {}
        for mtf_tf in getattr(strategy, "MTF_TIMEFRAMES", []):
            if mtf_tf == tf:
                continue
            mk = (sym, mtf_tf)
            if mk not in cache:
                try:
                    cache[mk] = await self.connector.fetch_ohlcv(sym, timeframe=mtf_tf, limit=150)
                except Exception as e:
                    logger.warning("[BOT] MTF %s/%s fetch failed: %s", sym, mtf_tf, e)
            if mk in cache:
                mtf_candles[mtf_tf] = cache[mk]

        try:
            signal = await strategy.analyze(candles, price, mtf_candles=mtf_candles)
        except Exception as e:
            logger.error("[%s] analyze error on %s: %s", strategy.name, sym, e)
            return None

        # ── Layer 1 result ────────────────────────────────────────────────────
        is_hold_l1 = signal.type == SignalType.HOLD
        side_hint  = "long" if signal.type == SignalType.BUY else ("short" if signal.type == SignalType.SELL else "?")
        sj         = (signal.metadata or {}).get("sj_score")
        l1_label   = "✗" if is_hold_l1 else (f"✓(sj={sj:.0f})" if isinstance(sj, (int, float)) else "✓")

        # ── Layer 3 — Pattern scanner (NO LONGER blocks entry; feeds TP1 optimizer) ──
        pattern_tier  = 0
        pattern_names = ""
        if is_hold_l1:
            l3_label = "—"
        else:
            try:
                _, pat_str = pattern_gate_passes(candles, side_hint)
                fired = [n for n in pat_str.split("+") if n and n != "no_pattern"]
                pattern_names = "+".join(fired)
                pattern_tier  = pattern_best_tier(fired)
                if pattern_tier > 0:
                    l3_label = f"T{pattern_tier}({pattern_names})"
                elif fired:
                    l3_label = f"—({pattern_names})"
                else:
                    l3_label = "—(no pat)"
            except Exception as _pe:
                logger.warning("[%s] Pattern scan error on %s: %s", strategy.name, sym, _pe)
                l3_label = "—(err)"

        sig_dict = {
            "strategy":       strategy.name,
            "symbol":         sym,
            "type":           signal.type.value,
            "price":          price,
            "confidence":     signal.confidence,
            "reason":         signal.reason,
            "ts":             int(time.time() * 1000),
            "metadata":       signal.metadata,
            "_l1_label":      l1_label,
            "_l3_label":      l3_label,
            "_pattern_tier":  pattern_tier,
            "_pattern_names": pattern_names,
        }

        if signal.type == SignalType.HOLD:
            logger.info("[%s] %s HOLD — %s", strategy.name, sym, signal.reason[:140])

        return signal, sig_dict

    # ── Signal routing ────────────────────────────────────────────────────

    async def _handle_signal(self, signal: Signal, sig_dict: dict, strategy):
        sym           = signal.symbol
        strategy_name = strategy.name if hasattr(strategy, "name") else str(strategy)
        is_buy        = signal.type == SignalType.BUY

        if self._entries_paused:
            logger.debug("[%s] entries paused (daily circuit breaker) — %s", strategy_name, sym)
            return

        if is_buy:
            slot = strategy_name                    # long slot key
            side = "long"
            label_side = "long"
        else:
            slot = f"{strategy_name}_short"         # short slot key
            side = "short"
            label_side = "short"

        if self._sig.is_locked_for_strategy(sym, slot):
            logger.debug("[%s] %s already %s — suppressing", strategy_name, sym, label_side)
            return

        _mp_key = f"{sym}||{slot}"
        if self._maxpos_cooldown.get(_mp_key, 0) > time.time():
            _rem = int((self._maxpos_cooldown[_mp_key] - time.time()) / 60)
            logger.debug("[%s] %s %s maxpos-cooldown: %dmin left", strategy_name, label_side, sym, _rem)
            return

        can, reason = self.risk.can_open(sym, strategy=slot)
        if not can:
            if "Max open positions" in reason:
                self._maxpos_cooldown[_mp_key] = time.time() + 60 * 60
                logger.info("[%s] %s %s maxpos blocked — cooldown 60m", strategy_name, label_side, sym)
            else:
                logger.debug("[%s] %s %s blocked: %s", strategy_name, label_side, sym, reason)
            return

        # ── Layer 2 — Health-score entry gate (TCI_HEALTH_ENTRY_MIN > 0) ──────
        l3_label  = sig_dict.get("_l3_label", "—")
        gate_this = self._health_entry_min > 0 and (
            not self._health_entry_syms or sym in self._health_entry_syms)
        l2_label  = "—(off)"
        if gate_this:
            try:
                hr = await self._entry_health.check(sym, side, entry_price=0.0, oneR=0.0, current_tp2=0.0)
                if hr.score < self._health_entry_min:
                    l2_label = f"✗(health={hr.score:.0f}<{self._health_entry_min:.0f})"
                    logger.info("[%s] %s %s L2 blocked — health=%.0f < %.0f",
                                strategy_name, sym, side.upper(),
                                hr.score, self._health_entry_min)
                    return
                l2_label = f"✓(health={hr.score:.0f})"
            except Exception as e:
                l2_label = "?(err)"
                logger.warning("[%s] %s health gate eval failed — allowing entry: %s",
                               strategy_name, sym, e)

        # All layers passed — log entry summary then execute
        logger.info("[%s] %s %s ENTRY ✓ — L2:%s | Pat:%s",
                    strategy_name, sym, side.upper(), l2_label, l3_label)

        self._sig.lock_strategy(sym, slot, signal.type.value)
        self._sig.record_signal(sym, signal.type.value, signal.price,
                                signal.confidence, strategy=slot)
        if self.telegram:
            self.telegram.notify_signal({**sig_dict, "strategy": slot})

        pattern_tier = sig_dict.get("_pattern_tier", 0)
        opened = await self._open_position(signal, slot, side, pattern_tier=pattern_tier)

        # Arm the whipsaw cooldown ONLY after a confirmed fill — a rejected order
        # (insufficient margin, sub-minimum size) must not strand the symbol.
        if opened and hasattr(strategy, "arm_cooldown"):
            strategy.arm_cooldown()

    # ── Order execution ───────────────────────────────────────────────────

    async def _open_position(self, signal: Signal, slot: str, side: str, pattern_tier: int = 0):
        """Open a LONG or SHORT futures position."""
        sym = signal.symbol
        try:
            balances = await self.connector.fetch_balance()
            usdt_free = next(
                (b.free for b in balances if b.asset in ("USDT", "USD", "BUSD")), 0.0
            )
            ticker = await self.connector.fetch_ticker(sym)
            price  = float(ticker["last"])

            meta = signal.metadata or {}
            # Re-anchor SL/TP to the ACTUAL entry price using the price-independent
            # 1R distance, so entry−SL == one_r exactly. The strategy's metadata levels
            # were anchored to its ticker; price may have drifted before this fill,
            # which would otherwise break the R-multiple math the protection stack relies on.
            one_r = float(meta.get("one_r", 0) or 0)
            if one_r > 0:
                rr_tp1 = float(meta.get("rr_tp1", 0.5))
                rr_tp2 = float(meta.get("rr_tp2", 2.5))
                if side == "long":
                    sl_p  = round(price - one_r, 8)
                    tp1_r = round(price + rr_tp1 * one_r, 8)
                    tp2_r = round(price + rr_tp2 * one_r, 8)
                else:
                    sl_p  = round(price + one_r, 8)
                    tp1_r = round(price - rr_tp1 * one_r, 8)
                    tp2_r = round(price - rr_tp2 * one_r, 8)
                tp_p = tp2_r
                meta = {**meta, "stop_loss": sl_p, "take_profit": tp_p,
                        "tp1": tp1_r, "tp2": tp2_r}
            else:
                sl_p, tp_p = self._calc_sl_tp(signal, price, side)
            if sl_p is None or tp_p is None:
                logger.error("[%s] No SL/TP available for %s %s — order refused", slot, side, sym)
                self._sig.unlock_strategy(sym, slot)
                return False

            # Dynamic sizing: size by SL distance so worst-case loss ≈ risk_pct of port.
            sl_dist_pct = meta.get("sl_dist_pct")
            risk_pct    = meta.get("risk_pct", 0.02)
            if sl_dist_pct and self._dynamic_sizing:
                amount = self.risk.size_by_risk(usdt_free, price, sl_dist_pct, risk_pct)
            else:
                amount = self.risk.size_position(usdt_free, price)
            # All entries open at the same configured amount regardless of
            # confidence level — no per-trade size scaling.
            if amount <= 0:
                logger.warning("[%s] Position size=0 for %s — skipping", slot, sym)
                self._sig.unlock_strategy(sym, slot)
                return False

            # Align size to whole contracts so partial (50%) closes are tradable.
            # Partial close (TP1) needs ≥2 contracts to split; otherwise the
            # position runs as a single-TP to TP2 (no partial, no breakeven step).
            ct = await self.connector.contract_size(sym)
            contracts = int(round(amount / ct)) if ct > 0 else 0
            if contracts < 1:
                # Sub-minimum-lot amount — sending it would just get rejected by
                # the exchange. Skip cleanly instead.
                logger.warning(
                    "[%s] %s: sizing rounds to 0 contracts (amount=%.6f, contract=%.6f) — skipping entry",
                    slot, sym, amount, ct,
                )
                self._sig.unlock_strategy(sym, slot)
                return False
            amount = round(contracts * ct, 8)
            partial_ok = (meta.get("tp1") is not None) and contracts >= 2
            tp1_val = meta.get("tp1") if partial_ok else None
            if meta.get("tp1") is not None and not partial_ok:
                logger.info("[%s] %s: %d contract(s) < 2 → single-TP (no partial close)",
                            slot, sym, contracts)

            order_side = "buy" if side == "long" else "sell"
            pos_side   = side if self.futures_mode else ""

            # Exchange-side TP backstop is placed at the health ladder's MAX
            # (3.0R) rather than the strategy's starting TP2 (2.5R). The bot
            # closes the runner at the *current* pos.tp2 via market order each
            # tick (and the health monitor can raise pos.tp2 up the ladder toward
            # 3.0R). Anchoring the exchange algo TP at the ladder max means the
            # exchange only auto-closes if the bot is offline AND price runs all
            # the way to 3.0R — so BULL TP2-extensions take real effect on live
            # instead of being pre-empted by an exchange TP at 2.5R.
            if one_r > 0:
                max_r = PositionHealthMonitor.TP_LADDER[-1]
                exchange_tp = (round(price + max_r * one_r, 8) if side == "long"
                               else round(price - max_r * one_r, 8))
            else:
                exchange_tp = tp_p  # no 1R info → keep strategy TP as the backstop

            order = await self.connector.create_order(
                sym, order_side, amount,
                tp_price=exchange_tp, sl_price=sl_p,
                pos_side=pos_side,
            )
            # Sync full_amount with the actual OKX position size after fill.
            # The bot's pre-calculated amount may differ from OKX's actual contracts
            # (floating-point rounding, pre-existing manual positions, etc.).
            # Using the wrong full_amount causes TP1 to close the wrong percentage.
            actual_amount = amount
            if hasattr(self.connector, "fetch_position_amount"):
                try:
                    okx_amount = await self.connector.fetch_position_amount(sym, side)
                    if okx_amount > 0 and abs(okx_amount - amount) > ct * 0.5:
                        logger.warning(
                            "[%s] %s position size mismatch: ordered=%.6f OKX=%.6f — "
                            "using OKX size for full_amount",
                            slot, sym, amount, okx_amount,
                        )
                        actual_amount = okx_amount
                except Exception as e:
                    logger.warning("[%s] fetch_position_amount failed — using ordered amount: %s", slot, e)
            self.risk.open_position(
                sym, side, price, actual_amount,
                strategy=slot, stop_loss=sl_p, take_profit=tp_p,
                tp1=tp1_val, tp2=meta.get("tp2", tp_p),
                partial_pct=meta.get("partial_pct", 0.5),
                contract_size=ct,
                one_r=one_r,
            )
            # Store pattern tier for TP1 upgrade logic
            pos = self.risk.get_position_obj(sym, strategy=slot)
            if pos is not None:
                pos.pattern_tier   = pattern_tier
                pos.tp1_original   = pos.tp1 or 0.0
            self._record_trade(TradeRecord(
                timestamp=int(time.time() * 1000),
                symbol=sym, side=order_side,
                price=order.price, amount=actual_amount,
                pnl=0.0, strategy=slot, reason=signal.reason,
                paper=self.paper,
            ))
            self.lessons.record_open(sym, slot, side, meta)
            logger.info("[%s] OPEN %s %s @ %.4f  SL=%.4f  TP=%.4f  amount=%.6f",
                        slot, side.upper(), sym, price, sl_p, tp_p, actual_amount)
            if self.telegram:
                label = "buy" if side == "long" else "sell_short"
                self.telegram.notify_order(sym, label, actual_amount, order.price, slot, self.paper,
                                           sl=sl_p, tp=tp_p)
            return True

        except Exception as e:
            logger.error("[%s] Open %s %s failed: %s", slot, side, sym, e)
            self._sig.unlock_strategy(sym, slot)
            return False

    async def _check_stop(self, pos_info: dict):
        """Staged exit: TP1 closes part + moves SL→breakeven; TP2/breakeven/SL close the rest."""
        sym           = pos_info["symbol"]
        strategy_name = pos_info.get("strategy", "")
        close_key     = f"{sym}||{strategy_name}"
        if close_key in self._closing_positions:
            return  # health monitor already sent a close order — skip
        pos = self.risk.get_position_obj(sym, strategy_name)
        if pos is None:
            return
        try:
            ticker = await self.connector.fetch_ticker(sym)
            price  = float(ticker["last"])
        except Exception as e:
            logger.warning("[BOT] Ticker fetch failed for %s: %s", sym, e)
            return

        trigger = pos.stage_check(price)
        if not trigger:
            return

        is_long    = pos.side == "long"
        close_side = "sell" if is_long else "buy"
        pos_side   = ("long" if is_long else "short") if self.futures_mode else ""
        pnl_mult   = 1 if is_long else -1

        # Claim the close-lock for the whole trigger handling so the health
        # monitor can't send a concurrent close for the same position (which
        # would double-count PnL). Re-check here because we awaited fetch_ticker
        # after the top guard. The check-and-add below has no await between, so
        # it is atomic under the single-threaded event loop.
        if close_key in self._closing_positions:
            return  # monitor grabbed it while we were fetching the ticker
        self._closing_positions.add(close_key)
        try:
            # ── TP1: partial close + move stop to breakeven, keep the runner open ──
            if trigger == "tp1":
                # Close a whole number of contracts (≈partial_pct of the position).
                ct = pos.contract_size or 1.0
                full_contracts  = int(round(pos.full_amount / ct)) if ct > 0 else 0
                # Use round (not truncate) so a 2-contract position still splits:
                # int(2*0.4)=0 silently disabled TP1; round→1 closes one, keeps one.
                # Always leave at least one runner and always close at least one.
                close_contracts = max(1, int(round(full_contracts * pos.partial_pct)))
                close_contracts = min(close_contracts, full_contracts - 1)
                close_amt = round(close_contracts * ct, 8)
                if close_amt <= 0 or close_amt >= pos.amount:
                    # Can't split into whole contracts → keep position, treat as single-TP:
                    # disable TP1 so the runner exits at TP2 / SL only (no phantom state).
                    pos.tp1 = None
                    logger.info("[%s] %s TP1 hit but not splittable (%d contracts) → single-TP",
                                strategy_name, sym, full_contracts)
                    return
                try:
                    await self.connector.create_order(
                        sym, close_side, close_amt, pos_side=pos_side, reduce_only=True)
                except Exception as e:
                    logger.warning("[%s] TP1 partial close failed — will retry next tick: %s",
                                   strategy_name, e)
                    return  # do NOT mutate state; next tick will retry
                pnl = pnl_mult * (price - pos.entry_price) * close_amt
                self.risk.register_pnl(pnl)
                pos.amount     = round(pos.amount - close_amt, 6)
                pos.tp1_hit    = True
                # Runner SL: entry + 0.25R buffer locks minimum profit instead of exact BE.
                _one_r   = pos.one_r or abs(pos.entry_price - (pos.stop_loss or pos.entry_price))
                _strat   = next((s for s in self.strategies
                                 if s.symbol == sym and s.name == strategy_name), None)
                _be_buf  = float((getattr(_strat, "_p", None) or {}).get("tp1_be_buffer_r", 0.25))
                be_price = (pos.entry_price + _be_buf * _one_r if pos.side == "long"
                            else pos.entry_price - _be_buf * _one_r)
                pos.stop_loss  = be_price
                # Move exchange algo SL to BE+buffer. Non-fatal if it fails —
                # bot-side BE stop + health monitor protect the runner.
                sl_ok = await self.connector.move_sl_to_breakeven(
                    sym, pos_side, be_price, pos.amount
                )
                if not sl_ok:
                    logger.warning(
                        "[%s] TP1 %s — exchange SL not moved to BE+buf; "
                        "bot-side BE stop + health monitor protecting runner",
                        strategy_name, sym,
                    )
                self._record_trade(TradeRecord(
                    timestamp=int(time.time() * 1000), symbol=sym, side=close_side,
                    price=price, amount=close_amt, pnl=pnl,
                    strategy=strategy_name, reason="take_profit1", paper=self.paper,
                ))
                logger.info("[%s] TP1 %s @ %.4f closed %.6f (%.0f%%) → SL→BE+buf %.4f  pnl≈%.2f",
                            strategy_name, sym, price, close_amt, pos.partial_pct * 100,
                            be_price, pnl)
                if self.telegram:
                    self.telegram.notify_trade_closed(
                        sym, "take_profit1", price, pos.entry_price,
                        pos.stop_loss, pos.tp1, self._sig.summary(), side=pos.side)
                return

            # ── Terminal: stop_loss / breakeven / take_profit2 → close remainder ──
            amount = pos.amount
            pnl    = pnl_mult * (price - pos.entry_price) * amount
            try:
                await self.connector.create_order(
                    sym, close_side, amount, pos_side=pos_side, reduce_only=True)
            except Exception as e:
                err_str = str(e)
                # OKX 51169: "you don't have any positions" — exchange-side algo already closed it.
                okx_no_pos = "51169" in err_str or "don't have any positions" in err_str.lower()
                if trigger == "stop_loss" or okx_no_pos:
                    logger.info("[%s] Close failed for %s (OKX already closed via algo order): %s",
                                strategy_name, trigger, e)
                    # Fall through — clean up internal state below.
                else:
                    # Transient network/exchange error — retry next tick.
                    logger.warning("[%s] Close failed for %s — will retry next tick: %s",
                                   strategy_name, trigger, e)
                    return

            self.risk.register_pnl(pnl)
            self._sig.record_outcome(
                symbol=sym, side=pos.side,
                entry=pos.entry_price, exit_price=price,
                sl=pos.stop_loss, tp=pos.take_profit,
                reason=trigger, strategy=strategy_name,
            )
            self._sig.unlock_strategy(sym, strategy_name)
            self.risk.close_position(sym, strategy=strategy_name)
            self._record_trade(TradeRecord(
                timestamp=int(time.time() * 1000), symbol=sym, side=close_side,
                price=price, amount=amount, pnl=pnl,
                strategy=strategy_name, reason=trigger, paper=self.paper,
            ))
            logger.info("[%s] CLOSED %s @ %.4f via %s  pnl≈%.2f USDT",
                        strategy_name, sym, price, trigger, pnl)
            if self.telegram:
                self.telegram.notify_trade_closed(
                    sym, trigger, price, pos.entry_price,
                    pos.stop_loss, pos.take_profit, self._sig.summary(), side=pos.side)
        finally:
            self._closing_positions.discard(close_key)

    # ── Position health monitor ───────────────────────────────────────────

    async def _monitor_loop(self):
        """Background task: re-evaluates every open position on a short interval."""
        monitor = PositionHealthMonitor(self.connector)

        # Stagger first run by half the interval so it doesn't fire at the same time as _tick.
        await asyncio.sleep(self._monitor_interval // 2)

        while self._running:
            try:
                await self._run_health_checks(monitor)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[MONITOR] Unexpected error: %s", e, exc_info=True)
            await asyncio.sleep(self._monitor_interval)

    async def _run_health_checks(self, monitor):
        positions = self.risk.get_positions()
        if not positions:
            return

        candle_cache: dict[tuple, list] = {}

        for pos_info in positions:
            sym      = pos_info["symbol"]
            strategy = pos_info.get("strategy", "")
            pos      = self.risk.get_position_obj(sym, strategy)
            if pos is None:
                continue

            # ── OKX position sync: detect positions closed externally (algo TP/SL) ──
            # fetch_position_amount returns 0 if OKX has no position; clean up stale state.
            if not self.paper:
                try:
                    actual_amt = await self.connector.fetch_position_amount(sym, pos.side)
                    if actual_amt == 0.0:
                        logger.warning(
                            "[MONITOR] %s %s %s — OKX shows NO position (closed externally). "
                            "Cleaning up internal state.",
                            strategy, sym, pos.side.upper())
                        self.risk.close_position(sym, strategy=strategy)
                        self._sig.unlock_strategy(sym, strategy)
                        if self.telegram:
                            self.telegram.notify(
                                f"⚠️ *Stale Position Cleared* `{sym}` {pos.side.upper()}\n"
                                f"OKX shows no position — likely closed by exchange algo order.\n"
                                f"Internal state cleaned up."
                            )
                        continue
                except Exception as e:
                    logger.warning("[MONITOR] %s %s position sync failed: %s", sym, sym, e)

            # Pre-fetch once per symbol across all TFs.
            # 15m is fetched for the Trend-Fade guard (ADX(15m) falling check).
            for tf in ("3m", "5m", "15m", "1h", "4h"):
                key = (sym, tf)
                if key not in candle_cache:
                    try:
                        lim = 80 if tf == "3m" else 120
                        candle_cache[key] = await self.connector.fetch_ohlcv(sym, tf, limit=lim)
                    except Exception as e:
                        logger.warning("[MONITOR] %s %s fetch failed: %s", sym, tf, e)
                        candle_cache[key] = []

            c3m  = candle_cache.get((sym, "3m"),  [])
            c5m  = candle_cache.get((sym, "5m"),  [])
            c15m = candle_cache.get((sym, "15m"), [])
            c1h  = candle_cache.get((sym, "1h"),  [])
            c4h  = candle_cache.get((sym, "4h"),  [])

            # ── Crash-guard (v2 strategy): closes ONLY when deeply underwater + momentum reversed
            strat_obj = next(
                (s for s in self.strategies if s.symbol == sym and s.name == strategy), None
            )
            if strat_obj and hasattr(strat_obj, "monitor_position"):
                one_r = pos.one_r or abs(pos.entry_price - (pos.stop_loss or pos.entry_price))
                cg_action, cg_reason = strat_obj.monitor_position(
                    pos.side, pos.entry_price, one_r, pos.tp1_hit,
                    c5m, c1h, c4h, candles_3m=c3m, candles_15m=c15m,
                )
                if cg_action == "CLOSE":
                    cg_result = HealthResult(score=0.0, label="CRASH_GUARD",
                                             action="CLOSE", details={"reason": cg_reason})
                    pos.health_label  = cg_result.label
                    pos.health_checks += 1
                    await self._health_action(pos, pos_info, strategy, cg_result, monitor)
                    continue

            # ── Weighted score (BULL → extend TP; WEAK is secondary, crash-guard is primary) ──
            result = await monitor.check(
                symbol=sym,
                side=pos.side,
                entry_price=pos.entry_price,
                oneR=pos.one_r or abs(pos.entry_price - (pos.stop_loss or pos.entry_price)),
                current_tp2=pos.tp2 or pos.take_profit or 0.0,
                candles_5m=c5m,
                candles_1h=c1h,
                candles_4h=c4h,
            )
            pos.health_label  = result.label
            pos.health_checks += 1

            # STRONG_WEAK (score < 25): sharp reversal / V-shape — close NOW, no wait.
            # WEAK (25-49): soft fade — must persist N consecutive cycles before close.
            weak_confirm = self._health_weak_confirm
            if result.label == "STRONG_WEAK":
                pos.health_weak_count = 0
                logger.info("[MONITOR] %s %s STRONG_WEAK (score=%.0f) — closing immediately",
                            strategy, sym, result.score)
            elif result.label == "WEAK":
                pos.health_weak_count += 1
                # Pattern-upgraded TP1: don't wait N confirms — exit now (momentum fading)
                if pos.tp1_pattern_upgraded and not pos.tp1_hit:
                    logger.info(
                        "[PAT-TP1] %s %s WEAK after TP1 upgrade (%.1fR target) — exit immediately",
                        strategy, sym, TP1_BY_TIER.get(pos.pattern_tier, 0))
                elif pos.health_weak_count < weak_confirm:
                    logger.info("[MONITOR] %s %s WEAK (score=%.0f) — %d/%d confirmation, holding",
                                strategy, sym, result.score, pos.health_weak_count, weak_confirm)
                    continue
            else:
                pos.health_weak_count = 0  # reset on any non-WEAK cycle

            await self._health_action(pos, pos_info, strategy, result, monitor)

    async def _health_action(self, pos, pos_info: dict, strategy: str,
                             result, monitor):
        """Act on a health check result: extend TP or close early."""
        sym = pos.symbol

        # ── BULL: advance TP1 (before hit, max 1.5R) or TP2 (after hit) ────────
        if result.action == "EXTEND_TP":
            one_r = pos.one_r or abs(pos.entry_price - (pos.stop_loss or pos.entry_price))
            if not pos.tp1_hit:
                # TP1 upgrade requires BOTH: pattern tier present AND health BULL (this call)
                # Health BULL alone → no adjustment (TP1 stays at 0.5R)
                # Pattern alone     → no adjustment (wait for BULL confirmation)
                if pos.pattern_tier > 0 and not pos.tp1_pattern_upgraded:
                    r_mult   = TP1_BY_TIER.get(pos.pattern_tier, 0.5)
                    new_tp1  = (round(pos.entry_price + r_mult * one_r, 8) if pos.side == "long"
                                else round(pos.entry_price - r_mult * one_r, 8))
                    old_tp1  = pos.tp1
                    pos.tp1                  = new_tp1
                    pos.tp1_pattern_upgraded = True
                    logger.info(
                        "[PAT-TP1] %s %s tier-%d → TP1 %.4f→%.4f (%.1fR) | WEAK exit armed",
                        strategy, sym, pos.pattern_tier, old_tp1 or 0, new_tp1, r_mult)
                    if self.telegram:
                        self.telegram.notify(
                            f"🎯 *Pattern TP1 Upgrade* `{sym}` {pos.side.upper()}\n"
                            f"Tier {pos.pattern_tier} + Health BULL → TP1: `{old_tp1 or 0:.2f}` → `{new_tp1:.2f}` _({r_mult}R)_\n"
                            f"Early-exit armed on WEAK | Score {result.score:.0f}%"
                        )
            else:
                # After TP1: runner is at BE — extend TP2 up the ladder (max 3.0R)
                new_tp = monitor.next_tp_level(pos.entry_price, one_r,
                                               pos.tp2 or pos.take_profit or 0.0, pos.side)
                if new_tp and new_tp != pos.tp2:
                    old_tp = pos.tp2
                    pos.tp2 = new_tp
                    logger.info("[MONITOR] %s %s BULL (runner) → TP2 extended %.4f → %.4f",
                                strategy, sym, old_tp or 0, new_tp)
                    if self.telegram:
                        self.telegram.notify(
                            f"📈 *Health BULL* `{sym}` {pos.side.upper()} _(runner)_\n"
                            f"Score {result.score:.0f}%  TP2 extended: `{old_tp or 0:.2f}` → `{new_tp:.2f}`\n"
                            f"Indicators: {_fmt_details(result.details)}"
                        )
            return

        # ── WEAK: force-close position ────────────────────────────────────────
        if result.action == "CLOSE":
            close_key = f"{sym}||{strategy}"
            if close_key in self._closing_positions:
                return  # already being closed by _check_stop or prior health check
            self._closing_positions.add(close_key)
            # try/finally guarantees the lock is released on EVERY exit path —
            # without it, a failed close order would leave the key set forever,
            # permanently blocking _check_stop and future health checks for this
            # position (stuck, unmanaged position).
            try:
                try:
                    ticker = await self.connector.fetch_ticker(sym)
                    price  = float(ticker["last"])
                except Exception as e:
                    logger.warning("[MONITOR] %s ticker failed during health-close: %s", sym, e)
                    return

                is_long    = pos.side == "long"
                close_side = "sell" if is_long else "buy"
                pos_side   = (pos.side if self.futures_mode else "")
                amount     = pos.amount

                logger.warning("[MONITOR] %s %s %s (score=%.0f) → force-closing %.6f @ %.4f",
                               strategy, sym, result.label, result.score, amount, price)
                try:
                    order = await self.connector.create_order(sym, close_side, amount,
                                                              pos_side=pos_side, reduce_only=True)
                except Exception as e:
                    logger.error("[MONITOR] %s health-close order failed: %s — will retry next cycle", sym, e)
                    return  # finally releases the lock so the next cycle can retry

                # Book PnL from the actual fill price when the connector provides it,
                # falling back to the ticker. Closing `amount` (whole runner) via
                # reduce_only is exact, so size is reliable.
                fill_price = float(getattr(order, "price", 0.0) or price)
                pnl = (1 if is_long else -1) * (fill_price - pos.entry_price) * amount
                self.risk.register_pnl(pnl)
                self._sig.record_outcome(sym, pos.side, pos.entry_price, fill_price,
                                         pos.stop_loss, pos.take_profit,
                                         reason="health_weak", strategy=strategy)
                self._sig.unlock_strategy(sym, strategy)
                self.risk.close_position(sym, strategy=strategy)
                _detail = (result.details.get("reason", "")
                           if result.label == "CRASH_GUARD"
                           else f"health {result.label} score={result.score:.0f}")
                self._record_trade(TradeRecord(
                    timestamp=int(time.time() * 1000), symbol=sym, side=close_side,
                    price=fill_price, amount=amount, pnl=pnl,
                    strategy=strategy, reason="health_weak", paper=self.paper,
                    detail=_detail,
                ))
                reason_str = result.details.get("reason", "") if result.label == "CRASH_GUARD" else ""
                if self.telegram:
                    self.telegram.notify(
                        f"⚠️ *Health {result.label}* `{sym}` {pos.side.upper()} → closed early\n"
                        f"Score {result.score:.0f}%  PnL≈`{pnl:+.2f}$`\n"
                        + (f"_{reason_str}_" if reason_str else f"Indicators: {_fmt_details(result.details)}")
                    )
            finally:
                self._closing_positions.discard(close_key)

    # ── SL/TP calculation ─────────────────────────────────────────────────

    def _calc_sl_tp(self, signal: Signal, price: float,
                    side: str) -> tuple[Optional[float], Optional[float]]:
        """
        Priority: strategy metadata → fixed percentage config.
        For shorts, SL is above price and TP is below price.
        """
        meta = signal.metadata or {}
        sl   = meta.get("stop_loss")
        tp   = meta.get("take_profit")

        if sl is not None and tp is not None:
            # Validate orientation for short (strategy should provide short-side values)
            if side == "short":
                if sl < price:  # SL should be above entry for shorts
                    sl = round(price * (1 + self.fixed_sl_pct), 8) if self.fixed_sl_pct else None
                if tp > price:  # TP should be below entry for shorts
                    tp = round(price * (1 - self.fixed_tp_pct), 8) if self.fixed_tp_pct else None
            if sl is not None and tp is not None:
                return sl, tp

        # Fallback: fixed percentage
        if self.fixed_sl_pct > 0 and self.fixed_tp_pct > 0:
            if side == "long":
                return (round(price * (1 - self.fixed_sl_pct), 8),
                        round(price * (1 + self.fixed_tp_pct), 8))
            else:
                return (round(price * (1 + self.fixed_sl_pct), 8),
                        round(price * (1 - self.fixed_tp_pct), 8))
        return None, None

    # ── Balance & state ───────────────────────────────────────────────────

    async def _refresh_balance(self):
        try:
            balances = await self.connector.fetch_balance()
            self._balance = sum(
                b.total for b in balances if b.asset in ("USDT", "USD", "BUSD")
            )
            if self._start_balance > 0:
                self._pnl_total = self._balance - self._start_balance
                self.risk.update_peak(self._balance)
        except Exception as e:
            logger.warning("[BOT] Balance refresh failed: %s", e)

    def _record_trade(self, trade: TradeRecord):
        self._trade_history.append(trade)
        # Route to the lesson tracker: TP1 partials accumulate into the pending
        # trade; terminal reasons close it out (win/loss judged on TOTAL pnl).
        try:
            if trade.reason == "take_profit1":
                self.lessons.record_partial(trade.symbol, trade.strategy, trade.pnl)
            elif trade.reason in TERMINAL_REASONS:
                self.lessons.record_close(trade.symbol, trade.strategy, trade.reason,
                                          trade.pnl, trade.price, trade.detail)
                self._sync_lesson_caution()
        except Exception as e:
            logger.warning("[LESSON] tracking failed: %s", e)

    def _sync_lesson_caution(self):
        """Apply/remove the adaptive min_score bump requested by the lesson tracker."""
        bump = self.lessons.score_bump_active()
        if bump == self._lesson_bump:
            return
        for s in self.strategies:
            p = getattr(s, "_p", None)
            if not isinstance(p, dict) or "min_score" not in p:
                continue
            base = self._lesson_base.setdefault(id(s), float(p["min_score"]))
            p["min_score"] = base + bump
        logger.info("[LESSON] caution min_score bump %.0f → %.0f", self._lesson_bump, bump)
        if bump == 0.0 and self.telegram:
            try:
                self.telegram.notify("🛡 *Caution mode OFF* — min_score กลับค่าปกติ")
            except Exception:
                pass
        self._lesson_bump = bump

    def _broadcast_state(self):
        try:
            self._broadcast({"type": "trading_update", "state": self.get_state()})
        except Exception:
            pass

    def get_state(self) -> dict:
        return {
            "running":      self._running,
            "paper":        self.paper,
            "balance":      round(self._balance, 2),
            "equity":       round(self._balance, 2),
            "pnl_total":    round(self._pnl_total, 2),
            "positions":    self.risk.get_positions(),
            "recent_trades": [
                {"ts": t.timestamp, "symbol": t.symbol, "side": t.side,
                 "price": t.price, "amount": t.amount, "pnl": t.pnl,
                 "strategy": t.strategy, "paper": t.paper}
                for t in self._trade_history[-20:]
            ],
            "signals":      self._signals_cache,
            "error":        self._error,
            "last_updated": int(time.time() * 1000),
        }

    def get_stats(self) -> dict:
        return self._sig.summary()
