"""DUALCORE V2.0 live bot entry point.

Loop: fetch closed 5M/15M/1H/4H bars, manage open positions every poll, and
for flat symbols evaluate 4H Macro -> 1H Bias -> 15M Structure/Location ->
5M EMA timing (Fast Pullback + Major/Base Breakout-Retest).  Entry logic runs once
per newly-closed 5M candle.  Position management keeps the slower EMA10/EMA20
5M reversal exit to avoid closing from the faster EMA8/EMA13 entry pair alone.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time

import pandas as pd

from config import load_config
from exchange_client import ExchangeClient
from data_engine import DataEngine, drop_unclosed_bar, _ohlcv_to_df
from pipeline import Pipeline, LONG, SHORT
from risk_manager import RiskManager
from position_manager import PositionManager
from telegram_notifier import TelegramNotifier
from chart_engine import build_entry_chart
from spike_guard import check_spike, CLOSE as SPIKE_CLOSE
from entry_engine import EMA_CROSS_REVERSAL, PRICE_OPEN_BEYOND_EMA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("main")


def _sym(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTC' for compact Telegram output."""
    return symbol.split("/")[0]


def _entry_score_text(entry) -> str:
    """Render N/A until a complete candidate was actually scored."""
    if entry is None or not getattr(entry, "score_evaluated", False):
        return "N/A"
    threshold = getattr(entry, "score_threshold", None)
    if threshold is None:
        return f"{entry.entry_score:.0f}"
    return f"{entry.entry_score:.0f}/{threshold:.0f}"


def _stats_reset_path(state_dir: str) -> str:
    return os.path.join(state_dir, "stats_reset.json")


def _load_stats_reset_ms(state_dir: str):
    """The /restats cursor, if one was ever set — None means "use
    STATS_SINCE_DATE". Missing/corrupt file is not an error, just no override."""
    try:
        with open(_stats_reset_path(state_dir)) as f:
            return int(json.load(f)["since_ms"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _save_stats_reset_ms(state_dir: str, since_ms: int) -> None:
    """Atomic write so a crash mid-write can never leave a truncated/corrupt
    marker file (same pattern as dual_entry_v14's state_store)."""
    os.makedirs(state_dir, exist_ok=True)
    path = _stats_reset_path(state_dir)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"since_ms": since_ms}, f)
    os.replace(tmp, path)


class Bot:
    def __init__(self):
        self.cfg = load_config()
        self.client = ExchangeClient(
            api_key=self.cfg.okx_api_key, api_secret=self.cfg.okx_secret,
            passphrase=self.cfg.okx_passphrase, paper=self.cfg.paper,
            leverage=self.cfg.leverage, margin_mode=self.cfg.margin_mode,
            fee_rate=self.cfg.fee_rate,
        )
        self.data = DataEngine(self.cfg, self.client)
        self.signal_engine = Pipeline(self.cfg)
        self.risk = RiskManager(self.cfg)
        self.positions = PositionManager(self.cfg, self.client, self.risk,
                                         self.signal_engine.entry_engine)
        self.telegram = TelegramNotifier(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id)

        self._last_entry_bar: dict[str, pd.Timestamp] = {}
        self._last_signal_by_symbol: dict[str, object] = {}
        self._symbol_cooldown_until: dict[str, float] = {}
        self._daily_alert_sent = False
        self._cooldown_alert_sent = False
        self._last_status_log_ts = 0.0
        self._last_reconcile_ts = 0.0         # periodic untracked-position safety net
        self._trade_log: list[dict] = []      # closed trades (for /trades, live view)
        # Persistent close-journal — one line per closed trade with the exit
        # bucket (TP/BE/SL) the bot observed. Survives restarts and is the ONLY
        # source of the TP/BE/SL breakdown (OKX has no "which target" concept).
        # Trade COUNT / WR / PnL always come from OKX, never from here.
        # Same format/role as the HTF bot's journal so /stats is unified.
        self._journal_path = os.path.join(self.cfg.state_dir, "trade_journal.jsonl")
        self.journal: list[dict] = self._load_journal()
        self._journaled_close_ms = {(e["symbol"], int(e["close_ms"] // 60000))
                                    for e in self.journal}   # dedup key: symbol+minute
        self._stats_reset_ms = _load_stats_reset_ms(self.cfg.state_dir)   # /restats cursor override
        self._cmd_task = None                 # Telegram command polling task
        self._tg_offset = 0
        self._running = False

    # ── close journal (exit-type breakdown source) ───────────────────────────

    def _load_journal(self) -> list[dict]:
        out: list[dict] = []
        try:
            with open(self._journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except (FileNotFoundError, OSError):
            pass
        except json.JSONDecodeError:
            logger.warning("[JOURNAL] corrupt line(s) skipped")
        return out

    def _journal_add(self, symbol: str, side: str, pnl: float, exit_type: str,
                     reason: str, close_ms: int) -> None:
        """Append one closed trade. Deduped by (symbol, close-minute) so a
        restart-time backfill can't double-count a trade already journaled.
        `reason` keeps the raw event (TP2_HIT / EMA_CROSS_REVERSAL / …) so the
        collapsed TP/BE/SL bucket can be re-derived or expanded later."""
        key = (symbol, int(close_ms // 60000))
        if key in self._journaled_close_ms:
            return
        entry = {"close_ms": int(close_ms), "symbol": symbol, "side": side,
                 "pnl": round(float(pnl), 4), "exit_type": exit_type, "reason": reason}
        try:
            os.makedirs(self.cfg.state_dir, exist_ok=True)
            with open(self._journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning("[JOURNAL] write failed: %s", e)
        self.journal.append(entry)
        self._journaled_close_ms.add(key)

    @staticmethod
    def _bucket_for_event(ev: str, pnl: float) -> str:
        """Collapse this 2-TP bot's exit reasons into the HTF bot's three
        buckets (TP / BE / SL) so /stats reads identically across bots. The raw
        reason is still journaled, so TP1/TP2 can be split back out later."""
        if ev == "TP2_HIT":
            return "TP"
        if ev == "BE_HIT":
            return "BE"
        if ev == "SL_HIT":
            return "SL"
        # TP1_THEN_EXTERNAL_CLOSE, EMA_CROSS_REVERSAL, PRICE_OPEN_BEYOND_EMA,
        # SPIKE_GUARD — a target-or-stop by outcome: sign of realized PnL.
        if pnl > 1e-8:
            return "TP"
        if pnl < -1e-8:
            return "SL"
        return "BE"

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start live: " + "; ".join(problems))

        if not self.cfg.paper:
            hedge_ok = await self.client.ensure_hedge_mode()
            if not hedge_ok:
                raise RuntimeError("Could not confirm OKX hedge mode — refusing to trade live.")

        balance = await self.client.fetch_balance_usdt()
        logger.info("=== Signal Regime Bias Bot starting [%s] symbols=%s lev=%dx balance=%.2f ===",
                   "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols, self.cfg.leverage, balance)
        if not self.telegram.enabled:
            logger.warning("Telegram not configured — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing")

        # Adopt any position that already exists on OKX but isn't tracked —
        # this bot's state is in-memory only, so a restart would otherwise
        # orphan every open position (no more SL/TP/health management for it).
        await self._reconcile_positions(context="STARTUP")

        self._running = True
        if self.telegram.enabled:
            self._cmd_task = asyncio.create_task(self._command_loop())
            logger.info("Telegram command interface active (/help)")
            # Confirm in Telegram that a (re)deploy actually came up healthy —
            # otherwise a clean restart is completely silent in the chat and
            # there's no way to tell "still starting" from "crashed".
            await self.telegram.send_text(
                f"🤖 *Bot started* [{'PAPER' if self.cfg.paper else 'LIVE'}]\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT\n"
                f"Architecture: 4H Regime → 1H Bias → 15M Context → 5M EMA Dual Entry "
                f"(Fast Pullback + Momentum Breakout/Retest)"
            )

    async def stop(self):
        self._running = False
        if self._cmd_task:
            self._cmd_task.cancel()
            try:
                await self._cmd_task
            except asyncio.CancelledError:
                pass
        await self.client.close()
        logger.info("Bot stopped cleanly.")

    async def run_forever(self):
        while self._running:
            # Safety net FIRST: adopt any position that exists on OKX but the
            # bot isn't tracking (e.g. an order that filled but whose tracking
            # was lost to a crash/timeout), BEFORE the entry logic runs — so a
            # symbol that's actually in a position is never re-opened, and the
            # user is always told about a live position within ~one cycle.
            await self._maybe_reconcile()
            for symbol in self.cfg.symbols:
                try:
                    await self._process_symbol(symbol)
                except Exception as e:
                    logger.error("[%s] unhandled error: %s", symbol, e, exc_info=True)
                    await self.telegram.error(symbol, str(e))
            await self._check_global_alerts()
            await self._maybe_log_status()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def _maybe_reconcile(self):
        now = time.time()
        if now - self._last_reconcile_ts < self.cfg.reconcile_interval_sec:
            return
        self._last_reconcile_ts = now
        try:
            await self._reconcile_positions(context="SAFETY-NET")
        except Exception as e:
            logger.warning("[RECONCILE] periodic sweep failed: %s", e)

    async def _reconcile_positions(self, context: str):
        """Adopt untracked OKX positions and ALERT for each. Runs at startup
        and every reconcile_interval_sec — the single guard against a position
        living on OKX while the bot flies blind (no SL/TP management, no
        notification, and — worst — re-opening because it thinks it's flat)."""
        adopted = await self.positions.reconcile_with_exchange(self.cfg.symbols)
        if not adopted:
            return
        logger.warning("[%s] Adopted %d untracked position(s): %s", context, len(adopted), adopted)
        for entry in adopted:
            # reconcile returns "SYMBOL SIDE" strings; _positions is keyed by
            # the bare symbol (which itself contains no spaces).
            sym = entry.rsplit(" ", 1)[0]
            pos = self.positions.get(sym)
            if pos is None:
                # e.g. a hedge-conflict warning string — surface it as-is so an
                # unmanaged leg is never silently dropped.
                await self.telegram.send_text(f"⚠️ *Reconcile* ({context})\n\n`{entry}`")
                continue
            await self.telegram.send_text(
                f"⚠️ *Adopted untracked position* `{sym}` ({context})\n\n"
                f"This position was live on OKX but the bot wasn't tracking it — "
                f"now managing SL/TP.\n"
                f"Side: `{pos.side}`  Entry: `{pos.entry_price:.6f}`\n"
                f"SL: `{pos.stop_loss:.6f}`  TP2: `{pos.tp2:.6f}`  "
                f"TP1: `{pos.tp1 if pos.tp1 else 'hit'}`\n"
                f"Amount: `{pos.amount:.6f}`"
            )

    # ── Per-symbol processing ────────────────────────────────────────────────

    async def _process_symbol(self, symbol: str):
        self.data.new_tick()
        frames = await self.data.fetch_all(symbol)
        if not self.data.has_min_bars(frames):
            logger.debug("[%s] insufficient bars (<%d) on one or more TFs — skip",
                        symbol, self.cfg.min_bars)
            return

        df_1h = frames[self.cfg.tf_bias]
        df_4h = frames[self.cfg.tf_regime]
        df_30m = frames[self.cfg.tf_entry]
        df_15m = frames[self.cfg.tf_fast]
        df_5m = frames[self.cfg.tf_micro]

        # Full pipeline computed once per symbol per tick and cached — reused by
        # the entry check below AND by the 5-minute status log, so the exit
        # branch and the entry branch never re-run the layers separately.
        sig = self.signal_engine.evaluate(df_1h, df_4h, df_15m, df_5m, df_30m=df_30m, symbol=symbol)
        self._last_signal_by_symbol[symbol] = sig

        if self.positions.has_position(symbol):
            await self._manage_open_position(symbol, df_15m, df_5m)
        else:
            await self._look_for_entry(symbol, df_15m, df_5m, sig)

    async def _manage_open_position(self, symbol: str, df_15m, df_5m):
        pos = self.positions.get(symbol)
        try:
            ticker = await self.client.fetch_ticker(symbol)
            price = float(ticker["last"])
        except Exception as e:
            logger.warning("[%s] ticker fetch failed: %s", symbol, e)
            return

        event = await self.positions.check_exits_live(symbol, price)
        if event:
            await self._handle_event(event)
            return   # position fully or partially closed this tick — exit check waits for next

        # SpikeGuard — EVERY tick, 5m/15m closed bars + live price. The fast
        # layer: force-close before a V-reversal eats the full SL (+slippage).
        if self.cfg.spike_guard_enabled:
            spike_event = await self._check_spike_guard(symbol, pos, price)
            if spike_event:
                await self._handle_event(spike_event)
                return

        # EMA early-exit check — once per newly-closed 5m bar only.
        eevent = await self.positions.process_closed_bar_exit_check(symbol, df_5m)
        if eevent:
            await self._handle_event(eevent)

    async def _check_spike_guard(self, symbol: str, pos, price: float):
        """Fetch 5m/15m closed bars and run the spike check. Non-fatal on data errors."""
        import time as _t
        c = self.cfg
        now_ms = int(_t.time() * 1000)
        try:
            raw5 = await self.client.fetch_ohlcv(symbol, c.spike_tf_fast, limit=c.spike_fetch_limit)
            raw15 = await self.client.fetch_ohlcv(symbol, c.spike_tf_slow, limit=c.spike_fetch_limit)
        except Exception as e:
            logger.warning("[%s] spike-guard data fetch failed (skipping this tick): %s", symbol, e)
            return None
        df5 = drop_unclosed_bar(_ohlcv_to_df(raw5), c.spike_tf_fast, now_ms)
        df15 = drop_unclosed_bar(_ohlcv_to_df(raw15), c.spike_tf_slow, now_ms)

        result = check_spike(pos.side, pos.entry_price, pos.one_r, df5, df15, price, c)
        if result.action != SPIKE_CLOSE:
            return None
        logger.warning("[%s] SPIKE GUARD firing: %s", symbol, result.reason)
        event = await self.positions._close_full(pos, price, "SPIKE_GUARD")
        if event.get("event") == "SPIKE_GUARD":
            event["spike_reason"] = result.reason
        return event

    async def _look_for_entry(self, symbol: str, df_15m, df_5m, sig):
        # Actual execution runs once per completed 5M candle. Repeated polls of
        # the same candle must never submit the same setup twice.
        bar_ts = df_5m.index[-1] if (df_5m is not None and len(df_5m)) else None
        if bar_ts is None or self._last_entry_bar.get(symbol) == bar_ts:
            return   # already evaluated this closed bar
        self._last_entry_bar[symbol] = bar_ts

        now = time.time()
        cooldown_until = self._symbol_cooldown_until.get(symbol, 0)
        if now < cooldown_until:
            logger.debug("[%s] symbol cooldown: %.0f min left", symbol, (cooldown_until - now) / 60)
            return

        balance = await self.client.fetch_balance_usdt()
        can_open, reason = self.risk.can_open_new(balance, now, self.positions.open_position_count())
        if not can_open:
            logger.debug("[%s] entry blocked: %s", symbol, reason)
            return

        if sig.direction not in (LONG, SHORT):
            self._log_pipeline_block(symbol, sig)
            return

        risk_pct = self.cfg.risk_per_trade * sig.regime.size_multiplier
        # Chart the actual 5M execution frame with EMA8/EMA13.
        chart_path = build_entry_chart(
            symbol, df_5m, sig.direction, sig.price,
            *self._preview_sl_tp(sig, df_5m),
            ema_fast_len=self.cfg.dual_entry_ema_fast,
            ema_slow_len=self.cfg.dual_entry_ema_slow,
            tf_label="5M",
        )

        # PositionManager's first frame is the execution frame used for fallback
        # ATR/swing calculations. EntryResult normally supplies the structure SL.
        pos = await self.positions.open_position(
            symbol, sig.direction, sig.price, df_5m, sig.regime, sig.bias,
            sig.entry_score, df_5m=df_5m, entry_result=sig.entry)
        if pos is None:
            return

        # Consume the deterministic setup key ONLY now that a position really opened —
        # a blocked/failed open above leaves the setup eligible until it expires.
        self.signal_engine.entry_engine.confirm_entry(
            symbol, sig.entry.cross_id if sig.entry is not None else None)

        # The position is OPEN and tracked — the alerts below must NOT be able
        # to raise back into the caller (which would look like an entry
        # failure) or let one broken message suppress the others. Send the
        # simple, always-valid "Order Opened" text FIRST, then the richer
        # chart signal; isolate and loudly log each.
        logger.info("[%s] position opened — notifying Telegram", symbol)
        try:
            await self.telegram.order_opened(
                symbol, sig.direction, pos.entry_price, pos.amount, pos.stop_loss, pos.tp1, pos.tp2)
        except Exception as e:
            logger.error("[%s] order_opened notify failed: %s", symbol, e, exc_info=True)
        try:
            await self.telegram.entry_signal(
                symbol, sig.direction, pos.entry_price, pos.stop_loss, pos.tp1, pos.tp2,
                sig.regime, sig.bias, sig.entry_score, risk_pct, self.cfg.leverage,
                chart_path, sig.entry)
        except Exception as e:
            logger.error("[%s] entry_signal notify failed: %s", symbol, e, exc_info=True)

    def _log_pipeline_block(self, symbol: str, sig):
        """Explain the exact layer and setup state that blocked a trade."""
        r = sig.regime
        if sig.blocked_layer == "BIAS" and sig.bias is not None:
            b = sig.bias
            logger.info(
                "[%s] regime=%s(4H=%s,1H=%s) BIAS NO-TRADE bull=%.0f bear=%.0f edge=%+.0f — %s",
                symbol, r.label, r.label_4h, r.label_1h,
                b.bull_score, b.bear_score, b.directional_edge, b.reason,
            )
        elif sig.blocked_layer == "ENTRY" and sig.entry is not None:
            e = sig.entry
            edge_text = f"{e.macd_hist:.1f}" if getattr(e, "score_evaluated", False) else "N/A"
            logger.info(
                "[%s] regime=%s bias=%s NO ENTRY EMA8/13=%.6f/%.6f edge=%s — %s",
                symbol, r.label, sig.bias.direction if sig.bias else "-",
                e.ema_fast, e.ema_slow, edge_text, e.reason,
            )
        elif sig.blocked_layer == "MARKET":
            logger.info("[%s] MARKET CLOSED — %s", symbol, sig.reason)
        else:
            logger.debug("[%s] no trade: %s", symbol, sig.reason)

    def _preview_sl_tp(self, sig, execution_df) -> tuple[float, float, float]:
        """Use the Entry Engine's structure plan when available."""
        import indicators as ind
        from position_manager import calc_stop_loss, calc_take_profits, _normalize_planned_stop
        c = self.cfg
        if sig.entry is not None and sig.entry.planned_stop is not None:
            atr_val = float(ind.atr(execution_df, c.sl_atr_period).iloc[-1])
            side = "long" if sig.direction == LONG else "short"
            sl, _, _ = _normalize_planned_stop(
                c, side, sig.price, float(sig.entry.planned_stop), atr_val
            )
            if sl <= 0:
                sl = float(sig.entry.planned_stop)
            one_r = abs(sig.price - sl)
            tp1 = sig.price + c.tp1_r * one_r if sig.direction == LONG else sig.price - c.tp1_r * one_r
            tp2 = float(sig.entry.planned_target) if sig.entry.planned_target is not None else (
                sig.price + c.tp2_r * one_r if sig.direction == LONG else sig.price - c.tp2_r * one_r
            )
            return sl, tp1, tp2
        atr_val = float(ind.atr(execution_df, c.sl_atr_period).iloc[-1])
        swing_high, swing_low = ind.recent_swing_levels(
            execution_df["high"], execution_df["low"], c.swing_lookback_left, c.swing_lookback_right)
        side = "long" if sig.direction == LONG else "short"
        sl = calc_stop_loss(side, sig.price, atr_val, c.sl_atr_mult, swing_high, swing_low,
                           c.sl_min_pct, c.sl_max_pct, c.sl_tighten_mult)
        tp1, tp2 = calc_take_profits(side, sig.price, sl, c.tp1_r, c.tp2_r)
        return sl, tp1, tp2

    # Events that fully close the position (as opposed to TP1_HIT, which only
    # partially closes and leaves the runner open) — each one starts this
    # symbol's post-close cooldown.
    _TERMINAL_EVENTS = {"TP2_HIT", "SL_HIT", "BE_HIT", EMA_CROSS_REVERSAL,
                        PRICE_OPEN_BEYOND_EMA, "TP1_THEN_EXTERNAL_CLOSE", "SPIKE_GUARD"}

    async def _handle_event(self, event: dict):
        ev = event.get("event")
        symbol = event.get("symbol", "")
        if ev == "TP1_HIT":
            await self.telegram.tp1_hit(symbol, event["price"], event["pnl"], event["new_sl"], ev=event)
        elif ev == "TP2_HIT":
            await self.telegram.tp2_hit(symbol, event["price"], event.get("trade_pnl", event["pnl"]), ev=event)
        elif ev == "SL_HIT":
            await self.telegram.sl_hit(symbol, event["price"], event.get("trade_pnl", event["pnl"]), at_breakeven=False, ev=event)
        elif ev == "BE_HIT":
            await self.telegram.sl_hit(symbol, event["price"], event.get("trade_pnl", event["pnl"]), at_breakeven=True, ev=event)
        elif ev in (EMA_CROSS_REVERSAL, PRICE_OPEN_BEYOND_EMA):
            await self.telegram.early_exit(symbol, event["price"], event.get("trade_pnl", event["pnl"]),
                                           ev, event.get("exit_detail", ""), ev=event)
        elif ev == "SPIKE_GUARD":
            await self.telegram.spike_guard(symbol, event["price"], event.get("trade_pnl", event["pnl"]),
                                            event.get("spike_reason", ""), ev=event)
        elif ev == "TP1_THEN_EXTERNAL_CLOSE":
            # The exchange-side algo closed the FULL position before our TP1
            # partial fired — no separate TP1 leg happened, just note the
            # approximate total pnl for the trade.
            await self.telegram.tp2_hit(symbol, event["price"], event.get("trade_pnl", event["pnl"]), ev=event)
        elif ev == "ERROR":
            await self.telegram.error(symbol, event.get("detail", "unknown error"))

        if ev in self._TERMINAL_EVENTS and symbol:
            if ev == "SL_HIT":
                cooldown_min = getattr(self.cfg, "symbol_sl_cooldown_min", 90)
            elif ev == "BE_HIT":
                cooldown_min = getattr(self.cfg, "symbol_be_cooldown_min", 45)
            else:
                cooldown_min = self.cfg.symbol_cooldown_min
            cooldown_sec = cooldown_min * 60
            self._symbol_cooldown_until[symbol] = time.time() + cooldown_sec
            logger.info("[%s] closed (%s) — cooldown %d min before next entry",
                       symbol, ev, cooldown_min)
            # Trade log for /trades and the live view. A trade is a win only
            # when its final post-fee PnL is positive; TP1 is tracked separately.
            pnl = float(event.get("trade_pnl", event.get("pnl", 0.0)) or 0.0)
            side = event.get("side", "")
            self._trade_log.append({
                "time": time.time(), "symbol": symbol,
                "side": side, "reason": ev,
                "entry": float(event.get("entry_price", 0.0) or 0.0),
                "exit": float(event.get("price", 0.0) or 0.0),
                "tp1_hit": bool(event.get("tp1_hit", False)),
                "pnl": pnl,
            })
            del self._trade_log[:-200]   # keep the last 200 trades
            # Persist the exit bucket for the /stats breakdown (restart-safe).
            # The event fires the moment the close is detected, so this close_ms
            # is within the 3-min match tolerance of OKX's own close time.
            self._journal_add(symbol, side, pnl, self._bucket_for_event(ev, pnl),
                              ev, int(time.time() * 1000))

    async def _check_global_alerts(self):
        now = time.time()
        balance = await self.client.fetch_balance_usdt()
        allowed, reason = self.risk.check_daily_limits(balance, now)
        if not allowed and not self._daily_alert_sent:
            self._daily_alert_sent = True
            pnl_pct = self.risk.state.day_realized_pnl / max(self.risk.state.day_start_balance, 1e-9)
            if "loss" in reason.lower():
                await self.telegram.daily_loss_limit(pnl_pct)
            else:
                await self.telegram.daily_profit_lock(pnl_pct)
        elif allowed:
            self._daily_alert_sent = False

        in_cd = self.risk.is_in_cooldown(now)
        if in_cd and not self._cooldown_alert_sent:
            self._cooldown_alert_sent = True
            await self.telegram.cooldown_activated(
                int(self.risk.cooldown_remaining_sec(now) / 60), self.risk.state.loss_streak)
        elif not in_cd:
            self._cooldown_alert_sent = False

    # ── Telegram command interface ───────────────────────────────────────────

    async def _command_loop(self):
        """Long-poll Telegram for /commands from the configured chat."""
        while self._running:
            try:
                updates = await self.telegram.get_updates(self._tg_offset + 1)
                for u in updates:
                    self._tg_offset = max(self._tg_offset, int(u.get("update_id", 0)))
                    msg = u.get("message") or {}
                    chat_id = str((msg.get("chat") or {}).get("id", ""))
                    text = (msg.get("text") or "").strip()
                    if chat_id != str(self.telegram.chat_id) or not text.startswith("/"):
                        continue
                    try:
                        await self._handle_command(text.split("@")[0].lower())
                    except Exception as ce:
                        logger.warning("[TG-CMD] %s failed: %s", text, ce)
                        await self.telegram.send_text(f"command failed: {ce}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[TG-CMD] loop error: %s", e)
                await asyncio.sleep(5)

    async def _handle_command(self, cmd: str):
        if cmd in ("/help", "/start"):
            await self.telegram.send_text(
                "🤖 *Signal Regime Bias Bot*\n\n"
                "/help — รายการคำสั่ง\n"
                "/stats — สถิติเทรด (จาก OKX order history, post-fee)\n"
                "/restats — reset stats แล้วเริ่มนับใหม่จากตอนนี้\n"
                "/balance — ยอด USDT ปัจจุบัน\n"
                "/positions — position ที่เปิดอยู่\n"
                "/trades — 5 เทรดล่าสุด\n"
                "/status — regime/bias/score ทุก symbol"
            )
        elif cmd == "/balance":
            bal = await self.client.fetch_balance_usdt()
            day = self.risk.state
            day_pnl = day.day_realized_pnl
            await self.telegram.send_text(
                f"💰 *Balance*: `{bal:.2f}` USDT\n"
                f"Today PnL: `{day_pnl:+.2f}` USDT\n"
                f"Open positions: `{self.positions.open_position_count()}`/"
                f"`{self.cfg.max_open_positions}`"
            )
        elif cmd == "/positions":
            lines = []
            for sym in self.cfg.symbols:
                pos = self.positions.get(sym)
                if pos is None:
                    continue
                lines.append(
                    f"`{sym}` *{pos.side.upper()}* @ `{pos.entry_price:.6g}`\n"
                    f"  amt `{pos.amount:.6g}`  SL `{pos.stop_loss:.6g}`"
                    f"  TP1 `{(f'{pos.tp1:.6g}' if pos.tp1 else '-')}`  TP2 `{pos.tp2:.6g}`"
                    f"  TP1hit: {'✅' if pos.tp1_hit else '—'}"
                )
            await self.telegram.send_text(
                "📊 *Open Positions*\n\n" + ("\n".join(lines) if lines else "no open positions"))
        elif cmd == "/trades":
            last = self._trade_log[-5:]
            if not last:
                await self.telegram.send_text("no closed trades yet")
                return
            lines = []
            for t in reversed(last):
                ts = time.strftime("%m-%d %H:%M", time.gmtime(t["time"]))
                win = t["pnl"] > 0
                lines.append(
                    f"{'🟢' if win else '🔴'} `{t['symbol']}` {t['side'].upper()} "
                    f"{t['reason']}  pnl `{t['pnl']:+.2f}`  {ts}")
            await self.telegram.send_text("🧾 *Last 5 Trades*\n\n" + "\n".join(lines))
        elif cmd == "/stats":
            # plain text (no markdown) — separators + emoji, so symbols/numbers
            # never get mangled by Telegram's Markdown parser. Same as HTF.
            await self.telegram._send_message(await self._build_stats_report(), _markdown=False)
        elif cmd == "/restats":
            now_ms = int(time.time() * 1000)
            self._stats_reset_ms = now_ms
            _save_stats_reset_ms(self.cfg.state_dir, now_ms)
            self._trade_log = []   # this process's own log; OKX history is untouched
            now_lbl = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now_ms / 1000))
            await self.telegram.send_text(
                f"🔄 *Stats reset*\n\n/stats now counts trades closed on/after `{now_lbl}` only."
            )
        elif cmd == "/status":
            lines = []
            now = time.time()
            for sym in self.cfg.symbols:
                sig = self._last_signal_by_symbol.get(sym)
                pos = self.positions.get(sym)
                pos_label = f"OPEN {pos.side.upper()}" if pos else "flat"
                if sig is None:
                    lines.append(f"`{sym}` {pos_label} — no data yet")
                    continue
                cd = self._symbol_cooldown_until.get(sym, 0)
                cd_lb = f" cd={max(0,(cd-now))/60:.0f}m" if cd > now else ""
                if sig.bias is not None:
                    b = sig.bias
                    bias_str = f"`{b.direction}` 1H`{b.score_1h:.0f}` 15M`{b.score_15m:.0f}` 5M`{b.score_5m:.0f}`"
                else:
                    bias_str = "`—`"
                entry_str = (
                    f"`{sig.entry.setup_type or 'WAIT'} score={_entry_score_text(sig.entry)} "
                    f"EMA8/13={sig.entry.ema_fast:.4f}/{sig.entry.ema_slow:.4f}`"
                    if sig.entry is not None else "`-`"
                )
                lines.append(
                    f"`{sym}` {pos_label}\n"
                    f"  regime `{sig.regime.label}`\n"
                    f"  bias {bias_str}\n"
                    f"  entry {entry_str} dir `{sig.direction}`{cd_lb}")
            await self.telegram.send_text("📡 *Status*\n\n" + "\n".join(lines))
        else:
            await self.telegram.send_text(f"unknown command: {cmd} — try /help")

    @staticmethod
    def _month_bounds(now_ms: int) -> tuple:
        """(this_month_start_ms, prev_month_start_ms, prev_month_label) in UTC."""
        import datetime as dt
        now = dt.datetime.fromtimestamp(now_ms / 1000, tz=dt.timezone.utc)
        m0 = dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
        py, pm = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        p0 = dt.datetime(py, pm, 1, tzinfo=dt.timezone.utc)
        return int(m0.timestamp() * 1000), int(p0.timestamp() * 1000), p0.strftime("%b")

    def _match_exit_types(self, okx_rows: list) -> dict:
        """One-to-one match OKX-closed trades to journal entries by nearest
        close time (same symbol, within 3 min), each journal entry consumed at
        most once — so two same-symbol trades minutes apart can't both borrow
        one journal row. Returns {id(row): exit_type} for matched rows."""
        pool = list(self.journal)
        used = [False] * len(pool)
        out = {}
        for row in sorted(okx_rows, key=lambda r: r.get("close_time_ms", 0)):
            cms = row.get("close_time_ms", 0)
            best_j, best_d = -1, 3 * 60_000 + 1
            for j, e in enumerate(pool):
                if used[j] or e["symbol"] != row["symbol"]:
                    continue
                d = abs(int(e["close_ms"]) - cms)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                used[best_j] = True
                out[id(row)] = pool[best_j]["exit_type"]
        return out

    async def _build_stats_report(self) -> str:
        """Three sections, plain text — identical layout to the HTF bot:
        OVERALL (current UTC month, resets on the 1st) with TP/BE/SL/Untracked
        broken out — counts/WR/PnL from OKX, breakdown from the local journal,
        denominators = OKX total (never the journal's); BY SYMBOL (all-time);
        LAST 5. Falls back to the journal with a warning if OKX is unreachable.
        The regime bot is a 2-TP system, so TP here means "hit a profit target"
        (TP2, or a net-positive TP1-then-exit); the raw reason stays journaled."""
        since = (self._stats_reset_ms if self._stats_reset_ms is not None
                 else self.cfg.stats_since_ms())
        now_ms = int(time.time() * 1000)
        m0, p0, prev_lbl = self._month_bounds(now_ms)

        okx_ok = True
        rows = []
        if not self.cfg.paper:
            try:
                rows = await self.client.fetch_trade_history(since, self.cfg.symbols)
            except Exception as e:
                logger.warning("[STATS] OKX history fetch failed: %s", e)
                okx_ok = False
        if not rows and (self.cfg.paper or not okx_ok):
            rows = [{"symbol": e["symbol"], "side": e.get("side", ""), "pnl": e["pnl"],
                     "close_time_ms": e["close_ms"], "_journal": True}
                    for e in self.journal if e["close_ms"] >= since]

        balance = await self.client.fetch_balance_usdt()
        open_lines = [f"📌 {_sym(sym)} {pos.side.upper()} @ {pos.entry_price:.6g}"
                      for sym in self.cfg.symbols if (pos := self.positions.get(sym)) is not None]
        header = (f"💰 Balance: ${balance:.2f}\n"
                  + ("\n".join(open_lines) if open_lines else "📌 No open positions"))
        if not okx_ok:
            header = "⚠️ OKX history unavailable — showing local journal\n" + header
        sep = "――――――――――――――――――"
        if not rows:
            since_lbl = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(since / 1000))
            return header + f"\n\n(no closed trades since {since_lbl})"

        # ── ช่อง 1: OVERALL — current month only ──
        month = [r for r in rows if r.get("close_time_ms", 0) >= m0]
        total = len(month)
        wins = sum(1 for r in month if r["pnl"] > 0)
        net = sum(r["pnl"] for r in month)
        prev_net = sum(r["pnl"] for r in rows if p0 <= r.get("close_time_ms", 0) < m0)
        cnt = {"TP": 0, "BE": 0, "SL": 0}
        matched = self._match_exit_types([r for r in month if not r.get("_journal")])
        tracked = 0
        for r in month:
            et = matched.get(id(r))
            if et in cnt:
                cnt[et] += 1
                tracked += 1
        untracked = total - tracked

        def pct(n):
            return f"{n}/{total} ({n / total * 100:.0f}%)" if total else "0/0"
        lines = [header, "", sep, "OVERALL (this month, OKX)", sep,
                 f"Trades   : {total}  ({wins}W / {total - wins}L)",
                 f"Win rate : {wins / total * 100:.0f}%" if total else "Win rate : —",
                 f"TP hit   : {pct(cnt['TP'])}   BE : {pct(cnt['BE'])}   SL : {pct(cnt['SL'])}"]
        if untracked:
            lines.append(f"Untracked: {untracked}/{total} (closed while bot offline)")
        lines.append(f"Net PnL  : ${net:+.2f}  (post-fee, from OKX)")
        lines.append(f"{prev_lbl} PnL  : ${prev_net:+.2f}")

        # ── ช่อง 2: BY SYMBOL — all-time ──
        lines += ["", sep, "BY SYMBOL (all-time)", sep]
        by = {}
        for r in rows:
            by.setdefault(r["symbol"], []).append(r["pnl"])
        for s in self.cfg.symbols:
            ps = by.get(s, [])
            if not ps:
                lines.append(f"{_sym(s):5s} 0 trades")
                continue
            w = sum(1 for p in ps if p > 0)
            lines.append(f"{_sym(s):5s} {len(ps)} trades  {w / len(ps) * 100:.0f}%WR  ${sum(ps):+.2f}")
        allp = [p for ps in by.values() for p in ps]
        if allp:
            wa = sum(1 for p in allp if p > 0)
            lines += ["――――――――",
                      f"TOTAL {len(allp)} trades  {wa / len(allp) * 100:.0f}%WR  ${sum(allp):+.2f}"]

        # ── ช่อง 3: LAST 5 TRADES ──
        lines += ["", sep, "LAST 5 TRADES", sep]
        now = time.time()
        for i, r in enumerate(sorted(rows, key=lambda x: -x.get("close_time_ms", 0))[:5], 1):
            age = now - r.get("close_time_ms", now_ms) / 1000
            age_lbl = f"{age / 3600:.1f}h ago" if age < 86400 else f"{age / 86400:.1f}d ago"
            e = "✅" if r["pnl"] > 0 else "❌"
            side = (r.get("side") or "").upper()
            lines.append(f"{i}. {e} {_sym(r['symbol'])} {side} ${r['pnl']:+.2f} — {age_lbl}")
        return "\n".join(lines)

    async def _maybe_log_status(self):
        """Per-symbol regime/bias/entry snapshot, every status_log_interval_sec
        (default 5 min) — uses the signal already computed this tick in
        _process_symbol (self._last_signal_by_symbol), no extra API calls."""
        now = time.time()
        if now - self._last_status_log_ts < self.cfg.status_log_interval_sec:
            return
        self._last_status_log_ts = now

        now_wall = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
        logger.info("=== STATUS [%s UTC] ===", now_wall)
        for symbol in self.cfg.symbols:
            sig = self._last_signal_by_symbol.get(symbol)
            pos = self.positions.get(symbol)
            pos_label = f"OPEN {pos.side.upper()} @ {pos.entry_price:.6g}" if pos else "flat"
            if sig is None:
                logger.info("  %-16s %-24s no data yet", symbol, pos_label)
                continue
            cd_until = self._symbol_cooldown_until.get(symbol, 0)
            cd_label = f" cooldown={max(0,(cd_until-now))/60:.0f}m" if cd_until > now else ""
            # sig.bias/sig.entry are None only when an earlier layer's own
            # dataclass wasn't reached (never happens in this pipeline since
            # Regime always classifies and Bias always runs) — guard anyway.
            if sig.bias is not None:
                bias_label = f"{sig.bias.direction}(1H={sig.bias.score_1h:.0f},15M={sig.bias.score_15m:.0f},5M={sig.bias.score_5m:.0f})"
            else:
                bias_label = "—"
            entry_label = (
                f"{sig.entry.setup_type or 'WAIT'} score={_entry_score_text(sig.entry)} "
                f"EMA8/13={sig.entry.ema_fast:.4f}/{sig.entry.ema_slow:.4f}"
                if sig.entry is not None else "-"
            )
            blk = f" blocked={sig.blocked_layer}" if sig.blocked_layer else ""
            logger.info(
                "  %-16s %-24s regime=%-20s bias=%-40s entry=%-5s dir=%s%s%s",
                symbol, pos_label, sig.regime.label, bias_label, entry_label,
                sig.direction, cd_label, blk,
            )


async def _main():
    bot = Bot()
    await bot.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig_name, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    run_task = asyncio.create_task(bot.run_forever())
    await stop_event.wait()
    logger.info("Shutdown signal received...")
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
