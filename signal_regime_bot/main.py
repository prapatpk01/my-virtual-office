"""
Signal Regime Bias Strategy — live bot entry point.

Loop: every `poll_interval_sec`, for each symbol —
  1. fetch 30m/1h/4h closed bars (skip symbol if any TF has < min_bars)
  2. if a position is open: check SL/TP1/TP2 against the live ticker every
     tick; run the 3-bar-confirm health monitor once per newly-closed 30m bar
  3. else: once per newly-closed 30m bar, evaluate SignalEngine and open a
     position if regime/bias/entry all clear their thresholds and risk
     manager allows a new entry (no cooldown, no daily limit breach)

Every branch that mutates state is wrapped so one symbol's exception can
never kill the loop or leave an order half-placed with silent failure.
"""
from __future__ import annotations

import asyncio
import logging
import signal
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


class Bot:
    def __init__(self):
        self.cfg = load_config()
        self.client = ExchangeClient(
            api_key=self.cfg.okx_api_key, api_secret=self.cfg.okx_secret,
            passphrase=self.cfg.okx_passphrase, paper=self.cfg.paper,
            leverage=self.cfg.leverage, margin_mode=self.cfg.margin_mode,
        )
        self.data = DataEngine(self.cfg, self.client)
        self.signal_engine = Pipeline(self.cfg)
        self.risk = RiskManager(self.cfg)
        self.positions = PositionManager(self.cfg, self.client, self.risk)
        self.telegram = TelegramNotifier(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id)

        self._last_entry_bar: dict[str, pd.Timestamp] = {}
        self._last_signal_by_symbol: dict[str, object] = {}
        self._symbol_cooldown_until: dict[str, float] = {}
        self._daily_alert_sent = False
        self._cooldown_alert_sent = False
        self._last_status_log_ts = 0.0
        self._trade_log: list[dict] = []      # closed trades (for /stats, /trades)
        self._cmd_task = None                 # Telegram command polling task
        self._tg_offset = 0
        self._running = False

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
        adopted = await self.positions.reconcile_with_exchange(self.cfg.symbols)
        if adopted:
            logger.warning("[STARTUP] Adopted %d orphaned position(s) from a prior run: %s",
                          len(adopted), adopted)
            # TP1 status is inferred per-position from the attached SL (at
            # entry = already hit, still a real distance away = recovered) —
            # see reconcile_with_exchange for the detail; this alert just
            # confirms adoption happened, the Railway log has the per-symbol mode.
            await self.telegram.error(
                "STARTUP",
                f"Adopted {len(adopted)} orphaned position(s) from a prior run: "
                f"{', '.join(adopted)}. TP1 status inferred from the attached "
                f"stop (already hit -> single-TP mode; still wide -> TP1 recovered)."
            )

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
                f"Architecture: Regime → Bias → Entry (strict 3-layer)"
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
            for symbol in self.cfg.symbols:
                try:
                    await self._process_symbol(symbol)
                except Exception as e:
                    logger.error("[%s] unhandled error: %s", symbol, e, exc_info=True)
                    await self.telegram.error(symbol, str(e))
            await self._check_global_alerts()
            await self._maybe_log_status()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    # ── Per-symbol processing ────────────────────────────────────────────────

    async def _process_symbol(self, symbol: str):
        self.data.new_tick()
        frames = await self.data.fetch_all(symbol)
        if not self.data.has_min_bars(frames):
            logger.debug("[%s] insufficient bars (<%d) on one or more TFs — skip",
                        symbol, self.cfg.min_bars)
            return

        df_30m = frames[self.cfg.tf_entry]
        df_1h = frames[self.cfg.tf_bias]
        df_4h = frames[self.cfg.tf_regime]
        df_15m = frames[self.cfg.tf_fast]
        df_5m = frames[self.cfg.tf_micro]

        # Full pipeline computed once per symbol per tick and cached — reused by
        # the entry check below AND by the 5-minute status log, so the health
        # branch and the entry branch never re-run the layers separately.
        sig = self.signal_engine.evaluate(df_30m, df_1h, df_4h, df_15m, df_5m)
        self._last_signal_by_symbol[symbol] = sig

        if self.positions.has_position(symbol):
            await self._manage_open_position(symbol, df_30m, df_1h, df_4h, df_15m, df_5m)
        else:
            await self._look_for_entry(symbol, df_30m, sig)

    async def _manage_open_position(self, symbol: str, df_30m, df_1h, df_4h, df_15m, df_5m):
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
            return   # position fully or partially closed this tick — health check waits for next

        # SpikeGuard — EVERY tick, 5m/15m closed bars + live price. The fast
        # layer: force-close before a V-reversal eats the full SL (+slippage).
        if self.cfg.spike_guard_enabled:
            spike_event = await self._check_spike_guard(symbol, pos, price)
            if spike_event:
                await self._handle_event(spike_event)
                return

        # Health monitor — once per newly-closed 30m bar only (the slow layer).
        bar_ts = df_30m.index[-1] if len(df_30m) else None
        if bar_ts is not None and pos.last_health_bar_ts != bar_ts:
            regime = self.signal_engine.regime_engine.analyze(df_4h, df_1h)
            bias = self.signal_engine.bias_engine.analyze(df_1h, df_15m, df_5m, regime.label)
            hevent = await self.positions.process_closed_bar_health(symbol, df_30m, regime, bias)
            if hevent:
                await self._handle_event(hevent)

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

    async def _look_for_entry(self, symbol: str, df_30m, sig):
        bar_ts = df_30m.index[-1] if len(df_30m) else None
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
        chart_path = build_entry_chart(
            symbol, df_30m, sig.direction, sig.price,
            *self._preview_sl_tp(sig, df_30m),
        )

        pos = await self.positions.open_position(
            symbol, sig.direction, sig.price, df_30m, sig.regime, sig.bias, sig.entry_score)
        if pos is None:
            return

        await self.telegram.entry_signal(
            symbol, sig.direction, pos.entry_price, pos.stop_loss, pos.tp1, pos.tp2,
            sig.regime, sig.bias, sig.entry_score, risk_pct, self.cfg.leverage, chart_path)
        await self.telegram.order_opened(
            symbol, sig.direction, pos.entry_price, pos.amount, pos.stop_loss, pos.tp1, pos.tp2)

    def _log_pipeline_block(self, symbol: str, sig):
        """Explain WHICH layer stopped the trade — never a generic 'no signal'."""
        r = sig.regime
        layer = sig.blocked_layer
        if layer == "BIAS" and sig.bias is not None:
            b = sig.bias
            logger.info("[%s] regime=%s(4H=%s,1H=%s)  BIAS NO-TRADE  1H=%.0f 15M=%.0f 5M=%.0f — %s",
                       symbol, r.label, r.label_4h, r.label_1h,
                       b.score_1h, b.score_15m, b.score_5m, b.reason)
        elif layer == "ENTRY" and sig.entry is not None:
            e = sig.entry
            logger.info("[%s] regime=%s dir=%s(bias)  ENTRY NOT READY  %d/5 categories — %s",
                       symbol, r.label, sig.bias.direction if sig.bias else "-",
                       e.passed_count, e.reason)
        elif layer == "CONFIRM":
            logger.info("[%s] regime=%s dir=%s(bias)  AWAITING 5M/15M CONFIRMATION — %s",
                       symbol, r.label, sig.bias.direction if sig.bias else "-", sig.reason)
        else:
            logger.debug("[%s] no trade: %s", symbol, sig.reason)

    def _preview_sl_tp(self, sig, df_30m) -> tuple[float, float, float]:
        """Compute SL/TP1/TP2 for the chart BEFORE the order is placed (same math as open_position)."""
        import indicators as ind
        from position_manager import calc_stop_loss, calc_take_profits
        c = self.cfg
        atr_val = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        swing_high, swing_low = ind.recent_swing_levels(
            df_30m["high"], df_30m["low"], c.swing_lookback_left, c.swing_lookback_right)
        side = "long" if sig.direction == LONG else "short"
        sl = calc_stop_loss(side, sig.price, atr_val, c.sl_atr_mult, swing_high, swing_low,
                           c.sl_min_pct, c.sl_max_pct, c.sl_tighten_mult)
        tp1, tp2 = calc_take_profits(side, sig.price, sl, c.tp1_r, c.tp2_r)
        return sl, tp1, tp2

    # Events that fully close the position (as opposed to TP1_HIT, which only
    # partially closes and leaves the runner open) — each one starts this
    # symbol's post-close cooldown.
    _TERMINAL_EVENTS = {"TP2_HIT", "SL_HIT", "BE_HIT", "HEALTH_CLOSE",
                        "TP1_THEN_EXTERNAL_CLOSE", "SPIKE_GUARD"}

    async def _handle_event(self, event: dict):
        ev = event.get("event")
        symbol = event.get("symbol", "")
        if ev == "TP1_HIT":
            await self.telegram.tp1_hit(symbol, event["price"], event["pnl"], event["new_sl"])
        elif ev == "TP2_HIT":
            await self.telegram.tp2_hit(symbol, event["price"], event["pnl"])
        elif ev == "SL_HIT":
            await self.telegram.sl_hit(symbol, event["price"], event["pnl"], at_breakeven=False)
        elif ev == "BE_HIT":
            await self.telegram.sl_hit(symbol, event["price"], event["pnl"], at_breakeven=True)
        elif ev == "HEALTH_CLOSE":
            await self.telegram.health_close(symbol, event["price"], event["pnl"],
                                             event.get("health_score", 0.0), event.get("weak_count", 0))
        elif ev == "SPIKE_GUARD":
            await self.telegram.spike_guard(symbol, event["price"], event["pnl"],
                                            event.get("spike_reason", ""))
        elif ev == "TP1_THEN_EXTERNAL_CLOSE":
            # The exchange-side algo closed the FULL position before our TP1
            # partial fired — no separate TP1 leg happened, just note the
            # approximate total pnl for the trade.
            await self.telegram.tp2_hit(symbol, event["price"], event["pnl"])
        elif ev == "ERROR":
            await self.telegram.error(symbol, event.get("detail", "unknown error"))
        # HEALTH_OK / HEALTH_WEAK (not yet confirmed) are logged only, not alerted —
        # avoids spamming Telegram every 30m bar while a position rides normally.

        if ev in self._TERMINAL_EVENTS and symbol:
            cooldown_sec = self.cfg.symbol_cooldown_min * 60
            self._symbol_cooldown_until[symbol] = time.time() + cooldown_sec
            logger.info("[%s] closed (%s) — cooldown %d min before next entry",
                       symbol, ev, self.cfg.symbol_cooldown_min)
            # Trade log for /stats and /trades. trade_pnl = FULL trade total
            # (TP1 partial leg + final leg), tp1_hit drives the win rule:
            # a TP1-then-breakeven exit counts as a WIN.
            self._trade_log.append({
                "time": time.time(), "symbol": symbol,
                "side": event.get("side", ""), "reason": ev,
                "entry": float(event.get("entry_price", 0.0) or 0.0),
                "exit": float(event.get("price", 0.0) or 0.0),
                "tp1_hit": bool(event.get("tp1_hit", False)),
                "pnl": float(event.get("trade_pnl", event.get("pnl", 0.0)) or 0.0),
            })
            del self._trade_log[:-200]   # keep the last 200 trades

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
                "/stats — สถิติเทรด (winrate, TP1/TP2 rate)\n"
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
                win = t["pnl"] > 0 or t["tp1_hit"]
                lines.append(
                    f"{'🟢' if win else '🔴'} `{t['symbol']}` {t['side'].upper()} "
                    f"{t['reason']}  pnl `{t['pnl']:+.2f}`  {ts}")
            await self.telegram.send_text("🧾 *Last 5 Trades*\n\n" + "\n".join(lines))
        elif cmd == "/stats":
            await self.telegram.send_text(self._stats_text())
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
                entry_str = f"`{sig.entry.passed_count}/5`" if sig.entry is not None else "`-`"
                lines.append(
                    f"`{sym}` {pos_label}\n"
                    f"  regime `{sig.regime.label}`\n"
                    f"  bias {bias_str}\n"
                    f"  entry {entry_str} dir `{sig.direction}`{cd_lb}")
            await self.telegram.send_text("📡 *Status*\n\n" + "\n".join(lines))
        else:
            await self.telegram.send_text(f"unknown command: {cmd} — try /help")

    def _stats_text(self) -> str:
        trades = self._trade_log
        total = len(trades)
        if total == 0:
            return "📈 *Stats*\n\nno closed trades yet"
        # Win rule (per spec): TP1 hit then stopped at breakeven still counts
        # as a WIN — the trade banked the TP1 partial.
        wins = [t for t in trades if t["pnl"] > 0 or t["tp1_hit"]]
        losses = total - len(wins)
        tp1 = [t for t in trades if t["tp1_hit"]]
        tp2 = [t for t in trades if t["reason"] == "TP2_HIT"]
        net = sum(t["pnl"] for t in trades)
        winrate = len(wins) / total * 100
        tp1_of_wins = (len(tp1) / len(wins) * 100) if wins else 0.0
        tp2_of_total = len(tp2) / total * 100
        return (
            "📈 *Stats*\n\n"
            f"Trades: `{total}`  Win: `{len(wins)}`  Loss: `{losses}`\n"
            f"Winrate: `{winrate:.0f}%` _(TP1→SL counts as win)_\n"
            f"TP1 hit: `{len(tp1)}` — `{tp1_of_wins:.0f}%` of wins "
            f"(win {len(wins)} tp1 {len(tp1)}={tp1_of_wins:.0f}%)\n"
            f"TP2 hit: `{len(tp2)}` — `{tp2_of_total:.0f}%` of trades\n"
            f"Net PnL: `{net:+.2f}` USDT"
        )

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
            entry_label = f"{sig.entry.passed_count}/5" if sig.entry is not None else "-"
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
