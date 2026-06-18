"""
Telegram Bot notifier + command handler for the Trading Bot.

Alerts sent:
  - Bot started / stopped
  - BUY / SELL signals (with confidence)
  - Order executed
  - Stop-loss / Take-profit triggered
  - Max drawdown halt

Commands received (polling loop):
  /status    — current bot state & balance
  /positions — open positions
  /trades    — last 5 trades
  /balance   — balance & P&L
  /start_bot — start the trading bot
  /stop_bot  — stop the trading bot
  /help      — command list
"""
import asyncio
import json
import logging
import time
import threading
from typing import Callable, Optional
import aiohttp

logger = logging.getLogger("telegram_notifier")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        get_state_fn: Optional[Callable] = None,
        get_stats_fn: Optional[Callable] = None,
        start_bot_fn: Optional[Callable] = None,
        stop_bot_fn: Optional[Callable] = None,
        backtest_fn: Optional[Callable] = None,
        min_confidence: float = 0.5,
    ):
        self.token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.get_state_fn = get_state_fn
        self.get_stats_fn = get_stats_fn
        self.backtest_fn  = backtest_fn
        self.start_bot_fn = start_bot_fn
        self.stop_bot_fn = stop_bot_fn
        self.min_confidence = min_confidence

        self._last_update_id = 0
        self._polling_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._enabled = bool(token and chat_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_polling(self, loop: asyncio.AbstractEventLoop):
        """Start the Telegram command polling loop on the given event loop."""
        if not self._enabled or self._polling_task:   # prevent double-start
            return
        self._loop = loop
        self._polling_task = loop.create_task(self._poll_loop())
        logger.info("Telegram polling started (chat_id=%s)", self.chat_id)

    def stop_polling(self):
        if self._polling_task:
            self._polling_task.cancel()

    # ------------------------------------------------------------------
    # Notification helpers (sync wrappers — safe to call from any thread)
    # ------------------------------------------------------------------

    def notify(self, text: str):
        """Fire-and-forget send from any thread."""
        if not self._enabled:
            return
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send(text), self._loop)
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._send(text))
                else:
                    loop.run_until_complete(self._send(text))
            except Exception:
                pass

    @staticmethod
    def _md_escape(s: str) -> str:
        """Escape characters that break Telegram Markdown v1 entity parsing."""
        return s.replace("[", "\\[").replace("]", "\\]").replace("_", "\\_").replace("*", "\\*")

    def notify_signal(self, signal_dict: dict):
        sig_type = signal_dict.get("type", "hold")
        if sig_type == "hold":
            return
        conf = signal_dict.get("confidence", 0)
        if conf < self.min_confidence:
            return
        emoji  = "🟢" if sig_type == "buy" else "🔴"
        sym    = signal_dict.get("symbol", "")
        strat  = signal_dict.get("strategy", "")
        price  = signal_dict.get("price", 0)
        reason = self._md_escape(signal_dict.get("reason", ""))
        meta   = signal_dict.get("metadata", {})

        sl  = meta.get("stop_loss")
        tp  = meta.get("take_profit")
        rr  = meta.get("rr")

        sl_tp_line = ""
        if sl and tp:
            rr_line = f"\n📐 R:R `1:{rr:.1f}`" if rr else ""
            sl_tp_line = (
                f"\n🛑 SL: `{sl:,.4f}`"
                f"\n🎯 TP: `{tp:,.4f}`"
                f"{rr_line}"
            )
        text = (
            f"{emoji} *{sig_type.upper()} Signal*\n"
            f"`{sym}` @ `{price:,.4f}`\n"
            f"Strategy: {strat} | Conf: {conf*100:.0f}%\n"
            f"_{reason}_"
            f"{sl_tp_line}"
        )
        self.notify(text)

    def notify_order(self, symbol: str, side: str, amount: float,
                     price: float, strategy: str, paper: bool,
                     sl: float = None, tp: float = None):
        emoji = "✅" if side == "buy" else "🏁"
        mode = "📄 PAPER" if paper else "💰 LIVE"
        sl_tp_line = ""
        if sl and tp:
            sl_tp_line = f"\n🛑 SL: `{sl:,.4f}` | 🎯 TP: `{tp:,.4f}`"
        text = (
            f"{emoji} *Order Executed* {mode}\n"
            f"`{symbol}` — *{side.upper()}*\n"
            f"Amount: `{amount}` @ `{price:,.4f}`\n"
            f"Strategy: {strategy}"
            f"{sl_tp_line}"
        )
        self.notify(text)

    def notify_trade_closed(self, symbol: str, reason: str, exit_price: float,
                            entry: float, sl, tp, stats: dict):
        """Called when a position closes via SL or TP. Full detail + running stats."""
        won   = reason == "take_profit"
        emoji = "✅" if won else "❌"
        label = "Take-Profit Hit" if won else "Stop-Loss Hit"
        risk  = abs(entry - sl) if sl else 1.0
        pnl_r = abs(exit_price - entry) / risk if won else -1.0
        sign  = "+" if pnl_r >= 0 else ""

        sl_str = f"`{sl:,.4f}`" if sl else "—"
        tp_str = f"`{tp:,.4f}`" if tp else "—"

        total   = stats.get("trades",    0)
        wins    = stats.get("wins",      0)
        losses  = stats.get("losses",    0)
        wr      = stats.get("win_rate",  0)
        pf      = stats.get("profit_factor", 0)
        total_r = stats.get("total_r",   0)
        streak  = stats.get("streak",    0)
        sig_day = stats.get("signals_per_day", 0)
        streak_str = (f"W{streak}" if streak > 0 else f"L{abs(streak)}") if streak else "—"
        sign_r = "+" if total_r >= 0 else ""

        text = (
            f"{emoji} *{label}*\n"
            f"`{symbol}`\n"
            f"Entry: `{entry:,.4f}` → Exit: `{exit_price:,.4f}`\n"
            f"SL: {sl_str} | TP: {tp_str}\n"
            f"Result: `{sign}{pnl_r:.1f}R`\n\n"
            f"📊 *Backtest Stats* ({total} trades)\n"
            f"Win/Loss: `{wins}W / {losses}L` | WR: `{wr:.1f}%`\n"
            f"Profit Factor: `{pf:.2f}` | Total: `{sign_r}{total_r:.1f}R`\n"
            f"Streak: `{streak_str}` | Avg signals/day: `{sig_day}`"
        )
        self.notify(text)

    def notify_virtual_closed(self, symbol: str, reason: str, exit_price: float, stats: dict):
        """Called when a virtual (signal-only / paper-0-balance) trade hits SL or TP."""
        won   = reason == "take_profit"
        emoji = "✅" if won else "❌"
        label = "Take-Profit Hit" if won else "Stop-Loss Hit"

        total   = stats.get("trades",           0)
        wins    = stats.get("wins",             0)
        losses  = stats.get("losses",           0)
        wr      = stats.get("win_rate",         0)
        pf      = stats.get("profit_factor",    0)
        total_r = stats.get("total_r",          0)
        streak  = stats.get("streak",           0)
        sig_day = stats.get("signals_per_day",  0)
        streak_str = (f"W{streak}" if streak > 0 else f"L{abs(streak)}") if streak else "—"
        sign_r = "+" if total_r >= 0 else ""

        text = (
            f"{emoji} *{label}* _(virtual)_\n"
            f"`{symbol}` @ `{exit_price:,.4f}`\n\n"
            f"📊 *Signal Stats* ({total} tracked)\n"
            f"Win/Loss: `{wins}W / {losses}L` | WR: `{wr:.1f}%`\n"
            f"Profit Factor: `{pf:.2f}` | Total: `{sign_r}{total_r:.1f}R`\n"
            f"Streak: `{streak_str}` | Avg signals/day: `{sig_day}`"
        )
        self.notify(text)

    def notify_stop_event(self, symbol: str, event: str, price: float, pnl: float):
        """Legacy — kept for compatibility."""
        emoji = "🛑" if event == "stop_loss" else "💰"
        label = "Stop-Loss Hit" if event == "stop_loss" else "Take-Profit Reached"
        sign = "+" if pnl >= 0 else ""
        text = (
            f"{emoji} *{label}*\n"
            f"`{symbol}` closed @ `{price:,.4f}`\n"
            f"P&L: `{sign}{pnl:,.2f} USD`"
        )
        self.notify(text)

    def notify_drawdown_halt(self, balance: float, peak: float):
        dd = (peak - balance) / peak * 100
        text = (
            f"⚠️ *Max Drawdown Reached — Trading Halted*\n"
            f"Balance: `${balance:,.2f}` (peak `${peak:,.2f}`)\n"
            f"Drawdown: `{dd:.1f}%`\n"
            f"Use /start\\_bot to resume after reviewing."
        )
        self.notify(text)

    def notify_bot_started(self, paper: bool, strategies: list[str], symbols: list[str]):
        mode = "📄 Paper" if paper else "💰 Live"
        text = (
            f"🤖 *Trading Bot Started* — {mode}\n"
            f"Symbols: `{'  '.join(symbols)}`\n"
            f"Strategies: `{'  '.join(strategies)}`"
        )
        self.notify(text)

    def notify_bot_stopped(self):
        self.notify("⏹ *Trading Bot Stopped*")

    # ------------------------------------------------------------------
    # Telegram HTTP helpers
    # ------------------------------------------------------------------

    async def _send(self, text: str, parse_mode: str = "Markdown") -> bool:
        url = TELEGRAM_API.format(token=self.token, method="sendMessage")
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 400 and parse_mode:
                        # Markdown parse error — retry as plain text
                        plain = {"chat_id": self.chat_id, "text": text}
                        async with session.post(url, json=plain,
                                                timeout=aiohttp.ClientTimeout(total=10)) as r2:
                            if r2.status != 200:
                                body = await r2.text()
                                logger.warning("Telegram send (plain) failed %s: %s", r2.status, body[:200])
                                return False
                            return True
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("Telegram send failed %s: %s", r.status, body[:200])
                        return False
                    return True
        except Exception as e:
            logger.warning("Telegram send error: %s", e)
            return False

    async def _get_updates(self) -> list:
        url = TELEGRAM_API.format(token=self.token, method="getUpdates")
        params = {"offset": self._last_update_id + 1, "timeout": 20, "allowed_updates": ["message"]}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status != 200:
                        return []
                    data = await r.json()
                    return data.get("result", [])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Telegram getUpdates error: %s", e)
            return []

    # ------------------------------------------------------------------
    # Command polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self):
        logger.info("Telegram command polling started")
        while True:
            try:
                updates = await self._get_updates()
                for update in updates:
                    self._last_update_id = max(self._last_update_id, update["update_id"])
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    # Only respond to our own chat_id for security
                    chat = str(msg.get("chat", {}).get("id", ""))
                    if chat != self.chat_id:
                        continue
                    if text:
                        await self._handle_command(text)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Poll loop error: %s", e)
                await asyncio.sleep(5)

    async def _handle_command(self, text: str):
        cmd = text.split()[0].lower().lstrip("/")
        if cmd == "help":
            await self._send(
                "📋 *Trading Bot Commands*\n\n"
                "/status — current state & balance\n"
                "/positions — open positions\n"
                "/trades — last 5 trades\n"
                "/balance — balance & P\\&L\n"
                "/stats — win rate & signal statistics\n"
                "/backtest — run historical backtest (takes 1-2 min)\n"
                "/start\\_bot — start the bot\n"
                "/stop\\_bot — stop the bot\n"
                "/help — this message"
            )

        elif cmd == "backtest":
            if not self.backtest_fn:
                await self._send("⚠️ Backtest not configured")
                return
            await self._send("⏳ Running backtest... (1-2 minutes)")
            try:
                result_text = await self.backtest_fn()
                await self._send(result_text)
            except Exception as e:
                await self._send(f"❌ Backtest failed: {e}")

        elif cmd == "status":
            state = self.get_state_fn() if self.get_state_fn else {}
            running = state.get("running", False)
            paper = state.get("paper", True)
            balance = state.get("balance", 0)
            pnl = state.get("pnl_total", 0)
            positions = len(state.get("positions", []))
            mode = "📄 Paper" if paper else "💰 Live"
            status = "🟢 Running" if running else "⏹ Stopped"
            sign = "+" if pnl >= 0 else ""
            await self._send(
                f"📊 *Bot Status*\n"
                f"Status: {status} | {mode}\n"
                f"Balance: `${balance:,.2f}`\n"
                f"P&L: `{sign}${pnl:,.2f}`\n"
                f"Open positions: `{positions}`"
            )

        elif cmd == "balance":
            state = self.get_state_fn() if self.get_state_fn else {}
            balance = state.get("balance", 0)
            equity = state.get("equity", 0)
            pnl = state.get("pnl_total", 0)
            sign = "+" if pnl >= 0 else ""
            await self._send(
                f"💳 *Balance*\n"
                f"Cash: `${balance:,.2f}`\n"
                f"Equity: `${equity:,.2f}`\n"
                f"P&L: `{sign}${pnl:,.2f}`"
            )

        elif cmd == "positions":
            state = self.get_state_fn() if self.get_state_fn else {}
            positions = state.get("positions", [])
            if not positions:
                await self._send("📭 No open positions")
                return
            lines = ["📌 *Open Positions*\n"]
            for p in positions:
                lines.append(
                    f"`{p['symbol']}` {p['side']}\n"
                    f"  Entry: `{p['entry']:,.4f}` | SL: `{p.get('stop_loss','—')}` | TP: `{p.get('take_profit','—')}`"
                )
            await self._send("\n".join(lines))

        elif cmd == "trades":
            state = self.get_state_fn() if self.get_state_fn else {}
            trades = state.get("recent_trades", [])[-5:]
            if not trades:
                await self._send("📭 No trades yet")
                return
            lines = ["📋 *Recent Trades*\n"]
            for t in reversed(trades):
                emoji = "🟢" if t["side"] == "buy" else "🔴"
                lines.append(
                    f"{emoji} `{t['symbol']}` {t['side'].upper()} @ `{t['price']:,.4f}`\n"
                    f"  [{t['strategy']}] {'📄' if t.get('paper') else '💰'}"
                )
            await self._send("\n".join(lines))

        elif cmd in ("start_bot", "start"):
            if self.start_bot_fn:
                result = self.start_bot_fn()
                msg = result.get("message", "Starting...") if isinstance(result, dict) else "Starting..."
                await self._send(f"▶️ {msg}")
            else:
                await self._send("⚠️ start\\_bot not configured")

        elif cmd in ("stop_bot", "stop"):
            if self.stop_bot_fn:
                result = self.stop_bot_fn()
                msg = result.get("message", "Stopping...") if isinstance(result, dict) else "Stopping..."
                await self._send(f"⏹ {msg}")
            else:
                await self._send("⚠️ stop\\_bot not configured")

        elif cmd == "stats":
            s       = self.get_stats_fn() if self.get_stats_fn else {}
            total   = s.get("trades",          0)
            wins    = s.get("wins",            0)
            losses  = s.get("losses",          0)
            wr      = s.get("win_rate",        0.0)
            pf      = s.get("profit_factor",   0.0)
            total_r = s.get("total_r",         0.0)
            streak  = s.get("streak",          0)
            recent  = s.get("recent",          [])
            weekly  = s.get("weekly",          {})
            sig_all = s.get("total_signals",   0)
            sig_day = s.get("signals_per_day", 0)
            pending = s.get("pending",         0)

            if total == 0 and sig_all == 0:
                await self._send("📭 ยังไม่มีเทรดเลย — รอ signal แรก")
                return

            sign_r     = "+" if total_r >= 0 else ""
            streak_str = (f"W{streak}" if streak > 0 else f"L{abs(streak)}") if streak else "—"

            lines = []

            # ── ทั้งหมด ──────────────────────────────────────────────
            lines.append("📊 *Stats ทั้งหมด*")
            if total > 0:
                lines.append(
                    f"Trades: `{total}`  Win: `{wins}`  Loss: `{losses}`  "
                    f"WR: `{wr:.1f}%`"
                )
                lines.append(
                    f"PF: `{pf:.2f}`  Total R: `{sign_r}{total_r:.1f}R`  "
                    f"Streak: `{streak_str}`"
                )
            else:
                lines.append(f"_Signals fired: {sig_all} — รอ SL/TP_")

            # ── 7 วันล่าสุด ──────────────────────────────────────────
            lines.append("")
            lines.append("📅 *7 วันล่าสุด*")
            w7  = weekly.get("trades",   0)
            ww  = weekly.get("wins",     0)
            wl  = weekly.get("losses",   0)
            wwr = weekly.get("win_rate", 0.0)
            if w7 > 0:
                lines.append(
                    f"Trades: `{w7}`  Win: `{ww}`  Loss: `{wl}`  WR: `{wwr:.1f}%`"
                )
            else:
                lines.append("_ยังไม่มีเทรดใน 7 วันนี้_")
            lines.append(f"Avg signals/day: `{sig_day}`")
            if pending:
                lines.append(f"Open virtual: `{pending}`")

            # ── 5 เทรดล่าสุด ─────────────────────────────────────────
            lines.append("")
            lines.append("🕐 *5 เทรดล่าสุด*")
            if not recent:
                lines.append("_(รอ SL/TP hit ครั้งแรก)_")
            else:
                for o in reversed(recent):
                    e     = "✅" if o["pnl_r"] > 0 else "❌"
                    sr    = "+" if o["pnl_r"] >= 0 else ""
                    label = "TP" if o["reason"] == "take_profit" else "SL"
                    strat = o.get("strategy", "")
                    tag   = f" `[{strat}]`" if strat else ""
                    lines.append(
                        f"{e} `{o['symbol']}` {o['side'].upper()} "
                        f"`{sr}{o['pnl_r']:.1f}R` \\[{label}]{tag}"
                    )

            await self._send("\n".join(lines))

        else:
            await self._send(f"❓ Unknown command: `{text}`\nType /help for commands.")
