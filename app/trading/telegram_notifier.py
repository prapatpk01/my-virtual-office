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
import os
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
        get_insights_fn: Optional[Callable] = None,
        start_bot_fn: Optional[Callable] = None,
        stop_bot_fn: Optional[Callable] = None,
        min_confidence: float = 0.5,
    ):
        self.token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.get_state_fn = get_state_fn
        self.get_stats_fn = get_stats_fn
        self.get_okx_stats_fn = None   # async; set by the bot runner (real OKX stats)
        self.get_insights_fn = get_insights_fn
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

    def notify_photo(self, photo_path: str, caption: str = ""):
        """Fire-and-forget photo send (with optional caption) from any thread.
        Falls back to a text-only message if the photo send fails."""
        if not self._enabled:
            return
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_photo(photo_path, caption), self._loop)
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._send_photo(photo_path, caption))
                else:
                    loop.run_until_complete(self._send_photo(photo_path, caption))
            except Exception:
                pass

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
        reason = signal_dict.get("reason", "")
        meta   = signal_dict.get("metadata", {})

        sl  = meta.get("stop_loss")
        tp  = meta.get("take_profit")
        rr  = meta.get("rr")

        sl_tp_line = ""
        if sl and tp and rr:
            sl_tp_line = (
                f"\n🛑 SL: `{sl:,.4f}`\n"
                f"🎯 TP: `{tp:,.4f}`\n"
                f"📐 R:R `1:{rr:.1f}`"
            )
        text = (
            f"{emoji} *{sig_type.upper()} Signal*\n"
            f"`{sym}` @ `{price:,.4f}`\n"
            f"Strategy: {strat} | Conf: {conf*100:.0f}%\n"
            f"_{reason}_"
            f"{sl_tp_line}"
        )
        self.notify(text)

    def build_order_caption(self, symbol: str, side: str, amount: float,
                            price: float, strategy: str, paper: bool,
                            sl: float = None, tp: float = None,
                            macro_score: float = None, macro_bias: str = None,
                            selected_strategy: str = None, strategy_confidence: float = None,
                            regime: str = None, direction: str = None,
                            early_trend: bool = False, reason: str = None,
                            fee: float = 0.0) -> str:
        emoji = "🟢" if side == "buy" else "🔴"
        mode  = "📄 PAPER" if paper else "💰 LIVE"
        dir_label = f"LONG" if direction == "long" else "SHORT" if direction == "short" else side.upper()
        if early_trend:
            dir_label += " 🌱 EARLY"

        # price/amount here are the exchange's post-fill avgPx / fillSz —
        # the actual fill, not the requested size.
        lines = [
            f"{emoji} *Order Executed* {mode}",
            f"`{symbol}` — *{dir_label}*",
            f"Fill: `{amount:.6g}` @ `{price:,.4f}`"
            + (f"  |  Fee: `${fee:,.4f}`" if fee else ""),
        ]
        if reason:
            lines.append(f"_{reason}_")
        if early_trend:
            lines.append("🌱 _Early-trend entry: HMA crossed before the 30m trend confirmed "
                         "(passed the stricter Layer2 quality gate)._")

        if sl and tp:
            risk   = abs(price - sl)
            reward = abs(tp - price)
            rr     = reward / risk if risk > 0 else 0
            lines += [
                f"🛑 SL: `{sl:,.4f}`  ({abs(price-sl)/price*100:.2f}%)",
                f"🎯 TP: `{tp:,.4f}`  ({abs(tp-price)/price*100:.2f}%)",
                f"📐 R:R `1:{rr:.2f}`",
            ]
        elif sl:
            lines.append(f"🛑 SL: `{sl:,.4f}`")

        lines.append("")
        if selected_strategy:
            lines.append(f"🧠 Strategy: `{selected_strategy}` ({strategy_confidence:.0f})" if strategy_confidence else f"🧠 Strategy: `{selected_strategy}`")
        if macro_score is not None and macro_bias:
            lines.append(f"📈 Macro: `{macro_bias}` ({macro_score:.0f}/100)")
        if regime:
            lines.append(f"🌍 Regime: `{regime}`")
        lines.append(f"⚙️ {strategy}")

        return "\n".join(lines)

    def notify_order(self, symbol: str, side: str, amount: float,
                     price: float, strategy: str, paper: bool,
                     sl: float = None, tp: float = None,
                     macro_score: float = None, macro_bias: str = None,
                     selected_strategy: str = None, strategy_confidence: float = None,
                     regime: str = None, direction: str = None,
                     chart_path: str = None, early_trend: bool = False,
                     reason: str = None, fee: float = 0.0):
        """Send the order alert. If chart_path is given, sends it as a photo
        with the order details as caption; otherwise sends text only."""
        caption = self.build_order_caption(
            symbol, side, amount, price, strategy, paper,
            sl=sl, tp=tp, macro_score=macro_score, macro_bias=macro_bias,
            selected_strategy=selected_strategy, strategy_confidence=strategy_confidence,
            regime=regime, direction=direction, early_trend=early_trend,
            reason=reason, fee=fee,
        )
        if chart_path:
            self.notify_photo(chart_path, caption=caption)
        else:
            self.notify(caption)

    def _account_stats_block(self, stats: dict, title: str) -> str:
        """Shared '📊 Paper Account' footer for close notifications."""
        total     = stats.get("trades", 0)
        wins      = stats.get("wins", 0)
        losses    = stats.get("losses", 0)
        wr        = stats.get("win_rate", 0)
        pf        = stats.get("profit_factor", 0)
        total_usd = stats.get("total_pnl_usd", 0.0)
        ret_pct   = stats.get("return_pct", 0.0)
        start_bal = stats.get("start_balance", 1000.0)
        streak    = stats.get("streak", 0)
        streak_str = (f"W{streak}" if streak > 0 else f"L{abs(streak)}") if streak else "—"
        tot_sign  = "+" if total_usd >= 0 else "-"
        ret_sign  = "+" if ret_pct >= 0 else ""
        return (
            f"📊 *{title}* ({total} trades, start `${start_bal:,.0f}`)\n"
            f"Win/Loss: `{wins}W / {losses}L` | WR: `{wr:.1f}%`\n"
            f"Total P&L: `{tot_sign}${abs(total_usd):,.2f}` "
            f"(`{ret_sign}{ret_pct:.2f}%`) | PF: `{pf:.2f}`\n"
            f"Streak: `{streak_str}`"
        )

    def notify_trade_closed(self, symbol: str, outcome: dict, stats: dict):
        """Called when a position closes. Reports the ACTUAL exit reason.
        When the exchange's post-fill data is present (outcome["fill"]:
        avgPx/fillSz/fee/realized pnl), the trade result shown IS that data —
        Net PnL = realized_pnl - entry_fee_alloc - exit_fee. The $1000
        paper-account model remains as the running-stats footer."""
        won          = outcome.get("won", outcome.get("pnl_usd", 0) > 0)
        result_emoji = "✅" if won else "❌"
        label        = outcome.get("reason_label", "Position Closed")
        reason_emoji = outcome.get("emoji", "")
        raw_reason   = outcome.get("reason", "")
        side         = (outcome.get("side") or "").upper()
        entry        = outcome.get("entry", 0.0)
        exit_price   = outcome.get("exit", 0.0)
        sl           = outcome.get("sl")
        tp           = outcome.get("tp")
        pnl_usd      = outcome.get("pnl_usd", 0.0)
        pnl_pct      = outcome.get("pnl_pct", 0.0)
        pnl_r        = outcome.get("pnl_r", 0.0)
        bal_after    = outcome.get("balance_after", stats.get("paper_balance", 0.0))
        fill         = outcome.get("fill") or {}

        sl_str = f"`{sl:,.4f}`" if sl else "—"
        tp_str = f"`{tp:,.4f}`" if tp else "—"
        usd_sign = "+" if pnl_usd >= 0 else "-"

        # ── Trade result from the exchange fill (the real numbers) ─────────
        if fill.get("net_pnl") is not None:
            net = fill["net_pnl"]
            n_sign = "+" if net >= 0 else "-"
            fees_total = (fill.get("entry_fee_alloc") or 0) + (fill.get("exit_fee") or 0)
            result_block = (
                f"Fill: `{fill.get('exit_sz', 0):.6g}` @ `{fill.get('exit_avg_px', exit_price):,.4f}`\n"
                f"💵 Net P&L: `{n_sign}${abs(net):,.4f}`\n"
                f"Fees: entry `${fill.get('entry_fee_alloc', 0):,.4f}` + "
                f"exit `${fill.get('exit_fee', 0):,.4f}` = `${fees_total:,.4f}`\n"
            )
        else:
            result_block = (
                f"💵 P&L: `{usd_sign}${abs(pnl_usd):,.2f}`  "
                f"(`{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%`, `{pnl_r:+.2f}R`)\n"
                f"💰 Balance: `${bal_after:,.2f}`\n"
            )

        text = (
            f"{result_emoji} *{label}* {reason_emoji}\n"
            f"`{symbol}`  {side}\n"
            f"Entry: `{entry:,.4f}` → Exit: `{exit_price:,.4f}`\n"
            f"SL: {sl_str} | TP: {tp_str}\n"
            + result_block +
            f"_Reason: {raw_reason}_\n"
            f"_Use /stats for the OKX post-fee summary._"
        )
        self.notify(text)

    def notify_virtual_closed(self, symbol: str, reason: str, exit_price: float,
                              stats: dict, outcome: Optional[dict] = None):
        """Called when a virtual (signal-only / paper-0-balance) trade hits SL or TP."""
        outcome = outcome or {}
        won          = outcome.get("won", reason == "take_profit")
        result_emoji = "✅" if won else "❌"
        label        = outcome.get("reason_label",
                                    "Take-Profit Hit" if won else "Stop-Loss Hit")
        reason_emoji = outcome.get("emoji", "")
        pnl_usd      = outcome.get("pnl_usd")
        pnl_pct      = outcome.get("pnl_pct")
        bal_after    = outcome.get("balance_after", stats.get("paper_balance", 0.0))

        pnl_line = ""
        if pnl_usd is not None:
            usd_sign = "+" if pnl_usd >= 0 else "-"
            pnl_line = (
                f"💵 P&L: `{usd_sign}${abs(pnl_usd):,.2f}` "
                f"(`{'+' if (pnl_pct or 0) >= 0 else ''}{pnl_pct:.2f}%`)\n"
                f"💰 Balance: `${bal_after:,.2f}`\n"
            )

        text = (
            f"{result_emoji} *{label}* {reason_emoji} _(virtual)_\n"
            f"`{symbol}` @ `{exit_price:,.4f}`\n"
            f"{pnl_line}"
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

    def notify_cooldown_halt(self, streak: int, hours: float):
        text = (
            f"🧊 *Cooldown Started — New Entries Paused*\n"
            f"`{streak}` consecutive losing closes in a row.\n"
            f"Resumes automatically in `{hours:.1f}h` — existing positions "
            f"are still managed normally, no action needed."
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

    def notify_reconciled_position(self, symbol: str, strategy_name: str, side: str,
                                    entry_price: float, amount: float,
                                    stop_loss: Optional[float], take_profit: Optional[float]):
        # SL/TP may be None (e.g. the pure cross-back system runs with no TP) —
        # format defensively so this notification never crashes and get silently
        # swallowed, which would leave a restart-adopted position unannounced.
        sl_str = f"`{stop_loss:.4f}`" if stop_loss else "—"
        tp_str = f"`{take_profit:.4f}`" if take_profit else "—"
        text = (
            f"🔄 *Reconciled Position on Restart*\n"
            f"`{strategy_name}` on `{symbol}`\n"
            f"Side: *{side.upper()}*  Entry: `{entry_price:.4f}`  Amount: `{amount:.6f}`\n"
            f"SL: {sl_str}  TP: {tp_str} _(adopted after restart)_"
        )
        self.notify(text)

    # ------------------------------------------------------------------
    # Telegram HTTP helpers
    # ------------------------------------------------------------------

    async def _send(self, text: str, parse_mode: str = "Markdown") -> bool:
        url = TELEGRAM_API.format(token=self.token, method="sendMessage")
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("Telegram send failed %s: %s", r.status, body[:200])
                        return False
                    return True
        except Exception as e:
            logger.warning("Telegram send error: %s", e)
            return False

    async def _send_photo(self, photo_path: str, caption: str = "",
                          parse_mode: str = "Markdown") -> bool:
        url = TELEGRAM_API.format(token=self.token, method="sendPhoto")
        try:
            with open(photo_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("chat_id", self.chat_id)
                if caption:
                    data.add_field("caption", caption[:1024])  # Telegram caption limit
                    data.add_field("parse_mode", parse_mode)
                data.add_field("photo", f, filename=os.path.basename(photo_path),
                               content_type="image/png")
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=data,
                                            timeout=aiohttp.ClientTimeout(total=20)) as r:
                        if r.status != 200:
                            body = await r.text()
                            logger.warning("Telegram sendPhoto failed %s: %s", r.status, body[:200])
                            # Fall back to text-only so the alert isn't lost
                            await self._send(caption or "(chart failed to send)")
                            return False
                        return True
        except Exception as e:
            logger.warning("Telegram sendPhoto error: %s", e)
            try:
                await self._send(caption or "(chart failed to send)")
            except Exception:
                pass
            return False
        finally:
            try:
                os.remove(photo_path)
            except OSError:
                pass

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

    @staticmethod
    def _age_str(seconds: int) -> str:
        if seconds < 3600:
            return f"{max(1, seconds // 60)}m ago"
        if seconds < 86400:
            return f"{seconds / 3600:.1f}h ago"
        return f"{seconds / 86400:.1f}d ago"

    def _render_stats(self, s: dict) -> str:
        """Render /stats. OKX-sourced format (OVERALL / BY SYMBOL / LAST 5)
        when source == 'okx'; otherwise a short internal-fallback block."""
        SEP = "—" * 16
        bal = s.get("balance")
        bal_line = f"💰 Balance: `${bal:,.2f}`" if bal is not None else "💰 Balance: `—`"
        open_pos = s.get("open_positions", 0)
        pos_line = f"📌 Open positions: `{open_pos}`" if open_pos else "📌 No open positions"

        if s.get("source") != "okx":
            # Fallback (paper mode / no exchange history yet)
            return "\n".join([
                bal_line, pos_line, "",
                "_Live OKX stats unavailable (paper mode or no fills yet)._",
                f"Internal record: `{s.get('trades', 0)}` trades, "
                f"WR `{s.get('win_rate', 0):.0f}%`, "
                f"net `{'+' if s.get('total_pnl_usd', 0) >= 0 else ''}"
                f"${s.get('total_pnl_usd', 0):,.2f}`",
            ])

        n = s.get("trades", 0)
        wins = s.get("wins", 0); losses = s.get("losses", 0)
        net = s.get("net_pnl", 0.0)
        net_sign = "+" if net >= 0 else "-"
        tp1 = s.get("tp1_hits", 0); tp2 = s.get("tp2_hits", 0); sl = s.get("sl_only", 0)
        den = max(1, s.get("partial_denom", 0))

        lines = [bal_line, pos_line, "", SEP, "OVERALL (OKX)", SEP,
                 f"Trades   : `{n}`  (`{wins}W / {losses}L`)",
                 f"Win rate : `{s.get('win_rate', 0):.0f}%`",
                 f"TP1 hit  : `{tp1}/{den}` (`{tp1/den*100:.0f}%`)   "
                 f"TP2 hit : `{tp2}/{den}` (`{tp2/den*100:.0f}%`)",
                 f"SL only  : `{sl}/{n or 1}` (`{sl/(n or 1)*100:.0f}%`)",
                 f"Net PnL  : `{net_sign}${abs(net):,.2f}`  (post-fee, from OKX)"]

        open_list = s.get("open_list", [])
        if open_list:
            lines += ["", SEP, "OPEN POSITIONS", SEP]
            for p in open_list:
                arrow = "🟢" if p["side"] == "long" else "🔴"
                upnl = p.get("upnl")
                if upnl is None:
                    upnl_str = ""
                else:
                    us = "+" if upnl >= 0 else "-"
                    upnl_str = f"  uPnL `{us}${abs(upnl):,.2f}`"
                lines.append(
                    f"{arrow} `{p['short']:<5}` {p['side'].upper()} "
                    f"`{p['amount']:g}` @ `${p['entry']:,.4f}`  "
                    f"(≈`${p['notional']:,.0f}`){upnl_str}")

        per = s.get("per_symbol", {})
        if per:
            lines += ["", SEP, "BY SYMBOL", SEP]
            for sym, d in sorted(per.items(), key=lambda kv: -kv[1]["net"]):
                if d["trades"] == 0:
                    lines.append(f"`{sym:<5}` 0 trades")
                else:
                    ns = "+" if d["net"] >= 0 else "-"
                    lines.append(f"`{sym:<5}` {d['trades']} trades  {d['win_rate']:.0f}%WR  "
                                 f"`{ns}${abs(d['net']):,.2f}`")

        last = s.get("last_trades", [])
        if last:
            lines += ["", SEP, "LAST 5 TRADES", SEP]
            for i, t in enumerate(last, 1):
                e = "✅" if t["won"] else "❌"
                ns = "+" if t["net"] >= 0 else "-"
                lines.append(f"{i}. {e} {t['short']} {t['side'].upper()} "
                             f"`{ns}${abs(t['net']):,.2f}` — {self._age_str(t['age_s'])}")
        return "\n".join(lines)

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
                "/insights — deep learning analysis & recommendations\n"
                "/start\\_bot — start the bot\n"
                "/stop\\_bot — stop the bot\n"
                "/help — this message"
            )

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
            # Real stats from OKX order history (post-fee), sectioned display.
            fn = getattr(self, "get_okx_stats_fn", None)
            s = (await fn()) if fn else (self.get_stats_fn() if self.get_stats_fn else {})
            await self._send(self._render_stats(s))

        elif cmd == "insights":
            insights = self.get_insights_fn() if self.get_insights_fn else {}
            if not insights or not insights.get("total_closed"):
                await self._send(
                    "📭 Not enough closed trades yet for deep analysis.\n"
                    "Insights unlock once trades start closing."
                )
                return

            period = insights.get("period_days", 30)
            closed = insights.get("total_closed", 0)
            total_sig = insights.get("total_signals", 0)
            trend = insights.get("trend", {})
            by_strategy = insights.get("by_strategy", {})
            by_symbol = insights.get("by_symbol", {})
            by_confidence = insights.get("by_confidence", {})
            recs = insights.get("recommendations", [])

            lines = [f"🧠 *Learning Analysis — ย้อนหลัง {period} วัน*\n"]
            lines.append(f"Signals: `{total_sig}` | Closed trades: `{closed}`")

            direction = trend.get("direction")
            if direction and direction != "insufficient_data":
                arrow = {"improving": "📈", "declining": "📉", "flat": "➡️"}.get(direction, "➡️")
                rwr = trend.get("recent_win_rate")
                pwr = trend.get("prior_win_rate")
                compare = f" vs prior `{pwr}%`" if pwr is not None else ""
                lines.append(f"{arrow} Trend: *{direction}* (recent `{rwr}%`{compare})")

            if by_strategy:
                lines.append("\n*By strategy:*")
                for s, d in sorted(by_strategy.items(), key=lambda kv: kv[1].get("win_rate") or 0, reverse=True):
                    wr = f"{d['win_rate']}%" if d["win_rate"] is not None else "—"
                    lines.append(f"`{s}` — {wr} ({d['wins']}W/{d['losses']}L)")

            if by_symbol:
                lines.append("\n*By symbol (Total R):*")
                for sym, d in sorted(by_symbol.items(), key=lambda kv: kv[1].get("total_r", 0), reverse=True)[:5]:
                    lines.append(f"`{sym}` — {d['total_r']}R ({d['trades']} trades, {d['win_rate']}% WR)")

            if by_confidence:
                lines.append("\n*By confidence:*")
                for label, d in by_confidence.items():
                    wr = f"{d['win_rate']}%" if d["win_rate"] is not None else "—"
                    lines.append(f"`{label}` — {wr} ({d['trades']} trades)")

            if recs:
                lines.append("\n*💡 Recommendations:*")
                for r in recs:
                    lines.append(f"• {r}")

            await self._send("\n".join(lines))

        else:
            await self._send(f"❓ Unknown command: `{text}`\nType /help for commands.")
