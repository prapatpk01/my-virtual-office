"""
Telegram Notifier — sends every alert the spec requires: entry signal,
order opened, TP1 hit, SL moved to BE, TP2 hit, SL hit, daily limit hit,
cooldown activated, OKX API error. Photos (charts) sent via sendPhoto.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger("telegram_notifier")

API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    # Prevent more than one long-poll request for the same token inside a
    # single Python process. This does not replace the Railway requirement
    # that only one replica/service may poll a Telegram bot token.
    _poll_locks: dict[str, asyncio.Lock] = {}

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._conflict_count = 0
        self._conflict_backoff_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def _send_message(self, text: str, _markdown: bool = True) -> bool:
        if not self.enabled:
            return False
        url = API.format(token=self.token, method="sendMessage")
        payload = {"chat_id": self.chat_id, "text": text}
        if _markdown:
            payload["parse_mode"] = "Markdown"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("[TG] sendMessage %s (markdown=%s): %s",
                                       r.status, _markdown, body[:200])
                        # A 400 is almost always a Markdown parse error (an
                        # unbalanced `*`/`_`/backtick in a symbol or reason
                        # string). Retry ONCE as plain text so the alert still
                        # lands instead of vanishing silently.
                        if r.status == 400 and _markdown:
                            return await self._send_message(text, _markdown=False)
                        return False
                    return True
        except Exception as e:
            logger.warning("[TG] sendMessage failed: %s", e)
            return False

    async def _send_photo(self, path: str, caption: str) -> bool:
        # A missing/broken chart must NEVER cost us the alert — fall back to the
        # caption as a text message on any failure.
        if not self.enabled:
            return False
        url = API.format(token=self.token, method="sendPhoto")
        try:
            data = aiohttp.FormData()
            data.add_field("chat_id", self.chat_id)
            data.add_field("caption", caption[:1024])
            data.add_field("parse_mode", "Markdown")
            with open(path, "rb") as f:
                data.add_field("photo", f.read(), filename="chart.png", content_type="image/png")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data,
                                        timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("[TG] sendPhoto %s: %s — falling back to text", r.status, body[:200])
                        return await self._send_message(caption)
                    return True
        except Exception as e:
            logger.warning("[TG] sendPhoto failed: %s — falling back to text", e)
            return await self._send_message(caption)

    async def send_text(self, text: str) -> bool:
        """Plain reply for the command interface."""
        return await self._send_message(text)

    async def get_updates(self, offset: int, timeout: int = 25) -> list:
        """Long-poll Telegram for incoming commands.

        The method is deliberately non-fatal: command polling failures must
        never stop the trading loop. A per-token lock prevents duplicate
        getUpdates calls inside one process. Telegram HTTP 409 responses are
        backed off exponentially so deployment overlap does not create a hot
        retry loop.
        """
        if not self.enabled:
            await asyncio.sleep(timeout)
            return []

        loop = asyncio.get_running_loop()
        now = loop.time()
        if now < self._conflict_backoff_until:
            await asyncio.sleep(min(timeout, self._conflict_backoff_until - now))
            return []

        poll_lock = self._poll_locks.setdefault(self.token, asyncio.Lock())
        url = API.format(token=self.token, method="getUpdates")
        params = {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": '["message"]',
        }

        async with poll_lock:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=timeout + 15),
                    ) as r:
                        if r.status == 409:
                            body = await r.text()
                            self._conflict_count += 1
                            backoff = min(300, 30 * (2 ** (self._conflict_count - 1)))
                            self._conflict_backoff_until = loop.time() + backoff
                            logger.warning(
                                "[TG] getUpdates 409 conflict: another process/replica "
                                "is polling this bot token. Commands paused for %ss; "
                                "trading remains active. Detail: %s",
                                backoff,
                                body[:200],
                            )
                            return []

                        if r.status != 200:
                            body = await r.text()
                            logger.warning("[TG] getUpdates %s: %s", r.status, body[:200])
                            await asyncio.sleep(3)
                            return []

                        data = await r.json()
                        if not data.get("ok"):
                            logger.warning("[TG] getUpdates returned ok=false: %s", str(data)[:200])
                            return []

                        self._conflict_count = 0
                        self._conflict_backoff_until = 0.0
                        result = data.get("result", [])
                        return result if isinstance(result, list) else []

            except asyncio.TimeoutError:
                return []
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[TG] getUpdates failed: %s", e)
                await asyncio.sleep(3)
                return []

    # ── Formatted alerts ──────────────────────────────────────────────────────

    async def entry_signal(self, symbol: str, direction: str, price: float, sl: float,
                           tp1: float, tp2: float, regime, bias, entry_score: float,
                           margin_usdt: float, leverage: int, chart_path: str | None = None,
                           entry_result=None):
        if bias is not None:
            bias_line = f"Bias: `{bias.bias}` ({bias.bull_score if direction=='LONG' else bias.bear_score:.0f})"
        else:
            bias_line = f"Style: `{getattr(regime, 'style', '—')}` ({getattr(regime, 'regime_type', '—')})"
        setup_line = ""
        score_line = f"Entry Score: `{entry_score:.0f}`"
        edge_line = ""
        if entry_result is not None:
            setup = getattr(entry_result, "setup_type", "") or "—"
            trigger = getattr(entry_result, "trigger", "") or "—"
            threshold = getattr(entry_result, "score_threshold", None)
            setup_line = f"Setup: `{setup}` | Trigger: `{trigger}`\n"
            if threshold is not None:
                score_line = f"Entry Score: `{entry_score:.0f}/{threshold:.0f}`"
            components = getattr(entry_result, "score_components", {}) or {}
            local_edge = components.get("local_direction_edge")
            if local_edge is not None:
                edge_line = f"\nLocal Edge: `{float(local_edge):+.0f}`"
        text = (
            f"🎯 *Entry Signal*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: *{direction}*\n"
            f"{setup_line}"
            f"Entry: `{price:.6f}`\n"
            f"SL: `{sl:.6f}`\n"
            f"TP1: `{tp1:.6f}`\n"
            f"TP2: `{tp2:.6f}`\n"
            f"Regime: `{regime.name}` ({regime.score:.0f})\n"
            f"{bias_line}\n"
            f"{score_line}{edge_line}\n"
            f"Margin: `{margin_usdt:.2f} USDT`\n"
            f"Leverage: `{leverage}x`\n"
            f"Target Notional: `~{margin_usdt * leverage:.2f} USDT`"
        )
        if chart_path:
            await self._send_photo(chart_path, text)
        else:
            await self._send_message(text)

    async def order_opened(self, symbol: str, direction: str, price: float, amount: float,
                           sl: float, tp1: float, tp2: float):
        await self._send_message(
            f"✅ *Order Opened*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: *{direction}*\n"
            f"Entry: `{price:.6f}`\n"
            f"Amount: `{amount:.6f}`\n"
            f"SL: `{sl:.6f}` | TP1: `{tp1:.6f}` | TP2: `{tp2:.6f}`"
        )

    @staticmethod
    def _net_block(pnl: float, ev: Optional[dict] = None) -> str:
        """Clearly distinguish a partial-leg result from the whole-trade result."""
        if ev and ev.get("realized") is not None:
            breakdown = (
                f"  exit-leg realized `{ev['realized']:+.4f}` − openFeeAlloc "
                f"`{ev.get('entry_fee_alloc', 0.0):.4f}` − closeFee `{ev.get('exit_fee', 0.0):.4f}`"
            )
            if ev.get("trade_pnl") is not None:
                return (
                    f"Trade Net PnL: `{float(ev['trade_pnl']):+.4f}` USDT\n"
                    f"This exit leg: `{float(ev.get('pnl', pnl)):+.4f}` USDT\n"
                    f"{breakdown}"
                )
            if ev.get("cumulative_pnl") is not None:
                return (
                    f"TP1 leg net: `{pnl:+.4f}` USDT\n"
                    f"Banked cumulative net: `{float(ev['cumulative_pnl']):+.4f}` USDT\n"
                    f"{breakdown}"
                )
            return f"Net PnL: `{pnl:+.4f}` USDT\n{breakdown}"
        return f"PnL: `{pnl:+.2f}` USDT"

    async def tp1_hit(self, symbol: str, price: float, pnl: float, new_sl: float,
                      ev: Optional[dict] = None):
        await self._send_message(
            f"🎯 *TP1 Hit* `{symbol}`\n\n"
            f"Price: `{price:.6f}`\n"
            f"{self._net_block(pnl, ev)}\n"
            f"Fee-adjusted runner SL: `{new_sl:.6f}`"
        )

    async def tp2_hit(self, symbol: str, price: float, pnl: float, ev: Optional[dict] = None):
        await self._send_message(
            f"🏁 *TP2 Hit — Position Closed* `{symbol}`\n\n"
            f"Price: `{price:.6f}`\n{self._net_block(pnl, ev)}"
        )

    async def sl_hit(self, symbol: str, price: float, pnl: float, at_breakeven: bool = False,
                     ev: Optional[dict] = None):
        label = "Breakeven Stop" if at_breakeven else "Stop Loss Hit"
        await self._send_message(
            f"🛑 *{label}* `{symbol}`\n\n"
            f"Price: `{price:.6f}`\n{self._net_block(pnl, ev)}"
        )

    async def ai_exit(self, symbol: str, price: float, pnl: float, reason: str,
                      emergency: bool = False, ev: Optional[dict] = None):
        title = "AI Exit — Emergency Close" if emergency else "AI Exit — Confirmed Reversal"
        icon = "🚨" if emergency else "🧠"
        await self._send_message(
            f"{icon} *{title}* `{symbol}`\n\n"
            f"Multi-factor exit decision.\n"
            f"`{reason}`\n"
            f"Price: `{price:.6f}`\n{self._net_block(pnl, ev)}"
        )

    async def spike_guard(self, symbol: str, price: float, pnl: float, reason: str,
                          ev: Optional[dict] = None):
        await self._send_message(
            f"⚡️ *Spike Guard — Emergency Close* `{symbol}`\n\n"
            f"Reversal spike detected — closed before full SL.\n"
            f"`{reason}`\n"
            f"Price: `{price:.6f}`\n{self._net_block(pnl, ev)}"
        )

    async def early_exit(self, symbol: str, price: float, pnl: float,
                         reason: str, detail: str = "", ev: Optional[dict] = None):
        label = "EMA Cross Reversal" if reason == "EMA_CROSS_REVERSAL" else "Price Opened Beyond EMA20"
        await self._send_message(
            f"⚠️ *Early Exit — {label}* `{symbol}`\n\n"
            f"`{detail}`\n"
            f"Price: `{price:.6f}`\n{self._net_block(pnl, ev)}"
        )

    async def daily_loss_limit(self, day_pnl_pct: float):
        await self._send_message(
            f"🚨 *Daily Loss Limit Hit*\n\nDay PnL: `{day_pnl_pct*100:.1f}%`\n"
            f"New entries paused until next UTC day."
        )

    async def daily_profit_lock(self, day_pnl_pct: float):
        await self._send_message(
            f"🔒 *Daily Profit Lock Hit*\n\nDay PnL: `{day_pnl_pct*100:.1f}%`\n"
            f"New entries paused until next UTC day."
        )

    async def cooldown_activated(self, minutes: int, loss_streak: int):
        await self._send_message(
            f"⏸ *Cooldown Activated*\n\n{loss_streak} consecutive losses — "
            f"pausing new entries for `{minutes}` min."
        )

    async def error(self, context: str, detail: str):
        await self._send_message(f"❌ *Error* `{context}`\n\n`{detail[:500]}`")
