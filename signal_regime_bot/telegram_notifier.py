"""
Telegram Notifier — sends every alert the spec requires: entry signal,
order opened, TP1 hit, SL moved to BE, TP2 hit, SL hit, daily limit hit,
cooldown activated, OKX API error. Photos (charts) sent via sendPhoto.
"""
from __future__ import annotations

import asyncio
import logging

import aiohttp

logger = logging.getLogger("telegram_notifier")

API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def _send_message(self, text: str) -> bool:
        if not self.enabled:
            return False
        url = API.format(token=self.token, method="sendMessage")
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("[TG] sendMessage %s: %s", r.status, body[:200])
                        return False
                    return True
        except Exception as e:
            logger.warning("[TG] sendMessage failed: %s", e)
            return False

    async def _send_photo(self, path: str, caption: str) -> bool:
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
                        logger.warning("[TG] sendPhoto %s: %s", r.status, body[:200])
                        return False
                    return True
        except Exception as e:
            logger.warning("[TG] sendPhoto failed: %s", e)
            return False

    async def send_text(self, text: str) -> bool:
        """Plain reply for the command interface."""
        return await self._send_message(text)

    async def get_updates(self, offset: int, timeout: int = 25) -> list:
        """
        Long-poll Telegram getUpdates for incoming commands. Returns the raw
        update list ([] on error/disabled). Caller tracks the offset.
        """
        if not self.enabled:
            await asyncio.sleep(timeout)
            return []
        url = API.format(token=self.token, method="getUpdates")
        params = {"offset": offset, "timeout": timeout, "allowed_updates": '["message"]'}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=timeout + 15)) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("[TG] getUpdates %s: %s", r.status, body[:200])
                        await asyncio.sleep(3)
                        return []
                    data = await r.json()
                    return data.get("result", []) if data.get("ok") else []
        except asyncio.TimeoutError:
            return []
        except Exception as e:
            logger.warning("[TG] getUpdates failed: %s", e)
            await asyncio.sleep(3)
            return []

    # ── Formatted alerts ──────────────────────────────────────────────────────

    async def entry_signal(self, symbol: str, direction: str, price: float, sl: float,
                           tp1: float, tp2: float, regime, bias, entry_score: float,
                           risk_pct: float, leverage: int, chart_path: str | None = None):
        # bias is None for MEANREV / BREAKOUT style trades (they bypass the
        # momentum-bias gate) — show the trade style instead.
        if bias is not None:
            bias_line = f"Bias: `{bias.bias}` ({bias.bull_score if direction=='LONG' else bias.bear_score:.0f})"
        else:
            bias_line = f"Style: `{getattr(regime, 'style', '—')}` ({getattr(regime, 'regime_type', '—')})"
        text = (
            f"🎯 *Entry Signal*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: *{direction}*\n"
            f"Entry: `{price:.6f}`\n"
            f"SL: `{sl:.6f}`\n"
            f"TP1: `{tp1:.6f}`\n"
            f"TP2: `{tp2:.6f}`\n"
            f"Regime: `{regime.name}` ({regime.score:.0f})\n"
            f"{bias_line}\n"
            f"Entry Score: `{entry_score:.0f}`\n"
            f"Risk: `{risk_pct*100:.1f}%`\n"
            f"Leverage: `{leverage}x`"
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

    async def tp1_hit(self, symbol: str, price: float, pnl: float, new_sl: float):
        await self._send_message(
            f"🎯 *TP1 Hit* `{symbol}`\n\n"
            f"Price: `{price:.6f}`\n"
            f"PnL (50%): `{pnl:+.2f}` USDT\n"
            f"SL moved to Breakeven: `{new_sl:.6f}`"
        )

    async def tp2_hit(self, symbol: str, price: float, pnl: float):
        await self._send_message(
            f"🏁 *TP2 Hit — Position Closed* `{symbol}`\n\n"
            f"Price: `{price:.6f}`\nPnL: `{pnl:+.2f}` USDT"
        )

    async def sl_hit(self, symbol: str, price: float, pnl: float, at_breakeven: bool = False):
        label = "Breakeven Stop" if at_breakeven else "Stop Loss Hit"
        await self._send_message(
            f"🛑 *{label}* `{symbol}`\n\n"
            f"Price: `{price:.6f}`\nPnL: `{pnl:+.2f}` USDT"
        )

    async def spike_guard(self, symbol: str, price: float, pnl: float, reason: str):
        await self._send_message(
            f"⚡️ *Spike Guard — Emergency Close* `{symbol}`\n\n"
            f"Reversal spike detected — closed before full SL.\n"
            f"`{reason}`\n"
            f"Price: `{price:.6f}`\nPnL: `{pnl:+.2f}` USDT"
        )

    async def health_close(self, symbol: str, price: float, pnl: float,
                           health_score: float, weak_count: int):
        await self._send_message(
            f"⚠️ *Health Close* `{symbol}`\n\n"
            f"Closed after {weak_count} consecutive WEAK reads (score={health_score:.0f})\n"
            f"Price: `{price:.6f}`\nPnL: `{pnl:+.2f}` USDT"
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
