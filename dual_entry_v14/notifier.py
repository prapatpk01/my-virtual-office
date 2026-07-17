"""Telegram Notifier (spec §32) — resilient: markdown 400 falls back to plain
text so an alert can never vanish on a formatting error.
"""
from __future__ import annotations

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger("dual_entry.notify")
API = "https://api.telegram.org/bot{token}/{method}"


class Notifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def _send(self, text: str, markdown: bool = True) -> bool:
        if not self.enabled:
            return False
        payload = {"chat_id": self.chat_id, "text": text}
        if markdown:
            payload["parse_mode"] = "Markdown"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(API.format(token=self.token, method="sendMessage"),
                                  json=payload,
                                  timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning("[TG] %s: %s", r.status, body[:150])
                        if r.status == 400 and markdown:
                            return await self._send(text, markdown=False)
                        return False
                    return True
        except Exception as e:
            logger.warning("[TG] send failed: %s", e)
            return False

    # ── message shapes (spec §32) ────────────────────────────────────────────

    async def signal(self, cand, plan, risk_pct: float) -> None:
        z = cand.active_zone
        await self._send(
            f"{'🟢 LONG' if cand.direction == 'LONG' else '🔴 SHORT'} *SIGNAL*\n"
            f"Symbol: `{cand.symbol}`\nSetup: `{cand.setup_type}`\n"
            f"Score: `{cand.score:.0f}/100`  Threshold: `{cand.threshold:.0f}`  "
            f"Edge: `{cand.edge_score:+.0f}`\n\n"
            f"Entry Reference: `{cand.entry_reference:.6f}`\n"
            f"SL: `{plan.stop_price:.6f}`\nTP: `{plan.target_price:.6f}`\n"
            f"Risk: `{risk_pct * 100:.2f}%`  Planned RR: `{plan.planned_rr:.2f}`\n\n"
            f"4H: `{cand.htf_structure}`  1H Bias: `{cand.bias}`\n"
            f"15M Regime: `{cand.regime}`\n"
            f"Zone: `{(z.timeframe + ' ' + z.zone_type) if z else '-'}`  "
            f"Zone Score: `{cand.zone_score:.0f}`\n"
            f"Trigger: `{cand.candle_pattern or '-'}`\n"
            f"Structure Room: `{cand.structure_room_r:.2f}R`")

    async def fill(self, symbol: str, cand, plan, fill_price: float, qty: float,
                   slip_atr: float) -> None:
        await self._send(
            f"✅ *{cand.direction} FILLED* `{symbol}`\n"
            f"Entry: `{fill_price:.6f}`\nQuantity: `{qty:.6f}`\n"
            f"SL: `{plan.stop_price:.6f}`  TP: `{plan.target_price:.6f}`\n"
            f"Risk Distance: `{abs(fill_price - plan.stop_price):.6f}`\n"
            f"Actual RR: `{abs(plan.target_price - fill_price) / max(plan.effective_risk_distance, 1e-9):.2f}`\n"
            f"Slippage: `{slip_atr:.2f} ATR`")

    async def breakeven(self, symbol: str, setup: str, new_sl: float, r: float) -> None:
        await self._send(
            f"🔒 *BREAK-EVEN MOVED* `{symbol}`\nSetup: `{setup}`\n"
            f"New SL: `{new_sl:.6f}`\nCurrent R: `{r:+.2f}R`")

    async def exit(self, symbol: str, setup: str, reason: str, r: float,
                   exit_price: float) -> None:
        emoji = "🏁" if r > 0 else "🛑"
        await self._send(
            f"{emoji} *POSITION CLOSED* `{symbol}`\nSetup: `{setup}`\n"
            f"Reason: `{reason}`\nResult: `{r:+.2f}R`\nExit: `{exit_price:.6f}`")

    async def warn(self, text: str) -> None:
        await self._send(f"⚠️ {text}")

    async def critical(self, text: str) -> None:
        await self._send(f"🚨 *CRITICAL*\n{text}")

    async def send_critical_error(self, symbol: str, error: Exception) -> None:
        await self.critical(f"`{symbol}` error: `{str(error)[:300]}`")

    async def info(self, text: str) -> None:
        await self._send(text)
