"""HMA Expert MTF V3.2 recovery-safe runtime.

Production safeguards:
1) Positions adopted after a Railway restart have no reliable local entry/SL/TP.
   They remain protected by OKX-native SL/TP and are never passed into percentage
   or R calculations that require a non-zero entry.
2) Repeated per-poll failures are rate-limited in Telegram while full tracebacks
   remain available in Railway logs.
3) Startup and status output clearly identify V3.2 and adopted positions.
"""
from __future__ import annotations

import asyncio
import time

import main_v3 as v3


class Bot(v3.Bot):
    ERROR_NOTIFY_COOLDOWN_SEC = 15 * 60

    def __init__(self):
        super().__init__()
        self._error_notified_at: dict[str, float] = {}

    async def start(self):
        """Start V3.2 directly so Telegram does not show the inherited V3.1 banner."""
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))

        if not self.cfg.paper:
            if not await self.client.ensure_hedge_mode():
                raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        v3.base.logger.info(
            "=== HMA EXPERT MTF V3.2 RECOVERY-SAFE [%s] symbols=%s margin=$%.2f "
            "leverage=x%d max_pos=%d balance=%.2f ===",
            "PAPER" if self.cfg.paper else "LIVE",
            self.cfg.symbols,
            self.cfg.margin_per_position_usd,
            self.cfg.leverage,
            self.cfg.max_positions,
            balance,
        )

        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            await self.tg.send_text(
                f"🤖 *HMA Expert MTF V3.2 Recovery-Safe started* "
                f"[{'PAPER' if self.cfg.paper else 'LIVE'}]\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin "
                f"`${self.cfg.margin_per_position_usd:.2f}`/position | "
                f"Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n"
                f"4H Direction → 1H Q+soft DMI → 15M Location → recent 5M Execution\n"
                f"SL: 15M structure + ATR buffer | T1 +0.6%→lock +0.3% | "
                f"T2 +1.0%→lock +0.7% | TP +1.5%\n"
                f"Restarted positions: OKX-native SL/TP management until close"
            )

        v3.base.logger.info(
            "HMA V3.2 recovery guard active: adopted positions are exchange-managed; "
            "Telegram error alerts limited to once per symbol per 15 minutes"
        )

    @staticmethod
    def _is_adopted_position(pos: dict) -> bool:
        try:
            entry = float(pos.get("entry") or 0.0)
        except (TypeError, ValueError):
            entry = 0.0
        return bool(pos.get("adopted")) or entry <= 0.0

    def _view_line(self, symbol: str) -> str:
        """Never display adopted positions as a fake entry price of zero."""
        st = self.state.get(symbol) or {}
        pos = st.get("pos")
        if pos and self._is_adopted_position(pos):
            side = str(pos.get("side") or "?").upper()
            amount = float(pos.get("amount") or 0.0)
            live = self._view.get(symbol, "OKX native SL/TP managing")
            return f"OPEN {side} adopted/restart | amount={amount:.8g} | {live}"
        return super()._view_line(symbol)

    async def _manage(self, symbol: str, st: dict):
        """Safely manage locally tracked and restart-adopted positions."""
        pos = st.get("pos") or {}
        side = str(pos.get("side") or "").lower()

        if self._is_adopted_position(pos):
            if side not in ("long", "short"):
                v3.base.logger.warning("[%s] invalid adopted position side=%r", symbol, side)
                return

            amount = await self.client.fetch_position_amount(symbol, side)
            if amount <= 0:
                await self._report_close(symbol, st)
                return

            ticker = await self.client.fetch_ticker(symbol)
            price = float(ticker.get("last") or 0.0)
            self._view[symbol] = (
                f"px={price:.8g} | OKX native SL/TP managing"
            )
            return

        await super()._manage(symbol, st)

    async def run_forever(self):
        """Run normally, but prevent one recurring fault from flooding Telegram."""
        while self._running:
            for symbol in self.cfg.symbols:
                try:
                    await self._process(symbol)
                except Exception as exc:
                    v3.base.logger.error("[%s] unhandled: %s", symbol, exc, exc_info=True)
                    now = time.time()
                    last = self._error_notified_at.get(symbol, 0.0)
                    if now - last >= self.ERROR_NOTIFY_COOLDOWN_SEC:
                        self._error_notified_at[symbol] = now
                        try:
                            await self.tg.send_text(
                                f"❌ `{v3.base._sym(symbol)}` error: {str(exc)[:150]}\n"
                                f"Telegram repeats muted for 15 minutes; Railway log has traceback."
                            )
                        except Exception:
                            pass
            self._maybe_status_log()
            await asyncio.sleep(self.cfg.poll_interval_sec)


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v3.base._signal, sig_name),
                lambda: asyncio.ensure_future(bot.stop()),
            )
        except (NotImplementedError, AttributeError):
            pass

    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
