"""HMA Expert MTF V3.2 recovery-safe runtime.

Fixes two production issues without changing entry logic:
1) Positions adopted after a Railway restart have entry=0 in local state. They
   must stay exchange-managed until close; running percentage/R calculations on
   them caused ``float division by zero``.
2) Repeated per-poll failures are rate-limited in Telegram while full errors
   continue to be written to Railway logs.
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
        await super().start()
        v3.base.logger.info(
            "HMA V3.2 recovery guard active: adopted positions are exchange-managed; "
            "Telegram error alerts limited to once per symbol per 15 minutes"
        )

    async def _manage(self, symbol: str, st: dict):
        """Safely manage locally tracked and restart-adopted positions.

        A position discovered on OKX after restart is stored with entry/sl/tp=0
        because the old process state is unavailable. The inherited manager calls
        locked_stop() before checking ``adopted`` and therefore divides by zero.
        Keep such positions protected by their existing OKX-native SL/TP and only
        watch for the exchange position to close.
        """
        pos = st.get("pos") or {}
        side = str(pos.get("side") or "").lower()
        entry = float(pos.get("entry") or 0.0)
        adopted = bool(pos.get("adopted")) or entry <= 0.0

        if adopted:
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
                f"OPEN {side.upper()} adopted/restart | amount={amount:.8g} "
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
