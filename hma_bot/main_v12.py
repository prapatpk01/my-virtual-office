"""HMA Expert MTF V4.2 — clean single-pipeline production runtime."""
from __future__ import annotations

import asyncio

import main_v11 as v11
import strategy_v8 as S

_LOG = v11._LOG


def _v42_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace("HMA Expert MTF V4.1 Adaptive Sentinel", "HMA Expert MTF V4.2 Clean Sentinel")
        .replace("HMA Expert MTF V4.1", "HMA Expert MTF V4.2")
        .replace("HMA V4.1 Bot Stats", "HMA V4.2 Bot Stats")
    )


class Bot(v11.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV8(self.cfg.strategy_config())

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v42_text(text: str) -> bool:
            return await previous_send_text(_v42_text(text))

        async def v42_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v42_text(caption))

        self.tg.send_text = v42_text
        self.tg._send_photo = v42_photo

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        """Show only the authoritative V4.2 decision; remove legacy duplicate fields."""
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"V4.2 POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            self._view[symbol] = f"5M px={px:.6g} | {self.strat.entry_status(df4h, df1h, df15, df5)}"
        except Exception as exc:
            self._view[symbol] = f"V4.2 view error: {str(exc)[:120]}"

    async def start(self):
        await super().start()
        _LOG.info(
            "HMA V4.2 clean pipeline active: one decision object drives status and entry"
        )
        await self.tg.send_text(
            "🧭 *HMA V4.2 Clean Sentinel — ACTIVE*\n"
            "One decision pipeline: `Trend → Quality → Location → Execution → Score`\n"
            "Legacy 4H/HMA/Q fields are no longer duplicated in Railway status.\n"
            "Status and actual order eligibility now use the same evaluation result."
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v11.v10.v9.v8.v7.v5.v4.v3.base._signal, sig_name),
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
