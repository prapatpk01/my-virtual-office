"""HMA Expert MTF V4.3 — Sentinel S/R entry-zone production runtime."""
from __future__ import annotations

import asyncio

import main_v12 as v12
import strategy_v9 as S

_LOG = v12._LOG


def _v43_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace(
            "HMA Expert MTF V4.2 Clean Sentinel",
            "HMA Expert MTF V4.3 Sentinel S/R Entry",
        )
        .replace("HMA Expert MTF V4.2", "HMA Expert MTF V4.3")
        .replace("HMA V4.2 Bot Stats", "HMA V4.3 Bot Stats")
    )


class Bot(v12.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV9(self.cfg.strategy_config())

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v43_text(text: str) -> bool:
            return await previous_send_text(_v43_text(text))

        async def v43_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v43_text(caption))

        self.tg.send_text = v43_text
        self.tg._send_photo = v43_photo

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"V4.3 POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            self._view[symbol] = (
                f"5M px={px:.6g} | "
                f"{self.strat.entry_status(df4h, df1h, df15, df5)}"
            )
        except Exception as exc:
            self._view[symbol] = f"V4.3 view error: {str(exc)[:120]}"

    async def start(self):
        await super().start()
        _LOG.info(
            "HMA V4.3 Sentinel S/R active: long S1/S2, short R1/R2; "
            "moderate first-level min=%.0f rejection=%s",
            self.strat.moderate_first_level_min,
            self.strat.first_level_requires_rejection,
        )
        await self.tg.send_text(
            "🧭 *HMA V4.3 Sentinel S/R Entry — ACTIVE*\n"
            "15M Sentinel levels are now the authoritative entry zones.\n"
            "LONG: `S1/S2` | SHORT: `R1/R2`\n"
            "S/R only arms a setup; a recent closed-5M EMA8/13, BOS/CHOCH or continuation trigger is still required.\n"
            f"MODERATE S1/R1 requires Location `≥{self.strat.moderate_first_level_min:.0f}` plus rejection/sweep.\n"
            "EARLY uses only S2/R2; STRONG can use S1/S2 or R1/R2."
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal, sig_name),
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
