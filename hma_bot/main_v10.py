"""HMA Expert MTF V4.0 — Sentinel X location-aware production runtime.

Inherits V3.7 production infrastructure:
- FX 24/5 new-entry gate
- OKX restart recovery and native SL/TP sync
- auditable OKX stats
- 5M chart alerts
- balanced two-stage profit locks

Only the strategy/context layer changes to Sentinel X v2.3 adaptive S/R.
"""
from __future__ import annotations

import asyncio

import main_v9 as v9
import strategy_v6 as S

_LOG = v9._LOG


def _v40_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace(
            "HMA Expert MTF V3.7 FX-24/5 Two-Stage",
            "HMA Expert MTF V4.0 Sentinel Location-Aware",
        )
        .replace("HMA Expert MTF V3.7", "HMA Expert MTF V4.0")
        .replace("HMA V3.7 Bot Stats", "HMA V4.0 Bot Stats")
    )


class Bot(v9.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV6(self.cfg.strategy_config())

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v40_text(text: str) -> bool:
            return await previous_send_text(_v40_text(text))

        async def v40_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v40_text(caption))

        self.tg.send_text = v40_text
        self.tg._send_photo = v40_photo

    async def start(self):
        await super().start()
        _LOG.info(
            "HMA V4.0 Sentinel context active: strong trend uses S1/S2 or R1/R2; "
            "moderate trend uses S2/R2; location score >= %.0f",
            self.strat.sentinel_location_min,
        )
        await self.tg.send_text(
            "🧭 *Sentinel X Location Engine — ACTIVE*\n"
            "15M adaptive zones: `S1 / S2 / R1 / R2`\n"
            "Strong trend: `LONG S1/S2 · SHORT R1/R2`\n"
            "Moderate trend: `LONG S2 · SHORT R2`\n"
            f"Minimum location score: `{self.strat.sentinel_location_min:.0f}/100`\n"
            "Execution remains recent closed-5M EMA8/13, micro BOS/CHOCH or continuation."
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v9.v8.v7.v5.v4.v3.base._signal, sig_name),
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
