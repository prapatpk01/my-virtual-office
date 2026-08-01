"""HMA Expert MTF V3.4 — one-target runner runtime.

Keeps the V3.3 chart/directional-alert layer and V3.2 restart safety while
simplifying position management to:

    Target 1 +0.8% -> SL +0.5%
    Runner          -> final TP (default +1.5%)

No second target or second lock stage remains.
"""
from __future__ import annotations

import asyncio

import main_v5 as v5
import strategy_v4 as S


def _one_target_text(text: str) -> str:
    """Update inherited Telegram wording to the active one-target model."""
    if not isinstance(text, str):
        return text

    replacements = (
        (
            "HMA Expert MTF V3.2 Recovery-Safe",
            "HMA Expert MTF V3.4 One-Target Runner",
        ),
        (
            "T1 `+0.6%` → lock `+0.3%` | T2 `+1.0%` → lock `+0.7%`",
            "Target 1 `+0.8%` → lock `+0.5%` | Runner → Final TP",
        ),
        (
            "T1 +0.6%→lock +0.3% | T2 +1.0%→lock +0.7% | TP +1.5%",
            "Target 1 +0.8%→lock +0.5% | Runner→TP +1.5%",
        ),
        (
            " T1 reached* `+0.6%`",
            " Target 1 reached* `+0.8%`",
        ),
        (
            "SL moved to lock `+0.3%`",
            "SL moved to lock `+0.5%`\nRunner active to final TP",
        ),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


class Bot(v5.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV4(self.cfg.strategy_config())

        # Preserve the V3.3 LONG/SHORT color styling and chart delivery, then
        # rewrite all inherited T1/T2 wording to the one-target runner model.
        original_send_photo = self.tg._send_photo
        original_send_text = self.tg.send_text

        async def one_target_photo(path: str, caption: str) -> bool:
            return await original_send_photo(path, _one_target_text(caption))

        async def one_target_text(text: str) -> bool:
            return await original_send_text(_one_target_text(text))

        self.tg._send_photo = one_target_photo
        self.tg.send_text = one_target_text

    async def start(self):
        await super().start()
        v5.v4.v3.base.logger.info(
            "HMA V3.4 one-target runner active: +0.8%% -> lock +0.5%% -> final TP"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v5.v4.v3.base._signal, sig_name),
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
