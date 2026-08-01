"""HMA Expert MTF V3.5 — balanced two-stage runner runtime.

Keeps V3.3 chart/directional alerts and V3.2 restart safety while using:

    Stage 1 +0.7% -> SL +0.4%
    Stage 2 +1.1% -> SL +0.75%
    Runner         -> final TP (default +1.5%)

No partial close is performed at either stage.
"""
from __future__ import annotations

import asyncio

import main_v5 as v5
import strategy_v5 as S


def _two_stage_text(text: str) -> str:
    """Rewrite inherited labels to the active V3.5 two-stage model."""
    if not isinstance(text, str):
        return text

    replacements = (
        (
            "HMA Expert MTF V3.2 Recovery-Safe",
            "HMA Expert MTF V3.5 Balanced Two-Stage",
        ),
        (
            "T1 `+0.6%` → lock `+0.3%` | T2 `+1.0%` → lock `+0.7%`",
            "Stage 1 `+0.7%` → lock `+0.4%` | Stage 2 `+1.1%` → lock `+0.75%`",
        ),
        (
            "T1 +0.6%→lock +0.3% | T2 +1.0%→lock +0.7% | TP +1.5%",
            "Stage 1 +0.7%→lock +0.4% | Stage 2 +1.1%→lock +0.75% | TP +1.5%",
        ),
        (
            " T1 reached* `+0.6%`",
            " Stage 1 reached* `+0.7%`",
        ),
        (
            "SL moved to lock `+0.3%`",
            "SL moved to lock `+0.4%`",
        ),
        (
            " T2 reached* `+1.0%`",
            " Stage 2 reached* `+1.1%`",
        ),
        (
            "SL moved to lock `+0.7%`",
            "SL moved to lock `+0.75%`",
        ),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


class Bot(v5.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV5(self.cfg.strategy_config())

        # V3.3 already styles LONG green / SHORT red and delivers the 5M chart.
        # This wrapper only updates stage labels and values.
        original_send_photo = self.tg._send_photo
        original_send_text = self.tg.send_text

        async def two_stage_photo(path: str, caption: str) -> bool:
            return await original_send_photo(path, _two_stage_text(caption))

        async def two_stage_text(text: str) -> bool:
            return await original_send_text(_two_stage_text(text))

        self.tg._send_photo = two_stage_photo
        self.tg.send_text = two_stage_text

    async def start(self):
        await super().start()
        v5.v4.v3.base.logger.info(
            "HMA V3.5 balanced two-stage active: "
            "+0.7%% -> lock +0.4%% | +1.1%% -> lock +0.75%% | final TP"
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
