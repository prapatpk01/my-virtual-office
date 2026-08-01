"""HMA Expert MTF V3.5 — balanced two-stage runner runtime.

Keeps restart safety while using:

    Stage 1 +0.7% -> SL +0.4%
    Stage 2 +1.1% -> SL +0.75%
    Runner         -> final TP (default +1.5%)

Entry alerts use the local Matplotlib 5M chart renderer. Photo delivery is sent
without Telegram Markdown parsing so a caption-format error can never downgrade
the alert to text-only.
"""
from __future__ import annotations

import asyncio
import os

import aiohttp

import main_v5 as v5
import strategy_v5 as S
from chart_engine import build_entry_chart as hma_build_entry_chart


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


def _plain_caption(text: str) -> str:
    """Remove Markdown tokens and stay safely below Telegram's 1024 limit."""
    text = v5._style_direction(_two_stage_text(text))
    text = text.replace("`", "").replace("*", "")
    if len(text) > 1000:
        text = text[:997].rstrip() + "..."
    return text


class Bot(v5.Bot):
    def __init__(self):
        # Force main_v3's direct base.build_entry_chart call to use the HMA-local
        # renderer, independent of Python import order or module shadowing.
        v5.v4.v3.base.build_entry_chart = hma_build_entry_chart
        super().__init__()
        self.strat = S.PrecisionTrendStructureV5(self.cfg.strategy_config())

        original_send_text = self.tg.send_text

        async def two_stage_text(text: str) -> bool:
            return await original_send_text(_two_stage_text(text))

        async def reliable_photo(path: str, caption: str) -> bool:
            """Send a real photo without Markdown parse-mode failure fallback."""
            clean_caption = _plain_caption(caption)
            if not self.tg.enabled:
                return False

            try:
                if not path or not os.path.isfile(path) or os.path.getsize(path) <= 0:
                    v5.v4.v3.base.logger.warning(
                        "[HMA CHART] invalid output before Telegram send: %s", path
                    )
                    return await original_send_text(clean_caption)

                with open(path, "rb") as file_obj:
                    photo_bytes = file_obj.read()

                url = f"https://api.telegram.org/bot{self.tg.token}/sendPhoto"
                form = aiohttp.FormData()
                form.add_field("chat_id", str(self.tg.chat_id))
                form.add_field("caption", clean_caption)
                form.add_field(
                    "photo",
                    photo_bytes,
                    filename="hma_entry_chart.png",
                    content_type="image/png",
                )

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        data=form,
                        timeout=aiohttp.ClientTimeout(total=35),
                    ) as response:
                        if response.status == 200:
                            v5.v4.v3.base.logger.info(
                                "[HMA CHART] Telegram photo sent bytes=%d", len(photo_bytes)
                            )
                            return True

                        body = await response.text()
                        v5.v4.v3.base.logger.warning(
                            "[HMA CHART] Telegram sendPhoto failed status=%s body=%s",
                            response.status,
                            body[:300],
                        )
            except Exception as exc:
                v5.v4.v3.base.logger.error(
                    "[HMA CHART] Telegram photo exception: %s", exc, exc_info=True
                )
            finally:
                try:
                    if path and os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass

            # The alert is never lost. Text fallback happens only after the full
            # photo error is written to Railway logs.
            return await original_send_text(clean_caption)

        self.tg.send_text = two_stage_text
        self.tg._send_photo = reliable_photo

    async def start(self):
        await super().start()
        v5.v4.v3.base.logger.info(
            "HMA V3.5 balanced two-stage active: "
            "+0.7%% -> lock +0.4%% | +1.1%% -> lock +0.75%% | final TP"
        )
        v5.v4.v3.base.logger.info(
            "HMA chart pipeline active: renderer=%s; Telegram photo uses plain caption",
            hma_build_entry_chart.__module__,
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
