"""HMA Expert MTF V3.3 alert presentation layer.

Keeps V3.2 recovery safety and trading logic unchanged while ensuring:
- LONG entry alerts use a green marker.
- SHORT entry alerts use a red marker.
- HMA mode uses the local 5M chart renderer from ``hma_bot/chart_engine.py``.
"""
from __future__ import annotations

import asyncio

import main_v4 as v4


def _style_direction(text: str) -> str:
    """Apply directional color to entry captions without altering other alerts."""
    if not isinstance(text, str):
        return text
    upper = text.upper()
    if "*SHORT*" in upper or " SHORT*" in upper:
        if text.startswith("🟢"):
            return "🔴" + text[len("🟢"):]
    elif "*LONG*" in upper or " LONG*" in upper:
        if text.startswith("🔴"):
            return "🟢" + text[len("🔴"):]
    return text


class Bot(v4.Bot):
    def __init__(self):
        super().__init__()

        # main_v3 constructs the entry caption. Wrap Telegram delivery once so
        # every SHORT entry is red even when sendPhoto falls back to sendMessage.
        original_send_photo = self.tg._send_photo
        original_send_text = self.tg.send_text

        async def directional_photo(path: str, caption: str) -> bool:
            return await original_send_photo(path, _style_direction(caption))

        async def directional_text(text: str) -> bool:
            return await original_send_text(_style_direction(text))

        self.tg._send_photo = directional_photo
        self.tg.send_text = directional_text

    async def start(self):
        await super().start()
        v4.v3.base.logger.info(
            "HMA V3.3 alert layer active: local 5M chart + green LONG / red SHORT"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v4.v3.base._signal, sig_name),
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
