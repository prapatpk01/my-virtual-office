"""Canonical HMA production entrypoint.

All future HMA changes must be applied through the canonical production
entrypoint/strategy instead of creating another ``main_vXX.py`` launcher.

The current implementation delegates to the proven V5.2 runtime while the
versioned implementation chain is retained as compatibility code.  Railway and
all operators should launch this file only.
"""
from __future__ import annotations

import asyncio
import signal

from main_v16 import Bot

BOT_VERSION = "5.2"
BOT_NAME = "HMA Gate Sentinel"


async def main() -> None:
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(signal, sig_name),
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
    asyncio.run(main())
