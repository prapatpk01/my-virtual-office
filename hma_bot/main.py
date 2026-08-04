"""Canonical HMA Gate Sentinel production entrypoint.

Railway launches this file for MODE=hma. Version numbers live in BOT_VERSION;
new releases must update the implementation instead of creating main_vXX.py.
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
