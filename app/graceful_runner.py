"""Graceful entrypoint for Adaptive SMC v14.

Tracks exchange-backed connector instances created by run_bot and explicitly
closes them before asyncio.run() closes the event loop. This prevents CCXT's
"requires to release all resources" warning and aiohttp's unclosed session /
connector errors during Railway redeploys and SIGTERM shutdowns.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

import run_bot
from trading.connectors.binance_conn import BinanceConnector

logger = logging.getLogger("adaptive_smc_shutdown")
_TRACKED: list[Any] = []


def _track_class(cls: type) -> None:
    original = cls.__init__
    if getattr(original, "_adaptive_shutdown_wrapped", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _TRACKED.append(self)

    wrapped._adaptive_shutdown_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = wrapped  # type: ignore[method-assign]


async def _close_target(target: Any, label: str) -> bool:
    if target is None:
        return False
    close = getattr(target, "close", None)
    if not callable(close):
        return False
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
        logger.info("Closed %s", label)
        return True
    except Exception as exc:
        logger.warning("Failed to close %s: %s", label, exc)
        return False


async def _close_resource(resource: Any) -> None:
    seen: set[int] = set()
    candidates = (
        (resource, type(resource).__name__),
        (getattr(resource, "_exchange", None), f"{type(resource).__name__}._exchange"),
        (getattr(resource, "exchange", None), f"{type(resource).__name__}.exchange"),
        (getattr(resource, "client", None), f"{type(resource).__name__}.client"),
    )
    for target, label in candidates:
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        if await _close_target(target, label):
            break


async def main() -> None:
    _track_class(BinanceConnector)

    # Live mode creates this class lazily inside run_bot.main(). Track it too
    # when the module is available; paper mode remains unaffected.
    try:
        from trading.connectors.okx_adapter import OKXAdapter
        _track_class(OKXAdapter)
    except Exception:
        pass

    try:
        await run_bot.main()
    finally:
        for resource in reversed(_TRACKED):
            await _close_resource(resource)
        # Give aiohttp transports one event-loop turn to finish closing.
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
