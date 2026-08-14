"""Sentinel-only production runner.

This wrapper intentionally disables every other strategy (including
TrendConfirm) without deleting their source files. Railway can switch back to
the normal multi-strategy runner later simply by changing the start command.

Sentinel V3 uses its own SignalState file so paper statistics start from a clean
V3 baseline instead of mixing V1/V2 trades into the new architecture results.
"""
import asyncio
import logging
import os

# V3 journal isolation MUST be set before trading.bot / signal_state are imported.
# If Railway explicitly supplies SIGNAL_STATE_FILE, respect that override.
os.environ.setdefault("SIGNAL_STATE_FILE", "/app/signal_state_sentinel_v3.json")

import run_bot

logger = logging.getLogger("run_sentinel_bot")


def _sentinel_only_strategies(symbols: list[str], config: dict):
    # Import through the strategies package so all Sentinel lifecycle overlays
    # (two-target management, structure-v2, log clarity, V3 quality gate) are installed.
    from trading.strategies import SentinelStrategy

    strategies = [SentinelStrategy(symbol) for symbol in symbols]
    logger.info(
        "SENTINEL V3 ONLY | %d strategy instances | symbols=%s | "
        "TrendConfirm=DISABLED | journal=%s",
        len(strategies),
        symbols,
        os.environ.get("SIGNAL_STATE_FILE"),
    )
    return strategies


# Make status/startup output unambiguous and prevent stale Railway strategy
# variables from selecting another strategy inside run_bot.
os.environ["STRATEGY"] = "sentinel"
os.environ["STRATEGY_AI_EXPERT"] = "false"
os.environ["STRATEGY_MCDX"] = "false"
os.environ["STRATEGY_WT_ADX"] = "false"

# run_bot.main() resolves this global at runtime when it builds crypto/forex
# bots, so replacing it here guarantees Sentinel is the sole strategy.
run_bot._make_strategies = _sentinel_only_strategies


if __name__ == "__main__":
    asyncio.run(run_bot.main())
