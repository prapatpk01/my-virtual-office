"""Production entry point for merged Trend Confirm + WaveTrend entry.

The bot now loads one strategy family only:
- 4H Trend direction
- 1H Context + ADX/CHOP gate
- 15M entry: EMA8/13 cross OR WaveTrend extreme cross
- 15M price must remain on the correct side of EMA20

The existing filename and Railway start command are retained.
"""
from __future__ import annotations

import asyncio

import run_dual_bot  # keeps existing risk, sleep, exchange and lifecycle patches
import run_bot


def _make_merged_trend_confirm(symbols: list, config: dict):
    if not run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True):
        raise RuntimeError(
            "Merged Trend Confirm is disabled. Set ENABLE_TREND_CONFIRM=true"
        )

    from trading.strategies.trend_confirm_wt_strategy import TrendConfirmWTStrategy

    return [TrendConfirmWTStrategy(symbol) for symbol in symbols]


# Replace the old two-strategy factory. WT is now an entry trigger inside
# Trend Confirm, not a second strategy with separate 4H/1H state.
run_bot._make_strategies = _make_merged_trend_confirm


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
