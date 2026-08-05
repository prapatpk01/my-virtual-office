"""Production entry point for merged Trend Confirm + corrected WaveTrend entry.

One strategy family only:
- Layer 1: 4H trend direction
- Layer 2: 1H context + ADX/CHOP quality gate
- Layer 3: 15M EMA8/13 cross OR WaveTrend extreme cross
- 15M price must be on the correct side of EMA20

WaveTrend entry extremes used in production:
- Long cross from oversold <= -42
- Short cross from overbought >= +45

WT is an entry trigger inside Trend Confirm, not a second strategy. The existing
filename and Railway start command are retained.
"""
from __future__ import annotations

import asyncio
import os

import run_dual_bot  # keeps exchange, sleep, risk and lifecycle patches
import run_bot


def _make_merged_trend_confirm(symbols: list, config: dict):
    if not run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True):
        raise RuntimeError(
            "Merged Trend Confirm is disabled. Set ENABLE_TREND_CONFIRM=true"
        )

    from trading.strategies.trend_confirm_wt_fixed_strategy import (
        TrendConfirmWTFixedStrategy,
    )

    return [
        TrendConfirmWTFixedStrategy(
            symbol,
            wt_oversold=-42.0,
            wt_overbought=45.0,
        )
        for symbol in symbols
    ]


def _build_merged_config() -> dict:
    # Bypass the old dual quota sum. There is now only one strategy family.
    config = run_dual_bot._ORIGINAL_BUILD_CONFIG()
    if not run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True):
        raise RuntimeError(
            "Merged Trend Confirm is disabled. Set ENABLE_TREND_CONFIRM=true"
        )

    strategy_limit = max(1, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2")))
    requested_global = max(1, int(os.getenv("MAX_POSITIONS", str(strategy_limit))))
    config["max_positions"] = min(requested_global, strategy_limit)
    config["strategy_mode"] = "trend_confirm_ema_or_wt"
    config["enable_trend_confirm"] = True
    config["enable_wt_trend"] = False
    config["enable_ai_expert"] = False
    config["wt_oversold"] = -42.0
    config["wt_overbought"] = 45.0
    os.environ["CANDLE_TF"] = "15m"
    config["candle_tf"] = "15m"
    return config


run_bot._make_strategies = _make_merged_trend_confirm
run_bot.build_config = _build_merged_config


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
