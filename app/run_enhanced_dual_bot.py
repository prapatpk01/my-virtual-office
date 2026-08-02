"""Production entry point for Dual Bot with Safe Enhanced AI Expert.

Imports run_dual_bot first so all existing quota, hedge, enable/disable and
configuration patches remain active. It then replaces only the strategy
factory so AI Expert instances use SafeEnhancedAIExpertStrategy.
"""
from __future__ import annotations

import asyncio

import run_dual_bot
import run_bot


def _enhanced_make_strategies(symbols: list, config: dict):
    enable_ai = run_dual_bot._env_bool("ENABLE_AI_EXPERT", True)
    enable_tc = run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True)

    if not enable_ai and not enable_tc:
        raise RuntimeError(
            "No strategy enabled: set ENABLE_AI_EXPERT=true and/or "
            "ENABLE_TREND_CONFIRM=true"
        )

    strategies = []
    if enable_ai:
        from trading.strategies.safe_enhanced_ai_expert_strategy import (
            SafeEnhancedAIExpertStrategy,
        )
        for symbol in symbols:
            strategies.append(SafeEnhancedAIExpertStrategy(
                symbol,
                min_confidence=config.get("ai_expert_min_confidence", 70.0),
                require_all_checks=config.get("ai_expert_strict", False),
            ))

    if enable_tc:
        from trading.strategies.trend_confirm_strategy import TrendConfirmStrategy
        for symbol in symbols:
            strategies.append(TrendConfirmStrategy(symbol))

    return strategies


run_bot._make_strategies = _enhanced_make_strategies


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
