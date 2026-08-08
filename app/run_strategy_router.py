"""Safe production strategy router.

BOT_STRATEGY_MODE controls the non-XAU strategy family explicitly:
  trend_confirm  -> Trend Confirm only
  adaptive       -> Adaptive Multi-Trigger only
  both           -> Trend Confirm + Adaptive Multi-Trigger

XAU/UTBot remains controlled independently by ENABLE_UTBOT_XAU.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("run_strategy_router")


def _normalize_mode(raw: str) -> str:
    value = str(raw or "").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "trend": "trend_confirm",
        "trendconfirm": "trend_confirm",
        "trend_confirm": "trend_confirm",
        "tc": "trend_confirm",
        "adaptive": "adaptive",
        "adaptive_multi_trigger": "adaptive",
        "adaptivemultitrigger": "adaptive",
        "both": "both",
        "combined": "both",
        "trend_confirm_adaptive": "both",
        "adaptive_trend_confirm": "both",
    }
    return aliases.get(value, value)


def _apply_strategy_mode() -> str:
    raw = os.getenv("BOT_STRATEGY_MODE", "trend_confirm")
    mode = _normalize_mode(raw)

    if mode == "trend_confirm":
        os.environ["ENABLE_TREND_CONFIRM"] = "true"
        os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "false"
    elif mode == "adaptive":
        os.environ["ENABLE_TREND_CONFIRM"] = "false"
        os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "true"
    elif mode == "both":
        os.environ["ENABLE_TREND_CONFIRM"] = "true"
        os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "true"
    else:
        raise RuntimeError(
            "BOT_STRATEGY_MODE must be trend_confirm, adaptive, or both; "
            f"got {raw!r}"
        )

    logger.warning(
        "[STRATEGY ROUTER] BOT_STRATEGY_MODE=%s -> TrendConfirm=%s Adaptive=%s UTBotXAU=%s",
        mode,
        os.environ.get("ENABLE_TREND_CONFIRM"),
        os.environ.get("ENABLE_ADAPTIVE_MULTI_TRIGGER"),
        os.environ.get("ENABLE_UTBOT_XAU", "false"),
    )
    return mode


_mode = _apply_strategy_mode()

# Install V5 before the combined runner captures Trend Confirm's factory.
# The ADX patch changes only Layer2 ADX contribution + detailed component log.
import trend_confirm_v5_patch  # noqa: E402,F401
import trading.strategies.trend_confirm_v5_adx_patch  # noqa: E402,F401
import run_trendconfirm_utbot  # noqa: E402,F401
import run_bot  # noqa: E402


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
