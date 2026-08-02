"""Dual-strategy production runner.

Runs AIExpertStrategy and TrendConfirmStrategy in one TradingBot process while
keeping independent position quotas:

- AI Expert:      AI_EXPERT_MAX_POSITIONS (default 1)
- Trend Confirm:  TREND_CONFIRM_MAX_POSITIONS (default 2)
- Global:         MAX_POSITIONS (default sum of the two quotas = 3)
- One live position per symbol across both strategies

The underlying run_bot.py remains the single source of truth for connectors,
Telegram, sleep mode, reconciliation, order execution and lifecycle handling.
"""
from __future__ import annotations

import asyncio
import os

import run_bot
from trading.risk_manager import RiskManager


def _strategy_family(strategy_key: str) -> str:
    """Map a live position key to its quota family.

    Keys may include hedge suffixes such as ``:L`` or ``:S``.
    """
    name = str(strategy_key or "")
    if name.endswith((":L", ":S")):
        name = name[:-2]
    if name.startswith("AIExpert("):
        return "ai_expert"
    if name.startswith("TrendConfirm("):
        return "trend_confirm"
    return "other"


def _install_dual_risk_limits() -> None:
    """Patch RiskManager.can_open with strategy quotas and symbol ownership."""
    if getattr(RiskManager, "_dual_limits_installed", False):
        return

    def _dual_can_open(self: RiskManager, symbol: str, strategy: str = ""):
        if self._halted:
            return False, "Trading halted: max drawdown reached"

        in_cd, remaining = self.in_cooldown(symbol)
        if in_cd:
            return False, (
                f"{symbol} cooldown after {self.max_consecutive_sl} consecutive losing closes "
                f"— resumes in {remaining/60:.0f} min"
            )

        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return False, f"{strategy} already has open position for {symbol}"

        # A symbol has one owner only. This prevents AI Expert and Trend Confirm
        # from opening duplicate or opposing positions on the same instrument.
        symbol_positions = [
            k for k in self._positions if k.startswith(f"{symbol}||")
        ]
        if symbol_positions:
            owner = symbol_positions[0].split("||", 1)[1]
            return False, f"{symbol} already managed by {owner}"

        family = _strategy_family(strategy)
        ai_limit = max(0, int(os.getenv("AI_EXPERT_MAX_POSITIONS", "1")))
        tc_limit = max(0, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2")))

        family_count = 0
        for position_key in self._positions:
            tracked_strategy = position_key.split("||", 1)[1] if "||" in position_key else ""
            if _strategy_family(tracked_strategy) == family:
                family_count += 1

        if family == "ai_expert" and family_count >= ai_limit:
            return False, f"AI Expert position quota reached ({family_count}/{ai_limit})"
        if family == "trend_confirm" and family_count >= tc_limit:
            return False, f"Trend Confirm position quota reached ({family_count}/{tc_limit})"

        if len(self._positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"

        return True, "ok"

    RiskManager.can_open = _dual_can_open
    RiskManager._dual_limits_installed = True


def _dual_make_strategies(symbols: list, config: dict):
    """Create both independent strategies for every configured symbol."""
    from trading.strategies.ai_expert_strategy import AIExpertStrategy
    from trading.strategies.trend_confirm_strategy import TrendConfirmStrategy

    strategies = []
    for symbol in symbols:
        strategies.append(AIExpertStrategy(
            symbol,
            min_confidence=config.get("ai_expert_min_confidence", 70.0),
            require_all_checks=config.get("ai_expert_strict", False),
        ))
        strategies.append(TrendConfirmStrategy(symbol))
    return strategies


def _dual_build_config() -> dict:
    config = _ORIGINAL_BUILD_CONFIG()
    config["strategy_mode"] = "dual"

    ai_limit = max(0, int(os.getenv("AI_EXPERT_MAX_POSITIONS", "1")))
    tc_limit = max(0, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2")))
    default_global = ai_limit + tc_limit
    config["max_positions"] = int(os.getenv("MAX_POSITIONS", str(default_global)))

    # Both strategies consume closed 15M candles as the runner base timeframe.
    os.environ["CANDLE_TF"] = "15m"
    config["candle_tf"] = "15m"
    return config


_ORIGINAL_BUILD_CONFIG = run_bot.build_config
_install_dual_risk_limits()
run_bot._make_strategies = _dual_make_strategies
run_bot.build_config = _dual_build_config


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
