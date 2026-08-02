"""Dual-strategy production runner.

Runs AIExpertStrategy and TrendConfirmStrategy in one TradingBot process while
keeping independent position quotas:

- AI Expert:      ENABLE_AI_EXPERT + AI_EXPERT_MAX_POSITIONS (default 1)
- Trend Confirm:  ENABLE_TREND_CONFIRM + TREND_CONFIRM_MAX_POSITIONS (default 2)
- Global:         MAX_POSITIONS, capped by the enabled-strategy quota sum
- Per symbol:     MAX_POSITIONS_PER_SYMBOL (default 2)

Safe hedge rule:
- One position per strategy family per symbol.
- A second position on the same symbol is allowed only from the other strategy
  and only on the opposite side (LONG + SHORT).
- Same-side duplicate positions are blocked because OKX aggregates positions
  on the same side, which would make strategy ownership and TP/SL ambiguous.

The underlying run_bot.py remains the single source of truth for connectors,
Telegram, sleep mode, reconciliation, order execution and lifecycle handling.
"""
from __future__ import annotations

import asyncio
import os

import run_bot
from trading.risk_manager import RiskManager


def _env_bool(name: str, default: bool = True) -> bool:
    """Read a Railway boolean variable safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(
        f"{name} must be true/false, got {raw!r}"
    )


def _strip_side_suffix(strategy_key: str) -> str:
    """Remove the hedge-side suffix from a live strategy key."""
    name = str(strategy_key or "")
    if name.endswith((":L", ":S")):
        return name[:-2]
    return name


def _strategy_family(strategy_key: str) -> str:
    """Map a live position key to its quota family."""
    name = _strip_side_suffix(strategy_key)
    if name.startswith("AIExpert("):
        return "ai_expert"
    if name.startswith("TrendConfirm("):
        return "trend_confirm"
    return "other"


def _strategy_side(strategy_key: str, position=None) -> str:
    """Resolve long/short from hedge suffix, falling back to Position.side."""
    key = str(strategy_key or "")
    if key.endswith(":L"):
        return "long"
    if key.endswith(":S"):
        return "short"
    side = str(getattr(position, "side", "") or "").lower()
    if side in ("buy", "long"):
        return "long"
    if side in ("sell", "short"):
        return "short"
    return ""


def _install_dual_risk_limits() -> None:
    """Patch RiskManager.can_open with dual quotas and safe hedge rules."""
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

        candidate_family = _strategy_family(strategy)
        enable_ai = _env_bool("ENABLE_AI_EXPERT", True)
        enable_tc = _env_bool("ENABLE_TREND_CONFIRM", True)
        if candidate_family == "ai_expert" and not enable_ai:
            return False, "AI Expert is disabled by ENABLE_AI_EXPERT=false"
        if candidate_family == "trend_confirm" and not enable_tc:
            return False, "Trend Confirm is disabled by ENABLE_TREND_CONFIRM=false"

        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return False, f"{strategy} already has open position for {symbol}"

        candidate_side = _strategy_side(strategy)
        per_symbol_limit = max(1, int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "2")))

        symbol_positions = []
        for position_key, position in self._positions.items():
            if not position_key.startswith(f"{symbol}||"):
                continue
            tracked_strategy = position_key.split("||", 1)[1]
            symbol_positions.append((tracked_strategy, position))

        if len(symbol_positions) >= per_symbol_limit:
            return False, (
                f"{symbol} per-symbol position limit reached "
                f"({len(symbol_positions)}/{per_symbol_limit})"
            )

        # Each strategy family may own only one position per symbol.
        for tracked_strategy, _position in symbol_positions:
            if _strategy_family(tracked_strategy) == candidate_family:
                return False, f"{candidate_family} already has a position for {symbol}"

        # A second position on a symbol must be a true opposite-side hedge.
        if symbol_positions:
            if candidate_side not in ("long", "short"):
                return False, f"Cannot determine hedge side for {strategy}"
            for tracked_strategy, position in symbol_positions:
                existing_side = _strategy_side(tracked_strategy, position)
                if existing_side == candidate_side:
                    return False, (
                        f"{symbol} already has {existing_side.upper()} exposure — "
                        "second strategy must take the opposite side for a hedge"
                    )

        ai_limit = max(0, int(os.getenv("AI_EXPERT_MAX_POSITIONS", "1")))
        tc_limit = max(0, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2")))

        family_count = 0
        for position_key in self._positions:
            tracked_strategy = (
                position_key.split("||", 1)[1] if "||" in position_key else ""
            )
            if _strategy_family(tracked_strategy) == candidate_family:
                family_count += 1

        if candidate_family == "ai_expert" and family_count >= ai_limit:
            return False, f"AI Expert position quota reached ({family_count}/{ai_limit})"
        if candidate_family == "trend_confirm" and family_count >= tc_limit:
            return False, f"Trend Confirm position quota reached ({family_count}/{tc_limit})"

        if len(self._positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"

        return True, "ok"

    RiskManager.can_open = _dual_can_open
    RiskManager._dual_limits_installed = True


def _dual_make_strategies(symbols: list, config: dict):
    """Create only the strategy families enabled in Railway variables."""
    enable_ai = _env_bool("ENABLE_AI_EXPERT", True)
    enable_tc = _env_bool("ENABLE_TREND_CONFIRM", True)

    if not enable_ai and not enable_tc:
        raise RuntimeError(
            "No strategy enabled: set ENABLE_AI_EXPERT=true and/or "
            "ENABLE_TREND_CONFIRM=true"
        )

    strategies = []
    if enable_ai:
        from trading.strategies.ai_expert_strategy import AIExpertStrategy
        for symbol in symbols:
            strategies.append(AIExpertStrategy(
                symbol,
                min_confidence=config.get("ai_expert_min_confidence", 70.0),
                require_all_checks=config.get("ai_expert_strict", False),
            ))

    if enable_tc:
        from trading.strategies.trend_confirm_strategy import TrendConfirmStrategy
        for symbol in symbols:
            strategies.append(TrendConfirmStrategy(symbol))

    return strategies


def _dual_build_config() -> dict:
    config = _ORIGINAL_BUILD_CONFIG()

    enable_ai = _env_bool("ENABLE_AI_EXPERT", True)
    enable_tc = _env_bool("ENABLE_TREND_CONFIRM", True)
    if not enable_ai and not enable_tc:
        raise RuntimeError(
            "No strategy enabled: set ENABLE_AI_EXPERT=true and/or "
            "ENABLE_TREND_CONFIRM=true"
        )

    ai_limit = max(0, int(os.getenv("AI_EXPERT_MAX_POSITIONS", "1"))) if enable_ai else 0
    tc_limit = max(0, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2"))) if enable_tc else 0
    enabled_quota_sum = ai_limit + tc_limit
    if enabled_quota_sum <= 0:
        raise RuntimeError(
            "Enabled strategies have zero total position quota; increase "
            "AI_EXPERT_MAX_POSITIONS or TREND_CONFIRM_MAX_POSITIONS"
        )

    requested_global = max(1, int(os.getenv("MAX_POSITIONS", str(enabled_quota_sum))))
    config["max_positions"] = min(requested_global, enabled_quota_sum)
    config["strategy_mode"] = (
        "dual" if enable_ai and enable_tc
        else "ai_expert" if enable_ai
        else "trend_confirm"
    )
    config["enable_ai_expert"] = enable_ai
    config["enable_trend_confirm"] = enable_tc

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
