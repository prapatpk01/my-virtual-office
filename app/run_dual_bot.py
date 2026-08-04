"""Dual-strategy production runner.

Runs WTTrendEntryStrategy and TrendConfirmStrategy in one TradingBot process.

- WT Trend Entry: ENABLE_WT_TREND + WT_TREND_MAX_POSITIONS (default 1)
- Trend Confirm:  ENABLE_TREND_CONFIRM + TREND_CONFIRM_MAX_POSITIONS (default 2)
- Global:         MAX_POSITIONS, capped by enabled-strategy quota sum
- Per symbol:     MAX_POSITIONS_PER_SYMBOL (default 2)

Compatibility: when the new WT variables are absent, the runner falls back to
ENABLE_AI_EXPERT and AI_EXPERT_MAX_POSITIONS so an existing Railway deployment
can transition without failing its first restart.
"""
from __future__ import annotations

import asyncio
import os

import run_bot
from trading.risk_manager import RiskManager


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"{name} must be true/false, got {raw!r}")


def _wt_enabled() -> bool:
    if os.getenv("ENABLE_WT_TREND") is not None:
        return _env_bool("ENABLE_WT_TREND", True)
    return _env_bool("ENABLE_AI_EXPERT", True)


def _wt_limit() -> int:
    raw = os.getenv("WT_TREND_MAX_POSITIONS")
    if raw is None:
        raw = os.getenv("AI_EXPERT_MAX_POSITIONS", "1")
    return max(0, int(raw))


def _strip_side_suffix(strategy_key: str) -> str:
    name = str(strategy_key or "")
    return name[:-2] if name.endswith((":L", ":S")) else name


def _strategy_family(strategy_key: str) -> str:
    name = _strip_side_suffix(strategy_key)
    if name.startswith("WTTrendEntry("):
        return "wt_trend"
    if name.startswith("TrendConfirm("):
        return "trend_confirm"
    return "other"


def _strategy_side(strategy_key: str, position=None) -> str:
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
        enable_wt = _wt_enabled()
        enable_tc = _env_bool("ENABLE_TREND_CONFIRM", True)
        if candidate_family == "wt_trend" and not enable_wt:
            return False, "WT Trend Entry disabled by ENABLE_WT_TREND=false"
        if candidate_family == "trend_confirm" and not enable_tc:
            return False, "Trend Confirm disabled by ENABLE_TREND_CONFIRM=false"

        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return False, f"{strategy} already has open position for {symbol}"

        candidate_side = _strategy_side(strategy)
        per_symbol_limit = max(1, int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "2")))
        symbol_positions = []
        for position_key, position in self._positions.items():
            if position_key.startswith(f"{symbol}||"):
                tracked_strategy = position_key.split("||", 1)[1]
                symbol_positions.append((tracked_strategy, position))

        if len(symbol_positions) >= per_symbol_limit:
            return False, f"{symbol} per-symbol position limit reached ({len(symbol_positions)}/{per_symbol_limit})"

        for tracked_strategy, _position in symbol_positions:
            if _strategy_family(tracked_strategy) == candidate_family:
                return False, f"{candidate_family} already has a position for {symbol}"

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

        wt_limit = _wt_limit()
        tc_limit = max(0, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2")))
        family_count = 0
        for position_key in self._positions:
            tracked_strategy = position_key.split("||", 1)[1] if "||" in position_key else ""
            if _strategy_family(tracked_strategy) == candidate_family:
                family_count += 1

        if candidate_family == "wt_trend" and family_count >= wt_limit:
            return False, f"WT Trend position quota reached ({family_count}/{wt_limit})"
        if candidate_family == "trend_confirm" and family_count >= tc_limit:
            return False, f"Trend Confirm position quota reached ({family_count}/{tc_limit})"

        if len(self._positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"
        return True, "ok"

    RiskManager.can_open = _dual_can_open
    RiskManager._dual_limits_installed = True


def _dual_make_strategies(symbols: list, config: dict):
    enable_wt = _wt_enabled()
    enable_tc = _env_bool("ENABLE_TREND_CONFIRM", True)
    if not enable_wt and not enable_tc:
        raise RuntimeError("No strategy enabled: set ENABLE_WT_TREND=true and/or ENABLE_TREND_CONFIRM=true")

    strategies = []
    if enable_wt:
        from trading.strategies.wt_trend_entry_strategy import WTTrendEntryStrategy
        strategies.extend(WTTrendEntryStrategy(symbol) for symbol in symbols)
    if enable_tc:
        from trading.strategies.trend_confirm_strategy import TrendConfirmStrategy
        strategies.extend(TrendConfirmStrategy(symbol) for symbol in symbols)
    return strategies


def _dual_build_config() -> dict:
    config = _ORIGINAL_BUILD_CONFIG()
    enable_wt = _wt_enabled()
    enable_tc = _env_bool("ENABLE_TREND_CONFIRM", True)
    if not enable_wt and not enable_tc:
        raise RuntimeError("No strategy enabled: set ENABLE_WT_TREND=true and/or ENABLE_TREND_CONFIRM=true")

    wt_limit = _wt_limit() if enable_wt else 0
    tc_limit = max(0, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2"))) if enable_tc else 0
    enabled_quota_sum = wt_limit + tc_limit
    if enabled_quota_sum <= 0:
        raise RuntimeError("Enabled strategies have zero total position quota")

    requested_global = max(1, int(os.getenv("MAX_POSITIONS", str(enabled_quota_sum))))
    config["max_positions"] = min(requested_global, enabled_quota_sum)
    config["strategy_mode"] = "dual" if enable_wt and enable_tc else "wt_trend" if enable_wt else "trend_confirm"
    config["enable_wt_trend"] = enable_wt
    config["enable_trend_confirm"] = enable_tc
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
