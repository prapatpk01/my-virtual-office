"""Canonical production router for Sentinel V3.1 — 15M Unified.

Railway starts this file. Legacy strategies remain in the repository for
comparison/backtests but are not instantiated in production.
"""
from __future__ import annotations

import asyncio
import logging
import os

import run_bot
from trading.bot import TradingBot
from trading.strategies.sentinel_v31_strategy import SentinelV31Strategy

logger = logging.getLogger("run_strategy_router")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_symbols(name: str, fallback: list[str]) -> list[str]:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return list(dict.fromkeys(fallback))
    return list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))


_BASE_BUILD_CONFIG = run_bot.build_config


def _build_config() -> dict:
    config = _BASE_BUILD_CONFIG()
    # Keep existing Railway env names for backward compatibility.
    config["symbols"] = _env_symbols("SIMPLE_PRECISION_SYMBOLS", config.get("symbols") or [])
    config["strategy_mode"] = "simple_precision"
    config["candle_tf"] = "15m"
    os.environ["CANDLE_TF"] = "15m"
    logger.warning(
        "[PRODUCTION CONFIG] Sentinel V%s | symbols=%s | 15M only: quality + Sentinel X + trigger direction",
        SentinelV31Strategy.VERSION,
        config["symbols"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV31Strategy]:
    strategies = [
        SentinelV31Strategy(
            symbol,
            quality_threshold=_env_float("SP_QUALITY_THRESHOLD", 55.0),
            adx_min=_env_float("SP_ADX_MIN", 15.0),
            chop_max=_env_float("SP_CHOP_MAX", 62.0),
            max_entry_distance_atr=_env_float("SP_MAX_ENTRY_DISTANCE_ATR", 1.50),
            min_room_r=_env_float("SP_MIN_ROOM_R", 1.20),
            stop_atr_min=_env_float("SP_STOP_ATR_MIN", 0.70),
            stop_atr_max=_env_float("SP_STOP_ATR_MAX", 1.40),
            target_r=_env_float("SP_TARGET_R", 2.0),
            tp1_r=_env_float("SP_TP1_R", 1.0),
            tp1_trim_pct=_env_float("SP_TP1_TRIM_PCT", 0.40),
            exit_cooldown_bars=_env_int("SP_EXIT_COOLDOWN_BARS", 2),
        )
        for symbol in symbols
    ]
    if not strategies:
        raise RuntimeError("SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


_ORIGINAL_LOG_SCAN = TradingBot._log_scan
_LAST_SENTINEL_SCAN: dict[str, dict] = {}


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V3":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    reason = str(getattr(signal, "reason", "") or "")
    has_metrics = bool(meta.get("market_quality_15m") or meta.get("entry_15m") or meta.get("structure_15m"))
    if has_metrics:
        _LAST_SENTINEL_SCAN[symbol] = meta

    repeated_bar = reason == "15M bar already evaluated"
    view = _LAST_SENTINEL_SCAN.get(symbol, meta) if repeated_bar else meta

    market = view.get("market_quality_15m") or {}
    entry = view.get("entry_15m") or {}
    structure = entry.get("structure") or view.get("structure_15m") or {}

    q_value = entry.get("quality_score", market.get("score", "-"))
    q_threshold = entry.get("quality_threshold", "-")
    trigger_value = entry.get("candidate_trigger", entry.get("trigger", "WAIT"))
    direction_value = entry.get("direction", "WAIT")
    repeat_tag = " | cached=same-15M-bar" if repeated_bar and view is not meta else ""

    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL] %s px=%.4f sig=%s | "
        "15M Q=%s/%s ADX=%s CHOP=%s | "
        "trigger=%s dir=%s struct=%s room=%sR dist=%sATR fast=%s tp2=%s blocks=%s | %s%s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        q_value,
        q_threshold,
        market.get("adx", "-"),
        market.get("chop", "-"),
        trigger_value,
        direction_value,
        structure.get("label", "-"),
        entry.get("room_r", "-"),
        entry.get("distance_atr", "-"),
        entry.get("fast_impulse", "-"),
        entry.get("target_source", "-"),
        ",".join(entry.get("hard_blocks", [])) or ",".join(market.get("hard_blocks", [])) or "none",
        reason,
        repeat_tag,
    )


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _sentinel_log_scan

logger.warning(
    "[PRODUCTION] Sentinel V%s installed; 15M unified + structure-aware hold + same-side rearm",
    SentinelV31Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
