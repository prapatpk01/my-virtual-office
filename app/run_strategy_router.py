"""Canonical production router for Sentinel V2.

Railway starts this file. Legacy strategies remain in the repository for
comparison/backtests but are not instantiated in production.
"""
from __future__ import annotations

import asyncio
import logging
import os

import run_bot
from trading.bot import TradingBot
from trading.strategies.sentinel_v2_strategy import SentinelV2Strategy

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
    # Keep the existing env name for Railway backward compatibility.
    config["symbols"] = _env_symbols("SIMPLE_PRECISION_SYMBOLS", config.get("symbols") or [])
    # Keep strategy_mode stable so surrounding execution code sees no mode migration.
    config["strategy_mode"] = "simple_precision"
    config["candle_tf"] = "15m"
    os.environ["CANDLE_TF"] = "15m"
    logger.warning(
        "[PRODUCTION CONFIG] Sentinel V%s | symbols=%s | "
        "4H direction -> 1H quality -> 15M structure/location + 3 triggers",
        SentinelV2Strategy.VERSION,
        config["symbols"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV2Strategy]:
    strategies = [
        SentinelV2Strategy(
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


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V2":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    macro = meta.get("macro_4h") or {}
    quality = meta.get("quality_1h") or {}
    entry = meta.get("entry_15m") or {}
    structure = entry.get("structure") or {}
    location = entry.get("location") or {}
    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL] %s px=%.4f sig=%s | 4H=%s score=%s | "
        "1H Q=%s/%s ADX=%s CHOP=%s blocks=%s | "
        "15M trigger=%s struct=%s dist=%sATR room=%sR fast=%s tp2=%s | %s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        macro.get("direction", "WAIT"),
        macro.get("score", "?"),
        quality.get("score", "?"),
        quality.get("threshold", "?"),
        quality.get("adx", "?"),
        quality.get("chop", "?"),
        ",".join(quality.get("hard_blocks", [])) or "none",
        entry.get("trigger", "WAIT"),
        structure.get("label", "?"),
        entry.get("distance_atr", "?"),
        entry.get("room_r", location.get("room_r", "?")),
        entry.get("fast_impulse", "?"),
        entry.get("target_source", "?"),
        getattr(signal, "reason", ""),
    )


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _sentinel_log_scan

logger.warning(
    "[PRODUCTION] Sentinel V%s installed; Simple Precision risk/exits retained; legacy strategies disabled",
    SentinelV2Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
