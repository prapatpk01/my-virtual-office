"""Canonical production router for Sentinel V4.4 — Responsive PA + Position Defense.

Railway starts this file. Legacy strategies remain in the repository for
comparison/backtests but are not instantiated in production.
"""
from __future__ import annotations

import asyncio
import logging
import os

import run_bot
from trading.bot import TradingBot
from trading.telegram_notifier import TelegramNotifier
from trading.strategies.sentinel_v44_strategy import SentinelV44Strategy

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
    legacy_symbols = _env_symbols("SIMPLE_PRECISION_SYMBOLS", config.get("symbols") or [])
    config["symbols"] = _env_symbols("SENTINEL_SYMBOLS", legacy_symbols)
    config["strategy_mode"] = "simple_precision"
    config["candle_tf"] = "15m"
    os.environ["CANDLE_TF"] = "15m"
    logger.warning(
        "[PRODUCTION CONFIG] Sentinel V%s | symbols=%s | 15M responsive PA | 1H major S/R only | SL 1.0-1.8ATR | hard-SL rearm 3 bars | target>=1.5R",
        SentinelV44Strategy.VERSION,
        config["symbols"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV44Strategy]:
    strategies = [
        SentinelV44Strategy(
            symbol,
            quality_threshold=_env_float("SP_QUALITY_THRESHOLD", 55.0),
            adx_min=12.0,
            chop_max=65.0,
            max_entry_distance_atr=1.60,
            min_room_r=1.50,
            stop_atr_min=1.00,
            stop_atr_max=1.80,
            target_r=2.0,
            tp1_r=1.0,
            tp1_trim_pct=0.0,
            exit_cooldown_bars=_env_int("SENTINEL_EXIT_COOLDOWN_BARS", 2),
        )
        for symbol in symbols
    ]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


_ORIGINAL_LOG_SCAN = TradingBot._log_scan
_LAST_SENTINEL_SCAN: dict[str, dict] = {}


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V4":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    reason = str(getattr(signal, "reason", "") or "")
    has_metrics = bool(meta.get("market_15m") or meta.get("entry_15m") or meta.get("structure_15m"))
    if has_metrics:
        _LAST_SENTINEL_SCAN[symbol] = meta

    repeated_bar = reason == "15M bar already evaluated"
    view = _LAST_SENTINEL_SCAN.get(symbol, meta) if repeated_bar else meta
    market = view.get("market_15m") or {}
    entry = view.get("entry_15m") or {}
    structure = entry.get("structure") or view.get("structure_15m") or {}
    major1h = entry.get("major_sr_1h") or view.get("major_sr_1h") or {}
    rearm = entry.get("sl_rearm") or {}

    setup = entry.get("candidate_trigger", entry.get("trigger", "WAIT"))
    direction = entry.get("direction", "WAIT")
    gate = "PASS" if market.get("ready") else "BLOCK" if market else "-"
    blocks = entry.get("blocks", []) or market.get("blocks", []) or []
    repeat_tag = " | cached=same-15M-bar" if repeated_bar and view is not meta else ""

    room15 = entry.get("room_15m_r", entry.get("room_r", "-"))
    room1h = entry.get("room_1h_r", major1h.get("room_r", "-"))
    effective_room = entry.get("room_r", "-")
    sr_source = entry.get("target_source", major1h.get("target_source", "-"))
    rearm_str = "-"
    if rearm.get("active"):
        rearm_str = f"{rearm.get('bars_since_sl', 0)}/{rearm.get('min_bars', 3)} reset={int(bool(rearm.get('fresh_reset')))}"

    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL] %s px=%.4f sig=%s | "
        "15M gate=%s ADX=%s CHOP=%s ATRx=%s | "
        "setup=%s dir=%s struct=%s SL=%sATR room15=%sR room1H=%sR eff=%sR rr=%sR SR=%s rearm=%s dist=%sATR "
        "RSI=%s/%s blocks=%s | %s%s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        gate,
        market.get("adx", "-"),
        market.get("chop", "-"),
        market.get("atr_ratio", "-"),
        setup,
        direction,
        structure.get("label", "-"),
        entry.get("sl_atr", "-"),
        room15,
        room1h if room1h is not None else "-",
        effective_room,
        entry.get("target_rr", "-"),
        sr_source,
        rearm_str,
        entry.get("distance_atr", "-"),
        entry.get("rsi", market.get("rsi", "-")),
        entry.get("rsi_sma", market.get("rsi_sma", "-")),
        ",".join(blocks) or "none",
        reason,
        repeat_tag,
    )


# The legacy notifier caption still described EMA8/13 cross-back exits. Sentinel
# V4.x no longer uses that exit, so patch only Sentinel order cards to show the
# actual production position-management rule without changing other strategies.
_ORIGINAL_BUILD_ORDER_CAPTION = TelegramNotifier.build_order_caption


def _sentinel_build_order_caption(self, *args, **kwargs):
    text = _ORIGINAL_BUILD_ORDER_CAPTION(self, *args, **kwargs)
    strategy = kwargs.get("strategy")
    if strategy is None and len(args) >= 5:
        strategy = args[4]
    if "SentinelV4.4" in str(strategy or "") or "SentinelV4.4" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🏁 Exit : structure invalidation / 2 closes beyond EMA20 + HMA16 flip",
        )
    return text


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _sentinel_log_scan
TelegramNotifier.build_order_caption = _sentinel_build_order_caption

logger.warning(
    "[PRODUCTION] Sentinel V%s installed; SL>=1.0ATR; wider structure buffer; hard-SL same-side rearm=3 bars + fresh reset; technical exit unchanged",
    SentinelV44Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
