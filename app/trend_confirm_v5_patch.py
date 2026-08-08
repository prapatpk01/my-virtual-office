"""Production patch that activates Trend Confirm V5 without disturbing UTBot routing."""
from __future__ import annotations

import logging
import os

import run_bot
import run_enhanced_dual_bot as enhanced
from trading.bot import TradingBot

logger = logging.getLogger("trend_confirm_v5_patch")
_ORIGINAL_FACTORY = enhanced._make_merged_trend_confirm
_ORIGINAL_LOG_SCAN = TradingBot._log_scan


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _factory(symbols: list, config: dict):
    # Preserve the established 1H-only bypass implementation when Layer1 is
    # deliberately disabled. V5 is the new 4H+1H production path.
    if not _env_bool("USE_LAYER1_4H", True):
        return _ORIGINAL_FACTORY(symbols, config)

    from trading.strategies.trend_confirm_v5_strategy import TrendConfirmV5Strategy

    def env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    def env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return int(default)

    return [
        TrendConfirmV5Strategy(
            symbol=symbol,
            wt_oversold=-42.0,
            wt_overbought=45.0,
            structure_swing_span=env_int("STRUCTURE_SWING_SPAN", 3),
            structure_retest_min_bars=env_int("STRUCTURE_RETEST_MIN_BARS", 1),
            structure_retest_max_bars=env_int("STRUCTURE_RETEST_MAX_BARS", 3),
            structure_bos_buffer_atr=env_float("STRUCTURE_BOS_BUFFER_ATR", 0.05),
            structure_touch_tolerance_atr=env_float("STRUCTURE_TOUCH_TOLERANCE_ATR", 0.15),
            structure_invalidation_tolerance_atr=env_float("STRUCTURE_INVALIDATION_TOLERANCE_ATR", 0.25),
            structure_max_close_distance_atr=env_float("STRUCTURE_MAX_CLOSE_DISTANCE_ATR", 0.50),
            structure_max_fill_slippage_atr=env_float("STRUCTURE_MAX_FILL_SLIPPAGE_ATR", 0.35),
        )
        for symbol in symbols
    ]


def _log_scan(self, symbol, strategy_name, price, signal):
    if str(strategy_name).startswith("TrendConfirm("):
        meta = getattr(signal, "metadata", None) or {}
        macro = meta.get("macro_4h") if isinstance(meta.get("macro_4h"), dict) else {}
        ctx = meta.get("context_1h") if isinstance(meta.get("context_1h"), dict) else {}
        if meta.get("trend_confirm_version") == "5.0" or macro.get("layer_role") == "DIRECTION_ONLY":
            sig_type = getattr(getattr(signal, "type", None), "value", "hold").upper()
            logger.info(
                "[SCAN] %-28s %-22s px=%-11.4f sig=%-5s | L1 4H=%s score=%s (B=%s/S=%s) | L2 1H=%s Q=%s ADX=%s CHOP=%s Struct=%s Room=%sR | 15M=%s | %s",
                strategy_name,
                symbol,
                price,
                sig_type,
                macro.get("state", "?"),
                macro.get("score", "?"),
                macro.get("bull_score", "?"),
                macro.get("bear_score", "?"),
                ctx.get("label", "?"),
                ctx.get("score", "?"),
                ctx.get("adx", "?"),
                ctx.get("chop", "?"),
                ctx.get("structure", "?"),
                ctx.get("room_r", "?"),
                meta.get("entry_trigger_owner") or meta.get("entry_trigger") or meta.get("direction_15m", "WAIT"),
                getattr(signal, "reason", ""),
            )
            return
    return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)


def install() -> None:
    enhanced._make_merged_trend_confirm = _factory
    run_bot._make_strategies = _factory
    if not getattr(TradingBot, "_trend_confirm_v5_log_installed", False):
        TradingBot._log_scan = _log_scan
        TradingBot._trend_confirm_v5_log_installed = True
    logger.warning("[TREND CONFIRM V5] installed: simplified 4H direction + 1H quality + RR 1:2")


install()
