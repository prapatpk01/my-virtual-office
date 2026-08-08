"""Production router — Trend Confirm V5.1 + Sentinel.

Railway controls:
  ENABLE_TREND_CONFIRM=true
  ENABLE_SENTINEL=true
  ENABLE_UTBOT=false

Trend Confirm V5.1 is the only Trend Confirm implementation instantiated here.
UTBot is never instantiated when ENABLE_UTBOT=false.
"""
from __future__ import annotations

import asyncio
import logging
import os

# Keep the production Trend Confirm execution/Telegram/chart hooks.
import run_enhanced_dual_bot  # noqa: F401
import run_bot
from trading.bot import TradingBot
from trading.strategies.trend_confirm_v5_strategy import TrendConfirmV5Strategy
from trading.strategies.sentinel_strategy import SentinelStrategy

logger = logging.getLogger("run_strategy_router")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


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
    return list(dict.fromkeys(x.strip() for x in raw.split(",") if x.strip()))


_BASE_BUILD_CONFIG = run_bot.build_config


def _build_config() -> dict:
    config = _BASE_BUILD_CONFIG()
    base_symbols = list(dict.fromkeys(config.get("symbols") or []))
    tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
    sentinel_enabled = _env_bool("ENABLE_SENTINEL", True)
    ut_enabled = _env_bool("ENABLE_UTBOT", False)

    tc_symbols = _env_symbols("TREND_CONFIRM_SYMBOLS", base_symbols) if tc_enabled else []
    sentinel_symbols = _env_symbols("SENTINEL_SYMBOLS", base_symbols) if sentinel_enabled else []

    config["trend_confirm_symbols"] = tc_symbols
    config["sentinel_symbols"] = sentinel_symbols
    config["symbols"] = list(dict.fromkeys(tc_symbols + sentinel_symbols))
    config["enable_trend_confirm"] = tc_enabled
    config["enable_sentinel"] = sentinel_enabled
    config["enable_utbot_xau"] = False
    config["strategy_mode"] = "trend_confirm_v5+sentinel"
    config["candle_tf"] = "15m"
    os.environ["CANDLE_TF"] = "15m"
    os.environ["ENABLE_UTBOT_XAU"] = "false"

    if ut_enabled:
        logger.warning("ENABLE_UTBOT=true ignored by this router; UTBot is intentionally disabled")

    logger.warning(
        "[ROUTER CONFIG] TrendConfirmV5.1=%s Sentinel=%s UTBot=false | TC=%s | Sentinel=%s",
        tc_enabled, sentinel_enabled, tc_symbols, sentinel_symbols,
    )
    return config


def _make_strategies(symbols: list, config: dict):
    strategies = []
    if config.get("enable_trend_confirm", True):
        for symbol in config.get("trend_confirm_symbols", []):
            strategies.append(TrendConfirmV5Strategy(symbol=symbol))

    if config.get("enable_sentinel", True):
        for symbol in config.get("sentinel_symbols", []):
            strategies.append(SentinelStrategy(
                symbol=symbol,
                min_context_score=_env_float("SENTINEL_MIN_CONTEXT_SCORE", 65.0),
                min_location_atr=_env_float("SENTINEL_MIN_LOCATION_ATR", 1.20),
                min_rr=_env_float("SENTINEL_MIN_RR", 1.50),
                entry_zone_atr=_env_float("SENTINEL_ENTRY_ZONE_ATR", 0.30),
                sl_buffer_atr=_env_float("SENTINEL_SL_BUFFER_ATR", 0.15),
                sr_merge_atr=_env_float("SENTINEL_SR_MERGE_ATR", 0.65),
                pivot_span=_env_int("SENTINEL_PIVOT_SPAN", 4),
            ))

    if not strategies:
        raise RuntimeError("Both Trend Confirm V5.1 and Sentinel are disabled")

    logger.warning(
        "[STRATEGY INSTANCES] TrendConfirmV5.1=%d Sentinel=%d UTBot=0 total=%d",
        sum(isinstance(s, TrendConfirmV5Strategy) for s in strategies),
        sum(isinstance(s, SentinelStrategy) for s in strategies),
        len(strategies),
    )
    return strategies


# Detailed ViewLog: V5 component scores + Sentinel S/R/MCDX detail.
_ORIGINAL_LOG = TradingBot._log_scan
scan_logger = logging.getLogger("trading_bot")


def _router_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}

    if meta.get("trend_confirm_version") == "5.1":
        macro = meta.get("macro_4h") if isinstance(meta.get("macro_4h"), dict) else {}
        ctx = meta.get("context_1h") if isinstance(meta.get("context_1h"), dict) else {}
        if not ctx:
            ctx = meta.get("quality_1h") if isinstance(meta.get("quality_1h"), dict) else {}
        comp = ctx.get("components") if isinstance(ctx.get("components"), dict) else {}
        sig_type = getattr(getattr(signal, "type", None), "value", "hold").upper()
        trigger = meta.get("entry_trigger_owner") or meta.get("entry_trigger") or meta.get("direction_15m", "WAIT")
        scan_logger.info(
            "[SCAN V5.1] %s %s px=%.4f sig=%s | L1 4H=%s %s/100 | "
            "L2 1H=%s Q=%s/100 [ADX %s=%s/25 | CHOP %s=%s/20 | STRUCT %s=%s/20 | MOM %s=%s/15 | ROOM %sR=%s/20] | 15M=%s | %s",
            strategy_name, symbol, price, sig_type,
            macro.get("state", meta.get("trend_4h", "?")), macro.get("score", "?"),
            ctx.get("label", meta.get("trend_1h", "?")), ctx.get("score", "?"),
            ctx.get("adx", "?"), comp.get("adx", "?"),
            ctx.get("chop", "?"), comp.get("chop", "?"),
            str(ctx.get("structure", "?")).upper(), comp.get("structure", "?"),
            "ALIGNED" if ctx.get("momentum_aligned") else "OPPOSED", comp.get("momentum", "?"),
            ctx.get("room_r", "?"), comp.get("room", "?"), trigger,
            getattr(signal, "reason", ""),
        )
        return

    if str(strategy_name).startswith("Sentinel("):
        mc = meta.get("mcdx") if isinstance(meta.get("mcdx"), dict) else {}
        sx = meta.get("sentinel_x") if isinstance(meta.get("sentinel_x"), dict) else {}
        scan_logger.info(
            "[SCAN SENTINEL] %s %s px=%.4f sig=%s | S2=%s S1=%s R1=%s R2=%s | room=%sATR | SX=%s/%s | MCDX L=%s S=%s flow=%s | RR L=%s S=%s | %s",
            strategy_name, symbol, price,
            getattr(getattr(signal, "type", None), "value", "hold").upper(),
            meta.get("s2", "?"), meta.get("s1", "?"), meta.get("r1", "?"), meta.get("r2", "?"),
            meta.get("location_atr", "?"), sx.get("bias", "?"), sx.get("structure", "?"),
            mc.get("long_score", "?"), mc.get("short_score", "?"), mc.get("smart_flow", "?"),
            meta.get("long_rr", "?"), meta.get("short_rr", "?"), getattr(signal, "reason", ""),
        )
        return

    return _ORIGINAL_LOG(self, symbol, strategy_name, price, signal)


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _router_log_scan

logger.warning("[PRODUCTION] Trend Confirm V5.1 + Sentinel router installed; UTBot disabled")
logger.warning("[TREND CONFIRM] canonical runtime = trading/strategies/trend_confirm_v5_strategy.py")
logger.warning("[SENTINEL] canonical runtime = trading/strategies/sentinel_strategy.py")


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
