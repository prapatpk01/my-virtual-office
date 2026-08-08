"""Production strategy router: canonical Trend Confirm + Sentinel + optional Adaptive/UTBot.

Canonical Railway toggles:
    ENABLE_TREND_CONFIRM=true|false
    ENABLE_SENTINEL=true|false
    ENABLE_ADAPTIVE_MULTI_TRIGGER=true|false
    ENABLE_UTBOT=true|false

Trend Confirm is intentionally sourced from the existing production
trend_confirm_strategy.py path.  No V5 shadow strategy or runtime patch is
loaded here.  This keeps one authoritative Trend Confirm implementation.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("run_strategy_router")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise RuntimeError(f"{name} must be true/false, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


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
        "sentinel": "sentinel",
        "both": "trend_sentinel",
        "combined": "trend_sentinel",
        "trend_sentinel": "trend_sentinel",
        "sentinel_trend": "trend_sentinel",
        "trend_confirm_sentinel": "trend_sentinel",
        "sentinel_trend_confirm": "trend_sentinel",
    }
    return aliases.get(value, value)


def _canonical_ut_enabled() -> bool:
    if os.getenv("ENABLE_UTBOT") is not None:
        return _env_bool("ENABLE_UTBOT", False)
    return _env_bool("ENABLE_UTBOT_XAU", False)


def _apply_strategy_mode() -> str:
    raw = os.getenv("BOT_STRATEGY_MODE", "")
    mode = _normalize_mode(raw) if raw else "manual"

    if mode == "trend_confirm":
        os.environ["ENABLE_TREND_CONFIRM"] = "true"
        os.environ["ENABLE_SENTINEL"] = "false"
        os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "false"
    elif mode == "adaptive":
        os.environ["ENABLE_TREND_CONFIRM"] = "false"
        os.environ["ENABLE_SENTINEL"] = "false"
        os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "true"
    elif mode == "sentinel":
        os.environ["ENABLE_TREND_CONFIRM"] = "false"
        os.environ["ENABLE_SENTINEL"] = "true"
        os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "false"
    elif mode == "trend_sentinel":
        os.environ["ENABLE_TREND_CONFIRM"] = "true"
        os.environ["ENABLE_SENTINEL"] = "true"
        os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "false"
    elif mode not in {"manual", ""}:
        raise RuntimeError(
            "BOT_STRATEGY_MODE must be trend_confirm, sentinel, trend_sentinel, adaptive, or be unset; "
            f"got {raw!r}"
        )

    ut_enabled = _canonical_ut_enabled()
    os.environ["ENABLE_UTBOT_XAU"] = "true" if ut_enabled else "false"

    logger.warning(
        "[STRATEGY ROUTER] mode=%s TrendConfirm=%s Sentinel=%s Adaptive=%s UTBot=%s",
        mode,
        os.getenv("ENABLE_TREND_CONFIRM", "true"),
        os.getenv("ENABLE_SENTINEL", "false"),
        os.getenv("ENABLE_ADAPTIVE_MULTI_TRIGGER", "false"),
        ut_enabled,
    )
    return mode


_mode = _apply_strategy_mode()

# Load the existing production runtime after Railway toggles are normalized.
# IMPORTANT: do not import any trend_confirm_v5_* module here.
import run_trendconfirm_utbot  # noqa: E402
import run_bot  # noqa: E402
from trading.bot import TradingBot  # noqa: E402
from trading.strategies.sentinel_strategy import SentinelStrategy  # noqa: E402

# Canonical Trend Confirm factory captured from the real production path.
_TREND_FACTORY = run_trendconfirm_utbot._TREND_MAKE_STRATEGIES
_TREND_BUILD_CONFIG = run_trendconfirm_utbot._TREND_BUILD_CONFIG
XAU = "XAU/USDT:USDT"


def _sentinel_symbols(base_symbols: list[str]) -> list[str]:
    raw = str(os.getenv("SENTINEL_SYMBOLS", "") or "").strip()
    if not raw:
        return list(dict.fromkeys(base_symbols))
    wanted = [x.strip() for x in raw.split(",") if x.strip()]
    return list(dict.fromkeys(wanted))


def _make_strategies(symbols: list, config: dict):
    tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
    sentinel_enabled = _env_bool("ENABLE_SENTINEL", False)
    adaptive_enabled = _env_bool("ENABLE_ADAPTIVE_MULTI_TRIGGER", False)
    ut_enabled = _canonical_ut_enabled()

    base_symbols = list(config.get("base_symbols") or symbols or [])
    tc_symbols = list(config.get("trend_confirm_symbols") or [])
    sentinel_symbols = list(config.get("sentinel_symbols") or [])
    strategies = []

    if tc_enabled and tc_symbols:
        strategies.extend(_TREND_FACTORY(tc_symbols, config))

    if sentinel_enabled:
        for symbol in sentinel_symbols:
            strategies.append(
                SentinelStrategy(
                    symbol=symbol,
                    min_context_score=_env_float("SENTINEL_MIN_CONTEXT_SCORE", 65.0),
                    min_location_atr=_env_float("SENTINEL_MIN_LOCATION_ATR", 1.20),
                    min_rr=_env_float("SENTINEL_MIN_RR", 1.50),
                    entry_zone_atr=_env_float("SENTINEL_ENTRY_ZONE_ATR", 0.30),
                    sl_buffer_atr=_env_float("SENTINEL_SL_BUFFER_ATR", 0.15),
                    sr_merge_atr=_env_float("SENTINEL_SR_MERGE_ATR", 0.65),
                    pivot_span=_env_int("SENTINEL_PIVOT_SPAN", 4),
                )
            )

    if adaptive_enabled:
        from trading.strategies.adaptive_multi_trigger_strategy import AdaptiveMultiTriggerStrategy
        for symbol in [s for s in base_symbols if s != XAU]:
            strategies.append(AdaptiveMultiTriggerStrategy(symbol=symbol))

    if ut_enabled:
        from trading.strategies.utbot_xau_strategy import UTBotXAUStrategy
        strategies.append(
            UTBotXAUStrategy(
                symbol=XAU,
                multiplier=_env_float("UTBOT_MULTIPLIER", 1.0),
                atr_period=_env_int("UTBOT_ATR_PERIOD", 10),
                timeframe="15m",
                use_date_filter=_env_bool("UTBOT_USE_DATE_FILTER", True),
            )
        )

    if not strategies:
        raise RuntimeError("No enabled strategy produced an instance")

    logger.warning(
        "[STRATEGY INSTANCES] TrendConfirm=%d Sentinel=%d Adaptive=%d UTBot=%d total=%d",
        sum(str(getattr(s, "name", "")).startswith("TrendConfirm(") for s in strategies),
        sum(str(getattr(s, "name", "")).startswith("Sentinel(") for s in strategies),
        sum(str(getattr(s, "name", "")).startswith("AdaptiveMultiTrigger") for s in strategies),
        sum(str(getattr(s, "name", "")).startswith("UTBotXAU(") for s in strategies),
        len(strategies),
    )
    return strategies


def _build_config() -> dict:
    tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
    sentinel_enabled = _env_bool("ENABLE_SENTINEL", False)
    adaptive_enabled = _env_bool("ENABLE_ADAPTIVE_MULTI_TRIGGER", False)
    ut_enabled = _canonical_ut_enabled()

    config = _TREND_BUILD_CONFIG()
    base_symbols = list(dict.fromkeys(config.get("symbols") or []))
    config["base_symbols"] = base_symbols

    tc_exclude_xau = _env_bool("TREND_CONFIRM_EXCLUDE_XAU", True)
    tc_symbols = [s for s in base_symbols if not (tc_exclude_xau and s == XAU)] if tc_enabled else []
    sentinel_symbols = _sentinel_symbols(base_symbols) if sentinel_enabled else []

    market_symbols = []
    for group in (tc_symbols, sentinel_symbols):
        market_symbols.extend(group)
    if adaptive_enabled:
        market_symbols.extend(s for s in base_symbols if s != XAU)
    if ut_enabled:
        market_symbols.append(XAU)

    config["trend_confirm_symbols"] = list(dict.fromkeys(tc_symbols))
    config["sentinel_symbols"] = list(dict.fromkeys(sentinel_symbols))
    config["symbols"] = list(dict.fromkeys(market_symbols))
    config["enable_trend_confirm"] = tc_enabled
    config["enable_sentinel"] = sentinel_enabled
    config["enable_adaptive_multi_trigger"] = adaptive_enabled
    config["enable_utbot_xau"] = ut_enabled
    config["candle_tf"] = "15m"
    os.environ["CANDLE_TF"] = "15m"

    names = []
    if tc_enabled:
        names.append("trend_confirm")
    if sentinel_enabled:
        names.append("sentinel")
    if adaptive_enabled:
        names.append("adaptive")
    if ut_enabled:
        names.append("utbot")
    config["strategy_mode"] = "+".join(names)

    logger.warning(
        "[ROUTER CONFIG] TC=%s Sentinel=%s UTBot=%s | TC symbols=%s | Sentinel symbols=%s",
        tc_enabled,
        sentinel_enabled,
        ut_enabled,
        config["trend_confirm_symbols"],
        config["sentinel_symbols"],
    )
    return config


run_bot._make_strategies = _make_strategies
run_bot.build_config = _build_config

# Preserve the canonical Trend Confirm logger.  Add only Sentinel-specific detail.
_FALLBACK_LOG = TradingBot._log_scan
_scan_logger = logging.getLogger("trading_bot")


def _router_log_scan(self, symbol, strategy_name, price, signal):
    if str(strategy_name).startswith("Sentinel("):
        meta = getattr(signal, "metadata", None) or {}
        mc = meta.get("mcdx") if isinstance(meta.get("mcdx"), dict) else {}
        sx = meta.get("sentinel_x") if isinstance(meta.get("sentinel_x"), dict) else {}
        _scan_logger.info(
            "[SCAN SENTINEL] %s %s px=%.4f sig=%s | S2=%s S1=%s R1=%s R2=%s | room=%sATR | "
            "SX=%s/%s | MCDX L=%s S=%s flow=%s | RR L=%s S=%s | %s",
            strategy_name,
            symbol,
            price,
            getattr(getattr(signal, "type", None), "value", "hold").upper(),
            meta.get("s2", "?"),
            meta.get("s1", "?"),
            meta.get("r1", "?"),
            meta.get("r2", "?"),
            meta.get("location_atr", "?"),
            sx.get("bias", "?"),
            sx.get("structure", "?"),
            mc.get("long_score", "?"),
            mc.get("short_score", "?"),
            mc.get("smart_flow", "?"),
            meta.get("long_rr", "?"),
            meta.get("short_rr", "?"),
            getattr(signal, "reason", ""),
        )
        return
    return _FALLBACK_LOG(self, symbol, strategy_name, price, signal)


TradingBot._log_scan = _router_log_scan
logger.warning("[VIEWLOG] Canonical Trend Confirm + Sentinel logger installed")
logger.warning("[TREND CONFIRM] canonical implementation = trading/strategies/trend_confirm_strategy.py")


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
