"""Production strategy router: Trend Confirm + Sentinel + optional Adaptive/UTBot.

Canonical Railway toggles:
    ENABLE_TREND_CONFIRM=true|false
    ENABLE_SENTINEL=true|false
    ENABLE_ADAPTIVE_MULTI_TRIGGER=true|false
    ENABLE_UTBOT=true|false

Legacy ENABLE_UTBOT_XAU is still read for backwards compatibility, but the
canonical ENABLE_UTBOT value wins when it is present. This prevents an old
ENABLE_UTBOT_XAU=true variable from silently keeping UTBot alive after
ENABLE_UTBOT=false is set in Railway.
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
        "trend": "trend_confirm", "trendconfirm": "trend_confirm", "trend_confirm": "trend_confirm", "tc": "trend_confirm",
        "adaptive": "adaptive", "adaptive_multi_trigger": "adaptive", "adaptivemultitrigger": "adaptive",
        "sentinel": "sentinel",
        "both": "trend_sentinel", "combined": "trend_sentinel", "trend_sentinel": "trend_sentinel", "sentinel_trend": "trend_sentinel",
        "trend_confirm_sentinel": "trend_sentinel", "sentinel_trend_confirm": "trend_sentinel",
    }
    return aliases.get(value, value)


def _canonical_ut_enabled() -> bool:
    # New variable has priority. This is the bug fix for Railway screenshots
    # where ENABLE_UTBOT=false existed while stale ENABLE_UTBOT_XAU=true still
    # caused UTBotXAU to scan.
    if os.getenv("ENABLE_UTBOT") is not None:
        return _env_bool("ENABLE_UTBOT", False)
    return _env_bool("ENABLE_UTBOT_XAU", False)


def _apply_strategy_mode() -> str:
    raw = os.getenv("BOT_STRATEGY_MODE", "")
    mode = _normalize_mode(raw) if raw else "manual"

    # If a mode is explicitly supplied it controls the main families. Otherwise
    # Railway ENABLE_* toggles are used directly.
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
    # Synchronize legacy variable so every imported legacy helper sees the same
    # state. ENABLE_UTBOT=false therefore ALWAYS disables UTBotXAU.
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

# Import production patches/runtimes only after Railway toggles are normalized.
import trend_confirm_v5_patch  # noqa: E402
import trading.strategies.trend_confirm_v5_adx_patch  # noqa: E402,F401
import run_trendconfirm_utbot  # noqa: E402
import run_bot  # noqa: E402
from trading.bot import TradingBot  # noqa: E402
from trading.strategies.sentinel_strategy import SentinelStrategy  # noqa: E402

# Capture current Trend Confirm V5 factory/config.
run_trendconfirm_utbot._TREND_MAKE_STRATEGIES = trend_confirm_v5_patch._factory
_TREND_FACTORY = trend_confirm_v5_patch._factory
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

    # Preserve the previous design: Trend Confirm does not trade XAU. Sentinel
    # may trade XAU, so XAU remains in market data whenever Sentinel is enabled.
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
    if tc_enabled: names.append("trend_confirm")
    if sentinel_enabled: names.append("sentinel")
    if adaptive_enabled: names.append("adaptive")
    if ut_enabled: names.append("utbot")
    config["strategy_mode"] = "+".join(names)

    logger.warning(
        "[ROUTER CONFIG] TC=%s Sentinel=%s UTBot=%s | TC symbols=%s | Sentinel symbols=%s",
        tc_enabled, sentinel_enabled, ut_enabled,
        config["trend_confirm_symbols"], config["sentinel_symbols"],
    )
    return config


# Authoritative hooks. Do NOT delegate strategy creation back to the old
# TrendConfirm+UTBot runner; this is what previously kept UTBot alive.
run_bot._make_strategies = _make_strategies
run_bot.build_config = _build_config

# Detailed V5 Trend Confirm viewlog remains installed last.
_FINAL_FALLBACK_LOG = TradingBot._log_scan
_final_logger = logging.getLogger("trading_bot")


def _final_component_log(self, symbol, strategy_name, price, signal):
    if str(strategy_name).startswith("TrendConfirm("):
        meta = getattr(signal, "metadata", None) or {}
        macro = meta.get("macro_4h") if isinstance(meta.get("macro_4h"), dict) else {}
        ctx = meta.get("context_1h") if isinstance(meta.get("context_1h"), dict) else {}
        comp = ctx.get("components") if isinstance(ctx.get("components"), dict) else {}
        version = str(meta.get("trend_confirm_version", "?"))
        sig_type = getattr(getattr(signal, "type", None), "value", "hold").upper()
        mom_label = "ALIGNED" if ctx.get("momentum_aligned") is True else "OPPOSED" if ctx.get("momentum_aligned") is False else "?"
        trigger = meta.get("entry_trigger_owner") or meta.get("entry_trigger") or meta.get("direction_15m", "WAIT")
        _final_logger.info(
            "[SCAN V5.%s] %s %s px=%.4f sig=%s | L1 4H=%s score=%s/100 (B=%s S=%s) | "
            "L2 1H=%s Q=%s/100 [ADX %s=%s/25 | CHOP %s=%s/20 | STRUCT %s=%s/20 | "
            "MOM %s=%s/15 | ROOM %sR=%s/20] hard=%s | 15M=%s | %s",
            version, strategy_name, symbol, price, sig_type,
            macro.get("state", "?"), macro.get("score", "?"), macro.get("bull_score", "?"), macro.get("bear_score", "?"),
            ctx.get("label", "?"), ctx.get("score", "?"), ctx.get("adx", "?"), comp.get("adx", "?"),
            ctx.get("chop", "?"), comp.get("chop", "?"), str(ctx.get("structure", "?")).upper(), comp.get("structure", "?"),
            mom_label, comp.get("momentum", "?"), ctx.get("room_r", "?"), comp.get("room", "?"),
            ctx.get("hard_block", "?"), trigger, getattr(signal, "reason", ""),
        )
        return
    if str(strategy_name).startswith("Sentinel("):
        meta = getattr(signal, "metadata", None) or {}
        mc = meta.get("mcdx") if isinstance(meta.get("mcdx"), dict) else {}
        sx = meta.get("sentinel_x") if isinstance(meta.get("sentinel_x"), dict) else {}
        _final_logger.info(
            "[SCAN SENTINEL] %s %s px=%.4f sig=%s | S2=%s S1=%s R1=%s R2=%s | room=%sATR | "
            "SX=%s/%s | MCDX L=%s S=%s flow=%s | RR L=%s S=%s | %s",
            strategy_name, symbol, price,
            getattr(getattr(signal, "type", None), "value", "hold").upper(),
            meta.get("s2", "?"), meta.get("s1", "?"), meta.get("r1", "?"), meta.get("r2", "?"),
            meta.get("location_atr", "?"), sx.get("bias", "?"), sx.get("structure", "?"),
            mc.get("long_score", "?"), mc.get("short_score", "?"), mc.get("smart_flow", "?"),
            meta.get("long_rr", "?"), meta.get("short_rr", "?"), getattr(signal, "reason", ""),
        )
        return
    return _FINAL_FALLBACK_LOG(self, symbol, strategy_name, price, signal)


TradingBot._log_scan = _final_component_log
logger.warning("[VIEWLOG] Trend Confirm V5 + Sentinel detailed logger installed")


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
