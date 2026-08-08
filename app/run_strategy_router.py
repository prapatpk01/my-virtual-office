"""Safe production strategy router.

BOT_STRATEGY_MODE controls the non-XAU strategy family explicitly:
  trend_confirm  -> Trend Confirm only
  adaptive       -> Adaptive Multi-Trigger only
  both           -> Trend Confirm + Adaptive Multi-Trigger

XAU/UTBot remains controlled independently by ENABLE_UTBOT_XAU.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("run_strategy_router")


def _normalize_mode(raw: str) -> str:
    value = str(raw or "").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "trend": "trend_confirm", "trendconfirm": "trend_confirm", "trend_confirm": "trend_confirm", "tc": "trend_confirm",
        "adaptive": "adaptive", "adaptive_multi_trigger": "adaptive", "adaptivemultitrigger": "adaptive",
        "both": "both", "combined": "both", "trend_confirm_adaptive": "both", "adaptive_trend_confirm": "both",
    }
    return aliases.get(value, value)


def _apply_strategy_mode() -> str:
    raw = os.getenv("BOT_STRATEGY_MODE", "trend_confirm")
    mode = _normalize_mode(raw)
    if mode == "trend_confirm":
        os.environ["ENABLE_TREND_CONFIRM"] = "true"; os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "false"
    elif mode == "adaptive":
        os.environ["ENABLE_TREND_CONFIRM"] = "false"; os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "true"
    elif mode == "both":
        os.environ["ENABLE_TREND_CONFIRM"] = "true"; os.environ["ENABLE_ADAPTIVE_MULTI_TRIGGER"] = "true"
    else:
        raise RuntimeError(f"BOT_STRATEGY_MODE must be trend_confirm, adaptive, or both; got {raw!r}")
    logger.warning("[STRATEGY ROUTER] BOT_STRATEGY_MODE=%s -> TrendConfirm=%s Adaptive=%s UTBotXAU=%s", mode, os.environ.get("ENABLE_TREND_CONFIRM"), os.environ.get("ENABLE_ADAPTIVE_MULTI_TRIGGER"), os.environ.get("ENABLE_UTBOT_XAU", "false"))
    return mode


_mode = _apply_strategy_mode()

# Install strategy/risk/UI patches first.
import trend_confirm_v5_patch  # noqa: E402,F401
import trading.strategies.trend_confirm_v5_adx_patch  # noqa: E402,F401
import run_trendconfirm_utbot  # noqa: E402,F401
import run_bot  # noqa: E402

# IMPORTANT: install the detailed Trend Confirm logger LAST. Some combined-runner
# patches wrap TradingBot._log_scan during import. Doing this last guarantees the
# production viewlog cannot silently fall back to the legacy compact formatter.
from trading.bot import TradingBot  # noqa: E402

_FINAL_FALLBACK_LOG = TradingBot._log_scan
_final_logger = logging.getLogger("trading_bot")


def _final_component_log(self, symbol, strategy_name, price, signal):
    if str(strategy_name).startswith("TrendConfirm("):
        meta = getattr(signal, "metadata", None) or {}
        macro = meta.get("macro_4h") if isinstance(meta.get("macro_4h"), dict) else {}
        ctx = meta.get("context_1h") if isinstance(meta.get("context_1h"), dict) else {}
        comp = ctx.get("components") if isinstance(ctx.get("components"), dict) else {}

        # V5 should always provide components. If a HOLD was produced before the
        # full context dict exists, still print explicit placeholders so it is
        # obvious which component is unavailable rather than reverting to old log.
        if str(meta.get("trend_confirm_version", "")).startswith("5") or macro.get("layer_role") == "DIRECTION_ONLY" or comp:
            sig_type = getattr(getattr(signal, "type", None), "value", "hold").upper()
            mom_label = "ALIGNED" if ctx.get("momentum_aligned") else "OPPOSED"
            trigger = meta.get("entry_trigger_owner") or meta.get("entry_trigger") or meta.get("direction_15m", "WAIT")
            _final_logger.info(
                "[SCAN] %s %s px=%.4f sig=%s | L1 4H=%s score=%s/100 (B=%s S=%s) | "
                "L2 1H=%s Q=%s/100 [ADX %s=%s/25 | CHOP %s=%s/20 | STRUCT %s=%s/20 | "
                "MOM %s=%s/15 | ROOM %sR=%s/20] hard=%s | 15M=%s | %s",
                strategy_name, symbol, price, sig_type,
                macro.get("state", "?"), macro.get("score", "?"), macro.get("bull_score", "?"), macro.get("bear_score", "?"),
                ctx.get("label", "?"), ctx.get("score", "?"),
                ctx.get("adx", "?"), comp.get("adx", "?"),
                ctx.get("chop", "?"), comp.get("chop", "?"),
                str(ctx.get("structure", "?")).upper(), comp.get("structure", "?"),
                mom_label, comp.get("momentum", "?"),
                ctx.get("room_r", "?"), comp.get("room", "?"),
                ctx.get("hard_block", "?"), trigger, getattr(signal, "reason", ""),
            )
            return
    return _FINAL_FALLBACK_LOG(self, symbol, strategy_name, price, signal)


TradingBot._log_scan = _final_component_log
TradingBot._trend_confirm_final_component_log_installed = True
logger.warning("[VIEWLOG] Final Trend Confirm component logger installed AFTER all runtime patches")


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
