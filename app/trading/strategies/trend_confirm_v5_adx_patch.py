"""Trend Confirm V5 ADX scoring + detailed component viewlog.

ADX contribution (25 pts):
  <15      -> 0
  15-<20   -> 10
  20-<25   -> 18
  25-46    -> 25
  >46-50   -> 22
  >50-54   -> 18
  >54-58   -> 14
  >58-62   -> 10
  >62-65   -> 5
  >65      -> 0

Only the ADX component is changed. CHOP / structure / momentum / room scoring,
hard blocks, entry triggers and position management remain unchanged.
"""
from __future__ import annotations

import logging

from trading.bot import TradingBot
from trading.strategies.trend_confirm_v5_strategy import TrendConfirmV5Strategy

logger = logging.getLogger("trend_confirm_v5_adx_patch")

_ORIGINAL_CONTEXT_1H = TrendConfirmV5Strategy._context_1h
_ORIGINAL_LOG_SCAN = TradingBot._log_scan


def _adx_score(adx: float) -> float:
    value = float(adx)
    if value < 15.0:
        return 0.0
    if value < 20.0:
        return 10.0
    if value < 25.0:
        return 18.0
    if value <= 46.0:
        return 25.0
    if value <= 50.0:
        return 22.0
    if value <= 54.0:
        return 18.0
    if value <= 58.0:
        return 14.0
    if value <= 62.0:
        return 10.0
    if value <= 65.0:
        return 5.0
    return 0.0


def _context_1h(self, candles_1h: list, direction: str):
    ctx = _ORIGINAL_CONTEXT_1H(self, candles_1h, direction)
    if not isinstance(ctx, dict):
        return ctx

    components = dict(ctx.get("components") or {})
    adx_value = float(ctx.get("adx", 0.0) or 0.0)
    components["adx"] = _adx_score(adx_value)

    # Preserve the other V5 component scores exactly as calculated by V5.
    for key in ("chop", "structure", "momentum", "room"):
        components[key] = float(components.get(key, 0.0) or 0.0)

    total = sum(float(components.get(key, 0.0)) for key in (
        "adx", "chop", "structure", "momentum", "room"
    ))
    hard_block = bool(ctx.get("hard_block", False))

    if total >= 70.0:
        label = "STRONG"
    elif total >= 55.0:
        label = "NORMAL"
    elif total >= 45.0:
        label = "WEAK"
    else:
        label = "BLOCK"

    ctx.update({
        "score": round(total, 1),
        "label": label,
        "ready": bool(total >= 55.0 and not hard_block),
        "components": components,
        "adx_score": components["adx"],
        "adx_score_rule": "STEP_15_25_THEN_DECAY_46_65_ZERO_GT65",
    })
    return ctx


def _log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    macro = meta.get("macro_4h") if isinstance(meta.get("macro_4h"), dict) else {}
    ctx = meta.get("context_1h") if isinstance(meta.get("context_1h"), dict) else {}

    if (
        str(strategy_name).startswith("TrendConfirm(")
        and (meta.get("trend_confirm_version") == "5.0" or macro.get("layer_role") == "DIRECTION_ONLY")
        and isinstance(ctx.get("components"), dict)
    ):
        comp = ctx["components"]
        sig_type = getattr(getattr(signal, "type", None), "value", "hold").upper()
        trigger = (
            meta.get("entry_trigger_owner")
            or meta.get("entry_trigger")
            or meta.get("direction_15m", "WAIT")
        )
        logger.info(
            "[SCAN] %-28s %-22s px=%-11.4f sig=%-5s | "
            "L1 4H=%s %s/100 (B=%s S=%s) | "
            "L2 1H=%s Q=%s/100 [ADX %.1f=%.0f/25 | CHOP %.1f=%.0f/20 | "
            "STRUCT %s=%.0f/20 | MOM %s=%.0f/15 | ROOM %sR=%.0f/20] | "
            "15M=%s | %s",
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
            float(ctx.get("adx", 0.0) or 0.0),
            float(comp.get("adx", 0.0) or 0.0),
            float(ctx.get("chop", 0.0) or 0.0),
            float(comp.get("chop", 0.0) or 0.0),
            str(ctx.get("structure", "?")).upper(),
            float(comp.get("structure", 0.0) or 0.0),
            "ALIGNED" if ctx.get("momentum_aligned") else "OPPOSED",
            float(comp.get("momentum", 0.0) or 0.0),
            ctx.get("room_r", "?"),
            float(comp.get("room", 0.0) or 0.0),
            trigger,
            getattr(signal, "reason", ""),
        )
        return

    return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)


def install() -> None:
    if getattr(TrendConfirmV5Strategy, "_adx_decay_patch_installed", False):
        return
    TrendConfirmV5Strategy._context_1h = _context_1h
    TrendConfirmV5Strategy._adx_decay_patch_installed = True
    TradingBot._log_scan = _log_scan
    TradingBot._trend_confirm_v5_component_log_installed = True
    logger.warning(
        "[TREND CONFIRM V5] ADX score patch installed: peak 25 pts at ADX 25-46, decay 46-65, >65 = 0"
    )


install()
