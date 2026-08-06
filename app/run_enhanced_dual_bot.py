"""Production entry point for unified Trend Confirm.

Only closed 15M triggers may open a position:
- EMA8/13 cross,
- WaveTrend extreme cross (-42 / +45), or
- confirmed Structure BOS + retest.

4H and 1H remain direction/quality gates only. Telegram text and PNG charts use
entry-trigger metadata so every position shows the correct owner and exit rule.
"""
from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar

import run_dual_bot
import run_bot
import trading.chart_renderer as chart_renderer
from trading.bot import TradingBot
from trading.telegram_notifier import TelegramNotifier


_TG_ENTRY_CONTEXT: ContextVar[dict | None] = ContextVar(
    "trend_confirm_tg_entry_context",
    default=None,
)


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


def _entry_trigger_context(signal) -> dict:
    """Resolve the exact Layer-3 owner and chart metadata from a Signal."""
    metadata = getattr(signal, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    owner = str(
        metadata.get("entry_trigger_owner")
        or metadata.get("entry_trigger")
        or ""
    ).upper()

    if "STRUCTURE" in owner or "BOS_RETEST" in owner:
        label = "Structure BOS + Retest"
    elif "WT" in owner:
        label = "WT Cross"
    elif "EMA" in owner:
        label = "EMA8/13 Cross"
    else:
        label = None

    return {
        "label": label,
        "metadata": dict(metadata),
    }


def _signal_exit_text(label: str) -> str:
    if label == "WT Cross":
        return "🏁 Signal Exit : `WT opposite cross`"
    if label == "Structure BOS + Retest":
        return "🏁 Signal Exit : `Structure invalidation / opposite CHOCH`"
    return "🏁 Signal Exit : `EMA8/13 reverse cross`"


def _append_entry_trigger(text: str, label: str | None) -> str:
    """Rewrite the order caption to match the actual entry owner."""
    if not label:
        return text

    lines = str(text).splitlines()
    if not any("Entry Trigger" in line for line in lines):
        trigger_line = f"⚡ Entry Trigger : `{label}`"
        insert_at = 1
        for index, line in enumerate(lines):
            if "Entry :" in line or "Entry:" in line or "Fill:" in line:
                insert_at = index + 1
                break
        lines.insert(insert_at, trigger_line)

    target_index = None
    for index, line in enumerate(lines):
        if line.startswith("🎯 Target") or line.startswith("🎯 T1"):
            target_index = index
            break
    if target_index is not None:
        lines[target_index] = "🎯 T1 : `+0.6%` → take profit `40%`"
        runner_line = "🔒 Runner `60%` : SL → `+0.3%` | TP Final → `+1.3%`"
        if not any(line.startswith("🔒 Runner") for line in lines):
            lines.insert(target_index + 1, runner_line)

    exit_text = _signal_exit_text(label)
    exit_replaced = False
    for index, line in enumerate(lines):
        if line.startswith("🏁 Exit") or line.startswith("🏁 Signal Exit"):
            lines[index] = exit_text
            exit_replaced = True
            break
    if not exit_replaced:
        lines.append(exit_text)

    for index, line in enumerate(lines):
        if line.startswith("📊 Strategy:"):
            regime_suffix = ""
            if "| Regime:" in line:
                regime_suffix = " | Regime:" + line.split("| Regime:", 1)[1]
            lines[index] = f"📊 Strategy: `Trend Confirm`{regime_suffix}"
            break

    for index, line in enumerate(lines):
        if line.startswith("🧭 4H Macro:") and " (0/100)" in line:
            lines[index] = line.replace(" (0/100)", "")

    return "\n".join(lines)


def _normalize_notification_metadata(signal) -> None:
    """Expose Trend Confirm's native 4H metadata to the generic notifier."""
    metadata = getattr(signal, "metadata", None)
    if not isinstance(metadata, dict):
        return

    metadata["selected_strategy"] = "Trend Confirm"
    macro = metadata.get("macro_4h")
    if isinstance(macro, dict):
        state = str(macro.get("state") or "").upper()
        direction = str(macro.get("direction") or "").upper()
        bias = state or direction
        macro_trend = metadata.get("macro_trend")
        if not isinstance(macro_trend, dict):
            macro_trend = {}
        if bias:
            macro_trend["bias"] = bias
        score = macro.get("score")
        if isinstance(score, (int, float)):
            macro_trend["score"] = float(score)
        metadata["macro_trend"] = macro_trend
        if state:
            metadata["regime"] = {"state": state}


def _install_trigger_aware_telegram_patch() -> None:
    """Attach trigger metadata to caption, fallback text and PNG chart."""
    if getattr(TradingBot, "_tg_entry_trigger_patch_installed", False):
        return

    original_execute_signal = TradingBot._execute_signal
    original_build_caption = TelegramNotifier.build_order_caption
    original_notify = TelegramNotifier.notify
    original_render_entry_chart = chart_renderer.render_entry_chart

    async def _execute_signal_with_trigger(self, signal, *args, **kwargs):
        _normalize_notification_metadata(signal)
        token = _TG_ENTRY_CONTEXT.set(_entry_trigger_context(signal))
        try:
            return await original_execute_signal(self, signal, *args, **kwargs)
        finally:
            _TG_ENTRY_CONTEXT.reset(token)

    def _build_caption_with_trigger(self, *args, **kwargs):
        caption = original_build_caption(self, *args, **kwargs)
        context = _TG_ENTRY_CONTEXT.get() or {}
        return _append_entry_trigger(caption, context.get("label"))

    def _notify_with_trigger(self, text: str):
        context = _TG_ENTRY_CONTEXT.get() or {}
        label = context.get("label")
        if label and (
            "Order Executed" in str(text)
            or "OPEN LONG" in str(text)
            or "OPEN SHORT" in str(text)
        ):
            text = _append_entry_trigger(str(text), label)
        return original_notify(self, text)

    def _render_entry_chart_with_trigger(*args, **kwargs):
        context = _TG_ENTRY_CONTEXT.get() or {}
        label = context.get("label")
        metadata = context.get("metadata") or {}
        if label:
            kwargs["entry_trigger"] = label
            kwargs["strategy"] = "Trend Confirm"
            kwargs["t1_pct"] = 0.006
            kwargs["t1_trim_pct"] = 0.40
            kwargs["t1_lock_pct"] = 0.003

            if label == "WT Cross":
                kwargs["lower_panel"] = "wt"
                kwargs["wt_channel_length"] = 10
                kwargs["wt_average_length"] = 21
                kwargs["wt_signal_length"] = 4
                kwargs["wt_oversold"] = -42.0
                kwargs["wt_overbought"] = 45.0
            elif label == "Structure BOS + Retest":
                structure = metadata.get("structure_15m")
                structure = structure if isinstance(structure, dict) else {}
                kwargs["lower_panel"] = "structure"
                kwargs["structure_level"] = (
                    metadata.get("structure_level")
                    or structure.get("level")
                )
                kwargs["structure_breakout_ts"] = (
                    metadata.get("structure_breakout_ts")
                    or structure.get("breakout_ts")
                )
                kwargs["structure_retest_ts"] = (
                    metadata.get("structure_retest_ts")
                    or structure.get("retest_ts")
                )
            else:
                kwargs["lower_panel"] = "macd"

        return original_render_entry_chart(*args, **kwargs)

    TradingBot._execute_signal = _execute_signal_with_trigger
    TelegramNotifier.build_order_caption = _build_caption_with_trigger
    TelegramNotifier.notify = _notify_with_trigger
    chart_renderer.render_entry_chart = _render_entry_chart_with_trigger
    TradingBot._tg_entry_trigger_patch_installed = True


def _make_merged_trend_confirm(symbols: list, config: dict):
    if not run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True):
        raise RuntimeError(
            "Unified Trend Confirm is disabled. Set ENABLE_TREND_CONFIRM=true"
        )

    from trading.strategies.trend_confirm_wt_fixed_strategy import (
        TrendConfirmWTFixedStrategy,
    )

    return [
        TrendConfirmWTFixedStrategy(
            symbol,
            wt_oversold=-42.0,
            wt_overbought=45.0,
            structure_swing_span=_env_int("STRUCTURE_SWING_SPAN", 3),
            structure_retest_min_bars=_env_int("STRUCTURE_RETEST_MIN_BARS", 1),
            structure_retest_max_bars=_env_int("STRUCTURE_RETEST_MAX_BARS", 3),
            structure_bos_buffer_atr=_env_float("STRUCTURE_BOS_BUFFER_ATR", 0.05),
            structure_touch_tolerance_atr=_env_float(
                "STRUCTURE_TOUCH_TOLERANCE_ATR", 0.15
            ),
            structure_invalidation_tolerance_atr=_env_float(
                "STRUCTURE_INVALIDATION_TOLERANCE_ATR", 0.25
            ),
            structure_max_close_distance_atr=_env_float(
                "STRUCTURE_MAX_CLOSE_DISTANCE_ATR", 0.50
            ),
            structure_max_fill_slippage_atr=_env_float(
                "STRUCTURE_MAX_FILL_SLIPPAGE_ATR", 0.35
            ),
        )
        for symbol in symbols
    ]


def _build_merged_config() -> dict:
    config = run_dual_bot._ORIGINAL_BUILD_CONFIG()
    if not run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True):
        raise RuntimeError(
            "Unified Trend Confirm is disabled. Set ENABLE_TREND_CONFIRM=true"
        )

    strategy_limit = max(1, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2")))
    requested_global = max(1, int(os.getenv("MAX_POSITIONS", str(strategy_limit))))
    config["max_positions"] = min(requested_global, strategy_limit)
    config["strategy_mode"] = "trend_confirm_ema_wt_structure"
    config["enable_trend_confirm"] = True
    config["enable_wt_trend"] = False
    config["enable_ai_expert"] = False
    config["wt_oversold"] = -42.0
    config["wt_overbought"] = 45.0
    config["structure_entry_enabled"] = True
    os.environ["CANDLE_TF"] = "15m"
    config["candle_tf"] = "15m"
    return config


_install_trigger_aware_telegram_patch()
run_bot._make_strategies = _make_merged_trend_confirm
run_bot.build_config = _build_merged_config


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
