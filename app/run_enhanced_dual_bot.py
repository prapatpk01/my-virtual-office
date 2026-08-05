"""Production entry point for merged Trend Confirm + corrected WaveTrend entry.

One strategy family only:
- Layer 1: 4H trend direction
- Layer 2: 1H context + ADX/CHOP quality gate
- Layer 3: 15M EMA8/13 cross OR WaveTrend extreme cross
- 15M price must be on the correct side of EMA20

WaveTrend entry extremes used in production:
- Long cross from oversold <= -42
- Short cross from overbought >= +45

WT is an entry trigger inside Trend Confirm, not a second strategy. Telegram
order alerts receive the exact entry-trigger owner from signal metadata and
show the matching entry trigger and signal-exit rule.
"""
from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar

import run_dual_bot  # keeps exchange, sleep, risk and lifecycle patches
import run_bot
from trading.bot import TradingBot
from trading.telegram_notifier import TelegramNotifier


_TG_ENTRY_TRIGGER: ContextVar[str | None] = ContextVar(
    "trend_confirm_tg_entry_trigger",
    default=None,
)


def _entry_trigger_label(signal) -> str | None:
    """Resolve the exact Layer-3 trigger from signal metadata."""
    metadata = getattr(signal, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    owner = str(
        metadata.get("entry_trigger_owner")
        or metadata.get("entry_trigger")
        or ""
    ).upper()
    if "WT" in owner:
        return "WT Cross"
    if "EMA" in owner:
        return "EMA8/13 Cross"
    return None


def _append_entry_trigger(text: str, label: str | None) -> str:
    """Show the entry trigger and its matching signal exit in an order alert."""
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

    exit_text = (
        "🏁 Signal Exit : `WT opposite cross`"
        if label == "WT Cross"
        else "🏁 Signal Exit : `EMA8/13 reverse cross`"
    )
    for index, line in enumerate(lines):
        if line.startswith("🏁 Exit") or line.startswith("🏁 Signal Exit"):
            lines[index] = exit_text
            break

    return "\n".join(lines)


def _install_telegram_entry_trigger_patch() -> None:
    """Bridge Signal metadata to the existing Telegram notifier safely.

    ContextVar keeps the trigger attached to the correct async order task even
    if multiple symbols are evaluated close together. The notifier's public API
    remains backward compatible for all other strategies and callers.
    """
    if getattr(TradingBot, "_tg_entry_trigger_patch_installed", False):
        return

    original_execute_signal = TradingBot._execute_signal
    original_build_caption = TelegramNotifier.build_order_caption
    original_notify = TelegramNotifier.notify

    async def _execute_signal_with_trigger(self, signal, *args, **kwargs):
        token = _TG_ENTRY_TRIGGER.set(_entry_trigger_label(signal))
        try:
            return await original_execute_signal(self, signal, *args, **kwargs)
        finally:
            _TG_ENTRY_TRIGGER.reset(token)

    def _build_caption_with_trigger(self, *args, **kwargs):
        caption = original_build_caption(self, *args, **kwargs)
        return _append_entry_trigger(caption, _TG_ENTRY_TRIGGER.get())

    def _notify_with_trigger(self, text: str):
        # Also covers bot.py's minimal fallback alert if chart/caption delivery
        # fails after the live order has already opened.
        label = _TG_ENTRY_TRIGGER.get()
        if label and ("Order Executed" in str(text) or "OPEN LONG" in str(text)
                      or "OPEN SHORT" in str(text)):
            text = _append_entry_trigger(str(text), label)
        return original_notify(self, text)

    TradingBot._execute_signal = _execute_signal_with_trigger
    TelegramNotifier.build_order_caption = _build_caption_with_trigger
    TelegramNotifier.notify = _notify_with_trigger
    TradingBot._tg_entry_trigger_patch_installed = True


def _make_merged_trend_confirm(symbols: list, config: dict):
    if not run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True):
        raise RuntimeError(
            "Merged Trend Confirm is disabled. Set ENABLE_TREND_CONFIRM=true"
        )

    from trading.strategies.trend_confirm_wt_fixed_strategy import (
        TrendConfirmWTFixedStrategy,
    )

    return [
        TrendConfirmWTFixedStrategy(
            symbol,
            wt_oversold=-42.0,
            wt_overbought=45.0,
        )
        for symbol in symbols
    ]


def _build_merged_config() -> dict:
    # Bypass the old dual quota sum. There is now only one strategy family.
    config = run_dual_bot._ORIGINAL_BUILD_CONFIG()
    if not run_dual_bot._env_bool("ENABLE_TREND_CONFIRM", True):
        raise RuntimeError(
            "Merged Trend Confirm is disabled. Set ENABLE_TREND_CONFIRM=true"
        )

    strategy_limit = max(1, int(os.getenv("TREND_CONFIRM_MAX_POSITIONS", "2")))
    requested_global = max(1, int(os.getenv("MAX_POSITIONS", str(strategy_limit))))
    config["max_positions"] = min(requested_global, strategy_limit)
    config["strategy_mode"] = "trend_confirm_ema_or_wt"
    config["enable_trend_confirm"] = True
    config["enable_wt_trend"] = False
    config["enable_ai_expert"] = False
    config["wt_oversold"] = -42.0
    config["wt_overbought"] = 45.0
    os.environ["CANDLE_TF"] = "15m"
    config["candle_tf"] = "15m"
    return config


_install_telegram_entry_trigger_patch()
run_bot._make_strategies = _make_merged_trend_confirm
run_bot.build_config = _build_merged_config


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
