"""Production entry point for unified Trend Confirm.

Closed 15M is always the only entry authority:
- EMA8/13 cross,
- WaveTrend extreme cross (-42 / +45), or
- confirmed Structure BOS + retest.

Direction mode is runtime-selectable from Railway:
- USE_LAYER1_4H=true  -> 4H direction -> 1H quality -> 15M trigger.
- USE_LAYER1_4H=false -> 1H bias+quality -> 15M trigger (4H fully bypassed).

Telegram text and PNG charts use entry-trigger metadata so every position shows
the correct owner and exit rule.
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


def _layer1_4h_enabled() -> bool:
    """Default ON so existing Railway deployments keep current behaviour."""
    return run_dual_bot._env_bool("USE_LAYER1_4H", True)


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


def _append_entry_trigger(
    text: str,
    label: str | None,
    metadata: dict | None = None,
) -> str:
    """Rewrite the order caption to match the actual entry owner and bias mode."""
    if not label:
        return text

    metadata = metadata if isinstance(metadata, dict) else {}
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

    layer1_enabled = metadata.get("layer1_4h_enabled") is not False
    for index, line in enumerate(lines):
        if line.startswith("🧭 4H Macro:"):
            if " (0/100)" in line:
                line = line.replace(" (0/100)", "")
            if not layer1_enabled:
                line = line.replace("🧭 4H Macro:", "🧭 1H Bias:", 1)
            lines[index] = line

    return "\n".join(lines)


def _normalize_notification_metadata(signal) -> None:
    """Expose the active direction authority to the generic notifier."""
    metadata = getattr(signal, "metadata", None)
    if not isinstance(metadata, dict):
        return

    metadata["selected_strategy"] = "Trend Confirm"
    layer1_enabled = metadata.get("layer1_4h_enabled") is not False

    if not layer1_enabled:
        ctx = metadata.get("context_1h")
        ctx = ctx if isinstance(ctx, dict) else {}
        bias = str(ctx.get("bias") or "").upper()
        score = ctx.get("score")
        macro_trend = {}
        if bias in ("LONG", "SHORT"):
            macro_trend["bias"] = f"1H_{bias}"
            metadata["regime"] = {"state": f"1H_{bias}"}
        if isinstance(score, (int, float)):
            macro_trend["score"] = float(score)
        metadata["macro_trend"] = macro_trend
        return

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
        return _append_entry_trigger(
            caption,
            context.get("label"),
            context.get("metadata"),
        )

    def _notify_with_trigger(self, text: str):
        context = _TG_ENTRY_CONTEXT.get() or {}
        label = context.get("label")
        if label and (
            "Order Executed" in str(text)
            or "OPEN LONG" in str(text)
            or "OPEN SHORT" in str(text)
        ):
            text = _append_entry_trigger(
                str(text),
                label,
                context.get("metadata"),
            )
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

    class RuntimeLayer1TrendConfirm(TrendConfirmWTFixedStrategy):
        """Production strategy with Railway-switchable 4H direction authority."""

        def __init__(self, *args, use_layer1_4h: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_layer1_4h = bool(use_layer1_4h)
            self._layer1_override_macro = None

        def _macro_trend_4h(self, candles_4h: list) -> dict:
            if self.use_layer1_4h:
                return super()._macro_trend_4h(candles_4h)
            if isinstance(self._layer1_override_macro, dict):
                return dict(self._layer1_override_macro)
            return {
                "state": "LAYER1_OFF_WAIT_1H",
                "direction": None,
                "score": None,
                "bull_votes": 0,
                "bear_votes": 0,
                "signals": {"source": "1h_bias_quality"},
                "bars": 0,
                "layer1_enabled": False,
                "source": "1h_bias_quality",
            }

        def _diag_update(self, **values) -> None:
            if not self.use_layer1_4h:
                if "trend_4h" in values:
                    values["trend_4h"] = "OFF"
                if values.get("entry_state") == "4H_TREND":
                    values["entry_state"] = "1H_BIAS"
                mtf_text = values.get("mtf")
                if isinstance(mtf_text, str) and mtf_text.startswith("4H="):
                    if " | " in mtf_text:
                        values["mtf"] = "4H=OFF | " + mtf_text.split(" | ", 1)[1]
                    else:
                        values["mtf"] = "4H=OFF"
            return super()._diag_update(**values)

        async def analyze(
            self,
            candles: list,
            current_price: float,
            mtf_candles: dict = None,
        ):
            if self.use_layer1_4h:
                signal = await super().analyze(candles, current_price, mtf_candles)
                if isinstance(getattr(signal, "metadata", None), dict):
                    signal.metadata["layer1_4h_enabled"] = True
                    signal.metadata["direction_source"] = "4H_DIRECTION_THEN_1H_QUALITY"
                return signal

            # Layer1 OFF: keep only the 15M + 1H data-quality checks. 4H data
            # availability must never block this mode because 4H is not used.
            self._diag_reset()
            mtf = mtf_candles or {}
            if not candles:
                return self._hold(
                    current_price,
                    "Data Quality: empty 15M candle series",
                    metadata={
                        "layer1_4h_enabled": False,
                        "direction_source": "1H_BIAS_QUALITY",
                    },
                )

            c15 = self._closed_candle_series(
                candles,
                15 * 60_000,
                self.closed_bar_grace_ms,
            )
            c1h_raw = mtf.get("1h", []) or self._resample_timeframe(
                c15,
                60 * 60_000,
                15 * 60_000,
            )
            c1h = self._closed_candle_series(
                c1h_raw,
                60 * 60_000,
                self.closed_bar_grace_ms,
            )

            self._latest_candles = c15
            self._latest_15m = c15
            self._latest_5m = c15

            min15 = max(self.ema_slow + 3, self.atr_period + 3)
            if len(c15) < min15:
                return self._hold(
                    current_price,
                    f"15M warm-up: need {min15}+ closed bars, have {len(c15)}",
                    metadata={
                        "layer1_4h_enabled": False,
                        "direction_source": "1H_BIAS_QUALITY",
                    },
                )

            data_quality = {}
            if self.use_data_quality_gate:
                for tf_name, series, expected_ms in (
                    ("15m", c15, 15 * 60_000),
                    ("1h", c1h, 60 * 60_000),
                ):
                    quality = (
                        self._data_quality_context(series, expected_ms)
                        if series
                        else {"valid": False, "reason": "missing", "bars": 0}
                    )
                    data_quality[tf_name] = quality
                    if not quality.get("valid"):
                        return self._hold(
                            current_price,
                            f"Data Quality FAIL {tf_name}: {quality.get('reason')}",
                            metadata={
                                "data_quality": data_quality,
                                "layer1_4h_enabled": False,
                                "direction_source": "1H_BIAS_QUALITY",
                            },
                        )

            long_ctx = self._context_1h(c1h, "long")
            short_ctx = self._context_1h(c1h, "short")
            if long_ctx is None and short_ctx is None:
                self._diag_update(
                    trend_4h="OFF",
                    trend_1h="WARMUP",
                    regime="1H_WARMUP",
                    aligned=False,
                    direction_15m="WAIT_CROSS",
                    mtf="4H=OFF | 1H=WARMUP",
                    entry_state="1H_BIAS",
                )
                return self._hold(
                    current_price,
                    "Layer1 4H OFF — 1H bias/quality warming up",
                    metadata={
                        "data_quality": data_quality,
                        "layer1_4h_enabled": False,
                        "direction_source": "1H_BIAS_QUALITY",
                    },
                )

            reference_ctx = long_ctx or short_ctx or {}
            bias = str(reference_ctx.get("bias") or "neutral").lower()
            if bias == "long":
                direction = "long"
                ctx = long_ctx
            elif bias == "short":
                direction = "short"
                ctx = short_ctx
            else:
                ctx = long_ctx or short_ctx or {}
                self._diag_update(
                    trend_4h="OFF",
                    trend_1h="NEUTRAL",
                    regime="1H_NEUTRAL",
                    aligned=False,
                    direction_15m="WAIT_CROSS",
                    mtf=(
                        "4H=OFF | 1H=NEUTRAL "
                        f"ADX={float(ctx.get('adx', 0.0)):.1f} "
                        f"CHOP={float(ctx.get('chop', 100.0)):.1f}"
                    ),
                    entry_state="1H_BIAS",
                )
                return self._hold(
                    current_price,
                    "Layer1 4H OFF — 1H bias is NEUTRAL; no trade",
                    metadata={
                        "context_1h": ctx,
                        "quality_1h": ctx,
                        "data_quality": data_quality,
                        "layer1_4h_enabled": False,
                        "direction_source": "1H_BIAS_QUALITY",
                    },
                )

            ctx = ctx or {}
            bull_votes = int(ctx.get("bull_votes", 0))
            bear_votes = int(ctx.get("bear_votes", 0))
            state = "1H_BULL" if direction == "long" else "1H_BEAR"
            synthetic_macro = {
                "state": state,
                "direction": direction,
                "score": ctx.get("score"),
                "bull_votes": bull_votes,
                "bear_votes": bear_votes,
                "signals": {
                    "source": "1h_bias_quality",
                    "bias": ctx.get("bias"),
                    "adx_ok": ctx.get("adx_ok"),
                    "chop_ok": ctx.get("chop_ok"),
                },
                "bars": len(c1h),
                "layer1_enabled": False,
                "source": "1h_bias_quality",
            }

            self._layer1_override_macro = synthetic_macro
            original_data_quality_gate = self.use_data_quality_gate
            self.use_data_quality_gate = False
            try:
                signal = await super().analyze(candles, current_price, mtf_candles)
            finally:
                self.use_data_quality_gate = original_data_quality_gate
                self._layer1_override_macro = None

            metadata = (
                signal.metadata
                if isinstance(getattr(signal, "metadata", None), dict)
                else {}
            )
            metadata.update({
                "macro_4h": synthetic_macro,
                "context_1h": ctx,
                "quality_1h": ctx,
                "data_quality": data_quality,
                "layer1_4h_enabled": False,
                "direction_source": "1H_BIAS_QUALITY",
            })
            signal.metadata = metadata

            reason = str(getattr(signal, "reason", "") or "")
            reason = reason.replace(
                "4H/1H trend aligned",
                "1H bias/quality aligned",
            ).replace(
                "4H/1H gates passed",
                "1H bias/quality gate passed",
            )
            if "+ 1H bias aligned" in reason and reason.startswith("4H "):
                suffix = reason.split("+ 1H bias aligned", 1)[1]
                reason = "Layer1 4H OFF + 1H bias aligned" + suffix
            signal.reason = reason
            return signal

    use_layer1_4h = _layer1_4h_enabled()
    return [
        RuntimeLayer1TrendConfirm(
            symbol,
            use_layer1_4h=use_layer1_4h,
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

    use_layer1_4h = _layer1_4h_enabled()
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
    config["use_layer1_4h"] = use_layer1_4h
    config["direction_mode"] = (
        "4H_DIRECTION_THEN_1H_QUALITY"
        if use_layer1_4h
        else "1H_BIAS_QUALITY_ONLY"
    )
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
