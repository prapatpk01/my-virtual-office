"""Defensive production wrapper for EnhancedAIExpertStrategy.

Normalizes only fields that are truly mappings and maps explainability to the
actual metadata contract emitted by AIExpertStrategy._base_metadata().

It also applies a final 15M stop-distance guard after a valid AI Expert signal:
- preserve the strategy's structural stop when it is already wide enough;
- widen stops that are too close using both ATR and percentage floors;
- reject entries whose required structural stop exceeds the configured cap;
- rebuild TP from the final stop distance and the strategy's original R:R.
"""
from __future__ import annotations

from typing import Any

from .base import Signal, SignalType
from .enhanced_ai_expert_strategy import EnhancedAIExpertStrategy


_STOP_RULES = {
    "trend_continuation": {"pct_floor": 0.55, "atr_floor": 0.80, "cap_pct": 1.20},
    "momentum_expansion": {"pct_floor": 0.65, "atr_floor": 1.00, "cap_pct": 1.20},
    "breakout": {"pct_floor": 0.70, "atr_floor": 1.00, "cap_pct": 1.20},
    "mean_reversion": {"pct_floor": 0.60, "atr_floor": 0.90, "cap_pct": 1.20},
    "swing_reversal": {"pct_floor": 0.75, "atr_floor": 1.10, "cap_pct": 1.20},
}


class SafeEnhancedAIExpertStrategy(EnhancedAIExpertStrategy):
    """Enhanced AI Expert with safe diagnostics and 15M stop-distance floors."""

    _MAPPING_FIELDS = (
        "market_quality",
        "macro_trend",
        "context_1h",
        "regime_scores",
        "mtf_combined",
        "strategy_setup",
        "confidence",
        "expectancy",
        "dynamic_risk",
        "entry_quality",
        "decision_trace",
    )

    @staticmethod
    def _as_dict(value: Any) -> dict:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _normalize_metadata(self, signal: Signal) -> dict:
        metadata = self._as_dict(getattr(signal, "metadata", None))
        for field in self._MAPPING_FIELDS:
            metadata[field] = self._as_dict(metadata.get(field))
        # Do not normalize scalar fields such as regime, regime_secondary,
        # selected_strategy or direction_filter into dictionaries.
        signal.metadata = metadata
        return metadata

    @staticmethod
    def _setup_checks(reason: str) -> tuple[list[str], str]:
        """Extract compact passed checks and the current waiting condition."""
        text = " ".join(str(reason or "").split())
        if not text:
            return [], "No setup detail"

        passed: list[str] = []
        lower = text.lower()
        known = (
            ("ema20_50_trend", "EMA20/50 trend"),
            ("ema20_pullback_reclaim", "EMA20 pullback/reclaim"),
            ("fresh_ema8_13_cross", "EMA8/13 fresh cross"),
            ("adx_di", "ADX/DI confirmation"),
            ("liquidity_sweep", "Liquidity sweep"),
            ("fresh_ema8_13_reversal", "EMA8/13 reversal"),
            ("closed_bos", "Closed BOS"),
            ("fresh_retest_hold", "Retest hold"),
            ("body_or_retest_confirm", "Candle/retest confirmation"),
            ("volatility_expanding", "Volatility expansion"),
            ("fresh_micro_break", "Fresh micro breakout"),
            ("ema_stack", "EMA alignment"),
            ("choch", "CHOCH"),
            ("engulfing", "Engulfing candle"),
        )
        for token, label in known:
            if token in lower:
                passed.append(label)

        return passed[:6], text

    def _build_decision_trace(self, signal: Signal) -> dict:
        metadata = self._normalize_metadata(signal)
        macro = metadata["macro_trend"]
        context = metadata["context_1h"]
        mtf = metadata["mtf_combined"]
        setup = metadata["strategy_setup"]
        confidence = metadata["confidence"]
        expectancy = metadata["expectancy"]
        dynamic = metadata["dynamic_risk"]
        entry_quality = metadata["entry_quality"]

        setup_score = self._safe_float(setup.get("raw_score"), 0.0)
        setup_progress = max(0, min(100, int(round(setup_score))))
        original_reason = str(setup.get("reason") or getattr(signal, "reason", "") or "")
        passed_checks, waiting_for = self._setup_checks(original_reason)

        macro_score = macro.get("score")
        macro_bias = macro.get("bias", "?")
        macro_stage = macro.get("stage", "?")
        context_bias = context.get("dominant_bias", "?")
        bull_score = self._safe_float(context.get("bull_score"), 0.0)
        bear_score = self._safe_float(context.get("bear_score"), 0.0)
        context_score = max(bull_score, bear_score)

        selected_strategy = metadata.get("selected_strategy") or entry_quality.get("strategy") or "none"
        regime = metadata.get("regime") or "?"
        regime_secondary = metadata.get("regime_secondary") or "?"

        confidence_score = confidence.get("score")
        if confidence_score is None:
            confidence_score = metadata.get("strategy_confidence")

        rr_value = entry_quality.get("live_rr")
        if rr_value is None:
            rr_value = metadata.get("rr_ratio")

        return {
            "macro_state": str(macro_bias),
            "macro_stage": str(macro_stage),
            "macro_score": macro_score,
            "context_bias": str(context_bias),
            "context_stage": str(context.get("stage", "?")),
            "context_score": round(context_score, 1),
            "regime": str(regime),
            "volatility_state": str(regime_secondary),
            "regime_confidence": metadata.get("regime_confidence"),
            "strategy": str(selected_strategy),
            "direction_filter": metadata.get("direction_filter", "?"),
            "mtf_pct": mtf.get("pct"),
            "mtf_aligned": mtf.get("aligned_1h_4h"),
            "setup_valid": bool(setup.get("valid", False)),
            "setup_score": round(setup_score, 1),
            "setup_progress_pct": setup_progress,
            "passed_checks": passed_checks,
            "waiting_for": self._short_reason(waiting_for, limit=140),
            "confidence": confidence_score,
            "confidence_level": confidence.get("level"),
            "expectancy_r": expectancy.get("expectancy_r"),
            "expectancy_samples": self._safe_int(expectancy.get("sample_size"), 0),
            "expected_rr": metadata.get("rr_ratio"),
            "live_rr": rr_value,
            "risk_multiplier": dynamic.get("risk_multiplier"),
            "decision": str(getattr(signal.type, "value", signal.type)).upper(),
        }

    def _trace_summary(self, trace: dict) -> str:
        macro_score = self._fmt_value(trace.get("macro_score"), 0)
        context_score = self._fmt_value(trace.get("context_score"), 0)
        confidence = self._fmt_value(trace.get("confidence"), 0, "---")
        rr_text = self._fmt_value(trace.get("live_rr") or trace.get("expected_rr"), 2, "---")
        exp_text = self._fmt_value(trace.get("expectancy_r"), 2, "---")
        mtf_text = self._fmt_value(trace.get("mtf_pct"), 0, "---")

        checks = trace.get("passed_checks") or []
        checks_text = ", ".join(f"✓{item}" for item in checks[:3])
        if checks_text:
            checks_text = f" [{checks_text}]"

        return (
            f"L1={trace.get('macro_state')}/{trace.get('macro_stage')}({macro_score}) "
            f"L2={trace.get('context_bias')}/{trace.get('context_stage')}({context_score}) "
            f"L3={trace.get('regime')}/{trace.get('volatility_state')} "
            f"L4={trace.get('strategy')}({trace.get('direction_filter')}) "
            f"MTF={mtf_text}% SETUP={trace.get('setup_progress_pct', 0)}% "
            f"CONF={confidence} EXP={exp_text}R RR={rr_text}{checks_text} | "
            f"{trace.get('waiting_for') or trace.get('decision')}"
        )

    def _attach_decision_trace(self, signal: Signal, replace_hold_reason: bool = True) -> Signal:
        metadata = self._normalize_metadata(signal)
        trace = self._build_decision_trace(signal)
        metadata["decision_trace"] = trace
        metadata["setup_progress_pct"] = trace["setup_progress_pct"]
        metadata["waiting_for"] = trace["waiting_for"]
        metadata["expected_rr"] = trace.get("live_rr") or trace.get("expected_rr")
        signal.metadata = metadata

        if replace_hold_reason and signal.type == SignalType.HOLD:
            signal.reason = self._trace_summary(trace)
        return signal

    def _blocked_hold(self, signal: Signal, reason: str, extra: dict) -> Signal:
        self._cancel_generated_entry(reason)
        metadata = self._normalize_metadata(signal)
        entry_quality = metadata["entry_quality"]
        entry_quality.update(self._as_dict(extra))
        entry_quality["passed"] = False
        entry_quality["block_reason"] = reason
        metadata["entry_quality"] = entry_quality
        hold = self._hold(float(signal.price), reason=reason, metadata=metadata)
        return self._attach_decision_trace(hold, replace_hold_reason=True)

    def _stop_rule(self, strategy_type: str) -> dict:
        base = dict(_STOP_RULES.get(strategy_type, {
            "pct_floor": 0.60, "atr_floor": 0.90, "cap_pct": 1.20,
        }))
        key = strategy_type.upper() if strategy_type else "DEFAULT"
        base["pct_floor"] = self._env_float(
            f"AI_SL_MIN_PCT_{key}", base["pct_floor"]
        )
        base["atr_floor"] = self._env_float(
            f"AI_SL_MIN_ATR_{key}", base["atr_floor"]
        )
        base["cap_pct"] = self._env_float(
            f"AI_SL_MAX_PCT_{key}", base["cap_pct"]
        )
        return base

    def _apply_stop_distance_guard(self, signal: Signal, candles: list) -> Signal:
        """Widen too-tight SLs and rebuild TP while preserving strategy R:R."""
        if signal.type not in (SignalType.BUY, SignalType.SELL):
            return signal

        metadata = self._normalize_metadata(signal)
        entry = self._open_entry or {}
        strategy_type = str(
            entry.get("strategy_type") or metadata.get("selected_strategy") or ""
        ).lower()
        direction = str(entry.get("direction") or (
            "long" if signal.type == SignalType.BUY else "short"
        )).lower()
        price = float(signal.price)
        original_sl = self._safe_float(
            metadata.get("stop_loss", entry.get("stop_loss")), 0.0
        )
        original_tp = self._safe_float(
            metadata.get("take_profit", entry.get("take_profit")), 0.0
        )
        original_rr = self._safe_float(metadata.get("rr_ratio"), 0.0)

        if original_sl <= 0 or price <= 0:
            return self._blocked_hold(
                signal,
                "Stop guard: invalid entry or stop price",
                {"strategy": strategy_type, "original_sl": original_sl},
            )

        atr14 = self._atr(candles, 14)
        rule = self._stop_rule(strategy_type)
        pct_floor_distance = price * rule["pct_floor"] / 100.0
        atr_floor_distance = atr14 * rule["atr_floor"] if atr14 > 0 else 0.0
        min_distance = max(pct_floor_distance, atr_floor_distance)
        max_distance = price * rule["cap_pct"] / 100.0
        original_distance = abs(price - original_sl)

        # A structure stop beyond the cap is not compressed toward price;
        # the setup is rejected because doing so would invalidate the strategy.
        if original_distance > max_distance:
            return self._blocked_hold(
                signal,
                (
                    f"Stop guard: structural SL {original_distance / price * 100:.2f}% "
                    f"> max {rule['cap_pct']:.2f}%"
                ),
                {
                    "strategy": strategy_type,
                    "original_sl_pct": round(original_distance / price * 100, 3),
                    "max_sl_pct": rule["cap_pct"],
                },
            )

        final_distance = max(original_distance, min(min_distance, max_distance))
        final_sl = price - final_distance if direction == "long" else price + final_distance

        # Preserve the engine's intended R:R. If metadata is unavailable,
        # derive it from its original TP/SL, then fall back to 1.2R.
        if original_rr <= 0 and original_tp > 0 and original_distance > 0:
            original_reward = abs(original_tp - price)
            original_rr = original_reward / original_distance
        target_rr = max(original_rr, 1.0)
        final_tp = price + target_rr * final_distance if direction == "long" else price - target_rr * final_distance

        metadata["stop_loss"] = round(final_sl, 8)
        metadata["take_profit"] = round(final_tp, 8)
        metadata["rr_ratio"] = round(target_rr, 2)
        entry_quality = metadata["entry_quality"]
        entry_quality["stop_guard"] = {
            "applied": final_distance > original_distance + 1e-12,
            "strategy": strategy_type,
            "original_sl": round(original_sl, 8),
            "final_sl": round(final_sl, 8),
            "final_tp": round(final_tp, 8),
            "original_sl_pct": round(original_distance / price * 100, 3),
            "final_sl_pct": round(final_distance / price * 100, 3),
            "pct_floor": rule["pct_floor"],
            "atr_floor": rule["atr_floor"],
            "max_pct": rule["cap_pct"],
            "atr14": round(atr14, 8),
            "target_rr": round(target_rr, 2),
        }
        entry_quality["live_rr"] = round(target_rr, 3)
        metadata["entry_quality"] = entry_quality
        signal.metadata = metadata

        if self._open_entry is not None:
            self._open_entry["stop_loss"] = round(final_sl, 8)
            self._open_entry["take_profit"] = round(final_tp, 8)
            self._open_entry["stop_guard_applied"] = final_distance > original_distance + 1e-12

        return self._attach_decision_trace(signal, replace_hold_reason=False)

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        signal = await super().analyze(candles, current_price, mtf_candles)
        return self._apply_stop_distance_guard(signal, candles)
