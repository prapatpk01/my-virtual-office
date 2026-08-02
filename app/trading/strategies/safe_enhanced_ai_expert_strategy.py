"""Defensive production wrapper for EnhancedAIExpertStrategy.

Normalizes only fields that are truly mappings and maps explainability to the
actual metadata contract emitted by AIExpertStrategy._base_metadata().
"""
from __future__ import annotations

from typing import Any

from .base import Signal, SignalType
from .enhanced_ai_expert_strategy import EnhancedAIExpertStrategy


class SafeEnhancedAIExpertStrategy(EnhancedAIExpertStrategy):
    """Enhanced AI Expert with type-safe, contract-aware diagnostics."""

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

        # Keep the engine's exact reason as the authoritative missing/waiting
        # explanation; merely shorten it for Railway readability.
        waiting = text
        return passed[:6], waiting

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
        # Before Layer 6 runs, show the selector/regime confidence rather than
        # an empty value. This is diagnostic only and does not alter gating.
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
