"""Defensive production wrapper for EnhancedAIExpertStrategy.

Normalizes signal metadata and nested diagnostic fields before the enhanced
entry-quality and explainability layers consume them. This prevents malformed
legacy metadata (string/list/tuple values) from crashing an entire AI Expert
scan cycle.
"""
from __future__ import annotations

from typing import Any

from .base import Signal, SignalType
from .enhanced_ai_expert_strategy import EnhancedAIExpertStrategy


class SafeEnhancedAIExpertStrategy(EnhancedAIExpertStrategy):
    """Enhanced AI Expert with type-safe metadata handling."""

    _DICT_FIELDS = (
        "macro",
        "context",
        "regime",
        "selection",
        "strategy_setup",
        "confidence",
        "expectancy",
        "dynamic_risk",
        "entry_quality",
        "decision_trace",
    )

    @staticmethod
    def _as_dict(value: Any) -> dict:
        """Return a shallow dict copy only when value is actually a mapping."""
        return dict(value) if isinstance(value, dict) else {}

    def _normalize_metadata(self, signal: Signal) -> dict:
        metadata = self._as_dict(getattr(signal, "metadata", None))
        for field in self._DICT_FIELDS:
            metadata[field] = self._as_dict(metadata.get(field))
        signal.metadata = metadata
        return metadata

    def _build_decision_trace(self, signal: Signal) -> dict:
        """Build the same trace as the parent, using normalized dictionaries."""
        metadata = self._normalize_metadata(signal)
        macro = metadata["macro"]
        context = metadata["context"]
        regime = metadata["regime"]
        selection = metadata["selection"]
        setup = metadata["strategy_setup"]
        confidence = metadata["confidence"]
        expectancy = metadata["expectancy"]
        dynamic = metadata["dynamic_risk"]
        entry_quality = metadata["entry_quality"]

        strategy_name = str(
            self._pick(
                selection,
                "strategy",
                "selected",
                "selected_strategy",
                default=entry_quality.get("strategy", "none"),
            )
            or "none"
        )
        try:
            setup_score = float(setup.get("raw_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            setup_score = 0.0
        setup_progress = max(0, min(100, int(round(setup_score))))
        setup_reason = str(setup.get("reason") or getattr(signal, "reason", "") or "")
        try:
            expectancy_samples = int(expectancy.get("sample_size", 0) or 0)
        except (TypeError, ValueError):
            expectancy_samples = 0

        return {
            "macro_state": str(self._pick(macro, "state", "label", "trend", default="?")),
            "macro_score": self._pick(macro, "score", default=None),
            "context_bias": str(self._pick(context, "dominant_bias", "bias", "state", default="?")),
            "context_score": self._pick(context, "quality", "score", "bias_score", default=None),
            "regime": str(self._pick(regime, "primary", "state", "label", default="?")),
            "volatility_state": str(self._pick(regime, "secondary", "volatility", default="?")),
            "strategy": strategy_name,
            "setup_valid": bool(setup.get("valid", False)),
            "setup_score": round(setup_score, 1),
            "setup_progress_pct": setup_progress,
            "waiting_for": self._short_reason(setup_reason),
            "confidence": self._pick(confidence, "score", default=None),
            "confidence_level": self._pick(confidence, "level", default=None),
            "expectancy_r": self._pick(expectancy, "expectancy_r", default=None),
            "expectancy_samples": expectancy_samples,
            "expected_rr": metadata.get("rr_ratio"),
            "live_rr": entry_quality.get("live_rr"),
            "risk_multiplier": self._pick(dynamic, "risk_multiplier", default=None),
            "decision": str(getattr(signal.type, "value", signal.type)).upper(),
        }

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
