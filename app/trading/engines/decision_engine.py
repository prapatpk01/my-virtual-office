"""
Layer 3 + 4: Decision Engine & Confidence Filter

Combines expert scores into Long/Short scores using regime-adaptive weights.
Applies confidence thresholds before passing signals downstream.

Weights (default, adaptive per regime):
  Trend      30%
  Momentum   15%
  Liquidity  20%
  Volume     15%
  Volatility 10%
  Session    10%

Confidence Levels:
  < 70  → IGNORE
  70-80 → WEAK
  80-90 → GOOD
  90+   → HIGH_CONVICTION
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .market_intelligence import MarketRegime, RegimeResult
from .expert_analysis import ExpertScores


class ConfidenceLevel(str, Enum):
    IGNORE          = "ignore"          # < 70
    WEAK            = "weak"            # 70-80
    GOOD            = "good"            # 80-90
    HIGH_CONVICTION = "high_conviction" # 90+


@dataclass
class DecisionResult:
    long_score:  float = 0.0
    short_score: float = 0.0
    direction:   str   = "hold"   # "long" | "short" | "hold"
    confidence:  float = 0.0      # 0-100
    confidence_level: ConfidenceLevel = ConfidenceLevel.IGNORE
    weights_used: dict = field(default_factory=dict)
    score_breakdown: dict = field(default_factory=dict)


# Default weights (sum = 1.0)
_DEFAULT_WEIGHTS = {
    "trend":      0.30,
    "momentum":   0.15,
    "liquidity":  0.20,
    "volume":     0.15,
    "volatility": 0.10,
    "session":    0.10,
}

# Regime-specific weight overrides
_REGIME_WEIGHTS: dict[MarketRegime, dict] = {
    MarketRegime.TREND: {
        "trend":      0.35,
        "momentum":   0.20,
        "liquidity":  0.15,
        "volume":     0.15,
        "volatility": 0.05,
        "session":    0.10,
    },
    MarketRegime.RANGE: {
        "trend":      0.15,
        "momentum":   0.20,
        "liquidity":  0.30,
        "volume":     0.15,
        "volatility": 0.10,
        "session":    0.10,
    },
    MarketRegime.BREAKOUT: {
        "trend":      0.20,
        "momentum":   0.15,
        "liquidity":  0.25,
        "volume":     0.25,
        "volatility": 0.05,
        "session":    0.10,
    },
    MarketRegime.REVERSAL: {
        "trend":      0.10,
        "momentum":   0.20,
        "liquidity":  0.30,
        "volume":     0.20,
        "volatility": 0.10,
        "session":    0.10,
    },
    MarketRegime.HIGH_VOLATILITY: {
        "trend":      0.20,
        "momentum":   0.15,
        "liquidity":  0.20,
        "volume":     0.15,
        "volatility": 0.20,
        "session":    0.10,
    },
    MarketRegime.LOW_VOLATILITY: {
        "trend":      0.30,
        "momentum":   0.10,
        "liquidity":  0.25,
        "volume":     0.10,
        "volatility": 0.15,
        "session":    0.10,
    },
}


class DecisionEngine:
    """Aggregates expert scores into a final Long/Short decision with confidence."""

    def __init__(self, min_score_threshold: float = 70.0):
        self.min_score = min_score_threshold
        # Adaptive weights (learned over time, initialized to defaults)
        self._adaptive_weights: dict[str, dict] = {}

    def decide(
        self,
        experts: ExpertScores,
        regime: RegimeResult,
        adaptive_weights: Optional[dict] = None,
    ) -> DecisionResult:
        """Compute Long/Short scores and confidence level."""
        weights = self._resolve_weights(regime.regime, adaptive_weights)

        # Expert scores are 0-100 where >50 is bullish, <50 is bearish
        long_components = {
            "trend":      experts.trend,
            "momentum":   experts.momentum,
            "liquidity":  experts.liquidity,
            "volume":     experts.volume,
            "volatility": experts.volatility,
            "session":    experts.session,
        }

        # Long score: weighted sum of bullish expert signals
        long_score = sum(long_components[k] * weights[k] for k in weights)

        # Short score: invert directional experts, keep neutral ones (vol/session)
        short_components = {
            "trend":      100 - experts.trend,
            "momentum":   100 - experts.momentum,
            "liquidity":  100 - experts.liquidity,
            "volume":     100 - experts.volume,
            "volatility": experts.volatility,   # same for both
            "session":    experts.session,       # same for both
        }
        short_score = sum(short_components[k] * weights[k] for k in weights)

        # Apply correlation expert as a multiplier (0.8-1.2x)
        corr_factor = 0.8 + (experts.correlation / 100.0) * 0.4
        if experts.direction_bias > 0:
            long_score  = min(100.0, long_score  * corr_factor)
        else:
            short_score = min(100.0, short_score * corr_factor)

        long_score  = round(long_score,  1)
        short_score = round(short_score, 1)

        # Direction and confidence
        if long_score >= short_score:
            direction  = "long"
            confidence = long_score
        else:
            direction  = "short"
            confidence = short_score

        # Apply minimum threshold gate
        if confidence < self.min_score:
            direction = "hold"

        confidence_level = self._classify_confidence(confidence)

        return DecisionResult(
            long_score=long_score,
            short_score=short_score,
            direction=direction,
            confidence=confidence,
            confidence_level=confidence_level,
            weights_used=weights,
            score_breakdown={
                "long":  {k: round(long_components[k]  * weights[k], 2) for k in weights},
                "short": {k: round(short_components[k] * weights[k], 2) for k in weights},
            },
        )

    def update_adaptive_weights(
        self, regime: MarketRegime, performance_map: dict[str, float]
    ) -> None:
        """Adjust weights based on which experts contributed to wins.
        performance_map: {expert_name: contribution_score} from learning engine.
        """
        base = dict(_REGIME_WEIGHTS.get(regime, _DEFAULT_WEIGHTS))
        total_perf = sum(abs(v) for v in performance_map.values()) + 1e-9

        for key in base:
            if key in performance_map:
                contribution = performance_map[key] / total_perf
                # Nudge weight towards performance contribution (10% learning rate)
                base[key] = base[key] * 0.9 + contribution * 0.1

        # Normalize so weights sum to 1
        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}

        self._adaptive_weights[regime.value] = base

    def get_weights(self, regime: MarketRegime) -> dict:
        return self._adaptive_weights.get(regime.value, _REGIME_WEIGHTS.get(regime, _DEFAULT_WEIGHTS))

    def _resolve_weights(self, regime: MarketRegime, override: Optional[dict]) -> dict:
        if override:
            return override
        # Use adaptive if available, else regime-specific, else default
        return (
            self._adaptive_weights.get(regime.value) or
            _REGIME_WEIGHTS.get(regime) or
            _DEFAULT_WEIGHTS
        )

    @staticmethod
    def _classify_confidence(score: float) -> ConfidenceLevel:
        if score >= 90:
            return ConfidenceLevel.HIGH_CONVICTION
        if score >= 80:
            return ConfidenceLevel.GOOD
        if score >= 70:
            return ConfidenceLevel.WEAK
        return ConfidenceLevel.IGNORE
