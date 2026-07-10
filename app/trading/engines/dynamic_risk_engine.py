"""
Layer 8: Dynamic Risk Engine.

Risk is never fixed. Given everything the pipeline has established by
this point — market quality, regime confidence, trade confidence,
historical expectancy, current volatility state, and portfolio
correlation — this layer outputs a single risk_multiplier (applied to
the account's base risk-per-trade %) plus regime-aware partial-TP /
break-even parameters. It does not touch entry/SL/TP prices — those
belong to Layer 5's strategy-specific logic — only how much size and
how the position gets managed once open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .market_quality_engine import MarketQualityResult
from .regime_classifier import RegimeResult, PrimaryRegime, SecondaryState
from .confidence_engine import ConfidenceResult
from .expectancy_engine import ExpectancyResult


@dataclass
class RiskDecision:
    risk_multiplier: float       # applied on top of the account's base risk %
    tp1_rr: float                 # partial-TP trigger, in R
    tp2_rr: float                 # full-TP trigger, in R
    partial_close_pct: float      # fraction closed at TP1
    trail_after_tp1: bool         # trail the remainder instead of a fixed TP2
    detail: dict = field(default_factory=dict)


class DynamicRiskEngine:
    def __init__(
        self,
        base_tp1_rr: float = 0.6,
        base_tp2_rr: float = 1.2,
        min_multiplier: float = 0.35,
        max_multiplier: float = 1.5,
    ):
        self.base_tp1_rr = base_tp1_rr
        self.base_tp2_rr = base_tp2_rr
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def compute(
        self,
        quality: MarketQualityResult,
        regime: RegimeResult,
        confidence: ConfidenceResult,
        expectancy: ExpectancyResult,
        correlation_score: float = 50.0,   # ExpertScores.correlation, 0-100
    ) -> RiskDecision:
        detail: dict = {}

        # ── Size multiplier: blend quality/confidence/expectancy signals ────
        quality_factor = quality.score / 100.0
        confidence_factor = confidence.score / 100.0
        # Expectancy nudges size up/down around 1.0x; capped so one hot streak
        # can't blow sizing out, and a cold/unknown streak doesn't zero it.
        expectancy_factor = 1.0 + max(-0.3, min(0.3, expectancy.expectancy_r * 2.0))
        # High correlation with the rest of the portfolio trims size (the
        # Portfolio Engine still hard-gates concentration separately).
        correlation_factor = 1.0 - max(0.0, (correlation_score - 50.0) / 100.0) * 0.3

        secondary_factor = {
            SecondaryState.LOW_VOLATILITY: 1.05,
            SecondaryState.NORMAL_VOLATILITY: 1.0,
            SecondaryState.HIGH_VOLATILITY: 0.75,
            SecondaryState.EXPANSION: 0.85,
        }.get(regime.secondary, 1.0)

        multiplier = quality_factor * confidence_factor * expectancy_factor \
            * correlation_factor * secondary_factor
        multiplier = round(max(self.min_multiplier, min(self.max_multiplier, multiplier)), 3)

        detail.update({
            "quality_factor": round(quality_factor, 3),
            "confidence_factor": round(confidence_factor, 3),
            "expectancy_factor": round(expectancy_factor, 3),
            "correlation_factor": round(correlation_factor, 3),
            "secondary_factor": secondary_factor,
        })

        # ── Position management: regime-aware partial TP / trail ────────────
        tp1_rr, tp2_rr, partial_pct, trail = self._position_plan(regime.primary, regime.secondary)

        return RiskDecision(
            risk_multiplier=multiplier, tp1_rr=tp1_rr, tp2_rr=tp2_rr,
            partial_close_pct=partial_pct, trail_after_tp1=trail, detail=detail,
        )

    def _position_plan(self, primary: PrimaryRegime, secondary: SecondaryState):
        """Trending/expanding regimes get more room to run (trail instead of
        a hard TP2); mean-reversion/reversal regimes take profit faster."""
        if primary in (PrimaryRegime.BULL_TREND, PrimaryRegime.BEAR_TREND) or secondary == SecondaryState.EXPANSION:
            return self.base_tp1_rr, self.base_tp2_rr * 1.5, 0.5, True
        if primary in (PrimaryRegime.RANGE,):
            return self.base_tp1_rr * 0.8, self.base_tp2_rr * 0.8, 0.5, False
        if primary == PrimaryRegime.REVERSAL:
            return self.base_tp1_rr * 0.75, self.base_tp2_rr, 0.5, False
        return self.base_tp1_rr, self.base_tp2_rr, 0.5, False
