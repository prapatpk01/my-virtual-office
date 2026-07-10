"""
Layer 6: Confidence Engine — Trade Quality Gate (part 1 of 2, with Layer 7).

Takes the ONE valid StrategySignal that survived Layer 5 and blends it
with everything else the pipeline has learned about this bar (macro
alignment, regime confidence, RR quality, and the underlying expert
scores) into a single 0-100 Confidence Score.

Components:
  Signal Quality     25 pts — the strategy's own raw_score (Layer 5)
  Market Alignment   15 pts — does macro/regime direction agree with the trade
  Risk Alignment     15 pts — R:R quality of the proposed entry
  Volume Quality     10 pts — ExpertScores.volume
  Momentum Quality    10 pts — ExpertScores.momentum
  Pattern/Structure  15 pts — ExpertScores.liquidity as a structure/pattern proxy
  Trend Alignment    10 pts — macro score distance from neutral, signed with direction

Threshold:
  <75    SKIP
  75-84  GOOD
  85-100 HIGH_CONFIDENCE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .strategy_engine import StrategySignal
from .macro_trend_engine import MacroTrendResult
from .regime_classifier import RegimeResult


class ConfidenceLevel(str, Enum):
    SKIP            = "skip"
    GOOD            = "good"
    HIGH_CONFIDENCE = "high_confidence"


@dataclass
class ConfidenceResult:
    score: float
    level: ConfidenceLevel
    breakdown: dict = field(default_factory=dict)

    def passes(self) -> bool:
        return self.level != ConfidenceLevel.SKIP


class ConfidenceEngine:
    def __init__(self, skip_threshold: float = 75.0, high_threshold: float = 85.0):
        self.skip_threshold = skip_threshold
        self.high_threshold = high_threshold

    def score(
        self,
        signal: StrategySignal,
        macro: MacroTrendResult,
        regime: RegimeResult,
        experts,  # ExpertScores — trend/momentum/volatility/liquidity/volume/session/correlation
    ) -> ConfidenceResult:
        breakdown: dict = {}

        # Signal Quality (25 pts) — scaled from Layer 5's own 0-100 setup score
        signal_quality = signal.raw_score / 100.0 * 25.0
        breakdown["signal_quality"] = round(signal_quality, 1)

        # Market Alignment (15 pts) — macro direction agrees with trade direction
        macro_aligned = (
            (signal.direction == "long" and macro.score >= 50) or
            (signal.direction == "short" and macro.score < 50)
        )
        alignment_strength = abs(macro.score - 50) / 50.0  # 0-1
        market_alignment = (10.0 + alignment_strength * 5.0) if macro_aligned else (alignment_strength * 5.0)
        breakdown["market_alignment"] = round(market_alignment, 1)

        # Risk Alignment (15 pts) — reward the RR the strategy actually produced
        rr = max(0.0, signal.rr)
        risk_alignment = min(15.0, rr / 2.0 * 15.0)  # RR>=2 maxes this out
        breakdown["risk_alignment"] = round(risk_alignment, 1)

        # Volume Quality (10 pts)
        volume_quality = (experts.volume / 100.0) * 10.0
        breakdown["volume_quality"] = round(volume_quality, 1)

        # Momentum Quality (10 pts) — distance from neutral 50, signed with direction
        mom = experts.momentum
        mom_aligned = mom if signal.direction == "long" else (100 - mom)
        momentum_quality = (mom_aligned / 100.0) * 10.0
        breakdown["momentum_quality"] = round(momentum_quality, 1)

        # Pattern/Structure/Liquidity Quality (15 pts)
        structure_quality = (experts.liquidity / 100.0) * 15.0
        breakdown["structure_quality"] = round(structure_quality, 1)

        # Trend Alignment (10 pts) — regime classifier's own confidence in its call
        trend_alignment = (regime.confidence / 100.0) * 10.0
        breakdown["trend_alignment"] = round(trend_alignment, 1)

        total = sum(breakdown.values())
        total = round(min(100.0, max(0.0, total)), 1)

        level = self._classify(total)
        return ConfidenceResult(score=total, level=level, breakdown=breakdown)

    def _classify(self, score: float) -> ConfidenceLevel:
        if score >= self.high_threshold:
            return ConfidenceLevel.HIGH_CONFIDENCE
        if score >= self.skip_threshold:
            return ConfidenceLevel.GOOD
        return ConfidenceLevel.SKIP
