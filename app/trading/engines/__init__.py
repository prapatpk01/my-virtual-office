"""AI Decision Engine — Layer 0-8 systematic multi-strategy pipeline.

Legacy engines (MarketIntelligenceEngine, DecisionEngine, EntryTimingEngine)
are kept for backward compatibility / standalone reuse but are no longer
part of the active AIExpertStrategy pipeline — see ai_expert_strategy.py
for the current Layer 0-8 flow (MarketQualityEngine -> MacroTrendEngine ->
ContextBiasEngine -> RegimeClassifier -> RegimeStrategySelector ->
strategy_engine -> ConfidenceEngine -> ExpectancyEngine -> DynamicRiskEngine).
"""
from .market_intelligence import MarketIntelligenceEngine, RegimeResult, MarketRegime
from .expert_analysis import ExpertAnalysisEngine, ExpertScores
from .decision_engine import DecisionEngine, DecisionResult, ConfidenceLevel
from .entry_timing import EntryTimingEngine, EntrySignal
from .position_manager import PositionManager, PositionUpdate
from .exit_engine import ExitEngine, ExitSignal
from .adaptive_learning import AdaptiveLearningEngine
from .feature_store import FeatureStore
from .model_registry import ModelRegistry
from .portfolio_engine import PortfolioEngine
from .drift_detector import DriftDetector

from .market_quality_engine import MarketQualityEngine, MarketQualityResult, QualityBand
from .macro_trend_engine import MacroTrendEngine, MacroTrendResult, TrendBias
from .context_bias_engine import ContextBiasEngine, ContextBiasResult, ContextType
from .regime_classifier import RegimeClassifier, RegimeResult as PrimaryRegimeResult, PrimaryRegime, SecondaryState
from .regime_strategy_selector import RegimeStrategySelector, StrategySelectionResult, StrategyType
from .strategy_engine import (
    TrendContinuationStrategy, MeanReversionStrategy, BreakoutStrategy,
    SwingReversalStrategy, MomentumExpansionStrategy, StrategySignal,
)
from .confidence_engine import ConfidenceEngine, ConfidenceResult, ConfidenceLevel as TradeConfidenceLevel
from .expectancy_engine import ExpectancyEngine, ExpectancyResult
from .dynamic_risk_engine import DynamicRiskEngine, RiskDecision

__all__ = [
    "MarketIntelligenceEngine", "RegimeResult", "MarketRegime",
    "ExpertAnalysisEngine", "ExpertScores",
    "DecisionEngine", "DecisionResult", "ConfidenceLevel",
    "EntryTimingEngine", "EntrySignal",
    "PositionManager", "PositionUpdate",
    "ExitEngine", "ExitSignal",
    "AdaptiveLearningEngine",
    "FeatureStore",
    "ModelRegistry",
    "PortfolioEngine",
    "DriftDetector",
    "MarketQualityEngine", "MarketQualityResult", "QualityBand",
    "MacroTrendEngine", "MacroTrendResult", "TrendBias",
    "ContextBiasEngine", "ContextBiasResult", "ContextType",
    "RegimeClassifier", "PrimaryRegimeResult", "PrimaryRegime", "SecondaryState",
    "RegimeStrategySelector", "StrategySelectionResult", "StrategyType",
    "TrendContinuationStrategy", "MeanReversionStrategy", "BreakoutStrategy",
    "SwingReversalStrategy", "MomentumExpansionStrategy", "StrategySignal",
    "ConfidenceEngine", "ConfidenceResult", "TradeConfidenceLevel",
    "ExpectancyEngine", "ExpectancyResult",
    "DynamicRiskEngine", "RiskDecision",
]
