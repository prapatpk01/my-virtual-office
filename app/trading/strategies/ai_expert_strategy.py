"""
AI Expert Strategy — AI Decision Engine Architecture (Systematic Multi-Strategy)

Modular, event-driven, context-aware pipeline. Every market condition
routes through a different, purpose-built strategy — never one blended
indicator soup for every regime.

  Market Data
    v
  Layer 0  Market Quality Engine        — HARD GATE (untradeable market? stop.)
    v
  Layer 1  4H Macro Trend Engine        — DIRECTION GATE (never picks entries)
    v
  Layer 2  1H Context & Bias Engine     — score-based, informs Layer 4/6, no hard gate
    v
  Layer 3  Market Regime Classifier     — Primary Regime + Secondary Volatility State
    v
  Layer 4  Dynamic Strategy Selector    — REGIME GATE (exactly one strategy survives)
    v
  Layer 5  Strategy Engine              — score-based concrete entry/SL/TP logic
    v
  Layer 6  Confidence Engine            — TRADE QUALITY GATE (<75 skip)
    v
  Layer 7  Expectancy Engine            — TRADE QUALITY GATE (historical edge required)
    v
  Layer 8  Dynamic Risk Engine          — position size multiplier + TP1/TP2/trail plan
    v
  Signal -> Position Manager -> Exit Engine -> Learning Engine

Rule-based adaptive throughout — no machine learning. The Learning
Engine's journal feeds the Expectancy Engine (Layer 7) directly; there
is no separate weight-adaptation step left to hand-tune, the pipeline
self-selects strategies by regime and self-gates by real edge.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

from ..strategies.base import BaseStrategy, Signal, SignalType
from ..engines.expert_analysis import ExpertAnalysisEngine
from ..engines.exit_engine import ExitEngine
from ..engines.position_manager import PositionManager
from ..engines.adaptive_learning import AdaptiveLearningEngine
from ..engines.feature_store import FeatureStore
from ..engines.model_registry import ModelRegistry
from ..engines.drift_detector import DriftDetector, DriftAction
from ..engines.macro_trend_engine import MacroTrendEngine
from ..engines.context_bias_engine import ContextBiasEngine
from ..engines.market_quality_engine import MarketQualityEngine
from ..engines.regime_classifier import RegimeClassifier
from ..engines.regime_strategy_selector import RegimeStrategySelector, StrategyType
from ..engines.strategy_engine import (
    TrendContinuationStrategy, MeanReversionStrategy, BreakoutStrategy,
    SwingReversalStrategy, MomentumExpansionStrategy, StrategySignal,
)
from ..engines.confidence_engine import ConfidenceEngine
from ..engines.expectancy_engine import ExpectancyEngine
from ..engines.dynamic_risk_engine import DynamicRiskEngine
from ..engines import indicators as ind

logger = logging.getLogger("ai_expert_strategy")

_STRATEGY_ENGINES = {
    StrategyType.TREND_CONTINUATION: TrendContinuationStrategy(),
    StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
    StrategyType.BREAKOUT: BreakoutStrategy(),
    StrategyType.SWING_REVERSAL: SwingReversalStrategy(),
    StrategyType.MOMENTUM_EXPANSION: MomentumExpansionStrategy(),
}


class AIExpertStrategy(BaseStrategy):
    """AI Decision Engine — selects and runs exactly one strategy per bar,
    matched to the current market regime, gated by confidence and
    historical expectancy, sized by dynamic risk."""

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        min_confidence: float   = 70.0,   # kept for API compat; Layer 6 has its own threshold
        require_all_checks: bool = False,  # unused in the new pipeline; kept for API compat
        journal_path: str = "trade_journal.json",
        registry_path: str = "model_registry.json",
    ):
        super().__init__(symbol, params)
        self.name = f"AIExpert({symbol})"
        self.min_confidence = min_confidence

        # ── Layer engines ────────────────────────────────────────────────────
        self._quality_engine    = MarketQualityEngine()               # Layer 0
        self._macro_engine      = MacroTrendEngine()                  # Layer 1
        self._context_engine    = ContextBiasEngine()                 # Layer 2
        self._regime_classifier = RegimeClassifier()                  # Layer 3
        self._selector          = RegimeStrategySelector()            # Layer 4
        # Layer 5 strategy instances are shared module-level singletons (stateless)
        self._confidence_engine = ConfidenceEngine()                  # Layer 6
        self._expectancy_engine = ExpectancyEngine()                  # Layer 7
        self._risk_engine       = DynamicRiskEngine(                  # Layer 8
            base_tp1_rr=float(os.getenv("TP1_RR", "0.6")),
            base_tp2_rr=float(os.getenv("TP2_RR", "1.2")),
        )

        # ── Supporting engines (Confidence Engine inputs + post-entry mgmt) ───
        self._expert_engine    = ExpertAnalysisEngine()
        self._exit_engine      = ExitEngine(soft_threshold=70.0, hard_threshold=85.0)
        self._position_manager = PositionManager(
            partial_tp_1_rr=float(os.getenv("TP1_RR", "0.6")),
            partial_tp_2_rr=float(os.getenv("TP2_RR", "1.2")),
        )
        self._learning_engine  = AdaptiveLearningEngine(journal_path=journal_path)
        self._feature_store    = FeatureStore(ttl_seconds=60)
        self._model_registry   = ModelRegistry(registry_path=registry_path)
        self._drift_detector   = DriftDetector()

        # Internal state
        self._open_entry:     Optional[dict] = None
        self._signal_count    = 0
        self._latest_candles: list            = []

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        if len(candles) < 60:
            return self._hold(current_price, "Not enough candles")

        self._latest_candles = candles
        symbol = self.symbol
        mtf    = mtf_candles or {}

        # ── Layer 0: Market Quality — hard gate ───────────────────────────────
        quality = self._quality_engine.analyze(candles)
        if quality.blocks_trading():
            return self._hold(
                current_price,
                reason=f"Market Quality too low ({quality.score:.0f}/{quality.band.value}) — no analysis run",
                metadata={"market_quality": {"score": quality.score, "band": quality.band.value}},
            )

        # ── Layer 1: 4H Macro Trend — direction gate only, never picks entries ─
        macro = self._macro_engine.analyze(mtf.get("4h", []))

        # ── Layer 2: 1H Context & Bias — score-based, no hard gate ────────────
        context = self._context_engine.analyze(mtf.get("1h", []), macro.score)

        # ── Layer 3: Market Regime Classifier — the heart of the system ──────
        regime = self._regime_classifier.analyze(candles)

        # ── Layer 4: Dynamic Strategy Selector — exactly one strategy survives ─
        selection = self._selector.select(macro, context, regime)

        base_meta = self._base_metadata(quality, macro, context, regime, selection)

        # ── Exit check for an already-open position runs regardless of
        # whether Layer 4 allows new entries this bar ────────────────────────
        if self._open_entry:
            exit_sig = self._exit_engine.evaluate(
                candles,
                direction=self._open_entry["direction"],
                entry_price=self._open_entry["entry_price"],
                current_price=current_price,
                regime=None,
            )
            if exit_sig.forced_exit:
                return self._generate_exit_signal(
                    current_price,
                    reason=f"Exit AI forced close: {', '.join(exit_sig.reasons)}",
                    confidence=exit_sig.score / 100,
                )

        if not selection.is_tradeable():
            return self._hold(current_price, reason=f"Strategy selector: {selection.block_reason}", metadata=base_meta)

        # ── Layer 5: Strategy Engine — one strategy, its own entry/SL/TP logic ─
        engine = _STRATEGY_ENGINES.get(selection.selected)
        if engine is None:
            return self._hold(current_price, reason=f"No engine wired for {selection.selected.value}", metadata=base_meta)

        strat_signal: StrategySignal = engine.evaluate(candles, selection.direction_filter)
        base_meta["strategy_setup"] = {
            "valid": strat_signal.valid, "raw_score": strat_signal.raw_score,
            "reason": strat_signal.reason, "detail": strat_signal.detail,
        }
        if not strat_signal.valid:
            return self._hold(current_price, reason=strat_signal.reason, metadata=base_meta)

        # ── Supporting expert scores (feed Layer 6's quality sub-scores) ─────
        experts = self._expert_engine.analyze(candles, symbol=symbol, mtf_candles=mtf)

        candle_ts = int(candles[-1].timestamp) if hasattr(candles[-1], "timestamp") else int(time.time())
        self._feature_store.store(
            symbol=symbol, timeframe="15m", candle_ts=candle_ts,
            features={"strategy": selection.selected.value, "price": current_price},
            regime=regime.primary.value,
        )

        # ── Layer 6: Confidence Engine — trade quality gate ───────────────────
        confidence = self._confidence_engine.score(strat_signal, macro, regime, experts)
        base_meta["confidence"] = {
            "score": confidence.score, "level": confidence.level.value,
            "breakdown": confidence.breakdown,
        }
        if not confidence.passes():
            return self._hold(
                current_price,
                reason=f"Confidence {confidence.score:.0f} below threshold (skip)",
                metadata=base_meta,
            )

        # ── Layer 7: Expectancy Engine — trade quality gate ───────────────────
        regime_key = f"{regime.primary.value}:{selection.selected.value}"
        expectancy = self._expectancy_engine.evaluate(self._learning_engine._journal, regime_key)
        base_meta["expectancy"] = {
            "tradeable": expectancy.tradeable, "expectancy_r": expectancy.expectancy_r,
            "win_rate": expectancy.win_rate, "profit_factor": expectancy.profit_factor,
            "kelly_fraction": expectancy.kelly_fraction, "sample_size": expectancy.sample_size,
        }
        if not expectancy.tradeable:
            return self._hold(current_price, reason=f"Expectancy gate: {expectancy.reason}", metadata=base_meta)

        # ── Drift check (non-blocking informational veto) ────────────────────
        if self._drift_detector.highest_severity_action() == DriftAction.PAUSE:
            return self._hold(current_price, reason="Drift detector: PAUSE — model performance degraded", metadata=base_meta)

        # ── Layer 8: Dynamic Risk Engine — sizing + position-management plan ──
        risk_decision = self._risk_engine.compute(quality, regime, confidence, expectancy, experts.correlation)
        base_meta["dynamic_risk"] = {
            "risk_multiplier": risk_decision.risk_multiplier,
            "tp1_rr": risk_decision.tp1_rr, "tp2_rr": risk_decision.tp2_rr,
            "partial_close_pct": risk_decision.partial_close_pct,
            "trail_after_tp1": risk_decision.trail_after_tp1,
        }

        # ── Build Signal ───────────────────────────────────────────────────
        signal_type = SignalType.BUY if strat_signal.direction == "long" else SignalType.SELL
        base_meta.update({
            "stop_loss": strat_signal.stop_loss,
            "take_profit": strat_signal.take_profit,
            "rr_ratio": strat_signal.rr,
        })

        self._open_entry = {
            "direction": strat_signal.direction,
            "entry_price": current_price,
            "stop_loss": strat_signal.stop_loss,
            "take_profit": strat_signal.take_profit,
            "regime": regime_key,
            "session": experts.detail.get("session", {}).get("session", "unknown"),
            "expert_scores": {
                "trend": experts.trend, "momentum": experts.momentum,
                "volatility": experts.volatility, "liquidity": experts.liquidity,
                "volume": experts.volume, "session": experts.session,
                "correlation": experts.correlation,
            },
            "decision_score": confidence.score,
            "confidence_level": confidence.level.value,
            "strategy_type": selection.selected.value,
            "macro_score": macro.score,
            "risk_multiplier": risk_decision.risk_multiplier,
            "tp1_rr": risk_decision.tp1_rr,
            "tp2_rr": risk_decision.tp2_rr,
            "opened_at": time.time(),
        }

        self._signal_count += 1
        logger.info(
            "[%s] %s SIGNAL | Regime=%s/%s | Strategy=%s | Conf=%.0f(%s) | Expectancy=%+.3fR | "
            "RiskMult=%.2fx | R:R=%.2f | SL=%.4f TP=%.4f",
            symbol, signal_type.value.upper(), regime.primary.value, regime.secondary.value,
            selection.selected.value, confidence.score, confidence.level.value,
            expectancy.expectancy_r, risk_decision.risk_multiplier,
            strat_signal.rr, strat_signal.stop_loss, strat_signal.take_profit,
        )

        return Signal(
            type=signal_type, symbol=symbol, price=current_price, amount=0.0,
            reason=strat_signal.reason, confidence=min(1.0, confidence.score / 100.0),
            metadata=base_meta,
        )

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Feed the learning engine (and therefore Layer 7's Expectancy Engine) after a close."""
        entry = self._open_entry
        if not entry:
            return
        self._learning_engine.record_trade(
            symbol=self.symbol, direction=entry["direction"],
            regime=entry["regime"],   # composite "primary_regime:strategy" key
            session=entry["session"],
            entry_price=entry["entry_price"], exit_price=exit_price,
            stop_loss=entry["stop_loss"], take_profit=entry["take_profit"],
            expert_scores=entry["expert_scores"],
            decision_score=entry["decision_score"], confidence_level=entry["confidence_level"],
            exit_reason=exit_reason, duration_min=duration_min,
        )
        self._open_entry = None

        n = len(self._learning_engine._journal)
        if n > 0 and n % 10 == 0:
            self._run_adaptation()

    def tick_open_position(
        self,
        current_price: float,
        position_key: Optional[str] = None,
    ) -> Optional["PositionUpdate"]:
        """Layer 8's plan (tp1_rr/tp2_rr) drives PositionManager per-position."""
        if not self._open_entry or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate

        entry   = self._open_entry
        candles = self._latest_candles
        pos_id  = position_key or self.symbol

        closes = np.array([float(c.close) for c in candles], dtype=float)
        highs  = np.array([float(c.high) for c in candles], dtype=float)
        lows   = np.array([float(c.low) for c in candles], dtype=float)
        atr_arr = ind.atr(closes, highs, lows, 14)
        valid = atr_arr[~np.isnan(atr_arr)]
        atr = float(valid[-1]) if len(valid) > 0 else 0.0
        if atr <= 0:
            return PositionUpdate(action="hold", reason="ATR unavailable")

        exit_sig = self._exit_engine.evaluate(
            candles, direction=entry["direction"], entry_price=entry["entry_price"],
            current_price=current_price, regime=None,
        )

        if pos_id not in self._position_manager._positions:
            self._position_manager.register_position(
                position_id=pos_id, direction=entry["direction"],
                entry_price=entry["entry_price"], stop_loss=entry["stop_loss"],
                take_profit=entry["take_profit"], atr=atr,
                tp1_rr=entry.get("tp1_rr"), tp2_rr=entry.get("tp2_rr"),
            )

        return self._position_manager.update(
            position_id=pos_id, current_price=current_price,
            current_atr=atr, exit_score=exit_sig.score,
        )

    def get_analysis_state(self) -> dict:
        return {
            "symbol": self.symbol,
            "signals_fired": self._signal_count,
            "journal_entries": len(self._learning_engine._journal),
            "drift_action": self._drift_detector.highest_severity_action().value,
            "performance": self._learning_engine.performance_summary(),
            "feature_cache": self._feature_store.stats(),
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _hold(self, price: float, reason: str = "", metadata: dict = None) -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata=metadata or {},
        )

    def _generate_exit_signal(self, price: float, reason: str, confidence: float) -> Signal:
        self._open_entry = None
        return Signal(
            type=SignalType.SELL, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=confidence, metadata={"exit_ai": True},
        )

    @staticmethod
    def _base_metadata(quality, macro, context, regime, selection) -> dict:
        return {
            "market_quality": {"score": quality.score, "band": quality.band.value},
            "macro_trend": {
                "score": macro.score, "bias": macro.bias.value,
                "structure": macro.structure, "allowed_direction": macro.allowed_direction(),
            },
            "context_1h": {
                "type": context.context.value,
                "bull_score": context.bull_score, "bear_score": context.bear_score,
                "dominant_bias": context.dominant_bias,
            },
            "regime": regime.primary.value,
            "regime_secondary": regime.secondary.value,
            "regime_confidence": regime.confidence,
            "regime_scores": regime.scores,
            "selected_strategy": selection.selected.value,
            "strategy_confidence": regime.confidence,
            "direction_filter": selection.direction_filter,
        }

    def _run_adaptation(self) -> None:
        """Periodic drift check + model registry snapshot. Strategy selection
        and weighting are rule-based on the live regime/expectancy each bar —
        there's no separate weight table left to hand-adapt here."""
        try:
            journal = self._learning_engine._journal
            perf = self._learning_engine.performance_summary()

            self._drift_detector.evaluate(
                recent_trades=journal,
                current_regime=self._open_entry["regime"] if self._open_entry else "",
            )

            win_rate = perf.get("win_rate_pct", 0)
            if win_rate > 0:
                self._model_registry.save_model(
                    name=f"AIExpert_{self.symbol}",
                    weights={},
                    performance={"win_rate": win_rate, "expectancy_r": perf.get("expectancy_r", 0)},
                    description=f"Auto-saved after {len(journal)} trades",
                )

            logger.info(
                "[%s] Adaptation run: WR=%.1f%% Exp=%.2fR Drift=%s",
                self.symbol, win_rate, perf.get("expectancy_r", 0),
                self._drift_detector.highest_severity_action().value,
            )
        except Exception as e:
            logger.warning("Adaptation run failed: %s", e)
