"""
AI Expert Strategy — Institutional-Grade 9-Layer Analysis Pipeline

Pipeline flow:
  Candle Data
    ↓
  Feature Store (cache features)
    ↓
  Layer 1: Market Intelligence (Regime Classification)
    ↓
  Layer 2: Expert Analysis (7 expert modules → scores 0-100)
    ↓
  Layer 3+4: Decision Engine (weighted scoring + confidence filter)
    ↓
  Layer 5+6: Entry Timing (MTF cascade + trigger checklist)
    ↓
  Signal (with SL/TP calculated from ATR + structure)
    ↓
  Layer 7: Position Manager (BE, trail, partial TP) — post-entry
    ↓
  Layer 8: Exit AI (dynamic exit scoring) — tick by tick
    ↓
  Layer 9: Learning Engine (record trade, adapt weights)

Indicators are used only as SUPPORTING features — not primary decision makers.
Primary signals come from: Price Action, Market Structure, Liquidity,
Volume Profile, Session, Volatility Regime, Statistical Edge.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

from ..strategies.base import BaseStrategy, Signal, SignalType
from ..engines.market_intelligence import MarketIntelligenceEngine, MarketRegime
from ..engines.expert_analysis import ExpertAnalysisEngine
from ..engines.decision_engine import DecisionEngine, ConfidenceLevel
from ..engines.entry_timing import EntryTimingEngine
from ..engines.exit_engine import ExitEngine
from ..engines.position_manager import PositionManager
from ..engines.adaptive_learning import AdaptiveLearningEngine
from ..engines.feature_store import FeatureStore
from ..engines.model_registry import ModelRegistry
from ..engines.portfolio_engine import PortfolioEngine
from ..engines.drift_detector import DriftDetector, DriftAction

logger = logging.getLogger("ai_expert_strategy")


class AIExpertStrategy(BaseStrategy):
    """
    Adaptive AI Expert Strategy with full 9-layer institutional analysis.
    Replaces simple indicator-based strategies with ensemble expert scoring.
    """

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        min_confidence: float   = 70.0,   # Ignore signals below this
        require_all_checks: bool = False,  # True = strictest mode
        journal_path: str = "trade_journal.json",
        registry_path: str = "model_registry.json",
    ):
        super().__init__(symbol, params)
        self.name = f"AIExpert({symbol})"
        self.min_confidence = min_confidence

        # ── Engine instantiation ───────────────────────────────────────────────
        self._regime_engine    = MarketIntelligenceEngine()
        self._expert_engine    = ExpertAnalysisEngine()
        self._decision_engine  = DecisionEngine(min_score_threshold=min_confidence)
        self._timing_engine    = EntryTimingEngine(
            min_rr=float(os.getenv("MIN_RR", "2.0")),
            require_all_checks=require_all_checks,
        )
        self._exit_engine      = ExitEngine(soft_threshold=70.0, hard_threshold=85.0)
        self._position_manager = PositionManager(
            be_atr_mult=1.0, trail_atr_mult=2.0,
            partial_tp_1_rr=1.0, partial_tp_2_rr=2.0,
        )
        self._learning_engine  = AdaptiveLearningEngine(journal_path=journal_path)
        self._feature_store    = FeatureStore(ttl_seconds=60)
        self._model_registry   = ModelRegistry(registry_path=registry_path)
        self._drift_detector   = DriftDetector()

        # Internal state per signal
        self._open_entry:    Optional[dict]  = None
        self._open_position: Optional[str]   = None
        self._signal_count   = 0
        self._last_adapt_at  = 0
        self._latest_candles: list           = []   # updated each analyze(); used by tick_open_position

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        """Run full 9-layer pipeline and return a Signal."""
        if len(candles) < 60:
            return self._hold(current_price, "Not enough candles")

        self._latest_candles = candles  # cache for tick_open_position()
        symbol = self.symbol
        mtf    = mtf_candles or {}

        # ── Layer 1: Market Intelligence (Regime) ─────────────────────────────
        regime = self._regime_engine.analyze(candles, mtf_candles=mtf)

        # ── Layer 2: Expert Analysis ─────────────────────────────────────────
        experts = self._expert_engine.analyze(
            candles, symbol=symbol, mtf_candles=mtf
        )

        # ── Feature Store: cache current state ─────────────────────────────────
        candle_ts = int(candles[-1].timestamp) if hasattr(candles[-1], "timestamp") else int(time.time())
        self._feature_store.store(
            symbol=symbol, timeframe="15m", candle_ts=candle_ts,
            features={
                "regime_scores": regime.scores(),
                "expert_scores": {
                    "trend":       experts.trend,
                    "momentum":    experts.momentum,
                    "volatility":  experts.volatility,
                    "liquidity":   experts.liquidity,
                    "volume":      experts.volume,
                    "session":     experts.session,
                    "correlation": experts.correlation,
                },
                "price": current_price,
            },
            regime=regime.regime.value,
        )

        # ── Adaptive weights from learning engine ──────────────────────────────
        adapt_weights = None
        if len(self._learning_engine._journal) >= self._learning_engine.MIN_TRADES_TO_ADAPT:
            recs = self._learning_engine.expert_weight_recommendations()
            if regime.regime.value in recs:
                adapt_weights = recs[regime.regime.value].get("weights")

        # ── Layer 3+4: Decision Engine ─────────────────────────────────────────
        decision = self._decision_engine.decide(
            experts, regime, adaptive_weights=adapt_weights
        )

        # ── Layer 8: Exit AI (check open position first) ──────────────────────
        if self._open_entry:
            exit_sig = self._exit_engine.evaluate(
                candles,
                direction=self._open_entry["direction"],
                entry_price=self._open_entry["entry_price"],
                current_price=current_price,
                regime=regime.regime,
            )
            if exit_sig.forced_exit:
                return self._generate_exit_signal(
                    current_price,
                    reason=f"Exit AI forced close: {', '.join(exit_sig.reasons)}",
                    confidence=exit_sig.score / 100,
                )

        # ── Layer 5+6: Entry Timing ────────────────────────────────────────────
        entry_signal = self._timing_engine.evaluate(
            candles_15m=candles,
            decision=decision,
            regime=regime,
            experts=experts,
            mtf_candles=mtf,
            current_price=current_price,
        )

        if not entry_signal.valid:
            logger.debug(
                "[%s] No entry: %s | Regime=%s | Long=%.0f Short=%.0f",
                symbol, entry_signal.reason,
                regime.regime.value, decision.long_score, decision.short_score,
            )
            return self._hold(
                current_price,
                reason=entry_signal.reason,
                metadata=self._build_metadata(regime, experts, decision, entry_signal),
            )

        # ── Drift check (non-blocking, informational) ───────────────────────────
        drift_action = self._drift_detector.highest_severity_action()
        if drift_action == DriftAction.PAUSE:
            return self._hold(
                current_price,
                reason="Drift detector: PAUSE — model performance degraded",
                metadata=self._build_metadata(regime, experts, decision, entry_signal),
            )

        # ── Build Signal ──────────────────────────────────────────────────
        signal_type = SignalType.BUY if entry_signal.direction == "long" else SignalType.SELL
        confidence  = min(1.0, entry_signal.confidence / 100.0)

        metadata = self._build_metadata(regime, experts, decision, entry_signal)
        metadata.update({
            "stop_loss":   entry_signal.stop_loss,
            "take_profit": entry_signal.take_profit,
            "rr_ratio":    entry_signal.rr_ratio,
        })

        # Save entry context for position management
        self._open_entry = {
            "direction":   entry_signal.direction,
            "entry_price": current_price,
            "stop_loss":   entry_signal.stop_loss,
            "take_profit": entry_signal.take_profit,
            "regime":      regime.regime.value,
            "session":     experts.detail.get("session", {}).get("session", "unknown"),
            "expert_scores": {
                "trend":       experts.trend,
                "momentum":    experts.momentum,
                "volatility":  experts.volatility,
                "liquidity":   experts.liquidity,
                "volume":      experts.volume,
                "session":     experts.session,
                "correlation": experts.correlation,
            },
            "decision_score":   decision.confidence,
            "confidence_level": decision.confidence_level.value,
            "opened_at":        time.time(),
        }

        self._signal_count += 1
        logger.info(
            "[%s] %s SIGNAL | Regime=%s | Conf=%s %.0f | R:R=%.1f | SL=%.4f TP=%.4f",
            symbol, signal_type.value.upper(),
            regime.regime.value, decision.confidence_level.value,
            decision.confidence, entry_signal.rr_ratio,
            entry_signal.stop_loss, entry_signal.take_profit,
        )

        return Signal(
            type=signal_type,
            symbol=symbol,
            price=current_price,
            amount=0.0,  # sized by risk_manager
            reason=entry_signal.reason,
            confidence=confidence,
            metadata=metadata,
        )

    def record_closed_trade(
        self,
        exit_price:   float,
        exit_reason:  str,
        duration_min: float = 0.0,
    ) -> None:
        """Call after a position is closed to feed the learning engine."""
        entry = self._open_entry
        if not entry:
            return
        self._learning_engine.record_trade(
            symbol=self.symbol,
            direction=entry["direction"],
            regime=entry["regime"],
            session=entry["session"],
            entry_price=entry["entry_price"],
            exit_price=exit_price,
            stop_loss=entry["stop_loss"],
            take_profit=entry["take_profit"],
            expert_scores=entry["expert_scores"],
            decision_score=entry["decision_score"],
            confidence_level=entry["confidence_level"],
            exit_reason=exit_reason,
            duration_min=duration_min,
        )
        self._open_entry = None

        # Periodically run drift detection and weight adaptation
        n = len(self._learning_engine._journal)
        if n > 0 and n % 10 == 0:
            self._run_adaptation()

    def tick_open_position(
        self,
        current_price: float,
        position_key: Optional[str] = None,
    ) -> Optional["PositionUpdate"]:
        """
        Called each bot tick when this strategy has an open position.
        Runs Exit AI + Position Manager and returns a recommended action.
        Returns None if no open entry is tracked or candles are unavailable.
        """
        if not self._open_entry or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate
        from ..engines.market_intelligence import MarketRegime

        entry   = self._open_entry
        candles = self._latest_candles
        pos_id  = position_key or self.symbol

        atr_arr = self._regime_engine._atr(candles)
        valid = atr_arr[~np.isnan(atr_arr)]
        atr = float(valid[-1]) if len(valid) > 0 else 0.0
        if atr <= 0:
            return PositionUpdate(action="hold", reason="ATR unavailable")

        try:
            regime_val = MarketRegime(entry["regime"])
        except ValueError:
            regime_val = MarketRegime.TREND

        exit_sig = self._exit_engine.evaluate(
            candles,
            direction=entry["direction"],
            entry_price=entry["entry_price"],
            current_price=current_price,
            regime=regime_val,
        )

        if pos_id not in self._position_manager._positions:
            self._position_manager.register_position(
                position_id=pos_id,
                direction=entry["direction"],
                entry_price=entry["entry_price"],
                stop_loss=entry["stop_loss"],
                take_profit=entry["take_profit"],
                atr=atr,
            )

        return self._position_manager.update(
            position_id=pos_id,
            current_price=current_price,
            current_atr=atr,
            exit_score=exit_sig.score,
        )

    def get_analysis_state(self) -> dict:
        """Return current pipeline state for dashboard display."""
        return {
            "symbol":         self.symbol,
            "signals_fired":  self._signal_count,
            "journal_entries":len(self._learning_engine._journal),
            "drift_action":   self._drift_detector.highest_severity_action().value,
            "performance":    self._learning_engine.performance_summary(),
            "feature_cache":  self._feature_store.stats(),
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _hold(self, price: float, reason: str = "", metadata: dict = None) -> Signal:
        return Signal(
            type=SignalType.HOLD,
            symbol=self.symbol,
            price=price,
            amount=0.0,
            reason=reason,
            confidence=0.0,
            metadata=metadata or {},
        )

    def _generate_exit_signal(self, price: float, reason: str, confidence: float) -> Signal:
        self._open_entry = None
        return Signal(
            type=SignalType.SELL,
            symbol=self.symbol,
            price=price,
            amount=0.0,
            reason=reason,
            confidence=confidence,
            metadata={"exit_ai": True},
        )

    def _build_metadata(
        self, regime, experts, decision, entry_signal
    ) -> dict:
        return {
            "regime":         regime.regime.value,
            "regime_scores":  regime.scores(),
            "expert_scores": {
                "trend":      experts.trend,
                "momentum":   experts.momentum,
                "volatility": experts.volatility,
                "liquidity":  experts.liquidity,
                "volume":     experts.volume,
                "session":    experts.session,
            },
            "long_score":     decision.long_score,
            "short_score":    decision.short_score,
            "confidence":     decision.confidence,
            "confidence_level": decision.confidence_level.value,
            "mtf_bias":       getattr(entry_signal, "mtf_bias", {}),
            "checklist":      getattr(entry_signal, "checklist", {}),
            "regime_detail":  regime.detail,
            "direction_bias": experts.direction_bias,
        }

    def _run_adaptation(self) -> None:
        """Periodic learning: update decision engine weights + check drift."""
        try:
            journal = self._learning_engine._journal
            perf    = self._learning_engine.performance_summary()

            # Drift check
            self._drift_detector.evaluate(
                recent_trades=journal,
                current_regime=self._open_entry["regime"] if self._open_entry else "",
            )

            # Update weights per regime from recommendations
            recs = self._learning_engine.expert_weight_recommendations()
            for regime_str, rec in recs.items():
                try:
                    regime_enum = MarketRegime(regime_str)
                    self._decision_engine.update_adaptive_weights(
                        regime_enum, rec["weights"]
                    )
                except ValueError:
                    pass

            # Save to model registry if performance improved
            win_rate = perf.get("win_rate_pct", 0)
            if win_rate > 0:
                self._model_registry.save_model(
                    name=f"AIExpert_{self.symbol}",
                    weights=recs,
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
