"""
Dynamic Strategy Selection + Confidence Engine

Layer 3 aggregator: takes MacroTrend + ContextBias + MarketRegime and:
  1. Scores each of 4 strategy types (0-100)
  2. Selects the highest-confidence strategy type (winner-takes-all)
  3. Returns regime-specific indicator weights for the DecisionEngine

Strategy types:
  TREND_CONTINUATION  — ride the trend via EMA pullbacks and momentum
  MEAN_REVERSION      — fade extremes in range / compression markets
  BREAKOUT            — enter on compression breakout with volume confirmation
  SWING_REVERSAL      — catch reversals via divergence, QM, CHOCH patterns

Dynamic weights per strategy type:
  TREND_CONTINUATION:
    EMA 25%, ADX 20%, Momentum 20%, Volume 15%, Liquidity 10%, Pattern 10%
  MEAN_REVERSION:
    EMA 5%, RSI/Momentum 20%, VWAP/Liquidity 20%, Structure/BB 20%, Volume 15%, Session 20%
  BREAKOUT:
    Volume 25%, Momentum 20%, Volatility 20%, Liquidity 15%, Pattern 10%, Session 10%
  SWING_REVERSAL:
    Pattern 25%, Liquidity 25%, Momentum 15%, Volume 15%, Structure 10%, Session 10%
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .macro_trend_engine import MacroTrendResult, TrendBias
from .context_bias_engine import ContextBiasResult, ContextType
from .market_intelligence import MarketRegime, RegimeResult


class StrategyType(str, Enum):
    TREND_CONTINUATION = "trend_continuation"
    MEAN_REVERSION     = "mean_reversion"
    BREAKOUT           = "breakout"
    SWING_REVERSAL     = "swing_reversal"
    NO_TRADE           = "no_trade"


# Indicator weights per strategy type (maps to DecisionEngine expert names)
_STRATEGY_WEIGHTS: dict[StrategyType, dict[str, float]] = {
    StrategyType.TREND_CONTINUATION: {
        "trend":      0.25,
        "momentum":   0.20,
        "liquidity":  0.10,
        "volume":     0.15,
        "volatility": 0.10,
        "session":    0.10,
        "correlation": 0.10,
    },
    StrategyType.MEAN_REVERSION: {
        "trend":      0.05,
        "momentum":   0.20,
        "liquidity":  0.20,
        "volume":     0.15,
        "volatility": 0.05,
        "session":    0.15,
        "correlation": 0.20,
    },
    StrategyType.BREAKOUT: {
        "trend":      0.10,
        "momentum":   0.20,
        "liquidity":  0.15,
        "volume":     0.25,
        "volatility": 0.20,
        "session":    0.10,
        "correlation": 0.00,
    },
    StrategyType.SWING_REVERSAL: {
        "trend":      0.05,
        "momentum":   0.15,
        "liquidity":  0.25,
        "volume":     0.15,
        "volatility": 0.10,
        "session":    0.10,
        "correlation": 0.20,
    },
}

# Ensure all weight dicts sum to 1.0
for _st, _w in _STRATEGY_WEIGHTS.items():
    _total = sum(_w.values())
    if abs(_total - 1.0) > 0.001:
        _STRATEGY_WEIGHTS[_st] = {k: v / _total for k, v in _w.items()}


@dataclass
class StrategySelectionResult:
    selected: StrategyType
    scores: dict[str, float]            # {strategy_type_value: score 0-100}
    weights: dict[str, float]           # indicator weights for DecisionEngine
    confidence: float                   # score of the winner (0-100)
    direction_filter: str               # "long_only" | "short_only" | "both" | "none"
    block_reason: str = ""
    detail: dict = field(default_factory=dict)

    def allows_long(self) -> bool:
        return self.direction_filter in ("long_only", "both")

    def allows_short(self) -> bool:
        return self.direction_filter in ("short_only", "both")

    def is_tradeable(self) -> bool:
        return self.selected != StrategyType.NO_TRADE and self.confidence >= 50


class RegimeStrategySelector:
    """
    Scores all 4 strategy types and selects the highest-confidence one.
    Implements the Confidence Engine — only the winner is eligible to trade.
    """

    def __init__(self, min_strategy_score: float = 55.0):
        self.min_strategy_score = min_strategy_score

    def select(
        self,
        macro: MacroTrendResult,
        context: ContextBiasResult,
        regime: RegimeResult,
    ) -> StrategySelectionResult:
        """Score all strategy types and return the winner."""

        scores: dict[str, float] = {}
        detail: dict = {}

        # ── Score each strategy type ─────────────────────────────────────────
        s, d = self._score_trend_continuation(macro, context, regime)
        scores[StrategyType.TREND_CONTINUATION.value] = s
        detail[StrategyType.TREND_CONTINUATION.value] = d

        s, d = self._score_mean_reversion(macro, context, regime)
        scores[StrategyType.MEAN_REVERSION.value] = s
        detail[StrategyType.MEAN_REVERSION.value] = d

        s, d = self._score_breakout(macro, context, regime)
        scores[StrategyType.BREAKOUT.value] = s
        detail[StrategyType.BREAKOUT.value] = d

        s, d = self._score_swing_reversal(macro, context, regime)
        scores[StrategyType.SWING_REVERSAL.value] = s
        detail[StrategyType.SWING_REVERSAL.value] = d

        # ── Winner-takes-all selection ────────────────────────────────────────
        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]
        selected   = StrategyType(best_type)

        if best_score < self.min_strategy_score:
            return StrategySelectionResult(
                selected=StrategyType.NO_TRADE,
                scores=scores,
                weights=_STRATEGY_WEIGHTS[StrategyType.TREND_CONTINUATION],
                confidence=best_score,
                direction_filter="none",
                block_reason=f"No strategy cleared min score {self.min_strategy_score:.0f}",
                detail=detail,
            )

        # ── Direction filter from macro + counter-trend block ─────────────────
        direction_filter = self._direction_filter(macro, context, selected)

        if direction_filter == "none":
            return StrategySelectionResult(
                selected=StrategyType.NO_TRADE,
                scores=scores,
                weights=_STRATEGY_WEIGHTS[selected],
                confidence=best_score,
                direction_filter="none",
                block_reason="Counter-trend blocked by macro trend",
                detail=detail,
            )

        weights = _STRATEGY_WEIGHTS[selected]

        return StrategySelectionResult(
            selected=selected,
            scores=scores,
            weights=weights,
            confidence=round(best_score, 1),
            direction_filter=direction_filter,
            detail=detail,
        )

    # ── Strategy scorers ─────────────────────────────────────────────────────

    def _score_trend_continuation(
        self,
        macro: MacroTrendResult,
        ctx: ContextBiasResult,
        regime: RegimeResult,
    ) -> tuple[float, str]:
        score = 0.0
        notes = []

        # Needs: trending macro (not neutral/range), pullback context, momentum
        trend_factor = (macro.score - 50.0) / 50.0  # -1 to +1

        # Macro trend alignment (40 pts)
        if macro.bias in (TrendBias.STRONG_BULL, TrendBias.STRONG_BEAR):
            score += 40
            notes.append(f"strong_macro {macro.bias.value}")
        elif macro.bias in (TrendBias.BULL, TrendBias.BEAR):
            score += 28
            notes.append(f"macro {macro.bias.value}")
        else:
            score += 5
            notes.append("macro_neutral (weak)")

        # Regime confirms trend (20 pts)
        if regime.regime == MarketRegime.TREND:
            score += 20
            notes.append("regime=TREND")
        elif regime.regime in (MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY):
            score += 8
        else:
            score += 2

        # Context supports continuation (25 pts)
        if ctx.context in (ContextType.PULLBACK, ContextType.CONTINUATION):
            score += 25
            notes.append(f"ctx={ctx.context.value}")
        elif ctx.context == ContextType.ACCUMULATION:
            score += 15
        else:
            score += 5

        # 1H bias alignment (15 pts)
        if (macro.score >= 50 and ctx.dominant_bias == "bull"):
            score += 15
            notes.append("1h_bull_aligned")
        elif (macro.score < 50 and ctx.dominant_bias == "bear"):
            score += 15
            notes.append("1h_bear_aligned")
        else:
            score += 3

        return round(min(100.0, score), 1), ", ".join(notes)

    def _score_mean_reversion(
        self,
        macro: MacroTrendResult,
        ctx: ContextBiasResult,
        regime: RegimeResult,
    ) -> tuple[float, str]:
        score = 0.0
        notes = []

        # Needs: ranging regime, near-neutral macro, range context
        if regime.regime == MarketRegime.RANGE:
            score += 35
            notes.append("regime=RANGE")
        elif regime.regime == MarketRegime.LOW_VOLATILITY:
            score += 20
            notes.append("regime=LOW_VOL")
        else:
            score += 5

        # Macro near neutral (30 pts — mean reversion works best in choppy macro)
        neutrality = 100 - abs(macro.score - 50) * 2  # 100 at score=50, 0 at extremes
        mean_rev_macro = neutrality * 0.3
        score += mean_rev_macro
        notes.append(f"macro_neutrality={neutrality:.0f}")

        # Range context (25 pts)
        if ctx.context == ContextType.RANGE:
            score += 25
            notes.append("ctx=RANGE")
        elif ctx.context in (ContextType.DISTRIBUTION, ContextType.ACCUMULATION):
            score += 15
        else:
            score += 3

        # Counter-trend RSI extreme (10 pts bonus)
        rsi_extreme = (
            ctx.score_breakdown.get("rsi", {}).get("bull", 0) >= 9 or
            ctx.score_breakdown.get("rsi", {}).get("bear", 0) >= 9
        )
        if rsi_extreme:
            score += 10
            notes.append("rsi_extreme")

        return round(min(100.0, score), 1), ", ".join(notes)

    def _score_breakout(
        self,
        macro: MacroTrendResult,
        ctx: ContextBiasResult,
        regime: RegimeResult,
    ) -> tuple[float, str]:
        score = 0.0
        notes = []

        # Needs: breakout regime, volume spike, compression beforehand
        if regime.regime == MarketRegime.BREAKOUT:
            score += 40
            notes.append("regime=BREAKOUT")
        elif regime.regime == MarketRegime.HIGH_VOLATILITY:
            score += 20
            notes.append("regime=HIGH_VOL")
        else:
            score += 3

        # Context breakout (30 pts)
        if ctx.context == ContextType.BREAKOUT:
            score += 30
            notes.append("ctx=BREAKOUT")
        elif ctx.context == ContextType.CONTINUATION:
            score += 12

        # Volume confirms (15 pts)
        vol_bull = ctx.score_breakdown.get("volume", {}).get("bull", 0)
        vol_bear = ctx.score_breakdown.get("volume", {}).get("bear", 0)
        if max(vol_bull, vol_bear) >= 8:
            score += 15
            notes.append("vol_confirmed")
        else:
            score += 3

        # Macro alignment bonus (15 pts)
        if macro.bias in (TrendBias.STRONG_BULL, TrendBias.BULL, TrendBias.STRONG_BEAR, TrendBias.BEAR):
            score += 15
            notes.append("macro_directional")
        else:
            score += 5

        return round(min(100.0, score), 1), ", ".join(notes)

    def _score_swing_reversal(
        self,
        macro: MacroTrendResult,
        ctx: ContextBiasResult,
        regime: RegimeResult,
    ) -> tuple[float, str]:
        score = 0.0
        notes = []

        # Needs: reversal regime, distribution/accumulation context, liquidity sweep
        if regime.regime == MarketRegime.REVERSAL:
            score += 40
            notes.append("regime=REVERSAL")
        elif regime.regime == MarketRegime.HIGH_VOLATILITY:
            score += 10
        else:
            score += 3

        # Distribution / Accumulation contexts (30 pts)
        if ctx.context in (ContextType.DISTRIBUTION, ContextType.ACCUMULATION):
            score += 30
            notes.append(f"ctx={ctx.context.value}")
        else:
            score += 5

        # Liquidity sweep detected (20 pts)
        liq_bull = ctx.score_breakdown.get("liquidity", {}).get("bull", 0)
        liq_bear = ctx.score_breakdown.get("liquidity", {}).get("bear", 0)
        if max(liq_bull, liq_bear) >= 13:
            score += 20
            notes.append("liquidity_sweep")
        elif max(liq_bull, liq_bear) >= 9:
            score += 10
            notes.append("near_liquidity")

        # Candlestick pattern confirms (10 pts)
        pat_bull = ctx.score_breakdown.get("pattern", {}).get("bull", 0)
        pat_bear = ctx.score_breakdown.get("pattern", {}).get("bear", 0)
        if max(pat_bull, pat_bear) >= 10:
            score += 10
            notes.append("pattern_confirmed")

        return round(min(100.0, score), 1), ", ".join(notes)

    # ── Direction filter ─────────────────────────────────────────────────────

    def _direction_filter(
        self,
        macro: MacroTrendResult,
        ctx: ContextBiasResult,
        selected: StrategyType,
    ) -> str:
        """
        Determine which directions are allowed.
        Strong macro bias blocks counter-trend entries for trend-following strategies.
        Mean reversion and swing reversal can trade against macro but with lower size.
        """
        strong_bull = macro.bias == TrendBias.STRONG_BULL
        strong_bear = macro.bias == TrendBias.STRONG_BEAR

        if selected == StrategyType.TREND_CONTINUATION:
            if strong_bull:
                return "long_only"
            if strong_bear:
                return "short_only"
            if macro.score >= 65:
                return "long_only"
            if macro.score <= 35:
                return "short_only"
            return "both"

        if selected == StrategyType.BREAKOUT:
            # Breakout goes with macro direction
            if macro.score >= 60:
                return "long_only"
            if macro.score <= 40:
                return "short_only"
            return "both"

        if selected in (StrategyType.MEAN_REVERSION, StrategyType.SWING_REVERSAL):
            # Counter-trend is allowed, but block if macro is EXTREMELY one-sided
            if macro.score >= 90 or macro.score <= 10:
                return "none"  # market is running — too risky to fade
            return "both"

        return "both"
