"""
Layer 4: Dynamic Strategy Selector.

Maps the Layer 3 winning Primary Regime to exactly ONE strategy —
never blend multiple strategies' logic together. Momentum Expansion is
picked over plain Trend Continuation when Layer 3's Secondary State is
EXPANSION (a trending market that's also actively expanding calls for
a different entry cadence than a routine pullback-continuation).

  BULL_TREND / BEAR_TREND  -> TREND_CONTINUATION (or MOMENTUM_EXPANSION
                               if Secondary State == EXPANSION)
  RANGE                    -> MEAN_REVERSION
  BREAKOUT                 -> BREAKOUT
  REVERSAL                 -> SWING_REVERSAL
  COMPRESSION              -> BREAKOUT_PREP   (readiness only, no entries)
  EXHAUSTION               -> PROFIT_PROTECTION (manage existing only, no entries)
  TRANSITION               -> NO_TRADE

This layer is a hard gate at the regime boundary (exactly one strategy
survives) but the weights it hands to Layer 5/6 are score-based, not
further hard-gated — indicator categories are still blended smoothly
within whichever single strategy got selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .macro_trend_engine import MacroTrendResult, TrendBias
from .context_bias_engine import ContextBiasResult, ContextType
from .regime_classifier import RegimeResult, PrimaryRegime, SecondaryState


class StrategyType(str, Enum):
    TREND_CONTINUATION  = "trend_continuation"
    MEAN_REVERSION       = "mean_reversion"
    BREAKOUT              = "breakout"
    SWING_REVERSAL        = "swing_reversal"
    MOMENTUM_EXPANSION    = "momentum_expansion"
    BREAKOUT_PREP         = "breakout_prep"       # Compression — no entries, readiness only
    PROFIT_PROTECTION     = "profit_protection"   # Exhaustion — manage existing only
    NO_TRADE               = "no_trade"


# Indicator-category weights per strategy type. Categories map onto the
# ExpertScores 7-category surface (trend/momentum/liquidity/volume/
# volatility/session/correlation) that Layer 6 blends — the underlying
# strategy-specific indicators (EMA/ADX/RSI/VWAP/BB/CHOCH/etc.) used for
# actual entry/SL/TP live in Layer 5's StrategyEngine, not here.
_STRATEGY_WEIGHTS: dict[StrategyType, dict[str, float]] = {
    StrategyType.TREND_CONTINUATION: {
        "trend": 0.25, "momentum": 0.20, "liquidity": 0.10,
        "volume": 0.15, "volatility": 0.10, "session": 0.10, "correlation": 0.10,
    },
    StrategyType.MEAN_REVERSION: {
        "trend": 0.05, "momentum": 0.20, "liquidity": 0.20,
        "volume": 0.15, "volatility": 0.05, "session": 0.15, "correlation": 0.20,
    },
    StrategyType.BREAKOUT: {
        "trend": 0.10, "momentum": 0.20, "liquidity": 0.15,
        "volume": 0.25, "volatility": 0.20, "session": 0.10, "correlation": 0.00,
    },
    StrategyType.SWING_REVERSAL: {
        "trend": 0.05, "momentum": 0.15, "liquidity": 0.25,
        "volume": 0.15, "volatility": 0.10, "session": 0.10, "correlation": 0.20,
    },
    StrategyType.MOMENTUM_EXPANSION: {
        "trend": 0.15, "momentum": 0.30, "liquidity": 0.05,
        "volume": 0.25, "volatility": 0.15, "session": 0.05, "correlation": 0.05,
    },
}
for _st, _w in _STRATEGY_WEIGHTS.items():
    _total = sum(_w.values())
    if abs(_total - 1.0) > 0.001:
        _STRATEGY_WEIGHTS[_st] = {k: v / _total for k, v in _w.items()}

_NEUTRAL_WEIGHTS = {
    "trend": 1/7, "momentum": 1/7, "liquidity": 1/7, "volume": 1/7,
    "volatility": 1/7, "session": 1/7, "correlation": 1/7,
}


@dataclass
class StrategySelectionResult:
    selected: StrategyType
    regime: PrimaryRegime
    secondary: SecondaryState
    weights: dict[str, float]
    direction_filter: str               # "long_only" | "short_only" | "both" | "none"
    block_reason: str = ""
    detail: dict = field(default_factory=dict)

    def allows_long(self) -> bool:
        return self.direction_filter in ("long_only", "both")

    def allows_short(self) -> bool:
        return self.direction_filter in ("short_only", "both")

    def is_tradeable(self) -> bool:
        return self.selected not in (
            StrategyType.NO_TRADE, StrategyType.BREAKOUT_PREP, StrategyType.PROFIT_PROTECTION,
        ) and self.direction_filter != "none"


class RegimeStrategySelector:
    """Layer 4 — one regime, one strategy, always."""

    _REGIME_MAP: dict[PrimaryRegime, StrategyType] = {
        PrimaryRegime.RANGE: StrategyType.MEAN_REVERSION,
        PrimaryRegime.BREAKOUT: StrategyType.BREAKOUT,
        PrimaryRegime.REVERSAL: StrategyType.SWING_REVERSAL,
        PrimaryRegime.COMPRESSION: StrategyType.BREAKOUT_PREP,
        PrimaryRegime.EXHAUSTION: StrategyType.PROFIT_PROTECTION,
        PrimaryRegime.TRANSITION: StrategyType.NO_TRADE,
    }

    def select(
        self,
        macro: MacroTrendResult,
        context: ContextBiasResult,
        regime: RegimeResult,
    ) -> StrategySelectionResult:
        primary = regime.primary

        if primary in (PrimaryRegime.BULL_TREND, PrimaryRegime.BEAR_TREND):
            selected = (
                StrategyType.MOMENTUM_EXPANSION
                if regime.secondary == SecondaryState.EXPANSION
                else StrategyType.TREND_CONTINUATION
            )
        else:
            selected = self._REGIME_MAP.get(primary, StrategyType.NO_TRADE)

        weights = _STRATEGY_WEIGHTS.get(selected, _NEUTRAL_WEIGHTS)

        if selected in (StrategyType.NO_TRADE, StrategyType.BREAKOUT_PREP, StrategyType.PROFIT_PROTECTION):
            reason = {
                StrategyType.NO_TRADE: "Transition regime — no directional edge",
                StrategyType.BREAKOUT_PREP: "Compression regime — awaiting breakout, no entries yet",
                StrategyType.PROFIT_PROTECTION: "Exhaustion regime — manage existing positions only",
            }[selected]
            return StrategySelectionResult(
                selected=selected, regime=primary, secondary=regime.secondary,
                weights=weights, direction_filter="none", block_reason=reason,
                detail={"regime_confidence": regime.confidence},
            )

        direction_filter = self._direction_filter(macro, selected, primary)
        block_reason = "" if direction_filter != "none" else "Counter-trend blocked by macro direction gate"

        return StrategySelectionResult(
            selected=selected, regime=primary, secondary=regime.secondary,
            weights=weights, direction_filter=direction_filter, block_reason=block_reason,
            detail={"regime_confidence": regime.confidence, "context": context.context.value},
        )

    @staticmethod
    def _direction_filter(macro: MacroTrendResult, selected: StrategyType, primary: PrimaryRegime) -> str:
        """Layer 1's allowed_direction() is the hard fence; trend-following
        strategies obey it directly, counter-trend strategies (mean
        reversion / swing reversal) may fade it but not against an extreme."""
        macro_dir = macro.allowed_direction()

        if selected in (StrategyType.TREND_CONTINUATION, StrategyType.MOMENTUM_EXPANSION, StrategyType.BREAKOUT):
            if macro_dir == "no_trade":
                return "none"
            if macro_dir == "long_only":
                return "long_only" if primary != PrimaryRegime.BEAR_TREND else "none"
            if macro_dir == "short_only":
                return "short_only" if primary != PrimaryRegime.BULL_TREND else "none"
            # macro_dir == "both": follow the regime's own directional lean
            if primary == PrimaryRegime.BULL_TREND:
                return "long_only"
            if primary == PrimaryRegime.BEAR_TREND:
                return "short_only"
            return "both"

        if selected in (StrategyType.MEAN_REVERSION, StrategyType.SWING_REVERSAL):
            if macro.bias in (TrendBias.STRONG_BULL, TrendBias.STRONG_BEAR):
                return "none"  # market running too hard to fade
            return "both"

        return "both"
