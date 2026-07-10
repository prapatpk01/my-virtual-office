"""
Decision Pipeline — the single orchestrator both main.py (live) and
backtest.py call, so the two can never diverge in logic.

    Layer 1  Regime   (4H+1H)  HARD GATE   -> may we trade? which side?
    Layer 2  Bias     (1H+15M) SOFT+veto   -> which side has better odds?
    Layer 3  Context  (30M)    SOFT SCORE  -> is this a quality location?
    Layer 4  Entry    (30M)    setup+score -> valid trigger, strong enough?
    Layer 5  Booster  (15M)    timing lift -> rescue a near-miss (never alone)

Each layer can only narrow what the next sees. The first hard gate that
fails short-circuits the rest and records which layer blocked, with its
reason, so the caller can log a specific cause — never a generic "no signal".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import Config
from regime_engine import RegimeEngine, RegimeResult, LONG as R_LONG, SHORT as R_SHORT
from bias_engine import BiasEngine, BiasResult, BIAS_BULL, BIAS_BEAR
from context_engine import ContextEngine, ContextResult
from entry_engine import EntryEngine, EntryResult, LONG, SHORT, NONE
from early_booster import EarlyBooster, BoosterResult


@dataclass
class PipelineResult:
    direction: str                    # LONG | SHORT | NONE  (final decision)
    price: float
    entry_score: float                # final score (after booster if used)
    size_multiplier: float
    blocked_layer: Optional[str]      # REGIME | BIAS | CONTEXT | ENTRY | BOOSTER | None
    reason: str                       # final human-readable cause
    regime: RegimeResult
    bias: Optional[BiasResult] = None
    context: Optional[ContextResult] = None
    entry: Optional[EntryResult] = None
    booster: Optional[BoosterResult] = None
    round_id: object = None
    round_age_bars: object = None
    used_booster: bool = False


class Pipeline:
    """Hard Gate + Soft Score + Adaptive Threshold, in strict layer order."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.context_engine = ContextEngine(cfg)
        self.entry_engine = EntryEngine(cfg)
        self.booster = EarlyBooster(cfg)
        from style_engines import MeanReversionEntry, BreakoutEntry
        self.meanrev_engine = MeanReversionEntry(cfg)
        self.breakout_engine = BreakoutEntry(cfg)

    def evaluate(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame,
                 df_4h: pd.DataFrame, df_15m: Optional[pd.DataFrame] = None) -> PipelineResult:
        price = float(df_30m["close"].iloc[-1]) if len(df_30m) else 0.0

        # ── Layer 1 — Regime (HARD GATE + style routing) ─────────────────────
        regime = self.regime_engine.analyze(df_4h, df_1h)
        base = dict(price=price, size_multiplier=regime.size_multiplier, regime=regime)
        if not regime.allow_trade:
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="REGIME",
                                  reason=regime.reason, **base)

        # RANGE -> mean-reversion, COMPRESSION -> breakout: their own entry
        # logic determines the side and bypasses the trend-momentum bias gate.
        if regime.style in ("MEANREV", "BREAKOUT"):
            return self._evaluate_style(regime, df_30m, base)

        # trend styles (TREND / SWING) — regime already fixed the side
        side = LONG if regime.direction == R_LONG else SHORT

        # ── Layer 2 — Bias (SOFT confirmation of the regime side + hard veto) ─
        # The spec's rule is side-specific: for a LONG regime, allow when the
        # WEIGHTED BULL score clears the threshold AND there's no strong bear.
        # We do NOT require bias to independently "win" the side — momentum
        # (bias) and structure (regime) rarely pick the identical side on the
        # same bar (a pullback in an uptrend reads bearish on momentum), so a
        # winner-take-all match would gate out almost everything.
        bias = self.bias_engine.analyze(df_1h, df_15m)
        c = self.cfg
        if side == LONG:
            side_score, opp_score = bias.bull_score, bias.bear_score
        else:
            side_score, opp_score = bias.bear_score, bias.bull_score
        if opp_score >= c.bias_strong_opposite:
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="BIAS",
                                  reason=f"strong opposite bias {opp_score:.0f} >= {c.bias_strong_opposite:.0f}",
                                  bias=bias, **base)
        if side_score < c.bias_min_threshold:
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="BIAS",
                                  reason=f"{side} bias {side_score:.0f} < {c.bias_min_threshold:.0f}",
                                  bias=bias, **base)

        # ── Layer 3 — Context (SOFT SCORE, adaptive threshold) ───────────────
        context = self.context_engine.analyze(df_30m, side, regime.quality)
        if not context.context_pass:
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="CONTEXT",
                                  reason=context.reason, bias=bias, context=context, **base)

        # ── Layer 4 — Entry (30M setup + adaptive threshold) ─────────────────
        # Pass the REGIME/trade side (not bias's winner-take-all direction) —
        # the bias gate above already confirmed this side is supported. Passing
        # bias.direction would reject a valid setup whenever short-horizon
        # momentum's winner differs from the structural side (e.g. a pullback).
        entry = self.entry_engine.analyze(df_30m, side,
                                          regime.adaptive_threshold_adj, context.context_score)
        common = dict(bias=bias, context=context, entry=entry,
                      round_id=entry.round_id, round_age_bars=entry.setup_age, **base)

        if entry.allow_entry:
            return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                                  reason=entry.reason, **common)

        if not entry.near_miss:
            return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="ENTRY",
                                  reason=entry.reason, **common)

        # ── Layer 5 — Early Booster (rescue the near-miss on 15M timing) ─────
        booster = self.booster.analyze(df_15m, df_30m, side, entry.entry_score,
                                       entry.entry_threshold, regime.quality)
        common["booster"] = booster
        common["used_booster"] = True
        if booster.cancel_setup:
            return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="BOOSTER",
                                  reason=booster.reason, **common)
        if booster.allow_early_entry:
            return PipelineResult(side, entry_score=booster.final_score, blocked_layer=None,
                                  reason=f"early-boost: {booster.reason}", **common)
        return PipelineResult(NONE, entry_score=booster.final_score, blocked_layer="BOOSTER",
                              reason=booster.reason, **common)


    def _evaluate_style(self, regime, df_30m, base) -> PipelineResult:
        """MEANREV / BREAKOUT — the style engine picks the side, then Context
        confirms. The trend-momentum Bias gate is intentionally skipped: a
        mean-reversion trade is counter-trend, so momentum would always veto it."""
        eng = self.meanrev_engine if regime.style == "MEANREV" else self.breakout_engine
        entry = eng.analyze(df_30m)
        common = dict(entry=entry, round_id=entry.round_id,
                      round_age_bars=entry.setup_age, **base)
        if entry.setup_direction not in (LONG, SHORT):
            return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="ENTRY",
                                  reason=f"{regime.style}: {entry.reason}", **common)

        side = entry.setup_direction
        # Context is ADVISORY for style trades — the trend-continuation context
        # (BOS/CHOCH in the trade direction) never aligns at a mean-reversion
        # extreme, so hard-gating on it blocks every fade. The style engine's
        # own multi-component score (RSI extreme + rejection + sweep for
        # MEANREV; break + volume for BREAKOUT) is the quality gate here.
        context = self.context_engine.analyze(df_30m, side, regime.quality)
        common["context"] = context
        return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                              reason=f"{regime.style}: {entry.reason} (ctx {context.context_score:.0f})",
                              **common)


# Backward-compat alias — main.py / backtest.py imported `SignalEngine`.
SignalEngine = Pipeline
