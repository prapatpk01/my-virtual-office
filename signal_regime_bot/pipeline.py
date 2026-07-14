"""
Decision Pipeline — the single orchestrator both main.py (live) and
backtest.py call, so the two can never diverge in logic.

    Regime -> Bias -> Direction -> Entry (3.1 -> 3.2 -> 3.3) -> Execution

"Directional Trading Architecture, not Signal-Driven Trading":
  - Layer 1 (Regime, 4H+1H) ONLY classifies market state. It never opens a
    position and never picks a side.
  - Layer 2 (Bias, 1H+15M+5M) ONLY picks Trading Direction (Long Only /
    Short Only / No Trade) via a Dynamic Combined Bias Score. After this
    layer, no other layer may change direction.
  - Layer 3 (Entry) receives the fixed direction and searches exclusively
    for a trigger on that side — it has no right to open the opposite
    side even if one appears. Three sequential sub-layers, all must clear
    (see entry_engine.py for the full detail):
      3.1  15M 5-category quality pre-filter
      3.2  15M+5M prior-acceleration wait-rounds
      3.3  15M HMA10/HMA16 fresh-cross timing trigger + anti-chase

Each layer can only narrow what the next sees. The first block short-
circuits the rest and records which layer blocked, with its reason, so the
caller can log a specific cause — never a generic "no signal".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import Config
from regime_engine import RegimeEngine, RegimeResult
from bias_engine import BiasEngine, BiasResult, LONG as B_LONG, SHORT as B_SHORT
from entry_engine import EntryEngine, EntryResult, LONG, SHORT, NONE


@dataclass
class PipelineResult:
    direction: str                    # LONG | SHORT | NONE  (final decision)
    price: float
    entry_score: float
    blocked_layer: Optional[str]      # BIAS | ENTRY | None
    reason: str                       # final human-readable cause
    regime: RegimeResult
    bias: Optional[BiasResult] = None
    entry: Optional[EntryResult] = None
    # backward-compat fields some call sites (main.py logging, telegram,
    # position sizing) still read.
    size_multiplier: float = 1.0
    round_id: object = None
    round_age_bars: object = None
    used_booster: bool = False
    context: object = None
    booster: object = None


class Pipeline:
    """Regime (classify) -> Bias (direction) -> Entry (3.1 -> 3.2 -> 3.3)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.entry_engine = EntryEngine(cfg)

    def evaluate(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                 df_15m: pd.DataFrame, df_5m: Optional[pd.DataFrame] = None,
                 symbol: str = "") -> PipelineResult:
        has_15m = df_15m is not None and len(df_15m)
        price = float(df_15m["close"].iloc[-1]) if has_15m else 0.0

        # HMA cross bookkeeping runs on EVERY evaluation, before any layer can
        # short-circuit — a cross that fires while Bias reads NO TRADE is
        # still a real cross event, and missing it would leave the Layer 3.3
        # cycle state (waiting_for_new_cross) stuck. Idempotent per bar.
        if has_15m:
            self.entry_engine.observe(df_15m, symbol)

        # ── Layer 1 — Regime (classification only) ────────────────────────────
        regime = self.regime_engine.analyze(df_4h, df_1h)
        base = dict(price=price, regime=regime)

        # ── Layer 2 — Bias (Trading Direction, Dynamic Combined Bias Score) ───
        bias = self.bias_engine.analyze(df_1h, df_15m, df_5m, regime.label)
        if bias.direction not in (B_LONG, B_SHORT):
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="BIAS",
                                  reason=bias.reason, bias=bias, **base)

        side = LONG if bias.direction == B_LONG else SHORT

        # ── Layer 3 — Entry (3.1 15M pre-filter -> 3.2 accel wait -> 3.3 HMA) ──
        if not has_15m:
            entry = EntryResult(NONE, False, "missing 15m frame for entry")
        else:
            entry = self.entry_engine.analyze(df_15m, df_5m, side, symbol)
        common = dict(bias=bias, entry=entry, **base)

        if entry.allow_entry:
            return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                                  reason=entry.reason, **common)
        return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="ENTRY",
                              reason=entry.reason, **common)


# Backward-compat alias — main.py / backtest.py imported `SignalEngine`.
SignalEngine = Pipeline
