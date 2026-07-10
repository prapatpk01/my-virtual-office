"""
Decision Pipeline — the single orchestrator both main.py (live) and
backtest.py call, so the two can never diverge in logic.

    Regime -> Bias -> Direction -> Entry -> Execution

"Directional Trading Architecture, not Signal-Driven Trading":
  - Layer 1 (Regime, 4H+1H) ONLY classifies market state. It never opens a
    position and never picks a side.
  - Layer 2 (Bias, 1H+15M+5M) ONLY picks Trading Direction (Long Only /
    Short Only / No Trade) via a STRICT all-timeframe gate. After this
    layer, no other layer may change direction.
  - Layer 3 (Entry, 30M) ONLY times the trigger. It receives the fixed
    direction and searches exclusively for a trigger on that side — it has
    no right to open the opposite side even if one appears.

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
    blocked_layer: Optional[str]      # REGIME | BIAS | ENTRY | None
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
    """Regime (classify) -> Bias (direction, strict AND) -> Entry (timing)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.entry_engine = EntryEngine(cfg)

    def evaluate(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame,
                 df_4h: pd.DataFrame, df_15m: Optional[pd.DataFrame] = None,
                 df_5m: Optional[pd.DataFrame] = None) -> PipelineResult:
        price = float(df_30m["close"].iloc[-1]) if len(df_30m) else 0.0

        # ── Layer 1 — Regime (classification only) ────────────────────────────
        regime = self.regime_engine.analyze(df_4h, df_1h)
        base = dict(price=price, regime=regime)

        # ── Layer 2 — Bias (Trading Direction, strict all-TF AND) ─────────────
        bias = self.bias_engine.analyze(df_1h, df_15m, df_5m, regime.label)
        if bias.direction not in (B_LONG, B_SHORT):
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="BIAS",
                                  reason=bias.reason, bias=bias, **base)

        side = LONG if bias.direction == B_LONG else SHORT

        # ── Layer 3 — Entry (timing only; direction is FIXED, cannot change) ──
        entry = self.entry_engine.analyze(df_30m, side)
        common = dict(bias=bias, entry=entry, **base)

        if entry.allow_entry:
            return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                                  reason=entry.reason, **common)
        return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="ENTRY",
                              reason=entry.reason, **common)


# Backward-compat alias — main.py / backtest.py imported `SignalEngine`.
SignalEngine = Pipeline
