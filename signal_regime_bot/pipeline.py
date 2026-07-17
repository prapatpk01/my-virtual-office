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
    side even if one appears. Entry is a 3-layer multi-timeframe cross
    confluence (see entry_engine.py for the full detail):
      L3a HMA10/16 30M + L3b EMA5/9 15M + L3c EMA10/20 5M;
      >= 2 crossing the bias direction within 15 min -> enter

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
                 df_30m: Optional[pd.DataFrame] = None,
                 symbol: str = "") -> PipelineResult:
        has_15m = df_15m is not None and len(df_15m)
        price = float(df_15m["close"].iloc[-1]) if has_15m else 0.0

        # Cross bookkeeping (all 3 layers) runs on EVERY evaluation, before any
        # layer can short-circuit — a cross that fires while Bias reads NO TRADE
        # is still a real cross event, and dropping it would leave the
        # confluence state blind to it. Idempotent per (layer, bar).
        if df_5m is not None and len(df_5m):
            self.entry_engine.observe(df_30m, df_15m, df_5m, symbol)

        # ── Layer 1 — Regime (classification only) ────────────────────────────
        regime = self.regime_engine.analyze(df_4h, df_1h)
        base = dict(price=price, regime=regime)

        # ── Commodity weekend halt (XAU/XAG): the underlying metals market is
        # closed — the OKX perp still quotes but it's illiquid, so NEW entries
        # are blocked from Fri commodity_halt_hour_utc until Sun
        # commodity_resume_hour_utc (= 3h before the Monday session open the
        # user referenced). Open positions keep being managed elsewhere; the
        # observe() above already kept cross bookkeeping alive. Uses the bar's
        # CLOSE time so live and backtest gate identically.
        c = self.cfg
        if (c.commodity_weekend_block_enabled and has_15m
                and any(k in symbol.upper() for k in c.commodity_symbol_keywords)):
            bar_close = df_15m.index[-1] + pd.Timedelta(c.tf_fast)
            wd, hr = bar_close.weekday(), bar_close.hour
            halted = ((wd == 4 and hr >= c.commodity_halt_hour_utc) or wd == 5
                     or (wd == 6 and hr < c.commodity_resume_hour_utc))
            if halted:
                return PipelineResult(NONE, entry_score=0.0, blocked_layer="MARKET",
                                      reason=f"commodity weekend halt ({bar_close.strftime('%a %H:%M')} UTC; "
                                             f"entries resume Sun {c.commodity_resume_hour_utc:02d}:00 UTC)",
                                      **base)

        # ── Layer 2 — Bias (Trading Direction, Dynamic Combined Bias Score) ───
        bias = self.bias_engine.analyze(df_1h, df_15m, df_5m, regime.label)
        if bias.direction not in (B_LONG, B_SHORT):
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="BIAS",
                                  reason=bias.reason, bias=bias, **base)

        side = LONG if bias.direction == B_LONG else SHORT

        # ── Layer 3 — Entry (3-layer cross confluence: L3a 30M + L3b 15M + L3c 5M) ──
        if df_5m is None or not len(df_5m):
            entry = EntryResult(NONE, False, "missing 5m frame for entry")
        else:
            entry = self.entry_engine.analyze(df_30m, df_15m, df_5m, side, symbol)
        common = dict(bias=bias, entry=entry, **base)

        if entry.allow_entry:
            return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                                  reason=entry.reason, **common)
        return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="ENTRY",
                              reason=entry.reason, **common)


# Backward-compat alias — main.py / backtest.py imported `SignalEngine`.
SignalEngine = Pipeline
