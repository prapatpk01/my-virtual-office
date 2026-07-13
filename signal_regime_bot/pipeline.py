"""
Decision Pipeline — the single orchestrator both main.py (live) and
backtest.py call, so the two can never diverge in logic.

    Regime -> Bias -> Direction -> Entry -> Execution

"Directional Trading Architecture, not Signal-Driven Trading":
  - Layer 1 (Regime, 4H+1H) ONLY classifies market state. It never opens a
    position and never picks a side.
  - Layer 2 (Bias, 1H+15M+5M) ONLY picks Trading Direction (Long Only /
    Short Only / No Trade) via a Dynamic Combined Bias Score. After this
    layer, no other layer may change direction.
  - Layer 3 (Entry, 30M) ONLY times the trigger. It receives the fixed
    direction and searches exclusively for a trigger on that side — it has
    no right to open the opposite side even if one appears.

Context Engine was tried here as a post-Entry quality filter and REMOVED:
measured on the local BTC/XAU backtest (Jan-May 2026), it cut trade volume
~30% but also dropped win rate ~4pp and worsened avg R — it wasn't
discriminating quality, just trimming volume indiscriminately, which a
direct Bias/Entry threshold change does more honestly. See git history
(pipeline.py) if this needs revisiting with a different Context scoring.

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
    """Regime (classify) -> Bias (direction) -> Entry (timing) -> Micro-confirm."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.entry_engine = EntryEngine(cfg)

    # ── Layer 3b helpers — 5M/15M micro-confirmation ─────────────────────────
    def _needs_micro_confirm(self, df_30m: pd.DataFrame, bias: BiasResult,
                             side: str) -> tuple[bool, str]:
        """A trigger needs extra 5M/15M confirmation when it comes from a
        one-bar volatility spike (a single impulsive bounce can satisfy every
        30M category and reverse the next bar — the ETH case) or when the
        fast-TF bias is still ambiguous/two-sided despite clearing its floor."""
        import indicators as ind
        c = self.cfg
        if not c.confirm_enabled:
            return False, ""
        # volatility spike: the 30M trigger bar's range vs its own ATR
        atr30 = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        rng = float(df_30m["high"].iloc[-1] - df_30m["low"].iloc[-1])
        if atr30 > 0 and rng >= c.confirm_spike_atr_mult * atr30:
            return True, f"trigger bar is a {rng/atr30:.1f}xATR spike"
        # ambiguous fast bias: 5M inside the neutral zone (cleared its loose
        # floor but not actually leaning), or 5M leaning AGAINST the trade
        # side while 15M leans with it (two-sided read)
        s5, s15 = bias.score_5m, bias.score_15m
        if c.confirm_ambig_lo <= s5 <= c.confirm_ambig_hi:
            return True, f"5M bias ambiguous ({s5:.0f} in {c.confirm_ambig_lo:.0f}-{c.confirm_ambig_hi:.0f})"
        if side == LONG and s5 < 50 <= s15:
            return True, f"5M ({s5:.0f}) leaning against LONG while 15M ({s15:.0f}) leans with"
        if side == SHORT and s5 > 50 >= s15:
            return True, f"5M ({s5:.0f}) leaning against SHORT while 15M ({s15:.0f}) leans with"
        return False, ""

    def _micro_confirm(self, df_15m: Optional[pd.DataFrame],
                       df_5m: Optional[pd.DataFrame], side: str) -> tuple[bool, str]:
        """Require >= confirm_min_agree of the last confirm_bars CLOSED bars
        on BOTH 15M and 5M to close in the trade direction — sustained follow-
        through, not a single impulsive bounce. Missing frames fail closed
        (no confirmation data -> wait) since this gate only runs when the
        setup was already flagged as spike/ambiguous."""
        c = self.cfg
        n, need = c.confirm_bars, c.confirm_min_agree
        counts = {}
        for name, df in (("15m", df_15m), ("5m", df_5m)):
            if df is None or len(df) < n:
                return False, f"no {name} data for confirmation — wait"
            closes = df["close"].iloc[-n:].values
            opens = df["open"].iloc[-n:].values
            fav = (closes > opens) if side == LONG else (closes < opens)
            counts[name] = int(fav.sum())
            if counts[name] < need:
                return False, (f"{name} only {counts[name]}/{n} bars closed with {side} "
                               f"(need >= {need}) — wait for follow-through")
        return True, f"confirmed: 15m {counts['15m']}/{n}, 5m {counts['5m']}/{n} bars with {side}"

    def evaluate(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame,
                 df_4h: pd.DataFrame, df_15m: Optional[pd.DataFrame] = None,
                 df_5m: Optional[pd.DataFrame] = None) -> PipelineResult:
        price = float(df_30m["close"].iloc[-1]) if len(df_30m) else 0.0

        # ── Layer 1 — Regime (classification only) ────────────────────────────
        regime = self.regime_engine.analyze(df_4h, df_1h)
        base = dict(price=price, regime=regime)

        # ── Layer 2 — Bias (Trading Direction, Dynamic Combined Bias Score) ───
        bias = self.bias_engine.analyze(df_1h, df_15m, df_5m, regime.label)
        if bias.direction not in (B_LONG, B_SHORT):
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="BIAS",
                                  reason=bias.reason, bias=bias, **base)

        side = LONG if bias.direction == B_LONG else SHORT

        # ── Layer 3 — Entry (timing only; direction is FIXED, cannot change) ──
        entry = self.entry_engine.analyze(df_30m, side)
        common = dict(bias=bias, entry=entry, **base)

        if not entry.allow_entry:
            return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="ENTRY",
                                  reason=entry.reason, **common)

        # ── Layer 3b — Micro-confirmation (5M/15M follow-through) ─────────────
        # Only for spike/ambiguous triggers: an impulsive one-bar bounce must
        # show sustained fast-TF follow-through before it's tradeable.
        need_confirm, why = self._needs_micro_confirm(df_30m, bias, side)
        if need_confirm:
            ok, detail = self._micro_confirm(df_15m, df_5m, side)
            if not ok:
                return PipelineResult(NONE, entry_score=entry.entry_score,
                                      blocked_layer="CONFIRM",
                                      reason=f"{why} -> {detail}", **common)
            entry_reason = f"{entry.reason} | {why} -> {detail}"
        else:
            entry_reason = entry.reason

        return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                              reason=entry_reason, **common)


# Backward-compat alias — main.py / backtest.py imported `SignalEngine`.
SignalEngine = Pipeline
