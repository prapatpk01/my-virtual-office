"""
Decision Pipeline — the single orchestrator both main.py (live) and
backtest.py call, so the two can never diverge in logic.

    Regime -> Bias -> Direction -> Entry -> [Accel confirm] -> Execution

"Directional Trading Architecture, not Signal-Driven Trading":
  - Layer 1 (Regime, 4H+1H) ONLY classifies market state. It never opens a
    position and never picks a side.
  - Layer 2 (Bias, 1H+15M+5M) ONLY picks Trading Direction (Long Only /
    Short Only / No Trade) via a Dynamic Combined Bias Score. After this
    layer, no other layer may change direction.
  - Layer 3 (Entry, 30M) ONLY times the trigger. It receives the fixed
    direction and searches exclusively for a trigger on that side — it has
    no right to open the opposite side even if one appears.
  - Layer 3.2 (Accel confirm, 15M+5M) is a CONDITIONAL gate: if price/
    volatility is normal, a valid Layer-3 trigger enters IMMEDIATELY, no
    extra check. ONLY when excessive prior acceleration is detected (last
    4x15M / 10x5M) does it hold and require 1-2 wait rounds of fast-TF
    follow-through before entry — don't chase a violent burst that may
    snap back. See the wait-round machine below.

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

import numpy as np
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
    """Regime (classify) -> Bias (direction) -> Entry (timing) -> Accel wait-rounds."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.entry_engine = EntryEngine(cfg)
        # per-symbol acceleration wait-round state:
        #   {symbol: {"side", "flag_ts", "round", "last_bar"}}
        # In-memory only (like every other runtime counter) — a restart drops
        # a pending wait, which fails safe: the setup is simply re-evaluated.
        self._accel_wait: dict = {}

    # ── Layer 3b helpers — prior-acceleration detection + wait rounds ────────
    def _recent_acceleration(self, df_15m: Optional[pd.DataFrame],
                             df_5m: Optional[pd.DataFrame]) -> tuple[bool, str]:
        """Was there excessive price acceleration JUST BEFORE this entry?
        Looks at the last accel_15m_window (4) closed 15M bars and the last
        accel_5m_window (10) closed 5M bars: net move across the window or
        any single bar's range beyond its own TF's ATR flags the setup —
        entering right after a violent burst is chasing an unstable move
        (the ETH bounce-then-dump case)."""
        import indicators as ind
        c = self.cfg
        for name, df, win in (("15m", df_15m, c.accel_15m_window),
                              ("5m", df_5m, c.accel_5m_window)):
            if df is None or len(df) < win + 15:
                continue
            atr_v = float(ind.atr(df, 14).iloc[-1])
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            seg = df.iloc[-win:]
            net = abs(float(seg["close"].iloc[-1]) - float(seg["open"].iloc[0]))
            max_rng = float((seg["high"] - seg["low"]).max())
            if net >= c.accel_net_atr_mult * atr_v:
                return True, f"{name} net move {net/atr_v:.1f}xATR over last {win} bars"
            if max_rng >= c.accel_bar_atr_mult * atr_v:
                return True, f"{name} bar range {max_rng/atr_v:.1f}xATR within last {win} bars"
        return False, ""

    def _judge_round(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
                     side: str, flag_ts: pd.Timestamp, rnd: int) -> tuple[Optional[bool], str]:
        """Judge wait-round `rnd` (1 or 2). Each round consumes the NEXT
        1x15M + 4x5M bars closed after flag_ts (round 2 = the second 15M bar
        and 5M bars 5-8). Returns (None, ...) while the round's bars haven't
        all closed yet; (True/False, detail) once judgeable. Pass = the
        round's 15M bar closed in the trade direction AND >= accel_round_5m_min
        of its four 5M bars did — i.e. the move held instead of stalling or
        reversing."""
        c = self.cfg
        need15, need5 = rnd, 4 * rnd
        b15 = df_15m[df_15m.index >= flag_ts]
        b5 = df_5m[df_5m.index >= flag_ts]
        if len(b15) < need15 or len(b5) < need5:
            return None, (f"round {rnd}: waiting for post-flag bars "
                          f"(15m {len(b15)}/{need15}, 5m {len(b5)}/{need5})")
        r15 = b15.iloc[need15 - 1]
        r5 = b5.iloc[4 * (rnd - 1): 4 * rnd]
        fav15 = (float(r15["close"]) > float(r15["open"])) if side == LONG \
            else (float(r15["close"]) < float(r15["open"]))
        fav5 = ((r5["close"].values > r5["open"].values) if side == LONG
                else (r5["close"].values < r5["open"].values))
        n5 = int(fav5.sum())
        ok = fav15 and n5 >= c.accel_round_5m_min
        detail = (f"round {rnd}: 15m {'with' if fav15 else 'against'} {side}, "
                  f"5m {n5}/4 with {side}")
        return ok, detail

    def evaluate(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame,
                 df_4h: pd.DataFrame, df_15m: Optional[pd.DataFrame] = None,
                 df_5m: Optional[pd.DataFrame] = None,
                 symbol: str = "") -> PipelineResult:
        price = float(df_30m["close"].iloc[-1]) if len(df_30m) else 0.0

        # ── Layer 1 — Regime (classification only) ────────────────────────────
        regime = self.regime_engine.analyze(df_4h, df_1h)
        base = dict(price=price, regime=regime)

        # ── Layer 2 — Bias (Trading Direction, Dynamic Combined Bias Score) ───
        bias = self.bias_engine.analyze(df_1h, df_15m, df_5m, regime.label)
        if bias.direction not in (B_LONG, B_SHORT):
            # layers changed while a wait was pending -> that setup is dead
            self._accel_wait.pop(symbol, None)
            return PipelineResult(NONE, entry_score=0.0, blocked_layer="BIAS",
                                  reason=bias.reason, bias=bias, **base)

        side = LONG if bias.direction == B_LONG else SHORT

        # ── Layer 3 — Entry (timing only; direction is FIXED, cannot change) ──
        entry = self.entry_engine.analyze(df_30m, side)
        common = dict(bias=bias, entry=entry, **base)
        bar_ts = df_30m.index[-1] if len(df_30m) else None
        state = self._accel_wait.get(symbol)

        # direction flipped while waiting = Layer 2 changed -> setup failed
        if state is not None and state["side"] != side:
            self._accel_wait.pop(symbol, None)
            state = None

        # ── Layer 3b — pending wait-round takes precedence over fresh entries ─
        if state is not None:
            c = self.cfg
            # advance the machine only on a NEW closed 30M bar so live ticks
            # (every ~30s) can't burn a round mid-bar — keeps live == backtest.
            if state["last_bar"] == bar_ts:
                return PipelineResult(NONE, entry_score=entry.entry_score,
                                      blocked_layer="CONFIRM",
                                      reason=f"accel wait round {state['round']} pending (same bar)",
                                      **common)
            state["last_bar"] = bar_ts
            # core signal must still hold while waiting (window-staleness is
            # exempt — this state machine is the timing authority now, capped
            # at 2 rounds; the ENTRY layer's other conditions are not).
            if not entry.core_ok:
                if state["round"] >= c.accel_max_rounds:
                    self._accel_wait.pop(symbol, None)
                    return PipelineResult(NONE, entry_score=entry.entry_score,
                                          blocked_layer="CONFIRM",
                                          reason="accel wait: signal degraded in final round — setup failed, "
                                                 "wait for the next one", **common)
                state["round"] += 1
                return PipelineResult(NONE, entry_score=entry.entry_score,
                                      blocked_layer="CONFIRM",
                                      reason=f"accel wait: signal weakened, extending to round {state['round']}",
                                      **common)
            verdict, detail = self._judge_round(df_15m, df_5m, side,
                                                state["flag_ts"], state["round"])
            if verdict is None:
                return PipelineResult(NONE, entry_score=entry.entry_score,
                                      blocked_layer="CONFIRM",
                                      reason=f"accel wait: {detail}", **common)
            if verdict:
                self._accel_wait.pop(symbol, None)
                return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                                      reason=f"{entry.reason} | accel confirmed ({detail})",
                                      **common)
            if state["round"] >= c.accel_max_rounds:
                self._accel_wait.pop(symbol, None)
                return PipelineResult(NONE, entry_score=entry.entry_score,
                                      blocked_layer="CONFIRM",
                                      reason=f"accel wait failed twice ({detail}) — setup abandoned",
                                      **common)
            state["round"] += 1
            return PipelineResult(NONE, entry_score=entry.entry_score,
                                  blocked_layer="CONFIRM",
                                  reason=f"accel wait: {detail} — pullback/reversal, "
                                         f"extending to round {state['round']}", **common)

        if not entry.allow_entry:
            return PipelineResult(NONE, entry_score=entry.entry_score, blocked_layer="ENTRY",
                                  reason=entry.reason, **common)

        # fresh in-window trigger: check for excessive PRIOR acceleration
        if self.cfg.accel_confirm_enabled:
            accel, why = self._recent_acceleration(df_15m, df_5m)
            if accel:
                flag_ts = (bar_ts + pd.Timedelta(self.cfg.tf_entry)) if bar_ts is not None else None
                self._accel_wait[symbol] = {"side": side, "flag_ts": flag_ts,
                                            "round": 1, "last_bar": bar_ts}
                return PipelineResult(NONE, entry_score=entry.entry_score,
                                      blocked_layer="CONFIRM",
                                      reason=f"prior acceleration ({why}) — waiting round 1 "
                                             f"(1x15m + 4x5m after this bar)", **common)

        return PipelineResult(side, entry_score=entry.entry_score, blocked_layer=None,
                              reason=entry.reason, **common)


# Backward-compat alias — main.py / backtest.py imported `SignalEngine`.
SignalEngine = Pipeline
