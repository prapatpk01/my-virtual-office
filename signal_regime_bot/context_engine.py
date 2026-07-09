"""
Layer 3 — Context Engine (SOFT SCORE).  30M.

Answers: "Is this a high-quality SETUP location?"

Pure soft score — no single component is required. It measures whether the
30M chart is at a place worth acting from (a sweep, a structure break, a
VWAP reclaim, an EMA pullback, expansion, a clean retest, good session)
rather than whether momentum is aligned (that's Bias) or a trigger fired
(that's Entry). The pass threshold ADAPTS to regime quality: a strong
regime earns an easier context bar, a transition demands a much higher one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import price_action as pa
from config import Config

LONG = "LONG"
SHORT = "SHORT"


@dataclass
class ContextResult:
    context_score: float
    context_pass: bool
    components: dict
    threshold: float
    reason: str


class ContextEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _threshold_for(self, regime_quality: str) -> float:
        c = self.cfg
        return {
            "strong": c.context_thr_strong,
            "normal": c.context_thr_normal,
            "weak": c.context_thr_weak,
            "transition": c.context_thr_transition,
        }.get(regime_quality, c.context_base_threshold)

    def analyze(self, df_30m: pd.DataFrame, side: str, regime_quality: str) -> ContextResult:
        c = self.cfg
        threshold = self._threshold_for(regime_quality)

        if len(df_30m) < 30:
            return ContextResult(0.0, False, {}, threshold, "insufficient 30m history")

        comp: dict = {}

        sweep = pa.liquidity_sweep(df_30m, side, c.entry_sweep_lookback)
        comp["liquidity_sweep"] = c.context_w_sweep if sweep else 0.0

        bos, choch = pa.bos_choch(df_30m, side, c.swing_lookback_left, c.swing_lookback_right)
        comp["bos_choch"] = c.context_w_bos if (bos or choch) else 0.0

        vwap = pa.vwap_reclaim(df_30m, side)
        comp["vwap"] = c.context_w_vwap if vwap else 0.0

        pullback = pa.ema_pullback(df_30m, side, c.entry_ema_ref)
        comp["ema_pullback"] = c.context_w_pullback if pullback else 0.0

        vol = pa.volume_expansion(df_30m, c.entry_vol_expansion_mult)
        comp["volume_expansion"] = c.context_w_volume if vol else 0.0

        retest = pa.successful_retest(df_30m, side, c.entry_sweep_lookback)
        comp["retest"] = c.context_w_retest if retest else 0.0

        sess = pa.session_quality(df_30m)
        comp["session"] = round(c.context_w_session * sess, 1)

        score = round(sum(comp.values()), 1)
        passed = score >= threshold

        present = [k for k, v in comp.items() if v > 0]
        reason = (f"context {score:.0f} >= {threshold:.0f} [{', '.join(present)}]" if passed
                  else f"context {score:.0f} < {threshold:.0f} — weak setup location "
                       f"[have: {', '.join(present) or 'none'}]")

        return ContextResult(score, passed, comp, threshold, reason)
