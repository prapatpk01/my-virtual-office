"""
Layer 1 — Regime Engine (HARD GATE).  4H + 1H.

Answers one question: "Should the system trade at all, and which side?"
This is the only hard gate at the top of the pipeline — if it blocks,
nothing downstream runs. It also emits the adaptive_threshold_adj and
size_multiplier that the Entry and Risk layers read.

Design: each timeframe gets a directional regime label + 0-100 quality
score from the SAME feature set (EMA structure, slope, ADX, chop, ATR).
The combined score is a weighted blend; the gate opens only when both
timeframes lean the same way, they aren't strongly conflicting, and price
isn't overextended past EMA20 (anti-chase).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

# directional regime labels per timeframe
BULL        = "BULL"
EARLY_BULL  = "EARLY_BULL"
BEAR        = "BEAR"
EARLY_BEAR  = "EARLY_BEAR"
RANGE       = "RANGE"
TRANSITION  = "TRANSITION"

LONG    = "LONG"
SHORT   = "SHORT"
NEUTRAL = "NEUTRAL"


@dataclass
class TFRegime:
    label: str
    score: float          # 0-100 quality
    direction: str        # LONG | SHORT | NEUTRAL
    extension_atr: float
    components: dict = field(default_factory=dict)


@dataclass
class RegimeResult:
    allow_trade: bool
    direction: str
    regime_4h: str
    regime_1h: str
    score_4h: float
    score_1h: float
    combined_score: float
    aligned: bool
    extension_atr: float
    price_extension_ok: bool
    adaptive_threshold_adj: float
    size_multiplier: float
    quality: str          # 'strong' | 'normal' | 'weak' | 'transition'
    reason: str
    # kept for backward-compat with health monitor / status log call sites
    name: str = ""
    score: float = 0.0
    components: dict = field(default_factory=dict)


class RegimeEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── One timeframe -> directional label + quality score ───────────────────
    def _tf_regime(self, df: pd.DataFrame) -> TFRegime:
        c = self.cfg
        if len(df) < c.regime_ema_slow + 6:
            return TFRegime(TRANSITION, 0.0, NEUTRAL, 0.0)

        closes = df["close"]
        e_fast = ind.ema(closes, c.regime_ema_fast)
        e_slow = ind.ema(closes, c.regime_ema_slow)
        e_fast_v, e_slow_v = float(e_fast.iloc[-1]), float(e_slow.iloc[-1])
        price = float(closes.iloc[-1])

        adx_s, _, _ = ind.adx(df, c.regime_adx_period)
        adx_v = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0.0
        chop_s = ind.choppiness_index(df, c.regime_chop_period)
        chop_v = float(chop_s.iloc[-1]) if not np.isnan(chop_s.iloc[-1]) else 50.0
        slope_v = float(ind.slope_pct(e_fast, c.regime_ema_slope_lookback).iloc[-1] or 0.0)
        atr_v = float(ind.atr(df, c.regime_atr_period).iloc[-1])
        structure = ind.market_structure(df["high"], df["low"],
                                         c.bias_structure_left, c.bias_structure_right)

        # ── Direction + label come from EMA STRUCTURE, not the quality score.
        # "Bull regime" is a structural fact (EMA20>EMA50); the 0-100 score
        # only measures HOW strong it is, and drives the adaptive threshold.
        # Coupling the label to a high score made alignment unreachable on
        # real data (per-TF score rarely cleared 60, so nothing was ever
        # labelled even EARLY_BULL).
        price_above = price > e_fast_v
        slope_up = slope_v > 0
        if e_fast_v > e_slow_v:
            direction = LONG
            label = BULL if (price_above and slope_up) else EARLY_BULL
        elif e_fast_v < e_slow_v:
            direction = SHORT
            label = BEAR if ((not price_above) and (not slope_up)) else EARLY_BEAR
        else:
            direction = NEUTRAL
            label = RANGE

        # quality score 0-100 (30 structure / 20 slope / 25 ADX / 15 chop / 10 struct-HH-LL)
        comp = {}
        comp["trend_align"] = 30.0 if direction != NEUTRAL else 0.0
        slope_ok = (direction == LONG and slope_v > 0) or (direction == SHORT and slope_v < 0)
        comp["slope"] = 20.0 if slope_ok else 0.0
        # ADX >= lo means "trending" — no upper cap (a strong 4H trend shouldn't
        # be scored down for being strong; the old 18-40 band zeroed most trends).
        comp["adx"] = 25.0 if adx_v >= c.regime_adx_trend_lo else 0.0
        comp["chop"] = 15.0 if chop_v < 50.0 else 0.0
        struct_ok = (direction == LONG and structure == "HH_HL") or \
                    (direction == SHORT and structure == "LH_LL")
        comp["structure"] = 10.0 if struct_ok else 0.0
        score = sum(comp.values())

        ext = abs(price - e_fast_v) / atr_v if (atr_v and atr_v > 0 and not np.isnan(atr_v)) else 0.0
        return TFRegime(label, score, direction, ext, comp)

    # ── Combined 4H + 1H hard gate ───────────────────────────────────────────
    def analyze(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame | None = None) -> RegimeResult:
        c = self.cfg
        r4 = self._tf_regime(df_4h)
        # 1H is optional so single-TF callers (e.g. legacy backtest) still work;
        # if absent, 4H alone drives the gate.
        r1 = self._tf_regime(df_1h) if df_1h is not None and len(df_1h) else r4

        combined = round(r4.score * c.regime_weight_4h + r1.score * c.regime_weight_1h, 1)
        # extension anti-chase uses the FASTER (1H) frame — that's where a chase shows first
        ext = r1.extension_atr
        extension_ok = ext <= c.regime_extension_atr_max

        long_labels = {BULL, EARLY_BULL}
        short_labels = {BEAR, EARLY_BEAR}
        both_long = r4.label in long_labels and r1.label in long_labels
        both_short = r4.label in short_labels and r1.label in short_labels
        aligned = both_long or both_short

        # strong conflict: the two frames point opposite with real conviction
        strong_conflict = (r4.direction == LONG and r1.direction == SHORT and r1.score >= c.regime_normal_score) or \
                          (r4.direction == SHORT and r1.direction == LONG and r1.score >= c.regime_normal_score)

        direction = LONG if both_long else SHORT if both_short else NEUTRAL

        # ── adaptive threshold + quality band from combined score ────────────
        if combined >= c.regime_strong_score:
            adj, quality, size_mult = -5.0, "strong", c.size_mult_strong
        elif combined >= c.regime_normal_score:
            adj, quality, size_mult = 0.0, "normal", c.size_mult_normal
        elif combined >= c.regime_block_below_score:
            adj, quality, size_mult = 5.0, "weak", c.size_mult_weak
        else:
            adj, quality, size_mult = 0.0, "transition", c.size_mult_transition

        # a TRANSITION label on either frame pins quality to transition and
        # NEVER relaxes the entry bar (spec: regime_transition_relax = 0)
        if r4.label == TRANSITION or r1.label == TRANSITION:
            quality = "transition"
            size_mult = c.size_mult_transition
            adj = max(adj, c.regime_transition_relax)   # never negative in transition

        # ── the hard gate ─────────────────────────────────────────────────────
        reasons = []
        allow = True
        if combined < c.regime_block_below_score:
            allow = False; reasons.append(f"combined score {combined:.0f} < {c.regime_block_below_score:.0f}")
        if not aligned:
            allow = False; reasons.append(f"4H={r4.label} 1H={r1.label} not aligned")
        if strong_conflict:
            allow = False; reasons.append("4H/1H strong conflict")
        if not extension_ok:
            allow = False; reasons.append(f"overextended {ext:.1f}ATR past EMA20 (max {c.regime_extension_atr_max})")

        reason = "regime OK" if allow else "; ".join(reasons)

        return RegimeResult(
            allow_trade=allow, direction=direction if allow else NEUTRAL,
            regime_4h=r4.label, regime_1h=r1.label, score_4h=r4.score, score_1h=r1.score,
            combined_score=combined, aligned=aligned, extension_atr=ext,
            price_extension_ok=extension_ok, adaptive_threshold_adj=adj,
            size_multiplier=size_mult, quality=quality, reason=reason,
            name=r4.label, score=combined,
            components={"4h": r4.components, "1h": r1.components},
        )
