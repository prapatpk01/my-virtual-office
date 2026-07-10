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
    # 6-regime + trade style
    regime_type: str = "TRANSITION"   # STRONG_TREND | HEALTHY_TREND | EARLY_TREND | RANGE | COMPRESSION | TRANSITION
    style: str = "BLOCK"              # TREND | SWING | MEANREV | BREAKOUT | BLOCK
    adx_1h: float = 0.0
    chop_1h: float = 0.0
    atr_pct_1h: float = 50.0
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
        ext = r1.extension_atr
        extension_ok = ext <= c.regime_extension_atr_max

        # 1H context indicators for RANGE / COMPRESSION detection
        src = df_1h if (df_1h is not None and len(df_1h)) else df_4h
        adx_1h = float(ind.adx(src, c.regime_adx_period)[0].iloc[-1] or 0.0)
        chop_1h = float(ind.choppiness_index(src, c.regime_chop_period).iloc[-1] or 50.0)
        atr_s = ind.atr(src, c.regime_atr_period)
        atr_pct_1h = float(ind.atr_percentile(atr_s, c.regime_atr_pct_lookback).iloc[-1])
        if np.isnan(atr_pct_1h):
            atr_pct_1h = 50.0

        long_labels = {BULL, EARLY_BULL}
        short_labels = {BEAR, EARLY_BEAR}
        both_long = r4.label in long_labels and r1.label in long_labels
        both_short = r4.label in short_labels and r1.label in short_labels
        aligned = both_long or both_short
        direction = LONG if both_long else SHORT if both_short else NEUTRAL

        strong_conflict = (r4.direction == LONG and r1.direction == SHORT and r1.score >= c.regime_normal_score) or \
                          (r4.direction == SHORT and r1.direction == LONG and r1.score >= c.regime_normal_score)

        # ── 6-regime classification (priority order) ─────────────────────────
        if strong_conflict or r4.label == TRANSITION or r1.label == TRANSITION:
            regime_type = "TRANSITION"
        elif aligned and combined >= c.regime_strong_trend_min:
            regime_type = "STRONG_TREND"
        elif aligned and combined >= c.regime_healthy_trend_min:
            regime_type = "HEALTHY_TREND"
        elif aligned and combined >= c.regime_early_trend_min:
            regime_type = "EARLY_TREND"
        elif atr_pct_1h <= c.regime_compression_atrpct_max and chop_1h >= c.regime_range_chop_min:
            regime_type = "COMPRESSION"
        elif adx_1h < c.regime_range_adx_max and chop_1h >= c.regime_range_chop_min:
            regime_type = "RANGE"
        else:
            regime_type = "TRANSITION"

        # ── regime type -> style + adaptive threshold + quality/size ─────────
        adj, quality, size_mult, style, allow = 0.0, "normal", c.size_mult_normal, "BLOCK", False
        if regime_type == "STRONG_TREND":
            adj, quality, size_mult, style, allow = c.regime_strong_trend_adj, "strong", c.size_mult_strong, "TREND", True
        elif regime_type == "HEALTHY_TREND":
            adj, quality, size_mult, style, allow = 0.0, "normal", c.size_mult_normal, "TREND", True
        elif regime_type == "EARLY_TREND":
            adj, quality, size_mult, style, allow = c.regime_early_trend_adj, "weak", c.size_mult_weak, "SWING", True
        elif regime_type == "RANGE" and c.style_range_enabled:
            adj, quality, size_mult, style, allow = 0.0, "weak", c.meanrev_size_mult, "MEANREV", True
        elif regime_type == "COMPRESSION" and c.style_compression_enabled:
            adj, quality, size_mult, style, allow = 0.0, "weak", c.breakout_size_mult, "BREAKOUT", True
        else:  # TRANSITION or disabled style
            adj, quality, size_mult, style, allow = 0.0, "transition", c.size_mult_transition, "BLOCK", False

        # trend styles carry the extension anti-chase hard gate; mean-reversion
        # WANTS extension (it fades stretched price) so it's exempt.
        if style in ("TREND", "SWING") and not extension_ok:
            allow = False
            reason = f"{regime_type}: overextended {ext:.1f}ATR past EMA20 (max {c.regime_extension_atr_max})"
        elif not allow:
            reason = f"{regime_type}: no tradeable style"
        else:
            reason = f"{regime_type} -> {style}"

        return RegimeResult(
            allow_trade=allow, direction=direction if style in ("TREND", "SWING") else NEUTRAL,
            regime_4h=r4.label, regime_1h=r1.label, score_4h=r4.score, score_1h=r1.score,
            combined_score=combined, aligned=aligned, extension_atr=ext,
            price_extension_ok=extension_ok, adaptive_threshold_adj=adj,
            size_multiplier=size_mult, quality=quality, reason=reason,
            regime_type=regime_type, style=style, adx_1h=adx_1h, chop_1h=chop_1h, atr_pct_1h=atr_pct_1h,
            name=regime_type, score=combined,
            components={"4h": r4.components, "1h": r1.components},
        )
