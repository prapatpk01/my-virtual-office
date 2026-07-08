"""
Market Regime Engine — TF 4H.

Classifies the market BEFORE any entry is considered. This is the outermost
gate: if the regime says don't trade, nothing downstream matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

REGIME_TREND          = "TREND"
REGIME_EARLY_TREND    = "EARLY_TREND"
REGIME_RANGE          = "RANGE"
REGIME_HIGH_VOL       = "HIGH_VOLATILITY"
REGIME_COMPRESSION    = "COMPRESSION"
REGIME_TRANSITION     = "TRANSITION"


@dataclass
class RegimeResult:
    name: str
    score: float
    direction: str              # 'long' | 'short' | 'neutral' — EMA-structure hint
    adx: float
    chop: float
    atr_pct: float
    components: dict = field(default_factory=dict)
    allow_trade: bool = False
    breakout_confirmed: bool = False   # only meaningful in COMPRESSION
    entry_threshold_adj: float = 0.0   # applied to entry_score_min downstream
    bias_threshold_adj: float = 0.0    # applied to bias_score_min downstream
    size_multiplier: float = 1.0       # applied to position size downstream
    note: str = ""


class RegimeEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_4h: pd.DataFrame) -> RegimeResult:
        c = self.cfg
        n = len(df_4h)
        if n < max(c.regime_ema_slow, c.regime_atr_pct_lookback // 2) + 5:
            return RegimeResult(REGIME_TRANSITION, 0.0, "neutral", 0.0, 50.0, 50.0,
                                allow_trade=False, note="insufficient 4h history")

        closes = df_4h["close"]
        e_fast = ind.ema(closes, c.regime_ema_fast)
        e_slow = ind.ema(closes, c.regime_ema_slow)
        e_fast_v, e_slow_v = float(e_fast.iloc[-1]), float(e_slow.iloc[-1])

        adx_s, plus_di, minus_di = ind.adx(df_4h, c.regime_adx_period)
        adx_v = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0.0

        chop_s = ind.choppiness_index(df_4h, c.regime_chop_period)
        chop_v = float(chop_s.iloc[-1]) if not np.isnan(chop_s.iloc[-1]) else 50.0

        atr_s = ind.atr(df_4h, c.regime_atr_period)
        atr_pct_s = ind.atr_percentile(atr_s, c.regime_atr_pct_lookback)
        atr_pct_v = float(atr_pct_s.iloc[-1]) if not np.isnan(atr_pct_s.iloc[-1]) else 50.0

        slope_s = ind.slope_pct(e_fast, c.regime_ema_slope_lookback)
        slope_v = float(slope_s.iloc[-1]) if not np.isnan(slope_s.iloc[-1]) else 0.0

        # Reference direction from EMA structure (score is direction-agnostic:
        # it measures "how strongly trending", using whichever side the EMAs
        # currently point).
        if e_fast_v > e_slow_v:
            direction = "long"
        elif e_fast_v < e_slow_v:
            direction = "short"
        else:
            direction = "neutral"

        comp_trend_align = 30.0 if direction != "neutral" else 0.0
        slope_aligned = (direction == "long" and slope_v > 0) or (direction == "short" and slope_v < 0)
        comp_slope = 20.0 if slope_aligned else 0.0
        comp_adx = 25.0 if c.regime_adx_trend_lo <= adx_v <= c.regime_adx_trend_hi else 0.0
        comp_chop = 15.0 if chop_v < 50.0 else 0.0
        comp_atr = 10.0 if 30.0 <= atr_pct_v <= 85.0 else 0.0

        score = comp_trend_align + comp_slope + comp_adx + comp_chop + comp_atr
        components = {
            "trend_align": comp_trend_align, "slope": comp_slope, "adx": comp_adx,
            "chop": comp_chop, "atr_pct": comp_atr,
        }

        # ── Classification (priority order matters) ─────────────────────────
        if score >= c.regime_trend_score_min:
            name = REGIME_TREND
        elif score >= c.regime_early_trend_score_min:
            name = REGIME_EARLY_TREND
        elif chop_v >= c.regime_chop_range_min and adx_v < c.regime_adx_trend_lo:
            name = REGIME_RANGE
        elif atr_pct_v >= c.regime_atr_pct_high_vol:
            name = REGIME_HIGH_VOL
        elif atr_pct_v <= c.regime_atr_pct_compression and chop_v >= c.regime_chop_compression_min:
            name = REGIME_COMPRESSION
        else:
            name = REGIME_TRANSITION

        result = RegimeResult(name=name, score=score, direction=direction, adx=adx_v,
                              chop=chop_v, atr_pct=atr_pct_v, components=components)

        # ── Trading rule per regime ──────────────────────────────────────────
        if name in (REGIME_TREND, REGIME_EARLY_TREND):
            result.allow_trade = True
            result.note = f"{name}: trading allowed"
        elif name == REGIME_RANGE:
            result.allow_trade = False
            result.note = "RANGE: no new entries"
        elif name == REGIME_HIGH_VOL:
            result.allow_trade = True
            result.size_multiplier = 1.0 - c.regime_high_vol_size_cut
            result.note = f"HIGH_VOLATILITY: size cut {c.regime_high_vol_size_cut*100:.0f}%"
        elif name == REGIME_COMPRESSION:
            breakout = self._breakout_confirmed(df_4h)
            result.breakout_confirmed = breakout
            result.allow_trade = breakout
            result.note = "COMPRESSION: breakout confirmed" if breakout else "COMPRESSION: waiting for breakout"
        else:  # TRANSITION
            result.allow_trade = True
            result.entry_threshold_adj = -c.regime_transition_entry_relax
            result.bias_threshold_adj  = c.regime_transition_bias_tighten
            result.note = "TRANSITION: relaxed entry, tighter bias required"

        return result

    def _breakout_confirmed(self, df_4h: pd.DataFrame) -> bool:
        """COMPRESSION regime: only allow trade if price just broke the prior N-bar range."""
        c = self.cfg
        lb = c.regime_compression_breakout_lookback
        if len(df_4h) < lb + 2:
            return False
        prior_high = df_4h["high"].iloc[-(lb + 1):-1].max()
        prior_low  = df_4h["low"].iloc[-(lb + 1):-1].min()
        close = float(df_4h["close"].iloc[-1])
        return bool(close > prior_high or close < prior_low)
