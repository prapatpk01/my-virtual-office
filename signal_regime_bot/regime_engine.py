"""
Layer 1 — Market Regime Engine (CLASSIFICATION ONLY).  4H macro + 1H mid.

Answers ONE question: "What state is the market in right now?" This layer
NEVER decides to open a position and never picks a trade direction — it only
emits one label out of seven. Direction is Layer 2's job (Bias); timing is
Layer 3's job (Entry). See pipeline.py for the full "Directional Trading
Architecture" flow this implements.

Output — exactly one of:
    STRONG_BULL_TREND | EARLY_BULL_TREND | RANGE | COMPRESSION |
    EARLY_BEAR_TREND | STRONG_BEAR_TREND | HIGH_VOLATILITY
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

STRONG_BULL = "STRONG_BULL_TREND"
EARLY_BULL  = "EARLY_BULL_TREND"
RANGE       = "RANGE"
COMPRESSION = "COMPRESSION"
EARLY_BEAR  = "EARLY_BEAR_TREND"
STRONG_BEAR = "STRONG_BEAR_TREND"
HIGH_VOL    = "HIGH_VOLATILITY"

BULL_LABELS = (STRONG_BULL, EARLY_BULL)
BEAR_LABELS = (STRONG_BEAR, EARLY_BEAR)

LONG    = "LONG"
SHORT   = "SHORT"
NEUTRAL = "NEUTRAL"


@dataclass
class TFRegime:
    label: str
    checks: dict = field(default_factory=dict)   # criterion name -> bool, for logging
    pass_count: int = 0
    pass_total: int = 6


@dataclass
class RegimeResult:
    label: str            # the 7-way classification (final combined call)
    label_4h: str
    label_1h: str
    checks_4h: dict = field(default_factory=dict)
    checks_1h: dict = field(default_factory=dict)
    reason: str = ""
    # backward-compat fields other modules key off of (health monitor,
    # telegram, status logging) — `name`/`score` are display-only here, this
    # layer no longer gates or sizes trades.
    name: str = ""
    score: float = 0.0
    style: str = ""
    size_multiplier: float = 1.0


class RegimeEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── One timeframe -> 7-way label + the checks that produced it ───────────
    def _tf_regime(self, df: pd.DataFrame) -> TFRegime:
        c = self.cfg
        min_len = max(c.regime_ema_slow, 60) + 10
        if len(df) < min_len:
            return TFRegime(RANGE, {}, 0, 6)

        closes, highs, lows, vols = df["close"], df["high"], df["low"], df["volume"]
        price = float(closes.iloc[-1])

        e20 = ind.ema(closes, c.regime_ema_fast)
        e50 = ind.ema(closes, c.regime_ema_slow)
        e20_v, e50_v = float(e20.iloc[-1]), float(e50.iloc[-1])
        slope20 = float(ind.slope_pct(e20, c.regime_ema_slope_lookback).iloc[-1] or 0.0)

        adx_s, _, _ = ind.adx(df, c.regime_adx_period)
        adx_now = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0.0
        adx_prev = float(adx_s.iloc[-2]) if not np.isnan(adx_s.iloc[-2]) else 0.0
        adx_rising = adx_now > adx_prev

        _, _, hist = ind.macd(closes)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0

        roc9 = float(ind.roc(closes, 9).iloc[-1] or 0.0)
        structure = ind.market_structure(highs, lows, c.bias_structure_left, c.bias_structure_right)
        sflags = ind.structure_flags(highs, lows, c.bias_structure_left, c.bias_structure_right)

        atr_s = ind.atr(df, c.regime_atr_period)
        atr_pctl = float(ind.atr_percentile(atr_s, c.regime_atr_pct_lookback).iloc[-1])
        atr_pctl = 50.0 if np.isnan(atr_pctl) else atr_pctl
        bb_w = ind.bollinger_width(df, 20, 2.0)
        bb_pctl = float(ind.rolling_percentile(bb_w, c.regime_atr_pct_lookback).iloc[-1])
        bb_pctl = 50.0 if np.isnan(bb_pctl) else bb_pctl

        vol_now = float(vols.iloc[-1])
        vol_ma20 = float(vols.iloc[-21:-1].mean()) if len(vols) >= 21 else 0.0
        vol_expansion = vol_ma20 > 0 and vol_now >= c.rg_highvol_vol_mult * vol_ma20
        rng_now = float(highs.iloc[-1] - lows.iloc[-1])
        rng_ma20 = float((highs.iloc[-21:-1] - lows.iloc[-21:-1]).mean()) if len(df) >= 21 else 0.0
        candle_expansion = rng_ma20 > 0 and rng_now >= c.rg_highvol_range_mult * rng_ma20
        bull_vol = float(closes.iloc[-1]) >= float(df["open"].iloc[-1]) and vol_expansion
        bear_vol = float(closes.iloc[-1]) <= float(df["open"].iloc[-1]) and vol_expansion

        # ── Strong Bull Trend: pass >= 4/6 ────────────────────────────────────
        strong_bull_checks = {
            "ema20>ema50":      e20_v > e50_v,
            "close>ema20":      price > e20_v,
            "ema20_slope_up":   slope20 > 0,
            "adx_trending":     adx_now > c.rg_adx_trend_min or adx_rising,
            "hh_hl":            structure == "HH_HL",
            "macd_hist>0":      h_now > 0,
        }
        # ── Strong Bear Trend: mirror ─────────────────────────────────────────
        strong_bear_checks = {
            "ema20<ema50":      e20_v < e50_v,
            "close<ema20":      price < e20_v,
            "ema20_slope_dn":   slope20 < 0,
            "adx_trending":     adx_now > c.rg_adx_trend_min or adx_rising,
            "lh_ll":            structure == "LH_LL",
            "macd_hist<0":      h_now < 0,
        }
        # ── Early Bull: pass >= 4/6 ────────────────────────────────────────────
        early_bull_checks = {
            "reclaim_ema20":    ind.recent_cross_above(closes, e20, lookback=5),
            "ema20_turning_up": slope20 > 0,
            "macd_improving":   h_now > h_prev,
            "roc9>0":           roc9 > 0,
            "higher_low":       sflags["higher_low"],
            "bullish_volume":   bull_vol,
        }
        early_bear_checks = {
            "reclaim_ema20_dn": ind.recent_cross_below(closes, e20, lookback=5),
            "ema20_turning_dn": slope20 < 0,
            "macd_falling":     h_now < h_prev,
            "roc9<0":           roc9 < 0,
            "lower_high":       sflags["lower_high"],
            "bearish_volume":   bear_vol,
        }
        # ── Compression / Range / High Volatility ─────────────────────────────
        compression_checks = {
            "atr_low":  atr_pctl <= c.rg_compression_pctile_max,
            "bbw_low":  bb_pctl <= c.rg_compression_pctile_max,
            "adx_low":  adx_now < c.rg_range_adx_max,
        }
        range_checks = {
            "adx_low":         adx_now < c.rg_range_adx_max,
            "structure_mixed": structure == "MIXED",
            "ema_flat":        abs(slope20) < c.rg_range_flat_slope_pct,
        }
        highvol_checks = {
            "atr_expansion":    atr_pctl >= c.rg_highvol_atr_pctile_min,
            "volume_expansion": vol_expansion,
            "candle_expansion": candle_expansion,
        }

        sb_n = sum(strong_bull_checks.values())
        be_n = sum(strong_bear_checks.values())
        eb_n = sum(early_bull_checks.values())
        er_n = sum(early_bear_checks.values())
        cp_n = sum(compression_checks.values())
        rg_n = sum(range_checks.values())
        hv_n = sum(highvol_checks.values())

        # Priority: volatility expansion overrides trend/range calls (it's an
        # urgent condition, not a steady-state one) -> then strong trend ->
        # early trend -> compression -> range (the default catch-all).
        if hv_n >= 2:
            return TFRegime(HIGH_VOL, highvol_checks, hv_n, 3)
        if sb_n >= 4:
            return TFRegime(STRONG_BULL, strong_bull_checks, sb_n, 6)
        if be_n >= 4:
            return TFRegime(STRONG_BEAR, strong_bear_checks, be_n, 6)
        if eb_n >= 4:
            return TFRegime(EARLY_BULL, early_bull_checks, eb_n, 6)
        if er_n >= 4:
            return TFRegime(EARLY_BEAR, early_bear_checks, er_n, 6)
        if cp_n >= 2:
            return TFRegime(COMPRESSION, compression_checks, cp_n, 3)
        return TFRegime(RANGE, range_checks, rg_n, 3)

    # ── Combine 4H (macro) + 1H (mid) into ONE final classification ──────────
    def analyze(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame | None = None) -> RegimeResult:
        r4 = self._tf_regime(df_4h)
        r1 = self._tf_regime(df_1h) if df_1h is not None and len(df_1h) else r4

        if r4.label == HIGH_VOL or r1.label == HIGH_VOL:
            label, reason = HIGH_VOL, f"volatility expansion (4H={r4.label}, 1H={r1.label})"
        elif r4.label in BULL_LABELS and r1.label in BULL_LABELS:
            label = STRONG_BULL if (r4.label == STRONG_BULL and r1.label == STRONG_BULL) else EARLY_BULL
            reason = f"4H={r4.label} 1H={r1.label} -> {label}"
        elif r4.label in BEAR_LABELS and r1.label in BEAR_LABELS:
            label = STRONG_BEAR if (r4.label == STRONG_BEAR and r1.label == STRONG_BEAR) else EARLY_BEAR
            reason = f"4H={r4.label} 1H={r1.label} -> {label}"
        elif (r4.label in BULL_LABELS and r1.label in BEAR_LABELS) or \
             (r4.label in BEAR_LABELS and r1.label in BULL_LABELS):
            label, reason = RANGE, f"4H/1H directional conflict (4H={r4.label}, 1H={r1.label}) -> RANGE"
        elif r4.label == COMPRESSION or r1.label == COMPRESSION:
            label, reason = COMPRESSION, f"4H={r4.label} 1H={r1.label} -> COMPRESSION"
        else:
            label, reason = RANGE, f"4H={r4.label} 1H={r1.label} -> RANGE"

        # display-only quality score (0-100) — NOT used for gating anymore,
        # kept so status logs / telegram still show something meaningful.
        if label in (STRONG_BULL, STRONG_BEAR):
            score = max(r4.pass_count, r1.pass_count) / 6.0 * 100.0
        elif label in (EARLY_BULL, EARLY_BEAR):
            score = max(r4.pass_count, r1.pass_count) / 6.0 * 70.0
        else:
            score = 40.0

        return RegimeResult(
            label=label, label_4h=r4.label, label_1h=r1.label,
            checks_4h=r4.checks, checks_1h=r1.checks, reason=reason,
            name=label, score=round(score, 1),
        )
