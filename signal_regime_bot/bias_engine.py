"""
Layer 2 — MTF Bias Engine (DIRECTION, Dynamic Combined Bias Score).
1H + 15M + 5M.

Answers: "Now that we know the Regime, which side — if any — is actually
allowed to trade?" This layer picks the Trading Direction and nothing else:
it never scores an entry trigger (that's Layer 3) and once it decides, no
downstream layer may override it.

Each timeframe gets ONE 0-100 "bull-lean" score from 9 equally-structured
components (EMA Position, EMA Alignment, MACD Direction, ROC Direction, RSI
Zone, VWAP Position, Volume, Relative Volume, Market Structure) — 100 =
every component bullish, 0 = every component bearish, ~50 = mixed.

Combined Bias = 1H*w1h + 15M*w15m + 5M*w5m — weights and the pass bar are
DYNAMIC, keyed off the Regime tier:
    Confirmed Trend (STRONG_BULL/BEAR_TREND): 1H=50% 15M=35% 5M=15%, pass 65
      -> 1H-heavy: an established trend wants continuity, not noise.
    Early Trend (EARLY_BULL/BEAR_TREND):      1H=35% 15M=45% 5M=20%, pass 60
      -> faster TFs weighted up so an early trend gets caught sooner.

The weighted average alone can't pass — every TF must ALSO individually
clear a floor and not be flagged the opposite direction:
    LONG_BIAS_PASS  = bull regime AND Combined Bull Bias >= threshold
                       AND 1H >= 55 AND 15M >= 55 AND 5M >= 40
                       AND 1H Direction != BEAR AND 15M Direction != BEAR
    SHORT_BIAS_PASS = mirror (bear regime, Combined <= 100-threshold,
                       1H <= 45, 15M <= 45, 5M <= 60, no BULL-flagged TF)
    NO TRADE        <- anything else (regime/bias mismatch, TF conflict,
                       Range, Compression, High Volatility, below threshold)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from regime_engine import (BULL_LABELS, BEAR_LABELS, STRONG_BULL, STRONG_BEAR,
                           EARLY_BULL, EARLY_BEAR)

BIAS_BULL    = "BULL"
BIAS_BEAR    = "BEAR"
BIAS_NEUTRAL = "NEUTRAL"

LONG    = "LONG"
SHORT   = "SHORT"
NEUTRAL = "NEUTRAL"


@dataclass
class TFBiasScore:
    score: float                          # 0-100 bull-lean
    components: dict = field(default_factory=dict)


@dataclass
class BiasResult:
    direction: str            # LONG | SHORT | NEUTRAL  (Trading Direction)
    score_1h: float
    score_15m: float
    score_5m: float
    reason: str
    components: dict = field(default_factory=dict)
    # backward-compat fields (health monitor / status log / telegram read these)
    bias: str = BIAS_NEUTRAL
    bull_score: float = 0.0
    bear_score: float = 0.0
    confidence: float = 0.0
    weighted_score: float = 0.0
    aligned: bool = False
    allow_entry: bool = False
    structure: str = "—"


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── One timeframe -> single 0-100 bull-lean score, 9 components ──────────
    def _tf_score(self, df: pd.DataFrame) -> TFBiasScore:
        c = self.cfg
        if len(df) < max(c.bias_ema_slow, 30):
            return TFBiasScore(50.0)

        closes, highs, lows, vols = df["close"], df["high"], df["low"], df["volume"]
        price = float(closes.iloc[-1])
        comp = {}

        e20 = ind.ema(closes, c.bias_ema_fast)
        e50 = ind.ema(closes, c.bias_ema_slow)
        e20_v, e50_v = float(e20.iloc[-1]), float(e50.iloc[-1])
        comp["ema_position"] = 10.0 if price > e20_v else 0.0
        comp["ema_alignment"] = 15.0 if e20_v > e50_v else 0.0

        _, _, hist = ind.macd(closes)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        comp["macd_direction"] = 15.0 if h_now > h_prev else 0.0

        roc_v = float(ind.roc(closes, c.bias_roc_period).iloc[-1] or 0.0)
        comp["roc_direction"] = 10.0 if roc_v > 0 else 0.0

        rsi_v = float(ind.rsi(closes, 14).iloc[-1])
        comp["rsi_zone"] = 10.0 if rsi_v >= 50 else 0.0

        vwap_s = ind.vwap(df, window=min(48, len(df) - 1))
        vwap_v = float(vwap_s.iloc[-1])
        comp["vwap_position"] = 10.0 if (not np.isnan(vwap_v) and price > vwap_v) else 0.0

        o = float(df["open"].iloc[-1])
        vol_now = float(vols.iloc[-1])
        vol_ma20 = float(vols.iloc[-21:-1].mean()) if len(vols) >= 21 else 0.0
        vol_up_dir = vol_ma20 > 0 and vol_now > vol_ma20 and price >= o
        comp["volume"] = 10.0 if vol_up_dir else 0.0

        rel_vol = (vol_now / vol_ma20) if vol_ma20 > 0 else 1.0
        comp["relative_volume"] = 10.0 if rel_vol >= c.bias_rel_vol_min else 0.0

        sflags = ind.structure_flags(highs, lows, c.bias_structure_left, c.bias_structure_right)
        comp["market_structure"] = 10.0 if sflags["higher_high"] or sflags["higher_low"] else 0.0

        return TFBiasScore(round(sum(comp.values()), 1), comp)

    def _tf_direction(self, score: float) -> str:
        c = self.cfg
        if score >= c.bias_direction_bull_min:
            return BIAS_BULL
        if score <= c.bias_direction_bear_max:
            return BIAS_BEAR
        return BIAS_NEUTRAL

    def _weight_profile(self, regime_label: str) -> tuple[float, float, float, float]:
        """(w1h, w15m, w5m, long_pass_threshold) for the regime's trend tier."""
        c = self.cfg
        if regime_label in (STRONG_BULL, STRONG_BEAR):
            return c.bias_w1h_confirmed, c.bias_w15m_confirmed, c.bias_w5m_confirmed, c.bias_threshold_confirmed
        if regime_label in (EARLY_BULL, EARLY_BEAR):
            return c.bias_w1h_early, c.bias_w15m_early, c.bias_w5m_early, c.bias_threshold_early
        return c.bias_w1h_default, c.bias_w15m_default, c.bias_w5m_default, c.bias_threshold_default

    def analyze(self, df_1h: pd.DataFrame, df_15m: pd.DataFrame | None = None,
               df_5m: pd.DataFrame | None = None, regime_label: str = "") -> BiasResult:
        c = self.cfg
        s1h = self._tf_score(df_1h)
        # 15M/5M optional in callers with only 1H available (legacy health-check
        # call sites) — fall back to the 1H score so those degrade safely to
        # "same as 1H" rather than crashing, but LIVE/backtest entries always
        # pass real 15M/5M frames since the combined score needs them for real.
        s15 = self._tf_score(df_15m) if df_15m is not None and len(df_15m) else s1h
        s5 = self._tf_score(df_5m) if df_5m is not None and len(df_5m) else s15

        bull_regime = regime_label in BULL_LABELS
        bear_regime = regime_label in BEAR_LABELS
        w1h, w15m, w5m, long_thr = self._weight_profile(regime_label)
        short_thr = 100.0 - long_thr
        combined = s1h.score * w1h + s15.score * w15m + s5.score * w5m

        dir_1h, dir_15m = self._tf_direction(s1h.score), self._tf_direction(s15.score)

        long_ok = (bull_regime and combined >= long_thr
                  and s1h.score >= c.bias_tf_floor_1h and s15.score >= c.bias_tf_floor_15m
                  and s5.score >= c.bias_tf_floor_5m
                  and dir_1h != BIAS_BEAR and dir_15m != BIAS_BEAR)
        short_ok = (bear_regime and combined <= short_thr
                   and s1h.score <= 100.0 - c.bias_tf_floor_1h and s15.score <= 100.0 - c.bias_tf_floor_15m
                   and s5.score <= 100.0 - c.bias_tf_floor_5m
                   and dir_1h != BIAS_BULL and dir_15m != BIAS_BULL)

        tag = f"regime={regime_label} combined={combined:.1f} (w1h={w1h:.2f} w15m={w15m:.2f} w5m={w5m:.2f}) 1H={s1h.score:.0f} 15M={s15.score:.0f} 5M={s5.score:.0f}"
        if long_ok:
            direction = LONG
            reason = f"LONG ONLY: {tag} combined>={long_thr:.0f} + all TF floors clear"
        elif short_ok:
            direction = SHORT
            reason = f"SHORT ONLY: {tag} combined<={short_thr:.0f} + all TF floors clear"
        else:
            direction = NEUTRAL
            if not (bull_regime or bear_regime):
                reason = f"NO TRADE: regime={regime_label} is not a bull/bear trend"
            elif bull_regime:
                reason = f"NO TRADE: bull regime but LONG_BIAS_PASS failed — {tag} (need >={long_thr:.0f})"
            else:
                reason = f"NO TRADE: bear regime but SHORT_BIAS_PASS failed — {tag} (need <={short_thr:.0f})"

        bias_str = BIAS_BULL if direction == LONG else BIAS_BEAR if direction == SHORT else BIAS_NEUTRAL

        return BiasResult(
            direction=direction, score_1h=s1h.score, score_15m=s15.score, score_5m=s5.score,
            reason=reason, components={"1h": s1h.components, "15m": s15.components, "5m": s5.components},
            bias=bias_str, bull_score=round(combined, 1), bear_score=round(100.0 - combined, 1),
            confidence=round(abs(combined - 50.0) * 2.0, 1), weighted_score=round(combined, 1),
            aligned=long_ok or short_ok, allow_entry=long_ok or short_ok,
        )
