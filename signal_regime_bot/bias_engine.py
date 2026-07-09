"""
Layer 2 — Bias Engine (SOFT confirmation + minimum gate).  1H + 15M.

Answers: "Which side has the better probability right now?"

Deliberately momentum-focused, NOT trend-focused — trend/structure is the
Regime layer's job and re-scoring it here would just double-count. This
layer reads ROC, MACD histogram, RSI, momentum slope, volume, and EMA
pullback: the shorter-horizon push behind the regime.

Weighted, not strict-AND: weighted = score_1h*0.7 + score_15m*0.3. The
only HARD parts are the two vetoes — a strongly-opposite 1H or 15M forces
NEUTRAL regardless of the weighted number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

BIAS_BULL    = "BULL"
BIAS_BEAR    = "BEAR"
BIAS_NEUTRAL = "NEUTRAL"


@dataclass
class TFBias:
    bull: float
    bear: float
    components: dict = field(default_factory=dict)


@dataclass
class BiasResult:
    direction: str            # LONG | SHORT | NEUTRAL  (kept as .bias too for compat)
    score_1h: float
    score_15m: float
    weighted_score: float
    aligned: bool
    allow_entry: bool
    reason: str
    # backward-compat fields (health monitor / status log read these)
    bias: str = BIAS_NEUTRAL
    bull_score: float = 0.0
    bear_score: float = 0.0
    confidence: float = 0.0
    structure: str = "MIXED"
    components: dict = field(default_factory=dict)
    conf_components: dict = field(default_factory=dict)


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _tf_bias(self, df: pd.DataFrame) -> TFBias:
        """Momentum-only bull/bear score 0-100 for one timeframe."""
        c = self.cfg
        if len(df) < max(c.bias_ema_slow, 30):
            return TFBias(0.0, 0.0)

        closes = df["close"]
        comp_bull, comp_bear = {}, {}

        # ROC direction (25)
        roc_v = float(ind.roc(closes, c.bias_roc_period).iloc[-1] or 0.0)
        comp_bull["roc"] = 25.0 if roc_v > 0 else 0.0
        comp_bear["roc"] = 25.0 if roc_v < 0 else 0.0

        # MACD histogram direction (25) — rising vs falling
        _, _, hist = ind.macd(closes)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        comp_bull["macd"] = 25.0 if h_now > h_prev else 0.0
        comp_bear["macd"] = 25.0 if h_now < h_prev else 0.0

        # RSI regime (20) — above/below 50, not overbought/oversold extremes
        rsi_v = float(ind.rsi(closes, 14).iloc[-1])
        comp_bull["rsi"] = 20.0 if rsi_v >= 50 else 0.0
        comp_bear["rsi"] = 20.0 if rsi_v < 50 else 0.0

        # momentum slope (15) — EMA9 slope
        e9 = ind.ema(closes, 9)
        slope_v = float(ind.slope_pct(e9, 3).iloc[-1] or 0.0)
        comp_bull["mom_slope"] = 15.0 if slope_v > 0 else 0.0
        comp_bear["mom_slope"] = 15.0 if slope_v < 0 else 0.0

        # volume confirmation (15) — expansion credited to the bar's direction
        o = float(df["open"].iloc[-1]); cl = float(closes.iloc[-1])
        vol = float(df["volume"].iloc[-1])
        vol_ma = float(df["volume"].iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
        vol_up = vol_ma > 0 and vol > vol_ma
        comp_bull["volume"] = 15.0 if (vol_up and cl >= o) else 0.0
        comp_bear["volume"] = 15.0 if (vol_up and cl <= o) else 0.0

        return TFBias(sum(comp_bull.values()), sum(comp_bear.values()),
                      {"bull": comp_bull, "bear": comp_bear})

    def analyze(self, df_1h: pd.DataFrame, df_15m: pd.DataFrame | None = None) -> BiasResult:
        c = self.cfg
        b1 = self._tf_bias(df_1h)
        # 15M optional — falls back to 1H if the caller has no 15M frame
        b15 = self._tf_bias(df_15m) if df_15m is not None and len(df_15m) else b1

        # net directional score per TF (bull minus bear, floored at 0 per side)
        s1_bull, s1_bear = b1.bull, b1.bear
        s15_bull, s15_bear = b15.bull, b15.bear

        w1, w15 = c.bias_weight_1h, c.bias_weight_15m
        weighted_bull = s1_bull * w1 + s15_bull * w15
        weighted_bear = s1_bear * w1 + s15_bear * w15

        # single reported per-TF score = the dominant side of that TF
        score_1h = max(s1_bull, s1_bear)
        score_15m = max(s15_bull, s15_bear)

        # strong-opposite HARD vetoes
        veto_long = s1_bear >= c.bias_strong_opposite or s15_bear >= c.bias_strong_opposite
        veto_short = s1_bull >= c.bias_strong_opposite or s15_bull >= c.bias_strong_opposite

        direction = BIAS_NEUTRAL
        weighted_score = 0.0
        reason = "no side cleared the bias threshold"
        allow = False

        if weighted_bull >= c.bias_min_threshold and weighted_bull > weighted_bear and not veto_long:
            direction, weighted_score, allow = BIAS_BULL, weighted_bull, True
            reason = f"weighted bull {weighted_bull:.1f} >= {c.bias_min_threshold:.0f}"
        elif weighted_bear >= c.bias_min_threshold and weighted_bear > weighted_bull and not veto_short:
            direction, weighted_score, allow = BIAS_BEAR, weighted_bear, True
            reason = f"weighted bear {weighted_bear:.1f} >= {c.bias_min_threshold:.0f}"
        else:
            weighted_score = max(weighted_bull, weighted_bear)
            if veto_long or veto_short:
                reason = "strong opposite bias on 1H/15M — NEUTRAL"

        # aligned = both TFs lean the same way on the chosen side
        if direction == BIAS_BULL:
            aligned = s1_bull >= s1_bear and s15_bull >= s15_bear
        elif direction == BIAS_BEAR:
            aligned = s1_bear >= s1_bull and s15_bear >= s15_bull
        else:
            aligned = False

        # confidence (compat): how backed the chosen side is, 0-100
        confidence = min(100.0, weighted_score) if direction != BIAS_NEUTRAL else 0.0

        return BiasResult(
            direction=direction, score_1h=score_1h, score_15m=score_15m,
            weighted_score=round(weighted_score, 1), aligned=aligned, allow_entry=allow,
            reason=reason, bias=direction,
            bull_score=weighted_bull, bear_score=weighted_bear,
            confidence=confidence, structure="—",
            components={"1h": b1.components, "15m": b15.components},
        )
