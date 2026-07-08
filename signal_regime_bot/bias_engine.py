"""
Bias Engine — TF 1H.

Decides which SIDE (long/short) is allowed. This runs after the regime
gate passes and before the entry engine looks for a trigger.
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
class BiasResult:
    bias: str
    bull_score: float
    bear_score: float
    structure: str
    components: dict = field(default_factory=dict)
    confidence: float = 50.0        # 0-100 — HOW trustworthy the bias call is
    conf_components: dict = field(default_factory=dict)


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_1h: pd.DataFrame) -> BiasResult:
        c = self.cfg
        if len(df_1h) < c.bias_ema_slow + 5:
            return BiasResult(BIAS_NEUTRAL, 0.0, 0.0, "MIXED", {})

        closes = df_1h["close"]
        ema_fast_s = ind.ema(closes, c.bias_ema_fast)
        e_fast = float(ema_fast_s.iloc[-1])
        e_slow = float(ind.ema(closes, c.bias_ema_slow).iloc[-1])
        close_v = float(closes.iloc[-1])

        structure = ind.market_structure(df_1h["high"], df_1h["low"],
                                         c.bias_structure_left, c.bias_structure_right)

        roc_s = ind.roc(closes, c.bias_roc_period)
        roc_v = float(roc_s.iloc[-1]) if not np.isnan(roc_s.iloc[-1]) else 0.0

        comps = {}
        comps["ema_bull"] = 40.0 if e_fast > e_slow else 0.0
        comps["ema_bear"] = 40.0 if e_fast < e_slow else 0.0
        comps["structure_bull"] = 30.0 if structure == "HH_HL" else 0.0
        comps["structure_bear"] = 30.0 if structure == "LH_LL" else 0.0
        comps["roc_bull"] = 20.0 if roc_v > 0 else 0.0
        comps["roc_bear"] = 20.0 if roc_v < 0 else 0.0
        comps["close_bull"] = 10.0 if close_v > e_fast else 0.0
        comps["close_bear"] = 10.0 if close_v < e_fast else 0.0

        bull = comps["ema_bull"] + comps["structure_bull"] + comps["roc_bull"] + comps["close_bull"]
        bear = comps["ema_bear"] + comps["structure_bear"] + comps["roc_bear"] + comps["close_bear"]

        if bull >= c.bias_score_min and bull > bear:
            bias = BIAS_BULL
        elif bear >= c.bias_score_min and bear > bull:
            bias = BIAS_BEAR
        else:
            bias = BIAS_NEUTRAL

        confidence, conf_comps = self._confidence(df_1h, bias, ema_fast_s)
        return BiasResult(bias=bias, bull_score=bull, bear_score=bear,
                          structure=structure, components=comps,
                          confidence=confidence, conf_components=conf_comps)

    def _confidence(self, df_1h: pd.DataFrame, bias: str, ema_fast_s) -> tuple[float, dict]:
        """
        Confidence 0-100 — how much to TRUST the bias call (the bias score says
        which side; this says how hard the side is backed):
          ADX strength   (25): 1H ADX >= 25 full, >= 18 partial
          RSI slope      (20): RSI(14) 3-bar slope in the bias direction
          EMA slope      (20): EMA20 5-bar slope in the bias direction
          Volume confirm (15): last volume above its 20-bar average
          Pullback zone  (20): price within 1 ATR of EMA20 (entering at value,
                               not at an extreme far from the mean)
        NEUTRAL bias always gets confidence 0 — there is nothing to trust.
        """
        c = self.cfg
        comps: dict = {"adx": 0.0, "rsi_slope": 0.0, "ema_slope": 0.0,
                       "volume": 0.0, "pullback_zone": 0.0}
        if bias == BIAS_NEUTRAL:
            return 0.0, comps
        is_bull = bias == BIAS_BULL
        closes = df_1h["close"]

        adx_s, _, _ = ind.adx(df_1h, 14)
        adx_v = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0.0
        if adx_v >= c.bias_conf_adx_strong:
            comps["adx"] = 25.0
        elif adx_v >= c.bias_conf_adx_ok:
            comps["adx"] = 15.0

        rsi_s = ind.rsi(closes, 14)
        if len(rsi_s) >= 4:
            rsi_slope = float(rsi_s.iloc[-1]) - float(rsi_s.iloc[-4])
            if (rsi_slope > 0) if is_bull else (rsi_slope < 0):
                comps["rsi_slope"] = 20.0

        slope_s = ind.slope_pct(ema_fast_s, 5)
        slope_v = float(slope_s.iloc[-1]) if not np.isnan(slope_s.iloc[-1]) else 0.0
        if (slope_v > 0) if is_bull else (slope_v < 0):
            comps["ema_slope"] = 20.0

        vol = float(df_1h["volume"].iloc[-1])
        vol_ma = float(df_1h["volume"].iloc[-21:-1].mean()) if len(df_1h) >= 21 else 0.0
        if vol_ma > 0 and vol > vol_ma:
            comps["volume"] = 15.0

        atr_v = float(ind.atr(df_1h, 14).iloc[-1])
        e20_v = float(ema_fast_s.iloc[-1])
        price = float(closes.iloc[-1])
        if atr_v > 0 and not np.isnan(atr_v) and abs(price - e20_v) <= atr_v:
            comps["pullback_zone"] = 20.0

        return sum(comps.values()), comps
