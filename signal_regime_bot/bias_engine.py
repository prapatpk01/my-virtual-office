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


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_1h: pd.DataFrame) -> BiasResult:
        c = self.cfg
        if len(df_1h) < c.bias_ema_slow + 5:
            return BiasResult(BIAS_NEUTRAL, 0.0, 0.0, "MIXED", {})

        closes = df_1h["close"]
        e_fast = float(ind.ema(closes, c.bias_ema_fast).iloc[-1])
        e_slow = float(ind.ema(closes, c.bias_ema_slow).iloc[-1])
        close_v = float(closes.iloc[-1])

        structure = ind.market_structure(df_1h["high"], df_1h["low"],
                                         c.bias_structure_left, c.bias_structure_right)

        roc_s = ind.roc(closes, c.bias_roc_period)
        roc_v = float(roc_s.iloc[-1]) if not np.isnan(roc_s.iloc[-1]) else 0.0

        bull = 0.0
        bear = 0.0
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

        return BiasResult(bias=bias, bull_score=bull, bear_score=bear,
                          structure=structure, components=comps)
