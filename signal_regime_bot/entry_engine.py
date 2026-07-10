"""
Layer 3 — Entry Engine (TIMING ONLY).  30M.

Answers: "Has the moment to enter arrived?" — nothing else. This layer has
NO right to pick Long or Short: it receives a fixed `direction` from the
Bias layer and may ONLY search for a trigger matching that direction. Even
if the opposite trigger fires (e.g. a MACD Cross Down while direction is
LONG), it must be ignored — the correct response is to keep waiting, never
to open the opposite side.

Five trigger categories, each true/false:
    Momentum      MACD cross / histogram improving / ROC9 direction
    Trend         HMA10 vs HMA20 / Close vs EMA20
    Structure     BOS / CHOCH / Higher-Low (Lower-High for shorts)
    Liquidity     Sweep / rejection wick
    Participation Volume expansion / elevated relative volume

Needs >= 4/5 categories, with Momentum AND Structure mandatory regardless
of the total count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
import price_action as pa
from config import Config

LONG  = "LONG"
SHORT = "SHORT"
NONE  = "NONE"


@dataclass
class EntryResult:
    direction: str             # LONG | SHORT | NONE  (echoes the fixed input direction, or NONE)
    allow_entry: bool
    passed_count: int
    categories: dict = field(default_factory=dict)   # category -> bool
    reason: str = ""
    price: float = 0.0
    entry_score: float = 0.0   # passed_count/5 * 100, for logging/telegram compat


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_30m: pd.DataFrame, direction: str) -> EntryResult:
        c = self.cfg
        if direction not in (LONG, SHORT):
            return EntryResult(NONE, False, 0, {}, "no direction from Bias layer — nothing to time")
        if len(df_30m) < max(c.entry_hma_slow, c.entry_macd_slow) + 10:
            return EntryResult(NONE, False, 0, {}, "insufficient 30m history")

        closes, opens = df_30m["close"], df_30m["open"]
        price = float(closes.iloc[-1])
        is_long = direction == LONG

        hma_f = ind.hma(closes, c.entry_hma_fast)
        hma_s = ind.hma(closes, c.entry_hma_slow)
        e20 = ind.ema(closes, c.entry_ema_ref)
        roc9 = ind.roc(closes, c.entry_roc_period)
        _, _, hist = ind.macd(closes, c.entry_macd_fast, c.entry_macd_slow, c.entry_macd_signal)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        macd_cross_up = h_prev <= 0 and h_now > 0
        macd_cross_dn = h_prev >= 0 and h_now < 0

        sflags = ind.structure_flags(df_30m["high"], df_30m["low"],
                                     c.bias_structure_left, c.bias_structure_right)
        side = "LONG" if is_long else "SHORT"
        bos, choch = pa.bos_choch(df_30m, side, c.bias_structure_left, c.bias_structure_right)

        vol_now = float(df_30m["volume"].iloc[-1])
        vol_ma20 = float(df_30m["volume"].iloc[-21:-1].mean()) if len(df_30m) >= 21 else 0.0
        rel_vol = (vol_now / vol_ma20) if vol_ma20 > 0 else 1.0

        if is_long:
            categories = {
                "momentum":      macd_cross_up or (h_now > h_prev) or (float(roc9.iloc[-1] or 0.0) > 0),
                "trend":         (float(hma_f.iloc[-1]) > float(hma_s.iloc[-1])) or
                                  (price > float(e20.iloc[-1])),
                "structure":     bos or choch or sflags["higher_low"],
                "liquidity":     pa.liquidity_sweep(df_30m, "LONG", c.entry_sweep_lookback) or
                                  pa.rejection_candle(df_30m, "LONG", c.entry_wick_reject_frac),
                "participation": pa.volume_expansion(df_30m, c.entry_vol_expansion_mult) or
                                  rel_vol >= c.entry_rel_vol_min,
            }
        else:
            categories = {
                "momentum":      macd_cross_dn or (h_now < h_prev) or (float(roc9.iloc[-1] or 0.0) < 0),
                "trend":         (float(hma_f.iloc[-1]) < float(hma_s.iloc[-1])) or
                                  (price < float(e20.iloc[-1])),
                "structure":     bos or choch or sflags["lower_high"],
                "liquidity":     pa.liquidity_sweep(df_30m, "SHORT", c.entry_sweep_lookback) or
                                  pa.rejection_candle(df_30m, "SHORT", c.entry_wick_reject_frac),
                "participation": pa.volume_expansion(df_30m, c.entry_vol_expansion_mult) or
                                  rel_vol >= c.entry_rel_vol_min,
            }

        passed = sum(categories.values())
        mandatory_ok = categories["momentum"] and categories["structure"]
        allow = passed >= c.entry_min_categories and mandatory_ok
        score = round(passed / 5.0 * 100.0, 1)

        if allow:
            reason = f"{direction} trigger: {passed}/5 categories ({', '.join(k for k, v in categories.items() if v)})"
        elif not mandatory_ok:
            missing = [k for k in ("momentum", "structure") if not categories[k]]
            reason = f"{direction} trigger blocked: mandatory {'/'.join(missing)} not met ({passed}/5 passed)"
        else:
            reason = f"{direction} trigger not ready: {passed}/5 categories (need >= {c.entry_min_categories})"

        return EntryResult(direction if allow else NONE, allow, passed, categories, reason,
                           price=price, entry_score=score)
