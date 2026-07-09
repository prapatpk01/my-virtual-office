"""
Layer 5 — Early Entry Booster (15M).

Improves TIMING on a valid 30M near-miss — it can never open a trade on its
own. It only runs when the 30M Entry Engine returned near_miss (score in
[floor, threshold)) and regime/bias/context all still pass.

15M EXECUTION features only — engulf, EMA bounce, VWAP reclaim, break of
the previous 15M extreme, liquidity-sweep confirmation, volume spike,
rejection candle, successful retest. It deliberately does NOT re-use HMA /
ROC / MACD — those already scored on 30M.

early_score 0-20  ->  bonus = min(early_score * 0.5, max_bonus)
final_score = entry_score + bonus.  Enter only if final >= entry_threshold.
Also emits a cancel flag if the 30M setup reversed while we waited.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
import price_action as pa
from config import Config

LONG = "LONG"
SHORT = "SHORT"


@dataclass
class BoosterResult:
    early_score: float
    early_bonus: float
    final_score: float
    max_bonus: float
    allow_early_entry: bool
    cancel_setup: bool
    components: dict
    reason: str


class EarlyBooster:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _max_bonus(self, regime_quality: str) -> float:
        c = self.cfg
        return {
            "strong": c.booster_max_bonus_strong,
            "normal": c.booster_max_bonus_normal,
            "weak": c.booster_max_bonus_weak,
            "transition": c.booster_max_bonus_transition,
        }.get(regime_quality, c.booster_max_bonus_weak)

    def _reversed_on_30m(self, df_30m: pd.DataFrame, side: str) -> bool:
        """Cancel the setup if the 30M turned against us while we waited on 15M."""
        c = self.cfg
        closes = df_30m["close"]
        hf = ind.hma(closes, c.entry_hma_fast)
        hs = ind.hma(closes, c.entry_hma_slow)
        hf_now, hs_now = float(hf.iloc[-1]), float(hs.iloc[-1])
        # HMA crossed back the other way
        hma_against = (hf_now < hs_now) if side == LONG else (hf_now > hs_now)
        # ROC flipped against
        roc_v = float(ind.roc(closes, c.entry_roc_period).iloc[-1] or 0.0)
        roc_against = (roc_v < 0) if side == LONG else (roc_v > 0)
        # MACD hist flipped strongly against
        _, _, hist = ind.macd(closes, c.entry_macd_fast, c.entry_macd_slow, c.entry_macd_signal)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        macd_against = (h_now < 0) if side == LONG else (h_now > 0)
        return hma_against or (roc_against and macd_against)

    def analyze(self, df_15m: pd.DataFrame, df_30m: pd.DataFrame, side: str,
                entry_score: float, entry_threshold: float,
                regime_quality: str) -> BoosterResult:
        c = self.cfg
        max_bonus = self._max_bonus(regime_quality)

        # cancel first — a reversed 30M kills the setup outright
        if self._reversed_on_30m(df_30m, side):
            return BoosterResult(0.0, 0.0, entry_score, max_bonus, False, True, {},
                                 "30M reversed while waiting — cancel setup")

        if df_15m is None or len(df_15m) < 30:
            return BoosterResult(0.0, 0.0, entry_score, max_bonus, False, False, {},
                                 "no 15m data — booster cannot confirm")

        # ── 15M execution confirmations (each 0..~4 pts toward early_score) ──
        comp: dict = {}
        engulf = pa.bull_engulf(df_15m) if side == LONG else pa.bear_engulf(df_15m)
        comp["engulf"] = 4.0 if engulf else 0.0
        comp["ema_bounce"] = 3.0 if pa.ema_bounce(df_15m, side, c.entry_ema_ref) else 0.0
        comp["vwap"] = 3.0 if pa.vwap_reclaim(df_15m, side) else 0.0
        comp["break_prev"] = 3.0 if pa.break_prev_extreme(df_15m, side) else 0.0
        comp["sweep"] = 3.0 if pa.liquidity_sweep(df_15m, side, c.entry_sweep_lookback) else 0.0
        comp["vol_spike"] = 2.0 if pa.volume_spike(df_15m, c.spike_vol_mult) else 0.0
        comp["rejection"] = 2.0 if pa.rejection_candle(df_15m, side, c.entry_wick_reject_frac) else 0.0

        early_score = round(min(20.0, sum(comp.values())), 1)
        bonus = round(min(early_score * c.booster_score_to_bonus, max_bonus), 1)
        final = round(entry_score + bonus, 1)
        allow = final >= entry_threshold

        present = [k for k, v in comp.items() if v > 0]
        reason = (f"early {early_score:.0f} -> +{bonus:.0f} -> final {final:.0f} "
                  f"{'>=' if allow else '<'} thr {entry_threshold:.0f} [{', '.join(present) or 'none'}]")
        return BoosterResult(early_score, bonus, final, max_bonus, allow, False, comp, reason)
