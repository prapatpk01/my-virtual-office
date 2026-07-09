"""
Layer 4 — Entry Engine (30M setup + score + adaptive threshold).

Answers: "Is there a valid 30M trigger right now, and is it strong enough?"

HMA10 x HMA20 cross is NOT an entry — it only RESETS a setup and opens a
3-bar window. Inside that window the entry score (HMA alignment/slope, ROC,
MACD-hist direction, EMA position, candle quality, ATR extension) has to
clear an adaptive threshold. Liquidity / VWAP / BOS deliberately live in the
Context Engine, not here — no duplication.

Score ≥ threshold        -> enter
Score in [floor, thr)    -> near-miss, hand to the Early Booster (Layer 5)
Score < floor            -> cancel / wait
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

LONG  = "LONG"
SHORT = "SHORT"
NONE  = "NONE"


@dataclass
class EntryResult:
    setup_direction: str      # LONG | SHORT | NONE
    setup_age: Optional[int]
    entry_score: float
    entry_threshold: float
    allow_entry: bool
    near_miss: bool
    cancel_setup: bool
    reason: str
    round_id: object = None   # trigger-bar timestamp — one entry per HMA cross cycle
    price: float = 0.0


def _bars_since_sign_flip(series: pd.Series, lookback: int = 20) -> Optional[int]:
    """Bars since `series` last crossed zero. 0 = flipped into the current bar."""
    vals = series.values
    n = len(vals)
    if n < 2:
        return None
    cur = np.sign(vals[-1])
    if cur == 0:
        return None
    for back in range(0, min(lookback, n - 2) + 1):
        newer, older = np.sign(vals[n - 1 - back]), np.sign(vals[n - 2 - back])
        if newer == cur and older != 0 and older != cur:
            return back
    return None


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_30m: pd.DataFrame, bias_direction: str,
                regime_adj: float, context_score: float) -> EntryResult:
        c = self.cfg
        if len(df_30m) < max(c.entry_hma_slow, c.entry_macd_slow) + 10:
            return EntryResult(NONE, None, 0.0, c.entry_base_threshold, False, False, True,
                               "insufficient 30m history")

        closes = df_30m["close"]
        opens = df_30m["open"]
        price = float(closes.iloc[-1])

        hma_fast = ind.hma(closes, c.entry_hma_fast)
        hma_slow = ind.hma(closes, c.entry_hma_slow)
        hf_now = float(hma_fast.iloc[-1]); hs_now = float(hma_slow.iloc[-1])
        hf_prev = float(hma_fast.iloc[-1 - c.entry_hma_slope_lookback])
        hma_slope_up = hf_now > hf_prev
        hma_slope_dn = hf_now < hf_prev

        # ── HMA cross = setup reset trigger; a 3-bar window from the cross ────
        hma_diff = hma_fast - hma_slow
        flip = _bars_since_sign_flip(hma_diff, lookback=20)
        if flip is None:
            return EntryResult(NONE, None, 0.0, c.entry_base_threshold, False, False, True,
                               "no HMA cross in lookback — no open setup")
        setup_age = flip + 1                                   # cross bar is age 1
        setup_dir = LONG if hf_now > hs_now else SHORT
        round_id = df_30m.index[-1 - flip]

        if setup_age > c.entry_setup_window_bars:
            return EntryResult(NONE, setup_age, 0.0, c.entry_base_threshold, False, False, True,
                               f"setup expired (age {setup_age} > {c.entry_setup_window_bars})",
                               round_id=round_id, price=price)

        # setup must agree with the bias side — otherwise it's not our trade
        if (setup_dir == LONG and bias_direction not in ("LONG", "BULL")) or \
           (setup_dir == SHORT and bias_direction not in ("SHORT", "BEAR")):
            return EntryResult(NONE, setup_age, 0.0, c.entry_base_threshold, False, False, False,
                               f"30M setup {setup_dir} disagrees with bias {bias_direction}",
                               round_id=round_id, price=price)

        is_long = setup_dir == LONG

        # ── entry score components (momentum/trigger only) ───────────────────
        roc_v = float(ind.roc(closes, c.entry_roc_period).iloc[-1] or 0.0)
        _, _, hist = ind.macd(closes, c.entry_macd_fast, c.entry_macd_slow, c.entry_macd_signal)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        e15 = float(ind.ema(closes, c.entry_ema_ref).iloc[-1])
        atr_v = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        ext_atr = abs(price - e15) / atr_v if (atr_v and atr_v > 0 and not np.isnan(atr_v)) else 0.0
        candle_up = float(closes.iloc[-1]) > float(opens.iloc[-1])

        comp = {}
        comp["hma_align"] = 25.0 if (hf_now > hs_now if is_long else hf_now < hs_now) else 0.0
        comp["hma_slope"] = 15.0 if (hma_slope_up if is_long else hma_slope_dn) else 0.0
        comp["roc"] = 20.0 if (roc_v > 0 if is_long else roc_v < 0) else 0.0
        comp["macd"] = 20.0 if (h_now > h_prev if is_long else h_now < h_prev) else 0.0
        comp["ema_pos"] = 10.0 if (price > e15 if is_long else price < e15) else 0.0
        comp["candle"] = 10.0 if (candle_up if is_long else not candle_up) else 0.0
        # ATR-extension is a PENALTY slot: award the 0 pts (chasing) only when overextended
        overext = ext_atr > c.entry_max_ext_atr
        comp["atr_ok"] = 0.0 if overext else 0.0   # neutral; overextension handled as hard block below
        entry_score = round(sum(comp.values()), 1)

        # ── adaptive threshold ────────────────────────────────────────────────
        thr = c.entry_base_threshold + regime_adj
        if context_score >= c.entry_context_strong:
            thr += c.entry_context_strong_adj
        elif context_score < c.entry_context_weak:
            thr += c.entry_context_weak_adj
        thr = max(c.entry_threshold_floor, thr)

        # anti-chase hard block even inside a fresh window
        if overext:
            return EntryResult(NONE, setup_age, entry_score, thr, False, False, False,
                               f"overextended {ext_atr:.1f}ATR past EMA15 — no chase",
                               round_id=round_id, price=price)

        if entry_score >= thr:
            return EntryResult(setup_dir, setup_age, entry_score, thr, True, False, False,
                               f"entry {entry_score:.0f} >= threshold {thr:.0f}",
                               round_id=round_id, price=price)
        if entry_score >= c.entry_near_miss_floor:
            return EntryResult(setup_dir, setup_age, entry_score, thr, False, True, False,
                               f"near-miss {entry_score:.0f} in [{c.entry_near_miss_floor:.0f}, {thr:.0f}) "
                               f"— hand to Early Booster",
                               round_id=round_id, price=price)
        return EntryResult(NONE, setup_age, entry_score, thr, False, False, True,
                           f"entry {entry_score:.0f} < floor {c.entry_near_miss_floor:.0f} — wait",
                           round_id=round_id, price=price)
