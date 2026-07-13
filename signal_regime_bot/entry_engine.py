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

Needs >= entry_min_categories (3/5) categories, with Momentum AND Structure
mandatory regardless of the total count.

Two additional HARD gates, config-toggled:
    entry_adx_gate_enabled     30M ADX must clear entry_adx_min OR be rising
                               (mirrors Regime's own "adx_trending" check) —
                               confirms the move has real directional force
                               behind it, not just a low-conviction chop
                               bounce. Enabled, threshold 18.
    entry_participation_mandatory  Participation (volume) required alongside
                               Momentum + Structure, not just optional.
                               Measured worse in testing — stays off.

Setup freshness window: the category check alone has no memory of WHEN
momentum first turned, so it can fire many bars after the actual shift —
by then price has already run and the "entry" is really a chase. A setup
opens on whichever of {ROC>0, MACD favorable, HMA cross} turns true FIRST
in the trade direction; only within `entry_setup_window_bars` bars of that
first trigger is an entry allowed. If full confirmation (>=4/5 categories)
hasn't arrived by the window's end, the setup goes stale and is skipped —
the bot waits for a fresh trigger rather than chasing an old one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
    adx: float = 0.0
    setup_age: Optional[int] = None    # bars since the earliest active momentum trigger
    # categories + mandatory + ADX all pass, ignoring the freshness window —
    # used by the pipeline's acceleration wait-rounds: a setup already in a
    # confirmation round must not be killed as "stale" mid-round (the round
    # machine is the timing authority there), but its core signal must hold.
    core_ok: bool = False


def _consecutive_true_run(values: np.ndarray, cap: int = 20) -> Optional[int]:
    """Age (0 = just turned true this bar) of the current True run in a bool
    array, or None if the last value is False. Capped at `cap` bars back."""
    n = len(values)
    if n == 0 or not values[-1]:
        return None
    count = 0
    for i in range(n - 1, max(-1, n - 1 - cap), -1):
        if values[i]:
            count += 1
        else:
            break
    return count - 1


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

        adx_s, _, _ = ind.adx(df_30m, c.regime_adx_period)
        adx_now = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0.0
        adx_prev = float(adx_s.iloc[-2]) if not np.isnan(adx_s.iloc[-2]) else 0.0
        adx_rising = adx_now > adx_prev

        # ── setup freshness: age of the EARLIEST currently-active momentum
        # trigger among {ROC>0, MACD favorable, HMA cross}, in the trade
        # direction. Whichever fired first sets the window's start.
        roc_v = roc9.values
        hist_v = hist.values
        hma_diff_v = (hma_f - hma_s).values
        if is_long:
            roc_fav = roc_v > 0
            macd_fav = hist_v > 0
            hma_fav = hma_diff_v > 0
        else:
            roc_fav = roc_v < 0
            macd_fav = hist_v < 0
            hma_fav = hma_diff_v < 0
        roc_fav = np.nan_to_num(roc_fav.astype(float), nan=0.0).astype(bool)
        macd_fav = np.nan_to_num(macd_fav.astype(float), nan=0.0).astype(bool)
        hma_fav = np.nan_to_num(hma_fav.astype(float), nan=0.0).astype(bool)

        ages = [a for a in (
            _consecutive_true_run(roc_fav, c.entry_setup_window_bars + 15),
            _consecutive_true_run(macd_fav, c.entry_setup_window_bars + 15),
            _consecutive_true_run(hma_fav, c.entry_setup_window_bars + 15),
        ) if a is not None]
        setup_age = max(ages) if ages else None
        in_window = setup_age is not None and setup_age < c.entry_setup_window_bars

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
        mandatory_cats = ["momentum", "structure"]
        if c.entry_participation_mandatory:
            mandatory_cats.append("participation")
        mandatory_ok = all(categories[k] for k in mandatory_cats)
        adx_ok = (not c.entry_adx_gate_enabled) or adx_now >= c.entry_adx_min or adx_rising
        core_ok = passed >= c.entry_min_categories and mandatory_ok and adx_ok
        allow = core_ok and in_window
        score = round(passed / 5.0 * 100.0, 1)

        if allow:
            reason = (f"{direction} trigger: {passed}/5 categories ({', '.join(k for k, v in categories.items() if v)}) "
                      f"ADX={adx_now:.0f} setup_age={setup_age}")
        elif setup_age is None:
            reason = f"{direction} trigger not ready: no active momentum trigger (ROC/MACD/HMA)"
        elif not in_window:
            reason = (f"{direction} trigger stale: momentum fired {setup_age} bars ago "
                      f">= window {c.entry_setup_window_bars} — skip, wait for a fresh trigger")
        elif not mandatory_ok:
            missing = [k for k in mandatory_cats if not categories[k]]
            reason = f"{direction} trigger blocked: mandatory {'/'.join(missing)} not met ({passed}/5 passed, setup_age={setup_age})"
        elif not adx_ok:
            reason = (f"{direction} trigger blocked: ADX {adx_now:.0f} < {c.entry_adx_min:.0f} and not rising "
                      f"({passed}/5 categories passed)")
        else:
            reason = f"{direction} trigger not ready: {passed}/5 categories (need >= {c.entry_min_categories}, setup_age={setup_age})"

        return EntryResult(direction if allow else NONE, allow, passed, categories, reason,
                           price=price, entry_score=score, adx=adx_now, setup_age=setup_age,
                           core_ok=core_ok)
