"""
SpikeGuard — fast reversal-spike / V-sharp protection engine.

Purpose (from the live BTC incident, Jul 9): a short at 62,083 was hit by a
~400-point V-reversal green candle in <15 minutes. The 30m health monitor
(closed-bar + 3-bar confirm) is structurally too slow for this — by design it
reacts in hours, not minutes. SpikeGuard is the fast layer: it runs EVERY
poll tick (~30s) against CLOSED 5m and 15m bars plus the live ticker price,
and force-closes the position before the full SL (plus slippage — the
incident filled 212 points PAST the stop) is eaten.

Detection (all against the position's direction):
  1. 5m single-bar spike: last closed 5m bar range >= spike_5m_atr_mult x
     ATR14(5m), closing hard against us (close in the extreme
     spike_close_frac of its range).
  2. 15m single-bar spike: same idea, spike_15m_atr_mult x ATR14(15m).
  3. 15m V-move: cumulative 3-bar move against us >= spike_15m_cum_atr_mult
     x ATR14(15m).
  4. Live-price acceleration: ticker has already moved >= spike_live_atr_mult
     x ATR14(5m) against us beyond the last closed 5m close (the spike is
     happening RIGHT NOW, mid-bar).

Arming rule: a detected spike only fires CLOSE when the position is actually
threatened — adverse excursion >= spike_min_adverse_r of 1R — OR the spike is
huge (>= spike_hard_atr_mult x ATR). A volume blowout (>= spike_vol_mult x
20-bar avg) softens the ATR bars by 20% (a spike ON volume is more real).

Pure function — no exchange calls — so a backtest can replay it bar-by-bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

HOLD = "HOLD"
CLOSE = "CLOSE"


@dataclass
class SpikeResult:
    action: str          # HOLD | CLOSE
    reason: str
    adverse_r: float = 0.0


def _last_closed_bar_stats(df: pd.DataFrame, atr_period: int = 14):
    """(bar_range, direction(+1 up/-1 down), close_frac, atr, vol_ratio) of the last closed bar."""
    o = float(df["open"].iloc[-1]);  h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1]);   c = float(df["close"].iloc[-1])
    rng = max(h - l, 1e-12)
    direction = 1 if c >= o else -1
    # close_frac: how near the close sits to the bar's extreme in the bar's
    # own direction (1.0 = closed exactly at the extreme = full momentum)
    close_frac = (c - l) / rng if direction > 0 else (h - c) / rng
    atr_s = ind.atr(df, atr_period)
    atr_v = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0
    vol = float(df["volume"].iloc[-1])
    vol_ma = float(df["volume"].iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
    vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0
    return rng, direction, close_frac, atr_v, vol_ratio


def check_spike(side: str, entry_price: float, one_r: float,
                df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                current_price: float, cfg: Config) -> SpikeResult:
    """
    side: 'long' | 'short'. Returns CLOSE when a reversal spike against the
    position is detected AND the arming rule passes.
    """
    if not cfg.spike_guard_enabled or one_r <= 0 or current_price <= 0:
        return SpikeResult(HOLD, "disabled/invalid")
    if len(df_5m) < 25 or len(df_15m) < 25:
        return SpikeResult(HOLD, "insufficient 5m/15m history")

    is_long = side == "long"
    against = -1 if is_long else 1   # bar direction that hurts us (+1 = up-bar hurts shorts)

    # Adverse excursion in R (how deep into the stop distance price has gone)
    adverse = (entry_price - current_price) if is_long else (current_price - entry_price)
    adverse_r = adverse / one_r

    armed_by_depth = adverse_r >= cfg.spike_min_adverse_r

    # ── 1) 5m single-bar spike ────────────────────────────────────────────────
    rng5, dir5, cf5, atr5, volr5 = _last_closed_bar_stats(df_5m)
    if atr5 > 0:
        mult5 = cfg.spike_5m_atr_mult * (0.8 if volr5 >= cfg.spike_vol_mult else 1.0)
        spike5 = (dir5 == against and rng5 >= mult5 * atr5
                  and cf5 >= cfg.spike_close_frac)
        if spike5 and (armed_by_depth or rng5 >= cfg.spike_hard_atr_mult * atr5):
            return SpikeResult(CLOSE,
                f"5m reversal spike {rng5/atr5:.1f}xATR against {side} "
                f"(close_frac={cf5:.2f}, vol={volr5:.1f}x, adverse={adverse_r:.2f}R)",
                adverse_r)

    # ── 2) 15m single-bar spike ───────────────────────────────────────────────
    rng15, dir15, cf15, atr15, volr15 = _last_closed_bar_stats(df_15m)
    if atr15 > 0:
        mult15 = cfg.spike_15m_atr_mult * (0.8 if volr15 >= cfg.spike_vol_mult else 1.0)
        spike15 = (dir15 == against and rng15 >= mult15 * atr15
                   and cf15 >= cfg.spike_close_frac)
        if spike15 and (armed_by_depth or rng15 >= cfg.spike_hard_atr_mult * atr15):
            return SpikeResult(CLOSE,
                f"15m reversal spike {rng15/atr15:.1f}xATR against {side} "
                f"(close_frac={cf15:.2f}, vol={volr15:.1f}x, adverse={adverse_r:.2f}R)",
                adverse_r)

        # ── 3) 15m V-move: cumulative 3-bar thrust against us ────────────────
        if len(df_15m) >= 4:
            c_now = float(df_15m["close"].iloc[-1])
            c_3ago = float(df_15m["close"].iloc[-4])
            move = (c_now - c_3ago) * against   # positive = 3-bar move hurts us
            if move >= cfg.spike_15m_cum_atr_mult * atr15 and armed_by_depth:
                return SpikeResult(CLOSE,
                    f"15m V-move {move/atr15:.1f}xATR over 3 bars against {side} "
                    f"(adverse={adverse_r:.2f}R)", adverse_r)

    # ── 4) Live-price acceleration (mid-bar, vs last closed 5m close) ─────────
    if atr5 > 0:
        c5 = float(df_5m["close"].iloc[-1])
        live_move = (c5 - current_price) if is_long else (current_price - c5)
        if live_move >= cfg.spike_live_atr_mult * atr5 and armed_by_depth:
            return SpikeResult(CLOSE,
                f"live-price acceleration {live_move/atr5:.1f}xATR(5m) beyond last close "
                f"against {side} (adverse={adverse_r:.2f}R)", adverse_r)

    return SpikeResult(HOLD, f"no spike (adverse={adverse_r:.2f}R)", adverse_r)
