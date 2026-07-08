"""
Entry Engine — TF 30M.

Fast trigger, deliberately not indicator-heavy: HMA cross/slope, ROC sign,
MACD histogram direction, EMA15 position, candle close direction. No extra
confirmation layers — regime + bias already did the heavy filtering.

`SignalEngine` composes RegimeEngine + BiasEngine + EntryEngine into the
single call used by BOTH main.py (live) and backtest.py, so live and
backtest can never diverge in logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from regime_engine import RegimeEngine, RegimeResult
from bias_engine import BiasEngine, BiasResult, BIAS_BULL, BIAS_BEAR

LONG  = "LONG"
SHORT = "SHORT"
NONE  = "NONE"


@dataclass
class EntryResult:
    signal: str          # LONG | SHORT | NONE
    score: float
    price: float
    components: dict = field(default_factory=dict)
    fresh: bool = False
    fresh_reason: str = ""


def _bars_since_sign_flip(series: pd.Series, lookback: int = 20) -> Optional[int]:
    """
    How many bars ago did `series` (e.g. HMA-fast minus HMA-slow, or the MACD
    histogram) last cross zero? Returns None if no flip within `lookback` bars
    (i.e. the current state has held for a long time — NOT a fresh signal).
    0 means the flip happened on the transition INTO the current bar (freshest
    possible); 3 means the bar 3 positions back was the first bar with the
    current sign.
    """
    vals = series.values
    n = len(vals)
    if n < 2:
        return None
    cur_sign = np.sign(vals[-1])
    if cur_sign == 0:
        return None
    max_back = min(lookback, n - 2)
    for back in range(0, max_back + 1):
        newer_idx = n - 1 - back
        older_idx = n - 2 - back
        if older_idx < 0:
            break
        newer_sign = np.sign(vals[newer_idx])
        older_sign = np.sign(vals[older_idx])
        if newer_sign == cur_sign and older_sign != 0 and older_sign != cur_sign:
            return back
    return None


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_30m: pd.DataFrame) -> tuple[EntryResult, EntryResult]:
        """Returns (long_result, short_result) — caller picks based on bias."""
        c = self.cfg
        if len(df_30m) < max(c.entry_hma_slow, c.entry_macd_slow) + 10:
            empty = lambda: EntryResult(NONE, 0.0, 0.0, {})
            return empty(), empty()

        closes = df_30m["close"]
        opens  = df_30m["open"]
        price  = float(closes.iloc[-1])

        hma_fast = ind.hma(closes, c.entry_hma_fast)
        hma_slow = ind.hma(closes, c.entry_hma_slow)
        hf_now, hf_prev = float(hma_fast.iloc[-1]), float(hma_fast.iloc[-1 - c.entry_hma_slope_lookback])
        hs_now = float(hma_slow.iloc[-1])
        hma_slope_up = hf_now > hf_prev
        hma_slope_dn = hf_now < hf_prev

        # cross detection (this bar vs previous bar)
        hf_prev1, hs_prev1 = float(hma_fast.iloc[-2]), float(hma_slow.iloc[-2])
        cross_up   = hf_prev1 <= hs_prev1 and hf_now > hs_now
        cross_down = hf_prev1 >= hs_prev1 and hf_now < hs_now

        hma_long_ok  = cross_up   or (hf_now > hs_now and hma_slope_up)
        hma_short_ok = cross_down or (hf_now < hs_now and hma_slope_dn)

        roc_s = ind.roc(closes, c.entry_roc_period)
        roc_v = float(roc_s.iloc[-1]) if not np.isnan(roc_s.iloc[-1]) else 0.0

        _, _, hist = ind.macd(closes, c.entry_macd_fast, c.entry_macd_slow, c.entry_macd_signal)
        hist_now  = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        hist_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        macd_rising  = hist_now > hist_prev
        macd_falling = hist_now < hist_prev

        e15 = float(ind.ema(closes, c.entry_ema_ref).iloc[-1])

        close_up   = float(closes.iloc[-1]) > float(opens.iloc[-1])
        close_down = float(closes.iloc[-1]) < float(opens.iloc[-1])

        # ── Freshness gate 1: the setup must be something HAPPENING NOW, not
        # something that already ran its course many bars ago while the score
        # merely stayed in the entry zone. Anchor on whichever trigger is most
        # recent — HMA fast/slow crossing, or the MACD histogram crossing zero
        # (a fresh momentum re-acceleration even without a literal HMA cross,
        # e.g. a pullback-then-resume). If neither happened within
        # entry_freshness_bars, the move is "old news" — wait for the next round.
        hma_diff = hma_fast - hma_slow
        hma_flip_bars  = _bars_since_sign_flip(hma_diff, lookback=20)
        macd_flip_bars = _bars_since_sign_flip(hist, lookback=20)

        def _fresh_within(bars: Optional[int]) -> bool:
            return bars is not None and bars <= c.entry_freshness_bars

        flip_fresh_long  = (_fresh_within(hma_flip_bars)  and hf_now > hs_now) or \
                           (_fresh_within(macd_flip_bars) and hist_now > 0)
        flip_fresh_short = (_fresh_within(hma_flip_bars)  and hf_now < hs_now) or \
                           (_fresh_within(macd_flip_bars) and hist_now < 0)

        # ── Freshness gate 2: don't chase a spike that already happened. If
        # price is already far (in ATR terms) from EMA15, the move's "meat" is
        # behind us — this is what let the bot short XAG right at the bottom
        # of a capitulation candle with a huge volume spike (bias/regime were
        # legitimately BEAR/TREND from the established multi-hour decline, but
        # the ENTRY itself fired chasing the final spike, not near its start).
        atr_val = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        ext_atr = abs(price - e15) / atr_val if (atr_val and atr_val > 0 and not np.isnan(atr_val)) else 0.0
        not_overextended = ext_atr <= c.entry_max_ext_atr

        fresh_long  = flip_fresh_long  and not_overextended
        fresh_short = flip_fresh_short and not_overextended
        fresh_reason = (f"hma_flip={hma_flip_bars} macd_flip={macd_flip_bars} "
                       f"(need <= {c.entry_freshness_bars} bars)  ext={ext_atr:.1f}ATR "
                       f"(max {c.entry_max_ext_atr})")

        # ── LONG components ───────────────────────────────────────────────────
        lc = {
            "hma":   30.0 if hma_long_ok else 0.0,
            "roc":   20.0 if roc_v > 0 else 0.0,
            "macd":  20.0 if macd_rising else 0.0,
            "ema15": 15.0 if price > e15 else 0.0,
            "close": 15.0 if close_up else 0.0,
        }
        long_score = sum(lc.values())

        # ── SHORT components ──────────────────────────────────────────────────
        sc = {
            "hma":   30.0 if hma_short_ok else 0.0,
            "roc":   20.0 if roc_v < 0 else 0.0,
            "macd":  20.0 if macd_falling else 0.0,
            "ema15": 15.0 if price < e15 else 0.0,
            "close": 15.0 if close_down else 0.0,
        }
        short_score = sum(sc.values())

        long_res  = EntryResult(LONG  if long_score  > 0 else NONE, long_score,  price, lc,
                                fresh=fresh_long, fresh_reason=fresh_reason)
        short_res = EntryResult(SHORT if short_score > 0 else NONE, short_score, price, sc,
                                fresh=fresh_short, fresh_reason=fresh_reason)
        return long_res, short_res


@dataclass
class FinalSignal:
    direction: str        # LONG | SHORT | NONE
    entry_score: float
    entry_components: dict
    regime: RegimeResult
    bias: BiasResult
    price: float
    reason: str = ""


class SignalEngine:
    """Single source of truth for signal generation — used by main.py AND backtest.py."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.entry_engine = EntryEngine(cfg)

    def evaluate(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame,
                df_4h: pd.DataFrame) -> FinalSignal:
        c = self.cfg
        regime = self.regime_engine.analyze(df_4h)
        bias = self.bias_engine.analyze(df_1h)
        long_res, short_res = self.entry_engine.analyze(df_30m)

        price = long_res.price or short_res.price or (
            float(df_30m["close"].iloc[-1]) if len(df_30m) else 0.0)

        if not regime.allow_trade:
            return FinalSignal(NONE, 0.0, {}, regime, bias, price, f"regime blocked: {regime.note}")
        if regime.score < c.regime_score_min_to_trade:
            return FinalSignal(NONE, 0.0, {}, regime, bias, price,
                               f"regime score {regime.score:.0f} < {c.regime_score_min_to_trade:.0f}")

        entry_thr = c.entry_score_min + regime.entry_threshold_adj
        bias_thr  = c.bias_score_min + regime.bias_threshold_adj

        if bias.bias == BIAS_BULL:
            if bias.bull_score < bias_thr:
                return FinalSignal(NONE, long_res.score, long_res.components, regime, bias,
                                   price, f"bias score {bias.bull_score:.0f} < {bias_thr:.0f}")
            if long_res.score >= entry_thr:
                if not long_res.fresh:
                    return FinalSignal(NONE, long_res.score, long_res.components, regime, bias,
                                       price, f"score qualifies but not fresh — {long_res.fresh_reason} "
                                       f"— waiting for next signal round")
                return FinalSignal(LONG, long_res.score, long_res.components, regime, bias,
                                   price, "long entry confirmed")
            return FinalSignal(NONE, long_res.score, long_res.components, regime, bias, price,
                               f"entry score {long_res.score:.0f} < {entry_thr:.0f}")

        if bias.bias == BIAS_BEAR:
            if bias.bear_score < bias_thr:
                return FinalSignal(NONE, short_res.score, short_res.components, regime, bias,
                                   price, f"bias score {bias.bear_score:.0f} < {bias_thr:.0f}")
            if short_res.score >= entry_thr:
                if not short_res.fresh:
                    return FinalSignal(NONE, short_res.score, short_res.components, regime, bias,
                                       price, f"score qualifies but not fresh — {short_res.fresh_reason} "
                                       f"— waiting for next signal round")
                return FinalSignal(SHORT, short_res.score, short_res.components, regime, bias,
                                   price, "short entry confirmed")
            return FinalSignal(NONE, short_res.score, short_res.components, regime, bias, price,
                               f"entry score {short_res.score:.0f} < {entry_thr:.0f}")

        return FinalSignal(NONE, 0.0, {}, regime, bias, price, "bias NEUTRAL — no new entries")
