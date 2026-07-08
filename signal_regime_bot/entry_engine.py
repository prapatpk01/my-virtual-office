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

        long_res  = EntryResult(LONG  if long_score  > 0 else NONE, long_score,  price, lc)
        short_res = EntryResult(SHORT if short_score > 0 else NONE, short_score, price, sc)
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
                return FinalSignal(LONG, long_res.score, long_res.components, regime, bias,
                                   price, "long entry confirmed")
            return FinalSignal(NONE, long_res.score, long_res.components, regime, bias, price,
                               f"entry score {long_res.score:.0f} < {entry_thr:.0f}")

        if bias.bias == BIAS_BEAR:
            if bias.bear_score < bias_thr:
                return FinalSignal(NONE, short_res.score, short_res.components, regime, bias,
                                   price, f"bias score {bias.bear_score:.0f} < {bias_thr:.0f}")
            if short_res.score >= entry_thr:
                return FinalSignal(SHORT, short_res.score, short_res.components, regime, bias,
                                   price, "short entry confirmed")
            return FinalSignal(NONE, short_res.score, short_res.components, regime, bias, price,
                               f"entry score {short_res.score:.0f} < {entry_thr:.0f}")

        return FinalSignal(NONE, 0.0, {}, regime, bias, price, "bias NEUTRAL — no new entries")
