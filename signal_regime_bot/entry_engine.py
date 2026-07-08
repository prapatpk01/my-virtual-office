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
    # Signal Round Gate: identity of the trigger event (HMA cross / MACD
    # zero-cross) that opened the current entry window — the timestamp of the
    # bar where the flip landed. One round = at most ONE entry: the caller
    # records the round_id it traded and refuses to re-enter the same round,
    # so a stop-out can't be immediately re-chased on the same (failed) setup.
    # None = no qualifying trigger within the lookback (no open round).
    round_id: object = None
    round_age_bars: object = None


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


def _structure_confirms(df_30m: pd.DataFrame, c: Config) -> tuple[list, list]:
    """
    Market-structure confirmations on the last CLOSED bar, per direction.
    Returns (long_confirms, short_confirms) — lists of short labels:
      'break'  — close beyond the previous candle's high (long) / low (short)
      'vol'    — volume expansion vs the 20-bar average
      'wick'   — rejection wick (lower wick for long, upper for short)
      'sweep'  — liquidity sweep: pierced the prior N-bar low (long) / high
                 (short) intrabar, then closed back inside
    """
    long_c: list = []
    short_c: list = []
    if len(df_30m) < max(22, c.entry_sweep_lookback + 2):
        return long_c, short_c

    o = float(df_30m["open"].iloc[-1])
    h = float(df_30m["high"].iloc[-1])
    l = float(df_30m["low"].iloc[-1])
    cl = float(df_30m["close"].iloc[-1])
    prev_h = float(df_30m["high"].iloc[-2])
    prev_l = float(df_30m["low"].iloc[-2])

    # break of previous candle high/low
    if cl > prev_h:
        long_c.append("break")
    if cl < prev_l:
        short_c.append("break")

    # volume expansion (direction-neutral signal, credited to the bar's side)
    vol = float(df_30m["volume"].iloc[-1])
    vol_ma = float(df_30m["volume"].iloc[-21:-1].mean())
    if vol_ma > 0 and vol >= c.entry_vol_expansion_mult * vol_ma:
        if cl >= o:
            long_c.append("vol")
        if cl <= o:
            short_c.append("vol")

    # wick rejection on the current bar
    rng = max(h - l, 1e-12)
    lower_wick = (min(o, cl) - l) / rng
    upper_wick = (h - max(o, cl)) / rng
    if lower_wick >= c.entry_wick_reject_frac:
        long_c.append("wick")
    if upper_wick >= c.entry_wick_reject_frac:
        short_c.append("wick")

    # liquidity sweep of the prior N-bar extreme
    lb = c.entry_sweep_lookback
    prior_low = float(df_30m["low"].iloc[-(lb + 1):-1].min())
    prior_high = float(df_30m["high"].iloc[-(lb + 1):-1].max())
    if l < prior_low and cl > prior_low:
        long_c.append("sweep")
    if h > prior_high and cl < prior_high:
        short_c.append("sweep")

    return long_c, short_c


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

        # ── Signal Round Gate: HMA10 x HMA20 cross = start of a new round.
        # The cross is the ONLY round trigger (the state machine the user
        # specified: cross -> setup opens -> the entry score gets
        # entry_setup_window_bars closed bars to reach threshold -> enter, or
        # the round is cancelled; a cross back the other way kills the old
        # round and starts a fresh one automatically — the flip age resets).
        # setup_age is 1-based: the cross bar itself is age 1, per the spec's
        # "count the entry score within the next 1-5 bars".
        hma_diff = hma_fast - hma_slow
        hma_flip_bars = _bars_since_sign_flip(hma_diff, lookback=20)

        long_anchor  = hma_flip_bars if hf_now > hs_now else None
        short_anchor = hma_flip_bars if hf_now < hs_now else None
        round_id_long  = df_30m.index[-1 - long_anchor]  if long_anchor  is not None else None
        round_id_short = df_30m.index[-1 - short_anchor] if short_anchor is not None else None
        setup_age_long  = (long_anchor + 1)  if long_anchor  is not None else None
        setup_age_short = (short_anchor + 1) if short_anchor is not None else None

        window = c.entry_setup_window_bars
        in_window_long  = setup_age_long  is not None and setup_age_long  <= window
        in_window_short = setup_age_short is not None and setup_age_short <= window

        # ── Anti-chase: don't enter a round whose move already ran away. If
        # price is far (in ATR terms) from EMA15, the spike already happened —
        # this is what let the bot short XAG at the bottom of a capitulation
        # candle. A young round can still be a bad round.
        atr_val = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        ext_atr = abs(price - e15) / atr_val if (atr_val and atr_val > 0 and not np.isnan(atr_val)) else 0.0
        not_overextended = ext_atr <= c.entry_max_ext_atr

        # ── Market-structure confirmation (reduce false triggers): at least
        # entry_structure_confirm_min of these must back the direction —
        # break of previous candle high/low, volume expansion, wick rejection,
        # liquidity sweep.
        struct_long, struct_short = _structure_confirms(df_30m, c)
        struct_ok_long  = len(struct_long)  >= c.entry_structure_confirm_min
        struct_ok_short = len(struct_short) >= c.entry_structure_confirm_min

        fresh_long  = in_window_long  and not_overextended and struct_ok_long
        fresh_short = in_window_short and not_overextended and struct_ok_short
        fresh_reason = (f"setup_age: long={setup_age_long} short={setup_age_short} "
                       f"(window <= {window})  ext={ext_atr:.1f}ATR (max {c.entry_max_ext_atr})  "
                       f"struct: long={struct_long or '-'} short={struct_short or '-'} "
                       f"(need >= {c.entry_structure_confirm_min})")

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
                                fresh=fresh_long, fresh_reason=fresh_reason,
                                round_id=round_id_long, round_age_bars=setup_age_long)
        short_res = EntryResult(SHORT if short_score > 0 else NONE, short_score, price, sc,
                                fresh=fresh_short, fresh_reason=fresh_reason,
                                round_id=round_id_short, round_age_bars=setup_age_short)
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
    round_id: object = None      # trigger-bar timestamp of the signal round
    round_age_bars: object = None


class SignalEngine:
    """Single source of truth for signal generation — used by main.py AND backtest.py."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.entry_engine = EntryEngine(cfg)

    def evaluate(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame,
                df_4h: pd.DataFrame) -> FinalSignal:
        sig = self._evaluate_core(df_30m, df_1h, df_4h)
        # Always expose round info (even on NONE results) so the status log
        # and callers can see where the current round stands for the active
        # bias side — not just on confirmed entries.
        if sig.round_id is None and self._last_results is not None:
            long_res, short_res = self._last_results
            active = (long_res if sig.bias.bias == BIAS_BULL
                      else short_res if sig.bias.bias == BIAS_BEAR else None)
            if active is not None:
                sig.round_id = active.round_id
                sig.round_age_bars = active.round_age_bars
        return sig

    def _evaluate_core(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame,
                       df_4h: pd.DataFrame) -> FinalSignal:
        c = self.cfg
        regime = self.regime_engine.analyze(df_4h)
        bias = self.bias_engine.analyze(df_1h)
        long_res, short_res = self.entry_engine.analyze(df_30m)
        self._last_results = (long_res, short_res)

        price = long_res.price or short_res.price or (
            float(df_30m["close"].iloc[-1]) if len(df_30m) else 0.0)

        if not regime.allow_trade:
            return FinalSignal(NONE, 0.0, {}, regime, bias, price, f"regime blocked: {regime.note}")
        if regime.score < c.regime_score_min_to_trade:
            return FinalSignal(NONE, 0.0, {}, regime, bias, price,
                               f"regime score {regime.score:.0f} < {c.regime_score_min_to_trade:.0f}")

        entry_thr = c.entry_score_min + regime.entry_threshold_adj
        bias_thr  = c.bias_score_min + regime.bias_threshold_adj

        # Bias confidence modulates entry strictness: a well-backed bias
        # (strong ADX, aligned RSI/EMA slopes, volume, price at value) may
        # enter slightly easier; a shaky one must clear a higher bar.
        if bias.confidence >= c.bias_conf_high:
            entry_thr += c.bias_conf_high_adj
        elif bias.confidence < c.bias_conf_low:
            entry_thr += c.bias_conf_low_adj

        if bias.bias == BIAS_BULL:
            if bias.bull_score < bias_thr:
                return FinalSignal(NONE, long_res.score, long_res.components, regime, bias,
                                   price, f"bias score {bias.bull_score:.0f} < {bias_thr:.0f}")
            if long_res.score >= entry_thr:
                if not long_res.fresh:
                    return FinalSignal(NONE, long_res.score, long_res.components, regime, bias,
                                       price, f"score qualifies but round not open — {long_res.fresh_reason} "
                                       f"— waiting for next signal round")
                return FinalSignal(LONG, long_res.score, long_res.components, regime, bias,
                                   price, "long entry confirmed",
                                   round_id=long_res.round_id, round_age_bars=long_res.round_age_bars)
            return FinalSignal(NONE, long_res.score, long_res.components, regime, bias, price,
                               f"entry score {long_res.score:.0f} < {entry_thr:.0f}")

        if bias.bias == BIAS_BEAR:
            if bias.bear_score < bias_thr:
                return FinalSignal(NONE, short_res.score, short_res.components, regime, bias,
                                   price, f"bias score {bias.bear_score:.0f} < {bias_thr:.0f}")
            if short_res.score >= entry_thr:
                if not short_res.fresh:
                    return FinalSignal(NONE, short_res.score, short_res.components, regime, bias,
                                       price, f"score qualifies but round not open — {short_res.fresh_reason} "
                                       f"— waiting for next signal round")
                return FinalSignal(SHORT, short_res.score, short_res.components, regime, bias,
                                   price, "short entry confirmed",
                                   round_id=short_res.round_id, round_age_bars=short_res.round_age_bars)
            return FinalSignal(NONE, short_res.score, short_res.components, regime, bias, price,
                               f"entry score {short_res.score:.0f} < {entry_thr:.0f}")

        return FinalSignal(NONE, 0.0, {}, regime, bias, price, "bias NEUTRAL — no new entries")
