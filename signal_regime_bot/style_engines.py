"""
Style engines — alternative entry logic for the non-trend regimes.

The trend styles (TREND / SWING) use the HMA-cross Entry Engine. RANGE and
COMPRESSION need fundamentally different triggers:

  MEANREV  (RANGE)       — fade a stretched, exhausted move back to the mean.
                           Direction is COUNTER to the stretch; the trend-
                           momentum bias gate is bypassed (it would always
                           veto a counter-trend entry).
  BREAKOUT (COMPRESSION) — enter the direction of a range break, but only
                           WITH volume expansion (a break on no volume is a
                           fake-out).

Both return the same EntryResult shape as entry_engine so the pipeline can
treat them uniformly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
import price_action as pa
from config import Config
from entry_engine import EntryResult, LONG, SHORT, NONE


class MeanReversionEntry:
    """RANGE regime — fade the extreme."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_30m: pd.DataFrame) -> EntryResult:
        c = self.cfg
        thr = c.meanrev_entry_threshold
        if len(df_30m) < 40:
            return EntryResult(NONE, None, 0.0, thr, False, False, True, "insufficient 30m history")

        closes = df_30m["close"]
        price = float(closes.iloc[-1])
        e20 = float(ind.ema(closes, c.regime_ema_fast).iloc[-1])
        atr_v = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        rsi_v = float(ind.rsi(closes, 14).iloc[-1])
        if np.isnan(atr_v) or atr_v <= 0:
            return EntryResult(NONE, None, 0.0, thr, False, False, True, "no ATR")
        ext = (price - e20) / atr_v      # signed: + = stretched above, - = below

        # decide the fade side
        if ext <= -c.meanrev_ext_atr_min and rsi_v <= c.meanrev_rsi_long_max:
            side = LONG                   # oversold + stretched below -> fade up
        elif ext >= c.meanrev_ext_atr_min and rsi_v >= c.meanrev_rsi_short_min:
            side = SHORT                  # overbought + stretched above -> fade down
        else:
            return EntryResult(NONE, None, 0.0, thr, False, False, True,
                               f"no extreme (ext={ext:+.1f}ATR rsi={rsi_v:.0f})", price=price)

        # reversal trigger + score
        comp = {}
        comp["rsi_extreme"] = 30.0
        comp["stretch"] = min(20.0, abs(ext) * 8.0)                       # more stretch = stronger fade
        comp["rejection"] = 25.0 if pa.rejection_candle(df_30m, side, c.entry_wick_reject_frac) else 0.0
        engulf = pa.bull_engulf(df_30m) if side == LONG else pa.bear_engulf(df_30m)
        comp["engulf"] = 15.0 if engulf else 0.0
        comp["sweep"] = 10.0 if pa.liquidity_sweep(df_30m, side, c.entry_sweep_lookback) else 0.0
        score = round(sum(comp.values()), 1)

        # round id = the bar the extreme reversal triggers on
        round_id = df_30m.index[-1]
        if score >= thr:
            return EntryResult(side, 1, score, thr, True, False, False,
                               f"mean-reversion {side} score {score:.0f} >= {thr:.0f} "
                               f"(ext={ext:+.1f}ATR rsi={rsi_v:.0f})", round_id=round_id, price=price)
        return EntryResult(NONE, 1, score, thr, False, False, True,
                           f"reversal not confirmed ({score:.0f} < {thr:.0f})",
                           round_id=round_id, price=price)


class BreakoutEntry:
    """COMPRESSION regime — enter a volume-confirmed range break."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, df_30m: pd.DataFrame) -> EntryResult:
        c = self.cfg
        thr = c.breakout_entry_threshold
        lb = c.breakout_lookback
        if len(df_30m) < lb + 5:
            return EntryResult(NONE, None, 0.0, thr, False, False, True, "insufficient 30m history")

        price = float(df_30m["close"].iloc[-1])
        prior_high = float(df_30m["high"].iloc[-(lb + 1):-1].max())
        prior_low = float(df_30m["low"].iloc[-(lb + 1):-1].min())

        if price > prior_high:
            side = LONG
        elif price < prior_low:
            side = SHORT
        else:
            return EntryResult(NONE, None, 0.0, thr, False, False, True,
                               "no range break", price=price)

        vol_ok = pa.volume_expansion(df_30m, c.breakout_vol_mult)
        o = float(df_30m["open"].iloc[-1])
        strong_candle = (price > o) if side == LONG else (price < o)

        comp = {}
        comp["break"] = 40.0
        comp["volume"] = 35.0 if vol_ok else 0.0            # the make-or-break component
        comp["candle"] = 25.0 if strong_candle else 0.0
        score = round(sum(comp.values()), 1)

        round_id = df_30m.index[-1]
        if score >= thr and vol_ok:
            return EntryResult(side, 1, score, thr, True, False, False,
                               f"breakout {side} score {score:.0f} >= {thr:.0f} (vol confirmed)",
                               round_id=round_id, price=price)
        return EntryResult(NONE, 1, score, thr, False, False, True,
                           f"breakout not confirmed (score {score:.0f}, vol_ok={vol_ok})",
                           round_id=round_id, price=price)
