"""
Price-action feature library — shared by Context Engine (30M) and Early
Entry Booster (15M). Kept in one module so the two engines can't drift on
what "a bull engulf" or "a liquidity sweep" means, and so neither
duplicates the other's detection code.

Every function reads the LAST CLOSED bar of the passed frame. Callers own
the no-lookahead contract (never pass an in-progress candle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind


# ── Candlestick patterns ──────────────────────────────────────────────────────

def bull_engulf(df: pd.DataFrame) -> bool:
    """Current close > current open, and the body engulfs the prior (red) body."""
    if len(df) < 2:
        return False
    o, c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    po, pc = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    return c > o and pc < po and c >= po and o <= pc


def bear_engulf(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    o, c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    po, pc = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    return c < o and pc > po and c <= po and o >= pc


def rejection_candle(df: pd.DataFrame, side: str, wick_frac: float = 0.5) -> bool:
    """Long lower wick (bullish rejection) / upper wick (bearish rejection)."""
    o = float(df["open"].iloc[-1]); h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1]);  c = float(df["close"].iloc[-1])
    rng = max(h - l, 1e-12)
    if side == "LONG":
        return (min(o, c) - l) / rng >= wick_frac
    return (h - max(o, c)) / rng >= wick_frac


# ── Volume ─────────────────────────────────────────────────────────────────────

def volume_expansion(df: pd.DataFrame, mult: float = 1.5, ma: int = 20) -> bool:
    if len(df) < ma + 1:
        return False
    vol = float(df["volume"].iloc[-1])
    vol_ma = float(df["volume"].iloc[-(ma + 1):-1].mean())
    return vol_ma > 0 and vol >= mult * vol_ma


def volume_spike(df: pd.DataFrame, mult: float = 2.0, ma: int = 20) -> bool:
    return volume_expansion(df, mult, ma)


# ── VWAP ───────────────────────────────────────────────────────────────────────

def vwap_reclaim(df: pd.DataFrame, side: str, window: int = 48) -> bool:
    """Price crossed back to the favorable side of VWAP this bar (reclaim/reject)."""
    if len(df) < 3:
        return False
    v = ind.vwap(df, window)
    c_now, c_prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
    v_now, v_prev = float(v.iloc[-1]), float(v.iloc[-2])
    if np.isnan(v_now) or np.isnan(v_prev):
        return False
    if side == "LONG":
        return c_prev <= v_prev and c_now > v_now      # reclaimed above VWAP
    return c_prev >= v_prev and c_now < v_now           # rejected below VWAP


# ── Structure: BOS / CHOCH ────────────────────────────────────────────────────

def bos_choch(df: pd.DataFrame, side: str, left: int = 3, right: int = 3) -> tuple[bool, bool]:
    """
    Returns (bos, choch) for the given side.
      BOS  — trend continuation: close breaks the most recent confirmed swing
             high (long) / low (short) IN the trend direction.
      CHOCH— change of character: close breaks the opposite swing, i.e. the
             first structural break against the prior micro-trend.
    Both read confirmed pivots only (need `right` bars after a pivot), so
    neither uses lookahead.
    """
    ph, pl = ind.swing_pivots(df["high"], df["low"], left, right)
    close = float(df["close"].iloc[-1])
    h_vals, l_vals = df["high"].values, df["low"].values
    last_ph = float(h_vals[ph[-1]]) if ph else np.nan
    last_pl = float(l_vals[pl[-1]]) if pl else np.nan

    if side == "LONG":
        bos = (not np.isnan(last_ph)) and close > last_ph      # broke swing high up
        choch = (not np.isnan(last_pl)) and close > last_ph if False else False
        # CHOCH for a long = we WERE making lower highs and just broke one upward
        if len(ph) >= 2:
            prior_ph = float(h_vals[ph[-2]])
            choch = last_ph < prior_ph and close > last_ph
        return bos, choch
    else:
        bos = (not np.isnan(last_pl)) and close < last_pl      # broke swing low down
        choch = False
        if len(pl) >= 2:
            prior_pl = float(l_vals[pl[-2]])
            choch = last_pl > prior_pl and close < last_pl
        return bos, choch


# ── Liquidity sweep ───────────────────────────────────────────────────────────

def liquidity_sweep(df: pd.DataFrame, side: str, lookback: int = 10) -> bool:
    """
    Pierced the prior N-bar extreme intrabar, then closed back inside it —
    a stop-run that failed to hold. Long: sweep the prior low. Short: sweep
    the prior high.
    """
    if len(df) < lookback + 2:
        return False
    h = float(df["high"].iloc[-1]); l = float(df["low"].iloc[-1]); c = float(df["close"].iloc[-1])
    prior_low = float(df["low"].iloc[-(lookback + 1):-1].min())
    prior_high = float(df["high"].iloc[-(lookback + 1):-1].max())
    if side == "LONG":
        return l < prior_low and c > prior_low
    return h > prior_high and c < prior_high


# ── EMA pullback / bounce ─────────────────────────────────────────────────────

def ema_pullback(df: pd.DataFrame, side: str, ema_period: int = 20,
                 zone_atr: float = 0.6, atr_period: int = 14) -> bool:
    """
    Price pulled back INTO the EMA zone and is turning back with trend — a
    long wants price to have dipped near/below EMA20 then closed above it;
    a short is the mirror.
    """
    if len(df) < ema_period + 2:
        return False
    e = ind.ema(df["close"], ema_period)
    a = ind.atr(df, atr_period)
    e_now = float(e.iloc[-1]); a_now = float(a.iloc[-1])
    c = float(df["close"].iloc[-1]); lo = float(df["low"].iloc[-1]); hi = float(df["high"].iloc[-1])
    if np.isnan(a_now) or a_now <= 0:
        return False
    zone = zone_atr * a_now
    if side == "LONG":
        touched = lo <= e_now + zone
        holding = c > e_now
        return touched and holding
    touched = hi >= e_now - zone
    holding = c < e_now
    return touched and holding


def ema_bounce(df: pd.DataFrame, side: str, ema_period: int = 20) -> bool:
    """A single-bar bounce off the EMA (wick tags the EMA, body closes away with trend)."""
    if len(df) < ema_period + 1:
        return False
    e = float(ind.ema(df["close"], ema_period).iloc[-1])
    o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
    lo = float(df["low"].iloc[-1]); hi = float(df["high"].iloc[-1])
    if side == "LONG":
        return lo <= e <= max(o, c) and c > o
    return hi >= e >= min(o, c) and c < o


# ── Retest of a broken level ──────────────────────────────────────────────────

def successful_retest(df: pd.DataFrame, side: str, lookback: int = 10) -> bool:
    """
    Broke the prior N-bar extreme a few bars ago, pulled back to it, and held —
    a break-and-retest. Approximation: the level was broken within the window
    and the current bar's low (long) / high (short) revisited it without
    closing back through.
    """
    if len(df) < lookback + 3:
        return False
    prior_high = float(df["high"].iloc[-(lookback + 3):-3].max())
    prior_low = float(df["low"].iloc[-(lookback + 3):-3].min())
    c = float(df["close"].iloc[-1]); lo = float(df["low"].iloc[-1]); hi = float(df["high"].iloc[-1])
    recent_high = float(df["high"].iloc[-3:].max())
    recent_low = float(df["low"].iloc[-3:].min())
    if side == "LONG":
        broke = recent_high > prior_high
        retested = lo <= prior_high * 1.001 and c > prior_high
        return broke and retested
    broke = recent_low < prior_low
    retested = hi >= prior_low * 0.999 and c < prior_low
    return broke and retested


def break_prev_extreme(df: pd.DataFrame, side: str) -> bool:
    """Close breaks the previous bar's high (long) / low (short)."""
    if len(df) < 2:
        return False
    c = float(df["close"].iloc[-1])
    if side == "LONG":
        return c > float(df["high"].iloc[-2])
    return c < float(df["low"].iloc[-2])


# ── Session quality ───────────────────────────────────────────────────────────

def session_quality(df: pd.DataFrame) -> float:
    """
    0..1 — favor the liquid overlap hours (London 07-16 UTC, New York
    12-21 UTC). The last bar's UTC hour drives it; dead Asian-afternoon
    hours score lower. Timestamp comes from the frame's index, so it stays
    correct in both live and backtest.
    """
    try:
        hour = int(df.index[-1].hour)
    except Exception:
        return 0.6
    london = 7 <= hour < 16
    newyork = 12 <= hour < 21
    if london and newyork:      # 12-16 UTC overlap — best liquidity
        return 1.0
    if london or newyork:
        return 0.8
    if 0 <= hour < 7:           # Asian session — thinner
        return 0.5
    return 0.6
