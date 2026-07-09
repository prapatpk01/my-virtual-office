"""
Pure indicator functions — pandas/numpy only, no external TA library.

Every function operates on already-CLOSED bars only. Callers (live loop,
backtest) are responsible for never passing an in-progress candle — that
is the single no-lookahead contract this module relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Moving averages ──────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n)) — fast, low-lag."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(round(np.sqrt(period))))
    raw = 2 * wma(series, half) - wma(series, period)
    return wma(raw, sqrt_p)


def slope_pct(series: pd.Series, lookback: int = 5) -> pd.Series:
    """% change of `series` over `lookback` bars — used for EMA/HMA slope direction."""
    prior = series.shift(lookback)
    return (series - prior) / prior.replace(0, np.nan) * 100


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def roc(series: pd.Series, period: int = 9) -> pd.Series:
    """Rate of change, %."""
    prior = series.shift(period)
    return (series - prior) / prior.replace(0, np.nan) * 100


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    line = ema(series, fast) - ema(series, slow)
    sig  = ema(line, signal)
    return line, sig, line - sig


def vwap(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """
    Rolling VWAP over the last `window` bars — a session-anchored VWAP needs
    exchange session boundaries we don't track per-symbol, so a rolling window
    (default 48 = one 24h day on 30m) is the robust, symbol-agnostic stand-in.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).rolling(window, min_periods=1).sum()
    vol = df["volume"].rolling(window, min_periods=1).sum().replace(0, np.nan)
    return pv / vol


# ── Volatility / trend strength ──────────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-smoothed ATR."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_percentile(atr_series: pd.Series, lookback: int = 100) -> pd.Series:
    """Percentile rank (0-100) of the current ATR within the trailing `lookback` window."""
    def _rank(window: np.ndarray) -> float:
        cur = window[-1]
        return float((window <= cur).sum() - 1) / max(len(window) - 1, 1) * 100.0
    return atr_series.rolling(lookback, min_periods=max(20, lookback // 4)).apply(_rank, raw=True)


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI), Wilder-smoothed."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    dn_move = prev_low - low
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)

    atr_s = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_s  = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False,
                                                          min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False,
                                                           min_periods=period).mean()

    plus_di  = 100 * plus_dm_s  / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_val, plus_di, minus_di


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index: 100 * log10(sum(TR, n) / (max(high,n) - min(low,n))) / log10(n).
    High (>60) = choppy/ranging. Low (<38) = trending.
    """
    tr = true_range(df)
    tr_sum = tr.rolling(period).sum()
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    return 100 * np.log10(tr_sum / rng) / np.log10(period)


# ── Market structure (swing HH/HL vs LH/LL) ──────────────────────────────────

def swing_pivots(highs: pd.Series, lows: pd.Series, left: int = 3, right: int = 3):
    """
    Fractal pivot detection. A pivot at index i needs `right` bars AFTER it to
    confirm — so the most recent `right` bars can never have a confirmed pivot
    yet. This is a natural lag, not lookahead: at live time T we only see bars
    up to T, so a pivot at T-right is the newest one we could possibly know.
    Returns (pivot_high_idx, pivot_low_idx) — lists of positional indices.
    """
    n = len(highs)
    ph, pl = [], []
    h = highs.values
    l = lows.values
    for i in range(left, n - right):
        window_h = h[i - left:i + right + 1]
        window_l = l[i - left:i + right + 1]
        if h[i] == window_h.max() and np.argmax(window_h) == left:
            ph.append(i)
        if l[i] == window_l.min() and np.argmin(window_l) == left:
            pl.append(i)
    return ph, pl


def market_structure(highs: pd.Series, lows: pd.Series,
                     left: int = 3, right: int = 3) -> str:
    """
    Classify structure from the two most recent CONFIRMED swing highs and
    the two most recent CONFIRMED swing lows:
      'HH_HL' — higher high AND higher low  (bullish structure)
      'LH_LL' — lower high AND lower low    (bearish structure)
      'MIXED' — anything else / not enough confirmed pivots
    """
    ph, pl = swing_pivots(highs, lows, left, right)
    if len(ph) < 2 or len(pl) < 2:
        return "MIXED"
    h_vals = highs.values
    l_vals = lows.values
    higher_high = h_vals[ph[-1]] > h_vals[ph[-2]]
    higher_low  = l_vals[pl[-1]] > l_vals[pl[-2]]
    lower_high  = h_vals[ph[-1]] < h_vals[ph[-2]]
    lower_low   = l_vals[pl[-1]] < l_vals[pl[-2]]
    if higher_high and higher_low:
        return "HH_HL"
    if lower_high and lower_low:
        return "LH_LL"
    return "MIXED"


def recent_swing_levels(highs: pd.Series, lows: pd.Series,
                        left: int = 3, right: int = 3) -> tuple[float, float]:
    """Most recent confirmed swing high / swing low price — used for SL safety check."""
    ph, pl = swing_pivots(highs, lows, left, right)
    swing_high = float(highs.values[ph[-1]]) if ph else float("nan")
    swing_low  = float(lows.values[pl[-1]])  if pl else float("nan")
    return swing_high, swing_low
