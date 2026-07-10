"""
Shared numpy indicator toolkit used by the Layer 0/3/5 engines
(Market Quality, Regime Classifier, Strategy Engine).

Kept dependency-free (numpy only) and side-effect-free — every function
takes arrays, returns arrays or scalars. No candle/dataclass coupling,
so any layer can reuse the same math without re-deriving it.
"""
from __future__ import annotations

import numpy as np


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def hma(arr: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average — WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    def wma(a: np.ndarray, p: int) -> np.ndarray:
        out = np.full(len(a), np.nan)
        if len(a) < p:
            return out
        weights = np.arange(1, p + 1)
        for i in range(p - 1, len(a)):
            window = a[i - p + 1: i + 1]
            out[i] = np.dot(window, weights) / weights.sum()
        return out

    if len(arr) < period:
        return np.full(len(arr), np.nan)
    half = max(1, period // 2)
    sqrt_p = max(1, int(round(period ** 0.5)))
    wma_half = wma(arr, half)
    wma_full = wma(arr, period)
    raw = 2 * wma_half - wma_full
    raw = np.nan_to_num(raw, nan=0.0)
    return wma(raw, sqrt_p)


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, 50.0)
    if n < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, n - 1):
        g, l = gains[i], losses[i]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    macd_line = np.nan_to_num(macd_line, nan=0.0)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)
    out = np.full(n, np.nan)
    if n > period:
        out[period] = float(np.nanmean(tr[1:period + 1]))
        for i in range(period + 1, n):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14):
    n = len(closes)
    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    adx_arr = np.full(n, np.nan)
    if n < period * 2:
        return adx_arr, pdi, mdi

    tr_arr = np.full(n, 0.0)
    pdm = np.full(n, 0.0)
    mdm = np.full(n, 0.0)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr_arr[i] = max(hl, hpc, lpc)
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        pdm[i] = up if up > down and up > 0 else 0.0
        mdm[i] = down if down > up and down > 0 else 0.0

    atr14 = np.full(n, 0.0)
    pdi14 = np.full(n, 0.0)
    mdi14 = np.full(n, 0.0)
    atr14[period] = float(np.sum(tr_arr[1:period + 1]))
    pdi14[period] = float(np.sum(pdm[1:period + 1]))
    mdi14[period] = float(np.sum(mdm[1:period + 1]))
    for i in range(period + 1, n):
        atr14[i] = atr14[i - 1] - atr14[i - 1] / period + tr_arr[i]
        pdi14[i] = pdi14[i - 1] - pdi14[i - 1] / period + pdm[i]
        mdi14[i] = mdi14[i - 1] - mdi14[i - 1] / period + mdm[i]

    for i in range(period, n):
        if atr14[i] > 0:
            pdi[i] = 100 * pdi14[i] / atr14[i]
            mdi[i] = 100 * mdi14[i] / atr14[i]

    dx = np.full(n, np.nan)
    for i in range(period, n):
        denom = pdi[i] + mdi[i]
        if denom > 0:
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / denom

    if n >= period * 2:
        adx_arr[period * 2 - 1] = float(np.nanmean(dx[period:period * 2]))
        for i in range(period * 2, n):
            if not np.isnan(adx_arr[i - 1]) and not np.isnan(dx[i]):
                adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period
    return adx_arr, pdi, mdi


def bollinger(closes: np.ndarray, period: int = 20, mult: float = 2.0):
    n = len(closes)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        m = float(np.mean(window))
        sd = float(np.std(window))
        mid[i] = m
        upper[i] = m + mult * sd
        lower[i] = m - mult * sd
    width = np.where(mid > 0, (upper - lower) / mid, np.nan)
    return mid, upper, lower, width


def roc(closes: np.ndarray, period: int = 10) -> np.ndarray:
    n = len(closes)
    out = np.full(n, 0.0)
    for i in range(period, n):
        prev = closes[i - period]
        if prev != 0:
            out[i] = (closes[i] - prev) / prev * 100.0
    return out


def percentile_rank(arr: np.ndarray, value: float) -> float:
    """% of values in arr that are <= value. Used for ATR percentile etc."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 50.0
    return float(np.sum(valid <= value) / len(valid) * 100.0)


def swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int = 3):
    """Simple fractal swing high/low detector: index i is a swing high if it's
    the max of its +/- lookback neighborhood (swing low analogously)."""
    n = len(highs)
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max():
            swing_highs.append((i, highs[i]))
        if lows[i] == window_l.min():
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def bullish_engulfing(o1, c1, o2, c2) -> bool:
    return c1 < o1 and c2 > o2 and c2 >= o1 and o2 <= c1


def bearish_engulfing(o1, c1, o2, c2) -> bool:
    return c1 > o1 and c2 < o2 and c2 <= o1 and o2 >= c1
