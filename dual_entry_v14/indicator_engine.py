"""Indicator Engine — all formulas from spec section 6, numpy arrays over
CLOSED candles only. Every ratio is epsilon-guarded against divide-by-zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .models import Candle

EPS = 1e-12


# ── primitives ───────────────────────────────────────────────────────────────

def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) == 0:
        return out
    alpha = 2.0 / (n + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def wma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    w = np.arange(1, n + 1, dtype=float)
    denom = w.sum()
    for i in range(n - 1, len(x)):
        out[i] = float(np.dot(x[i - n + 1: i + 1], w) / denom)
    return out


def hma(x: np.ndarray, n: int) -> np.ndarray:
    half = max(1, n // 2)
    sqrt_n = max(1, int(round(np.sqrt(n))))
    raw = 2.0 * wma(x, half) - wma(x, n)
    # leading NaNs poison the final wma; fill with first finite value
    first = np.argmax(np.isfinite(raw)) if np.isfinite(raw).any() else len(raw)
    raw2 = raw.copy()
    if first < len(raw2):
        raw2[:first] = raw2[first]
    return wma(raw2, sqrt_n)


def roc(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) > n:
        out[n:] = (x[n:] - x[:-n]) / np.maximum(np.abs(x[:-n]), EPS) * 100.0
    return out


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    tr = h - l
    if len(c) > 1:
        prev = np.roll(c, 1)
        tr1 = np.abs(h - prev)
        tr2 = np.abs(l - prev)
        tr = np.maximum(tr, np.maximum(tr1, tr2))
        tr[0] = h[0] - l[0]
    return tr


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    tr = true_range(h, l, c)
    out = np.full(len(tr), np.nan)
    if len(tr) < n:
        return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def dmi(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int, smooth: int):
    """Returns (+DI, -DI, ADX)."""
    m = len(h)
    plus_di = np.full(m, np.nan)
    minus_di = np.full(m, np.nan)
    adx = np.full(m, np.nan)
    if m < n + smooth + 1:
        return plus_di, minus_di, adx
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)[1:]

    def wilder(x: np.ndarray, p: int) -> np.ndarray:
        o = np.full(len(x), np.nan)
        if len(x) < p:
            return o
        o[p - 1] = x[:p].sum()
        for i in range(p, len(x)):
            o[i] = o[i - 1] - o[i - 1] / p + x[i]
        return o

    atr_s = wilder(tr, n)
    pdm_s = wilder(plus_dm, n)
    mdm_s = wilder(minus_dm, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100.0 * pdm_s / np.maximum(atr_s, EPS)
        mdi = 100.0 * mdm_s / np.maximum(atr_s, EPS)
        dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, EPS)
    adx_raw = np.full(len(dx), np.nan)
    valid = np.where(np.isfinite(dx))[0]
    if len(valid) >= smooth:
        s = valid[0]
        adx_raw[s + smooth - 1] = np.nanmean(dx[s:s + smooth])
        for i in range(s + smooth, len(dx)):
            adx_raw[i] = (adx_raw[i - 1] * (smooth - 1) + dx[i]) / smooth
    plus_di[1:] = pdi
    minus_di[1:] = mdi
    adx[1:] = adx_raw
    return plus_di, minus_di, adx


def choppiness(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(h), np.nan)
    tr = true_range(h, l, c)
    for i in range(n, len(h)):
        tr_sum = tr[i - n + 1: i + 1].sum()
        hh = h[i - n + 1: i + 1].max()
        ll = l[i - n + 1: i + 1].min()
        rng = max(hh - ll, EPS)
        out[i] = 100.0 * np.log10(max(tr_sum / rng, EPS)) / np.log10(n)
    return out


# ── bundles ──────────────────────────────────────────────────────────────────

@dataclass
class EntryIndicators:
    """15M bundle (last CLOSED bar values plus arrays where needed)."""
    hma_fast: np.ndarray
    hma_slow: np.ndarray
    atr: np.ndarray
    roc: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray
    adx: np.ndarray
    chop: np.ndarray
    volume_sma: np.ndarray
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    opens: np.ndarray
    volumes: np.ndarray
    timestamps: np.ndarray

    # derived scalars (last closed bar)
    volume_ratio: float = 1.0
    body_atr: float = 0.0
    body_ratio: float = 0.0
    bull_close_quality: float = 0.0
    bear_close_quality: float = 0.0
    long_extension_atr: float = 0.0
    short_extension_atr: float = 0.0
    hma_spread_atr: float = 0.0
    roc_acceleration: float = 0.0
    di_spread_long: float = 0.0
    di_spread_short: float = 0.0

    @property
    def last_atr(self) -> float:
        v = float(self.atr[-1])
        return v if np.isfinite(v) and v > 0 else 0.0

    def val(self, arr: np.ndarray, k: int = -1) -> float:
        v = float(arr[k])
        return v if np.isfinite(v) else 0.0


@dataclass
class ContextIndicators:
    """1H / 4H bundle."""
    ema20: np.ndarray
    ema50: np.ndarray
    hma20: np.ndarray
    atr: np.ndarray
    roc9: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray
    adx: np.ndarray
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    opens: np.ndarray
    volumes: np.ndarray
    timestamps: np.ndarray

    @property
    def hma20_slope(self) -> float:
        if len(self.hma20) < 2 or not (np.isfinite(self.hma20[-1]) and np.isfinite(self.hma20[-2])):
            return 0.0
        return float(self.hma20[-1] - self.hma20[-2])

    def val(self, arr: np.ndarray, k: int = -1) -> float:
        v = float(arr[k])
        return v if np.isfinite(v) else 0.0


def _arrays(candles: list) -> tuple:
    o = np.array([c.open for c in candles], dtype=float)
    h = np.array([c.high for c in candles], dtype=float)
    l = np.array([c.low for c in candles], dtype=float)
    cl = np.array([c.close for c in candles], dtype=float)
    v = np.array([c.volume for c in candles], dtype=float)
    ts = np.array([c.timestamp for c in candles], dtype=np.int64)
    return o, h, l, cl, v, ts


class IndicatorEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def calculate_entry(self, candles: list) -> Optional[EntryIndicators]:
        c = self.cfg
        if len(candles) < max(c.hma_slow * 3, c.adx_length * 3, c.chop_length + 2,
                              c.volume_sma + 2, c.roc_length + 2):
            return None
        o, h, l, cl, v, ts = _arrays(candles)
        atr_a = atr(h, l, cl, c.atr_length)
        pdi, mdi, adx_a = dmi(h, l, cl, c.adx_length, c.adx_smoothing)
        ind = EntryIndicators(
            hma_fast=hma(cl, c.hma_fast), hma_slow=hma(cl, c.hma_slow),
            atr=atr_a, roc=roc(cl, c.roc_length),
            plus_di=pdi, minus_di=mdi, adx=adx_a,
            chop=choppiness(h, l, cl, c.chop_length),
            volume_sma=sma(v, c.volume_sma),
            closes=cl, highs=h, lows=l, opens=o, volumes=v, timestamps=ts,
        )
        a = ind.last_atr or EPS
        last = candles[-1]
        vs = float(ind.volume_sma[-1]) if np.isfinite(ind.volume_sma[-1]) else 0.0
        ind.volume_ratio = float(last.volume / max(vs, EPS)) if vs > 0 else 1.0
        ind.body_atr = last.body / a
        ind.body_ratio = last.body / last.range
        ind.bull_close_quality = last.bull_close_quality
        ind.bear_close_quality = last.bear_close_quality
        hs = ind.val(ind.hma_slow)
        ind.long_extension_atr = (last.close - hs) / a
        ind.short_extension_atr = (hs - last.close) / a
        ind.hma_spread_atr = abs(ind.val(ind.hma_fast) - hs) / a
        r_now, r_prev = ind.val(ind.roc), ind.val(ind.roc, -2)
        ind.roc_acceleration = r_now - r_prev
        ind.di_spread_long = ind.val(ind.plus_di) - ind.val(ind.minus_di)
        ind.di_spread_short = -ind.di_spread_long
        return ind

    def calculate_context(self, candles: list) -> Optional[ContextIndicators]:
        c = self.cfg
        if len(candles) < 60:
            return None
        o, h, l, cl, v, ts = _arrays(candles)
        pdi, mdi, adx_a = dmi(h, l, cl, c.adx_length, c.adx_smoothing)
        return ContextIndicators(
            ema20=ema(cl, 20), ema50=ema(cl, 50), hma20=hma(cl, 20),
            atr=atr(h, l, cl, c.atr_length), roc9=roc(cl, c.roc_1h),
            plus_di=pdi, minus_di=mdi, adx=adx_a,
            closes=cl, highs=h, lows=l, opens=o, volumes=v, timestamps=ts,
        )

    # 4H uses the same bundle shape
    calculate_macro = calculate_context
