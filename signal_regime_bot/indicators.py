"""Shared technical-analysis primitives for live trading and backtesting.

All functions expect *closed* candles.  No function fetches data or uses future
bars.  Swing points are only returned after their right-hand confirmation bars
exist, preventing look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

EPSILON = 1e-12


# ── Moving averages ──────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=max(1, period), adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).rolling(max(1, period)).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    period = max(1, int(period))
    weights = np.arange(1, period + 1, dtype=float)
    return series.astype(float).rolling(period).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    period = max(2, int(period))
    half = max(1, period // 2)
    root = max(1, int(round(np.sqrt(period))))
    return wma(2.0 * wma(series, half) - wma(series, period), root)


def slope_pct(series: pd.Series, lookback: int = 5) -> pd.Series:
    prior = series.shift(max(1, lookback))
    return (series - prior) / prior.replace(0, np.nan) * 100.0


def normalized_slope(series: pd.Series, atr_series: pd.Series, lookback: int = 3) -> pd.Series:
    """Slope measured in ATR units, comparable across BTC, metals and alts."""
    return (series - series.shift(max(1, lookback))) / atr_series.replace(0, np.nan)


# ── Momentum ────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI with symmetric zero-gain/zero-loss handling.

    A one-sided advance must produce RSI=100 and a one-sided decline RSI=0.
    The previous implementation replaced a zero average loss with NaN and then
    filled it with 50, which incorrectly weakened strong bullish trends while
    strong bearish trends still approached zero.  Both-zero windows remain 50.
    """
    period = max(1, int(period))
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.where(avg_loss > EPSILON)
    out = 100.0 - 100.0 / (1.0 + rs)

    gain_zero = avg_gain.abs() <= EPSILON
    loss_zero = avg_loss.abs() <= EPSILON
    out = out.mask(loss_zero & ~gain_zero, 100.0)
    out = out.mask(gain_zero & ~loss_zero, 0.0)
    out = out.mask(gain_zero & loss_zero, 50.0)
    return out.fillna(50.0).clip(0.0, 100.0)


def roc(series: pd.Series, period: int = 9) -> pd.Series:
    prior = series.shift(max(1, period))
    return (series - prior) / prior.replace(0, np.nan) * 100.0


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def vwap(df: pd.DataFrame, window: int = 48) -> pd.Series:
    window = max(1, min(int(window), max(len(df), 1)))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).rolling(window, min_periods=1).sum()
    vol = df["volume"].rolling(window, min_periods=1).sum().replace(0, np.nan)
    return pv / vol


# ── Volatility and trend strength ───────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(
        alpha=1.0 / max(1, period),
        adjust=False,
        min_periods=max(1, period),
    ).mean()


def rolling_percentile(series: pd.Series, lookback: int = 100) -> pd.Series:
    lookback = max(5, int(lookback))

    def _rank(window: np.ndarray) -> float:
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            return np.nan
        current = valid[-1]
        return float((valid <= current).sum() - 1) / max(len(valid) - 1, 1) * 100.0

    return series.rolling(lookback, min_periods=max(20, lookback // 4)).apply(_rank, raw=True)


def atr_percentile(atr_series: pd.Series, lookback: int = 100) -> pd.Series:
    return rolling_percentile(atr_series, lookback)


def bollinger_width(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    mid = sma(df["close"], period)
    std = df["close"].rolling(period).std()
    return (2.0 * mult * std) / mid.replace(0, np.nan) * 100.0


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    tr = true_range(df)
    atr_smoothed = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_smoothed = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_smoothed = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_smoothed / atr_smoothed.replace(0, np.nan)
    minus_di = 100.0 * minus_smoothed / atr_smoothed.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_value = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_value, plus_di, minus_di


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr_sum = true_range(df).rolling(period).sum()
    price_range = (
        df["high"].rolling(period).max() - df["low"].rolling(period).min()
    ).replace(0, np.nan)
    return 100.0 * np.log10(tr_sum / price_range) / np.log10(max(period, 2))


# ── Candle quality ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CandleMetrics:
    body: float
    candle_range: float
    body_atr: float
    body_ratio: float
    bull_close_quality: float
    bear_close_quality: float
    upper_wick: float
    lower_wick: float
    volume_ratio: float
    bullish: bool
    bearish: bool


def candle_metrics(df: pd.DataFrame, atr_value: Optional[float] = None, volume_period: int = 20) -> CandleMetrics:
    if df is None or len(df) == 0:
        return CandleMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, False, False)
    row = df.iloc[-1]
    op, hi, lo, cl = map(float, (row["open"], row["high"], row["low"], row["close"]))
    rng = max(hi - lo, EPSILON)
    body = abs(cl - op)
    if atr_value is None:
        atr_value = float(atr(df, 14).iloc[-1]) if len(df) >= 14 else 0.0
    body_atr = body / max(float(atr_value or 0.0), EPSILON)
    volume = float(row.get("volume", 0.0))
    if len(df) > 1:
        base = float(df["volume"].iloc[-(volume_period + 1):-1].mean())
    else:
        base = 0.0
    volume_ratio = volume / max(base, EPSILON) if base > 0 else 1.0
    return CandleMetrics(
        body=body,
        candle_range=rng,
        body_atr=body_atr,
        body_ratio=body / rng,
        bull_close_quality=(cl - lo) / rng,
        bear_close_quality=(hi - cl) / rng,
        upper_wick=hi - max(op, cl),
        lower_wick=min(op, cl) - lo,
        volume_ratio=volume_ratio,
        bullish=cl > op,
        bearish=cl < op,
    )


def bullish_trigger_candle(
    metrics: CandleMetrics,
    min_body_atr: float = 0.15,
    min_close_quality: float = 0.62,
) -> bool:
    return (
        metrics.bullish
        and metrics.body_atr >= min_body_atr
        and metrics.bull_close_quality >= min_close_quality
    )


def bearish_trigger_candle(
    metrics: CandleMetrics,
    min_body_atr: float = 0.15,
    min_close_quality: float = 0.62,
) -> bool:
    return (
        metrics.bearish
        and metrics.body_atr >= min_body_atr
        and metrics.bear_close_quality >= min_close_quality
    )


# ── Confirmed market structure ─────────────────────────────────────────────

@dataclass(frozen=True)
class SwingPoint:
    position: int
    timestamp: object
    price: float
    kind: str  # HIGH | LOW


def swing_pivots(
    highs: pd.Series,
    lows: pd.Series,
    left: int = 3,
    right: int = 3,
) -> tuple[list[int], list[int]]:
    left, right = max(1, int(left)), max(1, int(right))
    high_values = highs.astype(float).to_numpy()
    low_values = lows.astype(float).to_numpy()
    pivot_highs: list[int] = []
    pivot_lows: list[int] = []
    for i in range(left, len(high_values) - right):
        hw = high_values[i - left : i + right + 1]
        lw = low_values[i - left : i + right + 1]
        if np.isfinite(high_values[i]) and high_values[i] == np.nanmax(hw) and np.nanargmax(hw) == left:
            pivot_highs.append(i)
        if np.isfinite(low_values[i]) and low_values[i] == np.nanmin(lw) and np.nanargmin(lw) == left:
            pivot_lows.append(i)
    return pivot_highs, pivot_lows


def confirmed_swings(
    highs: pd.Series,
    lows: pd.Series,
    left: int = 3,
    right: int = 3,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    ph, pl = swing_pivots(highs, lows, left, right)
    high_points = [
        SwingPoint(i, highs.index[i], float(highs.iloc[i]), "HIGH") for i in ph
    ]
    low_points = [
        SwingPoint(i, lows.index[i], float(lows.iloc[i]), "LOW") for i in pl
    ]
    return high_points, low_points


def market_structure(
    highs: pd.Series,
    lows: pd.Series,
    left: int = 3,
    right: int = 3,
) -> str:
    ph, pl = swing_pivots(highs, lows, left, right)
    if len(ph) < 2 or len(pl) < 2:
        return "MIXED"
    hh = float(highs.iloc[ph[-1]]) > float(highs.iloc[ph[-2]])
    hl = float(lows.iloc[pl[-1]]) > float(lows.iloc[pl[-2]])
    lh = float(highs.iloc[ph[-1]]) < float(highs.iloc[ph[-2]])
    ll = float(lows.iloc[pl[-1]]) < float(lows.iloc[pl[-2]])
    if hh and hl:
        return "HH_HL"
    if lh and ll:
        return "LH_LL"
    return "MIXED"


def structure_flags(
    highs: pd.Series,
    lows: pd.Series,
    left: int = 3,
    right: int = 3,
) -> dict[str, bool]:
    ph, pl = swing_pivots(highs, lows, left, right)
    return {
        "higher_high": len(ph) >= 2 and float(highs.iloc[ph[-1]]) > float(highs.iloc[ph[-2]]),
        "lower_high": len(ph) >= 2 and float(highs.iloc[ph[-1]]) < float(highs.iloc[ph[-2]]),
        "higher_low": len(pl) >= 2 and float(lows.iloc[pl[-1]]) > float(lows.iloc[pl[-2]]),
        "lower_low": len(pl) >= 2 and float(lows.iloc[pl[-1]]) < float(lows.iloc[pl[-2]]),
    }


def recent_swing_levels(
    highs: pd.Series,
    lows: pd.Series,
    left: int = 3,
    right: int = 3,
) -> tuple[float, float]:
    ph, pl = swing_pivots(highs, lows, left, right)
    swing_high = float(highs.iloc[ph[-1]]) if ph else float("nan")
    swing_low = float(lows.iloc[pl[-1]]) if pl else float("nan")
    return swing_high, swing_low


def nearest_confirmed_levels(
    df: Optional[pd.DataFrame],
    price: float,
    left: int = 3,
    right: int = 3,
) -> tuple[Optional[float], Optional[float]]:
    """Nearest confirmed support below and resistance above `price`."""
    if df is None or len(df) < left + right + 3:
        return None, None
    highs, lows = confirmed_swings(df["high"], df["low"], left, right)
    supports = [p.price for p in lows if p.price < price]
    resistances = [p.price for p in highs if p.price > price]
    return (max(supports) if supports else None, min(resistances) if resistances else None)


def latest_bos(
    df: pd.DataFrame,
    direction: str,
    left: int = 3,
    right: int = 3,
    min_body_atr: float = 0.18,
) -> tuple[bool, Optional[float]]:
    if df is None or len(df) < left + right + 5:
        return False, None
    atr_value = float(atr(df, 14).iloc[-1])
    metrics = candle_metrics(df, atr_value)
    ph, pl = swing_pivots(df["high"], df["low"], left, right)
    close = float(df["close"].iloc[-1])
    previous_close = float(df["close"].iloc[-2])
    if direction.upper() == "LONG" and ph:
        level = float(df["high"].iloc[ph[-1]])
        return close > level and previous_close <= level and metrics.body_atr >= min_body_atr, level
    if direction.upper() == "SHORT" and pl:
        level = float(df["low"].iloc[pl[-1]])
        return close < level and previous_close >= level and metrics.body_atr >= min_body_atr, level
    return False, None


def sweep_reclaim(
    df: pd.DataFrame,
    direction: str,
    level: Optional[float],
    min_wick_body_ratio: float = 1.2,
) -> bool:
    if df is None or len(df) == 0 or level is None:
        return False
    metrics = candle_metrics(df)
    row = df.iloc[-1]
    if direction.upper() == "LONG":
        return (
            float(row["low"]) < level < float(row["close"])
            and metrics.lower_wick >= max(metrics.body * min_wick_body_ratio, EPSILON)
        )
    return (
        float(row["high"]) > level > float(row["close"])
        and metrics.upper_wick >= max(metrics.body * min_wick_body_ratio, EPSILON)
    )


def recent_cross_above(price: pd.Series, level: pd.Series, lookback: int = 5) -> bool:
    if len(price) < lookback + 2 or pd.isna(level.iloc[-1]) or not price.iloc[-1] > level.iloc[-1]:
        return False
    return any(price.iloc[-1 - back] <= level.iloc[-1 - back] for back in range(1, lookback + 1))


def recent_cross_below(price: pd.Series, level: pd.Series, lookback: int = 5) -> bool:
    if len(price) < lookback + 2 or pd.isna(level.iloc[-1]) or not price.iloc[-1] < level.iloc[-1]:
        return False
    return any(price.iloc[-1 - back] >= level.iloc[-1 - back] for back in range(1, lookback + 1))


def cross_count(fast: pd.Series, slow: pd.Series, lookback: int = 8) -> int:
    relation = np.sign((fast - slow).fillna(0.0).to_numpy())
    relation = relation[-max(2, lookback + 1) :]
    return int(np.sum(relation[1:] * relation[:-1] < 0))



def latest_bos_event(
    df: pd.DataFrame,
    direction: str,
    left: int = 3,
    right: int = 3,
    min_body_atr: float = 0.18,
    scan_bars: int = 40,
) -> tuple[Optional[pd.Timestamp], Optional[float]]:
    """Return the newest closed-candle BOS event without repeated pivot scans.

    A pivot at position ``p`` is eligible only when ``p + right <= end``.
    This is equivalent to evaluating :func:`latest_bos` on every historical
    prefix, but computes pivots and ATR once, making restart/re-entry checks
    practical in both live trading and backtests.
    """
    if df is None or len(df) < left + right + 8:
        return None, None
    ph, pl = swing_pivots(df["high"], df["low"], left, right)
    pivots = ph if direction.upper() == "LONG" else pl
    if not pivots:
        return None, None
    atr_s = atr(df, 14)
    opens = df["open"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()
    start_end = max(left + right + 7, len(df) - max(int(scan_bars), 10))
    event_ts: Optional[pd.Timestamp] = None
    event_level: Optional[float] = None
    pivot_pos = -1
    for end_pos in range(start_end, len(df)):
        eligible_limit = end_pos - max(1, int(right))
        while pivot_pos + 1 < len(pivots) and pivots[pivot_pos + 1] <= eligible_limit:
            pivot_pos += 1
        if pivot_pos < 0 or end_pos < 1:
            continue
        p = pivots[pivot_pos]
        level = float(highs[p] if direction.upper() == "LONG" else lows[p])
        atr_value = safe_float(atr_s.iloc[end_pos], 0.0)
        body_atr = abs(closes[end_pos] - opens[end_pos]) / max(atr_value, EPSILON)
        if body_atr < min_body_atr:
            continue
        hit = (
            closes[end_pos] > level and closes[end_pos - 1] <= level
            if direction.upper() == "LONG"
            else closes[end_pos] < level and closes[end_pos - 1] >= level
        )
        if hit:
            event_ts = pd.Timestamp(df.index[end_pos])
            event_level = level
    return event_ts, event_level


def latest_confirmed_swing_confirmation(
    df: pd.DataFrame,
    kind: str,
    left: int = 3,
    right: int = 3,
) -> tuple[Optional[pd.Timestamp], Optional[float]]:
    """Return confirmation time and price of the newest confirmed swing.

    A pivot at position ``i`` becomes knowable only after ``right`` additional
    candles.  The returned timestamp is therefore the confirmation candle, not
    the pivot candle itself.
    """
    if df is None or len(df) < left + right + 3:
        return None, None
    highs, lows = confirmed_swings(df["high"], df["low"], left, right)
    points = highs if kind.upper() == "HIGH" else lows
    if not points:
        return None, None
    point = points[-1]
    confirmation_pos = point.position + max(1, int(right))
    if confirmation_pos >= len(df):
        return None, None
    return pd.Timestamp(df.index[confirmation_pos]), float(point.price)


def compression_ratio(df: pd.DataFrame, recent: int = 4, normal: int = 20) -> float:
    """Recent average candle range divided by its longer baseline."""
    if df is None or len(df) < normal + 2:
        return 1.0
    ranges = (df["high"] - df["low"]).astype(float)
    recent_avg = safe_float(ranges.iloc[-recent - 1 : -1].mean(), 0.0)
    normal_avg = safe_float(ranges.iloc[-normal - 1 : -1].mean(), 0.0)
    return recent_avg / max(normal_avg, EPSILON) if normal_avg > 0 else 1.0

def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(result) else result
