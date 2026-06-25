"""
Pattern Gate — Layer 3 entry confirmation (13 candlestick/structure patterns).

Any 1 of 13 patterns must fire in the signal direction for the gate to pass.
Operates on the primary timeframe (15m) OHLCV list.

Enable via env var:  TCI_PATTERN_GATE=1

Pattern library (fire rates estimated on BTC+XAU 15m Jan-May 2026):
  ── Candlestick ──────────────────────────────────────────────────────────
  Rejection Candle     ~10%   Hammer / Shooting Star (wick ≥ 2× body)
  Engulfing            ~ 6%   Body fully engulfs prior body
  Three Soldiers/Crows ~ 4%   3 trending bars closing near extreme
  Morning/Evening Star ~ 3%   3-bar reversal with small middle body
  Tweezer Top/Bottom   ~ 5%   Two bars with matching high or low (±0.1%)
  ── Structure ────────────────────────────────────────────────────────────
  BOS-20               ~ 9%   Close breaks 20-bar high/low
  Inside Bar Breakout  ~ 5%   Breakout after inside-bar consolidation
  Bull/Bear Flag       ~ 6%   Tight consolidation after impulsive move
  Double Top/Bottom    ~ 4%   Two equal peaks/troughs then breakout
  ── Momentum / Indicator ─────────────────────────────────────────────────
  Volume Spike         ~18%   Volume ≥ 1.5× 20-bar avg
  MACD Fresh Cross     ~15%   MACD histogram sign change (1-2 bars)
  RSI Divergence       ~ 5%   Price extreme not confirmed by RSI
  Squeeze Breakout     ~ 8%   BB compression → expansion with direction

Backtest result (BTC+XAU Jan-May 2026, $500 balance):
  Baseline (no gate)   T=219  WR=53.4%  PnL=+$399  MaxDD=$50  PnL/DD=7.97
  8-pattern gate       T=191  WR=55.0%  PnL=+$417  MaxDD=$38  PnL/DD=11.08
"""
from __future__ import annotations

import logging
import pandas as pd

from .connectors.base import OHLCV

logger = logging.getLogger(__name__)


# ── Candlestick patterns ──────────────────────────────────────────────────────

def _pat_rejection(candles: list) -> tuple[bool, bool]:
    """Hammer (long) / Shooting Star (short): wick ≥ 2× body."""
    if len(candles) < 1:
        return False, False
    b = candles[-1]
    body = abs(b.close - b.open)
    if body < 1e-10:
        return False, False
    bl = min(b.open, b.close); bh = max(b.open, b.close)
    lw = bl - b.low; uw = b.high - bh
    return (lw >= 2 * body and uw <= body), (uw >= 2 * body and lw <= body)


def _pat_engulfing(candles: list) -> tuple[bool, bool]:
    """Bullish / Bearish engulfing: current body fully covers prior body."""
    if len(candles) < 2:
        return False, False
    cur = candles[-1]; prev = candles[-2]
    clo = min(cur.open, cur.close); chi = max(cur.open, cur.close)
    plo = min(prev.open, prev.close); phi = max(prev.open, prev.close)
    bull = cur.close > cur.open and prev.close < prev.open and clo <= plo and chi >= phi
    bear = cur.close < cur.open and prev.close > prev.open and clo <= plo and chi >= phi
    return bull, bear


def _pat_three_soldiers(candles: list) -> tuple[bool, bool]:
    """Three White Soldiers (long) / Three Black Crows (short).
    3 consecutive bars each closing near their extreme."""
    if len(candles) < 3:
        return False, False
    bars = candles[-3:]

    def rng(b):
        return b.high - b.low + 1e-10

    bull = (
        all(b.close > b.open for b in bars)
        and all((b.high - b.close) / rng(b) <= 0.30 for b in bars)
        and bars[1].close > bars[0].close
        and bars[2].close > bars[1].close
    )
    bear = (
        all(b.close < b.open for b in bars)
        and all((b.close - b.low) / rng(b) <= 0.30 for b in bars)
        and bars[1].close < bars[0].close
        and bars[2].close < bars[1].close
    )
    return bull, bear


def _pat_morning_evening_star(candles: list) -> tuple[bool, bool]:
    """Morning Star (bull) / Evening Star (bear) — 3-bar reversal.
    Bar 1: large body, Bar 2: small star body, Bar 3: large opposite body."""
    if len(candles) < 3:
        return False, False
    b1 = candles[-3]; b2 = candles[-2]; b3 = candles[-1]

    def body_frac(b):
        return abs(b.close - b.open) / (b.high - b.low + 1e-10)

    morning = (
        b1.close < b1.open and body_frac(b1) > 0.50
        and body_frac(b2) < 0.35
        and b3.close > b3.open and body_frac(b3) > 0.50
    )
    evening = (
        b1.close > b1.open and body_frac(b1) > 0.50
        and body_frac(b2) < 0.35
        and b3.close < b3.open and body_frac(b3) > 0.50
    )
    return morning, evening


def _pat_tweezer(candles: list) -> tuple[bool, bool]:
    """Tweezer Bottom (long) / Tweezer Top (short).
    Two consecutive bars with matching lows (bottom) or highs (top) within 0.1%."""
    if len(candles) < 2:
        return False, False
    cur = candles[-1]; prev = candles[-2]
    tol = 0.001  # 0.1%
    bottom = (
        abs(cur.low - prev.low) / (prev.low + 1e-10) <= tol
        and cur.close > cur.open  # second bar bullish confirms reversal
    )
    top = (
        abs(cur.high - prev.high) / (prev.high + 1e-10) <= tol
        and cur.close < cur.open  # second bar bearish confirms reversal
    )
    return bottom, top


# ── Structure patterns ────────────────────────────────────────────────────────

def _pat_bos20(candles: list) -> tuple[bool, bool]:
    """Break of Structure: close breaks above/below the 20-bar high/low."""
    if len(candles) < 21:
        return False, False
    last = candles[-1]; lb = candles[-21:-1]
    return (last.close > max(b.high for b in lb)), (last.close < min(b.low for b in lb))


def _pat_inside_bar_breakout(candles: list) -> tuple[bool, bool]:
    """Inside Bar Breakout: prior bar range inside the mother bar, then breakout."""
    if len(candles) < 3:
        return False, False
    mother = candles[-3]; inside = candles[-2]; cur = candles[-1]
    if not (inside.high <= mother.high and inside.low >= mother.low):
        return False, False
    return (cur.close > mother.high), (cur.close < mother.low)


def _pat_flag(candles: list) -> tuple[bool, bool]:
    """Bull Flag (long) / Bear Flag (short).
    Impulsive flagpole move followed by tight consolidation, then breakout."""
    if len(candles) < 20:
        return False, False
    pole = candles[-20:-5]    # 15-bar flagpole
    flag = candles[-5:-1]     # 4-bar flag consolidation
    cur  = candles[-1]

    pole_high = max(b.high for b in pole)
    pole_low  = min(b.low  for b in pole)
    pole_move = (pole_high - pole_low) / (pole_low + 1e-10)
    if pole_move < 0.005:     # flagpole must be at least 0.5% move
        return False, False

    flag_high = max(b.high for b in flag)
    flag_low  = min(b.low  for b in flag)
    flag_range = (flag_high - flag_low) / (flag_low + 1e-10)
    is_tight = flag_range < pole_move * 0.40  # flag ≤ 40% of pole width

    pole_bull = pole[-1].close > pole[0].close
    pole_bear = pole[-1].close < pole[0].close

    bull_flag = pole_bull and is_tight and cur.close > flag_high
    bear_flag = pole_bear and is_tight and cur.close < flag_low
    return bull_flag, bear_flag


def _pat_double_top_bottom(candles: list) -> tuple[bool, bool]:
    """Double Top (short) / Double Bottom (long).
    Two swing peaks/troughs within 0.5%, then confirmed breakout through neckline."""
    if len(candles) < 40:
        return False, False
    window = candles[-40:]
    highs = [b.high  for b in window]
    lows  = [b.low   for b in window]
    tol   = 0.005   # 0.5% matching tolerance

    # Swing-high detection: higher than 2 bars on each side
    swing_highs = [(i, highs[i]) for i in range(2, len(window) - 2)
                   if highs[i] > highs[i-1] and highs[i] > highs[i-2]
                   and highs[i] > highs[i+1] and highs[i] > highs[i+2]]
    swing_lows  = [(i, lows[i])  for i in range(2, len(window) - 2)
                   if lows[i]  < lows[i-1]  and lows[i]  < lows[i-2]
                   and lows[i]  < lows[i+1]  and lows[i]  < lows[i+2]]

    cur = window[-1]

    double_top = False
    if len(swing_highs) >= 2:
        (i1, h1), (i2, h2) = swing_highs[-2], swing_highs[-1]
        if i2 > i1 + 3 and abs(h1 - h2) / (h1 + 1e-10) <= tol:
            neckline = min(lows[i1:i2 + 1])
            double_top = cur.close < neckline

    double_bottom = False
    if len(swing_lows) >= 2:
        (i1, l1), (i2, l2) = swing_lows[-2], swing_lows[-1]
        if i2 > i1 + 3 and abs(l1 - l2) / (l1 + 1e-10) <= tol:
            neckline = max(highs[i1:i2 + 1])
            double_bottom = cur.close > neckline

    return double_bottom, double_top


# ── Momentum / Indicator patterns ─────────────────────────────────────────────

def _pat_volume_spike(candles: list) -> tuple[bool, bool]:
    """Volume ≥ 1.5× 20-bar average (confirms momentum, non-directional)."""
    if len(candles) < 21:
        return False, False
    avg = sum(b.volume for b in candles[-21:-1]) / 20
    if avg <= 0:
        return False, False
    spike = candles[-1].volume >= 1.5 * avg
    return spike, spike


def _pat_macd_cross(candles: list) -> tuple[bool, bool]:
    """Fresh MACD cross: histogram changed sign within the last 2 bars."""
    if len(candles) < 35:
        return False, False
    closes = pd.Series([c.close for c in candles])
    macd = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    bull_cross = float(hist.iloc[-2]) <= 0 < float(hist.iloc[-1])
    bear_cross = float(hist.iloc[-2]) >= 0 > float(hist.iloc[-1])
    return bull_cross, bear_cross


def _pat_rsi_divergence(candles: list) -> tuple[bool, bool]:
    """RSI Divergence over last 20 bars.
    Bullish: price makes lower low but RSI makes higher low (hidden strength).
    Bearish: price makes higher high but RSI makes lower high (hidden weakness)."""
    if len(candles) < 30:
        return False, False
    closes = pd.Series([c.close for c in candles])
    delta  = closes.diff()
    rsi    = 100 - 100 / (1 + delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                              / ((-delta).clip(lower=0).ewm(com=13, adjust=False).mean() + 1e-10))

    w      = 20
    rsi_w  = rsi.iloc[-w:].values
    lows_w = [c.low  for c in candles[-w:]]
    highs_w= [c.high for c in candles[-w:]]
    mid    = w // 2

    # Bullish divergence: price lower low, RSI higher low
    bull_div = (
        lows_w[-1] < min(lows_w[:mid])      # price new low
        and rsi_w[-1] > min(rsi_w[:mid])    # RSI NOT new low
        and float(rsi.iloc[-1]) < 50        # still in oversold zone
    )
    # Bearish divergence: price higher high, RSI lower high
    bear_div = (
        highs_w[-1] > max(highs_w[:mid])    # price new high
        and rsi_w[-1] < max(rsi_w[:mid])    # RSI NOT new high
        and float(rsi.iloc[-1]) > 50        # still in overbought zone
    )
    return bull_div, bear_div


def _pat_squeeze_breakout(candles: list) -> tuple[bool, bool]:
    """Squeeze Breakout: Bollinger Bands were compressed then started expanding.
    Direction determined by price relative to the 20-bar moving average."""
    if len(candles) < 35:
        return False, False
    closes = pd.Series([c.close for c in candles])
    ma20   = closes.rolling(20).mean()
    std20  = closes.rolling(20).std()
    bb_w   = (2 * std20) / (ma20 + 1e-10)   # normalised BB width

    recent = bb_w.iloc[-30:]
    w_min  = float(recent.min()); w_max = float(recent.max())
    w_rng  = w_max - w_min + 1e-10

    # Was squeezed (narrow) anywhere in the 3–5 bars ago window, now expanding for 2+ bars
    squeeze_threshold = w_min + w_rng * 0.25
    was_squeezed = any(float(bb_w.iloc[i]) < squeeze_threshold for i in (-3, -4, -5))
    expanding    = float(bb_w.iloc[-1]) > float(bb_w.iloc[-2]) > float(bb_w.iloc[-3])

    if not (was_squeezed and expanding):
        return False, False

    price = float(closes.iloc[-1]); ma = float(ma20.iloc[-1])
    return (price > ma), (price < ma)


# ── Gate entry point ──────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, object]] = [
    # Candlestick
    ("Rejection",        _pat_rejection),
    ("Engulfing",        _pat_engulfing),
    ("3-Soldiers",       _pat_three_soldiers),
    ("Morning/Eve-Star", _pat_morning_evening_star),
    ("Tweezer",          _pat_tweezer),
    # Structure
    ("BOS-20",           _pat_bos20),
    ("InsideBarBreak",   _pat_inside_bar_breakout),
    ("Flag",             _pat_flag),
    ("DoubleTop/Bot",    _pat_double_top_bottom),
    # Momentum / Indicator
    ("Vol-Spike",        _pat_volume_spike),
    ("MACD-Cross",       _pat_macd_cross),
    ("RSI-Div",          _pat_rsi_divergence),
    ("Squeeze",          _pat_squeeze_breakout),
]


def pattern_gate_passes(candles: list, side: str) -> tuple[bool, str]:
    """Check whether any of the 13 patterns fires for the given direction.

    Args:
        candles: List of OHLCV objects (primary timeframe, most-recent last).
        side:    "long" or "short"

    Returns:
        (passes, reason)
          passes=True  → at least one pattern confirmed; reason lists which ones fired.
          passes=False → no pattern fired; reason = "no_pattern".
    """
    if side not in ("long", "short"):
        raise ValueError(f"pattern_gate_passes: side must be 'long' or 'short', got {side!r}")
    fired: list[str] = []
    for name, fn in _PATTERNS:
        try:
            bl, bs = fn(candles)
            if bl if side == "long" else bs:
                fired.append(name)
        except Exception as _e:
            logger.warning("[PatternGate] %s raised %s", name, _e)
    if fired:
        return True, "+".join(fired)
    return False, "no_pattern"
