"""
Pattern Gate — Layer 3 entry confirmation (8 candlestick/structure patterns).

Any 1 of 8 patterns must fire in the signal direction for the gate to pass.
Operates on the primary timeframe (15m) OHLCV list.

Enable via env var:  TCI_PATTERN_GATE=1

Fire rates on BTC+XAU 15m data (Jan-May 2026):
  Rejection Candle     ~10%   Hammer / Shooting Star
  Engulfing            ~ 6%   Body fully engulfs prior body
  BOS-20               ~ 9%   Close breaks 20-bar high/low
  Three Soldiers/Crows ~ 4%   3 trending bars closing near extreme
  Volume Spike         ~18%   Volume ≥ 1.5× 20-bar avg
  Inside Bar Breakout  ~ 5%   Breakout of inside-bar consolidation
  Morning/Evening Star ~ 3%   3-bar reversal with small middle body
  MACD Fresh Cross     ~15%   MACD histogram sign change (1-2 bars)

Backtest result (BTC+XAU Jan-May 2026, $500 balance):
  Baseline (no gate)   T=219  WR=53.4%  PnL=+$399  MaxDD=$50  PnL/DD=7.97
  8-pattern gate       T=191  WR=55.0%  PnL=+$417  MaxDD=$38  PnL/DD=11.08
"""
from __future__ import annotations

import pandas as pd

from .connectors.base import OHLCV


# ── Individual pattern detectors ──────────────────────────────────────────────
# Each returns (fires_long: bool, fires_short: bool).

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
    bull = cur.close > cur.open and prev.close < prev.open and clo < plo and chi > phi
    bear = cur.close < cur.open and prev.close > prev.open and clo < plo and chi > phi
    return bull, bear


def _pat_bos20(candles: list) -> tuple[bool, bool]:
    """Break of Structure: close breaks above/below the 20-bar high/low."""
    if len(candles) < 21:
        return False, False
    last = candles[-1]; lb = candles[-21:-1]
    return (last.close > max(b.high for b in lb)), (last.close < min(b.low for b in lb))


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


def _pat_volume_spike(candles: list) -> tuple[bool, bool]:
    """Volume ≥ 1.5× 20-bar average (confirms momentum, non-directional)."""
    if len(candles) < 21:
        return False, False
    avg = sum(b.volume for b in candles[-21:-1]) / 20
    if avg <= 0:
        return False, False
    spike = candles[-1].volume >= 1.5 * avg
    return spike, spike


def _pat_inside_bar_breakout(candles: list) -> tuple[bool, bool]:
    """Inside Bar Breakout: prior bar range inside the mother bar, then breakout."""
    if len(candles) < 3:
        return False, False
    mother = candles[-3]; inside = candles[-2]; breakout = candles[-1]
    if not (inside.high <= mother.high and inside.low >= mother.low):
        return False, False
    return (breakout.close > mother.high), (breakout.close < mother.low)


def _pat_morning_evening_star(candles: list) -> tuple[bool, bool]:
    """Morning Star (bull) / Evening Star (bear) — 3-bar reversal.
    Bar 1: large body, Bar 2: small star body, Bar 3: large opposite body."""
    if len(candles) < 3:
        return False, False
    b1 = candles[-3]; b2 = candles[-2]; b3 = candles[-1]

    def body_frac(b):
        rng = b.high - b.low + 1e-10
        return abs(b.close - b.open) / rng

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


# ── Gate entry point ──────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, object]] = [
    ("Rejection",        _pat_rejection),
    ("Engulfing",        _pat_engulfing),
    ("BOS-20",           _pat_bos20),
    ("3-Soldiers",       _pat_three_soldiers),
    ("Vol-Spike",        _pat_volume_spike),
    ("InsideBarBreak",   _pat_inside_bar_breakout),
    ("Morning/Eve-Star", _pat_morning_evening_star),
    ("MACD-Cross",       _pat_macd_cross),
]


def pattern_gate_passes(candles: list, side: str) -> tuple[bool, str]:
    """Check whether any of the 8 patterns fires for the given direction.

    Args:
        candles: List of OHLCV objects (primary timeframe, most-recent last).
        side:    "long" or "short"

    Returns:
        (passes, reason)
          passes=True  → at least one pattern confirmed; reason lists which ones.
          passes=False → no pattern fired; reason = "no_pattern".
    """
    fired: list[str] = []
    for name, fn in _PATTERNS:
        try:
            bl, bs = fn(candles)
            if bl if side == "long" else bs:
                fired.append(name)
        except Exception:
            pass
    if fired:
        return True, "+".join(fired)
    return False, "no_pattern"
