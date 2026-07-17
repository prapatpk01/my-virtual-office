"""Confirmed Swing Engine (spec §7) — pivots confirm only after right bars.

No future pivots: a swing's confirmed_at is the OPEN timestamp of the bar
that completes the right-side window, and callers must only consume swings
whose confirmed_at <= the bar being evaluated.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .models import Candle, SwingPoint


class SwingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def calculate(self, candles: list, timeframe: str) -> list:
        """Return confirmed SwingPoints (oldest→newest). O(n·(L+R))."""
        L, R = self.cfg.swing_left_bars, self.cfg.swing_right_bars
        n = len(candles)
        if n < L + R + 3:
            return []
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        atr_ref = float(np.mean(highs[-50:] - lows[-50:])) if n >= 50 else float(np.mean(highs - lows))
        atr_ref = max(atr_ref, 1e-12)

        out: list = []
        # pivots are only knowable up to index n-1-R
        for i in range(L, n - R):
            h, l = highs[i], lows[i]
            if all(h > highs[j] for j in range(i - L, i)) and \
               all(h >= highs[j] for j in range(i + 1, i + R + 1)):
                disp = (h - float(np.min(lows[max(0, i - L): i + 1]))) / atr_ref
                out.append(SwingPoint(
                    timeframe=timeframe, swing_type="high", price=float(h),
                    timestamp=int(candles[i].timestamp),
                    confirmed_at=int(candles[i + R].timestamp),
                    strength=min(disp, 5.0), displacement_atr=disp,
                    broken=bool(np.any(closes[i + R:] > h)),
                ))
            if all(l < lows[j] for j in range(i - L, i)) and \
               all(l <= lows[j] for j in range(i + 1, i + R + 1)):
                disp = (float(np.max(highs[max(0, i - L): i + 1])) - l) / atr_ref
                out.append(SwingPoint(
                    timeframe=timeframe, swing_type="low", price=float(l),
                    timestamp=int(candles[i].timestamp),
                    confirmed_at=int(candles[i + R].timestamp),
                    strength=min(disp, 5.0), displacement_atr=disp,
                    broken=bool(np.any(closes[i + R:] < l)),
                ))
        out.sort(key=lambda s: s.timestamp)
        return out[-40:]


def swings_of(swings: list, kind: str, unbroken_only: bool = False) -> list:
    xs = [s for s in swings if s.swing_type == kind]
    if unbroken_only:
        xs = [s for s in xs if not s.broken]
    return xs
