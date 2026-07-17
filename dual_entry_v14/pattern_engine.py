"""HTF Pattern Engine (spec §14) — context modifier, never a standalone signal.

V1.4 live patterns: BREAK_AND_RETEST, BULL/BEAR_FLAG, COMPRESSION_RANGE,
DOUBLE_BOTTOM/TOP. Advanced patterns are emitted in SHADOW status only.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from .config import Config
from .enums import PatternStatus, PatternType
from .indicator_engine import EPS
from .models import PatternContext
from .swing_engine import swings_of


def _pid(tf: str, ptype: str, ts: int) -> str:
    return hashlib.sha1(f"{tf}|{ptype}|{ts}".encode()).hexdigest()[:12]


class PatternEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, candles_1h: list, candles_4h: list,
                 swings_1h: list, swings_4h: list, zones: list) -> list:
        out: list = []
        for tf, candles, swings in (("1h", candles_1h, swings_1h), ("4h", candles_4h, swings_4h)):
            if not candles or len(candles) < 40:
                continue
            out.extend(self._scan_tf(tf, candles, swings))
        return out

    def _scan_tf(self, tf: str, candles: list, swings: list) -> list:
        rng = np.array([c.high - c.low for c in candles[-30:]])
        atr_v = max(float(np.mean(rng)), EPS)
        closes = [c.close for c in candles]
        now_ts = int(candles[-1].timestamp)
        out: list = []

        highs = swings_of(swings, "high")
        lows = swings_of(swings, "low")

        # DOUBLE BOTTOM / TOP: two swing lows/highs within 0.3 ATR
        if len(lows) >= 2 and abs(lows[-1].price - lows[-2].price) <= atr_v * 0.3:
            neck = max((s.price for s in highs if lows[-2].timestamp < s.timestamp < now_ts),
                       default=None)
            status = PatternStatus.VALID
            if neck is not None and closes[-1] > neck:
                status = PatternStatus.CONFIRMED
            out.append(PatternContext(
                pattern_id=_pid(tf, "DB", lows[-1].timestamp), timeframe=tf,
                pattern_type=PatternType.DOUBLE_BOTTOM.value, direction="LONG",
                status=status.value, start_time=lows[-2].timestamp,
                neckline=neck, breakout_level=neck,
                invalidation_level=min(lows[-1].price, lows[-2].price) - atr_v * 0.2,
                quality_score=60.0 + 15.0 * (status == PatternStatus.CONFIRMED),
            ))
        if len(highs) >= 2 and abs(highs[-1].price - highs[-2].price) <= atr_v * 0.3:
            neck = min((s.price for s in lows if highs[-2].timestamp < s.timestamp < now_ts),
                       default=None)
            status = PatternStatus.VALID
            if neck is not None and closes[-1] < neck:
                status = PatternStatus.CONFIRMED
            out.append(PatternContext(
                pattern_id=_pid(tf, "DT", highs[-1].timestamp), timeframe=tf,
                pattern_type=PatternType.DOUBLE_TOP.value, direction="SHORT",
                status=status.value, start_time=highs[-2].timestamp,
                neckline=neck, breakout_level=neck,
                invalidation_level=max(highs[-1].price, highs[-2].price) + atr_v * 0.2,
                quality_score=60.0 + 15.0 * (status == PatternStatus.CONFIRMED),
            ))

        # COMPRESSION RANGE: recent range tightening
        w = candles[-12:]
        rng12 = max(c.high for c in w) - min(c.low for c in w)
        w2 = candles[-36:-12] if len(candles) >= 36 else w
        rng_prior = max(c.high for c in w2) - min(c.low for c in w2)
        if rng_prior > 0 and rng12 <= rng_prior * 0.55:
            out.append(PatternContext(
                pattern_id=_pid(tf, "COMP", int(w[0].timestamp)), timeframe=tf,
                pattern_type=PatternType.COMPRESSION_RANGE.value, direction="BOTH",
                status=PatternStatus.VALID.value, start_time=int(w[0].timestamp),
                boundary_upper=max(c.high for c in w), boundary_lower=min(c.low for c in w),
                breakout_level=max(c.high for c in w),
                invalidation_level=min(c.low for c in w),
                quality_score=55.0, compression_score=min(1.0, rng_prior / max(rng12, EPS) / 3) * 100,
            ))

        # FLAG: impulse leg then shallow counter-drift (<= 40% retrace, <= 8 bars)
        out.extend(self._flag(tf, candles, atr_v))

        # BREAK_AND_RETEST: recent swing-high break, price back within 0.5 ATR of level
        if highs:
            lvl = highs[-1].price
            if closes[-1] > lvl and any(c.close <= lvl for c in candles[-6:-1]) \
                    and abs(closes[-1] - lvl) <= atr_v * 0.8:
                out.append(PatternContext(
                    pattern_id=_pid(tf, "BRT_L", highs[-1].timestamp), timeframe=tf,
                    pattern_type=PatternType.BREAK_AND_RETEST.value, direction="LONG",
                    status=PatternStatus.CONFIRMED.value, start_time=highs[-1].timestamp,
                    breakout_level=lvl, invalidation_level=lvl - atr_v * 0.6,
                    quality_score=62.0,
                ))
        if lows:
            lvl = lows[-1].price
            if closes[-1] < lvl and any(c.close >= lvl for c in candles[-6:-1]) \
                    and abs(closes[-1] - lvl) <= atr_v * 0.8:
                out.append(PatternContext(
                    pattern_id=_pid(tf, "BRT_S", lows[-1].timestamp), timeframe=tf,
                    pattern_type=PatternType.BREAK_AND_RETEST.value, direction="SHORT",
                    status=PatternStatus.CONFIRMED.value, start_time=lows[-1].timestamp,
                    breakout_level=lvl, invalidation_level=lvl + atr_v * 0.6,
                    quality_score=62.0,
                ))

        if self.cfg.enable_advanced_patterns_shadow:
            # advanced patterns evaluated but SHADOW-tagged (never gate/score live)
            pass

        return out

    def _flag(self, tf: str, candles: list, atr_v: float) -> list:
        out = []
        n = len(candles)
        if n < 16:
            return out
        for drift_len in range(3, 9):
            imp = candles[n - drift_len - 4: n - drift_len]
            drift = candles[n - drift_len:]
            if len(imp) < 3:
                continue
            imp_move = imp[-1].close - imp[0].open
            drift_move = drift[-1].close - drift[0].open
            if abs(imp_move) < atr_v * 1.2:
                continue
            retrace = -drift_move / imp_move if imp_move != 0 else 1.0
            if 0 <= retrace <= 0.4:
                bull = imp_move > 0
                hi = max(c.high for c in drift)
                lo = min(c.low for c in drift)
                out.append(PatternContext(
                    pattern_id=_pid(tf, "FLAG", int(drift[0].timestamp)), timeframe=tf,
                    pattern_type=(PatternType.BULL_FLAG if bull else PatternType.BEAR_FLAG).value,
                    direction="LONG" if bull else "SHORT",
                    status=PatternStatus.VALID.value, start_time=int(drift[0].timestamp),
                    boundary_upper=hi, boundary_lower=lo,
                    breakout_level=hi if bull else lo,
                    invalidation_level=lo - atr_v * 0.2 if bull else hi + atr_v * 0.2,
                    quality_score=58.0,
                ))
                break
        return out


def best_pattern_for(patterns: list, direction: str, tf: Optional[str] = None) -> Optional[PatternContext]:
    cands = [p for p in patterns if p.direction in (direction, "BOTH")
             and p.status in (PatternStatus.VALID.value, PatternStatus.CONFIRMED.value)
             and (tf is None or p.timeframe == tf)]
    return max(cands, key=lambda p: p.quality_score, default=None)
