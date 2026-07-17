"""Support/Resistance Zone Engine (spec §11) — zones, not lines.

Builds PriceZones from confirmed swings + structure events across 15m/1h/4h,
scores them (0-100), tracks touches/freshness/breaks/flips.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from .config import Config
from .enums import ZoneType
from .indicator_engine import EPS
from .models import PriceZone
from .swing_engine import swings_of

_TF_IMPORTANCE = {"15m": 10.0, "1h": 18.0, "4h": 25.0}
_ZONE_ATR = {"15m": "zone_width_15m_atr", "1h": "zone_width_1h_atr", "4h": "zone_width_4h_atr"}


def _zid(tf: str, kind: str, price: float) -> str:
    return hashlib.sha1(f"{tf}|{kind}|{price:.8f}".encode()).hexdigest()[:12]


class SupportResistanceEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build_zones(self, symbol: str, candles_by_timeframe: dict,
                    swings_by_timeframe: dict, structures: dict) -> list:
        zones: list = []
        for tf, candles in candles_by_timeframe.items():
            if not candles or tf not in swings_by_timeframe:
                continue
            zones.extend(self._tf_zones(tf, candles, swings_by_timeframe[tf], structures.get(tf)))
        # previous-day high/low from 1h candles
        c1h = candles_by_timeframe.get("1h") or []
        zones.extend(self._prev_day_zones(c1h))
        return zones

    # ── per-TF zone construction ────────────────────────────────────────────

    def _tf_zones(self, tf: str, candles: list, swings: list, structure) -> list:
        cfg = self.cfg
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        n = len(candles)
        atr_v = float(np.mean((highs - lows)[-30:])) if n >= 30 else float(np.mean(highs - lows))
        atr_v = max(atr_v, EPS)
        wick_avg = float(np.mean([max(c.upper_wick, c.lower_wick) for c in candles[-30:]])) if n >= 30 else atr_v * 0.2
        width = max(atr_v * getattr(cfg, _ZONE_ATR[tf]), wick_avg, atr_v * 0.05)
        now_ts = int(candles[-1].timestamp)
        last_close = float(closes[-1])

        out: list = []
        strength_rank = sorted(swings, key=lambda s: -s.strength)
        major_cut = strength_rank[max(0, len(strength_rank) // 3)].strength if strength_rank else 0.0

        for s in swings[-24:]:
            kind_res = s.swing_type == "high"
            major = s.strength >= major_cut and s.strength > 0.8
            if kind_res:
                ztype = ZoneType.MAJOR_RESISTANCE if major else ZoneType.MINOR_RESISTANCE
                upper, lower = s.price + width * 0.4, s.price - width
            else:
                ztype = ZoneType.MAJOR_SUPPORT if major else ZoneType.MINOR_SUPPORT
                upper, lower = s.price + width, s.price - width * 0.4
            z = PriceZone(
                zone_id=_zid(tf, ztype.value, s.price), timeframe=tf, zone_type=ztype.value,
                upper_price=float(upper), lower_price=float(lower),
                center_price=float(s.price), width=float(width),
                created_at=s.timestamp, source_swing_id=f"{s.timeframe}:{s.timestamp}",
                displacement_strength=min(s.displacement_atr / 2.0, 1.0),
            )
            self._score_touches_and_breaks(z, candles, s.confirmed_at)
            # flipped zone: broken then retested from the other side
            if z.broken:
                if kind_res and last_close > z.upper_price:
                    z2 = PriceZone(zone_id=_zid(tf, "FLIP_S", s.price), timeframe=tf,
                                   zone_type=ZoneType.FLIPPED_SUPPORT.value,
                                   upper_price=z.upper_price, lower_price=z.lower_price,
                                   center_price=z.center_price, width=z.width,
                                   created_at=now_ts, flipped=True)
                    self._score_touches_and_breaks(z2, candles, now_ts, flipped=True)
                    out.append(z2)
                elif not kind_res and last_close < z.lower_price:
                    z2 = PriceZone(zone_id=_zid(tf, "FLIP_R", s.price), timeframe=tf,
                                   zone_type=ZoneType.FLIPPED_RESISTANCE.value,
                                   upper_price=z.upper_price, lower_price=z.lower_price,
                                   center_price=z.center_price, width=z.width,
                                   created_at=now_ts, flipped=True)
                    self._score_touches_and_breaks(z2, candles, now_ts, flipped=True)
                    out.append(z2)
                continue    # broken, unflipped zones are dead
            out.append(z)

        # breakout-retest zone from the latest BOS
        if structure is not None and structure.last_bos is not None:
            e = structure.last_bos
            z = PriceZone(
                zone_id=_zid(tf, "BR", e.level), timeframe=tf,
                zone_type=ZoneType.BREAKOUT_RETEST.value,
                upper_price=e.level + width * 0.5, lower_price=e.level - width * 0.5,
                center_price=e.level, width=width, created_at=e.confirmed_at,
                source_structure_event=f"{e.event_type}:{e.confirmed_at}",
                displacement_strength=e.displacement_quality,
            )
            self._score_touches_and_breaks(z, candles, e.confirmed_at)
            if not z.broken:
                out.append(z)

        # range high/low over the recent window
        if n >= 40:
            w = candles[-40:]
            rh, rl = max(c.high for c in w), min(c.low for c in w)
            for price, ztype in ((rh, ZoneType.RANGE_HIGH), (rl, ZoneType.RANGE_LOW)):
                z = PriceZone(zone_id=_zid(tf, ztype.value, price), timeframe=tf,
                              zone_type=ztype.value, upper_price=price + width * 0.5,
                              lower_price=price - width * 0.5, center_price=price,
                              width=width, created_at=now_ts)
                self._score_touches_and_breaks(z, candles, now_ts - 1)
                if not z.broken:
                    out.append(z)

        self._score_all(out)
        out = [z for z in out if z.strength >= self.cfg.zone_min_score or z.zone_type in
               (ZoneType.BREAKOUT_RETEST.value, ZoneType.RANGE_HIGH.value, ZoneType.RANGE_LOW.value)]
        out.sort(key=lambda z: -z.strength)
        return out[: self.cfg.max_zones_per_tf]

    def _prev_day_zones(self, c1h: list) -> list:
        if len(c1h) < 48:
            return []
        day_ms = 86_400_000
        last_ts = int(c1h[-1].timestamp)
        day_start = (last_ts // day_ms) * day_ms
        prev = [c for c in c1h if day_start - day_ms <= c.timestamp < day_start]
        if not prev:
            return []
        hi = max(c.high for c in prev)
        lo = min(c.low for c in prev)
        rng = np.array([c.high - c.low for c in c1h[-30:]])
        width = float(np.mean(rng)) * 0.25
        return [
            PriceZone(zone_id=_zid("1h", "PDH", hi), timeframe="1h",
                      zone_type=ZoneType.PREVIOUS_DAY_HIGH.value, upper_price=hi + width,
                      lower_price=hi - width, center_price=hi, width=width,
                      created_at=day_start, strength=55.0),
            PriceZone(zone_id=_zid("1h", "PDL", lo), timeframe="1h",
                      zone_type=ZoneType.PREVIOUS_DAY_LOW.value, upper_price=lo + width,
                      lower_price=lo - width, center_price=lo, width=width,
                      created_at=day_start, strength=55.0),
        ]

    # ── scoring ─────────────────────────────────────────────────────────────

    def _score_touches_and_breaks(self, z: PriceZone, candles: list,
                                  active_from_ts: int, flipped: bool = False) -> None:
        touches = 0
        reactions = 0.0
        last_test = None
        broken = False
        body_break = self.cfg.zone_break_body_atr
        rng = np.array([c.high - c.low for c in candles[-30:]])
        atr_v = max(float(np.mean(rng)), EPS)
        after = [c for c in candles if c.timestamp > active_from_ts]
        for i, c in enumerate(after):
            touched = c.low <= z.upper_price and c.high >= z.lower_price
            if not touched:
                continue
            touches += 1
            last_test = int(c.timestamp)
            body_atr = abs(c.close - c.open) / atr_v
            if z.is_resistance_like and not flipped:
                if c.close > z.upper_price and body_atr >= body_break:
                    broken = True
                elif c.close < z.center_price:
                    reactions += min(body_atr, 1.0)
            elif z.is_support_like or flipped:
                if c.close < z.lower_price and body_atr >= body_break:
                    broken = True
                elif c.close > z.center_price:
                    reactions += min(body_atr, 1.0)
        z.touches = touches
        z.last_tested_at = last_test
        z.broken = broken
        z.reaction_strength = min(reactions / 3.0, 1.0)
        z.freshness = 15.0 if touches == 0 else 10.0 if touches == 1 else 5.0 if touches == 2 else 0.0

    def _score_all(self, zones: list) -> None:
        for z in zones:
            score = 0.0
            score += _TF_IMPORTANCE.get(z.timeframe, 10.0)                      # 0-25
            score += min(z.touches, 3) * 5.0                                     # 0-15 valid reactions
            score += z.reaction_strength * 15.0                                  # 0-15
            score += z.freshness                                                 # 0-15
            score += 15.0 if z.zone_type.startswith("MAJOR") or z.flipped else 7.0   # structure confluence 0-15
            score += z.displacement_strength * 10.0                              # 0-10
            score += 2.5                                                          # liquidity context baseline 0-5
            z.strength = min(score, 100.0)


# ── queries used by the entry engines ────────────────────────────────────────

def nearest_opposing_zone(zones: list, direction: str, price: float,
                          min_score: float = 45.0) -> Optional[PriceZone]:
    if direction == "LONG":
        cands = [z for z in zones if z.is_resistance_like and not z.broken
                 and z.lower_price > price and z.strength >= min_score]
        return min(cands, key=lambda z: z.lower_price - price, default=None)
    cands = [z for z in zones if z.is_support_like and not z.broken
             and z.upper_price < price and z.strength >= min_score]
    return min(cands, key=lambda z: price - z.upper_price, default=None)


def zones_at_price(zones: list, price: float, support_side: bool) -> list:
    out = [z for z in zones if not z.broken and z.contains(price)
           and (z.is_support_like if support_side else z.is_resistance_like)]
    out.sort(key=lambda z: -z.strength)
    return out
