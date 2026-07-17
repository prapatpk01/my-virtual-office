"""Supply/Demand Engine (spec §12) — base + strong departure zones only."""
from __future__ import annotations

import hashlib

import numpy as np

from .config import Config
from .indicator_engine import EPS
from .models import SupplyDemandZone


class SupplyDemandEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, candles_1h: list, candles_4h: list, structures: dict) -> list:
        out: list = []
        for tf, candles in (("1h", candles_1h), ("4h", candles_4h)):
            if candles and len(candles) >= 40:
                out.extend(self._scan(tf, candles, structures.get(tf)))
        return out

    def _scan(self, tf: str, candles: list, structure) -> list:
        cfg = self.cfg
        rng = np.array([c.high - c.low for c in candles[-40:]])
        atr_v = max(float(np.mean(rng)), EPS)
        out: list = []
        n = len(candles)
        # scan last ~80 bars for base -> departure
        for i in range(max(5, n - 80), n - 2):
            # base = up to max_base_bars small candles
            for blen in range(1, cfg.max_base_bars + 1):
                s = i - blen
                if s < 0:
                    break
                base = candles[s:i]
                if not base or any(abs(c.close - c.open) / atr_v > 0.45 for c in base):
                    continue
                dep = candles[i]
                move = (dep.close - dep.open) / atr_v
                if move >= cfg.min_departure_atr:            # demand
                    prox = max(c.high for c in base)
                    dist = min(c.low for c in base)
                    direction = "LONG"
                elif move <= -cfg.min_departure_atr:          # supply
                    prox = min(c.low for c in base)
                    dist = max(c.high for c in base)
                    direction = "SHORT"
                else:
                    continue
                # mitigation count after departure
                mits = 0
                for c in candles[i + 1:]:
                    lo, hi = min(prox, dist), max(prox, dist)
                    if c.low <= hi and c.high >= lo:
                        mits += 1
                if mits > cfg.max_hq_mitigations + 1:
                    continue
                bos_ok = bool(structure and structure.last_bos
                              and structure.last_bos.direction == direction)
                zid = hashlib.sha1(f"{tf}|SD|{direction}|{candles[s].timestamp}".encode()).hexdigest()[:12]
                out.append(SupplyDemandZone(
                    zone_id=zid, direction=direction,
                    base_start=int(candles[s].timestamp), base_end=int(candles[i - 1].timestamp),
                    proximal_line=float(prox), distal_line=float(dist),
                    departure_strength=min(abs(move) / (cfg.min_departure_atr * 2), 1.0),
                    bos_confirmed=bos_ok,
                    freshness=15.0 if mits == 0 else 8.0 if mits == 1 else 0.0,
                    mitigation_count=mits,
                ))
                break
        # newest, freshest first; cap
        out.sort(key=lambda z: (-z.freshness, -z.base_end))
        return out[:6]
