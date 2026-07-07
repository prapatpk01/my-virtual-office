"""
Module 6: Feature Store

Caches computed features to ensure training/serving consistency.
Prevents re-computing expensive features across the pipeline.
Supports versioned snapshots for backtesting replay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FeatureSnapshot:
    symbol:    str
    timeframe: str
    timestamp: float
    candle_ts: int          # last candle timestamp
    features:  dict         # all computed features
    regime:    str  = ""
    version:   int  = 1


class FeatureStore:
    """In-memory feature cache with TTL and versioning."""

    def __init__(self, ttl_seconds: int = 60):
        self._ttl = ttl_seconds
        self._cache: Dict[str, FeatureSnapshot] = {}
        self._history: list = []  # for walk-forward replay
        self._max_history = 1000

    def store(
        self,
        symbol:    str,
        timeframe: str,
        candle_ts: int,
        features:  dict,
        regime:    str = "",
    ) -> FeatureSnapshot:
        key      = f"{symbol}:{timeframe}"
        version  = (self._cache[key].version + 1) if key in self._cache else 1
        snapshot = FeatureSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=time.time(),
            candle_ts=candle_ts,
            features=features,
            regime=regime,
            version=version,
        )
        self._cache[key] = snapshot
        # Keep rolling history for backtesting replay
        self._history.append({"key": key, "ts": snapshot.timestamp, "snap": snapshot})
        if len(self._history) > self._max_history:
            self._history.pop(0)
        return snapshot

    def get(
        self,
        symbol:    str,
        timeframe: str,
        max_age:   Optional[int] = None,
    ) -> Optional[FeatureSnapshot]:
        key  = f"{symbol}:{timeframe}"
        snap = self._cache.get(key)
        if snap is None:
            return None
        age = time.time() - snap.timestamp
        ttl = max_age or self._ttl
        return snap if age <= ttl else None

    def invalidate(self, symbol: str, timeframe: str) -> None:
        self._cache.pop(f"{symbol}:{timeframe}", None)

    def get_history(self, symbol: str, timeframe: str, limit: int = 100) -> list:
        key = f"{symbol}:{timeframe}"
        return [
            h["snap"] for h in self._history
            if h["key"] == key
        ][-limit:]

    def build_feature_vector(self, snapshot: FeatureSnapshot) -> dict:
        """Flatten nested features into a single dict for ML model input."""
        flat: dict = {}
        for section, values in snapshot.features.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[f"{section}_{k}"] = v
            else:
                flat[section] = values
        flat["regime"]    = snapshot.regime
        flat["symbol"]    = snapshot.symbol
        flat["timeframe"] = snapshot.timeframe
        return flat

    def stats(self) -> dict:
        return {
            "cached_keys":    len(self._cache),
            "history_length": len(self._history),
            "keys":           list(self._cache.keys()),
        }
