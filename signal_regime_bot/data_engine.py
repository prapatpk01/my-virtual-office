"""
Data Engine — fetches/caches the 3 timeframes per symbol as pandas DataFrames.

Contract with the rest of the system: every DataFrame this module hands out
contains ONLY closed bars — the exchange's `fetch_ohlcv` naturally excludes
the still-forming candle for most exchanges, but we defensively drop the
last bar if its close_time is in the future relative to "now" to guarantee
it everywhere (live AND backtest use the same drop rule).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from config import Config
from exchange_client import ExchangeClient

logger = logging.getLogger("data_engine")

_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
}


class MarketDataUnavailable(RuntimeError):
    """Raised only when a timeframe has no usable fresh OR cached data."""


@dataclass
class _CacheEntry:
    frame: pd.DataFrame
    fetched_ms: int
    bucket: int


# Last-known-good data may be reused temporarily during a transient OKX/Railway
# outage.  The values are intentionally bounded by timeframe: a 5M entry frame
# can never remain stale for hours, while a 4H macro frame remains meaningful
# longer.  Trading is skipped once a cache exceeds this age.
_MAX_STALE_SECONDS = {
    "1m": 180,
    "3m": 420,
    "5m": 900,
    "15m": 2_700,
    "30m": 5_400,
    "1h": 10_800,
    "2h": 21_600,
    "4h": 43_200,
    "1d": 172_800,
}


def _ohlcv_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df.astype(float)


def drop_unclosed_bar(df: pd.DataFrame, timeframe: str, now_ms: int) -> pd.DataFrame:
    """Guarantee the last row is a CLOSED bar as of `now_ms`."""
    if df.empty:
        return df
    tf_sec = _TF_SECONDS.get(timeframe, 60)
    last_open_ms = int(df.index[-1].value // 1_000_000)
    close_ms = last_open_ms + tf_sec * 1000
    if close_ms > now_ms:
        return df.iloc[:-1]
    return df


class DataEngine:
    def __init__(self, cfg: Config, client: ExchangeClient):
        self.cfg = cfg
        self.client = client
        self._cache: dict[tuple, _CacheEntry] = {}
        self._cache_tick = 0
        self._stale_use_count: dict[tuple, int] = {}

    def new_tick(self):
        """Call once per main-loop iteration.

        Cache is intentionally NOT cleared.  A timeframe is fetched only once
        per exchange candle bucket, so polling every 30 seconds no longer asks
        OKX for the same 1H/4H candles dozens of times.  This is the main fix for
        avoidable timeout pressure on Railway.
        """
        self._cache_tick += 1

    @staticmethod
    def _bucket(timeframe: str, now_ms: int) -> int:
        return now_ms // (_TF_SECONDS.get(timeframe, 60) * 1000)

    def _cached_frame(self, key: tuple) -> pd.DataFrame | None:
        entry = self._cache.get(key)
        return None if entry is None else entry.frame

    def _cache_is_usable(self, key: tuple, timeframe: str, now_ms: int) -> bool:
        entry = self._cache.get(key)
        if entry is None or entry.frame.empty:
            return False
        max_stale_ms = _MAX_STALE_SECONDS.get(timeframe, 3_600) * 1000
        return now_ms - entry.fetched_ms <= max_stale_ms

    async def fetch_all(self, symbol: str) -> dict[str, pd.DataFrame]:
        c = self.cfg
        now_ms = int(time.time() * 1000)
        out: dict[str, pd.DataFrame] = {}
        # Preserve order but remove duplicate timeframe requests (for example,
        # an ENV override may make tf_entry equal tf_fast).
        requests = []
        seen = set()
        for tf, limit_attr in (
            (c.tf_micro, "fetch_limit_micro"),      # 5M  — Bias tertiary + Entry L3c (EMA10/20) + exit
            (c.tf_fast, "fetch_limit_fast"),        # 15M — Bias secondary + Entry L3b (EMA5/9)
            (c.tf_entry, "fetch_limit_entry"),      # 30M — Entry L3a (HMA10/16)
            (c.tf_bias, "fetch_limit_bias"),        # 1H  — Regime mid + Bias primary
            (c.tf_regime, "fetch_limit_regime"),    # 4H  — Regime macro
        ):
            if tf not in seen:
                requests.append((tf, limit_attr))
                seen.add(tf)

        for tf, limit_attr in requests:
            key = (symbol, tf)
            bucket = self._bucket(tf, now_ms)
            cached = self._cache.get(key)

            # Same candle bucket = no new closed candle can exist yet.  Reuse
            # the frame without another HTTP request.
            if cached is not None and cached.bucket == bucket:
                out[tf] = cached.frame
                continue

            limit = getattr(c, limit_attr)
            try:
                raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)
                df = _ohlcv_to_df(raw)
                df = drop_unclosed_bar(df, tf, now_ms)
                if df.empty:
                    raise MarketDataUnavailable(f"empty OHLCV response for {symbol} {tf}")
                self._cache[key] = _CacheEntry(df, now_ms, bucket)
                self._stale_use_count.pop(key, None)
                out[tf] = df
            except Exception as exc:
                if self._cache_is_usable(key, tf, now_ms):
                    cached = self._cache[key]
                    count = self._stale_use_count.get(key, 0) + 1
                    self._stale_use_count[key] = count
                    # Warn on first stale use and then every 10 uses; avoid one
                    # identical stack trace every poll cycle.
                    if count == 1 or count % 10 == 0:
                        age_sec = max(0, (now_ms - cached.fetched_ms) // 1000)
                        logger.warning(
                            "[DATA] using last-known-good %s %s cache age=%ss after fetch error: %s",
                            symbol, tf, age_sec, exc,
                        )
                    out[tf] = cached.frame
                    continue
                raise MarketDataUnavailable(
                    f"{symbol} {tf} unavailable and no usable cache: {exc}"
                ) from exc
        return out

    def has_min_bars(self, frames: dict[str, pd.DataFrame]) -> bool:
        return all(len(df) >= self.cfg.min_bars for df in frames.values())
