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

import pandas as pd

from config import Config
from exchange_client import ExchangeClient

logger = logging.getLogger("data_engine")

_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
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
        self._cache: dict[tuple, pd.DataFrame] = {}
        self._cache_tick = 0

    def new_tick(self):
        """Call once per main-loop iteration to invalidate the per-tick cache."""
        self._cache = {}
        self._cache_tick += 1

    async def fetch_all(self, symbol: str) -> dict[str, pd.DataFrame]:
        c = self.cfg
        now_ms = int(time.time() * 1000)
        out = {}
        for tf, limit_attr in (
            (c.tf_micro, "fetch_limit_micro"),      # 5M  — Bias tertiary + Entry Layer 3.2
            (c.tf_fast, "fetch_limit_fast"),        # 15M — Bias secondary + Entry Layers 3.1/3.2/3.3
            (c.tf_bias, "fetch_limit_bias"),        # 1H  — Regime mid + Bias primary
            (c.tf_regime, "fetch_limit_regime"),    # 4H  — Regime macro
        ):
            key = (symbol, tf)
            if key in self._cache:
                out[tf] = self._cache[key]
                continue
            limit = getattr(c, limit_attr)
            raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)
            df = _ohlcv_to_df(raw)
            df = drop_unclosed_bar(df, tf, now_ms)
            self._cache[key] = df
            out[tf] = df
        return out

    def has_min_bars(self, frames: dict[str, pd.DataFrame]) -> bool:
        return all(len(df) >= self.cfg.min_bars for df in frames.values())
