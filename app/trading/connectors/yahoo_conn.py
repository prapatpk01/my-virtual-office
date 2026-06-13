"""
Yahoo Finance connector — signal-only, no real orders.
Fetches OHLCV for forex/gold via yfinance (free, no API key needed).

Supported symbols:
  XAUUSD, EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, USDCAD, NZDUSD
"""
import asyncio
import time
from typing import List
import numpy as np

from .base import BaseConnector, OHLCV, OrderResult, Balance

SYMBOL_MAP = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

TF_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "60m", "4h": "1h", "1d": "1d",
}

PERIOD_MAP = {
    "1m": "1d", "5m": "5d", "15m": "5d",
    "30m": "60d", "60m": "60d", "1h": "60d", "1d": "1y",
}


class YahooConnector(BaseConnector):
    """Read-only connector using yfinance. Orders are simulated (signal-only)."""

    def __init__(self, **kwargs):
        super().__init__(api_key="", api_secret="", paper=True)
        self._price_cache: dict = {}

    def _yf_symbol(self, symbol: str) -> str:
        return SYMBOL_MAP.get(symbol.upper(), symbol)

    def _fetch_sync(self, symbol: str, interval: str, limit: int):
        import yfinance as yf
        yf_sym  = self._yf_symbol(symbol)
        yf_int  = TF_MAP.get(interval, "15m")
        period  = PERIOD_MAP.get(yf_int, "5d")
        ticker  = yf.Ticker(yf_sym)
        df = ticker.history(period=period, interval=yf_int, auto_adjust=True)
        if df.empty:
            return []
        df = df.tail(limit)
        candles = []
        for ts, row in df.iterrows():
            try:
                t = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(ts) // 10**9
            except Exception:
                t = int(time.time())
            candles.append(OHLCV(
                timestamp=t,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 1000)),
            ))
        return candles

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 250) -> List[OHLCV]:
        candles = await asyncio.to_thread(self._fetch_sync, symbol, timeframe, limit)
        if candles:
            # Reject stale data: forex closes Fri night → Sun night (~48h gap max)
            age_hours = (time.time() - candles[-1].timestamp) / 3600
            if age_hours > 50:
                return []   # strategy returns HOLD → no signal during market closure
            self._price_cache[symbol] = candles[-1].close
        return candles

    async def fetch_ticker(self, symbol: str) -> dict:
        if symbol in self._price_cache:
            return {"last": self._price_cache[symbol], "symbol": symbol}
        candles = await self.fetch_ohlcv(symbol, "15m", 5)
        price = candles[-1].close if candles else 0.0
        return {"last": price, "symbol": symbol}

    async def create_order(self, symbol: str, side: str, amount: float, price: float = None) -> OrderResult:
        last = self._price_cache.get(symbol, price or 0.0)
        return OrderResult(
            id=f"signal-{int(time.time())}",
            symbol=symbol, side=side,
            price=last, amount=amount,
            filled=amount, status="signal_only",
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        return True

    async def fetch_open_orders(self, symbol=None) -> list:
        return []

    async def fetch_balance(self):
        return [Balance(asset="USD", free=0.0, used=0.0, total=0.0)]
