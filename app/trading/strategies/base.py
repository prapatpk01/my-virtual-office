"""Base strategy classes and shared indicator helpers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    type: SignalType
    symbol: str
    price: float
    amount: float              # position size in base asset
    reason: str = ""
    confidence: float = 1.0   # 0-1
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """All strategies inherit from this."""

    def __init__(self, symbol: str, params: Optional[dict] = None):
        self.symbol = symbol
        self.params = params or {}
        self.name = self.__class__.__name__

    @abstractmethod
    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        """Return a Signal given OHLCV candles and current price.
        mtf_candles: optional dict of higher-TF candles, e.g. {"1h": [...], "4h": [...]}
        """

    # ------------------------------------------------------------------
    # Shared indicator helpers (no external TA library needed)
    # ------------------------------------------------------------------

    @staticmethod
    def ema(values: list[float], period: int) -> np.ndarray:
        arr = np.array(values, dtype=float)
        result = np.full_like(arr, np.nan)
        k = 2.0 / (period + 1)
        result[period - 1] = arr[:period].mean()
        for i in range(period, len(arr)):
            result[i] = arr[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def sma(values: list[float], period: int) -> np.ndarray:
        arr = np.array(values, dtype=float)
        result = np.full_like(arr, np.nan)
        for i in range(period - 1, len(arr)):
            result[i] = arr[i - period + 1:i + 1].mean()
        return result

    @staticmethod
    def rsi(values: list[float], period: int = 14) -> np.ndarray:
        arr = np.array(values, dtype=float)
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.full(len(arr), np.nan)
        avg_loss = np.full(len(arr), np.nan)
        avg_gain[period] = gain[:period].mean()
        avg_loss[period] = loss[:period].mean()
        for i in range(period + 1, len(arr)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
        rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
        rsi_arr = 100 - (100 / (1 + rs))
        rsi_arr[:period] = np.nan
        return rsi_arr

    @staticmethod
    def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
        fast_ema = BaseStrategy.ema(values, fast)
        slow_ema = BaseStrategy.ema(values, slow)
        macd_line = fast_ema - slow_ema
        signal_line = BaseStrategy.ema(
            [v for v in macd_line if not np.isnan(v)], signal
        )
        # Pad signal_line to match macd_line length
        pad = len(macd_line) - len(signal_line)
        signal_padded = np.concatenate([np.full(pad, np.nan), signal_line])
        histogram = macd_line - signal_padded
        return macd_line, signal_padded, histogram

    @staticmethod
    def bollinger_bands(values: list[float], period: int = 20, std_dev: float = 2.0):
        sma = BaseStrategy.sma(values, period)
        arr = np.array(values, dtype=float)
        rolling_std = np.full_like(arr, np.nan)
        for i in range(period - 1, len(arr)):
            rolling_std[i] = arr[i - period + 1:i + 1].std()
        upper = sma + std_dev * rolling_std
        lower = sma - std_dev * rolling_std
        return upper, sma, lower

    @staticmethod
    def atr(candles: list, period: int = 14) -> np.ndarray:
        """Average True Range using Wilder's smoothing (RMA)."""
        n = len(candles)
        tr = np.full(n, np.nan)
        for i in range(1, n):
            h  = candles[i].high
            l  = candles[i].low
            pc = candles[i - 1].close
            tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        result = np.full(n, np.nan)
        if n > period:
            result[period] = float(np.nanmean(tr[1:period + 1]))
            for i in range(period + 1, n):
                result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
        return result

    @staticmethod
    def wma(values: list[float], period: int) -> np.ndarray:
        """Weighted Moving Average — linearly weighted, newest bar has highest weight."""
        arr = np.array(values, dtype=float)
        result = np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1, dtype=float)
        denom = weights.sum()
        for i in range(period - 1, len(arr)):
            result[i] = np.dot(arr[i - period + 1:i + 1], weights) / denom
        return result

    @staticmethod
    def hma(values: list[float], period: int) -> np.ndarray:
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
        import math
        half  = max(2, period // 2)
        sqrtn = max(2, int(round(math.sqrt(period))))
        wma_half = BaseStrategy.wma(values, half)
        wma_full = BaseStrategy.wma(values, period)
        raw  = 2.0 * wma_half - wma_full
        return BaseStrategy.wma(list(raw), sqrtn)
