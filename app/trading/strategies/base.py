"""Base strategy classes and shared indicator helpers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class _HACandle:
    """Lightweight Heikin Ashi candle object compatible with all indicator helpers."""
    __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp = ts
        self.open = o; self.high = h; self.low = l; self.close = c; self.volume = v


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
        # Allow callers to pass name="..." in params to create distinct strategy slots
        self.name = self.params.get("name", self.__class__.__name__)

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
        if len(arr) < period:
            return np.full(len(arr), np.nan)
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
        if len(arr) <= period:
            return np.full(len(arr), np.nan)
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
        valid = [v for v in macd_line if not np.isnan(v)]
        if len(valid) < signal:
            nan_arr = np.full(len(macd_line), np.nan)
            return macd_line, nan_arr, nan_arr
        signal_line = BaseStrategy.ema(valid, signal)
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

    @staticmethod
    def adx(candles: list, period: int = 14) -> tuple:
        """ADX, +DI, -DI using Wilder's smoothing. Returns (adx_arr, plus_di, minus_di)."""
        n = len(candles)
        pdm = np.zeros(n)
        mdm = np.zeros(n)
        tr  = np.zeros(n)
        for i in range(1, n):
            h, l   = candles[i].high, candles[i].low
            ph, pl, pc = candles[i-1].high, candles[i-1].low, candles[i-1].close
            up, dn = h - ph, pl - l
            pdm[i] = up if up > dn and up > 0 else 0.0
            mdm[i] = dn if dn > up and dn > 0 else 0.0
            tr[i]  = max(h - l, abs(h - pc), abs(l - pc))
        s_tr = np.full(n, np.nan)
        s_pd = np.full(n, np.nan)
        s_md = np.full(n, np.nan)
        if n > period:
            s_tr[period] = tr[1:period+1].sum()
            s_pd[period] = pdm[1:period+1].sum()
            s_md[period] = mdm[1:period+1].sum()
            for i in range(period+1, n):
                s_tr[i] = s_tr[i-1] - s_tr[i-1]/period + tr[i]
                s_pd[i] = s_pd[i-1] - s_pd[i-1]/period + pdm[i]
                s_md[i] = s_md[i-1] - s_md[i-1]/period + mdm[i]
        plus_di  = np.where(s_tr > 0, 100 * s_pd / s_tr, 0.0)
        minus_di = np.where(s_tr > 0, 100 * s_md / s_tr, 0.0)
        dsum = plus_di + minus_di
        dx = np.where(
            dsum > 0,
            100 * np.abs(plus_di - minus_di) / np.where(dsum > 0, dsum, 1.0),
            0.0,
        )
        adx_arr = np.full(n, np.nan)
        start = 2 * period
        if n > start:
            adx_arr[start] = float(np.nanmean(dx[period:start+1]))
            for i in range(start+1, n):
                if not np.isnan(adx_arr[i-1]):
                    adx_arr[i] = (adx_arr[i-1] * (period-1) + dx[i]) / period
        return adx_arr, plus_di, minus_di

    def risk_metadata(self, current_price: float, atr_v: float, side: str,
                      sl_atr: float = 1.2, tp1_atr: float = 1.5, tp2_atr: float = 3.0,
                      sl_min_pct: float = 0.010, sl_max_pct: float = 0.040,
                      risk_pct: float = 0.02, partial_pct: float = 0.5) -> dict:
        """
        Build the standard Global-Risk metadata block for an entry signal.

        Initial SL = sl_atr×ATR (clamped to [sl_min_pct, sl_max_pct] of price).
        TP1 = tp1_atr×ATR (close `partial_pct` of position, then move SL → breakeven).
        TP2 = tp2_atr×ATR (close the remainder, ride the trend).
        After clamping, TP distances are scaled by the *effective* ATR so the
        1.2 / 1.5 / 2.2 ratios are preserved even when the SL is widened/narrowed.

        `sl_dist_pct` and `risk_pct` let the RiskManager size the position so the
        worst-case loss (full size hitting initial SL) ≈ risk_pct of the portfolio.
        """
        sl_raw   = sl_atr * atr_v
        sl_dist  = max(current_price * sl_min_pct,
                       min(sl_raw, current_price * sl_max_pct))
        atr_unit = sl_dist / max(sl_atr, 1e-9)      # effective ATR after clamp
        tp1_dist = tp1_atr * atr_unit
        tp2_dist = tp2_atr * atr_unit
        if side == "long":
            sl  = current_price - sl_dist
            tp1 = current_price + tp1_dist
            tp2 = current_price + tp2_dist
        else:
            sl  = current_price + sl_dist
            tp1 = current_price - tp1_dist
            tp2 = current_price - tp2_dist
        return {
            "stop_loss":   round(sl, 2),    # back-compat: initial SL
            "take_profit": round(tp2, 2),   # back-compat: full TP = TP2
            "sl_init":     round(sl, 2),
            "tp1":         round(tp1, 2),
            "tp2":         round(tp2, 2),
            "atr":         round(atr_v, 2),
            "sl_dist_pct": round(sl_dist / current_price, 6),
            "partial_pct": partial_pct,     # fraction to close at TP1
            "risk_pct":    risk_pct,        # portfolio risk budget for sizing
            "breakeven":   round(current_price, 2),
            "rr_tp1":      round(tp1_dist / sl_dist, 2),
            "rr_tp2":      round(tp2_dist / sl_dist, 2),
        }

    @staticmethod
    def supertrend(candles: list, period: int = 10, multiplier: float = 3.0) -> tuple:
        """
        Supertrend indicator. Returns (st_line, direction) where
          direction = +1  → uptrend  (price above line, "green")
          direction = -1  → downtrend (price below line, "red")
        Uses ATR(period) with Wilder smoothing and the classic band-flip logic.
        """
        n = len(candles)
        st  = np.full(n, np.nan)
        dir_ = np.zeros(n, dtype=int)
        if n <= period:
            return st, dir_
        atr = BaseStrategy.atr(candles, period)
        hl2 = np.array([(c.high + c.low) / 2.0 for c in candles], dtype=float)
        basic_up = hl2 + multiplier * atr
        basic_dn = hl2 - multiplier * atr
        f_up = np.full(n, np.nan)
        f_dn = np.full(n, np.nan)
        start = period  # first bar with a valid ATR
        f_up[start] = basic_up[start]
        f_dn[start] = basic_dn[start]
        dir_[start] = 1
        st[start]   = f_dn[start]
        for i in range(start + 1, n):
            prev_close = candles[i - 1].close
            f_up[i] = (basic_up[i] if (basic_up[i] < f_up[i - 1] or prev_close > f_up[i - 1])
                       else f_up[i - 1])
            f_dn[i] = (basic_dn[i] if (basic_dn[i] > f_dn[i - 1] or prev_close < f_dn[i - 1])
                       else f_dn[i - 1])
            close = candles[i].close
            if dir_[i - 1] == 1:
                dir_[i] = -1 if close < f_dn[i] else 1
            else:
                dir_[i] = 1 if close > f_up[i] else -1
            st[i] = f_dn[i] if dir_[i] == 1 else f_up[i]
        return st, dir_

    @staticmethod
    def obv(candles: list) -> np.ndarray:
        """On-Balance Volume: cumulative volume signed by close-to-close direction."""
        n = len(candles)
        out = np.zeros(n, dtype=float)
        for i in range(1, n):
            c, pc = candles[i].close, candles[i - 1].close
            if   c > pc: out[i] = out[i - 1] + candles[i].volume
            elif c < pc: out[i] = out[i - 1] - candles[i].volume
            else:        out[i] = out[i - 1]
        return out

    @staticmethod
    def _heikin_ashi(candles: list):
        """Convert OHLCV candles to Heikin Ashi. Returns (ha_candles, ha_open_arr, ha_close_arr)."""
        n = len(candles)
        ha_o = np.zeros(n); ha_c = np.zeros(n)
        ha_h = np.zeros(n); ha_l = np.zeros(n)
        for i, c in enumerate(candles):
            ha_c[i] = (c.open + c.high + c.low + c.close) / 4.0
            ha_o[i] = ((c.open + c.close) / 2.0 if i == 0 else (ha_o[i-1] + ha_c[i-1]) / 2.0)
            ha_h[i] = max(c.high, ha_o[i], ha_c[i])
            ha_l[i] = min(c.low,  ha_o[i], ha_c[i])
        ha_candles = [
            _HACandle(candles[i].timestamp, ha_o[i], ha_h[i], ha_l[i], ha_c[i], candles[i].volume)
            for i in range(n)
        ]
        return ha_candles, ha_o, ha_c

    @staticmethod
    def compute_mtf_bias(
        candles_15m: list,
        mtf_candles: dict,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_bull: float = 55.0,
        rsi_bear: float = 45.0,
        w1: float = 1.0,
        w2: float = 2.0,
        w3: float = 3.0,
    ) -> tuple[float, str]:
        """
        Port of Pine Script MTF Bias Monitor (15m / 1H / 4H).

        Per-TF score  -3..+3:
          close > EMA20   → +1  else -1
          EMA20 > EMA50   → +1  else -1
          RSI > rsi_bull  → +1, RSI < rsi_bear → -1, else 0

        Weighted composite (weights default 1/2/3):
          comp_pct = (s15m*w1 + s1h*w2 + s4h*w3) / (3*(w1+w2+w3)) * 100
          range -100..+100 %

        Gate:  BUY  → comp_pct > 0
               SELL → comp_pct < 0
        """
        def _tf_score(closes: list) -> float:
            if len(closes) < ema_slow + rsi_period + 2:
                return 0.0
            e20 = float(BaseStrategy.ema(closes, ema_fast)[-1])
            e50 = float(BaseStrategy.ema(closes, ema_slow)[-1])
            r   = float(BaseStrategy.rsi(closes, rsi_period)[-1])
            if any(np.isnan(v) for v in [e20, e50, r]):
                return 0.0
            c = closes[-1]
            return (
                (1.0 if c > e20 else -1.0) +
                (1.0 if e20 > e50 else -1.0) +
                (1.0 if r > rsi_bull else (-1.0 if r < rsi_bear else 0.0))
            )

        def _arrow(s: float) -> str:
            return "↑↑" if s >= 2 else "↑" if s > 0 else "↓↓" if s <= -2 else "↓" if s < 0 else "→"

        mtf = mtf_candles or {}
        c15 = [c.close for c in (candles_15m or [])]
        c1h = [c.close for c in mtf.get("1h", [])]
        c4h = [c.close for c in mtf.get("4h", [])]

        s1 = _tf_score(c15)
        s2 = _tf_score(c1h)
        s3 = _tf_score(c4h)

        wt_max   = 3.0 * (w1 + w2 + w3)
        comp_pct = (s1 * w1 + s2 * w2 + s3 * w3) / wt_max * 100.0

        label = (
            f"15m{_arrow(s1)} 1H{_arrow(s2)} 4H{_arrow(s3)} | "
            f"{comp_pct:+.0f}%"
        )
        return comp_pct, label
