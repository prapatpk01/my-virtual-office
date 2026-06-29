"""MACD + EMA Strategy with Heikin Ashi confirmation.

Combines HMA price filter, EMA/SMA trend, and MACD momentum on HA candles.

BUY:  HA close > HA open (green HA bar) AND close > HMA AND MACD histogram positive
SELL: HA close < HA open (red HA bar)   AND close < HMA AND MACD histogram negative

Public interface used by backtest_50usd.py / tune_1h.py:
    _build_arrays(candles) → (ha_o, ha_c, ha_highs, ha_lows, hma, ema, sma, ml, sl_line, hist, atr_arr)
    _signal_at(i, ha_o, ha_c, ha_highs, ha_lows, hma, ema, sma, ml, sl_line) → 1 | -1 | 0
    attributes: macd_slow, macd_sig, hma_period, sl_atr_mult, rr_ratio
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MACDEMAStrategy(BaseStrategy):
    """MACD + EMA/HMA Heikin Ashi strategy."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.hma_period   = int(self.params.get("hma_period",   30))
        self.ema_fast     = int(self.params.get("ema_fast",     21))
        self.sma_slow     = int(self.params.get("sma_slow",     50))
        self.macd_fast    = int(self.params.get("macd_fast",    12))
        self.macd_slow    = int(self.params.get("macd_slow",    26))
        self.macd_sig     = int(self.params.get("macd_sig",      9))
        self.sl_atr_mult  = float(self.params.get("sl_atr_mult", 2.0))
        self.rr_ratio     = float(self.params.get("rr_ratio",    1.5))
        self.position_pct = float(self.params.get("position_pct", 0.05))

    # ------------------------------------------------------------------ #

    def _build_arrays(self, candles: list):
        """Pre-compute all indicator arrays over the candle history.

        Returns
        -------
        ha_o      : np.ndarray  — Heikin Ashi open
        ha_c      : np.ndarray  — Heikin Ashi close
        ha_highs  : np.ndarray  — Heikin Ashi high
        ha_lows   : np.ndarray  — Heikin Ashi low
        hma       : np.ndarray  — HMA of HA close
        ema       : np.ndarray  — EMA(ema_fast) of HA close
        sma       : np.ndarray  — SMA(sma_slow) of HA close
        ml        : np.ndarray  — MACD line
        sl_line   : np.ndarray  — MACD signal line
        hist      : np.ndarray  — MACD histogram
        atr_arr   : np.ndarray  — ATR(14) on HA candles
        """
        ha_candles, ha_o_arr, ha_c_arr = self._heikin_ashi(candles)

        ha_highs = np.array([c.high for c in ha_candles], dtype=float)
        ha_lows  = np.array([c.low  for c in ha_candles], dtype=float)

        closes_list = ha_c_arr.tolist()

        hma_arr  = np.array(self.hma(closes_list, self.hma_period), dtype=float)
        ema_arr  = np.array(self.ema(closes_list, self.ema_fast),   dtype=float)
        sma_arr  = np.array(self.sma(closes_list, self.sma_slow),   dtype=float)
        ml, sl_line, hist = self.macd(closes_list, self.macd_fast, self.macd_slow, self.macd_sig)
        atr_arr  = np.array(self.atr(ha_candles, 14), dtype=float)

        return (ha_o_arr, ha_c_arr, ha_highs, ha_lows,
                hma_arr, ema_arr, sma_arr,
                np.asarray(ml, dtype=float),
                np.asarray(sl_line, dtype=float),
                np.asarray(hist, dtype=float),
                atr_arr)

    def _signal_at(self, i: int,
                   ha_o, ha_c, ha_highs, ha_lows,
                   hma, ema, sma, ml, sl_line) -> int:
        """Return 1 (buy), -1 (sell), or 0 (hold) for bar index i."""
        if i < 1:
            return 0

        for arr in (hma, ema, sma, ml, sl_line):
            if i >= len(arr) or np.isnan(arr[i]):
                return 0

        ha_green = float(ha_c[i]) > float(ha_o[i])
        ha_red   = float(ha_c[i]) < float(ha_o[i])
        hist_pos = float(ml[i])  > float(sl_line[i])
        hist_neg = float(ml[i])  < float(sl_line[i])
        above_hma = float(ha_c[i]) > float(hma[i])
        below_hma = float(ha_c[i]) < float(hma[i])
        ema_bull  = float(ema[i]) > float(sma[i])
        ema_bear  = float(ema[i]) < float(sma[i])

        # Previous bar HA direction for crossover detection
        prev_green = float(ha_c[i - 1]) > float(ha_o[i - 1])

        # BUY: first green HA bar + above HMA + MACD positive + EMA > SMA
        if ha_green and not prev_green and above_hma and hist_pos and ema_bull:
            return 1
        # Also buy on continued green with all filters
        if ha_green and above_hma and hist_pos and ema_bull:
            return 1

        # SELL: red HA bar + below HMA + MACD negative + EMA < SMA
        if ha_red and not (float(ha_c[i - 1]) < float(ha_o[i - 1])) and below_hma and hist_neg and ema_bear:
            return -1
        if ha_red and below_hma and hist_neg and ema_bear:
            return -1

        return 0

    # ------------------------------------------------------------------ #

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.hma_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        (ha_o, ha_c, ha_highs, ha_lows,
         hma, ema, sma, ml, sl_line, hist, atr_arr) = self._build_arrays(candles)

        i = len(candles) - 1
        d = self._signal_at(i, ha_o, ha_c, ha_highs, ha_lows, hma, ema, sma, ml, sl_line)

        curr_hma = float(hma[i])  if not np.isnan(hma[i])  else current_price
        curr_atr = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0

        if d == 1:
            sl = current_price - self.sl_atr_mult * curr_atr
            tp = current_price + self.sl_atr_mult * curr_atr * self.rr_ratio
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[MACD/EMA] HA green + above HMA + MACD+ + EMA>SMA",
                confidence=0.7,
                metadata={"hma": curr_hma, "atr": curr_atr, "sl": sl, "tp": tp},
            )
        if d == -1:
            sl = current_price + self.sl_atr_mult * curr_atr
            tp = current_price - self.sl_atr_mult * curr_atr * self.rr_ratio
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[MACD/EMA] HA red + below HMA + MACD- + EMA<SMA",
                confidence=0.7,
                metadata={"hma": curr_hma, "atr": curr_atr, "sl": sl, "tp": tp},
            )

        above = current_price > curr_hma
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MACD/EMA] Price {'above' if above else 'below'} HMA={curr_hma:.2f}",
            metadata={"hma": curr_hma},
        )
