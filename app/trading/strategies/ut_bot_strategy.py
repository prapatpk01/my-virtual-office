"""UT Bot Strategy — ATR-based trailing stop with Heikin Ashi confirmation.

Uses the UT Bot key level (ATR trailing stop) to detect trend changes.
BUY:  price crosses above the trailing stop with HA close > HA open
SELL: price crosses below the trailing stop with HA close < HA open

Public interface used by backtest_50usd.py / tune_1h.py:
    _build_signals(candles) → (buy_arr, sell_arr, key_level_arr, atr_arr)
    attributes: ut_atr_len, sl_atr_mult, rr_ratio
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class UTBotStrategy(BaseStrategy):
    """UT Bot — ATR trailing stop with Heikin Ashi bar direction filter."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ut_mult      = float(self.params.get("ut_mult",      1.0))
        self.ut_atr_len   = int(self.params.get("ut_atr_len",    14))
        self.sl_atr_mult  = float(self.params.get("sl_atr_mult",  2.0))
        self.rr_ratio     = float(self.params.get("rr_ratio",     1.5))
        self.position_pct = float(self.params.get("position_pct", 0.05))

    # ------------------------------------------------------------------ #

    def _build_signals(self, candles: list):
        """Build signal arrays over the full candle history.

        Returns
        -------
        buy_arr  : np.ndarray[bool]
        sell_arr : np.ndarray[bool]
        key_level: np.ndarray[float]  — ATR trailing stop values
        atr_arr  : np.ndarray[float]  — ATR(14) for position sizing
        """
        n = len(candles)
        ha_candles, ha_o, ha_c = self._heikin_ashi(candles)

        # ATR for trailing stop
        atr_arr = np.array(self.atr(ha_candles, self.ut_atr_len), dtype=float)
        # ATR(14) for stop sizing (same array here)
        atr14   = atr_arr.copy()

        key_level = np.full(n, np.nan)
        buy_arr   = np.zeros(n, dtype=bool)
        sell_arr  = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
                continue

            atr_band = self.ut_mult * atr_arr[i]
            src      = float(ha_c[i])
            src_prev = float(ha_c[i - 1])
            kl_prev  = float(key_level[i - 1]) if not np.isnan(key_level[i - 1]) else src_prev

            # Trailing stop update
            if src_prev > kl_prev:
                kl = max(kl_prev, src - atr_band)
            else:
                kl = min(kl_prev, src + atr_band)

            key_level[i] = kl

            # Signal: crossover / crossunder of key level
            ha_bull = float(ha_c[i]) > float(ha_o[i])
            ha_bear = float(ha_c[i]) < float(ha_o[i])

            cross_up   = src_prev <= kl_prev and src > kl
            cross_down = src_prev >= kl_prev and src < kl

            buy_arr[i]  = cross_up   and ha_bull
            sell_arr[i] = cross_down and ha_bear

        return buy_arr, sell_arr, key_level, atr14

    # ------------------------------------------------------------------ #

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.ut_atr_len + 14 + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        buy_arr, sell_arr, key_level, atr_arr = self._build_signals(candles)

        curr_kl  = float(key_level[-1]) if not np.isnan(key_level[-1]) else current_price
        curr_atr = float(atr_arr[-1])   if not np.isnan(atr_arr[-1])  else 0.0

        if buy_arr[-1]:
            sl = current_price - self.sl_atr_mult * curr_atr
            tp = current_price + self.sl_atr_mult * curr_atr * self.rr_ratio
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[UTBot] Price crossed above key level {curr_kl:.2f}",
                confidence=0.7,
                metadata={"key_level": curr_kl, "atr": curr_atr, "sl": sl, "tp": tp},
            )
        if sell_arr[-1]:
            sl = current_price + self.sl_atr_mult * curr_atr
            tp = current_price - self.sl_atr_mult * curr_atr * self.rr_ratio
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[UTBot] Price crossed below key level {curr_kl:.2f}",
                confidence=0.7,
                metadata={"key_level": curr_kl, "atr": curr_atr, "sl": sl, "tp": tp},
            )

        above = current_price > curr_kl
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[UTBot] Price {'above' if above else 'below'} key level {curr_kl:.2f}",
            metadata={"key_level": curr_kl, "atr": curr_atr},
        )
