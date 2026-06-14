"""
WaveTrend Strategy — WT1/WT2 cross in OB/OS zones + ATR SL/TP.

BUY:  WT1 crosses above WT2  AND  WT1 < -25  (oversold zone)
SELL: WT1 crosses below WT2  AND  WT1 > +25  (overbought zone)

SL/TP: ATR-based, R:R = 1:1.5
State machine prevents duplicate consecutive signals.
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_ATR_PERIOD = 14


class WTADXStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.n1          = self.params.get("wt_channel_len", 10)
        self.n2          = self.params.get("wt_avg_len",     21)
        self.ob_level    = self.params.get("ob_level",       25.0)   # overbought threshold
        self.os_level    = self.params.get("os_level",      -25.0)   # oversold threshold
        self.sl_atr_mult = self.params.get("sl_atr_mult",    1.5)
        self.rr_ratio    = self.params.get("rr_ratio",       1.5)
        self._last_signal = 0   # state machine: 1=buy, -1=sell

    # ------------------------------------------------------------------ #

    def _rma(self, arr: np.ndarray, length: int) -> np.ndarray:
        """Wilder's Moving Average (alpha = 1/length)."""
        alpha = 1.0 / length
        result = np.full(len(arr), np.nan)
        if len(arr) < length:
            return result
        result[length - 1] = float(np.mean(arr[:length]))
        for i in range(length, len(arr)):
            if not np.isnan(arr[i]):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
        return result

    def _wavetrend(self, highs, lows, closes):
        h = np.array(highs); l = np.array(lows); c = np.array(closes)
        ap  = (h + l + c) / 3.0
        esa = np.array(self.ema(ap.tolist(), self.n1))
        diff_abs = np.abs(ap - esa)
        first_ok = np.where(~np.isnan(diff_abs))[0]
        if len(first_ok):
            diff_filled = diff_abs.copy()
            diff_filled[:first_ok[0]] = diff_abs[first_ok[0]]
        else:
            diff_filled = diff_abs
        d   = np.array(self.ema(diff_filled.tolist(), self.n1))
        ci  = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
        tci = np.array(self.ema(ci.tolist(), self.n2))
        wt1 = tci
        wt2 = np.array(self.sma(wt1.tolist(), 4))
        return wt1, wt2

    # ------------------------------------------------------------------ #

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.n1 + self.n2 + _ATR_PERIOD + 10
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        highs  = [c.high  for c in candles]
        lows   = [c.low   for c in candles]
        closes = [c.close for c in candles]

        wt1, wt2 = self._wavetrend(highs, lows, closes)
        atr_arr  = self.atr(candles, _ATR_PERIOD)

        if np.isnan(wt1[-1]) or np.isnan(wt2[-1]) or np.isnan(atr_arr[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        curr_wt1 = float(wt1[-1]); prev_wt1 = float(wt1[-2])
        curr_wt2 = float(wt2[-1]); prev_wt2 = float(wt2[-2])
        current_atr = float(atr_arr[-1])

        # WT1 cross WT2 detection
        cross_up   = prev_wt1 <= prev_wt2 and curr_wt1 > curr_wt2
        cross_down = prev_wt1 >= prev_wt2 and curr_wt1 < curr_wt2

        # Zone gates: cross must happen in OS/OB territory
        buy_signal  = cross_up   and curr_wt1 < self.os_level and self._last_signal != 1
        sell_signal = cross_down and curr_wt1 > self.ob_level and self._last_signal != -1

        p = current_price
        conf = round(min(0.90, 0.55 + abs(curr_wt1) / 200), 2)

        if buy_signal:
            self._last_signal = 1
            sl_price = round(p - self.sl_atr_mult * current_atr, 4)
            tp_price = round(p + self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=p, amount=0.0,
                confidence=conf,
                reason=f"[WT] WT1 cross↑WT2 in OS | WT1={curr_wt1:.1f} (<{self.os_level})",
                metadata={
                    "wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2),
                    "atr": round(current_atr, 4),
                    "stop_loss": sl_price, "take_profit": tp_price, "rr": self.rr_ratio,
                },
            )

        if sell_signal:
            self._last_signal = -1
            sl_price = round(p + self.sl_atr_mult * current_atr, 4)
            tp_price = round(p - self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=p, amount=0.0,
                confidence=conf,
                reason=f"[WT] WT1 cross↓WT2 in OB | WT1={curr_wt1:.1f} (>{self.ob_level})",
                metadata={
                    "wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2),
                    "atr": round(current_atr, 4),
                    "stop_loss": sl_price, "take_profit": tp_price, "rr": self.rr_ratio,
                },
            )

        zone = f"OB({curr_wt1:.1f})" if curr_wt1 > self.ob_level else \
               f"OS({curr_wt1:.1f})" if curr_wt1 < self.os_level else \
               f"neutral({curr_wt1:.1f})"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[WT] {zone} | no cross",
            metadata={"wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2)},
        )
