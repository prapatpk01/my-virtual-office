"""
SJ WaveTrend + ADX Filter — Python port of Pine Script v6.

BUY:
  A) wtDiff crosses above +5 AND ADX > threshold
  B) wtDiff already > 5 AND ADX just crossed above threshold
SELL:
  A) wtDiff crosses below -5 AND ADX > threshold
  B) wtDiff already < -5 AND ADX just crossed above threshold

State machine prevents duplicate consecutive signals.
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class WTADXStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.n1            = self.params.get("wt_channel_len",  10)
        self.n2            = self.params.get("wt_avg_len",      21)
        self.adx_period    = self.params.get("adx_period",      14)
        self.adx_threshold = self.params.get("adx_threshold",   20.0)
        self.wt_level      = self.params.get("wt_cross_level",  5.0)
        self.position_pct  = self.params.get("position_pct",    0.08)
        self._last_signal  = 0   # state machine: 1=buy, -1=sell

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
        d   = np.array(self.ema(np.abs(ap - esa).tolist(), self.n1))
        ci  = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
        tci = np.array(self.ema(ci.tolist(), self.n2))
        wt1 = tci
        wt2 = np.array(self.sma(wt1.tolist(), 4))
        return wt1, wt2, wt1 - wt2

    def _adx(self, highs, lows, closes):
        h = np.array(highs); l = np.array(lows); c = np.array(closes)
        n = len(c)

        up   = np.diff(h, prepend=h[0])
        down = -(np.diff(l, prepend=l[0]))
        plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        hl = h - l
        hc = np.abs(h - np.concatenate([[c[0]], c[:-1]]))
        lc = np.abs(l - np.concatenate([[c[0]], c[:-1]]))
        tr = np.maximum(hl, np.maximum(hc, lc))

        sm_tr    = self._rma(tr,       self.adx_period)
        sm_plus  = self._rma(plus_dm,  self.adx_period)
        sm_minus = self._rma(minus_dm, self.adx_period)

        denom    = np.clip(sm_tr, 1e-10, None)
        plus_di  = 100 * sm_plus  / denom
        minus_di = 100 * sm_minus / denom
        di_sum   = np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1.0)
        dx       = 100 * np.abs(plus_di - minus_di) / di_sum
        adx      = self._rma(np.nan_to_num(dx), self.adx_period)
        return adx, plus_di, minus_di

    # ------------------------------------------------------------------ #

    async def analyze(self, candles: list, current_price: float) -> Signal:
        min_len = self.n1 + self.n2 + self.adx_period + 10
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        highs  = [c.high   for c in candles]
        lows   = [c.low    for c in candles]
        closes = [c.close  for c in candles]

        wt1, wt2, wt_diff = self._wavetrend(highs, lows, closes)
        adx, plus_di, minus_di = self._adx(highs, lows, closes)

        if np.isnan(wt_diff[-1]) or np.isnan(adx[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        curr_diff = float(wt_diff[-1])
        prev_diff = float(wt_diff[-2])
        curr_adx  = float(adx[-1])
        prev_adx  = float(adx[-2])
        curr_wt1  = float(wt1[-1])

        # Cross helpers (Pine: crossover = was below, now above)
        wt_cross_up   = prev_diff <= self.wt_level  and curr_diff > self.wt_level
        wt_cross_down = prev_diff >= -self.wt_level and curr_diff < -self.wt_level
        adx_cross_up  = prev_adx <= self.adx_threshold and curr_adx > self.adx_threshold
        adx_above     = curr_adx > self.adx_threshold

        # Signal logic (robust, mirrors Pine state machine)
        buy_a = wt_cross_up   and adx_above
        buy_b = curr_diff > self.wt_level  and adx_cross_up
        sell_a = wt_cross_down and adx_above
        sell_b = curr_diff < -self.wt_level and adx_cross_up

        buy_signal  = (buy_a  or buy_b)  and self._last_signal != 1
        sell_signal = (sell_a or sell_b) and self._last_signal != -1

        conf = min(1.0, 0.5 + (curr_adx - self.adx_threshold) / 100)

        if buy_signal:
            self._last_signal = 1
            tag = "WT cross +5 + ADX" if buy_a else "WT>+5 + ADX cross"
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[WT+ADX] {tag} | WT={curr_diff:.2f} ADX={curr_adx:.1f}",
                confidence=conf,
                metadata={"wt_diff": curr_diff, "wt1": curr_wt1, "adx": curr_adx},
            )

        if sell_signal:
            self._last_signal = -1
            tag = "WT cross -5 + ADX" if sell_a else "WT<-5 + ADX cross"
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[WT+ADX] {tag} | WT={curr_diff:.2f} ADX={curr_adx:.1f}",
                confidence=conf,
                metadata={"wt_diff": curr_diff, "wt1": curr_wt1, "adx": curr_adx},
            )

        trend = "bull" if curr_diff > 0 else "bear"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[WT+ADX] WT={curr_diff:.2f} ({trend}) ADX={curr_adx:.1f} "
            f"{'(trend)' if adx_above else '(weak)'}",
            metadata={"wt_diff": curr_diff, "wt1": curr_wt1, "adx": curr_adx},
        )
