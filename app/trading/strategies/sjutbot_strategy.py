"""
SJ-UTBot v2 Strategy — 1H timeframe
ATR Trailing Stop (mult=0.30, len=14) on Heikin-Ashi close.
BUY : HA close crosses above TSL
SELL: HA close crosses below TSL
SL = entry ± sl_mult × ATR(14),  TP = SL × rr
Sequential (reversal) mode: SELL closes LONG, BUY closes SHORT.
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class SJUTBotStrategy(BaseStrategy):
    """
    Heikin-Ashi ATR Trailing Stop strategy.
    BUY/SELL fire when HA close crosses the TSL.
    Works best on 1H with REVERSAL_MODE=true in the bot.
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ut_mult   = self.params.get("ut_mult",   0.30)   # TSL ATR multiplier
        self.ut_len    = self.params.get("ut_len",    14)      # TSL ATR period
        self.sl_len    = self.params.get("sl_len",    14)      # SL/TP ATR period
        self.sl_mult   = self.params.get("sl_mult",   2.0)     # SL = sl_mult × ATR
        self.rr        = self.params.get("rr",        2.0)     # TP = SL × rr
        self.position_pct = self.params.get("position_pct", 0.08)

    def _wilder_atr(self, highs, lows, closes, period: int) -> np.ndarray:
        h, l, c = (np.asarray(x, dtype=float) for x in (highs, lows, closes))
        n = len(c)
        tr = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr = np.zeros(n)
        if n >= period:
            atr[period-1] = tr[:period].mean()
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr

    def _ha_close(self, candles) -> np.ndarray:
        o = np.array([c.open  for c in candles], dtype=float)
        h = np.array([c.high  for c in candles], dtype=float)
        l = np.array([c.low   for c in candles], dtype=float)
        c = np.array([c.close for c in candles], dtype=float)
        return (o + h + l + c) / 4.0

    def _tsl(self, ha_c: np.ndarray, ut_atr: np.ndarray) -> np.ndarray:
        """ATR trailing stop, Pine Script v2 logic."""
        n = len(ha_c)
        slval = ut_atr * self.ut_mult
        tsl = np.zeros(n)
        tsl[0] = ha_c[0] + slval[0]
        for i in range(1, n):
            src, src_p, pt, sv = ha_c[i], ha_c[i-1], tsl[i-1], slval[i]
            if src > pt and src_p > pt:
                tsl[i] = max(pt, src - sv)
            elif src < pt and src_p < pt:
                tsl[i] = min(pt, src + sv)
            elif src > pt:
                tsl[i] = src - sv
            else:
                tsl[i] = src + sv
        return tsl

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = max(self.ut_len, self.sl_len) + 20
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        h = [c.high  for c in candles]
        l = [c.low   for c in candles]
        c = [c.close for c in candles]

        ut_atr = self._wilder_atr(h, l, c, self.ut_len)
        sl_atr = self._wilder_atr(h, l, c, self.sl_len)
        ha_c   = self._ha_close(candles)
        tsl    = self._tsl(ha_c, ut_atr)

        # Crossover on last completed bar (i=-2 vs i=-1)
        buy_sig  = ha_c[-2] < tsl[-2] and ha_c[-1] > tsl[-1]
        sell_sig = ha_c[-2] > tsl[-2] and ha_c[-1] < tsl[-1]

        curr_atr = float(sl_atr[-1]) if sl_atr[-1] > 0 else current_price * 0.005
        sl_long  = round(current_price - self.sl_mult * curr_atr, 2)
        tp_long  = round(current_price + self.sl_mult * self.rr * curr_atr, 2)
        sl_short = round(current_price + self.sl_mult * curr_atr, 2)
        tp_short = round(current_price - self.sl_mult * self.rr * curr_atr, 2)
        tsl_val  = float(tsl[-1])

        if buy_sig:
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[SJUTBot] HA×TSL↑  TSL={tsl_val:.0f}  ATR={curr_atr:.0f}",
                confidence=0.65,
                metadata={"stop_loss": sl_long, "take_profit": tp_long,
                          "tsl": tsl_val, "atr": curr_atr},
            )
        if sell_sig:
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[SJUTBot] HA×TSL↓  TSL={tsl_val:.0f}  ATR={curr_atr:.0f}",
                confidence=0.65,
                metadata={"stop_loss": sl_short, "take_profit": tp_short,
                          "tsl": tsl_val, "atr": curr_atr},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[SJUTBot] HA={ha_c[-1]:.0f}  TSL={tsl_val:.0f}  ATR={curr_atr:.0f}",
            metadata={"tsl": tsl_val, "atr": curr_atr},
        )
