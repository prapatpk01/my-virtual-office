"""
MACD + EMA + HMA15 Confluence Strategy — tuned for 15m timeframe.

HMA15 filter (primary gate):
  Price close > HMA15 → BUY side only
  Price close < HMA15 → SELL side only

Signal tiers (most → least strict):
  A) Full confluence: EMA9>21>50 + MACD cross up + price > EMA21
  B) MACD crossover alone with price > EMA9
  C) EMA9 > EMA21 + MACD histogram crosses zero upward

Same logic inverted for SELL.
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MACDEMAStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ema_fast     = self.params.get("ema_fast",     9)
        self.ema_mid      = self.params.get("ema_mid",      21)
        self.ema_slow     = self.params.get("ema_slow",     50)
        self.hma_period   = self.params.get("hma_period",   15)
        self.macd_fast    = self.params.get("macd_fast",    12)
        self.macd_slow    = self.params.get("macd_slow",    26)
        self.macd_sig     = self.params.get("macd_signal",  9)
        self.position_pct = self.params.get("position_pct", 0.08)

    async def analyze(self, candles: list, current_price: float) -> Signal:
        min_len = self.ema_slow + self.macd_slow + self.macd_sig + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]

        ema9  = np.array(self.ema(closes, self.ema_fast))
        ema21 = np.array(self.ema(closes, self.ema_mid))
        ema50 = np.array(self.ema(closes, self.ema_slow))
        hma15 = np.array(self.hma(closes, self.hma_period))
        macd_line, signal_line, histogram = self.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_sig
        )

        e9    = float(ema9[-1])
        e21   = float(ema21[-1])
        e50   = float(ema50[-1])
        hma   = float(hma15[-1])
        ml_c  = float(macd_line[-1]);  ml_p = float(macd_line[-2])
        sl_c  = float(signal_line[-1]); sl_p = float(signal_line[-2])
        h_c   = float(histogram[-1]);   h_p  = float(histogram[-2])
        p     = current_price

        if any(np.isnan(v) for v in [e9, e21, e50, hma, ml_c, sl_c, h_c]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        # HMA15 gate — primary filter
        above_hma = p > hma
        below_hma = p < hma

        # ── Helpers ───────────────────────────────────────────────────
        macd_cross_up   = ml_p < sl_p and ml_c > sl_c
        macd_cross_down = ml_p > sl_p and ml_c < sl_c
        hist_up   = h_c > h_p   # histogram rising
        hist_down = h_c < h_p   # histogram falling

        # ── BUY tiers (only when price > HMA15) ──────────────────────
        buy_A = above_hma and (e9 > e21 > e50) and macd_cross_up and p > e21
        buy_B = above_hma and macd_cross_up and p > e9
        buy_C = above_hma and (e9 > e21) and (h_c > 0 > h_p) and p > e21

        if buy_A or buy_B or buy_C:
            conf = 0.80 if buy_A else (0.65 if buy_B else 0.55)
            tag  = ("Full stack+MACD cross" if buy_A
                    else "MACD crossover" if buy_B
                    else "EMA9>21 + hist cross 0")
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=p,
                amount=self.position_pct,
                reason=f"[MACD+EMA] {tag} | HMA15={hma:.0f} EMA9={e9:.0f} EMA21={e21:.0f}",
                confidence=conf,
                metadata={"hma15": hma, "ema9": e9, "ema21": e21, "ema50": e50,
                          "macd": ml_c, "signal": sl_c, "hist": h_c},
            )

        # ── SELL tiers (only when price < HMA15) ─────────────────────
        sell_A = below_hma and (e9 < e21 < e50) and macd_cross_down and p < e21
        sell_B = below_hma and macd_cross_down and p < e9
        sell_C = below_hma and (e9 < e21) and (h_c < 0 < h_p) and p < e21

        if sell_A or sell_B or sell_C:
            conf = 0.80 if sell_A else (0.65 if sell_B else 0.55)
            tag  = ("Full stack+MACD cross" if sell_A
                    else "MACD crossover" if sell_B
                    else "EMA9<21 + hist cross 0")
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=p,
                amount=self.position_pct,
                reason=f"[MACD+EMA] {tag} | HMA15={hma:.0f} EMA9={e9:.0f} EMA21={e21:.0f}",
                confidence=conf,
                metadata={"hma15": hma, "ema9": e9, "ema21": e21, "ema50": e50,
                          "macd": ml_c, "signal": sl_c, "hist": h_c},
            )

        # ── HOLD ──────────────────────────────────────────────────────
        stack    = "bull" if e9 > e21 > e50 else ("bear" if e9 < e21 < e50 else "mixed")
        hma_side = "above" if above_hma else "below"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MACD+EMA] {stack} | price {hma_side} HMA15={hma:.0f} "
            f"EMA9={e9:.0f} hist={'▲' if h_c > 0 else '▼'}{h_c:.4f}",
            metadata={"hma15": hma, "ema9": e9, "ema21": e21, "ema50": e50,
                      "macd": ml_c, "signal": sl_c, "hist": h_c},
        )
