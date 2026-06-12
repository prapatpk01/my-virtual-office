"""
MACD + EMA Confluence Strategy.

BUY  — EMA bullish stack (EMA9 > EMA21 > EMA50)
       + MACD line crosses above signal line
       + MACD histogram turns positive
       + price above EMA50

SELL — EMA bearish stack (EMA9 < EMA21 < EMA50)
       + MACD line crosses below signal line
       + MACD histogram turns negative
       + price below EMA50
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MACDEMAStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ema_fast     = self.params.get("ema_fast",     9)
        self.ema_mid      = self.params.get("ema_mid",      21)
        self.ema_slow     = self.params.get("ema_slow",     50)
        self.macd_fast    = self.params.get("macd_fast",    12)
        self.macd_slow    = self.params.get("macd_slow",    26)
        self.macd_sig     = self.params.get("macd_signal",  9)
        self.position_pct = self.params.get("position_pct", 0.08)

    async def analyze(self, candles: list, current_price: float) -> Signal:
        min_len = self.ema_slow + self.macd_slow + self.macd_sig + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]
        c_arr  = np.array(closes)

        # EMA stack
        ema9  = np.array(self.ema(closes, self.ema_fast))
        ema21 = np.array(self.ema(closes, self.ema_mid))
        ema50 = np.array(self.ema(closes, self.ema_slow))

        # MACD
        macd_line, signal_line, histogram = self.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_sig
        )

        # Current values
        e9_c  = float(ema9[-1]);  e9_p  = float(ema9[-2])
        e21_c = float(ema21[-1]); e21_p = float(ema21[-2])
        e50_c = float(ema50[-1])
        ml_c  = float(macd_line[-1]); ml_p = float(macd_line[-2])
        sl_c  = float(signal_line[-1]); sl_p = float(signal_line[-2])
        h_c   = float(histogram[-1]);  h_p  = float(histogram[-2])
        price = current_price

        if any(np.isnan(v) for v in [e9_c, e21_c, e50_c, ml_c, sl_c, h_c]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        # ── Conditions ────────────────────────────────────────────────
        ema_bull_stack  = e9_c > e21_c > e50_c
        ema_bear_stack  = e9_c < e21_c < e50_c
        price_above_50  = price > e50_c
        price_below_50  = price < e50_c
        macd_cross_up   = ml_p < sl_p and ml_c > sl_c   # MACD line crosses above signal
        macd_cross_down = ml_p > sl_p and ml_c < sl_c   # MACD line crosses below signal
        hist_positive   = h_c > 0
        hist_negative   = h_c < 0
        hist_turning_up  = h_c > h_p   # histogram rising
        hist_turning_dn  = h_c < h_p   # histogram falling

        # EMA slope strength (how aligned the stack is)
        ema_bull_strength = ((e9_c - e21_c) / e21_c + (e21_c - e50_c) / e50_c) * 100
        ema_bear_strength = ((e21_c - e9_c) / e21_c + (e50_c - e21_c) / e50_c) * 100

        # ── BUY ───────────────────────────────────────────────────────
        # Full confluence: EMA stack + MACD cross up + above EMA50
        buy_full = ema_bull_stack and macd_cross_up and price_above_50
        # Partial: EMA stack + histogram just turned positive
        buy_partial = ema_bull_stack and hist_positive and hist_turning_up and price_above_50

        if buy_full or buy_partial:
            conf = min(1.0, 0.5 + min(ema_bull_strength * 5, 0.3) + (0.2 if buy_full else 0.0))
            tag  = "EMA stack + MACD crossover" if buy_full else "EMA stack + MACD hist rising"
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[MACD+EMA] {tag} | EMA9={e9_c:.0f} EMA21={e21_c:.0f} EMA50={e50_c:.0f}",
                confidence=conf,
                metadata={"ema9": e9_c, "ema21": e21_c, "ema50": e50_c,
                          "macd": ml_c, "signal": sl_c, "hist": h_c},
            )

        # ── SELL ──────────────────────────────────────────────────────
        sell_full    = ema_bear_stack and macd_cross_down and price_below_50
        sell_partial = ema_bear_stack and hist_negative and hist_turning_dn and price_below_50

        if sell_full or sell_partial:
            conf = min(1.0, 0.5 + min(ema_bear_strength * 5, 0.3) + (0.2 if sell_full else 0.0))
            tag  = "EMA stack + MACD crossover" if sell_full else "EMA stack + MACD hist falling"
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[MACD+EMA] {tag} | EMA9={e9_c:.0f} EMA21={e21_c:.0f} EMA50={e50_c:.0f}",
                confidence=conf,
                metadata={"ema9": e9_c, "ema21": e21_c, "ema50": e50_c,
                          "macd": ml_c, "signal": sl_c, "hist": h_c},
            )

        # ── HOLD ──────────────────────────────────────────────────────
        stack = "bull" if ema_bull_stack else ("bear" if ema_bear_stack else "mixed")
        macd_pos = "▲" if h_c > 0 else "▼"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MACD+EMA] EMA {stack} | MACD {macd_pos}{h_c:.4f} | "
            f"EMA9={e9_c:.0f} EMA21={e21_c:.0f} EMA50={e50_c:.0f}",
            metadata={"ema9": e9_c, "ema21": e21_c, "ema50": e50_c,
                      "macd": ml_c, "signal": sl_c, "hist": h_c},
        )
