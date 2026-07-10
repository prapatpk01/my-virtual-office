"""
EMA(12/26) Cross + SMA50 Trend Filter — pure TF15m, no MACD confirmation.

A simpler variant of ema_macd_strategy for comparison: same EMA/SMA50 core,
but with the 30m MACD confirmation layer removed entirely. Everything is
evaluated on TF15m only.

Entry LONG:
  1. EMA12 crosses above EMA26          (15m)
  2. Candle opens above SMA50           (15m, uptrend filter)

Entry SHORT: mirror (cross down, candle opens below SMA50).

Exit LONG:  EMA12 crosses back below EMA26  OR  price closes below SMA50
Exit SHORT: mirror.

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze() — in
hedge mode a SELL Signal always OPENS a new short rather than closing an
open long. tick_open_position()'s PositionUpdate("close") always closes
whichever position is actually open, regardless of hedge mode.

TP/SL: 1:1 R:R, distance = ATR(14) on 15m x 2.0 (not specified in the
request; ATR-based is the same convention used elsewhere in this repo —
the 2x multiplier avoids stops that are too tight for a single 15m bar's
ATR; adjust atr_mult for a different distance).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


class EMASMAStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        ema_fast: int = 12,
        ema_slow: int = 26,
        sma_trend: int = 50,
        rr_ratio: float = 1.0,
        atr_mult: float = 2.0,
    ):
        super().__init__(symbol, params)
        self.name = f"EMASMA({symbol})"
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sma_trend = sma_trend
        self.rr_ratio = rr_ratio
        self.atr_mult = atr_mult

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._latest_candles: list = []

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        min_needed = max(self.ema_slow, self.sma_trend) + 5
        if len(candles) < min_needed:
            return self._hold(current_price, f"Need {min_needed}+ candles, have {len(candles)}")

        self._latest_candles = candles  # cached for tick_open_position()

        if self._open_position is not None:
            return self._hold(current_price, f"Holding {self._open_position.upper()} — managed via tick_open_position()")

        closes = [c.close for c in candles]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        sma50 = self.sma(closes, self.sma_trend)

        if np.isnan(ema_f[-2]) or np.isnan(ema_s[-2]) or np.isnan(sma50[-1]):
            return self._hold(current_price, "Indicators still warming up")

        cross_up = ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1]
        cross_down = ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1]

        last = candles[-1]
        open_above_sma50 = last.open > sma50[-1]
        open_below_sma50 = last.open < sma50[-1]

        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else current_price * 0.01

        # ── Look for a new entry ─────────────────────────────────────────────
        if cross_up and open_above_sma50:
            sl = current_price - self.atr_mult * atr_val
            tp = current_price + self.atr_mult * atr_val * self.rr_ratio
            self._open_position = "long"
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                reason="EMA12↑EMA26 (15m) + candle opened above SMA50 (15m)",
                confidence=1.0,
                metadata={"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio},
            )

        if cross_down and open_below_sma50:
            sl = current_price + self.atr_mult * atr_val
            tp = current_price - self.atr_mult * atr_val * self.rr_ratio
            self._open_position = "short"
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
                reason="EMA12↓EMA26 (15m) + candle opened below SMA50 (15m)",
                confidence=1.0,
                metadata={"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio},
            )

        return self._hold(current_price, "No entry: waiting for EMA cross + SMA50 open-side alignment")

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Called every bot tick while a position is open. Hedge-mode-safe
        exit path — closes via PositionUpdate, never a SELL/BUY Signal."""
        if self._open_position is None or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate

        candles = self._latest_candles
        closes = [c.close for c in candles]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        sma50 = self.sma(closes, self.sma_trend)
        if np.isnan(ema_f[-2]) or np.isnan(ema_s[-2]) or np.isnan(sma50[-1]):
            return PositionUpdate(action="hold", reason="Indicators warming up")

        cross_up = ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1]
        cross_down = ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1]
        last = candles[-1]
        close_below_sma50 = last.close < sma50[-1]
        close_above_sma50 = last.close > sma50[-1]

        if self._open_position == "long":
            if cross_down or close_below_sma50:
                reason = "EMA12 crossed below EMA26 (15m)" if cross_down \
                    else "Price closed below SMA50 (15m)"
                self._open_position = None
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit LONG: {reason}")
            return PositionUpdate(action="hold", reason="Holding LONG")

        if self._open_position == "short":
            if cross_up or close_above_sma50:
                reason = "EMA12 crossed above EMA26 (15m)" if cross_up \
                    else "Price closed above SMA50 (15m)"
                self._open_position = None
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit SHORT: {reason}")
            return PositionUpdate(action="hold", reason="Holding SHORT")

        return PositionUpdate(action="hold")

    def _hold(self, price: float, reason: str = "") -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata={},
        )
