"""
EMA(12/26) Cross + SMA50 Trend Filter — pure TF15m, no MACD confirmation.

A simpler variant of ema_macd_strategy for comparison: same EMA/SMA50 core,
but with the 30m MACD confirmation layer removed entirely. Everything is
evaluated on TF15m only.

Entry LONG:
  1. EMA12 crosses above EMA26          (15m)
  2. Candle opens above SMA50           (15m, uptrend filter)

Entry SHORT: mirror (cross down, candle opens below SMA50).

SL/TP price levels use the same formula as ai_expert_strategy.py's Layer 5
Strategy Engine (entry_timing.py's _compute_sl_tp):
  - SL = wider of (recent swing high/low + 0.3xATR buffer) and (ATR x1.5),
    capped at 3xATR max width. ai_expert additionally scales the 1.5x
    multiplier by market regime (1.0-2.5x); this strategy has no regime
    classifier, so it always uses the TREND-regime default of 1.5x.
  - TP  = entry + 1.2R (R = |entry - SL|) — same rr_target as ai_expert.
  These levels are only the hard-stop safety net (checked by bot.py's
  risk-manager fallback against live price) — actual position closing is
  NOT R:R/TP1/TP2-driven like ai_expert. It stays purely signal-based:

Exit LONG:  EMA12 crosses back below EMA26  OR  price closes below SMA50
Exit SHORT: mirror.

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze() — in
hedge mode a SELL Signal always OPENS a new short rather than closing an
open long. tick_open_position()'s PositionUpdate("close") always closes
whichever position is actually open, regardless of hedge mode.
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
        sl_atr_mult: float = 1.5,
        rr_target: float = 1.2,
    ):
        super().__init__(symbol, params)
        self.name = f"EMASMA({symbol})"
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sma_trend = sma_trend
        self.sl_atr_mult = sl_atr_mult
        self.rr_target = rr_target

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
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)

        # ── Look for a new entry ─────────────────────────────────────────────
        if cross_up and open_above_sma50:
            sl, tp = self._compute_sl_tp("long", current_price, atr_val, highs, lows)
            self._open_position = "long"
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                reason="EMA12↑EMA26 (15m) + candle opened above SMA50 (15m)",
                confidence=1.0,
                metadata={"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_target},
            )

        if cross_down and open_below_sma50:
            sl, tp = self._compute_sl_tp("short", current_price, atr_val, highs, lows)
            self._open_position = "short"
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
                reason="EMA12↓EMA26 (15m) + candle opened below SMA50 (15m)",
                confidence=1.0,
                metadata={"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_target},
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

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Called by bot.py after ANY close, including ones this strategy
        didn't decide itself (e.g. the risk-manager's hard SL/TP fallback
        firing before EMA/SMA50 flips). Without this, _open_position would
        stay set forever and analyze() would refuse all future entries."""
        self._open_position = None

    def cancel_pending_entry(self, reason: str = "") -> None:
        """Called by bot.py when a signal this strategy just emitted failed
        to actually open (rejected by risk/portfolio gates, insufficient
        balance, or an order error) — undoes the optimistic _open_position
        set made in analyze() so the strategy isn't stuck believing a
        position exists that was never really opened."""
        self._open_position = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_sl_tp(self, direction: str, price: float, atr: float,
                        highs: np.ndarray, lows: np.ndarray) -> tuple[float, float]:
        """Same structure+ATR SL/TP formula as ai_expert_strategy's Layer 5
        Strategy Engine (entry_timing.py's _compute_sl_tp), minus the
        regime-based multiplier scaling (fixed at the TREND-regime default)."""
        lookback = min(20, len(highs))
        if direction == "long":
            swing_sl  = float(lows[-lookback:].min()) if lookback > 0 else price * 0.98
            struct_sl = swing_sl - 0.3 * atr
            atr_sl    = price - self.sl_atr_mult * atr
            sl        = min(struct_sl, atr_sl)          # wider of the two
            sl        = max(sl, price - 3.0 * atr)      # cap max width
            tp        = price + self.rr_target * (price - sl)
        else:  # short
            swing_sl  = float(highs[-lookback:].max()) if lookback > 0 else price * 1.02
            struct_sl = swing_sl + 0.3 * atr
            atr_sl    = price + self.sl_atr_mult * atr
            sl        = max(struct_sl, atr_sl)          # wider of the two
            sl        = min(sl, price + 3.0 * atr)      # cap max width
            tp        = price - self.rr_target * (sl - price)
        return sl, tp

    def _hold(self, price: float, reason: str = "") -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata={},
        )
