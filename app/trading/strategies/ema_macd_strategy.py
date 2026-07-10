"""
EMA(12/26) Cross + SMA50 Trend Filter + 30m MACD Confirmation.

Simple, transparent, classic crossover strategy — separate from the
AI Decision Engine pipeline, selectable via STRATEGY=ema_macd.

Entry LONG (all three must align):
  1. EMA12 crosses above EMA26            (15m)
  2. Price closes above SMA50             (15m, uptrend filter)
  3. MACD(12,26,9) line > signal AND histogram > 0   (30m, trend confirmation)

Entry SHORT: mirror of the above (cross down, below SMA50, MACD 30m downtrend).

Exit LONG:  EMA12 crosses back below EMA26
            OR (MACD 30m turns bearish AND price pulls back down to touch SMA50)
Exit SHORT: mirror.

TP/SL: 1:1 R:R, distance = ATR(14) on 15m (SL/TP method wasn't specified in
the request — ATR-based is the same convention used elsewhere in this repo;
adjust rr_ratio or the ATR multiplier via params if you want a different spec).

30m candles aren't fetched separately — this strategy resamples them
directly from the 15m series it's given, so it needs no changes to the
bot's data-fetching loop.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


class EMAMacdStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        ema_fast: int = 12,
        ema_slow: int = 26,
        sma_trend: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rr_ratio: float = 1.0,
        atr_mult: float = 1.0,
    ):
        super().__init__(symbol, params)
        self.name = f"EMAMacd({symbol})"
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sma_trend = sma_trend
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rr_ratio = rr_ratio
        self.atr_mult = atr_mult

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._entry_price: float = 0.0

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        min_needed = max(self.ema_slow, self.sma_trend) + 5
        if len(candles) < min_needed:
            return self._hold(current_price, f"Need {min_needed}+ candles, have {len(candles)}")

        closes = [c.close for c in candles]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        sma50 = self.sma(closes, self.sma_trend)

        if np.isnan(ema_f[-2]) or np.isnan(ema_s[-2]) or np.isnan(sma50[-1]):
            return self._hold(current_price, "Indicators still warming up")

        cross_up = ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1]
        cross_down = ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1]
        price_above_sma50 = current_price > sma50[-1]
        price_below_sma50 = current_price < sma50[-1]

        macd_up, macd_down = self._macd_30m_trend(candles)

        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else current_price * 0.01
        last = candles[-1]
        touched_sma50 = last.low <= sma50[-1] <= last.high

        # ── Manage an open position first ───────────────────────────────────
        if self._open_position == "long":
            if cross_down or (macd_down and touched_sma50):
                reason = "EMA12 crossed below EMA26 (15m)" if cross_down \
                    else "MACD 30m turned bearish + price touched SMA50 (15m)"
                self._open_position = None
                return Signal(
                    type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
                    reason=f"Exit LONG: {reason}", confidence=1.0, metadata={"exit": True},
                )
            return self._hold(current_price, "Holding LONG — no exit condition yet")

        if self._open_position == "short":
            if cross_up or (macd_up and touched_sma50):
                reason = "EMA12 crossed above EMA26 (15m)" if cross_up \
                    else "MACD 30m turned bullish + price touched SMA50 (15m)"
                self._open_position = None
                return Signal(
                    type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                    reason=f"Exit SHORT: {reason}", confidence=1.0, metadata={"exit": True},
                )
            return self._hold(current_price, "Holding SHORT — no exit condition yet")

        # ── Look for a new entry ─────────────────────────────────────────────
        if cross_up and price_above_sma50 and macd_up:
            sl = current_price - self.atr_mult * atr_val
            tp = current_price + self.atr_mult * atr_val * self.rr_ratio
            self._open_position = "long"
            self._entry_price = current_price
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                reason="EMA12↑EMA26 (15m) + price>SMA50 (15m) + MACD 30m uptrend",
                confidence=1.0,
                metadata={"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio},
            )

        if cross_down and price_below_sma50 and macd_down:
            sl = current_price + self.atr_mult * atr_val
            tp = current_price - self.atr_mult * atr_val * self.rr_ratio
            self._open_position = "short"
            self._entry_price = current_price
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
                reason="EMA12↓EMA26 (15m) + price<SMA50 (15m) + MACD 30m downtrend",
                confidence=1.0,
                metadata={"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio},
            )

        return self._hold(current_price, "No entry: waiting for EMA cross + SMA50 + MACD30m alignment")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _macd_30m_trend(self, candles_15m: list) -> tuple[bool, bool]:
        """Resample 15m candles into 30m bars, then read MACD trend off them."""
        c30 = self._resample(candles_15m, minutes=30)
        needed = self.macd_slow + self.macd_signal + 5
        if len(c30) < needed:
            return False, False
        closes30 = [c.close for c in c30]
        macd_line, signal_line, hist = self.macd(closes30, self.macd_fast, self.macd_slow, self.macd_signal)
        if np.isnan(macd_line[-1]) or np.isnan(signal_line[-1]) or np.isnan(hist[-1]):
            return False, False
        macd_up = macd_line[-1] > signal_line[-1] and hist[-1] > 0
        macd_down = macd_line[-1] < signal_line[-1] and hist[-1] < 0
        return macd_up, macd_down

    @staticmethod
    def _resample(candles: list, minutes: int) -> list:
        """Group candles into `minutes`-wide buckets by timestamp (ms epoch)."""
        if not candles:
            return []
        bucket_ms = minutes * 60_000
        buckets: dict[int, list] = {}
        for c in candles:
            key = (c.timestamp // bucket_ms) * bucket_ms
            buckets.setdefault(key, []).append(c)

        class _Bar:
            __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
            def __init__(self, ts, o, h, l, cl, v):
                self.timestamp = ts; self.open = o; self.high = h
                self.low = l; self.close = cl; self.volume = v

        out = []
        for key in sorted(buckets):
            grp = buckets[key]
            out.append(_Bar(
                key, grp[0].open, max(g.high for g in grp), min(g.low for g in grp),
                grp[-1].close, sum(g.volume for g in grp),
            ))
        return out

    def _hold(self, price: float, reason: str = "") -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata={},
        )
