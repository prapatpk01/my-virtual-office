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

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze(). In
hedge mode a SELL Signal always OPENS a new short (independent of any open
long) — returning one to mean "close" would silently open the wrong
position. tick_open_position()'s PositionUpdate("close") always closes the
position that's actually open, regardless of hedge mode.

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
        price_above_sma50 = current_price > sma50[-1]
        price_below_sma50 = current_price < sma50[-1]

        macd_up, macd_down = self._macd_30m_trend(candles)

        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else current_price * 0.01

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
        macd_up, macd_down = self._macd_30m_trend(candles)
        last = candles[-1]
        touched_sma50 = last.low <= sma50[-1] <= last.high

        if self._open_position == "long":
            if cross_down or (macd_down and touched_sma50):
                reason = "EMA12 crossed below EMA26 (15m)" if cross_down \
                    else "MACD 30m turned bearish + price touched SMA50 (15m)"
                self._open_position = None
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit LONG: {reason}")
            return PositionUpdate(action="hold", reason="Holding LONG")

        if self._open_position == "short":
            if cross_up or (macd_up and touched_sma50):
                reason = "EMA12 crossed above EMA26 (15m)" if cross_up \
                    else "MACD 30m turned bullish + price touched SMA50 (15m)"
                self._open_position = None
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit SHORT: {reason}")
            return PositionUpdate(action="hold", reason="Holding SHORT")

        return PositionUpdate(action="hold")

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Called by bot.py after ANY close, including ones this strategy
        didn't decide itself (e.g. the risk-manager's hard SL/TP fallback
        firing before EMA/MACD30m flips). Without this, _open_position would
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
        """Group candles into `minutes`-wide buckets by timestamp (ms epoch).
        Drops the last bucket if it hasn't fully elapsed yet, so MACD 30m
        confirmation only ever reads a genuinely closed 30m candle."""
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

        keys = sorted(buckets)
        out = []
        for key in keys:
            grp = buckets[key]
            out.append(_Bar(
                key, grp[0].open, max(g.high for g in grp), min(g.low for g in grp),
                grp[-1].close, sum(g.volume for g in grp),
            ))

        last_c15_end = candles[-1].timestamp + 15 * 60_000
        last_bucket_end = keys[-1] + bucket_ms
        if last_c15_end < last_bucket_end:
            out = out[:-1]

        return out

    def _hold(self, price: float, reason: str = "") -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata={},
        )
