"""Simple Precision — one deterministic 4H/1H/15M trading strategy.

The strategy intentionally keeps each timeframe responsible for one job:

* 4H selects direction. It never triggers an order.
* 1H checks whether the trend is tradable. It never changes direction.
* 15M owns entry timing, stop placement, and technical exits.

Only closed candles are used for decisions.  There is one strategy instance
per symbol and at most one position per instance.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType
from ..engines.position_manager import PositionUpdate


class SimplePrecisionStrategy(BaseStrategy):
    """Fast, compact trend-continuation strategy for 15-minute execution."""

    VERSION = "1.0"
    entry_tf = "15m"

    def __init__(
        self,
        symbol: str,
        *,
        quality_threshold: float = 55.0,
        adx_min: float = 15.0,
        chop_max: float = 62.0,
        max_entry_distance_atr: float = 1.50,
        min_room_r: float = 1.20,
        stop_atr_min: float = 0.70,
        stop_atr_max: float = 1.40,
        target_r: float = 2.0,
        tp1_r: float = 1.0,
        tp1_trim_pct: float = 0.40,
        exit_cooldown_bars: int = 2,
    ):
        super().__init__(symbol)
        self.name = f"SimplePrecision({symbol})"

        self.quality_threshold = float(quality_threshold)
        self.adx_min = float(adx_min)
        self.chop_max = float(chop_max)
        self.max_entry_distance_atr = float(max_entry_distance_atr)
        self.min_room_r = float(min_room_r)
        self.stop_atr_min = float(stop_atr_min)
        self.stop_atr_max = float(stop_atr_max)
        self.target_r = float(target_r)
        self.tp1_r = float(tp1_r)
        self.tp1_trim_pct = float(tp1_trim_pct)
        self.exit_cooldown_bars = max(0, int(exit_cooldown_bars))

        # TradingBot temporarily raises this after a loss-streak cooldown.
        self._entry_threshold_bonus = 0.0

        self._open_position: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._entry_sl: Optional[float] = None
        self._entry_tp: Optional[float] = None
        self._initial_risk: Optional[float] = None
        self._tp1_done = False
        self._pending_entry = False

        self._latest_15m: list = []
        self._last_evaluated_bar_ts: Optional[int] = None
        self._last_exit_check_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None

    # ------------------------------------------------------------------
    # Candle and indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_ts(candle) -> int:
        return int(getattr(candle, "timestamp", 0) or 0)

    @classmethod
    def _closed_candle_series(
        cls,
        candles: list,
        timeframe_ms: int,
        grace_ms: int = 1500,
    ) -> list:
        """Return candles whose full interval has elapsed.

        Historical/backtest timestamps below 1e11 are treated as already
        closed; live exchange timestamps are milliseconds since epoch.
        """
        data = list(candles or [])
        if not data:
            return []
        ts = cls._bar_ts(data[-1])
        if ts < 100_000_000_000:
            return data
        now_ms = int(time.time() * 1000)
        if ts + int(timeframe_ms) > now_ms - int(grace_ms):
            return data[:-1]
        return data

    @staticmethod
    def _finite(*values) -> bool:
        return all(v is not None and np.isfinite(float(v)) for v in values)

    @staticmethod
    def _choppiness(candles: list, period: int = 14) -> Optional[float]:
        if len(candles) < period + 1:
            return None
        window = candles[-period:]
        prev = candles[-period - 1]
        tr_sum = 0.0
        previous_close = float(prev.close)
        for candle in window:
            high = float(candle.high)
            low = float(candle.low)
            tr_sum += max(high - low, abs(high - previous_close), abs(low - previous_close))
            previous_close = float(candle.close)
        price_range = max(float(c.high) for c in window) - min(float(c.low) for c in window)
        if tr_sum <= 0 or price_range <= 0:
            return 100.0
        return 100.0 * math.log10(tr_sum / price_range) / math.log10(period)

    @staticmethod
    def _recent_pivots(candles: list, span: int = 2) -> tuple[list[float], list[float]]:
        highs: list[float] = []
        lows: list[float] = []
        for i in range(span, len(candles) - span):
            high = float(candles[i].high)
            low = float(candles[i].low)
            before = candles[i - span:i]
            after = candles[i + 1:i + span + 1]
            if high >= max(float(c.high) for c in before + after):
                highs.append(high)
            if low <= min(float(c.low) for c in before + after):
                lows.append(low)
        return highs, lows

    # ------------------------------------------------------------------
    # Layer 1 — 4H direction only
    # ------------------------------------------------------------------

    def _macro_4h(self, candles: list) -> dict:
        if len(candles) < 55:
            return {"ready": False, "direction": None, "score": 0.0, "reason": "4H warmup"}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)
        atr = self.atr(candles, 14)
        if not self._finite(ema20[-1], ema20[-4], ema50[-1], atr[-1]):
            return {"ready": False, "direction": None, "score": 0.0, "reason": "4H indicators unavailable"}

        close = closes[-1]
        atr_now = max(float(atr[-1]), 1e-12)
        slope_atr = (float(ema20[-1]) - float(ema20[-4])) / atr_now

        bull = {
            "ema_stack": float(ema20[-1]) > float(ema50[-1]),
            "ema_slope": slope_atr > 0.0,
            "price_side": close > float(ema20[-1]),
        }
        bear = {
            "ema_stack": float(ema20[-1]) < float(ema50[-1]),
            "ema_slope": slope_atr < 0.0,
            "price_side": close < float(ema20[-1]),
        }
        weights = {"ema_stack": 40.0, "ema_slope": 30.0, "price_side": 30.0}
        bull_score = sum(weights[k] for k, passed in bull.items() if passed)
        bear_score = sum(weights[k] for k, passed in bear.items() if passed)

        # The EMA stack is mandatory; one of slope/price then confirms it.
        if bull["ema_stack"] and bull_score >= 70.0:
            direction, score = "long", bull_score
        elif bear["ema_stack"] and bear_score >= 70.0:
            direction, score = "short", bear_score
        else:
            direction, score = None, max(bull_score, bear_score)

        return {
            "ready": direction is not None,
            "direction": direction,
            "score": round(score, 1),
            "bull_score": round(bull_score, 1),
            "bear_score": round(bear_score, 1),
            "ema20": round(float(ema20[-1]), 8),
            "ema50": round(float(ema50[-1]), 8),
            "slope_atr": round(float(slope_atr), 3),
            "votes": bull if direction == "long" else bear if direction == "short" else {},
            "reason": "direction confirmed" if direction else "4H direction mixed",
        }

    # ------------------------------------------------------------------
    # Layer 2 — 1H quality only
    # ------------------------------------------------------------------

    def _quality_1h(self, candles: list, direction: str) -> dict:
        if direction not in {"long", "short"} or len(candles) < 55:
            return {"ready": False, "score": 0.0, "reason": "1H warmup"}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        adx, plus_di, minus_di = self.adx(candles, 14)
        macd, macd_signal, macd_hist = self.macd(closes)
        chop = self._choppiness(candles, 14)
        if chop is None or not self._finite(
            ema20[-1], adx[-1], plus_di[-1], minus_di[-1], macd[-1], macd_signal[-1]
        ):
            return {"ready": False, "score": 0.0, "reason": "1H indicators unavailable"}

        close = closes[-1]
        adx_now = float(adx[-1])
        plus_now = float(plus_di[-1])
        minus_now = float(minus_di[-1])
        hist_now = float(macd_hist[-1]) if np.isfinite(macd_hist[-1]) else float(macd[-1] - macd_signal[-1])

        if adx_now >= 25.0:
            adx_score = 25.0
        elif adx_now >= 20.0:
            adx_score = 20.0
        elif adx_now >= self.adx_min:
            adx_score = 12.0
        else:
            adx_score = 0.0

        if chop < 45.0:
            chop_score = 25.0
        elif chop < 50.0:
            chop_score = 20.0
        elif chop < 55.0:
            chop_score = 12.0
        elif chop < self.chop_max:
            chop_score = 5.0
        else:
            chop_score = 0.0

        di_aligned = plus_now > minus_now if direction == "long" else minus_now > plus_now
        momentum_aligned = hist_now >= 0.0 if direction == "long" else hist_now <= 0.0
        price_aligned = close >= float(ema20[-1]) if direction == "long" else close <= float(ema20[-1])

        components = {
            "adx": adx_score,
            "chop": chop_score,
            "directional": 20.0 if di_aligned else 0.0,
            "momentum": 15.0 if momentum_aligned else 0.0,
            "price": 15.0 if price_aligned else 0.0,
        }
        score = sum(components.values())
        threshold = min(70.0, self.quality_threshold + max(0.0, float(self._entry_threshold_bonus)))

        # Only obvious non-trend conditions are hard blocks. A single lagging
        # momentum reading merely lowers the score and cannot veto a good setup.
        hard_blocks = []
        if adx_now < self.adx_min:
            hard_blocks.append("ADX")
        if chop >= self.chop_max:
            hard_blocks.append("CHOP")
        if not di_aligned and not momentum_aligned and not price_aligned:
            hard_blocks.append("FULL_OPPOSITION")

        ready = score >= threshold and not hard_blocks
        return {
            "ready": ready,
            "score": round(score, 1),
            "threshold": round(threshold, 1),
            "adx": round(adx_now, 1),
            "chop": round(float(chop), 1),
            "plus_di": round(plus_now, 1),
            "minus_di": round(minus_now, 1),
            "momentum_aligned": momentum_aligned,
            "price_aligned": price_aligned,
            "components": components,
            "hard_blocks": hard_blocks,
            "reason": "quality pass" if ready else ("blocked: " + ",".join(hard_blocks) if hard_blocks else "quality below threshold"),
        }

    # ------------------------------------------------------------------
    # Layer 3 — closed-15M entry timing
    # ------------------------------------------------------------------

    def _entry_15m(self, candles: list, direction: str, current_price: float) -> dict:
        if len(candles) < 35:
            return {"trigger": None, "reason": "15M warmup"}

        closes = [float(c.close) for c in candles]
        ema8 = self.ema(closes, 8)
        ema13 = self.ema(closes, 13)
        ema20 = self.ema(closes, 20)
        atr = self.atr(candles, 14)
        if not self._finite(ema8[-1], ema13[-1], ema20[-1], atr[-1]):
            return {"trigger": None, "reason": "15M indicators unavailable"}

        atr_now = max(float(atr[-1]), 1e-12)
        close = closes[-1]
        previous_close = closes[-2]
        candle = candles[-1]
        body = abs(float(candle.close) - float(candle.open))

        long = direction == "long"
        aligned = (ema8[-1] > ema13[-1] and close > ema8[-1]) if long else (ema8[-1] < ema13[-1] and close < ema8[-1])
        cross_now = (ema8[-2] <= ema13[-2] and ema8[-1] > ema13[-1]) if long else (ema8[-2] >= ema13[-2] and ema8[-1] < ema13[-1])
        cross_prev = (ema8[-3] <= ema13[-3] and ema8[-2] > ema13[-2]) if long else (ema8[-3] >= ema13[-3] and ema8[-2] < ema13[-2])

        touched = (
            float(candle.low) <= float(ema13[-1]) or previous_close <= float(ema13[-2])
        ) if long else (
            float(candle.high) >= float(ema13[-1]) or previous_close >= float(ema13[-2])
        )
        reclaim = touched and aligned and (
            float(candle.close) > float(candle.open) if long else float(candle.close) < float(candle.open)
        )

        lookback = 7
        level = (
            max(float(c.high) for c in candles[-lookback - 1:-1])
            if long else min(float(c.low) for c in candles[-lookback - 1:-1])
        )
        breakout = (close > level and previous_close <= level) if long else (close < level and previous_close >= level)
        volumes = [float(c.volume or 0.0) for c in candles[-21:-1]]
        median_volume = float(np.median(volumes)) if volumes else 0.0
        volume_ok = median_volume <= 0.0 or float(candle.volume or 0.0) >= 0.90 * median_volume
        breakout = breakout and volume_ok and body >= 0.15 * atr_now

        if aligned and (cross_now or cross_prev):
            trigger = "EMA8_13_CROSS"
        elif reclaim:
            trigger = "EMA13_PULLBACK_RECLAIM"
        elif aligned and breakout:
            trigger = "STRUCTURE_BREAKOUT"
        else:
            trigger = None

        distance_atr = abs(close - float(ema20[-1])) / atr_now
        chase_block = distance_atr > self.max_entry_distance_atr

        # Risk is volatility-aware but bounded, which avoids both tiny stops
        # and stops so wide that leverage dominates the trade.
        if long:
            raw_risk = close - min(float(c.low) for c in candles[-7:]) + 0.10 * atr_now
        else:
            raw_risk = max(float(c.high) for c in candles[-7:]) - close + 0.10 * atr_now
        risk = min(max(raw_risk, self.stop_atr_min * atr_now), self.stop_atr_max * atr_now)
        risk = max(risk, close * 0.001)

        pivots_high, pivots_low = self._recent_pivots(candles[-50:])
        opposing = [p for p in (pivots_high if long else pivots_low) if p > close] if long else [p for p in pivots_low if p < close]
        nearest = min(opposing) if long and opposing else max(opposing) if (not long and opposing) else None
        room_r = abs(nearest - close) / risk if nearest is not None else 3.0
        room_block = room_r < self.min_room_r

        if trigger is None:
            reason = "waiting for EMA cross, pullback reclaim, or structure breakout"
        elif chase_block:
            reason = "trigger ready but price is extended"
            trigger = None
        elif room_block:
            reason = "trigger ready but opposing structure is too close"
            trigger = None
        else:
            reason = "entry trigger confirmed"

        entry = float(current_price or close)
        stop = entry - risk if long else entry + risk
        target = entry + self.target_r * risk if long else entry - self.target_r * risk
        return {
            "trigger": trigger,
            "reason": reason,
            "entry": entry,
            "stop_loss": float(stop),
            "take_profit": float(target),
            "risk": float(risk),
            "atr": atr_now,
            "distance_atr": round(float(distance_atr), 2),
            "room_r": round(float(room_r), 2),
            "nearest_opposing": nearest,
            "ema8": round(float(ema8[-1]), 8),
            "ema13": round(float(ema13[-1]), 8),
            "ema20": round(float(ema20[-1]), 8),
            "aligned": bool(aligned),
        }

    # ------------------------------------------------------------------
    # Public strategy lifecycle
    # ------------------------------------------------------------------

    def _hold(self, price: float, reason: str, metadata: dict) -> Signal:
        return Signal(SignalType.HOLD, self.symbol, float(price), 0.0, reason, 0.0, metadata)

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c15 = self._closed_candle_series(candles, 15 * 60_000)
        c1h = self._closed_candle_series(mtf.get("1h", []), 60 * 60_000)
        c4h = self._closed_candle_series(mtf.get("4h", []), 4 * 60 * 60_000)
        self._latest_15m = c15

        metadata = {
            "strategy": "SIMPLE_PRECISION",
            "version": self.VERSION,
            "architecture": "4H_DIRECTION__1H_QUALITY__15M_TRIGGER",
        }
        if not c15 or not c1h or not c4h:
            return self._hold(current_price, "waiting for complete MTF candles", metadata)

        macro = self._macro_4h(c4h)
        metadata["macro_4h"] = macro
        if not macro["ready"]:
            return self._hold(current_price, macro["reason"], metadata)

        quality = self._quality_1h(c1h, macro["direction"])
        metadata["quality_1h"] = quality
        if not quality["ready"]:
            return self._hold(current_price, quality["reason"], metadata)

        entry = self._entry_15m(c15, macro["direction"], current_price)
        metadata["entry_15m"] = entry

        bar_ts = self._bar_ts(c15[-1])
        if self._open_position is not None:
            return self._hold(current_price, f"managing open {self._open_position} position", metadata)
        if self._last_evaluated_bar_ts == bar_ts:
            return self._hold(current_price, "15M bar already evaluated", metadata)
        self._last_evaluated_bar_ts = bar_ts

        if self._last_exit_bar_ts is not None:
            elapsed = bar_ts - self._last_exit_bar_ts
            if elapsed < self.exit_cooldown_bars * 15 * 60_000:
                return self._hold(current_price, "post-exit cooldown", metadata)

        if not entry.get("trigger"):
            return self._hold(current_price, entry["reason"], metadata)

        direction = macro["direction"]
        signal_type = SignalType.BUY if direction == "long" else SignalType.SELL
        self._open_position = direction
        self._pending_entry = True
        self._entry_price = float(entry["entry"])
        self._entry_sl = float(entry["stop_loss"])
        self._entry_tp = float(entry["take_profit"])
        self._initial_risk = abs(self._entry_price - self._entry_sl)
        self._tp1_done = False

        metadata.update({
            "direction": direction,
            "entry_trigger": entry["trigger"],
            "stop_loss": round(self._entry_sl, 8),
            "take_profit": round(self._entry_tp, 8),
            "rr_ratio": self.target_r,
            "tp1_r": self.tp1_r,
            "tp1_close_pct": self.tp1_trim_pct,
            "risk_plan": f"SL_{self._initial_risk:.8f}__T1_{self.tp1_r:.1f}R_TRIM{self.tp1_trim_pct:.0%}_BE__TP2_{self.target_r:.1f}R",
        })
        confidence = min(0.95, max(0.70, (float(macro["score"]) + float(quality["score"])) / 200.0))
        reason = (
            f"{direction.upper()} {entry['trigger']} | 4H={macro['score']:.0f} "
            f"1H={quality['score']:.0f}/{quality['threshold']:.0f} "
            f"dist={entry['distance_atr']:.2f}ATR room={entry['room_r']:.2f}R"
        )
        return Signal(signal_type, self.symbol, self._entry_price, 0.0, reason, confidence, metadata)

    def cancel_pending_entry(self, reason: str = "") -> None:
        if self._pending_entry:
            self._reset_position()

    def attach_existing_position(
        self,
        direction: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        self._open_position = direction
        self._pending_entry = False
        self._entry_price = float(entry_price)
        self._entry_sl = float(stop_loss) if stop_loss is not None else None
        self._entry_tp = float(take_profit) if take_profit is not None else None
        self._initial_risk = abs(self._entry_price - self._entry_sl) if self._entry_sl is not None else None
        self._tp1_done = bool(
            self._entry_sl is not None
            and ((direction == "long" and self._entry_sl >= self._entry_price)
                 or (direction == "short" and self._entry_sl <= self._entry_price))
        )

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        if self._open_position is None:
            return None

        candles = self._latest_15m
        if len(candles) >= 22:
            bar_ts = self._bar_ts(candles[-1])
            if bar_ts != self._last_exit_check_ts:
                self._last_exit_check_ts = bar_ts
                closes = [float(c.close) for c in candles]
                ema8 = self.ema(closes, 8)
                ema13 = self.ema(closes, 13)
                if self._finite(ema8[-1], ema8[-2], ema13[-1], ema13[-2]):
                    long = self._open_position == "long"
                    reverse_cross = (
                        ema8[-2] >= ema13[-2] and ema8[-1] < ema13[-1]
                    ) if long else (
                        ema8[-2] <= ema13[-2] and ema8[-1] > ema13[-1]
                    )
                    close_confirm = closes[-1] < ema13[-1] if long else closes[-1] > ema13[-1]
                    two_close_fail = (
                        closes[-1] < ema13[-1] and closes[-2] < ema13[-2]
                    ) if long else (
                        closes[-1] > ema13[-1] and closes[-2] > ema13[-2]
                    )
                    if (reverse_cross and close_confirm) or two_close_fail:
                        side = self._open_position
                        self._last_exit_bar_ts = bar_ts
                        self._reset_position(keep_exit_ts=True)
                        return PositionUpdate(
                            action="close",
                            close_pct=1.0,
                            reason=f"15M trend failed ({'reverse cross' if reverse_cross else '2 closes beyond EMA13'}) — close {side.upper()}",
                        )

        if (
            not self._tp1_done
            and self._entry_price is not None
            and self._initial_risk is not None
            and self._initial_risk > 0
        ):
            profit = (
                float(current_price) - self._entry_price
                if self._open_position == "long"
                else self._entry_price - float(current_price)
            )
            current_r = profit / self._initial_risk
            if current_r >= self.tp1_r:
                self._tp1_done = True
                return PositionUpdate(
                    action="partial_tp",
                    close_pct=self.tp1_trim_pct,
                    new_sl=self._entry_price,
                    reason=f"T1 {current_r:.2f}R — trim {self.tp1_trim_pct:.0%}, move SL to breakeven",
                )

        return PositionUpdate(
            action="hold",
            reason=f"Holding {self._open_position.upper()} — hard SL, T1 {self.tp1_r:.1f}R, TP2 {self.target_r:.1f}R",
        )

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        if self._latest_15m:
            self._last_exit_bar_ts = self._bar_ts(self._latest_15m[-1])
        self._reset_position(keep_exit_ts=True)

    def _reset_position(self, keep_exit_ts: bool = False) -> None:
        exit_ts = self._last_exit_bar_ts if keep_exit_ts else None
        self._open_position = None
        self._pending_entry = False
        self._entry_price = None
        self._entry_sl = None
        self._entry_tp = None
        self._initial_risk = None
        self._tp1_done = False
        self._last_exit_check_ts = None
        self._last_exit_bar_ts = exit_ts
