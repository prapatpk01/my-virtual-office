"""Sentinel V6 — Trend-Filtered RSI Pullback.

A full rewrite after V5's raw RSI-cross churn:
- 1H chooses direction only: EMA20/EMA50 alignment + EMA20 slope.
- 15M market gate only: ADX >= 12, CHOP < 65, ATR activity >= 0.65.
- 15M entry trigger: fresh RSI(14) cross of SMA(14) of RSI on a CLOSED bar.
- LONG: 1H bull + RSI cross up with RSI/SMA both < 55.
- SHORT: 1H bear + RSI cross down with RSI/SMA both >= 65.
- No BOS, sweep, S/R room, chase, HMA, MACD or 4H decision logic.
- Initial SL = 1.25 ATR.
- TP1 = +1.00R, close 50%, move remaining SL to +0.10R.
- TP2 = +2.00R, close the remaining 50% via the hard TP.
- Before TP1, opposite RSI/SMA cross exits only after neutral confirmation
  (LONG: RSI < 50, SHORT: RSI > 50) to avoid V5's cross-noise churn.
- After TP1, any opposite closed-bar RSI/SMA cross closes the runner.
"""
from __future__ import annotations

import numpy as np

from .base import Signal, SignalType
from .simple_precision_strategy import SimplePrecisionStrategy
from ..engines.position_manager import PositionUpdate


class SentinelV6Strategy(SimplePrecisionStrategy):
    VERSION = "6.0"
    entry_tf = "15m"

    RSI_PERIOD = 14
    RSI_SMA_PERIOD = 14
    LONG_ZONE_MAX = 55.0
    SHORT_ZONE_MIN = 65.0
    NEUTRAL = 50.0

    ADX_FLOOR = 12.0
    CHOP_CEILING = 65.0
    ATR_ACTIVITY_FLOOR = 0.65

    STOP_ATR = 1.25
    TP1_R = 1.00
    TP1_CLOSE_PCT = 0.50
    TP1_LOCK_R = 0.10
    TP2_R = 2.00

    MIN_15M_BARS = 45
    MIN_1H_BARS = 55

    def __init__(self, symbol: str, *, exit_cooldown_bars: int = 2, **kwargs):
        # Inherit only shared closed-candle/state utilities; none of the old
        # SimplePrecision 4H/1H/EMA entry stack is called by this class.
        super().__init__(
            symbol,
            adx_min=self.ADX_FLOOR,
            chop_max=self.CHOP_CEILING,
            stop_atr_min=self.STOP_ATR,
            stop_atr_max=self.STOP_ATR,
            target_r=self.TP2_R,
            tp1_r=self.TP1_R,
            tp1_trim_pct=self.TP1_CLOSE_PCT,
            exit_cooldown_bars=exit_cooldown_bars,
        )
        self.name = f"SentinelV6({symbol})"
        self.target_r = self.TP2_R
        self.tp1_r = self.TP1_R
        self.tp1_trim_pct = self.TP1_CLOSE_PCT
        self.use_be_trail = False

    # ------------------------------------------------------------------
    # 1H direction — intentionally tiny
    # ------------------------------------------------------------------
    def _trend_1h(self, candles: list) -> dict:
        if len(candles) < self.MIN_1H_BARS:
            return {"ready": False, "direction": None, "reason": "1H warmup"}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)
        if not self._finite(ema20[-1], ema20[-4], ema50[-1]):
            return {"ready": False, "direction": None, "reason": "1H indicators unavailable"}

        e20 = float(ema20[-1])
        e50 = float(ema50[-1])
        e20_prev = float(ema20[-4])
        slope_up = e20 > e20_prev
        slope_down = e20 < e20_prev

        if e20 > e50 and slope_up:
            direction = "long"
        elif e20 < e50 and slope_down:
            direction = "short"
        else:
            direction = None

        return {
            "ready": direction is not None,
            "direction": direction,
            "ema20": round(e20, 8),
            "ema50": round(e50, 8),
            "slope": "UP" if slope_up else "DOWN" if slope_down else "FLAT",
            "reason": f"1H {direction} bias" if direction else "1H EMA trend mixed",
        }

    # ------------------------------------------------------------------
    # 15M gate + RSI cross snapshot
    # ------------------------------------------------------------------
    def _snapshot_15m(self, candles: list) -> dict:
        if len(candles) < self.MIN_15M_BARS:
            return {"ready": False, "market_ready": False, "blocks": ["WARMUP"], "reason": "15M warmup"}

        closes = [float(c.close) for c in candles]
        atr = self.atr(candles, 14)
        adx, _, _ = self.adx(candles, 14)
        chop = self._choppiness(candles, 14)
        rsi = self.rsi(closes, self.RSI_PERIOD)
        rsi_sma = self.sma(list(rsi), self.RSI_SMA_PERIOD)

        if chop is None or not self._finite(
            atr[-1], adx[-1], rsi[-1], rsi[-2], rsi_sma[-1], rsi_sma[-2]
        ):
            return {"ready": False, "market_ready": False, "blocks": ["INDICATORS"], "reason": "15M indicators unavailable"}

        atr_now = max(float(atr[-1]), 1e-12)
        atr_values = [float(v) for v in atr[-21:-1] if np.isfinite(v)]
        atr_median = float(np.median(atr_values)) if atr_values else atr_now
        atr_ratio = atr_now / max(atr_median, 1e-12)

        adx_now = float(adx[-1])
        chop_now = float(chop)
        prev_rsi = float(rsi[-2])
        prev_sma = float(rsi_sma[-2])
        curr_rsi = float(rsi[-1])
        curr_sma = float(rsi_sma[-1])

        cross_up = prev_rsi <= prev_sma and curr_rsi > curr_sma
        cross_down = prev_rsi >= prev_sma and curr_rsi < curr_sma
        long_zone = curr_rsi < self.LONG_ZONE_MAX and curr_sma < self.LONG_ZONE_MAX
        short_zone = curr_rsi >= self.SHORT_ZONE_MIN and curr_sma >= self.SHORT_ZONE_MIN

        blocks = []
        if adx_now < self.ADX_FLOOR:
            blocks.append("ADX")
        if chop_now >= self.CHOP_CEILING:
            blocks.append("CHOP")
        if atr_ratio < self.ATR_ACTIVITY_FLOOR:
            blocks.append("DEAD_VOL")

        return {
            "ready": True,
            "market_ready": not blocks,
            "blocks": blocks,
            "reason": "market gate pass" if not blocks else "blocked: " + ",".join(blocks),
            "atr": atr_now,
            "atr_ratio": round(float(atr_ratio), 2),
            "adx": round(adx_now, 1),
            "chop": round(chop_now, 1),
            "rsi": round(curr_rsi, 2),
            "rsi_sma": round(curr_sma, 2),
            "prev_rsi": round(prev_rsi, 2),
            "prev_rsi_sma": round(prev_sma, 2),
            "cross_up": bool(cross_up),
            "cross_down": bool(cross_down),
            "cross": "UP" if cross_up else "DOWN" if cross_down else "NONE",
            "long_zone": bool(long_zone),
            "short_zone": bool(short_zone),
        }

    def _build_entry(self, current_price: float, trend: dict, snap: dict) -> dict:
        direction = trend.get("direction")
        if not snap.get("ready"):
            return {"trigger": None, "direction": direction, "reason": snap.get("reason", "15M unavailable"), "blocks": list(snap.get("blocks", []))}
        if not snap.get("market_ready"):
            return {"trigger": None, "direction": direction, "reason": snap["reason"], "blocks": list(snap.get("blocks", []))}
        if direction not in {"long", "short"}:
            return {"trigger": None, "direction": None, "reason": "waiting for clear 1H trend", "blocks": ["TREND_1H"]}

        trigger = None
        if direction == "long" and snap.get("cross_up") and snap.get("long_zone"):
            trigger = "LONG_RSI14_CROSS_UP_SMA14"
        elif direction == "short" and snap.get("cross_down") and snap.get("short_zone"):
            trigger = "SHORT_RSI14_CROSS_DOWN_SMA14"

        if trigger is None:
            if direction == "long" and snap.get("cross_up") and not snap.get("long_zone"):
                reason = "bullish RSI cross outside LONG zone (<55 required)"
            elif direction == "short" and snap.get("cross_down") and not snap.get("short_zone"):
                reason = "bearish RSI cross outside SHORT zone (>=65 required)"
            elif (direction == "long" and snap.get("cross_down")) or (direction == "short" and snap.get("cross_up")):
                reason = "fresh RSI cross opposes 1H trend"
            else:
                reason = "waiting for RSI14/SMA14 cross with 1H trend"
            return {"trigger": None, "direction": direction, "reason": reason, "blocks": []}

        entry = float(current_price)
        risk = max(self.STOP_ATR * float(snap["atr"]), entry * 0.001)
        stop = entry - risk if direction == "long" else entry + risk
        target = entry + self.TP2_R * risk if direction == "long" else entry - self.TP2_R * risk
        tp1 = entry + self.TP1_R * risk if direction == "long" else entry - self.TP1_R * risk

        return {
            "trigger": trigger,
            "direction": direction,
            "reason": "1H trend + 15M RSI cross confirmed",
            "entry": entry,
            "risk": float(risk),
            "stop_loss": float(stop),
            "tp1_price": float(tp1),
            "take_profit": float(target),
            "sl_atr": self.STOP_ATR,
            "tp1_r": self.TP1_R,
            "tp1_close_pct": self.TP1_CLOSE_PCT,
            "tp1_lock_r": self.TP1_LOCK_R,
            "tp2_r": self.TP2_R,
            "blocks": [],
        }

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c15 = self._closed_candle_series(candles, 15 * 60_000)
        c1h = self._closed_candle_series(mtf.get("1h", []), 60 * 60_000)
        self._latest_15m = c15

        meta = {
            "strategy": "SENTINEL_V6",
            "version": self.VERSION,
            "architecture": "1H_EMA_DIRECTION__15M_GATE__RSI_CROSS",
            "entry_tf": "15m_closed",
            "mtf_used": "1h_direction_only",
        }
        if len(c15) < self.MIN_15M_BARS or len(c1h) < self.MIN_1H_BARS:
            return self._hold(current_price, "waiting for closed 15M/1H warmup", meta)

        trend = self._trend_1h(c1h)
        snap = self._snapshot_15m(c15)
        meta["trend_1h"] = trend
        meta["market_15m"] = snap

        if self._open_position is not None:
            return self._hold(current_price, f"managing open {self._open_position} position", meta)

        bar_ts = self._bar_ts(c15[-1])
        if self._last_evaluated_bar_ts == bar_ts:
            return self._hold(current_price, "15M bar already evaluated", meta)
        self._last_evaluated_bar_ts = bar_ts

        if self._last_exit_bar_ts is not None:
            elapsed = bar_ts - self._last_exit_bar_ts
            if elapsed < self.exit_cooldown_bars * 15 * 60_000:
                return self._hold(current_price, "post-exit cooldown", meta)

        entry = self._build_entry(float(current_price), trend, snap)
        meta["entry_15m"] = entry
        if not entry.get("trigger"):
            return self._hold(current_price, entry["reason"], meta)

        direction = str(entry["direction"])
        signal_type = SignalType.BUY if direction == "long" else SignalType.SELL
        self._open_position = direction
        self._pending_entry = True
        self._entry_price = float(entry["entry"])
        self._entry_sl = float(entry["stop_loss"])
        self._entry_tp = float(entry["take_profit"])
        self._initial_risk = abs(self._entry_price - self._entry_sl)
        self._tp1_done = False

        meta.update({
            "direction": direction,
            "entry_trigger": entry["trigger"],
            "stop_loss": round(self._entry_sl, 8),
            "take_profit": round(self._entry_tp, 8),
            "tp1_price": round(float(entry["tp1_price"]), 8),
            "rr_ratio": self.TP2_R,
            "tp1_r": self.TP1_R,
            "tp1_close_pct": self.TP1_CLOSE_PCT,
            "tp1_lock_r": self.TP1_LOCK_R,
            "risk_plan": "SL_1.25ATR__TP1_1.0R_CLOSE50_LOCK+0.10R__TP2_2.0R",
        })
        reason = (
            f"{direction.upper()} {entry['trigger']} | 1H={trend['direction']} "
            f"RSI={snap['rsi']:.2f}/{snap['rsi_sma']:.2f} "
            f"ADX={snap['adx']:.1f} CHOP={snap['chop']:.1f} ATRx={snap['atr_ratio']:.2f}"
        )
        return Signal(signal_type, self.symbol, self._entry_price, 0.0, reason, 0.82, meta)

    def tick_open_position(self, current_price: float, position_key: str | None = None):
        if self._open_position is None:
            return None

        # Closed-15M RSI reversal management. Before TP1, demand neutral-line
        # confirmation so a tiny cross does not churn the trade like V5 did.
        candles = self._latest_15m
        if len(candles) >= self.MIN_15M_BARS:
            bar_ts = self._bar_ts(candles[-1])
            if bar_ts != self._last_exit_check_ts:
                self._last_exit_check_ts = bar_ts
                snap = self._snapshot_15m(candles)
                if snap.get("ready"):
                    if self._open_position == "long":
                        opposite = bool(snap.get("cross_down"))
                        confirmed = float(snap.get("rsi", 50.0)) < self.NEUTRAL
                    else:
                        opposite = bool(snap.get("cross_up"))
                        confirmed = float(snap.get("rsi", 50.0)) > self.NEUTRAL

                    if opposite and (self._tp1_done or confirmed):
                        side = self._open_position
                        reason = (
                            "RSI_RUNNER_EXIT: opposite RSI14/SMA14 cross after TP1"
                            if self._tp1_done
                            else "RSI_REVERSAL_EXIT: opposite RSI14/SMA14 cross + neutral confirmation"
                        )
                        self._last_exit_bar_ts = bar_ts
                        self._reset_position(keep_exit_ts=True)
                        return PositionUpdate(action="close", close_pct=1.0, reason=f"{reason} — close {side.upper()}")

        # TP1 is handled here because the bot's partial_tp path can actually
        # close 50% and re-place the exchange SL on the remaining 50%.
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
            if current_r >= self.TP1_R:
                self._tp1_done = True
                new_sl = (
                    self._entry_price + self.TP1_LOCK_R * self._initial_risk
                    if self._open_position == "long"
                    else self._entry_price - self.TP1_LOCK_R * self._initial_risk
                )
                return PositionUpdate(
                    action="partial_tp",
                    close_pct=self.TP1_CLOSE_PCT,
                    new_sl=round(float(new_sl), 8),
                    reason=(
                        f"TP1 {current_r:.2f}R — close 50%, "
                        f"move remaining SL to +{self.TP1_LOCK_R:.2f}R"
                    ),
                )

        return PositionUpdate(
            action="hold",
            reason=(
                f"Holding {self._open_position.upper()} — SL 1.25ATR | "
                f"TP1 1.0R/50% | TP2 2.0R"
            ),
        )

    def attach_existing_position(self, direction: str, entry_price: float,
                                 stop_loss: float | None = None,
                                 take_profit: float | None = None) -> None:
        super().attach_existing_position(direction, entry_price, stop_loss, take_profit)
        # If the recovered stop is already beyond entry, treat TP1 as complete.
        self._tp1_done = bool(
            stop_loss is not None
            and ((direction == "long" and float(stop_loss) > float(entry_price))
                 or (direction == "short" and float(stop_loss) < float(entry_price)))
        )
