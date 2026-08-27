"""Sentinel V5 — Simple RSI14/SMA14 Cross.

A deliberately small 15M strategy:
- Entry trigger: fresh RSI(14) cross of SMA(14) of RSI on a CLOSED 15M bar.
- LONG only when both RSI and RSI-SMA are below 55 at the bullish cross.
- SHORT only when both RSI and RSI-SMA are at/above 65 at the bearish cross.
- Market gate only: ADX >= 12, CHOP < 65, ATR activity >= 0.65.
- No EMA/HMA, structure, BOS, sweep, room, chase, 1H or 4H decision logic.
- Initial SL = 1.20 ATR; final TP = 1.80R.
- +0.80R -> move SL to +0.25R (no trim).
- +1.30R -> move SL to +0.70R (no trim).
- Opposite RSI/SMA cross on a CLOSED 15M bar exits early, regardless of zone.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Signal, SignalType
from .simple_precision_strategy import SimplePrecisionStrategy
from ..engines.position_manager import PositionUpdate


class SentinelV5Strategy(SimplePrecisionStrategy):
    VERSION = "5.0"
    entry_tf = "15m"

    MIN_BARS = 45

    RSI_PERIOD = 14
    RSI_SMA_PERIOD = 14
    LONG_ZONE_MAX = 55.0
    SHORT_ZONE_MIN = 65.0

    ADX_FLOOR = 12.0
    CHOP_CEILING = 65.0
    ATR_ACTIVITY_FLOOR = 0.65

    STOP_ATR = 1.20
    FINAL_TP_R = 1.80

    T1_R = 0.80
    T1_LOCK_R = 0.25
    T2_R = 1.30
    T2_LOCK_R = 0.70

    def __init__(self, symbol: str, *, exit_cooldown_bars: int = 2, **kwargs):
        # Keep only the shared lifecycle/state helpers. None of the inherited
        # 4H/1H/EMA entry logic is called by this class.
        super().__init__(
            symbol,
            adx_min=self.ADX_FLOOR,
            chop_max=self.CHOP_CEILING,
            stop_atr_min=self.STOP_ATR,
            stop_atr_max=self.STOP_ATR,
            target_r=self.FINAL_TP_R,
            tp1_r=self.T1_R,
            tp1_trim_pct=0.0,
            exit_cooldown_bars=exit_cooldown_bars,
        )
        self.name = f"SentinelV5({symbol})"
        self.target_r = self.FINAL_TP_R
        self.tp1_r = self.T1_R
        self.tp1_trim_pct = 0.0
        self.use_be_trail = False
        self._t2_done = False

    def _indicator_snapshot(self, candles: list) -> dict:
        closes = [float(c.close) for c in candles]
        atr = self.atr(candles, 14)
        adx, _, _ = self.adx(candles, 14)
        chop = self._choppiness(candles, 14)
        rsi = self.rsi(closes, self.RSI_PERIOD)
        rsi_sma = self.sma(list(rsi), self.RSI_SMA_PERIOD)

        if (
            chop is None
            or len(candles) < self.MIN_BARS
            or not self._finite(
                atr[-1], adx[-1],
                rsi[-1], rsi[-2],
                rsi_sma[-1], rsi_sma[-2],
            )
        ):
            return {
                "ready": False,
                "market_ready": False,
                "blocks": ["INDICATORS"],
                "reason": "15M indicators unavailable",
            }

        atr_values = [float(v) for v in atr[-21:-1] if np.isfinite(v)]
        atr_median = float(np.median(atr_values)) if atr_values else float(atr[-1])
        atr_now = max(float(atr[-1]), 1e-12)
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

        market_ready = not blocks
        return {
            "ready": True,
            "market_ready": market_ready,
            "blocks": blocks,
            "reason": "market gate pass" if market_ready else "blocked: " + ",".join(blocks),
            "atr": atr_now,
            "atr_ratio": round(float(atr_ratio), 2),
            "adx": round(adx_now, 1),
            "adx_floor": self.ADX_FLOOR,
            "chop": round(chop_now, 1),
            "chop_ceiling": self.CHOP_CEILING,
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

    def _entry_from_snapshot(self, current_price: float, snap: dict) -> dict:
        if not snap.get("ready"):
            return {"trigger": None, "reason": snap.get("reason", "indicators unavailable")}

        if not snap.get("market_ready"):
            return {
                "trigger": None,
                "reason": snap["reason"],
                "blocks": list(snap.get("blocks", [])),
            }

        direction = None
        trigger = None
        zone = None

        if snap.get("cross_up") and snap.get("long_zone"):
            direction = "long"
            trigger = "RSI14_CROSS_UP_SMA14"
            zone = "BELOW_55"
        elif snap.get("cross_down") and snap.get("short_zone"):
            direction = "short"
            trigger = "RSI14_CROSS_DOWN_SMA14"
            zone = "AT_OR_ABOVE_65"

        if trigger is None:
            if snap.get("cross_up"):
                reason = "RSI cross up outside LONG zone (<55 required)"
            elif snap.get("cross_down"):
                reason = "RSI cross down outside SHORT zone (>=65 required)"
            else:
                reason = "waiting for fresh RSI14/SMA14 cross"
            return {
                "trigger": None,
                "direction": None,
                "zone": None,
                "reason": reason,
                "blocks": [],
            }

        entry = float(current_price)
        atr_now = float(snap["atr"])
        risk = max(self.STOP_ATR * atr_now, entry * 0.001)
        stop = entry - risk if direction == "long" else entry + risk
        target = entry + self.FINAL_TP_R * risk if direction == "long" else entry - self.FINAL_TP_R * risk

        return {
            "trigger": trigger,
            "direction": direction,
            "zone": zone,
            "reason": "fresh RSI14/SMA14 cross confirmed",
            "entry": entry,
            "risk": float(risk),
            "stop_loss": float(stop),
            "take_profit": float(target),
            "sl_atr": self.STOP_ATR,
            "target_rr": self.FINAL_TP_R,
            "blocks": [],
        }

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        c15 = self._closed_candle_series(candles, 15 * 60_000)
        self._latest_15m = c15

        meta = {
            "strategy": "SENTINEL_V5",
            "version": self.VERSION,
            "architecture": "15M_MARKET_GATE__RSI14_SMA14_CROSS",
            "entry_tf": "15m_closed",
            "mtf_used": False,
        }

        if len(c15) < self.MIN_BARS:
            return self._hold(current_price, "waiting for 15M warmup", meta)

        snap = self._indicator_snapshot(c15)
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
                meta["cooldown"] = {
                    "active": True,
                    "bars": self.exit_cooldown_bars,
                    "elapsed_bars": max(0, int(elapsed // (15 * 60_000))),
                }
                return self._hold(current_price, "post-exit cooldown", meta)

        entry = self._entry_from_snapshot(float(current_price), snap)
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
        self._t2_done = False

        meta.update({
            "direction": direction,
            "entry_trigger": entry["trigger"],
            "stop_loss": round(self._entry_sl, 8),
            "take_profit": round(self._entry_tp, 8),
            "rr_ratio": self.FINAL_TP_R,
            "sl_atr": self.STOP_ATR,
            "t1_r": self.T1_R,
            "t1_lock_r": self.T1_LOCK_R,
            "t2_r": self.T2_R,
            "t2_lock_r": self.T2_LOCK_R,
            "final_tp_r": self.FINAL_TP_R,
            "risk_plan": (
                f"SL_{self.STOP_ATR:.2f}ATR__"
                f"T1_{self.T1_R:.2f}R_LOCK_{self.T1_LOCK_R:.2f}R__"
                f"T2_{self.T2_R:.2f}R_LOCK_{self.T2_LOCK_R:.2f}R__"
                f"TP_{self.FINAL_TP_R:.2f}R__"
                "EARLY_EXIT_OPPOSITE_RSI_SMA_CROSS"
            ),
        })

        confidence = 0.80
        if float(snap.get("adx", 0.0)) >= 20.0:
            confidence += 0.05
        if float(snap.get("atr_ratio", 0.0)) >= 1.0:
            confidence += 0.03
        confidence = min(confidence, 0.90)

        reason = (
            f"{direction.upper()} {entry['trigger']} | "
            f"RSI={snap['rsi']:.1f}/{snap['rsi_sma']:.1f} "
            f"ADX={snap['adx']:.1f} CHOP={snap['chop']:.1f} "
            f"SL={self.STOP_ATR:.1f}ATR TP={self.FINAL_TP_R:.1f}R"
        )
        return Signal(signal_type, self.symbol, self._entry_price, 0.0, reason, confidence, meta)

    def _closed_rsi_cross(self) -> tuple[bool, bool, dict]:
        candles = self._latest_15m
        if len(candles) < self.MIN_BARS:
            return False, False, {}

        closes = [float(c.close) for c in candles]
        rsi = self.rsi(closes, self.RSI_PERIOD)
        rsi_sma = self.sma(list(rsi), self.RSI_SMA_PERIOD)
        if not self._finite(rsi[-1], rsi[-2], rsi_sma[-1], rsi_sma[-2]):
            return False, False, {}

        prev_rsi = float(rsi[-2])
        prev_sma = float(rsi_sma[-2])
        curr_rsi = float(rsi[-1])
        curr_sma = float(rsi_sma[-1])
        cross_up = prev_rsi <= prev_sma and curr_rsi > curr_sma
        cross_down = prev_rsi >= prev_sma and curr_rsi < curr_sma
        return cross_up, cross_down, {
            "rsi": curr_rsi,
            "sma": curr_sma,
            "prev_rsi": prev_rsi,
            "prev_sma": prev_sma,
        }

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        if self._open_position is None:
            return None

        # Closed-bar opposite cross exits first. Hard SL/TP is checked outside
        # the strategy before this method in PAPER, and exchange-side in LIVE.
        if self._latest_15m:
            bar_ts = self._bar_ts(self._latest_15m[-1])
            if bar_ts != self._last_exit_check_ts:
                self._last_exit_check_ts = bar_ts
                cross_up, cross_down, info = self._closed_rsi_cross()
                opposite = (
                    self._open_position == "long" and cross_down
                ) or (
                    self._open_position == "short" and cross_up
                )
                if opposite:
                    side = str(self._open_position)
                    self._last_exit_bar_ts = bar_ts
                    reason = (
                        f"15M RSI14/SMA14 opposite cross — close {side.upper()} "
                        f"(RSI={info.get('rsi', 0.0):.1f} SMA={info.get('sma', 0.0):.1f})"
                    )
                    self._reset_position(keep_exit_ts=True)
                    return PositionUpdate(action="close", close_pct=1.0, reason=reason)

        if self._entry_price is None or self._initial_risk is None or self._initial_risk <= 0:
            return PositionUpdate(action="hold", reason="Holding — waiting for valid risk state")

        profit = (
            float(current_price) - self._entry_price
            if self._open_position == "long"
            else self._entry_price - float(current_price)
        )
        current_r = profit / self._initial_risk

        # If price jumps across both milestones between ticks, apply the stronger
        # T2 lock directly rather than briefly installing the weaker T1 lock.
        if not self._t2_done and current_r >= self.T2_R:
            self._tp1_done = True
            self._t2_done = True
            new_sl = (
                self._entry_price + self.T2_LOCK_R * self._initial_risk
                if self._open_position == "long"
                else self._entry_price - self.T2_LOCK_R * self._initial_risk
            )
            return PositionUpdate(
                action="move_sl",
                close_pct=0.0,
                new_sl=round(float(new_sl), 8),
                reason=f"T2 {current_r:.2f}R — hold 100%; lock SL at +{self.T2_LOCK_R:.2f}R",
            )

        if not self._tp1_done and current_r >= self.T1_R:
            self._tp1_done = True
            new_sl = (
                self._entry_price + self.T1_LOCK_R * self._initial_risk
                if self._open_position == "long"
                else self._entry_price - self.T1_LOCK_R * self._initial_risk
            )
            return PositionUpdate(
                action="move_sl",
                close_pct=0.0,
                new_sl=round(float(new_sl), 8),
                reason=f"T1 {current_r:.2f}R — hold 100%; lock SL at +{self.T1_LOCK_R:.2f}R",
            )

        return PositionUpdate(
            action="hold",
            reason=(
                f"Holding {self._open_position.upper()} — RSI/SMA opposite-cross exit; "
                f"T1 {self.T1_R:.1f}R→+{self.T1_LOCK_R:.2f}R, "
                f"T2 {self.T2_R:.1f}R→+{self.T2_LOCK_R:.2f}R, "
                f"TP {self.FINAL_TP_R:.1f}R"
            ),
        )

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

        # TP is the best way to reconstruct ORIGINAL R after a restart if the
        # stop has already been ratcheted into profit.
        if self._entry_tp is not None:
            self._initial_risk = abs(self._entry_tp - self._entry_price) / self.FINAL_TP_R
        elif self._entry_sl is not None:
            self._initial_risk = abs(self._entry_price - self._entry_sl)
        else:
            self._initial_risk = None

        self._tp1_done = False
        self._t2_done = False
        if self._entry_sl is not None and self._initial_risk and self._initial_risk > 0:
            lock_r = (
                (self._entry_sl - self._entry_price) / self._initial_risk
                if direction == "long"
                else (self._entry_price - self._entry_sl) / self._initial_risk
            )
            self._tp1_done = lock_r >= self.T1_LOCK_R - 1e-9
            self._t2_done = lock_r >= self.T2_LOCK_R - 1e-9

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        if self._latest_15m:
            self._last_exit_bar_ts = self._bar_ts(self._latest_15m[-1])
        self._reset_position(keep_exit_ts=True)

    def _reset_position(self, keep_exit_ts: bool = False) -> None:
        super()._reset_position(keep_exit_ts=keep_exit_ts)
        self._t2_done = False
