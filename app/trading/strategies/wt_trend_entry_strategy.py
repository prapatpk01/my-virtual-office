"""WT Trend Entry Strategy.

Simple three-layer architecture:
  Layer 1 (4H): choose bull/bear direction with EMA20/50 + EMA20 slope.
  Layer 2 (1H): context must agree; ADX and Choppiness must pass.
  Layer 3 (15M): price must be on the correct side of EMA20 and entry fires on
                 either a fresh EMA8/13 cross OR a WaveTrend cross from an
                 extreme zone.

WaveTrend core and extreme levels are derived from the uploaded SJ WaveTrend
strategy: WT(10,21), signal SMA(4), long extreme < -45, short extreme > +53.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType
from ..engines.position_manager import PositionUpdate


class WTTrendEntryStrategy(BaseStrategy):
    def __init__(self, symbol: str, params: Optional[dict] = None):
        super().__init__(symbol, params)
        self.name = f"WTTrendEntry({symbol})"

        self.wt_channel_len = int(os.getenv("WT_CHANNEL_LEN", "10"))
        self.wt_avg_len = int(os.getenv("WT_AVG_LEN", "21"))
        self.wt_ob = float(os.getenv("WT_OB_LEVEL", "53"))
        self.wt_os = float(os.getenv("WT_OS_LEVEL", "-45"))
        self.adx_min = float(os.getenv("WT_CONTEXT_ADX_MIN", "18"))
        self.chop_max = float(os.getenv("WT_CONTEXT_CHOP_MAX", "61.8"))
        self.final_rr = float(os.getenv("WT_FINAL_RR", "1.8"))
        self.min_stop_pct = float(os.getenv("WT_MIN_STOP_PCT", "0.70"))
        self.min_stop_atr = float(os.getenv("WT_MIN_STOP_ATR", "1.0"))
        self.max_stop_pct = float(os.getenv("WT_MAX_STOP_PCT", "1.20"))
        self.t1_rr = float(os.getenv("WT_T1_RR", "0.60"))
        self.t1_lock_rr = float(os.getenv("WT_T1_LOCK_RR", "0.30"))

        self._last_entry_bar_ts: Optional[int] = None
        self._open_position: Optional[dict] = None
        self._latest_candles: list = []
        self._t1_done = False

    @staticmethod
    def _closes(candles: list) -> list[float]:
        return [float(c.close) for c in candles]

    @staticmethod
    def _cross(fast: np.ndarray, slow: np.ndarray, direction: str) -> bool:
        if len(fast) < 2 or any(np.isnan(v) for v in (fast[-2], fast[-1], slow[-2], slow[-1])):
            return False
        if direction == "long":
            return bool(fast[-2] <= slow[-2] and fast[-1] > slow[-1])
        return bool(fast[-2] >= slow[-2] and fast[-1] < slow[-1])

    def _wavetrend(self, candles: list) -> tuple[np.ndarray, np.ndarray]:
        ha_candles, _, ha_close = self._heikin_ashi(candles)
        highs = np.asarray([float(c.high) for c in ha_candles], dtype=float)
        lows = np.asarray([float(c.low) for c in ha_candles], dtype=float)
        ap = (highs + lows + ha_close) / 3.0
        esa = self.ema(ap.tolist(), self.wt_channel_len)
        deviation = np.abs(ap - esa)
        valid = np.where(~np.isnan(deviation))[0]
        if len(valid):
            deviation[:valid[0]] = deviation[valid[0]]
        d = self.ema(deviation.tolist(), self.wt_channel_len)
        ci = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
        wt1 = self.ema(ci.tolist(), self.wt_avg_len)
        wt2 = self.sma(wt1.tolist(), 4)
        return wt1, wt2

    @staticmethod
    def _choppiness(candles: list, period: int = 14) -> float:
        if len(candles) < period + 1:
            return float("nan")
        sample = candles[-period:]
        tr_sum = 0.0
        for i in range(len(candles) - period, len(candles)):
            if i == 0:
                continue
            c = candles[i]
            prev = candles[i - 1]
            tr_sum += max(
                float(c.high) - float(c.low),
                abs(float(c.high) - float(prev.close)),
                abs(float(c.low) - float(prev.close)),
            )
        highest = max(float(c.high) for c in sample)
        lowest = min(float(c.low) for c in sample)
        width = highest - lowest
        if width <= 0 or tr_sum <= 0:
            return 100.0
        return 100.0 * math.log10(tr_sum / width) / math.log10(period)

    def _four_hour_direction(self, candles: list) -> tuple[str, dict]:
        if len(candles) < 55:
            return "neutral", {"reason": "4H insufficient data"}
        closes = self._closes(candles)
        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)
        price = closes[-1]
        slope = ema20[-1] - ema20[-4]
        bull = price > ema20[-1] and ema20[-1] > ema50[-1] and slope > 0
        bear = price < ema20[-1] and ema20[-1] < ema50[-1] and slope < 0
        direction = "long" if bull else "short" if bear else "neutral"
        return direction, {
            "price": price, "ema20": float(ema20[-1]), "ema50": float(ema50[-1]),
            "ema20_slope": float(slope), "direction": direction,
        }

    def _one_hour_context(self, candles: list, direction: str) -> tuple[bool, dict]:
        if len(candles) < 55:
            return False, {"reason": "1H insufficient data"}
        closes = self._closes(candles)
        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)
        adx, plus_di, minus_di = self.adx(candles, 14)
        adx_value = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
        chop = self._choppiness(candles, 14)
        price = closes[-1]
        if direction == "long":
            aligned = price > ema20[-1] and ema20[-1] > ema50[-1] and plus_di[-1] > minus_di[-1]
        else:
            aligned = price < ema20[-1] and ema20[-1] < ema50[-1] and minus_di[-1] > plus_di[-1]
        passed = bool(aligned and adx_value >= self.adx_min and chop <= self.chop_max)
        return passed, {
            "aligned": bool(aligned), "adx": round(adx_value, 2),
            "adx_min": self.adx_min, "chop": round(chop, 2), "chop_max": self.chop_max,
        }

    def _build_levels(self, candles: list, price: float, direction: str, atr_value: float):
        lookback = candles[-10:]
        if direction == "long":
            structure_sl = min(float(c.low) for c in lookback) - 0.15 * atr_value
            structure_distance = price - structure_sl
        else:
            structure_sl = max(float(c.high) for c in lookback) + 0.15 * atr_value
            structure_distance = structure_sl - price

        minimum = max(price * self.min_stop_pct / 100.0, atr_value * self.min_stop_atr)
        distance = max(structure_distance, minimum)
        maximum = price * self.max_stop_pct / 100.0
        if distance <= 0 or distance > maximum:
            return None
        sl = price - distance if direction == "long" else price + distance
        tp = price + self.final_rr * distance if direction == "long" else price - self.final_rr * distance
        return round(sl, 8), round(tp, 8), distance

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles
        mtf = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])
        if len(candles) < 60:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0.0, "WT Trend: insufficient 15M data")

        direction, trend_detail = self._four_hour_direction(candles_4h)
        if direction == "neutral":
            return Signal(SignalType.HOLD, self.symbol, current_price, 0.0,
                          "WT Trend: 4H neutral", metadata={"trend_4h": trend_detail})

        context_ok, context_detail = self._one_hour_context(candles_1h, direction)
        if not context_ok:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0.0,
                          f"WT Trend: 1H context failed ADX/CHOP ({context_detail})",
                          metadata={"trend_4h": trend_detail, "context_1h": context_detail})

        closes = self._closes(candles)
        ema8 = self.ema(closes, 8)
        ema13 = self.ema(closes, 13)
        ema20 = self.ema(closes, 20)
        wt1, wt2 = self._wavetrend(candles)
        atr = self.atr(candles, 14)
        atr_value = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
        if atr_value <= 0 or any(np.isnan(v) for v in (ema20[-1], wt1[-1], wt2[-1])):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0.0, "WT Trend: indicators unavailable")

        price_side_ok = current_price > ema20[-1] if direction == "long" else current_price < ema20[-1]
        ema_trigger = self._cross(ema8, ema13, direction)
        wt_cross = self._cross(wt1, wt2, direction)
        wt_extreme = wt1[-1] < self.wt_os if direction == "long" else wt1[-1] > self.wt_ob
        wt_trigger = bool(wt_cross and wt_extreme)
        trigger = "EMA8/13" if ema_trigger else "WT extreme cross" if wt_trigger else ""

        metadata = {
            "trend_4h": trend_detail,
            "context_1h": context_detail,
            "entry_15m": {
                "direction": direction, "price_side_ema20": bool(price_side_ok),
                "ema_cross": bool(ema_trigger), "wt_cross": bool(wt_cross),
                "wt_extreme": bool(wt_extreme), "wt1": round(float(wt1[-1]), 2),
                "wt2": round(float(wt2[-1]), 2), "ema20": round(float(ema20[-1]), 8),
            },
        }

        if not price_side_ok or not (ema_trigger or wt_trigger):
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0.0,
                f"WT Trend: waiting 15M EMA8/13 OR WT extreme cross; price-side={price_side_ok}",
                metadata=metadata,
            )

        bar_ts = int(getattr(candles[-1], "timestamp", 0) or 0)
        if bar_ts and self._last_entry_bar_ts == bar_ts:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0.0,
                          "WT Trend: entry trigger already processed on this 15M bar", metadata=metadata)

        levels = self._build_levels(candles, float(current_price), direction, atr_value)
        if levels is None:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0.0,
                          f"WT Trend: structural stop exceeds {self.max_stop_pct:.2f}% cap", metadata=metadata)
        sl, tp, risk = levels

        self._last_entry_bar_ts = bar_ts or self._last_entry_bar_ts
        self._open_position = {
            "direction": direction, "entry": float(current_price), "sl": sl,
            "tp": tp, "risk": risk, "trigger": trigger,
        }
        self._t1_done = False
        metadata.update({
            "stop_loss": sl, "take_profit": tp, "rr_ratio": self.final_rr,
            "tp1_rr": self.t1_rr, "tp1_lock_rr": self.t1_lock_rr,
            "entry_trigger": trigger, "sizing_mode": "margin", "margin_pct": 0.05,
        })
        signal_type = SignalType.BUY if direction == "long" else SignalType.SELL
        return Signal(
            signal_type, self.symbol, current_price, 0.0,
            f"WT Trend {direction.upper()} via {trigger} | 4H+1H aligned, ADX/CHOP pass",
            confidence=0.78 if ema_trigger and wt_trigger else 0.72,
            metadata=metadata,
        )

    def attach_existing_position(self, direction: str, entry_price: float,
                                 stop_loss: Optional[float] = None,
                                 take_profit: Optional[float] = None) -> None:
        risk = abs(float(entry_price) - float(stop_loss or entry_price))
        self._open_position = {
            "direction": direction, "entry": float(entry_price),
            "sl": float(stop_loss or entry_price), "tp": float(take_profit or entry_price),
            "risk": risk, "trigger": "reconciled",
        }
        self._t1_done = False

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None) -> Optional[PositionUpdate]:
        pos = self._open_position
        if not pos or pos.get("risk", 0.0) <= 0:
            return None
        direction = pos["direction"]
        profit = current_price - pos["entry"] if direction == "long" else pos["entry"] - current_price
        current_r = profit / pos["risk"]
        if not self._t1_done and current_r >= self.t1_rr:
            self._t1_done = True
            new_sl = pos["entry"] + self.t1_lock_rr * pos["risk"] if direction == "long" else pos["entry"] - self.t1_lock_rr * pos["risk"]
            pos["sl"] = round(new_sl, 8)
            return PositionUpdate(
                action="modify_sl", new_sl=pos["sl"], close_pct=0.0,
                reason=f"WT T1 {current_r:.2f}R → lock +{self.t1_lock_rr:.2f}R, no partial close",
            )
        return PositionUpdate(action="hold", reason=f"WT holding {current_r:.2f}R")

    def record_closed_trade(self, *args, **kwargs) -> None:
        self._open_position = None
        self._t1_done = False
