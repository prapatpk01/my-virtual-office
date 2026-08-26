"""Sentinel V4 — 15M Price-Action Core.

A clean rewrite focused on fresh price-action setups instead of indicator-cross churn.

Architecture:
- 15M only.
- Small market gate: ADX / CHOP / active ATR.
- Three explicit setups: pullback continuation, breakout-retest,
  liquidity sweep -> micro structure reversal.
- EMA8/13 cross is neither an entry trigger nor an exit trigger.
- RSI/SMA, DMI and MACD are soft confidence context only.
- Oversized structure stops are rejected instead of clipped inside invalidation.
- Trades need >= 1.6R room to the first opposing S/R.
- T1 protects without trimming so winners can run.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Signal, SignalType
from .simple_precision_strategy import SimplePrecisionStrategy
from ..engines.position_manager import PositionUpdate


class SentinelV4Strategy(SimplePrecisionStrategy):
    VERSION = "4.0"
    entry_tf = "15m"

    MIN_BARS = 90
    MIN_TARGET_R = 1.60
    TARGET_R = 2.00
    ENTRY_GRACE_BARS = 2
    T1_R = 1.00
    T1_LOCK_R = 0.20
    TRAIL_R = 1.50
    TRAIL_LOCK_R = 0.80

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV4({symbol})"
        self.tp1_r = self.T1_R
        self.tp1_trim_pct = 0.0
        self.target_r = self.TARGET_R

        self._entry_bar_ts: Optional[int] = None
        self._trail_lock_done = False
        self._reentry_side: Optional[str] = None
        self._reentry_exit_ts: Optional[int] = None
        self._last_setup: str = ""

    @staticmethod
    def _merge_levels(levels: list[float], tolerance: float) -> list[float]:
        clean = sorted(float(v) for v in levels if v is not None and np.isfinite(float(v)))
        if not clean:
            return []
        merged = [clean[0]]
        for value in clean[1:]:
            if abs(value - merged[-1]) <= max(tolerance, 1e-12):
                merged[-1] = (merged[-1] + value) / 2.0
            else:
                merged.append(value)
        return merged

    @staticmethod
    def _nearest_above(price: float, levels: list[float]) -> Optional[float]:
        values = [float(v) for v in levels if float(v) > price]
        return min(values) if values else None

    @staticmethod
    def _nearest_below(price: float, levels: list[float]) -> Optional[float]:
        values = [float(v) for v in levels if float(v) < price]
        return max(values) if values else None

    @classmethod
    def _pivot_points(cls, candles: list, span: int = 2):
        highs = []
        lows = []
        for i in range(span, len(candles) - span):
            before = candles[i - span:i]
            after = candles[i + 1:i + span + 1]
            high = float(candles[i].high)
            low = float(candles[i].low)
            if high >= max(float(c.high) for c in before + after):
                highs.append((cls._bar_ts(candles[i]), high))
            if low <= min(float(c.low) for c in before + after):
                lows.append((cls._bar_ts(candles[i]), low))
        return highs, lows

    def _structure(self, candles: list, atr_now: float) -> dict:
        recent = candles[-120:]
        highs2, lows2 = self._pivot_points(recent, span=2)
        highs4, lows4 = self._pivot_points(recent, span=4)

        last_ph = highs2[-1][1] if highs2 else None
        prev_ph = highs2[-2][1] if len(highs2) >= 2 else None
        last_pl = lows2[-1][1] if lows2 else None
        prev_pl = lows2[-2][1] if len(lows2) >= 2 else None

        hh = last_ph is not None and prev_ph is not None and last_ph > prev_ph
        lh = last_ph is not None and prev_ph is not None and last_ph < prev_ph
        hl = last_pl is not None and prev_pl is not None and last_pl > prev_pl
        ll = last_pl is not None and prev_pl is not None and last_pl < prev_pl

        if hh and hl:
            state, label = 2, "HH/HL"
        elif lh and ll:
            state, label = -2, "LH/LL"
        elif hh or hl:
            state, label = 1, "BULLISH"
        elif lh or ll:
            state, label = -1, "BEARISH"
        else:
            state, label = 0, "MIXED"

        tolerance = atr_now * 0.30
        resistances = self._merge_levels(
            [p for _, p in highs2[-6:]] + [p for _, p in highs4[-4:]], tolerance
        )
        supports = self._merge_levels(
            [p for _, p in lows2[-6:]] + [p for _, p in lows4[-4:]], tolerance
        )

        return {
            "state": state,
            "label": label,
            "last_ph": last_ph,
            "last_pl": last_pl,
            "resistances": resistances,
            "supports": supports,
        }

    def _market_gate(self, candles: list) -> dict:
        closes = [float(c.close) for c in candles]
        atr = self.atr(candles, 14)
        adx, plus_di, minus_di = self.adx(candles, 14)
        chop = self._choppiness(candles, 14)
        rsi = self.rsi(closes, 14)
        rsi_sma = self.sma(list(rsi), 9)
        macd, macd_signal, macd_hist = self.macd(closes)

        if chop is None or not self._finite(
            atr[-1], adx[-1], plus_di[-1], minus_di[-1], rsi[-1], rsi_sma[-1]
        ):
            return {"ready": False, "blocks": ["INDICATORS"], "reason": "15M indicators unavailable"}

        atr_values = [float(v) for v in atr[-21:-1] if np.isfinite(v)]
        atr_median = float(np.median(atr_values)) if atr_values else float(atr[-1])
        atr_ratio = float(atr[-1]) / max(atr_median, 1e-12)

        adx_now = float(adx[-1])
        chop_now = float(chop)
        adx_floor = max(13.0, min(float(self.adx_min), 16.0))
        chop_ceiling = min(max(float(self.chop_max), 58.0), 62.0)

        blocks = []
        if adx_now < adx_floor:
            blocks.append("ADX")
        if chop_now >= chop_ceiling:
            blocks.append("CHOP")
        if atr_ratio < 0.65:
            blocks.append("DEAD_VOL")

        hist = float(macd_hist[-1]) if np.isfinite(macd_hist[-1]) else float(macd[-1] - macd_signal[-1])
        return {
            "ready": not blocks,
            "blocks": blocks,
            "adx": round(adx_now, 1),
            "adx_floor": round(adx_floor, 1),
            "chop": round(chop_now, 1),
            "chop_ceiling": round(chop_ceiling, 1),
            "atr_ratio": round(atr_ratio, 2),
            "plus_di": round(float(plus_di[-1]), 1),
            "minus_di": round(float(minus_di[-1]), 1),
            "macd_hist": hist,
            "rsi": round(float(rsi[-1]), 1),
            "rsi_sma": round(float(rsi_sma[-1]), 1),
            "reason": "market gate pass" if not blocks else "blocked: " + ",".join(blocks),
        }

    @staticmethod
    def _bar_quality(candle):
        rng = max(float(candle.high) - float(candle.low), 1e-12)
        body_eff = abs(float(candle.close) - float(candle.open)) / rng
        clv = (float(candle.close) - float(candle.low)) / rng
        return body_eff, clv

    def _volume_ratio(self, candles: list) -> float:
        history = [float(c.volume or 0.0) for c in candles[-21:-1]]
        median = float(np.median(history)) if history else 0.0
        return 1.0 if median <= 0 else float(candles[-1].volume or 0.0) / max(median, 1e-12)

    def _pullback_setup(self, candles, structure, atr_now, ema20, ema50, hma16):
        close = float(candles[-1].close)
        candle = candles[-1]
        body_eff, clv = self._bar_quality(candle)
        candidates = []
        touch_long = None
        touch_short = None

        for i in range(len(candles) - 3, len(candles)):
            if not self._finite(ema20[i], hma16[i]):
                continue
            zone_low = min(float(ema20[i]), float(hma16[i]))
            zone_high = max(float(ema20[i]), float(hma16[i]))
            if float(candles[i].low) <= zone_high + 0.10 * atr_now:
                touch_long = i
            if float(candles[i].high) >= zone_low - 0.10 * atr_now:
                touch_short = i

        prev_high = max(float(c.high) for c in candles[-3:-1])
        prev_low = min(float(c.low) for c in candles[-3:-1])

        long_ok = (
            structure["state"] > 0
            and float(ema20[-1]) > float(ema50[-1])
            and touch_long is not None
            and close > float(ema20[-1])
            and close > float(hma16[-1])
            and close > float(candle.open)
            and clv >= 0.60
            and body_eff >= 0.30
            and close > prev_high
        )
        short_ok = (
            structure["state"] < 0
            and float(ema20[-1]) < float(ema50[-1])
            and touch_short is not None
            and close < float(ema20[-1])
            and close < float(hma16[-1])
            and close < float(candle.open)
            and clv <= 0.40
            and body_eff >= 0.30
            and close < prev_low
        )

        if long_ok:
            base = float(candles[touch_long].low)
            pivot = float(structure["last_pl"]) if structure["last_pl"] is not None else base
            candidates.append({
                "direction": "long", "setup": "PULLBACK_CONTINUATION", "priority": 20,
                "invalidation": min(base, pivot) - 0.08 * atr_now,
                "event_ts": self._bar_ts(candles[touch_long]),
            })
        if short_ok:
            base = float(candles[touch_short].high)
            pivot = float(structure["last_ph"]) if structure["last_ph"] is not None else base
            candidates.append({
                "direction": "short", "setup": "PULLBACK_CONTINUATION", "priority": 20,
                "invalidation": max(base, pivot) + 0.08 * atr_now,
                "event_ts": self._bar_ts(candles[touch_short]),
            })
        return candidates

    def _breakout_retest_setup(self, candles, structure, atr_now):
        candidates = []
        current = candles[-1]
        close = float(current.close)
        body_eff, clv = self._bar_quality(current)
        last_ph = structure["last_ph"]
        last_pl = structure["last_pl"]

        if last_ph is not None:
            breakout_idx = None
            for i in range(len(candles) - 4, len(candles) - 1):
                bar_body, _ = self._bar_quality(candles[i])
                if (
                    float(candles[i].close) > float(last_ph) + 0.05 * atr_now
                    and float(candles[i - 1].close) <= float(last_ph)
                    and bar_body >= 0.35
                ):
                    breakout_idx = i
            if (
                breakout_idx is not None
                and float(current.low) <= float(last_ph) + 0.18 * atr_now
                and close > float(last_ph)
                and close > float(current.open)
                and clv >= 0.58
                and body_eff >= 0.25
            ):
                candidates.append({
                    "direction": "long", "setup": "BREAKOUT_RETEST", "priority": 30,
                    "invalidation": float(last_ph) - 0.22 * atr_now,
                    "event_ts": self._bar_ts(current),
                })

        if last_pl is not None:
            breakout_idx = None
            for i in range(len(candles) - 4, len(candles) - 1):
                bar_body, _ = self._bar_quality(candles[i])
                if (
                    float(candles[i].close) < float(last_pl) - 0.05 * atr_now
                    and float(candles[i - 1].close) >= float(last_pl)
                    and bar_body >= 0.35
                ):
                    breakout_idx = i
            if (
                breakout_idx is not None
                and float(current.high) >= float(last_pl) - 0.18 * atr_now
                and close < float(last_pl)
                and close < float(current.open)
                and clv <= 0.42
                and body_eff >= 0.25
            ):
                candidates.append({
                    "direction": "short", "setup": "BREAKOUT_RETEST", "priority": 30,
                    "invalidation": float(last_pl) + 0.22 * atr_now,
                    "event_ts": self._bar_ts(current),
                })
        return candidates

    def _sweep_reversal_setup(self, candles, structure, atr_now, hma16):
        candidates = []
        current = candles[-1]
        close = float(current.close)
        body_eff, clv = self._bar_quality(current)
        last_ph = structure["last_ph"]
        last_pl = structure["last_pl"]
        low_sweep = None
        high_sweep = None

        for i in range(len(candles) - 4, len(candles) - 1):
            bar = candles[i]
            if last_pl is not None and float(bar.low) < float(last_pl) - 0.05 * atr_now and float(bar.close) > float(last_pl):
                low_sweep = i
            if last_ph is not None and float(bar.high) > float(last_ph) + 0.05 * atr_now and float(bar.close) < float(last_ph):
                high_sweep = i

        micro_high = max(float(c.high) for c in candles[-3:-1])
        micro_low = min(float(c.low) for c in candles[-3:-1])

        if (
            low_sweep is not None
            and close > micro_high
            and close > float(hma16[-1])
            and close > float(current.open)
            and clv >= 0.62
            and body_eff >= 0.32
        ):
            sweep = candles[low_sweep]
            candidates.append({
                "direction": "long", "setup": "SWEEP_STRUCTURE_REVERSAL", "priority": 40,
                "invalidation": float(sweep.low) - 0.08 * atr_now,
                "event_ts": self._bar_ts(sweep),
            })
        if (
            high_sweep is not None
            and close < micro_low
            and close < float(hma16[-1])
            and close < float(current.open)
            and clv <= 0.38
            and body_eff >= 0.32
        ):
            sweep = candles[high_sweep]
            candidates.append({
                "direction": "short", "setup": "SWEEP_STRUCTURE_REVERSAL", "priority": 40,
                "invalidation": float(sweep.high) + 0.08 * atr_now,
                "event_ts": self._bar_ts(sweep),
            })
        return candidates

    def _candidate_context(self, candles, structure, atr_now):
        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)
        hma16 = self.hma(closes, 16)
        if not self._finite(ema20[-1], ema50[-1], hma16[-1]):
            return []

        candidates = []
        candidates.extend(self._pullback_setup(candles, structure, atr_now, ema20, ema50, hma16))
        candidates.extend(self._breakout_retest_setup(candles, structure, atr_now))
        candidates.extend(self._sweep_reversal_setup(candles, structure, atr_now, hma16))
        volume_ratio = self._volume_ratio(candles)
        for item in candidates:
            item.update({
                "ema20": float(ema20[-1]), "ema50": float(ema50[-1]),
                "hma16": float(hma16[-1]), "volume_ratio": volume_ratio,
            })
        return candidates

    def _build_entry(self, candles, current_price, market, structure):
        close = float(candles[-1].close)
        atr_now = max(float(self.atr(candles, 14)[-1]), 1e-12)
        candidates = self._candidate_context(candles, structure, atr_now)
        if not candidates:
            return {"trigger": None, "reason": "waiting for pullback, breakout-retest, or sweep reversal"}
        if len({c["direction"] for c in candidates}) > 1:
            return {"trigger": None, "reason": "conflicting fresh price-action setups", "blocks": ["CONFLICT"]}

        candidate = max(candidates, key=lambda x: int(x["priority"]))
        direction = str(candidate["direction"])
        setup = str(candidate["setup"])
        entry = float(current_price or close)
        invalidation = float(candidate["invalidation"])
        raw_risk = entry - invalidation if direction == "long" else invalidation - entry

        blocks = []
        if raw_risk <= 0:
            blocks.append("INVALID_STOP")
        min_stop = max(0.80 * atr_now, entry * 0.001)
        max_stop = 1.60 * atr_now
        if raw_risk > max_stop:
            blocks.append("STOP_TOO_WIDE")
        risk = max(raw_risk, min_stop) if raw_risk > 0 else min_stop
        stop = entry - risk if direction == "long" else entry + risk

        resistance = self._nearest_above(entry, structure["resistances"])
        support = self._nearest_below(entry, structure["supports"])
        opposing = resistance if direction == "long" else support
        room_r = abs(float(opposing) - entry) / max(risk, 1e-12) if opposing is not None else 2.50
        required_room = max(self.MIN_TARGET_R, float(self.min_room_r))
        if room_r < required_room:
            blocks.append("ROOM")

        distance_atr = abs(close - float(candidate["ema20"])) / atr_now
        chase_limit = {
            "PULLBACK_CONTINUATION": 1.00,
            "BREAKOUT_RETEST": 1.30,
            "SWEEP_STRUCTURE_REVERSAL": 1.45,
        }.get(setup, 1.20)
        if distance_atr > chase_limit:
            blocks.append("CHASE")

        if (
            self._reentry_side == direction
            and self._reentry_exit_ts is not None
            and int(candidate.get("event_ts") or 0) <= int(self._reentry_exit_ts)
        ):
            blocks.append("STALE_REENTRY")

        fixed_target = entry + self.TARGET_R * risk if direction == "long" else entry - self.TARGET_R * risk
        target = fixed_target
        target_source = f"{self.TARGET_R:.1f}R"
        if opposing is not None and room_r >= required_room:
            sr_target = float(opposing) - 0.05 * atr_now if direction == "long" else float(opposing) + 0.05 * atr_now
            if direction == "long" and sr_target > entry:
                target = min(fixed_target, sr_target)
            elif direction == "short" and sr_target < entry:
                target = max(fixed_target, sr_target)
            actual_rr = abs(target - entry) / max(risk, 1e-12)
            if actual_rr < required_room:
                blocks.append("TARGET_TOO_CLOSE")
            elif abs(target - fixed_target) > 0.03 * risk:
                target_source = "NEAREST_S/R"

        rsi = float(market.get("rsi", 50.0))
        rsi_sma = float(market.get("rsi_sma", 50.0))
        dmi_ok = float(market.get("plus_di", 0.0)) > float(market.get("minus_di", 0.0)) if direction == "long" else float(market.get("minus_di", 0.0)) > float(market.get("plus_di", 0.0))
        macd_hist = float(market.get("macd_hist", 0.0))
        macd_ok = macd_hist >= 0 if direction == "long" else macd_hist <= 0
        rsi_ok = (rsi >= 50.0 and rsi > rsi_sma) if direction == "long" else (rsi <= 50.0 and rsi < rsi_sma)
        rsi_extreme = rsi >= 75.0 if direction == "long" else rsi <= 25.0

        confidence = 0.68
        confidence += 0.05 if dmi_ok else 0.0
        confidence += 0.05 if macd_ok else 0.0
        confidence += 0.05 if rsi_ok else 0.0
        confidence += 0.04 if float(candidate.get("volume_ratio", 1.0)) >= 1.0 else 0.0
        confidence += 0.03 if room_r >= 2.0 else 0.0
        confidence -= 0.04 if rsi_extreme else 0.0
        confidence = float(np.clip(confidence, 0.60, 0.92))

        trigger = setup if market.get("ready") and not blocks else None
        reason = "fresh price-action setup confirmed" if trigger else ("blocked: " + ",".join(blocks) if blocks else str(market.get("reason") or "market gate blocked"))
        return {
            "trigger": trigger, "candidate_trigger": setup, "direction": direction, "reason": reason,
            "entry": entry, "stop_loss": float(stop), "take_profit": float(target), "risk": float(risk),
            "atr": atr_now, "structure": structure, "room_r": round(float(room_r), 2),
            "distance_atr": round(float(distance_atr), 2), "chase_limit_atr": chase_limit,
            "target_source": target_source, "target_rr": round(abs(target - entry) / max(risk, 1e-12), 2),
            "confidence": round(confidence, 3),
            "boosters": {"dmi": dmi_ok, "macd": macd_ok, "rsi_sma": rsi_ok, "volume": float(candidate.get("volume_ratio", 1.0)) >= 1.0},
            "rsi": rsi, "rsi_sma": rsi_sma, "volume_ratio": round(float(candidate.get("volume_ratio", 1.0)), 2),
            "blocks": blocks, "event_ts": int(candidate.get("event_ts") or 0),
        }

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        c15 = self._closed_candle_series(candles, 15 * 60_000)
        self._latest_15m = c15
        meta = {"strategy": "SENTINEL_V4", "version": self.VERSION, "architecture": "15M_MARKET_GATE__PRICE_ACTION_SETUP__STRUCTURE_RISK"}
        if len(c15) < self.MIN_BARS:
            return self._hold(current_price, "waiting for 15M warmup", meta)

        bar_ts = self._bar_ts(c15[-1])
        if self._open_position is not None:
            return self._hold(current_price, f"managing open {self._open_position} position", meta)
        if self._last_evaluated_bar_ts == bar_ts:
            return self._hold(current_price, "15M bar already evaluated", meta)
        self._last_evaluated_bar_ts = bar_ts
        if self._last_exit_bar_ts is not None and bar_ts - self._last_exit_bar_ts < self.exit_cooldown_bars * 15 * 60_000:
            return self._hold(current_price, "post-exit cooldown", meta)

        atr_now = float(self.atr(c15, 14)[-1])
        if not self._finite(atr_now):
            return self._hold(current_price, "15M ATR unavailable", meta)

        market = self._market_gate(c15)
        structure = self._structure(c15, max(atr_now, 1e-12))
        meta["market_15m"] = market
        meta["structure_15m"] = {"state": structure["state"], "label": structure["label"], "last_ph": structure["last_ph"], "last_pl": structure["last_pl"]}
        if not market["ready"]:
            return self._hold(current_price, market["reason"], meta)

        entry = self._build_entry(c15, current_price, market, structure)
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
        self._trail_lock_done = False
        self._entry_bar_ts = bar_ts
        self._last_setup = str(entry["trigger"])
        self._reentry_side = None
        self._reentry_exit_ts = None

        actual_rr = abs(self._entry_tp - self._entry_price) / max(self._initial_risk, 1e-12)
        meta.update({
            "direction": direction, "entry_trigger": entry["trigger"],
            "stop_loss": round(self._entry_sl, 8), "take_profit": round(self._entry_tp, 8),
            "rr_ratio": round(actual_rr, 2), "tp1_r": self.T1_R, "tp1_close_pct": 0.0,
            "tp1_lock_r": self.T1_LOCK_R, "trail_r": self.TRAIL_R, "trail_lock_r": self.TRAIL_LOCK_R,
            "target_source": entry["target_source"], "boosters": entry["boosters"],
            "risk_plan": f"STRUCTURE_SL__T1_{self.T1_R:.1f}R_LOCK_{self.T1_LOCK_R:.1f}R__TRAIL_{self.TRAIL_R:.1f}R_LOCK_{self.TRAIL_LOCK_R:.1f}R__TP_{actual_rr:.2f}R_{entry['target_source']}",
        })
        reason = (
            f"{direction.upper()} {entry['trigger']} | Struct={structure['label']} "
            f"ADX={market['adx']:.1f} CHOP={market['chop']:.1f} "
            f"Room={entry['room_r']:.2f}R Dist={entry['distance_atr']:.2f}ATR "
            f"RSI={entry['rsi']:.0f}/{entry['rsi_sma']:.0f} TP={actual_rr:.2f}R"
        )
        return Signal(signal_type, self.symbol, self._entry_price, 0.0, reason, float(entry["confidence"]), meta)

    def _technical_exit(self, side: str, bar_ts: int, reason: str) -> PositionUpdate:
        self._last_exit_bar_ts = int(bar_ts)
        self._reentry_side = side
        self._reentry_exit_ts = int(bar_ts)
        self._reset_position(keep_exit_ts=True)
        return PositionUpdate(action="close", close_pct=1.0, reason=reason)

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        if self._open_position is None:
            return None

        candles = self._latest_15m
        if len(candles) >= 50:
            bar_ts = self._bar_ts(candles[-1])
            atr_now = float(self.atr(candles, 14)[-1])
            if self._finite(atr_now):
                side = str(self._open_position)
                long = side == "long"
                closes = [float(c.close) for c in candles]
                ema20 = self.ema(closes, 20)
                hma16 = self.hma(closes, 16)
                structure = self._structure(candles, max(atr_now, 1e-12))
                close = closes[-1]
                current = candles[-1]

                if long:
                    pivot_broken = structure["last_pl"] is not None and close < float(structure["last_pl"]) - 0.05 * atr_now
                    strong_invalid = pivot_broken and close < float(ema20[-1]) and close < float(current.open)
                else:
                    pivot_broken = structure["last_ph"] is not None and close > float(structure["last_ph"]) + 0.05 * atr_now
                    strong_invalid = pivot_broken and close > float(ema20[-1]) and close > float(current.open)

                if strong_invalid:
                    return self._technical_exit(side, bar_ts, f"15M structure invalidation + EMA20 loss — close {side.upper()}")

                bars_after_entry = sum(1 for candle in candles if self._entry_bar_ts is not None and self._bar_ts(candle) > self._entry_bar_ts)
                grace_active = self._entry_bar_ts is not None and bars_after_entry < self.ENTRY_GRACE_BARS
                if not grace_active and self._finite(ema20[-1], ema20[-2], hma16[-1], hma16[-2]):
                    two_ema20_fail = (closes[-1] < ema20[-1] and closes[-2] < ema20[-2]) if long else (closes[-1] > ema20[-1] and closes[-2] > ema20[-2])
                    hma_opposite = float(hma16[-1]) < float(hma16[-2]) if long else float(hma16[-1]) > float(hma16[-2])
                    if two_ema20_fail and hma_opposite:
                        return self._technical_exit(side, bar_ts, f"15M trend failure: 2 closes beyond EMA20 + HMA16 flip — close {side.upper()}")

        if self._entry_price is not None and self._initial_risk is not None and self._initial_risk > 0:
            profit = float(current_price) - self._entry_price if self._open_position == "long" else self._entry_price - float(current_price)
            current_r = profit / self._initial_risk
            if not self._tp1_done and current_r >= self.T1_R:
                self._tp1_done = True
                locked_sl = self._entry_price + self.T1_LOCK_R * self._initial_risk if self._open_position == "long" else self._entry_price - self.T1_LOCK_R * self._initial_risk
                return PositionUpdate(action="partial_tp", close_pct=0.0, new_sl=round(float(locked_sl), 8), reason=f"T1 {current_r:.2f}R — no trim; lock +{self.T1_LOCK_R:.2f}R and let winner run")
            if self._tp1_done and not self._trail_lock_done and current_r >= self.TRAIL_R:
                self._trail_lock_done = True
                locked_sl = self._entry_price + self.TRAIL_LOCK_R * self._initial_risk if self._open_position == "long" else self._entry_price - self.TRAIL_LOCK_R * self._initial_risk
                return PositionUpdate(action="partial_tp", close_pct=0.0, new_sl=round(float(locked_sl), 8), reason=f"Runner {current_r:.2f}R — no trim; raise profit lock to +{self.TRAIL_LOCK_R:.2f}R")

        grace_note = ""
        if self._entry_bar_ts is not None and self._latest_15m:
            bars_after_entry = sum(1 for candle in self._latest_15m if self._bar_ts(candle) > self._entry_bar_ts)
            if bars_after_entry < self.ENTRY_GRACE_BARS:
                grace_note = f" | entry grace {bars_after_entry}/{self.ENTRY_GRACE_BARS}"
        return PositionUpdate(action="hold", reason="Holding — exits use structure/EMA20 confirmation; EMA8/13 cross ignored" + grace_note)

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        side = self._open_position
        exit_ts = self._bar_ts(self._latest_15m[-1]) if self._latest_15m else None
        if side in {"long", "short"} and exit_ts is not None:
            self._reentry_side = str(side)
            self._reentry_exit_ts = int(exit_ts)
        super().record_closed_trade(exit_price, reason, duration_min)

    def _reset_position(self, keep_exit_ts: bool = False) -> None:
        super()._reset_position(keep_exit_ts=keep_exit_ts)
        self._entry_bar_ts = None
        self._trail_lock_done = False
        self._last_setup = ""
