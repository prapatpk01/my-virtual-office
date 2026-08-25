"""Sentinel V3 — 15M Unified execution engine.

Single-timeframe architecture:
- 15M Market Quality decides whether conditions are tradable.
- 15M Sentinel X intelligence reads structure, location, S/R, BOS/CHOCH/sweeps.
- Entry triggers decide LONG/SHORT; there is no separate direction layer.
- DMI, MACD and Fast Impulse are confidence boosters, never hard entry gates.
- Risk lifecycle remains the proven Simple Precision structure/ATR SL, T1 partial,
  breakeven move, adaptive TP2, and technical exits.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Signal, SignalType
from .simple_precision_strategy import SimplePrecisionStrategy
from ..engines.position_manager import PositionUpdate


class SentinelV3Strategy(SimplePrecisionStrategy):
    """Production Sentinel: 15M quality + trigger-driven direction."""

    VERSION = "3.0"
    entry_tf = "15m"

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV3({symbol})"

    @staticmethod
    def _merge_levels(levels: list[float], tolerance: float) -> list[float]:
        cleaned = sorted(float(v) for v in levels if v is not None and np.isfinite(float(v)))
        if not cleaned:
            return []
        merged = [cleaned[0]]
        for value in cleaned[1:]:
            if abs(value - merged[-1]) <= max(tolerance, 1e-12):
                merged[-1] = (merged[-1] + value) / 2.0
            else:
                merged.append(value)
        return merged

    @staticmethod
    def _nearest_above(price: float, levels: list[float]) -> Optional[float]:
        values = [v for v in levels if v > price]
        return min(values) if values else None

    @staticmethod
    def _nearest_below(price: float, levels: list[float]) -> Optional[float]:
        values = [v for v in levels if v < price]
        return max(values) if values else None

    def _structure_snapshot(self, candles: list, atr_now: float) -> dict:
        recent = candles[-100:]
        highs2, lows2 = self._recent_pivots(recent, span=2)
        highs4, lows4 = self._recent_pivots(recent, span=4)
        last_ph = highs2[-1] if highs2 else None
        prev_ph = highs2[-2] if len(highs2) >= 2 else None
        last_pl = lows2[-1] if lows2 else None
        prev_pl = lows2[-2] if len(lows2) >= 2 else None

        hh = last_ph is not None and prev_ph is not None and last_ph > prev_ph
        lh = last_ph is not None and prev_ph is not None and last_ph < prev_ph
        hl = last_pl is not None and prev_pl is not None and last_pl > prev_pl
        ll = last_pl is not None and prev_pl is not None and last_pl < prev_pl
        state = 2 if hh and hl else -2 if lh and ll else 1 if (hh or hl) else -1 if (lh or ll) else 0
        label = "HH/HL" if state == 2 else "LH/LL" if state == -2 else "BULLISH" if state == 1 else "BEARISH" if state == -1 else "MIXED"

        candle = candles[-1]
        previous_close = float(candles[-2].close)
        close = float(candle.close)
        rng = max(float(candle.high) - float(candle.low), 1e-12)
        body_eff = abs(float(candle.close) - float(candle.open)) / rng
        clv = (close - float(candle.low)) / rng

        bos_up = last_ph is not None and close > last_ph + atr_now * 0.06 and previous_close <= last_ph and body_eff >= 0.36
        bos_down = last_pl is not None and close < last_pl - atr_now * 0.06 and previous_close >= last_pl and body_eff >= 0.36
        sweep_up = last_pl is not None and float(candle.low) < last_pl - atr_now * 0.04 and close > last_pl and clv > 0.62 and close > float(candle.open)
        sweep_down = last_ph is not None and float(candle.high) > last_ph + atr_now * 0.04 and close < last_ph and clv < 0.38 and close < float(candle.open)
        choch_up = state <= 0 and bos_up
        choch_down = state >= 0 and bos_down

        local_high4 = max(float(c.high) for c in candles[-5:-1])
        local_low4 = min(float(c.low) for c in candles[-5:-1])
        micro_bos_up = close > local_high4 and close > float(candle.open) and body_eff >= 0.36
        micro_bos_down = close < local_low4 and close < float(candle.open) and body_eff >= 0.36

        tolerance = atr_now * 0.35
        resistances = self._merge_levels(highs2[-5:] + highs4[-4:], tolerance)
        supports = self._merge_levels(lows2[-5:] + lows4[-4:], tolerance)

        return {
            "state": state, "label": label, "last_ph": last_ph, "last_pl": last_pl,
            "bos_up": bool(bos_up), "bos_down": bool(bos_down),
            "choch_up": bool(choch_up), "choch_down": bool(choch_down),
            "sweep_up": bool(sweep_up), "sweep_down": bool(sweep_down),
            "micro_bos_up": bool(micro_bos_up), "micro_bos_down": bool(micro_bos_down),
            "resistances": resistances, "supports": supports, "body_eff": float(body_eff),
        }

    def _fast_impulse(self, candles: list) -> float:
        if len(candles) < 35:
            return 0.0
        closes = [float(c.close) for c in candles]
        atr = self.atr(candles, 14)
        rsi = self.rsi(closes, 14)
        macd, macd_signal, macd_hist = self.macd(closes)
        _, plus_di, minus_di = self.adx(candles, 14)
        if not self._finite(atr[-1], rsi[-1], rsi[-3], plus_di[-1], minus_di[-1]):
            return 0.0

        den = max(float(atr[-1]), 1e-12)
        roc1 = (closes[-1] - closes[-2]) / den * 35.0
        roc3 = (closes[-1] - closes[-4]) / den * 18.0
        old_roc1 = (closes[-3] - closes[-4]) / den * 35.0
        price_accel = (roc1 - old_roc1) * 1.40
        rsi_vel = (float(rsi[-1]) - float(rsi[-3])) * 2.20
        hist_now = float(macd_hist[-1]) if np.isfinite(macd_hist[-1]) else float(macd[-1] - macd_signal[-1])
        hist_old = float(macd_hist[-3]) if np.isfinite(macd_hist[-3]) else 0.0
        macd_hist_accel = (hist_now - hist_old) / den * 120.0
        di_den = max(float(plus_di[-1] + minus_di[-1]), 1.0)
        di_now = (float(plus_di[-1]) - float(minus_di[-1])) / di_den * 100.0
        old_den = max(float(plus_di[-3] + minus_di[-3]), 1.0)
        di_old = (float(plus_di[-3]) - float(minus_di[-3])) / old_den * 100.0
        di_accel = (di_now - di_old) * 0.95
        candle = candles[-1]
        displacement = float(np.clip((float(candle.close) - float(candle.open)) / den * 42.0, -100.0, 100.0))
        raw = roc1 * 0.18 + roc3 * 0.12 + price_accel * 0.18 + rsi_vel * 0.12 + macd_hist_accel * 0.18 + di_accel * 0.10 + displacement * 0.12
        return float(np.clip(raw, -100.0, 100.0))

    def _market_quality(self, candles: list) -> dict:
        if len(candles) < 60:
            return {"ready": False, "score": 0.0, "hard_blocks": ["WARMUP"], "reason": "15M warmup"}

        closes = [float(c.close) for c in candles]
        atr = self.atr(candles, 14)
        adx, plus_di, minus_di = self.adx(candles, 14)
        macd, macd_signal, macd_hist = self.macd(closes)
        chop = self._choppiness(candles, 14)
        if chop is None or not self._finite(atr[-1], adx[-1], plus_di[-1], minus_di[-1]):
            return {"ready": False, "score": 0.0, "hard_blocks": ["INDICATORS"], "reason": "15M indicators unavailable"}

        adx_now = float(adx[-1])
        chop_now = float(chop)
        adx_score = 20.0 if adx_now >= 25.0 else 16.0 if adx_now >= 20.0 else 12.0 if adx_now >= 15.0 else 7.0 if adx_now >= 12.0 else 0.0
        chop_score = 20.0 if chop_now < 45.0 else 17.0 if chop_now < 50.0 else 13.0 if chop_now < 55.0 else 8.0 if chop_now < 60.0 else 4.0 if chop_now < 65.0 else 0.0

        atr_values = [float(v) for v in atr[-21:-1] if np.isfinite(v)]
        atr_median = float(np.median(atr_values)) if atr_values else float(atr[-1])
        atr_ratio = float(atr[-1]) / max(atr_median, 1e-12)
        volatility_score = 10.0 if atr_ratio >= 1.10 else 8.0 if atr_ratio >= 0.85 else 5.0 if atr_ratio >= 0.70 else 2.0

        hard_blocks = []
        if adx_now < 12.0:
            hard_blocks.append("ADX")
        if chop_now >= 65.0:
            hard_blocks.append("CHOP")
        hist_now = float(macd_hist[-1]) if np.isfinite(macd_hist[-1]) else float(macd[-1] - macd_signal[-1])

        return {
            "ready": not hard_blocks,
            "score": round(adx_score + chop_score + volatility_score, 1),
            "adx": round(adx_now, 1), "chop": round(chop_now, 1), "atr_ratio": round(atr_ratio, 2),
            "plus_di": round(float(plus_di[-1]), 1), "minus_di": round(float(minus_di[-1]), 1),
            "macd_hist": hist_now,
            "components": {"adx": adx_score, "chop": chop_score, "volatility": volatility_score},
            "hard_blocks": hard_blocks,
            "reason": "market tradable" if not hard_blocks else "blocked: " + ",".join(hard_blocks),
        }

    def _trigger_candidates(self, candles: list, structure: dict, atr_now: float) -> list[dict]:
        closes = [float(c.close) for c in candles]
        ema8 = self.ema(closes, 8)
        ema13 = self.ema(closes, 13)
        ema20 = self.ema(closes, 20)
        hma16 = self.hma(closes, 16)
        if not self._finite(ema8[-1], ema13[-1], ema20[-1], hma16[-1], hma16[-2]):
            return []

        close = closes[-1]
        previous_close = closes[-2]
        candle = candles[-1]
        body = abs(float(candle.close) - float(candle.open))
        volumes = [float(c.volume or 0.0) for c in candles[-21:-1]]
        median_volume = float(np.median(volumes)) if volumes else 0.0
        volume_ratio = float(candle.volume or 0.0) / max(median_volume, 1e-12) if median_volume > 0 else 1.0
        volume_ok = median_volume <= 0.0 or volume_ratio >= 0.90

        candidates = []
        long_aligned = ema8[-1] > ema13[-1] and close > ema8[-1]
        short_aligned = ema8[-1] < ema13[-1] and close < ema8[-1]
        long_cross = (ema8[-2] <= ema13[-2] and ema8[-1] > ema13[-1]) or (ema8[-3] <= ema13[-3] and ema8[-2] > ema13[-2])
        short_cross = (ema8[-2] >= ema13[-2] and ema8[-1] < ema13[-1]) or (ema8[-3] >= ema13[-3] and ema8[-2] < ema13[-2])

        if long_aligned and long_cross:
            candidates.append({"direction": "long", "trigger": "EMA8_13_CROSS", "priority": 30})
        if short_aligned and short_cross:
            candidates.append({"direction": "short", "trigger": "EMA8_13_CROSS", "priority": 30})

        long_touch = float(candle.low) <= float(ema13[-1]) or previous_close <= float(ema13[-2])
        short_touch = float(candle.high) >= float(ema13[-1]) or previous_close >= float(ema13[-2])
        if long_touch and long_aligned and close > float(candle.open):
            candidates.append({"direction": "long", "trigger": "EMA13_PULLBACK_RECLAIM", "priority": 20})
        if short_touch and short_aligned and close < float(candle.open):
            candidates.append({"direction": "short", "trigger": "EMA13_PULLBACK_RECLAIM", "priority": 20})

        upper = max(float(c.high) for c in candles[-8:-1])
        lower = min(float(c.low) for c in candles[-8:-1])
        breakout_up = (((close > upper and previous_close <= upper) or structure["bos_up"]) and body >= 0.15 * atr_now and volume_ok)
        breakout_down = (((close < lower and previous_close >= lower) or structure["bos_down"]) and body >= 0.15 * atr_now and volume_ok)
        if breakout_up:
            candidates.append({"direction": "long", "trigger": "STRUCTURE_BREAKOUT", "priority": 40})
        if breakout_down:
            candidates.append({"direction": "short", "trigger": "STRUCTURE_BREAKOUT", "priority": 40})

        hma_up = close > float(hma16[-1]) and float(hma16[-1]) >= float(hma16[-2])
        hma_down = close < float(hma16[-1]) and float(hma16[-1]) <= float(hma16[-2])
        if structure["sweep_up"] and (structure["choch_up"] or structure["micro_bos_up"]) and hma_up:
            candidates.append({"direction": "long", "trigger": "SWEEP_CHOCH_REVERSAL", "priority": 50})
        if structure["sweep_down"] and (structure["choch_down"] or structure["micro_bos_down"]) and hma_down:
            candidates.append({"direction": "short", "trigger": "SWEEP_CHOCH_REVERSAL", "priority": 50})

        for item in candidates:
            item.update({"ema20": float(ema20[-1]), "volume_ratio": float(volume_ratio)})
        return candidates

    def _risk_for_direction(self, candles: list, direction: str, close: float, atr_now: float) -> float:
        raw_risk = close - min(float(c.low) for c in candles[-7:]) + 0.10 * atr_now if direction == "long" else max(float(c.high) for c in candles[-7:]) - close + 0.10 * atr_now
        risk = min(max(raw_risk, self.stop_atr_min * atr_now), self.stop_atr_max * atr_now)
        return max(float(risk), close * 0.001)

    def _location(self, close: float, risk: float, atr_now: float, direction: str, structure: dict) -> dict:
        resistance = self._nearest_above(close, structure["resistances"])
        support = self._nearest_below(close, structure["supports"])
        opposing = resistance if direction == "long" else support
        room_r = abs(float(opposing) - close) / max(risk, 1e-12) if opposing is not None else 3.0
        return {
            "nearest_support": support, "nearest_resistance": resistance, "nearest_opposing": opposing,
            "room_r": float(room_r),
            "near_support": bool(support is not None and abs(close - support) <= atr_now * 0.55),
            "near_resistance": bool(resistance is not None and abs(close - resistance) <= atr_now * 0.55),
        }

    def _setup_quality(self, candles: list, candidate: dict, structure: dict, location: dict) -> dict:
        direction = candidate["direction"]
        long = direction == "long"
        state = int(structure["state"])
        room_r = float(location["room_r"])

        if (long and state > 0) or ((not long) and state < 0):
            structure_score = 15.0
        elif state == 0:
            structure_score = 9.0
        elif (long and structure["choch_up"]) or ((not long) and structure["choch_down"]):
            structure_score = 12.0
        elif abs(state) == 1:
            structure_score = 5.0
        else:
            structure_score = 0.0

        location_score = (15.0 if location["near_support"] else 0.0 if location["near_resistance"] else 8.0) if long else (15.0 if location["near_resistance"] else 0.0 if location["near_support"] else 8.0)
        room_score = 10.0 if room_r >= 2.0 else 8.0 if room_r >= 1.5 else 6.0 if room_r >= self.min_room_r else 0.0

        candle = candles[-1]
        rng = max(float(candle.high) - float(candle.low), 1e-12)
        body_eff = abs(float(candle.close) - float(candle.open)) / rng
        volume_ratio = float(candidate.get("volume_ratio", 1.0))
        candle_score = 5.0 if body_eff >= 0.55 else 3.0 if body_eff >= 0.35 else 1.0
        volume_score = 5.0 if volume_ratio >= 1.10 else 3.0 if volume_ratio >= 0.90 else 1.0

        return {
            "score": round(structure_score + location_score + room_score + candle_score + volume_score, 1),
            "components": {"structure": structure_score, "location": location_score, "room": room_score, "candle": candle_score, "volume": volume_score},
            "strong_opposition": bool((long and state <= -2) or ((not long) and state >= 2)),
            "bad_location": bool(location["near_resistance"] if long else location["near_support"]),
        }

    def _chase_limit(self, trigger: str) -> float:
        if trigger == "EMA8_13_CROSS":
            return min(self.max_entry_distance_atr, 1.20)
        if trigger == "STRUCTURE_BREAKOUT":
            return max(self.max_entry_distance_atr, 1.80)
        return self.max_entry_distance_atr

    def _build_entry(self, candles: list, current_price: float, market: dict, structure: dict) -> dict:
        close = float(candles[-1].close)
        atr_now = max(float(self.atr(candles, 14)[-1]), 1e-12)
        candidates = self._trigger_candidates(candles, structure, atr_now)
        if not candidates:
            return {"trigger": None, "reason": "waiting for cross, reclaim, structure breakout, or sweep/CHOCH reversal"}
        if len({c["direction"] for c in candidates}) > 1:
            return {"trigger": None, "reason": "conflicting LONG/SHORT triggers on same 15M bar"}

        candidate = max(candidates, key=lambda x: int(x["priority"]))
        direction = candidate["direction"]
        risk = self._risk_for_direction(candles, direction, close, atr_now)
        location = self._location(close, risk, atr_now, direction, structure)
        setup = self._setup_quality(candles, candidate, structure, location)
        quality = float(market["score"]) + float(setup["score"])
        threshold = min(70.0, self.quality_threshold + max(0.0, float(self._entry_threshold_bonus)))
        distance_atr = abs(close - float(candidate["ema20"])) / atr_now
        chase_limit = self._chase_limit(candidate["trigger"])

        blocks = list(market.get("hard_blocks", []))
        if float(location["room_r"]) < self.min_room_r:
            blocks.append("ROOM")
        if distance_atr > chase_limit:
            blocks.append("CHASE")
        if setup["strong_opposition"] and setup["bad_location"]:
            blocks.append("STRUCTURE_LOCATION_CONFLICT")

        trigger = candidate["trigger"] if quality >= threshold and not blocks else None
        reason = "15M quality + trigger confirmed" if trigger else ("blocked: " + ",".join(blocks) if blocks else f"quality {quality:.0f} below {threshold:.0f}")

        entry_price = float(current_price or close)
        stop = entry_price - risk if direction == "long" else entry_price + risk
        fixed_target = entry_price + self.target_r * risk if direction == "long" else entry_price - self.target_r * risk
        nearest = location["nearest_opposing"]
        target = fixed_target
        source = f"{self.target_r:.1f}R"
        if nearest is not None and float(location["room_r"]) >= self.min_room_r:
            sr_target = float(nearest) - 0.05 * atr_now if direction == "long" else float(nearest) + 0.05 * atr_now
            if direction == "long" and sr_target > entry_price:
                target = min(fixed_target, sr_target)
            elif direction == "short" and sr_target < entry_price:
                target = max(fixed_target, sr_target)
            if abs(target - fixed_target) > max(1e-12, risk * 0.03):
                source = "NEAREST_S/R"

        return {
            "trigger": trigger, "candidate_trigger": candidate["trigger"], "direction": direction, "reason": reason,
            "entry": entry_price, "stop_loss": float(stop), "take_profit": float(target), "risk": float(risk),
            "quality_score": round(quality, 1), "quality_threshold": round(threshold, 1),
            "market_quality": market, "setup_quality": setup, "structure": structure, "location": location,
            "room_r": round(float(location["room_r"]), 2), "distance_atr": round(float(distance_atr), 2),
            "chase_limit_atr": round(float(chase_limit), 2), "fast_impulse": round(self._fast_impulse(candles), 1),
            "target_source": source, "target_rr": round(abs(target - entry_price) / max(risk, 1e-12), 2),
            "hard_blocks": blocks,
        }

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        c15 = self._closed_candle_series(candles, 15 * 60_000)
        self._latest_15m = c15
        meta = {"strategy": "SENTINEL_V3", "version": self.VERSION, "architecture": "15M_QUALITY__15M_SENTINEL_X__TRIGGER_DIRECTION"}

        if len(c15) < 60:
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

        market = self._market_quality(c15)
        structure = self._structure_snapshot(c15, max(atr_now, 1e-12))
        meta["market_quality_15m"] = market
        meta["structure_15m"] = {k: v for k, v in structure.items() if k not in {"resistances", "supports"}}
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

        hist = float(market.get("macd_hist", 0.0))
        dmi_ok = float(market.get("plus_di", 0.0)) > float(market.get("minus_di", 0.0)) if direction == "long" else float(market.get("minus_di", 0.0)) > float(market.get("plus_di", 0.0))
        macd_ok = hist >= 0.0 if direction == "long" else hist <= 0.0
        fast = float(entry.get("fast_impulse", 0.0))
        fast_ok = fast > 10.0 if direction == "long" else fast < -10.0
        structure_ok = int(structure["state"]) > 0 if direction == "long" else int(structure["state"]) < 0
        location = entry["location"]
        location_ok = bool(location["near_support"]) if direction == "long" else bool(location["near_resistance"])

        confidence = min(0.90, max(0.62, 0.55 + float(entry["quality_score"]) * 0.004))
        confidence += 0.03 if dmi_ok else 0.0
        confidence += 0.03 if macd_ok else 0.0
        confidence += 0.04 if fast_ok else 0.0
        confidence += 0.025 if structure_ok else 0.0
        confidence += 0.015 if location_ok else 0.0
        confidence = min(0.98, confidence)

        actual_rr = abs(self._entry_tp - self._entry_price) / max(self._initial_risk, 1e-12)
        meta.update({
            "direction": direction, "entry_trigger": entry["trigger"],
            "stop_loss": round(self._entry_sl, 8), "take_profit": round(self._entry_tp, 8),
            "rr_ratio": round(actual_rr, 2), "tp1_r": self.tp1_r, "tp1_close_pct": self.tp1_trim_pct,
            "target_source": entry["target_source"],
            "boosters": {"dmi": dmi_ok, "macd": macd_ok, "fast_impulse": fast_ok, "structure": structure_ok, "location": location_ok},
            "risk_plan": f"SL_{self._initial_risk:.8f}__T1_{self.tp1_r:.1f}R_TRIM{self.tp1_trim_pct:.0%}_BE__TP2_{actual_rr:.2f}R_{entry['target_source']}",
        })
        reason = f"{direction.upper()} {entry['trigger']} | Q={entry['quality_score']:.0f}/{entry['quality_threshold']:.0f} Struct={structure['label']} Room={entry['room_r']:.2f}R Dist={entry['distance_atr']:.2f}ATR Fast={fast:+.0f} TP2={entry['target_source']}"
        return Signal(signal_type, self.symbol, self._entry_price, 0.0, reason, confidence, meta)

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        if self._open_position is not None and len(self._latest_15m) >= 35:
            candles = self._latest_15m
            atr_now = float(self.atr(candles, 14)[-1])
            if self._finite(atr_now):
                structure = self._structure_snapshot(candles, max(atr_now, 1e-12))
                invalid = structure["choch_down"] if self._open_position == "long" else structure["choch_up"]
                if invalid:
                    side = self._open_position
                    self._last_exit_bar_ts = self._bar_ts(candles[-1])
                    self._reset_position(keep_exit_ts=True)
                    return PositionUpdate(action="close", close_pct=1.0, reason=f"15M structure invalidation (CHOCH) — close {side.upper()}")
        return super().tick_open_position(current_price, position_key)
