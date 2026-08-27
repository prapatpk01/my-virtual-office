"""Sentinel V4.2 — Responsive 15M Price-Action.

Keeps V4.1 risk/position-management philosophy while removing the entry
bottlenecks that made V4.1 too quiet:
- market gate: ADX >= 12, CHOP < 65, active ATR retained
- RSI(14) + SMA(14) of RSI remains soft context
- pullback requires structure + EMA20/HMA16 reclaim; micro break/candle/volume are boosters
- breakout accepts fresh BOS OR BOS->retest within five bars
- sweep reversal requires sweep+reclaim and (micro BOS OR HMA16 confirmation)
- max structure stop 1.8 ATR
- min target/room remains 1.5R; base target remains 2.0R
- wider setup-specific chase limits
"""
from __future__ import annotations

import numpy as np

from .sentinel_v41_strategy import SentinelV41Strategy


class SentinelV42Strategy(SentinelV41Strategy):
    VERSION = "4.2"
    ADX_FLOOR = 12.0
    CHOP_CEILING = 65.0
    MAX_STOP_ATR = 1.80

    def __init__(self, symbol: str, **kwargs):
        kwargs["min_room_r"] = 1.50
        kwargs["adx_min"] = self.ADX_FLOOR
        kwargs["chop_max"] = self.CHOP_CEILING
        kwargs["stop_atr_max"] = self.MAX_STOP_ATR
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV4.2({symbol})"
        self.min_room_r = 1.50

    def _market_gate(self, candles: list) -> dict:
        closes = [float(c.close) for c in candles]
        atr = self.atr(candles, 14)
        adx, plus_di, minus_di = self.adx(candles, 14)
        chop = self._choppiness(candles, 14)
        rsi = self.rsi(closes, 14)
        rsi_sma = self.sma(list(rsi), 14)
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

        blocks = []
        if adx_now < self.ADX_FLOOR:
            blocks.append("ADX")
        if chop_now >= self.CHOP_CEILING:
            blocks.append("CHOP")
        if atr_ratio < 0.65:
            blocks.append("DEAD_VOL")

        hist = float(macd_hist[-1]) if np.isfinite(macd_hist[-1]) else float(macd[-1] - macd_signal[-1])
        return {
            "ready": not blocks,
            "blocks": blocks,
            "adx": round(adx_now, 1),
            "adx_floor": self.ADX_FLOOR,
            "chop": round(chop_now, 1),
            "chop_ceiling": self.CHOP_CEILING,
            "atr_ratio": round(atr_ratio, 2),
            "plus_di": round(float(plus_di[-1]), 1),
            "minus_di": round(float(minus_di[-1]), 1),
            "macd_hist": hist,
            "rsi": round(float(rsi[-1]), 1),
            "rsi_sma": round(float(rsi_sma[-1]), 1),
            "reason": "market gate pass" if not blocks else "blocked: " + ",".join(blocks),
        }

    def _pullback_setup(self, candles, structure, atr_now, ema20, ema50, hma16):
        current = candles[-1]
        close = float(current.close)
        body_eff, clv = self._bar_quality(current)
        prev_high = max(float(c.high) for c in candles[-3:-1])
        prev_low = min(float(c.low) for c in candles[-3:-1])
        candidates = []

        touch_long = None
        touch_short = None
        for i in range(len(candles) - 4, len(candles)):
            if not self._finite(ema20[i], hma16[i]):
                continue
            zone_low = min(float(ema20[i]), float(hma16[i]))
            zone_high = max(float(ema20[i]), float(hma16[i]))
            if float(candles[i].low) <= zone_high + 0.12 * atr_now:
                touch_long = i
            if float(candles[i].high) >= zone_low - 0.12 * atr_now:
                touch_short = i

        long_required = (
            structure["state"] > 0
            and float(ema20[-1]) > float(ema50[-1])
            and touch_long is not None
            and close > max(float(ema20[-1]), float(hma16[-1]))
        )
        short_required = (
            structure["state"] < 0
            and float(ema20[-1]) < float(ema50[-1])
            and touch_short is not None
            and close < min(float(ema20[-1]), float(hma16[-1]))
        )

        if long_required:
            base = min(float(c.low) for c in candles[touch_long:])
            pivot = float(structure["last_pl"]) if structure["last_pl"] is not None else base
            candidates.append({
                "direction": "long", "setup": "PULLBACK_CONTINUATION", "priority": 20,
                "invalidation": min(base, pivot) - 0.08 * atr_now,
                "event_ts": self._bar_ts(current),
                "micro_confirm": close > prev_high,
                "candle_strength": close > float(current.open) and clv >= 0.55 and body_eff >= 0.25,
            })
        if short_required:
            base = max(float(c.high) for c in candles[touch_short:])
            pivot = float(structure["last_ph"]) if structure["last_ph"] is not None else base
            candidates.append({
                "direction": "short", "setup": "PULLBACK_CONTINUATION", "priority": 20,
                "invalidation": max(base, pivot) + 0.08 * atr_now,
                "event_ts": self._bar_ts(current),
                "micro_confirm": close < prev_low,
                "candle_strength": close < float(current.open) and clv <= 0.45 and body_eff >= 0.25,
            })
        return candidates

    def _breakout_retest_setup(self, candles, structure, atr_now):
        current = candles[-1]
        previous = candles[-2]
        close = float(current.close)
        body_eff, clv = self._bar_quality(current)
        candidates = []
        last_ph = structure["last_ph"]
        last_pl = structure["last_pl"]

        if last_ph is not None:
            fresh_bos = float(previous.close) <= float(last_ph) and close > float(last_ph) + 0.05 * atr_now and body_eff >= 0.20
            prior_bos = None
            for i in range(max(1, len(candles) - 6), len(candles) - 1):
                b_eff, _ = self._bar_quality(candles[i])
                if float(candles[i - 1].close) <= float(last_ph) and float(candles[i].close) > float(last_ph) + 0.05 * atr_now and b_eff >= 0.22:
                    prior_bos = i
            retest = (
                prior_bos is not None
                and float(current.low) <= float(last_ph) + 0.22 * atr_now
                and close > float(last_ph)
            )
            if fresh_bos or retest:
                candidates.append({
                    "direction": "long",
                    "setup": "BREAKOUT_BOS" if fresh_bos else "BREAKOUT_RETEST",
                    "priority": 32 if fresh_bos else 30,
                    "invalidation": float(last_ph) - 0.22 * atr_now,
                    "event_ts": self._bar_ts(current),
                    "micro_confirm": True,
                    "candle_strength": close > float(current.open) and clv >= 0.55 and body_eff >= 0.25,
                })

        if last_pl is not None:
            fresh_bos = float(previous.close) >= float(last_pl) and close < float(last_pl) - 0.05 * atr_now and body_eff >= 0.20
            prior_bos = None
            for i in range(max(1, len(candles) - 6), len(candles) - 1):
                b_eff, _ = self._bar_quality(candles[i])
                if float(candles[i - 1].close) >= float(last_pl) and float(candles[i].close) < float(last_pl) - 0.05 * atr_now and b_eff >= 0.22:
                    prior_bos = i
            retest = (
                prior_bos is not None
                and float(current.high) >= float(last_pl) - 0.22 * atr_now
                and close < float(last_pl)
            )
            if fresh_bos or retest:
                candidates.append({
                    "direction": "short",
                    "setup": "BREAKOUT_BOS" if fresh_bos else "BREAKOUT_RETEST",
                    "priority": 32 if fresh_bos else 30,
                    "invalidation": float(last_pl) + 0.22 * atr_now,
                    "event_ts": self._bar_ts(current),
                    "micro_confirm": True,
                    "candle_strength": close < float(current.open) and clv <= 0.45 and body_eff >= 0.25,
                })
        return candidates

    def _sweep_reversal_setup(self, candles, structure, atr_now, hma16):
        current = candles[-1]
        close = float(current.close)
        body_eff, clv = self._bar_quality(current)
        candidates = []
        last_ph = structure["last_ph"]
        last_pl = structure["last_pl"]
        low_sweep = None
        high_sweep = None

        for i in range(len(candles) - 4, len(candles)):
            bar = candles[i]
            if last_pl is not None and float(bar.low) < float(last_pl) - 0.05 * atr_now and float(bar.close) > float(last_pl):
                low_sweep = i
            if last_ph is not None and float(bar.high) > float(last_ph) + 0.05 * atr_now and float(bar.close) < float(last_ph):
                high_sweep = i

        micro_high = max(float(c.high) for c in candles[-3:-1])
        micro_low = min(float(c.low) for c in candles[-3:-1])
        micro_up = close > micro_high
        micro_down = close < micro_low
        hma_up = self._finite(hma16[-1]) and close > float(hma16[-1])
        hma_down = self._finite(hma16[-1]) and close < float(hma16[-1])

        if low_sweep is not None and last_pl is not None and close > float(last_pl) and (micro_up or hma_up):
            sweep = candles[low_sweep]
            candidates.append({
                "direction": "long", "setup": "SWEEP_STRUCTURE_REVERSAL", "priority": 40,
                "invalidation": float(sweep.low) - 0.08 * atr_now,
                "event_ts": self._bar_ts(current),
                "micro_confirm": micro_up,
                "candle_strength": close > float(current.open) and clv >= 0.55 and body_eff >= 0.25,
            })
        if high_sweep is not None and last_ph is not None and close < float(last_ph) and (micro_down or hma_down):
            sweep = candles[high_sweep]
            candidates.append({
                "direction": "short", "setup": "SWEEP_STRUCTURE_REVERSAL", "priority": 40,
                "invalidation": float(sweep.high) + 0.08 * atr_now,
                "event_ts": self._bar_ts(current),
                "micro_confirm": micro_down,
                "candle_strength": close < float(current.open) and clv <= 0.45 and body_eff >= 0.25,
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
            return {"trigger": None, "reason": "waiting for responsive pullback, BOS/retest, or sweep reversal"}
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
        max_stop = self.MAX_STOP_ATR * atr_now
        if raw_risk > max_stop:
            blocks.append("STOP_TOO_WIDE")
        risk = max(raw_risk, min_stop) if raw_risk > 0 else min_stop
        stop = entry - risk if direction == "long" else entry + risk

        resistance = self._nearest_above(entry, structure["resistances"])
        support = self._nearest_below(entry, structure["supports"])
        opposing = resistance if direction == "long" else support
        room_r = abs(float(opposing) - entry) / max(risk, 1e-12) if opposing is not None else 2.50
        required_room = 1.50
        if room_r + 1e-9 < required_room:
            blocks.append("ROOM")

        distance_atr = abs(close - float(candidate["ema20"])) / atr_now
        chase_limit = {
            "PULLBACK_CONTINUATION": 1.30,
            "BREAKOUT_BOS": 1.60,
            "BREAKOUT_RETEST": 1.60,
            "SWEEP_STRUCTURE_REVERSAL": 1.60,
        }.get(setup, 1.40)
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
            buffered = float(opposing) - 0.05 * atr_now if direction == "long" else float(opposing) + 0.05 * atr_now
            buffered_rr = abs(buffered - entry) / max(risk, 1e-12)
            if buffered_rr >= required_room:
                target = min(fixed_target, buffered) if direction == "long" else max(fixed_target, buffered)
            elif room_r >= required_room:
                # Preserve the explicit production policy that a real 1.50R
                # target is valid even when the cosmetic S/R buffer would
                # otherwise shrink it below the floor.
                target = entry + required_room * risk if direction == "long" else entry - required_room * risk
            if abs(target - fixed_target) > 0.03 * risk:
                target_source = "NEAREST_S/R"

        actual_rr = abs(target - entry) / max(risk, 1e-12)
        if actual_rr + 1e-9 < required_room:
            blocks.append("TARGET_TOO_CLOSE")

        rsi = float(market.get("rsi", 50.0))
        rsi_sma = float(market.get("rsi_sma", 50.0))
        dmi_ok = float(market.get("plus_di", 0.0)) > float(market.get("minus_di", 0.0)) if direction == "long" else float(market.get("minus_di", 0.0)) > float(market.get("plus_di", 0.0))
        macd_hist = float(market.get("macd_hist", 0.0))
        macd_ok = macd_hist >= 0.0 if direction == "long" else macd_hist <= 0.0
        rsi_ok = (rsi >= 50.0 and rsi > rsi_sma) if direction == "long" else (rsi <= 50.0 and rsi < rsi_sma)
        rsi_extreme = rsi >= 75.0 if direction == "long" else rsi <= 25.0
        volume_ok = float(candidate.get("volume_ratio", 1.0)) >= 1.0
        micro_ok = bool(candidate.get("micro_confirm", False))
        candle_ok = bool(candidate.get("candle_strength", False))

        confidence = 0.66
        confidence += 0.05 if dmi_ok else 0.0
        confidence += 0.05 if macd_ok else 0.0
        confidence += 0.05 if rsi_ok else 0.0
        confidence += 0.04 if volume_ok else 0.0
        confidence += 0.04 if micro_ok else 0.0
        confidence += 0.03 if candle_ok else 0.0
        confidence += 0.03 if room_r >= 2.0 else 0.0
        confidence -= 0.04 if rsi_extreme else 0.0
        confidence = float(np.clip(confidence, 0.60, 0.94))

        trigger = setup if market.get("ready") and not blocks else None
        reason = "responsive price-action setup confirmed" if trigger else ("blocked: " + ",".join(blocks) if blocks else str(market.get("reason") or "market gate blocked"))
        return {
            "trigger": trigger, "candidate_trigger": setup, "direction": direction, "reason": reason,
            "entry": entry, "stop_loss": float(stop), "take_profit": float(target), "risk": float(risk),
            "atr": atr_now, "structure": structure, "room_r": round(float(room_r), 2),
            "distance_atr": round(float(distance_atr), 2), "chase_limit_atr": chase_limit,
            "target_source": target_source, "target_rr": round(float(actual_rr), 2),
            "confidence": round(confidence, 3),
            "boosters": {"dmi": dmi_ok, "macd": macd_ok, "rsi_sma": rsi_ok, "volume": volume_ok, "micro": micro_ok, "candle": candle_ok},
            "rsi": rsi, "rsi_sma": rsi_sma, "volume_ratio": round(float(candidate.get("volume_ratio", 1.0)), 2),
            "blocks": blocks, "event_ts": int(candidate.get("event_ts") or 0),
        }
