"""Sentinel V2 — Simple Precision execution with Sentinel X structure/location intelligence.

Keeps the production 4H -> 1H -> 15M architecture and risk lifecycle from
SimplePrecisionStrategy, while adding:
- confirmed 15M structure state (HH/HL/LH/LL, BOS, CHOCH, liquidity sweep)
- local + 1H + 4H support/resistance location and room
- fast impulse as a confidence booster only (never a hard gate)
- adaptive TP2 capped by the nearest opposing S/R, fallback to the base 2R target

No RSI/SMA cross is used as a hard entry gate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import SignalType
from .simple_precision_strategy import SimplePrecisionStrategy


class SentinelV2Strategy(SimplePrecisionStrategy):
    """Production Sentinel execution core."""

    VERSION = "2.0"

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV2({symbol})"
        self._context_1h: list = []
        self._context_4h: list = []

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
        candidates = [v for v in levels if v > price]
        return min(candidates) if candidates else None

    @staticmethod
    def _nearest_below(price: float, levels: list[float]) -> Optional[float]:
        candidates = [v for v in levels if v < price]
        return max(candidates) if candidates else None

    def _structure_state(self, candles: list) -> dict:
        highs, lows = self._recent_pivots(candles[-90:], span=2)
        last_ph = highs[-1] if highs else None
        prev_ph = highs[-2] if len(highs) >= 2 else None
        last_pl = lows[-1] if lows else None
        prev_pl = lows[-2] if len(lows) >= 2 else None

        hh = last_ph is not None and prev_ph is not None and last_ph > prev_ph
        lh = last_ph is not None and prev_ph is not None and last_ph < prev_ph
        hl = last_pl is not None and prev_pl is not None and last_pl > prev_pl
        ll = last_pl is not None and prev_pl is not None and last_pl < prev_pl

        state = 2 if hh and hl else -2 if lh and ll else 1 if (hh or hl) else -1 if (lh or ll) else 0
        label = "HH/HL" if state == 2 else "LH/LL" if state == -2 else "BULLISH" if state == 1 else "BEARISH" if state == -1 else "MIXED"
        return {
            "state": state,
            "label": label,
            "last_ph": last_ph,
            "prev_ph": prev_ph,
            "last_pl": last_pl,
            "prev_pl": prev_pl,
            "highs": highs,
            "lows": lows,
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
        raw = (
            roc1 * 0.18
            + roc3 * 0.12
            + price_accel * 0.18
            + rsi_vel * 0.12
            + macd_hist_accel * 0.18
            + di_accel * 0.10
            + displacement * 0.12
        )
        return float(np.clip(raw, -100.0, 100.0))

    def _structure_location(self, candles: list, direction: str, close: float, risk: float, atr_now: float) -> dict:
        structure = self._structure_state(candles)
        previous_close = float(candles[-2].close)
        candle = candles[-1]
        rng = max(float(candle.high) - float(candle.low), 1e-12)
        body_eff = abs(float(candle.close) - float(candle.open)) / rng
        clv = (float(candle.close) - float(candle.low)) / rng

        last_ph = structure["last_ph"]
        last_pl = structure["last_pl"]
        bos_up = last_ph is not None and close > last_ph + atr_now * 0.06 and previous_close <= last_ph and body_eff >= 0.36
        bos_dn = last_pl is not None and close < last_pl - atr_now * 0.06 and previous_close >= last_pl and body_eff >= 0.36
        sweep_up = (
            last_pl is not None
            and float(candle.low) < last_pl - atr_now * 0.04
            and close > last_pl
            and clv > 0.62
            and close > float(candle.open)
        )
        sweep_dn = (
            last_ph is not None
            and float(candle.high) > last_ph + atr_now * 0.04
            and close < last_ph
            and clv < 0.38
            and close < float(candle.open)
        )
        choch_up = structure["state"] <= 0 and bos_up
        choch_dn = structure["state"] >= 0 and bos_dn

        local_highs, local_lows = structure["highs"], structure["lows"]
        h1_highs, h1_lows = self._recent_pivots(self._context_1h[-90:], span=2) if len(self._context_1h) >= 7 else ([], [])
        h4_highs, h4_lows = self._recent_pivots(self._context_4h[-90:], span=2) if len(self._context_4h) >= 7 else ([], [])

        tolerance = atr_now * 0.35
        resistances = self._merge_levels(local_highs[-4:] + h1_highs[-3:] + h4_highs[-3:], tolerance)
        supports = self._merge_levels(local_lows[-4:] + h1_lows[-3:] + h4_lows[-3:], tolerance)

        nearest_resistance = self._nearest_above(close, resistances)
        nearest_support = self._nearest_below(close, supports)
        nearest_opposing = nearest_resistance if direction == "long" else nearest_support
        room_r = abs(nearest_opposing - close) / max(risk, 1e-12) if nearest_opposing is not None else 3.0

        near_support = nearest_support is not None and abs(close - nearest_support) <= atr_now * 0.55
        near_resistance = nearest_resistance is not None and abs(close - nearest_resistance) <= atr_now * 0.55
        aligned_structure = structure["state"] > 0 if direction == "long" else structure["state"] < 0
        opposing_structure = structure["state"] <= -2 if direction == "long" else structure["state"] >= 2

        return {
            "structure": {
                "state": structure["state"],
                "label": structure["label"],
                "bos_up": bool(bos_up),
                "bos_down": bool(bos_dn),
                "choch_up": bool(choch_up),
                "choch_down": bool(choch_dn),
                "sweep_up": bool(sweep_up),
                "sweep_down": bool(sweep_dn),
                "aligned": bool(aligned_structure),
                "opposing": bool(opposing_structure),
            },
            "location": {
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "nearest_opposing": nearest_opposing,
                "near_support": bool(near_support),
                "near_resistance": bool(near_resistance),
                "room_r": float(room_r),
            },
        }

    def _entry_15m(self, candles: list, direction: str, current_price: float) -> dict:
        entry = super()._entry_15m(candles, direction, current_price)
        if len(candles) < 35 or "risk" not in entry:
            return entry

        close = float(candles[-1].close)
        atr_now = float(entry.get("atr") or 0.0)
        risk = max(float(entry.get("risk") or 0.0), 1e-12)
        context = self._structure_location(candles, direction, close, risk, atr_now)
        structure = context["structure"]
        location = context["location"]
        room_r = float(location["room_r"])

        # Extend the existing 7-bar breakout with confirmed pivot BOS. This is
        # still one of the original three entry families, not a new fourth gate.
        if entry.get("trigger") is None and entry.get("reason", "").startswith("waiting"):
            candle = candles[-1]
            volumes = [float(c.volume or 0.0) for c in candles[-21:-1]]
            median_volume = float(np.median(volumes)) if volumes else 0.0
            body = abs(float(candle.close) - float(candle.open))
            volume_ok = median_volume <= 0.0 or float(candle.volume or 0.0) >= 0.90 * median_volume
            bos_aligned = structure["bos_up"] if direction == "long" else structure["bos_down"]
            if bool(entry.get("aligned")) and bos_aligned and volume_ok and body >= 0.15 * max(atr_now, 1e-12):
                entry["trigger"] = "STRUCTURE_BREAKOUT"
                entry["reason"] = "confirmed pivot BOS breakout"

        # HTF/local combined room is authoritative. It can block an entry that
        # looked clear on 15M alone when a 1H/4H obstacle sits immediately ahead.
        if entry.get("trigger") is not None and room_r < self.min_room_r:
            entry["trigger"] = None
            entry["reason"] = "trigger ready but HTF/local opposing structure is too close"

        fast_impulse = self._fast_impulse(candles)

        entry_price = float(entry.get("entry") or current_price or close)
        fixed_target = entry_price + self.target_r * risk if direction == "long" else entry_price - self.target_r * risk
        nearest = location["nearest_opposing"]
        target = fixed_target
        target_source = f"{self.target_r:.1f}R"

        # Never place TP2 beyond the first meaningful opposing structure.
        # A small ATR buffer aims to exit just before the actual S/R level.
        if nearest is not None and room_r >= self.min_room_r:
            buffer = 0.05 * max(atr_now, 1e-12)
            sr_target = float(nearest) - buffer if direction == "long" else float(nearest) + buffer
            if direction == "long" and sr_target > entry_price:
                target = min(fixed_target, sr_target)
            elif direction == "short" and sr_target < entry_price:
                target = max(fixed_target, sr_target)
            if abs(target - fixed_target) > max(1e-12, risk * 0.03):
                target_source = "NEAREST_S/R"

        entry["take_profit"] = float(target)
        entry["room_r"] = round(room_r, 2)
        entry["nearest_opposing"] = nearest
        entry["target_source"] = target_source
        entry["target_rr"] = round(abs(target - entry_price) / risk, 2)
        entry["fast_impulse"] = round(fast_impulse, 1)
        entry["structure"] = structure
        entry["location"] = location
        return entry

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        mtf = mtf_candles or {}
        self._context_1h = self._closed_candle_series(mtf.get("1h", []), 60 * 60_000)
        self._context_4h = self._closed_candle_series(mtf.get("4h", []), 4 * 60 * 60_000)

        signal = await super().analyze(candles, current_price, mtf_candles)
        meta = signal.metadata or {}
        meta["strategy"] = "SENTINEL_V2"
        meta["version"] = self.VERSION
        meta["architecture"] = "4H_DIRECTION__1H_QUALITY__15M_STRUCTURE_LOCATION_TRIGGER"
        signal.metadata = meta

        entry = meta.get("entry_15m") or {}
        if signal.type in {SignalType.BUY, SignalType.SELL} and entry:
            direction = meta.get("direction") or ("long" if signal.type == SignalType.BUY else "short")
            fast = float(entry.get("fast_impulse") or 0.0)
            structure = entry.get("structure") or {}
            location = entry.get("location") or {}

            fast_aligned = fast > 10.0 if direction == "long" else fast < -10.0
            structure_aligned = bool(structure.get("aligned"))
            location_aligned = bool(location.get("near_support")) if direction == "long" else bool(location.get("near_resistance"))

            # Booster only: never reduces confidence and never vetoes a valid setup.
            boost = (0.04 if fast_aligned else 0.0) + (0.025 if structure_aligned else 0.0) + (0.015 if location_aligned else 0.0)
            signal.confidence = min(0.98, float(signal.confidence) + boost)

            if self._entry_price is not None and self._initial_risk:
                self._entry_tp = float(entry["take_profit"])
                actual_rr = abs(self._entry_tp - self._entry_price) / max(float(self._initial_risk), 1e-12)
                meta["take_profit"] = round(self._entry_tp, 8)
                meta["rr_ratio"] = round(actual_rr, 2)
                meta["target_source"] = entry.get("target_source", f"{self.target_r:.1f}R")
                meta["risk_plan"] = (
                    f"SL_{self._initial_risk:.8f}__T1_{self.tp1_r:.1f}R_TRIM{self.tp1_trim_pct:.0%}_BE"
                    f"__TP2_{actual_rr:.2f}R_{meta['target_source']}"
                )

            signal.reason += (
                f" | Struct={structure.get('label', 'MIXED')}"
                f" Fast={fast:+.0f}"
                f" TP2={entry.get('target_source', f'{self.target_r:.1f}R')}"
            )

        return signal
