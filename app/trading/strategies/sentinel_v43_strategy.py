"""Sentinel V4.3 — Responsive 15M Price Action + 1H Major S/R.

V4.3 keeps every V4.2 entry/direction rule on 15M and adds one narrowly
scoped higher-timeframe input: confirmed 1H support/resistance as a major
obstacle for room/target calculation only.

Important:
- 1H NEVER chooses LONG/SHORT.
- 1H NEVER becomes a trend/alignment gate.
- 4H is not used by the strategy.
- 15M remains the only entry/structure engine.
- The nearer of 15M S/R and confirmed 1H major S/R controls usable room/TP.
- Minimum usable target remains 1.5R; base target remains 2.0R.
"""
from __future__ import annotations

from .sentinel_v42_strategy import SentinelV42Strategy


class SentinelV43Strategy(SentinelV42Strategy):
    VERSION = "4.3"
    H1_MIN_BARS = 30
    H1_MERGE_ATR = 0.35
    H1_TARGET_BUFFER_ATR = 0.04

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV4.3({symbol})"
        self._major_sr_1h: dict = {
            "ready": False,
            "reason": "1H not loaded",
            "supports": [],
            "resistances": [],
        }

    def _major_sr_snapshot(self, mtf_candles: dict | None) -> dict:
        raw = list((mtf_candles or {}).get("1h") or [])
        c1h = self._closed_candle_series(raw, 60 * 60_000)
        if len(c1h) < self.H1_MIN_BARS:
            return {
                "ready": False,
                "reason": "1H warmup/unavailable",
                "supports": [],
                "resistances": [],
            }

        atr_values = self.atr(c1h, 14)
        atr_1h = float(atr_values[-1])
        if not self._finite(atr_1h) or atr_1h <= 0:
            return {
                "ready": False,
                "reason": "1H ATR unavailable",
                "supports": [],
                "resistances": [],
            }

        recent = c1h[-100:]
        highs2, lows2 = self._pivot_points(recent, span=2)
        highs4, lows4 = self._pivot_points(recent, span=4)
        tolerance = atr_1h * self.H1_MERGE_ATR

        resistances = self._merge_levels(
            [p for _, p in highs2[-7:]] + [p for _, p in highs4[-5:]],
            tolerance,
        )
        supports = self._merge_levels(
            [p for _, p in lows2[-7:]] + [p for _, p in lows4[-5:]],
            tolerance,
        )

        return {
            "ready": bool(resistances or supports),
            "reason": "confirmed 1H pivots" if (resistances or supports) else "no confirmed 1H pivots",
            "atr": atr_1h,
            "supports": supports,
            "resistances": resistances,
        }

    def _build_entry(self, candles, current_price, market, structure):
        entry = super()._build_entry(candles, current_price, market, structure)
        direction = entry.get("direction")
        if direction not in {"long", "short"} or "entry" not in entry or "risk" not in entry:
            entry["major_sr_1h"] = {
                "ready": bool(self._major_sr_1h.get("ready")),
                "reason": self._major_sr_1h.get("reason", "1H unavailable"),
            }
            return entry

        px = float(entry["entry"])
        risk = max(float(entry["risk"]), 1e-12)
        major = self._major_sr_1h or {}
        resistance_1h = self._nearest_above(px, major.get("resistances", []))
        support_1h = self._nearest_below(px, major.get("supports", []))
        obstacle_1h = resistance_1h if direction == "long" else support_1h

        room_15m = float(entry.get("room_r", 2.50))
        room_1h = None
        used_for_target = False
        major_target = None
        major_target_rr = None
        target_source = str(entry.get("target_source", "2.0R"))

        if obstacle_1h is not None:
            room_1h = abs(float(obstacle_1h) - px) / risk
            effective_room = min(room_15m, room_1h)
            entry["room_r"] = round(float(effective_room), 2)

            blocks = list(entry.get("blocks", []))
            if room_1h + 1e-9 < 1.50 and "MAJOR_1H_SR" not in blocks:
                blocks.append("MAJOR_1H_SR")

            if room_1h + 1e-9 >= 1.50:
                atr_1h = max(float(major.get("atr") or 0.0), 0.0)
                buffer = self.H1_TARGET_BUFFER_ATR * atr_1h
                buffered = (
                    float(obstacle_1h) - buffer
                    if direction == "long"
                    else float(obstacle_1h) + buffer
                )
                buffered_rr = abs(buffered - px) / risk

                # Explicit production policy: a genuine 1.50R obstacle remains
                # tradable. If the small pre-S/R buffer would shrink it below
                # 1.50R, target the 1.50R floor instead of rejecting the trade.
                if buffered_rr + 1e-9 >= 1.50:
                    major_target = buffered
                    major_target_rr = buffered_rr
                    major_source = "1H_MAJOR_S/R"
                else:
                    major_target = px + 1.50 * risk if direction == "long" else px - 1.50 * risk
                    major_target_rr = 1.50
                    major_source = "1H_MAJOR_S/R_1.5R_FLOOR"

                current_target = float(entry.get("take_profit", major_target))
                current_rr = abs(current_target - px) / risk
                if major_target_rr + 1e-9 < current_rr:
                    entry["take_profit"] = float(major_target)
                    entry["target_rr"] = round(float(major_target_rr), 2)
                    entry["target_source"] = major_source
                    target_source = major_source
                    used_for_target = True

            entry["blocks"] = blocks
            if blocks:
                entry["trigger"] = None
                entry["reason"] = "blocked: " + ",".join(blocks)
        else:
            entry["room_r"] = round(room_15m, 2)

        entry["room_15m_r"] = round(room_15m, 2)
        entry["room_1h_r"] = round(float(room_1h), 2) if room_1h is not None else None
        entry["major_sr_1h"] = {
            "ready": bool(major.get("ready")),
            "reason": major.get("reason", "1H unavailable"),
            "nearest_support": support_1h,
            "nearest_resistance": resistance_1h,
            "major_obstacle": obstacle_1h,
            "room_r": round(float(room_1h), 2) if room_1h is not None else None,
            "used_for_target": used_for_target,
            "target_source": target_source,
        }
        return entry

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        # Prepare 1H major levels before the inherited V4.2 15M entry engine
        # runs. No 1H trend/direction information is consumed anywhere.
        self._major_sr_1h = self._major_sr_snapshot(mtf_candles)
        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)

        meta = signal.metadata or {}
        meta["version"] = self.VERSION
        meta["sr_architecture"] = "15M_ENTRY_STRUCTURE__1H_MAJOR_OBSTACLE_ONLY"
        meta["major_sr_1h"] = {
            "ready": bool(self._major_sr_1h.get("ready")),
            "reason": self._major_sr_1h.get("reason", "1H unavailable"),
            "atr": self._major_sr_1h.get("atr"),
            "support_count": len(self._major_sr_1h.get("supports", [])),
            "resistance_count": len(self._major_sr_1h.get("resistances", [])),
        }
        signal.metadata = meta
        return signal
