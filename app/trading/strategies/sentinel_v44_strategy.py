"""Sentinel V4.4 — Position Defense.

Refines V4.3 without changing its entry philosophy:
- 15M remains the only entry/direction engine.
- 1H remains major S/R obstacle only.
- Minimum initial stop is 1.0 ATR; maximum structure stop remains 1.8 ATR.
- Pullback/sweep structure buffers widen from 0.08 -> 0.12 ATR.
- Breakout invalidation buffer widens from 0.22 -> 0.25 ATR.
- Existing V4.3 technical exits stay unchanged (no EMA8/13 cross exit).
- After a HARD stop-loss, same-side re-entry is locked for at least three
  closed 15M bars and requires a genuinely fresh structure reset.
"""
from __future__ import annotations

from .sentinel_v43_strategy import SentinelV43Strategy


class SentinelV44Strategy(SentinelV43Strategy):
    VERSION = "4.4"

    MIN_STOP_ATR = 1.00
    MAX_STOP_ATR = 1.80
    SL_REARM_BARS = 3

    def __init__(self, symbol: str, **kwargs):
        kwargs["stop_atr_min"] = self.MIN_STOP_ATR
        kwargs["stop_atr_max"] = self.MAX_STOP_ATR
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV4.4({symbol})"
        self._sl_rearm_side: str | None = None
        self._sl_rearm_exit_ts: int | None = None

    # ------------------------------------------------------------------
    # Wider structure invalidation buffers
    # ------------------------------------------------------------------
    @staticmethod
    def _widen_candidates(candidates: list[dict], extra_atr: float, atr_now: float) -> list[dict]:
        for item in candidates:
            direction = str(item.get("direction") or "")
            if "invalidation" not in item:
                continue
            if direction == "long":
                item["invalidation"] = float(item["invalidation"]) - extra_atr * atr_now
            elif direction == "short":
                item["invalidation"] = float(item["invalidation"]) + extra_atr * atr_now
        return candidates

    def _pullback_setup(self, candles, structure, atr_now, ema20, ema50, hma16):
        # V4.2 uses 0.08 ATR; add 0.04 -> 0.12 ATR total.
        candidates = super()._pullback_setup(candles, structure, atr_now, ema20, ema50, hma16)
        return self._widen_candidates(candidates, 0.04, atr_now)

    def _breakout_retest_setup(self, candles, structure, atr_now):
        # V4.2 uses 0.22 ATR; add 0.03 -> 0.25 ATR total.
        candidates = super()._breakout_retest_setup(candles, structure, atr_now)
        return self._widen_candidates(candidates, 0.03, atr_now)

    def _sweep_reversal_setup(self, candles, structure, atr_now, hma16):
        # V4.2 uses 0.08 ATR; add 0.04 -> 0.12 ATR total.
        candidates = super()._sweep_reversal_setup(candles, structure, atr_now, hma16)
        return self._widen_candidates(candidates, 0.04, atr_now)

    # ------------------------------------------------------------------
    # Hard-SL re-arm logic
    # ------------------------------------------------------------------
    def _fresh_rearm_structure(self, candles: list, direction: str, entry: dict) -> bool:
        """Require a fresh event after the hard SL before same-side re-entry.

        BOS/retest/sweep are intrinsically fresh structure events. A pullback
        continuation is allowed only after a new LH (short) / HL (long) pivot
        has actually formed after the SL, preventing repeated entries off the
        same failed swing.
        """
        exit_ts = self._sl_rearm_exit_ts
        if exit_ts is None:
            return True

        event_ts = int(entry.get("event_ts") or 0)
        setup = str(entry.get("candidate_trigger") or entry.get("trigger") or "")

        if setup in {"BREAKOUT_BOS", "BREAKOUT_RETEST", "SWEEP_STRUCTURE_REVERSAL"}:
            return event_ts > exit_ts

        recent = candles[-80:]
        highs2, lows2 = self._pivot_points(recent, span=2)
        if direction == "short" and len(highs2) >= 2:
            prev_ts, prev_high = highs2[-2]
            last_ts, last_high = highs2[-1]
            return int(last_ts) > exit_ts and float(last_high) < float(prev_high)
        if direction == "long" and len(lows2) >= 2:
            prev_ts, prev_low = lows2[-2]
            last_ts, last_low = lows2[-1]
            return int(last_ts) > exit_ts and float(last_low) > float(prev_low)
        return False

    def _apply_sl_rearm(self, candles: list, entry: dict, blocks: list[str]) -> tuple[list[str], dict]:
        direction = str(entry.get("direction") or "")
        if direction not in {"long", "short"} or self._sl_rearm_side != direction or self._sl_rearm_exit_ts is None:
            entry["sl_rearm"] = {"active": False}
            return blocks, entry

        bar_ts = self._bar_ts(candles[-1])
        bars_since = max(0, int((bar_ts - self._sl_rearm_exit_ts) // (15 * 60_000)))
        fresh_reset = self._fresh_rearm_structure(candles, direction, entry)

        if bars_since < self.SL_REARM_BARS:
            if "SL_REARM_WAIT" not in blocks:
                blocks.append("SL_REARM_WAIT")
        elif not fresh_reset:
            if "SL_REARM_SETUP" not in blocks:
                blocks.append("SL_REARM_SETUP")

        entry["sl_rearm"] = {
            "active": True,
            "side": direction,
            "bars_since_sl": bars_since,
            "min_bars": self.SL_REARM_BARS,
            "fresh_reset": bool(fresh_reset),
        }
        return blocks, entry

    # ------------------------------------------------------------------
    # Risk / target rebuild with 1.0 ATR minimum stop
    # ------------------------------------------------------------------
    def _build_entry(self, candles, current_price, market, structure):
        entry = super()._build_entry(candles, current_price, market, structure)
        direction = str(entry.get("direction") or "")
        if direction not in {"long", "short"} or "entry" not in entry or "risk" not in entry:
            return entry

        px = float(entry["entry"])
        atr_now = max(float(self.atr(candles, 14)[-1]), 1e-12)

        # Never tighten a structure stop. Only widen it to the 1.0 ATR floor.
        risk = max(float(entry.get("risk") or 0.0), self.MIN_STOP_ATR * atr_now, px * 0.001)
        stop = px - risk if direction == "long" else px + risk
        entry["risk"] = float(risk)
        entry["stop_loss"] = float(stop)
        entry["sl_atr"] = round(float(risk / atr_now), 2)
        entry["sl_policy"] = "STRUCTURE__MIN_1.0ATR__MAX_1.8ATR"

        blocks = [
            b for b in list(entry.get("blocks", []))
            if b not in {"ROOM", "MAJOR_1H_SR", "TARGET_TOO_CLOSE"}
        ]

        # Recompute room after widening the stop. This is essential: a wider
        # protective stop must not silently turn a 1.5R trade into a sub-1.5R trade.
        resistance_15m = self._nearest_above(px, structure.get("resistances", []))
        support_15m = self._nearest_below(px, structure.get("supports", []))
        obstacle_15m = resistance_15m if direction == "long" else support_15m
        room_15m = abs(float(obstacle_15m) - px) / risk if obstacle_15m is not None else 2.50

        major = self._major_sr_1h or {}
        resistance_1h = self._nearest_above(px, major.get("resistances", []))
        support_1h = self._nearest_below(px, major.get("supports", []))
        obstacle_1h = resistance_1h if direction == "long" else support_1h
        room_1h = abs(float(obstacle_1h) - px) / risk if obstacle_1h is not None else None

        if room_15m + 1e-9 < 1.50:
            blocks.append("ROOM")
        if room_1h is not None and room_1h + 1e-9 < 1.50:
            blocks.append("MAJOR_1H_SR")

        effective_room = min(room_15m, room_1h) if room_1h is not None else room_15m
        entry["room_15m_r"] = round(float(room_15m), 2)
        entry["room_1h_r"] = round(float(room_1h), 2) if room_1h is not None else None
        entry["room_r"] = round(float(effective_room), 2)

        # Start at the normal 2R target, then respect whichever valid 15M/1H
        # obstacle is closer. A genuine 1.50R obstacle remains tradable.
        fixed_target = px + self.TARGET_R * risk if direction == "long" else px - self.TARGET_R * risk
        best_target = fixed_target
        best_rr = self.TARGET_R
        best_source = f"{self.TARGET_R:.1f}R"

        def consider_obstacle(obstacle, room_r, buffer, source):
            nonlocal best_target, best_rr, best_source
            if obstacle is None or room_r is None or room_r + 1e-9 < 1.50:
                return
            buffered = float(obstacle) - buffer if direction == "long" else float(obstacle) + buffer
            buffered_rr = abs(buffered - px) / risk
            if buffered_rr + 1e-9 < 1.50:
                candidate_rr = 1.50
                candidate_target = px + 1.50 * risk if direction == "long" else px - 1.50 * risk
                candidate_source = source + "_1.5R_FLOOR"
            else:
                candidate_rr = min(float(buffered_rr), self.TARGET_R)
                candidate_target = px + candidate_rr * risk if direction == "long" else px - candidate_rr * risk
                candidate_source = source
            if candidate_rr + 1e-9 < best_rr:
                best_target = float(candidate_target)
                best_rr = float(candidate_rr)
                best_source = candidate_source

        consider_obstacle(obstacle_15m, room_15m, 0.05 * atr_now, "NEAREST_S/R")
        atr_1h = max(float(major.get("atr") or 0.0), 0.0)
        consider_obstacle(obstacle_1h, room_1h, self.H1_TARGET_BUFFER_ATR * atr_1h, "1H_MAJOR_S/R")

        entry["take_profit"] = float(best_target)
        entry["target_rr"] = round(float(best_rr), 2)
        entry["target_source"] = best_source
        if best_rr + 1e-9 < 1.50:
            blocks.append("TARGET_TOO_CLOSE")

        major_meta = dict(entry.get("major_sr_1h") or {})
        major_meta.update({
            "nearest_support": support_1h,
            "nearest_resistance": resistance_1h,
            "major_obstacle": obstacle_1h,
            "room_r": round(float(room_1h), 2) if room_1h is not None else None,
            "used_for_target": best_source.startswith("1H_MAJOR_S/R"),
            "target_source": best_source,
        })
        entry["major_sr_1h"] = major_meta

        blocks, entry = self._apply_sl_rearm(candles, entry, blocks)
        # Keep order stable and avoid duplicate diagnostic labels.
        entry["blocks"] = list(dict.fromkeys(blocks))
        if entry["blocks"]:
            entry["trigger"] = None
            entry["reason"] = "blocked: " + ",".join(entry["blocks"])
        return entry

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)
        meta = signal.metadata or {}
        meta["version"] = self.VERSION
        meta["position_defense"] = {
            "min_stop_atr": self.MIN_STOP_ATR,
            "max_stop_atr": self.MAX_STOP_ATR,
            "sl_rearm_bars": self.SL_REARM_BARS,
            "sl_rearm_side": self._sl_rearm_side,
        }
        signal.metadata = meta

        # A successfully emitted re-armed trade consumes the hard-SL lock.
        if getattr(getattr(signal, "type", None), "value", "hold") != "hold":
            direction = "long" if signal.type.value == "buy" else "short"
            if self._sl_rearm_side == direction:
                self._sl_rearm_side = None
                self._sl_rearm_exit_ts = None
        return signal

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        side = self._open_position
        exit_ts = self._bar_ts(self._latest_15m[-1]) if self._latest_15m else None
        text = str(reason or "").lower()
        hard_stop = any(token in text for token in ("stop_loss", "stop loss", "stop-loss", "hard_sl"))
        # A profit-protecting stop after T1 is not treated as a failed setup.
        hard_stop = bool(hard_stop and not self._tp1_done)

        super().record_closed_trade(exit_price, reason, duration_min)

        if hard_stop and side in {"long", "short"} and exit_ts is not None:
            self._sl_rearm_side = str(side)
            self._sl_rearm_exit_ts = int(exit_ts)
