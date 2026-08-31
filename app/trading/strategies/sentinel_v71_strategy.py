"""Sentinel V7.1 — rotation-memory + low-latency execution refinement.

Keeps V7 architecture intact and fixes the bottlenecks observed in PAPER logs:
- 1H remains direction only (EMA20/EMA50 + EMA20 slope).
- 15M market gate remains ADX/CHOP/ATR.
- RSI setup becomes a true rotation-memory setup instead of requiring the
  fresh cross to happen at an extreme RSI level.
    LONG: recent 4-bar RSI min <=45, fresh cross UP, current RSI <=55,
          RSI slope >0, positive cross spread magnitude >=0.30.
    SHORT: recent 4-bar RSI max >=55, fresh cross DOWN, current RSI >=45,
           RSI slope <0, negative cross spread magnitude >=0.30.
- SMA slope, spread acceleration and classic RSI divergence are context only;
  they never veto an otherwise valid rotation.
- ARM lasts 4 CLOSED 5M bars (20 minutes), and an opposite 15M RSI cross
  cancels a stale arm immediately.
- 5M execution is still the V7 micro price breakout, but entries that have
  already moved >0.25 ATR5 away from the trigger close are skipped as chase.
- Structure SL and 2TP remain unchanged:
  structure +/-0.20 ATR5, min 0.80 ATR, max 1.60 ATR;
  TP1 +1R close 50% -> runner SL +0.15R; TP2 +2R.
"""
from __future__ import annotations

from .sentinel_v7_strategy import SentinelV7Strategy


class SentinelV71Strategy(SentinelV7Strategy):
    VERSION = "7.1"

    ARM_BARS_5M = 4
    ROTATION_LOOKBACK = 4
    LONG_MEMORY_LEVEL = 45.0
    SHORT_MEMORY_LEVEL = 55.0
    LONG_CURRENT_MAX = 55.0
    SHORT_CURRENT_MIN = 45.0
    MIN_CROSS_SPREAD = 0.30
    MAX_TRIGGER_CHASE_ATR = 0.25

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV7.1({symbol})"
        self._last_arm_decision: dict = {}

    # ------------------------------------------------------------------
    # 15M RSI rotation memory
    # ------------------------------------------------------------------
    def _snapshot_15m(self, candles: list) -> dict:
        snap = super()._snapshot_15m(candles)
        if not snap.get("ready"):
            return snap

        closes = [float(c.close) for c in candles]
        rsi = self.rsi(closes, self.RSI_PERIOD)
        recent = [
            float(v) for v in rsi[-self.ROTATION_LOOKBACK:]
            if self._finite(v)
        ]
        if not recent:
            return snap

        current_rsi = float(snap.get("rsi"))
        spread = float(snap.get("spread", 0.0))
        rsi_slope = float(snap.get("rsi_slope", 0.0))
        sma_slope = float(snap.get("rsi_sma_slope", 0.0))
        spread_delta = float(snap.get("spread_delta", 0.0))

        recent_min = min(recent)
        recent_max = max(recent)
        long_memory_ok = recent_min <= self.LONG_MEMORY_LEVEL
        short_memory_ok = recent_max >= self.SHORT_MEMORY_LEVEL
        long_current_ok = current_rsi <= self.LONG_CURRENT_MAX
        short_current_ok = current_rsi >= self.SHORT_CURRENT_MIN
        long_slope_ok = rsi_slope > 0.0
        short_slope_ok = rsi_slope < 0.0
        long_spread_ok = spread >= self.MIN_CROSS_SPREAD
        short_spread_ok = spread <= -self.MIN_CROSS_SPREAD

        # Context-only fields: useful for analysis/logging but never hard gates.
        sma_context = (
            "LONG_ALIGNED" if sma_slope >= 0.0
            else "SHORT_ALIGNED"
        )
        spread_accel_context = (
            "LONG_ALIGNED" if spread_delta > 0.0
            else "SHORT_ALIGNED" if spread_delta < 0.0
            else "FLAT"
        )

        snap.update({
            "rotation_recent_min": round(recent_min, 2),
            "rotation_recent_max": round(recent_max, 2),
            "rotation_long_memory_ok": bool(long_memory_ok),
            "rotation_short_memory_ok": bool(short_memory_ok),
            "rotation_long_current_ok": bool(long_current_ok),
            "rotation_short_current_ok": bool(short_current_ok),
            "rotation_long_slope_ok": bool(long_slope_ok),
            "rotation_short_slope_ok": bool(short_slope_ok),
            "rotation_long_spread_ok": bool(long_spread_ok),
            "rotation_short_spread_ok": bool(short_spread_ok),
            "sma_slope_context": sma_context,
            "spread_accel_context": spread_accel_context,
            "min_cross_spread": self.MIN_CROSS_SPREAD,
        })
        return snap

    def _remember_arm_decision(self, *, bar_ts: int, decision: str,
                               reason: str, blocks: list[str], side: str | None = None) -> None:
        self._last_arm_decision = {
            "bar_ts": int(bar_ts),
            "decision": str(decision),
            "side": side,
            "reason": str(reason),
            "blocks": list(blocks or []),
        }

    def _try_arm(self, trend: dict, snap: dict, bar_ts: int) -> tuple[bool, str, list[str]]:
        direction = trend.get("direction")

        if not snap.get("ready"):
            blocks = list(snap.get("blocks", []))
            reason = snap.get("reason", "15M unavailable")
            self._remember_arm_decision(
                bar_ts=bar_ts, decision="REJECTED", reason=reason,
                blocks=blocks, side=direction,
            )
            return False, reason, blocks

        if not snap.get("market_ready"):
            blocks = list(snap.get("blocks", []))
            reason = snap.get("reason", "15M market gate blocked")
            self._remember_arm_decision(
                bar_ts=bar_ts, decision="REJECTED", reason=reason,
                blocks=blocks, side=direction,
            )
            return False, reason, blocks

        # A fresh opposite RSI cross invalidates a still-pending setup before
        # the 5M executor is allowed to act on it.
        if self._arm:
            arm_side = str(self._arm.get("side") or "")
            opposite = (
                arm_side == "long" and bool(snap.get("cross_down"))
            ) or (
                arm_side == "short" and bool(snap.get("cross_up"))
            )
            if opposite:
                self._clear_arm()
                reason = "ARM cancelled: opposite 15M RSI cross"
                blocks = ["OPPOSITE_RSI"]
                self._remember_arm_decision(
                    bar_ts=bar_ts, decision="CANCELLED", reason=reason,
                    blocks=blocks, side=arm_side,
                )
                return False, reason, blocks

        if direction not in {"long", "short"}:
            reason = "waiting for clear 1H direction"
            blocks = ["TREND_1H"]
            self._remember_arm_decision(
                bar_ts=bar_ts, decision="REJECTED", reason=reason,
                blocks=blocks, side=None,
            )
            return False, reason, blocks

        raw_cross = bool(snap.get("cross_up")) if direction == "long" else bool(snap.get("cross_down"))
        if not raw_cross:
            reason = f"waiting for {direction.upper()} 15M RSI rotation"
            self._remember_arm_decision(
                bar_ts=bar_ts, decision="WAIT", reason=reason,
                blocks=[], side=direction,
            )
            return False, reason, []

        blocks: list[str] = []
        if direction == "long":
            if not snap.get("rotation_long_memory_ok"):
                blocks.append("RSI_MEMORY")
            if not snap.get("rotation_long_current_ok"):
                blocks.append("RSI_CURRENT_ZONE")
            if not snap.get("rotation_long_slope_ok"):
                blocks.append("RSI_SLOPE")
            if not snap.get("rotation_long_spread_ok"):
                blocks.append("RSI_SPREAD_MAG")
            div_bonus = bool((snap.get("divergence") or {}).get("bullish"))
        else:
            if not snap.get("rotation_short_memory_ok"):
                blocks.append("RSI_MEMORY")
            if not snap.get("rotation_short_current_ok"):
                blocks.append("RSI_CURRENT_ZONE")
            if not snap.get("rotation_short_slope_ok"):
                blocks.append("RSI_SLOPE")
            if not snap.get("rotation_short_spread_ok"):
                blocks.append("RSI_SPREAD_MAG")
            div_bonus = bool((snap.get("divergence") or {}).get("bearish"))

        if blocks:
            reason = "RSI cross found but rotation quality failed"
            self._remember_arm_decision(
                bar_ts=bar_ts, decision="REJECTED", reason=reason,
                blocks=blocks, side=direction,
            )
            return False, reason, blocks

        ready_close_ts = int(bar_ts + self.FIFTEEN_MIN_MS)
        self._arm = {
            "side": direction,
            "armed_15m_ts": int(bar_ts),
            "ready_close_ts": ready_close_ts,
            "expires_close_ts": ready_close_ts + self.ARM_BARS_5M * self.FIVE_MIN_MS,
            "divergence_bonus": div_bonus,
            "sma_slope_context": snap.get("sma_slope_context"),
            "spread_accel_context": snap.get("spread_accel_context"),
        }
        reason = f"{direction.upper()} RSI rotation ARMED for next {self.ARM_BARS_5M}x5M bars"
        self._remember_arm_decision(
            bar_ts=bar_ts, decision="ARMED", reason=reason,
            blocks=[], side=direction,
        )
        return True, reason, []

    # ------------------------------------------------------------------
    # 5M execution freshness / anti-chase
    # ------------------------------------------------------------------
    def _snapshot_5m_trigger(self, candles: list, current_price: float) -> dict:
        out = super()._snapshot_5m_trigger(candles, current_price)

        if out.get("expired"):
            out["reason"] = f"ARM expired after {self.ARM_BARS_5M}x5M bars"
            return out

        if not out.get("trigger"):
            return out

        atr5 = max(float(out.get("atr5") or 0.0), 1e-12)
        trigger_close = float(out.get("close") or current_price)
        side = str(out.get("side") or "")
        if side == "long":
            adverse_distance = max(0.0, float(current_price) - trigger_close)
        else:
            adverse_distance = max(0.0, trigger_close - float(current_price))
        chase_atr = adverse_distance / atr5
        out["chase_atr"] = round(chase_atr, 3)
        out["max_chase_atr"] = self.MAX_TRIGGER_CHASE_ATR

        if chase_atr > self.MAX_TRIGGER_CHASE_ATR:
            out["trigger"] = False
            out["blocks"] = list(dict.fromkeys(list(out.get("blocks", [])) + ["ANTI_CHASE"]))
            out["reason"] = (
                f"5M breakout skipped: entry moved {chase_atr:.2f}ATR from trigger close "
                f"> {self.MAX_TRIGGER_CHASE_ATR:.2f}ATR"
            )
        return out

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)
        meta = signal.metadata or {}
        meta["strategy"] = "SENTINEL_V7_1"
        meta["version"] = self.VERSION
        meta["architecture"] = "1H_DIRECTION__15M_RSI_ROTATION_MEMORY__5M_PRICE_TRIGGER"
        meta["arm_bars_5m"] = self.ARM_BARS_5M
        meta["max_trigger_chase_atr"] = self.MAX_TRIGGER_CHASE_ATR
        meta["arm_decision_cached"] = dict(self._last_arm_decision)

        market = meta.get("market_15m") or {}
        trend = meta.get("trend_1h") or {}
        divergence = market.get("divergence") or {}
        direction = str(trend.get("direction") or "")
        counter_div = (
            direction == "long" and bool(divergence.get("bearish"))
        ) or (
            direction == "short" and bool(divergence.get("bullish"))
        )
        meta["counter_divergence_soft"] = bool(counter_div)

        if "within 15 minutes" in str(signal.reason):
            signal.reason = str(signal.reason).replace("within 15 minutes", "within 20 minutes")

        signal.metadata = meta
        return signal
