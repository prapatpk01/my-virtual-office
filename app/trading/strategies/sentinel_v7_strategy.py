"""Sentinel V7 — RSI Rotation + 5M Price Trigger.

Design goal: borrow EMA Hybrid's clean separation of setup vs execution without
copying its EMA trigger.

- 1H = direction only: EMA20/EMA50 + EMA20 slope.
- 15M = market gate + RSI rotation ARM:
    ADX >= 12, CHOP < 65, ATR activity >= 0.65.
    LONG arm: RSI14 crosses above SMA14(RSI), both < 55, positive RSI/SMA
    slopes and expanding positive RSI-SMA spread.
    SHORT arm: mirror, both >= 65.
    Classic RSI divergence remains a soft confidence bonus only.
- ARM is valid for the next 3 CLOSED 5M bars (15 minutes).
- 5M = actual execution trigger, no EMA cross:
    LONG: bullish close above the highs of the previous 2 closed 5M bars.
    SHORT: bearish close below the lows of the previous 2 closed 5M bars.
- SL = recent 5M structure +/- 0.20 ATR5M.
    Minimum distance 0.80 ATR5M; structure risk > 1.60 ATR5M blocks the trade
    rather than tightening the stop inside invalidation.
- TP1 = +1.00R, close 50%, runner SL -> +0.15R.
- TP2 = +2.00R, close remaining 50%.
- Exit remains deliberately slower than entry: closed-15M RSI reversal logic
  inherited from V6 (neutral confirmation before TP1; opposite cross after TP1).
"""
from __future__ import annotations

from .base import Signal, SignalType
from .sentinel_v61_strategy import SentinelV61Strategy
from ..engines.position_manager import PositionUpdate


class SentinelV7Strategy(SentinelV61Strategy):
    VERSION = "7.0"
    entry_tf = "5m"  # tells TradingBot to fetch closed 5M candles for execution

    ARM_BARS_5M = 3
    FIVE_MIN_MS = 5 * 60_000
    FIFTEEN_MIN_MS = 15 * 60_000
    MIN_5M_BARS = 30

    STRUCTURE_LOOKBACK_5M = 6
    SL_BUFFER_ATR = 0.20
    MIN_SL_ATR = 0.80
    MAX_SL_ATR = 1.60

    TP1_R = 1.00
    TP1_CLOSE_PCT = 0.50
    TP1_LOCK_R = 0.15
    TP2_R = 2.00

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV7({symbol})"
        self.tp1_r = self.TP1_R
        self.tp1_trim_pct = self.TP1_CLOSE_PCT
        self.target_r = self.TP2_R

        self._arm: dict | None = None
        self._last_5m_evaluated_ts: int | None = None

    def _clear_arm(self) -> None:
        self._arm = None

    def _arm_view(self) -> dict:
        if not self._arm:
            return {"active": False, "side": None}
        return {
            "active": True,
            "side": self._arm.get("side"),
            "armed_15m_ts": self._arm.get("armed_15m_ts"),
            "ready_close_ts": self._arm.get("ready_close_ts"),
            "expires_close_ts": self._arm.get("expires_close_ts"),
            "divergence_bonus": bool(self._arm.get("divergence_bonus")),
        }

    def _try_arm(self, trend: dict, snap: dict, bar_ts: int) -> tuple[bool, str, list[str]]:
        """Create a fresh 15M RSI rotation arm; never submits an order here."""
        direction = trend.get("direction")
        blocks: list[str] = []

        if not snap.get("ready"):
            return False, snap.get("reason", "15M unavailable"), list(snap.get("blocks", []))
        if not snap.get("market_ready"):
            return False, snap.get("reason", "15M market gate blocked"), list(snap.get("blocks", []))
        if direction not in {"long", "short"}:
            return False, "waiting for clear 1H direction", ["TREND_1H"]

        if direction == "long":
            raw_cross = bool(snap.get("cross_up"))
            zone_ok = bool(snap.get("long_zone"))
            slope_ok = bool(snap.get("slope_long_ok"))
            spread_ok = bool(snap.get("spread_long_ok"))
            div_bonus = bool((snap.get("divergence") or {}).get("bullish"))
            if raw_cross and not zone_ok:
                blocks.append("RSI_ZONE")
            if raw_cross and not slope_ok:
                blocks.append("RSI_SLOPE")
            if raw_cross and not spread_ok:
                blocks.append("RSI_SPREAD")
            qualifies = raw_cross and zone_ok and slope_ok and spread_ok
        else:
            raw_cross = bool(snap.get("cross_down"))
            zone_ok = bool(snap.get("short_zone"))
            slope_ok = bool(snap.get("slope_short_ok"))
            spread_ok = bool(snap.get("spread_short_ok"))
            div_bonus = bool((snap.get("divergence") or {}).get("bearish"))
            if raw_cross and not zone_ok:
                blocks.append("RSI_ZONE")
            if raw_cross and not slope_ok:
                blocks.append("RSI_SLOPE")
            if raw_cross and not spread_ok:
                blocks.append("RSI_SPREAD")
            qualifies = raw_cross and zone_ok and slope_ok and spread_ok

        if not qualifies:
            if raw_cross and blocks:
                return False, "RSI cross found but ARM quality failed", blocks
            return False, f"waiting for {direction.upper()} 15M RSI rotation", []

        # bar_ts is the 15M candle OPEN timestamp. The RSI signal only exists
        # once that candle has closed, so 5M execution starts strictly after
        # ready_close_ts and expires after the next three closed 5M bars.
        ready_close_ts = int(bar_ts + self.FIFTEEN_MIN_MS)
        self._arm = {
            "side": direction,
            "armed_15m_ts": int(bar_ts),
            "ready_close_ts": ready_close_ts,
            "expires_close_ts": ready_close_ts + self.ARM_BARS_5M * self.FIVE_MIN_MS,
            "divergence_bonus": div_bonus,
        }
        return True, f"{direction.upper()} RSI rotation ARMED for next {self.ARM_BARS_5M}x5M bars", []

    def _snapshot_5m_trigger(self, candles: list, current_price: float) -> dict:
        if len(candles) < self.MIN_5M_BARS:
            return {"ready": False, "trigger": False, "reason": "5M warmup", "blocks": ["5M_WARMUP"]}
        if not self._arm:
            return {"ready": True, "trigger": False, "reason": "no active ARM", "blocks": []}

        bar = candles[-1]
        bar_ts = int(self._bar_ts(bar))
        bar_close_ts = bar_ts + self.FIVE_MIN_MS
        side = str(self._arm.get("side"))
        ready_close = int(self._arm.get("ready_close_ts") or 0)
        expires_close = int(self._arm.get("expires_close_ts") or 0)

        if bar_close_ts <= ready_close:
            return {
                "ready": True, "trigger": False, "side": side,
                "bar_ts": bar_ts, "bar_close_ts": bar_close_ts,
                "reason": "waiting for first 5M bar after ARM", "blocks": [],
            }
        if bar_close_ts > expires_close:
            return {
                "ready": True, "trigger": False, "side": side,
                "bar_ts": bar_ts, "bar_close_ts": bar_close_ts,
                "expired": True, "reason": "ARM expired after 3x5M bars", "blocks": ["ARM_EXPIRED"],
            }

        atr_arr = self.atr(candles, 14)
        if not self._finite(atr_arr[-1]):
            return {"ready": False, "trigger": False, "reason": "5M ATR unavailable", "blocks": ["5M_ATR"]}
        atr5 = max(float(atr_arr[-1]), 1e-12)

        prev2 = candles[-3:-1]
        close = float(bar.close)
        open_ = float(bar.open)
        high2 = max(float(c.high) for c in prev2)
        low2 = min(float(c.low) for c in prev2)
        bullish = close > open_
        bearish = close < open_
        micro_break_up = bullish and close > high2
        micro_break_down = bearish and close < low2
        trigger = micro_break_up if side == "long" else micro_break_down

        out = {
            "ready": True,
            "trigger": bool(trigger),
            "side": side,
            "bar_ts": bar_ts,
            "bar_close_ts": bar_close_ts,
            "close": round(close, 8),
            "prev2_high": round(high2, 8),
            "prev2_low": round(low2, 8),
            "candle": "BULL" if bullish else "BEAR" if bearish else "DOJI",
            "atr5": float(atr5),
            "reason": "5M micro price breakout confirmed" if trigger else f"waiting for 5M {side.upper()} micro breakout",
            "blocks": [],
        }

        if not trigger:
            return out

        recent = candles[-self.STRUCTURE_LOOKBACK_5M:]
        entry = float(current_price)
        if side == "long":
            structure = min(float(c.low) for c in recent)
            raw_stop = structure - self.SL_BUFFER_ATR * atr5
            raw_risk = entry - raw_stop
        else:
            structure = max(float(c.high) for c in recent)
            raw_stop = structure + self.SL_BUFFER_ATR * atr5
            raw_risk = raw_stop - entry

        min_risk = self.MIN_SL_ATR * atr5
        max_risk = self.MAX_SL_ATR * atr5
        if raw_risk <= 0:
            out.update({"trigger": False, "reason": "invalid 5M structure stop", "blocks": ["SL_STRUCTURE"]})
            return out
        if raw_risk > max_risk:
            out.update({
                "trigger": False,
                "reason": f"5M structure stop too wide ({raw_risk/atr5:.2f}ATR > {self.MAX_SL_ATR:.2f}ATR)",
                "blocks": ["SL_TOO_WIDE"],
                "structure": round(structure, 8),
                "raw_sl_atr": round(raw_risk / atr5, 2),
            })
            return out

        risk = max(raw_risk, min_risk)
        stop = entry - risk if side == "long" else entry + risk
        tp1 = entry + self.TP1_R * risk if side == "long" else entry - self.TP1_R * risk
        tp2 = entry + self.TP2_R * risk if side == "long" else entry - self.TP2_R * risk

        out.update({
            "entry": entry,
            "structure": round(structure, 8),
            "raw_stop": round(raw_stop, 8),
            "raw_sl_atr": round(raw_risk / atr5, 2),
            "sl_atr": round(risk / atr5, 2),
            "risk": float(risk),
            "stop_loss": float(stop),
            "tp1_price": float(tp1),
            "take_profit": float(tp2),
        })
        return out

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c15 = self._closed_candle_series(candles, self.FIFTEEN_MIN_MS)
        c1h = self._closed_candle_series(mtf.get("1h", []), 60 * 60_000)
        c5 = self._closed_candle_series(mtf.get("5m", []), self.FIVE_MIN_MS)
        self._latest_15m = c15

        meta = {
            "strategy": "SENTINEL_V7",
            "version": self.VERSION,
            "architecture": "1H_DIRECTION__15M_RSI_ARM__5M_PRICE_TRIGGER",
            "entry_tf": "5m_closed",
            "mtf_used": "1h_direction__15m_setup__5m_execution",
            "risk_plan": "5M_STRUCTURE+0.20ATR_MIN0.80_MAX1.60__TP1_1R_CLOSE50_LOCK+0.15R__TP2_2R",
        }

        if len(c15) < self.MIN_15M_BARS or len(c1h) < self.MIN_1H_BARS or len(c5) < self.MIN_5M_BARS:
            return self._hold(float(current_price), "waiting for closed 1H/15M/5M warmup", meta)

        trend = self._trend_1h(c1h)
        snap = self._snapshot_15m(c15)
        meta["trend_1h"] = trend
        meta["market_15m"] = snap

        if self._open_position is not None:
            self._clear_arm()
            meta["arm"] = self._arm_view()
            return self._hold(float(current_price), f"managing open {self._open_position} position", meta)

        bar15_ts = int(self._bar_ts(c15[-1]))

        # Exit cooldown remains based on completed 15M bars so an early exit
        # cannot immediately re-arm on the same RSI oscillation.
        if self._last_exit_bar_ts is not None:
            elapsed = bar15_ts - self._last_exit_bar_ts
            if elapsed < self.exit_cooldown_bars * self.FIFTEEN_MIN_MS:
                self._clear_arm()
                meta["arm"] = self._arm_view()
                return self._hold(float(current_price), "post-exit cooldown", meta)

        arm_event = None
        arm_blocks: list[str] = []
        # RSI ARM can be created only once per newly closed 15M bar.
        if self._last_evaluated_bar_ts != bar15_ts:
            self._last_evaluated_bar_ts = bar15_ts
            armed, arm_event, arm_blocks = self._try_arm(trend, snap, bar15_ts)
            # A new opposite 15M direction/rotation invalidates any stale arm.
            if not armed and self._arm and self._arm.get("side") != trend.get("direction"):
                self._clear_arm()

        meta["arm"] = self._arm_view()
        meta["arm_event"] = arm_event
        meta["arm_blocks"] = arm_blocks

        if not self._arm:
            reason = arm_event or f"waiting for 15M RSI rotation | 1H={trend.get('direction') or 'NEUTRAL'}"
            return self._hold(float(current_price), reason, meta)

        # Direction is checked again while armed; 1H may flip before execution.
        if trend.get("direction") != self._arm.get("side"):
            self._clear_arm()
            meta["arm"] = self._arm_view()
            return self._hold(float(current_price), "ARM cancelled: 1H direction changed", meta)
        if not snap.get("market_ready"):
            self._clear_arm()
            meta["arm"] = self._arm_view()
            return self._hold(float(current_price), "ARM cancelled: 15M market gate no longer valid", meta)

        trigger5 = self._snapshot_5m_trigger(c5, float(current_price))
        meta["trigger_5m"] = trigger5

        if trigger5.get("expired"):
            self._clear_arm()
            meta["arm"] = self._arm_view()
            return self._hold(float(current_price), "ARM expired: no 5M price confirmation within 15 minutes", meta)

        bar5_ts = trigger5.get("bar_ts")
        if bar5_ts is not None and self._last_5m_evaluated_ts == int(bar5_ts):
            return self._hold(float(current_price), "5M bar already evaluated while ARMED", meta)
        if bar5_ts is not None:
            self._last_5m_evaluated_ts = int(bar5_ts)

        if not trigger5.get("trigger"):
            return self._hold(float(current_price), trigger5.get("reason", "waiting for 5M price trigger"), meta)

        side = str(self._arm.get("side"))
        divergence_bonus = bool(self._arm.get("divergence_bonus"))
        entry = float(trigger5["entry"])
        stop = float(trigger5["stop_loss"])
        target = float(trigger5["take_profit"])
        risk = float(trigger5["risk"])

        self._open_position = side
        self._pending_entry = True
        self._entry_price = entry
        self._entry_sl = stop
        self._entry_tp = target
        self._initial_risk = risk
        self._tp1_done = False

        signal_type = SignalType.BUY if side == "long" else SignalType.SELL
        confidence = 0.82 + (self.DIVERGENCE_CONF_BONUS if divergence_bonus else 0.0)
        confidence = min(0.95, confidence)

        meta.update({
            "direction": side,
            "entry_trigger": f"5M_MICRO_BREAK_{'UP' if side == 'long' else 'DOWN'}",
            "stop_loss": round(stop, 8),
            "take_profit": round(target, 8),
            "tp1_price": round(float(trigger5["tp1_price"]), 8),
            "rr_ratio": self.TP2_R,
            "tp1_r": self.TP1_R,
            "tp1_close_pct": self.TP1_CLOSE_PCT,
            "tp1_lock_r": self.TP1_LOCK_R,
            "divergence_soft_bonus": divergence_bonus,
        })

        reason = (
            f"{side.upper()} RSI ROTATION -> 5M PRICE BREAK | 1H={trend.get('direction')} "
            f"RSI={snap.get('rsi')}/{snap.get('rsi_sma')} spreadΔ={snap.get('spread_delta')} "
            f"ADX={snap.get('adx')} CHOP={snap.get('chop')} | "
            f"SL={trigger5.get('sl_atr')}ATR5 structure+{self.SL_BUFFER_ATR:.2f}ATR"
            + (" | divergence bonus" if divergence_bonus else "")
        )
        self._clear_arm()
        meta["arm"] = self._arm_view()
        return Signal(signal_type, self.symbol, entry, 0.0, reason, confidence, meta)

    def tick_open_position(self, current_price: float, position_key: str | None = None):
        update = super().tick_open_position(current_price, position_key=position_key)
        if update is not None and getattr(update, "action", "") == "hold":
            return PositionUpdate(
                action="hold",
                reason=(
                    f"Holding {str(self._open_position or '').upper()} — 5M structure SL | "
                    f"TP1 1.0R/50% -> runner SL +{self.TP1_LOCK_R:.2f}R | TP2 2.0R"
                ),
            )
        return update
