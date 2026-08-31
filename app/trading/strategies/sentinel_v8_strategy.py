"""Sentinel V8 — Responsive Price Action Core.

A clean rewrite after V7/V7.1 produced too few entries.

Architecture:
- 15M = directional bias only, using a simple 2-of-3 vote:
    price vs EMA20, EMA20 slope, RSI14 vs 50.
  No 1H hard gate and no RSI-cross ARM state.
- 5M = execution timeframe.
  Light market gate blocks only obvious dead/choppy conditions:
    ADX >= 10, CHOP < 68, ATR activity >= 0.55.
- Three independent 5M price-action triggers can enter with the 15M bias:
    1) PULLBACK_RECLAIM — recent EMA20 touch + directional reclaim.
    2) MICRO_BREAKOUT — close through previous 3-bar extreme with body >=0.20 ATR.
    3) SWEEP_RECLAIM — liquidity sweep of previous 5-bar extreme and close back in.
- Entry must still be fresh: current price may not be >0.35 ATR beyond trigger close.
- SL = trigger-local 5M structure +/-0.15 ATR, minimum 0.85 ATR.
  If the real structure stop needs >2.00 ATR, skip instead of tightening inside invalidation.
- TP1 = +1R, close 50%, runner SL -> +0.15R.
- TP2 = +2R, close remaining 50%.
- Technical exit is deliberately slow: only a confirmed opposite 15M bias flip.

The goal is responsiveness without copying EMA Hybrid's EMA8/13-cross execution.
"""
from __future__ import annotations

import numpy as np

from .base import Signal, SignalType
from .simple_precision_strategy import SimplePrecisionStrategy
from ..engines.position_manager import PositionUpdate


class SentinelV8Strategy(SimplePrecisionStrategy):
    VERSION = "8.0"
    entry_tf = "5m"

    FIVE_MIN_MS = 5 * 60_000
    FIFTEEN_MIN_MS = 15 * 60_000

    MIN_15M_BARS = 40
    MIN_5M_BARS = 45

    # 5M market gate: intentionally permissive; only poor market state blocks.
    ADX_FLOOR = 10.0
    CHOP_CEILING = 68.0
    ATR_ACTIVITY_FLOOR = 0.55

    # Execution freshness.
    MAX_TRIGGER_CHASE_ATR = 0.35

    # Stop construction.
    SL_BUFFER_ATR = 0.15
    MIN_SL_ATR = 0.85
    MAX_SL_ATR = 2.00

    # Position management.
    TP1_R = 1.00
    TP1_CLOSE_PCT = 0.50
    TP1_LOCK_R = 0.15
    TP2_R = 2.00

    # Fast post-exit re-entry protection, based on 5M bars rather than 15M.
    EXIT_COOLDOWN_5M_BARS = 1

    def __init__(self, symbol: str, **kwargs):
        super().__init__(
            symbol,
            adx_min=self.ADX_FLOOR,
            chop_max=self.CHOP_CEILING,
            stop_atr_min=self.MIN_SL_ATR,
            stop_atr_max=self.MAX_SL_ATR,
            target_r=self.TP2_R,
            tp1_r=self.TP1_R,
            tp1_trim_pct=self.TP1_CLOSE_PCT,
            exit_cooldown_bars=0,
        )
        self.name = f"SentinelV8({symbol})"
        self.target_r = self.TP2_R
        self.tp1_r = self.TP1_R
        self.tp1_trim_pct = self.TP1_CLOSE_PCT
        self.use_be_trail = False

        self._latest_5m: list = []
        self._last_5m_evaluated_ts: int | None = None
        self._last_exit_5m_ts: int | None = None

    # ------------------------------------------------------------------
    # 15M bias — 2-of-3 vote, no higher-TF veto
    # ------------------------------------------------------------------
    def _bias_15m(self, candles: list) -> dict:
        if len(candles) < self.MIN_15M_BARS:
            return {"ready": False, "direction": None, "reason": "15M warmup"}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        atr = self.atr(candles, 14)
        rsi = self.rsi(closes, 14)
        if not self._finite(ema20[-1], ema20[-4], atr[-1], rsi[-1]):
            return {"ready": False, "direction": None, "reason": "15M indicators unavailable"}

        close = float(closes[-1])
        e20 = float(ema20[-1])
        e20_prev = float(ema20[-4])
        atr_now = max(float(atr[-1]), 1e-12)
        rsi_now = float(rsi[-1])
        slope_atr = (e20 - e20_prev) / atr_now

        long_votes = {
            "price": close > e20,
            "slope": slope_atr > 0.0,
            "rsi": rsi_now >= 50.0,
        }
        short_votes = {
            "price": close < e20,
            "slope": slope_atr < 0.0,
            "rsi": rsi_now <= 50.0,
        }
        long_score = sum(1 for v in long_votes.values() if v)
        short_score = sum(1 for v in short_votes.values() if v)

        if long_score >= 2 and long_score > short_score:
            direction = "long"
        elif short_score >= 2 and short_score > long_score:
            direction = "short"
        else:
            direction = None

        return {
            "ready": direction is not None,
            "direction": direction,
            "long_votes": long_score,
            "short_votes": short_score,
            "close": round(close, 8),
            "ema20": round(e20, 8),
            "ema20_slope_atr": round(float(slope_atr), 3),
            "rsi": round(rsi_now, 2),
            "votes_long": long_votes,
            "votes_short": short_votes,
            "reason": f"15M {direction.upper()} bias {max(long_score, short_score)}/3" if direction else "15M mixed bias",
        }

    # ------------------------------------------------------------------
    # 5M market + trigger snapshot
    # ------------------------------------------------------------------
    def _snapshot_5m(self, candles: list, direction: str | None, current_price: float) -> dict:
        if len(candles) < self.MIN_5M_BARS:
            return {"ready": False, "trigger": None, "blocks": ["5M_WARMUP"], "reason": "5M warmup"}
        if direction not in {"long", "short"}:
            return {"ready": True, "trigger": None, "blocks": ["BIAS_15M"], "reason": "waiting for 15M bias"}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        atr = self.atr(candles, 14)
        adx, _, _ = self.adx(candles, 14)
        chop = self._choppiness(candles, 14)
        if chop is None or not self._finite(ema20[-1], atr[-1], adx[-1]):
            return {"ready": False, "trigger": None, "blocks": ["5M_INDICATORS"], "reason": "5M indicators unavailable"}

        bar = candles[-1]
        close = float(bar.close)
        open_ = float(bar.open)
        high = float(bar.high)
        low = float(bar.low)
        e20 = float(ema20[-1])
        atr_now = max(float(atr[-1]), 1e-12)
        adx_now = float(adx[-1])
        chop_now = float(chop)

        atr_hist = [float(v) for v in atr[-21:-1] if np.isfinite(v)]
        atr_median = float(np.median(atr_hist)) if atr_hist else atr_now
        atr_ratio = atr_now / max(atr_median, 1e-12)

        gate_blocks: list[str] = []
        if adx_now < self.ADX_FLOOR:
            gate_blocks.append("ADX")
        if chop_now >= self.CHOP_CEILING:
            gate_blocks.append("CHOP")
        if atr_ratio < self.ATR_ACTIVITY_FLOOR:
            gate_blocks.append("DEAD_VOL")

        bullish = close > open_
        bearish = close < open_
        body = abs(close - open_)
        long = direction == "long"

        # Trigger 1: pullback to/through EMA20 within the last 3 bars, then
        # directional reclaim through the previous bar extreme.
        recent3 = candles[-3:]
        touched_ema = (
            any(float(c.low) <= float(ema20[len(ema20) - len(candles) + candles.index(c)]) for c in [])
            if False else False
        )
        # Avoid index gymnastics: use current EMA20 as a local reference with
        # a small ATR tolerance across the last three bars.
        if long:
            touched_ema = any(float(c.low) <= e20 + 0.10 * atr_now for c in recent3)
            pullback_reclaim = touched_ema and bullish and close > float(candles[-2].high) and close >= e20
        else:
            touched_ema = any(float(c.high) >= e20 - 0.10 * atr_now for c in recent3)
            pullback_reclaim = touched_ema and bearish and close < float(candles[-2].low) and close <= e20

        # Trigger 2: fresh micro breakout through the previous 3-bar extreme.
        prev3 = candles[-4:-1]
        prev3_high = max(float(c.high) for c in prev3)
        prev3_low = min(float(c.low) for c in prev3)
        if long:
            micro_breakout = bullish and close > prev3_high and body >= 0.20 * atr_now and close >= e20
        else:
            micro_breakout = bearish and close < prev3_low and body >= 0.20 * atr_now and close <= e20

        # Trigger 3: sweep/reclaim of the previous 5-bar liquidity extreme.
        prev5 = candles[-6:-1]
        prev5_high = max(float(c.high) for c in prev5)
        prev5_low = min(float(c.low) for c in prev5)
        if long:
            sweep_reclaim = low < prev5_low and close > prev5_low and bullish
        else:
            sweep_reclaim = high > prev5_high and close < prev5_high and bearish

        # Priority prefers location-aware setups over raw expansion.
        if sweep_reclaim:
            trigger = "SWEEP_RECLAIM"
        elif pullback_reclaim:
            trigger = "PULLBACK_RECLAIM"
        elif micro_breakout:
            trigger = "MICRO_BREAKOUT"
        else:
            trigger = None

        out = {
            "ready": True,
            "direction": direction,
            "market_ready": not gate_blocks,
            "gate_blocks": gate_blocks,
            "trigger": trigger,
            "close": round(close, 8),
            "ema20": round(e20, 8),
            "atr5": float(atr_now),
            "atr_ratio": round(float(atr_ratio), 2),
            "adx": round(adx_now, 1),
            "chop": round(chop_now, 1),
            "candle": "BULL" if bullish else "BEAR" if bearish else "DOJI",
            "pullback": bool(pullback_reclaim),
            "breakout": bool(micro_breakout),
            "sweep": bool(sweep_reclaim),
            "prev3_high": round(prev3_high, 8),
            "prev3_low": round(prev3_low, 8),
            "prev5_high": round(prev5_high, 8),
            "prev5_low": round(prev5_low, 8),
            "blocks": list(gate_blocks),
            "reason": "waiting for 5M price-action trigger" if trigger is None else f"5M {trigger}",
        }

        if trigger is None or gate_blocks:
            if trigger and gate_blocks:
                out["reason"] = f"5M {trigger} blocked by market gate"
            return out

        # Freshness: poll is 60s, but still reject a trigger if price has
        # already expanded too far from its closed-bar signal price.
        if long:
            adverse = max(0.0, float(current_price) - close)
        else:
            adverse = max(0.0, close - float(current_price))
        chase_atr = adverse / atr_now
        out["chase_atr"] = round(float(chase_atr), 3)
        if chase_atr > self.MAX_TRIGGER_CHASE_ATR:
            out["trigger"] = None
            out["blocks"] = ["ANTI_CHASE"]
            out["reason"] = f"5M trigger skipped: chase {chase_atr:.2f}ATR > {self.MAX_TRIGGER_CHASE_ATR:.2f}ATR"
            return out

        # Trigger-local invalidation. Use tighter local structure for breakout,
        # wider local structure for pullback/sweep, then add an ATR buffer.
        lookback = 3 if trigger == "MICRO_BREAKOUT" else 5
        recent = candles[-lookback:]
        entry = float(current_price)
        if long:
            structure = min(float(c.low) for c in recent)
            raw_stop = structure - self.SL_BUFFER_ATR * atr_now
            raw_risk = entry - raw_stop
        else:
            structure = max(float(c.high) for c in recent)
            raw_stop = structure + self.SL_BUFFER_ATR * atr_now
            raw_risk = raw_stop - entry

        if raw_risk <= 0:
            out["trigger"] = None
            out["blocks"] = ["SL_STRUCTURE"]
            out["reason"] = "invalid 5M structure stop"
            return out

        raw_sl_atr = raw_risk / atr_now
        out["structure"] = round(structure, 8)
        out["raw_sl_atr"] = round(float(raw_sl_atr), 2)
        if raw_sl_atr > self.MAX_SL_ATR:
            out["trigger"] = None
            out["blocks"] = ["SL_TOO_WIDE"]
            out["reason"] = f"structure stop {raw_sl_atr:.2f}ATR > {self.MAX_SL_ATR:.2f}ATR"
            return out

        risk = max(raw_risk, self.MIN_SL_ATR * atr_now)
        stop = entry - risk if long else entry + risk
        tp1 = entry + self.TP1_R * risk if long else entry - self.TP1_R * risk
        tp2 = entry + self.TP2_R * risk if long else entry - self.TP2_R * risk

        out.update({
            "entry": entry,
            "risk": float(risk),
            "stop_loss": float(stop),
            "tp1_price": float(tp1),
            "take_profit": float(tp2),
            "sl_atr": round(float(risk / atr_now), 2),
        })
        return out

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c15 = self._closed_candle_series(candles, self.FIFTEEN_MIN_MS)
        c5 = self._closed_candle_series(mtf.get("5m", []), self.FIVE_MIN_MS)
        self._latest_15m = c15
        self._latest_5m = c5

        meta = {
            "strategy": "SENTINEL_V8",
            "version": self.VERSION,
            "architecture": "15M_2OF3_BIAS__5M_MULTI_PRICE_ACTION",
            "entry_tf": "5m_closed",
            "mtf_used": "15m_bias__5m_execution",
            "risk_plan": "5M_STRUCTURE+0.15ATR_MIN0.85_MAX2.00__TP1_1R_CLOSE50_LOCK+0.15R__TP2_2R",
        }

        if len(c15) < self.MIN_15M_BARS or len(c5) < self.MIN_5M_BARS:
            return self._hold(float(current_price), "waiting for closed 15M/5M warmup", meta)

        bias = self._bias_15m(c15)
        setup = self._snapshot_5m(c5, bias.get("direction"), float(current_price))
        meta["bias_15m"] = bias
        meta["setup_5m"] = setup

        if self._open_position is not None:
            return self._hold(float(current_price), f"managing open {self._open_position} position", meta)

        bar5_ts = int(self._bar_ts(c5[-1]))
        if self._last_5m_evaluated_ts == bar5_ts:
            return self._hold(float(current_price), "5M bar already evaluated", meta)
        self._last_5m_evaluated_ts = bar5_ts

        if self._last_exit_5m_ts is not None:
            elapsed = bar5_ts - self._last_exit_5m_ts
            if elapsed < self.EXIT_COOLDOWN_5M_BARS * self.FIVE_MIN_MS:
                return self._hold(float(current_price), "post-exit 5M cooldown", meta)

        if bias.get("direction") not in {"long", "short"}:
            return self._hold(float(current_price), bias.get("reason", "waiting for 15M bias"), meta)
        if not setup.get("market_ready", False):
            return self._hold(float(current_price), setup.get("reason", "5M market gate blocked"), meta)
        if not setup.get("trigger"):
            return self._hold(float(current_price), setup.get("reason", "waiting for 5M trigger"), meta)

        direction = str(bias["direction"])
        entry = float(setup["entry"])
        stop = float(setup["stop_loss"])
        target = float(setup["take_profit"])
        risk = float(setup["risk"])

        self._open_position = direction
        self._pending_entry = True
        self._entry_price = entry
        self._entry_sl = stop
        self._entry_tp = target
        self._initial_risk = risk
        self._tp1_done = False

        meta.update({
            "direction": direction,
            "entry_trigger": setup["trigger"],
            "stop_loss": round(stop, 8),
            "take_profit": round(target, 8),
            "tp1_price": round(float(setup["tp1_price"]), 8),
            "rr_ratio": self.TP2_R,
            "tp1_r": self.TP1_R,
            "tp1_close_pct": self.TP1_CLOSE_PCT,
            "tp1_lock_r": self.TP1_LOCK_R,
        })

        # Confidence is descriptive only; no extra confidence gate is added.
        vote_strength = max(int(bias.get("long_votes", 0)), int(bias.get("short_votes", 0)))
        trigger_bonus = 0.04 if setup["trigger"] in {"PULLBACK_RECLAIM", "SWEEP_RECLAIM"} else 0.0
        confidence = min(0.92, 0.74 + 0.05 * max(0, vote_strength - 2) + trigger_bonus)
        signal_type = SignalType.BUY if direction == "long" else SignalType.SELL
        reason = (
            f"{direction.upper()} {setup['trigger']} | 15M bias={vote_strength}/3 "
            f"ADX5={setup['adx']} CHOP5={setup['chop']} ATRx5={setup['atr_ratio']} "
            f"SL={setup['sl_atr']}ATR"
        )
        return Signal(signal_type, self.symbol, entry, 0.0, reason, confidence, meta)

    def tick_open_position(self, current_price: float, position_key: str | None = None):
        if self._open_position is None:
            return None

        # Slow technical exit: only a completed 15M bias flip can close early.
        candles = self._latest_15m
        if len(candles) >= self.MIN_15M_BARS:
            bar_ts = int(self._bar_ts(candles[-1]))
            if bar_ts != self._last_exit_check_ts:
                self._last_exit_check_ts = bar_ts
                bias = self._bias_15m(candles)
                direction = bias.get("direction")
                opposite = (
                    self._open_position == "long" and direction == "short"
                ) or (
                    self._open_position == "short" and direction == "long"
                )
                if opposite:
                    side = self._open_position
                    if self._latest_5m:
                        self._last_exit_5m_ts = int(self._bar_ts(self._latest_5m[-1]))
                    self._last_exit_bar_ts = bar_ts
                    self._reset_position(keep_exit_ts=True)
                    return PositionUpdate(
                        action="close",
                        close_pct=1.0,
                        reason=f"BIAS_FLIP_EXIT: 15M bias flipped opposite — close {side.upper()}",
                    )

        if (
            not self._tp1_done
            and self._entry_price is not None
            and self._initial_risk is not None
            and self._initial_risk > 0
        ):
            profit = (
                float(current_price) - self._entry_price
                if self._open_position == "long"
                else self._entry_price - float(current_price)
            )
            current_r = profit / self._initial_risk
            if current_r >= self.TP1_R:
                self._tp1_done = True
                new_sl = (
                    self._entry_price + self.TP1_LOCK_R * self._initial_risk
                    if self._open_position == "long"
                    else self._entry_price - self.TP1_LOCK_R * self._initial_risk
                )
                return PositionUpdate(
                    action="partial_tp",
                    close_pct=self.TP1_CLOSE_PCT,
                    new_sl=round(float(new_sl), 8),
                    reason=f"TP1 {current_r:.2f}R — close 50%, runner SL +{self.TP1_LOCK_R:.2f}R",
                )

        return PositionUpdate(
            action="hold",
            reason=(
                f"Holding {self._open_position.upper()} — 5M structure SL | "
                f"TP1 1.0R/50% -> +{self.TP1_LOCK_R:.2f}R | TP2 2.0R"
            ),
        )

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        if self._latest_5m:
            self._last_exit_5m_ts = int(self._bar_ts(self._latest_5m[-1]))
        if self._latest_15m:
            self._last_exit_bar_ts = int(self._bar_ts(self._latest_15m[-1]))
        self._reset_position(keep_exit_ts=True)
