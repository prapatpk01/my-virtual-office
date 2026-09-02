"""Sentinel V10.2 — Responsive Bias + Execution Core.

Production keeps the historical SentinelV10(symbol) strategy key so open
positions and persisted state survive deploys. The running logic is V10.2.

V10.2 keeps V10's EMA20 + RSI/SMA14 + MACD + KDJ + ADX/CHOP/ATR activity +
S/R + Forecast architecture, but removes the remaining no-trade bottlenecks:
- exact 15M setups remain first priority;
- bias fallback is slightly more responsive while still requiring score edge,
  2-of-3 momentum, regime, location and no strong counter trend;
- 5M uses responsive pullback, 2-bar micro breakout and MOMENTUM_RECLAIM;
- fixed natural-risk >=0.40% is replaced by a fee-to-R economic stop floor;
- after hard SL, 3x5M cooldown + one-entry-per-15M are sufficient safeguards.

V10.1 source is preserved in sentinel_v101_strategy_rollback.py.
"""
from __future__ import annotations

import numpy as np

from .sentinel_v10_strategy import SentinelV10Strategy


class SentinelV101Strategy(SentinelV10Strategy):
    VERSION = "10.2"

    # 15M fallback: responsive but still selective.
    BIAS_MIN_SCORE = 6.50
    BIAS_MIN_EDGE = 1.00
    BIAS_MIN_MOMENTUM_VOTES = 2
    BIAS_MAX_COUNTER_SLOPE_ATR = 0.15
    BIAS_MIN_ROOM_ATR = 0.50
    BIAS_OPPOSING_FORECAST_CONF = 75.0

    # 5M execution / risk.
    MAX_TRIGGER_CHASE_ATR = 0.35
    SL_BUFFER_ATR = 0.18
    MIN_SL_ATR = 0.80
    MAX_SL_ATR = 2.20

    # Estimated market-order economics. 0.05% taker each side -> 0.10% round
    # trip. Planned stop distance is sized so estimated fee drag <=0.55R.
    ASSUMED_TAKER_FEE_SIDE = 0.0005
    ROUND_TRIP_FEE_PCT = ASSUMED_TAKER_FEE_SIDE * 2.0
    MAX_EST_FEE_R = 0.55
    MIN_ECONOMIC_RISK_PCT = ROUND_TRIP_FEE_PCT / MAX_EST_FEE_R

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV10({symbol})"

    # ------------------------------------------------------------------
    # 15M: exact setup first; otherwise qualified directional bias.
    # ------------------------------------------------------------------
    def _analysis_15m(self, candles: list) -> dict:
        a = super()._analysis_15m(candles)
        if not a.get("ready"):
            return a

        if a.get("direction") in {"long", "short"} and a.get("selected_setup"):
            out = dict(a)
            out["setup_mode"] = "EXACT_SETUP"
            out["bias_fallback"] = False
            out["bias_candidate_side"] = a.get("direction")
            out["bias_score_edge"] = round(abs(float(a.get("score_long") or 0.0) - float(a.get("score_short") or 0.0)), 2)
            out["bias_rejects"] = []
            return out

        long_score = float(a.get("score_long") or 0.0)
        short_score = float(a.get("score_short") or 0.0)
        edge = abs(long_score - short_score)
        candidate = "long" if long_score > short_score else "short" if short_score > long_score else None
        leading_score = max(long_score, short_score)

        regime = a.get("regime") or {}
        regime_ok = bool(regime.get("market_ready"))
        atr = max(float(regime.get("atr") or 0.0), 1e-12)
        slope = float(a.get("ema20_slope_atr") or 0.0)
        close = float(candles[-1].close)
        support = a.get("support")
        resistance = a.get("resistance")
        near_support = bool(a.get("near_support"))
        near_resistance = bool(a.get("near_resistance"))
        ema_value = a.get("location") == "EMA20_VALUE"

        rsi = float(a.get("rsi") or 50.0)
        rsi_sma = float(a.get("rsi_sma") or 50.0)
        macd_hist = float(a.get("macd_hist") or 0.0)
        macd_delta = float(a.get("macd_hist_delta") or 0.0)
        k = float(a.get("kdj_k") or 50.0)
        d = float(a.get("kdj_d") or 50.0)

        forecast = a.get("forecast") or {}
        fc_side = str(forecast.get("side") or "NEUTRAL")
        fc_conf = float(forecast.get("confidence") or 0.0)

        rejects: list[str] = []
        momentum_votes = 0
        room_atr = 99.0
        location_ok = False
        slope_ok = False
        forecast_ok = True

        if candidate == "long":
            momentum_votes = int(rsi > rsi_sma) + int(macd_hist > 0.0 or macd_delta > 0.0) + int(k > d)
            slope_ok = slope >= -self.BIAS_MAX_COUNTER_SLOPE_ATR
            if resistance is not None:
                room_atr = max(0.0, (float(resistance) - close) / atr)
            location_ok = bool(
                (near_support or ema_value or resistance is None or room_atr >= self.BIAS_MIN_ROOM_ATR)
                and not near_resistance
            )
            forecast_ok = not (fc_side == "BEARISH" and fc_conf >= self.BIAS_OPPOSING_FORECAST_CONF)
        elif candidate == "short":
            momentum_votes = int(rsi < rsi_sma) + int(macd_hist < 0.0 or macd_delta < 0.0) + int(k < d)
            slope_ok = slope <= self.BIAS_MAX_COUNTER_SLOPE_ATR
            if support is not None:
                room_atr = max(0.0, (close - float(support)) / atr)
            location_ok = bool(
                (near_resistance or ema_value or support is None or room_atr >= self.BIAS_MIN_ROOM_ATR)
                and not near_support
            )
            forecast_ok = not (fc_side == "BULLISH" and fc_conf >= self.BIAS_OPPOSING_FORECAST_CONF)

        if candidate is None:
            rejects.append("NO_SCORE_EDGE")
        if leading_score < self.BIAS_MIN_SCORE:
            rejects.append("BIAS_SCORE")
        if edge < self.BIAS_MIN_EDGE:
            rejects.append("BIAS_EDGE")
        if not regime_ok:
            rejects.append("REGIME_2OF3")
        if momentum_votes < self.BIAS_MIN_MOMENTUM_VOTES:
            rejects.append("MOMENTUM_2OF3")
        if not slope_ok:
            rejects.append("COUNTER_SLOPE")
        if not location_ok:
            rejects.append("S_R_LOCATION")
        if not forecast_ok:
            rejects.append("OPPOSING_FORECAST")

        out = dict(a)
        out.update({
            "setup_mode": "BIAS_FALLBACK" if not rejects else "NONE",
            "bias_fallback": not rejects,
            "bias_candidate_side": candidate,
            "bias_candidate_score": round(leading_score, 2),
            "bias_score_edge": round(edge, 2),
            "bias_momentum_votes": momentum_votes,
            "bias_room_atr": round(room_atr, 2) if room_atr < 90 else None,
            "bias_slope_ok": slope_ok,
            "bias_location_ok": location_ok,
            "bias_forecast_ok": forecast_ok,
            "bias_rejects": rejects,
        })

        if rejects or candidate not in {"long", "short"}:
            out["reason"] = (
                f"15M no exact setup; bias {candidate or 'NONE'} score={leading_score:.2f} "
                f"edge={edge:.2f} mom={momentum_votes}/3 rejects={','.join(rejects) or 'none'}"
            )
            return out

        components = a.get("components_long") if candidate == "long" else a.get("components_short")
        out.update({
            "direction": candidate,
            "selected_setup": "MOMENTUM_CONTINUATION",
            "selected_score": round(leading_score, 2),
            "score_threshold": self.BIAS_MIN_SCORE,
            "components": components or {},
            "reason": (
                f"15M MOMENTUM_CONTINUATION {candidate.upper()} score {leading_score:.2f}/{self.BIAS_MIN_SCORE:.2f} "
                f"edge={edge:.2f} momentum={momentum_votes}/3 room={room_atr:.2f}ATR"
            ),
        })
        self._bias_strength = 3 if leading_score >= 8.0 else 2 if leading_score >= 7.0 else 1
        return out

    # ------------------------------------------------------------------
    # 5M: responsive confirmation, not a duplicate analysis engine.
    # ------------------------------------------------------------------
    def _snapshot_5m_v10(self, candles: list, direction: str | None, current_price: float) -> dict:
        if len(candles) < self.MIN_5M_BARS:
            return {"ready": False, "market_ready": False, "trigger": None, "blocks": ["5M_WARMUP"]}
        if direction not in {"long", "short"}:
            return {"ready": True, "market_ready": False, "trigger": None, "blocks": ["NO_15M_BIAS"]}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        atr = self.atr(candles, 14)
        regime = self._regime_snapshot(candles)
        k, d, j = self._kdj(candles, 9)
        if not regime.get("ready") or not self._finite(ema20[-1], atr[-1], k[-1], d[-1], j[-1]):
            return {"ready": False, "market_ready": False, "trigger": None, "blocks": ["5M_INDICATORS"]}

        bar = candles[-1]
        prev = candles[-2]
        close = float(bar.close)
        open_ = float(bar.open)
        high = float(bar.high)
        low = float(bar.low)
        prev_close = float(prev.close)
        atr5 = max(float(atr[-1]), 1e-12)
        e20 = float(ema20[-1])
        long = direction == "long"
        bullish = close > open_
        bearish = close < open_
        body_atr = abs(close - open_) / atr5
        rng = max(high - low, 1e-12)
        close_pos = (close - low) / rng
        slope5 = (e20 - float(ema20[-4])) / atr5
        dist_ema = abs(close - e20) / atr5

        vols = [float(c.volume or 0.0) for c in candles[-21:-1]]
        med_vol = float(np.median(vols)) if vols else 0.0
        vol_ratio = float(bar.volume or 0.0) / med_vol if med_vol > 0 else 1.0

        kval = float(k[-1])
        dval = float(d[-1])
        jval = float(j[-1])
        jprev = float(j[-2]) if np.isfinite(j[-2]) else jval
        kdj_aligned = (kval > dval and jval >= jprev) if long else (kval < dval and jval <= jprev)

        recent3 = candles[-3:]
        if long:
            touched = any(float(c.low) <= e20 + 0.18 * atr5 for c in recent3)
            pullback = touched and bullish and close >= e20 and (close > prev_close or close_pos >= 0.60)
        else:
            touched = any(float(c.high) >= e20 - 0.18 * atr5 for c in recent3)
            pullback = touched and bearish and close <= e20 and (close < prev_close or close_pos <= 0.40)

        prev2 = candles[-3:-1]
        p2h = max(float(c.high) for c in prev2)
        p2l = min(float(c.low) for c in prev2)
        breakout = (
            bullish and close > p2h and body_atr >= 0.15 and close >= e20
            if long else
            bearish and close < p2l and body_atr >= 0.15 and close <= e20
        )

        prev5 = candles[-6:-1]
        p5h = max(float(c.high) for c in prev5)
        p5l = min(float(c.low) for c in prev5)
        sweep = (
            low < p5l and close > p5l and bullish
            if long else
            high > p5h and close < p5h and bearish
        )

        momentum_reclaim = (
            bullish and close >= e20 and close > prev_close and kdj_aligned
            and body_atr >= 0.12 and dist_ema <= 0.85
            if long else
            bearish and close <= e20 and close < prev_close and kdj_aligned
            and body_atr >= 0.12 and dist_ema <= 0.85
        )

        trigger = (
            "SWEEP_RECLAIM" if sweep else
            "PULLBACK_RECLAIM" if pullback else
            "MICRO_BREAKOUT" if breakout else
            "MOMENTUM_RECLAIM" if momentum_reclaim else
            None
        )
        candidate = trigger
        blocks: list[str] = []

        if not regime["market_ready"]:
            blocks.append("REGIME_2OF3")

        if trigger:
            close_quality = close_pos >= 0.56 if long else close_pos <= 0.44
            if not close_quality:
                blocks.append("WEAK_CLOSE")

            min_body = {
                "MICRO_BREAKOUT": 0.16,
                "PULLBACK_RECLAIM": 0.10,
                "SWEEP_RECLAIM": 0.08,
                "MOMENTUM_RECLAIM": 0.12,
            }[trigger]
            if body_atr < min_body:
                blocks.append("WEAK_BODY")

            max_dist = {
                "MICRO_BREAKOUT": 1.50,
                "PULLBACK_RECLAIM": 1.25,
                "SWEEP_RECLAIM": 1.35,
                "MOMENTUM_RECLAIM": 0.90,
            }[trigger]
            if dist_ema > max_dist:
                blocks.append("EXTENDED_FROM_EMA20")

            if trigger == "MICRO_BREAKOUT" and vol_ratio < 0.55:
                blocks.append("VERY_LOW_BREAKOUT_VOLUME")
            if trigger == "MICRO_BREAKOUT":
                strongly_opposed = (long and slope5 < -0.08) or ((not long) and slope5 > 0.08)
                if strongly_opposed:
                    blocks.append("STRONG_5M_COUNTER_SLOPE")

        chase = None
        if trigger and not blocks:
            adverse = max(0.0, float(current_price) - close) if long else max(0.0, close - float(current_price))
            chase = adverse / atr5
            if chase > self.MAX_TRIGGER_CHASE_ATR:
                blocks.append("ANTI_CHASE")

        structure = None
        raw_sl_atr = None
        raw_risk_pct = None
        planned_risk_pct = None
        fee_r_est = None
        stop = tp1 = tp2 = risk = None

        if trigger and not blocks:
            lookback = 3 if trigger == "MICRO_BREAKOUT" else 4 if trigger in {"PULLBACK_RECLAIM", "MOMENTUM_RECLAIM"} else 5
            recent = candles[-lookback:]
            entry = float(current_price)
            if long:
                structure = min(float(c.low) for c in recent)
                raw_stop = structure - self.SL_BUFFER_ATR * atr5
                raw_risk = entry - raw_stop
            else:
                structure = max(float(c.high) for c in recent)
                raw_stop = structure + self.SL_BUFFER_ATR * atr5
                raw_risk = raw_stop - entry

            if raw_risk <= 0:
                blocks.append("SL_STRUCTURE")
            else:
                raw_sl_atr = raw_risk / atr5
                raw_risk_pct = raw_risk / max(entry, 1e-12)

                # Widen stop outside structure only when needed for ATR/fee
                # economics; never pull it inside the invalidation level.
                economic_floor = self.MIN_ECONOMIC_RISK_PCT * entry
                risk = max(raw_risk, self.MIN_SL_ATR * atr5, economic_floor)
                planned_sl_atr = risk / atr5
                planned_risk_pct = risk / max(entry, 1e-12)
                fee_r_est = self.ROUND_TRIP_FEE_PCT / max(planned_risk_pct, 1e-12)

                if planned_sl_atr > self.MAX_SL_ATR:
                    blocks.append("SL_TOO_WIDE")
                if fee_r_est > self.MAX_EST_FEE_R + 1e-9:
                    blocks.append("FEE_R_TOO_HIGH")

                if not blocks:
                    stop = entry - risk if long else entry + risk
                    tp1 = entry + self.TP1_R * risk if long else entry - self.TP1_R * risk
                    tp2 = entry + self.DYNAMIC_TP_FALLBACK_R * risk if long else entry - self.DYNAMIC_TP_FALLBACK_R * risk

        out = {
            "ready": True,
            "direction": direction,
            "market_ready": bool(regime["market_ready"]),
            "regime_pass": regime["pass_count"],
            "adx": regime["adx"],
            "chop": regime["chop"],
            "atr_ratio": regime["atr_ratio"],
            "candidate": candidate,
            "trigger": trigger if trigger and not blocks else None,
            "trigger_candidate": candidate,
            "pullback": bool(pullback),
            "breakout": bool(breakout),
            "sweep": bool(sweep),
            "momentum_reclaim": bool(momentum_reclaim),
            "body_atr": round(float(body_atr), 2),
            "close_pos": round(float(close_pos), 2),
            "dist_ema_atr": round(float(dist_ema), 2),
            "ema20_slope_atr": round(float(slope5), 3),
            "volume_ratio": round(float(vol_ratio), 2),
            "kdj_k": round(kval, 2),
            "kdj_d": round(dval, 2),
            "kdj_j": round(jval, 2),
            "chase_atr": round(float(chase), 3) if chase is not None else None,
            "structure": round(float(structure), 8) if structure is not None else None,
            "raw_sl_atr": round(float(raw_sl_atr), 2) if raw_sl_atr is not None else None,
            "raw_risk_pct": round(float(raw_risk_pct * 100.0), 3) if raw_risk_pct is not None else None,
            "planned_risk_pct": round(float(planned_risk_pct * 100.0), 3) if planned_risk_pct is not None else None,
            "fee_r_est": round(float(fee_r_est), 2) if fee_r_est is not None else None,
            "economic_floor_pct": round(self.MIN_ECONOMIC_RISK_PCT * 100.0, 3),
            "blocks": list(dict.fromkeys(blocks)),
        }
        if risk is not None and stop is not None and not blocks:
            out.update({
                "entry": float(current_price),
                "risk": float(risk),
                "stop_loss": float(stop),
                "tp1_price": float(tp1),
                "take_profit": float(tp2),
                "sl_atr": round(float(risk / atr5), 2),
            })
        return out

    @staticmethod
    def _apply_setup_execution_map(setup: dict, family: str | None) -> dict:
        if not family or not setup.get("trigger"):
            return setup
        allowed = {
            "PULLBACK": {"PULLBACK_RECLAIM", "MICRO_BREAKOUT", "MOMENTUM_RECLAIM"},
            "BREAKOUT_RETEST": {"PULLBACK_RECLAIM", "MICRO_BREAKOUT"},
            "SWEEP_REVERSAL": {"SWEEP_RECLAIM", "MICRO_BREAKOUT"},
            "MOMENTUM_CONTINUATION": {"PULLBACK_RECLAIM", "MICRO_BREAKOUT", "MOMENTUM_RECLAIM"},
        }.get(family, set())
        if setup.get("trigger") in allowed:
            return setup
        out = dict(setup)
        out["trigger_candidate"] = setup.get("trigger")
        out["trigger"] = None
        out["blocks"] = list(dict.fromkeys(list(out.get("blocks", [])) + ["SETUP_EXEC_MISMATCH"]))
        return out

    # ------------------------------------------------------------------
    # Lifecycle / metadata.
    # ------------------------------------------------------------------
    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        # Remove duplicated hard-SL fresh-event latch. Parent still enforces the
        # 3x5M cooldown and one-entry-per-15M guard.
        if getattr(self, "_last_close_was_hard_sl", False):
            self._last_close_was_hard_sl = False

        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)
        meta = signal.metadata or {}
        analysis = meta.get("analysis_15m") or {}
        setup = meta.get("setup_5m") or {}
        meta["version"] = self.VERSION
        meta["architecture"] = "15M_EXACT_OR_RESPONSIVE_BIAS__5M_RESPONSIVE_PA__FEE_R_RISK"
        meta["setup_mode"] = analysis.get("setup_mode")
        meta["bias_fallback"] = bool(analysis.get("bias_fallback"))
        meta["bias_score_edge"] = analysis.get("bias_score_edge")
        meta["bias_momentum_votes"] = analysis.get("bias_momentum_votes")
        meta["bias_rejects"] = analysis.get("bias_rejects") or []
        meta["risk_plan"] = "STRUCTURE_OUTSIDE__MIN0.80ATR__MAX2.20ATR__EST_FEE<=0.55R__TP1_1R50_LOCK0.15R__TP2_SR"
        meta["economic_floor_pct"] = setup.get("economic_floor_pct")
        meta["fee_r_est"] = setup.get("fee_r_est")
        meta["planned_risk_pct"] = setup.get("planned_risk_pct")
        signal.metadata = meta
        return signal

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        super().record_closed_trade(exit_price, reason, duration_min)
        r = str(reason or "").lower()
        if "stop_loss" in r or "hard_sl" in r or "stop loss" in r:
            self._last_close_was_hard_sl = False
