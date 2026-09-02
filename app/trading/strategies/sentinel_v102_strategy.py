"""Sentinel V10.2 — Responsive Bias + Execution Core.

V10.2 keeps the V10/V10.1 analysis stack and lifecycle, but removes the
remaining no-trade bottlenecks without turning the strategy into an unfiltered
momentum bot.

Changes vs V10.1:
- Exact 15M setups still have first priority.
- Bias fallback is slightly more responsive, but still needs a clear score
  edge, 2-of-3 momentum, valid regime/location and no strong counter trend.
- 5M adds MOMENTUM_RECLAIM and uses a 2-bar micro breakout so a good 15M thesis
  does not have to wait for an unusually perfect 5M candle.
- 5M quality gates are softened, not removed.
- Fixed natural-risk >=0.40% is replaced by a fee-to-R economic floor. The stop
  may be widened beyond local structure (never inside it) to keep estimated
  round-trip taker fees <=0.55R, provided the final stop remains <=2.20 ATR.
- After a hard SL, the existing 3x5M cooldown + one-entry-per-15M protection is
  sufficient; a second fresh 15M RSI/SR event is no longer required.
"""
from __future__ import annotations

import numpy as np

from .sentinel_v101_strategy import SentinelV101Strategy


class SentinelV102Strategy(SentinelV101Strategy):
    VERSION = "10.2"

    # 15M fallback: still selective, but no longer requires a near-perfect
    # separation between long/short scores.
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

    # Paper/live entries are market-like in the current engine. 0.05% taker
    # per side -> ~0.10% round trip. Keep estimated fee drag <=0.55R instead of
    # enforcing one fixed risk percentage across BTC, metals and alts.
    ASSUMED_TAKER_FEE_SIDE = 0.0005
    ROUND_TRIP_FEE_PCT = ASSUMED_TAKER_FEE_SIDE * 2.0
    MAX_EST_FEE_R = 0.55
    MIN_ECONOMIC_RISK_PCT = ROUND_TRIP_FEE_PCT / MAX_EST_FEE_R

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        # Keep the long-lived position key compatible with V10/V10.1.
        self.name = f"SentinelV10({symbol})"

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

        kval = float(k[-1]); dval = float(d[-1]); jval = float(j[-1])
        jprev = float(j[-2]) if np.isfinite(j[-2]) else jval
        kdj_aligned = (kval > dval and jval >= jprev) if long else (kval < dval and jval <= jprev)

        # Responsive reclaim: price only needs to prove that the pullback has
        # stopped and reclaimed value; it no longer must close beyond the whole
        # previous candle high/low.
        recent3 = candles[-3:]
        if long:
            touched = any(float(c.low) <= e20 + 0.18 * atr5 for c in recent3)
            pullback = (
                touched and bullish and close >= e20
                and (close > prev_close or close_pos >= 0.60)
            )
        else:
            touched = any(float(c.high) >= e20 - 0.18 * atr5 for c in recent3)
            pullback = (
                touched and bearish and close <= e20
                and (close < prev_close or close_pos <= 0.40)
            )

        # Two-bar micro breakout instead of three-bar. 15M already supplies the
        # thesis; 5M only needs to confirm immediate order-flow continuation.
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

        # New fast timing path for a qualified 15M bias / exact pullback. It is
        # still directional, EMA20-located, KDJ-aligned and candle-confirmed.
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

            # Volume and 5M slope are now vetoes only when clearly poor. They
            # are not allowed to duplicate the 15M momentum/regime engine.
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

                # Economic stop floor is applied by widening OUTSIDE structure,
                # never by moving the stop inside the invalidation level.
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

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        # V10.1's hard-SL fresh-event latch duplicated the 3x5M cooldown and
        # often required a 15M event and a 5M trigger to coincide. Keep the
        # cooldown, but remove this second latch.
        if getattr(self, "_last_close_was_hard_sl", False):
            self._last_close_was_hard_sl = False

        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)
        meta = signal.metadata or {}
        analysis = meta.get("analysis_15m") or {}
        setup = meta.get("setup_5m") or {}
        meta["version"] = self.VERSION
        meta["architecture"] = "15M_EXACT_OR_RESPONSIVE_BIAS__5M_RESPONSIVE_PA__FEE_R_RISK"
        meta["risk_plan"] = "STRUCTURE_OUTSIDE__MIN0.80ATR__MAX2.20ATR__EST_FEE<=0.55R__TP1_1R50_LOCK0.15R__TP2_SR"
        meta["economic_floor_pct"] = setup.get("economic_floor_pct")
        meta["fee_r_est"] = setup.get("fee_r_est")
        meta["planned_risk_pct"] = setup.get("planned_risk_pct")
        meta["bias_fallback"] = bool(analysis.get("bias_fallback"))
        signal.metadata = meta
        return signal

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        super().record_closed_trade(exit_price, reason, duration_min)
        # Parent keeps 3x5M cooldown for hard SL. Do not also require a fresh
        # 15M RSI/SR event; one-entry-per-15M remains active as a second guard.
        r = str(reason or "").lower()
        if "stop_loss" in r or "hard_sl" in r or "stop loss" in r:
            self._last_close_was_hard_sl = False
