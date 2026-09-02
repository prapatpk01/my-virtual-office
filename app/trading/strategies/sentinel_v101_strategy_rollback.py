"""Sentinel V10.1 rollback snapshot — Bias-Separated Momentum + Location Core."""
from __future__ import annotations

from .sentinel_v10_strategy import SentinelV10Strategy


class SentinelV101RollbackStrategy(SentinelV10Strategy):
    VERSION = "10.1"
    BIAS_MIN_SCORE = 7.0
    BIAS_MIN_EDGE = 1.50
    BIAS_MIN_MOMENTUM_VOTES = 2
    BIAS_MAX_COUNTER_SLOPE_ATR = 0.10
    BIAS_MIN_ROOM_ATR = 0.75
    BIAS_OPPOSING_FORECAST_CONF = 65.0

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV10({symbol})"

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
            location_ok = bool((near_support or ema_value or resistance is None or room_atr >= self.BIAS_MIN_ROOM_ATR) and not near_resistance)
            forecast_ok = not (fc_side == "BEARISH" and fc_conf >= self.BIAS_OPPOSING_FORECAST_CONF)
        elif candidate == "short":
            momentum_votes = int(rsi < rsi_sma) + int(macd_hist < 0.0 or macd_delta < 0.0) + int(k < d)
            slope_ok = slope <= self.BIAS_MAX_COUNTER_SLOPE_ATR
            if support is not None:
                room_atr = max(0.0, (close - float(support)) / atr)
            location_ok = bool((near_resistance or ema_value or support is None or room_atr >= self.BIAS_MIN_ROOM_ATR) and not near_support)
            forecast_ok = not (fc_side == "BULLISH" and fc_conf >= self.BIAS_OPPOSING_FORECAST_CONF)

        if candidate is None: rejects.append("NO_SCORE_EDGE")
        if leading_score < self.BIAS_MIN_SCORE: rejects.append("BIAS_SCORE")
        if edge < self.BIAS_MIN_EDGE: rejects.append("BIAS_EDGE")
        if not regime_ok: rejects.append("REGIME_2OF3")
        if momentum_votes < self.BIAS_MIN_MOMENTUM_VOTES: rejects.append("MOMENTUM_2OF3")
        if not slope_ok: rejects.append("COUNTER_SLOPE")
        if not location_ok: rejects.append("S_R_LOCATION")
        if not forecast_ok: rejects.append("OPPOSING_FORECAST")

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
            out["reason"] = f"15M no exact setup; bias {candidate or 'NONE'} score={leading_score:.2f} edge={edge:.2f} mom={momentum_votes}/3 rejects={','.join(rejects) or 'none'}"
            return out

        components = a.get("components_long") if candidate == "long" else a.get("components_short")
        out.update({
            "direction": candidate,
            "selected_setup": "MOMENTUM_CONTINUATION",
            "selected_score": round(leading_score, 2),
            "score_threshold": self.BIAS_MIN_SCORE,
            "components": components or {},
            "reason": f"15M MOMENTUM_CONTINUATION {candidate.upper()} score {leading_score:.2f}/{self.BIAS_MIN_SCORE:.2f} edge={edge:.2f} momentum={momentum_votes}/3 room={room_atr:.2f}ATR",
        })
        self._bias_strength = 3 if leading_score >= 8.0 else 2 if leading_score >= 7.0 else 1
        return out

    @staticmethod
    def _apply_setup_execution_map(setup: dict, family: str | None) -> dict:
        if family != "MOMENTUM_CONTINUATION":
            return SentinelV10Strategy._apply_setup_execution_map(setup, family)
        if not setup.get("trigger"):
            return setup
        allowed = {"PULLBACK_RECLAIM", "MICRO_BREAKOUT"}
        if setup.get("trigger") in allowed:
            return setup
        out = dict(setup)
        out["trigger_candidate"] = setup.get("trigger")
        out["trigger"] = None
        out["blocks"] = list(dict.fromkeys(list(out.get("blocks", [])) + ["BIAS_EXEC_MISMATCH"]))
        return out

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)
        meta = signal.metadata or {}
        analysis = meta.get("analysis_15m") or {}
        meta["version"] = self.VERSION
        meta["architecture"] = "15M_EXACT_SETUP_OR_QUALIFIED_BIAS__5M_PRICE_ACTION_EXECUTION"
        meta["setup_mode"] = analysis.get("setup_mode")
        meta["bias_fallback"] = bool(analysis.get("bias_fallback"))
        meta["bias_score_edge"] = analysis.get("bias_score_edge")
        meta["bias_momentum_votes"] = analysis.get("bias_momentum_votes")
        meta["bias_rejects"] = analysis.get("bias_rejects") or []
        signal.metadata = meta
        return signal
