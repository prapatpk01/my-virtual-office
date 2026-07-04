"""
AI Signal strategy v4 — API-free, multi-timeframe, regime-adaptive (hardened).

Architecture (top-down):
  4h candles  → detect_regime()      — TRENDING/RANGING/VOLATILE/LOW_CONVICTION
  1h candles  → directional_bias()   — bull/bear bias score  (-3 … +3)
  15m candles → volume_ok()          — skip entry when volume is too thin
  15m candles → _score_factors()     — entry trigger score (RSI/MACD/ST/Vol/EMA/HA)
  price+ATR   → build_trade_plan()   — SL=1R, T1=0.5R … T4=1.2R, SL-ratchet ladder

The Signal.metadata returned is fully compatible with TradingBot._open_position():
  sl_ladder_enabled=True activates SL-ratchet mode (no partial closes; SL steps
  up as each ladder level is hit: T1→+0.3R, T2→+0.5R, T3→+0.8R, T4→full close).

Improvements vs v3:
  - Parameter validation / sanitization with safe defaults.
  - Standardized decision metadata across HOLD / BUY / SELL paths.
  - Deterministic handling for both-valid long/short cases.
  - Confidence score now reflects threshold edge, side separation, regime/bias
    alignment, and remains deterministic / bounded.
  - Explicit sizing intent metadata for downstream consumers.
"""
import math
import numpy as np
from .base import BaseStrategy, Signal, SignalType
from .mtf_regime import (
    RegimeType,
    detect_regime,
    directional_bias,
    volume_ok,
    entry_threshold,
    _score_factors,
    build_trade_plan,
    BIAS_MISALIGN_LONG_MIN,
    BIAS_MISALIGN_SHORT_MAX,
)


class AISignalStrategy(BaseStrategy):
    """
    Multi-timeframe, regime-adaptive signal strategy.

    No external API dependency — fully deterministic, indicator-based.
    Keeps the same class name and interface as the original AISignalStrategy
    so all existing bot wiring continues to work without changes.

    Params (all optional, with sensible defaults):
      position_pct       — legacy position fraction (0.05); overridden by
                            RiskManager dynamic sizing when sl_dist_pct and
                            risk_pct are present.
      vol_period          — lookback for volume filter (default 20)
      vol_threshold       — minimum vol_ratio to trade (default 0.70)
      min_15m             — minimum 15m candles required (default 40)
      min_1h              — minimum 1h candles required (default 55)
      min_4h              — minimum 4h candles required (default 40)

      tie_break_policy    — how to resolve near-equal long/short scores that
                            both clear the entry threshold:
                              "hold"   — do nothing, stay flat (default, safest)
                              "bias"   — take the side the 1h bias favors
                              "regime" — take the side the 4h regime favors,
                                         falling back to bias if regime is
                                         non-directional (e.g. RANGING)
      tie_tolerance       — abs(long_score - short_score) <= tolerance is
                            considered a tie (default 1e-6)

      atr_guard_enabled   — if True, an invalid ATR aborts trade-plan
                            construction and returns HOLD instead (default True)
      atr_min_value       — minimum ATR value accepted as valid (default 1e-10)
    """

    MTF_TIMEFRAMES = ["1h", "4h"]

    DEFAULTS = {
        "position_pct": 0.05,
        "vol_period": 20,
        "vol_threshold": 0.70,
        "min_15m": 40,
        "min_1h": 55,
        "min_4h": 40,
        "tie_break_policy": "hold",
        "tie_tolerance": 1e-6,
        "atr_guard_enabled": True,
        "atr_min_value": 1e-10,
    }
    VALID_TIE_BREAK_POLICIES = {"hold", "bias", "regime"}

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)

        self._param_warnings = []
        self.position_pct = self._sanitize_float(
            "position_pct", self.DEFAULTS["position_pct"], minimum=0.0, maximum=1.0, strict_min=True
        )
        self.vol_period = self._sanitize_int(
            "vol_period", self.DEFAULTS["vol_period"], minimum=1
        )
        self.vol_threshold = self._sanitize_float(
            "vol_threshold", self.DEFAULTS["vol_threshold"], minimum=0.0, strict_min=True
        )
        self.min_15m = self._sanitize_int(
            "min_15m", self.DEFAULTS["min_15m"], minimum=20
        )
        self.min_1h = self._sanitize_int(
            "min_1h", self.DEFAULTS["min_1h"], minimum=20
        )
        self.min_4h = self._sanitize_int(
            "min_4h", self.DEFAULTS["min_4h"], minimum=20
        )

        raw_policy = str(self.params.get("tie_break_policy", self.DEFAULTS["tie_break_policy"]))
        policy = raw_policy.strip().lower()
        if policy not in self.VALID_TIE_BREAK_POLICIES:
            self._param_warnings.append(
                f"tie_break_policy='{raw_policy}' invalid; fallback='{self.DEFAULTS['tie_break_policy']}'"
            )
            policy = self.DEFAULTS["tie_break_policy"]
        self.tie_break_policy = policy

        self.tie_tolerance = self._sanitize_float(
            "tie_tolerance", self.DEFAULTS["tie_tolerance"], minimum=0.0
        )
        self.atr_guard_enabled = self._sanitize_bool(
            "atr_guard_enabled", self.DEFAULTS["atr_guard_enabled"]
        )
        self.atr_min_value = self._sanitize_float(
            "atr_min_value", self.DEFAULTS["atr_min_value"], minimum=0.0, strict_min=True
        )

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        """
        Produce a BUY / SELL / HOLD signal with full metadata.
        """
        mtf = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])

        if len(candles) < self.min_15m:
            return self._hold(
                current_price,
                f"Insufficient 15m data ({len(candles)} bars < {self.min_15m})",
                metadata=self._decision_meta(
                    stage="data_guard",
                    reason_code="insufficient_15m_data",
                    actual_15m=len(candles),
                    required_15m=self.min_15m,
                ),
            )

        if len(candles_4h) < self.min_4h:
            return self._hold(
                current_price,
                f"Insufficient 4h data ({len(candles_4h)} bars < {self.min_4h})",
                metadata=self._decision_meta(
                    stage="data_guard",
                    reason_code="insufficient_4h_data",
                    actual_4h=len(candles_4h),
                    required_4h=self.min_4h,
                ),
            )

        if len(candles_1h) < self.min_1h:
            return self._hold(
                current_price,
                f"Insufficient 1h data ({len(candles_1h)} bars < {self.min_1h})",
                metadata=self._decision_meta(
                    stage="data_guard",
                    reason_code="insufficient_1h_data",
                    actual_1h=len(candles_1h),
                    required_1h=self.min_1h,
                ),
            )

        regime, regime_debug = detect_regime(candles_4h, min_candles=self.min_4h)
        if regime == RegimeType.LOW_CONVICTION:
            return self._hold(
                current_price,
                "Regime=LOW_CONVICTION (ADX weak) — skipping entry",
                metadata=self._decision_meta(
                    stage="regime_filter",
                    reason_code="low_conviction_regime",
                    regime=regime,
                    regime_debug=regime_debug,
                ),
            )

        vol_valid, vol_ratio = volume_ok(
            candles, period=self.vol_period, threshold=self.vol_threshold
        )
        if not vol_valid:
            return self._hold(
                current_price,
                f"Volume too low (ratio={vol_ratio:.2f} < {self.vol_threshold})",
                metadata=self._decision_meta(
                    stage="volume_filter",
                    reason_code="volume_below_threshold",
                    regime=regime,
                    regime_debug=regime_debug,
                    vol_ratio=vol_ratio,
                ),
            )

        bias_score, bias_debug = directional_bias(candles_1h, min_candles=self.min_1h)
        threshold = entry_threshold(regime)

        long_score, long_factors, long_debug = _score_factors(
            candles, "long", regime, bias_score, min_candles=self.min_15m
        )
        short_score, short_factors, short_debug = _score_factors(
            candles, "short", regime, bias_score, min_candles=self.min_15m
        )

        if regime == RegimeType.TRENDING_UP:
            short_score = -1.0
        elif regime == RegimeType.TRENDING_DOWN:
            long_score = -1.0

        base_meta = self._decision_meta(
            stage="decision",
            reason_code="pending",
            regime=regime,
            regime_debug=regime_debug,
            bias_score=bias_score,
            bias_debug=bias_debug,
            vol_ratio=vol_ratio,
            threshold=threshold,
            long_score=long_score,
            short_score=short_score,
            long_factors=long_factors,
            short_factors=short_factors,
            long_debug=long_debug,
            short_debug=short_debug,
        )

        if regime == RegimeType.TRENDING_UP and bias_score < BIAS_MISALIGN_LONG_MIN:
            return self._hold(
                current_price,
                f"Regime={regime.value} but 1h bias={bias_score:.1f} — misaligned",
                metadata=self._decision_meta(
                    stage="bias_gate",
                    reason_code="trend_up_bias_misaligned",
                    **base_meta,
                ),
            )
        if regime == RegimeType.TRENDING_DOWN and bias_score > BIAS_MISALIGN_SHORT_MAX:
            return self._hold(
                current_price,
                f"Regime={regime.value} but 1h bias={bias_score:.1f} — misaligned",
                metadata=self._decision_meta(
                    stage="bias_gate",
                    reason_code="trend_down_bias_misaligned",
                    **base_meta,
                ),
            )

        long_ok = long_score >= threshold
        short_ok = short_score >= threshold
        score_gap = abs(long_score - short_score)
        is_tie = score_gap <= self.tie_tolerance

        decision_meta = self._decision_meta(
            **base_meta,
            eligible_long=long_ok,
            eligible_short=short_ok,
            score_gap=score_gap,
            tie_detected=is_tie,
        )

        if long_ok and short_ok:
            side = None
            if is_tie:
                side = self._resolve_tie(regime, bias_score)
            else:
                side = "long" if long_score > short_score else "short"

            if side is None:
                return self._hold(
                    current_price,
                    (f"Both sides valid but unresolved (long={long_score:.4f}, short={short_score:.4f}, "
                     f"tolerance={self.tie_tolerance})"),
                    metadata=self._decision_meta(
                        stage="tie_break",
                        reason_code="both_sides_valid_unresolved",
                        tie_break_policy=self.tie_break_policy,
                        **decision_meta,
                    ),
                )

            chosen_score = long_score if side == "long" else short_score
            chosen_factors = long_factors if side == "long" else short_factors
            chosen_debug = long_debug if side == "long" else short_debug
            return self._build_signal(
                candles,
                current_price,
                side,
                chosen_score,
                chosen_factors,
                chosen_debug,
                regime,
                bias_score,
                self._decision_meta(
                    stage="signal_build",
                    reason_code="both_sides_valid_resolved",
                    tie_break_policy=self.tie_break_policy,
                    tie_break_used=is_tie,
                    **decision_meta,
                ),
                threshold,
                long_score,
                short_score,
            )

        if long_ok:
            return self._build_signal(
                candles,
                current_price,
                "long",
                long_score,
                long_factors,
                long_debug,
                regime,
                bias_score,
                self._decision_meta(
                    stage="signal_build",
                    reason_code="long_signal_selected",
                    **decision_meta,
                ),
                threshold,
                long_score,
                short_score,
            )

        if short_ok:
            return self._build_signal(
                candles,
                current_price,
                "short",
                short_score,
                short_factors,
                short_debug,
                regime,
                bias_score,
                self._decision_meta(
                    stage="signal_build",
                    reason_code="short_signal_selected",
                    **decision_meta,
                ),
                threshold,
                long_score,
                short_score,
            )

        best_side = "long" if long_score >= short_score else "short"
        best_score = max(long_score, short_score)
        return self._hold(
            current_price,
            f"Score below threshold ({best_side}={best_score:.2f} < {threshold:.2f}) regime={regime.value}",
            metadata=self._decision_meta(
                stage="decision",
                reason_code="score_below_threshold",
                best_side=best_side,
                best_score=best_score,
                **decision_meta,
            ),
        )

    def _resolve_tie(self, regime: RegimeType, bias_score: float):
        """
        Resolve a long/short score tie according to `self.tie_break_policy`.

        Returns "long", "short", or None (None => stay flat / HOLD).
        """
        policy = (self.tie_break_policy or "hold").lower()

        if policy == "hold":
            return None

        if policy == "bias":
            if bias_score > 0:
                return "long"
            if bias_score < 0:
                return "short"
            return None

        if policy == "regime":
            if regime == RegimeType.TRENDING_UP:
                return "long"
            if regime == RegimeType.TRENDING_DOWN:
                return "short"
            if bias_score > 0:
                return "long"
            if bias_score < 0:
                return "short"
            return None

        return None

    def _sanitize_bool(self, key: str, default: bool) -> bool:
        value = self.params.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        self._param_warnings.append(f"{key} invalid; fallback={default}")
        return default

    def _sanitize_int(self, key: str, default: int, minimum: int = None, maximum: int = None) -> int:
        value = self.params.get(key, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            self._param_warnings.append(f"{key} invalid; fallback={default}")
            return default
        if minimum is not None and parsed < minimum:
            self._param_warnings.append(f"{key}={parsed} below minimum {minimum}; fallback={default}")
            return default
        if maximum is not None and parsed > maximum:
            self._param_warnings.append(f"{key}={parsed} above maximum {maximum}; fallback={default}")
            return default
        return parsed

    def _sanitize_float(
        self,
        key: str,
        default: float,
        minimum: float = None,
        maximum: float = None,
        strict_min: bool = False,
    ) -> float:
        value = self.params.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            self._param_warnings.append(f"{key} invalid; fallback={default}")
            return default
        if math.isnan(parsed) or math.isinf(parsed):
            self._param_warnings.append(f"{key} non-finite; fallback={default}")
            return default
        if minimum is not None:
            if strict_min and parsed <= minimum:
                self._param_warnings.append(f"{key}={parsed} must be > {minimum}; fallback={default}")
                return default
            if not strict_min and parsed < minimum:
                self._param_warnings.append(f"{key}={parsed} below minimum {minimum}; fallback={default}")
                return default
        if maximum is not None and parsed > maximum:
            self._param_warnings.append(f"{key}={parsed} above maximum {maximum}; fallback={default}")
            return default
        return parsed

    def _hold(self, price: float, reason: str, metadata: dict = None) -> Signal:
        return Signal(
            type=SignalType.HOLD,
            symbol=self.symbol,
            price=price,
            amount=0,
            reason=f"[MTF] {reason}",
            confidence=0.0,
            metadata=self._decision_meta(decision="hold", **(metadata or {})),
        )

    def _decision_meta(self, **extra) -> dict:
        meta = {
            "strategy": "AISignalStrategy",
            "api_free": True,
            "tie_break_policy": self.tie_break_policy,
            "position_pct": self.position_pct,
            "vol_period": self.vol_period,
            "vol_threshold": self.vol_threshold,
            "min_15m": self.min_15m,
            "min_1h": self.min_1h,
            "min_4h": self.min_4h,
            "atr_guard_enabled": self.atr_guard_enabled,
            "atr_min_value": self.atr_min_value,
            "param_warnings": list(self._param_warnings),
        }
        meta.update(extra)
        if "regime" in meta and isinstance(meta["regime"], RegimeType):
            meta["regime"] = meta["regime"].value
        if "bias_score" in meta and meta["bias_score"] is not None:
            meta["bias_score"] = round(float(meta["bias_score"]), 3)
        if "vol_ratio" in meta and meta["vol_ratio"] is not None:
            meta["vol_ratio"] = round(float(meta["vol_ratio"]), 4)
        if "long_score" in meta and meta["long_score"] is not None:
            meta["long_score"] = round(float(meta["long_score"]), 4)
        if "short_score" in meta and meta["short_score"] is not None:
            meta["short_score"] = round(float(meta["short_score"]), 4)
        if "threshold" in meta and meta["threshold"] is not None:
            meta["threshold"] = round(float(meta["threshold"]), 4)
        if "score_gap" in meta and meta["score_gap"] is not None:
            meta["score_gap"] = round(float(meta["score_gap"]), 6)
        return meta

    def _build_signal(
        self,
        candles: list,
        price: float,
        side: str,
        score: float,
        factors: list[str],
        entry_debug: dict,
        regime: RegimeType,
        bias_score: float,
        base_meta: dict,
        threshold: float,
        long_score: float,
        short_score: float,
    ) -> Signal:
        sig_type = SignalType.BUY if side == "long" else SignalType.SELL

        conf = self._compute_confidence(
            score=score,
            threshold=threshold,
            side=side,
            regime=regime,
            bias_score=bias_score,
            long_score=long_score,
            short_score=short_score,
        )

        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if len(atr_arr) > 0 and not np.isnan(atr_arr[-1]) else 0.0

        if self.atr_guard_enabled and atr_val <= self.atr_min_value:
            return self._hold(
                price,
                f"Invalid ATR for trade plan (atr={atr_val:.10f}) — skipping entry",
                metadata=self._decision_meta(
                    stage="trade_plan",
                    reason_code="invalid_atr",
                    side=side,
                    entry_score=score,
                    entry_factors=factors,
                    entry_debug=entry_debug,
                    atr_value=atr_val,
                    **base_meta,
                ),
            )

        trade_plan = build_trade_plan(price, atr_val, side)
        uses_risk_managed_sizing = bool(trade_plan.get("sl_dist_pct"))

        sj = round(conf * 100 + score * 20, 2)
        factor_str = "+".join(factors[:4]) if factors else "no_factors"
        reason = (
            f"[MTF] {side.upper()} regime={regime.value} "
            f"score={score:.2f} bias={bias_score:.1f} [{factor_str}]"
        )

        metadata = self._decision_meta(
            decision="buy" if side == "long" else "sell",
            side=side,
            entry_factors=factors,
            entry_debug=entry_debug,
            entry_score=score,
            confidence=conf,
            confidence_components=self._confidence_breakdown(
                score=score,
                threshold=threshold,
                side=side,
                regime=regime,
                bias_score=bias_score,
                long_score=long_score,
                short_score=short_score,
            ),
            sj_score=sj,
            atr_value=atr_val,
            sizing_mode="risk_manager" if uses_risk_managed_sizing else "fixed_fraction",
            requested_amount=self.position_pct,
            amount_is_placeholder=uses_risk_managed_sizing,
            sizing_note=(
                "amount=0 intentionally delegates sizing to downstream risk manager via sl_dist_pct/risk_pct"
                if uses_risk_managed_sizing
                else "amount uses fixed legacy position_pct"
            ),
            **base_meta,
            **trade_plan,
        )

        return Signal(
            type=sig_type,
            symbol=self.symbol,
            price=price,
            amount=0 if uses_risk_managed_sizing else self.position_pct,
            reason=reason,
            confidence=conf,
            metadata=metadata,
        )

    def _compute_confidence(
        self,
        score: float,
        threshold: float,
        side: str,
        regime: RegimeType,
        bias_score: float,
        long_score: float,
        short_score: float,
    ) -> float:
        components = self._confidence_breakdown(
            score=score,
            threshold=threshold,
            side=side,
            regime=regime,
            bias_score=bias_score,
            long_score=long_score,
            short_score=short_score,
        )
        conf = (
            components["base"]
            + components["edge_bonus"]
            + components["separation_bonus"]
            + components["regime_bonus"]
            + components["bias_bonus"]
        )
        return round(max(0.0, min(conf, 0.97)), 4)

    def _confidence_breakdown(
        self,
        score: float,
        threshold: float,
        side: str,
        regime: RegimeType,
        bias_score: float,
        long_score: float,
        short_score: float,
    ) -> dict:
        edge = max(0.0, score - threshold)
        separation = abs(long_score - short_score)
        base = 0.50
        edge_bonus = min(edge * 0.18, 0.18)
        separation_bonus = min(separation * 0.08, 0.12)
        regime_bonus = 0.0
        if (regime == RegimeType.TRENDING_UP and side == "long") or (
            regime == RegimeType.TRENDING_DOWN and side == "short"
        ):
            regime_bonus = 0.07
        bias_bonus = 0.0
        if (side == "long" and bias_score > 0) or (side == "short" and bias_score < 0):
            bias_bonus = min(abs(bias_score) * 0.015, 0.06)
        return {
            "base": round(base, 4),
            "edge": round(edge, 4),
            "edge_bonus": round(edge_bonus, 4),
            "separation": round(separation, 4),
            "separation_bonus": round(separation_bonus, 4),
            "regime_bonus": round(regime_bonus, 4),
            "bias_bonus": round(bias_bonus, 4),
        }
