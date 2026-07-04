"""
AI Signal strategy v3 — API-free, multi-timeframe, regime-adaptive (hardened).

Architecture (top-down):
  4h candles  → detect_regime()      — TRENDING/RANGING/VOLATILE/LOW_CONVICTION
  1h candles  → directional_bias()   — bull/bear bias score  (-3 … +3)
  15m candles → volume_ok()          — skip entry when volume is too thin
  15m candles → _score_factors()     — entry trigger score (RSI/MACD/ST/Vol/EMA/HA)
  price+ATR   → build_trade_plan()   — SL=1R, T1=0.5R … T4=1.2R, SL-ratchet ladder

The Signal.metadata returned is fully compatible with TradingBot._open_position():
  sl_ladder_enabled=True activates SL-ratchet mode (no partial closes; SL steps
  up as each ladder level is hit: T1→+0.3R, T2→+0.5R, T3→+0.8R, T4→full close).

Improvements vs v2:
  - Explicit min-candles guards for 1h/4h before regime/bias compute (previously
    relied on downstream functions to silently handle insufficient data).
  - ATR invalid guard (<=0 or NaN) -> HOLD instead of building a malformed
    trade plan with a zero/garbage ATR value.
  - Deterministic tie-break policy (configurable via `tie_break_policy`):
    "hold" (default, safest), "bias", or "regime".
  - Richer telemetry metadata (threshold, atr_value, tie_break_policy, etc.)
    for easier debugging / analytics.
"""
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

# Minimum bias alignment required to trade against the 1h bias
# (now imported from mtf_regime: BIAS_MISALIGN_LONG_MIN / BIAS_MISALIGN_SHORT_MAX)


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

    # Tells the bot to also fetch "1h" and "4h" candles and pass them via mtf_candles.
    MTF_TIMEFRAMES = ["1h", "4h"]
    DEFAULT_POSITION_PCT = 0.05
    DEFAULT_VOL_PERIOD = 20
    DEFAULT_VOL_THRESHOLD = 0.70
    DEFAULT_MIN_15M = 40
    DEFAULT_MIN_1H = 55
    DEFAULT_MIN_4H = 40
    DEFAULT_TIE_BREAK_POLICY = "hold"
    ALLOWED_TIE_BREAK_POLICIES = {"hold", "bias", "regime"}
    DEFAULT_TIE_TOLERANCE = 1e-6
    DEFAULT_ATR_GUARD_ENABLED = True
    DEFAULT_ATR_MIN_VALUE = 1e-10
    CONF_SCORE_NORMALIZER = 4.0
    CONF_EDGE_NORMALIZER = 2.0
    CONF_BASE = 0.35
    CONF_SCORE_WEIGHT = 0.35
    CONF_THRESHOLD_WEIGHT = 0.20
    CONF_SEPARATION_WEIGHT = 0.10

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.position_pct = self._safe_float_in_range(
            self.params.get("position_pct"),
            self.DEFAULT_POSITION_PCT,
            lo=0.0,
            hi=1.0,
            include_lo=False,
            include_hi=True,
        )
        self.vol_period = self._safe_int_min(
            self.params.get("vol_period"),
            self.DEFAULT_VOL_PERIOD,
            min_value=1,
        )
        self.vol_threshold = self._safe_float_in_range(
            self.params.get("vol_threshold"),
            self.DEFAULT_VOL_THRESHOLD,
            lo=0.0,
            hi=10.0,
            include_lo=True,
            include_hi=True,
        )
        self.min_15m = self._safe_int_min(
            self.params.get("min_15m"),
            self.DEFAULT_MIN_15M,
            min_value=1,
        )
        self.min_1h = self._safe_int_min(
            self.params.get("min_1h"),
            self.DEFAULT_MIN_1H,
            min_value=1,
        )
        self.min_4h = self._safe_int_min(
            self.params.get("min_4h"),
            self.DEFAULT_MIN_4H,
            min_value=1,
        )

        # Tie-break configuration
        self.tie_break_policy = self._safe_choice(
            self.params.get("tie_break_policy"),
            self.DEFAULT_TIE_BREAK_POLICY,
            self.ALLOWED_TIE_BREAK_POLICIES,
        )
        self.tie_tolerance = self._safe_float_in_range(
            self.params.get("tie_tolerance"),
            self.DEFAULT_TIE_TOLERANCE,
            lo=0.0,
            hi=1.0,
            include_lo=True,
            include_hi=True,
        )

        # ATR guard configuration
        self.atr_guard_enabled = self._safe_bool(
            self.params.get("atr_guard_enabled"),
            self.DEFAULT_ATR_GUARD_ENABLED,
        )
        self.atr_min_value = self._safe_float_in_range(
            self.params.get("atr_min_value"),
            self.DEFAULT_ATR_MIN_VALUE,
            lo=0.0,
            hi=10.0,
            include_lo=True,
            include_hi=True,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    async def analyze(
        self,
        candles: list,           # 15m candles (primary entry TF)
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        """
        Produce a BUY / SELL / HOLD signal with full metadata.

        Flow:
          1. Guard: require enough candles on all timeframes (15m/1h/4h).
          2. Regime detection (4h).  LOW_CONVICTION → HOLD immediately.
          3. Volume filter (15m).    Low volume → HOLD.
          4. Directional bias (1h).
          5. Entry scoring (15m) for the regime-preferred direction(s).
          6. Decide signal type and confidence, resolving ties deterministically.
          7. Build TP/SL ladder trade plan (guarded against invalid ATR).
          8. Return Signal with rich metadata.
        """
        mtf = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])

        # ── 1. Data guards ────────────────────────────────────────────────────
        if len(candles) < self.min_15m:
            return self._hold(
                current_price,
                f"Insufficient 15m data ({len(candles)} bars)",
                metadata={"required_15m": self.min_15m, "actual_15m": len(candles)},
                stage="guard",
                reason_code="insufficient_15m",
            )

        if len(candles_4h) < self.min_4h:
            return self._hold(
                current_price,
                f"Insufficient 4h data ({len(candles_4h)} bars < {self.min_4h})",
                metadata={
                    "required_4h": self.min_4h,
                    "actual_4h":   len(candles_4h),
                },
                stage="guard",
                reason_code="insufficient_4h",
            )

        if len(candles_1h) < self.min_1h:
            return self._hold(
                current_price,
                f"Insufficient 1h data ({len(candles_1h)} bars < {self.min_1h})",
                metadata={
                    "required_1h": self.min_1h,
                    "actual_1h":   len(candles_1h),
                },
                stage="guard",
                reason_code="insufficient_1h",
            )

        # ── 2. Market regime (4h) ─────────────────────────────────────────────
        regime, regime_debug = detect_regime(candles_4h, min_candles=self.min_4h)

        if regime == RegimeType.LOW_CONVICTION:
            return self._hold(
                current_price,
                f"Regime=LOW_CONVICTION (ADX weak) — skipping entry",
                metadata={"regime": regime.value, "regime_debug": regime_debug},
                stage="regime",
                reason_code="low_conviction_regime",
            )

        # ── 3. Volume filter (15m) ────────────────────────────────────────────
        vol_valid, vol_ratio = volume_ok(
            candles, period=self.vol_period, threshold=self.vol_threshold
        )
        if not vol_valid:
            return self._hold(
                current_price,
                f"Volume too low (ratio={vol_ratio:.2f} < {self.vol_threshold})",
                metadata={
                    "regime": regime.value,
                    "vol_ratio": vol_ratio,
                    "regime_debug": regime_debug,
                },
                stage="volume",
                reason_code="volume_below_threshold",
            )

        # ── 4. Directional bias (1h) ──────────────────────────────────────────
        bias_score, bias_debug = directional_bias(candles_1h, min_candles=self.min_1h)

        # ── 5. Entry scoring (15m) per regime ────────────────────────────────
        threshold = entry_threshold(regime)

        long_score, long_factors, long_debug = _score_factors(
            candles, "long", regime, bias_score, min_candles=self.min_15m
        )
        short_score, short_factors, short_debug = _score_factors(
            candles, "short", regime, bias_score, min_candles=self.min_15m
        )

        # ── 6. Direction eligibility by regime ───────────────────────────────
        # TRENDING_UP  → prefer long, disallow short
        # TRENDING_DOWN → prefer short, disallow long
        # RANGING       → both directions allowed (mean reversion)
        # VOLATILE      → both but higher threshold (already reflected in threshold)
        if regime == RegimeType.TRENDING_UP:
            short_score = -1.0   # disabled in strong uptrend
        elif regime == RegimeType.TRENDING_DOWN:
            long_score  = -1.0   # disabled in strong downtrend

        # ── Bias alignment gate ────────────────────────────────────────────────
        # In RANGING regime the 1h bias can support either direction.
        # In TRENDING regimes require the 1h bias to not actively oppose direction.
        if regime == RegimeType.TRENDING_UP and bias_score < BIAS_MISALIGN_LONG_MIN:
            # Strong 1h bear bias in a 4h uptrend → wait for alignment
            return self._hold(
                current_price,
                f"Regime={regime.value} but 1h bias={bias_score:.1f} — misaligned",
                metadata=self._base_meta(regime, regime_debug, bias_score, bias_debug,
                                         vol_ratio, long_score, short_score),
                stage="bias",
                reason_code="bias_misaligned_long",
            )
        if regime == RegimeType.TRENDING_DOWN and bias_score > BIAS_MISALIGN_SHORT_MAX:
            return self._hold(
                current_price,
                f"Regime={regime.value} but 1h bias={bias_score:.1f} — misaligned",
                metadata=self._base_meta(regime, regime_debug, bias_score, bias_debug,
                                         vol_ratio, long_score, short_score),
                stage="bias",
                reason_code="bias_misaligned_short",
            )

        # ── 7. Signal decision ────────────────────────────────────────────────
        base_meta = self._base_meta(
            regime, regime_debug, bias_score, bias_debug,
            vol_ratio, long_score, short_score,
        )
        base_meta["threshold"] = round(float(threshold), 4)

        long_ok  = long_score  >= threshold
        short_ok = short_score >= threshold
        score_delta = long_score - short_score
        score_gap = abs(score_delta)
        is_exact_tie = long_score == short_score
        is_near_tie = score_gap <= self.tie_tolerance
        is_tie = is_exact_tie or is_near_tie

        # ── 7a. Deterministic tie handling ───────────────────────────────────
        # Any both-valid state is resolved deterministically:
        #  - near/equal scores use tie policy
        #  - otherwise pick stronger side directly
        if long_ok and short_ok:
            side = None
            tie_break_used = False
            if is_tie:
                side = self._resolve_tie(regime, bias_score)
                tie_break_used = True
            elif score_delta > 0:
                side = "long"
            elif score_delta < 0:
                side = "short"
            else:
                # Defensive fallback: if both sides are valid and score ordering is
                # still unresolved, pick deterministically instead of falling through.
                side = "long" if long_score >= short_score else "short"

            if side is None:
                return self._hold(
                    current_price,
                    (f"Both sides valid but unresolved (long={long_score:.4f}, "
                     f"short={short_score:.4f}, tolerance={self.tie_tolerance})"),
                    metadata={
                        **base_meta,
                        "long_factors":     long_factors,
                        "short_factors":    short_factors,
                        "long_debug":       long_debug,
                        "short_debug":      short_debug,
                        "tie_break_policy": self.tie_break_policy,
                        "tie_detected":     is_tie,
                    },
                    stage="decision",
                    reason_code="both_valid_unresolved",
                )
            if side == "long":
                return self._build_signal(
                    candles, current_price, "long",
                    long_score, long_factors, long_debug,
                    regime, bias_score, short_score,
                    {**base_meta, "tie_detected": is_tie, "tie_break_used": tie_break_used},
                )
            return self._build_signal(
                candles, current_price, "short",
                short_score, short_factors, short_debug,
                regime, bias_score, long_score,
                {**base_meta, "tie_detected": is_tie, "tie_break_used": tie_break_used},
            )

        if long_ok and long_score > short_score:
            return self._build_signal(
                candles, current_price, "long",
                long_score, long_factors, long_debug,
                regime, bias_score, short_score, base_meta,
            )

        if short_ok and short_score > long_score:
            return self._build_signal(
                candles, current_price, "short",
                short_score, short_factors, short_debug,
                regime, bias_score, long_score, base_meta,
            )

        # Neither side clears threshold
        best_side  = "long" if long_score >= short_score else "short"
        best_score = max(long_score, short_score)
        return self._hold(
            current_price,
            (f"Score below threshold ({best_side}={best_score:.2f} < {threshold:.2f}) "
             f"regime={regime.value}"),
            metadata={
                **base_meta,
                "long_factors":  long_factors,
                "short_factors": short_factors,
                "long_debug":    long_debug,
                "short_debug":   short_debug,
            },
            stage="decision",
            reason_code="below_threshold",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

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
            return None  # bias is exactly neutral → still ambiguous

        if policy == "regime":
            if regime == RegimeType.TRENDING_UP:
                return "long"
            if regime == RegimeType.TRENDING_DOWN:
                return "short"
            # RANGING / VOLATILE have no directional regime preference —
            # fall back to bias as a secondary tiebreaker.
            if bias_score > 0:
                return "long"
            if bias_score < 0:
                return "short"
            return None

        # Unknown policy value → safest fallback is to stay flat.
        return None

    @staticmethod
    def _safe_int_min(value: object, default: int, min_value: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= min_value else default

    @staticmethod
    def _safe_float_in_range(
        value: object,
        default: float,
        lo: float,
        hi: float,
        include_lo: bool = True,
        include_hi: bool = True,
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(parsed):
            return default
        if parsed < lo or (parsed == lo and not include_lo):
            return default
        if parsed > hi or (parsed == hi and not include_hi):
            return default
        return parsed

    @staticmethod
    def _safe_choice(value: object, default: str, allowed: set) -> str:
        if not isinstance(value, str):
            return default
        normalized = value.strip().lower()
        return normalized if normalized in allowed else default

    @staticmethod
    def _safe_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _decision_meta(
        decision: str,
        stage: str,
        reason_code: str,
        metadata: dict = None,
        **extra,
    ) -> dict:
        merged = dict(metadata or {})
        merged.update(
            {
                "decision": decision,
                "decision_stage": stage,
                "decision_reason_code": reason_code,
            }
        )
        if extra:
            merged.update(extra)
        return merged

    def _hold(
        self,
        price: float,
        reason: str,
        metadata: dict = None,
        stage: str = "decision",
        reason_code: str = "hold",
    ) -> Signal:
        return Signal(
            type=SignalType.HOLD,
            symbol=self.symbol,
            price=price,
            amount=0,
            reason=f"[MTF] {reason}",
            confidence=0.0,
            metadata=self._decision_meta(
                decision=SignalType.HOLD.value,
                stage=stage,
                reason_code=reason_code,
                metadata=metadata,
            ),
        )

    @staticmethod
    def _base_meta(
        regime: RegimeType, regime_debug: dict,
        bias_score: float, bias_debug: dict,
        vol_ratio: float,
        long_score: float, short_score: float,
    ) -> dict:
        return {
            "regime":       regime.value,
            "regime_debug": regime_debug,
            "bias_score":   round(bias_score, 3),
            "bias_debug":   bias_debug,
            "vol_ratio":    vol_ratio,
            "long_score":   round(long_score,  4),
            "short_score":  round(short_score, 4),
        }

    def _build_signal(
        self,
        candles: list,
        price: float,
        side: str,           # "long" | "short"
        score: float,
        factors: list[str],
        entry_debug: dict,
        regime: RegimeType,
        bias_score: float,
        opposing_score: float,
        base_meta: dict,
    ) -> Signal:
        """Build a BUY/SELL Signal with trade plan metadata."""
        sig_type = SignalType.BUY if side == "long" else SignalType.SELL

        # ── Confidence ────────────────────────────────────────────────────────
        # Confidence combines raw score quality, threshold distance, and
        # separation from the opposite side score in a bounded/simple formula.
        threshold = float(base_meta.get("threshold", 0.0))
        score_component = np.clip(score / self.CONF_SCORE_NORMALIZER, 0.0, 1.0)
        threshold_edge = max(score - threshold, 0.0)
        threshold_component = np.clip(threshold_edge / self.CONF_EDGE_NORMALIZER, 0.0, 1.0)
        separation = max(score - opposing_score, 0.0)
        separation_component = np.clip(separation / self.CONF_EDGE_NORMALIZER, 0.0, 1.0)

        conf = (
            self.CONF_BASE
            + self.CONF_SCORE_WEIGHT * score_component
            + self.CONF_THRESHOLD_WEIGHT * threshold_component
            + self.CONF_SEPARATION_WEIGHT * separation_component
        )
        if regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN):
            if (regime == RegimeType.TRENDING_UP and side == "long") or \
               (regime == RegimeType.TRENDING_DOWN and side == "short"):
                conf = min(conf + 0.05, 0.95)   # regime alignment bonus
        bias_aligned = (side == "long" and bias_score > 0) or \
                       (side == "short" and bias_score < 0)
        if bias_aligned:
            conf = min(conf + 0.03, 0.95)
        conf = round(conf, 4)

        # ── Trade plan (ATR-based SL/TP ladder) ───────────────────────────────
        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if len(atr_arr) > 0 and not np.isnan(atr_arr[-1]) else 0.0

        # ── ATR validity guard ────────────────────────────────────────────────
        # An ATR of zero (or effectively zero / NaN) would produce a malformed
        # trade plan (e.g. zero-width SL/TP). Prefer a safe HOLD over opening a
        # position with a broken risk plan.
        if self.atr_guard_enabled and atr_val <= self.atr_min_value:
            return self._hold(
                price,
                f"Invalid ATR for trade plan (atr={atr_val:.10f}) — skipping entry",
                metadata={
                    **base_meta,
                    "side":               side,
                    "entry_score":        round(score, 4),
                    "entry_factors":      factors,
                    "entry_debug":        entry_debug,
                    "atr_value":          atr_val,
                    "atr_guard_enabled":  self.atr_guard_enabled,
                    "atr_min_value":      self.atr_min_value,
                },
                stage="trade_plan",
                reason_code="invalid_atr",
            )

        trade_plan = build_trade_plan(price, atr_val, side)

        # ── sj_score for bot trade-ranking (higher = better quality) ─────────
        sj = round(conf * 100 + score * 20, 2)
        uses_sl_based_sizing = trade_plan.get("sl_dist_pct") is not None
        sizing_mode = "risk_managed" if uses_sl_based_sizing else "fixed_fraction"
        amount = 0 if uses_sl_based_sizing else self.position_pct

        # ── Reason string ──────────────────────────────────────────────────────
        factor_str = "+".join(factors[:4]) if factors else "no_factors"
        reason = (
            f"[MTF] {side.upper()} regime={regime.value} "
            f"score={score:.2f} bias={bias_score:.1f} [{factor_str}]"
        )

        metadata = {
            **base_meta,
            **trade_plan,
            "entry_factors":     factors,
            "entry_debug":       entry_debug,
            "entry_score":       round(score, 4),
            "side":              side,
            "sj_score":          sj,
            "api_free":          True,
            "atr_value":         atr_val,
            "tie_break_policy":  self.tie_break_policy,
            "sizing_mode":       sizing_mode,
            "position_pct":      self.position_pct,
            "risk_sizing_enabled": uses_sl_based_sizing,
            "confidence_breakdown": {
                "score_component": round(score_component, 4),
                "threshold_component": round(threshold_component, 4),
                "separation_component": round(separation_component, 4),
                "score_threshold_edge": round(threshold_edge, 4),
                "score_separation": round(separation, 4),
            },
        }
        reason_code = "signal_long" if side == "long" else "signal_short"
        metadata = self._decision_meta(
            decision=sig_type.value,
            stage="entry",
            reason_code=reason_code,
            metadata=metadata,
        )

        return Signal(
            type=sig_type,
            symbol=self.symbol,
            price=price,
            # Keep integration compatibility: when trade_plan includes sl_dist_pct,
            # RiskManager computes size and `amount=0` is an intentional placeholder.
            amount=amount,
            reason=reason,
            confidence=conf,
            metadata=metadata,
        )
