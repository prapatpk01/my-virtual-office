"""
AI Signal strategy v7 — API-free, multi-timeframe, regime-adaptive with presets,
reversal probes, and execution telemetry.

Architecture (top-down):
  4h candles  → detect_regime()         — TRENDING/RANGING/VOLATILE/LOW_CONVICTION
  1h candles  → directional_bias()      — bull/bear bias score (-3 … +3)
  15m candles → volume_ok()             — skip entry when volume is too thin
  15m candles → _score_factors()        — entry trigger score (RSI/MACD/ST/Vol/EMA/HA)
  score layer → adaptive thresholding   — context-aware trigger sensitivity
  profile     → preset tuning           — all_weather / aggressive / sniper
  price+ATR   → build_trade_plan()      — SL=1R, T1=0.5R … T4=1.2R, SL-ratchet ladder

Execution goals:
  - Stay tradable across trending, ranging, and volatile regimes.
  - Balance responsiveness with trade quality by default.
  - Allow profile-based tuning without changing public class/interface.
  - Emit rich telemetry so live-debugging, backtests, and analytics can explain every decision.
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
    MTF_TIMEFRAMES = ["1h", "4h"]

    BASE_DEFAULTS = {
        "profile": "all_weather",
        "position_pct": 0.05,
        "vol_period": 20,
        "vol_threshold": 0.70,
        "min_15m": 40,
        "min_1h": 55,
        "min_4h": 40,
        "tie_break_policy": "bias",
        "tie_tolerance": 0.03,
        "atr_guard_enabled": True,
        "atr_min_value": 1e-10,
        "fast_entry_enabled": True,
        "fast_entry_max_bonus": 0.18,
        "countertrend_penalty": 0.26,
        "min_score_separation": 0.22,
        "range_threshold_discount": 0.08,
        "volatile_threshold_buffer": 0.10,
        "trend_threshold_discount": 0.06,
        "bias_neutral_band": 0.35,
        "min_confidence_floor": 0.52,
        "reversal_probe_enabled": False,
        "reversal_probe_max_bonus": 0.0,
        "reentry_cooldown_bars": 0,
        "max_same_side_reentries": 0,
    }
    PROFILE_OVERRIDES = {
        "all_weather": {},
        "aggressive": {
            "tie_break_policy": "bias",
            "tie_tolerance": 0.02,
            "fast_entry_max_bonus": 0.30,
            "countertrend_penalty": 0.18,
            "min_score_separation": 0.14,
            "range_threshold_discount": 0.10,
            "volatile_threshold_buffer": 0.06,
            "trend_threshold_discount": 0.09,
            "bias_neutral_band": 0.28,
            "min_confidence_floor": 0.50,
            "reversal_probe_enabled": True,
            "reversal_probe_max_bonus": 0.06,
        },
        "sniper": {
            "tie_break_policy": "hold",
            "tie_tolerance": 0.01,
            "fast_entry_max_bonus": 0.10,
            "countertrend_penalty": 0.34,
            "min_score_separation": 0.32,
            "range_threshold_discount": 0.04,
            "volatile_threshold_buffer": 0.14,
            "trend_threshold_discount": 0.04,
            "bias_neutral_band": 0.20,
            "min_confidence_floor": 0.58,
            "reversal_probe_enabled": False,
            "reversal_probe_max_bonus": 0.0,
        },
    }
    VALID_TIE_BREAK_POLICIES = {"hold", "bias", "regime"}
    VALID_PROFILES = set(PROFILE_OVERRIDES.keys())

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)

        self._param_warnings = []
        self._runtime_state = {
            "last_signal_side": None,
            "last_signal_bar_index": None,
            "same_side_reentries": 0,
        }
        self.profile = self._resolve_profile()
        self.DEFAULTS = self._build_defaults_for_profile(self.profile)

        self.position_pct = self._sanitize_float(
            "position_pct", self.DEFAULTS["position_pct"], minimum=0.0, maximum=1.0, strict_min=True
        )
        self.vol_period = self._sanitize_int("vol_period", self.DEFAULTS["vol_period"], minimum=1)
        self.vol_threshold = self._sanitize_float(
            "vol_threshold", self.DEFAULTS["vol_threshold"], minimum=0.0, strict_min=True
        )
        self.min_15m = self._sanitize_int("min_15m", self.DEFAULTS["min_15m"], minimum=20)
        self.min_1h = self._sanitize_int("min_1h", self.DEFAULTS["min_1h"], minimum=20)
        self.min_4h = self._sanitize_int("min_4h", self.DEFAULTS["min_4h"], minimum=20)

        raw_policy = str(self.params.get("tie_break_policy", self.DEFAULTS["tie_break_policy"]))
        policy = raw_policy.strip().lower()
        if policy not in self.VALID_TIE_BREAK_POLICIES:
            self._param_warnings.append(
                f"tie_break_policy='{raw_policy}' invalid; fallback='{self.DEFAULTS['tie_break_policy']}'"
            )
            policy = self.DEFAULTS["tie_break_policy"]
        self.tie_break_policy = policy

        self.tie_tolerance = self._sanitize_float("tie_tolerance", self.DEFAULTS["tie_tolerance"], minimum=0.0)
        self.atr_guard_enabled = self._sanitize_bool("atr_guard_enabled", self.DEFAULTS["atr_guard_enabled"])
        self.atr_min_value = self._sanitize_float(
            "atr_min_value", self.DEFAULTS["atr_min_value"], minimum=0.0, strict_min=True
        )
        self.fast_entry_enabled = self._sanitize_bool("fast_entry_enabled", self.DEFAULTS["fast_entry_enabled"])
        self.fast_entry_max_bonus = self._sanitize_float(
            "fast_entry_max_bonus", self.DEFAULTS["fast_entry_max_bonus"], minimum=0.0
        )
        self.countertrend_penalty = self._sanitize_float(
            "countertrend_penalty", self.DEFAULTS["countertrend_penalty"], minimum=0.0
        )
        self.min_score_separation = self._sanitize_float(
            "min_score_separation", self.DEFAULTS["min_score_separation"], minimum=0.0
        )
        self.range_threshold_discount = self._sanitize_float(
            "range_threshold_discount", self.DEFAULTS["range_threshold_discount"], minimum=0.0
        )
        self.volatile_threshold_buffer = self._sanitize_float(
            "volatile_threshold_buffer", self.DEFAULTS["volatile_threshold_buffer"], minimum=0.0
        )
        self.trend_threshold_discount = self._sanitize_float(
            "trend_threshold_discount", self.DEFAULTS["trend_threshold_discount"], minimum=0.0
        )
        self.bias_neutral_band = self._sanitize_float(
            "bias_neutral_band", self.DEFAULTS["bias_neutral_band"], minimum=0.0
        )
        self.min_confidence_floor = self._sanitize_float(
            "min_confidence_floor", self.DEFAULTS["min_confidence_floor"], minimum=0.0, maximum=0.95
        )
        self.reversal_probe_enabled = self._sanitize_bool(
            "reversal_probe_enabled", self.DEFAULTS["reversal_probe_enabled"]
        )
        self.reversal_probe_max_bonus = self._sanitize_float(
            "reversal_probe_max_bonus", self.DEFAULTS["reversal_probe_max_bonus"], minimum=0.0
        )
        self.reentry_cooldown_bars = self._sanitize_int(
            "reentry_cooldown_bars", self.DEFAULTS["reentry_cooldown_bars"], minimum=0
        )
        self.max_same_side_reentries = self._sanitize_int(
            "max_same_side_reentries", self.DEFAULTS["max_same_side_reentries"], minimum=0
        )

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])
        current_bar_index = max(len(candles) - 1, 0)

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

        vol_valid, vol_ratio = volume_ok(candles, period=self.vol_period, threshold=self.vol_threshold)
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
        base_threshold = entry_threshold(regime)

        long_score, long_factors, long_debug = _score_factors(
            candles, "long", regime, bias_score, min_candles=self.min_15m
        )
        short_score, short_factors, short_debug = _score_factors(
            candles, "short", regime, bias_score, min_candles=self.min_15m
        )

        long_disabled_by_regime = False
        short_disabled_by_regime = False
        if regime == RegimeType.TRENDING_UP:
            if self.reversal_probe_enabled and bias_score <= BIAS_MISALIGN_SHORT_MAX:
                short_score = max(short_score - 0.35, -0.25)
            else:
                short_score = -1.0
                short_disabled_by_regime = True
        elif regime == RegimeType.TRENDING_DOWN:
            if self.reversal_probe_enabled and bias_score >= BIAS_MISALIGN_LONG_MIN:
                long_score = max(long_score - 0.35, -0.25)
            else:
                long_score = -1.0
                long_disabled_by_regime = True

        long_threshold, long_threshold_meta = self._effective_threshold(
            side="long",
            regime=regime,
            base_threshold=base_threshold,
            bias_score=bias_score,
            vol_ratio=vol_ratio,
            score=long_score,
            opposing_score=short_score,
        )
        short_threshold, short_threshold_meta = self._effective_threshold(
            side="short",
            regime=regime,
            base_threshold=base_threshold,
            bias_score=bias_score,
            vol_ratio=vol_ratio,
            score=short_score,
            opposing_score=long_score,
        )

        base_meta = self._decision_meta(
            stage="decision",
            reason_code="pending",
            regime=regime,
            regime_debug=regime_debug,
            bias_score=bias_score,
            bias_debug=bias_debug,
            vol_ratio=vol_ratio,
            threshold=base_threshold,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            long_threshold_meta=long_threshold_meta,
            short_threshold_meta=short_threshold_meta,
            long_score=long_score,
            short_score=short_score,
            long_factors=long_factors,
            short_factors=short_factors,
            long_debug=long_debug,
            short_debug=short_debug,
            long_disabled_by_regime=long_disabled_by_regime,
            short_disabled_by_regime=short_disabled_by_regime,
            bar_index=current_bar_index,
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

        cooldown_blocked, cooldown_meta = self._cooldown_gate(current_bar_index)
        if cooldown_blocked:
            return self._hold(
                current_price,
                "Reentry cooldown active — skipping entry",
                metadata=self._decision_meta(
                    stage="cooldown_gate",
                    reason_code="reentry_cooldown_active",
                    **base_meta,
                    **cooldown_meta,
                ),
            )

        long_ok = long_score >= long_threshold
        short_ok = short_score >= short_threshold
        score_gap = abs(long_score - short_score)
        tie_band = max(self.tie_tolerance, self.min_score_separation)
        is_tie = score_gap <= self.tie_tolerance
        near_tie = score_gap <= tie_band

        decision_meta = self._decision_meta(
            **base_meta,
            eligible_long=long_ok,
            eligible_short=short_ok,
            score_gap=score_gap,
            tie_detected=is_tie,
            near_tie=near_tie,
            tie_band=tie_band,
            **cooldown_meta,
        )

        if long_ok and short_ok:
            side = None
            used_resolution = False
            resolution_reason = "both_sides_valid_resolved"

            if is_tie or near_tie:
                side = self._resolve_tie(regime, bias_score)
                used_resolution = True
                resolution_reason = "near_tie_resolved" if side is not None else "both_sides_valid_unresolved"
            else:
                side = "long" if long_score > short_score else "short"

            if side is None:
                return self._hold(
                    current_price,
                    (f"Both sides valid but unresolved (long={long_score:.4f}, short={short_score:.4f}, "
                     f"gap={score_gap:.4f})"),
                    metadata=self._decision_meta(
                        stage="tie_break",
                        reason_code="both_sides_valid_unresolved",
                        tie_break_policy=self.tie_break_policy,
                        **decision_meta,
                    ),
                )

            same_side_limit_hit, reentry_meta = self._same_side_reentry_gate(side)
            if same_side_limit_hit:
                return self._hold(
                    current_price,
                    f"Same-side reentry limit hit for {side}",
                    metadata=self._decision_meta(
                        stage="reentry_gate",
                        reason_code="same_side_reentry_limit",
                        side=side,
                        **decision_meta,
                        **reentry_meta,
                    ),
                )

            chosen_score = long_score if side == "long" else short_score
            chosen_factors = long_factors if side == "long" else short_factors
            chosen_debug = long_debug if side == "long" else short_debug
            chosen_threshold = long_threshold if side == "long" else short_threshold
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
                    reason_code=resolution_reason,
                    tie_break_policy=self.tie_break_policy,
                    tie_break_used=used_resolution,
                    **decision_meta,
                    **reentry_meta,
                ),
                chosen_threshold,
                long_score,
                short_score,
                current_bar_index,
            )

        if long_ok:
            same_side_limit_hit, reentry_meta = self._same_side_reentry_gate("long")
            if same_side_limit_hit:
                return self._hold(
                    current_price,
                    "Same-side reentry limit hit for long",
                    metadata=self._decision_meta(
                        stage="reentry_gate",
                        reason_code="same_side_reentry_limit",
                        side="long",
                        **decision_meta,
                        **reentry_meta,
                    ),
                )
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
                    **reentry_meta,
                ),
                long_threshold,
                long_score,
                short_score,
                current_bar_index,
            )

        if short_ok:
            same_side_limit_hit, reentry_meta = self._same_side_reentry_gate("short")
            if same_side_limit_hit:
                return self._hold(
                    current_price,
                    "Same-side reentry limit hit for short",
                    metadata=self._decision_meta(
                        stage="reentry_gate",
                        reason_code="same_side_reentry_limit",
                        side="short",
                        **decision_meta,
                        **reentry_meta,
                    ),
                )
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
                    **reentry_meta,
                ),
                short_threshold,
                long_score,
                short_score,
                current_bar_index,
            )

        best_side = "long" if long_score >= short_score else "short"
        best_score = max(long_score, short_score)
        best_threshold = long_threshold if best_side == "long" else short_threshold
        return self._hold(
            current_price,
            f"Score below threshold ({best_side}={best_score:.2f} < {best_threshold:.2f}) regime={regime.value}",
            metadata=self._decision_meta(
                stage="decision",
                reason_code="score_below_threshold",
                best_side=best_side,
                best_score=best_score,
                best_threshold=best_threshold,
                **decision_meta,
            ),
        )

    def _resolve_profile(self) -> str:
        raw_profile = str(self.params.get("profile", self.BASE_DEFAULTS["profile"]))
        profile = raw_profile.strip().lower()
        if profile not in self.VALID_PROFILES:
            self._param_warnings.append(
                f"profile='{raw_profile}' invalid; fallback='{self.BASE_DEFAULTS['profile']}'"
            )
            return self.BASE_DEFAULTS["profile"]
        return profile

    def _build_defaults_for_profile(self, profile: str) -> dict:
        defaults = dict(self.BASE_DEFAULTS)
        defaults.update(self.PROFILE_OVERRIDES.get(profile, {}))
        return defaults

    def _cooldown_gate(self, current_bar_index: int):
        last_bar_index = self._runtime_state.get("last_signal_bar_index")
        bars_since_last = None if last_bar_index is None else current_bar_index - last_bar_index
        blocked = (
            self.reentry_cooldown_bars > 0 and
            last_bar_index is not None and
            bars_since_last is not None and
            bars_since_last < self.reentry_cooldown_bars
        )
        return blocked, {
            "last_signal_side": self._runtime_state.get("last_signal_side"),
            "last_signal_bar_index": last_bar_index,
            "bars_since_last_signal": bars_since_last,
            "reentry_cooldown_active": blocked,
        }

    def _same_side_reentry_gate(self, side: str):
        same_side_reentries = self._runtime_state.get("same_side_reentries", 0)
        last_signal_side = self._runtime_state.get("last_signal_side")
        blocked = (
            self.max_same_side_reentries > 0 and
            last_signal_side == side and
            same_side_reentries >= self.max_same_side_reentries
        )
        return blocked, {
            "last_signal_side": last_signal_side,
            "same_side_reentries": same_side_reentries,
            "max_same_side_reentries": self.max_same_side_reentries,
            "same_side_reentry_blocked": blocked,
        }

    def _record_signal_state(self, side: str, bar_index: int):
        if self._runtime_state.get("last_signal_side") == side:
            self._runtime_state["same_side_reentries"] = self._runtime_state.get("same_side_reentries", 0) + 1
        else:
            self._runtime_state["same_side_reentries"] = 1
        self._runtime_state["last_signal_side"] = side
        self._runtime_state["last_signal_bar_index"] = bar_index

    def _effective_threshold(
        self,
        side: str,
        regime: RegimeType,
        base_threshold: float,
        bias_score: float,
        vol_ratio: float,
        score: float,
        opposing_score: float,
    ):
        threshold = float(base_threshold)
        bias_aligned = (side == "long" and bias_score > 0) or (side == "short" and bias_score < 0)
        bias_strength = abs(bias_score)
        score_advantage = score - opposing_score
        healthy_volume = vol_ratio >= max(self.vol_threshold + 0.12, 0.90)
        strong_volume = vol_ratio >= max(self.vol_threshold + 0.28, 1.05)
        directional_regime_match = (
            (regime == RegimeType.TRENDING_UP and side == "long") or
            (regime == RegimeType.TRENDING_DOWN and side == "short")
        )

        regime_adjustment = 0.0
        if regime in (RegimeType.RANGING, RegimeType.VOLATILE):
            regime_adjustment -= self.range_threshold_discount if regime == RegimeType.RANGING else 0.0
            regime_adjustment += self.volatile_threshold_buffer if regime == RegimeType.VOLATILE else 0.0
        elif directional_regime_match:
            regime_adjustment -= self.trend_threshold_discount

        alignment_bonus = 0.0
        if self.fast_entry_enabled and bias_aligned and bias_strength >= 1.0:
            alignment_bonus += min(self.fast_entry_max_bonus * 0.35, 0.08)
        if self.fast_entry_enabled and bias_aligned and bias_strength >= 2.0:
            alignment_bonus += min(self.fast_entry_max_bonus * 0.20, 0.04)
        if self.fast_entry_enabled and healthy_volume:
            alignment_bonus += min(self.fast_entry_max_bonus * 0.20, 0.04)
        if self.fast_entry_enabled and strong_volume:
            alignment_bonus += min(self.fast_entry_max_bonus * 0.15, 0.03)
        if self.fast_entry_enabled and directional_regime_match:
            alignment_bonus += min(self.fast_entry_max_bonus * 0.20, 0.04)

        quality_bonus = 0.0
        if self.fast_entry_enabled and score >= base_threshold + 0.60:
            quality_bonus += min(self.fast_entry_max_bonus * 0.10, 0.02)
        if self.fast_entry_enabled and score_advantage >= self.min_score_separation * 1.5:
            quality_bonus += min(self.fast_entry_max_bonus * 0.10, 0.02)

        countertrend_penalty = 0.0
        if regime in (RegimeType.RANGING, RegimeType.VOLATILE) and not bias_aligned and bias_strength >= 1.0:
            countertrend_penalty += self.countertrend_penalty
        if directional_regime_match is False and regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN):
            countertrend_penalty += self.countertrend_penalty * 0.5

        neutral_bias_bonus = 0.0
        if bias_strength <= self.bias_neutral_band and regime == RegimeType.RANGING:
            neutral_bias_bonus = min(self.fast_entry_max_bonus * 0.15, 0.02)

        reversal_probe_bonus = 0.0
        if self.reversal_probe_enabled and regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN):
            counter_side = (
                (regime == RegimeType.TRENDING_UP and side == "short") or
                (regime == RegimeType.TRENDING_DOWN and side == "long")
            )
            if counter_side and bias_strength <= max(self.bias_neutral_band, 0.40):
                reversal_probe_bonus = min(self.reversal_probe_max_bonus, 0.08)

        threshold_bonus = min(
            alignment_bonus + quality_bonus + neutral_bias_bonus + reversal_probe_bonus,
            self.fast_entry_max_bonus + self.reversal_probe_max_bonus,
        )
        threshold = max(0.12, threshold + regime_adjustment - threshold_bonus + countertrend_penalty)

        return threshold, {
            "base_threshold": round(float(base_threshold), 4),
            "effective_threshold": round(float(threshold), 4),
            "regime_adjustment": round(float(regime_adjustment), 4),
            "fast_entry_enabled": self.fast_entry_enabled,
            "bias_aligned": bias_aligned,
            "bias_strength": round(float(bias_strength), 4),
            "healthy_volume": healthy_volume,
            "strong_volume": strong_volume,
            "directional_regime_match": directional_regime_match,
            "score_advantage": round(float(score_advantage), 4),
            "alignment_bonus": round(float(alignment_bonus), 4),
            "quality_bonus": round(float(quality_bonus), 4),
            "neutral_bias_bonus": round(float(neutral_bias_bonus), 4),
            "reversal_probe_bonus": round(float(reversal_probe_bonus), 4),
            "threshold_bonus": round(float(threshold_bonus), 4),
            "countertrend_penalty": round(float(countertrend_penalty), 4),
        }

    def _resolve_tie(self, regime: RegimeType, bias_score: float):
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

    def _sanitize_float(self, key: str, default: float, minimum: float = None, maximum: float = None, strict_min: bool = False) -> float:
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
            "profile": self.profile,
            "available_profiles": sorted(self.VALID_PROFILES),
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
            "fast_entry_enabled": self.fast_entry_enabled,
            "fast_entry_max_bonus": self.fast_entry_max_bonus,
            "countertrend_penalty": self.countertrend_penalty,
            "min_score_separation": self.min_score_separation,
            "range_threshold_discount": self.range_threshold_discount,
            "volatile_threshold_buffer": self.volatile_threshold_buffer,
            "trend_threshold_discount": self.trend_threshold_discount,
            "bias_neutral_band": self.bias_neutral_band,
            "min_confidence_floor": self.min_confidence_floor,
            "reversal_probe_enabled": self.reversal_probe_enabled,
            "reversal_probe_max_bonus": self.reversal_probe_max_bonus,
            "reentry_cooldown_bars": self.reentry_cooldown_bars,
            "max_same_side_reentries": self.max_same_side_reentries,
            "runtime_state": dict(self._runtime_state),
            "param_warnings": list(self._param_warnings),
        }
        meta.update(extra)
        if "regime" in meta and isinstance(meta["regime"], RegimeType):
            meta["regime"] = meta["regime"].value
        for key, digits in {
            "bias_score": 3,
            "vol_ratio": 4,
            "long_score": 4,
            "short_score": 4,
            "threshold": 4,
            "long_threshold": 4,
            "short_threshold": 4,
            "best_threshold": 4,
            "score_gap": 6,
            "tie_band": 6,
            "entry_score": 4,
            "atr_value": 10,
            "confidence": 4,
        }.items():
            if key in meta and meta[key] is not None:
                meta[key] = round(float(meta[key]), digits)
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
        current_bar_index: int,
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
        reason = f"[MTF] {side.upper()} regime={regime.value} score={score:.2f} bias={bias_score:.1f} [{factor_str}]"

        self._record_signal_state(side, current_bar_index)

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
            amount_value=0 if uses_risk_managed_sizing else self.position_pct,
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

    def _compute_confidence(self, score: float, threshold: float, side: str, regime: RegimeType, bias_score: float, long_score: float, short_score: float) -> float:
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
            + components["stability_bonus"]
        )
        conf = max(self.min_confidence_floor, conf)
        return round(max(0.0, min(conf, 0.95)), 4)

    def _confidence_breakdown(self, score: float, threshold: float, side: str, regime: RegimeType, bias_score: float, long_score: float, short_score: float) -> dict:
        edge = max(0.0, score - threshold)
        separation = abs(long_score - short_score)
        base = 0.48
        edge_bonus = min(edge * 0.16, 0.16)
        separation_bonus = min(separation * 0.06, 0.10)
        regime_bonus = 0.0
        if (regime == RegimeType.TRENDING_UP and side == "long") or (regime == RegimeType.TRENDING_DOWN and side == "short"):
            regime_bonus = 0.06
        elif regime == RegimeType.RANGING:
            regime_bonus = 0.03
        bias_bonus = 0.0
        if (side == "long" and bias_score > 0) or (side == "short" and bias_score < 0):
            bias_bonus = min(abs(bias_score) * 0.0125, 0.05)
        stability_bonus = 0.0
        if edge >= 0.20 and separation >= self.min_score_separation:
            stability_bonus = 0.03
        return {
            "base": round(base, 4),
            "edge": round(edge, 4),
            "edge_bonus": round(edge_bonus, 4),
            "separation": round(separation, 4),
            "separation_bonus": round(separation_bonus, 4),
            "regime_bonus": round(regime_bonus, 4),
            "bias_bonus": round(bias_bonus, 4),
            "stability_bonus": round(stability_bonus, 4),
        }
