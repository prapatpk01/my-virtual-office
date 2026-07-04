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

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.position_pct  = self.params.get("position_pct",  0.05)
        self.vol_period    = self.params.get("vol_period",    20)
        self.vol_threshold = self.params.get("vol_threshold", 0.70)
        self.min_15m       = self.params.get("min_15m",       40)
        self.min_1h        = self.params.get("min_1h",        55)
        self.min_4h        = self.params.get("min_4h",        40)

        # Tie-break configuration
        self.tie_break_policy = self.params.get("tie_break_policy", "hold")
        self.tie_tolerance    = float(self.params.get("tie_tolerance", 1e-6))

        # ATR guard configuration
        self.atr_guard_enabled = bool(self.params.get("atr_guard_enabled", True))
        self.atr_min_value     = float(self.params.get("atr_min_value", 1e-10))

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
            return self._hold(current_price, f"Insufficient 15m data ({len(candles)} bars)")

        if len(candles_4h) < self.min_4h:
            return self._hold(
                current_price,
                f"Insufficient 4h data ({len(candles_4h)} bars < {self.min_4h})",
                metadata={
                    "required_4h": self.min_4h,
                    "actual_4h":   len(candles_4h),
                },
            )

        if len(candles_1h) < self.min_1h:
            return self._hold(
                current_price,
                f"Insufficient 1h data ({len(candles_1h)} bars < {self.min_1h})",
                metadata={
                    "required_1h": self.min_1h,
                    "actual_1h":   len(candles_1h),
                },
            )

        # ── 2. Market regime (4h) ─────────────────────────────────────────────
        regime, regime_debug = detect_regime(candles_4h, min_candles=self.min_4h)

        if regime == RegimeType.LOW_CONVICTION:
            return self._hold(
                current_price,
                f"Regime=LOW_CONVICTION (ADX weak) — skipping entry",
                metadata={"regime": regime.value, "regime_debug": regime_debug},
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
            )
        if regime == RegimeType.TRENDING_DOWN and bias_score > BIAS_MISALIGN_SHORT_MAX:
            return self._hold(
                current_price,
                f"Regime={regime.value} but 1h bias={bias_score:.1f} — misaligned",
                metadata=self._base_meta(regime, regime_debug, bias_score, bias_debug,
                                         vol_ratio, long_score, short_score),
            )

        # ── 7. Signal decision ────────────────────────────────────────────────
        base_meta = self._base_meta(
            regime, regime_debug, bias_score, bias_debug,
            vol_ratio, long_score, short_score,
        )
        base_meta["threshold"] = round(float(threshold), 4)

        long_ok  = long_score  >= threshold
        short_ok = short_score >= threshold
        is_tie   = abs(long_score - short_score) <= self.tie_tolerance

        # ── 7a. Deterministic tie handling ───────────────────────────────────
        # Both sides clear the threshold AND are within tolerance of each other.
        if long_ok and short_ok and is_tie:
            side = self._resolve_tie(regime, bias_score)
            if side is None:
                return self._hold(
                    current_price,
                    (f"Tie score (long={long_score:.4f}, short={short_score:.4f}, "
                     f"tolerance={self.tie_tolerance}) — policy='{self.tie_break_policy}' "
                     f"could not resolve"),
                    metadata={
                        **base_meta,
                        "long_factors":     long_factors,
                        "short_factors":    short_factors,
                        "long_debug":       long_debug,
                        "short_debug":      short_debug,
                        "tie_break_policy": self.tie_break_policy,
                        "tie_detected":     True,
                    },
                )
            if side == "long":
                return self._build_signal(
                    candles, current_price, "long",
                    long_score, long_factors, long_debug,
                    regime, bias_score,
                    {**base_meta, "tie_detected": True, "tie_break_used": True},
                )
            return self._build_signal(
                candles, current_price, "short",
                short_score, short_factors, short_debug,
                regime, bias_score,
                {**base_meta, "tie_detected": True, "tie_break_used": True},
            )

        if long_ok and long_score > short_score:
            return self._build_signal(
                candles, current_price, "long",
                long_score, long_factors, long_debug,
                regime, bias_score, base_meta,
            )

        if short_ok and short_score > long_score:
            return self._build_signal(
                candles, current_price, "short",
                short_score, short_factors, short_debug,
                regime, bias_score, base_meta,
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

    def _hold(self, price: float, reason: str, metadata: dict = None) -> Signal:
        return Signal(
            type=SignalType.HOLD,
            symbol=self.symbol,
            price=price,
            amount=0,
            reason=f"[MTF] {reason}",
            confidence=0.0,
            metadata=metadata or {},
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
        base_meta: dict,
    ) -> Signal:
        """Build a BUY/SELL Signal with trade plan metadata."""
        sig_type = SignalType.BUY if side == "long" else SignalType.SELL

        # ── Confidence ────────────────────────────────────────────────────────
        # Base: raw score → 0.50–0.90 range; bonus for regime/bias alignment.
        conf = 0.45 + min(score * 0.55, 0.45)
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
            )

        trade_plan = build_trade_plan(price, atr_val, side)

        # ── sj_score for bot trade-ranking (higher = better quality) ─────────
        sj = round(conf * 100 + score * 20, 2)

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
        }

        return Signal(
            type=sig_type,
            symbol=self.symbol,
            price=price,
            amount=self.position_pct if not trade_plan.get("sl_dist_pct") else 0,
            reason=reason,
            confidence=conf,
            metadata=metadata,
        )
