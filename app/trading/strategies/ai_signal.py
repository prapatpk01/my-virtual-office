"""
AI Signal strategy v2 — API-free, multi-timeframe, regime-adaptive.

Architecture (top-down):
  4h candles  → detect_regime()      — TRENDING/RANGING/VOLATILE/LOW_CONVICTION
  1h candles  → directional_bias()   — bull/bear bias score  (-3 … +3)
  15m candles → volume_ok()          — skip entry when volume is too thin
  15m candles → _score_factors()     — entry trigger score (RSI/MACD/ST/Vol/EMA/HA)
  price+ATR   → build_trade_plan()   — SL=1R, T1=0.5R … T4=1.2R, SL-ratchet ladder

The Signal.metadata returned is fully compatible with TradingBot._open_position():
  sl_ladder_enabled=True activates SL-ratchet mode (no partial closes; SL steps
  up as each ladder level is hit: T1→+0.3R, T2→+0.5R, T3→+0.8R, T4→full close).
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
)

# Minimum bias alignment required to trade against the 1h bias
# (e.g. regime=TRENDING_UP but bias score is negative → skip)
_BIAS_ALIGN_MIN = 0.0


class AISignalStrategy(BaseStrategy):
    """
    Multi-timeframe, regime-adaptive signal strategy.

    No external API dependency — fully deterministic, indicator-based.
    Keeps the same class name and interface as the original AISignalStrategy
    so all existing bot wiring continues to work without changes.

    Params (all optional, with sensible defaults):
      position_pct  — legacy position fraction (0.05); overridden by RiskManager
                      dynamic sizing when sl_dist_pct and risk_pct are present.
      vol_period    — lookback for volume filter (default 20)
      vol_threshold — minimum vol_ratio to trade (default 0.70)
      min_15m       — minimum 15m candles required (default 40)
      min_1h        — minimum 1h candles required (default 55)
      min_4h        — minimum 4h candles required (default 40)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def analyze(
        self,
        candles: list,           # 15m candles (primary entry TF)
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        """
        Produce a BUY / SELL / HOLD signal with full metadata.

        Flow:
          1. Guard: require enough candles on all timeframes.
          2. Regime detection (4h).  LOW_CONVICTION → HOLD immediately.
          3. Volume filter (15m).    Low volume → HOLD.
          4. Directional bias (1h).
          5. Entry scoring (15m) for the regime-preferred direction(s).
          6. Decide signal type and confidence.
          7. Build TP/SL ladder trade plan.
          8. Return Signal with rich metadata.
        """
        mtf = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])

        # ── 1. Data guards ────────────────────────────────────────────────────
        if len(candles) < self.min_15m:
            return self._hold(current_price, f"Insufficient 15m data ({len(candles)} bars)")

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
        if regime == RegimeType.TRENDING_UP and bias_score < -1.0:
            # Strong 1h bear bias in a 4h uptrend → wait for alignment
            return self._hold(
                current_price,
                f"Regime={regime.value} but 1h bias={bias_score:.1f} — misaligned",
                metadata=self._base_meta(regime, regime_debug, bias_score, bias_debug,
                                         vol_ratio, long_score, short_score),
            )
        if regime == RegimeType.TRENDING_DOWN and bias_score > 1.0:
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

        if long_score >= threshold and long_score > short_score:
            return self._build_signal(
                candles, current_price, "long",
                long_score, long_factors, long_debug,
                regime, bias_score, base_meta,
            )

        if short_score >= threshold and short_score > long_score:
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

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

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
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        trade_plan = build_trade_plan(price, atr_val, side)

        # ── sj_score for bot trade-ranking (higher = better quality) ─────────
        sj = round(conf * 100 + score * 20, 2)

        # ── Reason string ──────────────────────────────────────────────────────
        factor_str = "+".join(factors[:4])
        reason = (
            f"[MTF] {side.upper()} regime={regime.value} "
            f"score={score:.2f} bias={bias_score:.1f} [{factor_str}]"
        )

        metadata = {
            **base_meta,
            **trade_plan,
            "entry_factors":  factors,
            "entry_debug":    entry_debug,
            "entry_score":    round(score, 4),
            "side":           side,
            "sj_score":       sj,
            "api_free":       True,
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
