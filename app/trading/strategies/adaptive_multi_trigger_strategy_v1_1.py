"""Adaptive Multi-Trigger Entry Engine V1.1.

V1.1 is a conservative refinement of V1.0 based on BTC Jan-May 2026 research
backtests.  It keeps the same architecture and trigger ownership, but reduces
low-quality overtrading without turning the engine into an indicator stack.

Changes from V1.0:
- 1H WEAK context is not tradable (STRONG/NORMAL only; CHOP still blocked).
- Pullback continuation requires BOTH contracting pullback volume and a real
  rejection/reclaim candle; reclaim-bar RVOL must be <= 0.90.
- Direct BREAKOUT requires RVOL >= 1.20.
- MOMENTUM_EXPANSION requires RVOL >= 1.10.
- COMPRESSION_BREAK requires RVOL >= 1.00.
- Location is stricter: minimum room >= 1.40R.
- Chase guard is stricter: EMA20 extension <= 0.80 ATR.

Important:
- Pattern still answers WHAT is happening.
- Volume DNA still scores participation; the extra pattern-specific rules only
  reject the clearest mismatch cases found in research.
- Trigger still answers WHEN to execute.
- EMA8/13 and HMA remain optional; structure/reclaim/momentum triggers retain
  priority when they appear first.
- This module preserves the V1 family name so it can replace V1.0 later without
  changing position-family ownership semantics.
"""
from __future__ import annotations

from typing import Optional

from .adaptive_multi_trigger_strategy import AdaptiveMultiTriggerStrategy


class AdaptiveMultiTriggerV11Strategy(AdaptiveMultiTriggerStrategy):
    """Tuned V1.1 wrapper around the V1.0 pattern-first engine."""

    VERSION = "1.1"

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        pullback_max_rvol: float = 0.90,
        breakout_min_rvol: float = 1.20,
        momentum_min_rvol: float = 1.10,
        compression_min_rvol: float = 1.00,
        v11_min_structure_room_r: float = 1.40,
        v11_max_ema20_extension_atr: float = 0.80,
        **kwargs,
    ):
        # Keep the original 60-quality architecture.  V1.1 improves candidate
        # quality through setup-specific validation rather than simply raising
        # one universal score threshold for every archetype.
        kwargs.setdefault("entry_quality_threshold", 60.0)
        kwargs.setdefault("weak_context_threshold", 70.0)
        super().__init__(symbol=symbol, params=params, **kwargs)

        # Preserve the production family prefix used by risk ownership.
        self.name = f"AdaptiveMultiTriggerV1({symbol})"

        self.pullback_max_rvol = max(0.10, float(pullback_max_rvol))
        self.breakout_min_rvol = max(0.10, float(breakout_min_rvol))
        self.momentum_min_rvol = max(0.10, float(momentum_min_rvol))
        self.compression_min_rvol = max(0.10, float(compression_min_rvol))
        self.v11_min_structure_room_r = max(0.50, float(v11_min_structure_room_r))
        self.v11_max_ema20_extension_atr = max(
            0.20,
            float(v11_max_ema20_extension_atr),
        )

    @staticmethod
    def _rvol(candles: list) -> float:
        if len(candles) < 21:
            return 0.0
        current = max(float(candles[-1].volume), 0.0)
        baseline = sum(max(float(c.volume), 0.0) for c in candles[-21:-1]) / 20.0
        return current / max(baseline, 1e-12)

    def _context_1h(self, candles: list, direction: str) -> dict:
        context = super()._context_1h(candles, direction)
        # Research result: allowing WEAK through a higher score threshold still
        # created low-expectancy entries.  V1.1 therefore trades only NORMAL or
        # STRONG context.  This does not add another trigger; it is permission
        # only, exactly like CHOP handling in V1.0.
        if context.get("status") == "WEAK":
            context = dict(context)
            context["status"] = "CHOP"
            context["v11_block"] = "WEAK_CONTEXT"
        return context

    def _detect_patterns(self, candles: list, direction: str):
        patterns = super()._detect_patterns(candles, direction)
        if not patterns:
            return patterns

        rvol = self._rvol(candles)
        filtered = []
        for pattern in patterns:
            name = str(pattern.name)
            diag = pattern.diagnostics if isinstance(pattern.diagnostics, dict) else {}

            if name == "PULLBACK_CONTINUATION":
                # Pullback should actually contract, then reject/reclaim.  A
                # high-RVOL reclaim tended to be late/impulsive rather than a
                # clean low-energy pullback continuation in BTC research.
                if not bool(diag.get("pullback_volume_contract")):
                    continue
                if not bool(diag.get("rejection")):
                    continue
                if rvol > self.pullback_max_rvol:
                    continue

            elif name == "BREAKOUT":
                if rvol < self.breakout_min_rvol:
                    continue

            elif name == "MOMENTUM_EXPANSION":
                if rvol < self.momentum_min_rvol:
                    continue

            elif name == "COMPRESSION_BREAK":
                if rvol < self.compression_min_rvol:
                    continue

            # BREAKOUT_RETEST, STRUCTURE_CONTINUATION and LIQUIDITY_SWEEP keep
            # their V1.0 pattern logic.  V1.1 intentionally does not disable a
            # whole archetype based on a small single-symbol sample.
            filtered.append(pattern)

        return filtered

    def _location_and_chase(
        self,
        candles: list,
        direction: str,
        current_price: float,
        pattern,
    ) -> dict:
        result = dict(
            super()._location_and_chase(
                candles,
                direction,
                current_price,
                pattern,
            )
        )

        room_r = float(result.get("room_r", 0.0) or 0.0)
        ema_ext = float(result.get("ema20_extension_atr", 999.0) or 999.0)
        room_ok = room_r >= self.v11_min_structure_room_r
        extension_ok = ema_ext <= self.v11_max_ema20_extension_atr

        result.update({
            "v11_min_room_r": self.v11_min_structure_room_r,
            "v11_max_ema20_extension_atr": self.v11_max_ema20_extension_atr,
            "v11_room_ok": bool(room_ok),
            "v11_extension_ok": bool(extension_ok),
        })
        result["valid"] = bool(result.get("valid") and room_ok and extension_ok)
        if not room_ok:
            result["v11_reject_reason"] = "ROOM_LT_1_4R"
        elif not extension_ok:
            result["v11_reject_reason"] = "EMA20_EXTENSION_GT_0_8ATR"
        return result
