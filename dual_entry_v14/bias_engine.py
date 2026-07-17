"""1H Bias Engine (spec §10) — grouped scoring + Soft Bias Mode.

Never "not strong-bear so long is fine": soft-pass paths are explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .enums import BiasState, StructureState
from .indicator_engine import ContextIndicators
from .macro_context_engine import MacroContext
from .models import StructureView


@dataclass
class BiasResult:
    bias: str = BiasState.NEUTRAL.value
    score_long: float = 0.0
    score_short: float = 0.0
    detail: dict = field(default_factory=dict)

    def allows(self, direction: str, structure_15m: StructureView,
               zones: list, price: float, structure_1h: StructureView,
               now_ms: int, bar_ms_1h: int) -> tuple:
        """Soft Bias Mode — returns (allowed, risk_modifier, why)."""
        b = self.bias
        opp = "SHORT" if direction == "LONG" else "LONG"
        aligned = {"LONG": (BiasState.BULL.value, BiasState.STRONG_BULL.value),
                   "SHORT": (BiasState.BEAR.value, BiasState.STRONG_BEAR.value)}[direction]
        against = {"LONG": (BiasState.BEAR.value,), "SHORT": (BiasState.BULL.value,)}[direction]
        strong_against = {"LONG": BiasState.STRONG_BEAR.value,
                          "SHORT": BiasState.STRONG_BULL.value}[direction]

        recent_opp_choch = structure_1h.recent_choch_against(direction, bar_ms_1h * 12, now_ms)

        # path 1: aligned bias
        if b in aligned:
            return True, 1.0, "bias_aligned"
        # path 2: neutral + 15M structure agrees + HTF support/demand + no opposite 1H CHOCH
        if b == BiasState.NEUTRAL.value:
            s15_ok = (structure_15m.state in (StructureState.BULL.value, StructureState.STRONG_BULL.value)
                      if direction == "LONG" else
                      structure_15m.state in (StructureState.BEAR.value, StructureState.STRONG_BEAR.value))
            at_zone = any(z.timeframe in ("1h", "4h") and not z.broken and z.contains(price)
                          and (z.is_support_like if direction == "LONG" else z.is_resistance_like)
                          for z in zones)
            if s15_ok and at_zone and not recent_opp_choch:
                return True, 0.85, "neutral_soft_pass"
            return False, 1.0, "neutral_no_context"
        # path 3: weak counter-bias pullback — stabilization + reclaim, reduced risk
        if b in against and b != strong_against:
            stabilized = not structure_1h.recent_choch_against(direction, bar_ms_1h * 6, now_ms)
            reclaim_ok = (structure_15m.last_choch is not None
                          and structure_15m.last_choch.direction == direction)
            if stabilized and reclaim_ok:
                return True, 0.6, "counter_bias_reduced_risk"
        return False, 1.0, f"bias_{b}_blocks_{direction}"


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, indicators_1h: Optional[ContextIndicators],
                 structure_1h: StructureView, macro_context: MacroContext) -> BiasResult:
        res = BiasResult()
        if indicators_1h is None:
            return res
        i = indicators_1h
        price = i.val(i.closes)

        def side_score(direction: str) -> float:
            s = 0.0
            st = structure_1h.state
            # Structure 0-35
            if direction == "LONG":
                if st == StructureState.STRONG_BULL.value:
                    s += 35
                elif st == StructureState.BULL.value:
                    s += 27
                elif st == StructureState.TRANSITION.value:
                    s += 12
                if structure_1h.last_choch and structure_1h.last_choch.direction == "SHORT":
                    s -= 12
            else:
                if st == StructureState.STRONG_BEAR.value:
                    s += 35
                elif st == StructureState.BEAR.value:
                    s += 27
                elif st == StructureState.TRANSITION.value:
                    s += 12
                if structure_1h.last_choch and structure_1h.last_choch.direction == "LONG":
                    s -= 12
            # Trend alignment 0-25
            e20, e50 = i.val(i.ema20), i.val(i.ema50)
            slope = i.hma20_slope
            if direction == "LONG":
                s += 10 if price > e20 else 0
                s += 8 if e20 > e50 else 0
                s += 7 if slope > 0 else 0
            else:
                s += 10 if price < e20 else 0
                s += 8 if e20 < e50 else 0
                s += 7 if slope < 0 else 0
            # Momentum 0-15
            r_now = i.val(i.roc9)
            r_prev = i.val(i.roc9, -2)
            if direction == "LONG" and r_now > 0:
                s += 10 + (5 if r_now > r_prev else 0)
            if direction == "SHORT" and r_now < 0:
                s += 10 + (5 if r_now < r_prev else 0)
            # DMI 0-15 (weight down when ADX very low)
            pdi, mdi, adx = i.val(i.plus_di), i.val(i.minus_di), i.val(i.adx)
            spread = (pdi - mdi) if direction == "LONG" else (mdi - pdi)
            dmi_pts = min(15.0, max(0.0, spread))
            if adx < self.cfg.min_adx:
                dmi_pts *= 0.5
            s += dmi_pts
            # ADX 0-10
            s += min(10.0, adx / 3.0)
            return max(0.0, min(100.0, s))

        res.score_long = side_score("LONG")
        res.score_short = side_score("SHORT")
        top, bottom = max(res.score_long, res.score_short), min(res.score_long, res.score_short)
        gap = top - bottom
        long_side = res.score_long >= res.score_short
        if top >= 72 and gap >= 15:
            res.bias = BiasState.STRONG_BULL.value if long_side else BiasState.STRONG_BEAR.value
        elif top >= 55 and gap >= 8:
            res.bias = BiasState.BULL.value if long_side else BiasState.BEAR.value
        else:
            res.bias = BiasState.NEUTRAL.value
        res.detail = {"long": round(res.score_long, 1), "short": round(res.score_short, 1),
                      "macro": macro_context.classification}
        return res
