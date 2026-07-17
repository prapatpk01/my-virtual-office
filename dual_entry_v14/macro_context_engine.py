"""4H Macro Context Engine (spec §9) — score, classification, conflict level.

4H is a context MODIFIER except for Strong Conflict, which hard-rejects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .enums import ConflictLevel, StructureState
from .indicator_engine import ContextIndicators
from .models import StructureView
from .support_resistance_engine import nearest_opposing_zone


@dataclass
class MacroContext:
    score: float = 50.0
    classification: str = "NEUTRAL"          # STRONG_BULL/BULL/TRANSITION/RANGE/BEAR/STRONG_BEAR/NEUTRAL
    direction: Optional[str] = None          # LONG | SHORT | None
    structure_state: str = StructureState.RANGE.value
    premium_discount: str = "EQ"             # PREMIUM | DISCOUNT | EQ
    detail: dict = field(default_factory=dict)

    def conflict_for(self, direction: str, price: float, zones: list,
                     structure_4h: StructureView) -> ConflictLevel:
        """Strong conflict (spec): 4H strong-opposite + under active opposing
        4H zone + fresh opposite BOS/CHOCH + continuing LH/LL (or mirror)."""
        opposite = "SHORT" if direction == "LONG" else "LONG"
        strong_states = {StructureState.STRONG_BEAR.value} if direction == "LONG" \
            else {StructureState.STRONG_BULL.value}
        if self.structure_state in strong_states or self.classification in strong_states:
            fresh_opp_event = (
                (structure_4h.last_bos is not None and structure_4h.last_bos.direction == opposite)
                or (structure_4h.last_choch is not None and structure_4h.last_choch.direction == opposite)
            )
            opp_zone = nearest_opposing_zone(
                [z for z in zones if z.timeframe == "4h"], direction, price, min_score=55.0)
            if fresh_opp_event and opp_zone is not None:
                return ConflictLevel.STRONG
            return ConflictLevel.MILD
        mild_states = {"TRANSITION", "NEUTRAL", "RANGE", "MIXED"}
        if self.classification in mild_states:
            return ConflictLevel.MILD
        # 4H aligned-opposite but not strong
        if (direction == "LONG" and self.classification in ("BEAR",)) or \
           (direction == "SHORT" and self.classification in ("BULL",)):
            return ConflictLevel.MILD
        return ConflictLevel.NONE


class MacroContextEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, indicators_4h: Optional[ContextIndicators],
                 structure_4h: StructureView, zones: list,
                 supply_demand_zones: list) -> MacroContext:
        ctx = MacroContext()
        if indicators_4h is None:
            return ctx
        i = indicators_4h
        price = i.val(i.closes)
        st = structure_4h.state
        ctx.structure_state = st

        bull_pts, bear_pts = 0.0, 0.0

        # Structure Direction 0-30
        if st in (StructureState.STRONG_BULL.value, StructureState.BULL.value):
            bull_pts += 30 if st == StructureState.STRONG_BULL.value else 22
        elif st in (StructureState.STRONG_BEAR.value, StructureState.BEAR.value):
            bear_pts += 30 if st == StructureState.STRONG_BEAR.value else 22

        # EMA/HMA alignment 0-20
        e20, e50, h20 = i.val(i.ema20), i.val(i.ema50), i.val(i.hma20)
        if price > e20 > e50:
            bull_pts += 14
        if price < e20 < e50:
            bear_pts += 14
        if i.hma20_slope > 0:
            bull_pts += 6
        elif i.hma20_slope < 0:
            bear_pts += 6

        # ROC 0-10
        r = i.val(i.roc9)
        if r > 0:
            bull_pts += min(10.0, 4 + abs(r))
        elif r < 0:
            bear_pts += min(10.0, 4 + abs(r))

        # DMI 0-10 / ADX 0-10
        pdi, mdi, adx = i.val(i.plus_di), i.val(i.minus_di), i.val(i.adx)
        if pdi > mdi:
            bull_pts += min(10.0, (pdi - mdi) / 2)
        else:
            bear_pts += min(10.0, (mdi - pdi) / 2)
        adx_pts = min(10.0, adx / 3.0)

        # Major zone location 0-10 + BOS/CHOCH quality 0-10
        zones_4h = [z for z in zones if z.timeframe == "4h" and not z.broken]
        at_demand = any(z.is_support_like and z.contains(price) for z in zones_4h)
        at_supply = any(z.is_resistance_like and z.contains(price) for z in zones_4h)
        if at_demand:
            bull_pts += 8
        if at_supply:
            bear_pts += 8
        if structure_4h.last_bos is not None:
            q = 10.0 * structure_4h.last_bos.displacement_quality
            if structure_4h.last_bos.direction == "LONG":
                bull_pts += q
            else:
                bear_pts += q

        dominant_bull = bull_pts >= bear_pts
        score = (max(bull_pts, bear_pts) + adx_pts)
        score = float(min(100.0, score))
        ctx.score = score

        if score >= 80:
            ctx.classification = "STRONG_BULL" if dominant_bull else "STRONG_BEAR"
        elif score >= 65:
            ctx.classification = "BULL" if dominant_bull else "BEAR"
        elif score >= 40:
            ctx.classification = "TRANSITION"
        else:
            ctx.classification = "RANGE"
        ctx.direction = None if ctx.classification in ("TRANSITION", "RANGE") \
            else ("LONG" if dominant_bull else "SHORT")

        # premium/discount vs recent 4H range
        if len(i.closes) >= 60:
            hh = float(np.max(i.highs[-60:]))
            ll = float(np.min(i.lows[-60:]))
            if hh > ll:
                pos = (price - ll) / (hh - ll)
                ctx.premium_discount = "PREMIUM" if pos > 0.62 else "DISCOUNT" if pos < 0.38 else "EQ"

        ctx.detail = {"bull_pts": round(bull_pts, 1), "bear_pts": round(bear_pts, 1),
                      "adx": round(adx, 1), "at_demand": at_demand, "at_supply": at_supply}
        return ctx
