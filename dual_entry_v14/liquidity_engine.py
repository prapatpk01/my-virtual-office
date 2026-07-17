"""Liquidity Engine (spec §15) — sweeps + reclaim near meaningful levels only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .indicator_engine import EPS
from .swing_engine import swings_of


@dataclass
class LiquidityContext:
    bullish_sweep: bool = False
    bearish_sweep: bool = False
    sweep_level: Optional[float] = None
    sweep_location_ok: bool = False       # near zone/swing — mid-air sweeps don't count
    equal_highs: Optional[float] = None
    equal_lows: Optional[float] = None
    detail: dict = field(default_factory=dict)


class LiquidityEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, candles_15m: list, candles_1h: list,
                 swings_15m: list, swings_1h: list, zones: list) -> LiquidityContext:
        ctx = LiquidityContext()
        if len(candles_15m) < 10:
            return ctx
        rng = np.array([c.high - c.low for c in candles_15m[-30:]])
        atr_v = max(float(np.mean(rng)), EPS)
        last, prev = candles_15m[-1], candles_15m[-2]
        now_ts = int(last.timestamp)

        # equal highs/lows on 15m (liquidity pools)
        highs = [s.price for s in swings_of(swings_15m, "high")][-4:]
        lows = [s.price for s in swings_of(swings_15m, "low")][-4:]
        for a, b in zip(highs, highs[1:]):
            if abs(a - b) <= atr_v * 0.15:
                ctx.equal_highs = max(a, b)
        for a, b in zip(lows, lows[1:]):
            if abs(a - b) <= atr_v * 0.15:
                ctx.equal_lows = min(a, b)

        # reference levels a sweep can take out (confirmed swings + zones + equal lows/highs)
        low_refs = [s.price for s in swings_of(swings_15m, "low") if s.confirmed_at < now_ts][-3:]
        low_refs += [s.price for s in swings_of(swings_1h, "low") if s.confirmed_at < now_ts][-2:]
        if ctx.equal_lows:
            low_refs.append(ctx.equal_lows)
        high_refs = [s.price for s in swings_of(swings_15m, "high") if s.confirmed_at < now_ts][-3:]
        high_refs += [s.price for s in swings_of(swings_1h, "high") if s.confirmed_at < now_ts][-2:]
        if ctx.equal_highs:
            high_refs.append(ctx.equal_highs)

        def near_meaningful(level: float, support_side: bool) -> bool:
            for z in zones:
                if z.broken:
                    continue
                side_ok = z.is_support_like if support_side else z.is_resistance_like
                if side_ok and (z.lower_price - atr_v * 0.3) <= level <= (z.upper_price + atr_v * 0.3):
                    return True
            return False

        # bullish sweep: low pokes below ref, close back above, w/ quality wick or displacement
        for bar in (last, prev):
            for ref in low_refs:
                if bar.low < ref and bar.close > ref:
                    wick_ok = bar.lower_wick >= bar.body * 1.2
                    disp_ok = bar.is_bull and (bar.body / atr_v) >= 0.22
                    reclaim_ok = bar is prev and last.close > ref  # next-bar micro-hold
                    if wick_ok or disp_ok or reclaim_ok:
                        ctx.bullish_sweep = True
                        ctx.sweep_level = ref
                        ctx.sweep_location_ok = near_meaningful(ref, support_side=True)
                        break
            if ctx.bullish_sweep:
                break

        for bar in (last, prev):
            for ref in high_refs:
                if bar.high > ref and bar.close < ref:
                    wick_ok = bar.upper_wick >= bar.body * 1.2
                    disp_ok = (not bar.is_bull) and (bar.body / atr_v) >= 0.22
                    reclaim_ok = bar is prev and last.close < ref
                    if wick_ok or disp_ok or reclaim_ok:
                        ctx.bearish_sweep = True
                        ctx.sweep_level = ref
                        ctx.sweep_location_ok = near_meaningful(ref, support_side=False)
                        break
            if ctx.bearish_sweep:
                break

        ctx.detail = {"eq_highs": ctx.equal_highs, "eq_lows": ctx.equal_lows}
        return ctx
