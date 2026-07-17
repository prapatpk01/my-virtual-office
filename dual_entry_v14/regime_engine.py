"""15M Regime Engine (spec §16) — HMA10/16 regimes with persistence +
candidate-regime fast path so entries aren't delayed two bars when a hard
BOS/CHOCH or strong location trigger appears.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .enums import RegimeState, StructureState
from .indicator_engine import EntryIndicators
from .models import StructureView


@dataclass
class RegimeResult:
    confirmed_regime: str = RegimeState.TRANSITION.value
    candidate_regime: str = RegimeState.TRANSITION.value
    candidate_count: int = 0
    raw_regime: str = RegimeState.TRANSITION.value
    is_chop: bool = False
    detail: dict = field(default_factory=dict)

    def allows_pullback(self, direction: str) -> bool:
        ok_long = {RegimeState.BULL_TREND.value, RegimeState.STRONG_BULL_TREND.value,
                   RegimeState.PULLBACK_TRANSITION_BULL.value}
        ok_short = {RegimeState.BEAR_TREND.value, RegimeState.STRONG_BEAR_TREND.value,
                    RegimeState.PULLBACK_TRANSITION_BEAR.value}
        r = self.effective_regime()
        return r in (ok_long if direction == "LONG" else ok_short)

    def allows_momentum(self, direction: str, breakout_quality_high: bool,
                        confirmed_breakout: bool) -> bool:
        r = self.effective_regime()
        if direction == "LONG":
            if r == RegimeState.STRONG_BULL_TREND.value:
                return True
            if r == RegimeState.BULL_TREND.value and breakout_quality_high:
                return True
            if r == RegimeState.TRANSITION.value and confirmed_breakout:
                return True
            return False
        if r == RegimeState.STRONG_BEAR_TREND.value:
            return True
        if r == RegimeState.BEAR_TREND.value and breakout_quality_high:
            return True
        if r == RegimeState.TRANSITION.value and confirmed_breakout:
            return True
        return False

    def effective_regime(self) -> str:
        # candidate fast-path is resolved by the engine into confirmed_regime
        return self.confirmed_regime


class RegimeEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def classify(self, indicators_15m: Optional[EntryIndicators],
                 structure_15m: StructureView, zones: list,
                 previous_regime: Optional[str], candidate_regime: Optional[str],
                 candidate_count: int) -> RegimeResult:
        res = RegimeResult()
        if indicators_15m is None:
            return res
        i = indicators_15m
        c = self.cfg
        hf, hs = i.val(i.hma_fast), i.val(i.hma_slow)
        hf_p, hs_p = i.val(i.hma_fast, -2), i.val(i.hma_slow, -2)
        adx, chop = i.val(i.adx), i.val(i.chop)
        pdi, mdi = i.val(i.plus_di), i.val(i.minus_di)
        price = i.val(i.closes)
        hma_up_slope = hf - hf_p
        hma_slow_slope = hs - hs_p

        raw = RegimeState.TRANSITION

        # ── chop first (hard) ────────────────────────────────────────────────
        flips = self._flip_count(i.hma_fast, i.hma_slow, c.hma_flip_window)
        spread_avg = float(np.nanmean(np.abs(i.hma_fast[-c.hma_flip_window:]
                                             - i.hma_slow[-c.hma_flip_window:]))) \
            / max(i.last_atr, 1e-12)
        di_flips = self._flip_count(i.plus_di, i.minus_di, c.hma_flip_window)
        di_spread_avg = float(np.nanmean(np.abs(i.plus_di[-c.hma_flip_window:]
                                                - i.minus_di[-c.hma_flip_window:])))
        is_chop = ((adx < c.chop_adx and chop > c.chop_chop)
                   or (flips >= c.hma_flip_count and spread_avg < c.hma_flip_spread_atr)
                   or (di_flips >= c.di_flip_count and di_spread_avg < c.di_flip_min_spread))
        if is_chop:
            raw = RegimeState.CHOP
        else:
            bull_struct_ok = structure_15m.state not in (StructureState.BEAR.value,
                                                         StructureState.STRONG_BEAR.value)
            bear_struct_ok = structure_15m.state not in (StructureState.BULL.value,
                                                         StructureState.STRONG_BULL.value)
            strong_bull = (hf > hs and hma_up_slope > 0 and hma_slow_slope >= 0
                           and pdi > mdi and adx >= c.strong_adx and chop <= c.strong_chop)
            strong_bear = (hf < hs and hma_up_slope < 0 and hma_slow_slope <= 0
                           and mdi > pdi and adx >= c.strong_adx and chop <= c.strong_chop)
            bull = (hf > hs and bull_struct_ok
                    and (adx >= c.min_adx or structure_15m.quality >= 55) and chop <= c.max_chop)
            bear = (hf < hs and bear_struct_ok
                    and (adx >= c.min_adx or structure_15m.quality >= 55) and chop <= c.max_chop)

            at_support = any(z.is_support_like and not z.broken and z.contains(price)
                             for z in zones)
            at_resistance = any(z.is_resistance_like and not z.broken and z.contains(price)
                                for z in zones)
            reclaim_bull = (structure_15m.last_choch is not None
                            and structure_15m.last_choch.direction == "LONG")
            reclaim_bear = (structure_15m.last_choch is not None
                            and structure_15m.last_choch.direction == "SHORT")
            prior = previous_regime or ""

            pb_bull = (prior in (RegimeState.BULL_TREND.value, RegimeState.STRONG_BULL_TREND.value)
                       and (abs(hf - hs) / max(i.last_atr, 1e-12) < 0.35 or hf < hs)
                       and bull_struct_ok and (at_support or reclaim_bull))
            pb_bear = (prior in (RegimeState.BEAR_TREND.value, RegimeState.STRONG_BEAR_TREND.value)
                       and (abs(hf - hs) / max(i.last_atr, 1e-12) < 0.35 or hf > hs)
                       and bear_struct_ok and (at_resistance or reclaim_bear))

            crossed_recently = self._crossed_within(i.hma_fast, i.hma_slow, 2)
            transition = (crossed_recently and 9 <= adx <= 22 and chop <= 64
                          and (structure_15m.last_bos is not None
                               or structure_15m.last_choch is not None))

            if strong_bull:
                raw = RegimeState.STRONG_BULL_TREND
            elif strong_bear:
                raw = RegimeState.STRONG_BEAR_TREND
            elif bull:
                raw = RegimeState.BULL_TREND
            elif bear:
                raw = RegimeState.BEAR_TREND
            elif pb_bull:
                raw = RegimeState.PULLBACK_TRANSITION_BULL
            elif pb_bear:
                raw = RegimeState.PULLBACK_TRANSITION_BEAR
            elif transition:
                raw = RegimeState.TRANSITION
            else:
                raw = RegimeState.TRANSITION

        res.raw_regime = raw.value
        res.is_chop = raw == RegimeState.CHOP

        # ── persistence: 2 bars to confirm; candidate fast-path allowed ──────
        prev_confirmed = previous_regime or raw.value
        if raw.value == candidate_regime:
            count = candidate_count + 1
        else:
            count = 1
        candidate = raw.value

        hard_event = ((structure_15m.last_bos is not None
                       and structure_15m.last_bos.confirmed_at == int(i.timestamps[-1]))
                      or (structure_15m.last_choch is not None
                          and structure_15m.last_choch.confirmed_at == int(i.timestamps[-1])))
        if raw == RegimeState.CHOP:
            confirmed = raw.value                      # chop applies immediately
        elif count >= self.cfg.regime_confirmation_bars:
            confirmed = raw.value
        elif self.cfg.allow_candidate_regime_entry and hard_event:
            confirmed = raw.value                      # fast path on hard BOS/CHOCH
        else:
            confirmed = prev_confirmed

        res.confirmed_regime = confirmed
        res.candidate_regime = candidate
        res.candidate_count = count
        res.detail = {"adx": round(adx, 1), "chop": round(chop, 1),
                      "hma_spread_atr": round(i.hma_spread_atr, 3),
                      "flips": flips, "raw": raw.value}
        return res

    @staticmethod
    def _flip_count(a: np.ndarray, b: np.ndarray, window: int) -> int:
        if len(a) < window + 1:
            return 0
        d = np.sign(a[-window - 1:] - b[-window - 1:])
        d = d[np.isfinite(d)]
        return int(np.sum(d[1:] * d[:-1] < 0))

    @staticmethod
    def _crossed_within(a: np.ndarray, b: np.ndarray, bars: int) -> bool:
        if len(a) < bars + 2:
            return False
        d = np.sign(a[-bars - 2:] - b[-bars - 2:])
        d = d[np.isfinite(d)]
        return bool(np.any(d[1:] * d[:-1] < 0))
