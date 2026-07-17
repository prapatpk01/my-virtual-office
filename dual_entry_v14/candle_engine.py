"""15M Candle & Trigger Engine (spec §17) — entry confirmation, never HTF bias.

A bar can qualify through a NAMED pattern or a generic quality close
(strong/reclaim/breakout/rejection/displacement close).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .enums import CandleTrigger
from .indicator_engine import EPS, EntryIndicators
from .models import Candle


@dataclass
class CandleContext:
    bull_triggers: list = field(default_factory=list)   # [CandleTrigger.value]
    bear_triggers: list = field(default_factory=list)
    bull_quality: float = 0.0        # 0-1
    bear_quality: float = 0.0
    location_score: float = 0.0      # 0-10 (spec table)
    detail: dict = field(default_factory=dict)

    def best_bull(self) -> Optional[str]:
        return self.bull_triggers[0] if self.bull_triggers else None

    def best_bear(self) -> Optional[str]:
        return self.bear_triggers[0] if self.bear_triggers else None


class CandleEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, candles_15m: list, indicators_15m: Optional[EntryIndicators],
                 zones: list) -> CandleContext:
        ctx = CandleContext()
        if indicators_15m is None or len(candles_15m) < 3:
            return ctx
        last, prev = candles_15m[-1], candles_15m[-2]
        a = indicators_15m.last_atr or EPS
        body_atr = last.body / a
        body_ratio = last.body / last.range
        bull_q = last.bull_close_quality
        bear_q = last.bear_close_quality

        bt, st = [], []

        # bullish displacement
        if last.is_bull and body_atr >= 0.22 and body_ratio >= 0.58 and bull_q >= 0.70:
            bt.append(CandleTrigger.BULLISH_DISPLACEMENT.value)
        # bullish pin bar / hammer
        if last.lower_wick >= last.body * 1.8 and bull_q >= 0.62:
            bt.append(CandleTrigger.BULLISH_PIN_BAR.value)
            if last.is_bull:
                bt.append(CandleTrigger.HAMMER.value)
        # bullish engulfing
        if (prev.close < prev.open and last.is_bull and last.open <= prev.close
                and last.close >= prev.open and body_atr >= 0.15):
            bt.append(CandleTrigger.BULLISH_ENGULFING.value)
        # inside-bar breakout
        if (prev.high <= candles_15m[-3].high and prev.low >= candles_15m[-3].low
                and last.close > candles_15m[-3].high and last.is_bull):
            bt.append(CandleTrigger.INSIDE_BAR_BREAKOUT.value)
        # generic strong / reclaim close
        if last.is_bull and bull_q >= 0.70 and body_atr >= 0.15:
            bt.append(CandleTrigger.STRONG_CLOSE.value)
        if last.is_bull and last.close > prev.high:
            bt.append(CandleTrigger.BREAKOUT_CLOSE.value)

        # bearish mirrors
        if (not last.is_bull) and body_atr >= 0.22 and body_ratio >= 0.58 and bear_q >= 0.70:
            st.append(CandleTrigger.BEARISH_DISPLACEMENT.value)
        if last.upper_wick >= last.body * 1.8 and bear_q >= 0.62:
            st.append(CandleTrigger.BEARISH_PIN_BAR.value)
            if not last.is_bull:
                st.append(CandleTrigger.SHOOTING_STAR.value)
        if (prev.close > prev.open and (not last.is_bull) and last.open >= prev.close
                and last.close <= prev.open and body_atr >= 0.15):
            st.append(CandleTrigger.BEARISH_ENGULFING.value)
        if (prev.high <= candles_15m[-3].high and prev.low >= candles_15m[-3].low
                and last.close < candles_15m[-3].low and not last.is_bull):
            st.append(CandleTrigger.INSIDE_BAR_BREAKOUT.value)
        if (not last.is_bull) and bear_q >= 0.70 and body_atr >= 0.15:
            st.append(CandleTrigger.STRONG_CLOSE.value)
        if (not last.is_bull) and last.close < prev.low:
            st.append(CandleTrigger.BREAKOUT_CLOSE.value)

        # zone rejection triggers + location score
        loc = 1.0    # mid-air baseline (0-2)
        for z in zones:
            if z.broken or not z.contains(last.close) and not (z.lower_price <= last.low <= z.upper_price
                                                               or z.lower_price <= last.high <= z.upper_price):
                continue
            tf_pts = {"15m": 4.0, "1h": 7.0, "4h": 9.0}.get(z.timeframe, 3.0)
            loc = max(loc, tf_pts)
            if z.is_support_like and last.is_bull and last.lower_wick >= last.body:
                bt.append(CandleTrigger.SUPPORT_REJECTION.value)
            if z.is_resistance_like and (not last.is_bull) and last.upper_wick >= last.body:
                st.append(CandleTrigger.RESISTANCE_REJECTION.value)

        ctx.bull_triggers = list(dict.fromkeys(bt))
        ctx.bear_triggers = list(dict.fromkeys(st))
        ctx.bull_quality = min(1.0, (0.4 if ctx.bull_triggers else 0.0)
                               + bull_q * 0.35 + min(body_atr / 0.4, 1.0) * 0.25)
        ctx.bear_quality = min(1.0, (0.4 if ctx.bear_triggers else 0.0)
                               + bear_q * 0.35 + min(body_atr / 0.4, 1.0) * 0.25)
        ctx.location_score = min(loc, 10.0)
        ctx.detail = {"body_atr": round(body_atr, 3), "body_ratio": round(body_ratio, 3),
                      "bull_close_q": round(bull_q, 3), "bear_close_q": round(bear_q, 3)}
        return ctx
