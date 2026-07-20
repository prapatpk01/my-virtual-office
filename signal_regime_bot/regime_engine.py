"""Layer 1: multi-timeframe market-regime classification.

The engine is deliberately structure-first.  4H provides macro context and 1H
provides active confirmation.  A 4H trend is not allowed to force a trade when
1H is clearly opposite or structure is ranging.  The engine classifies only;
Bias selects the side and Entry selects timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

STRONG_BULL = "STRONG_BULL_TREND"
EARLY_BULL = "EARLY_BULL_TREND"
RANGE = "RANGE"
COMPRESSION = "COMPRESSION"
EARLY_BEAR = "EARLY_BEAR_TREND"
STRONG_BEAR = "STRONG_BEAR_TREND"
HIGH_VOL = "HIGH_VOLATILITY"

BULL_LABELS = (STRONG_BULL, EARLY_BULL)
BEAR_LABELS = (STRONG_BEAR, EARLY_BEAR)

LONG = "LONG"
SHORT = "SHORT"
NEUTRAL = "NEUTRAL"


@dataclass
class TFRegime:
    label: str
    bull_score: float = 0.0
    bear_score: float = 0.0
    checks: dict = field(default_factory=dict)
    pass_count: int = 0
    pass_total: int = 100
    adx: float = 0.0
    chop: float = 100.0
    structure: str = "MIXED"
    volatility_shock: bool = False


@dataclass
class RegimeResult:
    label: str
    label_4h: str
    label_1h: str
    checks_4h: dict = field(default_factory=dict)
    checks_1h: dict = field(default_factory=dict)
    reason: str = ""
    name: str = ""
    score: float = 0.0
    style: str = ""
    size_multiplier: float = 1.0
    bull_score_4h: float = 0.0
    bear_score_4h: float = 0.0
    bull_score_1h: float = 0.0
    bear_score_1h: float = 0.0
    direction_edge: float = 0.0
    volatility_shock: bool = False


class RegimeEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _tf_regime(self, df: pd.DataFrame) -> TFRegime:
        c = self.cfg
        if df is None or len(df) < max(c.regime_ema_slow + 20, 80):
            return TFRegime(RANGE, checks={"insufficient_data": True})

        close = df["close"]
        price = ind.safe_float(close.iloc[-1])
        ema_fast = ind.ema(close, c.regime_ema_fast)
        ema_slow = ind.ema(close, c.regime_ema_slow)
        atr_series = ind.atr(df, c.regime_atr_period)
        atr_value = ind.safe_float(atr_series.iloc[-1])
        slope = ind.safe_float(ind.normalized_slope(ema_slow, atr_series, 3).iloc[-1])
        adx_s, plus_di_s, minus_di_s = ind.adx(df, c.regime_adx_period)
        adx_now = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_di_s.iloc[-1])
        minus_di = ind.safe_float(minus_di_s.iloc[-1])
        chop_now = ind.safe_float(ind.choppiness_index(df, c.regime_chop_period).iloc[-1], 100.0)
        line, signal, hist = ind.macd(close)
        macd_line = ind.safe_float(line.iloc[-1])
        macd_signal = ind.safe_float(signal.iloc[-1])
        macd_hist = ind.safe_float(hist.iloc[-1])
        roc9 = ind.safe_float(ind.roc(close, 9).iloc[-1])
        structure = ind.market_structure(
            df["high"], df["low"], c.bias_structure_left, c.bias_structure_right
        )
        bull_bos, _ = ind.latest_bos(
            df, LONG, c.bias_structure_left, c.bias_structure_right, 0.18
        )
        bear_bos, _ = ind.latest_bos(
            df, SHORT, c.bias_structure_left, c.bias_structure_right, 0.18
        )

        # Independent, symmetric 0-100 scores.  No bearish score is derived as
        # 100-bull; missing evidence contributes zero to both sides.
        bull_components = {
            "structure": 25.0 if structure == "HH_HL" else 10.0 if bull_bos else 0.0,
            "ema_alignment": 20.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 0.0,
            "ema_slope": 15.0 if slope > 0.03 else 7.0 if slope > 0 else 0.0,
            "price_location": 10.0 if price > ema_fast.iloc[-1] else 0.0,
            "macd": 10.0 if macd_line > macd_signal and macd_hist > 0 else 0.0,
            "roc": 5.0 if roc9 > 0 else 0.0,
            "dmi": 10.0 if plus_di > minus_di else 0.0,
            "adx": 5.0 if adx_now >= 15 else 0.0,
        }
        bear_components = {
            "structure": 25.0 if structure == "LH_LL" else 10.0 if bear_bos else 0.0,
            "ema_alignment": 20.0 if ema_fast.iloc[-1] < ema_slow.iloc[-1] else 0.0,
            "ema_slope": 15.0 if slope < -0.03 else 7.0 if slope < 0 else 0.0,
            "price_location": 10.0 if price < ema_fast.iloc[-1] else 0.0,
            "macd": 10.0 if macd_line < macd_signal and macd_hist < 0 else 0.0,
            "roc": 5.0 if roc9 < 0 else 0.0,
            "dmi": 10.0 if minus_di > plus_di else 0.0,
            "adx": 5.0 if adx_now >= 15 else 0.0,
        }
        bull_score = float(sum(bull_components.values()))
        bear_score = float(sum(bear_components.values()))
        edge = bull_score - bear_score

        bull_core = (
            ema_fast.iloc[-1] > ema_slow.iloc[-1]
            and price > ema_fast.iloc[-1]
            and slope > 0
        )
        bear_core = (
            ema_fast.iloc[-1] < ema_slow.iloc[-1]
            and price < ema_fast.iloc[-1]
            and slope < 0
        )

        atr_pct = ind.safe_float(
            ind.atr_percentile(atr_series, c.regime_atr_pct_lookback).iloc[-1], 50.0
        )
        ranges = (df["high"] - df["low"]).astype(float)
        median_range = ind.safe_float(ranges.iloc[-21:-1].median()) if len(df) >= 21 else 0.0
        range_shock = median_range > 0 and ranges.iloc[-1] > 2.5 * median_range
        atr_baseline = ind.safe_float(atr_series.iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
        atr_shock = atr_baseline > 0 and atr_value > 1.40 * atr_baseline
        volatility_shock = bool(range_shock or atr_shock)

        bb_width = ind.bollinger_width(df, 20)
        bb_pct = ind.safe_float(
            ind.rolling_percentile(bb_width, c.regime_atr_pct_lookback).iloc[-1], 50.0
        )
        compression = (
            atr_pct <= c.rg_compression_pctile_max
            and bb_pct <= c.rg_compression_pctile_max
            and adx_now < c.rg_range_adx_max
        )
        range_state = (
            (adx_now < 11 and chop_now > 62)
            or (structure == "MIXED" and adx_now < c.rg_range_adx_max and chop_now > 55)
        )

        if volatility_shock and max(bull_score, bear_score) < 75:
            label = HIGH_VOL
        elif bull_core and bull_score >= 78 and edge >= 15:
            label = STRONG_BULL
        elif bear_core and bear_score >= 78 and edge <= -15:
            label = STRONG_BEAR
        elif bull_score >= 60 and edge >= 10 and not range_state:
            label = EARLY_BULL
        elif bear_score >= 60 and edge <= -10 and not range_state:
            label = EARLY_BEAR
        elif compression:
            label = COMPRESSION
        else:
            label = RANGE

        checks = {
            "bull": bull_components,
            "bear": bear_components,
            "bull_core": bull_core,
            "bear_core": bear_core,
            "edge": round(edge, 1),
            "atr_percentile": round(atr_pct, 1),
            "volatility_shock": volatility_shock,
        }
        return TFRegime(
            label=label,
            bull_score=round(bull_score, 1),
            bear_score=round(bear_score, 1),
            checks=checks,
            pass_count=int(max(bull_score, bear_score)),
            adx=round(adx_now, 1),
            chop=round(chop_now, 1),
            structure=structure,
            volatility_shock=volatility_shock,
        )

    def analyze(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame | None = None) -> RegimeResult:
        r4 = self._tf_regime(df_4h)
        r1 = self._tf_regime(df_1h) if df_1h is not None and len(df_1h) else r4

        four_edge = r4.bull_score - r4.bear_score
        one_edge = r1.bull_score - r1.bear_score
        conflict = (
            (four_edge >= 15 and one_edge <= -15)
            or (four_edge <= -15 and one_edge >= 15)
        )

        if conflict:
            label = RANGE
            reason = (
                f"4H/1H strong conflict: 4H edge={four_edge:+.0f}, "
                f"1H edge={one_edge:+.0f}"
            )
        elif r4.label in BULL_LABELS and r1.label in BULL_LABELS:
            label = (
                STRONG_BULL
                if r4.label == STRONG_BULL and r1.label == STRONG_BULL
                else EARLY_BULL
            )
            reason = f"aligned bull context: 4H={r4.label}, 1H={r1.label}"
        elif r4.label in BEAR_LABELS and r1.label in BEAR_LABELS:
            label = (
                STRONG_BEAR
                if r4.label == STRONG_BEAR and r1.label == STRONG_BEAR
                else EARLY_BEAR
            )
            reason = f"aligned bear context: 4H={r4.label}, 1H={r1.label}"
        elif r1.label in BULL_LABELS and r4.bear_score < 60 and one_edge >= 15:
            # Allows an early 1H trend when 4H is neutral, but never against a
            # real 4H bear context.  This preserves frequency without using the
            # old unsafe rule “not strong bear means long”.
            label = EARLY_BULL
            reason = f"1H bull emerging inside non-bearish 4H context ({r4.label})"
        elif r1.label in BEAR_LABELS and r4.bull_score < 60 and one_edge <= -15:
            label = EARLY_BEAR
            reason = f"1H bear emerging inside non-bullish 4H context ({r4.label})"
        elif r4.label == HIGH_VOL or r1.label == HIGH_VOL:
            label = HIGH_VOL
            reason = f"volatility shock without confirmed aligned trend: 4H={r4.label}, 1H={r1.label}"
        elif r4.label == COMPRESSION or r1.label == COMPRESSION:
            label = COMPRESSION
            reason = f"compression context: 4H={r4.label}, 1H={r1.label}"
        else:
            label = RANGE
            reason = f"no aligned directional edge: 4H={r4.label}, 1H={r1.label}"

        if label in BULL_LABELS:
            score = 0.55 * r4.bull_score + 0.45 * r1.bull_score
            edge = 0.55 * four_edge + 0.45 * one_edge
            style = "TREND"
        elif label in BEAR_LABELS:
            score = 0.55 * r4.bear_score + 0.45 * r1.bear_score
            edge = -(0.55 * four_edge + 0.45 * one_edge)
            style = "TREND"
        else:
            score = max(r4.bull_score, r4.bear_score, r1.bull_score, r1.bear_score)
            edge = 0.0
            style = "NO_TRADE"

        # User requested 5% balance risk per accepted trade.  The regime may
        # block a trade but does not silently scale accepted trades below 5%.
        size_multiplier = 1.0
        return RegimeResult(
            label=label,
            label_4h=r4.label,
            label_1h=r1.label,
            checks_4h=r4.checks,
            checks_1h=r1.checks,
            reason=reason,
            name=label,
            score=round(score, 1),
            style=style,
            size_multiplier=size_multiplier,
            bull_score_4h=r4.bull_score,
            bear_score_4h=r4.bear_score,
            bull_score_1h=r1.bull_score,
            bear_score_1h=r1.bear_score,
            direction_edge=round(edge, 1),
            volatility_shock=r4.volatility_shock or r1.volatility_shock,
        )
