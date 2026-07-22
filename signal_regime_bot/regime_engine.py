"""DUALCORE V2.0 market-regime classification.

4H is a macro conflict filter; 1H is the active directional regime.  A neutral
4H does not block a valid 1H trend, while a confirmed strong opposite 4H does.
The engine never triggers an order by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
        ema_fast_s = ind.ema(close, c.regime_ema_fast)
        ema_slow_s = ind.ema(close, c.regime_ema_slow)
        ema_fast = ind.safe_float(ema_fast_s.iloc[-1])
        ema_slow = ind.safe_float(ema_slow_s.iloc[-1])
        atr_s = ind.atr(df, c.regime_atr_period)
        atr_value = ind.safe_float(atr_s.iloc[-1])
        slope = ind.safe_float(ind.normalized_slope(ema_slow_s, atr_s, 3).iloc[-1])
        adx_s, plus_s, minus_s = ind.adx(df, c.regime_adx_period)
        adx_now = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_s.iloc[-1])
        minus_di = ind.safe_float(minus_s.iloc[-1])
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

        bull = {
            "structure": 24.0 if structure == "HH_HL" else 11.0 if bull_bos else 0.0,
            "ema_alignment": 20.0 if ema_fast > ema_slow else 0.0,
            "ema_slope": 14.0 if slope > 0.03 else 7.0 if slope > 0 else 0.0,
            "price_hold": 10.0 if price > ema_fast else 0.0,
            "macd": 10.0 if macd_line > macd_signal and macd_hist > 0 else 0.0,
            "roc": 7.0 if roc9 > 0 else 0.0,
            "dmi": 10.0 if plus_di > minus_di else 0.0,
            "adx": 5.0 if adx_now >= 15 else 2.0 if adx_now >= 11 else 0.0,
        }
        bear = {
            "structure": 24.0 if structure == "LH_LL" else 11.0 if bear_bos else 0.0,
            "ema_alignment": 20.0 if ema_fast < ema_slow else 0.0,
            "ema_slope": 14.0 if slope < -0.03 else 7.0 if slope < 0 else 0.0,
            "price_hold": 10.0 if price < ema_fast else 0.0,
            "macd": 10.0 if macd_line < macd_signal and macd_hist < 0 else 0.0,
            "roc": 7.0 if roc9 < 0 else 0.0,
            "dmi": 10.0 if minus_di > plus_di else 0.0,
            "adx": 5.0 if adx_now >= 15 else 2.0 if adx_now >= 11 else 0.0,
        }
        bull_score = min(100.0, float(sum(bull.values())))
        bear_score = min(100.0, float(sum(bear.values())))
        edge = bull_score - bear_score

        bull_core = ema_fast > ema_slow and price > ema_fast and slope > 0
        bear_core = ema_fast < ema_slow and price < ema_fast and slope < 0

        ranges = (df["high"] - df["low"]).astype(float)
        median_range = ind.safe_float(ranges.iloc[-21:-1].median()) if len(df) >= 21 else 0.0
        atr_baseline = ind.safe_float(atr_s.iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
        volatility_shock = bool(
            (median_range > 0 and ranges.iloc[-1] > 2.5 * median_range)
            or (atr_baseline > 0 and atr_value > 1.40 * atr_baseline)
        )
        atr_pct = ind.safe_float(
            ind.atr_percentile(atr_s, c.regime_atr_pct_lookback).iloc[-1], 50.0
        )
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
            (adx_now < 10 and chop_now > 64)
            or (structure == "MIXED" and adx_now < 13 and chop_now > 60)
        )

        if bull_core and bull_score >= 80 and edge >= 18:
            label = STRONG_BULL
        elif bear_core and bear_score >= 80 and edge <= -18:
            label = STRONG_BEAR
        elif bull_score >= getattr(c, "dual_regime_early_score_min", 58.0) and edge >= getattr(c, "dual_regime_early_edge_min", 8.0) and not range_state:
            label = EARLY_BULL
        elif bear_score >= getattr(c, "dual_regime_early_score_min", 58.0) and edge <= -getattr(c, "dual_regime_early_edge_min", 8.0) and not range_state:
            label = EARLY_BEAR
        elif compression:
            label = COMPRESSION
        elif volatility_shock:
            label = HIGH_VOL
        else:
            label = RANGE

        checks = {
            "bull": bull,
            "bear": bear,
            "bull_core": bull_core,
            "bear_core": bear_core,
            "edge": round(edge, 1),
            "atr_percentile": round(atr_pct, 1),
            "volatility_shock": volatility_shock,
            "bull_bos": bool(bull_bos),
            "bear_bos": bool(bear_bos),
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

        strong_bear_conflict = (
            r4.label == STRONG_BEAR
            and r4.bear_score >= 80
            and bool(r4.checks.get("bear_bos") or r4.structure == "LH_LL")
        )
        strong_bull_conflict = (
            r4.label == STRONG_BULL
            and r4.bull_score >= 80
            and bool(r4.checks.get("bull_bos") or r4.structure == "HH_HL")
        )

        if r1.label in BULL_LABELS and not strong_bear_conflict:
            label = (
                STRONG_BULL
                if r1.label == STRONG_BULL and r4.label in BULL_LABELS
                else EARLY_BULL
            )
            reason = f"1H bull active; 4H context={r4.label}"
        elif r1.label in BEAR_LABELS and not strong_bull_conflict:
            label = (
                STRONG_BEAR
                if r1.label == STRONG_BEAR and r4.label in BEAR_LABELS
                else EARLY_BEAR
            )
            reason = f"1H bear active; 4H context={r4.label}"
        elif r4.label == STRONG_BULL and one_edge >= 8 and not strong_bear_conflict:
            label = EARLY_BULL
            reason = "4H strong bull with stabilizing 1H"
        elif r4.label == STRONG_BEAR and one_edge <= -8 and not strong_bull_conflict:
            label = EARLY_BEAR
            reason = "4H strong bear with stabilizing 1H"
        elif r4.label == COMPRESSION or r1.label == COMPRESSION:
            label = COMPRESSION
            reason = f"compression context: 4H={r4.label}, 1H={r1.label}"
        elif r4.volatility_shock or r1.volatility_shock:
            label = HIGH_VOL
            reason = f"volatility shock without confirmed 1H direction: 4H={r4.label}, 1H={r1.label}"
        else:
            label = RANGE
            reason = f"no active 1H directional regime: 4H={r4.label}, 1H={r1.label}"

        if label in BULL_LABELS:
            score = 0.40 * r4.bull_score + 0.60 * r1.bull_score
            edge = 0.40 * four_edge + 0.60 * one_edge
            style = "TREND"
        elif label in BEAR_LABELS:
            score = 0.40 * r4.bear_score + 0.60 * r1.bear_score
            edge = -(0.40 * four_edge + 0.60 * one_edge)
            style = "TREND"
        else:
            score = max(r4.bull_score, r4.bear_score, r1.bull_score, r1.bear_score)
            edge = 0.0
            style = "NO_TRADE"

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
            size_multiplier=1.0,
            bull_score_4h=r4.bull_score,
            bear_score_4h=r4.bear_score,
            bull_score_1h=r1.bull_score,
            bear_score_1h=r1.bear_score,
            direction_edge=round(edge, 1),
            volatility_shock=r4.volatility_shock or r1.volatility_shock,
        )
