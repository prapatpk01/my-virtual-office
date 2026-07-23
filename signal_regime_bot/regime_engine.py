"""DUALCORE V3.0 regime router.

4H is a macro conflict filter. 1H is the active regime. Range and compression
are tradable modes, but EntryEngine restricts them to specialized SMC/sweep or
breakout setups instead of trend-continuation signals.
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
        if df is None or len(df) < 80:
            return TFRegime(RANGE, checks={"insufficient_data": True})

        close = df["close"].astype(float)
        price = ind.safe_float(close.iloc[-1])
        ema10_s = ind.ema(close, 10)
        ema20_s = ind.ema(close, 20)
        ema50_s = ind.ema(close, 50)
        ema10 = ind.safe_float(ema10_s.iloc[-1])
        ema20 = ind.safe_float(ema20_s.iloc[-1])
        ema50 = ind.safe_float(ema50_s.iloc[-1])
        atr_s = ind.atr(df, 14)
        atr_v = max(ind.safe_float(atr_s.iloc[-1]), 1e-12)
        slope20 = ind.safe_float(ind.normalized_slope(ema20_s, atr_s, 3).iloc[-1])
        slope50 = ind.safe_float(ind.normalized_slope(ema50_s, atr_s, 5).iloc[-1])
        adx_s, plus_s, minus_s = ind.adx(df, 14)
        adx_v = ind.safe_float(adx_s.iloc[-1])
        plus = ind.safe_float(plus_s.iloc[-1])
        minus = ind.safe_float(minus_s.iloc[-1])
        chop = ind.safe_float(ind.choppiness_index(df, 14).iloc[-1], 100.0)
        macd_l, macd_s, macd_h = ind.macd(close)
        hist = ind.safe_float(macd_h.iloc[-1])
        roc9 = ind.safe_float(ind.roc(close, 9).iloc[-1])
        structure = ind.market_structure(df["high"], df["low"], 3, 3)
        bull_bos, bull_level = ind.latest_bos(df, LONG, 3, 3, 0.14)
        bear_bos, bear_level = ind.latest_bos(df, SHORT, 3, 3, 0.14)

        bull = {
            "structure": 22.0 if structure == "HH_HL" else 11.0 if bull_bos else 0.0,
            "ema_stack": 20.0 if ema10 > ema20 > ema50 else 13.0 if ema10 > ema20 else 0.0,
            "price_hold": 10.0 if price > ema20 else 5.0 if price > ema50 else 0.0,
            "slope": 13.0 if slope20 > 0.04 and slope50 >= 0 else 7.0 if slope20 > 0 else 0.0,
            "momentum": 10.0 if hist > 0 and roc9 > 0 else 5.0 if hist > 0 or roc9 > 0 else 0.0,
            "dmi": 10.0 if plus > minus else 4.0 if plus + 2 >= minus else 0.0,
            "adx": 8.0 if adx_v >= 18 else 5.0 if adx_v >= 13 else 2.0 if adx_v >= 10 else 0.0,
            "anti_chop": 7.0 if chop <= 55 else 3.0 if chop <= 62 else 0.0,
        }
        bear = {
            "structure": 22.0 if structure == "LH_LL" else 11.0 if bear_bos else 0.0,
            "ema_stack": 20.0 if ema10 < ema20 < ema50 else 13.0 if ema10 < ema20 else 0.0,
            "price_hold": 10.0 if price < ema20 else 5.0 if price < ema50 else 0.0,
            "slope": 13.0 if slope20 < -0.04 and slope50 <= 0 else 7.0 if slope20 < 0 else 0.0,
            "momentum": 10.0 if hist < 0 and roc9 < 0 else 5.0 if hist < 0 or roc9 < 0 else 0.0,
            "dmi": 10.0 if minus > plus else 4.0 if minus + 2 >= plus else 0.0,
            "adx": 8.0 if adx_v >= 18 else 5.0 if adx_v >= 13 else 2.0 if adx_v >= 10 else 0.0,
            "anti_chop": 7.0 if chop <= 55 else 3.0 if chop <= 62 else 0.0,
        }
        bull_score = min(100.0, sum(bull.values()))
        bear_score = min(100.0, sum(bear.values()))
        edge = bull_score - bear_score

        width = ind.bollinger_width(df, 20)
        width_pct = ind.safe_float(ind.rolling_percentile(width, 100).iloc[-1], 50.0)
        atr_pct = ind.safe_float(ind.atr_percentile(atr_s, 100).iloc[-1], 50.0)
        ranges = (df["high"] - df["low"]).astype(float)
        median_range = ind.safe_float(ranges.iloc[-21:-1].median()) if len(df) >= 21 else atr_v
        vol_ma = ind.safe_float(df["volume"].iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
        vol_ratio = ind.safe_float(df["volume"].iloc[-1]) / max(vol_ma, 1e-12)
        shock = bool(
            ranges.iloc[-1] >= 2.7 * max(median_range, 1e-12)
            or (atr_pct >= 90 and vol_ratio >= 1.8)
        )
        compression = bool(width_pct <= 27 and atr_pct <= 30 and adx_v <= 18)
        range_state = bool(
            (adx_v < 12 and chop >= 60)
            or (structure == "MIXED" and adx_v < 16 and chop >= 57)
        )

        if shock:
            label = HIGH_VOL
        elif compression:
            label = COMPRESSION
        elif range_state and abs(edge) < 18:
            label = RANGE
        elif bull_score >= 68 and edge >= 16:
            label = STRONG_BULL
        elif bear_score >= 68 and edge <= -16:
            label = STRONG_BEAR
        elif bull_score >= 53 and edge >= 6:
            label = EARLY_BULL
        elif bear_score >= 53 and edge <= -6:
            label = EARLY_BEAR
        else:
            label = RANGE

        return TFRegime(
            label=label,
            bull_score=round(bull_score, 1),
            bear_score=round(bear_score, 1),
            checks={
                "bull": bull,
                "bear": bear,
                "edge": round(edge, 1),
                "ema10": ema10,
                "ema20": ema20,
                "ema50": ema50,
                "adx": round(adx_v, 1),
                "chop": round(chop, 1),
                "atr_pct": round(atr_pct, 1),
                "bb_width_pct": round(width_pct, 1),
                "bull_bos": bool(bull_bos),
                "bear_bos": bool(bear_bos),
                "bull_bos_level": bull_level,
                "bear_bos_level": bear_level,
            },
            pass_count=int(max(bull_score, bear_score)),
            adx=adx_v,
            chop=chop,
            structure=structure,
            volatility_shock=shock,
        )

    def analyze(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> RegimeResult:
        c = self.cfg
        r4 = self._tf_regime(df_4h)
        r1 = self._tf_regime(df_1h)
        edge4 = r4.bull_score - r4.bear_score
        edge1 = r1.bull_score - r1.bear_score

        conflict_bear = r4.bear_score >= c.expert_htf_conflict_score and edge4 <= -c.expert_htf_conflict_edge
        conflict_bull = r4.bull_score >= c.expert_htf_conflict_score and edge4 >= c.expert_htf_conflict_edge

        label = r1.label
        reason_bits = [f"4H={r4.label} {r4.bull_score:.0f}/{r4.bear_score:.0f}",
                       f"1H={r1.label} {r1.bull_score:.0f}/{r1.bear_score:.0f}"]

        # Only a confirmed opposite 4H macro blocks the active 1H direction.
        if label in BULL_LABELS and conflict_bear:
            label = RANGE
            reason_bits.append("strong opposite 4H bear conflict -> range-only routing")
        elif label in BEAR_LABELS and conflict_bull:
            label = RANGE
            reason_bits.append("strong opposite 4H bull conflict -> range-only routing")
        elif label == HIGH_VOL:
            # Route high-volatility bars using the underlying 1H directional edge.
            if edge1 >= 8 and not conflict_bear:
                label = EARLY_BULL
                reason_bits.append("high-volatility bullish routing; retest preferred")
            elif edge1 <= -8 and not conflict_bull:
                label = EARLY_BEAR
                reason_bits.append("high-volatility bearish routing; retest preferred")
            else:
                label = RANGE
                reason_bits.append("high-volatility neutral routing")

        if label in (STRONG_BULL, STRONG_BEAR):
            style, size = "TREND", 1.0
        elif label in (EARLY_BULL, EARLY_BEAR):
            style, size = "SWING", 0.85
        elif label == COMPRESSION:
            style, size = "BREAKOUT", 0.70
        else:
            style, size = "MEANREV", 0.60

        active_score = max(r1.bull_score, r1.bear_score)
        return RegimeResult(
            label=label,
            label_4h=r4.label,
            label_1h=r1.label,
            checks_4h=r4.checks,
            checks_1h=r1.checks,
            reason=" | ".join(reason_bits),
            name=label,
            score=round(active_score, 1),
            style=style,
            size_multiplier=size,
            bull_score_4h=r4.bull_score,
            bear_score_4h=r4.bear_score,
            bull_score_1h=r1.bull_score,
            bear_score_1h=r1.bear_score,
            direction_edge=round(edge1, 1),
            volatility_shock=r4.volatility_shock or r1.volatility_shock,
        )
