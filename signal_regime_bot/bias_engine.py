"""DUALCORE V3.0 adaptive bias engine.

Trend regimes produce a preferred side. Range/compression regimes return BOTH
so the EntryEngine may choose only specialized SMC edge/sweep or compression
breakout setups. Ordinary 15M disagreement raises setup thresholds; only a
strong opposite structure/momentum cluster is a hard veto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

import indicators as ind
from config import Config
from regime_engine import (
    STRONG_BULL, EARLY_BULL, STRONG_BEAR, EARLY_BEAR,
    RANGE, COMPRESSION, BULL_LABELS, BEAR_LABELS,
)

LONG = "LONG"
SHORT = "SHORT"
BOTH = "BOTH"
NEUTRAL = "NEUTRAL"
BIAS_BULL = "BULL"
BIAS_BEAR = "BEAR"
BIAS_NEUTRAL = "NEUTRAL"


@dataclass
class TFBiasScore:
    score: float
    bull_score: float = 0.0
    bear_score: float = 0.0
    direction: str = BIAS_NEUTRAL
    components: dict = field(default_factory=dict)


@dataclass
class BiasResult:
    direction: str
    score_1h: float = 50.0
    score_15m: float = 50.0
    score_5m: float = 50.0
    reason: str = ""
    components: dict = field(default_factory=dict)
    bias: str = BIAS_NEUTRAL
    bull_score: float = 0.0
    bear_score: float = 0.0
    confidence: float = 0.0
    weighted_score: float = 50.0
    aligned: bool = False
    allow_entry: bool = False
    structure: str = "—"
    directional_edge: float = 0.0


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _tf_score(self, df: pd.DataFrame) -> TFBiasScore:
        c = self.cfg
        if df is None or len(df) < 70:
            return TFBiasScore(50.0, components={"insufficient": True})

        close = df["close"].astype(float)
        price = ind.safe_float(close.iloc[-1])
        ema8_s = ind.ema(close, 8)
        ema20_s = ind.ema(close, 20)
        ema50_s = ind.ema(close, 50)
        ema8 = ind.safe_float(ema8_s.iloc[-1])
        ema20 = ind.safe_float(ema20_s.iloc[-1])
        ema50 = ind.safe_float(ema50_s.iloc[-1])
        atr_s = ind.atr(df, 14)
        atr_v = max(ind.safe_float(atr_s.iloc[-1]), 1e-12)
        slope20 = ind.safe_float(ind.normalized_slope(ema20_s, atr_s, 3).iloc[-1])
        line, signal, hist_s = ind.macd(close)
        hist = ind.safe_float(hist_s.iloc[-1])
        roc9 = ind.safe_float(ind.roc(close, 9).iloc[-1])
        rsi14 = ind.safe_float(ind.rsi(close, 14).iloc[-1], 50.0)
        adx_s, plus_s, minus_s = ind.adx(df, 14)
        adx_v = ind.safe_float(adx_s.iloc[-1])
        plus = ind.safe_float(plus_s.iloc[-1])
        minus = ind.safe_float(minus_s.iloc[-1])
        structure = ind.market_structure(df["high"], df["low"], 3, 3)
        bull_bos, bull_level = ind.latest_bos(df, LONG, 3, 3, 0.12)
        bear_bos, bear_level = ind.latest_bos(df, SHORT, 3, 3, 0.12)
        vwap_v = ind.safe_float(ind.vwap(df, min(48, max(2, len(df) - 1))).iloc[-1], price)
        candle = ind.candle_metrics(df, atr_v)

        bull = {
            "structure": 20.0 if structure == "HH_HL" else 10.0 if bull_bos else 0.0,
            "ema_stack": 17.0 if ema8 > ema20 > ema50 else 11.0 if ema8 > ema20 else 0.0,
            "price_hold": 10.0 if price > ema20 else 5.0 if price > ema50 else 0.0,
            "slope": 10.0 if slope20 > 0.03 else 5.0 if slope20 > 0 else 0.0,
            "macd_roc": 12.0 if hist > 0 and roc9 > 0 else 6.0 if hist > 0 or roc9 > 0 else 0.0,
            "dmi": 10.0 if plus > minus else 4.0 if plus + 2 >= minus else 0.0,
            "adx": 6.0 if adx_v >= 16 else 3.0 if adx_v >= 11 else 0.0,
            "rsi": 5.0 if 50 <= rsi14 <= 72 else 2.0 if rsi14 > 48 else 0.0,
            "vwap": 4.0 if price >= vwap_v else 0.0,
            "candle": 6.0 if candle.bullish and candle.bull_close_quality >= 0.60 else 3.0 if candle.bullish else 0.0,
        }
        bear = {
            "structure": 20.0 if structure == "LH_LL" else 10.0 if bear_bos else 0.0,
            "ema_stack": 17.0 if ema8 < ema20 < ema50 else 11.0 if ema8 < ema20 else 0.0,
            "price_hold": 10.0 if price < ema20 else 5.0 if price < ema50 else 0.0,
            "slope": 10.0 if slope20 < -0.03 else 5.0 if slope20 < 0 else 0.0,
            "macd_roc": 12.0 if hist < 0 and roc9 < 0 else 6.0 if hist < 0 or roc9 < 0 else 0.0,
            "dmi": 10.0 if minus > plus else 4.0 if minus + 2 >= plus else 0.0,
            "adx": 6.0 if adx_v >= 16 else 3.0 if adx_v >= 11 else 0.0,
            "rsi": 5.0 if 28 <= rsi14 <= 50 else 2.0 if rsi14 < 52 else 0.0,
            "vwap": 4.0 if price <= vwap_v else 0.0,
            "candle": 6.0 if candle.bearish and candle.bear_close_quality >= 0.60 else 3.0 if candle.bearish else 0.0,
        }
        bull_score = min(100.0, sum(bull.values()))
        bear_score = min(100.0, sum(bear.values()))
        edge = bull_score - bear_score
        display = max(0.0, min(100.0, 50.0 + edge / 2.0))
        direction = BIAS_BULL if edge >= 7 else BIAS_BEAR if edge <= -7 else BIAS_NEUTRAL
        return TFBiasScore(
            score=round(display, 1),
            bull_score=round(bull_score, 1),
            bear_score=round(bear_score, 1),
            direction=direction,
            components={
                "bull": bull,
                "bear": bear,
                "edge": round(edge, 1),
                "structure": structure,
                "bull_bos": bool(bull_bos),
                "bear_bos": bool(bear_bos),
                "bull_bos_level": bull_level,
                "bear_bos_level": bear_level,
                "adx": round(adx_v, 1),
                "rsi": round(rsi14, 1),
            },
        )

    def analyze(
        self,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame | None = None,
        df_5m: pd.DataFrame | None = None,
        regime_label: str = "",
    ) -> BiasResult:
        c = self.cfg
        s1 = self._tf_score(df_1h)
        s15 = self._tf_score(df_15m) if df_15m is not None and len(df_15m) else s1
        s5 = self._tf_score(df_5m) if df_5m is not None and len(df_5m) >= 70 else s15

        if regime_label in (STRONG_BULL, STRONG_BEAR):
            w1, w15, w5 = 0.55, 0.35, 0.10
        elif regime_label in (EARLY_BULL, EARLY_BEAR):
            w1, w15, w5 = 0.45, 0.45, 0.10
        else:
            w1, w15, w5 = 0.40, 0.45, 0.15

        bull = s1.bull_score * w1 + s15.bull_score * w15 + s5.bull_score * w5
        bear = s1.bear_score * w1 + s15.bear_score * w15 + s5.bear_score * w5
        edge = bull - bear
        detail = (
            f"regime={regime_label} B/S={bull:.1f}/{bear:.1f} edge={edge:+.1f} | "
            f"1H={s1.bull_score:.0f}/{s1.bear_score:.0f} "
            f"15M={s15.bull_score:.0f}/{s15.bear_score:.0f} "
            f"5M={s5.bull_score:.0f}/{s5.bear_score:.0f}"
        )

        # Range/compression are intentionally passed to EntryEngine in BOTH
        # mode. Only its regime-specific setup families may fire.
        if regime_label in (RANGE, COMPRESSION):
            return BiasResult(
                direction=BOTH,
                score_1h=s1.score,
                score_15m=s15.score,
                score_5m=s5.score,
                reason=f"BOTH SIDES ROUTED: {detail}",
                components={"1h": s1.components, "15m": s15.components, "5m": s5.components},
                bias=BIAS_NEUTRAL,
                bull_score=round(bull, 1),
                bear_score=round(bear, 1),
                confidence=round(abs(edge), 1),
                weighted_score=round(max(0.0, min(100.0, 50.0 + edge / 2)), 1),
                aligned=True,
                allow_entry=True,
                structure=str(s1.components.get("structure", "—")),
                directional_edge=round(edge, 1),
            )

        long_regime = regime_label in BULL_LABELS
        short_regime = regime_label in BEAR_LABELS
        min_score = c.expert_bias_score_min
        min_edge = c.expert_bias_edge_min
        fifteen_edge = s15.bull_score - s15.bear_score
        strong_15m_bear = (
            s15.bear_score >= c.expert_htf_conflict_score
            and fifteen_edge <= -c.expert_15m_opposite_veto_edge
        )
        strong_15m_bull = (
            s15.bull_score >= c.expert_htf_conflict_score
            and fifteen_edge >= c.expert_15m_opposite_veto_edge
        )

        long_ok = (
            long_regime
            and bull >= min_score
            and edge >= min_edge
            and s1.bull_score >= 48
            and not strong_15m_bear
        )
        short_ok = (
            short_regime
            and bear >= min_score
            and edge <= -min_edge
            and s1.bear_score >= 48
            and not strong_15m_bull
        )

        if long_ok:
            direction, bias_name, reason = LONG, BIAS_BULL, f"LONG PREFERRED: {detail}"
        elif short_ok:
            direction, bias_name, reason = SHORT, BIAS_BEAR, f"SHORT PREFERRED: {detail}"
        else:
            # Soft fallback: keep the regime side tradable when 1H remains at
            # least neutral and 15M is not a strong opposite cluster. The entry
            # setup then pays a weak-context threshold surcharge.
            if long_regime and s1.bull_score >= 45 and not strong_15m_bear:
                direction, bias_name = LONG, BIAS_BULL
                reason = f"LONG SOFT PERMISSION: {detail}"
            elif short_regime and s1.bear_score >= 45 and not strong_15m_bull:
                direction, bias_name = SHORT, BIAS_BEAR
                reason = f"SHORT SOFT PERMISSION: {detail}"
            else:
                direction, bias_name = NEUTRAL, BIAS_NEUTRAL
                reason = f"NO TRADE: strong opposite bias/conflict; {detail}"

        allow = direction in (LONG, SHORT, BOTH)
        return BiasResult(
            direction=direction,
            score_1h=s1.score,
            score_15m=s15.score,
            score_5m=s5.score,
            reason=reason,
            components={"1h": s1.components, "15m": s15.components, "5m": s5.components},
            bias=bias_name,
            bull_score=round(bull, 1),
            bear_score=round(bear, 1),
            confidence=round(abs(edge), 1),
            weighted_score=round(max(0.0, min(100.0, 50.0 + edge / 2)), 1),
            aligned=allow,
            allow_entry=allow,
            structure=str(s1.components.get("structure", "—")),
            directional_edge=round(edge, 1),
        )
