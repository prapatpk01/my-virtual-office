"""Layer 2: independent Bull/Bear multi-timeframe bias scoring.

Unlike the previous implementation, Bear score is not computed as 100-Bull.
Every directional condition has an exact inverse. Missing or ambiguous evidence
adds zero to both sides and can never silently become a Short signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import indicators as ind
from config import Config
from regime_engine import BULL_LABELS, BEAR_LABELS, STRONG_BULL, STRONG_BEAR

BIAS_BULL = "BULL"
BIAS_BEAR = "BEAR"
BIAS_NEUTRAL = "NEUTRAL"

LONG = "LONG"
SHORT = "SHORT"
NEUTRAL = "NEUTRAL"


@dataclass
class TFBiasScore:
    score: float  # bull-lean display score, 0..100
    bull_score: float = 0.0
    bear_score: float = 0.0
    direction: str = BIAS_NEUTRAL
    components: dict = field(default_factory=dict)


@dataclass
class BiasResult:
    direction: str
    score_1h: float
    score_15m: float
    score_5m: float
    reason: str
    components: dict = field(default_factory=dict)
    bias: str = BIAS_NEUTRAL
    bull_score: float = 0.0
    bear_score: float = 0.0
    confidence: float = 0.0
    weighted_score: float = 0.0
    aligned: bool = False
    allow_entry: bool = False
    structure: str = "—"
    directional_edge: float = 0.0


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _tf_score(self, df: pd.DataFrame) -> TFBiasScore:
        c = self.cfg
        if df is None or len(df) < max(c.bias_ema_slow + 10, 70):
            return TFBiasScore(50.0, direction=BIAS_NEUTRAL, components={"insufficient": True})

        close = df["close"]
        price = ind.safe_float(close.iloc[-1])
        ema20 = ind.ema(close, c.bias_ema_fast)
        ema50 = ind.ema(close, c.bias_ema_slow)
        atr_s = ind.atr(df, 14)
        slope20 = ind.safe_float(ind.normalized_slope(ema20, atr_s, 3).iloc[-1])
        line, signal, hist = ind.macd(close)
        macd_line = ind.safe_float(line.iloc[-1])
        macd_signal = ind.safe_float(signal.iloc[-1])
        macd_hist = ind.safe_float(hist.iloc[-1])
        roc9 = ind.safe_float(ind.roc(close, c.bias_roc_period).iloc[-1])
        rsi14 = ind.safe_float(ind.rsi(close, 14).iloc[-1], 50.0)
        adx_s, plus_di_s, minus_di_s = ind.adx(df, 14)
        adx_now = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_di_s.iloc[-1])
        minus_di = ind.safe_float(minus_di_s.iloc[-1])
        structure = ind.market_structure(
            df["high"], df["low"], c.bias_structure_left, c.bias_structure_right
        )
        bull_bos, _ = ind.latest_bos(
            df, LONG, c.bias_structure_left, c.bias_structure_right, 0.18
        )
        bear_bos, _ = ind.latest_bos(
            df, SHORT, c.bias_structure_left, c.bias_structure_right, 0.18
        )
        rolling_vwap = ind.vwap(df, min(48, max(len(df) - 1, 1)))
        vwap_value = ind.safe_float(rolling_vwap.iloc[-1], price)
        candle = ind.candle_metrics(df, ind.safe_float(atr_s.iloc[-1]))

        bull = {
            "structure": 25.0 if structure == "HH_HL" else 12.0 if bull_bos else 0.0,
            "trend": 15.0 if ema20.iloc[-1] > ema50.iloc[-1] else 0.0,
            "price": 10.0 if price > ema20.iloc[-1] else 0.0,
            "slope": 10.0 if slope20 > 0.03 else 5.0 if slope20 > 0 else 0.0,
            "macd": 10.0 if macd_line > macd_signal and macd_hist > 0 else 0.0,
            "roc": 7.5 if roc9 > 0 else 0.0,
            "rsi": 5.0 if 50 <= rsi14 <= 72 else 2.5 if rsi14 > 50 else 0.0,
            "dmi": 10.0 if plus_di > minus_di else 0.0,
            "adx": 2.5 if adx_now >= 13 else 0.0,
            "vwap": 2.5 if price > vwap_value else 0.0,
            "directional_volume": 2.5 if candle.volume_ratio >= c.bias_rel_vol_min and candle.bullish else 0.0,
        }
        bear = {
            "structure": 25.0 if structure == "LH_LL" else 12.0 if bear_bos else 0.0,
            "trend": 15.0 if ema20.iloc[-1] < ema50.iloc[-1] else 0.0,
            "price": 10.0 if price < ema20.iloc[-1] else 0.0,
            "slope": 10.0 if slope20 < -0.03 else 5.0 if slope20 < 0 else 0.0,
            "macd": 10.0 if macd_line < macd_signal and macd_hist < 0 else 0.0,
            "roc": 7.5 if roc9 < 0 else 0.0,
            "rsi": 5.0 if 28 <= rsi14 <= 50 else 2.5 if rsi14 < 50 else 0.0,
            "dmi": 10.0 if minus_di > plus_di else 0.0,
            "adx": 2.5 if adx_now >= 13 else 0.0,
            "vwap": 2.5 if price < vwap_value else 0.0,
            "directional_volume": 2.5 if candle.volume_ratio >= c.bias_rel_vol_min and candle.bearish else 0.0,
        }
        bull_score = min(100.0, float(sum(bull.values())))
        bear_score = min(100.0, float(sum(bear.values())))
        edge = bull_score - bear_score
        display = max(0.0, min(100.0, 50.0 + edge / 2.0))
        direction = (
            BIAS_BULL if edge >= 10 and bull_score >= 55
            else BIAS_BEAR if edge <= -10 and bear_score >= 55
            else BIAS_NEUTRAL
        )
        return TFBiasScore(
            score=round(display, 1),
            bull_score=round(bull_score, 1),
            bear_score=round(bear_score, 1),
            direction=direction,
            components={"bull": bull, "bear": bear, "structure": structure, "edge": round(edge, 1)},
        )

    def _weight_profile(self, regime_label: str) -> tuple[float, float, float, float]:
        c = self.cfg
        if regime_label in (STRONG_BULL, STRONG_BEAR):
            return (
                c.bias_w1h_confirmed,
                c.bias_w15m_confirmed,
                c.bias_w5m_confirmed,
                c.bias_threshold_confirmed,
            )
        if regime_label in BULL_LABELS + BEAR_LABELS:
            return (
                c.bias_w1h_early,
                c.bias_w15m_early,
                c.bias_w5m_early,
                c.bias_threshold_early,
            )
        return (
            c.bias_w1h_default,
            c.bias_w15m_default,
            c.bias_w5m_default,
            c.bias_threshold_default,
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
        s5 = self._tf_score(df_5m) if df_5m is not None and len(df_5m) else s15
        w1, w15, w5, legacy_threshold = self._weight_profile(regime_label)

        combined_bull = s1.bull_score * w1 + s15.bull_score * w15 + s5.bull_score * w5
        combined_bear = s1.bear_score * w1 + s15.bear_score * w15 + s5.bear_score * w5
        edge = combined_bull - combined_bear

        bull_regime = regime_label in BULL_LABELS
        bear_regime = regime_label in BEAR_LABELS
        strong = regime_label in (STRONG_BULL, STRONG_BEAR)
        threshold = max(60.0 if strong else 58.0, legacy_threshold - 5.0)

        # 5M is a timing modifier, not a hard direction gate.  It only blocks if
        # it is very strongly opposite; this avoids missing clean 15M pullbacks.
        five_min_strong_bear = s5.bear_score - s5.bull_score >= 25
        five_min_strong_bull = s5.bull_score - s5.bear_score >= 25

        long_ok = (
            bull_regime
            and combined_bull >= threshold
            and edge >= getattr(c, "bias_min_directional_edge", 8.0)
            and s1.bull_score >= getattr(c, "bias_1h_min_bull", 55.0)
            and s1.bull_score - s1.bear_score >= 5.0
            and s15.bull_score >= getattr(c, "bias_15m_min_bull", 52.0)
            and not five_min_strong_bear
        )
        short_ok = (
            bear_regime
            and combined_bear >= threshold
            and edge <= -getattr(c, "bias_min_directional_edge", 8.0)
            and s1.bear_score >= getattr(c, "bias_1h_min_bull", 55.0)
            and s1.bear_score - s1.bull_score >= 5.0
            and s15.bear_score >= getattr(c, "bias_15m_min_bull", 52.0)
            and not five_min_strong_bull
        )

        detail = (
            f"regime={regime_label} bull={combined_bull:.1f} bear={combined_bear:.1f} "
            f"edge={edge:+.1f} | 1H B/S={s1.bull_score:.0f}/{s1.bear_score:.0f} "
            f"15M={s15.bull_score:.0f}/{s15.bear_score:.0f} "
            f"5M={s5.bull_score:.0f}/{s5.bear_score:.0f}"
        )
        if long_ok and not short_ok:
            direction, reason, bias = LONG, f"LONG ONLY: {detail}", BIAS_BULL
        elif short_ok and not long_ok:
            direction, reason, bias = SHORT, f"SHORT ONLY: {detail}", BIAS_BEAR
        else:
            direction, bias = NEUTRAL, BIAS_NEUTRAL
            if not (bull_regime or bear_regime):
                reason = f"NO TRADE: non-directional regime; {detail}"
            elif abs(edge) < getattr(c, "bias_min_directional_edge", 8.0):
                reason = f"NO TRADE: directional edge too small; {detail}"
            else:
                reason = f"NO TRADE: HTF/15M bias quality failed; {detail}"

        bull_lean = max(0.0, min(100.0, 50.0 + edge / 2.0))
        return BiasResult(
            direction=direction,
            score_1h=s1.score,
            score_15m=s15.score,
            score_5m=s5.score,
            reason=reason,
            components={"1h": s1.components, "15m": s15.components, "5m": s5.components},
            bias=bias,
            bull_score=round(combined_bull, 1),
            bear_score=round(combined_bear, 1),
            confidence=round(min(100.0, abs(edge)), 1),
            weighted_score=round(bull_lean, 1),
            aligned=long_ok or short_ok,
            allow_entry=long_ok or short_ok,
            structure=str(s1.components.get("structure", "—")),
            directional_edge=round(edge, 1),
        )
