"""DUALCORE V2.0 directional bias engine.

1H chooses the active side, 15M confirms that the side is not structurally
invalid, and 5M is informational only.  Bull and bear evidence are calculated
independently with mirrored rules.  A noisy 5M pullback can never flip the HTF
permission by itself.
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
    score: float
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
            return TFBiasScore(
                50.0,
                direction=BIAS_NEUTRAL,
                components={"insufficient": True},
            )

        close = df["close"]
        price = ind.safe_float(close.iloc[-1])
        ema20_s = ind.ema(close, c.bias_ema_fast)
        ema50_s = ind.ema(close, c.bias_ema_slow)
        ema20 = ind.safe_float(ema20_s.iloc[-1])
        ema50 = ind.safe_float(ema50_s.iloc[-1])
        atr_s = ind.atr(df, 14)
        atr_value = ind.safe_float(atr_s.iloc[-1])
        slope20 = ind.safe_float(ind.normalized_slope(ema20_s, atr_s, 3).iloc[-1])

        line, signal, hist = ind.macd(close)
        macd_line = ind.safe_float(line.iloc[-1])
        macd_signal = ind.safe_float(signal.iloc[-1])
        macd_hist = ind.safe_float(hist.iloc[-1])
        roc9 = ind.safe_float(ind.roc(close, c.bias_roc_period).iloc[-1])
        rsi14 = ind.safe_float(ind.rsi(close, 14).iloc[-1], 50.0)
        adx_s, plus_s, minus_s = ind.adx(df, 14)
        adx_now = ind.safe_float(adx_s.iloc[-1])
        plus_di = ind.safe_float(plus_s.iloc[-1])
        minus_di = ind.safe_float(minus_s.iloc[-1])

        structure = ind.market_structure(
            df["high"], df["low"], c.bias_structure_left, c.bias_structure_right
        )
        bull_bos, bull_level = ind.latest_bos(
            df, LONG, c.bias_structure_left, c.bias_structure_right, 0.18
        )
        bear_bos, bear_level = ind.latest_bos(
            df, SHORT, c.bias_structure_left, c.bias_structure_right, 0.18
        )
        rolling_vwap = ind.vwap(df, min(48, max(len(df) - 1, 1)))
        vwap_value = ind.safe_float(rolling_vwap.iloc[-1], price)
        candle = ind.candle_metrics(df, atr_value)

        # Evidence groups are independent and mirrored.  No side is derived as
        # the complement of the other.
        bull = {
            "structure": 22.0 if structure == "HH_HL" else 11.0 if bull_bos else 0.0,
            "ema_alignment": 16.0 if ema20 > ema50 else 0.0,
            "price_hold": 10.0 if price > ema20 else 0.0,
            "ema_slope": 10.0 if slope20 > 0.03 else 5.0 if slope20 > 0 else 0.0,
            "macd": 10.0 if macd_line > macd_signal and macd_hist > 0 else 0.0,
            "roc": 7.0 if roc9 > 0 else 0.0,
            "dmi": 10.0 if plus_di > minus_di else 0.0,
            "adx": 5.0 if adx_now >= 15 else 2.5 if adx_now >= 11 else 0.0,
            "rsi": 4.0 if 50 <= rsi14 <= 72 else 2.0 if rsi14 > 50 else 0.0,
            "vwap": 3.0 if price > vwap_value else 0.0,
            "directional_volume": (
                3.0
                if candle.volume_ratio >= c.bias_rel_vol_min and candle.bullish
                else 0.0
            ),
        }
        bear = {
            "structure": 22.0 if structure == "LH_LL" else 11.0 if bear_bos else 0.0,
            "ema_alignment": 16.0 if ema20 < ema50 else 0.0,
            "price_hold": 10.0 if price < ema20 else 0.0,
            "ema_slope": 10.0 if slope20 < -0.03 else 5.0 if slope20 < 0 else 0.0,
            "macd": 10.0 if macd_line < macd_signal and macd_hist < 0 else 0.0,
            "roc": 7.0 if roc9 < 0 else 0.0,
            "dmi": 10.0 if minus_di > plus_di else 0.0,
            "adx": 5.0 if adx_now >= 15 else 2.5 if adx_now >= 11 else 0.0,
            "rsi": 4.0 if 28 <= rsi14 <= 50 else 2.0 if rsi14 < 50 else 0.0,
            "vwap": 3.0 if price < vwap_value else 0.0,
            "directional_volume": (
                3.0
                if candle.volume_ratio >= c.bias_rel_vol_min and candle.bearish
                else 0.0
            ),
        }
        bull_score = min(100.0, float(sum(bull.values())))
        bear_score = min(100.0, float(sum(bear.values())))
        edge = bull_score - bear_score
        display = max(0.0, min(100.0, 50.0 + edge / 2.0))
        direction = (
            BIAS_BULL
            if edge >= 10 and bull_score >= 55
            else BIAS_BEAR
            if edge <= -10 and bear_score >= 55
            else BIAS_NEUTRAL
        )
        return TFBiasScore(
            score=round(display, 1),
            bull_score=round(bull_score, 1),
            bear_score=round(bear_score, 1),
            direction=direction,
            components={
                "bull": bull,
                "bear": bear,
                "structure": structure,
                "edge": round(edge, 1),
                "bull_bos": bool(bull_bos),
                "bear_bos": bool(bear_bos),
                "bull_bos_level": bull_level,
                "bear_bos_level": bear_level,
                "adx": round(adx_now, 1),
            },
        )

    def _weights(self, regime_label: str) -> tuple[float, float]:
        # 5M is deliberately excluded from permission. It remains visible in
        # diagnostics and is compared locally inside EntryEngine.
        if regime_label in (STRONG_BULL, STRONG_BEAR):
            return 0.60, 0.40
        if regime_label in BULL_LABELS + BEAR_LABELS:
            return 0.52, 0.48
        return 0.55, 0.45

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
        # 5M permission is evaluated by EntryEngine; avoid duplicate full score.
        s5 = s15
        w1, w15 = self._weights(regime_label)

        combined_bull = s1.bull_score * w1 + s15.bull_score * w15
        combined_bear = s1.bear_score * w1 + s15.bear_score * w15
        edge = combined_bull - combined_bear
        minimum_edge = getattr(c, "bias_min_directional_edge", 8.0)

        bull_regime = regime_label in BULL_LABELS
        bear_regime = regime_label in BEAR_LABELS
        one_hour_bull_edge = s1.bull_score - s1.bear_score
        one_hour_bear_edge = -one_hour_bull_edge
        fifteen_bull_edge = s15.bull_score - s15.bear_score
        fifteen_bear_edge = -fifteen_bull_edge

        one_hour_bear_shift = bool(s1.components.get("bear_bos"))
        one_hour_bull_shift = bool(s1.components.get("bull_bos"))
        fifteen_bear_shift = bool(s15.components.get("bear_bos"))
        fifteen_bull_shift = bool(s15.components.get("bull_bos"))

        long_ok = (
            bull_regime
            and combined_bull >= getattr(c, "bias_combined_min", 55.0)
            and edge >= minimum_edge
            and s1.bull_score >= getattr(c, "bias_1h_min_bull", 56.0)
            and one_hour_bull_edge >= getattr(c, "bias_1h_edge_min", 4.0)
            and s15.bull_score >= getattr(c, "bias_15m_min_bull", 50.0)
            and fifteen_bull_edge >= getattr(c, "bias_15m_edge_floor", -5.0)
            and not (one_hour_bear_shift and s1.bear_score >= s1.bull_score + getattr(c, "bias_opposite_bos_margin", 5.0))
            and not (fifteen_bear_shift and s15.bear_score >= s15.bull_score + getattr(c, "bias_opposite_bos_margin", 5.0))
        )
        short_ok = (
            bear_regime
            and combined_bear >= getattr(c, "bias_combined_min", 55.0)
            and edge <= -minimum_edge
            and s1.bear_score >= getattr(c, "bias_1h_min_bull", 56.0)
            and one_hour_bear_edge >= getattr(c, "bias_1h_edge_min", 4.0)
            and s15.bear_score >= getattr(c, "bias_15m_min_bull", 50.0)
            and fifteen_bear_edge >= getattr(c, "bias_15m_edge_floor", -5.0)
            and not (one_hour_bull_shift and s1.bull_score >= s1.bear_score + getattr(c, "bias_opposite_bos_margin", 5.0))
            and not (fifteen_bull_shift and s15.bull_score >= s15.bear_score + getattr(c, "bias_opposite_bos_margin", 5.0))
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
            elif abs(edge) < minimum_edge:
                reason = f"NO TRADE: directional edge too small; {detail}"
            elif one_hour_bear_shift or one_hour_bull_shift:
                reason = f"NO TRADE: fresh opposite 1H structure shift; {detail}"
            else:
                reason = f"NO TRADE: 1H/15M bias quality failed; {detail}"

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
