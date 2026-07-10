"""
Layer 2 — MTF Bias Engine (DIRECTION, strict).  1H + 15M + 5M.

Answers: "Now that we know the Regime, which side — if any — is actually
allowed to trade?" This layer picks the Trading Direction and nothing else:
it never scores an entry trigger (that's Layer 3) and once it decides, no
downstream layer may override it.

Each timeframe gets ONE 0-100 "bull-lean" score from 9 equally-structured
components (EMA Position, EMA Alignment, MACD Direction, ROC Direction, RSI
Zone, VWAP Position, Volume, Relative Volume, Market Structure) — 100 =
every component bullish, 0 = every component bearish, ~50 = mixed.

Trading Direction (STRICT AND — every timeframe must individually clear the
bar; this is deliberately NOT a weighted blend):
    LONG ONLY   <- regime in {STRONG_BULL_TREND, EARLY_BULL_TREND}
                   AND 1H > 70 AND 15M > 70 AND 5M > 70
    SHORT ONLY  <- regime in {STRONG_BEAR_TREND, EARLY_BEAR_TREND}
                   AND 1H < 30 AND 15M < 30 AND 5M < 30
    NO TRADE    <- anything else (regime/bias mismatch, TF conflict, Range,
                   Compression, High Volatility, below threshold)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from regime_engine import BULL_LABELS, BEAR_LABELS

BIAS_BULL    = "BULL"
BIAS_BEAR    = "BEAR"
BIAS_NEUTRAL = "NEUTRAL"

LONG    = "LONG"
SHORT   = "SHORT"
NEUTRAL = "NEUTRAL"


@dataclass
class TFBiasScore:
    score: float                          # 0-100 bull-lean
    components: dict = field(default_factory=dict)


@dataclass
class BiasResult:
    direction: str            # LONG | SHORT | NEUTRAL  (Trading Direction)
    score_1h: float
    score_15m: float
    score_5m: float
    reason: str
    components: dict = field(default_factory=dict)
    # backward-compat fields (health monitor / status log / telegram read these)
    bias: str = BIAS_NEUTRAL
    bull_score: float = 0.0
    bear_score: float = 0.0
    confidence: float = 0.0
    weighted_score: float = 0.0
    aligned: bool = False
    allow_entry: bool = False
    structure: str = "—"


class BiasEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── One timeframe -> single 0-100 bull-lean score, 9 components ──────────
    def _tf_score(self, df: pd.DataFrame) -> TFBiasScore:
        c = self.cfg
        if len(df) < max(c.bias_ema_slow, 30):
            return TFBiasScore(50.0)

        closes, highs, lows, vols = df["close"], df["high"], df["low"], df["volume"]
        price = float(closes.iloc[-1])
        comp = {}

        e20 = ind.ema(closes, c.bias_ema_fast)
        e50 = ind.ema(closes, c.bias_ema_slow)
        e20_v, e50_v = float(e20.iloc[-1]), float(e50.iloc[-1])
        comp["ema_position"] = 10.0 if price > e20_v else 0.0
        comp["ema_alignment"] = 15.0 if e20_v > e50_v else 0.0

        _, _, hist = ind.macd(closes)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        comp["macd_direction"] = 15.0 if h_now > h_prev else 0.0

        roc_v = float(ind.roc(closes, c.bias_roc_period).iloc[-1] or 0.0)
        comp["roc_direction"] = 10.0 if roc_v > 0 else 0.0

        rsi_v = float(ind.rsi(closes, 14).iloc[-1])
        comp["rsi_zone"] = 10.0 if rsi_v >= 50 else 0.0

        vwap_s = ind.vwap(df, window=min(48, len(df) - 1))
        vwap_v = float(vwap_s.iloc[-1])
        comp["vwap_position"] = 10.0 if (not np.isnan(vwap_v) and price > vwap_v) else 0.0

        o = float(df["open"].iloc[-1])
        vol_now = float(vols.iloc[-1])
        vol_ma20 = float(vols.iloc[-21:-1].mean()) if len(vols) >= 21 else 0.0
        vol_up_dir = vol_ma20 > 0 and vol_now > vol_ma20 and price >= o
        comp["volume"] = 10.0 if vol_up_dir else 0.0

        rel_vol = (vol_now / vol_ma20) if vol_ma20 > 0 else 1.0
        comp["relative_volume"] = 10.0 if rel_vol >= c.bias_rel_vol_min else 0.0

        sflags = ind.structure_flags(highs, lows, c.bias_structure_left, c.bias_structure_right)
        comp["market_structure"] = 10.0 if sflags["higher_high"] or sflags["higher_low"] else 0.0

        return TFBiasScore(round(sum(comp.values()), 1), comp)

    def analyze(self, df_1h: pd.DataFrame, df_15m: pd.DataFrame | None = None,
               df_5m: pd.DataFrame | None = None, regime_label: str = "") -> BiasResult:
        c = self.cfg
        s1h = self._tf_score(df_1h)
        # 15M/5M optional in callers with only 1H available (legacy health-check
        # call sites) — fall back to the 1H score so those degrade safely to
        # "same as 1H" rather than crashing, but LIVE/backtest entries always
        # pass real 15M/5M frames since the strict gate needs them for real.
        s15 = self._tf_score(df_15m) if df_15m is not None and len(df_15m) else s1h
        s5 = self._tf_score(df_5m) if df_5m is not None and len(df_5m) else s15

        bull_regime = regime_label in BULL_LABELS
        bear_regime = regime_label in BEAR_LABELS
        thr_hi, thr_lo = c.bias_long_threshold, c.bias_short_threshold

        long_ok = bull_regime and s1h.score > thr_hi and s15.score > thr_hi and s5.score > thr_hi
        short_ok = bear_regime and s1h.score < thr_lo and s15.score < thr_lo and s5.score < thr_lo

        if long_ok:
            direction = LONG
            reason = f"LONG ONLY: regime={regime_label} 1H={s1h.score:.0f} 15M={s15.score:.0f} 5M={s5.score:.0f} (all > {thr_hi:.0f})"
        elif short_ok:
            direction = SHORT
            reason = f"SHORT ONLY: regime={regime_label} 1H={s1h.score:.0f} 15M={s15.score:.0f} 5M={s5.score:.0f} (all < {thr_lo:.0f})"
        else:
            direction = NEUTRAL
            if not (bull_regime or bear_regime):
                reason = f"NO TRADE: regime={regime_label} is not a bull/bear trend"
            elif bull_regime:
                reason = (f"NO TRADE: bull regime but TF bias not all > {thr_hi:.0f} "
                          f"(1H={s1h.score:.0f} 15M={s15.score:.0f} 5M={s5.score:.0f})")
            else:
                reason = (f"NO TRADE: bear regime but TF bias not all < {thr_lo:.0f} "
                          f"(1H={s1h.score:.0f} 15M={s15.score:.0f} 5M={s5.score:.0f})")

        bias_str = BIAS_BULL if direction == LONG else BIAS_BEAR if direction == SHORT else BIAS_NEUTRAL
        avg_score = (s1h.score + s15.score + s5.score) / 3.0

        return BiasResult(
            direction=direction, score_1h=s1h.score, score_15m=s15.score, score_5m=s5.score,
            reason=reason, components={"1h": s1h.components, "15m": s15.components, "5m": s5.components},
            bias=bias_str, bull_score=round(avg_score, 1), bear_score=round(100.0 - avg_score, 1),
            confidence=round(abs(avg_score - 50.0) * 2.0, 1), weighted_score=round(avg_score, 1),
            aligned=long_ok or short_ok, allow_entry=long_ok or short_ok,
        )
