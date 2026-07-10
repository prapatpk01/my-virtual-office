"""
Layer 0: Market Quality Engine — the hard gate.

Runs before any directional/regime analysis. If the market itself is
untradeable (illiquid, noisy, dead session, unstable ticks), nothing
downstream should be trusted — so this layer can veto the whole pipeline
before Layer 1 even runs.

Score components (100 pts total):
  Liquidity Score      20 pts  — relative volume vs recent average
  Spread/Noise Score    20 pts  — wick-to-body ratio, choppiness
  ATR Quality           15 pts  — ATR in a sane band (not dead, not insane)
  Session Quality       15 pts  — hour-of-day activity heuristic
  Relative Volume       15 pts  — current vs rolling average volume
  Tick Stability        15 pts  — consecutive-candle price stability (no gaps/spikes)

Bands:
  90-100  Excellent
  75-89   Good
  60-74   Acceptable
  40-59   Poor
  <40     No Trade — hard veto
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import indicators as ind


class QualityBand(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    NO_TRADE = "no_trade"


@dataclass
class MarketQualityResult:
    score: float
    band: QualityBand
    tradeable: bool
    detail: dict = field(default_factory=dict)

    def blocks_trading(self) -> bool:
        return not self.tradeable


class MarketQualityEngine:
    def __init__(self, no_trade_threshold: float = 40.0):
        self.no_trade_threshold = no_trade_threshold

    def analyze(self, candles: list) -> MarketQualityResult:
        if not candles or len(candles) < 30:
            return MarketQualityResult(
                score=50.0, band=QualityBand.ACCEPTABLE, tradeable=True,
                detail={"reason": "insufficient_candles_default_pass"},
            )

        closes = np.array([float(c.close) for c in candles], dtype=float)
        highs = np.array([float(c.high) for c in candles], dtype=float)
        lows = np.array([float(c.low) for c in candles], dtype=float)
        opens = np.array([float(c.open) for c in candles], dtype=float)
        vols = np.array([float(c.volume) for c in candles], dtype=float)

        score = 0.0
        detail: dict = {}

        # ── Liquidity / relative volume (20 pts) ────────────────────────────
        recent_vol = float(np.mean(vols[-5:]))
        base_vol = float(np.mean(vols[-30:-5])) if len(vols) >= 30 else float(np.mean(vols))
        rel_vol = recent_vol / base_vol if base_vol > 0 else 1.0
        if rel_vol >= 0.5:
            liq_pts = min(20.0, 10.0 + rel_vol * 8)
        else:
            liq_pts = max(0.0, rel_vol * 20)
        score += liq_pts
        detail["liquidity"] = f"rel_vol={rel_vol:.2f} ({liq_pts:.0f}pts)"

        # ── Spread / noise: wick-to-body ratio (20 pts) ─────────────────────
        bodies = np.abs(closes[-20:] - opens[-20:])
        ranges = (highs[-20:] - lows[-20:])
        ranges_safe = np.where(ranges > 0, ranges, 1e-9)
        body_ratio = float(np.mean(bodies / ranges_safe))
        # High body ratio (candles mostly "body", little wick) = cleaner moves
        noise_pts = min(20.0, body_ratio * 28)
        score += noise_pts
        detail["noise"] = f"body_ratio={body_ratio:.2f} ({noise_pts:.0f}pts)"

        # ── ATR quality: not dead, not insane (15 pts) ──────────────────────
        atr_arr = ind.atr(closes, highs, lows, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        atr_pct = (atr_val / closes[-1] * 100) if closes[-1] > 0 else 0.0
        if 0.15 <= atr_pct <= 3.0:
            atr_pts = 15.0
        elif 0.08 <= atr_pct < 0.15 or 3.0 < atr_pct <= 5.0:
            atr_pts = 8.0
        else:
            atr_pts = 2.0
        score += atr_pts
        detail["atr_quality"] = f"atr%={atr_pct:.3f} ({atr_pts:.0f}pts)"

        # ── Session quality (15 pts) — hour-of-day activity heuristic ───────
        ts = candles[-1].timestamp
        hour_utc = int((ts // 3600000) % 24) if ts > 1e10 else int((ts // 3600) % 24)
        # Rough liquid-session windows (UTC): London 7-16, NY 12-21, overlap 12-16 best
        if 12 <= hour_utc <= 16:
            session_pts = 15.0
        elif 7 <= hour_utc <= 21:
            session_pts = 11.0
        else:
            session_pts = 6.0
        score += session_pts
        detail["session"] = f"hour_utc={hour_utc} ({session_pts:.0f}pts)"

        # ── Relative volume trend (15 pts) — is volume declining/rising ─────
        vol_trend = recent_vol / (float(np.mean(vols[-15:-5])) + 1e-9)
        if vol_trend >= 0.8:
            vol_pts = min(15.0, 8.0 + vol_trend * 5)
        else:
            vol_pts = max(0.0, vol_trend * 15)
        score += vol_pts
        detail["relative_volume"] = f"trend={vol_trend:.2f} ({vol_pts:.0f}pts)"

        # ── Tick stability: no extreme single-candle spikes (15 pts) ────────
        pct_moves = np.abs(np.diff(closes[-15:]) / closes[-15:-1])
        max_move = float(np.max(pct_moves)) * 100 if len(pct_moves) > 0 else 0.0
        if max_move <= 1.5:
            stability_pts = 15.0
        elif max_move <= 3.0:
            stability_pts = 9.0
        else:
            stability_pts = 3.0
        score += stability_pts
        detail["tick_stability"] = f"max_move%={max_move:.2f} ({stability_pts:.0f}pts)"

        score = round(max(0.0, min(100.0, score)), 1)
        band = self._classify(score)
        tradeable = score >= self.no_trade_threshold

        return MarketQualityResult(score=score, band=band, tradeable=tradeable, detail=detail)

    @staticmethod
    def _classify(score: float) -> QualityBand:
        if score >= 90:
            return QualityBand.EXCELLENT
        if score >= 75:
            return QualityBand.GOOD
        if score >= 60:
            return QualityBand.ACCEPTABLE
        if score >= 40:
            return QualityBand.POOR
        return QualityBand.NO_TRADE
