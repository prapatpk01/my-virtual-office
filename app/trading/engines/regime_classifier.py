"""
Layer 3: Market Regime Classifier — the heart of the system.

Scores every candidate Primary Regime independently, then picks the
single highest-scoring one (winner-takes-all, mirrors Layer 4's
selection style one level up). A Secondary State tag (volatility
character) is computed independently and travels alongside the
Primary Regime — it does not compete for the same slot.

Primary Regime (mutually exclusive, exactly one wins):
  BULL_TREND | BEAR_TREND | RANGE | COMPRESSION | BREAKOUT |
  REVERSAL | EXHAUSTION | TRANSITION

Secondary State (independent volatility character):
  LOW_VOLATILITY | NORMAL_VOLATILITY | HIGH_VOLATILITY | EXPANSION

This is a score-based classifier feeding Layer 4 (hard regime gate at
the Layer-3 boundary — Layer 4 picks exactly one strategy from the
winning regime), not a hard gate itself; component scores are exposed
in `detail` for observability/learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import indicators as ind


class PrimaryRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE = "range"
    COMPRESSION = "compression"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    EXHAUSTION = "exhaustion"
    TRANSITION = "transition"


class SecondaryState(str, Enum):
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    HIGH_VOLATILITY = "high_volatility"
    EXPANSION = "expansion"


@dataclass
class RegimeResult:
    primary: PrimaryRegime
    confidence: float                 # 0-100, the winning regime's score
    secondary: SecondaryState
    scores: dict[str, float]          # every regime's score, for observability
    detail: dict = field(default_factory=dict)

    def is_trend(self) -> bool:
        return self.primary in (PrimaryRegime.BULL_TREND, PrimaryRegime.BEAR_TREND)


class RegimeClassifier:

    def analyze(self, candles: list) -> RegimeResult:
        if not candles or len(candles) < 60:
            return RegimeResult(
                primary=PrimaryRegime.TRANSITION, confidence=0.0,
                secondary=SecondaryState.NORMAL_VOLATILITY,
                scores={}, detail={"reason": "insufficient_candles"},
            )

        closes = np.array([float(c.close) for c in candles], dtype=float)
        highs = np.array([float(c.high) for c in candles], dtype=float)
        lows = np.array([float(c.low) for c in candles], dtype=float)
        vols = np.array([float(c.volume) for c in candles], dtype=float)

        # ── Shared indicator computations (each regime scorer reuses these) ──
        ema20 = ind.ema(closes, 20)
        ema50 = ind.ema(closes, 50)
        adx_arr, pdi, mdi = ind.adx(closes, highs, lows, 14)
        atr_arr = ind.atr(closes, highs, lows, 14)
        _, _, _, bb_width = ind.bollinger(closes, 20, 2.0)
        rsi_arr = ind.rsi(closes, 14)
        roc_arr = ind.roc(closes, 10)
        swing_highs, swing_lows = ind.swing_points(highs, lows, lookback=3)

        adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 15.0
        pdi_val = float(pdi[-1]) if not np.isnan(pdi[-1]) else 50.0
        mdi_val = float(mdi[-1]) if not np.isnan(mdi[-1]) else 50.0
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else closes[-1] * 0.01
        atr_pct_now = atr_val / closes[-1] * 100 if closes[-1] > 0 else 1.0
        atr_series_pct = 100 * atr_arr[~np.isnan(atr_arr)] / closes[~np.isnan(atr_arr)] \
            if np.any(~np.isnan(atr_arr)) else np.array([1.0])
        atr_percentile = ind.percentile_rank(atr_series_pct, atr_pct_now)

        bb_now = float(bb_width[-1]) if not np.isnan(bb_width[-1]) else 0.05
        bb_valid = bb_width[~np.isnan(bb_width)]
        bb_percentile = ind.percentile_rank(bb_valid, bb_now) if len(bb_valid) else 50.0
        bb_prev_avg = float(np.nanmean(bb_width[-10:-3])) if len(bb_width) >= 13 else bb_now
        bb_expanding = bb_now > bb_prev_avg * 1.15

        price = closes[-1]
        ema_bull_aligned = price > ema20[-1] > ema50[-1] if not np.isnan(ema50[-1]) else False
        ema_bear_aligned = price < ema20[-1] < ema50[-1] if not np.isnan(ema50[-1]) else False

        rel_vol = float(np.mean(vols[-5:])) / (float(np.mean(vols[-25:-5])) + 1e-9)

        # HH/HL vs LL/LH from recent swing points
        structure = self._structure_from_swings(swing_highs, swing_lows)

        # Trend persistence: consecutive bars on the same side of EMA20
        persistence = self._trend_persistence(closes, ema20)

        # Divergence: price HH but RSI lower-high (bear div) / price LL but RSI higher-low (bull div)
        bull_div, bear_div = self._divergence(closes, rsi_arr, swing_highs, swing_lows)

        # BOS: close beyond recent swing high/low (break of structure)
        bos_up, bos_down = self._bos(closes, swing_highs, swing_lows)

        # Liquidity sweep: wick beyond a recent equal high/low then closes back inside
        sweep_up, sweep_down = self._liquidity_sweep(highs, lows, closes)

        scores: dict[str, float] = {}
        detail: dict = {}

        scores[PrimaryRegime.BULL_TREND.value], detail["bull_trend"] = self._score_bull_trend(
            ema_bull_aligned, adx_val, pdi_val, mdi_val, structure, persistence, roc_arr[-1]
        )
        scores[PrimaryRegime.BEAR_TREND.value], detail["bear_trend"] = self._score_bear_trend(
            ema_bear_aligned, adx_val, pdi_val, mdi_val, structure, persistence, roc_arr[-1]
        )
        scores[PrimaryRegime.RANGE.value], detail["range"] = self._score_range(
            adx_val, bb_percentile, structure, rel_vol
        )
        scores[PrimaryRegime.COMPRESSION.value], detail["compression"] = self._score_compression(
            bb_percentile, atr_percentile, rel_vol, bb_expanding
        )
        scores[PrimaryRegime.BREAKOUT.value], detail["breakout"] = self._score_breakout(
            bb_expanding, bb_percentile, rel_vol, bos_up or bos_down, atr_percentile
        )
        scores[PrimaryRegime.REVERSAL.value], detail["reversal"] = self._score_reversal(
            bull_div, bear_div, sweep_up or sweep_down, adx_val
        )
        scores[PrimaryRegime.EXHAUSTION.value], detail["exhaustion"] = self._score_exhaustion(
            persistence, rsi_arr[-1], atr_percentile, bull_div, bear_div
        )
        scores[PrimaryRegime.TRANSITION.value], detail["transition"] = self._score_transition(scores)

        best_key = max(scores, key=lambda k: scores[k])
        primary = PrimaryRegime(best_key)
        confidence = round(scores[best_key], 1)

        secondary = self._secondary_state(atr_percentile, bb_expanding)

        return RegimeResult(
            primary=primary, confidence=confidence, secondary=secondary,
            scores={k: round(v, 1) for k, v in scores.items()}, detail=detail,
        )

    # ── Regime scorers ──────────────────────────────────────────────────────

    @staticmethod
    def _score_bull_trend(aligned, adx_val, pdi_val, mdi_val, structure, persistence, roc_last):
        score = 0.0
        notes = []
        if aligned:
            score += 35; notes.append("ema_bull_aligned")
        if adx_val >= 25 and pdi_val > mdi_val:
            score += 30; notes.append(f"adx={adx_val:.0f}_+DI")
        elif adx_val >= 18 and pdi_val > mdi_val:
            score += 15
        if structure == "hh_hl":
            score += 20; notes.append("hh_hl")
        if persistence > 0:
            score += min(15.0, persistence * 1.5); notes.append(f"persist={persistence}")
        if roc_last > 0:
            score += 0  # momentum already implied by ADX/structure; avoid double count
        return round(min(100.0, score), 1), ", ".join(notes)

    @staticmethod
    def _score_bear_trend(aligned, adx_val, pdi_val, mdi_val, structure, persistence, roc_last):
        score = 0.0
        notes = []
        if aligned:
            score += 35; notes.append("ema_bear_aligned")
        if adx_val >= 25 and mdi_val > pdi_val:
            score += 30; notes.append(f"adx={adx_val:.0f}_-DI")
        elif adx_val >= 18 and mdi_val > pdi_val:
            score += 15
        if structure == "ll_lh":
            score += 20; notes.append("ll_lh")
        if persistence < 0:
            score += min(15.0, abs(persistence) * 1.5); notes.append(f"persist={persistence}")
        return round(min(100.0, score), 1), ", ".join(notes)

    @staticmethod
    def _score_range(adx_val, bb_percentile, structure, rel_vol):
        score = 0.0
        notes = []
        if adx_val < 20:
            score += 40; notes.append(f"adx_low={adx_val:.0f}")
        elif adx_val < 25:
            score += 20
        if 25 <= bb_percentile <= 70:
            score += 30; notes.append(f"bb_mid_pctile={bb_percentile:.0f}")
        if structure == "mixed":
            score += 20; notes.append("structure_mixed")
        if rel_vol < 1.1:
            score += 10
        return round(min(100.0, score), 1), ", ".join(notes)

    @staticmethod
    def _score_compression(bb_percentile, atr_percentile, rel_vol, bb_expanding):
        score = 0.0
        notes = []
        if bb_percentile <= 20:
            score += 45; notes.append(f"bb_pctile={bb_percentile:.0f}")
        elif bb_percentile <= 35:
            score += 20
        if atr_percentile <= 25:
            score += 30; notes.append(f"atr_pctile={atr_percentile:.0f}")
        if rel_vol < 0.8:
            score += 15; notes.append("low_volume")
        if bb_expanding:
            score -= 20  # already breaking out of compression, not compressed anymore
        return round(max(0.0, min(100.0, score)), 1), ", ".join(notes)

    @staticmethod
    def _score_breakout(bb_expanding, bb_percentile, rel_vol, bos, atr_percentile):
        score = 0.0
        notes = []
        if bb_expanding:
            score += 35; notes.append("bb_expanding")
        if rel_vol >= 1.4:
            score += 30; notes.append(f"rel_vol={rel_vol:.2f}")
        elif rel_vol >= 1.1:
            score += 15
        if bos:
            score += 25; notes.append("bos")
        if atr_percentile >= 60:
            score += 10
        return round(min(100.0, score), 1), ", ".join(notes)

    @staticmethod
    def _score_reversal(bull_div, bear_div, sweep, adx_val):
        score = 0.0
        notes = []
        if bull_div or bear_div:
            score += 40; notes.append(f"div bull={bull_div} bear={bear_div}")
        if sweep:
            score += 35; notes.append("liquidity_sweep")
        if adx_val >= 25:
            score += 15  # reversal against an established trend is more meaningful
        return round(min(100.0, score), 1), ", ".join(notes)

    @staticmethod
    def _score_exhaustion(persistence, rsi_last, atr_percentile, bull_div, bear_div):
        score = 0.0
        notes = []
        if abs(persistence) >= 8:
            score += 30; notes.append(f"persist={persistence}")
        if rsi_last >= 75 or rsi_last <= 25:
            score += 30; notes.append(f"rsi={rsi_last:.0f}")
        if atr_percentile >= 80:
            score += 20; notes.append(f"atr_pctile={atr_percentile:.0f}")
        if bull_div or bear_div:
            score += 20; notes.append("divergence_present")
        return round(min(100.0, score), 1), ", ".join(notes)

    @staticmethod
    def _score_transition(other_scores: dict) -> tuple[float, str]:
        # Transition wins when nothing else has a decisive edge.
        best_other = max(other_scores.values()) if other_scores else 0.0
        score = max(0.0, 55.0 - best_other * 0.5)
        return round(score, 1), f"best_other={best_other:.0f}"

    # ── Secondary state ──────────────────────────────────────────────────────

    @staticmethod
    def _secondary_state(atr_percentile: float, bb_expanding: bool) -> SecondaryState:
        if bb_expanding and atr_percentile >= 55:
            return SecondaryState.EXPANSION
        if atr_percentile >= 75:
            return SecondaryState.HIGH_VOLATILITY
        if atr_percentile <= 30:
            return SecondaryState.LOW_VOLATILITY
        return SecondaryState.NORMAL_VOLATILITY

    # ── Structure / pattern helpers ─────────────────────────────────────────

    @staticmethod
    def _structure_from_swings(swing_highs, swing_lows) -> str:
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "mixed"
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1] > swing_lows[-2][1]
        ll = swing_lows[-1][1] < swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        if hh and hl:
            return "hh_hl"
        if ll and lh:
            return "ll_lh"
        return "mixed"

    @staticmethod
    def _trend_persistence(closes: np.ndarray, ema20: np.ndarray) -> int:
        """Signed count of consecutive recent bars closing on one side of EMA20."""
        count = 0
        for i in range(len(closes) - 1, max(0, len(closes) - 30), -1):
            if np.isnan(ema20[i]):
                break
            side = 1 if closes[i] > ema20[i] else -1
            if count == 0:
                count = side
            elif (count > 0 and side > 0) or (count < 0 and side < 0):
                count += side
            else:
                break
        return count

    @staticmethod
    def _divergence(closes, rsi_arr, swing_highs, swing_lows) -> tuple[bool, bool]:
        bull_div = False
        bear_div = False
        if len(swing_highs) >= 2:
            i1, h1 = swing_highs[-2]
            i2, h2 = swing_highs[-1]
            if h2 > h1 and rsi_arr[i2] < rsi_arr[i1]:
                bear_div = True
        if len(swing_lows) >= 2:
            i1, l1 = swing_lows[-2]
            i2, l2 = swing_lows[-1]
            if l2 < l1 and rsi_arr[i2] > rsi_arr[i1]:
                bull_div = True
        return bull_div, bear_div

    @staticmethod
    def _bos(closes, swing_highs, swing_lows) -> tuple[bool, bool]:
        price = closes[-1]
        bos_up = bool(swing_highs) and price > swing_highs[-1][1]
        bos_down = bool(swing_lows) and price < swing_lows[-1][1]
        return bos_up, bos_down

    @staticmethod
    def _liquidity_sweep(highs, lows, closes, lookback: int = 15) -> tuple[bool, bool]:
        if len(highs) < lookback + 2:
            return False, False
        window_h = highs[-lookback - 1:-1]
        window_l = lows[-lookback - 1:-1]
        prior_high = float(window_h.max())
        prior_low = float(window_l.min())
        last_high, last_low, last_close = highs[-1], lows[-1], closes[-1]
        sweep_up = last_high > prior_high and last_close < prior_high
        sweep_down = last_low < prior_low and last_close > prior_low
        return sweep_up, sweep_down
