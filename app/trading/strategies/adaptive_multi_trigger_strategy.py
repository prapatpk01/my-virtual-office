"""Adaptive Multi-Trigger Entry Engine V1.0 (dormant strategy module).

Architecture:
    4H trend direction
      -> 1H context / quality
      -> 15M pattern detection
      -> Volume DNA
      -> entry archetype classification
      -> adaptive trigger selection
      -> location / room / chase validation
      -> Signal

Design contract:
- Trend answers WHICH SIDE may trade.
- Context answers whether that side is currently tradeable.
- Pattern answers WHAT is happening.
- Volume DNA answers whether participation supports that setup.
- Trigger answers WHEN to execute.
- EMA8/13 and HMA16 are optional trigger candidates only.
- One symbol = one candidate = one entry signal; multiple simultaneous triggers
  are collapsed into a single candidate.

This file is intentionally NOT wired into any production runner.  It can be
backtested or enabled later without changing the currently deployed strategy.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


@dataclass
class _Pattern:
    name: str
    direction: str
    score: float
    level: Optional[float]
    trigger_candidates: list[str]
    diagnostics: dict


class AdaptiveMultiTriggerStrategy(BaseStrategy):
    """Pattern-first, setup-aware 4H/1H/15M execution engine."""

    VERSION = "1.0"
    ENTRY_TF = "15m"

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        entry_quality_threshold: float = 60.0,
        weak_context_threshold: float = 70.0,
        trigger_freshness_bars: int = 3,
        breakout_rvol_preferred: float = 1.20,
        strong_rvol: float = 1.50,
        max_ema20_extension_atr: float = 1.50,
        minimum_structure_room_r: float = 1.20,
        preferred_structure_room_r: float = 1.50,
        adx_period: int = 14,
        chop_period: int = 14,
        hma_period: int = 16,
        atr_period: int = 14,
        swing_span: int = 3,
        closed_bar_grace_ms: int = 1500,
    ):
        super().__init__(symbol=symbol, params=params)
        self.name = f"AdaptiveMultiTriggerV1({symbol})"
        self.entry_tf = self.ENTRY_TF

        self.entry_quality_threshold = float(entry_quality_threshold)
        self.weak_context_threshold = float(weak_context_threshold)
        self.trigger_freshness_bars = max(1, int(trigger_freshness_bars))
        self.breakout_rvol_preferred = max(0.1, float(breakout_rvol_preferred))
        self.strong_rvol = max(self.breakout_rvol_preferred, float(strong_rvol))
        self.max_ema20_extension_atr = max(0.25, float(max_ema20_extension_atr))
        self.minimum_structure_room_r = max(0.5, float(minimum_structure_room_r))
        self.preferred_structure_room_r = max(
            self.minimum_structure_room_r,
            float(preferred_structure_room_r),
        )
        self.adx_period = max(5, int(adx_period))
        self.chop_period = max(5, int(chop_period))
        self.hma_period = max(5, int(hma_period))
        self.atr_period = max(5, int(atr_period))
        self.swing_span = max(2, int(swing_span))
        self.closed_bar_grace_ms = max(0, int(closed_bar_grace_ms))

        self._last_entry_bar_ts: Optional[int] = None
        self._setup_state = "IDLE"
        self._latest_candles: list = []
        self._latest_15m: list = []

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ts_ms(timestamp: int) -> int:
        value = int(timestamp)
        return value * 1000 if value < 10_000_000_000 else value

    def _closed(self, candles: list, timeframe_ms: int) -> list:
        if not candles:
            return []
        cutoff = int(time.time() * 1000) - self.closed_bar_grace_ms
        return [
            c for c in candles
            if self._ts_ms(c.timestamp) + timeframe_ms <= cutoff
        ]

    @staticmethod
    def _finite(value) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_last(arr, default=0.0) -> float:
        if arr is None or len(arr) == 0:
            return float(default)
        value = arr[-1]
        return float(value) if np.isfinite(value) else float(default)

    def _choppiness(self, candles: list, period: Optional[int] = None) -> float:
        p = max(2, int(period or self.chop_period))
        if len(candles) < p + 2:
            return 100.0
        segment = candles[-p:]
        tr_sum = 0.0
        for i in range(len(candles) - p, len(candles)):
            c = candles[i]
            pc = candles[i - 1].close
            tr_sum += max(
                float(c.high - c.low),
                abs(float(c.high - pc)),
                abs(float(c.low - pc)),
            )
        high = max(float(c.high) for c in segment)
        low = min(float(c.low) for c in segment)
        span = max(high - low, 1e-12)
        ratio = max(tr_sum / span, 1.0)
        return float(100.0 * math.log10(ratio) / math.log10(p))

    @staticmethod
    def _vwap(candles: list, period: int = 30) -> float:
        segment = candles[-period:] if len(candles) > period else candles
        if not segment:
            return 0.0
        pv = 0.0
        vol = 0.0
        for c in segment:
            typical = (float(c.high) + float(c.low) + float(c.close)) / 3.0
            v = max(float(c.volume), 0.0)
            pv += typical * v
            vol += v
        return pv / vol if vol > 0 else float(segment[-1].close)

    def _confirmed_swings(self, candles: list) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        span = self.swing_span
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        sh: list[tuple[int, float]] = []
        sl: list[tuple[int, float]] = []
        for i in range(span, len(candles) - span):
            h = highs[i]
            l = lows[i]
            if h > max(highs[i - span:i]) and h >= max(highs[i + 1:i + span + 1]):
                sh.append((i, h))
            if l < min(lows[i - span:i]) and l <= min(lows[i + 1:i + span + 1]):
                sl.append((i, l))
        return sh, sl

    @staticmethod
    def _body(candle) -> float:
        return abs(float(candle.close) - float(candle.open))

    @staticmethod
    def _range(candle) -> float:
        return max(float(candle.high) - float(candle.low), 1e-12)

    # ------------------------------------------------------------------
    # Layer 1 — 4H direction only
    # ------------------------------------------------------------------

    def _trend_4h(self, candles: list) -> dict:
        if len(candles) < 60:
            return {"direction": None, "state": "WARMUP"}
        closes = [float(c.close) for c in candles]
        e20 = self.ema(closes, 20)
        e50 = self.ema(closes, 50)
        if not all(self._finite(v) for v in (e20[-1], e20[-4], e50[-1])):
            return {"direction": None, "state": "WARMUP"}

        ema20 = float(e20[-1])
        ema50 = float(e50[-1])
        slope = ema20 - float(e20[-4])
        close = closes[-1]
        sh, sl = self._confirmed_swings(candles)

        structurally_bearish = False
        structurally_bullish = False
        if len(sh) >= 2 and len(sl) >= 2:
            structurally_bullish = sh[-1][1] > sh[-2][1] and sl[-1][1] > sl[-2][1]
            structurally_bearish = sh[-1][1] < sh[-2][1] and sl[-1][1] < sl[-2][1]

        if ema20 > ema50 and slope > 0 and not structurally_bearish:
            direction = "long"
            state = "BULL"
        elif ema20 < ema50 and slope < 0 and not structurally_bullish:
            direction = "short"
            state = "BEAR"
        else:
            direction = None
            state = "UNCLEAR"

        return {
            "direction": direction,
            "state": state,
            "ema20": round(ema20, 8),
            "ema50": round(ema50, 8),
            "ema20_slope": round(slope, 8),
            "close": round(close, 8),
            "structure_bullish": structurally_bullish,
            "structure_bearish": structurally_bearish,
        }

    # ------------------------------------------------------------------
    # Layer 2 — 1H market context / quality
    # ------------------------------------------------------------------

    def _context_1h(self, candles: list, direction: str) -> dict:
        if len(candles) < max(60, self.adx_period * 2 + 5):
            return {"status": "WEAK", "score": 0.0, "reason": "warmup"}
        closes = [float(c.close) for c in candles]
        adx_arr, pdi, mdi = self.adx(candles, self.adx_period)
        adx = self._safe_last(adx_arr, 0.0)
        chop = self._choppiness(candles, self.chop_period)
        macd_line, macd_signal, hist = self.macd(closes)
        hist_now = self._safe_last(hist, 0.0)
        sh, sl = self._confirmed_swings(candles)

        trend_di_ok = (
            self._safe_last(pdi, 0.0) >= self._safe_last(mdi, 0.0)
            if direction == "long"
            else self._safe_last(mdi, 0.0) >= self._safe_last(pdi, 0.0)
        )
        momentum_ok = hist_now >= 0 if direction == "long" else hist_now <= 0

        structure_ok = True
        if len(sh) >= 2 and len(sl) >= 2:
            if direction == "long":
                structure_ok = not (sh[-1][1] < sh[-2][1] and sl[-1][1] < sl[-2][1])
            else:
                structure_ok = not (sh[-1][1] > sh[-2][1] and sl[-1][1] > sl[-2][1])

        score = 0.0
        score += min(max((adx - 10.0) / 20.0, 0.0), 1.0) * 30.0
        score += min(max((61.8 - chop) / 25.0, 0.0), 1.0) * 25.0
        score += 15.0 if trend_di_ok else 5.0
        score += 15.0 if momentum_ok else 5.0
        score += 15.0 if structure_ok else 0.0

        if chop >= 61.8 or adx < 12:
            status = "CHOP"
        elif score >= 78:
            status = "STRONG"
        elif score >= 58:
            status = "NORMAL"
        else:
            status = "WEAK"

        return {
            "status": status,
            "score": round(score, 1),
            "adx": round(adx, 2),
            "chop": round(chop, 2),
            "di_aligned": trend_di_ok,
            "momentum_ok": momentum_ok,
            "structure_ok": structure_ok,
        }

    # ------------------------------------------------------------------
    # Layer 4 — Volume DNA (pattern-aware score, never universal hard gate)
    # ------------------------------------------------------------------

    def _volume_dna(self, candles: list, pattern_name: str, direction: str) -> dict:
        if len(candles) < 25:
            return {"score": 0.0, "rvol": 0.0}
        vols = np.asarray([max(float(c.volume), 0.0) for c in candles], dtype=float)
        avg20 = max(float(np.mean(vols[-21:-1])), 1e-12)
        current = float(vols[-1])
        rvol = current / avg20

        recent5 = float(np.mean(vols[-5:]))
        prior5 = float(np.mean(vols[-10:-5])) if len(vols) >= 10 else avg20
        expansion_ratio = recent5 / max(prior5, 1e-12)

        up_vol = 0.0
        down_vol = 0.0
        for c in candles[-10:]:
            if float(c.close) >= float(c.open):
                up_vol += max(float(c.volume), 0.0)
            else:
                down_vol += max(float(c.volume), 0.0)
        agreement_ratio = (
            up_vol / max(down_vol, 1e-12)
            if direction == "long"
            else down_vol / max(up_vol, 1e-12)
        )

        relative_score = min(max((rvol - 0.6) / 1.0, 0.0), 1.0) * 10.0
        expansion_score = min(max((expansion_ratio - 0.8) / 0.7, 0.0), 1.0) * 10.0
        agreement_score = min(max((agreement_ratio - 0.7) / 1.0, 0.0), 1.0) * 10.0

        pattern_score = 5.0
        if pattern_name in {"BREAKOUT", "MOMENTUM_EXPANSION", "COMPRESSION_BREAK"}:
            if rvol >= self.strong_rvol:
                pattern_score = 10.0
            elif rvol >= self.breakout_rvol_preferred:
                pattern_score = 8.0
            elif rvol >= 0.9:
                pattern_score = 5.0
            else:
                pattern_score = 2.0
        elif pattern_name in {"PULLBACK_CONTINUATION", "BREAKOUT_RETEST"}:
            pullback_slice = vols[-4:-1]
            pullback_avg = float(np.mean(pullback_slice)) if len(pullback_slice) else avg20
            contraction = pullback_avg < avg20 and current >= pullback_avg
            pattern_score = 10.0 if contraction else 6.0 if current >= pullback_avg else 3.0
        elif pattern_name == "LIQUIDITY_SWEEP_REVERSAL":
            pattern_score = 10.0 if rvol >= 1.25 else 7.0 if rvol >= 0.9 else 4.0

        total = min(40.0, relative_score + expansion_score + agreement_score + pattern_score)
        return {
            "score": round(total, 1),
            "rvol": round(rvol, 2),
            "expansion_ratio": round(expansion_ratio, 2),
            "agreement_ratio": round(agreement_ratio, 2),
            "components": {
                "relative_volume": round(relative_score, 1),
                "volume_expansion": round(expansion_score, 1),
                "price_volume_agreement": round(agreement_score, 1),
                "pattern_behavior": round(pattern_score, 1),
            },
        }

    # ------------------------------------------------------------------
    # Layer 3 — Pattern detection / archetype classification
    # ------------------------------------------------------------------

    def _detect_patterns(self, candles: list, direction: str) -> list[_Pattern]:
        if len(candles) < 60:
            return []
        closes = np.asarray([float(c.close) for c in candles], dtype=float)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        vols = np.asarray([max(float(c.volume), 0.0) for c in candles], dtype=float)
        atr_arr = self.atr(candles, self.atr_period)
        atr = self._safe_last(atr_arr, 0.0)
        if atr <= 0:
            return []
        ema20_arr = self.ema(list(closes), 20)
        ema8 = self.ema(list(closes), 8)
        ema13 = self.ema(list(closes), 13)
        hma16 = self.hma(list(closes), self.hma_period)
        ema20 = self._safe_last(ema20_arr, closes[-1])
        vwap = self._vwap(candles, 30)
        sh, sl = self._confirmed_swings(candles)
        last_sh = sh[-1][1] if sh else None
        last_sl = sl[-1][1] if sl else None
        c = candles[-1]
        prev = candles[-2]
        patterns: list[_Pattern] = []

        bullish = direction == "long"
        direction_close = float(c.close) > float(c.open) if bullish else float(c.close) < float(c.open)
        rejection = (
            (float(c.close) - float(c.low)) / self._range(c) >= 0.65
            if bullish
            else (float(c.high) - float(c.close)) / self._range(c) >= 0.65
        )
        body_avg = float(np.mean([self._body(x) for x in candles[-21:-1]]))
        range_avg = float(np.mean([self._range(x) for x in candles[-21:-1]]))
        body_ratio = self._body(c) / max(body_avg, 1e-12)
        range_ratio = self._range(c) / max(range_avg, 1e-12)

        ema_cross = (
            ema8[-2] <= ema13[-2] and ema8[-1] > ema13[-1]
            if bullish
            else ema8[-2] >= ema13[-2] and ema8[-1] < ema13[-1]
        ) if all(self._finite(v) for v in (ema8[-2], ema8[-1], ema13[-2], ema13[-1])) else False
        hma_flip = (
            hma16[-2] <= hma16[-3] and hma16[-1] > hma16[-2]
            if bullish
            else hma16[-2] >= hma16[-3] and hma16[-1] < hma16[-2]
        ) if all(self._finite(v) for v in hma16[-3:]) else False

        # 1) Pullback continuation: return to EMA20/VWAP/last structure, then reclaim.
        refs = [ema20, vwap]
        if bullish and last_sl is not None:
            refs.append(last_sl)
        if (not bullish) and last_sh is not None:
            refs.append(last_sh)
        nearest_ref = min(refs, key=lambda x: abs(float(c.close) - x))
        touched_ref = (
            float(c.low) <= nearest_ref + 0.20 * atr and float(c.close) > nearest_ref
            if bullish
            else float(c.high) >= nearest_ref - 0.20 * atr and float(c.close) < nearest_ref
        )
        pullback_vol_contract = float(np.mean(vols[-4:-1])) < float(np.mean(vols[-10:-4]))
        if touched_ref and direction_close:
            score = 15.0 + (15.0 if rejection else 7.0) + (10.0 if pullback_vol_contract else 4.0)
            score += 10.0 if abs(float(c.close) - nearest_ref) <= 0.5 * atr else 5.0
            score += 10.0 if ((float(c.close) > ema20) if bullish else (float(c.close) < ema20)) else 3.0
            triggers = ["EMA20_RECLAIM"]
            if rejection:
                triggers.append("REJECTION_CANDLE")
            if ema_cross:
                triggers.append("EMA8_13_CROSS")
            if hma_flip:
                triggers.append("HMA16_FLIP")
            patterns.append(_Pattern("PULLBACK_CONTINUATION", direction, min(score, 60.0), nearest_ref, triggers, {"pullback_volume_contract": pullback_vol_contract, "rejection": rejection}))

        # 2) Breakout of latest confirmed swing.
        breakout_level = last_sh if bullish else last_sl
        if breakout_level is not None:
            confirmed_break = (
                float(prev.close) <= breakout_level and float(c.close) > breakout_level
                if bullish
                else float(prev.close) >= breakout_level and float(c.close) < breakout_level
            )
            if confirmed_break:
                score = 25.0 + (15.0 if direction_close else 5.0) + min(range_ratio, 2.0) / 2.0 * 10.0
                score += 10.0 if body_ratio >= 1.2 else 5.0
                patterns.append(_Pattern("BREAKOUT", direction, min(score, 60.0), breakout_level, ["BOS_CLOSE", "RANGE_BREAKOUT"] + (["EMA8_13_CROSS"] if ema_cross else []) + (["HMA16_FLIP"] if hma_flip else []), {"body_ratio": round(body_ratio, 2), "range_ratio": round(range_ratio, 2)}))

        # 3) Breakout retest: breakout within last 1-3 bars + hold/reclaim now.
        if breakout_level is not None:
            for lag in range(1, min(self.trigger_freshness_bars, 3) + 1):
                idx = -1 - lag
                before = idx - 1
                if abs(before) > len(candles):
                    continue
                broke = (
                    float(candles[before].close) <= breakout_level and float(candles[idx].close) > breakout_level
                    if bullish
                    else float(candles[before].close) >= breakout_level and float(candles[idx].close) < breakout_level
                )
                held = (
                    float(c.low) <= breakout_level + 0.20 * atr and float(c.close) > breakout_level
                    if bullish
                    else float(c.high) >= breakout_level - 0.20 * atr and float(c.close) < breakout_level
                )
                if broke and held and rejection:
                    patterns.append(_Pattern("BREAKOUT_RETEST", direction, 52.0, breakout_level, ["RETEST_HOLD", "REJECTION_CANDLE"], {"breakout_lag": lag, "rejection": True}))
                    break

        # 4) Structure continuation via recent higher-low / lower-high reclaim.
        if len(sh) >= 2 and len(sl) >= 2:
            if bullish:
                structure_valid = sh[-1][1] >= sh[-2][1] and sl[-1][1] > sl[-2][1]
                structure_level = sl[-1][1]
                reclaim = float(c.close) > float(prev.high)
            else:
                structure_valid = sh[-1][1] < sh[-2][1] and sl[-1][1] <= sl[-2][1]
                structure_level = sh[-1][1]
                reclaim = float(c.close) < float(prev.low)
            if structure_valid and reclaim:
                patterns.append(_Pattern("STRUCTURE_CONTINUATION", direction, 50.0 + (8.0 if rejection else 0.0), structure_level, ["MICRO_BOS", "STRUCTURE_RECLAIM"], {"structure_valid": True, "rejection": rejection}))

        # 5) Liquidity sweep reversal in the higher-timeframe permitted direction.
        sweep_level = last_sl if bullish else last_sh
        if sweep_level is not None:
            swept = (
                float(c.low) < sweep_level and float(c.close) > sweep_level
                if bullish
                else float(c.high) > sweep_level and float(c.close) < sweep_level
            )
            displacement = body_ratio >= 1.15 and direction_close
            if swept and (rejection or displacement):
                triggers = ["SWEEP_RECLAIM"]
                if displacement:
                    triggers.append("DISPLACEMENT")
                patterns.append(_Pattern("LIQUIDITY_SWEEP_REVERSAL", direction, 48.0 + (10.0 if displacement else 5.0), sweep_level, triggers, {"swept": True, "displacement": displacement}))

        # 6) Momentum expansion.
        recent_structure = max(highs[-8:-1]) if bullish else min(lows[-8:-1])
        structure_break = float(c.close) > recent_structure if bullish else float(c.close) < recent_structure
        if direction_close and body_ratio >= 1.35 and range_ratio >= 1.25 and structure_break:
            patterns.append(_Pattern("MOMENTUM_EXPANSION", direction, min(60.0, 46.0 + min(body_ratio, 2.0) * 6.0), float(recent_structure), ["MOMENTUM_CANDLE", "MICRO_BOS"], {"body_ratio": round(body_ratio, 2), "range_ratio": round(range_ratio, 2)}))

        # 7) Compression break: last 4 ranges compressed vs prior baseline, then expansion.
        recent_ranges = np.asarray([self._range(x) for x in candles[-12:]], dtype=float)
        compressed = float(np.mean(recent_ranges[-5:-1])) <= 0.72 * float(np.mean(recent_ranges[:-5])) if len(recent_ranges) >= 10 else False
        comp_high = max(float(x.high) for x in candles[-5:-1])
        comp_low = min(float(x.low) for x in candles[-5:-1])
        comp_break = float(c.close) > comp_high if bullish else float(c.close) < comp_low
        if compressed and comp_break and range_ratio >= 1.10:
            patterns.append(_Pattern("COMPRESSION_BREAK", direction, 52.0 + (6.0 if range_ratio >= 1.4 else 0.0), comp_high if bullish else comp_low, ["COMPRESSION_BREAK"], {"compressed": True, "range_ratio": round(range_ratio, 2)}))

        return patterns

    # ------------------------------------------------------------------
    # Trigger selection / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _trigger_priority(trigger: str) -> int:
        structure = {"MICRO_BOS", "STRUCTURE_RECLAIM", "BOS_CLOSE", "SWEEP_RECLAIM"}
        reclaim = {"EMA20_RECLAIM", "RETEST_HOLD", "REJECTION_CANDLE"}
        momentum = {"RANGE_BREAKOUT", "MOMENTUM_CANDLE", "COMPRESSION_BREAK", "DISPLACEMENT"}
        if trigger in structure:
            return 1
        if trigger in reclaim:
            return 2
        if trigger in momentum:
            return 3
        return 4

    def _select_trigger(self, pattern: _Pattern) -> Optional[str]:
        valid = [t for t in pattern.trigger_candidates if t]
        if not valid:
            return None
        return sorted(valid, key=self._trigger_priority)[0]

    def _location_and_chase(self, candles: list, direction: str, current_price: float, pattern: _Pattern) -> dict:
        closes = [float(c.close) for c in candles]
        atr_arr = self.atr(candles, self.atr_period)
        atr = self._safe_last(atr_arr, 0.0)
        ema20 = self._safe_last(self.ema(closes, 20), current_price)
        if atr <= 0:
            return {"valid": False, "reason": "ATR warmup"}

        sh, sl = self._confirmed_swings(candles)
        if direction == "long":
            opposing = [level for _i, level in sh if level > current_price]
            nearest_opposing = min(opposing) if opposing else None
        else:
            opposing = [level for _i, level in sl if level < current_price]
            nearest_opposing = max(opposing) if opposing else None

        stop_reference = pattern.level
        if stop_reference is None:
            stop_reference = current_price - atr if direction == "long" else current_price + atr
        risk_distance = max(abs(current_price - float(stop_reference)), 0.50 * atr)
        room = (
            abs(float(nearest_opposing) - current_price)
            if nearest_opposing is not None
            else 2.0 * risk_distance
        )
        room_r = room / max(risk_distance, 1e-12)

        ema_extension = abs(current_price - ema20) / atr
        avg_range = float(np.mean([self._range(c) for c in candles[-21:-1]]))
        current_range_ratio = self._range(candles[-1]) / max(avg_range, 1e-12)

        expansion_count = 0
        for c in candles[-4:]:
            if self._range(c) > 1.25 * avg_range:
                expansion_count += 1

        strong_breakout_exception = pattern.name in {"BREAKOUT", "MOMENTUM_EXPANSION", "COMPRESSION_BREAK"} and room_r >= self.preferred_structure_room_r
        chase = (
            ema_extension > self.max_ema20_extension_atr
            or current_range_ratio > 1.8
            or expansion_count >= 3
        )
        if chase and strong_breakout_exception:
            chase = ema_extension > self.max_ema20_extension_atr * 1.25 or current_range_ratio > 2.20

        location_valid = room_r >= self.minimum_structure_room_r
        return {
            "valid": bool(location_valid and not chase),
            "room_r": round(room_r, 2),
            "ema20_extension_atr": round(ema_extension, 2),
            "current_range_ratio": round(current_range_ratio, 2),
            "recent_expansion_candles": expansion_count,
            "nearest_opposing_structure": nearest_opposing,
            "chase_rejected": bool(chase),
            "location_valid": bool(location_valid),
        }

    # ------------------------------------------------------------------
    # Public strategy entry point
    # ------------------------------------------------------------------

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._setup_state = "IDLE"
        mtf = mtf_candles or {}
        c15 = self._closed(candles, 15 * 60_000)
        c1h = self._closed(mtf.get("1h", []), 60 * 60_000)
        c4h = self._closed(mtf.get("4h", []), 4 * 60 * 60_000)
        self._latest_candles = c15
        self._latest_15m = c15

        metadata = {
            "strategy": "ADAPTIVE_MULTI_TRIGGER_V1",
            "selected_strategy": "Adaptive Multi-Trigger V1",
            "version": self.VERSION,
            "entry_tf": self.ENTRY_TF,
            "setup_state": self._setup_state,
        }

        if len(c15) < 60 or len(c1h) < 60 or len(c4h) < 60:
            metadata["bars"] = {"15m": len(c15), "1h": len(c1h), "4h": len(c4h)}
            return self._hold(current_price, "Adaptive Multi-Trigger warm-up", metadata)

        trend = self._trend_4h(c4h)
        metadata["trend_4h"] = trend
        direction = trend.get("direction")
        if direction not in {"long", "short"}:
            return self._hold(current_price, "Layer1 4H trend unclear — no trade", metadata)

        context = self._context_1h(c1h, direction)
        metadata["context_1h"] = context
        if context.get("status") == "CHOP":
            return self._hold(current_price, "Layer2 1H context = CHOP — no trade", metadata)

        patterns = self._detect_patterns(c15, direction)
        self._setup_state = "PATTERN_DETECTED" if patterns else "IDLE"
        metadata["setup_state"] = self._setup_state
        if not patterns:
            return self._hold(current_price, "15M: no valid adaptive entry archetype", metadata)

        candidates = []
        self._setup_state = "VALIDATING_VOLUME"
        for pattern in patterns:
            volume = self._volume_dna(c15, pattern.name, direction)
            total = min(100.0, pattern.score + float(volume.get("score", 0.0)))
            trigger = self._select_trigger(pattern)
            candidates.append({
                "pattern": pattern,
                "volume": volume,
                "quality": total,
                "trigger": trigger,
            })

        candidates.sort(key=lambda x: (x["quality"], -self._trigger_priority(x["trigger"] or "")), reverse=True)
        best = candidates[0]
        pattern: _Pattern = best["pattern"]
        quality = float(best["quality"])
        trigger = best["trigger"]
        metadata.update({
            "setup_state": "ARMED",
            "entry_archetype": pattern.name,
            "pattern_quality": round(pattern.score, 1),
            "volume_dna": best["volume"],
            "entry_quality": round(quality, 1),
            "adaptive_trigger": trigger,
            "pattern_diagnostics": pattern.diagnostics,
            "alternative_candidates": [
                {
                    "pattern": c["pattern"].name,
                    "quality": round(float(c["quality"]), 1),
                    "trigger": c["trigger"],
                }
                for c in candidates[1:4]
            ],
        })

        threshold = self.entry_quality_threshold
        if context.get("status") == "WEAK":
            threshold = max(threshold, self.weak_context_threshold)
        elif 55.0 <= quality < self.entry_quality_threshold and context.get("status") == "STRONG":
            threshold = 55.0

        if quality < threshold:
            return self._hold(current_price, f"Entry quality {quality:.1f} < required {threshold:.1f}", metadata)
        if trigger is None:
            return self._hold(current_price, f"{pattern.name}: pattern valid but no fresh trigger", metadata)

        location = self._location_and_chase(c15, direction, current_price, pattern)
        metadata["location_chase"] = location
        metadata["setup_state"] = "LOCATION_CHECK"
        if not location.get("valid"):
            reason = "chase" if location.get("chase_rejected") else "insufficient structure room"
            return self._hold(current_price, f"Adaptive candidate rejected by {reason} filter", metadata)

        bar_ts = int(c15[-1].timestamp)
        if self._last_entry_bar_ts == bar_ts:
            return self._hold(current_price, "Adaptive trigger bar already processed", metadata)

        # Anti-conflict: all simultaneous triggers collapse into this ONE signal.
        self._last_entry_bar_ts = bar_ts
        self._setup_state = "TRIGGERED"
        metadata.update({
            "setup_state": "TRIGGERED",
            "entry_trigger_owner": f"ADAPTIVE:{pattern.name}:{trigger}",
            "direction": direction,
            "freshness_bars": self.trigger_freshness_bars,
        })

        confidence = min(max(quality / 100.0, 0.0), 1.0)
        sig_type = SignalType.BUY if direction == "long" else SignalType.SELL
        return Signal(
            type=sig_type,
            symbol=self.symbol,
            price=float(current_price),
            amount=0.0,
            confidence=confidence,
            reason=(
                f"Adaptive {pattern.name} {direction.upper()} | "
                f"quality={quality:.1f}/100 | trigger={trigger} | "
                f"context={context.get('status')} | room={location.get('room_r')}R"
            ),
            metadata=metadata,
        )

    def _hold(self, current_price: float, reason: str, metadata: Optional[dict] = None) -> Signal:
        return Signal(
            type=SignalType.HOLD,
            symbol=self.symbol,
            price=float(current_price),
            amount=0.0,
            confidence=0.0,
            reason=reason,
            metadata=metadata or {
                "strategy": "ADAPTIVE_MULTI_TRIGGER_V1",
                "selected_strategy": "Adaptive Multi-Trigger V1",
                "version": self.VERSION,
                "entry_tf": self.ENTRY_TF,
                "setup_state": self._setup_state,
            },
        )
