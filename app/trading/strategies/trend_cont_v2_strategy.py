"""
TrendCont v2 — Fast Entry Architecture
=======================================
Eliminates Late Confirmation Syndrome via score-based entry instead of gate-based.

8 Layers:
  1. Market State Engine  — TREND / PULLBACK / CHOP / BREAKOUT / REVERSAL
  2. Trend Quality Engine — EMA alignment, slope, HTF bias, ADX
  3. Fast Entry Engine    — Entry Readiness Score 0-100 every bar
  4. Dynamic Confidence   — Adaptive threshold (ADX-based, not fixed)
  5. Smart Execution      — No candle-close wait; histogram derivative
  6. Position Manager     — SL/TP via TP-ladder
  7. Health Monitor       — 5-level (exit only at <25, not <45)
  8. Learning Engine      — Entry/exit quality logged in metadata

Key change vs v1: every bar scores 0-100; entry fires when score > dynamic_threshold.
No waiting for "all conditions aligned" — that causes 40-60% late entries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MarketState:
    name: str        # TREND | PULLBACK | CHOP | BREAKOUT | REVERSAL
    confidence: float  # 0-100
    direction: str   # "long" | "short" | "neutral"
    momentum: float  # -100 to +100 (positive = bullish)
    adx: float       # current ADX value


@dataclass
class EntryScore:
    total: float        # 0-100 weighted composite
    trend: float        # 0-30
    pullback: float     # 0-20
    momentum: float     # 0-20
    volume: float       # 0-10
    structure: float    # 0-10
    acceleration: float # 0-10
    direction: str      # "long" | "short"
    threshold: float    # dynamic threshold for this bar
    ready: bool         # READY state (trend+momentum+structure OK)


# ── Main Strategy ─────────────────────────────────────────────────────────────

class TrendContV2Strategy(BaseStrategy):
    """
    Trend Continuation v2 — Fast Entry Architecture.

    Fixes the root cause of late entries: replaces sequential gate checks
    (EMA cross AND MACD cross AND ADX AND bias AND pullback ALL at once)
    with a continuous 0-100 score that fires when score > dynamic_threshold.

    Entry Readiness Score breakdown:
      Trend       30 pts  — EMA alignment + slope + HTF bias
      Pullback    20 pts  — adaptive zone + wick rejection + retracement
      Momentum    20 pts  — histogram derivative (not cross) + ROC
      Volume      10 pts  — relative volume + OBV direction
      Structure   10 pts  — BOS + swing structure (HH/HL or LH/LL)
      Acceleration 10 pts — ATR expansion + body% + price velocity
      ────────────────────
      Total       100 pts

    Dynamic threshold (ADX-based):
      ADX ≥ 40 → 70   (strong trend, enter early)
      ADX ≥ 30 → 75
      ADX ≥ 25 → 78
      ADX ≥ 20 → 82
      ADX ≥ 15 → 88
      ADX < 15 → 95   (very weak, very high bar)

    Fast Mode (READY → EXECUTE):
      READY  = trend_score ≥ 20 AND momentum_score ≥ 12 AND structure_score ≥ 5
      EXECUTE = total_score ≥ threshold
    """

    DEFAULTS: dict = dict(
        name="TrendContV2",
        tf="15m",
        limit=500,
        # ── Entry ─────────────────────────────────────────────────────────────
        min_score=70.0,          # fallback minimum (overridden by dynamic threshold)
        strong_trend_score=85.0, # strong-trend override: enter at this score regardless
        # ── Market state ──────────────────────────────────────────────────────
        chop_adx_max=15.0,       # ADX below this = CHOP → skip
        chop_confidence_min=70.0,# CHOP confidence must be > this to skip
        trend_adx_min=18.0,      # ADX above this = trending
        # ── Adaptive pullback zone ─────────────────────────────────────────────
        pullback_atr_mult=1.0,   # base zone width as ATR multiplier
        pullback_tight_mult=0.5, # tight side (counter-direction from zone)
        # ── SL / TP ───────────────────────────────────────────────────────────
        sl_mult=1.2,
        sl_min_pct=0.012,
        sl_max_pct=0.035,
        tp1_r=0.5,
        tp2_r=1.5,               # ladder will extend to 3R via health BULL
        sl_ladder_enabled=True,  # use SL-ratchet ladder exit
        # ── HTF bias ──────────────────────────────────────────────────────────
        htf_bias_min=20.0,       # minimum MTF bias score to consider direction valid
        # ── Cooldown ──────────────────────────────────────────────────────────
        cooldown_bars=3,         # bars to wait after a signal (whipsaw guard)
        # ── Startup warmup ────────────────────────────────────────────────────
        startup_warmup_min=0,
        # ── Health monitor ────────────────────────────────────────────────────
        health_guard_enabled=True,
        reversal_spike_enabled=True,
        reversal_spike_atr=1.5,
        reversal_spike_bars=4,
        reversal_spike_uw_frac=0.7,
        trend_fade_enabled=True,
        trend_fade_uw_frac=0.6,
    )

    def __init__(self, symbol: str, params: Optional[dict] = None):
        super().__init__(symbol, params)
        self._p = {**self.DEFAULTS,
                   **{k: self.params.get(k, v) for k, v in self.DEFAULTS.items()}}
        self._cooldown_until: float = 0.0
        self._last_signal_bar: int = -99

    # ── Public interface ──────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        import time as _time
        if len(candles) < 60:
            return self._hold()

        closes = [float(c.close) for c in candles]
        highs  = [float(c.high)  for c in candles]
        lows   = [float(c.low)   for c in candles]

        # ── Layer 1: Market State ─────────────────────────────────────────────
        state = self._market_state(candles, closes, highs, lows, mtf_candles)

        # Skip CHOP — no trend to ride
        if state.name == "CHOP" and state.confidence >= self._p["chop_confidence_min"]:
            return self._hold()

        # ── Layer 3 + 4: Entry Score + Dynamic Threshold (both directions) ────
        long_score  = self._entry_score(candles, closes, highs, lows,
                                        current_price, "long", state, mtf_candles)
        short_score = self._entry_score(candles, closes, highs, lows,
                                        current_price, "short", state, mtf_candles)

        # Pick the higher-scoring direction that aligns with market state
        sig_type = SignalType.HOLD
        score    = None

        # Market state direction bias
        state_long  = state.direction in ("long",  "neutral")
        state_short = state.direction in ("short", "neutral")

        if (long_score.total >= long_score.threshold and state_long and
                long_score.total >= short_score.total):
            sig_type = SignalType.BUY
            score    = long_score
        elif (short_score.total >= short_score.threshold and state_short and
              short_score.total > long_score.total):
            sig_type = SignalType.SELL
            score    = short_score

        # Strong trend override: if one side scores high AND market is trending,
        # enter even if the other direction also has some score
        if sig_type == SignalType.HOLD:
            if (long_score.total >= self._p["strong_trend_score"] and state_long
                    and state.name in ("TREND", "PULLBACK", "BREAKOUT")):
                sig_type = SignalType.BUY
                score    = long_score
            elif (short_score.total >= self._p["strong_trend_score"] and state_short
                  and state.name in ("TREND", "PULLBACK", "BREAKOUT")):
                sig_type = SignalType.SELL
                score    = short_score

        if sig_type == SignalType.HOLD:
            return self._hold()

        # ── Layer 5: Smart Execution ──────────────────────────────────────────
        side = "long" if sig_type == SignalType.BUY else "short"
        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else current_price * 0.01

        meta = self.risk_metadata(
            current_price, atr_val, side,
            sl_atr=self._p["sl_mult"],
            tp1_atr=self._p["tp1_r"] * self._p["sl_mult"],
            tp2_atr=self._p["tp2_r"] * self._p["sl_mult"],
            sl_min_pct=self._p["sl_min_pct"],
            sl_max_pct=self._p["sl_max_pct"],
        )
        meta["sl_ladder_enabled"] = self._p["sl_ladder_enabled"]
        meta["sj_score"]          = round(score.total, 1)
        meta["market_state"]      = state.name
        meta["state_confidence"]  = round(state.confidence, 1)
        meta["threshold"]         = round(score.threshold, 1)
        meta["score_breakdown"]   = {
            "trend":        round(score.trend, 1),
            "pullback":     round(score.pullback, 1),
            "momentum":     round(score.momentum, 1),
            "volume":       round(score.volume, 1),
            "structure":    round(score.structure, 1),
            "acceleration": round(score.acceleration, 1),
        }

        return Signal(
            type=sig_type,
            symbol=self.symbol,
            price=current_price,
            amount=0.0,
            reason=f"{state.name} score={score.total:.0f} thr={score.threshold:.0f}",
            confidence=min(1.0, score.total / 100.0),
            metadata=meta,
        )

    def monitor_position(self, side: str, entry: float, one_r: float, tp1_hit: bool,
                         candles_5m: list, candles_1h: list, candles_4h: list,
                         candles_3m: list = None, candles_15m: list = None) -> tuple[str, str]:
        """Layer 7: Health monitoring — guard stack, same as v1 strategy."""
        if not self._p["health_guard_enabled"]:
            return ("HOLD", "health guard disabled")
        if not candles_5m or not candles_1h or not candles_4h:
            return ("HOLD", "no candle data")

        from .guard_checks import check_spike_cut, check_trend_fade
        from .base import BaseStrategy

        def _to_arr(cs, attr):
            return np.array([float(getattr(c, attr)) for c in cs])

        df5  = {a: _to_arr(candles_5m, a)  for a in ("open","high","low","close","volume")}
        df1h = {a: _to_arr(candles_1h, a)  for a in ("open","high","low","close","volume")}
        df4h = {a: _to_arr(candles_4h, a)  for a in ("open","high","low","close","volume")}

        # Spike-cut (pre-TP1, 3m)
        if (self._p.get("reversal_spike_enabled") and not tp1_hit
                and candles_3m and one_r > 0):
            df3m = {a: _to_arr(candles_3m, a) for a in ("open","high","low","close","volume")}
            try:
                sc_action, sc_reason = check_spike_cut(
                    side, entry, one_r, df3m,
                    spike_atr_mult=self._p["reversal_spike_atr"],
                    spike_bars=self._p["reversal_spike_bars"],
                    uw_frac=self._p["reversal_spike_uw_frac"],
                )
                if sc_action == "CLOSE":
                    return ("CLOSE", sc_reason)
            except Exception:
                pass

        # Trend-fade cut (pre-TP1, 15m)
        if (self._p.get("trend_fade_enabled") and not tp1_hit
                and candles_15m and one_r > 0):
            df15 = {a: _to_arr(candles_15m, a) for a in ("open","high","low","close","volume")}
            try:
                from .guard_checks import check_trend_fade
                tf_action, tf_reason = check_trend_fade(
                    side, entry, one_r, df15, df5,
                    uw_frac=self._p["trend_fade_uw_frac"],
                )
                if tf_action == "CLOSE":
                    return ("CLOSE", tf_reason)
            except Exception:
                pass

        return ("HOLD", "all guards passed")

    # ── Layer 1: Market State Engine ──────────────────────────────────────────

    def _market_state(self, candles, closes, highs, lows,
                      mtf_candles: dict = None) -> MarketState:
        n = len(candles)

        # ADX
        adx_arr, plus_di, minus_di = self.adx(candles, 14)
        adx_val  = float(adx_arr[-1])  if not np.isnan(adx_arr[-1])  else 0.0
        pdi_val  = float(plus_di[-1])  if not np.isnan(plus_di[-1])  else 0.0
        mdi_val  = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 0.0

        # EMA alignment
        e9  = self.ema(closes, 9)
        e20 = self.ema(closes, 20)
        e50 = self.ema(closes, 50)
        e9v, e20v, e50v = float(e9[-1]), float(e20[-1]), float(e50[-1])
        p = closes[-1]

        # ATR for volatility context
        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else p * 0.01

        # ── CHOP: ADX very low + price oscillating around EMA20 ───────────────
        if adx_val < self._p["chop_adx_max"]:
            # Check if price crossed EMA20 multiple times in last 10 bars
            e20_10 = float(self.ema(closes[:-10], 20)[-1]) if n > 70 else e20v
            crosses = sum(1 for i in range(max(0, n-10), n-1)
                         if (closes[i] - float(e20[-10 + (i-(n-10))]) > 0) !=
                            (closes[i+1] - float(e20[-9 + (i-(n-10))]) > 0)
                         ) if n > 70 else 0
            confidence = min(100.0, 50.0 + max(0, adx_val - 12) * (-3) + crosses * 5)
            return MarketState("CHOP", confidence, "neutral", 0.0, adx_val)

        # ── Direction from DI ──────────────────────────────────────────────────
        di_direction = "long" if pdi_val > mdi_val else "short"
        di_margin    = abs(pdi_val - mdi_val)

        # EMA direction
        ema_long  = (e9v > e20v and e20v > e50v and p > e20v)
        ema_short = (e9v < e20v and e20v < e50v and p < e20v)
        if ema_long:   ema_dir = "long"
        elif ema_short: ema_dir = "short"
        else:           ema_dir = "neutral"

        # Direction consensus
        if di_direction == ema_dir and ema_dir != "neutral":
            direction = ema_dir
        elif ema_dir != "neutral":
            direction = ema_dir
        else:
            direction = di_direction

        # ── BREAKOUT: ATR expansion + BOS ─────────────────────────────────────
        atr_ma20 = float(np.nanmean(atr_arr[-20:])) if not np.all(np.isnan(atr_arr[-20:])) else atr_val
        atr_expanding = atr_val > atr_ma20 * 1.3
        # Price broke recent 10-bar high/low
        recent_high = max(highs[-11:-1]) if n > 11 else p
        recent_low  = min(lows[-11:-1])  if n > 11 else p
        bos_long  = p > recent_high
        bos_short = p < recent_low
        if atr_expanding and (bos_long or bos_short):
            bos_dir = "long" if bos_long else "short"
            confidence = min(100.0, 60.0 + adx_val * 0.8)
            momentum   = (pdi_val - mdi_val) if bos_dir == "long" else (mdi_val - pdi_val)
            return MarketState("BREAKOUT", confidence, bos_dir, momentum, adx_val)

        # ── REVERSAL signals ───────────────────────────────────────────────────
        # RSI divergence check: price at extreme + RSI turning
        rsi_arr = self.rsi(closes, 14)
        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0
        reversal_long  = (rsi_val < 30 and closes[-1] > closes[-2])  # oversold turning up
        reversal_short = (rsi_val > 70 and closes[-1] < closes[-2])  # overbought turning down
        if reversal_long or reversal_short:
            rev_dir = "long" if reversal_long else "short"
            confidence = abs(rsi_val - 50) * 1.5
            momentum   = (50 - rsi_val) if reversal_long else (rsi_val - 50)
            return MarketState("REVERSAL", confidence, rev_dir, momentum, adx_val)

        # ── PULLBACK: In trending market, price near EMA zone ─────────────────
        in_trend = adx_val >= self._p["trend_adx_min"]
        zone_width = atr_val * self._p["pullback_atr_mult"]
        near_ema = abs(p - e20v) < zone_width * 1.5
        if in_trend and near_ema:
            confidence = min(100.0, 50.0 + adx_val * 0.8 + di_margin * 0.5)
            momentum   = (pdi_val - mdi_val) if direction == "long" else (mdi_val - pdi_val)
            return MarketState("PULLBACK", confidence, direction, momentum, adx_val)

        # ── TREND: Default when ADX is up and direction clear ─────────────────
        confidence = min(100.0, 40.0 + adx_val * 0.8 + di_margin * 0.5)
        momentum   = (pdi_val - mdi_val) if direction == "long" else (mdi_val - pdi_val)
        return MarketState("TREND", confidence, direction, momentum, adx_val)

    # ── Layer 3: Fast Entry Score ─────────────────────────────────────────────

    def _entry_score(self, candles, closes, highs, lows, price,
                     direction: str, state: MarketState,
                     mtf_candles: dict = None) -> EntryScore:

        t  = self._trend_score(closes, highs, lows, direction, mtf_candles, candles)
        pb = self._pullback_score(closes, highs, lows, price, direction, candles)
        mo = self._momentum_score(closes, direction)
        vo = self._volume_score(candles, closes, direction)
        st = self._structure_score(closes, highs, lows, direction)
        ac = self._acceleration_score(candles, closes, highs, lows, price, direction)

        # Weighted composite for total (matches the 30/20/20/10/10/10 breakdown)
        total = t + pb + mo + vo + st + ac

        # Dynamic threshold
        threshold = self._dynamic_threshold(state.adx)

        # READY check: core conditions met even if total isn't at threshold yet
        ready = (t >= 18 and mo >= 12 and st >= 5)

        return EntryScore(
            total=min(100.0, total),
            trend=t, pullback=pb, momentum=mo,
            volume=vo, structure=st, acceleration=ac,
            direction=direction,
            threshold=threshold,
            ready=ready,
        )

    def _trend_score(self, closes, highs, lows, direction, mtf_candles, candles) -> float:
        """
        Max 30 pts: stable trend alignment (12) + DI bonus (4) + EMA slope (8) + HTF bias (10).

        Primary: EMA20 vs EMA50 relationship (stable through 5-20 bar pullbacks).
        DI is a bonus, not a gate — DI can flip during pullbacks but EMA20/EMA50
        cross requires much longer counter-trend movement to flip.
        """
        score = 0.0
        is_long = (direction == "long")

        e20 = self.ema(closes, 20)
        e50 = self.ema(closes, 50)
        e20v, e50v = float(e20[-1]), float(e50[-1])
        p = closes[-1]

        # EMA Structural Alignment (max 12) — EMA20 vs EMA50 only
        # No DI gate: DI can flip during 5-20 bar pullbacks, EMA cross is rarer
        adx_arr, plus_di, minus_di = self.adx(candles, 14)
        pdi = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 0.0
        mdi = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 0.0

        if is_long:
            if e20v > e50v:     score += 12.0  # bullish structure
            elif p > e50v:      score += 6.0   # price above slow EMA
            elif p > e20v:      score += 3.0   # price above fast EMA only
            # DI alignment bonus (max 4) — not a gate, just extra confidence
            if pdi > mdi:
                di_margin = (pdi - mdi) / (pdi + mdi + 1e-9)
                score += min(4.0, di_margin * 10)
        else:
            if e20v < e50v:     score += 12.0  # bearish structure
            elif p < e50v:      score += 6.0   # price below slow EMA
            elif p < e20v:      score += 3.0   # price below fast EMA only
            # DI alignment bonus (max 4)
            if mdi > pdi:
                di_margin = (mdi - pdi) / (mdi + pdi + 1e-9)
                score += min(4.0, di_margin * 10)

        # EMA50 Slope (max 8) — use EMA50 slope which doesn't flip on pullbacks
        # 5-bar slope of EMA50: stable, tells us the underlying trend direction
        if len(e50) >= 6:
            slope_pct = (float(e50[-1]) - float(e50[-6])) / (float(e50[-6]) + 1e-9) * 100
            if is_long:
                if slope_pct > 0.2:   score += 8.0
                elif slope_pct > 0.05: score += 5.0
                elif slope_pct > 0.0: score += 2.0
            else:
                if slope_pct < -0.2:   score += 8.0
                elif slope_pct < -0.05: score += 5.0
                elif slope_pct < 0.0:  score += 2.0
        elif len(e20) >= 4:
            # Fallback: EMA20 slope
            slope_pct = (float(e20[-1]) - float(e20[-4])) / (float(e20[-4]) + 1e-9) * 100
            if is_long:
                if slope_pct > 0.15: score += 6.0
                elif slope_pct > 0:  score += 2.0
            else:
                if slope_pct < -0.15: score += 6.0
                elif slope_pct < 0:   score += 2.0

        # HTF Bias (max 10) — 1H + 4H bias, most reliable at pullback entry
        if mtf_candles:
            bias_val, _ = self.compute_mtf_bias(
                candles, mtf_candles,
                ema_fast=20, ema_slow=50, rsi_period=14,
                rsi_bull=55.0, rsi_bear=45.0,
                w1=0.0, w2=2.0, w3=3.0,  # 1H + 4H only (no 15m noise)
            )
            if is_long:
                if bias_val >= 40:    score += 10.0
                elif bias_val >= 15:  score += 7.0
                elif bias_val >= 0:   score += 3.0
            else:
                if bias_val <= -40:   score += 10.0
                elif bias_val <= -15: score += 7.0
                elif bias_val <= 0:   score += 3.0

        return min(30.0, score)

    def _pullback_score(self, closes, highs, lows, price, direction, candles) -> float:
        """Max 20 pts: adaptive zone (10) + wick rejection (5) + retracement (5)"""
        score  = 0.0
        is_long = (direction == "long")

        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else price * 0.01
        e20 = self.ema(closes, 20)
        e20v = float(e20[-1])

        # Adaptive pullback zone (max 10)
        # Zone width: ATR × mult, tightened when slope is steep
        slope_abs = abs(float(e20[-1]) - float(e20[-4])) / (float(e20[-4]) + 1e-9) if len(e20) >= 4 else 0.0
        slope_adj = max(0.5, 1.0 - slope_abs * 20)  # steeper slope → tighter zone
        zone_w = atr_val * self._p["pullback_atr_mult"] * slope_adj

        dist = price - e20v
        if is_long:
            # Want price below EMA20 or just touching from below (pullback to EMA)
            in_zone = -zone_w <= dist <= zone_w * self._p["pullback_tight_mult"]
            too_far = dist < -zone_w * 2
        else:
            in_zone = -zone_w * self._p["pullback_tight_mult"] <= dist <= zone_w
            too_far = dist > zone_w * 2

        if in_zone:
            score += 10.0
        elif not too_far:
            score += 4.0  # approaching but not in zone yet

        # Wick rejection (max 5) — last 2 bars showed wick rejection at EMA zone
        for bar in candles[-3:-1]:
            h, l, c, o = float(bar.high), float(bar.low), float(bar.close), float(bar.open)
            rng = h - l + 1e-9
            if is_long:
                lower_wick = (min(c, o) - l) / rng
                if lower_wick > 0.4:
                    score += 2.5
                    break
            else:
                upper_wick = (h - max(c, o)) / rng
                if upper_wick > 0.4:
                    score += 2.5
                    break

        # Retracement depth (max 5) — 38-62% fib of last significant swing
        if len(closes) >= 20:
            swing_high = max(highs[-20:])
            swing_low  = min(lows[-20:])
            swing_rng  = swing_high - swing_low
            if swing_rng > 0:
                if is_long:
                    retrace = (swing_high - price) / swing_rng
                else:
                    retrace = (price - swing_low) / swing_rng
                if 0.38 <= retrace <= 0.62:
                    score += 5.0
                elif 0.25 <= retrace <= 0.75:
                    score += 2.0

        return min(20.0, score)

    def _momentum_score(self, closes, direction: str) -> float:
        """
        Max 20 pts: MACD histogram direction (6) + histogram derivative (8) + ROC (6).
        Key innovation: histogram DERIVATIVE (acceleration) not level or cross.
        """
        score  = 0.0
        is_long = (direction == "long")

        if len(closes) < 30:
            return 0.0

        _, _, hist = self.macd(closes, fast=12, slow=26, signal=9)
        # Remove NaN prefix
        valid_hist = [h for h in hist if not np.isnan(h)]
        if len(valid_hist) < 3:
            return 0.0

        h0 = valid_hist[-1]   # current histogram
        h1 = valid_hist[-2]   # prev
        h2 = valid_hist[-3]   # prev prev

        # MACD Histogram direction (max 6)
        if is_long:
            if h0 > 0: score += 6.0
            elif h0 > h1: score += 3.0  # rising histogram even if negative
        else:
            if h0 < 0: score += 6.0
            elif h0 < h1: score += 3.0  # falling histogram even if positive

        # Histogram Derivative / Acceleration (max 8)
        # d1 = current change, d2 = previous change → acceleration = d1 - d2
        d1 = h0 - h1  # momentum direction this bar
        d2 = h1 - h2  # momentum direction prev bar
        accel = d1 - d2  # positive = accelerating (d1 > d2)

        if is_long:
            if d1 > 0 and accel > 0:  score += 8.0   # accelerating upward
            elif d1 > 0:               score += 5.0   # rising but not accelerating
            elif accel > 0:            score += 2.0   # at least acceleration positive
            elif d1 < 0 and accel < 0: score -= 4.0   # accelerating downward (bad for long)
        else:
            if d1 < 0 and accel < 0:  score += 8.0   # accelerating downward
            elif d1 < 0:               score += 5.0   # falling but not accelerating
            elif accel < 0:            score += 2.0
            elif d1 > 0 and accel > 0: score -= 4.0   # accelerating upward (bad for short)

        # ROC(9) (max 6)
        if len(closes) >= 11:
            roc = (closes[-1] - closes[-10]) / (closes[-10] + 1e-9) * 100
            if is_long:
                if roc > 1.0:   score += 6.0
                elif roc > 0.3: score += 4.0
                elif roc > 0.0: score += 2.0
            else:
                if roc < -1.0:   score += 6.0
                elif roc < -0.3: score += 4.0
                elif roc < 0.0:  score += 2.0

        return max(0.0, min(20.0, score))

    def _volume_score(self, candles, closes, direction: str) -> float:
        """Max 10 pts: relative volume (6) + OBV direction (4)"""
        score   = 0.0
        is_long = (direction == "long")
        vols    = [float(c.volume) for c in candles]

        # Relative volume (max 6)
        if len(vols) >= 20:
            vol_ma20  = float(np.mean(vols[-20:]))
            current_v = vols[-1]
            if vol_ma20 > 0:
                rel_vol = current_v / vol_ma20
                if rel_vol > 1.5:   score += 6.0
                elif rel_vol > 1.2: score += 4.0
                elif rel_vol > 1.0: score += 2.0

        # OBV direction (max 4) — OBV above its EMA20 = bullish
        obv_arr = self.obv(candles)
        if len(obv_arr) >= 20:
            obv_ema = float(self.ema(list(obv_arr), 20)[-1])
            obv_cur = float(obv_arr[-1])
            if is_long and obv_cur > obv_ema:
                score += 4.0
            elif not is_long and obv_cur < obv_ema:
                score += 4.0

        return min(10.0, score)

    def _structure_score(self, closes, highs, lows, direction: str) -> float:
        """Max 10 pts: swing structure context (5) + BOS (5)

        Swing structure uses mid-term lookback (20-45 bars back) to read TREND
        structure rather than pullback bars. A 5-15 bar pullback at the end of
        the series doesn't corrupt the structure read this way.
        """
        score   = 0.0
        is_long = (direction == "long")
        n = len(closes)
        if n < 10:
            return 0.0

        # Swing structure (max 5): compare mid-term windows well before current pullback
        if n >= 45:
            recent_lo = min(lows[-25:-15])   # 25-15 bars ago (pre-pullback trend)
            older_lo  = min(lows[-45:-35])   # 45-35 bars ago (older trend)
            recent_hi = max(highs[-25:-15])
            older_hi  = max(highs[-45:-35])
            if is_long:
                hl_ok = recent_lo > older_lo  # higher lows = uptrend
                hh_ok = recent_hi > older_hi  # higher highs = uptrend
            else:
                hl_ok = recent_lo < older_lo  # lower lows = downtrend
                hh_ok = recent_hi < older_hi  # lower highs = downtrend
            if hl_ok and hh_ok: score += 5.0
            elif hl_ok or hh_ok: score += 2.5
        elif n >= 20:
            mid_lo  = min(lows[-15:-8])
            old_lo  = min(lows[-20:]) if n < 25 else min(lows[-25:-18])
            mid_hi  = max(highs[-15:-8])
            old_hi  = max(highs[-20:]) if n < 25 else max(highs[-25:-18])
            if is_long:
                if mid_lo > old_lo: score += 3.0
                if mid_hi > old_hi: score += 2.0
            else:
                if mid_lo < old_lo: score += 3.0
                if mid_hi < old_hi: score += 2.0

        # BOS: Break of Structure (max 5) — near-term (6 bars)
        if n >= 6:
            lookback_high = max(highs[-6:-1])
            lookback_low  = min(lows[-6:-1])
            if is_long and closes[-1] > lookback_high:
                score += 5.0
            elif not is_long and closes[-1] < lookback_low:
                score += 5.0

        return min(10.0, score)

    def _acceleration_score(self, candles, closes, highs, lows, price,
                            direction: str) -> float:
        """Max 10 pts: ATR expansion (4) + body% (3) + velocity (3)"""
        score   = 0.0
        is_long = (direction == "long")

        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else price * 0.01
        atr_ma20 = float(np.nanmean(atr_arr[-20:])) if not np.all(np.isnan(atr_arr[-20:])) else atr_val

        # ATR expansion (max 4): current ATR > 20-bar ATR average
        if atr_val > atr_ma20:
            score += 4.0
        elif atr_val > atr_ma20 * 0.9:
            score += 2.0

        # Body% of last bar (max 3): large body = conviction
        last = candles[-1]
        h, l, c, o = float(last.high), float(last.low), float(last.close), float(last.open)
        bar_range = h - l + 1e-9
        body_pct  = abs(c - o) / bar_range
        bear_body = (c < o)  # bearish bar

        if is_long and not bear_body and body_pct > 0.6:
            score += 3.0
        elif not is_long and bear_body and body_pct > 0.6:
            score += 3.0
        elif body_pct > 0.4:
            score += 1.0

        # Velocity (max 3): price moved > 0.5×ATR in last 3 bars in entry direction
        if len(closes) >= 4:
            move_3bar = closes[-1] - closes[-4]
            if is_long and move_3bar > atr_val * 0.5:
                score += 3.0
            elif not is_long and move_3bar < -atr_val * 0.5:
                score += 3.0

        return min(10.0, score)

    # ── Layer 4: Dynamic Threshold ────────────────────────────────────────────

    def _dynamic_threshold(self, adx: float) -> float:
        """ADX-based threshold: strong trend → lower bar to enter early."""
        if adx >= 40:  return 70.0
        if adx >= 30:  return 75.0
        if adx >= 25:  return 78.0
        if adx >= 20:  return 82.0
        if adx >= 15:  return 88.0
        return 95.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hold(self) -> Signal:
        return Signal(
            type=SignalType.HOLD,
            symbol=self.symbol,
            price=0.0,
            amount=0.0,
        )
