"""
Layer 5: Strategy Engine.

Each strategy type gets its OWN concrete entry/SL/TP/invalidation logic
using its own named indicators — never a generic blended-indicator
score. Layer 4 has already picked exactly one of these to run; this
layer's job is purely "given that strategy, is there a valid setup
right now, and if so, where's the entry/stop/target."

  TrendContinuationStrategy  — EMA pullback + HMA slope + ADX + momentum + volume
  MeanReversionStrategy      — RSI extremes + VWAP distance + Bollinger + S/R
  BreakoutStrategy           — compression -> ATR/volume expansion + BOS + retest
  SwingReversalStrategy      — RSI divergence + CHOCH + liquidity sweep + engulfing
  MomentumExpansionStrategy  — ROC + ATR expansion + volume + EMA slope

Every strategy returns a StrategySignal with entry/stop/target/
invalidation/RR pre-computed — Layer 6 (Confidence) and Layer 7
(Expectancy) score the *setup*, they never invent price levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import indicators as ind


@dataclass
class StrategySignal:
    valid: bool
    direction: str = "long"          # "long" | "short"
    raw_score: float = 0.0           # 0-100, this strategy's own setup quality
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    invalidation: str = ""           # human-readable condition that kills the setup
    rr: float = 0.0
    reason: str = ""
    detail: dict = field(default_factory=dict)


def _rr(direction: str, entry: float, sl: float, tp: float) -> float:
    if direction == "long":
        risk, reward = entry - sl, tp - entry
    else:
        risk, reward = sl - entry, entry - tp
    return round(reward / risk, 2) if risk > 0 else 0.0


class TrendContinuationStrategy:
    """EMA pullback entries in the direction of an established trend."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 60:
            return StrategySignal(valid=False, reason="insufficient_candles")

        ema20 = ind.ema(closes, 20)
        ema50 = ind.ema(closes, 50)
        hma21 = ind.hma(closes, 21)
        adx_arr, pdi, mdi = ind.adx(closes, highs, lows, 14)
        rsi_arr = ind.rsi(closes, 14)
        atr_arr = ind.atr(closes, highs, lows, 14)
        vols = np.array([float(c.volume) for c in candles], dtype=float)

        price = closes[-1]
        atr_val = _safe(atr_arr[-1], price * 0.01)
        adx_val = _safe(adx_arr[-1], 15.0)
        hma_slope = hma21[-1] - hma21[-4] if len(hma21) > 4 and not np.isnan(hma21[-4]) else 0.0

        direction = "long" if direction_hint != "short_only" else "short"
        if direction_hint == "both":
            direction = "long" if price > ema50[-1] else "short"

        score = 0.0
        notes = []

        # Pullback to EMA20 (not too far from it — this is the "entry zone")
        dist_to_ema20 = abs(price - ema20[-1]) / price * 100
        pulled_back = dist_to_ema20 <= 0.6
        if pulled_back:
            score += 30; notes.append(f"pullback_to_ema20({dist_to_ema20:.2f}%)")

        # HMA slope confirms direction (faster/smoother trend confirmation than EMA)
        slope_ok = (hma_slope > 0) if direction == "long" else (hma_slope < 0)
        if slope_ok:
            score += 20; notes.append("hma_slope_confirms")

        # ADX + directional index confirm trend strength & direction
        di_ok = (pdi[-1] > mdi[-1]) if direction == "long" else (mdi[-1] > pdi[-1])
        if adx_val >= 20 and di_ok:
            score += 25; notes.append(f"adx={adx_val:.0f}_confirms")
        elif adx_val >= 15:
            score += 10

        # Momentum not overextended (avoid buying into an already-overbought pop)
        rsi_val = rsi_arr[-1]
        momentum_ok = (35 <= rsi_val <= 68) if direction == "long" else (32 <= rsi_val <= 65)
        if momentum_ok:
            score += 15; notes.append(f"rsi={rsi_val:.0f}_healthy")

        # Volume confirms continuation
        rel_vol = float(np.mean(vols[-3:])) / (float(np.mean(vols[-20:-3])) + 1e-9)
        if rel_vol >= 0.9:
            score += 10; notes.append(f"rel_vol={rel_vol:.2f}")

        valid = score >= 55 and pulled_back

        if not valid:
            return StrategySignal(valid=False, raw_score=round(score, 1),
                                  reason="No valid pullback setup: " + ", ".join(notes) if notes else "No setup")

        lookback = min(20, len(lows))
        if direction == "long":
            structure_sl = float(lows[-lookback:].min()) - 0.3 * atr_val
            sl = min(structure_sl, price - 1.5 * atr_val)
            sl = max(sl, price - 3.0 * atr_val)
            tp = price + 1.2 * (price - sl)
        else:
            structure_sl = float(highs[-lookback:].max()) + 0.3 * atr_val
            sl = max(structure_sl, price + 1.5 * atr_val)
            sl = min(sl, price + 3.0 * atr_val)
            tp = price - 1.2 * (sl - price)

        return StrategySignal(
            valid=True, direction=direction, raw_score=round(min(100.0, score), 1),
            entry=price, stop_loss=round(sl, 8), take_profit=round(tp, 8),
            invalidation=f"Close back beyond EMA50 ({ema50[-1]:.4f}) against {direction}",
            rr=_rr(direction, price, sl, tp),
            reason="Trend continuation: " + ", ".join(notes),
            detail={"adx": adx_val, "rsi": rsi_val, "hma_slope": hma_slope},
        )


class MeanReversionStrategy:
    """Fade extremes back toward VWAP/mid-Bollinger inside a range."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 40:
            return StrategySignal(valid=False, reason="insufficient_candles")

        vols = np.array([float(c.volume) for c in candles], dtype=float)
        rsi_arr = ind.rsi(closes, 14)
        mid, upper, lower, width = ind.bollinger(closes, 20, 2.0)
        atr_arr = ind.atr(closes, highs, lows, 14)
        vwap = _rolling_vwap(closes, vols, window=20)

        price = closes[-1]
        atr_val = _safe(atr_arr[-1], price * 0.01)
        rsi_val = rsi_arr[-1]

        score = 0.0
        notes = []
        direction: Optional[str] = None

        # RSI extreme picks the fade direction
        if rsi_val <= 32:
            direction = "long"; score += 30; notes.append(f"rsi_oversold={rsi_val:.0f}")
        elif rsi_val >= 68:
            direction = "short"; score += 30; notes.append(f"rsi_overbought={rsi_val:.0f}")
        else:
            return StrategySignal(valid=False, reason=f"RSI not at extreme ({rsi_val:.0f})")

        if direction_hint not in ("both", f"{direction}_only"):
            return StrategySignal(valid=False, reason=f"Direction {direction} blocked by macro gate")

        # Price at/beyond Bollinger band on the fade side
        band_val = float(lower[-1]) if direction == "long" else float(upper[-1])
        if not np.isnan(band_val):
            beyond_band = (price <= band_val) if direction == "long" else (price >= band_val)
            if beyond_band:
                score += 30; notes.append("beyond_bollinger_band")

        # Distance from VWAP (mean) — the further the better the reversion target
        vwap_dist_pct = abs(price - vwap) / price * 100 if vwap > 0 else 0
        if vwap_dist_pct >= 0.8:
            score += 25; notes.append(f"vwap_dist={vwap_dist_pct:.2f}%")
        elif vwap_dist_pct >= 0.4:
            score += 12

        # Support/resistance: recent swing extreme nearby confirms a fade zone
        lookback = min(25, len(lows))
        sr_level = float(lows[-lookback:].min()) if direction == "long" else float(highs[-lookback:].max())
        near_sr = abs(price - sr_level) / price * 100 <= 0.5
        if near_sr:
            score += 15; notes.append("near_support_resistance")

        valid = score >= 55

        if not valid:
            return StrategySignal(valid=False, raw_score=round(score, 1),
                                  reason="Mean reversion setup incomplete: " + ", ".join(notes))

        if direction == "long":
            sl = min(sr_level, price - 1.0 * atr_val) - 0.2 * atr_val
            tp = min(vwap, float(mid[-1]) if not np.isnan(mid[-1]) else vwap)
            tp = max(tp, price + 0.8 * atr_val)
        else:
            sl = max(sr_level, price + 1.0 * atr_val) + 0.2 * atr_val
            tp = max(vwap, float(mid[-1]) if not np.isnan(mid[-1]) else vwap)
            tp = min(tp, price - 0.8 * atr_val)

        return StrategySignal(
            valid=True, direction=direction, raw_score=round(min(100.0, score), 1),
            entry=price, stop_loss=round(sl, 8), take_profit=round(tp, 8),
            invalidation=f"RSI crosses back through 50 against {direction}, or SR level breaks",
            rr=_rr(direction, price, sl, tp),
            reason="Mean reversion: " + ", ".join(notes),
            detail={"rsi": rsi_val, "vwap": vwap, "vwap_dist_pct": vwap_dist_pct},
        )


class BreakoutStrategy:
    """Compression -> expansion breakout with volume confirmation + retest."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 50:
            return StrategySignal(valid=False, reason="insufficient_candles")

        vols = np.array([float(c.volume) for c in candles], dtype=float)
        atr_arr = ind.atr(closes, highs, lows, 14)
        _, _, _, bb_width = ind.bollinger(closes, 20, 2.0)

        price = closes[-1]
        atr_val = _safe(atr_arr[-1], price * 0.01)

        # Was the market compressed recently (low BB width) before this bar?
        prior_width = float(np.nanmean(bb_width[-10:-2])) if len(bb_width) >= 12 else np.nan
        now_width = float(bb_width[-1]) if not np.isnan(bb_width[-1]) else prior_width
        was_compressed = (not np.isnan(prior_width)) and prior_width < float(np.nanmedian(bb_width[~np.isnan(bb_width)]))
        expanding = (not np.isnan(now_width)) and (not np.isnan(prior_width)) and now_width > prior_width * 1.2

        lookback = min(20, len(highs) - 1)
        range_high = float(highs[-lookback - 1:-1].max())
        range_low = float(lows[-lookback - 1:-1].min())

        broke_up = price > range_high
        broke_down = price < range_low
        if not (broke_up or broke_down):
            return StrategySignal(valid=False, reason="No range breakout yet")

        direction = "long" if broke_up else "short"
        if direction_hint not in ("both", f"{direction}_only"):
            return StrategySignal(valid=False, reason=f"Direction {direction} blocked by macro gate")

        score = 0.0
        notes = ["bos"]
        score += 25

        if was_compressed:
            score += 20; notes.append("was_compressed")
        if expanding:
            score += 25; notes.append("volatility_expanding")

        rel_vol = float(np.mean(vols[-3:])) / (float(np.mean(vols[-20:-3])) + 1e-9)
        if rel_vol >= 1.5:
            score += 30; notes.append(f"volume_expansion={rel_vol:.2f}")
        elif rel_vol >= 1.15:
            score += 15

        valid = score >= 55

        if not valid:
            return StrategySignal(valid=False, raw_score=round(score, 1),
                                  reason="Breakout lacks confirmation: " + ", ".join(notes))

        breakout_level = range_high if direction == "long" else range_low
        if direction == "long":
            sl = min(breakout_level - 0.3 * atr_val, price - 1.2 * atr_val)
            tp = price + 2.0 * (price - sl)
        else:
            sl = max(breakout_level + 0.3 * atr_val, price + 1.2 * atr_val)
            tp = price - 2.0 * (sl - price)

        return StrategySignal(
            valid=True, direction=direction, raw_score=round(min(100.0, score), 1),
            entry=price, stop_loss=round(sl, 8), take_profit=round(tp, 8),
            invalidation=f"Close back inside the range (level {breakout_level:.4f})",
            rr=_rr(direction, price, sl, tp),
            reason="Breakout: " + ", ".join(notes),
            detail={"range_high": range_high, "range_low": range_low, "rel_vol": rel_vol},
        )


class SwingReversalStrategy:
    """RSI divergence + structure shift (CHOCH) + liquidity sweep + engulfing."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        opens = np.array([float(c.open) for c in candles], dtype=float)
        if len(closes) < 50:
            return StrategySignal(valid=False, reason="insufficient_candles")

        rsi_arr = ind.rsi(closes, 14)
        atr_arr = ind.atr(closes, highs, lows, 14)
        swing_highs, swing_lows = ind.swing_points(highs, lows, lookback=3)

        price = closes[-1]
        atr_val = _safe(atr_arr[-1], price * 0.01)

        bull_div = False
        bear_div = False
        if len(swing_lows) >= 2:
            (i1, l1), (i2, l2) = swing_lows[-2], swing_lows[-1]
            if l2 < l1 and rsi_arr[i2] > rsi_arr[i1]:
                bull_div = True
        if len(swing_highs) >= 2:
            (i1, h1), (i2, h2) = swing_highs[-2], swing_highs[-1]
            if h2 > h1 and rsi_arr[i2] < rsi_arr[i1]:
                bear_div = True

        # Liquidity sweep: wick beyond recent extreme, closes back inside
        lookback = 15
        sweep_up = sweep_down = False
        if len(highs) > lookback + 1:
            prior_high = float(highs[-lookback - 1:-1].max())
            prior_low = float(lows[-lookback - 1:-1].min())
            sweep_up = highs[-1] > prior_high and closes[-1] < prior_high
            sweep_down = lows[-1] < prior_low and closes[-1] > prior_low

        # Engulfing pattern (last 2 candles)
        bull_engulf = ind.bullish_engulfing(opens[-2], closes[-2], opens[-1], closes[-1])
        bear_engulf = ind.bearish_engulfing(opens[-2], closes[-2], opens[-1], closes[-1])

        # CHOCH proxy: break of the most recent minor swing in the opposite direction
        choch_up = bool(swing_highs) and price > swing_highs[-1][1] and bull_div
        choch_down = bool(swing_lows) and price < swing_lows[-1][1] and bear_div

        long_signals = [bull_div, sweep_down, bull_engulf, choch_up]
        short_signals = [bear_div, sweep_up, bear_engulf, choch_down]
        long_count = sum(long_signals)
        short_count = sum(short_signals)

        if long_count == 0 and short_count == 0:
            return StrategySignal(valid=False, reason="No reversal confluence present")

        direction = "long" if long_count >= short_count else "short"
        if direction_hint not in ("both", f"{direction}_only"):
            return StrategySignal(valid=False, reason=f"Direction {direction} blocked by macro gate")

        count = long_count if direction == "long" else short_count
        notes = []
        score = 0.0
        if (bull_div if direction == "long" else bear_div):
            score += 30; notes.append("divergence")
        if (sweep_down if direction == "long" else sweep_up):
            score += 25; notes.append("liquidity_sweep")
        if (bull_engulf if direction == "long" else bear_engulf):
            score += 20; notes.append("engulfing")
        if (choch_up if direction == "long" else choch_down):
            score += 25; notes.append("choch")

        valid = score >= 55 and count >= 2

        if not valid:
            return StrategySignal(valid=False, raw_score=round(score, 1),
                                  reason="Reversal confluence too weak: " + ", ".join(notes) if notes else "weak")

        lookback_sl = min(10, len(lows))
        if direction == "long":
            sl = float(lows[-lookback_sl:].min()) - 0.3 * atr_val
            tp = price + 1.5 * (price - sl)
        else:
            sl = float(highs[-lookback_sl:].max()) + 0.3 * atr_val
            tp = price - 1.5 * (sl - price)

        return StrategySignal(
            valid=True, direction=direction, raw_score=round(min(100.0, score), 1),
            entry=price, stop_loss=round(sl, 8), take_profit=round(tp, 8),
            invalidation="Price re-takes the swept liquidity level with conviction",
            rr=_rr(direction, price, sl, tp),
            reason="Swing reversal: " + ", ".join(notes),
            detail={"long_count": long_count, "short_count": short_count},
        )


class MomentumExpansionStrategy:
    """Chase active expansion — ROC + ATR expansion + volume + EMA slope."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 40:
            return StrategySignal(valid=False, reason="insufficient_candles")

        vols = np.array([float(c.volume) for c in candles], dtype=float)
        roc_arr = ind.roc(closes, 10)
        atr_arr = ind.atr(closes, highs, lows, 14)
        ema20 = ind.ema(closes, 20)

        price = closes[-1]
        atr_val = _safe(atr_arr[-1], price * 0.01)
        roc_val = roc_arr[-1]

        direction = "long" if roc_val >= 0 else "short"
        if direction_hint not in ("both", f"{direction}_only"):
            return StrategySignal(valid=False, reason=f"Direction {direction} blocked by macro gate")

        score = 0.0
        notes = []

        roc_abs = abs(roc_val)
        if roc_abs >= 3.0:
            score += 30; notes.append(f"roc={roc_val:.2f}%")
        elif roc_abs >= 1.5:
            score += 15

        atr_prev = float(np.nanmean(atr_arr[-15:-5])) if len(atr_arr) >= 15 else atr_val
        atr_expansion = atr_val / (atr_prev + 1e-9)
        if atr_expansion >= 1.3:
            score += 25; notes.append(f"atr_expansion={atr_expansion:.2f}x")

        rel_vol = float(np.mean(vols[-3:])) / (float(np.mean(vols[-20:-3])) + 1e-9)
        if rel_vol >= 1.3:
            score += 25; notes.append(f"rel_vol={rel_vol:.2f}")

        ema_slope = (ema20[-1] - ema20[-4]) / (ema20[-4] + 1e-9) if len(ema20) > 4 and not np.isnan(ema20[-4]) else 0.0
        slope_ok = (ema_slope > 0) if direction == "long" else (ema_slope < 0)
        if slope_ok:
            score += 20; notes.append("ema_slope_confirms")

        valid = score >= 55

        if not valid:
            return StrategySignal(valid=False, raw_score=round(score, 1),
                                  reason="Expansion momentum too weak: " + ", ".join(notes) if notes else "weak")

        if direction == "long":
            sl = price - 1.8 * atr_val
            tp = price + 1.3 * (price - sl)
        else:
            sl = price + 1.8 * atr_val
            tp = price - 1.3 * (sl - price)

        return StrategySignal(
            valid=True, direction=direction, raw_score=round(min(100.0, score), 1),
            entry=price, stop_loss=round(sl, 8), take_profit=round(tp, 8),
            invalidation="ROC flips sign or ATR contracts back below its prior average",
            rr=_rr(direction, price, sl, tp),
            reason="Momentum expansion: " + ", ".join(notes),
            detail={"roc": roc_val, "atr_expansion": atr_expansion},
        )


# ── Shared helpers ───────────────────────────────────────────────────────────

def _ohlc(candles: list):
    closes = np.array([float(c.close) for c in candles], dtype=float)
    highs = np.array([float(c.high) for c in candles], dtype=float)
    lows = np.array([float(c.low) for c in candles], dtype=float)
    return closes, highs, lows


def _safe(val: float, default: float) -> float:
    return float(val) if val is not None and not np.isnan(val) else default


def _rolling_vwap(closes: np.ndarray, vols: np.ndarray, window: int = 20) -> float:
    if len(closes) < window:
        window = len(closes)
    c = closes[-window:]
    v = vols[-window:]
    total_v = float(np.sum(v))
    if total_v <= 0:
        return float(np.mean(c))
    return float(np.sum(c * v) / total_v)
