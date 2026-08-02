"""Layer 5: strategy-specific entry engines (V2).

Each market regime is routed to one purpose-built strategy.  The engines keep
one stable interface (StrategySignal) so the AI Expert pipeline, confidence,
expectancy, risk, exits and dual runner remain unchanged.

V2 priorities:
- fresh closed-bar triggers instead of stale indicator state;
- ATR-normalized location and anti-chase checks;
- price-action structure confirmation;
- strategy-specific risk/reward and invalidation;
- no counter-trend mean reversion without directional permission;
- no swing reversal without both liquidity sweep and CHOCH.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import indicators as ind


@dataclass
class StrategySignal:
    valid: bool
    direction: str = "long"
    raw_score: float = 0.0
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    invalidation: str = ""
    rr: float = 0.0
    reason: str = ""
    detail: dict = field(default_factory=dict)


def _rr(direction: str, entry: float, sl: float, tp: float) -> float:
    if direction == "long":
        risk, reward = entry - sl, tp - entry
    else:
        risk, reward = sl - entry, entry - tp
    return round(reward / risk, 2) if risk > 0 else 0.0


def _blocked(direction: str, direction_hint: str) -> bool:
    return direction_hint not in ("both", f"{direction}_only")


def _ema_cross(fast: np.ndarray, slow: np.ndarray, direction: str) -> bool:
    if len(fast) < 2 or np.isnan(fast[-2]) or np.isnan(slow[-2]):
        return False
    if direction == "long":
        return bool(fast[-2] <= slow[-2] and fast[-1] > slow[-1])
    return bool(fast[-2] >= slow[-2] and fast[-1] < slow[-1])


def _body_ratio(candle) -> float:
    full = max(float(candle.high) - float(candle.low), 1e-12)
    return abs(float(candle.close) - float(candle.open)) / full


class TrendContinuationStrategy:
    """V2: fresh EMA8/13 reclaim after an EMA20 pullback in a mature trend."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 65:
            return StrategySignal(False, reason="insufficient_candles")

        ema8, ema13 = ind.ema(closes, 8), ind.ema(closes, 13)
        ema20, ema50 = ind.ema(closes, 20), ind.ema(closes, 50)
        adx_arr, pdi, mdi = ind.adx(closes, highs, lows, 14)
        rsi_arr = ind.rsi(closes, 14)
        atr_arr = ind.atr(closes, highs, lows, 14)
        vols = np.array([float(c.volume) for c in candles], dtype=float)

        price = float(closes[-1])
        atr_val = _safe(atr_arr[-1], price * 0.01)
        adx_val = _safe(adx_arr[-1], 15.0)
        rsi_val = _safe(rsi_arr[-1], 50.0)

        if direction_hint == "long_only":
            direction = "long"
        elif direction_hint == "short_only":
            direction = "short"
        else:
            direction = "long" if ema20[-1] > ema50[-1] else "short"

        trend_ok = (
            ema20[-1] > ema50[-1] and ema20[-1] > ema20[-4]
            if direction == "long"
            else ema20[-1] < ema50[-1] and ema20[-1] < ema20[-4]
        )
        di_ok = pdi[-1] > mdi[-1] if direction == "long" else mdi[-1] > pdi[-1]
        fresh_cross = _ema_cross(ema8, ema13, direction)

        # Pullback must have touched the EMA20 zone during the last four closed
        # candles and the latest candle must reclaim it in the trend direction.
        recent_touch = (
            float(np.min(lows[-4:])) <= ema20[-1] + 0.20 * atr_val
            if direction == "long"
            else float(np.max(highs[-4:])) >= ema20[-1] - 0.20 * atr_val
        )
        reclaim = price > ema20[-1] if direction == "long" else price < ema20[-1]
        distance_atr = abs(price - ema20[-1]) / max(atr_val, 1e-12)
        location_ok = distance_atr <= 1.0
        momentum_ok = 45 <= rsi_val <= 68 if direction == "long" else 32 <= rsi_val <= 55
        rel_vol = float(np.mean(vols[-3:])) / (float(np.mean(vols[-20:-3])) + 1e-9)

        score = 0.0
        notes = []
        for condition, points, label in (
            (trend_ok, 20, "ema20_50_trend"),
            (recent_touch and reclaim, 25, "ema20_pullback_reclaim"),
            (fresh_cross, 25, "fresh_ema8_13_cross"),
            (adx_val >= 17 and di_ok, 15, f"adx_di={adx_val:.0f}"),
            (momentum_ok, 10, f"rsi={rsi_val:.0f}"),
            (rel_vol >= 0.85, 5, f"rel_vol={rel_vol:.2f}"),
        ):
            if condition:
                score += points
                notes.append(label)

        mandatory = trend_ok and recent_touch and reclaim and fresh_cross and location_ok
        if not mandatory or score < 65:
            return StrategySignal(
                False, raw_score=round(score, 1),
                reason=("Trend continuation V2 waiting: " + ", ".join(notes))
                if notes else "Trend continuation V2: no fresh pullback trigger",
                detail={"distance_ema20_atr": round(distance_atr, 3)},
            )

        lookback = min(12, len(lows))
        if direction == "long":
            swing = float(np.min(lows[-lookback:]))
            sl = max(swing - 0.20 * atr_val, price - 2.2 * atr_val)
            sl = min(sl, price - 0.8 * atr_val)
            tp = price + 1.6 * (price - sl)
        else:
            swing = float(np.max(highs[-lookback:]))
            sl = min(swing + 0.20 * atr_val, price + 2.2 * atr_val)
            sl = max(sl, price + 0.8 * atr_val)
            tp = price - 1.6 * (sl - price)

        return StrategySignal(
            True, direction, round(min(score, 100.0), 1), price,
            round(sl, 8), round(tp, 8),
            f"15M close crosses back through EMA20/EMA13 against {direction}",
            _rr(direction, price, sl, tp),
            "Trend continuation V2: " + ", ".join(notes),
            {"adx": adx_val, "rsi": rsi_val, "distance_ema20_atr": distance_atr},
        )


class MeanReversionStrategy:
    """V2: range-edge fade only after sweep and EMA8/13 reversal trigger."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 55:
            return StrategySignal(False, reason="insufficient_candles")

        vols = np.array([float(c.volume) for c in candles], dtype=float)
        rsi_arr = ind.rsi(closes, 14)
        mid, upper, lower, _width = ind.bollinger(closes, 20, 2.0)
        atr_arr = ind.atr(closes, highs, lows, 14)
        adx_arr, _pdi, _mdi = ind.adx(closes, highs, lows, 14)
        ema8, ema13 = ind.ema(closes, 8), ind.ema(closes, 13)
        vwap = _rolling_vwap(closes, vols, 20)

        price = float(closes[-1])
        atr_val = _safe(atr_arr[-1], price * 0.01)
        rsi_val = _safe(rsi_arr[-1], 50.0)
        adx_val = _safe(adx_arr[-1], 20.0)

        prior_high = float(np.max(highs[-21:-1]))
        prior_low = float(np.min(lows[-21:-1]))
        sweep_down = lows[-1] < prior_low and closes[-1] > prior_low
        sweep_up = highs[-1] > prior_high and closes[-1] < prior_high

        long_candidate = rsi_val <= 38 and price <= lower[-1] + 0.15 * atr_val and sweep_down
        short_candidate = rsi_val >= 62 and price >= upper[-1] - 0.15 * atr_val and sweep_up
        if long_candidate:
            direction = "long"
        elif short_candidate:
            direction = "short"
        else:
            return StrategySignal(False, reason="Mean reversion V2: no swept range extreme")

        if _blocked(direction, direction_hint):
            return StrategySignal(False, reason=f"Direction {direction} blocked by macro gate")

        cross_back = _ema_cross(ema8, ema13, direction)
        range_ok = adx_val <= 22
        target = min(vwap, float(mid[-1])) if direction == "long" else max(vwap, float(mid[-1]))
        reward_room = (target - price) / atr_val if direction == "long" else (price - target) / atr_val

        score = 0.0
        notes = []
        for condition, points, label in (
            (range_ok, 20, f"adx_range={adx_val:.0f}"),
            (True, 25, "liquidity_sweep"),
            (cross_back, 25, "fresh_ema8_13_reversal"),
            (abs(price - vwap) / atr_val >= 0.8, 15, "vwap_extension"),
            (reward_room >= 0.9, 15, f"room={reward_room:.2f}ATR"),
        ):
            if condition:
                score += points
                notes.append(label)

        if not (range_ok and cross_back and reward_room >= 0.8) or score < 70:
            return StrategySignal(False, raw_score=round(score, 1),
                                  reason="Mean reversion V2 waiting: " + ", ".join(notes))

        if direction == "long":
            sl = min(float(lows[-1]), prior_low) - 0.25 * atr_val
            tp = max(price + 1.0 * atr_val, target)
        else:
            sl = max(float(highs[-1]), prior_high) + 0.25 * atr_val
            tp = min(price - 1.0 * atr_val, target)

        if _rr(direction, price, sl, tp) < 1.0:
            return StrategySignal(False, raw_score=round(score, 1),
                                  reason="Mean reversion V2 rejected: insufficient reward to mean")

        return StrategySignal(
            True, direction, round(min(score, 100.0), 1), price,
            round(sl, 8), round(tp, 8),
            "Range extreme breaks and closes outside after the sweep",
            _rr(direction, price, sl, tp),
            "Mean reversion V2: " + ", ".join(notes),
            {"rsi": rsi_val, "adx": adx_val, "vwap": vwap, "reward_room_atr": reward_room},
        )


class BreakoutStrategy:
    """V2: closed BOS with body/volume confirmation or a fresh retest hold."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 60:
            return StrategySignal(False, reason="insufficient_candles")

        vols = np.array([float(c.volume) for c in candles], dtype=float)
        atr_arr = ind.atr(closes, highs, lows, 14)
        _mid, _upper, _lower, width = ind.bollinger(closes, 20, 2.0)
        price = float(closes[-1])
        atr_val = _safe(atr_arr[-1], price * 0.01)

        range_high = float(np.max(highs[-22:-2]))
        range_low = float(np.min(lows[-22:-2]))
        prev_close = float(closes[-2])
        current = candles[-1]

        close_break_up = prev_close <= range_high and price > range_high
        close_break_down = prev_close >= range_low and price < range_low

        # Retest path: prior candle already broke; latest candle touches the
        # level and closes back on the breakout side.
        prior_broke_up = closes[-2] > range_high and closes[-3] <= range_high
        prior_broke_down = closes[-2] < range_low and closes[-3] >= range_low
        retest_up = prior_broke_up and lows[-1] <= range_high + 0.20 * atr_val and price > range_high
        retest_down = prior_broke_down and highs[-1] >= range_low - 0.20 * atr_val and price < range_low

        if close_break_up or retest_up:
            direction, level = "long", range_high
        elif close_break_down or retest_down:
            direction, level = "short", range_low
        else:
            return StrategySignal(False, reason="Breakout V2: waiting for fresh BOS or retest")

        if _blocked(direction, direction_hint):
            return StrategySignal(False, reason=f"Direction {direction} blocked by macro gate")

        rel_vol = float(np.mean(vols[-2:])) / (float(np.mean(vols[-22:-2])) + 1e-9)
        body_ok = _body_ratio(current) >= 0.55
        breakout_distance = abs(price - level) / max(atr_val, 1e-12)
        not_extended = breakout_distance <= 1.25
        prior_width = float(np.nanmean(width[-12:-3]))
        expanding = not np.isnan(width[-1]) and width[-1] > prior_width * 1.10
        true_retest = retest_up or retest_down

        score = 25.0
        notes = ["closed_bos" if not true_retest else "fresh_retest_hold"]
        for condition, points, label in (
            (body_ok or true_retest, 20, "body_or_retest_confirm"),
            (rel_vol >= 1.20, 25, f"rel_vol={rel_vol:.2f}"),
            (expanding, 15, "volatility_expanding"),
            (not_extended, 15, f"extension={breakout_distance:.2f}ATR"),
        ):
            if condition:
                score += points
                notes.append(label)

        mandatory = true_retest or (body_ok and rel_vol >= 1.20)
        if not mandatory or not_extended is False or score < 70:
            return StrategySignal(False, raw_score=round(score, 1),
                                  reason="Breakout V2 lacks confirmation: " + ", ".join(notes))

        if direction == "long":
            sl = min(level - 0.25 * atr_val, float(lows[-2:].min()) - 0.10 * atr_val)
            sl = max(sl, price - 2.0 * atr_val)
            tp = price + 1.8 * (price - sl)
        else:
            sl = max(level + 0.25 * atr_val, float(highs[-2:].max()) + 0.10 * atr_val)
            sl = min(sl, price + 2.0 * atr_val)
            tp = price - 1.8 * (sl - price)

        return StrategySignal(
            True, direction, round(min(score, 100.0), 1), price,
            round(sl, 8), round(tp, 8),
            f"Close back inside breakout range through {level:.4f}",
            _rr(direction, price, sl, tp),
            "Breakout V2: " + ", ".join(notes),
            {"level": level, "rel_vol": rel_vol, "extension_atr": breakout_distance,
             "entry_type": "retest" if true_retest else "direct_close"},
        )


class SwingReversalStrategy:
    """V2: liquidity sweep -> CHOCH -> fresh retest/candle confirmation."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        opens = np.array([float(c.open) for c in candles], dtype=float)
        if len(closes) < 65:
            return StrategySignal(False, reason="insufficient_candles")

        atr_arr = ind.atr(closes, highs, lows, 14)
        rsi_arr = ind.rsi(closes, 14)
        ema8, ema13 = ind.ema(closes, 8), ind.ema(closes, 13)
        price = float(closes[-1])
        atr_val = _safe(atr_arr[-1], price * 0.01)

        prior_high = float(np.max(highs[-18:-3]))
        prior_low = float(np.min(lows[-18:-3]))
        sweep_down_recent = bool(np.any((lows[-3:] < prior_low) & (closes[-3:] > prior_low)))
        sweep_up_recent = bool(np.any((highs[-3:] > prior_high) & (closes[-3:] < prior_high)))

        # CHOCH uses a minor structure level that existed before the sweep.
        minor_high = float(np.max(highs[-9:-3]))
        minor_low = float(np.min(lows[-9:-3]))
        choch_up = price > minor_high and closes[-2] <= minor_high
        choch_down = price < minor_low and closes[-2] >= minor_low

        bull_engulf = ind.bullish_engulfing(opens[-2], closes[-2], opens[-1], closes[-1])
        bear_engulf = ind.bearish_engulfing(opens[-2], closes[-2], opens[-1], closes[-1])
        cross_up = _ema_cross(ema8, ema13, "long")
        cross_down = _ema_cross(ema8, ema13, "short")

        long_core = sweep_down_recent and choch_up
        short_core = sweep_up_recent and choch_down
        if long_core:
            direction = "long"
        elif short_core:
            direction = "short"
        else:
            return StrategySignal(False, reason="Swing reversal V2 requires liquidity sweep + CHOCH")

        if _blocked(direction, direction_hint):
            return StrategySignal(False, reason=f"Direction {direction} blocked by macro gate")

        candle_confirm = bull_engulf if direction == "long" else bear_engulf
        cross_confirm = cross_up if direction == "long" else cross_down
        rsi_val = _safe(rsi_arr[-1], 50.0)
        rsi_ok = rsi_val >= 45 if direction == "long" else rsi_val <= 55

        score = 60.0
        notes = ["liquidity_sweep", "choch"]
        if candle_confirm:
            score += 15; notes.append("engulfing")
        if cross_confirm:
            score += 15; notes.append("fresh_ema8_13_cross")
        if rsi_ok:
            score += 10; notes.append(f"rsi={rsi_val:.0f}")

        if not (candle_confirm or cross_confirm) or score < 75:
            return StrategySignal(False, raw_score=round(score, 1),
                                  reason="Swing reversal V2 waiting for retest/candle trigger: " + ", ".join(notes))

        if direction == "long":
            sl = float(np.min(lows[-4:])) - 0.25 * atr_val
            tp = price + 1.7 * (price - sl)
        else:
            sl = float(np.max(highs[-4:])) + 0.25 * atr_val
            tp = price - 1.7 * (sl - price)

        return StrategySignal(
            True, direction, round(min(score, 100.0), 1), price,
            round(sl, 8), round(tp, 8),
            "Price closes beyond the swept extreme and invalidates CHOCH",
            _rr(direction, price, sl, tp),
            "Swing reversal V2: " + ", ".join(notes),
            {"prior_high": prior_high, "prior_low": prior_low, "rsi": rsi_val},
        )


class MomentumExpansionStrategy:
    """V2: early expansion only; fresh micro breakout with volume and EMA alignment."""

    def evaluate(self, candles: list, direction_hint: str) -> StrategySignal:
        closes, highs, lows = _ohlc(candles)
        if len(closes) < 60:
            return StrategySignal(False, reason="insufficient_candles")

        vols = np.array([float(c.volume) for c in candles], dtype=float)
        roc_arr = ind.roc(closes, 9)
        atr_arr = ind.atr(closes, highs, lows, 14)
        ema8, ema13, ema20 = ind.ema(closes, 8), ind.ema(closes, 13), ind.ema(closes, 20)
        price = float(closes[-1])
        atr_val = _safe(atr_arr[-1], price * 0.01)
        roc_val = _safe(roc_arr[-1], 0.0)

        micro_high = float(np.max(highs[-6:-1]))
        micro_low = float(np.min(lows[-6:-1]))
        break_up = closes[-2] <= micro_high and price > micro_high
        break_down = closes[-2] >= micro_low and price < micro_low
        if break_up:
            direction = "long"
        elif break_down:
            direction = "short"
        else:
            return StrategySignal(False, reason="Momentum expansion V2: waiting for fresh micro breakout")

        if _blocked(direction, direction_hint):
            return StrategySignal(False, reason=f"Direction {direction} blocked by macro gate")

        ema_stack = ema8[-1] > ema13[-1] > ema20[-1] if direction == "long" else ema8[-1] < ema13[-1] < ema20[-1]
        roc_ok = roc_val > 0.8 if direction == "long" else roc_val < -0.8
        atr_base = float(np.nanmean(atr_arr[-20:-5]))
        atr_expansion = atr_val / max(atr_base, 1e-12)
        # Early expansion: enough increase to matter, but not a late volatility spike.
        atr_ok = 1.08 <= atr_expansion <= 1.65
        rel_vol = float(np.mean(vols[-2:])) / (float(np.mean(vols[-22:-2])) + 1e-9)
        volume_ok = rel_vol >= 1.20
        distance_atr = abs(price - ema20[-1]) / max(atr_val, 1e-12)
        location_ok = distance_atr <= 1.35
        body_ok = _body_ratio(candles[-1]) >= 0.50

        score = 20.0
        notes = ["fresh_micro_break"]
        for condition, points, label in (
            (ema_stack, 20, "ema8_13_20_stack"),
            (roc_ok, 15, f"roc9={roc_val:.2f}"),
            (atr_ok, 15, f"atr_expansion={atr_expansion:.2f}"),
            (volume_ok, 20, f"rel_vol={rel_vol:.2f}"),
            (location_ok, 5, f"distance={distance_atr:.2f}ATR"),
            (body_ok, 5, "body_confirm"),
        ):
            if condition:
                score += points
                notes.append(label)

        mandatory = ema_stack and roc_ok and atr_ok and volume_ok and location_ok
        if not mandatory or score < 75:
            return StrategySignal(False, raw_score=round(score, 1),
                                  reason="Momentum expansion V2 incomplete: " + ", ".join(notes))

        level = micro_high if direction == "long" else micro_low
        if direction == "long":
            sl = min(level - 0.20 * atr_val, price - 1.0 * atr_val)
            sl = max(sl, price - 1.8 * atr_val)
            tp = price + 1.5 * (price - sl)
        else:
            sl = max(level + 0.20 * atr_val, price + 1.0 * atr_val)
            sl = min(sl, price + 1.8 * atr_val)
            tp = price - 1.5 * (sl - price)

        return StrategySignal(
            True, direction, round(min(score, 100.0), 1), price,
            round(sl, 8), round(tp, 8),
            "EMA8/13 loses alignment or price closes back through breakout level",
            _rr(direction, price, sl, tp),
            "Momentum expansion V2: " + ", ".join(notes),
            {"roc": roc_val, "atr_expansion": atr_expansion,
             "rel_vol": rel_vol, "distance_ema20_atr": distance_atr},
        )


# Shared helpers

def _ohlc(candles: list):
    closes = np.array([float(c.close) for c in candles], dtype=float)
    highs = np.array([float(c.high) for c in candles], dtype=float)
    lows = np.array([float(c.low) for c in candles], dtype=float)
    return closes, highs, lows


def _safe(val: float, default: float) -> float:
    return float(val) if val is not None and not np.isnan(val) else default


def _rolling_vwap(closes: np.ndarray, vols: np.ndarray, window: int = 20) -> float:
    window = min(window, len(closes))
    c, v = closes[-window:], vols[-window:]
    total_v = float(np.sum(v))
    return float(np.sum(c * v) / total_v) if total_v > 0 else float(np.mean(c))
