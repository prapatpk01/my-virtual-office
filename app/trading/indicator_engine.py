"""Adaptive SMC MTF indicator engine.

Pipeline (closed candles only):
  4H  -> TSS-style direction filter (EMA20/50 + HMA16 slope)
  15M -> Market Structure (HH/HL, LH/LL, BOS/CHOCH)
  5M  -> AMD setup (Accumulation -> Manipulation sweep -> Distribution)
  1M  -> IFVG execution (inverted FVG + fresh retest/rejection)

The engine is intentionally deterministic and testable.  "TSS-style" here is
an internal trend-tunnel approximation; it does not claim to reproduce any
proprietary TSS indicator.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

ENGINE_SCHEMA = "adaptive-smc-mtf-v1"


def _v(c: Any, name: str, idx: int) -> float:
    value = getattr(c, name, None)
    if value is None and isinstance(c, dict):
        value = c.get(name)
    if value is None and isinstance(c, (list, tuple)) and len(c) > idx:
        value = c[idx]
    return float(value or 0.0)


def _series(candles: List[Any], name: str, idx: int) -> List[float]:
    return [_v(c, name, idx) for c in candles]


def ema(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def _wma(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    length = max(1, int(length))
    out = [float(values[0])] * len(values)
    weights = list(range(1, length + 1))
    denom = float(sum(weights))
    for i in range(length - 1, len(values)):
        window = values[i - length + 1:i + 1]
        out[i] = sum(float(v) * w for v, w in zip(window, weights)) / denom
    if length > 1:
        seed = out[length - 1] if len(out) >= length else float(values[-1])
        for i in range(min(length - 1, len(out))):
            out[i] = seed
    return out


def _hma(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    half = max(2, length // 2)
    root = max(2, int(round(math.sqrt(length))))
    wh = _wma(values, half)
    wf = _wma(values, length)
    raw = [2.0 * a - b for a, b in zip(wh, wf)]
    return _wma(raw, root)


def _rma(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    alpha = 1.0 / max(length, 1)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def _true_range(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    if not closes:
        return []
    out = [max(highs[0] - lows[0], 0.0)]
    for i in range(1, len(closes)):
        out.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return out


def _atr(candles: List[Any], length: int = 14) -> Tuple[List[float], float]:
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    if not closes:
        return [], 0.0
    arr = _rma(_true_range(highs, lows, closes), length)
    return arr, max(arr[-1], closes[-1] * 0.00025)


def _rsi(closes: List[float], length: int = 14) -> List[float]:
    if not closes:
        return []
    gains, losses = [0.0], [0.0]
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = _rma(gains, length)
    avg_loss = _rma(losses, length)
    out: List[float] = []
    for gain, loss in zip(avg_gain, avg_loss):
        if loss <= 1e-12:
            out.append(100.0 if gain > 0 else 50.0)
        else:
            rs = gain / loss
            out.append(100.0 - 100.0 / (1.0 + rs))
    return out


def _pivots(values: List[float], kind: str, left: int = 2, right: int = 2) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    if len(values) < left + right + 1:
        return out
    for i in range(left, len(values) - right):
        v = values[i]
        window = values[i - left:i + right + 1]
        if kind == "high":
            if v == max(window) and window.count(v) == 1:
                out.append((i, v))
        else:
            if v == min(window) and window.count(v) == 1:
                out.append((i, v))
    return out


def _tss_4h(candles: List[Any]) -> Dict[str, Any]:
    closes = _series(candles, "close", 4)
    if len(closes) < 55:
        return {"bias": "NEUTRAL", "score": 0.0}
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    h16 = _hma(closes, 16)
    c = closes[-1]
    slope_span = min(3, len(e20) - 1)
    e20_up = e20[-1] > e20[-1 - slope_span]
    e20_dn = e20[-1] < e20[-1 - slope_span]
    hma_up = h16[-1] > h16[-2]
    hma_dn = h16[-1] < h16[-2]
    long_votes = sum((c > e20[-1] > e50[-1], e20_up, hma_up))
    short_votes = sum((c < e20[-1] < e50[-1], e20_dn, hma_dn))
    if long_votes >= 2 and long_votes > short_votes:
        bias = "LONG"
        score = 55.0 + long_votes * 15.0
    elif short_votes >= 2 and short_votes > long_votes:
        bias = "SHORT"
        score = 55.0 + short_votes * 15.0
    else:
        bias = "NEUTRAL"
        score = 40.0 + 5.0 * max(long_votes, short_votes)
    return {
        "bias": bias,
        "score": min(score, 100.0),
        "close": c,
        "ema20": e20[-1],
        "ema50": e50[-1],
        "hma16": h16[-1],
        "ema20_slope": e20[-1] - e20[-1 - slope_span],
        "hma16_slope": h16[-1] - h16[-2],
        "long_votes": long_votes,
        "short_votes": short_votes,
    }


def _structure_15m(candles: List[Any]) -> Dict[str, Any]:
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    if len(closes) < 30:
        return {"state": "UNKNOWN", "bias": "NEUTRAL"}
    ph = _pivots(highs, "high", 2, 2)
    pl = _pivots(lows, "low", 2, 2)
    if len(ph) < 2 or len(pl) < 2:
        return {"state": "UNKNOWN", "bias": "NEUTRAL"}
    h1, h2 = ph[-2][1], ph[-1][1]
    l1, l2 = pl[-2][1], pl[-1][1]
    hh, lh = h2 > h1, h2 < h1
    hl, ll = l2 > l1, l2 < l1
    if hh and hl:
        state, bias = "HH/HL", "LONG"
    elif lh and ll:
        state, bias = "LH/LL", "SHORT"
    else:
        state, bias = "TRANSITION", "NEUTRAL"
    last_close = closes[-1]
    bos_up = last_close > h2
    bos_down = last_close < l2
    choch_up = bias == "SHORT" and bos_up
    choch_down = bias == "LONG" and bos_down
    return {
        "state": state,
        "bias": bias,
        "last_swing_high": h2,
        "previous_swing_high": h1,
        "last_swing_low": l2,
        "previous_swing_low": l1,
        "bos_up": bos_up,
        "bos_down": bos_down,
        "choch_up": choch_up,
        "choch_down": choch_down,
        "allow_long": bias == "LONG" or choch_up,
        "allow_short": bias == "SHORT" or choch_down,
    }


def _amd_5m(candles: List[Any]) -> Dict[str, Any]:
    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    if len(closes) < 40:
        return {"phase": "WAIT", "long_ready": False, "short_ready": False}
    _, atr = _atr(candles, 14)

    # A fixed lookback makes the pattern deterministic: older block = accumulation,
    # recent block = potential manipulation and displacement/distribution.
    acc_start, acc_end = len(closes) - 26, len(closes) - 7
    acc_hi = max(highs[acc_start:acc_end])
    acc_lo = min(lows[acc_start:acc_end])
    acc_width = acc_hi - acc_lo
    accumulation_ok = acc_width <= max(4.0 * atr, closes[-1] * 0.018)
    tol = 0.05 * atr

    long_sweep_idx: Optional[int] = None
    short_sweep_idx: Optional[int] = None
    for i in range(acc_end, len(closes)):
        if lows[i] < acc_lo - tol and closes[i] > acc_lo:
            long_sweep_idx = i
        if highs[i] > acc_hi + tol and closes[i] < acc_hi:
            short_sweep_idx = i

    long_ready = False
    short_ready = False
    manipulation_low = min(lows[acc_end:])
    manipulation_high = max(highs[acc_end:])

    if accumulation_ok and long_sweep_idx is not None and long_sweep_idx < len(closes) - 1:
        sweep_high = highs[long_sweep_idx]
        post = range(long_sweep_idx + 1, len(closes))
        displacement = any(
            closes[i] > sweep_high
            or (closes[i] > opens[i] and (closes[i] - opens[i]) >= 0.55 * atr and closes[i] > (acc_hi + acc_lo) / 2)
            for i in post
        )
        long_ready = displacement
    if accumulation_ok and short_sweep_idx is not None and short_sweep_idx < len(closes) - 1:
        sweep_low = lows[short_sweep_idx]
        post = range(short_sweep_idx + 1, len(closes))
        displacement = any(
            closes[i] < sweep_low
            or (closes[i] < opens[i] and (opens[i] - closes[i]) >= 0.55 * atr and closes[i] < (acc_hi + acc_lo) / 2)
            for i in post
        )
        short_ready = displacement

    if long_ready and not short_ready:
        phase = "DISTRIBUTION_LONG"
    elif short_ready and not long_ready:
        phase = "DISTRIBUTION_SHORT"
    elif long_sweep_idx is not None or short_sweep_idx is not None:
        phase = "MANIPULATION"
    elif accumulation_ok:
        phase = "ACCUMULATION"
    else:
        phase = "WAIT"

    return {
        "phase": phase,
        "long_ready": long_ready,
        "short_ready": short_ready,
        "accumulation_ok": accumulation_ok,
        "range_high": acc_hi,
        "range_low": acc_lo,
        "range_width_atr": acc_width / max(atr, 1e-12),
        "manipulation_low": manipulation_low,
        "manipulation_high": manipulation_high,
        "long_sweep_age": (len(closes) - 1 - long_sweep_idx) if long_sweep_idx is not None else None,
        "short_sweep_age": (len(closes) - 1 - short_sweep_idx) if short_sweep_idx is not None else None,
        "atr": atr,
    }


def _ifvg_1m(candles: List[Any], direction: str) -> Dict[str, Any]:
    opens = _series(candles, "open", 1)
    highs = _series(candles, "high", 2)
    lows = _series(candles, "low", 3)
    closes = _series(candles, "close", 4)
    if len(closes) < 50:
        return {"valid": False, "direction": direction}
    _, atr = _atr(candles, 14)
    tol = 0.12 * atr
    start = max(2, len(closes) - 70)

    candidates: List[Dict[str, Any]] = []
    for i in range(start, len(closes) - 3):
        if direction == "LONG":
            # Bearish FVG: candle i trades entirely below candle i-2 low.
            if highs[i] >= lows[i - 2]:
                continue
            zone_low, zone_high = highs[i], lows[i - 2]
            invert_idx = next((j for j in range(i + 1, len(closes)) if closes[j] > zone_high), None)
            if invert_idx is None or invert_idx >= len(closes) - 1:
                continue
            # Fresh retest/rejection must happen in the last three closed bars.
            retests = [j for j in range(max(invert_idx + 1, len(closes) - 3), len(closes))
                       if lows[j] <= zone_high + tol and closes[j] >= zone_high]
            if not retests:
                continue
            j = retests[-1]
            rejection = closes[j] > opens[j] or closes[-1] > zone_high
            if not rejection:
                continue
        else:
            # Bullish FVG: candle i trades entirely above candle i-2 high.
            if lows[i] <= highs[i - 2]:
                continue
            zone_low, zone_high = highs[i - 2], lows[i]
            invert_idx = next((j for j in range(i + 1, len(closes)) if closes[j] < zone_low), None)
            if invert_idx is None or invert_idx >= len(closes) - 1:
                continue
            retests = [j for j in range(max(invert_idx + 1, len(closes) - 3), len(closes))
                       if highs[j] >= zone_low - tol and closes[j] <= zone_low]
            if not retests:
                continue
            j = retests[-1]
            rejection = closes[j] < opens[j] or closes[-1] < zone_low
            if not rejection:
                continue

        candidates.append({
            "valid": True,
            "direction": direction,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "fvg_index": i,
            "invert_index": invert_idx,
            "retest_index": j,
            "age": len(closes) - 1 - j,
            "atr": atr,
        })

    if not candidates:
        return {"valid": False, "direction": direction, "atr": atr}
    return candidates[-1]


def compute(c1m: List[Any], c5m: Optional[List[Any]] = None,
            c15m: Optional[List[Any]] = None, c4h: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Compute the complete Adaptive SMC MTF decision from closed candles."""
    # Backward-compatible guard: old callers passing one 15M series get no signal
    # rather than accidentally trading with mismatched semantics.
    if c5m is None or c15m is None or c4h is None:
        return {}
    if len(c1m) < 70 or len(c5m) < 50 or len(c15m) < 50 or len(c4h) < 60:
        return {}

    tss = _tss_4h(c4h)
    structure = _structure_15m(c15m)
    amd = _amd_5m(c5m)
    ifvg_long = _ifvg_1m(c1m, "LONG")
    ifvg_short = _ifvg_1m(c1m, "SHORT")

    closes1 = _series(c1m, "close", 4)
    closes15 = _series(c15m, "close", 4)
    ema20_15 = ema(closes15, 20)
    opens1 = _series(c1m, "open", 1)
    highs1 = _series(c1m, "high", 2)
    lows1 = _series(c1m, "low", 3)
    volumes1 = _series(c1m, "volume", 5)
    _, atr1 = _atr(c1m, 14)
    rsi1 = _rsi(closes1, 14)

    long_signal = (
        tss.get("bias") == "LONG"
        and bool(structure.get("allow_long"))
        and bool(amd.get("long_ready"))
        and bool(ifvg_long.get("valid"))
    )
    short_signal = (
        tss.get("bias") == "SHORT"
        and bool(structure.get("allow_short"))
        and bool(amd.get("short_ready"))
        and bool(ifvg_short.get("valid"))
    )

    direction = "LONG" if long_signal else "SHORT" if short_signal else "NONE"
    chosen = ifvg_long if direction == "LONG" else ifvg_short if direction == "SHORT" else {}

    # Stop is placed beyond the manipulation extreme with a small volatility
    # buffer.  It is structure-based, not an arbitrary fixed percentage.
    sl = 0.0
    if direction == "LONG":
        sl = float(amd["manipulation_low"]) - 0.15 * float(amd["atr"])
    elif direction == "SHORT":
        sl = float(amd["manipulation_high"]) + 0.15 * float(amd["atr"])

    trigger = ""
    if direction != "NONE":
        trigger = f"4H {tss['bias']} → M15 {structure['state']} → M5 {amd['phase']} → M1 {direction} IFVG retest"

    # Runner exits only on a meaningful higher-timeframe invalidation.
    runner_exit_long = bool(structure.get("choch_down")) or amd.get("phase") == "DISTRIBUTION_SHORT"
    runner_exit_short = bool(structure.get("choch_up")) or amd.get("phase") == "DISTRIBUTION_LONG"

    return {
        "schema": ENGINE_SCHEMA,
        "timeframe": "1M_EXECUTION",
        "open": opens1[-1], "high": highs1[-1], "low": lows1[-1], "close": closes1[-1],
        "volume": volumes1[-1], "atr1": atr1, "rsi1": rsi1[-1],
        "m15_close": closes15[-1], "m15_ema20": ema20_15[-1],
        "tss_bias": tss.get("bias", "NEUTRAL"),
        "tss_score": float(tss.get("score", 0.0)),
        "tss": tss,
        "structure": structure.get("state", "UNKNOWN"),
        "structure_bias": structure.get("bias", "NEUTRAL"),
        "m15": structure,
        "amd_phase": amd.get("phase", "WAIT"),
        "amd": amd,
        "ifvg_long": ifvg_long,
        "ifvg_short": ifvg_short,
        "ifvg_valid": bool(chosen.get("valid")),
        "ifvg_low": float(chosen.get("zone_low", 0.0) or 0.0),
        "ifvg_high": float(chosen.get("zone_high", 0.0) or 0.0),
        "manipulation_low": float(amd.get("manipulation_low", 0.0) or 0.0),
        "manipulation_high": float(amd.get("manipulation_high", 0.0) or 0.0),
        "sl": sl,
        "long_signal": long_signal,
        "short_signal": short_signal,
        "direction": direction,
        "trigger": trigger,
        "runner_exit_long": runner_exit_long,
        "runner_exit_short": runner_exit_short,
    }


class IndicatorEngine:
    def compute(self, c1m: List[Any], c5m: List[Any], c15m: List[Any], c4h: List[Any]):
        return compute(c1m, c5m, c15m, c4h)
