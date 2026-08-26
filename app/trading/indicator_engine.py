"""Adaptive Momentum v4.3 — 15M-only dual-entry indicator engine.

Two independent execution triggers:
1) Momentum entry: market-structure BOS/CHOCH.
2) Pullback entry: EMA13 reclaim/reject.

EMA8/13 is alignment/confirmation, not the mandatory trigger. ADX+CHOP validate
market quality. MACD12/26/9, ROC9, Bollinger location and market structure form
a 4-point confirmation score. ATR14 drives risk. No 1H/4H dependency.
"""
from __future__ import annotations
from typing import Any, Dict, List
import math

ENGINE_SCHEMA = "adaptive-momentum-v4.3-dual-entry-15m"


def _v(c: Any, name: str, idx: int) -> float:
    v = getattr(c, name, None)
    if v is None and isinstance(c, dict):
        v = c.get(name)
    if v is None and isinstance(c, (list, tuple)) and len(c) > idx:
        v = c[idx]
    return float(v or 0.0)


def _series(cs, name, idx):
    return [_v(c, name, idx) for c in cs]


def ema(xs: List[float], n: int) -> List[float]:
    if not xs:
        return []
    a = 2.0 / (n + 1.0)
    out = [float(xs[0])]
    for x in xs[1:]:
        out.append(a * float(x) + (1.0 - a) * out[-1])
    return out


def _sma(xs: List[float], n: int) -> List[float]:
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - n + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def _rma(xs: List[float], n: int) -> List[float]:
    if not xs:
        return []
    a = 1.0 / max(n, 1)
    out = [float(xs[0])]
    for x in xs[1:]:
        out.append(a * float(x) + (1.0 - a) * out[-1])
    return out


def _tr(h, l, c):
    out = [max(h[0] - l[0], 0.0)]
    for i in range(1, len(c)):
        out.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return out


def _adx(h, l, c, n=14):
    tr = _tr(h, l, c)
    pd, md = [0.0], [0.0]
    for i in range(1, len(c)):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        pd.append(up if up > dn and up > 0 else 0.0)
        md.append(dn if dn > up and dn > 0 else 0.0)
    ar, pr, mr = _rma(tr, n), _rma(pd, n), _rma(md, n)
    dx = []
    for a, p, m in zip(ar, pr, mr):
        if a <= 1e-12:
            dx.append(0.0)
            continue
        pdi, mdi = 100.0 * p / a, 100.0 * m / a
        total = pdi + mdi
        dx.append(100.0 * abs(pdi - mdi) / total if total > 1e-12 else 0.0)
    return _rma(dx, n)


def _chop(h, l, c, n=14):
    s = sum(_tr(h, l, c)[-n:])
    span = max(h[-n:]) - min(l[-n:])
    return 100.0 if s <= 0 or span <= 1e-12 else 100.0 * math.log10(s / span) / math.log10(n)


def _bb(c, n=20, mult=2.0):
    mid = _sma(c, n)
    upper, lower = [], []
    for i in range(len(c)):
        w = c[max(0, i - n + 1):i + 1]
        mean = mid[i]
        sd = math.sqrt(sum((z - mean) ** 2 for z in w) / len(w)) if w else 0.0
        upper.append(mean + mult * sd)
        lower.append(mean - mult * sd)
    return mid, upper, lower


def compute(candles: List[Any]) -> Dict[str, Any]:
    if len(candles) < 100:
        return {}

    o = _series(candles, "open", 1)
    h = _series(candles, "high", 2)
    l = _series(candles, "low", 3)
    c = _series(candles, "close", 4)
    v = _series(candles, "volume", 5)

    e8, e13 = ema(c, 8), ema(c, 13)
    macd_line = [a - b for a, b in zip(ema(c, 12), ema(c, 26))]
    macd_signal = ema(macd_line, 9)
    macd_hist = [a - b for a, b in zip(macd_line, macd_signal)]
    bb_mid, bb_upper, bb_lower = _bb(c, 20, 2.0)
    atrs = _rma(_tr(h, l, c), 14)
    atr = max(atrs[-1], c[-1] * 0.0005)
    adxs = _adx(h, l, c, 14)
    chop = _chop(h, l, c, 14)
    roc9 = ((c[-1] / c[-10]) - 1.0) * 100.0 if c[-10] else 0.0

    # EMA8/13 is alignment. A cross is still exposed for management/diagnostics.
    ema_bull = e8[-1] > e13[-1]
    ema_bear = e8[-1] < e13[-1]
    cross_up = ema_bull and e8[-2] <= e13[-2]
    cross_down = ema_bear and e8[-2] >= e13[-2]

    # Momentum tools.
    macd_long = macd_line[-1] > macd_signal[-1] or macd_hist[-1] > macd_hist[-2]
    macd_short = macd_line[-1] < macd_signal[-1] or macd_hist[-1] < macd_hist[-2]
    roc_long = roc9 > 0
    roc_short = roc9 < 0
    momentum_long = int(macd_long) + int(roc_long)
    momentum_short = int(macd_short) + int(roc_short)

    # Bollinger location: correct half of band, but never chase outside the band.
    bb_long = bb_lower[-1] <= c[-1] <= bb_upper[-1] and c[-1] >= bb_mid[-1]
    bb_short = bb_lower[-1] <= c[-1] <= bb_upper[-1] and c[-1] <= bb_mid[-1]
    bb_width = (bb_upper[-1] - bb_lower[-1]) / max(abs(bb_mid[-1]), 1e-12)
    prev_bb_width = (bb_upper[-2] - bb_lower[-2]) / max(abs(bb_mid[-2]), 1e-12)

    # Local structure from CLOSED 15M candles.
    prior_low = min(l[-10:-5])
    prior_high = max(h[-10:-5])
    recent_low = min(l[-5:-1])
    recent_high = max(h[-5:-1])
    hl = recent_low > prior_low
    lh = recent_high < prior_high

    # BOS is deliberately local to avoid waiting for a large swing break.
    bos_level_long = max(h[-5:-1])
    bos_level_short = min(l[-5:-1])
    bos_long = c[-1] > bos_level_long
    bos_short = c[-1] < bos_level_short

    # CHOCH: price breaks the older opposing swing after an opposing local structure developed.
    choch_long = lh and c[-1] > prior_high
    choch_short = hl and c[-1] < prior_low
    structure_break_long = bos_long or choch_long
    structure_break_short = bos_short or choch_short

    # Supportive structure used by the confirmation score.
    structure_long = hl or structure_break_long
    structure_short = lh or structure_break_short

    # Pullback engine: current CLOSED candle probes EMA13 and reclaims/rejects it.
    # Small ATR tolerance prevents missing a valid near-touch by a few ticks.
    touch_tol = 0.10 * atr
    ema13_reclaim_long = (
        ema_bull
        and l[-1] <= e13[-1] + touch_tol
        and c[-1] > e13[-1]
        and c[-1] > o[-1]
    )
    ema13_reclaim_short = (
        ema_bear
        and h[-1] >= e13[-1] - touch_tol
        and c[-1] < e13[-1]
        and c[-1] < o[-1]
    )

    # Dual entry triggers. Direction comes from trigger + EMA alignment.
    momentum_trigger_long = ema_bull and structure_break_long
    momentum_trigger_short = ema_bear and structure_break_short
    pullback_trigger_long = ema13_reclaim_long
    pullback_trigger_short = ema13_reclaim_short
    entry_trigger_long = momentum_trigger_long or pullback_trigger_long
    entry_trigger_short = momentum_trigger_short or pullback_trigger_short

    # Invalidation reference for position management.
    swing_low = min(l[-6:-1])
    swing_high = max(h[-6:-1])
    structure_invalid_long = c[-1] < swing_low
    structure_invalid_short = c[-1] > swing_high

    # Adaptive confirmation: minimum 3/4 across independent tools.
    confirmation_long = int(macd_long) + int(roc_long) + int(bb_long) + int(structure_long)
    confirmation_short = int(macd_short) + int(roc_short) + int(bb_short) + int(structure_short)

    return {
        "schema": ENGINE_SCHEMA,
        "timeframe": "15M",
        "open": o[-1], "high": h[-1], "low": l[-1], "close": c[-1], "prev_close": c[-2],
        "ema8": e8[-1], "ema13": e13[-1], "ema8_prev": e8[-2], "ema13_prev": e13[-2],
        "ema_bull": ema_bull, "ema_bear": ema_bear,
        "ema_cross_up": cross_up, "ema_cross_down": cross_down,
        "ema13_reclaim_long": ema13_reclaim_long, "ema13_reclaim_short": ema13_reclaim_short,
        "macd": macd_line[-1], "macd_signal": macd_signal[-1],
        "macd_hist": macd_hist[-1], "macd_hist_prev": macd_hist[-2],
        "macd_bull": macd_long, "macd_bear": macd_short,
        "roc9": roc9, "roc_long": roc_long, "roc_short": roc_short,
        "momentum_score_long": momentum_long, "momentum_score_short": momentum_short,
        "bb_mid": bb_mid[-1], "bb_upper": bb_upper[-1], "bb_lower": bb_lower[-1],
        "bb_width": bb_width, "bb_expanding": bb_width > prev_bb_width,
        "bb_long": bb_long, "bb_short": bb_short,
        "atr": atr, "adx": adxs[-1], "adx_prev": adxs[-2],
        "adx_rising": adxs[-1] > adxs[-2], "chop": chop,
        "structure_hl": hl, "structure_lh": lh,
        "structure_bos_long": bos_long, "structure_bos_short": bos_short,
        "structure_choch_long": choch_long, "structure_choch_short": choch_short,
        "structure_break_long": structure_break_long, "structure_break_short": structure_break_short,
        "structure_long": structure_long, "structure_short": structure_short,
        "structure_invalid_long": structure_invalid_long,
        "structure_invalid_short": structure_invalid_short,
        "bos_level_long": bos_level_long, "bos_level_short": bos_level_short,
        "momentum_trigger_long": momentum_trigger_long,
        "momentum_trigger_short": momentum_trigger_short,
        "pullback_trigger_long": pullback_trigger_long,
        "pullback_trigger_short": pullback_trigger_short,
        "entry_trigger_long": entry_trigger_long,
        "entry_trigger_short": entry_trigger_short,
        "confirmation_score_long": confirmation_long,
        "confirmation_score_short": confirmation_short,
        "recent_low": swing_low, "recent_high": swing_high,
        "volume": v[-1],
    }


class IndicatorEngine:
    def compute(self, c15m: List[Any], c1h: List[Any], c4h: List[Any]):
        return compute(c15m), {}, {}
