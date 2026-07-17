"""Data Quality Gate (spec §5) + volatility shock detection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config, TF_MS
from .enums import ReasonCode


@dataclass
class QualityResult:
    valid: bool
    reason_codes: list = field(default_factory=list)
    shock: bool = False
    severe_shock: bool = False
    detail: str = ""


def _check_series(candles: list, tf: str, min_n: int, codes: list) -> bool:
    if len(candles) < min_n:
        codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:{tf}_len_{len(candles)}<{min_n}")
        return False
    step = TF_MS[tf]
    ts = [c.timestamp for c in candles]
    if len(set(ts)) != len(ts):
        codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:{tf}_dup_bars")
        return False
    gaps = sum(1 for a, b in zip(ts, ts[1:]) if b - a != step)
    if gaps > max(3, len(ts) // 50):    # tolerate exchange maintenance gaps
        codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:{tf}_gaps_{gaps}")
        return False
    for c in candles[-50:]:
        if not (c.high >= max(c.open, c.close, c.low) and c.low <= min(c.open, c.close, c.high)):
            codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:{tf}_bad_ohlc")
            return False
    return True


class DataQualityGate:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, symbol: str, candles_15m: list, candles_1h: list,
                 candles_4h: list, exchange_state: Optional[dict] = None,
                 now_ms: Optional[int] = None) -> QualityResult:
        c = self.cfg
        codes: list = []
        ok = True
        ok &= _check_series(candles_15m, "15m", c.min_15m_candles, codes)
        ok &= _check_series(candles_1h, "1h", c.min_1h_candles, codes)
        ok &= _check_series(candles_4h, "4h", c.min_4h_candles, codes)
        if not ok:
            return QualityResult(False, codes)

        now_ms = now_ms or int(time.time() * 1000)
        last_close_ms = candles_15m[-1].timestamp + TF_MS["15m"]
        if now_ms - last_close_ms > (TF_MS["15m"] + c.max_price_staleness_sec * 1000):
            codes.append(ReasonCode.REJECT_STALE_DATA.value)
            return QualityResult(False, codes)

        # ATR > 0
        rng = np.array([b.high - b.low for b in candles_15m[-20:]])
        if float(np.mean(rng)) <= 0:
            codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:zero_atr")
            return QualityResult(False, codes)

        # exchange-state checks (spread / unknowns / rules) when provided
        if exchange_state:
            spread = exchange_state.get("spread_pct")
            if spread is not None and spread > c.max_spread_pct:
                codes.append(ReasonCode.REJECT_SPREAD.value)
                return QualityResult(False, codes)
            if exchange_state.get("unknown_position"):
                codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:unknown_position")
                return QualityResult(False, codes)
            if exchange_state.get("unknown_order"):
                codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:unknown_order")
                return QualityResult(False, codes)
            if exchange_state.get("rules_missing"):
                codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:rules_missing")
                return QualityResult(False, codes)
            skew = exchange_state.get("clock_skew_sec")
            if skew is not None and abs(skew) > c.max_clock_skew_sec:
                codes.append(f"{ReasonCode.REJECT_DATA_QUALITY.value}:clock_skew")
                return QualityResult(False, codes)

        # ── volatility shock ─────────────────────────────────────────────────
        shock = severe = False
        if len(candles_15m) >= 25:
            cur_rng = candles_15m[-1].high - candles_15m[-1].low
            med20 = float(np.median([b.high - b.low for b in candles_15m[-21:-1]]))
            atr14 = float(np.mean([b.high - b.low for b in candles_15m[-15:-1]]))
            atr_short = float(np.mean([b.high - b.low for b in candles_15m[-5:-1]]))
            if med20 > 0 and cur_rng > med20 * c.shock_range_median_mult:
                shock = True
            if atr14 > 0 and atr_short > atr14 * c.shock_atr_expansion:
                shock = True
            if atr14 > 0 and atr_short > atr14 * c.severe_shock_atr_expansion:
                severe = True
            if med20 > 0 and cur_rng > med20 * c.shock_range_median_mult * 1.6:
                severe = True
        if shock:
            codes.append(ReasonCode.VOLATILITY_SHOCK.value)
        return QualityResult(True, codes, shock=shock, severe_shock=severe)
