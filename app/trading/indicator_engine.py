"""Compact indicators for Adaptive Bot v12."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

def _values(candles, name: str):
    return [float(getattr(c, name, c[name] if isinstance(c, dict) else 0.0)) for c in candles]

def ema(values: List[float], length: int) -> List[float]:
    if not values: return []
    alpha = 2.0 / (length + 1.0)
    out = [float(values[0])]
    for value in values[1:]: out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out

def atr(candles, length: int = 14) -> float:
    if len(candles) < 2: return 0.0
    h, l, c = _values(candles, "high"), _values(candles, "low"), _values(candles, "close")
    tr = [h[0] - l[0]]
    for i in range(1, len(c)):
        tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    return float(np.mean(tr[-length:]))

def rsi(values: List[float], length: int = 14) -> float:
    if len(values) <= length: return 50.0
    delta = np.diff(np.asarray(values[-(length+1):], dtype=float))
    gain, loss = np.mean(np.clip(delta, 0, None)), np.mean(np.clip(-delta, 0, None))
    if loss <= 1e-12: return 100.0
    return float(100.0 - 100.0 / (1.0 + gain/loss))

def adx(candles, length: int = 14) -> float:
    if len(candles) < length + 2: return 20.0
    h, l, c = _values(candles, "high"), _values(candles, "low"), _values(candles, "close")
    pdm, mdm, tr = [], [], []
    for i in range(1, len(c)):
        up, down = h[i]-h[i-1], l[i-1]-l[i]
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
        tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    tr_sum = max(sum(tr[-length:]), 1e-12)
    pdi, mdi = 100.0*sum(pdm[-length:])/tr_sum, 100.0*sum(mdm[-length:])/tr_sum
    return float(100.0*abs(pdi-mdi)/max(pdi+mdi, 1e-12))

def bollinger(values: List[float], length: int = 20, multiplier: float = 2.0) -> Dict[str, float]:
    window = np.asarray(values[-length:], dtype=float)
    mid, sd = float(np.mean(window)), float(np.std(window))
    return {"mid": mid, "upper": mid+multiplier*sd, "lower": mid-multiplier*sd,
            "width": (2.0*multiplier*sd)/max(abs(mid), 1e-12)}

def compute(candles) -> Dict[str, Any]:
    if len(candles) < 60: return {}
    closes, opens = _values(candles, "close"), _values(candles, "open")
    highs, lows, volumes = _values(candles, "high"), _values(candles, "low"), _values(candles, "volume")
    e8, e13, e20, e50 = (ema(closes, n) for n in (8,13,20,50))
    e12, e26 = ema(closes,12), ema(closes,26)
    bb, a = bollinger(closes), atr(candles)
    return {
        "open": opens[-1], "high": highs[-1], "low": lows[-1], "close": closes[-1],
        "prev_close": closes[-2], "prev_high": highs[-2], "prev_low": lows[-2],
        "ema8": e8[-1], "ema13": e13[-1], "ema20": e20[-1], "ema50": e50[-1],
        "ema8_prev": e8[-2], "ema13_prev": e13[-2], "ema12": e12[-1], "ema26": e26[-1],
        "ema12_prev": e12[-2], "ema26_prev": e26[-2],
        "cdc_bull": e12[-1] > e26[-1], "cdc_bear": e12[-1] < e26[-1],
        "cdc_cross_up": e12[-2] <= e26[-2] and e12[-1] > e26[-1],
        "cdc_cross_down": e12[-2] >= e26[-2] and e12[-1] < e26[-1],
        "bb_mid": bb["mid"], "bb_upper": bb["upper"], "bb_lower": bb["lower"], "bb_width": bb["width"],
        "atr": a, "adx": adx(candles), "rsi": rsi(closes), "volume": volumes[-1],
        "vol_avg": float(np.mean(volumes[-20:])),
        "body_atr": abs(closes[-1]-opens[-1])/max(a,1e-12),
        "extension_atr": abs(closes[-1]-e20[-1])/max(a,1e-12),
        "swing_high": max(highs[-10:-1]), "swing_low": min(lows[-10:-1]),
    }

class IndicatorEngine:
    @staticmethod
    def _candle(candles) -> Dict[str, Any]:
        if not candles: return {}
        c = candles[-1]
        return {"open": float(c.open), "high": float(c.high), "low": float(c.low), "close": float(c.close), "volume": float(c.volume)}
    def compute(self, c15m, c1h, c4h) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
        return self._candle(c15m), self._candle(c1h), self._candle(c4h), compute(c15m), compute(c1h), compute(c4h)
