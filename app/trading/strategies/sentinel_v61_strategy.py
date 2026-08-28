"""Sentinel V6.1 — RSI Momentum Quality.

Keeps V6 intentionally small and adds only entry-quality information:
- HARD quality confirmation: RSI14 slope + RSI-SMA14 spread expansion.
- Soft bonus only: recent classic RSI divergence (never blocks a trade).
- 1H direction, 15M ADX/CHOP/ATR gate, zones, SL and 2TP stay unchanged.
"""
from __future__ import annotations

import numpy as np

from .sentinel_v6_strategy import SentinelV6Strategy


class SentinelV61Strategy(SentinelV6Strategy):
    VERSION = "6.1"
    DIVERGENCE_LOOKBACK = 50
    DIVERGENCE_RECENT_BARS = 12
    DIVERGENCE_RSI_DELTA = 0.50
    DIVERGENCE_CONF_BONUS = 0.05

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV6.1({symbol})"

    @classmethod
    def _classic_rsi_divergence(cls, candles: list, rsi: np.ndarray) -> dict:
        """Detect recent classic divergence from confirmed span-2 price pivots.

        Bullish: price lower-low while RSI makes a higher-low.
        Bearish: price higher-high while RSI makes a lower-high.
        This is context only; it never vetoes an entry.
        """
        n = len(candles)
        if n < 20 or len(rsi) != n:
            return {"bullish": False, "bearish": False, "label": "NONE"}

        start = max(2, n - cls.DIVERGENCE_LOOKBACK)
        highs: list[int] = []
        lows: list[int] = []
        for i in range(start, n - 2):
            if not np.isfinite(rsi[i]):
                continue
            hi = float(candles[i].high)
            lo = float(candles[i].low)
            before = candles[i - 2:i]
            after = candles[i + 1:i + 3]
            if hi >= max(float(c.high) for c in before + after):
                highs.append(i)
            if lo <= min(float(c.low) for c in before + after):
                lows.append(i)

        bullish = False
        bearish = False
        bull_meta = None
        bear_meta = None

        if len(lows) >= 2:
            a, b = lows[-2], lows[-1]
            if n - 1 - b <= cls.DIVERGENCE_RECENT_BARS:
                p1, p2 = float(candles[a].low), float(candles[b].low)
                r1, r2 = float(rsi[a]), float(rsi[b])
                bullish = p2 < p1 and r2 >= r1 + cls.DIVERGENCE_RSI_DELTA
                if bullish:
                    bull_meta = {
                        "price": [round(p1, 8), round(p2, 8)],
                        "rsi": [round(r1, 2), round(r2, 2)],
                        "bars_ago": n - 1 - b,
                    }

        if len(highs) >= 2:
            a, b = highs[-2], highs[-1]
            if n - 1 - b <= cls.DIVERGENCE_RECENT_BARS:
                p1, p2 = float(candles[a].high), float(candles[b].high)
                r1, r2 = float(rsi[a]), float(rsi[b])
                bearish = p2 > p1 and r2 <= r1 - cls.DIVERGENCE_RSI_DELTA
                if bearish:
                    bear_meta = {
                        "price": [round(p1, 8), round(p2, 8)],
                        "rsi": [round(r1, 2), round(r2, 2)],
                        "bars_ago": n - 1 - b,
                    }

        if bullish and not bearish:
            label = "BULLISH"
        elif bearish and not bullish:
            label = "BEARISH"
        elif bullish and bearish:
            label = "BOTH"
        else:
            label = "NONE"

        return {
            "bullish": bool(bullish),
            "bearish": bool(bearish),
            "label": label,
            "bullish_meta": bull_meta,
            "bearish_meta": bear_meta,
        }

    def _snapshot_15m(self, candles: list) -> dict:
        snap = super()._snapshot_15m(candles)
        if not snap.get("ready"):
            return snap

        closes = [float(c.close) for c in candles]
        rsi = self.rsi(closes, self.RSI_PERIOD)
        rsi_sma = self.sma(list(rsi), self.RSI_SMA_PERIOD)
        if not self._finite(rsi[-1], rsi[-2], rsi_sma[-1], rsi_sma[-2]):
            return snap

        curr_rsi = float(rsi[-1])
        prev_rsi = float(rsi[-2])
        curr_sma = float(rsi_sma[-1])
        prev_sma = float(rsi_sma[-2])

        rsi_slope = curr_rsi - prev_rsi
        sma_slope = curr_sma - prev_sma
        spread = curr_rsi - curr_sma
        prev_spread = prev_rsi - prev_sma
        spread_delta = spread - prev_spread

        slope_long_ok = rsi_slope > 0.0 and sma_slope >= 0.0
        slope_short_ok = rsi_slope < 0.0 and sma_slope <= 0.0
        spread_long_ok = spread > 0.0 and spread_delta > 0.0
        spread_short_ok = spread < 0.0 and spread_delta < 0.0

        divergence = self._classic_rsi_divergence(candles, rsi)
        snap.update({
            "rsi_slope": round(rsi_slope, 3),
            "rsi_sma_slope": round(sma_slope, 3),
            "spread": round(spread, 3),
            "prev_spread": round(prev_spread, 3),
            "spread_delta": round(spread_delta, 3),
            "slope_long_ok": bool(slope_long_ok),
            "slope_short_ok": bool(slope_short_ok),
            "spread_long_ok": bool(spread_long_ok),
            "spread_short_ok": bool(spread_short_ok),
            "divergence": divergence,
        })
        return snap

    def _build_entry(self, current_price: float, trend: dict, snap: dict) -> dict:
        entry = super()._build_entry(current_price, trend, snap)
        if not entry.get("trigger"):
            return entry

        direction = str(entry.get("direction") or "")
        quality_blocks: list[str] = []
        if direction == "long":
            if not snap.get("slope_long_ok"):
                quality_blocks.append("RSI_SLOPE")
            if not snap.get("spread_long_ok"):
                quality_blocks.append("RSI_SPREAD")
            div_bonus = bool((snap.get("divergence") or {}).get("bullish"))
        else:
            if not snap.get("slope_short_ok"):
                quality_blocks.append("RSI_SLOPE")
            if not snap.get("spread_short_ok"):
                quality_blocks.append("RSI_SPREAD")
            div_bonus = bool((snap.get("divergence") or {}).get("bearish"))

        entry["momentum_quality"] = {
            "rsi_slope": snap.get("rsi_slope"),
            "rsi_sma_slope": snap.get("rsi_sma_slope"),
            "spread": snap.get("spread"),
            "spread_delta": snap.get("spread_delta"),
            "divergence": (snap.get("divergence") or {}).get("label", "NONE"),
            "divergence_bonus": div_bonus,
        }

        if quality_blocks:
            entry["trigger"] = None
            entry["blocks"] = list(dict.fromkeys(list(entry.get("blocks", [])) + quality_blocks))
            entry["reason"] = "RSI cross detected but momentum quality not confirmed"
            return entry

        entry["reason"] = (
            "1H trend + RSI cross + slope/spread momentum confirmed"
            + (" + divergence bonus" if div_bonus else "")
        )
        return entry

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)
        meta = signal.metadata or {}
        meta["strategy"] = "SENTINEL_V6_1"
        meta["version"] = self.VERSION
        meta["architecture"] = "1H_EMA_DIRECTION__15M_GATE__RSI_CROSS_SLOPE_SPREAD"
        market = meta.get("market_15m") or {}
        entry = meta.get("entry_15m") or {}

        direction = str(meta.get("direction") or entry.get("direction") or "")
        divergence = market.get("divergence") or {}
        aligned_divergence = (
            (direction == "long" and divergence.get("bullish"))
            or (direction == "short" and divergence.get("bearish"))
        )
        meta["divergence_soft_bonus"] = bool(aligned_divergence)
        signal.metadata = meta

        if getattr(getattr(signal, "type", None), "value", "hold") != "hold":
            if aligned_divergence:
                signal.confidence = min(0.95, float(signal.confidence) + self.DIVERGENCE_CONF_BONUS)
            signal.reason += (
                f" | slope={market.get('rsi_slope', '-')} SMA_slope={market.get('rsi_sma_slope', '-')}"
                f" spread={market.get('spread', '-')} Δ={market.get('spread_delta', '-')}"
                f" div={divergence.get('label', 'NONE')}"
            )
        return signal
