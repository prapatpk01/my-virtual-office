"""Sentinel V4.1 — RSI14/SMA14 + 1.5R minimum target policy.

This is a narrow production refinement of Sentinel V4:
- Keep the same 15M price-action setup engine and position management.
- Use RSI(14) with SMA(14) of RSI, matching Sentinel X v5.6.2.
- Allow trades only when the first usable target / opposing S/R leaves >= 1.5R.
- Base target remains 2.0R; 0.8R and 1.2R remain blocked, while 1.5R+ is valid.
"""
from __future__ import annotations

from .sentinel_v4_strategy import SentinelV4Strategy


class SentinelV41Strategy(SentinelV4Strategy):
    """Sentinel V4 with corrected RSI baseline and 1.5R minimum room."""

    VERSION = "4.1"
    MIN_TARGET_R = 1.50

    def __init__(self, symbol: str, **kwargs):
        # The 1.5R floor is an intentional production rule rather than a
        # legacy environment default, so keep it consistent across deployments.
        kwargs["min_room_r"] = 1.50
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV4.1({symbol})"
        self.min_room_r = 1.50

    def _market_gate(self, candles: list) -> dict:
        market = super()._market_gate(candles)

        # Sentinel X v5.6.2 lower pane uses RSI(14) + SMA(14) of RSI.
        # RSI/SMA remains soft context only; it does not become an entry gate.
        closes = [float(c.close) for c in candles]
        rsi = self.rsi(closes, 14)
        rsi_sma = self.sma(list(rsi), 14)
        if not self._finite(rsi[-1], rsi_sma[-1]):
            market["ready"] = False
            blocks = list(market.get("blocks", []))
            if "INDICATORS" not in blocks:
                blocks.append("INDICATORS")
            market["blocks"] = blocks
            market["reason"] = "15M RSI14/SMA14 unavailable"
            return market

        market["rsi"] = round(float(rsi[-1]), 1)
        market["rsi_sma"] = round(float(rsi_sma[-1]), 1)
        return market
