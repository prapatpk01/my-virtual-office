"""Grid Trading Strategy.

Places virtual buy/sell grid levels around a reference price.
BUY:  price crosses down into a grid buy level
SELL: price crosses up into a grid sell level
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class GridTradingStrategy(BaseStrategy):
    """Static price grid — signals when price crosses a grid boundary."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.grid_levels   = int(self.params.get("grid_levels", 5))
        self.grid_pct      = float(self.params.get("grid_pct", 0.01))   # 1% between levels
        self.lookback      = int(self.params.get("lookback", 50))
        self.position_pct  = float(self.params.get("position_pct", 0.04))
        self._grid_ref: float | None = None

    def _build_grid(self, ref_price: float) -> tuple[list, list]:
        """Return (buy_levels, sell_levels) around ref_price."""
        buy_levels  = [ref_price * (1 - self.grid_pct * (i + 1))
                       for i in range(self.grid_levels)]
        sell_levels = [ref_price * (1 + self.grid_pct * (i + 1))
                       for i in range(self.grid_levels)]
        return buy_levels, sell_levels

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        if len(candles) < 2:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        # Recompute reference as SMA of recent closes
        recent = candles[-min(self.lookback, len(candles)):]
        sma_ref = float(np.mean([c.close for c in recent]))

        buy_levels, sell_levels = self._build_grid(sma_ref)

        prev_price = float(candles[-2].close)

        # Check if current price crossed any buy level from above
        for level in buy_levels:
            if prev_price > level >= current_price:
                dist_pct = abs(current_price - sma_ref) / sma_ref * 100
                return Signal(
                    SignalType.BUY, self.symbol, current_price, self.position_pct,
                    f"[Grid] Price hit buy level {level:.2f} (ref={sma_ref:.2f})",
                    confidence=min(1.0, 0.4 + dist_pct * 0.1),
                    metadata={"grid_ref": sma_ref, "level": level},
                )

        # Check if current price crossed any sell level from below
        for level in sell_levels:
            if prev_price < level <= current_price:
                dist_pct = abs(current_price - sma_ref) / sma_ref * 100
                return Signal(
                    SignalType.SELL, self.symbol, current_price, self.position_pct,
                    f"[Grid] Price hit sell level {level:.2f} (ref={sma_ref:.2f})",
                    confidence=min(1.0, 0.4 + dist_pct * 0.1),
                    metadata={"grid_ref": sma_ref, "level": level},
                )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Grid] Price={current_price:.2f} ref={sma_ref:.2f} — between grid levels",
            metadata={"grid_ref": sma_ref},
        )
