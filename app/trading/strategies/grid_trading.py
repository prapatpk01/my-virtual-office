"""Grid Trading strategy — profit from sideways markets."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .base import BaseStrategy, Signal, SignalType


@dataclass
class GridLevel:
    price: float
    order_id: Optional[str] = None
    filled: bool = False
    side: str = "buy"


class GridTradingStrategy(BaseStrategy):
    """
    Creates a grid of buy/sell orders between lower and upper price bounds.
    Each grid level alternates buy/sell; profit is captured on each bounce.

    Auto-range mode: if lower/upper not provided, uses Bollinger Bands
    from recent candles to set the grid range automatically.
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.grid_count = self.params.get("grid_count", 10)
        self.lower = self.params.get("lower", None)
        self.upper = self.params.get("upper", None)
        self.amount_per_grid = self.params.get("amount_per_grid", 0.01)
        self.levels: list[GridLevel] = []
        self._initialized = False

    def _build_grid(self, lower: float, upper: float) -> list[GridLevel]:
        step = (upper - lower) / self.grid_count
        levels = []
        for i in range(self.grid_count + 1):
            price = lower + i * step
            side = "buy" if i < self.grid_count // 2 else "sell"
            levels.append(GridLevel(price=round(price, 6), side=side))
        return levels

    async def analyze(self, candles: list, current_price: float) -> Signal:
        closes = [c.close for c in candles]

        # Auto-determine range if not set
        if not self._initialized:
            if self.lower is None or self.upper is None:
                if len(closes) < 20:
                    return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Waiting for BB data")
                upper_bb, _, lower_bb = self.bollinger_bands(closes, period=20, std_dev=1.5)
                self.upper = float(upper_bb[-1])
                self.lower = float(lower_bb[-1])
            self.levels = self._build_grid(self.lower, self.upper)
            self._initialized = True

        if not (self.lower <= current_price <= self.upper):
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"Price {current_price:.4f} outside grid [{self.lower:.4f}, {self.upper:.4f}]",
            )

        # Find the nearest unfilled level on each side
        buy_levels = [l for l in self.levels if l.side == "buy" and not l.filled]
        sell_levels = [l for l in self.levels if l.side == "sell" and not l.filled]

        nearest_buy = min(buy_levels, key=lambda l: abs(l.price - current_price), default=None)
        nearest_sell = min(sell_levels, key=lambda l: abs(l.price - current_price), default=None)

        if nearest_buy and abs(nearest_buy.price - current_price) / current_price < 0.005:
            nearest_buy.filled = True
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=nearest_buy.price,
                amount=self.amount_per_grid,
                reason=f"Grid BUY at level {nearest_buy.price:.4f}",
                confidence=0.8,
                metadata={"grid_level": nearest_buy.price, "grid_lower": self.lower, "grid_upper": self.upper},
            )

        if nearest_sell and abs(nearest_sell.price - current_price) / current_price < 0.005:
            nearest_sell.filled = True
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=nearest_sell.price,
                amount=self.amount_per_grid,
                reason=f"Grid SELL at level {nearest_sell.price:.4f}",
                confidence=0.8,
                metadata={"grid_level": nearest_sell.price, "grid_lower": self.lower, "grid_upper": self.upper},
            )

        filled_count = sum(1 for l in self.levels if l.filled)
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"Grid active [{self.lower:.2f}–{self.upper:.2f}], {filled_count}/{len(self.levels)} filled",
            metadata={"grid_lower": self.lower, "grid_upper": self.upper, "levels": len(self.levels)},
        )
