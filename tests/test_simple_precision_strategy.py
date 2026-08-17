from dataclasses import dataclass

from trading.strategies.simple_precision_strategy import SimplePrecisionStrategy
from trading.strategies.base import SignalType


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 100.0


def trend_candles(count: int, start: float = 100.0, step: float = 0.25, tf_ms: int = 900_000):
    candles = []
    price = start
    for i in range(count):
        open_price = price
        close = price + step
        candles.append(Candle(i * tf_ms, open_price, close + 0.12, open_price - 0.12, close, 100 + i))
        price = close
    return candles


def test_4h_direction_and_1h_quality_have_separate_jobs():
    strategy = SimplePrecisionStrategy("BTC/USDT:USDT")
    macro = strategy._macro_4h(trend_candles(80, step=0.8, tf_ms=14_400_000))
    quality = strategy._quality_1h(trend_candles(80, step=0.4, tf_ms=3_600_000), "long")

    assert macro["ready"] is True
    assert macro["direction"] == "long"
    assert quality["ready"] is True
    assert quality["score"] >= quality["threshold"]


def test_same_closed_bar_cannot_emit_two_entries():
    strategy = SimplePrecisionStrategy("BTC/USDT:USDT")
    c4h = trend_candles(80, step=0.8, tf_ms=14_400_000)
    c1h = trend_candles(80, step=0.4, tf_ms=3_600_000)
    c15 = trend_candles(75, step=0.20)
    c15 = [
        Candle(c.timestamp, c.open, c.close + 1.0, c.open - 1.0, c.close, c.volume)
        for c in c15
    ]

    # Fresh, non-extended structure breakout with sufficient volume.
    previous = c15[-2]
    c15[-1] = Candle(
        c15[-1].timestamp,
        previous.close,
        previous.close + 1.9,
        previous.close - 0.2,
        previous.close + 1.7,
        200,
    )

    import asyncio
    signal1 = asyncio.run(strategy.analyze(c15, c15[-1].close, {"1h": c1h, "4h": c4h}))
    assert signal1.type == SignalType.BUY
    assert signal1.metadata["entry_trigger"] == "STRUCTURE_BREAKOUT"
    strategy.cancel_pending_entry("test")
    signal2 = asyncio.run(strategy.analyze(c15, c15[-1].close, {"1h": c1h, "4h": c4h}))

    assert signal2.type == SignalType.HOLD
    assert "already evaluated" in signal2.reason


def test_cancel_pending_entry_releases_internal_position():
    strategy = SimplePrecisionStrategy("ETH/USDT:USDT")
    strategy._open_position = "long"
    strategy._pending_entry = True
    strategy._entry_price = 100.0
    strategy.cancel_pending_entry("risk gate")
    assert strategy._open_position is None
    assert strategy._pending_entry is False
