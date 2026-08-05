"""Graceful entrypoint for Adaptive SMC v15.

Besides closing CCXT/AIOHTTP resources cleanly, this entrypoint adds a very
light 15M market-quality filter without changing the SMC decision chain:
ADX14 < 12 AND CHOP14 > 65 blocks only new entries. Existing positions keep
normal SL/TP management.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
from typing import Any

import run_bot
from trading.connectors.binance_conn import BinanceConnector

logger = logging.getLogger("adaptive_smc_shutdown")
_TRACKED: list[Any] = []

ADX_MIN = float(os.getenv("V15_SOFT_ADX_MIN", "12"))
CHOP_MAX = float(os.getenv("V15_SOFT_CHOP_MAX", "65"))
QUALITY_FILTER_ENABLED = os.getenv("V15_SOFT_FILTER_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}


def _value(candle: Any, name: str, index: int) -> float:
    value = getattr(candle, name, None)
    if value is None and isinstance(candle, dict):
        value = candle.get(name)
    if value is None and isinstance(candle, (list, tuple)) and len(candle) > index:
        value = candle[index]
    return float(value or 0.0)


def _wilder(values: list[float], length: int) -> list[float]:
    if not values:
        return []
    output = [float(values[0])]
    alpha = 1.0 / max(length, 1)
    for value in values[1:]:
        output.append(output[-1] + alpha * (float(value) - output[-1]))
    return output


def _adx_chop(candles: list[Any], length: int = 14) -> tuple[float, float]:
    if len(candles) < length * 3:
        return 0.0, 100.0

    highs = [_value(c, "high", 2) for c in candles]
    lows = [_value(c, "low", 3) for c in candles]
    closes = [_value(c, "close", 4) for c in candles]

    tr: list[float] = [max(highs[0] - lows[0], 0.0)]
    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for index in range(1, len(candles)):
        up = highs[index] - highs[index - 1]
        down = lows[index - 1] - lows[index]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr.append(max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        ))

    atr_series = _wilder(tr, length)
    plus_series = _wilder(plus_dm, length)
    minus_series = _wilder(minus_dm, length)
    dx: list[float] = []
    for atr_value, plus_value, minus_value in zip(atr_series, plus_series, minus_series):
        if atr_value <= 1e-12:
            dx.append(0.0)
            continue
        plus_di = 100.0 * plus_value / atr_value
        minus_di = 100.0 * minus_value / atr_value
        denominator = plus_di + minus_di
        dx.append(100.0 * abs(plus_di - minus_di) / denominator if denominator > 1e-12 else 0.0)
    adx = _wilder(dx, length)[-1]

    tr_sum = sum(tr[-length:])
    highest = max(highs[-length:])
    lowest = min(lows[-length:])
    price_range = highest - lowest
    if tr_sum <= 0 or price_range <= 1e-12:
        chop = 100.0
    else:
        ratio = max(tr_sum / price_range, 1.0)
        chop = 100.0 * math.log10(ratio) / math.log10(length)
        chop = min(100.0, max(0.0, chop))
    return float(adx), float(chop)


def _install_soft_filter() -> None:
    original_compute = run_bot.compute

    def compute_with_quality(candles: list[Any]):
        result = original_compute(candles)
        if result:
            adx, chop = _adx_chop(candles, 14)
            result["adx"] = adx
            result["chop"] = chop
        return result

    run_bot.compute = compute_with_quality

    bot_class = run_bot.TradingBot
    original_on_bar = bot_class.on_bar
    if getattr(original_on_bar, "_v15_soft_filter_wrapped", False):
        return

    def on_bar_with_quality(self: Any, i15: dict, *args: Any, **kwargs: Any):
        # Never interfere with management of an already-open position.
        if QUALITY_FILTER_ENABLED and i15 and not self.position_open:
            adx = float(i15.get("adx", 0.0))
            chop = float(i15.get("chop", 100.0))
            if adx < ADX_MIN and chop > CHOP_MAX:
                if getattr(self, "setup", None) is not None:
                    self._reset("ADX_CHOP_DEAD_MARKET")
                counts = getattr(self, "counts", None)
                if isinstance(counts, dict):
                    counts["soft_filter"] = int(counts.get("soft_filter", 0)) + 1
                self.last_signal = (
                    f"SMC15 symbol={self.symbol} | QUALITY[ADX={adx:.1f}/{ADX_MIN:g},"
                    f"CHOP={chop:.1f}/{CHOP_MAX:g}] | RESULT[WAIT:DEAD_MARKET_SOFT_FILTER]"
                )
                return None
        return original_on_bar(self, i15, *args, **kwargs)

    on_bar_with_quality._v15_soft_filter_wrapped = True  # type: ignore[attr-defined]
    bot_class.on_bar = on_bar_with_quality
    logger.info(
        "SMC v15 soft filter enabled=%s: block new entries only when ADX<%.1f AND CHOP>%.1f",
        QUALITY_FILTER_ENABLED, ADX_MIN, CHOP_MAX,
    )


def _track_class(cls: type) -> None:
    original = cls.__init__
    if getattr(original, "_adaptive_shutdown_wrapped", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _TRACKED.append(self)

    wrapped._adaptive_shutdown_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = wrapped  # type: ignore[method-assign]


async def _close_target(target: Any, label: str) -> bool:
    if target is None:
        return False
    close = getattr(target, "close", None)
    if not callable(close):
        return False
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
        logger.info("Closed %s", label)
        return True
    except Exception as exc:
        logger.warning("Failed to close %s: %s", label, exc)
        return False


async def _close_resource(resource: Any) -> None:
    seen: set[int] = set()
    candidates = (
        (resource, type(resource).__name__),
        (getattr(resource, "_exchange", None), f"{type(resource).__name__}._exchange"),
        (getattr(resource, "exchange", None), f"{type(resource).__name__}.exchange"),
        (getattr(resource, "client", None), f"{type(resource).__name__}.client"),
    )
    for target, label in candidates:
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        if await _close_target(target, label):
            break


async def main() -> None:
    _install_soft_filter()
    _track_class(BinanceConnector)

    try:
        from trading.connectors.okx_adapter import OKXAdapter
        _track_class(OKXAdapter)
    except Exception:
        pass

    try:
        await run_bot.main()
    finally:
        for resource in reversed(_TRACKED):
            await _close_resource(resource)
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
