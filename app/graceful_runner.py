"""Graceful entrypoint for Adaptive Momentum v3."""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

import run_bot
import trading.adaptive_trading_bot as strategy
from trading.connectors.binance_conn import BinanceConnector

# Keep the legacy run_bot filename for Railway compatibility, but expose only
# the current strategy name in runtime logs and Telegram metadata.
run_bot.BUILD_ID = "adaptive-momentum-v3-2026-08-06"
run_bot.logger = logging.getLogger("adaptive_momentum_v3")

logger = logging.getLogger("adaptive_momentum_v3_shutdown")
_TRACKED: list[Any] = []


def _install_readable_logs() -> None:
    """Replace the dense raw debug dump with a gate-by-gate summary."""
    bot_class = run_bot.TradingBot
    original = bot_class._debug
    if getattr(original, "_momentum_v3_readable", False):
        return

    gate_order = [
        "trend", "cross", "macd", "hist", "adx", "chop", "structure", "location"
    ]
    reason_gate = {
        "EMA20_50_TREND": "trend",
        "EMA8_13_CROSS": "cross",
        "MACD_SIGNAL": "macd",
        "MACD_HIST_2BAR_EXPANSION": "hist",
        "ADX_NOT_STRONG_RISING": "adx",
        "CHOP_TOO_HIGH": "chop",
        "STRUCTURE_BREAK": "structure",
        "LOCATION": "location",
        "COOLDOWN": "cooldown",
        "RISK_BUILD": "risk",
    }

    def readable(self: Any, i15: dict, result: str, reason: str) -> str:
        direction = "LONG" if bool(i15.get("trend_bull")) else "SHORT" if bool(i15.get("trend_bear")) else "NONE"
        failed_gate = reason_gate.get(reason)
        fail_index = gate_order.index(failed_gate) if failed_gate in gate_order else len(gate_order)

        def icon(gate: str) -> str:
            index = gate_order.index(gate)
            if result == "ENTRY":
                return "✅"
            if failed_gate == gate:
                return "❌"
            if index < fail_index:
                return "✅"
            return "➖"

        cross_side = "UP" if direction == "LONG" else "DOWN"
        macd_side = "ABOVE" if direction == "LONG" else "BELOW"
        structure_level = float(i15.get("recent_high", 0)) if direction == "LONG" else float(i15.get("recent_low", 0))
        location = float(i15.get("distance_ema13_atr", 0))
        adx = float(i15.get("adx", 0))
        chop = float(i15.get("chop", 0))

        details = {
            "trend": f"EMA20 {'>' if direction == 'LONG' else '<' if direction == 'SHORT' else '?'} EMA50",
            "cross": f"EMA8/13 fresh cross {cross_side}",
            "macd": f"MACD {macd_side} Signal",
            "hist": "Histogram expanding 2 bars",
            "adx": f"ADX {adx:.1f} (need ≥{strategy.ADX_MIN:g} and rising)",
            "chop": f"CHOP {chop:.1f} (need ≤{strategy.CHOP_MAX:g})",
            "structure": f"Structure break @ {structure_level:.6f}",
            "location": f"Distance {location:.2f} ATR (max {strategy.LOCATION_MAX_ATR:g})",
        }
        labels = {
            "trend": "TREND", "cross": "EMA8/13", "macd": "MACD",
            "hist": "HISTOGRAM", "adx": "ADX", "chop": "CHOP",
            "structure": "STRUCTURE", "location": "LOCATION",
        }
        lines = [
            f"MOMENTUM V3 | {self.symbol} | 15M | Direction: {direction}",
        ]
        for gate in gate_order:
            lines.append(f"{icon(gate)} {labels[gate]} — {details[gate]}")

        if reason == "COOLDOWN":
            lines.append(f"⏳ COOLDOWN — {self.cooldown_remaining} bars remaining")
        elif reason == "RISK_BUILD":
            lines.append("❌ RISK — unable to build valid SL/size")

        readable_reason = {
            "EMA20_50_TREND": "WAIT TREND",
            "EMA8_13_CROSS": "WAIT EMA CROSS",
            "MACD_SIGNAL": "WAIT MACD",
            "MACD_HIST_2BAR_EXPANSION": "WAIT HISTOGRAM EXPANSION",
            "ADX_NOT_STRONG_RISING": "WAIT ADX",
            "CHOP_TOO_HIGH": "WAIT CHOP",
            "STRUCTURE_BREAK": "WAIT STRUCTURE BREAK",
            "LOCATION": "WAIT LOCATION",
            "COOLDOWN": "WAIT COOLDOWN",
            "RISK_BUILD": "WAIT RISK BUILD",
            "LONG": "ENTRY LONG",
            "SHORT": "ENTRY SHORT",
        }.get(reason, f"{result} {reason}")
        lines.append(f"RESULT: {readable_reason}")
        return "\n".join(lines)

    readable._momentum_v3_readable = True  # type: ignore[attr-defined]
    bot_class._debug = readable
    logger.info("Installed Momentum v3 gate-by-gate readable logs")


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
    _install_readable_logs()
    _track_class(BinanceConnector)

    try:
        from trading.connectors.okx_adapter import OKXAdapter
        _track_class(OKXAdapter)
    except Exception:
        pass

    logger.info("Adaptive Momentum v3 graceful runner started | build=%s", run_bot.BUILD_ID)
    try:
        await run_bot.main()
    finally:
        for resource in reversed(_TRACKED):
            await _close_resource(resource)
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
