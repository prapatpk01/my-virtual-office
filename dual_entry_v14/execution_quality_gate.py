"""Execution Quality Gate (spec §24)."""
from __future__ import annotations

import time
from typing import Optional

from .config import Config
from .enums import ReasonCode, SetupType
from .models import GateResult, SignalCandidate, TradePlan


class ExecutionQualityGate:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def evaluate(self, symbol: str, candidate: SignalCandidate,
                       trade_plan: TradePlan, market: dict,
                       now_ms: Optional[int] = None) -> GateResult:
        c = self.cfg
        now_ms = now_ms or int(time.time() * 1000)
        codes: list = []

        if now_ms > candidate.signal_expiry:
            return GateResult(False, [ReasonCode.REJECT_SIGNAL_EXPIRED.value])

        atr = float(market.get("atr", 0.0)) or 1e-12
        last = float(market.get("last_price", candidate.entry_reference))
        spread_pct = market.get("spread_pct")
        if spread_pct is not None and spread_pct > c.max_spread_pct:
            return GateResult(False, [ReasonCode.REJECT_SPREAD.value])
        if market.get("market_closed"):
            return GateResult(False, [ReasonCode.REJECT_MARKET_CLOSED.value])

        # deviation from the entry reference — momentum rejects sooner (no chasing)
        long = candidate.direction == "LONG"
        dev = (last - candidate.entry_reference) if long else (candidate.entry_reference - last)
        dev_atr = dev / atr
        max_dev = (c.momentum_max_deviation_atr
                   if candidate.setup_type == SetupType.MOMENTUM.value
                   else c.max_entry_deviation_atr)
        if dev_atr > max_dev:
            return GateResult(False, [ReasonCode.REJECT_SLIPPAGE.value + f":dev_{dev_atr:.2f}atr"])
        # price moved in our favor is always fine (dev < 0), but momentum that
        # fell back inside the breakout must be re-checked by the caller
        if candidate.setup_type == SetupType.MOMENTUM.value and candidate.breakout_level is not None:
            back_inside = (last <= candidate.breakout_level) if long else (last >= candidate.breakout_level)
            if back_inside:
                return GateResult(False, [ReasonCode.REJECT_FALSE_BREAKOUT.value + ":back_inside_pre_fill"])

        ob_liq = market.get("orderbook_ok", True)
        if not ob_liq:
            return GateResult(False, [ReasonCode.REJECT_EXECUTION_QUALITY.value + ":thin_book"])

        return GateResult(True, codes)
