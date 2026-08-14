"""Sentinel V3 quality architecture.

Purpose: stop location-only entries from firing against the dominant market
auction.  V1/V2 remain responsible for S/R mapping, Fib location, MCDX and
orders; this overlay is the final portfolio-quality gate.

V3 hierarchy
1. 4H Direction Gate: EMA20/50 + EMA20 slope + HMA16 slope.
2. 1H Structure Gate: structure/EMA context may be neutral, never strongly
   opposite the 4H trade direction.
3. 15M Execution Quality: require 2 of 3 (EMA8/13 alignment, HMA16 slope,
   structure/displacement confirmation).
4. Existing Sentinel location + MCDX + RR gates still apply.

This intentionally removes the old idea that touching S1/R1 plus one reversal
candle is enough.  A good location is necessary, not sufficient.
"""
from __future__ import annotations

import numpy as np
from .base import Signal, SignalType


def install_sentinel_quality_v3(strategy_cls) -> None:
    if getattr(strategy_cls, "_sentinel_quality_v3_installed", False):
        return

    original_analyze = strategy_cls.analyze
    strategy_cls._sentinel_quality_v3_installed = True
    strategy_cls.VERSION = "3.0"

    def _tf_state(self, candles: list) -> dict:
        if len(candles) < 60:
            return {"ready": False, "score": 0, "state": "WARMUP"}
        closes = [float(c.close) for c in candles]
        e20 = self.ema(closes, 20)
        e50 = self.ema(closes, 50)
        h16 = self.hma(closes, 16)
        if not all(np.isfinite(x) for x in (e20[-1], e20[-4], e50[-1], h16[-1], h16[-3])):
            return {"ready": False, "score": 0, "state": "WARMUP"}
        score = 0
        score += 1 if e20[-1] > e50[-1] else -1
        score += 1 if e20[-1] > e20[-4] else -1
        score += 1 if h16[-1] > h16[-3] else -1
        score += 1 if closes[-1] > e20[-1] else -1
        state = "BULL" if score >= 2 else "BEAR" if score <= -2 else "NEUTRAL"
        return {"ready": True, "score": score, "state": state}

    def _execution_quality(self, candles: list, side: str) -> dict:
        if len(candles) < 60:
            return {"ready": False, "passed": False, "votes": 0}
        closes = [float(c.close) for c in candles]
        e8 = self.ema(closes, 8)
        e13 = self.ema(closes, 13)
        h16 = self.hma(closes, 16)
        atr = self.atr(candles, 14)
        a = float(atr[-1]) if len(atr) and np.isfinite(atr[-1]) else 0.0
        bar = candles[-1]
        body = float(bar.close) - float(bar.open)
        structure = self._structure(candles[-100:])
        if side == "long":
            ema_vote = bool(e8[-1] > e13[-1] and closes[-1] >= e8[-1])
            hma_vote = bool(h16[-1] > h16[-3])
            pa_vote = bool(structure == "BULL" or (a > 0 and body >= 0.18*a))
        else:
            ema_vote = bool(e8[-1] < e13[-1] and closes[-1] <= e8[-1])
            hma_vote = bool(h16[-1] < h16[-3])
            pa_vote = bool(structure == "BEAR" or (a > 0 and -body >= 0.18*a))
        votes = int(ema_vote) + int(hma_vote) + int(pa_vote)
        return {
            "ready": True, "passed": votes >= 2, "votes": votes,
            "ema8_13": ema_vote, "hma16": hma_vote, "price_action": pa_vote,
            "structure": structure,
        }

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await original_analyze(self, candles, current_price, mtf_candles)
        md = dict(getattr(signal, "metadata", {}) or {})
        mtf = mtf_candles or {}
        t4 = _tf_state(self, list(mtf.get("4h") or []))
        t1 = _tf_state(self, list(mtf.get("1h") or []))
        md["sentinel_v3"] = {"trend_4h": t4, "context_1h": t1}

        if signal.type == SignalType.HOLD:
            signal.metadata = md
            return signal

        side = "long" if signal.type == SignalType.BUY else "short"
        exe = _execution_quality(self, candles, side)
        md["sentinel_v3"]["execution_15m"] = exe

        # Hard 4H direction; 1H can be neutral but cannot be strongly opposite.
        dir4_ok = t4.get("ready") and ((side == "long" and t4.get("score", 0) >= 2) or (side == "short" and t4.get("score", 0) <= -2))
        ctx1_ok = t1.get("ready") and not ((side == "long" and t1.get("score", 0) <= -2) or (side == "short" and t1.get("score", 0) >= 2))
        exe_ok = bool(exe.get("passed"))
        md["sentinel_v3"].update({"direction_pass": dir4_ok, "context_pass": ctx1_ok, "execution_pass": exe_ok})

        if dir4_ok and ctx1_ok and exe_ok:
            md["architecture"] = "SENTINEL_V3_TREND_LOCATION_EXECUTION"
            signal.metadata = md
            signal.reason = f"SENTINEL V3 PASS | 4H={t4['state']}({t4['score']:+d}) 1H={t1['state']}({t1['score']:+d}) 15M={exe['votes']}/3 | {signal.reason}"
            return signal

        # V1/V2 set internal position state before returning BUY/SELL. A veto
        # must roll it back or the strategy would think a phantom trade exists.
        self._reset_position_state()
        blockers = []
        if not dir4_ok:
            blockers.append(f"4H direction {t4.get('state')}({t4.get('score', 0):+d})")
        if not ctx1_ok:
            blockers.append(f"1H opposite {t1.get('state')}({t1.get('score', 0):+d})")
        if not exe_ok:
            blockers.append(f"15M quality {exe.get('votes', 0)}/3")
        md["sentinel_v3"]["blockers"] = blockers
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0.0,
            "SENTINEL V3 VETO | " + " ; ".join(blockers),
            confidence=0.0, metadata=md,
        )

    strategy_cls.analyze = analyze
