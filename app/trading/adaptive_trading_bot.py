"""Simple Structure Trading Bot V6 — one 15M strategy, no router.

ENTRY
- Direction: price vs EMA20 + EMA20 slope
- Trigger: 5-bar BOS OR one-candle EMA20 pullback continuation
- Momentum: RSI14 on the correct side of 50
- Anti-chase: entry within 1.5 ATR of EMA20

RISK
- ATR/structure stop
- TP1 1R close 50%, move stop to BE+0.10R
- TP2 2R close 25%
- Runner 25%, exit on a closed-bar EMA20 failure

Existing positions from older versions are preserved and managed after deployment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, Optional
import json
import logging
import os
import sys
import time

TP1_R = float(os.getenv("V6_TP1_R", "1.0"))
TP2_R = float(os.getenv("V6_TP2_R", "2.0"))
BE_LOCK_R = float(os.getenv("V6_BE_LOCK_R", "0.10"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
MIN_SL_PCT = float(os.getenv("MOM_MIN_SL_PCT", "0.004"))
COOLDOWN_BARS = int(os.getenv("V6_COOLDOWN_BARS", "1"))
SUPPORTED_SCHEMAS = {"simple-v6-structure-15m"}


@dataclass
class Position:
    direction: str
    entry: float
    sl: float
    initial_sl: float
    tp: float
    tp1: float
    tp2: float
    size: float
    initial_size: float
    strategy: str
    trigger: str
    opened_at: float
    tp1_hit: bool = False
    be_moved: bool = False
    style: str = "LEGACY"
    tp2_hit: bool = False
    runner_active: bool = False


class TradingBot:
    def __init__(
        self,
        symbol: str,
        margin_usdt: float = 20.0,
        leverage: int = 20,
        paper: bool = True,
        state_file: str = "",
        execution_callback: Optional[Callable] = None,
        risk_usdt: float = RISK_USDT,
        **_kwargs,
    ):
        self.symbol = symbol
        self.margin_usdt = float(margin_usdt)
        self.leverage = int(leverage)
        self.paper = bool(paper)
        self.state_file = state_file
        self.execution_callback = execution_callback
        self.risk_usdt = float(risk_usdt)
        self.position: Optional[Position] = None
        self.cooldown_remaining = 0
        self.last_signal = "WARMUP"
        self.counts = {"scans": 0, "entries": 0, "cooldown": 0, "wait": 0}
        self._identity()
        self.load_state()

    @staticmethod
    def _identity() -> None:
        try:
            runner = sys.modules.get("run_bot") or sys.modules.get("__main__")
            if runner is not None and hasattr(runner, "logger"):
                runner.logger = logging.getLogger("simple_structure_v6")
            if runner is not None and hasattr(runner, "BUILD_ID"):
                runner.BUILD_ID = "simple-structure-v6-2026-08-28"
        except Exception:
            pass

    @property
    def position_open(self) -> bool:
        return self.position is not None

    def load_state(self) -> None:
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, encoding="utf-8") as handle:
                raw = json.load(handle)
            position_raw = raw.get("position")
            if position_raw:
                allowed = {field.name for field in fields(Position)}
                clean = {key: value for key, value in position_raw.items() if key in allowed}
                self.position = Position(**clean)
            self.cooldown_remaining = int(raw.get("cooldown_remaining", 0))
        except Exception:
            self.position = None
            self.cooldown_remaining = 0

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({
                "position": asdict(self.position) if self.position else None,
                "cooldown_remaining": self.cooldown_remaining,
            }, handle)
        os.replace(temp, self.state_file)

    def _debug(self, i: Dict, reason: str) -> str:
        symbol = self.symbol.split("/")[0]
        if reason == "COOLDOWN":
            return f"V6 · {symbol} · COOLDOWN {self.cooldown_remaining} bar · WAIT"
        bias = "LONG" if i.get("bias_long") else "SHORT" if i.get("bias_short") else "FLAT"
        return (
            f"V6 · {symbol} · bias={bias} · RSI {float(i.get('rsi14', 50)):.1f} · "
            f"dist={float(i.get('distance_atr', 0)):.2f}ATR · WAIT"
        )

    def _build(self, i: Dict, direction: str, trigger: str):
        entry = float(i["close"])
        atr = max(float(i["atr"]), entry * 0.0005)
        min_risk = max(0.50 * atr, entry * MIN_SL_PCT)

        if direction == "LONG":
            structure_stop = float(i["recent_low"]) - 0.10 * atr
            sl = max(structure_stop, entry - 1.20 * atr)
            sl = min(sl, entry - min_risk)
            risk = entry - sl
            tp1, tp2 = entry + TP1_R * risk, entry + TP2_R * risk
        else:
            structure_stop = float(i["recent_high"]) + 0.10 * atr
            sl = min(structure_stop, entry + 1.20 * atr)
            sl = max(sl, entry + min_risk)
            risk = sl - entry
            tp1, tp2 = entry - TP1_R * risk, entry - TP2_R * risk

        if risk <= 0:
            return None
        size = min(
            self.risk_usdt / risk,
            (self.margin_usdt * self.leverage) / max(entry, 1e-12),
        )
        if size <= 0:
            return None

        return {
            "direction": direction,
            "style": "STRUCTURE",
            "strategy": "simple_structure_v6",
            "trigger": trigger,
            "entry": entry,
            "sl": sl,
            "tp": tp2,
            "tp1": tp1,
            "tp2": tp2,
            "size": size,
            "risk_usdt": size * risk,
            "sl_pct": 100.0 * risk / max(entry, 1e-12),
            "ema20": float(i["ema20"]),
            "rsi14": float(i["rsi14"]),
            "atr": atr,
            "distance_atr": float(i.get("distance_atr", 0.0)),
        }

    def _close(self, price: float, reason: str):
        p = self.position
        assert p
        pnl = (price - p.entry) * p.size if p.direction == "LONG" else (p.entry - price) * p.size
        initial_risk = abs(p.entry - p.initial_sl) * max(p.initial_size, 1e-12)
        r_multiple = pnl / initial_risk if initial_risk else 0.0
        payload = {
            "symbol": self.symbol,
            "direction": p.direction,
            "style": p.style,
            "price": price,
            "entry": p.entry,
            "sl": p.sl,
            "tp": p.tp2,
            "tp1": p.tp1,
            "tp2": p.tp2,
            "size": p.size,
            "strategy": p.strategy,
            "trigger": p.trigger,
            "reason": reason,
            "pnl": pnl,
            "r_multiple": r_multiple,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_" + p.direction, payload)
        self.position = None
        if reason == "SL":
            self.cooldown_remaining = COOLDOWN_BARS
        self.save_state()
        self.last_signal = f"CLOSE {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R"
        return {"event": "CLOSE", **payload}

    def _partial(self, price: float, qty: float, reason: str, r_multiple: float):
        p = self.position
        assert p
        pnl = (price - p.entry) * qty if p.direction == "LONG" else (p.entry - price) * qty
        payload = {
            "symbol": self.symbol,
            "direction": p.direction,
            "style": p.style,
            "price": price,
            "entry": p.entry,
            "size": qty,
            "trigger": p.trigger,
            "reason": reason,
            "pnl": pnl,
            "r_multiple": r_multiple,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_PARTIAL", payload)
        return {"event": "PARTIAL", **payload}

    def check_price(self, price: float):
        p = self.position
        if not p:
            return None

        if (p.direction == "LONG" and price <= p.sl) or (p.direction == "SHORT" and price >= p.sl):
            return self._close(price, "LOCKED_SL" if p.be_moved else "SL")

        # Preserve a legacy mean target if one survived from an older deployment.
        if p.style == "MEAN":
            if (p.direction == "LONG" and price >= p.tp) or (p.direction == "SHORT" and price <= p.tp):
                return self._close(price, "MEAN_TARGET")
            return None

        if not p.tp1_hit and (
            (p.direction == "LONG" and price >= p.tp1)
            or (p.direction == "SHORT" and price <= p.tp1)
        ):
            qty = min(p.initial_size * 0.50, p.size)
            event = self._partial(price, qty, "TP1", TP1_R)
            p.size -= qty
            p.tp1_hit = True
            risk_unit = abs(p.entry - p.initial_sl)
            lock = BE_LOCK_R * risk_unit
            p.sl = p.entry + lock if p.direction == "LONG" else p.entry - lock
            p.be_moved = True
            self.save_state()
            return event

        if p.tp1_hit and not p.tp2_hit and (
            (p.direction == "LONG" and price >= p.tp2)
            or (p.direction == "SHORT" and price <= p.tp2)
        ):
            qty = min(p.initial_size * 0.25, p.size)
            event = self._partial(price, qty, "TP2_PARTIAL", TP2_R)
            p.size -= qty
            p.tp2_hit = True
            p.runner_active = p.size > 1e-12
            risk_unit = abs(p.entry - p.initial_sl)
            p.sl = p.entry + risk_unit if p.direction == "LONG" else p.entry - risk_unit
            p.be_moved = True
            self.save_state()
            if not p.runner_active:
                return self._close(price, "TP2")
            return event

        return None

    def reconcile_flat(self, price: float, reason: str = "EXCHANGE_CLOSED"):
        return self._close(price, reason) if self.position else None

    def on_bar(self, i: Dict, _i1=None, _i4=None, price: float = 0.0):
        if not i:
            self.last_signal = "WAIT INDICATOR_WARMUP"
            return None
        if i.get("schema") not in SUPPORTED_SCHEMAS:
            raise RuntimeError(f"SIMPLE_V6_SCHEMA_MISMATCH: {i.get('schema')}")

        if self.position:
            event = self.check_price(price or float(i["close"]))
            if event:
                return event

            p = self.position
            if p.runner_active:
                # Closed-bar EMA20 failure ends the runner. This applies to both
                # V6 and preserved older positions for simple, deterministic management.
                if p.direction == "LONG" and float(i["close"]) < float(i["ema20"]):
                    return self._close(float(i["close"]), "RUNNER_EMA20_EXIT")
                if p.direction == "SHORT" and float(i["close"]) > float(i["ema20"]):
                    return self._close(float(i["close"]), "RUNNER_EMA20_EXIT")

            self.last_signal = (
                f"MANAGE {p.style} {p.direction} | SL={p.sl:.4f} | "
                f"TP1={p.tp1:.4f} | TP2={p.tp2:.4f} | Runner={int(p.runner_active)}"
            )
            return None

        self.counts["scans"] += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.counts["cooldown"] += 1
            self.save_state()
            self.last_signal = self._debug(i, "COOLDOWN")
            return None

        direction = "LONG" if i.get("long_signal") else "SHORT" if i.get("short_signal") else "NONE"
        if direction == "NONE":
            self.counts["wait"] += 1
            self.last_signal = self._debug(i, "WAIT")
            return None

        trigger = str(i.get("trigger") or "Structure continuation")
        payload = self._build(i, direction, trigger)
        if not payload:
            self.last_signal = f"V6 · {self.symbol.split('/')[0]} · WAIT RISK BUILD"
            return None

        payload["symbol"] = self.symbol
        if self.execution_callback:
            self.execution_callback("OPEN_" + direction, payload)

        self.position = Position(
            direction=direction,
            entry=payload["entry"],
            sl=payload["sl"],
            initial_sl=payload["sl"],
            tp=payload["tp2"],
            tp1=payload["tp1"],
            tp2=payload["tp2"],
            size=payload["size"],
            initial_size=payload["size"],
            strategy=payload["strategy"],
            trigger=payload["trigger"],
            opened_at=time.time(),
            style="STRUCTURE",
        )
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = f"ENTRY {direction} · {trigger}"
        return {"event": "OPEN", **payload}
