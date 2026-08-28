"""Adaptive Trading Bot V5.2 — 15M Dual Engine.

ACTIVE ENTRIES
- LEGACY_MOMENTUM: EMA8/13 directional bias + BOS/CHOCH + ADX/CHOP quality
  + 2-of-3 MACD/ROC9/structure confirmation.
- BREAKOUT: volatility expansion + local swing break + ROC9. Breakout has priority.

TREND and MEAN no longer open new positions. Existing positions from older versions
are preserved and continue to be managed safely after deployment.

Both active engines use TP1 1R (close 50%, lock BE+0.15R), TP2 2R (close 25%),
then a 25% runner. Legacy Momentum runners exit on EMA8/13 flip; Breakout runners
use a one-ATR trailing stop on closed 15M bars.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, Optional, Tuple
import json
import logging
import os
import sys
import time

TP1_R = float(os.getenv("V5_TP1_R", "1.0"))
TP2_R = float(os.getenv("V5_TP2_R", "2.0"))
TP_R = TP2_R
BE_LOCK_R = float(os.getenv("V5_BE_LOCK_R", "0.15"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
MIN_SL_PCT = float(os.getenv("MOM_MIN_SL_PCT", "0.004"))
COOLDOWN_BARS = int(os.getenv("V5_COOLDOWN_BARS", "1"))
SUPPORTED_SCHEMAS = {"adaptive-v5-three-style-15m"}


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
        self.counts = {key: 0 for key in (
            "scans", "entries", "cooldown", "wait_regime", "legacy_momentum", "breakout"
        )}
        self._identity()
        self.load_state()

    @staticmethod
    def _identity() -> None:
        try:
            runner = sys.modules.get("run_bot") or sys.modules.get("__main__")
            if runner is not None and hasattr(runner, "logger"):
                runner.logger = logging.getLogger("adaptive_trading_v5_2")
            if runner is not None and hasattr(runner, "BUILD_ID"):
                runner.BUILD_ID = "adaptive-v5.2-dual-engine-2026-08-28"
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

    @staticmethod
    def _entry_candidate(i: Dict) -> Tuple[str, str, str]:
        regime = str(i.get("regime", "WAIT"))

        # Breakout receives first priority from the indicator router.
        if regime == "BREAKOUT":
            if i.get("breakout_long"):
                return "LONG", "BREAKOUT", "Swing breakout + volatility expansion"
            if i.get("breakout_short"):
                return "SHORT", "BREAKOUT", "Swing breakdown + volatility expansion"

        if regime == "LEGACY_MOMENTUM":
            if i.get("legacy_long"):
                trigger = "Bullish BOS" if i.get("bullish_bos") else "Bullish CHOCH"
                return "LONG", "LEGACY_MOMENTUM", trigger
            if i.get("legacy_short"):
                trigger = "Bearish BOS" if i.get("bearish_bos") else "Bearish CHOCH"
                return "SHORT", "LEGACY_MOMENTUM", trigger

        return "NONE", regime, ""

    def _debug(self, i: Dict, reason: str) -> str:
        symbol = self.symbol.split("/")[0]
        regime = str(i.get("regime", "WAIT"))
        adx = float(i.get("adx", 0.0))
        chop = float(i.get("chop", 100.0))
        if reason == "COOLDOWN":
            return f"ADAPTIVE V5.2 · {symbol} · 15M · ⏳ COOLDOWN {self.cooldown_remaining} bar · WAIT"
        if regime == "BREAKOUT":
            return (
                f"ADAPTIVE V5.2 · {symbol} · BREAKOUT · ROC9 {float(i.get('roc9',0)):+.2f}% · "
                f"BBexpand={int(bool(i.get('bb_expanding')))} ATRrise={int(bool(i.get('atr_rising')))} · WAIT BREAK"
            )
        if regime == "LEGACY_MOMENTUM":
            score = max(int(i.get("legacy_long_score", 0)), int(i.get("legacy_short_score", 0)))
            return (
                f"ADAPTIVE V5.2 · {symbol} · LEGACY MOMENTUM · ADX {adx:.1f} · CHOP {chop:.1f} · "
                f"confirm={score}/3 · WAIT STRUCTURE"
            )
        return f"ADAPTIVE V5.2 · {symbol} · WAIT · ADX {adx:.1f} · CHOP {chop:.1f} · NO TRADE"

    def _build(self, i: Dict, direction: str, style: str, trigger: str):
        entry = float(i["close"])
        atr = max(float(i["atr"]), entry * 0.0005)
        min_risk = entry * MIN_SL_PCT

        if style == "BREAKOUT":
            if direction == "LONG":
                candidate = float(i["swing_high"]) - 0.30 * atr
                sl = max(candidate, entry - 1.20 * atr)
                sl = min(sl, entry - max(0.50 * atr, min_risk))
                risk = entry - sl
                tp1, tp2 = entry + TP1_R * risk, entry + TP2_R * risk
            else:
                candidate = float(i["swing_low"]) + 0.30 * atr
                sl = min(candidate, entry + 1.20 * atr)
                sl = max(sl, entry + max(0.50 * atr, min_risk))
                risk = sl - entry
                tp1, tp2 = entry - TP1_R * risk, entry - TP2_R * risk
        else:  # LEGACY_MOMENTUM
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

        score = int(i.get("legacy_long_score", 0) if direction == "LONG" else i.get("legacy_short_score", 0))
        return {
            "direction": direction,
            "style": style,
            "strategy": "adaptive_v5_2_dual_engine",
            "trigger": trigger,
            "entry": entry,
            "sl": sl,
            "tp": tp2,
            "tp1": tp1,
            "tp2": tp2,
            "size": size,
            "risk_usdt": size * risk,
            "sl_pct": 100.0 * risk / max(entry, 1e-12),
            "ema8": float(i["ema8"]),
            "ema13": float(i["ema13"]),
            "macd": float(i["macd"]),
            "macd_signal": float(i["macd_signal"]),
            "rsi14": float(i["rsi14"]),
            "roc9": float(i["roc9"]),
            "bb_mid": float(i["bb_mid"]),
            "bb_upper": float(i["bb_upper"]),
            "bb_lower": float(i["bb_lower"]),
            "atr": atr,
            "adx": float(i["adx"]),
            "chop": float(i["chop"]),
            "confirmation_score": score,
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
        self.last_signal = f"CLOSE {p.style} {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R"
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
            reason = "LOCKED_SL" if p.be_moved else "SL"
            return self._close(price, reason)

        # Preserve management for an old MEAN position if one survived deployment.
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
            raise RuntimeError(f"ADAPTIVE_V5_2_SCHEMA_MISMATCH: {i.get('schema')}")

        if self.position:
            event = self.check_price(price or float(i["close"]))
            if event:
                return event

            p = self.position
            if p.runner_active:
                # New Legacy Momentum and old LEGACY/TREND positions use the same
                # clean EMA8/13 runner exit. This also fixes old LEGACY runners
                # that previously had no closed-bar trend exit in V5.
                if p.style in {"LEGACY_MOMENTUM", "LEGACY", "TREND"}:
                    if p.direction == "LONG" and i.get("ema_bear"):
                        return self._close(float(i["close"]), "MOMENTUM_RUNNER_EMA_EXIT")
                    if p.direction == "SHORT" and i.get("ema_bull"):
                        return self._close(float(i["close"]), "MOMENTUM_RUNNER_EMA_EXIT")
                elif p.style == "BREAKOUT":
                    atr = float(i["atr"])
                    if p.direction == "LONG":
                        p.sl = max(p.sl, float(i["close"]) - atr)
                    else:
                        p.sl = min(p.sl, float(i["close"]) + atr)
                    self.save_state()

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

        direction, style, trigger = self._entry_candidate(i)
        if direction == "NONE":
            if style == "WAIT":
                self.counts["wait_regime"] += 1
            elif style.lower() in self.counts:
                self.counts[style.lower()] += 1
            self.last_signal = self._debug(i, "WAIT_SETUP")
            return None

        payload = self._build(i, direction, style, trigger)
        if not payload:
            self.last_signal = f"ADAPTIVE V5.2 · {self.symbol.split('/')[0]} · {style} · WAIT RISK BUILD"
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
            style=style,
        )
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = f"ENTRY {style} {direction} · {trigger}"
        return {"event": "OPEN", **payload}
