"""Adaptive Momentum v4.1 — 15M-only, EMA8/13 cross-directed trading bot.

A fresh EMA8/13 cross is the ONLY event that chooses trade direction:
- cross up   -> LONG candidate
- cross down -> SHORT candidate
There is no separate trend filter or EMA alignment bias.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json, logging, os, sys, time

TP1_R = float(os.getenv("MOM_TP1_R", "1.0"))
TP2_R = float(os.getenv("MOM_TP2_R", "2.0"))
TP_R = TP2_R
SL_ATR = float(os.getenv("MOM_SL_ATR", "1.0"))
MIN_SL_PCT = float(os.getenv("MOM_MIN_SL_PCT", "0.004"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
ADX_MIN = float(os.getenv("MOM_ADX_MIN", "15"))
CHOP_MAX = float(os.getenv("MOM_CHOP_MAX", "55"))
COOLDOWN_BARS = int(os.getenv("MOM_COOLDOWN_BARS", "3"))
SUPPORTED_SCHEMAS = {"adaptive-momentum-v4.0-15m"}


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


class TradingBot:
    def __init__(self, symbol: str, margin_usdt: float = 20.0, leverage: int = 20,
                 paper: bool = True, state_file: str = "",
                 execution_callback: Optional[Callable] = None,
                 risk_usdt: float = RISK_USDT, **_kwargs):
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
        self.counts = {k: 0 for k in ("scans", "entries", "cooldown", "cross", "quality", "momentum", "location")}
        self._identity()
        self.load_state()

    @staticmethod
    def _identity() -> None:
        try:
            runner = sys.modules.get("run_bot") or sys.modules.get("__main__")
            if runner is not None and hasattr(runner, "logger"):
                runner.logger = logging.getLogger("adaptive_momentum_v4_1")
            if runner is not None and hasattr(runner, "BUILD_ID"):
                runner.BUILD_ID = "adaptive-momentum-v4.1-cross-directed-2026-08-17"
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
            if raw.get("position"):
                self.position = Position(**raw["position"])
            self.cooldown_remaining = int(raw.get("cooldown_remaining", 0))
        except Exception:
            self.position = None
            self.cooldown_remaining = 0

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({
                "position": asdict(self.position) if self.position else None,
                "cooldown_remaining": self.cooldown_remaining,
            }, handle)
        os.replace(tmp, self.state_file)

    @staticmethod
    def _cross_direction(i: Dict) -> str:
        if bool(i.get("ema_cross_up")):
            return "LONG"
        if bool(i.get("ema_cross_down")):
            return "SHORT"
        return "NONE"

    def _debug(self, i: Dict, result: str, reason: str, direction: str = "NONE") -> str:
        symbol = self.symbol.split("/")[0]
        adx = float(i.get("adx", 0))
        chop = float(i.get("chop", 100))
        roc = float(i.get("roc9", 0))
        hist = float(i.get("macd_hist", 0))
        mom = int(i.get("momentum_score_long" if direction == "LONG" else "momentum_score_short", 0)) if direction in {"LONG", "SHORT"} else 0
        loc = int(i.get("location_score_long" if direction == "LONG" else "location_score_short", 0)) if direction in {"LONG", "SHORT"} else 0
        structure = (
            "HL/BOS" if direction == "LONG" and i.get("structure_long")
            else "LH/BOS" if direction == "SHORT" and i.get("structure_short")
            else "-"
        )
        if reason == "COOLDOWN":
            return f"MOMENTUM V4.1 · {symbol} · 15M · ⏳ COOLDOWN {self.cooldown_remaining} bars · RESULT: WAIT"
        if reason == "CROSS":
            return f"MOMENTUM V4.1 · {symbol} · 15M · ❌ No fresh EMA8/13 cross · RESULT: WAIT CROSS"

        parts = [
            f"MOMENTUM V4.1 · {symbol} · 15M · {direction}",
            f"✅ EMA8/13 CROSS {'UP' if direction == 'LONG' else 'DOWN'} → {direction}",
        ]
        if reason == "QUALITY":
            parts.append(f"❌ Quality ADX {adx:.1f} rising={'YES' if i.get('adx_rising') else 'NO'} · CHOP {chop:.1f}")
        else:
            parts.append(f"✅ Quality ADX {adx:.1f}↑ · CHOP {chop:.1f}")
            if reason == "MOMENTUM":
                parts.append(f"❌ Momentum 0/2 · MACD Hist {hist:.4f} · ROC9 {roc:.2f}%")
            else:
                parts.append(f"✅ Momentum {mom}/2{' STRONG' if mom == 2 else ''} · MACD Hist {hist:.4f} · ROC9 {roc:.2f}%")
                if reason == "LOCATION":
                    parts.append(f"❌ Location 0/2 · BB + Structure {structure}")
                else:
                    parts.append(f"✅ Location {loc}/2 · BB + Structure {structure}")
                    if result == "ENTRY":
                        parts.append("✅ ENTRY CONFIRMED")
        labels = {
            "QUALITY": "WAIT QUALITY",
            "MOMENTUM": "WAIT MOMENTUM",
            "LOCATION": "WAIT LOCATION",
            "LONG": "ENTRY LONG",
            "SHORT": "ENTRY SHORT",
            "RISK": "WAIT RISK",
        }
        parts.append(f"RESULT: {labels.get(reason, reason)}")
        return " · ".join(parts)

    def _build(self, i: Dict, direction: str):
        entry = float(i["close"])
        atr = max(float(i["atr"]), entry * 0.0005)
        minimum = entry * MIN_SL_PCT
        if direction == "LONG":
            sl = min(float(i.get("recent_low", entry - atr)), entry - SL_ATR * atr, entry - minimum)
            risk = entry - sl
            tp1 = entry + TP1_R * risk
            tp2 = entry + TP2_R * risk
        else:
            sl = max(float(i.get("recent_high", entry + atr)), entry + SL_ATR * atr, entry + minimum)
            risk = sl - entry
            tp1 = entry - TP1_R * risk
            tp2 = entry - TP2_R * risk
        if risk <= 0:
            return None
        size = min(self.risk_usdt / risk, (self.margin_usdt * self.leverage) / max(entry, 1e-12))
        if size <= 0:
            return None
        return {
            "direction": direction,
            "strategy": "adaptive_momentum_v4_1_cross_directed",
            "trigger": f"EMA8/13 cross {'up' if direction == 'LONG' else 'down'}",
            "entry": entry,
            "sl": sl,
            "tp": tp2,
            "tp1": tp1,
            "tp2": tp2,
            "size": size,
            "risk_usdt": size * risk,
            "sl_pct": 100 * risk / max(entry, 1e-12),
            "ema8": float(i["ema8"]),
            "ema13": float(i["ema13"]),
            "macd": float(i["macd"]),
            "macd_signal": float(i["macd_signal"]),
            "macd_hist": float(i["macd_hist"]),
            "bb_mid": float(i["bb_mid"]),
            "bb_upper": float(i["bb_upper"]),
            "bb_lower": float(i["bb_lower"]),
            "atr": atr,
            "adx": float(i["adx"]),
            "chop": float(i["chop"]),
            "roc9": float(i["roc9"]),
        }

    def _close(self, price: float, reason: str):
        p = self.position
        assert p
        pnl = (price - p.entry) * p.size if p.direction == "LONG" else (p.entry - price) * p.size
        initial = abs(p.entry - p.initial_sl) * max(p.initial_size, 1e-12)
        r = pnl / initial if initial else 0.0
        payload = {
            "symbol": self.symbol, "direction": p.direction, "price": price,
            "entry": p.entry, "sl": p.sl, "tp": p.tp2, "tp1": p.tp1, "tp2": p.tp2,
            "size": p.size, "strategy": p.strategy, "trigger": p.trigger,
            "reason": reason, "pnl": pnl, "r_multiple": r,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_" + p.direction, payload)
        self.position = None
        if reason in {"EMA_CROSS_BACK", "MOMENTUM_LOST"}:
            self.cooldown_remaining = COOLDOWN_BARS
        self.save_state()
        self.last_signal = f"CLOSE {reason} pnl=${pnl:+.2f} r={r:+.2f}R"
        return {"event": "CLOSE", **payload}

    def check_price(self, price: float):
        p = self.position
        if not p:
            return None
        if (p.direction == "LONG" and price <= p.sl) or (p.direction == "SHORT" and price >= p.sl):
            return self._close(price, "BE" if p.be_moved else "SL")
        if not p.tp1_hit and ((p.direction == "LONG" and price >= p.tp1) or (p.direction == "SHORT" and price <= p.tp1)):
            qty = p.size * 0.5
            pnl = (price - p.entry) * qty if p.direction == "LONG" else (p.entry - price) * qty
            payload = {
                "symbol": self.symbol, "direction": p.direction, "price": price,
                "entry": p.entry, "size": qty, "reason": "TP1", "pnl": pnl,
                "r_multiple": TP1_R,
            }
            if self.execution_callback:
                self.execution_callback("CLOSE_PARTIAL", payload)
            p.size -= qty
            p.tp1_hit = True
            p.sl = p.entry
            p.be_moved = True
            self.save_state()
            return {"event": "PARTIAL", **payload}
        if (p.direction == "LONG" and price >= p.tp2) or (p.direction == "SHORT" and price <= p.tp2):
            return self._close(price, "TP2")
        return None

    def reconcile_flat(self, price: float, reason: str = "EXCHANGE_CLOSED"):
        return self._close(price, reason) if self.position else None

    def on_bar(self, i: Dict, _i1=None, _i4=None, price: float = 0.0):
        if not i:
            self.last_signal = "WAIT INDICATOR_WARMUP"
            return None
        if i.get("schema") not in SUPPORTED_SCHEMAS:
            raise RuntimeError(f"MOMENTUM_V41_SCHEMA_MISMATCH: {i.get('schema')}")

        if self.position:
            event = self.check_price(price or float(i["close"]))
            if event:
                return event
            p = self.position
            px = price or float(i["close"])
            if p.direction == "LONG" and i.get("ema_cross_down"):
                return self._close(px, "EMA_CROSS_BACK")
            if p.direction == "SHORT" and i.get("ema_cross_up"):
                return self._close(px, "EMA_CROSS_BACK")
            if p.direction == "LONG" and int(i.get("momentum_score_long", 0)) == 0:
                return self._close(px, "MOMENTUM_LOST")
            if p.direction == "SHORT" and int(i.get("momentum_score_short", 0)) == 0:
                return self._close(px, "MOMENTUM_LOST")
            self.last_signal = f"MANAGE {p.direction} | SL={p.sl:.4f} | TP1={p.tp1:.4f} | TP2={p.tp2:.4f}"
            return None

        self.counts["scans"] += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.counts["cooldown"] += 1
            self.save_state()
            self.last_signal = self._debug(i, "WAIT", "COOLDOWN")
            return None

        # FIRST: the current closed 15M EMA8/13 cross decides the side.
        direction = self._cross_direction(i)
        if direction == "NONE":
            self.counts["cross"] += 1
            self.last_signal = self._debug(i, "WAIT", "CROSS")
            return None

        # SECOND: quality validates whether the cross occurs in a tradable environment.
        if float(i["adx"]) < ADX_MIN or not bool(i.get("adx_rising")) or float(i["chop"]) > CHOP_MAX:
            self.counts["quality"] += 1
            self.last_signal = self._debug(i, "WAIT", "QUALITY", direction)
            return None

        # THIRD: MACD 12/26/9 or ROC9 agreeing with the cross is enough (1/2).
        momentum = int(i.get("momentum_score_long" if direction == "LONG" else "momentum_score_short", 0))
        if momentum < 1:
            self.counts["momentum"] += 1
            self.last_signal = self._debug(i, "WAIT", "MOMENTUM", direction)
            return None

        # FOURTH: Bollinger location or market structure support is enough (1/2).
        location = int(i.get("location_score_long" if direction == "LONG" else "location_score_short", 0))
        if location < 1:
            self.counts["location"] += 1
            self.last_signal = self._debug(i, "WAIT", "LOCATION", direction)
            return None

        payload = self._build(i, direction)
        if not payload:
            self.last_signal = self._debug(i, "WAIT", "RISK", direction)
            return None
        payload["symbol"] = self.symbol
        if self.execution_callback:
            self.execution_callback("OPEN_" + direction, payload)
        self.position = Position(
            direction=direction, entry=payload["entry"], sl=payload["sl"], initial_sl=payload["sl"],
            tp=payload["tp2"], tp1=payload["tp1"], tp2=payload["tp2"], size=payload["size"],
            initial_size=payload["size"], strategy=payload["strategy"], trigger=payload["trigger"],
            opened_at=time.time(),
        )
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = self._debug(i, "ENTRY", direction, direction)
        return {"event": "OPEN", **payload}
