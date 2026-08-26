"""Adaptive Momentum v4.3 — 15M-only dual-entry trading bot.

Entry engines:
- Momentum: BOS/CHOCH + EMA8/13 alignment.
- Pullback: EMA13 reclaim/reject + EMA8/13 alignment.

ADX+CHOP is the quality gate. MACD, ROC9, Bollinger location and structure form
a 4-point confirmation score. Momentum weakness alone never closes a position;
early exit requires momentum loss plus structure invalidation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple
import json, logging, os, sys, time

TP1_R = float(os.getenv("MOM_TP1_R", "1.0"))
TP2_R = float(os.getenv("MOM_TP2_R", "2.0"))
TP_R = TP2_R
SL_ATR = float(os.getenv("MOM_SL_ATR", "1.0"))
MIN_SL_PCT = float(os.getenv("MOM_MIN_SL_PCT", "0.004"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
ADX_MIN = float(os.getenv("MOM_ADX_MIN", "15"))
CHOP_MAX = float(os.getenv("MOM_CHOP_MAX", "55"))
CONFIRM_MIN = int(os.getenv("MOM_CONFIRM_MIN", "3"))
COOLDOWN_BARS = int(os.getenv("MOM_COOLDOWN_BARS", "3"))
SUPPORTED_SCHEMAS = {
    "adaptive-momentum-v4.2-15m",
    "adaptive-momentum-v4.3-dual-entry-15m",
}


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
        self.counts = {k: 0 for k in ("scans", "entries", "cooldown", "trigger", "quality", "confirmation")}
        self._identity()
        self.load_state()

    @staticmethod
    def _identity() -> None:
        try:
            runner = sys.modules.get("run_bot") or sys.modules.get("__main__")
            if runner is not None and hasattr(runner, "logger"):
                runner.logger = logging.getLogger("adaptive_momentum_v4_3")
            if runner is not None and hasattr(runner, "BUILD_ID"):
                runner.BUILD_ID = "adaptive-momentum-v4.3-dual-entry-2026-08-26"
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
    def _score(i: Dict, direction: str) -> int:
        return int(i.get("confirmation_score_long" if direction == "LONG" else "confirmation_score_short", 0))

    @staticmethod
    def _entry_candidate(i: Dict) -> Tuple[str, str]:
        """Return direction and trigger. Momentum trigger gets priority if both fire."""
        if i.get("momentum_trigger_long"):
            trigger = "CHOCH bullish break" if i.get("structure_choch_long") else "Bullish BOS"
            return "LONG", trigger
        if i.get("momentum_trigger_short"):
            trigger = "CHOCH bearish break" if i.get("structure_choch_short") else "Bearish BOS"
            return "SHORT", trigger
        if i.get("pullback_trigger_long"):
            return "LONG", "EMA13 bullish reclaim"
        if i.get("pullback_trigger_short"):
            return "SHORT", "EMA13 bearish reject"
        return "NONE", ""

    def _debug(self, i: Dict, result: str, reason: str, direction: str = "NONE", trigger: str = "") -> str:
        symbol = self.symbol.split("/")[0]
        adx = float(i.get("adx", 0))
        chop = float(i.get("chop", 100))
        score = self._score(i, direction) if direction in {"LONG", "SHORT"} else 0
        if reason == "COOLDOWN":
            return f"MOMENTUM V4.3 · {symbol} · 15M · ⏳ COOLDOWN {self.cooldown_remaining} bars · RESULT: WAIT"
        if reason == "TRIGGER":
            return f"MOMENTUM V4.3 · {symbol} · 15M · ❌ No BOS/CHOCH or EMA13 reclaim/reject · RESULT: WAIT TRIGGER"

        macd_ok = bool(i.get("macd_bull" if direction == "LONG" else "macd_bear"))
        roc_ok = bool(i.get("roc_long" if direction == "LONG" else "roc_short"))
        bb_ok = bool(i.get("bb_long" if direction == "LONG" else "bb_short"))
        structure_ok = bool(i.get("structure_long" if direction == "LONG" else "structure_short"))
        alignment = "EMA8>EMA13" if direction == "LONG" else "EMA8<EMA13"
        parts = [
            f"MOMENTUM V4.3 · {symbol} · 15M · {direction}",
            f"✅ Trigger {trigger}",
            f"✅ Alignment {alignment}",
        ]
        if reason == "QUALITY":
            parts.append(f"❌ Quality ADX {adx:.1f}/{ADX_MIN:g} · CHOP {chop:.1f}/{CHOP_MAX:g}")
        else:
            parts.append(f"✅ Quality ADX {adx:.1f}{'↑' if i.get('adx_rising') else '→'} · CHOP {chop:.1f}")

        if reason == "CONFIRMATION":
            parts.append(
                f"❌ Confirm {score}/4 need {CONFIRM_MIN} · MACD {'✅' if macd_ok else '❌'} · "
                f"ROC9 {'✅' if roc_ok else '❌'} · BB {'✅' if bb_ok else '❌'} · Structure {'✅' if structure_ok else '❌'}"
            )
        elif reason not in {"QUALITY", "TRIGGER", "COOLDOWN"}:
            parts.append(
                f"✅ Confirm {score}/4 · MACD {'✅' if macd_ok else '❌'} · "
                f"ROC9 {'✅' if roc_ok else '❌'} · BB {'✅' if bb_ok else '❌'} · Structure {'✅' if structure_ok else '❌'}"
            )
            if result == "ENTRY":
                parts.append("✅ ENTRY CONFIRMED")

        labels = {
            "QUALITY": "WAIT QUALITY",
            "CONFIRMATION": "WAIT CONFIRMATION",
            "LONG": "ENTRY LONG",
            "SHORT": "ENTRY SHORT",
            "RISK": "WAIT RISK",
        }
        parts.append(f"RESULT: {labels.get(reason, reason)}")
        return " · ".join(parts)

    def _build(self, i: Dict, direction: str, trigger: str):
        entry = float(i["close"])
        atr = max(float(i["atr"]), entry * 0.0005)
        minimum = entry * MIN_SL_PCT
        if direction == "LONG":
            sl = min(float(i.get("recent_low", entry - atr)), entry - SL_ATR * atr, entry - minimum)
            risk = entry - sl
            tp1, tp2 = entry + TP1_R * risk, entry + TP2_R * risk
        else:
            sl = max(float(i.get("recent_high", entry + atr)), entry + SL_ATR * atr, entry + minimum)
            risk = sl - entry
            tp1, tp2 = entry - TP1_R * risk, entry - TP2_R * risk
        if risk <= 0:
            return None
        size = min(self.risk_usdt / risk, (self.margin_usdt * self.leverage) / max(entry, 1e-12))
        if size <= 0:
            return None
        return {
            "direction": direction,
            "strategy": "adaptive_momentum_v4_3_dual_entry",
            "trigger": trigger,
            "confirmation_score": self._score(i, direction),
            "entry": entry, "sl": sl, "tp": tp2, "tp1": tp1, "tp2": tp2,
            "size": size, "risk_usdt": size * risk, "sl_pct": 100 * risk / max(entry, 1e-12),
            "ema8": float(i["ema8"]), "ema13": float(i["ema13"]),
            "macd": float(i["macd"]), "macd_signal": float(i["macd_signal"]),
            "macd_hist": float(i["macd_hist"]),
            "bb_mid": float(i["bb_mid"]), "bb_upper": float(i["bb_upper"]), "bb_lower": float(i["bb_lower"]),
            "atr": atr, "adx": float(i["adx"]), "adx_rising": bool(i.get("adx_rising")),
            "chop": float(i["chop"]), "roc9": float(i["roc9"]),
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
        if reason in {"EMA_ALIGNMENT_FLIP", "MOMENTUM_STRUCTURE_EXIT"}:
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
            raise RuntimeError(f"MOMENTUM_V43_SCHEMA_MISMATCH: {i.get('schema')}")

        if self.position:
            event = self.check_price(price or float(i["close"]))
            if event:
                return event
            p = self.position
            px = price or float(i["close"])

            # EMA8/13 is now alignment; a confirmed flip against the position is a thesis exit.
            if p.direction == "LONG" and i.get("ema_bear"):
                return self._close(px, "EMA_ALIGNMENT_FLIP")
            if p.direction == "SHORT" and i.get("ema_bull"):
                return self._close(px, "EMA_ALIGNMENT_FLIP")

            momentum = int(i.get("momentum_score_long" if p.direction == "LONG" else "momentum_score_short", 0))
            structure_invalid = bool(i.get("structure_invalid_long" if p.direction == "LONG" else "structure_invalid_short"))
            if momentum == 0 and structure_invalid:
                return self._close(px, "MOMENTUM_STRUCTURE_EXIT")

            warning = " | ⚠ MOMENTUM WEAK" if momentum == 0 else ""
            self.last_signal = (
                f"MANAGE {p.direction} | SL={p.sl:.4f} | TP1={p.tp1:.4f} | TP2={p.tp2:.4f} | "
                f"Momentum={momentum}/2 | StructureInvalid={int(structure_invalid)}{warning}"
            )
            return None

        self.counts["scans"] += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.counts["cooldown"] += 1
            self.save_state()
            self.last_signal = self._debug(i, "WAIT", "COOLDOWN")
            return None

        # FIRST: one of the two price-action triggers must fire on the current closed 15M candle.
        direction, trigger = self._entry_candidate(i)
        if direction == "NONE":
            self.counts["trigger"] += 1
            self.last_signal = self._debug(i, "WAIT", "TRIGGER")
            return None

        # SECOND: market quality. ADX rising is informative, not mandatory.
        if float(i["adx"]) < ADX_MIN or float(i["chop"]) > CHOP_MAX:
            self.counts["quality"] += 1
            self.last_signal = self._debug(i, "WAIT", "QUALITY", direction, trigger)
            return None

        # THIRD: adaptive confirmation. Need at least 3/4 from MACD, ROC9, BB and Structure.
        score = self._score(i, direction)
        if score < CONFIRM_MIN:
            self.counts["confirmation"] += 1
            self.last_signal = self._debug(i, "WAIT", "CONFIRMATION", direction, trigger)
            return None

        payload = self._build(i, direction, trigger)
        if not payload:
            self.last_signal = self._debug(i, "WAIT", "RISK", direction, trigger)
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
        self.last_signal = self._debug(i, "ENTRY", direction, direction, trigger)
        return {"event": "OPEN", **payload}
