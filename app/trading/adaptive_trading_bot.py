"""Adaptive Bot v13: 4H trend -> 1H quality -> 15M structure + EMA trigger."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json
import os
import time


ADX_MIN = float(os.getenv("V13_ADX_MIN", "15"))
CHOP_MAX = float(os.getenv("V13_CHOP_MAX", "58"))
SLOPE_MIN_ATR = float(os.getenv("V13_SLOPE_MIN_ATR", "0.05"))
LOCATION_MAX_ATR = float(os.getenv("V13_LOCATION_MAX_ATR", "0.80"))
BODY_MAX_ATR = float(os.getenv("V13_BODY_MAX_ATR", "1.20"))
ROOM_MIN_R = float(os.getenv("V13_ROOM_MIN_R", "1.20"))
SL_ATR_BUFFER = float(os.getenv("V13_SL_ATR_BUFFER", "0.15"))
TP_R = float(os.getenv("V13_TP_R", "2.00"))
BE_TRIGGER_R = float(os.getenv("V13_BE_TRIGGER_R", "1.00"))


@dataclass
class Position:
    direction: str
    entry: float
    sl: float
    initial_sl: float
    tp: float
    size: float
    strategy: str
    opened_at: float
    be_moved: bool = False


class TradingBot:
    def __init__(
        self,
        symbol: str,
        margin_usdt: float = 20.0,
        leverage: int = 20,
        paper: bool = True,
        state_file: str = "",
        execution_callback: Optional[Callable] = None,
    ):
        self.symbol = symbol
        self.margin_usdt = float(margin_usdt)
        self.leverage = int(leverage)
        self.paper = bool(paper)
        self.state_file = state_file
        self.execution_callback = execution_callback
        self.position: Optional[Position] = None
        self.last_signal = "WARMUP"
        self.debug_counts: Dict[str, int] = {
            "scans": 0,
            "qualified": 0,
            "entries": 0,
            "4H": 0,
            "1H": 0,
            "LARGE_BAR": 0,
            "CHASE": 0,
            "LOCATION": 0,
            "STRUCTURE": 0,
            "CROSS": 0,
            "ROOM": 0,
        }
        self.load_state()

    @property
    def position_open(self) -> bool:
        return self.position is not None

    def load_state(self) -> None:
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, encoding="utf-8") as file:
                raw = json.load(file)
            position = raw.get("position")
            if position:
                if "initial_sl" not in position:
                    position["initial_sl"] = position.get("sl", 0.0)
                position.setdefault("be_moved", False)
                self.position = Position(**position)
        except Exception:
            self.position = None

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as file:
            json.dump({"position": asdict(self.position) if self.position else None}, file)
        os.replace(temp, self.state_file)

    @staticmethod
    def _macro(i4: Dict) -> str:
        bullish = (
            i4["close"] > i4["ema20"] > i4["ema50"]
            and i4["ema20_slope_atr"] >= SLOPE_MIN_ATR
        )
        bearish = (
            i4["close"] < i4["ema20"] < i4["ema50"]
            and i4["ema20_slope_atr"] <= -SLOPE_MIN_ATR
        )
        if bullish:
            return "BULL"
        if bearish:
            return "BEAR"
        return "NEUTRAL"

    @staticmethod
    def _context_parts(i1: Dict, direction: str) -> Dict[str, bool]:
        quality_adx = i1["adx"] >= ADX_MIN
        quality_chop = i1["chop"] <= CHOP_MAX
        if direction == "LONG":
            aligned = i1["close"] > i1["ema20"] > i1["ema50"] and i1["ema20_slope_atr"] > 0
            structure_ok = i1["structure"] != "BEAR"
        else:
            aligned = i1["close"] < i1["ema20"] < i1["ema50"] and i1["ema20_slope_atr"] < 0
            structure_ok = i1["structure"] != "BULL"
        return {
            "adx": bool(quality_adx),
            "chop": bool(quality_chop),
            "aligned": bool(aligned),
            "structure": bool(structure_ok),
            "passed": bool(quality_adx and quality_chop and aligned and structure_ok),
        }

    def _reject(self, reason: str) -> None:
        if reason in self.debug_counts:
            self.debug_counts[reason] += 1

    def _debug_block(
        self,
        *,
        macro: str,
        i15: Dict,
        i1: Dict,
        i4: Dict,
        direction: str = "NONE",
        result: str,
        reason: str,
        location_ok: Optional[bool] = None,
        structure_ok: Optional[bool] = None,
        trigger_ok: Optional[bool] = None,
        room_r: Optional[float] = None,
    ) -> str:
        context = self._context_parts(i1, direction) if direction in ("LONG", "SHORT") else None
        macro_align = (
            i4["close"] > i4["ema20"] > i4["ema50"]
            if macro == "BULL"
            else i4["close"] < i4["ema20"] < i4["ema50"]
            if macro == "BEAR"
            else False
        )
        checks = []
        if location_ok is not None:
            checks.append(f"location={'PASS' if location_ok else 'FAIL'}")
        if structure_ok is not None:
            checks.append(f"structure={'PASS' if structure_ok else 'FAIL'}")
        if trigger_ok is not None:
            checks.append(f"ema8/13={'PASS' if trigger_ok else 'WAIT'}")
        if room_r is not None:
            checks.append(f"room={room_r:.2f}R")
        setup_line = " | ".join(checks) if checks else "not evaluated"

        context_line = (
            f"ADX={i1['adx']:.1f}/{ADX_MIN:.1f} {'PASS' if context and context['adx'] else 'FAIL'} | "
            f"CHOP={i1['chop']:.1f}/{CHOP_MAX:.1f} {'PASS' if context and context['chop'] else 'FAIL'} | "
            f"EMA={'PASS' if context and context['aligned'] else 'FAIL'} | "
            f"Structure={i1['structure']} {'PASS' if context and context['structure'] else 'FAIL'}"
            if context else
            f"ADX={i1['adx']:.1f} | CHOP={i1['chop']:.1f} | Structure={i1['structure']}"
        )

        return (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{self.symbol} | 15M DECISION\n"
            f"4H  macro={macro} | EMA20/50={'PASS' if macro_align else 'FAIL'} | "
            f"slope={i4['ema20_slope_atr']:+.2f}ATR (need ±{SLOPE_MIN_ATR:.2f}) | structure={i4['structure']}\n"
            f"1H  {context_line}\n"
            f"15M price={i15['close']:.6f} | ext={i15['extension_atr']:.2f}/{LOCATION_MAX_ATR:.2f}ATR | "
            f"body={i15['body_atr']:.2f}/{BODY_MAX_ATR:.2f}ATR | structure={i15['structure']}\n"
            f"SETUP {direction}: {setup_line}\n"
            f"RESULT: {result} | reason={reason}\n"
            f"COUNTERS scans={self.debug_counts['scans']} qualified={self.debug_counts['qualified']} "
            f"entries={self.debug_counts['entries']} rejects="
            f"4H:{self.debug_counts['4H']} 1H:{self.debug_counts['1H']} "
            f"chase:{self.debug_counts['CHASE']} location:{self.debug_counts['LOCATION']} "
            f"structure:{self.debug_counts['STRUCTURE']} cross:{self.debug_counts['CROSS']} "
            f"room:{self.debug_counts['ROOM']}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    def _build(self, direction: str, entry: float, i15: Dict, i1: Dict, i4: Dict, macro: str) -> Optional[Dict]:
        atr = max(float(i15["atr"]), entry * 0.0005)
        if direction == "LONG":
            swing = float(i15["last_swing_low"])
            sl = swing - SL_ATR_BUFFER * atr
            if sl >= entry:
                self._reject("ROOM")
                self.last_signal = self._debug_block(
                    macro=macro, i15=i15, i1=i1, i4=i4, direction=direction,
                    result="WAIT", reason="INVALID_SL", room_r=0.0,
                )
                return None
            risk = entry - sl
            opposing_level = float(i15["last_swing_high"])
            room_r = (opposing_level - entry) / max(risk, 1e-12)
            if opposing_level > entry and room_r < ROOM_MIN_R:
                self._reject("ROOM")
                self.last_signal = self._debug_block(
                    macro=macro, i15=i15, i1=i1, i4=i4, direction=direction,
                    result="WAIT", reason=f"ROOM_LOW_{room_r:.2f}R", room_r=room_r,
                )
                return None
            tp = entry + TP_R * risk
        else:
            swing = float(i15["last_swing_high"])
            sl = swing + SL_ATR_BUFFER * atr
            if sl <= entry:
                self._reject("ROOM")
                self.last_signal = self._debug_block(
                    macro=macro, i15=i15, i1=i1, i4=i4, direction=direction,
                    result="WAIT", reason="INVALID_SL", room_r=0.0,
                )
                return None
            risk = sl - entry
            opposing_level = float(i15["last_swing_low"])
            room_r = (entry - opposing_level) / max(risk, 1e-12)
            if opposing_level < entry and room_r < ROOM_MIN_R:
                self._reject("ROOM")
                self.last_signal = self._debug_block(
                    macro=macro, i15=i15, i1=i1, i4=i4, direction=direction,
                    result="WAIT", reason=f"ROOM_LOW_{room_r:.2f}R", room_r=room_r,
                )
                return None
            tp = entry - TP_R * risk

        self.debug_counts["qualified"] += 1
        return {
            "direction": direction,
            "strategy": "structure_trend",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": (self.margin_usdt * self.leverage) / max(entry, 1e-12),
            "room_r": room_r,
        }

    def _signal(self, i15: Dict, i1: Dict, i4: Dict) -> Optional[Dict]:
        self.debug_counts["scans"] += 1
        macro = self._macro(i4)
        close = float(i15["close"])
        atr = max(float(i15["atr"]), close * 0.0005)

        if macro == "NEUTRAL":
            self._reject("4H")
            self.last_signal = self._debug_block(
                macro=macro, i15=i15, i1=i1, i4=i4,
                result="WAIT", reason="4H_NEUTRAL",
            )
            return None
        if i15["body_atr"] > BODY_MAX_ATR:
            self._reject("LARGE_BAR")
            self.last_signal = self._debug_block(
                macro=macro, i15=i15, i1=i1, i4=i4, direction="LONG" if macro == "BULL" else "SHORT",
                result="WAIT", reason=f"LARGE_BAR_{i15['body_atr']:.2f}ATR",
            )
            return None
        if i15["extension_atr"] > LOCATION_MAX_ATR:
            self._reject("CHASE")
            self.last_signal = self._debug_block(
                macro=macro, i15=i15, i1=i1, i4=i4, direction="LONG" if macro == "BULL" else "SHORT",
                result="WAIT", reason=f"CHASE_{i15['extension_atr']:.2f}ATR",
            )
            return None

        bullish_bar = i15["close"] > i15["open"]
        bearish_bar = i15["close"] < i15["open"]
        direction = "LONG" if macro == "BULL" else "SHORT"
        context = self._context_parts(i1, direction)
        if not context["passed"]:
            self._reject("1H")
            self.last_signal = self._debug_block(
                macro=macro, i15=i15, i1=i1, i4=i4, direction=direction,
                result="WAIT", reason="1H_CONTEXT",
            )
            return None

        if direction == "LONG":
            location_ok = (
                i15["low"] <= i15["ema20"] + 0.20 * atr
                and close > i15["ema13"]
                and close < i15["bb_upper"]
            )
            structure_ok = i15["higher_low"] and i15["structure"] != "BEAR"
            trigger_ok = i15["cross_up"] and bullish_bar
        else:
            location_ok = (
                i15["high"] >= i15["ema20"] - 0.20 * atr
                and close < i15["ema13"]
                and close > i15["bb_lower"]
            )
            structure_ok = i15["lower_high"] and i15["structure"] != "BULL"
            trigger_ok = i15["cross_down"] and bearish_bar

        if not location_ok:
            self._reject("LOCATION")
        if not structure_ok:
            self._reject("STRUCTURE")
        if not trigger_ok:
            self._reject("CROSS")

        if location_ok and structure_ok and trigger_ok:
            signal = self._build(direction, close, i15, i1, i4, macro)
            if signal:
                self.last_signal = self._debug_block(
                    macro=macro, i15=i15, i1=i1, i4=i4, direction=direction,
                    result="QUALIFIED", reason="ALL_GATES_PASS",
                    location_ok=True, structure_ok=True, trigger_ok=True,
                    room_r=float(signal.get("room_r", 0.0)),
                )
                return signal
            return None

        reasons = []
        if not location_ok:
            reasons.append("LOCATION")
        if not structure_ok:
            reasons.append("STRUCTURE")
        if not trigger_ok:
            reasons.append("EMA_CROSS")
        self.last_signal = self._debug_block(
            macro=macro, i15=i15, i1=i1, i4=i4, direction=direction,
            result="WAIT", reason="+".join(reasons),
            location_ok=location_ok, structure_ok=structure_ok, trigger_ok=trigger_ok,
        )
        return None

    def _close(self, price: float, reason: str) -> Dict:
        assert self.position is not None
        position = self.position
        pnl = (
            (price - position.entry) * position.size
            if position.direction == "LONG"
            else (position.entry - price) * position.size
        )
        initial_risk = abs(position.entry - position.initial_sl) * position.size
        r_multiple = pnl / initial_risk if initial_risk > 0 else 0.0
        payload = {
            "symbol": self.symbol,
            "direction": position.direction,
            "price": price,
            "entry": position.entry,
            "sl": position.sl,
            "tp": position.tp,
            "size": position.size,
            "strategy": position.strategy,
            "opened_at": position.opened_at,
            "closed_at": time.time(),
            "reason": reason,
            "pnl": pnl,
            "r_multiple": r_multiple,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_" + position.direction, payload)
        self.position = None
        self.save_state()
        self.last_signal = f"CLOSE {position.direction} {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R"
        return {"event": "CLOSE", **payload}

    def on_bar(self, i15: Dict, i1: Dict, i4: Dict, price: float) -> Optional[Dict]:
        if not i15 or not i1 or not i4:
            self.last_signal = "WAIT indicator warmup"
            return None

        if self.position:
            p = self.position
            risk_distance = abs(p.entry - p.initial_sl)
            current_r = (
                (price - p.entry) / max(risk_distance, 1e-12)
                if p.direction == "LONG"
                else (p.entry - price) / max(risk_distance, 1e-12)
            )

            if not p.be_moved and current_r >= BE_TRIGGER_R:
                p.sl = p.entry
                p.be_moved = True
                self.save_state()
                self.last_signal = f"MANAGE {p.direction} SL_TO_BE current={current_r:+.2f}R"

            hit_sl = price <= p.sl if p.direction == "LONG" else price >= p.sl
            hit_tp = price >= p.tp if p.direction == "LONG" else price <= p.tp
            structure_broken = (
                p.direction == "LONG" and price < i15["last_swing_low"] and price < i15["ema20"]
            ) or (
                p.direction == "SHORT" and price > i15["last_swing_high"] and price > i15["ema20"]
            )
            cross_back = (
                p.direction == "LONG" and i15["cross_down"]
            ) or (
                p.direction == "SHORT" and i15["cross_up"]
            )

            if hit_sl:
                return self._close(price, "BE" if p.be_moved and p.sl == p.entry else "SL")
            if hit_tp:
                return self._close(price, "TP")
            if cross_back and structure_broken:
                return self._close(price, "STRUCTURE_EXIT")

            self.last_signal = (
                f"MANAGE {p.direction} current={current_r:+.2f}R entry={p.entry:.6f} "
                f"sl={p.sl:.6f} tp={p.tp:.6f} be={int(p.be_moved)}"
            )
            return None

        signal = self._signal(i15, i1, i4)
        if not signal:
            return None

        payload = {"symbol": self.symbol, **signal}
        if self.execution_callback:
            self.execution_callback("OPEN_" + signal["direction"], payload)

        self.position = Position(
            direction=signal["direction"],
            entry=signal["entry"],
            sl=signal["sl"],
            initial_sl=signal["sl"],
            tp=signal["tp"],
            size=signal["size"],
            strategy=signal["strategy"],
            opened_at=time.time(),
        )
        self.debug_counts["entries"] += 1
        self.save_state()
        self.last_signal = (
            f"OPEN {signal['direction']} structure_trend entry={signal['entry']:.6f} "
            f"sl={signal['sl']:.6f} tp={signal['tp']:.6f} room={signal.get('room_r', 0):.2f}R"
        )
        return {"event": "OPEN", **payload}


_TRADEABLE_REGIMES = frozenset({"Trend"})


class ExpectancyEngine:
    pass
