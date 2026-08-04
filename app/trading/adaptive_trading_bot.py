"""Adaptive Bot v13.1: 4H direction -> 1H quality -> multi-bar 15M setup."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json
import os
import time


ADX_MIN = float(os.getenv("V13_ADX_MIN", "15"))
CHOP_MAX = float(os.getenv("V13_CHOP_MAX", "60"))
SLOPE_MIN_ATR = float(os.getenv("V13_SLOPE_MIN_ATR", "0.05"))
LOCATION_MAX_ATR = float(os.getenv("V13_LOCATION_MAX_ATR", "1.00"))
BODY_MAX_ATR = float(os.getenv("V13_BODY_MAX_ATR", "1.20"))
ROOM_MIN_R = float(os.getenv("V13_ROOM_MIN_R", "1.20"))
SL_ATR_BUFFER = float(os.getenv("V13_SL_ATR_BUFFER", "0.15"))
TP_R = float(os.getenv("V13_TP_R", "2.00"))
BE_TRIGGER_R = float(os.getenv("V13_BE_TRIGGER_R", "1.00"))
STRUCTURE_WINDOW = int(os.getenv("V13_STRUCTURE_WINDOW", "3"))
CROSS_WINDOW = int(os.getenv("V13_CROSS_WINDOW", "2"))
LOCATION_WINDOW = int(os.getenv("V13_LOCATION_WINDOW", "3"))
NEUTRAL_ADX_MIN = float(os.getenv("V13_NEUTRAL_ADX_MIN", "20"))
NEUTRAL_CHOP_MAX = float(os.getenv("V13_NEUTRAL_CHOP_MAX", "52"))


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
            "scans": 0, "qualified": 0, "entries": 0,
            "4H": 0, "1H": 0, "LARGE_BAR": 0, "CHASE": 0,
            "LOCATION": 0, "STRUCTURE": 0, "CROSS": 0, "ROOM": 0,
        }
        self.setup_age: Dict[str, int] = {
            "long_structure": 999,
            "short_structure": 999,
            "long_cross": 999,
            "short_cross": 999,
            "long_location": 999,
            "short_location": 999,
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
                position.setdefault("initial_sl", position.get("sl", 0.0))
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
        if (
            i4["close"] > i4["ema20"] > i4["ema50"]
            and i4["ema20_slope_atr"] >= SLOPE_MIN_ATR
        ):
            return "BULL"
        if (
            i4["close"] < i4["ema20"] < i4["ema50"]
            and i4["ema20_slope_atr"] <= -SLOPE_MIN_ATR
        ):
            return "BEAR"
        return "NEUTRAL"

    @staticmethod
    def _context_parts(i1: Dict, direction: str, strong: bool = False) -> Dict[str, bool]:
        adx_threshold = NEUTRAL_ADX_MIN if strong else ADX_MIN
        chop_threshold = NEUTRAL_CHOP_MAX if strong else CHOP_MAX
        quality_adx = i1["adx"] >= adx_threshold
        quality_chop = i1["chop"] <= chop_threshold
        if direction == "LONG":
            aligned = i1["close"] > i1["ema20"] > i1["ema50"] and i1["ema20_slope_atr"] > 0
            structure_ok = i1["structure"] != "BEAR"
        else:
            aligned = i1["close"] < i1["ema20"] < i1["ema50"] and i1["ema20_slope_atr"] < 0
            structure_ok = i1["structure"] != "BULL"
        return {
            "adx": bool(quality_adx), "chop": bool(quality_chop),
            "aligned": bool(aligned), "structure": bool(structure_ok),
            "passed": bool(quality_adx and quality_chop and aligned and structure_ok),
            "adx_threshold": adx_threshold, "chop_threshold": chop_threshold,
        }

    def _direction(self, macro: str, i1: Dict) -> tuple[str, bool]:
        if macro == "BULL":
            return "LONG", False
        if macro == "BEAR":
            return "SHORT", False
        long_strong = self._context_parts(i1, "LONG", strong=True)["passed"]
        short_strong = self._context_parts(i1, "SHORT", strong=True)["passed"]
        if long_strong and not short_strong:
            return "LONG", True
        if short_strong and not long_strong:
            return "SHORT", True
        return "NONE", True

    def _age_setups(self, i15: Dict) -> None:
        for key in self.setup_age:
            self.setup_age[key] = min(999, self.setup_age[key] + 1)
        atr = max(float(i15["atr"]), float(i15["close"]) * 0.0005)
        if i15["higher_low"] and i15["structure"] != "BEAR":
            self.setup_age["long_structure"] = 0
        if i15["lower_high"] and i15["structure"] != "BULL":
            self.setup_age["short_structure"] = 0
        if i15["cross_up"]:
            self.setup_age["long_cross"] = 0
        if i15["cross_down"]:
            self.setup_age["short_cross"] = 0
        long_pullback = i15["low"] <= i15["ema20"] + 0.30 * atr
        short_pullback = i15["high"] >= i15["ema20"] - 0.30 * atr
        if long_pullback:
            self.setup_age["long_location"] = 0
        if short_pullback:
            self.setup_age["short_location"] = 0

    def _reject(self, reason: str) -> None:
        if reason in self.debug_counts:
            self.debug_counts[reason] += 1

    def _debug_block(
        self, *, macro: str, i15: Dict, i1: Dict, i4: Dict,
        direction: str = "NONE", result: str, reason: str,
        neutral_override: bool = False,
        location_ok: Optional[bool] = None,
        structure_ok: Optional[bool] = None,
        trigger_ok: Optional[bool] = None,
        room_r: Optional[float] = None,
    ) -> str:
        context = self._context_parts(i1, direction, strong=neutral_override) if direction in ("LONG", "SHORT") else None
        macro_align = (
            i4["close"] > i4["ema20"] > i4["ema50"] if macro == "BULL"
            else i4["close"] < i4["ema20"] < i4["ema50"] if macro == "BEAR"
            else False
        )
        setup = []
        if direction in ("LONG", "SHORT"):
            side = direction.lower()
            setup += [
                f"locAge={self.setup_age[side + '_location']}/{LOCATION_WINDOW}",
                f"structAge={self.setup_age[side + '_structure']}/{STRUCTURE_WINDOW}",
                f"crossAge={self.setup_age[side + '_cross']}/{CROSS_WINDOW}",
            ]
        if location_ok is not None:
            setup.append(f"location={'PASS' if location_ok else 'FAIL'}")
        if structure_ok is not None:
            setup.append(f"structure={'PASS' if structure_ok else 'FAIL'}")
        if trigger_ok is not None:
            setup.append(f"trigger={'PASS' if trigger_ok else 'WAIT'}")
        if room_r is not None:
            setup.append(f"room={room_r:.2f}R")
        setup_text = ",".join(setup) if setup else f"SKIPPED({reason})"
        if context:
            context_text = (
                f"adx={i1['adx']:.1f}/{context['adx_threshold']:.1f}:{'P' if context['adx'] else 'F'},"
                f"chop={i1['chop']:.1f}/{context['chop_threshold']:.1f}:{'P' if context['chop'] else 'F'},"
                f"ema={'P' if context['aligned'] else 'F'},structure={i1['structure']}:{'P' if context['structure'] else 'F'}"
            )
        else:
            context_text = f"adx={i1['adx']:.1f},chop={i1['chop']:.1f},structure={i1['structure']},status=SKIPPED"
        return (
            f"DECISION symbol={self.symbol} tf=15m"
            f" | 4H[macro={macro},ema20/50={'PASS' if macro_align else 'FAIL'},slope={i4['ema20_slope_atr']:+.2f}/{SLOPE_MIN_ATR:.2f}ATR,structure={i4['structure']},neutralOverride={int(neutral_override)}]"
            f" | 1H[{context_text}]"
            f" | 15M[price={i15['close']:.6f},ext={i15['extension_atr']:.2f}/{LOCATION_MAX_ATR:.2f}ATR,body={i15['body_atr']:.2f}/{BODY_MAX_ATR:.2f}ATR,structure={i15['structure']}]"
            f" | SETUP[{direction}:{setup_text}]"
            f" | RESULT[{result}:{reason}]"
            f" | SYMBOL_COUNTERS[scans={self.debug_counts['scans']},qualified={self.debug_counts['qualified']},entries={self.debug_counts['entries']},reject4H={self.debug_counts['4H']},reject1H={self.debug_counts['1H']},largeBar={self.debug_counts['LARGE_BAR']},chase={self.debug_counts['CHASE']},location={self.debug_counts['LOCATION']},structure={self.debug_counts['STRUCTURE']},cross={self.debug_counts['CROSS']},room={self.debug_counts['ROOM']}]"
        )

    def _build(self, direction: str, entry: float, i15: Dict, i1: Dict, i4: Dict, macro: str, neutral_override: bool) -> Optional[Dict]:
        atr = max(float(i15["atr"]), entry * 0.0005)
        if direction == "LONG":
            sl = float(i15["last_swing_low"]) - SL_ATR_BUFFER * atr
            if sl >= entry:
                room_r = 0.0
            else:
                risk = entry - sl
                opposing = float(i15["last_swing_high"])
                room_r = (opposing - entry) / max(risk, 1e-12)
                if not (opposing > entry and room_r < ROOM_MIN_R):
                    tp = entry + TP_R * risk
                    self.debug_counts["qualified"] += 1
                    return {"direction": direction, "strategy": "structure_trend_v13_1", "entry": entry, "sl": sl, "tp": tp, "size": (self.margin_usdt * self.leverage) / max(entry, 1e-12), "room_r": room_r}
        else:
            sl = float(i15["last_swing_high"]) + SL_ATR_BUFFER * atr
            if sl <= entry:
                room_r = 0.0
            else:
                risk = sl - entry
                opposing = float(i15["last_swing_low"])
                room_r = (entry - opposing) / max(risk, 1e-12)
                if not (opposing < entry and room_r < ROOM_MIN_R):
                    tp = entry - TP_R * risk
                    self.debug_counts["qualified"] += 1
                    return {"direction": direction, "strategy": "structure_trend_v13_1", "entry": entry, "sl": sl, "tp": tp, "size": (self.margin_usdt * self.leverage) / max(entry, 1e-12), "room_r": room_r}
        self._reject("ROOM")
        self.last_signal = self._debug_block(macro=macro, i15=i15, i1=i1, i4=i4, direction=direction, result="WAIT", reason=f"ROOM_LOW_{room_r:.2f}R", neutral_override=neutral_override, room_r=room_r)
        return None

    def _signal(self, i15: Dict, i1: Dict, i4: Dict) -> Optional[Dict]:
        self.debug_counts["scans"] += 1
        self._age_setups(i15)
        macro = self._macro(i4)
        direction, neutral_override = self._direction(macro, i1)
        close = float(i15["close"])

        if direction == "NONE":
            self._reject("4H")
            self.last_signal = self._debug_block(macro=macro, i15=i15, i1=i1, i4=i4, result="WAIT", reason="4H_NEUTRAL_1H_NOT_STRONG")
            return None
        if i15["body_atr"] > BODY_MAX_ATR:
            self._reject("LARGE_BAR")
            self.last_signal = self._debug_block(macro=macro, i15=i15, i1=i1, i4=i4, direction=direction, result="WAIT", reason=f"LARGE_BAR_{i15['body_atr']:.2f}ATR", neutral_override=neutral_override)
            return None
        if i15["extension_atr"] > LOCATION_MAX_ATR:
            self._reject("CHASE")
            self.last_signal = self._debug_block(macro=macro, i15=i15, i1=i1, i4=i4, direction=direction, result="WAIT", reason=f"CHASE_{i15['extension_atr']:.2f}ATR", neutral_override=neutral_override)
            return None

        context = self._context_parts(i1, direction, strong=neutral_override)
        if not context["passed"]:
            self._reject("1H")
            self.last_signal = self._debug_block(macro=macro, i15=i15, i1=i1, i4=i4, direction=direction, result="WAIT", reason="1H_CONTEXT", neutral_override=neutral_override)
            return None

        side = direction.lower()
        location_ok = self.setup_age[side + "_location"] <= LOCATION_WINDOW
        structure_ok = self.setup_age[side + "_structure"] <= STRUCTURE_WINDOW
        trigger_ok = self.setup_age[side + "_cross"] <= CROSS_WINDOW
        candle_ok = i15["close"] > i15["open"] if direction == "LONG" else i15["close"] < i15["open"]
        momentum_ok = close > i15["ema13"] if direction == "LONG" else close < i15["ema13"]
        band_ok = close < i15["bb_upper"] if direction == "LONG" else close > i15["bb_lower"]
        location_ok = location_ok and momentum_ok and band_ok
        trigger_ok = trigger_ok and candle_ok

        if not location_ok:
            self._reject("LOCATION")
        if not structure_ok:
            self._reject("STRUCTURE")
        if not trigger_ok:
            self._reject("CROSS")

        if location_ok and structure_ok and trigger_ok:
            signal = self._build(direction, close, i15, i1, i4, macro, neutral_override)
            if signal:
                self.last_signal = self._debug_block(macro=macro, i15=i15, i1=i1, i4=i4, direction=direction, result="QUALIFIED", reason="MULTIBAR_SETUP_PASS", neutral_override=neutral_override, location_ok=True, structure_ok=True, trigger_ok=True, room_r=float(signal.get("room_r", 0.0)))
                return signal
            return None

        reasons = []
        if not location_ok:
            reasons.append("LOCATION")
        if not structure_ok:
            reasons.append("STRUCTURE_WINDOW")
        if not trigger_ok:
            reasons.append("CROSS_WINDOW")
        self.last_signal = self._debug_block(macro=macro, i15=i15, i1=i1, i4=i4, direction=direction, result="WAIT", reason="+".join(reasons), neutral_override=neutral_override, location_ok=location_ok, structure_ok=structure_ok, trigger_ok=trigger_ok)
        return None

    def _close(self, price: float, reason: str) -> Dict:
        assert self.position is not None
        position = self.position
        pnl = (price - position.entry) * position.size if position.direction == "LONG" else (position.entry - price) * position.size
        initial_risk = abs(position.entry - position.initial_sl) * position.size
        r_multiple = pnl / initial_risk if initial_risk > 0 else 0.0
        payload = {"symbol": self.symbol, "direction": position.direction, "price": price, "entry": position.entry, "sl": position.sl, "tp": position.tp, "size": position.size, "strategy": position.strategy, "opened_at": position.opened_at, "closed_at": time.time(), "reason": reason, "pnl": pnl, "r_multiple": r_multiple}
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
            current_r = (price - p.entry) / max(risk_distance, 1e-12) if p.direction == "LONG" else (p.entry - price) / max(risk_distance, 1e-12)
            if not p.be_moved and current_r >= BE_TRIGGER_R:
                p.sl = p.entry
                p.be_moved = True
                self.save_state()
            hit_sl = price <= p.sl if p.direction == "LONG" else price >= p.sl
            hit_tp = price >= p.tp if p.direction == "LONG" else price <= p.tp
            structure_broken = (p.direction == "LONG" and price < i15["last_swing_low"] and price < i15["ema20"]) or (p.direction == "SHORT" and price > i15["last_swing_high"] and price > i15["ema20"])
            cross_back = (p.direction == "LONG" and i15["cross_down"]) or (p.direction == "SHORT" and i15["cross_up"])
            if hit_sl:
                return self._close(price, "BE" if p.be_moved and p.sl == p.entry else "SL")
            if hit_tp:
                return self._close(price, "TP")
            if cross_back and structure_broken:
                return self._close(price, "STRUCTURE_EXIT")
            self.last_signal = f"MANAGE {p.direction} current={current_r:+.2f}R entry={p.entry:.6f} sl={p.sl:.6f} tp={p.tp:.6f} be={int(p.be_moved)}"
            return None

        signal = self._signal(i15, i1, i4)
        if not signal:
            return None
        payload = {"symbol": self.symbol, **signal}
        if self.execution_callback:
            self.execution_callback("OPEN_" + signal["direction"], payload)
        self.position = Position(direction=signal["direction"], entry=signal["entry"], sl=signal["sl"], initial_sl=signal["sl"], tp=signal["tp"], size=signal["size"], strategy=signal["strategy"], opened_at=time.time())
        self.debug_counts["entries"] += 1
        for key in self.setup_age:
            self.setup_age[key] = 999
        self.save_state()
        self.last_signal = f"OPEN {signal['direction']} structure_trend_v13_1 entry={signal['entry']:.6f} sl={signal['sl']:.6f} tp={signal['tp']:.6f} room={signal.get('room_r', 0):.2f}R"
        return {"event": "OPEN", **payload}


_TRADEABLE_REGIMES = frozenset({"Trend"})


class ExpectancyEngine:
    pass
