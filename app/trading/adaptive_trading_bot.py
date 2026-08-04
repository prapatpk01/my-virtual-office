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
    def _context(i1: Dict, direction: str) -> bool:
        quality = i1["adx"] >= ADX_MIN and i1["chop"] <= CHOP_MAX
        if direction == "LONG":
            aligned = i1["close"] > i1["ema20"] > i1["ema50"] and i1["ema20_slope_atr"] > 0
            structure_ok = i1["structure"] != "BEAR"
        else:
            aligned = i1["close"] < i1["ema20"] < i1["ema50"] and i1["ema20_slope_atr"] < 0
            structure_ok = i1["structure"] != "BULL"
        return bool(quality and aligned and structure_ok)

    def _build(self, direction: str, entry: float, i15: Dict) -> Optional[Dict]:
        atr = max(float(i15["atr"]), entry * 0.0005)
        if direction == "LONG":
            swing = float(i15["last_swing_low"])
            sl = swing - SL_ATR_BUFFER * atr
            if sl >= entry:
                return None
            risk = entry - sl
            opposing_level = float(i15["last_swing_high"])
            room_r = (opposing_level - entry) / max(risk, 1e-12)
            if opposing_level > entry and room_r < ROOM_MIN_R:
                self.last_signal = f"WAIT insufficient room long={room_r:.2f}R"
                return None
            tp = entry + TP_R * risk
        else:
            swing = float(i15["last_swing_high"])
            sl = swing + SL_ATR_BUFFER * atr
            if sl <= entry:
                return None
            risk = sl - entry
            opposing_level = float(i15["last_swing_low"])
            room_r = (entry - opposing_level) / max(risk, 1e-12)
            if opposing_level < entry and room_r < ROOM_MIN_R:
                self.last_signal = f"WAIT insufficient room short={room_r:.2f}R"
                return None
            tp = entry - TP_R * risk

        return {
            "direction": direction,
            "strategy": "structure_trend",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": (self.margin_usdt * self.leverage) / max(entry, 1e-12),
        }

    def _signal(self, i15: Dict, i1: Dict, i4: Dict) -> Optional[Dict]:
        macro = self._macro(i4)
        close = float(i15["close"])
        atr = max(float(i15["atr"]), close * 0.0005)

        if macro == "NEUTRAL":
            self.last_signal = "WAIT 4H_NEUTRAL"
            return None
        if i15["body_atr"] > BODY_MAX_ATR:
            self.last_signal = f"WAIT LARGE_BAR body={i15['body_atr']:.2f}ATR"
            return None
        if i15["extension_atr"] > LOCATION_MAX_ATR:
            self.last_signal = f"WAIT CHASE ext={i15['extension_atr']:.2f}ATR"
            return None

        bullish_bar = i15["close"] > i15["open"]
        bearish_bar = i15["close"] < i15["open"]

        if macro == "BULL":
            if not self._context(i1, "LONG"):
                self.last_signal = (
                    f"WAIT LONG_CONTEXT adx={i1['adx']:.1f} chop={i1['chop']:.1f} "
                    f"structure={i1['structure']}"
                )
                return None
            location_ok = (
                i15["low"] <= i15["ema20"] + 0.20 * atr
                and close > i15["ema13"]
                and close < i15["bb_upper"]
            )
            structure_ok = i15["higher_low"] and i15["structure"] != "BEAR"
            trigger_ok = i15["cross_up"] and bullish_bar
            if location_ok and structure_ok and trigger_ok:
                return self._build("LONG", close, i15)
            self.last_signal = (
                f"WAIT LONG_SETUP location={int(location_ok)} structure={int(structure_ok)} "
                f"cross={int(i15['cross_up'])}"
            )
            return None

        if not self._context(i1, "SHORT"):
            self.last_signal = (
                f"WAIT SHORT_CONTEXT adx={i1['adx']:.1f} chop={i1['chop']:.1f} "
                f"structure={i1['structure']}"
            )
            return None
        location_ok = (
            i15["high"] >= i15["ema20"] - 0.20 * atr
            and close < i15["ema13"]
            and close > i15["bb_lower"]
        )
        structure_ok = i15["lower_high"] and i15["structure"] != "BULL"
        trigger_ok = i15["cross_down"] and bearish_bar
        if location_ok and structure_ok and trigger_ok:
            return self._build("SHORT", close, i15)
        self.last_signal = (
            f"WAIT SHORT_SETUP location={int(location_ok)} structure={int(structure_ok)} "
            f"cross={int(i15['cross_down'])}"
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
        self.last_signal = f"CLOSE {position.direction} {reason}"
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
                self.last_signal = f"MANAGE {p.direction} SL_TO_BE"

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

            self.last_signal = f"MANAGE {p.direction} {current_r:+.2f}R"
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
        self.save_state()
        self.last_signal = f"OPEN {signal['direction']} {signal['strategy']}"
        return {"event": "OPEN", **payload}


_TRADEABLE_REGIMES = frozenset({"Trend"})


class ExpectancyEngine:
    pass
