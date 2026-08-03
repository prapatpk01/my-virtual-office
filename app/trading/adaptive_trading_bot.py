"""Adaptive Bot v12: simple Direction -> Location -> Trigger architecture."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json
import os
import time


@dataclass
class Position:
    direction: str
    entry: float
    sl: float
    tp: float
    size: float
    strategy: str
    opened_at: float


class TradingBot:
    def __init__(self, symbol: str, margin_usdt: float = 20.0, leverage: int = 20,
                 paper: bool = True, state_file: str = "",
                 execution_callback: Optional[Callable] = None):
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
            with open(self.state_file, encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("position"):
                self.position = Position(**raw["position"])
        except Exception:
            self.position = None

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({"position": asdict(self.position) if self.position else None}, f)

    @staticmethod
    def _macro(i4: Dict) -> str:
        if i4["close"] > i4["ema20"] > i4["ema50"]:
            return "BULL"
        if i4["close"] < i4["ema20"] < i4["ema50"]:
            return "BEAR"
        return "NEUTRAL"

    @staticmethod
    def _bias(i1: Dict) -> str:
        if i1["cdc_bull"] and i1["close"] > i1["ema20"]:
            return "BULL"
        if i1["cdc_bear"] and i1["close"] < i1["ema20"]:
            return "BEAR"
        return "NEUTRAL"

    def _build(self, direction: str, strategy: str, entry: float, i15: Dict) -> Dict:
        current_atr = max(i15["atr"], entry * 0.003)
        if direction == "LONG":
            sl = min(i15["swing_low"], entry - 1.2 * current_atr)
            tp = entry + 1.5 * (entry - sl)
        else:
            sl = max(i15["swing_high"], entry + 1.2 * current_atr)
            tp = entry - 1.5 * (sl - entry)
        return {
            "direction": direction,
            "strategy": strategy,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": (self.margin_usdt * self.leverage) / max(entry, 1e-12),
        }

    def _signal(self, i15: Dict, i1: Dict, i4: Dict) -> Optional[Dict]:
        macro, bias, close = self._macro(i4), self._bias(i1), i15["close"]
        if i15["extension_atr"] > 0.75 or i15["body_atr"] > 0.85:
            self.last_signal = f"WAIT late ext={i15['extension_atr']:.2f} body={i15['body_atr']:.2f}"
            return None
        long_ok, short_ok = macro == bias == "BULL", macro == bias == "BEAR"

        if long_ok and i15["cdc_bull"] and i15["adx"] >= 14 and i15["rsi"] <= 70:
            if (i15["prev_close"] <= i15["bb_mid"] < close or close > i15["prev_high"]) and close <= i15["bb_upper"]:
                return self._build("LONG", "trend_pullback", close, i15)
        if short_ok and i15["cdc_bear"] and i15["adx"] >= 14 and i15["rsi"] >= 30:
            if (i15["prev_close"] >= i15["bb_mid"] > close or close < i15["prev_low"]) and close >= i15["bb_lower"]:
                return self._build("SHORT", "trend_pullback", close, i15)

        if macro != "BEAR" and i1["cdc_bull"] and i15["cdc_cross_up"] and close > i15["bb_mid"]:
            return self._build("LONG", "cdc_transition", close, i15)
        if macro != "BULL" and i1["cdc_bear"] and i15["cdc_cross_down"] and close < i15["bb_mid"]:
            return self._build("SHORT", "cdc_transition", close, i15)

        vr = i15["volume"] / max(i15["vol_avg"], 1e-12)
        if long_ok and i15["cdc_bull"] and i15["prev_close"] <= i15["bb_upper"] < close and vr >= 1.15:
            return self._build("LONG", "bb_breakout", close, i15)
        if short_ok and i15["cdc_bear"] and i15["prev_close"] >= i15["bb_lower"] > close and vr >= 1.15:
            return self._build("SHORT", "bb_breakout", close, i15)

        self.last_signal = f"WAIT macro={macro} bias={bias} cdc={'BULL' if i15['cdc_bull'] else 'BEAR'}"
        return None

    def on_bar(self, i15: Dict, i1: Dict, i4: Dict, price: float) -> Optional[Dict]:
        if not i15 or not i1 or not i4:
            self.last_signal = "WAIT indicator warmup"
            return None
        if self.position:
            p = self.position
            hit_sl = price <= p.sl if p.direction == "LONG" else price >= p.sl
            hit_tp = price >= p.tp if p.direction == "LONG" else price <= p.tp
            cdc_flip = (p.direction == "LONG" and i15["cdc_bear"]) or (p.direction == "SHORT" and i15["cdc_bull"])
            if hit_sl or hit_tp or cdc_flip:
                reason = "SL" if hit_sl else ("TP" if hit_tp else "CDC_FLIP")
                pnl = (price - p.entry) * p.size if p.direction == "LONG" else (p.entry - price) * p.size
                risk = abs(p.entry - p.sl) * p.size
                r_multiple = pnl / risk if risk > 0 else 0.0
                payload = {
                    "symbol": self.symbol,
                    "direction": p.direction,
                    "price": price,
                    "entry": p.entry,
                    "sl": p.sl,
                    "tp": p.tp,
                    "size": p.size,
                    "strategy": p.strategy,
                    "opened_at": p.opened_at,
                    "closed_at": time.time(),
                    "reason": reason,
                    "pnl": pnl,
                    "r_multiple": r_multiple,
                }
                if self.execution_callback:
                    self.execution_callback("CLOSE_" + p.direction, payload)
                self.position = None
                self.save_state()
                self.last_signal = f"CLOSE {p.direction} {reason}"
                return {"event": "CLOSE", **payload}
            self.last_signal = f"MANAGE {p.direction}"
            return None

        signal = self._signal(i15, i1, i4)
        if not signal:
            return None
        payload = {"symbol": self.symbol, **signal}
        if self.execution_callback:
            self.execution_callback("OPEN_" + signal["direction"], payload)
        self.position = Position(signal["direction"], signal["entry"], signal["sl"], signal["tp"], signal["size"], signal["strategy"], time.time())
        self.last_signal = f"OPEN {signal['direction']} {signal['strategy']}"
        self.save_state()
        return {"event": "OPEN", **payload}


_TRADEABLE_REGIMES = frozenset({"Trend", "Transition", "Breakout"})
class ExpectancyEngine:
    pass
