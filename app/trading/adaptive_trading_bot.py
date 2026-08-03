"""Adaptive Bot v12: simple Direction -> Location -> Trigger architecture."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json
import os
import time


# Entry filters are intentionally configurable so Paper results can be tuned
# without rewriting the strategy again.
EXTENSION_MAX_ATR = float(os.getenv("V12_EXTENSION_MAX_ATR", "1.00"))
BODY_MAX_ATR = float(os.getenv("V12_BODY_MAX_ATR", "1.20"))
TREND_ADX_MIN = float(os.getenv("V12_TREND_ADX_MIN", "12"))
BREAKOUT_VOLUME_MIN = float(os.getenv("V12_BREAKOUT_VOLUME_MIN", "1.05"))


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

        # Chase guard remains active, but is no longer so tight that normal
        # continuation candles are rejected.
        if i15["extension_atr"] > EXTENSION_MAX_ATR or i15["body_atr"] > BODY_MAX_ATR:
            self.last_signal = (
                f"WAIT late ext={i15['extension_atr']:.2f}/{EXTENSION_MAX_ATR:.2f} "
                f"body={i15['body_atr']:.2f}/{BODY_MAX_ATR:.2f}"
            )
            return None

        # 4H is now a hard anti-trend gate rather than requiring exact 4H/1H
        # agreement. 1H controls direction; 4H only blocks the opposite side.
        long_ok = bias == "BULL" and macro != "BEAR"
        short_ok = bias == "BEAR" and macro != "BULL"
        atr = max(float(i15["atr"]), close * 0.001)

        # 1) Trend pullback / continuation.
        if long_ok and i15["cdc_bull"] and i15["adx"] >= TREND_ADX_MIN and i15["rsi"] <= 72:
            trigger = i15["prev_close"] <= i15["bb_mid"] < close or close > i15["prev_high"]
            location_ok = close <= i15["bb_upper"] + 0.10 * atr
            if trigger and location_ok:
                return self._build("LONG", "trend_pullback", close, i15)

        if short_ok and i15["cdc_bear"] and i15["adx"] >= TREND_ADX_MIN and i15["rsi"] >= 28:
            trigger = i15["prev_close"] >= i15["bb_mid"] > close or close < i15["prev_low"]
            location_ok = close >= i15["bb_lower"] - 0.10 * atr
            if trigger and location_ok:
                return self._build("SHORT", "trend_pullback", close, i15)

        # 2) Early CDC transition. Opposite 4H direction remains prohibited.
        if macro != "BEAR" and i1["cdc_bull"] and i15["cdc_cross_up"] and close > i15["bb_mid"]:
            return self._build("LONG", "cdc_transition", close, i15)
        if macro != "BULL" and i1["cdc_bear"] and i15["cdc_cross_down"] and close < i15["bb_mid"]:
            return self._build("SHORT", "cdc_transition", close, i15)

        # 3) Bollinger breakout. Require supportive 1H direction and only a
        # modest volume expansion; the global chase guard still prevents
        # entering oversized breakout candles.
        volume_ratio = i15["volume"] / max(i15["vol_avg"], 1e-12)
        if long_ok and i15["cdc_bull"] and i15["prev_close"] <= i15["bb_upper"] < close:
            if volume_ratio >= BREAKOUT_VOLUME_MIN:
                return self._build("LONG", "bb_breakout", close, i15)
        if short_ok and i15["cdc_bear"] and i15["prev_close"] >= i15["bb_lower"] > close:
            if volume_ratio >= BREAKOUT_VOLUME_MIN:
                return self._build("SHORT", "bb_breakout", close, i15)

        self.last_signal = (
            f"WAIT macro={macro} bias={bias} "
            f"cdc={'BULL' if i15['cdc_bull'] else 'BEAR'} "
            f"adx={i15['adx']:.1f} vr={volume_ratio:.2f}"
        )
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
