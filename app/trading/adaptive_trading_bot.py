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
    """Compact v12 engine with three entry paths and no score/voting stack."""

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
            with open(self.state_file, encoding="utf-8") as state_handle:
                raw = json.load(state_handle)
            if raw.get("position"):
                self.position = Position(**raw["position"])
        except Exception:
            self.position = None

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as state_handle:
            json.dump({"position": asdict(self.position) if self.position else None}, state_handle)

    @staticmethod
    def _macro(indicators_4h: Dict) -> str:
        close = indicators_4h["close"]
        ema20 = indicators_4h["ema20"]
        ema50 = indicators_4h["ema50"]
        if close > ema20 > ema50:
            return "BULL"
        if close < ema20 < ema50:
            return "BEAR"
        return "NEUTRAL"

    @staticmethod
    def _bias(indicators_1h: Dict) -> str:
        if indicators_1h["cdc_bull"] and indicators_1h["close"] > indicators_1h["ema20"]:
            return "BULL"
        if indicators_1h["cdc_bear"] and indicators_1h["close"] < indicators_1h["ema20"]:
            return "BEAR"
        return "NEUTRAL"

    def _signal(self, indicators_15m: Dict, indicators_1h: Dict, indicators_4h: Dict) -> Optional[Dict]:
        macro = self._macro(indicators_4h)
        bias = self._bias(indicators_1h)
        close = indicators_15m["close"]

        # Hard late-entry guard. No score stacking.
        if indicators_15m["extension_atr"] > 0.75 or indicators_15m["body_atr"] > 0.85:
            self.last_signal = (
                f"WAIT late ext={indicators_15m['extension_atr']:.2f}ATR "
                f"body={indicators_15m['body_atr']:.2f}ATR"
            )
            return None

        long_allowed = macro == "BULL" and bias == "BULL"
        short_allowed = macro == "BEAR" and bias == "BEAR"

        # 1) Trend pullback: HTF direction + CDC + immediate price-action trigger.
        if long_allowed and indicators_15m["cdc_bull"] and indicators_15m["adx"] >= 14 and indicators_15m["rsi"] <= 70:
            trigger = (
                indicators_15m["prev_close"] <= indicators_15m["bb_mid"] < close
                or close > indicators_15m["prev_high"]
            )
            if trigger and close <= indicators_15m["bb_upper"]:
                return self._build("LONG", "trend_pullback", close, indicators_15m)

        if short_allowed and indicators_15m["cdc_bear"] and indicators_15m["adx"] >= 14 and indicators_15m["rsi"] >= 30:
            trigger = (
                indicators_15m["prev_close"] >= indicators_15m["bb_mid"] > close
                or close < indicators_15m["prev_low"]
            )
            if trigger and close >= indicators_15m["bb_lower"]:
                return self._build("SHORT", "trend_pullback", close, indicators_15m)

        # 2) Early transition: CDC cross + Bollinger mid reclaim.
        if macro != "BEAR" and indicators_1h["cdc_bull"] and indicators_15m["cdc_cross_up"] and close > indicators_15m["bb_mid"]:
            return self._build("LONG", "cdc_transition", close, indicators_15m)
        if macro != "BULL" and indicators_1h["cdc_bear"] and indicators_15m["cdc_cross_down"] and close < indicators_15m["bb_mid"]:
            return self._build("SHORT", "cdc_transition", close, indicators_15m)

        # 3) Bollinger breakout: HTF direction + CDC + volume expansion.
        volume_ratio = indicators_15m["volume"] / max(indicators_15m["vol_avg"], 1e-12)
        if long_allowed and indicators_15m["cdc_bull"] and indicators_15m["prev_close"] <= indicators_15m["bb_upper"] < close and volume_ratio >= 1.15:
            return self._build("LONG", "bb_breakout", close, indicators_15m)
        if short_allowed and indicators_15m["cdc_bear"] and indicators_15m["prev_close"] >= indicators_15m["bb_lower"] > close and volume_ratio >= 1.15:
            return self._build("SHORT", "bb_breakout", close, indicators_15m)

        self.last_signal = (
            f"WAIT macro={macro} bias={bias} "
            f"cdc={'BULL' if indicators_15m['cdc_bull'] else 'BEAR'}"
        )
        return None

    def _build(self, direction: str, strategy: str, entry: float, indicators_15m: Dict) -> Dict:
        current_atr = max(indicators_15m["atr"], entry * 0.003)
        if direction == "LONG":
            sl = min(indicators_15m["swing_low"], entry - 1.2 * current_atr)
            risk = entry - sl
            tp = entry + 1.5 * risk
        else:
            sl = max(indicators_15m["swing_high"], entry + 1.2 * current_atr)
            risk = sl - entry
            tp = entry - 1.5 * risk

        notional = self.margin_usdt * self.leverage
        size = notional / max(entry, 1e-12)
        return {
            "direction": direction,
            "strategy": strategy,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": size,
        }

    def on_bar(self, indicators_15m: Dict, indicators_1h: Dict, indicators_4h: Dict, price: float) -> Optional[Dict]:
        if not indicators_15m or not indicators_1h or not indicators_4h:
            self.last_signal = "WAIT indicator warmup"
            return None

        if self.position:
            position = self.position
            hit_sl = price <= position.sl if position.direction == "LONG" else price >= position.sl
            hit_tp = price >= position.tp if position.direction == "LONG" else price <= position.tp
            cdc_flip = (
                position.direction == "LONG" and indicators_15m["cdc_bear"]
            ) or (
                position.direction == "SHORT" and indicators_15m["cdc_bull"]
            )
            if hit_sl or hit_tp or cdc_flip:
                reason = "SL" if hit_sl else ("TP" if hit_tp else "CDC_FLIP")
                payload = {
                    "symbol": self.symbol,
                    "direction": position.direction,
                    "price": price,
                    "size": position.size,
                    "reason": reason,
                }
                if self.execution_callback:
                    self.execution_callback("CLOSE_" + position.direction, payload)
                self.position = None
                self.save_state()
                self.last_signal = f"CLOSE {position.direction} {reason}"
                return {"event": "CLOSE", **payload}
            self.last_signal = f"MANAGE {position.direction}"
            return None

        signal = self._signal(indicators_15m, indicators_1h, indicators_4h)
        if not signal:
            return None

        payload = {"symbol": self.symbol, **signal}
        if self.execution_callback:
            self.execution_callback("OPEN_" + signal["direction"], payload)
        self.position = Position(
            signal["direction"],
            signal["entry"],
            signal["sl"],
            signal["tp"],
            signal["size"],
            signal["strategy"],
            time.time(),
        )
        self.last_signal = f"OPEN {signal['direction']} {signal['strategy']}"
        self.save_state()
        return {"event": "OPEN", **payload}


# Compatibility exports used by older imports. They are intentionally simple.
_TRADEABLE_REGIMES = frozenset({"Trend", "Transition", "Breakout"})


class ExpectancyEngine:
    pass
