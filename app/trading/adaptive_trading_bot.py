"""Adaptive Bot v12.1: simple Direction -> Location -> Trigger architecture."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json
import os
import time


EXTENSION_MAX_ATR = float(os.getenv("V12_EXTENSION_MAX_ATR", "1.00"))
BODY_MAX_ATR = float(os.getenv("V12_BODY_MAX_ATR", "1.20"))
TREND_ADX_MIN = float(os.getenv("V12_TREND_ADX_MIN", "12"))
BREAKOUT_VOLUME_MIN = float(os.getenv("V12_BREAKOUT_VOLUME_MIN", "1.05"))
BREAKOUT_HOLD_BUFFER_ATR = float(os.getenv("V12_BREAKOUT_HOLD_BUFFER_ATR", "0.05"))
STRUCTURE_ADX_MIN = float(os.getenv("V12_STRUCTURE_ADX_MIN", "10"))
EMA_PULLBACK_TOLERANCE_ATR = float(os.getenv("V12_EMA_PULLBACK_TOLERANCE_ATR", "0.18"))


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
        current_atr = max(float(i15["atr"]), entry * 0.003)
        if direction == "LONG":
            sl = min(float(i15["swing_low"]), entry - 1.2 * current_atr)
            tp = entry + 1.5 * (entry - sl)
        else:
            sl = max(float(i15["swing_high"]), entry + 1.2 * current_atr)
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
        macro = self._macro(i4)
        bias = self._bias(i1)
        close = float(i15["close"])
        open_ = float(i15.get("open", close))
        high = float(i15.get("high", close))
        low = float(i15.get("low", close))
        atr = max(float(i15["atr"]), close * 0.001)
        volume_ratio = float(i15["volume"]) / max(float(i15["vol_avg"]), 1e-12)

        if i15["extension_atr"] > EXTENSION_MAX_ATR or i15["body_atr"] > BODY_MAX_ATR:
            self.last_signal = (
                f"WAIT late ext={i15['extension_atr']:.2f}/{EXTENSION_MAX_ATR:.2f} "
                f"body={i15['body_atr']:.2f}/{BODY_MAX_ATR:.2f}"
            )
            return None

        long_ok = bias == "BULL" and macro != "BEAR"
        short_ok = bias == "BEAR" and macro != "BULL"

        bullish_bar = close > open_
        bearish_bar = close < open_
        ema8_bull = float(i15["ema8"]) > float(i15["ema13"])
        ema8_bear = float(i15["ema8"]) < float(i15["ema13"])

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

        ema13_distance = abs(close - float(i15["ema13"])) / atr
        if long_ok and i15["cdc_bull"] and ema8_bull and bullish_bar:
            touched_ema13 = low <= float(i15["ema13"]) + EMA_PULLBACK_TOLERANCE_ATR * atr
            reclaimed_ema13 = close > float(i15["ema13"])
            if touched_ema13 and reclaimed_ema13 and ema13_distance <= 0.45:
                return self._build("LONG", "ema13_pullback", close, i15)

        if short_ok and i15["cdc_bear"] and ema8_bear and bearish_bar:
            touched_ema13 = high >= float(i15["ema13"]) - EMA_PULLBACK_TOLERANCE_ATR * atr
            reclaimed_ema13 = close < float(i15["ema13"])
            if touched_ema13 and reclaimed_ema13 and ema13_distance <= 0.45:
                return self._build("SHORT", "ema13_pullback", close, i15)

        if long_ok and i15["cdc_bull"] and ema8_bull and i15["adx"] >= STRUCTURE_ADX_MIN:
            higher_low = low > float(i15["prev_low"])
            micro_break = close > float(i15["prev_high"])
            above_value = close > float(i15["ema20"])
            if higher_low and micro_break and above_value and bullish_bar:
                return self._build("LONG", "structure_pullback", close, i15)

        if short_ok and i15["cdc_bear"] and ema8_bear and i15["adx"] >= STRUCTURE_ADX_MIN:
            lower_high = high < float(i15["prev_high"])
            micro_break = close < float(i15["prev_low"])
            below_value = close < float(i15["ema20"])
            if lower_high and micro_break and below_value and bearish_bar:
                return self._build("SHORT", "structure_pullback", close, i15)

        if (
            macro != "BEAR"
            and i1["cdc_bull"]
            and i15["cdc_cross_up"]
            and ema8_bull
            and close > i15["bb_mid"]
        ):
            return self._build("LONG", "cdc_transition", close, i15)

        if (
            macro != "BULL"
            and i1["cdc_bear"]
            and i15["cdc_cross_down"]
            and ema8_bear
            and close < i15["bb_mid"]
        ):
            return self._build("SHORT", "cdc_transition", close, i15)

        long_breakout_hold = (
            i15["prev_close"] <= i15["bb_upper"]
            and close >= i15["bb_upper"] + BREAKOUT_HOLD_BUFFER_ATR * atr
        )
        short_breakout_hold = (
            i15["prev_close"] >= i15["bb_lower"]
            and close <= i15["bb_lower"] - BREAKOUT_HOLD_BUFFER_ATR * atr
        )
        if long_ok and i15["cdc_bull"] and ema8_bull and long_breakout_hold:
            if volume_ratio >= BREAKOUT_VOLUME_MIN:
                return self._build("LONG", "bb_breakout_hold", close, i15)

        if short_ok and i15["cdc_bear"] and ema8_bear and short_breakout_hold:
            if volume_ratio >= BREAKOUT_VOLUME_MIN:
                return self._build("SHORT", "bb_breakout_hold", close, i15)

        reasons = []
        if bias == "NEUTRAL":
            reasons.append("1H_BIAS_NEUTRAL")
        if macro == "BEAR" and i15["cdc_bull"]:
            reasons.append("4H_BLOCK_LONG")
        if macro == "BULL" and i15["cdc_bear"]:
            reasons.append("4H_BLOCK_SHORT")
        if i15["adx"] < STRUCTURE_ADX_MIN:
            reasons.append(f"ADX_LOW:{i15['adx']:.1f}")
        if volume_ratio < BREAKOUT_VOLUME_MIN:
            reasons.append(f"VOL_LOW:{volume_ratio:.2f}")
        if not reasons:
            reasons.append("NO_TRIGGER")

        self.last_signal = (
            f"WAIT macro={macro} bias={bias} "
            f"cdc={'BULL' if i15['cdc_bull'] else 'BEAR'} "
            f"adx={i15['adx']:.1f} vr={volume_ratio:.2f} "
            f"reason={','.join(reasons)}"
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
            cdc_flip = (
                p.direction == "LONG" and i15["cdc_bear"]
            ) or (
                p.direction == "SHORT" and i15["cdc_bull"]
            )
            if hit_sl or hit_tp or cdc_flip:
                reason = "SL" if hit_sl else ("TP" if hit_tp else "CDC_FLIP")
                pnl = (
                    (price - p.entry) * p.size
                    if p.direction == "LONG"
                    else (p.entry - price) * p.size
                )
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


_TRADEABLE_REGIMES = frozenset({"Trend", "Transition", "Breakout"})


class ExpectancyEngine:
    pass
