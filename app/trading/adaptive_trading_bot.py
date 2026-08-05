"""Adaptive Momentum v3: EMA cross + MACD expansion + ADX/CHOP + structure/location."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json
import os
import time

TP1_R = float(os.getenv("MOM_TP1_R", "1.0"))
TP2_R = float(os.getenv("MOM_TP2_R", "2.0"))
TP_R = TP2_R
SL_ATR = float(os.getenv("MOM_SL_ATR", "1.0"))
MIN_SL_PCT = float(os.getenv("MOM_MIN_SL_PCT", "0.004"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
ADX_MIN = float(os.getenv("MOM_ADX_MIN", "18"))
CHOP_MAX = float(os.getenv("MOM_CHOP_MAX", "52"))
LOCATION_MAX_ATR = float(os.getenv("MOM_LOCATION_MAX_ATR", "0.8"))
COOLDOWN_BARS = int(os.getenv("MOM_COOLDOWN_BARS", "3"))


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
        self.counts = {key: 0 for key in (
            "scans", "entries", "cooldown", "trend", "cross", "macd",
            "hist", "adx", "chop", "structure", "location"
        )}
        self.load_state()

    @property
    def position_open(self) -> bool:
        return self.position is not None

    def load_state(self) -> None:
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            raw = json.load(open(self.state_file, encoding="utf-8"))
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
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({
                "position": asdict(self.position) if self.position else None,
                "cooldown_remaining": self.cooldown_remaining,
            }, handle)
        os.replace(temp, self.state_file)

    def _debug(self, i15: Dict, result: str, reason: str) -> str:
        return (
            f"MOMENTUM_V3 symbol={self.symbol} | "
            f"TREND[20={float(i15['ema20']):.6f},50={float(i15['ema50']):.6f},bull={int(bool(i15.get('trend_bull')))},bear={int(bool(i15.get('trend_bear')))}] | "
            f"CROSS[up2={int(bool(i15.get('ema_cross_up_recent')))},down2={int(bool(i15.get('ema_cross_down_recent')))},8={float(i15['ema8']):.6f},13={float(i15['ema13']):.6f}] | "
            f"MACD[line={float(i15['macd']):+.6f},signal={float(i15['macd_signal']):+.6f},hist={float(i15['macd_hist']):+.6f},"
            f"expandUp2={int(bool(i15.get('macd_hist_expand_up_2')))},expandDown2={int(bool(i15.get('macd_hist_expand_down_2')))}] | "
            f"ADX[{float(i15['adx']):.1f}>={ADX_MIN:.1f},rising={int(bool(i15.get('adx_rising')))}] | "
            f"CHOP[{float(i15['chop']):.1f}<={CHOP_MAX:.1f}] | "
            f"STRUCTURE[long={int(bool(i15.get('structure_long')))},short={int(bool(i15.get('structure_short')))},"
            f"high={float(i15['recent_high']):.6f},low={float(i15['recent_low']):.6f}] | "
            f"LOCATION[dist={float(i15['distance_ema13_atr']):.2f}/{LOCATION_MAX_ATR:.2f}ATR] | "
            f"COOLDOWN[{self.cooldown_remaining}] | RESULT[{result}:{reason}] | "
            f"COUNTERS[{','.join(f'{key}={value}' for key, value in self.counts.items())}]"
        )

    def _build(self, i15: Dict, direction: str):
        entry = float(i15["close"])
        atr = max(float(i15["atr"]), entry * 0.0005)
        minimum = entry * MIN_SL_PCT
        if direction == "LONG":
            swing_sl = float(i15.get("recent_low", entry - atr))
            sl = min(swing_sl, entry - SL_ATR * atr, entry - minimum)
            risk = entry - sl
            tp1, tp2 = entry + TP1_R * risk, entry + TP2_R * risk
        else:
            swing_sl = float(i15.get("recent_high", entry + atr))
            sl = max(swing_sl, entry + SL_ATR * atr, entry + minimum)
            risk = sl - entry
            tp1, tp2 = entry - TP1_R * risk, entry - TP2_R * risk
        if risk <= 0:
            return None
        size = min(self.risk_usdt / risk, (self.margin_usdt * self.leverage) / max(entry, 1e-12))
        if size <= 0:
            return None
        trigger = "ema8_13_cross_macd_expand_structure"
        return {
            "direction": direction, "strategy": "momentum_v3_structure_confirmed",
            "trigger": trigger, "entry": entry, "sl": sl, "tp": tp2,
            "tp1": tp1, "tp2": tp2, "size": size,
            "risk_usdt": size * risk, "sl_pct": 100 * risk / max(entry, 1e-12),
            "ema8": float(i15["ema8"]), "ema13": float(i15["ema13"]),
            "ema20": float(i15["ema20"]), "ema50": float(i15["ema50"]),
            "macd": float(i15["macd"]), "macd_signal": float(i15["macd_signal"]),
            "macd_hist": float(i15["macd_hist"]), "adx": float(i15["adx"]),
            "chop": float(i15["chop"]),
            "distance_ema13_atr": float(i15["distance_ema13_atr"]),
            "structure_level": float(i15["recent_high"] if direction == "LONG" else i15["recent_low"]),
        }

    def _close(self, price: float, reason: str):
        position = self.position
        assert position
        pnl = ((price - position.entry) * position.size if position.direction == "LONG"
               else (position.entry - price) * position.size)
        initial_risk = abs(position.entry - position.initial_sl) * max(position.initial_size, 1e-12)
        r_multiple = pnl / initial_risk if initial_risk else 0.0
        payload = {
            "symbol": self.symbol, "direction": position.direction, "price": price,
            "entry": position.entry, "sl": position.sl, "tp": position.tp2,
            "tp1": position.tp1, "tp2": position.tp2, "size": position.size,
            "strategy": position.strategy, "trigger": position.trigger,
            "reason": reason, "pnl": pnl, "r_multiple": r_multiple,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_" + position.direction, payload)
        self.position = None
        if reason == "MOMENTUM_FLIP":
            self.cooldown_remaining = COOLDOWN_BARS
        self.save_state()
        self.last_signal = f"CLOSE {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R cooldown={self.cooldown_remaining}"
        return {"event": "CLOSE", **payload}

    def check_price(self, price: float):
        position = self.position
        if not position:
            return None
        if ((position.direction == "LONG" and price <= position.sl)
                or (position.direction == "SHORT" and price >= position.sl)):
            return self._close(price, "BE" if position.be_moved else "SL")
        if not position.tp1_hit and (
            (position.direction == "LONG" and price >= position.tp1)
            or (position.direction == "SHORT" and price <= position.tp1)
        ):
            close_size = position.size * 0.5
            pnl = ((price - position.entry) * close_size if position.direction == "LONG"
                   else (position.entry - price) * close_size)
            payload = {
                "symbol": self.symbol, "direction": position.direction,
                "price": price, "entry": position.entry, "size": close_size,
                "reason": "TP1", "pnl": pnl, "r_multiple": TP1_R,
            }
            if self.execution_callback:
                self.execution_callback("CLOSE_PARTIAL", payload)
            position.size -= close_size
            position.tp1_hit = True
            position.sl = position.entry
            position.be_moved = True
            self.save_state()
            return {"event": "PARTIAL", **payload}
        if ((position.direction == "LONG" and price >= position.tp2)
                or (position.direction == "SHORT" and price <= position.tp2)):
            return self._close(price, "TP2")
        return None

    def reconcile_flat(self, price: float, reason: str = "EXCHANGE_CLOSED"):
        return self._close(price, reason) if self.position else None

    def on_bar(self, i15: Dict, _i1=None, _i4=None, price: float = 0.0):
        if not i15:
            self.last_signal = "WAIT INDICATOR_WARMUP"
            return None
        if i15.get("schema") != "adaptive-momentum-v3-15m":
            raise RuntimeError(f"MOMENTUM_V3_SCHEMA_MISMATCH: {i15.get('schema')}")

        if self.position:
            event = self.check_price(price or float(i15["close"]))
            if event:
                return event
            position = self.position
            close_price = price or float(i15["close"])
            if position.direction == "LONG":
                ema_flip = bool(i15.get("ema_cross_down"))
                confirmed_macd_weakness = bool(
                    i15.get("macd_hist_weaken_long_2")
                    and float(i15["macd"]) <= float(i15["macd_signal"])
                )
            else:
                ema_flip = bool(i15.get("ema_cross_up"))
                confirmed_macd_weakness = bool(
                    i15.get("macd_hist_weaken_short_2")
                    and float(i15["macd"]) >= float(i15["macd_signal"])
                )
            if ema_flip or confirmed_macd_weakness:
                return self._close(close_price, "MOMENTUM_FLIP")
            self.last_signal = (
                f"MANAGE {position.direction} SL={position.sl:.6f} "
                f"TP1={position.tp1:.6f} TP2={position.tp2:.6f} "
                f"emaFlip={int(ema_flip)} macdWeak={int(confirmed_macd_weakness)}"
            )
            return None

        self.counts["scans"] += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.counts["cooldown"] += 1
            self.save_state()
            self.last_signal = self._debug(i15, "WAIT", "COOLDOWN")
            return None

        long_trend = bool(i15.get("trend_bull"))
        short_trend = bool(i15.get("trend_bear"))
        if not long_trend and not short_trend:
            self.counts["trend"] += 1
            self.last_signal = self._debug(i15, "WAIT", "EMA20_50_TREND")
            return None

        direction = "LONG" if long_trend else "SHORT"
        cross_ok = bool(i15.get("ema_cross_up_recent")) if direction == "LONG" else bool(i15.get("ema_cross_down_recent"))
        alignment_ok = bool(i15.get("entry_bull")) if direction == "LONG" else bool(i15.get("entry_bear"))
        if not cross_ok or not alignment_ok:
            self.counts["cross"] += 1
            self.last_signal = self._debug(i15, "WAIT", "EMA8_13_CROSS")
            return None

        macd_ok = bool(i15.get("macd_bull")) if direction == "LONG" else bool(i15.get("macd_bear"))
        if not macd_ok:
            self.counts["macd"] += 1
            self.last_signal = self._debug(i15, "WAIT", "MACD_SIGNAL")
            return None

        hist_ok = bool(i15.get("macd_hist_expand_up_2")) if direction == "LONG" else bool(i15.get("macd_hist_expand_down_2"))
        if not hist_ok:
            self.counts["hist"] += 1
            self.last_signal = self._debug(i15, "WAIT", "MACD_HIST_2BAR_EXPANSION")
            return None

        if float(i15["adx"]) < ADX_MIN or not bool(i15.get("adx_rising")):
            self.counts["adx"] += 1
            self.last_signal = self._debug(i15, "WAIT", "ADX_NOT_STRONG_RISING")
            return None

        if float(i15["chop"]) > CHOP_MAX:
            self.counts["chop"] += 1
            self.last_signal = self._debug(i15, "WAIT", "CHOP_TOO_HIGH")
            return None

        structure_ok = bool(i15.get("structure_long")) if direction == "LONG" else bool(i15.get("structure_short"))
        if not structure_ok:
            self.counts["structure"] += 1
            self.last_signal = self._debug(i15, "WAIT", "STRUCTURE_BREAK")
            return None

        location_ok = bool(i15.get("location_long")) if direction == "LONG" else bool(i15.get("location_short"))
        if not location_ok or float(i15["distance_ema13_atr"]) > LOCATION_MAX_ATR:
            self.counts["location"] += 1
            self.last_signal = self._debug(i15, "WAIT", "LOCATION")
            return None

        payload = self._build(i15, direction)
        if not payload:
            self.last_signal = self._debug(i15, "WAIT", "RISK_BUILD")
            return None
        payload["symbol"] = self.symbol
        if self.execution_callback:
            self.execution_callback("OPEN_" + direction, payload)
        self.position = Position(
            direction=direction, entry=payload["entry"], sl=payload["sl"],
            initial_sl=payload["sl"], tp=payload["tp2"], tp1=payload["tp1"],
            tp2=payload["tp2"], size=payload["size"], initial_size=payload["size"],
            strategy=payload["strategy"], trigger=payload["trigger"], opened_at=time.time(),
        )
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = self._debug(i15, "ENTRY", direction)
        return {"event": "OPEN", **payload}
