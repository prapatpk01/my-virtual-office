"""Adaptive SMC v15: 15M EMA trend -> sweep -> CHoCH/BOS -> OB/FVG -> trigger."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json
import os
import time

TP1_R = float(os.getenv("V15_TP1_R", "1.0"))
TP_R = float(os.getenv("V15_TP2_R", "2.0"))
SL_BUFFER_ATR = float(os.getenv("V15_SL_BUFFER_ATR", "0.12"))
MIN_SL_PCT = float(os.getenv("V15_MIN_SL_PCT", "0.004"))
RISK_USDT = float(os.getenv("V15_RISK_USDT", "5.0"))
SETUP_EXPIRY_BARS = int(os.getenv("V15_SETUP_EXPIRY_BARS", "10"))


@dataclass
class SetupState:
    direction: str
    sweep_level: float
    age: int = 0
    choch_level: float = 0.0
    bos_level: float = 0.0
    zone_low: float = 0.0
    zone_high: float = 0.0
    zone_type: str = "NONE"


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
    sweep_level: float
    choch_level: float
    bos_level: float
    zone_low: float
    zone_high: float
    zone_type: str
    tp1_hit: bool = False
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
        risk_usdt: float = RISK_USDT,
    ):
        self.symbol = symbol
        self.margin_usdt = float(margin_usdt)
        self.leverage = int(leverage)
        self.paper = bool(paper)
        self.state_file = state_file
        self.execution_callback = execution_callback
        self.risk_usdt = float(risk_usdt)
        self.position: Optional[Position] = None
        self.setup: Optional[SetupState] = None
        self.last_signal = "WARMUP"
        self.counts = {k: 0 for k in ("scans", "entries", "trend", "sweep", "structure", "zone", "trigger", "expired")}
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
            if raw.get("setup"):
                self.setup = SetupState(**raw["setup"])
        except Exception:
            self.position = None
            self.setup = None

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "position": asdict(self.position) if self.position else None,
                    "setup": asdict(self.setup) if self.setup else None,
                },
                handle,
            )
        os.replace(temp, self.state_file)

    @staticmethod
    def _trend(i15: Dict) -> str:
        if float(i15["ema20"]) > float(i15["ema50"]):
            return "BULL"
        if float(i15["ema20"]) < float(i15["ema50"]):
            return "BEAR"
        return "NEUTRAL"

    def _reset(self, reason: str) -> None:
        self.setup = None
        self.last_signal = f"SETUP_RESET {reason}"
        self.save_state()

    def _debug(self, trend: str, i15: Dict, result: str, reason: str) -> str:
        setup = self.setup
        state = "NONE" if not setup else (
            f"{setup.direction},age={setup.age}/{SETUP_EXPIRY_BARS},"
            f"zone={setup.zone_type}[{setup.zone_low:.6f}-{setup.zone_high:.6f}]"
        )
        return (
            f"SMC15 symbol={self.symbol} | TREND[{trend}:ema20={float(i15['ema20']):.6f},ema50={float(i15['ema50']):.6f}] | "
            f"SWEEP[sell={int(bool(i15.get('recent_sell_sweep')))},buy={int(bool(i15.get('recent_buy_sweep')))}] | "
            f"STRUCTURE[bullCHoCH={int(bool(i15.get('bullish_choch')))},bearCHoCH={int(bool(i15.get('bearish_choch')))},"
            f"bullBOS={int(bool(i15.get('bullish_bos')))},bearBOS={int(bool(i15.get('bearish_bos')))}] | "
            f"ZONE[OBbull={int(bool(i15.get('ob_bull')))},OBbear={int(bool(i15.get('ob_bear')))},"
            f"FVGbull={int(bool(i15.get('fvg_bull')))},FVGbear={int(bool(i15.get('fvg_bear')))}] | "
            f"STATE[{state}] | RESULT[{result}:{reason}] | COUNTERS[{','.join(f'{k}={v}' for k, v in self.counts.items())}]"
        )

    def _start_or_age_setup(self, trend: str, i15: Dict) -> None:
        if self.setup:
            self.setup.age += 1
            if self.setup.age > SETUP_EXPIRY_BARS:
                self.counts["expired"] += 1
                self._reset("EXPIRED")
                return
            if self.setup.direction == "LONG" and trend != "BULL":
                self._reset("EMA_TREND_CHANGED")
                return
            if self.setup.direction == "SHORT" and trend != "BEAR":
                self._reset("EMA_TREND_CHANGED")
                return

        if self.setup is None:
            if trend == "BULL" and i15.get("recent_sell_sweep"):
                self.setup = SetupState("LONG", float(i15["last_swing_low"]))
            elif trend == "BEAR" and i15.get("recent_buy_sweep"):
                self.setup = SetupState("SHORT", float(i15["last_swing_high"]))
            else:
                self.counts["sweep"] += 1
                return

        setup = self.setup
        if setup.direction == "LONG":
            if i15.get("bullish_choch"):
                setup.choch_level = float(i15["last_swing_high"])
            if i15.get("bullish_bos"):
                setup.bos_level = float(i15["last_swing_high"])
            if i15.get("ob_bull") or i15.get("fvg_bull"):
                setup.zone_low = float(i15.get("bull_zone_low", 0.0))
                setup.zone_high = float(i15.get("bull_zone_high", 0.0))
                setup.zone_type = "OB+FVG" if i15.get("bull_zone_overlap") else "OB" if i15.get("ob_bull") else "FVG"
        else:
            if i15.get("bearish_choch"):
                setup.choch_level = float(i15["last_swing_low"])
            if i15.get("bearish_bos"):
                setup.bos_level = float(i15["last_swing_low"])
            if i15.get("ob_bear") or i15.get("fvg_bear"):
                setup.zone_low = float(i15.get("bear_zone_low", 0.0))
                setup.zone_high = float(i15.get("bear_zone_high", 0.0))
                setup.zone_type = "OB+FVG" if i15.get("bear_zone_overlap") else "OB" if i15.get("ob_bear") else "FVG"
        self.save_state()

    @staticmethod
    def _has_structure(setup: SetupState) -> bool:
        return setup.choch_level > 0 or setup.bos_level > 0

    @staticmethod
    def _in_zone(i15: Dict, setup: SetupState) -> bool:
        return (
            setup.zone_low > 0
            and setup.zone_high > setup.zone_low
            and float(i15["low"]) <= setup.zone_high
            and float(i15["high"]) >= setup.zone_low
        )

    @staticmethod
    def _trigger(i15: Dict, direction: str):
        if direction == "LONG":
            if i15.get("bull_engulf"):
                return "bullish_engulfing"
            if i15.get("bull_pin"):
                return "bullish_pinbar"
            if i15.get("break_high"):
                return "strong_break_high"
            if i15.get("bull_volume") and float(i15["close"]) > float(i15["open"]):
                return "strong_bull_volume"
        else:
            if i15.get("bear_engulf"):
                return "bearish_engulfing"
            if i15.get("bear_pin"):
                return "bearish_pinbar"
            if i15.get("break_low"):
                return "strong_break_low"
            if i15.get("bear_volume") and float(i15["close"]) < float(i15["open"]):
                return "strong_bear_volume"
        return ""

    def _build(self, i15: Dict, trigger: str):
        setup = self.setup
        if not setup:
            return None
        entry = float(i15["close"])
        atr = max(float(i15["atr"]), entry * 0.0005)
        minimum = entry * MIN_SL_PCT
        if setup.direction == "LONG":
            invalid = min(setup.sweep_level, setup.zone_low or setup.sweep_level)
            sl = min(invalid - SL_BUFFER_ATR * atr, entry - minimum)
            risk = entry - sl
            if risk <= 0:
                return None
            tp1, tp2 = entry + TP1_R * risk, entry + TP_R * risk
        else:
            invalid = max(setup.sweep_level, setup.zone_high or setup.sweep_level)
            sl = max(invalid + SL_BUFFER_ATR * atr, entry + minimum)
            risk = sl - entry
            if risk <= 0:
                return None
            tp1, tp2 = entry - TP1_R * risk, entry - TP_R * risk
        size = min(self.risk_usdt / risk, (self.margin_usdt * self.leverage) / max(entry, 1e-12))
        if size <= 0:
            return None
        return {
            "direction": setup.direction,
            "strategy": "smc_15m",
            "trigger": trigger,
            "entry": entry,
            "sl": sl,
            "tp": tp2,
            "tp1": tp1,
            "tp2": tp2,
            "size": size,
            "risk_usdt": size * risk,
            "sl_pct": 100 * risk / max(entry, 1e-12),
            "sweep_level": setup.sweep_level,
            "choch_level": setup.choch_level,
            "bos_level": setup.bos_level,
            "zone_low": setup.zone_low,
            "zone_high": setup.zone_high,
            "zone_type": setup.zone_type,
        }

    def _close(self, price: float, reason: str):
        position = self.position
        assert position
        pnl = (price - position.entry) * position.size if position.direction == "LONG" else (position.entry - price) * position.size
        initial_risk = abs(position.entry - position.initial_sl) * max(position.initial_size, 1e-12)
        r_multiple = pnl / initial_risk if initial_risk else 0.0
        payload = {
            "symbol": self.symbol,
            "direction": position.direction,
            "price": price,
            "entry": position.entry,
            "sl": position.sl,
            "tp": position.tp2,
            "tp1": position.tp1,
            "tp2": position.tp2,
            "size": position.size,
            "strategy": position.strategy,
            "trigger": position.trigger,
            "reason": reason,
            "pnl": pnl,
            "r_multiple": r_multiple,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_" + position.direction, payload)
        self.position = None
        self.save_state()
        self.last_signal = f"CLOSE {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R"
        return {"event": "CLOSE", **payload}

    def check_price(self, price: float):
        position = self.position
        if not position:
            return None
        if price <= position.sl if position.direction == "LONG" else price >= position.sl:
            return self._close(price, "BE" if position.be_moved else "SL")
        if not position.tp1_hit and (price >= position.tp1 if position.direction == "LONG" else price <= position.tp1):
            close_size = position.size * 0.5
            pnl = (price - position.entry) * close_size if position.direction == "LONG" else (position.entry - price) * close_size
            payload = {"symbol": self.symbol, "direction": position.direction, "price": price, "entry": position.entry, "size": close_size, "reason": "TP1", "pnl": pnl, "r_multiple": TP1_R}
            if self.execution_callback:
                self.execution_callback("CLOSE_PARTIAL", payload)
            position.size -= close_size
            position.tp1_hit = True
            position.sl = position.entry
            position.be_moved = True
            self.save_state()
            return {"event": "PARTIAL", **payload}
        if price >= position.tp2 if position.direction == "LONG" else price <= position.tp2:
            return self._close(price, "TP2")
        return None

    def reconcile_flat(self, price: float, reason: str = "EXCHANGE_CLOSED"):
        return self._close(price, reason) if self.position else None

    def on_bar(self, i15: Dict, _i1=None, _i4=None, price: float = 0.0):
        if not i15:
            self.last_signal = "WAIT INDICATOR_WARMUP"
            return None
        if self.position:
            event = self.check_price(price or float(i15["close"]))
            if event:
                return event
            position = self.position
            opposite = (position.direction == "LONG" and i15.get("bearish_choch")) or (position.direction == "SHORT" and i15.get("bullish_choch"))
            if opposite:
                return self._close(price or float(i15["close"]), "OPPOSITE_CHOCH")
            self.last_signal = f"MANAGE {position.direction} SL={position.sl:.6f} TP1={position.tp1:.6f} TP2={position.tp2:.6f}"
            return None

        self.counts["scans"] += 1
        trend = self._trend(i15)
        if trend == "NEUTRAL":
            self.counts["trend"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "EMA20_50_EQUAL")
            return None

        self._start_or_age_setup(trend, i15)
        setup = self.setup
        if not setup:
            self.last_signal = self._debug(trend, i15, "WAIT", "LIQUIDITY_SWEEP")
            return None
        if not self._has_structure(setup):
            self.counts["structure"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "CHOCH_OR_BOS")
            return None
        if setup.zone_low <= 0 or setup.zone_high <= setup.zone_low:
            self.counts["zone"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "OB_OR_FVG")
            return None
        if not self._in_zone(i15, setup):
            self.counts["zone"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "RETRACE_TO_ZONE")
            return None
        trigger = self._trigger(i15, setup.direction)
        if not trigger:
            self.counts["trigger"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "PRICE_ACTION")
            return None

        payload = self._build(i15, trigger)
        if not payload:
            self.last_signal = self._debug(trend, i15, "WAIT", "RISK_BUILD")
            return None
        payload["symbol"] = self.symbol
        if self.execution_callback:
            self.execution_callback("OPEN_" + payload["direction"], payload)
        self.position = Position(
            direction=payload["direction"], entry=payload["entry"], sl=payload["sl"], initial_sl=payload["sl"],
            tp=payload["tp2"], tp1=payload["tp1"], tp2=payload["tp2"], size=payload["size"], initial_size=payload["size"],
            strategy=payload["strategy"], trigger=payload["trigger"], opened_at=time.time(), sweep_level=payload["sweep_level"],
            choch_level=payload["choch_level"], bos_level=payload["bos_level"], zone_low=payload["zone_low"],
            zone_high=payload["zone_high"], zone_type=payload["zone_type"],
        )
        self.setup = None
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = f"OPEN {payload['direction']} trigger={trigger} entry={payload['entry']:.6f}"
        return {"event": "OPEN", **payload}
