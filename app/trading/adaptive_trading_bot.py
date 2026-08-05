"""Adaptive SMC v15: 15M EMA trend -> sweep -> structure -> zone -> trigger."""
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
    phase: str = "WAIT_STRUCTURE"
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
    def __init__(self, symbol: str, margin_usdt: float = 20.0, leverage: int = 20,
                 paper: bool = True, state_file: str = "",
                 execution_callback: Optional[Callable] = None,
                 risk_usdt: float = RISK_USDT):
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
        self.counts = {k: 0 for k in (
            "scans", "entries", "trend", "sweep", "structure",
            "zone", "retrace", "trigger", "expired"
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
            p, s = raw.get("position"), raw.get("setup")
            if p:
                self.position = Position(**p)
            if s:
                s.setdefault("phase", self._infer_phase_from_raw(s))
                self.setup = SetupState(**s)
        except Exception:
            self.position = None
            self.setup = None

    @staticmethod
    def _infer_phase_from_raw(raw: Dict) -> str:
        if float(raw.get("zone_high", 0)) > float(raw.get("zone_low", 0)) > 0:
            return "WAIT_RETRACE"
        if float(raw.get("choch_level", 0)) > 0 or float(raw.get("bos_level", 0)) > 0:
            return "WAIT_ZONE"
        return "WAIT_STRUCTURE"

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({
                "position": asdict(self.position) if self.position else None,
                "setup": asdict(self.setup) if self.setup else None,
            }, handle)
        os.replace(temp, self.state_file)

    @staticmethod
    def _trend(i15: Dict) -> str:
        e20, e50 = float(i15["ema20"]), float(i15["ema50"])
        return "BULL" if e20 > e50 else "BEAR" if e20 < e50 else "NEUTRAL"

    def _reset(self, reason: str) -> None:
        self.setup = None
        self.last_signal = f"SETUP_RESET {reason}"
        self.save_state()

    def _expected_sweep(self, trend: str) -> tuple[str, bool]:
        if trend == "BULL":
            return "SELL_SIDE_LOW", bool(self._last_i15.get("recent_sell_sweep"))
        if trend == "BEAR":
            return "BUY_SIDE_HIGH", bool(self._last_i15.get("recent_buy_sweep"))
        return "NONE", False

    def _stage_status(self, trend: str, i15: Dict) -> str:
        expected, expected_seen = self._expected_sweep(trend)
        s = self.setup
        sweep_saved = s is not None
        structure_saved = bool(s and (s.choch_level > 0 or s.bos_level > 0))
        zone_saved = bool(s and s.zone_high > s.zone_low > 0)
        in_zone = bool(s and self._in_zone(i15, s))
        trigger = self._trigger(i15, s.direction) if s and in_zone else ""
        return (
            f"STAGES[trend=PASS,expectedSweep={expected},currentSweep={int(expected_seen)},"
            f"savedSweep={int(sweep_saved)},structure={int(structure_saved)},"
            f"zone={int(zone_saved)},retrace={int(in_zone)},trigger={trigger or '0'}]"
        )

    def _debug(self, trend: str, i15: Dict, result: str, reason: str) -> str:
        s = self.setup
        state = "WAIT_SWEEP" if not s else (
            f"{s.direction}:{s.phase},age={s.age}/{SETUP_EXPIRY_BARS},"
            f"sweep={s.sweep_level:.6f},choch={s.choch_level:.6f},"
            f"bos={s.bos_level:.6f},zone={s.zone_type}[{s.zone_low:.6f}-{s.zone_high:.6f}]"
        )
        quality = (
            f"QUALITY[adx={float(i15.get('adx', 0)):.1f},"
            f"chop={float(i15.get('chop', 0)):.1f}]"
            if "adx" in i15 or "chop" in i15 else "QUALITY[soft-filter-external]"
        )
        return (
            f"SMC15 symbol={self.symbol} | "
            f"TREND[{trend}:ema20={float(i15['ema20']):.6f},ema50={float(i15['ema50']):.6f}] | "
            f"CURRENT[ sellSweep={int(bool(i15.get('recent_sell_sweep')))},"
            f"buySweep={int(bool(i15.get('recent_buy_sweep')))},"
            f"bullCHoCH={int(bool(i15.get('bullish_choch')))},bearCHoCH={int(bool(i15.get('bearish_choch')))},"
            f"bullBOS={int(bool(i15.get('bullish_bos')))},bearBOS={int(bool(i15.get('bearish_bos')))},"
            f"OBbull={int(bool(i15.get('ob_bull')))},OBbear={int(bool(i15.get('ob_bear')))},"
            f"FVGbull={int(bool(i15.get('fvg_bull')))},FVGbear={int(bool(i15.get('fvg_bear')))}] | "
            f"{self._stage_status(trend, i15)} | {quality} | STATE[{state}] | "
            f"RESULT[{result}:{reason}] | COUNTERS[{','.join(f'{k}={v}' for k, v in self.counts.items())}]"
        )

    def _start_or_advance_setup(self, trend: str, i15: Dict) -> None:
        if self.setup:
            self.setup.age += 1
            if self.setup.age > SETUP_EXPIRY_BARS:
                self.counts["expired"] += 1
                self._reset("EXPIRED")
                return
            if (self.setup.direction == "LONG" and trend != "BULL") or (
                self.setup.direction == "SHORT" and trend != "BEAR"
            ):
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

        s = self.setup
        if s.direction == "LONG":
            if i15.get("bullish_choch"):
                s.choch_level = float(i15["last_swing_high"])
            if i15.get("bullish_bos"):
                s.bos_level = float(i15["last_swing_high"])
            if s.choch_level > 0 or s.bos_level > 0:
                s.phase = "WAIT_ZONE"
            if s.phase in {"WAIT_ZONE", "WAIT_RETRACE", "WAIT_TRIGGER"} and (
                i15.get("ob_bull") or i15.get("fvg_bull")
            ):
                s.zone_low = float(i15.get("bull_zone_low", 0))
                s.zone_high = float(i15.get("bull_zone_high", 0))
                s.zone_type = "OB+FVG" if i15.get("bull_zone_overlap") else "OB" if i15.get("ob_bull") else "FVG"
                if s.zone_high > s.zone_low > 0:
                    s.phase = "WAIT_RETRACE"
        else:
            if i15.get("bearish_choch"):
                s.choch_level = float(i15["last_swing_low"])
            if i15.get("bearish_bos"):
                s.bos_level = float(i15["last_swing_low"])
            if s.choch_level > 0 or s.bos_level > 0:
                s.phase = "WAIT_ZONE"
            if s.phase in {"WAIT_ZONE", "WAIT_RETRACE", "WAIT_TRIGGER"} and (
                i15.get("ob_bear") or i15.get("fvg_bear")
            ):
                s.zone_low = float(i15.get("bear_zone_low", 0))
                s.zone_high = float(i15.get("bear_zone_high", 0))
                s.zone_type = "OB+FVG" if i15.get("bear_zone_overlap") else "OB" if i15.get("ob_bear") else "FVG"
                if s.zone_high > s.zone_low > 0:
                    s.phase = "WAIT_RETRACE"
        self.save_state()

    @staticmethod
    def _in_zone(i15: Dict, s: SetupState) -> bool:
        return s.zone_high > s.zone_low > 0 and float(i15["low"]) <= s.zone_high and float(i15["high"]) >= s.zone_low

    @staticmethod
    def _trigger(i15: Dict, direction: str) -> str:
        if direction == "LONG":
            if i15.get("bull_engulf"): return "bullish_engulfing"
            if i15.get("bull_pin"): return "bullish_pinbar"
            if i15.get("break_high"): return "strong_break_high"
            if i15.get("bull_volume") and float(i15["close"]) > float(i15["open"]): return "strong_bull_volume"
        else:
            if i15.get("bear_engulf"): return "bearish_engulfing"
            if i15.get("bear_pin"): return "bearish_pinbar"
            if i15.get("break_low"): return "strong_break_low"
            if i15.get("bear_volume") and float(i15["close"]) < float(i15["open"]): return "strong_bear_volume"
        return ""

    def _build(self, i15: Dict, trigger: str):
        s = self.setup
        if not s: return None
        entry = float(i15["close"])
        atr = max(float(i15["atr"]), entry * 0.0005)
        minimum = entry * MIN_SL_PCT
        if s.direction == "LONG":
            invalid = min(s.sweep_level, s.zone_low or s.sweep_level)
            sl = min(invalid - SL_BUFFER_ATR * atr, entry - minimum)
            risk = entry - sl
            if risk <= 0: return None
            tp1, tp2 = entry + TP1_R * risk, entry + TP_R * risk
        else:
            invalid = max(s.sweep_level, s.zone_high or s.sweep_level)
            sl = max(invalid + SL_BUFFER_ATR * atr, entry + minimum)
            risk = sl - entry
            if risk <= 0: return None
            tp1, tp2 = entry - TP1_R * risk, entry - TP_R * risk
        size = min(self.risk_usdt / risk, (self.margin_usdt * self.leverage) / max(entry, 1e-12))
        if size <= 0: return None
        return {"direction": s.direction, "strategy": "smc_15m", "trigger": trigger,
                "entry": entry, "sl": sl, "tp": tp2, "tp1": tp1, "tp2": tp2,
                "size": size, "risk_usdt": size * risk, "sl_pct": 100 * risk / max(entry, 1e-12),
                "sweep_level": s.sweep_level, "choch_level": s.choch_level,
                "bos_level": s.bos_level, "zone_low": s.zone_low,
                "zone_high": s.zone_high, "zone_type": s.zone_type}

    def _close(self, price: float, reason: str):
        p = self.position
        assert p
        pnl = (price - p.entry) * p.size if p.direction == "LONG" else (p.entry - price) * p.size
        initial_risk = abs(p.entry - p.initial_sl) * max(p.initial_size, 1e-12)
        r_multiple = pnl / initial_risk if initial_risk else 0.0
        payload = {"symbol": self.symbol, "direction": p.direction, "price": price,
                   "entry": p.entry, "sl": p.sl, "tp": p.tp2, "tp1": p.tp1,
                   "tp2": p.tp2, "size": p.size, "strategy": p.strategy,
                   "trigger": p.trigger, "reason": reason, "pnl": pnl,
                   "r_multiple": r_multiple}
        if self.execution_callback: self.execution_callback("CLOSE_" + p.direction, payload)
        self.position = None
        self.save_state()
        self.last_signal = f"CLOSE {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R"
        return {"event": "CLOSE", **payload}

    def check_price(self, price: float):
        p = self.position
        if not p: return None
        if price <= p.sl if p.direction == "LONG" else price >= p.sl:
            return self._close(price, "BE" if p.be_moved else "SL")
        if not p.tp1_hit and (price >= p.tp1 if p.direction == "LONG" else price <= p.tp1):
            close_size = p.size * 0.5
            pnl = (price - p.entry) * close_size if p.direction == "LONG" else (p.entry - price) * close_size
            payload = {"symbol": self.symbol, "direction": p.direction, "price": price,
                       "entry": p.entry, "size": close_size, "reason": "TP1",
                       "pnl": pnl, "r_multiple": TP1_R}
            if self.execution_callback: self.execution_callback("CLOSE_PARTIAL", payload)
            p.size -= close_size
            p.tp1_hit = True
            p.sl = p.entry
            p.be_moved = True
            self.save_state()
            return {"event": "PARTIAL", **payload}
        if price >= p.tp2 if p.direction == "LONG" else price <= p.tp2:
            return self._close(price, "TP2")
        return None

    def reconcile_flat(self, price: float, reason: str = "EXCHANGE_CLOSED"):
        return self._close(price, reason) if self.position else None

    def on_bar(self, i15: Dict, _i1=None, _i4=None, price: float = 0.0):
        if not i15:
            self.last_signal = "WAIT INDICATOR_WARMUP"
            return None
        self._last_i15 = i15
        current_price = price or float(i15["close"])
        if self.position:
            event = self.check_price(current_price)
            if event: return event
            p = self.position
            opposite = (p.direction == "LONG" and i15.get("bearish_choch")) or (p.direction == "SHORT" and i15.get("bullish_choch"))
            if opposite: return self._close(current_price, "OPPOSITE_CHOCH")
            self.last_signal = f"MANAGE {p.direction} SL={p.sl:.6f} TP1={p.tp1:.6f} TP2={p.tp2:.6f}"
            return None

        self.counts["scans"] += 1
        trend = self._trend(i15)
        if trend == "NEUTRAL":
            self.counts["trend"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "EMA20_50_EQUAL")
            return None

        self._start_or_advance_setup(trend, i15)
        s = self.setup
        if not s:
            self.last_signal = self._debug(trend, i15, "WAIT", "EXPECTED_LIQUIDITY_SWEEP")
            return None
        if s.phase == "WAIT_STRUCTURE":
            self.counts["structure"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "CHOCH_OR_BOS")
            return None
        if s.phase == "WAIT_ZONE" or s.zone_high <= s.zone_low:
            self.counts["zone"] += 1
            self.last_signal = self._debug(trend, i15, "WAIT", "OB_OR_FVG")
            return None
        if not self._in_zone(i15, s):
            s.phase = "WAIT_RETRACE"
            self.counts["retrace"] += 1
            self.save_state()
            self.last_signal = self._debug(trend, i15, "WAIT", "RETRACE_TO_ZONE")
            return None
        s.phase = "WAIT_TRIGGER"
        trigger = self._trigger(i15, s.direction)
        if not trigger:
            self.counts["trigger"] += 1
            self.save_state()
            self.last_signal = self._debug(trend, i15, "WAIT", "PRICE_ACTION")
            return None

        payload = self._build(i15, trigger)
        if not payload:
            self.last_signal = self._debug(trend, i15, "WAIT", "RISK_BUILD")
            return None
        payload["symbol"] = self.symbol
        if self.execution_callback: self.execution_callback("OPEN_" + payload["direction"], payload)
        self.position = Position(
            direction=payload["direction"], entry=payload["entry"], sl=payload["sl"],
            initial_sl=payload["sl"], tp=payload["tp2"], tp1=payload["tp1"],
            tp2=payload["tp2"], size=payload["size"], initial_size=payload["size"],
            strategy=payload["strategy"], trigger=payload["trigger"], opened_at=time.time(),
            sweep_level=payload["sweep_level"], choch_level=payload["choch_level"],
            bos_level=payload["bos_level"], zone_low=payload["zone_low"],
            zone_high=payload["zone_high"], zone_type=payload["zone_type"])
        self.setup = None
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = f"OPEN {payload['direction']} trigger={trigger} entry={payload['entry']:.6f}"
        return {"event": "OPEN", **payload}
