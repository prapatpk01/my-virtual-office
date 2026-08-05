"""Adaptive Momentum v1: EMA5/9 + MACD + ADX rising + CHOP + location."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional
import json, os, time

TP1_R = float(os.getenv("MOM_TP1_R", "1.0"))
TP2_R = float(os.getenv("MOM_TP2_R", "2.0"))
SL_ATR = float(os.getenv("MOM_SL_ATR", "1.0"))
MIN_SL_PCT = float(os.getenv("MOM_MIN_SL_PCT", "0.004"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
ADX_MIN = float(os.getenv("MOM_ADX_MIN", "15"))
CHOP_MAX = float(os.getenv("MOM_CHOP_MAX", "55"))
LOCATION_MAX_ATR = float(os.getenv("MOM_LOCATION_MAX_ATR", "1.0"))

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
        self.last_signal = "WARMUP"
        self.counts = {k: 0 for k in ("scans","entries","ema","macd","adx","chop","location")}
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
        except Exception:
            self.position = None

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({"position": asdict(self.position) if self.position else None}, handle)
        os.replace(temp, self.state_file)

    def _debug(self, i15: Dict, result: str, reason: str) -> str:
        return (
            f"MOMENTUM15 symbol={self.symbol} | "
            f"EMA[5={float(i15['ema5']):.6f},9={float(i15['ema9']):.6f},"
            f"upRecent={int(bool(i15.get('ema_cross_up_recent')))},downRecent={int(bool(i15.get('ema_cross_down_recent')))}] | "
            f"MACD[line={float(i15['macd']):+.6f},signal={float(i15['macd_signal']):+.6f},hist={float(i15['macd_hist']):+.6f}] | "
            f"ADX[{float(i15['adx']):.1f}>{ADX_MIN:.1f},rising={int(bool(i15.get('adx_rising')))}] | "
            f"CHOP[{float(i15['chop']):.1f}<{CHOP_MAX:.1f}] | "
            f"LOCATION[dist={float(i15['distance_ema9_atr']):.2f}/{LOCATION_MAX_ATR:.2f}ATR] | "
            f"RESULT[{result}:{reason}] | COUNTERS[{','.join(f'{k}={v}' for k,v in self.counts.items())}]"
        )

    def _build(self, i15: Dict, direction: str):
        entry = float(i15["close"])
        atr = max(float(i15["atr"]), entry * 0.0005)
        minimum = entry * MIN_SL_PCT
        if direction == "LONG":
            swing_sl = float(i15.get("recent_low", entry - atr))
            sl = min(swing_sl, entry - SL_ATR * atr, entry - minimum)
            risk = entry - sl
            tp1, tp2 = entry + TP1_R*risk, entry + TP2_R*risk
        else:
            swing_sl = float(i15.get("recent_high", entry + atr))
            sl = max(swing_sl, entry + SL_ATR * atr, entry + minimum)
            risk = sl - entry
            tp1, tp2 = entry - TP1_R*risk, entry - TP2_R*risk
        if risk <= 0:
            return None
        size = min(self.risk_usdt/risk, (self.margin_usdt*self.leverage)/max(entry,1e-12))
        if size <= 0:
            return None
        return {
            "direction": direction, "strategy": "ema5_9_macd_adx_chop_location",
            "trigger": "ema5_9_cross", "entry": entry, "sl": sl,
            "tp": tp2, "tp1": tp1, "tp2": tp2, "size": size,
            "risk_usdt": size*risk, "sl_pct": 100*risk/max(entry,1e-12),
            "ema5": float(i15["ema5"]), "ema9": float(i15["ema9"]),
            "macd": float(i15["macd"]), "macd_signal": float(i15["macd_signal"]),
            "adx": float(i15["adx"]), "chop": float(i15["chop"]),
            "distance_ema9_atr": float(i15["distance_ema9_atr"]),
        }

    def _close(self, price: float, reason: str):
        p = self.position
        assert p
        pnl = (price-p.entry)*p.size if p.direction=="LONG" else (p.entry-price)*p.size
        initial_risk = abs(p.entry-p.initial_sl)*max(p.initial_size,1e-12)
        r_multiple = pnl/initial_risk if initial_risk else 0.0
        payload = {
            "symbol": self.symbol, "direction": p.direction, "price": price,
            "entry": p.entry, "sl": p.sl, "tp": p.tp2, "tp1": p.tp1, "tp2": p.tp2,
            "size": p.size, "strategy": p.strategy, "trigger": p.trigger,
            "reason": reason, "pnl": pnl, "r_multiple": r_multiple,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_"+p.direction, payload)
        self.position = None
        self.save_state()
        self.last_signal = f"CLOSE {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R"
        return {"event":"CLOSE", **payload}

    def check_price(self, price: float):
        p = self.position
        if not p:
            return None
        if (p.direction=="LONG" and price <= p.sl) or (p.direction=="SHORT" and price >= p.sl):
            return self._close(price, "BE" if p.be_moved else "SL")
        if not p.tp1_hit and ((p.direction=="LONG" and price >= p.tp1) or (p.direction=="SHORT" and price <= p.tp1)):
            close_size = p.size*0.5
            pnl = (price-p.entry)*close_size if p.direction=="LONG" else (p.entry-price)*close_size
            payload = {"symbol":self.symbol,"direction":p.direction,"price":price,"entry":p.entry,
                       "size":close_size,"reason":"TP1","pnl":pnl,"r_multiple":TP1_R}
            if self.execution_callback:
                self.execution_callback("CLOSE_PARTIAL", payload)
            p.size -= close_size
            p.tp1_hit = True
            p.sl = p.entry
            p.be_moved = True
            self.save_state()
            return {"event":"PARTIAL", **payload}
        if (p.direction=="LONG" and price >= p.tp2) or (p.direction=="SHORT" and price <= p.tp2):
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
            p = self.position
            close = float(i15["close"])
            if p.direction == "LONG":
                exit_now = bool(i15.get("ema_cross_down") or i15.get("macd_cross_down") or close < float(i15["ema9"]))
            else:
                exit_now = bool(i15.get("ema_cross_up") or i15.get("macd_cross_up") or close > float(i15["ema9"]))
            if exit_now:
                return self._close(price or close, "MOMENTUM_FLIP")
            self.last_signal = f"MANAGE {p.direction} SL={p.sl:.6f} TP1={p.tp1:.6f} TP2={p.tp2:.6f}"
            return None

        self.counts["scans"] += 1
        long_ema = bool(i15.get("ema_cross_up_recent")) and float(i15["ema5"]) > float(i15["ema9"])
        short_ema = bool(i15.get("ema_cross_down_recent")) and float(i15["ema5"]) < float(i15["ema9"])
        if not long_ema and not short_ema:
            self.counts["ema"] += 1
            self.last_signal = self._debug(i15, "WAIT", "EMA5_9_CROSS")
            return None

        direction = "LONG" if long_ema else "SHORT"
        macd_ok = bool(i15.get("macd_bull")) if direction=="LONG" else bool(i15.get("macd_bear"))
        if not macd_ok:
            self.counts["macd"] += 1
            self.last_signal = self._debug(i15, "WAIT", "MACD_SIGNAL")
            return None
        if float(i15["adx"]) < ADX_MIN or not bool(i15.get("adx_rising")):
            self.counts["adx"] += 1
            self.last_signal = self._debug(i15, "WAIT", "ADX_NOT_RISING")
            return None
        if float(i15["chop"]) > CHOP_MAX:
            self.counts["chop"] += 1
            self.last_signal = self._debug(i15, "WAIT", "CHOP_TOO_HIGH")
            return None
        location_ok = bool(i15.get("location_long")) if direction=="LONG" else bool(i15.get("location_short"))
        if not location_ok or float(i15["distance_ema9_atr"]) > LOCATION_MAX_ATR:
            self.counts["location"] += 1
            self.last_signal = self._debug(i15, "WAIT", "LOCATION")
            return None

        payload = self._build(i15, direction)
        if not payload:
            self.last_signal = self._debug(i15, "WAIT", "RISK_BUILD")
            return None
        payload["symbol"] = self.symbol
        if self.execution_callback:
            self.execution_callback("OPEN_"+direction, payload)
        self.position = Position(
            direction=direction, entry=payload["entry"], sl=payload["sl"], initial_sl=payload["sl"],
            tp=payload["tp2"], tp1=payload["tp1"], tp2=payload["tp2"], size=payload["size"],
            initial_size=payload["size"], strategy=payload["strategy"], trigger=payload["trigger"],
            opened_at=time.time(),
        )
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = self._debug(i15, "ENTRY", direction)
        return {"event":"OPEN", **payload}
