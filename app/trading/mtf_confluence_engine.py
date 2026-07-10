"""
MTF Trend Alignment & Three-Signal Fast Entry Logic
====================================================
Alternative, fully deterministic entry engine (no scoring, no learning) —
selectable via ADAPTIVE_ENTRY_ENGINE=mtf_confluence alongside the default
V9.2 L1/L2/L3/StrategyScorer pipeline in adaptive_trading_bot.py.

4H sets the macro trend, 1H confirms the mid trend, 15m only times entries
inside whichever direction 4H+1H both agree on:

    4H UPTREND   + 1H UPTREND   -> LONG_ONLY
    4H DOWNTREND + 1H DOWNTREND -> SHORT_ONLY
    anything else                -> NO_TRADE

On 15m, three independent signals — ROC9 zero-line cross, HMA10/HMA20
cross, MACD line/signal cross — must all fire, in any order, within a
3-bar window of whichever fires first. When all three have fired and the
indicators are STILL aligned on the entry bar (a signal having fired once
is not enough — the state must not have flipped back before entry), the
setup opens a position. Exit management (T1/T2/SL/post-T1 protection) is
untouched — this module only decides entry direction and timing.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .strategies.base import BaseStrategy

logger = logging.getLogger("mtf_confluence_engine")

# 15m signals must all complete within this many bars of the first one.
SETUP_WINDOW_BARS = 3


# ══════════════════════════════════════════════════════════════════════════
# Shared indicator math (HMA + ROC; EMA/MACD/ATR reuse BaseStrategy)
# ══════════════════════════════════════════════════════════════════════════

def _wma(values: List[float], period: int) -> np.ndarray:
    """Trailing weighted moving average — most recent sample gets the
    highest weight (1..period). Vectorized via convolution."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    weights = np.arange(1, period + 1, dtype=float)
    conv = np.convolve(arr, weights[::-1], mode="valid") / weights.sum()
    out[period - 1:] = conv
    return out


def hma(values: List[float], period: int) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half    = max(int(period / 2), 1)
    sqrt_p  = max(int(round(period ** 0.5)), 1)
    raw     = 2.0 * _wma(values, half) - _wma(values, period)
    return _wma(list(raw), sqrt_p)


def roc(values: List[float], period: int = 9) -> np.ndarray:
    """Rate of change (%) vs `period` bars ago."""
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) > period:
        base = arr[:-period]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[period:] = np.where(base != 0, (arr[period:] - base) / base * 100.0, np.nan)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Layer 1/2 — 4H Macro Trend / 1H Mid Trend (identical rule, different TF)
# ══════════════════════════════════════════════════════════════════════════

def classify_trend(candles: List, ema_fast: int = 15, ema_slow: int = 50,
                   hma_fast: int = 12, hma_slow: int = 20,
                   roc_period: int = 9, atr_period: int = 14,
                   atr_mult: float = 1.5) -> str:
    """Returns 'UPTREND' / 'DOWNTREND' / 'NO_TREND' for one timeframe."""
    min_bars = max(ema_slow, hma_slow + int(hma_slow ** 0.5) + 1, atr_period) + 1
    if len(candles) < min_bars:
        return "NO_TREND"

    closes = [float(c.close) for c in candles]
    ema_f  = BaseStrategy.ema(closes, ema_fast)
    ema_s  = BaseStrategy.ema(closes, ema_slow)
    hma_f  = hma(closes, hma_fast)
    hma_s  = hma(closes, hma_slow)
    r      = roc(closes, roc_period)
    macd_line, macd_sig, macd_hist = BaseStrategy.macd(closes, 12, 26, 9)
    atr_arr = BaseStrategy.atr(candles, atr_period)

    vals = (ema_f[-1], ema_s[-1], hma_f[-1], hma_s[-1], r[-1],
            macd_line[-1], macd_sig[-1], macd_hist[-1], atr_arr[-1])
    if any(np.isnan(v) for v in vals):
        return "NO_TREND"

    close = closes[-1]
    e15, e50, hf, hs, rr, ml, ms, mh, atr = vals

    uptrend = (
        close > e50 and close > e15
        and (close - e15) <= atr_mult * atr
        and hf > hs
        and rr > 0
        and ml > ms
        and mh > 0
    )
    if uptrend:
        return "UPTREND"

    downtrend = (
        close < e50 and close < e15
        and (e15 - close) <= atr_mult * atr
        and hf < hs
        and rr < 0
        and ml < ms
        and mh < 0
    )
    if downtrend:
        return "DOWNTREND"

    return "NO_TREND"


# ══════════════════════════════════════════════════════════════════════════
# Layer 4 — 15M Three-Signal Entry Engine
# ══════════════════════════════════════════════════════════════════════════

class SetupStatus(str, Enum):
    IDLE      = "IDLE"
    COLLECTING = "COLLECTING"
    READY     = "READY"
    EXPIRED   = "EXPIRED"
    CANCELLED = "CANCELLED"
    USED      = "USED"


@dataclass
class EntrySetup:
    direction: Optional[str] = None
    status: SetupStatus = SetupStatus.IDLE

    start_bar: Optional[int] = None
    expiry_bar: Optional[int] = None

    roc_received: bool = False
    hma_received: bool = False
    macd_received: bool = False

    roc_signal_bar: Optional[int] = None
    hma_signal_bar: Optional[int] = None
    macd_signal_bar: Optional[int] = None

    opposite_signal_detected: bool = False
    used: bool = False

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "EntrySetup":
        if not d:
            return cls()
        d = dict(d)
        d["status"] = SetupStatus(d.get("status", "IDLE"))
        return cls(**d)


def detect_entry_events(roc_arr: np.ndarray, hma_fast_arr: np.ndarray,
                        hma_slow_arr: np.ndarray, macd_line_arr: np.ndarray,
                        macd_sig_arr: np.ndarray) -> Dict[str, bool]:
    """Cross events between the CLOSED current bar and the one before it.
    NaN comparisons are always False, so insufficient warmup naturally
    yields "no event" rather than needing an explicit guard."""
    c_roc, p_roc = roc_arr[-1], roc_arr[-2]
    c_hf,  p_hf  = hma_fast_arr[-1], hma_fast_arr[-2]
    c_hs,  p_hs  = hma_slow_arr[-1], hma_slow_arr[-2]
    c_ml,  p_ml  = macd_line_arr[-1], macd_line_arr[-2]
    c_ms,  p_ms  = macd_sig_arr[-1], macd_sig_arr[-2]
    return {
        "roc_cross_up":    bool(c_roc > 0 and p_roc <= 0),
        "roc_cross_down":  bool(c_roc < 0 and p_roc >= 0),
        "hma_cross_up":    bool(c_hf > c_hs and p_hf <= p_hs),
        "hma_cross_down":  bool(c_hf < c_hs and p_hf >= p_hs),
        "macd_cross_up":   bool(c_ml > c_ms and p_ml <= p_ms),
        "macd_cross_down": bool(c_ml < c_ms and p_ml >= p_ms),
    }


def _start_setup(direction: str, current_bar: int, events: Dict[str, bool]) -> EntrySetup:
    setup = EntrySetup(
        direction=direction, status=SetupStatus.COLLECTING,
        start_bar=current_bar, expiry_bar=current_bar + SETUP_WINDOW_BARS,
    )
    up = direction == "LONG"
    if events["roc_cross_up" if up else "roc_cross_down"]:
        setup.roc_received, setup.roc_signal_bar = True, current_bar
    if events["hma_cross_up" if up else "hma_cross_down"]:
        setup.hma_received, setup.hma_signal_bar = True, current_bar
    if events["macd_cross_up" if up else "macd_cross_down"]:
        setup.macd_received, setup.macd_signal_bar = True, current_bar
    return setup


def _update_setup(setup: EntrySetup, current_bar: int, events: Dict[str, bool],
                  trade_direction: str) -> EntrySetup:
    if setup.status not in (SetupStatus.COLLECTING, SetupStatus.READY):
        return setup

    expected_direction = "LONG_ONLY" if setup.direction == "LONG" else "SHORT_ONLY"
    if trade_direction != expected_direction:
        setup.status = SetupStatus.CANCELLED
        return setup

    if current_bar > setup.expiry_bar:
        setup.status = SetupStatus.EXPIRED
        return setup

    up = setup.direction == "LONG"
    opposite = (events["roc_cross_down" if up else "roc_cross_up"]
                or events["hma_cross_down" if up else "hma_cross_up"]
                or events["macd_cross_down" if up else "macd_cross_up"])
    if opposite:
        setup.opposite_signal_detected = True
        setup.status = SetupStatus.CANCELLED
        return setup

    if events["roc_cross_up" if up else "roc_cross_down"] and not setup.roc_received:
        setup.roc_received, setup.roc_signal_bar = True, current_bar
    if events["hma_cross_up" if up else "hma_cross_down"] and not setup.hma_received:
        setup.hma_received, setup.hma_signal_bar = True, current_bar
    if events["macd_cross_up" if up else "macd_cross_down"] and not setup.macd_received:
        setup.macd_received, setup.macd_signal_bar = True, current_bar

    if setup.roc_received and setup.hma_received and setup.macd_received:
        setup.status = SetupStatus.READY

    return setup


def validate_current_alignment(direction: str, roc_now: float, hma_fast_now: float,
                               hma_slow_now: float, macd_line_now: float,
                               macd_sig_now: float, macd_hist_now: float) -> bool:
    if direction == "LONG":
        return bool(roc_now > 0 and hma_fast_now > hma_slow_now
                    and macd_line_now > macd_sig_now and macd_hist_now >= 0)
    if direction == "SHORT":
        return bool(roc_now < 0 and hma_fast_now < hma_slow_now
                    and macd_line_now < macd_sig_now and macd_hist_now <= 0)
    return False


def _evaluate_entry(setup: EntrySetup, alignment_ok: bool, trade_direction: str,
                    current_bar: int, position_is_open: bool,
                    cooldown_active: bool) -> Optional[str]:
    if position_is_open or cooldown_active or setup.used:
        return None
    if setup.status != SetupStatus.READY:
        return None
    if setup.expiry_bar is None:
        return None
    if current_bar > setup.expiry_bar:
        setup.status = SetupStatus.EXPIRED
        return None
    if setup.opposite_signal_detected:
        setup.status = SetupStatus.CANCELLED
        return None

    expected_direction = "LONG_ONLY" if setup.direction == "LONG" else "SHORT_ONLY"
    if trade_direction != expected_direction:
        setup.status = SetupStatus.CANCELLED
        return None
    if not alignment_ok:
        setup.status = SetupStatus.CANCELLED
        return None

    setup.status = SetupStatus.USED
    setup.used = True
    return setup.direction


# ══════════════════════════════════════════════════════════════════════════
# Engine facade — what adaptive_trading_bot.py talks to
# ══════════════════════════════════════════════════════════════════════════

class MTFConfluenceEngine:
    """Per-symbol instance. Call update_macro()/update_mid() once per closed
    4H/1H bar and process_15m() once per closed 15m bar, in that order (the
    caller — TradingBot.on_tick — already receives all three per tick)."""

    def __init__(self):
        self.macro_trend: str = "NO_TREND"
        self.mid_trend:   str = "NO_TREND"
        self.long_setup   = EntrySetup()
        self.short_setup  = EntrySetup()
        self.last_diag: Dict = {}

    @property
    def trade_direction(self) -> str:
        if self.macro_trend == "UPTREND" and self.mid_trend == "UPTREND":
            return "LONG_ONLY"
        if self.macro_trend == "DOWNTREND" and self.mid_trend == "DOWNTREND":
            return "SHORT_ONLY"
        return "NO_TRADE"

    def update_macro(self, candles_4h: List) -> None:
        self.macro_trend = classify_trend(candles_4h, hma_fast=12, hma_slow=20)

    def update_mid(self, candles_1h: List) -> None:
        self.mid_trend = classify_trend(candles_1h, hma_fast=12, hma_slow=20)

    def process_15m(self, candles_15m: List, current_bar: int,
                    position_is_open: bool, cooldown_active: bool) -> Optional[str]:
        """Returns 'LONG' / 'SHORT' / None. Mirrors spec section 30
        (process_15m_bar) — direction gate first, then setup state machine,
        then final entry evaluation on the same closed bar."""
        min_bars = 26 + 9 + 2   # MACD(12,26,9) warmup + 1 for prev-bar diff
        if len(candles_15m) < min_bars:
            self.last_diag = {"reason": "insufficient_15m_history"}
            return None

        closes = [float(c.close) for c in candles_15m]
        r      = roc(closes, 9)
        hma_f  = hma(closes, 10)
        hma_s  = hma(closes, 20)
        macd_line, macd_sig, macd_hist = BaseStrategy.macd(closes, 12, 26, 9)
        events = detect_entry_events(r, hma_f, hma_s, macd_line, macd_sig)

        trade_direction = self.trade_direction
        self.last_diag = {
            "macro": self.macro_trend, "mid": self.mid_trend,
            "trade_direction": trade_direction, "events": events,
            "long_status": self.long_setup.status.value,
            "short_status": self.short_setup.status.value,
        }

        if position_is_open:
            return None

        if trade_direction == "NO_TRADE":
            if self.long_setup.status in (SetupStatus.COLLECTING, SetupStatus.READY):
                self.long_setup.status = SetupStatus.CANCELLED
            if self.short_setup.status in (SetupStatus.COLLECTING, SetupStatus.READY):
                self.short_setup.status = SetupStatus.CANCELLED
            return None

        if trade_direction == "LONG_ONLY":
            if self.long_setup.status in (SetupStatus.IDLE, SetupStatus.EXPIRED,
                                          SetupStatus.CANCELLED, SetupStatus.USED):
                first_signal = events["roc_cross_up"] or events["hma_cross_up"] or events["macd_cross_up"]
                if first_signal:
                    self.long_setup = _start_setup("LONG", current_bar, events)
            else:
                self.long_setup = _update_setup(self.long_setup, current_bar, events, trade_direction)

            alignment_ok = validate_current_alignment(
                "LONG", r[-1], hma_f[-1], hma_s[-1],
                macd_line[-1], macd_sig[-1], macd_hist[-1],
            )
            return _evaluate_entry(self.long_setup, alignment_ok, trade_direction,
                                   current_bar, position_is_open, cooldown_active)

        if trade_direction == "SHORT_ONLY":
            if self.short_setup.status in (SetupStatus.IDLE, SetupStatus.EXPIRED,
                                           SetupStatus.CANCELLED, SetupStatus.USED):
                first_signal = events["roc_cross_down"] or events["hma_cross_down"] or events["macd_cross_down"]
                if first_signal:
                    self.short_setup = _start_setup("SHORT", current_bar, events)
            else:
                self.short_setup = _update_setup(self.short_setup, current_bar, events, trade_direction)

            alignment_ok = validate_current_alignment(
                "SHORT", r[-1], hma_f[-1], hma_s[-1],
                macd_line[-1], macd_sig[-1], macd_hist[-1],
            )
            return _evaluate_entry(self.short_setup, alignment_ok, trade_direction,
                                   current_bar, position_is_open, cooldown_active)

        return None

    # ── Persistence (folded into TradingBot.save_state/load_state) ──────────

    def to_dict(self) -> Dict:
        return {
            "macro_trend": self.macro_trend,
            "mid_trend": self.mid_trend,
            "long_setup": self.long_setup.to_dict(),
            "short_setup": self.short_setup.to_dict(),
        }

    def load_dict(self, d: Optional[Dict]) -> None:
        if not d:
            return
        self.macro_trend = d.get("macro_trend", "NO_TREND")
        self.mid_trend   = d.get("mid_trend", "NO_TREND")
        self.long_setup  = EntrySetup.from_dict(d.get("long_setup"))
        self.short_setup = EntrySetup.from_dict(d.get("short_setup"))
