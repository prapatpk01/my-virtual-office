"""
Layer 3 — Entry Engine (TF5M EMA5/9 + MACD timing trigger).

Regime (4H+1H) and Bias (1H+15M+5M) decide the SIDE. This layer receives
that fixed `direction` and only searches for a 5-minute entry trigger on
that side — it never picks Long vs Short. Everything below is evaluated on
the last CLOSED 5M bar (no intrabar cross).

LONG entry — all must hold:
  1. direction == LONG            (bullish regime + bias, from upstream)
  2. EMA5 crossed above EMA9 within the last entry_ema_cross_lookback bars,
     and EMA5 > EMA9 now
  3. MACD line > signal now, OR a bullish MACD cross within the last
     entry_macd_cross_lookback bars
  4. MACD histogram > 0
  5. MACD histogram rising: hist[-1] > hist[-2]
  6. bar OPEN or CLOSE above EMA9
SHORT mirrors every clause.

One entry per EMA cross; after a full close the engine waits for a
genuinely new cross before re-entering (cross-cycle guard, per symbol).

Early exit (`check_exit`) — HARD exits only, so noise doesn't shake us out:
    EMA_CROSS_REVERSAL     — EMA5 crosses back against the position
    PRICE_OPEN_BEYOND_EMA  — bar OPEN on the wrong side of EMA9
MACD weakening (line cross-back / histogram flip) is a WARNING only and
never closes the position. Suppressed for exit_grace_bars closed 5M bars
after entry so the EMAs can separate. Does NOT replace the hard SL/TP — an
additional, faster path evaluated once per closed 5M bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

LONG  = "LONG"
SHORT = "SHORT"
NONE  = "NONE"

# Exit reason identifiers (kept as stable strings consumed by main.py,
# report.py and telegram_notifier.py).
EMA_CROSS_REVERSAL = "EMA_CROSS_REVERSAL"
PRICE_OPEN_BEYOND_EMA = "PRICE_OPEN_BEYOND_EMA"


@dataclass
class EntryResult:
    direction: str             # LONG | SHORT | NONE
    allow_entry: bool
    reason: str = ""
    price: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    macd_hist: float = 0.0
    cross_id: object = None
    entry_score: float = 0.0   # 100 on a valid trigger, 0 otherwise — telegram/log compat


@dataclass
class ExitCheckResult:
    should_exit: bool
    reason: str = ""           # EMA_CROSS_REVERSAL | PRICE_OPEN_BEYOND_EMA | ""
    detail: str = ""


@dataclass
class _CrossState:
    """One-entry-per-EMA-cross cycle, per symbol."""
    cross_direction: Optional[str] = None    # LONG | SHORT | None — direction of the LAST EMA5/9 cross
    cross_id: object = None                  # closed-bar timestamp of that cross
    cross_used: bool = False                 # an entry was already opened from this cross
    waiting_for_new_cross: bool = False       # set True on any full close; cleared on the next fresh cross


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: dict[str, _CrossState] = {}

    # ── per-symbol state ──────────────────────────────────────────────────────
    def _get_state(self, symbol: str) -> _CrossState:
        return self._state.setdefault(symbol, _CrossState())

    def on_position_closed(self, symbol: str) -> None:
        """Call whenever a position FULLY closes (TP2/SL/BE/early-exit) — NOT
        on a TP1 partial. Blocks re-entry until a genuinely new EMA cross."""
        st = self._get_state(symbol)
        st.waiting_for_new_cross = True
        st.cross_used = True

    def reset_symbol(self, symbol: str) -> None:
        """Drop all cross-cycle memory for a symbol (e.g. manual flat)."""
        self._state.pop(symbol, None)

    # ── indicators (shared by analyze + check_exit) ──────────────────────────
    def _emas(self, df: pd.DataFrame):
        c = self.cfg
        close = df["close"]
        return ind.ema(close, c.entry_ema_fast), ind.ema(close, c.entry_ema_slow)

    def _macd(self, df: pd.DataFrame):
        c = self.cfg
        return ind.macd(df["close"], c.entry_macd_fast, c.entry_macd_slow, c.entry_macd_signal)

    @staticmethod
    def _crossed(fast: pd.Series, slow: pd.Series, lookback: int, bullish: bool) -> bool:
        """True if `fast` crossed `slow` in the given direction on any of the
        last `lookback` closed-bar transitions (the most recent transition is
        k=1: prev bar -> current bar)."""
        n = min(lookback, len(fast) - 1)
        for k in range(1, n + 1):
            pf, ps = float(fast.iloc[-k - 1]), float(slow.iloc[-k - 1])
            cf, cs = float(fast.iloc[-k]), float(slow.iloc[-k])
            if bullish and pf <= ps and cf > cs:
                return True
            if (not bullish) and pf >= ps and cf < cs:
                return True
        return False

    def _min_len(self) -> int:
        c = self.cfg
        return max(c.entry_macd_slow + c.entry_macd_signal, c.entry_ema_slow) + 5

    # ── cross bookkeeping ─────────────────────────────────────────────────────
    def observe(self, df_5m: pd.DataFrame, symbol: str) -> None:
        """Record any fresh EMA5/9 cross on the last closed 5M bar into the
        per-symbol cycle state. Called once per closed bar regardless of what
        Regime/Bias decide, so a cross that fires while Bias reads NO TRADE
        still clears waiting_for_new_cross. Idempotent per bar (keyed on the
        bar timestamp) — live re-evaluates every ~30s inside the same bar."""
        if df_5m is None or len(df_5m) < self._min_len():
            return
        state = self._get_state(symbol)
        bar_ts = df_5m.index[-1]
        if state.cross_id == bar_ts:
            return   # this bar's cross already recorded — never re-process
        ema_f, ema_s = self._emas(df_5m)
        cur_f, cur_s = float(ema_f.iloc[-1]), float(ema_s.iloc[-1])
        prev_f, prev_s = float(ema_f.iloc[-2]), float(ema_s.iloc[-2])
        long_cross = prev_f <= prev_s and cur_f > cur_s
        short_cross = prev_f >= prev_s and cur_f < cur_s
        if not (long_cross or short_cross):
            return
        state.cross_direction = LONG if long_cross else SHORT
        state.cross_id = bar_ts
        state.cross_used = False
        state.waiting_for_new_cross = False

    # ── combined entry decision ──────────────────────────────────────────────
    def analyze(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
               direction: str, symbol: str) -> EntryResult:
        c = self.cfg
        if direction not in (LONG, SHORT):
            return EntryResult(NONE, False, "no direction from Bias layer — nothing to time")

        # Cross bookkeeping first — a structural fact, recorded regardless of
        # what the trigger decides on this bar (idempotent per bar).
        self.observe(df_5m, symbol)

        if df_5m is None or len(df_5m) < self._min_len():
            return EntryResult(NONE, False, "insufficient 5m history")

        is_long = direction == LONG
        ema_f, ema_s = self._emas(df_5m)
        line, sig, hist = self._macd(df_5m)
        cur_f, cur_s = float(ema_f.iloc[-1]), float(ema_s.iloc[-1])
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        l_now, s_now = float(line.iloc[-1]), float(sig.iloc[-1])
        open_px = float(df_5m["open"].iloc[-1])
        close_px = float(df_5m["close"].iloc[-1])
        state = self._get_state(symbol)
        base = dict(price=close_px, ema_fast=cur_f, ema_slow=cur_s,
                    macd_hist=h_now, cross_id=state.cross_id)

        # ── condition 2 — EMA5 crossed the right way within the window, still aligned
        cross_pending = (state.cross_direction == direction and not state.cross_used
                        and not state.waiting_for_new_cross)
        bars_since = (int((df_5m.index > state.cross_id).sum())
                     if state.cross_id is not None else None)
        in_window = bars_since is not None and bars_since <= c.entry_ema_cross_lookback
        aligned = (cur_f > cur_s) if is_long else (cur_f < cur_s)

        if not cross_pending:
            reason = (f"L3: {direction} blocked — waiting for a new EMA cross after prior exit"
                      if state.waiting_for_new_cross else
                      f"L3: {direction} blocked — no pending EMA{c.entry_ema_fast}x{c.entry_ema_slow} cross")
            return EntryResult(NONE, False, reason, **base)
        if not in_window:
            return EntryResult(NONE, False,
                               f"L3: {direction} blocked — EMA cross was {bars_since} bar(s) ago "
                               f"(window {c.entry_ema_cross_lookback}) — wait for a fresh cross", **base)
        if not aligned:
            return EntryResult(NONE, False, f"L3: {direction} blocked — EMA{c.entry_ema_fast}/"
                               f"{c.entry_ema_slow} alignment not held", **base)

        # ── condition 3 — MACD line above signal, or crossed within window
        macd_side = (l_now > s_now) if is_long else (l_now < s_now)
        macd_cross = self._crossed(line, sig, c.entry_macd_cross_lookback, bullish=is_long)
        if not (macd_side or macd_cross):
            return EntryResult(NONE, False,
                               f"L3: {direction} blocked — MACD line not {'>' if is_long else '<'} signal "
                               f"and no cross within {c.entry_macd_cross_lookback} bars", **base)

        # ── condition 4 — histogram on the right side of zero
        hist_side = (h_now > 0) if is_long else (h_now < 0)
        if not hist_side:
            return EntryResult(NONE, False,
                               f"L3: {direction} blocked — MACD histogram {h_now:+.6f} wrong side of 0", **base)

        # ── condition 5 — histogram building in the trade direction
        hist_building = (h_now > h_prev) if is_long else (h_now < h_prev)
        if not hist_building:
            return EntryResult(NONE, False,
                               f"L3: {direction} blocked — MACD histogram not "
                               f"{'rising' if is_long else 'falling'} ({h_prev:+.6f}->{h_now:+.6f})", **base)

        # ── condition 6 — price (open OR close) on the correct side of EMA9
        price_ok = ((open_px > cur_s or close_px > cur_s) if is_long
                    else (open_px < cur_s or close_px < cur_s))
        if not price_ok:
            return EntryResult(NONE, False,
                               f"L3: {direction} blocked — neither open nor close past EMA{c.entry_ema_slow} "
                               f"(o={open_px:.6f} c={close_px:.6f} vs {cur_s:.6f})", **base)

        # ── all conditions clear — fire (one entry per cross) ─────────────────
        state.cross_used = True
        return EntryResult(direction, True,
                           f"ENTRY {direction}  EMA{c.entry_ema_fast}x{c.entry_ema_slow} cross "
                           f"{bars_since}b ago  MACD hist={h_now:+.5f} rising  CrossID={state.cross_id}",
                           entry_score=100.0, **base)

    # ── Early exit (EMA hard-exit; MACD is warning-only) ─────────────────────
    def check_exit(self, df_5m: pd.DataFrame, position_side: str,
                   bars_since_entry: Optional[int] = None) -> ExitCheckResult:
        """position_side: 'long' | 'short' (Position.side casing). While
        bars_since_entry < exit_grace_bars the HARD exit is suppressed so the
        EMAs can separate from the entry cross. SL/TP handled elsewhere are
        unaffected. Only EMA cross-back and an OPEN on the wrong side of EMA9
        close a position — MACD weakening is deliberately NOT an exit."""
        c = self.cfg
        if bars_since_entry is not None and bars_since_entry < c.exit_grace_bars:
            return ExitCheckResult(False)
        if df_5m is None or len(df_5m) < self._min_len():
            return ExitCheckResult(False)

        ema_f, ema_s = self._emas(df_5m)
        cur_f, cur_s = float(ema_f.iloc[-1]), float(ema_s.iloc[-1])
        prev_f, prev_s = float(ema_f.iloc[-2]), float(ema_s.iloc[-2])
        open_px = float(df_5m["open"].iloc[-1])
        is_long = position_side == "long"

        if is_long:
            cross_back = prev_f >= prev_s and cur_f < cur_s
            open_wrong = open_px < cur_s
        else:
            cross_back = prev_f <= prev_s and cur_f > cur_s
            open_wrong = open_px > cur_s

        if cross_back:
            return ExitCheckResult(True, EMA_CROSS_REVERSAL,
                                   f"EMA{c.entry_ema_fast} crossed back against {position_side}")
        if open_wrong:
            return ExitCheckResult(True, PRICE_OPEN_BEYOND_EMA,
                                   f"open={open_px:.6f} EMA{c.entry_ema_slow}={cur_s:.6f}")
        return ExitCheckResult(False)
