"""
Layer 3 — Entry Engine.  15M.  HMA10/HMA16 fresh-cross cycle.

Answers: "Has the moment to enter arrived?" — nothing else. This layer has
NO right to pick Long or Short: it receives a fixed `direction` from the
Bias layer (which itself only fires once Regime + Bias already agree) and
may ONLY search for a trigger matching that side.

HMA Cross is an EVENT, not a continuously-valid condition:
    long_cross  = previous_hma10 <= previous_hma16 AND current_hma10 > current_hma16
    short_cross = previous_hma10 >= previous_hma16 AND current_hma10 < current_hma16

Each cross opens exactly one entry cycle. At most ONE position may be
opened from a given cross. After a position closes, the engine will not
consider a new entry until a genuinely NEW cross has occurred — tracked
per-symbol via `waiting_for_new_cross`, cleared the next time ANY fresh
cross fires (mathematically the very next cross after a close is always
the opposite direction, since HMA10/16 must cross back before it can
cross forward again).

Entry also requires (both directions, mirrored):
  - regime_direction / bias_direction already == this side (Layer 1/2 own
    that decision, Entry never overrides it)
  - HMA alignment still holds on the signal (most recently CLOSED) 15M bar
  - the signal bar's close is on the correct side of HMA16
  - price is not overextended from HMA16 (anti-chase):
        extension_atr = |close - hma16| / atr  <=  entry_max_distance_from_hma_atr
  - the cross has not already produced an entry, and the engine isn't
    still waiting for a new cross after a prior exit

This module also owns the HMA-based EARLY EXIT check (`check_exit`) — the
same HMA10/HMA16/ATR values, so it lives next to the entry logic rather
than duplicating the indicator computation elsewhere:
    HMA_CROSS_REVERSAL      — HMA crosses back against the position
    PRICE_CLOSED_BEYOND_HMA — signal bar closes exit_hma_buffer_atr past
                               HMA16 against the position
Early exit does NOT replace the hard SL/TP — it is an additional, faster
path evaluated once per closed 15M bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config

LONG  = "LONG"
SHORT = "SHORT"
NONE  = "NONE"

HMA_CROSS_REVERSAL = "HMA_CROSS_REVERSAL"
PRICE_CLOSED_BEYOND_HMA = "PRICE_CLOSED_BEYOND_HMA"


@dataclass
class EntryResult:
    direction: str             # LONG | SHORT | NONE
    allow_entry: bool
    reason: str = ""
    price: float = 0.0
    hma_fast: float = 0.0
    hma_slow: float = 0.0
    atr: float = 0.0
    extension_atr: float = 0.0
    cross_id: object = None
    entry_score: float = 0.0   # 100 on a valid trigger, 0 otherwise — telegram/log compat


@dataclass
class ExitCheckResult:
    should_exit: bool
    reason: str = ""           # HMA_CROSS_REVERSAL | PRICE_CLOSED_BEYOND_HMA | ""
    detail: str = ""


@dataclass
class _CrossState:
    cross_direction: Optional[str] = None    # LONG | SHORT | None — direction of the LAST detected cross
    cross_id: object = None                  # closed-bar timestamp of that cross
    cross_used: bool = False                 # an entry was already opened from this cross
    waiting_for_new_cross: bool = False       # set True on any full close; cleared on the next fresh cross


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: dict[str, _CrossState] = {}

    # ── per-symbol cross-cycle state ─────────────────────────────────────────
    def _get_state(self, symbol: str) -> _CrossState:
        return self._state.setdefault(symbol, _CrossState())

    def on_position_closed(self, symbol: str) -> None:
        """Call whenever a position FULLY closes (TP2/SL/BE/early-exit) —
        NOT on a TP1 partial. Blocks re-entry until a genuinely new cross."""
        self._get_state(symbol).waiting_for_new_cross = True

    def reset_symbol(self, symbol: str) -> None:
        """Drop all cross-cycle memory for a symbol (e.g. on manual flat)."""
        self._state.pop(symbol, None)

    # ── shared HMA/ATR computation ───────────────────────────────────────────
    @staticmethod
    def _hma_atr(df_15m: pd.DataFrame, cfg: Config):
        closes = df_15m["close"]
        hma_f = ind.hma(closes, cfg.hma_fast_length)
        hma_s = ind.hma(closes, cfg.hma_slow_length)
        atr_s = ind.atr(df_15m, cfg.sl_atr_period)
        return hma_f, hma_s, atr_s

    # ── Entry ─────────────────────────────────────────────────────────────────
    def analyze(self, df_15m: pd.DataFrame, direction: str, symbol: str) -> EntryResult:
        c = self.cfg
        if direction not in (LONG, SHORT):
            return EntryResult(NONE, False, "no direction from Bias layer — nothing to time")
        min_len = max(c.hma_slow_length * 2, 30) + 5
        if len(df_15m) < min_len:
            return EntryResult(NONE, False, "insufficient 15m history")

        hma_f, hma_s, atr_s = self._hma_atr(df_15m, c)
        cur_f, cur_s = float(hma_f.iloc[-1]), float(hma_s.iloc[-1])
        prev_f, prev_s = float(hma_f.iloc[-2]), float(hma_s.iloc[-2])
        atr_now = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0
        close = float(df_15m["close"].iloc[-1])
        bar_ts = df_15m.index[-1]

        long_cross = prev_f <= prev_s and cur_f > cur_s
        short_cross = prev_f >= prev_s and cur_f < cur_s

        state = self._get_state(symbol)
        # a fresh cross (either direction) always means the wait is over —
        # the very next cross after any close is mathematically the opposite
        # of whatever was open, so this is exactly "a genuine opposite cross".
        if long_cross or short_cross:
            state.cross_direction = LONG if long_cross else SHORT
            state.cross_id = bar_ts
            state.cross_used = False
            state.waiting_for_new_cross = False

        if atr_now <= 0 or np.isnan(atr_now):
            return EntryResult(NONE, False, "ATR unavailable", price=close,
                               hma_fast=cur_f, hma_slow=cur_s)

        extension_atr = abs(close - cur_s) / atr_now
        is_long = direction == LONG
        base = dict(price=close, hma_fast=cur_f, hma_slow=cur_s, atr=atr_now,
                   extension_atr=extension_atr, cross_id=state.cross_id)

        this_cross_fresh = (long_cross if is_long else short_cross)
        aligned = (cur_f > cur_s) if is_long else (cur_f < cur_s)
        close_ok = (close > cur_s) if is_long else (close < cur_s)
        not_extended = extension_atr <= c.entry_max_distance_from_hma_atr
        cross_matches_side = state.cross_direction == direction

        if not this_cross_fresh:
            if state.waiting_for_new_cross:
                reason = f"{direction} blocked: waiting for a new cross after prior exit"
            elif not cross_matches_side:
                reason = f"{direction} blocked: no fresh {direction} HMA cross this bar"
            else:
                reason = f"{direction} blocked: no fresh HMA cross this bar"
            return EntryResult(NONE, False, reason, **base)
        if not aligned:
            return EntryResult(NONE, False, f"{direction} blocked: HMA alignment flipped intra-check", **base)
        if not close_ok:
            return EntryResult(NONE, False,
                               f"{direction} blocked: close not past HMA16 ({close:.6f} vs {cur_s:.6f})", **base)
        if not not_extended:
            return EntryResult(NONE, False,
                               f"ENTRY BLOCKED Reason=PRICE_TOO_EXTENDED ExtensionATR={extension_atr:.2f} "
                               f"Maximum={c.entry_max_distance_from_hma_atr:.2f}", **base)
        if state.cross_used:
            return EntryResult(NONE, False, f"{direction} blocked: this cross already used for an entry", **base)
        if state.waiting_for_new_cross:
            return EntryResult(NONE, False, f"{direction} blocked: waiting for a new cross after prior exit", **base)

        state.cross_used = True
        return EntryResult(direction, True,
                           f"ENTRY {direction}  HMA{c.hma_fast_length}xHMA{c.hma_slow_length} cross  "
                           f"ExtensionATR={extension_atr:.2f}  CrossID={bar_ts}",
                           entry_score=100.0, **base)

    # ── Early exit ────────────────────────────────────────────────────────────
    def check_exit(self, df_15m: pd.DataFrame, position_side: str) -> ExitCheckResult:
        """position_side: 'long' | 'short' (Position.side casing)."""
        c = self.cfg
        min_len = max(c.hma_slow_length * 2, 30) + 5
        if len(df_15m) < min_len:
            return ExitCheckResult(False)

        hma_f, hma_s, atr_s = self._hma_atr(df_15m, c)
        cur_f, cur_s = float(hma_f.iloc[-1]), float(hma_s.iloc[-1])
        prev_f, prev_s = float(hma_f.iloc[-2]), float(hma_s.iloc[-2])
        atr_now = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0
        close = float(df_15m["close"].iloc[-1])
        is_long = position_side == "long"

        if is_long:
            cross_back = prev_f >= prev_s and cur_f < cur_s
            price_failure = atr_now > 0 and close < cur_s - atr_now * c.exit_hma_buffer_atr
        else:
            cross_back = prev_f <= prev_s and cur_f > cur_s
            price_failure = atr_now > 0 and close > cur_s + atr_now * c.exit_hma_buffer_atr

        if cross_back:
            return ExitCheckResult(True, HMA_CROSS_REVERSAL,
                                   f"HMA{c.hma_fast_length} crossed back against {position_side}")
        if price_failure:
            return ExitCheckResult(True, PRICE_CLOSED_BEYOND_HMA,
                                   f"close={close:.6f} HMA{c.hma_slow_length}={cur_s:.6f} "
                                   f"ATRBuffer={atr_now * c.exit_hma_buffer_atr:.6f}")
        return ExitCheckResult(False)
