"""
Layer 3 — Entry Engine. Three sequential sub-layers; ALL THREE must clear
on the same evaluation for an entry to fire. This layer has NO right to
pick Long or Short: it receives a fixed `direction` from the Bias layer
and may only search for a trigger matching that side.

  Layer 3.1  30M — 5-category quality pre-filter (Momentum/Trend/Structure/
             Liquidity/Participation), needs >= entry_min_categories (3/5)
             with Momentum + Structure mandatory. A QUALITY CHECK, not a
             timing trigger — it doesn't decide WHEN, only whether the
             setup is good enough to consider at all.

  Layer 3.2  15M+5M — prior-acceleration check. Looks at the last
             accel_15m_window closed 15M bars and accel_5m_window closed 5M
             bars for excessive price acceleration (net move or a single
             bar beyond that TF's ATR). If flagged, a pending Layer 3.3
             trigger is HELD (not rejected) for up to accel_max_rounds
             confirmation rounds — round N judges the Nth 15M bar + 4 5M
             bars closed after the flag; holding the direction confirms,
             a pullback/reversal extends to the next round, and failing
             the final round abandons the setup.

  Layer 3.3  15M — HMA10/HMA16 fresh-cross timing trigger. HMA Cross is an
             EVENT (one entry per cross cycle), not a continuously-valid
             condition — tracked per-symbol so a position can only be
             opened once per cross, and after a close the engine won't
             fire again until a genuinely NEW cross occurs (mathematically
             guaranteed to be the opposite direction, since HMA10/16 must
             cross back before it can cross forward again). Also enforces
             HMA alignment, close on the correct side of HMA16, and the
             anti-chase extension cap (|close-HMA16|/ATR <= 0.8).

This module also owns the HMA-based EARLY EXIT check (`check_exit`) — same
HMA10/HMA16/ATR values as Layer 3.3, so it lives next to the entry logic
instead of duplicating the indicator computation:
    HMA_CROSS_REVERSAL      — HMA crosses back against the position
    PRICE_CLOSED_BEYOND_HMA — signal bar closes exit_hma_buffer_atr past
                               HMA16 against the position
Early exit does NOT replace the hard SL/TP — an additional, faster path
evaluated once per closed 15M bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
import price_action as pa
from config import Config

LONG  = "LONG"
SHORT = "SHORT"
NONE  = "NONE"

HMA_CROSS_REVERSAL = "HMA_CROSS_REVERSAL"
PRICE_CLOSED_BEYOND_HMA = "PRICE_CLOSED_BEYOND_HMA"


@dataclass
class Layer31Result:
    passed: bool
    passed_count: int
    categories: dict = field(default_factory=dict)
    reason: str = ""


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
    layer31_passed_count: int = 0
    layer32_status: str = ""   # "" (not applicable) | "waiting" | "failed" | "clear"


@dataclass
class ExitCheckResult:
    should_exit: bool
    reason: str = ""           # HMA_CROSS_REVERSAL | PRICE_CLOSED_BEYOND_HMA | ""
    detail: str = ""


@dataclass
class _CrossState:
    """Layer 3.3 — one-entry-per-cross cycle, per symbol."""
    cross_direction: Optional[str] = None    # LONG | SHORT | None — direction of the LAST detected cross
    cross_id: object = None                  # closed-bar timestamp of that cross
    cross_used: bool = False                 # an entry was already opened from this cross
    waiting_for_new_cross: bool = False       # set True on any full close; cleared on the next fresh cross


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: dict[str, _CrossState] = {}     # Layer 3.3 cross-cycle, per symbol
        self._accel_wait: dict[str, dict] = {}        # Layer 3.2 wait-round state, per symbol

    # ── per-symbol state ──────────────────────────────────────────────────────
    def _get_state(self, symbol: str) -> _CrossState:
        return self._state.setdefault(symbol, _CrossState())

    def on_position_closed(self, symbol: str) -> None:
        """Call whenever a position FULLY closes (TP2/SL/BE/early-exit) —
        NOT on a TP1 partial. Blocks re-entry until a genuinely new cross."""
        self._get_state(symbol).waiting_for_new_cross = True

    def reset_symbol(self, symbol: str) -> None:
        """Drop all cross-cycle/accel-wait memory for a symbol (e.g. manual flat)."""
        self._state.pop(symbol, None)
        self._accel_wait.pop(symbol, None)

    # ── Layer 3.1 — 5-category quality pre-filter (30M) ──────────────────────
    def _layer31(self, df_30m: pd.DataFrame, direction: str) -> Layer31Result:
        c = self.cfg
        if len(df_30m) < max(c.hma_slow_length, c.entry_macd_slow) + 10:
            return Layer31Result(False, 0, {}, "L3.1: insufficient 30m history")

        closes, opens = df_30m["close"], df_30m["open"]
        price = float(closes.iloc[-1])
        is_long = direction == LONG

        hma_f = ind.hma(closes, c.hma_fast_length)
        hma_s = ind.hma(closes, c.hma_slow_length)
        e_ref = ind.ema(closes, c.entry_ema_ref)
        roc_v = float(ind.roc(closes, c.entry_roc_period).iloc[-1] or 0.0)
        _, _, hist = ind.macd(closes, c.entry_macd_fast, c.entry_macd_slow, c.entry_macd_signal)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        macd_cross_up = h_prev <= 0 and h_now > 0
        macd_cross_dn = h_prev >= 0 and h_now < 0

        sflags = ind.structure_flags(df_30m["high"], df_30m["low"],
                                     c.bias_structure_left, c.bias_structure_right)
        side = "LONG" if is_long else "SHORT"
        bos, choch = pa.bos_choch(df_30m, side, c.bias_structure_left, c.bias_structure_right)

        vol_now = float(df_30m["volume"].iloc[-1])
        vol_ma20 = float(df_30m["volume"].iloc[-21:-1].mean()) if len(df_30m) >= 21 else 0.0
        rel_vol = (vol_now / vol_ma20) if vol_ma20 > 0 else 1.0

        if is_long:
            categories = {
                "momentum":      macd_cross_up or (h_now > h_prev) or (roc_v > 0),
                "trend":         (float(hma_f.iloc[-1]) > float(hma_s.iloc[-1])) or
                                  (price > float(e_ref.iloc[-1])),
                "structure":     bos or choch or sflags["higher_low"],
                "liquidity":     pa.liquidity_sweep(df_30m, "LONG", c.entry_sweep_lookback) or
                                  pa.rejection_candle(df_30m, "LONG", c.entry_wick_reject_frac),
                "participation": pa.volume_expansion(df_30m, c.entry_vol_expansion_mult) or
                                  rel_vol >= c.entry_rel_vol_min,
            }
        else:
            categories = {
                "momentum":      macd_cross_dn or (h_now < h_prev) or (roc_v < 0),
                "trend":         (float(hma_f.iloc[-1]) < float(hma_s.iloc[-1])) or
                                  (price < float(e_ref.iloc[-1])),
                "structure":     bos or choch or sflags["lower_high"],
                "liquidity":     pa.liquidity_sweep(df_30m, "SHORT", c.entry_sweep_lookback) or
                                  pa.rejection_candle(df_30m, "SHORT", c.entry_wick_reject_frac),
                "participation": pa.volume_expansion(df_30m, c.entry_vol_expansion_mult) or
                                  rel_vol >= c.entry_rel_vol_min,
            }

        passed = sum(categories.values())
        mandatory_ok = categories["momentum"] and categories["structure"]
        ok = passed >= c.entry_min_categories and mandatory_ok
        if ok:
            reason = f"L3.1 pass: {passed}/5 ({', '.join(k for k, v in categories.items() if v)})"
        elif not mandatory_ok:
            missing = [k for k in ("momentum", "structure") if not categories[k]]
            reason = f"L3.1 fail: mandatory {'/'.join(missing)} not met ({passed}/5 passed)"
        else:
            reason = f"L3.1 fail: {passed}/5 categories (need >= {c.entry_min_categories})"
        return Layer31Result(ok, passed, categories, reason)

    # ── Layer 3.2 — prior-acceleration wait rounds (15M+5M) ──────────────────
    def _recent_acceleration(self, df_15m: pd.DataFrame, df_5m: Optional[pd.DataFrame]) -> tuple[bool, str]:
        c = self.cfg
        for name, df, win in (("15m", df_15m, c.accel_15m_window),
                              ("5m", df_5m, c.accel_5m_window)):
            if df is None or len(df) < win + 15:
                continue
            atr_v = float(ind.atr(df, 14).iloc[-1])
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            seg = df.iloc[-win:]
            net = abs(float(seg["close"].iloc[-1]) - float(seg["open"].iloc[0]))
            max_rng = float((seg["high"] - seg["low"]).max())
            if net >= c.accel_net_atr_mult * atr_v:
                return True, f"{name} net move {net/atr_v:.1f}xATR over last {win} bars"
            if max_rng >= c.accel_bar_atr_mult * atr_v:
                return True, f"{name} bar range {max_rng/atr_v:.1f}xATR within last {win} bars"
        return False, ""

    def _judge_accel_round(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
                           side: str, flag_ts: pd.Timestamp, rnd: int) -> tuple[Optional[bool], str]:
        c = self.cfg
        need15, need5 = rnd, 4 * rnd
        b15 = df_15m[df_15m.index >= flag_ts]
        b5 = df_5m[df_5m.index >= flag_ts]
        if len(b15) < need15 or len(b5) < need5:
            return None, (f"round {rnd}: waiting for post-flag bars "
                          f"(15m {len(b15)}/{need15}, 5m {len(b5)}/{need5})")
        r15 = b15.iloc[need15 - 1]
        r5 = b5.iloc[4 * (rnd - 1): 4 * rnd]
        fav15 = (float(r15["close"]) > float(r15["open"])) if side == LONG \
            else (float(r15["close"]) < float(r15["open"]))
        fav5 = ((r5["close"].values > r5["open"].values) if side == LONG
                else (r5["close"].values < r5["open"].values))
        n5 = int(fav5.sum())
        ok = fav15 and n5 >= c.accel_round_5m_min
        detail = f"round {rnd}: 15m {'with' if fav15 else 'against'} {side}, 5m {n5}/4 with {side}"
        return ok, detail

    # ── shared HMA/ATR computation (Layer 3.3 + check_exit) ──────────────────
    @staticmethod
    def _hma_atr(df_15m: pd.DataFrame, cfg: Config):
        closes = df_15m["close"]
        hma_f = ind.hma(closes, cfg.hma_fast_length)
        hma_s = ind.hma(closes, cfg.hma_slow_length)
        atr_s = ind.atr(df_15m, cfg.sl_atr_period)
        return hma_f, hma_s, atr_s

    # ── combined entry decision ──────────────────────────────────────────────
    def analyze(self, df_30m: pd.DataFrame, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
               direction: str, symbol: str) -> EntryResult:
        c = self.cfg
        if direction not in (LONG, SHORT):
            return EntryResult(NONE, False, "no direction from Bias layer — nothing to time")

        # ── Layer 3.1 ──────────────────────────────────────────────────────────
        l31 = self._layer31(df_30m, direction)
        if not l31.passed:
            return EntryResult(NONE, False, l31.reason, layer31_passed_count=l31.passed_count)

        min_len = max(c.hma_slow_length * 2, 30) + 5
        if len(df_15m) < min_len:
            return EntryResult(NONE, False, "insufficient 15m history", layer31_passed_count=l31.passed_count)

        hma_f, hma_s, atr_s = self._hma_atr(df_15m, c)
        cur_f, cur_s = float(hma_f.iloc[-1]), float(hma_s.iloc[-1])
        prev_f, prev_s = float(hma_f.iloc[-2]), float(hma_s.iloc[-2])
        atr_now = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0
        close = float(df_15m["close"].iloc[-1])
        bar_ts = df_15m.index[-1]

        long_cross = prev_f <= prev_s and cur_f > cur_s
        short_cross = prev_f >= prev_s and cur_f < cur_s

        state = self._get_state(symbol)
        if long_cross or short_cross:
            state.cross_direction = LONG if long_cross else SHORT
            state.cross_id = bar_ts
            state.cross_used = False
            state.waiting_for_new_cross = False
            # a brand new cross invalidates any Layer 3.2 wait tied to the old one
            old_wait = self._accel_wait.get(symbol)
            if old_wait is not None and old_wait.get("cross_id") != state.cross_id:
                self._accel_wait.pop(symbol, None)

        if atr_now <= 0 or np.isnan(atr_now):
            return EntryResult(NONE, False, "ATR unavailable", price=close, hma_fast=cur_f,
                               hma_slow=cur_s, layer31_passed_count=l31.passed_count)

        extension_atr = abs(close - cur_s) / atr_now
        is_long = direction == LONG
        base = dict(price=close, hma_fast=cur_f, hma_slow=cur_s, atr=atr_now,
                   extension_atr=extension_atr, cross_id=state.cross_id,
                   layer31_passed_count=l31.passed_count)

        aligned = (cur_f > cur_s) if is_long else (cur_f < cur_s)
        close_ok = (close > cur_s) if is_long else (close < cur_s)
        not_extended = extension_atr <= c.entry_max_distance_from_hma_atr
        cross_pending = (state.cross_direction == direction and not state.cross_used
                        and not state.waiting_for_new_cross)

        # ── Layer 3.3 checks ────────────────────────────────────────────────────
        if not cross_pending:
            if state.waiting_for_new_cross:
                reason = f"L3.3: {direction} blocked — waiting for a new cross after prior exit"
            else:
                reason = f"L3.3: {direction} blocked — no pending HMA{c.hma_fast_length}xHMA{c.hma_slow_length} cross"
            return EntryResult(NONE, False, reason, **base)
        if not aligned:
            return EntryResult(NONE, False, f"L3.3: {direction} blocked — HMA alignment not held", **base)
        if not close_ok:
            return EntryResult(NONE, False,
                               f"L3.3: {direction} blocked — close not past HMA{c.hma_slow_length} "
                               f"({close:.6f} vs {cur_s:.6f})", **base)
        if not not_extended:
            return EntryResult(NONE, False,
                               f"L3.3 BLOCKED Reason=PRICE_TOO_EXTENDED ExtensionATR={extension_atr:.2f} "
                               f"Maximum={c.entry_max_distance_from_hma_atr:.2f}", **base)

        # ── Layer 3.2 — gates this now-pending Layer 3.3 trigger ────────────────
        if c.accel_confirm_enabled:
            accel_state = self._accel_wait.get(symbol)
            if accel_state is None:
                flag, why = self._recent_acceleration(df_15m, df_5m)
                if flag:
                    self._accel_wait[symbol] = {
                        "cross_id": state.cross_id, "side": direction, "round": 1,
                        "last_bar": bar_ts, "flag_ts": bar_ts + pd.Timedelta(c.tf_fast),
                    }
                    return EntryResult(NONE, False, f"L3.2 WAIT round 1: {why}",
                                       layer32_status="waiting", **base)
            else:
                if accel_state["last_bar"] == bar_ts:
                    return EntryResult(NONE, False,
                                       f"L3.2 WAIT round {accel_state['round']} pending (same bar)",
                                       layer32_status="waiting", **base)
                accel_state["last_bar"] = bar_ts
                verdict, detail = self._judge_accel_round(
                    df_15m, df_5m, direction, accel_state["flag_ts"], accel_state["round"])
                if verdict is None:
                    return EntryResult(NONE, False, f"L3.2 WAIT: {detail}",
                                       layer32_status="waiting", **base)
                if not verdict:
                    if accel_state["round"] >= c.accel_max_rounds:
                        self._accel_wait.pop(symbol, None)
                        return EntryResult(NONE, False,
                                           f"L3.2 FAILED both rounds ({detail}) — setup abandoned",
                                           layer32_status="failed", **base)
                    accel_state["round"] += 1
                    return EntryResult(NONE, False, f"L3.2 WAIT round {accel_state['round']}: {detail}",
                                       layer32_status="waiting", **base)
                self._accel_wait.pop(symbol, None)

        # ── all three layers cleared — fire ──────────────────────────────────────
        state.cross_used = True
        return EntryResult(direction, True,
                           f"ENTRY {direction}  L3.1={l31.passed_count}/5  "
                           f"L3.3 HMA{c.hma_fast_length}xHMA{c.hma_slow_length} ext={extension_atr:.2f}  "
                           f"CrossID={state.cross_id}",
                           entry_score=100.0, layer32_status="clear", **base)

    # ── Early exit (shares the Layer 3.3 HMA/ATR computation) ────────────────
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
