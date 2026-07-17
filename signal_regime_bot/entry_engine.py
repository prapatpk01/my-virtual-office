"""
Layer 3 — Entry Engine (3-layer multi-timeframe cross confluence).

Regime (4H+1H) and Bias (1H+15M+5M) decide the SIDE. This layer receives
that fixed `direction` and only searches for a timing trigger on that side.
Three cross layers, each on its own timeframe, watch for a cross in the
bias direction:

    L3a — HMA(l3a_hma_fast)/HMA(l3a_hma_slow) cross on 30M
    L3b — EMA(l3b_ema_fast)/EMA(l3b_ema_slow) cross on 15M
    L3c — EMA(l3c_ema_fast)/EMA(l3c_ema_slow) cross on 5M  (also the exit gate)

Any single cross ARMS a setup. The entry fires once >= entry_confluence_min
(2) of the three layers have crossed the SAME (bias) direction within
entry_confluence_window_min (15) minutes of "now" (the latest closed 5M
bar). If a second layer never confirms, the first cross ages out of the
window -> the setup fails and the engine waits for a new one. One entry per
setup: re-arming needs a genuinely newer cross than the one that last fired.

Early exit (`check_exit`) — the HARD gate uses L3c (5M) only, so noise on
the slower layers can't shake us out:
    EMA_CROSS_REVERSAL     — L3c fast EMA crosses back against the position
    PRICE_OPEN_BEYOND_EMA  — bar OPEN on the wrong side of the L3c slow EMA
Suppressed for exit_grace_bars closed 5M bars after entry. Does NOT replace
the hard SL/TP — an additional, faster path evaluated once per closed 5M bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import indicators as ind
from config import Config

LONG  = "LONG"
SHORT = "SHORT"
NONE  = "NONE"

EMA_CROSS_REVERSAL = "EMA_CROSS_REVERSAL"
PRICE_OPEN_BEYOND_EMA = "PRICE_OPEN_BEYOND_EMA"

# TF label -> minutes, for cross-freshness math across layers.
_TF_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


@dataclass
class EntryResult:
    direction: str             # LONG | SHORT | NONE
    allow_entry: bool
    reason: str = ""
    price: float = 0.0
    ema_fast: float = 0.0      # L3c fast EMA (for logging/telegram)
    ema_slow: float = 0.0      # L3c slow EMA
    macd_hist: float = 0.0     # repurposed: # of layers confirming (for log compat)
    cross_id: object = None
    entry_score: float = 0.0   # 100 on a valid trigger, 0 otherwise


@dataclass
class ExitCheckResult:
    should_exit: bool
    reason: str = ""           # EMA_CROSS_REVERSAL | PRICE_OPEN_BEYOND_EMA | ""
    detail: str = ""


@dataclass
class _LayerCross:
    """Most recent cross seen on ONE layer."""
    direction: Optional[str] = None      # LONG | SHORT | None
    close_ts: Optional[pd.Timestamp] = None   # close time of the bar the cross fired on
    bar_ts: Optional[pd.Timestamp] = None     # open ts of that bar (dedupe guard)


@dataclass
class _SetupState:
    """Per-symbol confluence state."""
    layers: dict = field(default_factory=lambda: {"L3a": _LayerCross(),
                                                   "L3b": _LayerCross(),
                                                   "L3c": _LayerCross()})
    last_entry_ts: Optional[pd.Timestamp] = None   # newest cross ts consumed by the last entry


class EntryEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: dict[str, _SetupState] = {}

    # ── per-symbol state ──────────────────────────────────────────────────────
    def _get_state(self, symbol: str) -> _SetupState:
        return self._state.setdefault(symbol, _SetupState())

    def on_position_closed(self, symbol: str) -> None:
        """Call whenever a position FULLY closes. last_entry_ts already blocks
        re-entry until a cross NEWER than it appears, so nothing else is
        needed — kept for API parity with callers."""
        # no-op: last_entry_ts is the re-arm guard.
        self._get_state(symbol)

    def reset_symbol(self, symbol: str) -> None:
        self._state.pop(symbol, None)

    # ── indicators ────────────────────────────────────────────────────────────
    def _l3c_emas(self, df_5m: pd.DataFrame):
        c = self.cfg
        close = df_5m["close"]
        return ind.ema(close, c.l3c_ema_fast), ind.ema(close, c.l3c_ema_slow)

    @staticmethod
    def _tf_min(tf: str) -> int:
        return _TF_MIN.get(tf, 5)

    def _detect_cross(self, df: pd.DataFrame, fast: pd.Series, slow: pd.Series,
                      tf: str) -> Optional[tuple]:
        """Return (direction, close_ts, bar_ts) if the LAST closed bar of `df`
        is a fresh fast/slow cross, else None."""
        if df is None or len(df) < 2:
            return None
        cur_f, cur_s = float(fast.iloc[-1]), float(slow.iloc[-1])
        prev_f, prev_s = float(fast.iloc[-2]), float(slow.iloc[-2])
        bar_ts = df.index[-1]
        close_ts = bar_ts + pd.Timedelta(minutes=self._tf_min(tf))
        if prev_f <= prev_s and cur_f > cur_s:
            return LONG, close_ts, bar_ts
        if prev_f >= prev_s and cur_f < cur_s:
            return SHORT, close_ts, bar_ts
        return None

    # ── cross bookkeeping ─────────────────────────────────────────────────────
    def observe(self, df_30m: pd.DataFrame, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
                symbol: str) -> None:
        """Record any fresh cross on each layer's last closed bar. Called once
        per evaluation, regardless of Regime/Bias. Idempotent per (layer, bar)."""
        c = self.cfg
        state = self._get_state(symbol)
        specs = (
            ("L3a", df_30m, c.l3a_tf, ind.hma, c.l3a_hma_fast, c.l3a_hma_slow),
            ("L3b", df_15m, c.l3b_tf, ind.ema, c.l3b_ema_fast, c.l3b_ema_slow),
            ("L3c", df_5m,  c.l3c_tf, ind.ema, c.l3c_ema_fast, c.l3c_ema_slow),
        )
        for name, df, tf, fn, pf, ps in specs:
            if df is None or len(df) < max(ps * 2, 5):
                continue
            lc = state.layers[name]
            bar_ts = df.index[-1]
            if lc.bar_ts == bar_ts:
                continue   # this bar already processed for this layer
            fast = fn(df["close"], pf)
            slow = fn(df["close"], ps)
            hit = self._detect_cross(df, fast, slow, tf)
            # advance the per-layer "seen" marker even when there's no cross, so
            # we don't re-scan the same bar every tick.
            if hit is None:
                lc.bar_ts = bar_ts
                continue
            lc.direction, lc.close_ts, lc.bar_ts = hit

    # ── combined entry decision ──────────────────────────────────────────────
    def analyze(self, df_30m: pd.DataFrame, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
               direction: str, symbol: str) -> EntryResult:
        c = self.cfg
        if direction not in (LONG, SHORT):
            return EntryResult(NONE, False, "no direction from Bias layer — nothing to time")

        self.observe(df_30m, df_15m, df_5m, symbol)

        if df_5m is None or len(df_5m) < max(c.l3c_ema_slow * 2, 20):
            return EntryResult(NONE, False, "insufficient 5m history")

        state = self._get_state(symbol)
        now = df_5m.index[-1] + pd.Timedelta(minutes=self._tf_min(c.l3c_tf))
        window = pd.Timedelta(minutes=c.entry_confluence_window_min)

        ema_f, ema_s = self._l3c_emas(df_5m)
        cur_f, cur_s = float(ema_f.iloc[-1]), float(ema_s.iloc[-1])
        close_px = float(df_5m["close"].iloc[-1])

        # layers that crossed in `direction` and are still within the 45-min window
        active = []
        for name, lc in state.layers.items():
            if (lc.direction == direction and lc.close_ts is not None
                    and (now - lc.close_ts) <= window and (now - lc.close_ts) >= pd.Timedelta(0)):
                active.append((name, lc.close_ts))
        n_active = len(active)
        base = dict(price=close_px, ema_fast=cur_f, ema_slow=cur_s,
                    macd_hist=float(n_active), cross_id=state.last_entry_ts)

        if n_active < c.entry_confluence_min:
            armed = ",".join(n for n, _ in active) if active else "none"
            return EntryResult(NONE, False,
                               f"L3: {direction} setup {n_active}/{c.entry_confluence_min} "
                               f"layers in {c.entry_confluence_window_min}m window (armed: {armed})", **base)

        newest_ts = max(ts for _, ts in active)
        # one entry per setup — require a cross newer than the last one consumed
        if state.last_entry_ts is not None and newest_ts <= state.last_entry_ts:
            return EntryResult(NONE, False,
                               f"L3: {direction} confluence {n_active}/3 but no new cross since last "
                               f"entry — waiting for a fresh setup", **base)

        # ── Sideways veto (ported from TrendConfirm) — a trend system bleeds
        # in ranges, and a range can still produce cross confluence on a
        # bounce. 4 independent range reads on 15M; >= min_signals = veto.
        vetoed, side_detail = self._sideways_context(df_15m)
        if vetoed:
            return EntryResult(NONE, False,
                               f"L3: {direction} blocked — SIDEWAYS veto ({side_detail})", **base)

        # ── Chase guard (ported from TrendConfirm) — never buy a move that
        # already left: entry must be within chase_max_dist_atr of the 5M
        # reference EMA, else wait for a pullback + fresh setup.
        if c.chase_guard_enabled and len(df_5m) >= c.chase_ema_ref + 5:
            ema_ref = float(ind.ema(df_5m["close"], c.chase_ema_ref).iloc[-1])
            atr5 = float(ind.atr(df_5m, 14).iloc[-1])
            if atr5 > 0:
                dist_atr = abs(close_px - ema_ref) / atr5
                if dist_atr > c.chase_max_dist_atr:
                    return EntryResult(NONE, False,
                                       f"L3: {direction} blocked — CHASE guard: "
                                       f"{dist_atr:.2f}xATR from EMA{c.chase_ema_ref} (5m) "
                                       f"(max {c.chase_max_dist_atr:.2f}) — wait for pullback", **base)

        # Do NOT consume the setup here: analyze() runs on EVERY poll tick, but
        # the actual open can still be blocked downstream (symbol cooldown, max
        # positions, a daily/streak limit, a balance-read blip, open_position
        # returning None). Burning last_entry_ts now would discard a still-valid
        # 45-min setup and make the bot wait for a brand-new cross. The caller
        # calls confirm_entry() ONLY after a position really opened.
        names = "+".join(sorted(n for n, _ in active))
        return EntryResult(direction, True,
                           f"ENTRY {direction}  confluence {n_active}/3 ({names}) within "
                           f"{c.entry_confluence_window_min}m", entry_score=100.0,
                           **{**base, "cross_id": newest_ts})

    def _sideways_context(self, df_15m: pd.DataFrame) -> tuple:
        """Range detector on 15M (ported from TrendConfirm). Returns
        (vetoed, detail). Signals lean on EMA compression / high chop / tight
        range — which stay range-y while ADX lags at trend starts — and ADX
        only counts when REALLY weak (< sideways_adx_max), so a fresh trend
        at ADX ~18 isn't mistaken for a range."""
        c = self.cfg
        if not c.sideways_filter_enabled:
            return False, ""
        n = 20
        min_needed = max(50 + 2, 14 * 2 + 2, n + 1)
        if df_15m is None or len(df_15m) < min_needed:
            return False, ""
        closes = df_15m["close"]
        ema20 = float(ind.ema(closes, 20).iloc[-1])
        ema50 = float(ind.ema(closes, 50).iloc[-1])
        atr15 = float(ind.atr(df_15m, 14).iloc[-1])
        if not (atr15 > 0) or pd.isna(ema20) or pd.isna(ema50):
            return False, ""
        adx_s, _, _ = ind.adx(df_15m, 14)
        adx_v = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 100.0
        chop_v = float(ind.choppiness_index(df_15m, 14).iloc[-1])
        window = df_15m.iloc[-n:]
        rng_atr = float((window["high"].max() - window["low"].min()) / atr15)
        ema_gap_atr = abs(ema20 - ema50) / atr15

        sig = {
            "ema_compressed": ema_gap_atr < c.sideways_ema_compression_atr,
            "high_chop": (not pd.isna(chop_v)) and chop_v > c.sideways_chop_threshold,
            "tight_range": rng_atr < c.sideways_range_atr,
            "weak_adx": adx_v < c.sideways_adx_max,
        }
        count = sum(1 for v in sig.values() if v)
        fired = ",".join(k for k, v in sig.items() if v)
        detail = (f"{count} signals: {fired} | gap={ema_gap_atr:.2f}ATR "
                  f"chop={chop_v:.0f} adx={adx_v:.0f} range={rng_atr:.2f}ATR")
        return count >= c.sideways_min_signals, detail

    def confirm_entry(self, symbol: str, cross_ts) -> None:
        """Mark the confluence setup that produced `cross_ts` (EntryResult.cross_id)
        as consumed — call ONLY after a position actually opened. A blocked or
        failed open then leaves the setup armed for the rest of its 45-min window
        instead of silently eating it."""
        if cross_ts is not None:
            self._get_state(symbol).last_entry_ts = cross_ts

    # ── Early exit — L3c (5M EMA10/20) hard gate ─────────────────────────────
    def check_exit(self, df_5m: pd.DataFrame, position_side: str,
                   bars_since_entry: Optional[int] = None) -> ExitCheckResult:
        """position_side: 'long' | 'short'. While bars_since_entry <
        exit_grace_bars the hard exit is suppressed so the L3c EMAs can
        separate from entry. Only an L3c EMA cross-back or an OPEN on the wrong
        side of the L3c slow EMA closes a position — nothing on L3a/L3b."""
        c = self.cfg
        if bars_since_entry is not None and bars_since_entry < c.exit_grace_bars:
            return ExitCheckResult(False)
        if df_5m is None or len(df_5m) < max(c.l3c_ema_slow * 2, 20):
            return ExitCheckResult(False)

        ema_f, ema_s = self._l3c_emas(df_5m)
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
                                   f"L3c EMA{c.l3c_ema_fast} crossed back against {position_side}")
        if open_wrong:
            return ExitCheckResult(True, PRICE_OPEN_BEYOND_EMA,
                                   f"open={open_px:.6f} L3c EMA{c.l3c_ema_slow}={cur_s:.6f}")
        return ExitCheckResult(False)
