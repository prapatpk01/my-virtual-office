"""
3-Layer Trend-Confirmed Multi-TF Strategy.

Layer 1 — Trend confirmation (TF30m):
  SMA30      : candle opens above SMA30 -> up, below -> down
  EMA10/20   : EMA10 > EMA20 -> up, EMA10 < EMA20 -> down
  EMA20 slope: EMA20 higher than it was `ema_slope_lookback` bars ago -> up,
               lower -> down
  MACD       : raw MACD LINE (fast EMA - slow EMA) vs zero, the "MACD 4C"
               zero-line read (not line-vs-signal) — matches the
               user-supplied Pine script, sign only.
  ALL FOUR must agree on the same direction, or there is no confirmed
  trend at all (self._trend_state = None) and nothing below runs.
  Tracked once per new 30m bar in self._trend_state, purely for
  logging/gating — a confirmed trend does not expire on its own; it just
  gets re-evaluated fresh every 30m bar.

Layer 2 — Bias + momentum filter (TF15m + TF1H):
  Bias score: BaseStrategy.compute_mtf_bias() (same formula ai_expert
  uses: per-TF score from close>EMA20 / EMA20>EMA50 / RSI lean, weighted
  15m x1 + 1H x2 — 4H excluded here via w3=0) is converted from its
  native -100..+100 scale to 0-100 (50 = neutral) via `50 + pct/2`. Must
  AGREE with and exceed Layer 1's confirmed direction:
    uptrend   needs bias_score > bias_threshold        (default 60)
    downtrend needs bias_score < 100 - bias_threshold  (default 40)

  Momentum gates — ADX/Choppiness/Volume are each computed on BOTH 15m
  AND 1H and must pass on both timeframes (not just one):
    ADX(14)        > adx_threshold (default 20)   — trend has real strength
    Choppiness(14) < chop_threshold (default 61.8) — market isn't ranging/chop
    Volume ratio   > volume_expansion_mult (default 1.0x recent SMA20 volume)
                                                    — the move has real participation

  Liquidity sweep (15m, opposite-side): before an entry is allowed, one
  of the last `sweep_recency` (default 5) 15m bars must have swept the
  prior `sweep_lookback`-bar (default 20) swing extreme on the STOP-HUNT
  side and closed back inside it — long needs a swept swing LOW
  (wick below the recent low, close back above it — sell-side stops
  grabbed before the real move up); short needs a swept swing HIGH
  (mirror). No sweep = no trade.

  Any one of bias/ADX(15m or 1H)/chop(15m or 1H)/volume(15m or
  1H)/liquidity-sweep failing = no trade, regardless of Layer 1.

Layer 3 — Entry (TF15m):
  HMA10/HMA20 cross in the confirmed direction. Normally the cross must
  fire on THIS bar going forward ("wait for the next cross") — but if
  Layer 1's trend JUST confirmed (within `fresh_trend_bars`, default 3,
  bars of when it flipped), a cross that already fired up to
  fresh_trend_bars ago still counts, catching the case where price
  crossed right as the 30m trend was still forming instead of forcing a
  second cross after the fact. Long only when Layer1+Layer2 both say
  "up"; short only when both say "down". The cross timestamp is
  consumed (reset to None) once it triggers an entry attempt (pass or
  fail), so a stale old cross can't silently satisfy a future setup.
  (EMA5/EMA10 is no longer part of the entry trigger — it's still
  computed and shown in logs, but no longer used for the exit either.)

  Chase-guard: even when the HMA cross fires, price must be within
  `max_dist_atr_mult` (default 1.5) x ATR(14, 15m) of Layer1's
  EMA20(30m) — the same trend reference Layer1 uses for its EMA10/20
  and slope checks. If price already ran too far from it, the setup
  fails outright (no entry) and waits for a pullback + fresh cross,
  rather than chasing an already-extended move.

Exit — OR logic, whichever fires first while SL/TP hasn't been hit yet:
  HMA10/20 cross-back  OR  candle opens on the wrong side of HMA20
  (long: open < HMA20, short: open > HMA20) -> close 100% immediately.
  EMA5/10 no longer drives entry OR exit — it's still computed and
  tracked purely for [SCAN] log diagnostics. The open-vs-HMA20 check is
  a faster warning than waiting for HMA10 to actually cross back — the
  open can already be on the wrong side while HMA10/HMA20 haven't
  crossed yet. SL/TP
  (ATR(14, 15m) x1.5, 1:1 R:R by default) remains the hard-stop safety
  net checked by bot.py's risk-manager fallback underneath all of this.

Once closed (by either exit condition or the hard SL/TP stop), the very
next bar where Layer1+Layer2 read confirmed again — same direction as
before or a fresh flip — is eligible for a new Layer3 entry. No
cooldown, no "used up this streak."

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze() — in
hedge mode a SELL Signal always OPENS a new short rather than closing an
open long. tick_open_position()'s PositionUpdate("close") always closes
whichever position is actually open, regardless of hedge mode.

TF30m candles for Layer 1 aren't fetched separately from the exchange —
they're resampled from the 15m series this strategy already receives,
and the last (possibly still-forming) 30m bucket is dropped so Layer 1
only ever evaluates against a genuinely closed 30m bar. Layer 2/3 run
directly on the raw 15m candles (and the 1H candles passed in via
mtf_candles), matching how every other pure-TF15m strategy in this
codebase evaluates crosses (no separate "closed 15m" bookkeeping).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


class TrendConfirmStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        # Layer 1 — trend confirmation (30m)
        sma_trend: int = 30,
        ema1_period: int = 10,
        ema2_period: int = 20,
        ema_slope_lookback: int = 5,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # Layer 2 — bias/momentum filter (15m + 1H)
        bias_threshold: float = 60.0,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        chop_period: int = 14,
        chop_threshold: float = 61.8,
        volume_sma_period: int = 20,
        volume_expansion_mult: float = 1.0,
        sweep_lookback: int = 20,
        sweep_recency: int = 5,
        # Layer 3 — entry (15m)
        ema_fast: int = 5,
        ema_slow: int = 10,
        hma_fast: int = 10,
        hma_slow: int = 20,
        fresh_trend_bars: int = 3,
        max_dist_atr_mult: float = 1.5,
        # Risk
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        rr_ratio: float = 1.0,
    ):
        super().__init__(symbol, params)
        self.name = f"TrendConfirm({symbol})"

        self.sma_trend = sma_trend
        self.ema1_period = ema1_period
        self.ema2_period = ema2_period
        self.ema_slope_lookback = ema_slope_lookback
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        self.bias_threshold = bias_threshold
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.chop_period = chop_period
        self.chop_threshold = chop_threshold
        self.volume_sma_period = volume_sma_period
        self.volume_expansion_mult = volume_expansion_mult
        self.sweep_lookback = sweep_lookback
        self.sweep_recency = sweep_recency

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.hma_fast = hma_fast
        self.hma_slow = hma_slow
        self.fresh_trend_bars = fresh_trend_bars
        self.max_dist_atr_mult = max_dist_atr_mult

        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._trend_state: Optional[str] = None      # "up" | "down" | None — Layer 1 result
        self._trend_confirmed_since_ts: Optional[int] = None  # bar_ts_15 when trend last CHANGED
        self._last_bar_ts_30: Optional[int] = None    # Layer 1 new-bar tracking
        self._last_bar_ts_15: Optional[int] = None     # Layer 3 new-bar tracking
        self._last_ema_cross_up_ts: Optional[int] = None
        self._last_ema_cross_down_ts: Optional[int] = None
        self._last_hma_cross_up_ts: Optional[int] = None
        self._last_hma_cross_down_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None  # owned by tick_open_position()
        self._latest_candles: list = []

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles  # cached for tick_open_position()
        mtf = mtf_candles or {}
        bar_ts_15 = candles[-1].timestamp

        # ── Layer 1: Trend confirmation (30m) ──────────────────────────────
        c30 = self._closed_30m_bars(candles)
        min_needed_30 = max(self.ema2_period + self.ema_slope_lookback, self.sma_trend,
                            self.macd_slow + self.macd_signal) + 5
        if len(c30) < min_needed_30:
            return self._hold(current_price, f"Layer1: need {min_needed_30}+ closed 30m bars, have {len(c30)}")

        bar_ts_30 = c30[-1].timestamp
        is_new_bar_30 = bar_ts_30 != self._last_bar_ts_30

        l1 = self._layer1_indicators(c30)
        if l1 is None:
            return self._hold(current_price, "Layer1: indicators still warming up (30m)")

        if is_new_bar_30:
            self._last_bar_ts_30 = bar_ts_30
            if l1["trend"] != self._trend_state:
                self._trend_confirmed_since_ts = bar_ts_15
            self._trend_state = l1["trend"]

        trend = self._trend_state

        # ── Layer 3 cross tracking — runs on every new 15m bar REGARDLESS of
        # Layer1/Layer2 gating below, so an HMA cross that fires before the
        # trend confirms (or while a position is open) is still remembered.
        # EMA5/10 is tracked too (still shown in logs), but it no longer
        # gates entry or exit. ─────────────────────────────────────────────
        is_new_bar_15 = bar_ts_15 != self._last_bar_ts_15
        l3 = self._layer3_indicators(candles)
        if is_new_bar_15:
            self._last_bar_ts_15 = bar_ts_15
            if l3 is not None:
                if l3["ema_cross_up"]:
                    self._last_ema_cross_up_ts = bar_ts_15
                if l3["ema_cross_down"]:
                    self._last_ema_cross_down_ts = bar_ts_15
                if l3["hma_cross_up"]:
                    self._last_hma_cross_up_ts = bar_ts_15
                if l3["hma_cross_down"]:
                    self._last_hma_cross_down_ts = bar_ts_15

        def _bars_ago_15(ts: Optional[int]) -> Optional[int]:
            return (bar_ts_15 - ts) // (15 * 60_000) if ts is not None else None

        ema_up_ago    = _bars_ago_15(self._last_ema_cross_up_ts)
        ema_down_ago  = _bars_ago_15(self._last_ema_cross_down_ts)
        hma_up_ago    = _bars_ago_15(self._last_hma_cross_up_ts)
        hma_down_ago  = _bars_ago_15(self._last_hma_cross_down_ts)

        # Entry trigger is HMA10/20 cross alone. Normally it must fire on
        # THIS bar going forward (0 bars ago — "wait for the next cross").
        # If the trend JUST confirmed (within fresh_trend_bars of Layer1
        # flipping), a cross that already fired up to fresh_trend_bars ago
        # still counts — catches the case where price already crossed
        # right as the 30m trend was forming, instead of forcing a second
        # fresh cross after the fact.
        fb = self.fresh_trend_bars
        trend_age = _bars_ago_15(self._trend_confirmed_since_ts)
        is_fresh_trend = trend_age is not None and trend_age <= fb
        lookback = fb if is_fresh_trend else 0
        dual_cross_up   = hma_up_ago is not None and hma_up_ago <= lookback
        dual_cross_down = hma_down_ago is not None and hma_down_ago <= lookback

        # ── Layer 2: Bias/momentum filter (15m + 1H) ───────────────────────
        c1h = mtf.get("1h", [])
        bias_pct, bias_label = self.compute_mtf_bias(candles, {"1h": c1h}, w3=0.0)
        bias_score = 50.0 + bias_pct / 2.0
        l2_15m = self._layer2_indicators(candles)
        l2_1h = self._layer2_indicators(c1h)
        sweep_up = self._liquidity_sweep(candles, "up")
        sweep_down = self._liquidity_sweep(candles, "down")

        close_price = candles[-1].close
        dist_atr = (abs(close_price - l1["ema20_val"]) / l3["atr_val"]
                   if (l3 is not None and l3["atr_val"] > 0) else None)

        def dbg(entry_status: str) -> dict:
            return {"trend_confirm": {
                "sma_trend": l1["sma_dir"], "ema10_20_trend": l1["ema1020_dir"],
                "ema20_slope": l1["slope_dir"], "macd_trend": l1["macd_dir"],
                "confirmed": trend,
                "bias_score": round(bias_score, 1), "bias_threshold": self.bias_threshold,
                "adx_15m": round(l2_15m["adx"], 1) if l2_15m else None,
                "adx_1h": round(l2_1h["adx"], 1) if l2_1h else None,
                "adx_threshold": self.adx_threshold,
                "chop_15m": round(l2_15m["chop"], 1) if l2_15m else None,
                "chop_1h": round(l2_1h["chop"], 1) if l2_1h else None,
                "chop_threshold": self.chop_threshold,
                "vol_ratio_15m": round(l2_15m["vol_ratio"], 2) if l2_15m else None,
                "vol_ratio_1h": round(l2_1h["vol_ratio"], 2) if l2_1h else None,
                "vol_expansion_mult": self.volume_expansion_mult,
                "sweep_up": sweep_up, "sweep_down": sweep_down,
                "open_position": self._open_position,
                "entry_status": entry_status,
                "fresh_trend_bars": fb, "trend_age_bars": trend_age, "is_fresh_trend": is_fresh_trend,
                "ema_cross_up_ago": ema_up_ago, "ema_cross_down_ago": ema_down_ago,
                "hma_cross_up_ago": hma_up_ago, "hma_cross_down_ago": hma_down_ago,
                "dist_atr": round(dist_atr, 2) if dist_atr is not None else None,
                "max_dist_atr": self.max_dist_atr_mult,
            }}

        if self._open_position is not None:
            return self._hold(current_price, f"Holding {self._open_position.upper()} — managed via tick_open_position()",
                              metadata=dbg("position_open"))

        if trend is None:
            return self._hold(current_price,
                "Layer1: SMA30/EMA10-20/EMA20 slope/MACD not confirmed or conflicting",
                metadata=dbg("no_trend"))

        bias_pass = (
            (trend == "up" and bias_score > self.bias_threshold) or
            (trend == "down" and bias_score < (100.0 - self.bias_threshold))
        )
        if not bias_pass:
            return self._hold(current_price,
                f"Layer2: bias score {bias_score:.0f} doesn't confirm {trend} "
                f"(need {'>' if trend == 'up' else '<'}"
                f"{self.bias_threshold if trend == 'up' else 100 - self.bias_threshold:.0f})",
                metadata=dbg("bias_fail"))

        if l2_15m is None or l2_1h is None:
            return self._hold(current_price, "Layer2: ADX/chop/volume indicators still warming up (15m/1H)",
                              metadata=dbg("no_trend"))
        if not (l2_15m["adx_pass"] and l2_1h["adx_pass"]):
            return self._hold(current_price,
                f"Layer2: ADX 15m={l2_15m['adx']:.1f} 1H={l2_1h['adx']:.1f} below {self.adx_threshold:.0f} "
                f"on at least one TF — trend too weak",
                metadata=dbg("momentum_fail"))
        if not (l2_15m["chop_pass"] and l2_1h["chop_pass"]):
            return self._hold(current_price,
                f"Layer2: Choppiness 15m={l2_15m['chop']:.1f} 1H={l2_1h['chop']:.1f} above {self.chop_threshold:.1f} "
                f"on at least one TF — market too choppy/ranging",
                metadata=dbg("momentum_fail"))
        if not (l2_15m["vol_pass"] and l2_1h["vol_pass"]):
            return self._hold(current_price,
                f"Layer2: volume 15m={l2_15m['vol_ratio']:.2f}x 1H={l2_1h['vol_ratio']:.2f}x below "
                f"{self.volume_expansion_mult:.2f}x on at least one TF — no volume expansion",
                metadata=dbg("momentum_fail"))

        sweep_pass = sweep_up if trend == "up" else sweep_down
        if not sweep_pass:
            swept_what = "swing LOW" if trend == "up" else "swing HIGH"
            return self._hold(current_price,
                f"Layer2: no liquidity sweep of the recent {swept_what} in the last "
                f"{self.sweep_recency} bars — no stop-hunt/reclaim confirming real participation",
                metadata=dbg("sweep_fail"))

        # ── Layer 3: Entry (15m) — HMA10/20 must have crossed in the
        # confirmed direction (fresh cross now, or within fresh_trend_bars
        # if the trend just confirmed), AND price must not have run more
        # than max_dist_atr_mult x ATR(15m) away from Layer1's EMA20(30m) —
        # a chase-guard: if the cross fires but price already ran too far
        # from the trend reference, skip this setup and wait for a
        # pullback + fresh cross instead of chasing. ───────────────────────
        if l3 is None:
            return self._hold(current_price, "Layer3: indicators still warming up (15m)", metadata=dbg("no_trend"))

        dist_ok = dist_atr is not None and dist_atr <= self.max_dist_atr_mult
        dist_disp = f"{dist_atr:.2f}" if dist_atr is not None else "n/a"

        if trend == "up":
            if dual_cross_up and not dist_ok:
                self._last_hma_cross_up_ts = None
                return self._hold(current_price,
                    f"Long setup FAILED: price {dist_disp}xATR from EMA20 (max {self.max_dist_atr_mult}x) "
                    f"— waiting for pullback + fresh cross",
                    metadata=dbg("cross_pass_distance_fail"))
            if dual_cross_up:
                sl, tp = self._compute_sl_tp("long", close_price, l3["atr_val"])
                self._open_position = "long"
                self._last_hma_cross_up_ts = None
                meta = dbg("entered")
                meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
                return Signal(
                    type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                    reason=f"Uptrend confirmed (Layer1 30m) + bias {bias_score:.0f}/ADX 15m={l2_15m['adx']:.0f} "
                           f"1H={l2_1h['adx']:.0f}/sweep-low ok (Layer2) + "
                           f"HMA10↑HMA20 crossed {hma_up_ago} bar(s) ago, "
                           f"{dist_atr:.2f}xATR from EMA20 (Layer3)",
                    confidence=1.0,
                    metadata=meta,
                )
            return self._hold(current_price, "Layer1+2 confirmed UP — waiting for HMA10/20 cross"
                              + (f" (trend fresh, within {fb} bars)" if is_fresh_trend else " (fresh cross only)"),
                              metadata=dbg("waiting_cross"))

        # trend == "down"
        if dual_cross_down and not dist_ok:
            self._last_hma_cross_down_ts = None
            return self._hold(current_price,
                f"Short setup FAILED: price {dist_disp}xATR from EMA20 (max {self.max_dist_atr_mult}x) "
                f"— waiting for pullback + fresh cross",
                metadata=dbg("cross_pass_distance_fail"))
        if dual_cross_down:
            sl, tp = self._compute_sl_tp("short", close_price, l3["atr_val"])
            self._open_position = "short"
            self._last_hma_cross_down_ts = None
            meta = dbg("entered")
            meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
                reason=f"Downtrend confirmed (Layer1 30m) + bias {bias_score:.0f}/ADX 15m={l2_15m['adx']:.0f} "
                       f"1H={l2_1h['adx']:.0f}/sweep-high ok (Layer2) + "
                       f"HMA10↓HMA20 crossed {hma_down_ago} bar(s) ago, "
                       f"{dist_atr:.2f}xATR from EMA20 (Layer3)",
                confidence=1.0,
                metadata=meta,
            )
        return self._hold(current_price, "Layer1+2 confirmed DOWN — waiting for HMA10/20 cross"
                          + (f" (trend fresh, within {fb} bars)" if is_fresh_trend else " (fresh cross only)"),
                          metadata=dbg("waiting_cross"))

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Exit = OR logic, whichever fires first, evaluated once per
        newly-formed 15m bar: HMA10/20 cross-back, OR the candle opens on
        the wrong side of HMA20 (a faster warning than waiting for the
        cross itself — the open can flip against the position while
        HMA10 hasn't crossed back yet). EMA5/10 is no longer part of the
        exit (or the entry) — it's still computed and tracked purely for
        [SCAN] log diagnostics. Hedge-mode-safe: always closes whichever
        position is actually open, never relies on signal.type semantics."""
        if self._open_position is None or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate

        candles = self._latest_candles
        bar_ts = candles[-1].timestamp
        if bar_ts == self._last_exit_bar_ts:
            return PositionUpdate(action="hold", reason="Waiting for the next 15m bar close")

        l3 = self._layer3_indicators(candles)
        if l3 is None:
            return PositionUpdate(action="hold", reason="Indicators warming up (15m)")
        self._last_exit_bar_ts = bar_ts

        if self._open_position == "long" and (l3["hma_cross_down"] or l3["open_below_hma20"]):
            reason = "HMA10 crossed below HMA20" if l3["hma_cross_down"] else "candle opened below HMA20"
            self._open_position = None
            return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit LONG: {reason} (15m)")
        if self._open_position == "short" and (l3["hma_cross_up"] or l3["open_above_hma20"]):
            reason = "HMA10 crossed above HMA20" if l3["hma_cross_up"] else "candle opened above HMA20"
            self._open_position = None
            return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit SHORT: {reason} (15m)")

        return PositionUpdate(action="hold", reason=f"Holding {self._open_position.upper()}")

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Called by bot.py after ANY close, including the risk-manager's
        hard SL/TP fallback firing before EMA5/10 or HMA10/20 crosses back —
        without this, _open_position would stay set forever and analyze()
        would refuse all future entries."""
        self._open_position = None

    def cancel_pending_entry(self, reason: str = "") -> None:
        """Called by bot.py when a signal this strategy just emitted failed
        to actually open (rejected by risk/portfolio gates, insufficient
        balance, or an order error)."""
        self._open_position = None

    def attach_existing_position(self, direction: str, entry_price: float,
                                  stop_loss: Optional[float] = None,
                                  take_profit: Optional[float] = None) -> None:
        """Called once on bot startup when a position for this symbol is
        already open on the exchange (from before a restart) — nothing in
        this strategy's in-memory state would otherwise know about it, so
        analyze() would try to open a duplicate and tick_open_position()
        would never manage the exit."""
        self._open_position = direction

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_sl_tp(self, direction: str, price: float, atr: float) -> tuple[float, float]:
        dist = self.sl_atr_mult * atr
        if direction == "long":
            sl = price - dist
            tp = price + dist * self.rr_ratio
        else:
            sl = price + dist
            tp = price - dist * self.rr_ratio
        return sl, tp

    def _layer1_indicators(self, c30: list) -> Optional[dict]:
        closes = [c.close for c in c30]
        ema1 = self.ema(closes, self.ema1_period)
        ema2 = self.ema(closes, self.ema2_period)
        sma_t = self.sma(closes, self.sma_trend)
        macd_line, macd_signal_line, hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)

        lb = self.ema_slope_lookback
        needed = [ema1[-1], ema2[-1], ema2[-1 - lb], sma_t[-1], macd_line[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        last = c30[-1]
        sma_up, sma_down = last.open > sma_t[-1], last.open < sma_t[-1]
        ema1020_up, ema1020_down = ema1[-1] > ema2[-1], ema1[-1] < ema2[-1]
        slope_up, slope_down = ema2[-1] > ema2[-1 - lb], ema2[-1] < ema2[-1 - lb]
        # "MACD 4C" trend read: raw MACD line vs zero, not vs its signal
        # line — matches the user-supplied Pine script (sign only).
        macd_up, macd_down = macd_line[-1] > 0, macd_line[-1] < 0

        trend_up = sma_up and ema1020_up and slope_up and macd_up
        trend_down = sma_down and ema1020_down and slope_down and macd_down

        return {
            "trend": "up" if trend_up else "down" if trend_down else None,
            "sma_dir": "up" if sma_up else "down" if sma_down else "flat",
            "ema1020_dir": "up" if ema1020_up else "down" if ema1020_down else "flat",
            "slope_dir": "up" if slope_up else "down" if slope_down else "flat",
            "macd_dir": "up" if macd_up else "down" if macd_down else "flat",
            "ema20_val": float(ema2[-1]),
        }

    def _layer2_indicators(self, candles: list) -> Optional[dict]:
        """Momentum context on top of the bias score: ADX (trend strength),
        Choppiness Index (trending vs ranging), and a volume-expansion
        ratio (current bar vs its recent average). Called on both the 15m
        candles and the 1H candles from mtf_candles — both must pass."""
        min_needed = max(2 * self.adx_period + 2, self.chop_period + 1, self.volume_sma_period)
        if len(candles) < min_needed:
            return None
        adx_arr, _plus_di, _minus_di = self.adx(candles, self.adx_period)
        if np.isnan(adx_arr[-1]):
            return None
        adx_val = float(adx_arr[-1])

        chop_val = self._choppiness(candles, self.chop_period)
        if chop_val is None:
            return None

        vols = [c.volume for c in candles]
        vol_sma = self.sma(vols, self.volume_sma_period)
        if np.isnan(vol_sma[-1]) or vol_sma[-1] <= 0:
            return None
        vol_ratio = float(candles[-1].volume) / float(vol_sma[-1])

        return {
            "adx": adx_val, "adx_pass": adx_val > self.adx_threshold,
            "chop": chop_val, "chop_pass": chop_val < self.chop_threshold,
            "vol_ratio": vol_ratio, "vol_pass": vol_ratio > self.volume_expansion_mult,
        }

    @staticmethod
    def _choppiness(candles: list, period: int) -> Optional[float]:
        """Choppiness Index over `period` bars: 100 = pure chop/ranging,
        0 = a strong sustained trend. CHOP = 100 * log10(sum(TR, period) /
        (highest_high - lowest_low)) / log10(period)."""
        if len(candles) < period + 1:
            return None
        window = candles[-(period + 1):]  # +1 seed candle for the first bar's prev close
        trs = []
        for i in range(1, len(window)):
            h, l, pc = window[i].high, window[i].low, window[i - 1].close
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        period_bars = window[1:]
        highest = max(c.high for c in period_bars)
        lowest = min(c.low for c in period_bars)
        rng = highest - lowest
        atr_sum = sum(trs)
        if rng <= 0 or atr_sum <= 0:
            return None
        return 100.0 * math.log10(atr_sum / rng) / math.log10(period)

    def _liquidity_sweep(self, candles: list, trend: str) -> bool:
        """True if, within the last `sweep_recency` 15m bars, one bar wicked
        beyond the `sweep_lookback`-bar swing extreme on the stop-hunt side
        and closed back inside it — a liquidity sweep + reclaim.
        trend="up"   -> looks for a swept swing LOW  (sell-side stops grabbed
                        below the recent low, then price closed back above it)
        trend="down" -> looks for a swept swing HIGH (mirror)."""
        lb, rec = self.sweep_lookback, self.sweep_recency
        if len(candles) < lb + rec + 1:
            return False
        for i in range(len(candles) - rec, len(candles)):
            window = candles[i - lb:i]  # the lb bars strictly before candidate bar i
            if len(window) < lb:
                continue
            bar = candles[i]
            if trend == "up":
                swing_low = min(c.low for c in window)
                if bar.low < swing_low and bar.close > swing_low:
                    return True
            else:
                swing_high = max(c.high for c in window)
                if bar.high > swing_high and bar.close < swing_high:
                    return True
        return False

    def _layer3_indicators(self, candles: list) -> Optional[dict]:
        closes = [c.close for c in candles]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        hma_f = self.hma(closes, self.hma_fast)
        hma_s = self.hma(closes, self.hma_slow)
        atr_arr = self.atr(candles, self.atr_period)

        needed = [ema_f[-1], ema_f[-2], ema_s[-1], ema_s[-2],
                 hma_f[-1], hma_f[-2], hma_s[-1], hma_s[-2], atr_arr[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        last = candles[-1]
        return {
            "ema_cross_up":   ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1],
            "ema_cross_down": ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1],
            "hma_cross_up":   hma_f[-2] <= hma_s[-2] and hma_f[-1] > hma_s[-1],
            "hma_cross_down": hma_f[-2] >= hma_s[-2] and hma_f[-1] < hma_s[-1],
            "hma20_val": float(hma_s[-1]),
            "open_below_hma20": last.open < hma_s[-1],
            "open_above_hma20": last.open > hma_s[-1],
            "atr_val": float(atr_arr[-1]),
        }

    @staticmethod
    def _closed_30m_bars(candles_15m: list) -> list:
        """Resample 15m -> 30m, dropping the last bucket if it isn't closed
        yet (i.e. the input series hasn't reached its 30m boundary)."""
        if not candles_15m:
            return []
        bucket_ms = 30 * 60_000
        buckets: dict[int, list] = {}
        for c in candles_15m:
            key = (c.timestamp // bucket_ms) * bucket_ms
            buckets.setdefault(key, []).append(c)

        class _Bar:
            __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
            def __init__(self, ts, o, h, l, cl, v):
                self.timestamp = ts; self.open = o; self.high = h
                self.low = l; self.close = cl; self.volume = v

        keys = sorted(buckets)
        out = []
        for key in keys:
            grp = buckets[key]
            out.append(_Bar(
                key, grp[0].open, max(g.high for g in grp), min(g.low for g in grp),
                grp[-1].close, sum(g.volume for g in grp),
            ))

        last_c15_end = candles_15m[-1].timestamp + 15 * 60_000
        last_bucket_end = keys[-1] + bucket_ms
        if last_c15_end < last_bucket_end:
            out = out[:-1]

        return out

    def _hold(self, price: float, reason: str = "", metadata: Optional[dict] = None) -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata=metadata or {},
        )
