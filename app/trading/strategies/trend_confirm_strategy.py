"""
3-Layer Trend-Confirmed Multi-TF Strategy.

Layer 1 — Trend direction (TF30m):
  Determines whether the market is currently in an uptrend or a
  downtrend. FOUR checks must ALL agree on the same direction, or there
  is no confirmed trend (self._trend_state = None) and nothing below runs:
    SMA30      : candle opens above SMA30 -> up, below -> down
    EMA10/20   : EMA10 > EMA20 -> up, EMA10 < EMA20 -> down
    EMA20 slope: EMA20 higher than `ema_slope_lookback` bars ago -> up
    MACD       : raw MACD LINE (fast EMA - slow EMA) vs zero ("MACD 4C"
                 zero-line read, sign only).
  Re-evaluated once per new 30m bar; a confirmed trend does not expire on
  its own, it just gets re-read every 30m bar.

Layer 2 — Trend quality: dynamic weighted score (TF15m + TF1H):
  Scores HOW GOOD the confirmed trend is on each timeframe (0-100 per TF),
  then combines them 15m x 65% + 1H x 35%. Must score > `layer2_threshold`
  (default 60) for an established-trend entry, or > `layer2_threshold_early`
  (default 75, stricter) for an early-trend entry (see Layer 3). Each
  per-TF quality score (in _tf_quality()) is:
    Alignment 40 pts — 4 x 10 pts: close vs EMA20, EMA20 vs EMA50, RSI
                       lean (>55 bull / <45 bear), MACD line sign — each
                       agreeing with Layer 1's direction scores 10.
    ADX       25 pts — trend strength, full credit at 2x adx_threshold (20).
    Chop      20 pts — inverted Choppiness Index (low chop = high score),
                       full credit at 100 - chop_threshold (61.8) points
                       of "not choppy".
    Volume    15 pts — volume / SMA20(volume), full credit at 2x
                       volume_expansion_mult (1.0).
  Soft point-scoring (ai_expert's ConfidenceEngine pattern): a weak
  reading on one component can be outweighed by strong readings elsewhere
  instead of vetoing the trade outright.

Layer 3 — Entry (TF15m), always WITH Layer 1's confirmed trend:
  LONG (only after Layer1+Layer2 confirm UP):
    1. HMA10 crosses above HMA20
    2. price (close) is above EMA10
    3. price is within `max_dist_atr_mult` (default 1.5) x ATR(14) of EMA20
  SHORT (only after Layer1+Layer2 confirm DOWN): the mirror —
    1. HMA10 crosses below HMA20
    2. price below EMA10
    3. price within 1.5 x ATR of EMA20
  Early-trend window: HMA is faster than Layer 1's 30m confirmation, so
  the HMA cross that starts a move often fires a bar or two BEFORE the
  trend confirms. When Layer 1's trend JUST confirmed (within
  `fresh_trend_bars`, default 2, bars of when it flipped) the entry counts
  an HMA cross up to fresh_trend_bars ago — which may predate the
  confirmation. Entering this early is riskier, so it must clear the
  STRICTER `layer2_threshold_early` (75) instead of the normal 60. Outside
  that window (established trend) the HMA cross must be on the current bar
  and only the normal 60 threshold applies. The cross timestamp is
  consumed (reset) once it triggers an entry attempt (pass or fail), so a
  stale cross can't silently satisfy a later setup. The EMA20 distance
  check is a chase-guard: no entry if price already ran too far from the
  trend reference — wait for a pullback + fresh cross.

Exit — primary is the HMA10/20 cross-back (a genuine trend reversal),
  which lets a healthy trend run instead of being whipsawed out. An
  optional faster exit (`exit_on_hma20_open`, OFF by default) closes on
  `exit_hma20_confirm_bars` consecutive closes past HMA20 — it's off
  because a single/near-single bar past HMA20 kept closing trades before
  the trend developed (visible on low-volatility symbols like XAU where
  ATR-tight stops + a twitchy exit stopped trades on noise). SL/TP
  (ATR(14, 15m) x2.5 SL, 2:1 R:R by default — wide enough that the
  signal exit, not noise, closes the trade) remains the hard-stop safety
  net checked by bot.py's risk-manager fallback underneath all of this.

Once closed (by either exit condition or the hard SL/TP stop), the next
bar where Layer1+Layer2 read confirmed again is eligible for a new entry.
No cooldown.

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze() — in
hedge mode a SELL Signal always OPENS a new short rather than closing an
open long. tick_open_position()'s PositionUpdate("close") always closes
whichever position is actually open, regardless of hedge mode.

TF30m candles for Layer 1 are resampled from the 15m series this strategy
receives (last, possibly-forming, 30m bucket dropped). Layer 2/3 run on
the raw 15m candles and the 1H candles passed in via mtf_candles.
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
        # Layer 1 — trend direction (30m)
        sma_trend: int = 30,
        ema1_period: int = 10,
        ema2_period: int = 20,
        ema_slope_lookback: int = 5,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # Layer 2 — trend quality (15m + 1H)
        quality_ema_fast: int = 20,
        quality_ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_bull: float = 55.0,
        rsi_bear: float = 45.0,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        chop_period: int = 14,
        chop_threshold: float = 61.8,
        volume_sma_period: int = 20,
        volume_expansion_mult: float = 1.0,
        tf_weight_15m: float = 0.65,
        tf_weight_1h: float = 0.35,
        layer2_threshold: float = 60.0,        # established-trend entries
        layer2_threshold_early: float = 75.0,  # stricter quality gate for early-trend entries (HMA led the confirm)
        # Layer 3 — entry (15m)
        hma_fast: int = 10,
        hma_slow: int = 20,
        entry_ema_ref: int = 10,    # price must be above (long) / below (short) this EMA
        dist_ema_ref: int = 20,     # distance-to-trend chase-guard is measured vs this EMA
        fresh_trend_bars: int = 2,  # HMA-cross lookback when the trend just confirmed (early trend)
        max_dist_atr_mult: float = 1.5,
        # Exit
        exit_on_hma20_open: bool = False,   # optional fast exit on N closes past HMA20 (off by default —
                                            #   it whipsawed trades out before the trend developed; a real
                                            #   HMA10/20 cross-back is the primary exit)
        exit_hma20_confirm_bars: int = 2,   # N consecutive closes past HMA20 required when the above is on
        # Risk
        atr_period: int = 14,
        sl_atr_mult: float = 2.5,           # wide enough that the signal exit, not noise, closes the trade
        rr_ratio: float = 2.0,              # TP at 2R so winners that reach a target aren't cut short at 1R
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

        self.quality_ema_fast = quality_ema_fast
        self.quality_ema_slow = quality_ema_slow
        self.rsi_period = rsi_period
        self.rsi_bull = rsi_bull
        self.rsi_bear = rsi_bear
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.chop_period = chop_period
        self.chop_threshold = chop_threshold
        self.volume_sma_period = volume_sma_period
        self.volume_expansion_mult = volume_expansion_mult
        self.tf_weight_15m = tf_weight_15m
        self.tf_weight_1h = tf_weight_1h
        self.layer2_threshold = layer2_threshold
        self.layer2_threshold_early = layer2_threshold_early

        self.hma_fast = hma_fast
        self.hma_slow = hma_slow
        self.entry_ema_ref = entry_ema_ref
        self.dist_ema_ref = dist_ema_ref
        self.fresh_trend_bars = fresh_trend_bars
        self.max_dist_atr_mult = max_dist_atr_mult

        self.exit_on_hma20_open = exit_on_hma20_open
        self.exit_hma20_confirm_bars = exit_hma20_confirm_bars

        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._trend_state: Optional[str] = None      # "up" | "down" | None — Layer 1 result
        self._trend_confirmed_since_ts: Optional[int] = None  # bar_ts_15 when trend last CHANGED
        self._last_bar_ts_30: Optional[int] = None    # Layer 1 new-bar tracking
        self._last_bar_ts_15: Optional[int] = None     # Layer 3 new-bar tracking
        self._last_hma_cross_up_ts: Optional[int] = None
        self._last_hma_cross_down_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None  # owned by tick_open_position()
        self._latest_candles: list = []

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles  # cached for tick_open_position()
        mtf = mtf_candles or {}
        bar_ts_15 = candles[-1].timestamp

        # ── Layer 1: Trend direction (30m) ─────────────────────────────────
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

        # ── Layer 3 HMA-cross tracking — runs every new 15m bar regardless of
        # Layer1/Layer2 gating, so an HMA cross that fires just before the
        # trend confirms is still remembered within the fresh-trend window. ─
        is_new_bar_15 = bar_ts_15 != self._last_bar_ts_15
        l3 = self._layer3_indicators(candles)
        if is_new_bar_15:
            self._last_bar_ts_15 = bar_ts_15
            if l3 is not None:
                if l3["hma_cross_up"]:
                    self._last_hma_cross_up_ts = bar_ts_15
                if l3["hma_cross_down"]:
                    self._last_hma_cross_down_ts = bar_ts_15

        def _bars_ago_15(ts: Optional[int]) -> Optional[int]:
            return (bar_ts_15 - ts) // (15 * 60_000) if ts is not None else None

        hma_up_ago   = _bars_ago_15(self._last_hma_cross_up_ts)
        hma_down_ago = _bars_ago_15(self._last_hma_cross_down_ts)

        # "Early trend": the trend just confirmed (within fresh_trend_bars).
        # Because HMA is faster than Layer1's 30m confirmation, the HMA
        # cross that kicks off the move often fires a bar or two BEFORE the
        # trend confirms — so in this window we count an HMA cross up to
        # fresh_trend_bars ago (which can predate the confirmation). Entering
        # this early is riskier, so it must clear a STRICTER Layer2 quality
        # gate (layer2_threshold_early) than an established-trend entry.
        fb = self.fresh_trend_bars
        trend_age = _bars_ago_15(self._trend_confirmed_since_ts)
        is_early_trend = trend_age is not None and trend_age <= fb
        lookback = fb if is_early_trend else 0
        hma_cross_up   = hma_up_ago is not None and hma_up_ago <= lookback
        hma_cross_down = hma_down_ago is not None and hma_down_ago <= lookback
        l2_thr = self.layer2_threshold_early if is_early_trend else self.layer2_threshold

        # ── Layer 2: Trend quality — weighted 15m (65%) + 1H (35%) score ───
        c1h = mtf.get("1h", [])
        q15 = self._tf_quality(candles, trend) if trend else None
        q1h = self._tf_quality(c1h, trend) if trend else None
        l2_score = None
        if q15 is not None and q1h is not None:
            l2_score = q15["score"] * self.tf_weight_15m + q1h["score"] * self.tf_weight_1h

        close_price = candles[-1].close
        dist_atr = (abs(close_price - l3["dist_ema_val"]) / l3["atr_val"]
                   if (l3 is not None and l3["atr_val"] > 0) else None)

        def dbg(entry_status: str) -> dict:
            return {"trend_confirm": {
                "sma_trend": l1["sma_dir"], "ema10_20_trend": l1["ema1020_dir"],
                "ema20_slope": l1["slope_dir"], "macd_trend": l1["macd_dir"],
                "confirmed": trend,
                "q15": round(q15["score"], 1) if q15 else None,
                "q1h": round(q1h["score"], 1) if q1h else None,
                "q15_breakdown": q15["breakdown"] if q15 else None,
                "q1h_breakdown": q1h["breakdown"] if q1h else None,
                "layer2_score": round(l2_score, 1) if l2_score is not None else None,
                "layer2_threshold": l2_thr,
                "open_position": self._open_position,
                "entry_status": entry_status,
                "fresh_trend_bars": fb, "trend_age_bars": trend_age, "is_early_trend": is_early_trend,
                "hma_cross_up_ago": hma_up_ago, "hma_cross_down_ago": hma_down_ago,
                "above_ema10": l3["above_ema_ref"] if l3 else None,
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

        if q15 is None or q1h is None:
            return self._hold(current_price, "Layer2: quality indicators still warming up (15m/1H)",
                              metadata=dbg("no_trend"))

        if l2_score <= l2_thr:
            score_note = (f"(15m={q15['score']:.0f} x{self.tf_weight_15m:.2f} + "
                          f"1H={q1h['score']:.0f} x{self.tf_weight_1h:.2f})")
            # Early-trend fail is DEFINITIVE: an HMA cross led the confirm and
            # the stricter gate rejected it — spend that cross so the same
            # setup can't retry on a later bar. A fresh cross is required for
            # another attempt. (An established-trend fail with no valid pending
            # cross is just "waiting", so nothing to consume there.)
            pending_cross = (trend == "up" and hma_cross_up) or (trend == "down" and hma_cross_down)
            if is_early_trend and pending_cross:
                if trend == "up":
                    self._last_hma_cross_up_ts = None
                else:
                    self._last_hma_cross_down_ts = None
                return self._hold(current_price,
                    f"Layer2 FAIL (early trend): quality {l2_score:.0f} <= {l2_thr:.0f} — "
                    f"early setup rejected, cross spent {score_note}",
                    metadata=dbg("early_quality_fail"))
            return self._hold(current_price,
                f"Layer2: trend quality {l2_score:.0f} <= {l2_thr:.0f} {score_note}",
                metadata=dbg("quality_fail"))

        # ── Layer 3: Entry (15m), always with the confirmed trend ──────────
        if l3 is None:
            return self._hold(current_price, "Layer3: indicators still warming up (15m)", metadata=dbg("no_trend"))

        dist_ok = dist_atr is not None and dist_atr <= self.max_dist_atr_mult
        dist_disp = f"{dist_atr:.2f}" if dist_atr is not None else "n/a"

        if trend == "up":
            if not hma_cross_up:
                return self._hold(current_price, "Layer1+2 confirmed UP — waiting for HMA10↑HMA20 cross"
                                  + (f" (early trend, within {fb} bars)" if is_early_trend else " (fresh cross only)"),
                                  metadata=dbg("waiting_cross"))
            # cross fired — consume it, then validate the two entry conditions
            self._last_hma_cross_up_ts = None
            if not l3["above_ema_ref"]:
                return self._hold(current_price,
                    f"Long setup FAILED: price not above EMA{self.entry_ema_ref} — waiting for fresh cross",
                    metadata=dbg("ema_ref_fail"))
            if not dist_ok:
                return self._hold(current_price,
                    f"Long setup FAILED: price {dist_disp}xATR from EMA{self.dist_ema_ref} "
                    f"(max {self.max_dist_atr_mult}x) — waiting for pullback + fresh cross",
                    metadata=dbg("cross_pass_distance_fail"))
            sl, tp = self._compute_sl_tp("long", close_price, l3["atr_val"])
            self._open_position = "long"
            meta = dbg("entered")
            meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                reason=f"Uptrend confirmed (Layer1 30m) + quality {l2_score:.0f}"
                       f"{' [early]' if is_early_trend else ''} >{l2_thr:.0f} (Layer2) + "
                       f"HMA10↑HMA20 {hma_up_ago}b ago, above EMA{self.entry_ema_ref}, "
                       f"{dist_atr:.2f}xATR from EMA{self.dist_ema_ref} (Layer3)",
                confidence=1.0,
                metadata=meta,
            )

        # trend == "down"
        if not hma_cross_down:
            return self._hold(current_price, "Layer1+2 confirmed DOWN — waiting for HMA10↓HMA20 cross"
                              + (f" (early trend, within {fb} bars)" if is_early_trend else " (fresh cross only)"),
                              metadata=dbg("waiting_cross"))
        self._last_hma_cross_down_ts = None
        if not l3["below_ema_ref"]:
            return self._hold(current_price,
                f"Short setup FAILED: price not below EMA{self.entry_ema_ref} — waiting for fresh cross",
                metadata=dbg("ema_ref_fail"))
        if not dist_ok:
            return self._hold(current_price,
                f"Short setup FAILED: price {dist_disp}xATR from EMA{self.dist_ema_ref} "
                f"(max {self.max_dist_atr_mult}x) — waiting for pullback + fresh cross",
                metadata=dbg("cross_pass_distance_fail"))
        sl, tp = self._compute_sl_tp("short", close_price, l3["atr_val"])
        self._open_position = "short"
        meta = dbg("entered")
        meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
        return Signal(
            type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
            reason=f"Downtrend confirmed (Layer1 30m) + quality {l2_score:.0f}"
                   f"{' [early]' if is_early_trend else ''} >{l2_thr:.0f} (Layer2) + "
                   f"HMA10↓HMA20 {hma_down_ago}b ago, below EMA{self.entry_ema_ref}, "
                   f"{dist_atr:.2f}xATR from EMA{self.dist_ema_ref} (Layer3)",
            confidence=1.0,
            metadata=meta,
        )

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Exit = OR logic, whichever fires first, evaluated once per
        newly-formed 15m bar: HMA10/20 cross-back (a genuine trend
        reversal), OR — only when `exit_on_hma20_open` is set —
        `exit_hma20_confirm_bars` consecutive closes on the wrong side of
        HMA20 (a faster warning than a full cross, but the confirmation
        window guards against a single whipsaw bar closing the trade
        prematurely). Hedge-mode-safe: always closes whichever position is
        actually open, never relies on signal.type semantics."""
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

        cb = self.exit_hma20_confirm_bars
        if self._open_position == "long":
            hma20_exit = self.exit_on_hma20_open and self._closes_past_hma20(candles, "long", cb)
            if l3["hma_cross_down"] or hma20_exit:
                reason = "HMA10 crossed below HMA20" if l3["hma_cross_down"] else f"{cb} close(s) below HMA20"
                self._open_position = None
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit LONG: {reason} (15m)")
        if self._open_position == "short":
            hma20_exit = self.exit_on_hma20_open and self._closes_past_hma20(candles, "short", cb)
            if l3["hma_cross_up"] or hma20_exit:
                reason = "HMA10 crossed above HMA20" if l3["hma_cross_up"] else f"{cb} close(s) above HMA20"
                self._open_position = None
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit SHORT: {reason} (15m)")

        return PositionUpdate(action="hold", reason=f"Holding {self._open_position.upper()}")

    def _closes_past_hma20(self, candles: list, side: str, n: int) -> bool:
        """True if the last `n` bars ALL closed on the wrong side of HMA20
        for the given position side (long: below, short: above)."""
        closes = [c.close for c in candles]
        hma_s = self.hma(closes, self.hma_slow)
        if len(closes) < n:
            return False
        for k in range(1, n + 1):
            if np.isnan(hma_s[-k]):
                return False
            if side == "long" and not (closes[-k] < hma_s[-k]):
                return False
            if side == "short" and not (closes[-k] > hma_s[-k]):
                return False
        return True

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Called by bot.py after ANY close, including the risk-manager's
        hard SL/TP fallback firing before HMA10/20 crosses back — without
        this, _open_position would stay set forever and analyze() would
        refuse all future entries."""
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
        # "MACD 4C" trend read: raw MACD line vs zero, not vs its signal line.
        macd_up, macd_down = macd_line[-1] > 0, macd_line[-1] < 0

        trend_up = sma_up and ema1020_up and slope_up and macd_up
        trend_down = sma_down and ema1020_down and slope_down and macd_down

        return {
            "trend": "up" if trend_up else "down" if trend_down else None,
            "sma_dir": "up" if sma_up else "down" if sma_down else "flat",
            "ema1020_dir": "up" if ema1020_up else "down" if ema1020_down else "flat",
            "slope_dir": "up" if slope_up else "down" if slope_down else "flat",
            "macd_dir": "up" if macd_up else "down" if macd_down else "flat",
        }

    def _tf_quality(self, candles: list, trend: str) -> Optional[dict]:
        """Per-timeframe 0-100 trend-quality score for the confirmed
        direction: Alignment 40 + ADX 25 + Choppiness 20 + Volume 15.
        Returns None if there aren't enough candles to compute everything
        (e.g. the 1H series from mtf_candles is short) — the caller treats
        that as 'still warming up'."""
        min_needed = max(self.quality_ema_slow + 2, 2 * self.adx_period + 2,
                         self.chop_period + 1, self.volume_sma_period, self.rsi_period + 1,
                         self.macd_slow + self.macd_signal + 1)
        if len(candles) < min_needed:
            return None

        closes = [c.close for c in candles]
        ema_fast = self.ema(closes, self.quality_ema_fast)
        ema_slow = self.ema(closes, self.quality_ema_slow)
        rsi = self.rsi(closes, self.rsi_period)
        macd_line, _sig, _hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        adx_arr, _p, _m = self.adx(candles, self.adx_period)
        chop_val = self._choppiness(candles, self.chop_period)
        vols = [c.volume for c in candles]
        vol_sma = self.sma(vols, self.volume_sma_period)

        needed = [ema_fast[-1], ema_slow[-1], rsi[-1], macd_line[-1], adx_arr[-1], vol_sma[-1]]
        if chop_val is None or any(np.isnan(x) for x in needed) or vol_sma[-1] <= 0:
            return None

        up = trend == "up"
        c = closes[-1]

        # Alignment (40 pts) — 4 x 10, each check agreeing with Layer1 direction
        checks = {
            "px_ema20": (c > ema_fast[-1]) == up,
            "ema20_50": (ema_fast[-1] > ema_slow[-1]) == up,
            "rsi":      (rsi[-1] > self.rsi_bull) if up else (rsi[-1] < self.rsi_bear),
            "macd":     (macd_line[-1] > 0) == up,
        }
        align_pts = sum(10.0 for v in checks.values() if v)

        # ADX (25 pts) — full credit at 2x threshold
        adx_val = float(adx_arr[-1])
        adx_pts = min(1.0, adx_val / max(1.0, self.adx_threshold * 2.0)) * 25.0

        # Choppiness (20 pts) — inverted (low chop = high score)
        chop_full_at = max(1.0, 100.0 - self.chop_threshold)
        chop_pts = max(0.0, min(1.0, (100.0 - chop_val) / chop_full_at)) * 20.0

        # Volume (15 pts) — full credit at 2x expansion multiple
        vol_ratio = float(candles[-1].volume) / float(vol_sma[-1])
        vol_pts = min(1.0, vol_ratio / max(0.01, self.volume_expansion_mult * 2.0)) * 15.0

        breakdown = {
            "align": round(align_pts, 1), "adx": round(adx_pts, 1),
            "chop": round(chop_pts, 1), "volume": round(vol_pts, 1),
            "adx_val": round(adx_val, 1), "chop_val": round(chop_val, 1),
            "vol_ratio": round(vol_ratio, 2),
        }
        score = align_pts + adx_pts + chop_pts + vol_pts
        return {"score": round(score, 1), "breakdown": breakdown}

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

    def _layer3_indicators(self, candles: list) -> Optional[dict]:
        closes = [c.close for c in candles]
        hma_f = self.hma(closes, self.hma_fast)
        hma_s = self.hma(closes, self.hma_slow)
        ema_ref = self.ema(closes, self.entry_ema_ref)
        dist_ema = self.ema(closes, self.dist_ema_ref)
        atr_arr = self.atr(candles, self.atr_period)

        needed = [hma_f[-1], hma_f[-2], hma_s[-1], hma_s[-2],
                 ema_ref[-1], dist_ema[-1], atr_arr[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        last = candles[-1]
        return {
            "hma_cross_up":   hma_f[-2] <= hma_s[-2] and hma_f[-1] > hma_s[-1],
            "hma_cross_down": hma_f[-2] >= hma_s[-2] and hma_f[-1] < hma_s[-1],
            "hma20_val": float(hma_s[-1]),
            "open_below_hma20": last.open < hma_s[-1],
            "open_above_hma20": last.open > hma_s[-1],
            "above_ema_ref": last.close > ema_ref[-1],
            "below_ema_ref": last.close < ema_ref[-1],
            "dist_ema_val": float(dist_ema[-1]),
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
