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

Layer 2 — Trade CONTEXT: trend quality (2a) + location/structure (2b).
  Both must pass before Layer 3 even waits for the cross.

  2a) Trend quality — dynamic weighted score (TF15m + TF1H):
    Scores HOW GOOD the confirmed trend is on each timeframe (0-100 per TF),
    then combines them 15m x 65% + 1H x 35%. Must score > `layer2_threshold`
    (default 60) established, or > `layer2_threshold_early` (68, stricter) for
    an early-trend entry. Each per-TF quality score (in _tf_quality()) is:
      Alignment 40 pts — 4 x 10 pts: close vs EMA20, EMA20 vs EMA50, RSI
                         lean (>55 bull / <45 bear), MACD line sign — each
                         agreeing with Layer 1's direction scores 10.
      ADX       25 pts — trend strength, full credit at 2x adx_threshold (20).
      Chop      20 pts — inverted Choppiness Index (low chop = high score),
                         full credit at 100 - chop_threshold (61.8) points.
      Volume    15 pts — volume / SMA20(volume), full credit at 2x
                         volume_expansion_mult (1.0).
    Soft point-scoring (ai_expert's ConfidenceEngine pattern): a weak reading
    on one component can be outweighed by strong readings elsewhere.

  2b) Location & structure-room filter (see _location_context()):
    Hard-rejects entering into an active HTF opposing pivot zone, opposite 1H
    swing structure, a wrong-side liquidity sweep, or critically low room.
    Borderline location doesn't reject — it raises 2a's threshold by
    `location_threshold_penalty` (+4). Only evaluated once 2a passes and the
    5m EMA50 stop reference (l3) is available.

Layer 3 — Entry TIMING (TF5m), reached only after Layer1 + Layer2 both pass:
  LONG (only after Layer1+Layer2 confirm UP):
    1. EMA5 crosses above EMA9 (5m)
    2. price (close) is above EMA9 (5m) — the same line the cross + exit use
    3. price is within `max_dist_atr_mult` (default 1.5) x ATR(14,5m) of EMA50
  SHORT (only after Layer1+Layer2 confirm DOWN): the mirror —
    1. EMA5 crosses below EMA9 (5m)
    2. price below EMA9 (5m)
    3. price within 1.5 x ATR of EMA50
  Early-trend window: the 5m EMA5/9 cross is faster than Layer 1's 30m
  confirmation, so the cross that starts a move often fires a bar or two
  BEFORE the trend confirms. When Layer 1's trend JUST confirmed (within
  `fresh_trend_bars`, default 2, 5m bars of when it flipped) the entry counts
  a cross up to fresh_trend_bars ago — which may predate the confirmation.
  Entering this early is riskier, so it must clear the STRICTER
  `layer2_threshold_early` (68) instead of the normal 60. Outside that window
  (established trend) only the normal 60 threshold applies.
  Cross validity: a cross stays usable for `cross_valid_bars` (default 3) 5m
  bars — the Layer2 gates (quality/location) often clear a bar or two AFTER
  the cross fires, and requiring both on the exact same bar silently wasted
  almost every signal. The price-position and chase-guard checks always run
  on the CURRENT bar, so a windowed cross can't produce a stale entry. The
  cross timestamp is consumed (reset) once it triggers an entry attempt
  (pass or fail), so one cross can't satisfy a later setup twice. The EMA50
  distance check is a chase-guard: no entry if price already ran too far
  from it — wait for a pullback + fresh cross.

Exit — a 2-TP + break-even scheme managed in tick_open_position():
  TP1 (partial): when price reaches `tp1_r` (0.75R, halfway to the 1.5R
    final TP), take `tp1_close_pct` (50%) off and move SL to break-even +
    `be_offset_r` (BE + 0.1R — a small locked profit on the runner). Checked
    every tick, fires once.
  Runner (remaining 50%): rides on until, on the 5m TF, EITHER the EMA5/9
    cross-back (long: EMA5 crosses below EMA9) OR a close past EMA9 (long:
    5m close below EMA9) — mirror for shorts — the hard final TP (1.5R), or
    the trailed SL (BE+0.1R).
  SL sits at the 5m EMA50 (the chase-guard keeps price within 1.5x ATR of it,
  so max risk ~1.5 ATR). Its distance from entry is R; TP2 = 1.5R (2.5R put
  the target too far — it was rarely hit). The
  distance is floored at min_sl_pct (0.5% of price) so an EMA50 sitting right
  at price can't produce a microscopic noise-stop. These are the hard bounds
  bot.py's risk manager checks underneath; TP1 fires the partial + BE-move
  via a PositionUpdate("partial_tp", new_sl=...).

Once closed (by either exit condition or the hard SL/TP stop), the next
bar where Layer1+Layer2 read confirmed again is eligible for a new entry.
No cooldown.

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze() — in
hedge mode a SELL Signal always OPENS a new short rather than closing an
open long. tick_open_position()'s PositionUpdate("close") always closes
whichever position is actually open, regardless of hedge mode.

Data: TF30m (Layer 1) is resampled from the 15m series this strategy
receives (last, possibly-forming, 30m bucket dropped). Layer 2 runs on the
raw 15m candles + the 1H series from mtf_candles. Layer 3 entry, SL/TP and
exit run on the 5m series passed in via mtf_candles["5m"].
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
        # Layer 2a quality is scored across 4 dimensions (sum = 100):
        bias_weight: float = 30.0,       # structural direction (price vs EMA20/50)
        trend_weight: float = 30.0,      # trend strength / not choppy (ADX + Choppiness)
        momentum_weight: float = 25.0,   # RSI lean + MACD histogram in-trend
        volume_weight: float = 15.0,     # volume expansion vs its SMA
        tf_weight_15m: float = 0.65,
        tf_weight_1h: float = 0.35,
        layer2_threshold: float = 60.0,        # established-trend entries
        layer2_threshold_early: float = 68.0,  # stricter quality gate for early-trend entries (5m cross led the confirm)
        # Layer 3 — entry (5m): EMA5/9 cross, price above/below EMA5, within 1.5xATR of EMA50
        entry_tf: str = "5m",       # timeframe (mtf key) the entry cross + exit run on
        ema_fast: int = 5,          # entry-cross fast EMA (5m)
        ema_slow: int = 9,          # entry-cross slow EMA (5m); also the "close past" exit reference
        entry_ema_ref: int = 9,     # price must be above (long) / below (short) this EMA (5m) — EMA9, same line the cross + exit use
        sl_ema_ref: int = 50,       # SL sits at this EMA (5m)
        chase_ema_ref: int = 50,    # chase-guard distance is measured vs this EMA (5m); decoupled from sl_ema_ref
        fresh_trend_bars: int = 2,  # EMA-cross lookback (in 5m bars) when the trend just confirmed (early trend)
        cross_valid_bars: int = 3,  # how many 5m bars a cross stays usable while Layer2 gates settle —
                                    #   without this, a cross was only good on the exact bar every gate was
                                    #   already open (quality/location often clear 1-2 bars AFTER the cross,
                                    #   which silently wasted almost every signal)
        max_dist_atr_mult: float = 1.5,  # max distance from the chase EMA in ATR(5m)
        # Location & structure-room filter (lightweight; avoids late/blocked entries)
        use_location_filter: bool = True,
        structure_pivot_left: int = 2,
        structure_pivot_right: int = 2,
        zone_width_atr_1h: float = 0.18,
        zone_width_atr_4h: float = 0.22,
        hard_zone_distance_atr: float = 0.30,
        min_structure_room_r: float = 0.50,   # hard-reject below this many R of room to the opposing zone
        preferred_structure_room_r: float = 0.75,  # below this (but >= min) just penalizes the quality gate
        location_threshold_penalty: float = 4.0,
        reject_midrange_when_choppy: bool = True,
        # Sideways / range veto (Layer 2) — hard-block entries when the 15m
        # context looks like a range, not a trend. Designed NOT to kill early
        # trends: it leans on EMA compression + high chop (which stay range-y
        # even as ADX lags), and only counts "really weak" ADX (< sideways_adx_max,
        # stricter than adx_threshold) so a fresh trend at ADX ~18 isn't vetoed.
        use_sideways_filter: bool = True,
        sideways_ema_compression_atr: float = 0.5,  # |EMA20-EMA50| < this x ATR = tangled/flat
        sideways_adx_max: float = 15.0,             # ADX below this = "really weak" (< adx_threshold on purpose)
        sideways_range_atr: float = 1.2,            # last-20-bar high-low range < this x ATR = tight consolidation
        sideways_min_signals: int = 2,              # how many of the 4 signals must fire to veto
        # Exit (5m): EMA5/9 cross-back OR a 5m close past EMA9 closes the runner
        use_close_past_exit: bool = True,   # enable the "close past EMA9" exit at all (cross-back always on)
        exit_close_confirm_bars: int = 1,   # N consecutive 5m closes past EMA9 required for that exit
        signal_exit_requires_tp1: bool = True,   # no signal exits before TP1 — only the hard SL (EMA50) / TP
                                                 #   bounds manage the trade until then. On 5m the single-close
                                                 #   EMA9 exits killed 75% of trades at ~-0.3R before TP1; arming
                                                 #   them only on the runner nearly doubled WR (25->62% BTC,
                                                 #   41->60% SOL) and cut losses ~2-3x in backtest
        # Partial take-profit + break-even (2-TP scheme)
        use_partial_tp: bool = True,        # TP1 -> take tp1_close_pct, move SL to BE+be_offset_r; runner rides on
        tp1_r: float = 0.75,                # TP1 at 0.75R (halfway to the 1.5R final TP)
        tp1_close_pct: float = 0.5,         # fraction closed at TP1
        be_offset_r: float = 0.1,           # after TP1, SL -> entry +/- this many R (BE + 0.1R, a small locked profit)
        # Risk
        atr_period: int = 14,               # ATR(5m) for the chase guard / min-distance sanity
        rr_ratio: float = 1.5,              # final TP (TP2) at 1.5R (R = |entry - EMA50(5m)|) — 2.5R was too far,
                                            #   ~30% WR at 1.5R vs ~21% at 2.5R for ~the same $ return (backtest)
        min_sl_pct: float = 0.005,          # floor the SL distance at 0.5% of price — if EMA50 sits right at
                                            #   price the raw SL would be a microscopic noise-stop; R (and TP)
                                            #   is measured from the floored distance
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
        self.bias_weight = bias_weight
        self.trend_weight = trend_weight
        self.momentum_weight = momentum_weight
        self.volume_weight = volume_weight
        self.tf_weight_15m = tf_weight_15m
        self.tf_weight_1h = tf_weight_1h
        self.layer2_threshold = layer2_threshold
        self.layer2_threshold_early = layer2_threshold_early

        self.entry_tf = entry_tf
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.entry_ema_ref = entry_ema_ref
        self.sl_ema_ref = sl_ema_ref
        self.chase_ema_ref = chase_ema_ref
        self.fresh_trend_bars = fresh_trend_bars
        self.cross_valid_bars = cross_valid_bars
        self.max_dist_atr_mult = max_dist_atr_mult

        self.use_location_filter = use_location_filter
        self.structure_pivot_left = structure_pivot_left
        self.structure_pivot_right = structure_pivot_right
        self.zone_width_atr_1h = zone_width_atr_1h
        self.zone_width_atr_4h = zone_width_atr_4h
        self.hard_zone_distance_atr = hard_zone_distance_atr
        self.min_structure_room_r = min_structure_room_r
        self.preferred_structure_room_r = preferred_structure_room_r
        self.location_threshold_penalty = location_threshold_penalty
        self.reject_midrange_when_choppy = reject_midrange_when_choppy
        self.use_sideways_filter = use_sideways_filter
        self.sideways_ema_compression_atr = sideways_ema_compression_atr
        self.sideways_adx_max = sideways_adx_max
        self.sideways_range_atr = sideways_range_atr
        self.sideways_min_signals = sideways_min_signals

        self.use_close_past_exit = use_close_past_exit
        self.exit_close_confirm_bars = exit_close_confirm_bars
        self.signal_exit_requires_tp1 = signal_exit_requires_tp1

        self.use_partial_tp = use_partial_tp
        self.tp1_r = tp1_r
        self.tp1_close_pct = tp1_close_pct
        self.be_offset_r = be_offset_r

        self.atr_period = atr_period
        self.rr_ratio = rr_ratio
        self.min_sl_pct = min_sl_pct

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._trend_state: Optional[str] = None      # "up" | "down" | None — Layer 1 result
        self._trend_confirmed_since_ts: Optional[int] = None  # 5m bar_ts when trend last CHANGED
        self._last_bar_ts_30: Optional[int] = None    # Layer 1 new-bar tracking
        self._last_bar_ts_5: Optional[int] = None     # Layer 3 (5m) new-bar tracking
        self._last_ema_cross_up_ts: Optional[int] = None
        self._last_ema_cross_down_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None  # owned by tick_open_position()
        self._latest_candles: list = []
        self._latest_5m: list = []       # 5m series cached for tick_open_position()
        # Partial-TP tracking for the open position (owned by tick_open_position())
        self._entry_price: Optional[float] = None
        self._entry_sl: Optional[float] = None
        self._tp1_done: bool = False

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles  # 15m base (Layer1 30m resample + Layer2 15m)
        mtf = mtf_candles or {}
        bar_ts_15 = candles[-1].timestamp
        # Entry, SL/TP and exit all run on the finer 5m series (mtf_candles).
        c5m = mtf.get(self.entry_tf, []) or []
        self._latest_5m = c5m  # cached for tick_open_position()
        bar_ts_5 = c5m[-1].timestamp if c5m else None

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
                self._trend_confirmed_since_ts = bar_ts_5
            self._trend_state = l1["trend"]

        trend = self._trend_state

        # ── Layer 3 EMA5/9-cross tracking (5m) — runs every new 5m bar
        # regardless of Layer1/Layer2 gating, so a cross that fires just
        # before the trend confirms is still remembered within the
        # fresh-trend window. ───────────────────────────────────────────────
        l3 = self._layer3_indicators(c5m) if c5m else None
        is_new_bar_5 = bar_ts_5 is not None and bar_ts_5 != self._last_bar_ts_5
        if is_new_bar_5:
            self._last_bar_ts_5 = bar_ts_5
            if l3 is not None:
                if l3["ema_cross_up"]:
                    self._last_ema_cross_up_ts = bar_ts_5
                if l3["ema_cross_down"]:
                    self._last_ema_cross_down_ts = bar_ts_5

        def _bars_ago_5(ts: Optional[int]) -> Optional[int]:
            return (bar_ts_5 - ts) // (5 * 60_000) if (ts is not None and bar_ts_5 is not None) else None

        ema_up_ago   = _bars_ago_5(self._last_ema_cross_up_ts)
        ema_down_ago = _bars_ago_5(self._last_ema_cross_down_ts)

        # "Early trend": the trend just confirmed (within fresh_trend_bars 5m
        # bars). Because the 5m EMA5/9 cross is faster than Layer1's 30m
        # confirmation, the cross that kicks off the move often fires a bar or
        # two BEFORE the trend confirms — so in this window we count a cross up
        # to fresh_trend_bars ago (which can predate the confirmation).
        # Entering this early is riskier, so it must clear a STRICTER Layer2
        # quality gate (layer2_threshold_early) than an established-trend entry.
        fb = self.fresh_trend_bars
        trend_age = _bars_ago_5(self._trend_confirmed_since_ts)
        is_early_trend = trend_age is not None and trend_age <= fb
        # A cross stays usable for cross_valid_bars while the Layer2 gates
        # settle (quality/location often clear 1-2 bars AFTER the cross; with
        # lookback 0 those crosses were silently wasted and the bot barely
        # traded). In the early-trend window the cross may additionally
        # predate the 30m confirmation by up to fresh_trend_bars.
        lookback = max(self.cross_valid_bars, fb) if is_early_trend else self.cross_valid_bars
        ema_cross_up   = ema_up_ago is not None and ema_up_ago <= lookback
        ema_cross_down = ema_down_ago is not None and ema_down_ago <= lookback
        l2_thr = self.layer2_threshold_early if is_early_trend else self.layer2_threshold

        # ── Layer 2: Trend quality — weighted 15m (65%) + 1H (35%) score ───
        c1h = mtf.get("1h", [])
        q15 = self._tf_quality(candles, trend) if trend else None
        q1h = self._tf_quality(c1h, trend) if trend else None
        l2_score = None
        if q15 is not None and q1h is not None:
            l2_score = q15["score"] * self.tf_weight_15m + q1h["score"] * self.tf_weight_1h

        # Entry price is the latest 5m close (the TF the cross fires on).
        close_price = c5m[-1].close if c5m else candles[-1].close
        dist_atr = (abs(close_price - l3["dist_ema_val"]) / l3["atr_val"]
                   if (l3 is not None and l3["atr_val"] > 0) else None)
        c4h = mtf.get("4h", [])

        # Location/structure defaults — overwritten in the Layer 2b block once
        # trend + base quality pass and l3 is available (location is evaluated
        # as part of Layer2, before Layer3 waits for the cross).
        has_long_candidate = trend == "up" and ema_cross_up
        has_short_candidate = trend == "down" and ema_cross_down
        location = self._neutral_location_context()
        location_adjusted_l2_thr = l2_thr
        sideways = (self._sideways_context(candles) if self.use_sideways_filter
                    else {"is_sideways": False, "signals": 0, "detail": {}})

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
                "layer2_threshold": location_adjusted_l2_thr,
                "base_layer2_threshold": l2_thr,
                "location": location,
                "sideways": sideways,
                "open_position": self._open_position,
                "entry_status": entry_status,
                "fresh_trend_bars": fb, "trend_age_bars": trend_age, "is_early_trend": is_early_trend,
                "ema_cross_up_ago": ema_up_ago, "ema_cross_down_ago": ema_down_ago,
                "above_ema_ref": l3["above_ema_ref"] if l3 else None,
                "dist_atr": round(dist_atr, 2) if dist_atr is not None else None,
                "max_dist_atr": self.max_dist_atr_mult,
                "entry_tf": self.entry_tf,
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

        pending_cross = has_long_candidate or has_short_candidate
        score_note = (f"(15m={q15['score']:.0f} x{self.tf_weight_15m:.2f} + "
                      f"1H={q1h['score']:.0f} x{self.tf_weight_1h:.2f})")

        def _spend_cross_if_early() -> bool:
            """Early-trend fail is DEFINITIVE: a 5m EMA cross led the confirm
            but Layer2 rejected the setup — spend that leading cross so the
            same setup can't retry on a later bar (a fresh cross is required
            for another attempt). No-op unless we're in the early window with
            a pending cross."""
            if not (is_early_trend and pending_cross):
                return False
            if trend == "up":
                self._last_ema_cross_up_ts = None
            else:
                self._last_ema_cross_down_ts = None
            return True

        # ── Layer 2 — sideways / range veto (hard) ────────────────────────
        # A trend-follower bleeds in ranges, so an explicit range read gets a
        # hard veto here regardless of the quality score (a range can still
        # score >60 on a bounce). Tuned not to fire on fresh trends (see
        # _sideways_context). Definitive for early-trend crosses.
        if sideways["is_sideways"]:
            spent = _spend_cross_if_early()
            d = sideways["detail"]
            fired = [k for k in ("ema_compressed", "high_chop", "tight_range", "weak_adx") if d.get(k)]
            tag = "FAIL (early trend, cross spent): " if spent else ""
            return self._hold(current_price,
                f"Layer2 {tag}SIDEWAYS veto ({sideways['signals']} signals: {', '.join(fired)}) — "
                f"EMA-gap {d.get('ema_gap_atr')}xATR, chop {d.get('chop')}, ADX {d.get('adx')}, "
                f"range {d.get('range_atr')}xATR",
                metadata=dbg("early_quality_fail" if spent else "sideways_veto"))

        # ── Layer 2a: base trend quality ──────────────────────────────────
        if l2_score <= l2_thr:
            spent = _spend_cross_if_early()
            tag = "FAIL (early trend, cross spent): " if spent else ""
            return self._hold(current_price,
                f"Layer2 {tag}trend quality {l2_score:.0f} <= {l2_thr:.0f} {score_note}",
                metadata=dbg("early_quality_fail" if spent else "quality_fail"))

        # ── Layer 2b: location & structure-room filter ────────────────────
        # Location is part of Layer2: the trade CONTEXT (trend quality AND
        # where we'd be entering vs HTF structure) must both be good before
        # Layer3 even waits for the cross. Needs the 5m EMA50 stop reference
        # for the room-R estimate, so it requires l3.
        if self.use_location_filter and l3 is not None:
            entry_risk = max(abs(close_price - l3["sl_ema_val"]), self.min_sl_pct * close_price)
            location = self._location_context(
                direction=trend, entry_price=close_price,
                atr_15m=self._last_atr(candles, self.atr_period) or l3["atr_val"],
                estimated_risk=entry_risk, candles_15m=candles,
                candles_1h=c1h, candles_4h=c4h, q15=q15,
            )
            location_adjusted_l2_thr = l2_thr + (
                self.location_threshold_penalty if location["penalize"] else 0.0)
            if not location["valid"]:
                return self._hold(current_price,
                    f"Layer2 Location/Structure REJECT: {location['reason']}",
                    metadata=dbg("location_reject"))
            if l2_score <= location_adjusted_l2_thr:
                spent = _spend_cross_if_early()
                tag = "FAIL (early trend, cross spent): " if spent else ""
                return self._hold(current_price,
                    f"Layer2 {tag}quality {l2_score:.0f} <= {location_adjusted_l2_thr:.0f} "
                    f"(location-adjusted) {score_note}",
                    metadata=dbg("early_quality_fail" if spent else "location_quality_fail"))

        # ── Layer 3: Entry timing (5m) — wait for the EMA5/9 cross ─────────
        # Reached only after Layer1 (trend) AND Layer2 (quality + location)
        # both pass. Layer3 just waits for the precise cross and confirms the
        # entry candle sits on the right side of EMA9, not too far from EMA50.
        if l3 is None:
            return self._hold(current_price, "Layer3: 5m indicators still warming up", metadata=dbg("no_trend"))

        dist_ok = dist_atr is not None and dist_atr <= self.max_dist_atr_mult
        dist_disp = f"{dist_atr:.2f}" if dist_atr is not None else "n/a"

        if trend == "up":
            if not has_long_candidate:
                return self._hold(current_price, f"Layer1+2 passed — waiting for EMA{self.ema_fast}↑EMA{self.ema_slow} cross (5m)"
                                  + (f" (early trend, within {fb} bars)" if is_early_trend else f" (cross valid {self.cross_valid_bars} bars)"),
                                  metadata=dbg("waiting_cross"))
            # cross fired — consume it, then validate the two entry conditions
            self._last_ema_cross_up_ts = None
            if not l3["above_ema_ref"]:
                return self._hold(current_price,
                    f"Long setup FAILED: price not above EMA{self.entry_ema_ref} (5m) — waiting for fresh cross",
                    metadata=dbg("ema_ref_fail"))
            if not dist_ok:
                return self._hold(current_price,
                    f"Long setup FAILED: price {dist_disp}xATR from EMA{self.sl_ema_ref} "
                    f"(max {self.max_dist_atr_mult}x, 5m) — waiting for pullback + fresh cross",
                    metadata=dbg("cross_pass_distance_fail"))
            sl, tp = self._compute_sl_tp("long", close_price, l3["sl_ema_val"])
            self._open_position = "long"
            self._entry_price, self._entry_sl, self._tp1_done = close_price, sl, False
            meta = dbg("entered")
            meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio,
                         "structure_room_r": location.get("structure_room_r"),
                         "nearest_opposing_zone": location.get("nearest_opposing_zone")})
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                reason=f"Uptrend confirmed (Layer1 30m) + quality {l2_score:.0f}"
                       f"{' [early]' if is_early_trend else ''} >{location_adjusted_l2_thr:.0f} + location OK (Layer2) + "
                       f"EMA{self.ema_fast}↑EMA{self.ema_slow} {ema_up_ago}b ago, above EMA{self.entry_ema_ref}, "
                       f"{dist_atr:.2f}xATR from EMA{self.sl_ema_ref} (Layer3 5m)",
                confidence=1.0,
                metadata=meta,
            )

        # trend == "down"
        if not has_short_candidate:
            return self._hold(current_price, f"Layer1+2 passed — waiting for EMA{self.ema_fast}↓EMA{self.ema_slow} cross (5m)"
                              + (f" (early trend, within {fb} bars)" if is_early_trend else f" (cross valid {self.cross_valid_bars} bars)"),
                              metadata=dbg("waiting_cross"))
        self._last_ema_cross_down_ts = None
        if not l3["below_ema_ref"]:
            return self._hold(current_price,
                f"Short setup FAILED: price not below EMA{self.entry_ema_ref} (5m) — waiting for fresh cross",
                metadata=dbg("ema_ref_fail"))
        if not dist_ok:
            return self._hold(current_price,
                f"Short setup FAILED: price {dist_disp}xATR from EMA{self.sl_ema_ref} "
                f"(max {self.max_dist_atr_mult}x, 5m) — waiting for pullback + fresh cross",
                metadata=dbg("cross_pass_distance_fail"))
        sl, tp = self._compute_sl_tp("short", close_price, l3["sl_ema_val"])
        self._open_position = "short"
        self._entry_price, self._entry_sl, self._tp1_done = close_price, sl, False
        meta = dbg("entered")
        meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio,
                     "structure_room_r": location.get("structure_room_r"),
                     "nearest_opposing_zone": location.get("nearest_opposing_zone")})
        return Signal(
            type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
            reason=f"Downtrend confirmed (Layer1 30m) + quality {l2_score:.0f}"
                   f"{' [early]' if is_early_trend else ''} >{location_adjusted_l2_thr:.0f} + location OK (Layer2) + "
                   f"EMA{self.ema_fast}↓EMA{self.ema_slow} {ema_down_ago}b ago, below EMA{self.entry_ema_ref}, "
                   f"{dist_atr:.2f}xATR from EMA{self.sl_ema_ref} (Layer3 5m)",
            confidence=1.0,
            metadata=meta,
        )

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Position management, evaluated every tick:

        1. TP1 partial (price-based, checked every tick): when price reaches
           tp1_r (1.25R, halfway to the 2.5R final TP), close tp1_close_pct
           (50%) and move SL to break-even + be_offset_r (BE + 0.1R). Once.
        2. Exit the runner (5m bar-based): EMA5/9 cross-back (a genuine trend
           reversal) OR a 5m close past EMA9 (long: close below EMA9; short:
           close above EMA9). The final TP (2.5R) and the trailed SL are the
           hard bounds bot.py's risk manager checks underneath.
        Hedge-mode-safe: always closes whichever position is actually open,
        never relies on signal.type semantics."""
        if self._open_position is None:
            return None

        from ..engines.position_manager import PositionUpdate

        # ── 1) TP1 partial take-profit + move SL to BE+offset (every tick) ──
        if (self.use_partial_tp and not self._tp1_done
                and self._entry_price is not None and self._entry_sl is not None):
            r = abs(self._entry_price - self._entry_sl)
            if r > 0:
                if self._open_position == "long" and current_price >= self._entry_price + self.tp1_r * r:
                    self._tp1_done = True
                    new_sl = self._entry_price + self.be_offset_r * r
                    return PositionUpdate(action="partial_tp", close_pct=self.tp1_close_pct, new_sl=new_sl,
                                          reason=f"TP1 {self.tp1_r:.1f}R hit — took {self.tp1_close_pct*100:.0f}%, "
                                                 f"SL -> BE+{self.be_offset_r:.1f}R")
                if self._open_position == "short" and current_price <= self._entry_price - self.tp1_r * r:
                    self._tp1_done = True
                    new_sl = self._entry_price - self.be_offset_r * r
                    return PositionUpdate(action="partial_tp", close_pct=self.tp1_close_pct, new_sl=new_sl,
                                          reason=f"TP1 {self.tp1_r:.1f}R hit — took {self.tp1_close_pct*100:.0f}%, "
                                                 f"SL -> BE+{self.be_offset_r:.1f}R")

        # ── 2) Runner exit — EMA5/9 cross-back OR 5m close past EMA9 ─────────
        candles = self._latest_5m
        if not candles:
            return PositionUpdate(action="hold", reason="Waiting for 5m data")
        bar_ts = candles[-1].timestamp
        if bar_ts == self._last_exit_bar_ts:
            return PositionUpdate(action="hold", reason="Waiting for the next 5m bar close")

        l3 = self._layer3_indicators(candles)
        if l3 is None:
            return PositionUpdate(action="hold", reason="Indicators warming up (5m)")
        self._last_exit_bar_ts = bar_ts

        # Optionally hold all signal exits until TP1 has banked — before that,
        # only the hard SL (EMA50) / TP bounds manage the trade. Cuts the
        # noise exits that killed trades at ~-0.3R before they could develop.
        if self.signal_exit_requires_tp1 and not self._tp1_done:
            return PositionUpdate(action="hold",
                                  reason=f"Holding {self._open_position.upper()} — signal exits armed after TP1")

        cb = self.exit_close_confirm_bars
        if self._open_position == "long":
            close_exit = (self.use_close_past_exit
                          and self._closes_past_ema_slow(candles, "long", cb))
            if l3["ema_cross_down"] or close_exit:
                reason = (f"EMA{self.ema_fast} crossed below EMA{self.ema_slow}" if l3["ema_cross_down"]
                          else f"{cb} close(s) below EMA{self.ema_slow}")
                self._reset_position_state()
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit LONG: {reason} (5m)")
        if self._open_position == "short":
            close_exit = (self.use_close_past_exit
                          and self._closes_past_ema_slow(candles, "short", cb))
            if l3["ema_cross_up"] or close_exit:
                reason = (f"EMA{self.ema_fast} crossed above EMA{self.ema_slow}" if l3["ema_cross_up"]
                          else f"{cb} close(s) above EMA{self.ema_slow}")
                self._reset_position_state()
                return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit SHORT: {reason} (5m)")

        return PositionUpdate(action="hold", reason=f"Holding {self._open_position.upper()}")

    def _closes_past_ema_slow(self, candles: list, side: str, n: int) -> bool:
        """True if the last `n` 5m bars ALL closed on the wrong side of EMA9
        for the position (long: below, short: above). n=1 reproduces the
        original single-close exit."""
        closes = [c.close for c in candles]
        ema_s = self.ema(closes, self.ema_slow)
        if len(closes) < n:
            return False
        for k in range(1, n + 1):
            if np.isnan(ema_s[-k]):
                return False
            if side == "long" and not (closes[-k] < ema_s[-k]):
                return False
            if side == "short" and not (closes[-k] > ema_s[-k]):
                return False
        return True

    def _reset_position_state(self) -> None:
        self._open_position = None
        self._entry_price = None
        self._entry_sl = None
        self._tp1_done = False

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Called by bot.py after ANY close, including the risk-manager's
        hard SL/TP fallback firing before the 5m EMA cross-back — without
        this, _open_position would stay set forever and analyze() would
        refuse all future entries."""
        self._reset_position_state()

    def cancel_pending_entry(self, reason: str = "") -> None:
        """Called by bot.py when a signal this strategy just emitted failed
        to actually open (rejected by risk/portfolio gates, insufficient
        balance, or an order error)."""
        self._reset_position_state()

    def attach_existing_position(self, direction: str, entry_price: float,
                                  stop_loss: Optional[float] = None,
                                  take_profit: Optional[float] = None) -> None:
        """Called once on bot startup when a position for this symbol is
        already open on the exchange (from before a restart) — nothing in
        this strategy's in-memory state would otherwise know about it, so
        analyze() would try to open a duplicate and tick_open_position()
        would never manage the exit. Seeds the partial-TP tracking from the
        reconciled entry/SL so TP1 can still fire on the recovered position
        (tp1_done left False — TP1 may not have triggered yet)."""
        self._open_position = direction
        self._entry_price = entry_price
        self._entry_sl = stop_loss
        self._tp1_done = False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _neutral_location_context(self) -> dict:
        return {
            "valid": True, "penalize": False, "reason": "filter_off_or_no_context",
            "structure_room_r": None, "nearest_opposing_zone": None,
            "location_type": "UNKNOWN", "structure_1h": "UNKNOWN",
            "bearish_sweep": False, "bullish_sweep": False,
        }

    def _location_context(self, direction: str, entry_price: float, atr_15m: float,
                          estimated_risk: float, candles_15m: list, candles_1h: list,
                          candles_4h: list, q15: Optional[dict] = None) -> dict:
        """Lightweight location and structure-room filter.

        It rejects only obvious conflicts: entering directly into an active HTF
        opposing pivot zone, confirmed opposite 1H swing structure, wrong-side
        liquidity sweep, or critically insufficient room. Borderline location
        merely raises Layer-2's threshold so trade frequency is preserved.

        `estimated_risk` is the candidate's actual entry->SL distance (EMA50 on
        5m, floored) so structure-room-R matches the real stop. `atr_15m` still
        sizes the 1h/4h pivot zone widths and the hard-zone-distance test.
        """
        if atr_15m <= 0 or estimated_risk <= 0:
            return self._neutral_location_context()

        p1h = self._confirmed_pivots(candles_1h, self.structure_pivot_left, self.structure_pivot_right)
        p4h = self._confirmed_pivots(candles_4h, self.structure_pivot_left, self.structure_pivot_right)
        structure_1h = self._swing_structure(p1h)

        # Strong opposite 1H structure is a hard conflict. A single lower high
        # is not enough; both highs and lows must confirm the opposing sequence.
        if direction == "up" and structure_1h == "bear":
            return {**self._neutral_location_context(), "valid": False,
                    "reason": "1H confirmed LH/LL structure", "structure_1h": structure_1h}
        if direction == "down" and structure_1h == "bull":
            return {**self._neutral_location_context(), "valid": False,
                    "reason": "1H confirmed HH/HL structure", "structure_1h": structure_1h}

        levels = []
        for tf, piv, bars, mult in (("1h", p1h, candles_1h, self.zone_width_atr_1h),
                                    ("4h", p4h, candles_4h, self.zone_width_atr_4h)):
            tf_atr = self._last_atr(bars, self.atr_period)
            width = max((tf_atr or atr_15m) * mult, atr_15m * 0.15)
            wanted = "high" if direction == "up" else "low"
            for pivot in piv:
                if pivot["type"] == wanted and self._pivot_still_active(bars, pivot, direction, width):
                    levels.append({"timeframe": tf, "price": pivot["price"], "width": width})

        if direction == "up":
            opposing = [z for z in levels if z["price"] > entry_price]
            nearest = min(opposing, key=lambda z: z["price"] - entry_price, default=None)
            room = (nearest["price"] - nearest["width"] - entry_price) if nearest else None
        else:
            opposing = [z for z in levels if z["price"] < entry_price]
            nearest = min(opposing, key=lambda z: entry_price - z["price"], default=None)
            room = (entry_price - (nearest["price"] + nearest["width"])) if nearest else None

        room_r = room / estimated_risk if (room is not None and estimated_risk > 0) else None
        zone_distance_atr = room / atr_15m if room is not None else None

        sweep = self._wrong_side_liquidity_sweep(candles_15m, direction, nearest)
        if sweep:
            return {**self._neutral_location_context(), "valid": False,
                    "reason": "wrong-side liquidity sweep/rejection at opposing zone",
                    "structure_1h": structure_1h, "nearest_opposing_zone": nearest,
                    "structure_room_r": round(room_r, 2) if room_r is not None else None,
                    "bearish_sweep": direction == "up", "bullish_sweep": direction == "down"}

        if nearest is not None and zone_distance_atr is not None and zone_distance_atr < self.hard_zone_distance_atr:
            return {**self._neutral_location_context(), "valid": False,
                    "reason": f"entry directly into {nearest['timeframe']} opposing zone",
                    "structure_1h": structure_1h, "nearest_opposing_zone": nearest,
                    "structure_room_r": round(room_r, 2) if room_r is not None else None}

        if room_r is not None and room_r < self.min_structure_room_r:
            return {**self._neutral_location_context(), "valid": False,
                    "reason": f"structure room {room_r:.2f}R below {self.min_structure_room_r:.2f}R",
                    "structure_1h": structure_1h, "nearest_opposing_zone": nearest,
                    "structure_room_r": round(room_r, 2)}

        midrange = self._is_midrange(candles_1h, entry_price)
        chop_val = (q15 or {}).get("breakdown", {}).get("chop_val")
        choppy_midrange = bool(self.reject_midrange_when_choppy and midrange and
                               chop_val is not None and chop_val >= self.chop_threshold - 3.0)

        penalize = choppy_midrange or (room_r is not None and room_r < self.preferred_structure_room_r)
        location_type = "MID_RANGE" if midrange else "EDGE_OR_TREND_LOCATION"
        reason = "acceptable location"
        if choppy_midrange:
            reason = "mid-range in choppy conditions; higher quality required"
        elif penalize:
            reason = f"limited room {room_r:.2f}R; higher quality required"

        return {
            "valid": True, "penalize": penalize, "reason": reason,
            "structure_room_r": round(room_r, 2) if room_r is not None else None,
            "nearest_opposing_zone": nearest, "location_type": location_type,
            "structure_1h": structure_1h, "bearish_sweep": False, "bullish_sweep": False,
        }

    @staticmethod
    def _confirmed_pivots(candles: list, left: int, right: int) -> list[dict]:
        if len(candles) < left + right + 3:
            return []
        out: list[dict] = []
        # Stop at len-right: pivots are only known after right-side bars close.
        for i in range(left, len(candles) - right):
            h = candles[i].high
            l = candles[i].low
            if all(h > candles[j].high for j in range(i-left, i)) and \
               all(h >= candles[j].high for j in range(i+1, i+right+1)):
                out.append({"type": "high", "price": float(h), "timestamp": candles[i].timestamp})
            if all(l < candles[j].low for j in range(i-left, i)) and \
               all(l <= candles[j].low for j in range(i+1, i+right+1)):
                out.append({"type": "low", "price": float(l), "timestamp": candles[i].timestamp})
        return out[-20:]

    @staticmethod
    def _pivot_still_active(candles: list, pivot: dict, direction: str, width: float) -> bool:
        """A pivot only still gates entries as live resistance/support if the
        market hasn't already closed decisively through it since it formed —
        an old high/low price has broken and moved on from shouldn't keep
        rejecting entries as though it were still a wall."""
        bars_after = [c for c in candles if c.timestamp > pivot["timestamp"]]
        if not bars_after:
            return True
        if direction == "up":
            # pivot is a resistance HIGH; broken once a close clears it + width
            return not any(c.close > pivot["price"] + width for c in bars_after)
        # pivot is a support LOW; broken once a close clears it - width
        return not any(c.close < pivot["price"] - width for c in bars_after)

    @staticmethod
    def _swing_structure(pivots: list[dict]) -> str:
        highs = [p["price"] for p in pivots if p["type"] == "high"]
        lows = [p["price"] for p in pivots if p["type"] == "low"]
        if len(highs) < 2 or len(lows) < 2:
            return "neutral"
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "bull"
        if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return "bear"
        return "transition"

    def _last_atr(self, candles: list, period: int) -> Optional[float]:
        if len(candles) < period + 2:
            return None
        arr = self.atr(candles, period)
        val = arr[-1]
        return None if np.isnan(val) or val <= 0 else float(val)

    @staticmethod
    def _is_midrange(candles_1h: list, price: float, lookback: int = 20) -> bool:
        if len(candles_1h) < lookback:
            return False
        window = candles_1h[-lookback:]
        hi = max(c.high for c in window)
        lo = min(c.low for c in window)
        if hi <= lo:
            return False
        pos = (price - lo) / (hi - lo)
        return 0.38 <= pos <= 0.62

    @staticmethod
    def _wrong_side_liquidity_sweep(candles_15m: list, direction: str, nearest: Optional[dict]) -> bool:
        """True if the last 2 bars show a stop-hunt at the opposing zone that
        hasn't since been reclaimed. Checking 2 bars (not just the latest)
        catches a sweep that fired the bar BEFORE the entry cross; but a sweep
        1 bar back that the market has already reclaimed (closed cleanly back
        on the trade's own side) is a false alarm, not a live rejection —
        don't let it block an otherwise-good entry."""
        if nearest is None or len(candles_15m) < 3:
            return False
        level = nearest["price"]
        width = nearest["width"]

        def is_sweep(c) -> bool:
            body = abs(c.close - c.open)
            rng = max(c.high - c.low, 1e-12)
            if direction == "up":
                # Price probes resistance but closes back below with a meaningful upper wick.
                return (c.high >= level - width and c.close < level - width * 0.25
                        and (c.high - max(c.open, c.close)) >= body * 1.2 and body / rng >= 0.15)
            return (c.low <= level + width and c.close > level + width * 0.25
                    and (min(c.open, c.close) - c.low) >= body * 1.2 and body / rng >= 0.15)

        last = candles_15m[-1]
        for idx in (-1, -2):
            c = candles_15m[idx]
            if not is_sweep(c):
                continue
            if idx == -2:
                reclaimed = (last.close > level + width if direction == "up"
                            else last.close < level - width)
                if reclaimed:
                    continue
            return True
        return False

    def _compute_sl_tp(self, direction: str, price: float, sl_ema_val: float) -> tuple[float, float]:
        # SL sits at the 5m EMA50 (sl_ema_val). Its distance from entry defines
        # R; TP2 = R x rr_ratio (2.5). If EMA50 sits right on price the raw SL
        # would be a microscopic noise-stop, so the distance is floored at
        # min_sl_pct (0.5% of price) and R/TP measured from the floored value.
        dist = max(abs(price - sl_ema_val), self.min_sl_pct * price)
        if direction == "long":
            sl = price - dist
            tp = price + dist * self.rr_ratio
        else:
            sl = price + dist
            tp = price - dist * self.rr_ratio
        return sl, tp

    def _sideways_context(self, candles: list) -> dict:
        """Range / sideways detector on the 15m context. Returns
        {is_sideways, signals, detail}. Fires up to 4 independent range
        signals; `sideways_min_signals` (default 2) of them = veto.

        Deliberately weighted toward signals that STAY range-y while ADX is
        still lagging at the start of a trend (EMA compression, high chop,
        tight price range), so a fresh/early trend isn't mistaken for a range.
        The ADX signal only counts when ADX is *really* weak (< sideways_adx_max,
        stricter than adx_threshold) for the same reason."""
        n = 20
        min_needed = max(self.quality_ema_slow + 2, self.adx_period * 2 + 2,
                         self.chop_period + 1, n + 1)
        if len(candles) < min_needed:
            return {"is_sideways": False, "signals": 0, "detail": {}}

        closes = [c.close for c in candles]
        ema20 = self.ema(closes, self.quality_ema_fast)
        ema50 = self.ema(closes, self.quality_ema_slow)
        atr_arr = self.atr(candles, self.atr_period)
        adx_arr, _p, _m = self.adx(candles, self.adx_period)
        chop_val = self._choppiness(candles, self.chop_period)
        atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        if atr <= 0 or np.isnan(ema20[-1]) or np.isnan(ema50[-1]):
            return {"is_sideways": False, "signals": 0, "detail": {}}

        ema_gap_atr = abs(ema20[-1] - ema50[-1]) / atr
        window = candles[-n:]
        rng_atr = (max(c.high for c in window) - min(c.low for c in window)) / atr
        adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 100.0

        sig = {
            "ema_compressed": ema_gap_atr < self.sideways_ema_compression_atr,
            "high_chop":      chop_val is not None and chop_val > self.chop_threshold,
            "tight_range":    rng_atr < self.sideways_range_atr,
            "weak_adx":       adx_val < self.sideways_adx_max,
        }
        count = sum(1 for v in sig.values() if v)
        return {
            "is_sideways": count >= self.sideways_min_signals,
            "signals": count,
            "detail": {**sig, "ema_gap_atr": round(ema_gap_atr, 2),
                       "range_atr": round(rng_atr, 2), "adx": round(adx_val, 1),
                       "chop": round(chop_val, 1) if chop_val is not None else None},
        }

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
        """Per-timeframe 0-100 trend-quality score for the confirmed direction,
        across 4 dimensions (sum of the weights = 100):

          BIAS     (bias_weight, 30) — structural direction: close vs EMA20,
                     EMA20 vs EMA50, EMA50 slope (each agreeing = 1/3). The
                     3rd is the EMA50's DIRECTION (rising/falling over
                     ema_slope_lookback bars), not price-vs-EMA50 — the level
                     check lagged badly on fresh trends (price hadn't reached
                     the slow EMA yet), the slope turns much sooner.
          TREND    (trend_weight, 30) — is there a real trend, not a range:
                     ADX strength (60%) + inverted Choppiness (40%).
          MOMENTUM (momentum_weight, 25) — push in the trend's direction:
                     RSI lean scaled (48%) + MACD histogram in-trend (52%).
          VOLUME   (volume_weight, 15) — participation: volume / SMA20(volume).

        Each dimension is scored 0..1 then multiplied by its weight, so a weak
        dimension can be outweighed by strong ones (soft scoring). Returns None
        if there aren't enough candles (caller treats that as 'warming up')."""
        min_needed = max(self.quality_ema_slow + self.ema_slope_lookback + 2, 2 * self.adx_period + 2,
                         self.chop_period + 1, self.volume_sma_period, self.rsi_period + 1,
                         self.macd_slow + self.macd_signal + 1)
        if len(candles) < min_needed:
            return None

        closes = [c.close for c in candles]
        ema_fast = self.ema(closes, self.quality_ema_fast)
        ema_slow = self.ema(closes, self.quality_ema_slow)
        rsi = self.rsi(closes, self.rsi_period)
        macd_line, macd_sig, _hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        adx_arr, _p, _m = self.adx(candles, self.adx_period)
        chop_val = self._choppiness(candles, self.chop_period)
        vols = [c.volume for c in candles]
        vol_sma = self.sma(vols, self.volume_sma_period)

        lb = self.ema_slope_lookback
        needed = [ema_fast[-1], ema_slow[-1], ema_slow[-1 - lb], rsi[-1], macd_line[-1],
                  macd_sig[-1], adx_arr[-1], vol_sma[-1]]
        if chop_val is None or any(np.isnan(x) for x in needed) or vol_sma[-1] <= 0:
            return None

        up = trend == "up"
        c = closes[-1]

        # ── BIAS (0..1): 3 structural checks, each agreeing with the trend ──
        bias_checks = [
            (c > ema_fast[-1]) == up,                     # price vs EMA20
            (ema_fast[-1] > ema_slow[-1]) == up,          # EMA20 vs EMA50
            (ema_slow[-1] > ema_slow[-1 - lb]) == up,     # EMA50 slope (less lag than price-vs-EMA50)
        ]
        bias01 = sum(1 for v in bias_checks if v) / 3.0

        # ── TREND (0..1): ADX strength (60%) + inverted Choppiness (40%) ────
        adx_val = float(adx_arr[-1])
        adx01 = min(1.0, adx_val / max(1.0, self.adx_threshold * 2.0))
        chop_full_at = max(1.0, 100.0 - self.chop_threshold)
        chop01 = max(0.0, min(1.0, (100.0 - chop_val) / chop_full_at))
        trend01 = 0.60 * adx01 + 0.40 * chop01

        # ── MOMENTUM (0..1): RSI lean (48%) + MACD histogram in-trend (52%) ─
        if up:
            rsi01 = max(0.0, min(1.0, (rsi[-1] - 50.0) / max(1.0, self.rsi_bull + 15.0 - 50.0)))
        else:
            rsi01 = max(0.0, min(1.0, (50.0 - rsi[-1]) / max(1.0, 50.0 - (self.rsi_bear - 15.0))))
        hist = macd_line[-1] - macd_sig[-1]
        macd01 = 1.0 if ((hist > 0) == up) else 0.0
        mom01 = 0.48 * rsi01 + 0.52 * macd01

        # ── VOLUME (0..1): volume vs its SMA, full at 2x expansion multiple ─
        vol_ratio = float(candles[-1].volume) / float(vol_sma[-1])
        vol01 = min(1.0, vol_ratio / max(0.01, self.volume_expansion_mult * 2.0))

        bias_pts = bias01 * self.bias_weight
        trend_pts = trend01 * self.trend_weight
        mom_pts = mom01 * self.momentum_weight
        vol_pts = vol01 * self.volume_weight

        breakdown = {
            "bias": round(bias_pts, 1), "trend": round(trend_pts, 1),
            "momentum": round(mom_pts, 1), "volume": round(vol_pts, 1),
            "adx_val": round(adx_val, 1), "chop_val": round(chop_val, 1),
            "rsi_val": round(float(rsi[-1]), 1), "vol_ratio": round(vol_ratio, 2),
        }
        score = bias_pts + trend_pts + mom_pts + vol_pts
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
        """Entry/exit indicators on the 5m series: EMA5/9 cross, price vs the
        entry EMA, the EMA50 stop reference + chase-guard distance, and ATR."""
        closes = [c.close for c in candles]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        ema_ref = self.ema(closes, self.entry_ema_ref)
        sl_ema = self.ema(closes, self.sl_ema_ref)
        chase_ema = self.ema(closes, self.chase_ema_ref)
        atr_arr = self.atr(candles, self.atr_period)

        needed = [ema_f[-1], ema_f[-2], ema_s[-1], ema_s[-2],
                 ema_ref[-1], sl_ema[-1], chase_ema[-1], atr_arr[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        last = candles[-1]
        return {
            "ema_cross_up":   ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1],
            "ema_cross_down": ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1],
            "ema_slow_val": float(ema_s[-1]),
            "close_below_ema_slow": last.close < ema_s[-1],
            "close_above_ema_slow": last.close > ema_s[-1],
            "above_ema_ref": last.close > ema_ref[-1],
            "below_ema_ref": last.close < ema_ref[-1],
            "sl_ema_val": float(sl_ema[-1]),
            "dist_ema_val": float(chase_ema[-1]),   # chase-guard reference
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
