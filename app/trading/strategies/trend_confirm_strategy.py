"""
Trend-Confirmed Multi-TF Strategy — Adaptive HTF Context V2.1.

V2 keeps the three 5M entry engines (EMA_CROSS, BREAKOUT_RETEST and
STRUCTURE_RETEST) and adds a production context architecture around them:
  Layer 0 — OHLCV/timestamp data-quality validation.
  Layer 1 — 30M confirmed trend direction.
  Layer 1B — 4H major + 1H minor swing structure.
  Layer 2 — 15M/1H quality score and explicit 15M regime router.
  Layer 2B — 4H/1H supply-demand zones with freshness, touch count,
             invalidation, role reversal and opposing-zone room in R.
  Layer 3 — 5M three-engine candidate router and candidate-specific stops.

Layer 1 — Trend direction (TF30m):
  Determines whether the market is currently in an uptrend or a
  downtrend. EMA10/20 alignment is mandatory and at least 3 of the 4
  direction checks must agree. This preserves trend confirmation without
  requiring slower SMA/MACD components to turn on the exact same bar:
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
    then combines them 15m x 70% + 1H x 30%. Must score above the balanced
    `layer2_threshold` (default 56) established, or 62 for
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

Layer 3 — Multi-entry router (TF5m), reached only after Layer1 + Layer2 pass:
  A) EMA_CROSS — original EMA10/20 cross timing.
  B) BREAKOUT_RETEST — a qualified breakout (body/close/volume) must return to
     the broken level and reclaim it; direct breakout chasing is not allowed.
  C) STRUCTURE_RETEST — confirmed HH/HL (long) or LH/LL (short), followed by a
     retest of the latest HL/LH and reclaim, engulf/pin, or micro-BOS.
  Every engine still requires EMA alignment, the EMA50 chase guard, HTF
  location/room validation, and a candidate-specific quality threshold.
  Early-trend window: the 5m EMA10/20 cross is faster than Layer 1's 30m
  confirmation, so the cross that starts a move often fires a bar or two
  BEFORE the trend confirms. When Layer 1's trend JUST confirmed (within
  `fresh_trend_bars`, default 2, 5m bars of when it flipped) the entry counts
  a cross up to fresh_trend_bars ago — which may predate the confirmation.
  Entering this early is riskier, so it must clear the STRICTER
  `layer2_threshold_early` (62) instead of the normal 56. Outside that
  window the established-trend threshold applies.
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
  Runner (remaining 50%): rides on until, on the 5m TF, EITHER the EMA10/20
    cross-back (long: EMA10 crosses below EMA20) OR a close past EMA20 (long:
    5m close below EMA20) — mirror for shorts — the hard final TP (1.5R), or
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
        layer1_min_agreement: int = 3,  # 3-of-4 direction vote; EMA10/20 remains mandatory
        layer1_require_ema_alignment: bool = True,
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
        tf_weight_15m: float = 0.70,
        tf_weight_1h: float = 0.30,
        layer2_threshold: float = 56.0,        # balanced live threshold; HTF context still protects location
        layer2_threshold_early: float = 62.0,  # early setups remain stricter without becoming practically unreachable
        allow_15m_quality_fallback: bool = True,
        single_tf_quality_penalty: float = 4.0,
        # Layer 3 — entry (5m): EMA10/20 cross, price above/below EMA20, within 1.5xATR of EMA50
        entry_tf: str = "15m",      # timeframe the entry cross + exit run on (15m base beats 5m fee drag)
        trend_tf: str = "1h",       # Layer1 trend timeframe: "1h" (uses mtf 1h) or "30m" (15m resample)
        ema_fast: int = 8,          # entry-cross fast EMA (on entry_tf)
        ema_slow: int = 13,         # entry-cross slow EMA (on entry_tf); also the cross-back exit reference
        entry_ema_ref: int = 13,    # price must be above (long) / below (short) this EMA — same line the cross + exit use
        sl_ema_ref: int = 50,       # SL sits at this EMA (5m)
        chase_ema_ref: int = 50,    # chase-guard distance is measured vs this EMA (5m); decoupled from sl_ema_ref
        fresh_trend_bars: int = 3,  # EMA-cross lookback (in 5m bars) when the trend just confirmed (early trend)
        cross_valid_bars: int = 6,  # how many 5m bars a cross stays usable while Layer2 gates settle —
                                    #   without this, a cross was only good on the exact bar every gate was
                                    #   already open (quality/location often clear 1-2 bars AFTER the cross,
                                    #   which silently wasted almost every signal)
        max_dist_atr_mult: float = 2.2,  # EMA-cross chase limit in ATR(5m)
        breakout_max_dist_atr_mult: float = 2.8,
        structure_max_dist_atr_mult: float = 2.4,
        # Layer 3 Entry Router — three independent triggers. Breakout uses
        # retest-only execution (no direct chasing) and structure entry requires
        # a confirmed HH/HL or LH/LL sequence plus a reclaim / micro-BOS trigger.
        use_ema_cross_entry: bool = True,
        use_breakout_retest_entry: bool = True,
        use_structure_retest_entry: bool = True,
        use_price_action_structure_entry: bool = True,
        # Price-action structure confirmation: sweep/reclaim or BOS/CHOCH with
        # candle displacement and micro-structure confirmation. This engine is
        # especially useful in TRANSITION, where EMA alignment is developing
        # but a clean price-action shift already exists.
        entry_trigger_valid_bars: int = 3,
        breakout_lookback: int = 6,
        breakout_arm_bars: int = 6,
        breakout_buffer_atr: float = 0.05,
        breakout_retest_tolerance_atr: float = 0.30,
        breakout_invalidation_atr: float = 0.35,
        breakout_min_body_atr: float = 0.20,
        breakout_min_close_quality: float = 0.65,
        breakout_min_volume_ratio: float = 0.95,
        breakout_entry_min_quality: float = 64.0,
        structure_retest_window_bars: int = 8,
        structure_level_max_age_bars: int = 60,
        structure_retest_tolerance_atr: float = 0.35,
        structure_invalidation_atr: float = 0.30,
        structure_micro_bos_lookback: int = 2,
        structure_entry_min_quality: float = 62.0,
        pa_structure_entry_min_quality: float = 64.0,
        pa_structure_lookback: int = 12,
        pa_structure_swing_lookback: int = 5,
        pa_min_body_atr: float = 0.18,
        pa_min_close_quality: float = 0.62,
        pa_min_volume_ratio: float = 0.80,
        pa_sweep_tolerance_atr: float = 0.12,
        pa_break_buffer_atr: float = 0.04,
        pa_require_sweep_or_choch: bool = True,
        entry_stop_buffer_atr: float = 0.10,
        entry_min_stop_atr: float = 0.50,
        entry_max_stop_atr: float = 2.20,
        # Position sizing (emitted in the signal so bot.py sizes live orders the
        # same way the paper account does): margin = margin_pct of balance,
        # notional = margin x leverage. e.g. $100 x 5% = $5 x 20 = $100 notional.
        sizing_mode: str = "margin",
        margin_pct: float = 0.05,
        # Location & structure-room filter (lightweight; avoids late/blocked entries)
        use_location_filter: bool = True,
        structure_pivot_left: int = 2,
        structure_pivot_right: int = 2,
        zone_width_atr_1h: float = 0.18,
        zone_width_atr_4h: float = 0.22,
        hard_zone_distance_atr: float = 0.15,
        min_structure_room_r: float = 0.25,   # hard-reject below this many R of room to the opposing zone
        preferred_structure_room_r: float = 0.75,  # below this (but >= min) just penalizes the quality gate
        location_threshold_penalty: float = 3.0,
        max_location_threshold_penalty: float = 8.0,
        reject_midrange_when_choppy: bool = True,
        # Production data / HTF context gates
        use_data_quality_gate: bool = True,
        max_recent_gap_mult: float = 20.0,
        require_4h_context: bool = False,
        min_4h_context_bars: int = 30,
        use_htf_macro_filter: bool = True,
        htf_transition_threshold_penalty: float = 2.0,
        htf_single_conflict_penalty: float = 4.0,
        htf_alignment_edge_bonus: float = 3.0,
        # 4H/1H supply-demand engine. Zones are detected from a compact base
        # followed by an ATR-qualified departure, then tracked for touches,
        # invalidation, freshness and support/resistance role reversal.
        use_supply_demand_zones: bool = True,
        sd_scan_lookback_bars: int = 180,
        sd_base_max_bars: int = 4,
        sd_base_max_range_atr: float = 1.00,
        sd_base_max_body_atr: float = 0.45,
        sd_departure_bars: int = 3,
        sd_min_departure_atr: float = 1.20,
        sd_break_buffer_atr: float = 0.10,
        sd_touch_tolerance_atr: float = 0.08,
        sd_max_touches: int = 3,
        sd_touch_freshness_decay: float = 0.25,
        sd_min_freshness: float = 0.25,
        sd_supportive_zone_distance_atr: float = 0.30,
        sd_supportive_zone_edge_bonus: float = 3.0,
        sd_role_reversal_edge_bonus: float = 1.5,
        # Explicit 15M regime router. Strong trends receive a small threshold
        # discount; transition regimes require more quality and do not permit
        # the structure-retest engine until the structure is established.
        use_regime_router: bool = True,
        regime_strong_adx: float = 25.0,
        regime_trend_adx: float = 17.0,
        regime_strong_chop_max: float = 50.0,
        regime_trend_chop_max: float = 61.8,
        regime_min_ema_gap_atr: float = 0.35,
        regime_strong_threshold_discount: float = 3.0,
        regime_transition_threshold_penalty: float = 1.5,
        allow_structure_entry_in_transition: bool = True,
        require_trend_regime: bool = False,     # allow TRANSITION through the dedicated confirmation gate (regime TREND/STRONG_TREND).
                                                #   TRANSITION/CHOP no longer trade — cuts the unclear-market
                                                #   churn. Set False to allow TRANSITION via the extra gate below.
        # TRANSITION extra-analysis gate — these regimes CAN trade, but need
        # real confirmation. Kept LIGHT on the quality axis (a fresh cross is
        # usually early-trend, which already carries the stricter early
        # threshold) — the extra confidence comes from momentum + volume, not a
        # sky-high score bar that would block essentially every cross.
        transition_extra_threshold: float = 0.0,    # no quality stacking (6 blocked ~every cross). The extra
                                                     #   analysis is momentum + volume below, not a higher bar.
        transition_min_vol_ratio: float = 0.75,      # only block genuinely dead volume (< 0.5x its SMA)
        transition_require_momentum: bool = True,   # need MACD histogram pushing in the trend's direction
        transition_momentum_frac: float = 0.35,      # momentum dimension must reach this fraction of its weight
        transition_require_clean_location: bool = True,  # reject borderline/penalized HTF location
        # Sideways / range veto (Layer 2) — hard-block entries when the 15m
        # context looks like a range, not a trend. Designed NOT to kill early
        # trends: it leans on EMA compression + high chop (which stay range-y
        # even as ADX lags), and only counts "really weak" ADX (< sideways_adx_max,
        # stricter than adx_threshold) so a fresh trend at ADX ~18 isn't vetoed.
        use_sideways_filter: bool = True,
        sideways_ema_compression_atr: float = 0.5,  # |EMA20-EMA50| < this x ATR = tangled/flat
        sideways_adx_max: float = 15.0,             # ADX below this = "really weak" (< adx_threshold on purpose)
        sideways_range_atr: float = 1.2,            # last-20-bar high-low range < this x ATR = tight consolidation
        sideways_min_signals: int = 3,              # how many of the 4 signals must fire to veto (clear ranges only)
        # Exit (5m): EMA10/20 cross-back OR a 5m close past EMA20 closes the runner
        # ── Exit style ────────────────────────────────────────────────────
        # STICKY by default: stay in the trade until the TREND actually reverses
        # (the 1h trend flips, or price closes past EMA50) — not on every fast
        # EMA8/13 wiggle, which was churning small losses in unclear markets.
        exit_on_trend_flip: bool = True,    # exit when the trend_tf (1h) Layer1 trend flips against us
        use_structural_exit: bool = True,   # exit when price closes past EMA50 (sl_ema_ref) — a real break
        exit_structural_confirm_bars: int = 1,   # N closes past EMA50 required
        use_ema_crossback_exit: bool = False,    # ALWAYS-on fast EMA8/13 cross-back exit — OFF (caused whipsaw)
        crossback_exit_after_target: bool = True,  # once IN PROFIT or past +0.8R (be_trailed), DO take the fast
                                                   #   EMA8/13 cross-back — lock the gain on a reversal. Before
                                                   #   profit it stays sticky (only trend-flip / EMA50 exits).
        use_close_past_exit: bool = False,  # the fast "close past EMA13" exit — OFF (too twitchy)
        exit_close_confirm_bars: int = 1,   # N consecutive closes past EMA_slow required (if the above is on)
        signal_exit_requires_tp1: bool = False,  # cross-back exit works immediately (no TP1 to wait for now)
                                                 #   bounds manage the trade until then. On 5m the single-close
                                                 #   slow-EMA exits killed 75% of trades at ~-0.3R before TP1; arming
                                                 #   them only on the runner nearly doubled WR (25->62% BTC,
                                                 #   41->60% SOL) and cut losses ~2-3x in backtest
        # Take-profit scheme. use_hard_tp=False + use_partial_tp=False = a pure
        # trend-follow cross system: no TP at all, ride the position until the
        # EMA8/13 cross-back (the SL at EMA50 is only a disaster stop).
        use_hard_tp: bool = False,          # emit a fixed TP2 (1.5R) with the entry? off = hold to cross-back
        use_partial_tp: bool = False,       # TP1 -> take tp1_close_pct, move SL to BE+be_offset_r; runner rides on
        # Break-even TRAIL (no partial close): once price reaches be_trail_trigger_r
        # of profit, ratchet the SL up to entry + be_trail_sl_r (locks a minimum
        # profit) and keep riding the full position until the EMA8/13 cross-back.
        use_be_trail: bool = True,
        be_trail_trigger_r: float = 0.8,    # target: move the SL once +0.8R is reached
        be_trail_sl_r: float = 0.5,         # new SL sits at entry +/- 0.5R (BE + 0.5R locked)
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
        self.layer1_min_agreement = max(2, min(4, int(layer1_min_agreement)))
        self.layer1_require_ema_alignment = bool(layer1_require_ema_alignment)

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
        self.allow_15m_quality_fallback = bool(allow_15m_quality_fallback)
        self.single_tf_quality_penalty = max(0.0, float(single_tf_quality_penalty))

        self.entry_tf = entry_tf
        self.trend_tf = trend_tf
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.entry_ema_ref = entry_ema_ref
        self.sl_ema_ref = sl_ema_ref
        self.chase_ema_ref = chase_ema_ref
        self.fresh_trend_bars = fresh_trend_bars
        self.cross_valid_bars = cross_valid_bars
        self.sizing_mode = sizing_mode
        self.margin_pct = margin_pct
        self.max_dist_atr_mult = max(0.5, float(max_dist_atr_mult))
        self.breakout_max_dist_atr_mult = max(self.max_dist_atr_mult, float(breakout_max_dist_atr_mult))
        self.structure_max_dist_atr_mult = max(self.max_dist_atr_mult, float(structure_max_dist_atr_mult))
        self.use_ema_cross_entry = use_ema_cross_entry
        self.use_breakout_retest_entry = use_breakout_retest_entry
        self.use_structure_retest_entry = use_structure_retest_entry
        self.use_price_action_structure_entry = use_price_action_structure_entry
        self.entry_trigger_valid_bars = max(1, entry_trigger_valid_bars)
        self.breakout_lookback = max(3, breakout_lookback)
        self.breakout_arm_bars = max(1, breakout_arm_bars)
        self.breakout_buffer_atr = breakout_buffer_atr
        self.breakout_retest_tolerance_atr = breakout_retest_tolerance_atr
        self.breakout_invalidation_atr = breakout_invalidation_atr
        self.breakout_min_body_atr = breakout_min_body_atr
        self.breakout_min_close_quality = breakout_min_close_quality
        self.breakout_min_volume_ratio = breakout_min_volume_ratio
        self.breakout_entry_min_quality = breakout_entry_min_quality
        self.structure_retest_window_bars = max(2, structure_retest_window_bars)
        self.structure_level_max_age_bars = max(5, structure_level_max_age_bars)
        self.structure_retest_tolerance_atr = structure_retest_tolerance_atr
        self.structure_invalidation_atr = structure_invalidation_atr
        self.structure_micro_bos_lookback = max(1, structure_micro_bos_lookback)
        self.structure_entry_min_quality = structure_entry_min_quality
        self.pa_structure_entry_min_quality = pa_structure_entry_min_quality
        self.pa_structure_lookback = max(6, int(pa_structure_lookback))
        self.pa_structure_swing_lookback = max(2, int(pa_structure_swing_lookback))
        self.pa_min_body_atr = max(0.05, float(pa_min_body_atr))
        self.pa_min_close_quality = max(0.50, min(0.95, float(pa_min_close_quality)))
        self.pa_min_volume_ratio = max(0.0, float(pa_min_volume_ratio))
        self.pa_sweep_tolerance_atr = max(0.0, float(pa_sweep_tolerance_atr))
        self.pa_break_buffer_atr = max(0.0, float(pa_break_buffer_atr))
        self.pa_require_sweep_or_choch = bool(pa_require_sweep_or_choch)
        self.entry_stop_buffer_atr = entry_stop_buffer_atr
        self.entry_min_stop_atr = entry_min_stop_atr
        self.entry_max_stop_atr = entry_max_stop_atr

        self.use_location_filter = use_location_filter
        self.structure_pivot_left = structure_pivot_left
        self.structure_pivot_right = structure_pivot_right
        self.zone_width_atr_1h = zone_width_atr_1h
        self.zone_width_atr_4h = zone_width_atr_4h
        self.hard_zone_distance_atr = hard_zone_distance_atr
        self.min_structure_room_r = min_structure_room_r
        self.preferred_structure_room_r = preferred_structure_room_r
        self.location_threshold_penalty = max(0.0, float(location_threshold_penalty))
        self.max_location_threshold_penalty = max(self.location_threshold_penalty, float(max_location_threshold_penalty))
        self.reject_midrange_when_choppy = reject_midrange_when_choppy
        self.use_data_quality_gate = use_data_quality_gate
        self.max_recent_gap_mult = max(2.0, max_recent_gap_mult)
        self.require_4h_context = require_4h_context
        self.min_4h_context_bars = max(20, min_4h_context_bars)
        self.use_htf_macro_filter = use_htf_macro_filter
        self.htf_transition_threshold_penalty = max(0.0, htf_transition_threshold_penalty)
        self.htf_single_conflict_penalty = max(0.0, htf_single_conflict_penalty)
        self.htf_alignment_edge_bonus = max(0.0, htf_alignment_edge_bonus)
        self.use_supply_demand_zones = use_supply_demand_zones
        self.sd_scan_lookback_bars = max(30, sd_scan_lookback_bars)
        self.sd_base_max_bars = max(1, sd_base_max_bars)
        self.sd_base_max_range_atr = max(0.10, sd_base_max_range_atr)
        self.sd_base_max_body_atr = max(0.05, sd_base_max_body_atr)
        self.sd_departure_bars = max(1, sd_departure_bars)
        self.sd_min_departure_atr = max(0.25, sd_min_departure_atr)
        self.sd_break_buffer_atr = max(0.0, sd_break_buffer_atr)
        self.sd_touch_tolerance_atr = max(0.0, sd_touch_tolerance_atr)
        self.sd_max_touches = max(0, sd_max_touches)
        self.sd_touch_freshness_decay = min(1.0, max(0.0, sd_touch_freshness_decay))
        self.sd_min_freshness = min(1.0, max(0.0, sd_min_freshness))
        self.sd_supportive_zone_distance_atr = max(0.0, sd_supportive_zone_distance_atr)
        self.sd_supportive_zone_edge_bonus = max(0.0, sd_supportive_zone_edge_bonus)
        self.sd_role_reversal_edge_bonus = max(0.0, sd_role_reversal_edge_bonus)
        self.use_regime_router = use_regime_router
        self.regime_strong_adx = max(1.0, regime_strong_adx)
        self.regime_trend_adx = max(1.0, min(regime_trend_adx, self.regime_strong_adx))
        self.regime_strong_chop_max = max(1.0, regime_strong_chop_max)
        self.regime_trend_chop_max = max(self.regime_strong_chop_max, regime_trend_chop_max)
        self.regime_min_ema_gap_atr = max(0.0, regime_min_ema_gap_atr)
        self.regime_strong_threshold_discount = max(0.0, regime_strong_threshold_discount)
        self.regime_transition_threshold_penalty = max(0.0, regime_transition_threshold_penalty)
        self.allow_structure_entry_in_transition = allow_structure_entry_in_transition
        self.require_trend_regime = require_trend_regime
        self.transition_extra_threshold = max(0.0, transition_extra_threshold)
        self.transition_min_vol_ratio = max(0.0, transition_min_vol_ratio)
        self.transition_require_momentum = transition_require_momentum
        self.transition_momentum_frac = max(0.0, min(1.0, transition_momentum_frac))
        self.transition_require_clean_location = transition_require_clean_location
        self.use_sideways_filter = use_sideways_filter
        self.sideways_ema_compression_atr = sideways_ema_compression_atr
        self.sideways_adx_max = sideways_adx_max
        self.sideways_range_atr = sideways_range_atr
        self.sideways_min_signals = sideways_min_signals

        self.exit_on_trend_flip = exit_on_trend_flip
        self.use_structural_exit = use_structural_exit
        self.exit_structural_confirm_bars = max(1, int(exit_structural_confirm_bars))
        self.use_ema_crossback_exit = use_ema_crossback_exit
        self.crossback_exit_after_target = crossback_exit_after_target
        self.use_close_past_exit = use_close_past_exit
        self.exit_close_confirm_bars = exit_close_confirm_bars
        self.signal_exit_requires_tp1 = signal_exit_requires_tp1

        self.use_hard_tp = use_hard_tp
        self.use_partial_tp = use_partial_tp
        self.use_be_trail = use_be_trail
        self.be_trail_trigger_r = be_trail_trigger_r
        self.be_trail_sl_r = be_trail_sl_r
        self._be_trailed: bool = False
        self._entry_threshold_bonus: float = 0.0  # bot raises this during post-cooldown strict window
        self.tp1_r = tp1_r
        self.tp1_close_pct = tp1_close_pct
        self.be_offset_r = be_offset_r

        self.atr_period = atr_period
        self.rr_ratio = rr_ratio
        self.min_sl_pct = min_sl_pct

        # Apply params passed by the bot/config. Older versions forwarded params
        # to BaseStrategy but silently ignored them in this class.
        self._apply_runtime_params(params)

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._trend_state: Optional[str] = None      # "up" | "down" | None — Layer 1 result
        self._trend_confirmed_since_ts: Optional[int] = None  # 5m bar_ts when trend last CHANGED
        self._last_bar_ts_30: Optional[int] = None    # Layer 1 new-bar tracking
        self._last_bar_ts_5: Optional[int] = None     # Layer 3 (5m) new-bar tracking
        self._last_ema_cross_up_ts: Optional[int] = None
        self._last_ema_cross_down_ts: Optional[int] = None
        self._last_breakout_trigger_up_ts: Optional[int] = None
        self._last_breakout_trigger_down_ts: Optional[int] = None
        self._last_structure_trigger_up_ts: Optional[int] = None
        self._last_structure_trigger_down_ts: Optional[int] = None
        self._last_pa_structure_trigger_up_ts: Optional[int] = None
        self._last_pa_structure_trigger_down_ts: Optional[int] = None
        self._last_entry_attempt_bar_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None  # owned by tick_open_position()
        self._latest_candles: list = []
        self._latest_5m: list = []       # 5m series cached for tick_open_position()
        # Partial-TP tracking for the open position (owned by tick_open_position())
        self._entry_price: Optional[float] = None
        self._entry_sl: Optional[float] = None
        self._tp1_done: bool = False
        # HTF zone cache: location is evaluated once per candidate, but the
        # expensive zone scan only needs to rerun when a 1H/4H bar changes.
        self._zone_cache_key: Optional[tuple] = None
        self._zone_cache: list[dict] = []


    def _apply_runtime_params(self, params: Optional[dict]) -> None:
        """Apply strategy settings supplied through the bot's params dict.

        The previous file accepted ``params`` but never copied those values to
        this strategy, so Railway/config changes appeared to work while the
        strict defaults kept running. Only existing public scalar attributes
        are accepted; unknown keys are ignored.
        """
        if not isinstance(params, dict):
            return
        cfg = params
        for nested_key in ("strategy_params", "trend_confirm", "trend_confirm_strategy"):
            nested = params.get(nested_key)
            if isinstance(nested, dict):
                cfg = nested
                break

        protected = {"name", "symbol", "params"}
        for key, raw in cfg.items():
            if key in protected or key.startswith("_") or not hasattr(self, key):
                continue
            current = getattr(self, key)
            if not isinstance(current, (bool, int, float, str)):
                continue
            try:
                if isinstance(current, bool):
                    if isinstance(raw, str):
                        value = raw.strip().lower() in {"1", "true", "yes", "on"}
                    else:
                        value = bool(raw)
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(raw)
                elif isinstance(current, float):
                    value = float(raw)
                else:
                    value = str(raw)
                setattr(self, key, value)
            except (TypeError, ValueError):
                continue

        # Re-normalize values that can make all signals impossible when a bad
        # environment value is supplied.
        self.layer1_min_agreement = max(2, min(4, int(self.layer1_min_agreement)))
        self.tf_weight_15m = max(0.0, float(self.tf_weight_15m))
        self.tf_weight_1h = max(0.0, float(self.tf_weight_1h))
        weight_sum = self.tf_weight_15m + self.tf_weight_1h
        if weight_sum <= 0:
            self.tf_weight_15m, self.tf_weight_1h = 0.70, 0.30
        else:
            self.tf_weight_15m /= weight_sum
            self.tf_weight_1h /= weight_sum
        self.layer2_threshold = max(35.0, min(85.0, float(self.layer2_threshold)))
        self.layer2_threshold_early = max(self.layer2_threshold, min(90.0, float(self.layer2_threshold_early)))
        self.cross_valid_bars = max(1, int(self.cross_valid_bars))
        self.entry_trigger_valid_bars = max(1, int(self.entry_trigger_valid_bars))
        self.max_dist_atr_mult = max(0.5, float(self.max_dist_atr_mult))
        self.breakout_max_dist_atr_mult = max(self.max_dist_atr_mult, float(self.breakout_max_dist_atr_mult))
        self.structure_max_dist_atr_mult = max(self.max_dist_atr_mult, float(self.structure_max_dist_atr_mult))
        self.sideways_min_signals = max(2, min(4, int(self.sideways_min_signals)))
        self.location_threshold_penalty = max(0.0, float(self.location_threshold_penalty))
        self.max_location_threshold_penalty = max(
            self.location_threshold_penalty, float(self.max_location_threshold_penalty)
        )
        self.min_4h_context_bars = max(10, int(self.min_4h_context_bars))

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles  # 15m base (Layer1 30m resample + Layer2 15m)
        mtf = mtf_candles or {}
        if not candles:
            return self._hold(current_price, "Data Quality: empty 15m candle series")
        bar_ts_15 = candles[-1].timestamp
        _TF_MS = {"5m": 5 * 60_000, "15m": 15 * 60_000, "30m": 30 * 60_000,
                  "1h": 60 * 60_000, "4h": 4 * 60 * 60_000}
        entry_ms = _TF_MS.get(self.entry_tf, 15 * 60_000)
        # Entry, SL/TP and exit all run on the entry_tf series. When entry_tf is
        # the 15m base, that IS `candles` — mtf usually only carries 5m/1h/4h.
        c5m = mtf.get(self.entry_tf, []) or []
        if not c5m and self.entry_tf == "15m":
            c5m = candles
        # Some bot deployments only request 5m/15m data. Build missing HTFs
        # from the 15m series instead of permanently blocking all entries.
        c1h = mtf.get("1h", []) or self._resample_timeframe(candles, 60 * 60_000, 15 * 60_000)
        c4h = mtf.get("4h", []) or self._resample_timeframe(candles, 4 * 60 * 60_000, 15 * 60_000)
        self._latest_5m = c5m  # cached for tick_open_position() (name kept; = entry_tf series)
        bar_ts_5 = c5m[-1].timestamp if c5m else None

        # ── Layer 0: production data-quality gate ──────────────────────────
        data_quality: dict = {}
        if self.use_data_quality_gate:
            for tf_name, series, expected_ms, required in (
                ("15m", candles, 15 * 60_000, True),
                (self.entry_tf, c5m, entry_ms, True),
                ("1h", c1h, 60 * 60_000, False),
                ("4h", c4h, 4 * 60 * 60_000, self.require_4h_context),
            ):
                if not series and not required:
                    data_quality[tf_name] = {"valid": True, "reason": "optional_context_missing", "bars": 0}
                    continue
                quality = self._data_quality_context(series, expected_ms)
                data_quality[tf_name] = quality
                if not quality["valid"]:
                    return self._hold(
                        current_price,
                        f"Data Quality FAIL {tf_name}: {quality['reason']}",
                        metadata={"data_quality": data_quality},
                    )
        if self.require_4h_context and len(c4h) < self.min_4h_context_bars:
            return self._hold(
                current_price,
                f"HTF Context: need {self.min_4h_context_bars}+ 4H bars, have {len(c4h)}",
                metadata={"data_quality": data_quality},
            )

        # ── Layer 1: Trend direction (trend_tf) ────────────────────────────
        # "1h" reads the mtf 1H series (falls back to a 15m→1h resample);
        # "30m" keeps the legacy 15m→30m resample.
        c30 = c1h if self.trend_tf == "1h" else self._closed_30m_bars(candles)
        min_needed_30 = max(self.ema2_period + self.ema_slope_lookback, self.sma_trend,
                            self.macd_slow + self.macd_signal) + 5
        if len(c30) < min_needed_30:
            return self._hold(current_price, f"Layer1: need {min_needed_30}+ closed {self.trend_tf} bars, have {len(c30)}")

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

        # ── Layer 3 EMA10/20-cross tracking (5m) — runs every new 5m bar
        # regardless of Layer1/Layer2 gating, so a cross that fires just
        # before the trend confirms is still remembered within the
        # fresh-trend window. ───────────────────────────────────────────────
        l3 = self._layer3_indicators(c5m) if c5m else None
        is_new_bar_5 = bar_ts_5 is not None and bar_ts_5 != self._last_bar_ts_5
        if is_new_bar_5:
            self._last_bar_ts_5 = bar_ts_5
        # Re-check the latest candle on every analysis call. Some exchanges
        # return the currently forming 5m bar; the old code inspected it only
        # once at bar open and therefore missed crosses that formed later.
        if bar_ts_5 is not None and l3 is not None:
            if l3["ema_cross_up"] and self._last_ema_cross_up_ts != bar_ts_5:
                self._last_ema_cross_up_ts = bar_ts_5
            if l3["ema_cross_down"] and self._last_ema_cross_down_ts != bar_ts_5:
                self._last_ema_cross_down_ts = bar_ts_5

        def _bars_ago_5(ts: Optional[int]) -> Optional[int]:
            if ts is None or bar_ts_5 is None:
                return None
            delta = bar_ts_5 - ts
            if delta < 0:
                return None
            return delta // (5 * 60_000)

        ema_up_ago   = _bars_ago_5(self._last_ema_cross_up_ts)
        ema_down_ago = _bars_ago_5(self._last_ema_cross_down_ts)

        # "Early trend": the trend just confirmed (within fresh_trend_bars 5m
        # bars). Because the 5m EMA10/20 cross is faster than Layer1's 30m
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
        base_l2_thr = self.layer2_threshold_early if is_early_trend else self.layer2_threshold
        # Post-cooldown tightening: bot raises this after a symbol resumes from a
        # losing-streak cooldown, so its first few re-entries need better quality.
        base_l2_thr += getattr(self, "_entry_threshold_bonus", 0.0)

        # ── Layer 2: Trend quality — weighted 15m (65%) + 1H (35%) score ───
        q15 = self._tf_quality(candles, trend) if trend else None
        q1h = self._tf_quality(c1h, trend) if trend else None
        l2_score = None
        quality_fallback = False
        if q15 is not None and q1h is not None:
            l2_score = q15["score"] * self.tf_weight_15m + q1h["score"] * self.tf_weight_1h
        elif q15 is not None and self.allow_15m_quality_fallback:
            # Startup-safe mode: use 15m quality until enough 1H bars exist.
            l2_score = q15["score"]
            quality_fallback = True

        # Entry price is the latest 5m close (the TF the cross fires on).
        close_price = c5m[-1].close if c5m else candles[-1].close
        dist_atr = (abs(close_price - l3["dist_ema_val"]) / l3["atr_val"]
                   if (l3 is not None and l3["atr_val"] > 0) else None)

        # Location/structure defaults. With multiple entry engines the actual
        # room-in-R calculation must use each candidate's own stop distance, so
        # the final location gate is evaluated after candidates are built.
        has_long_candidate = trend == "up" and ema_cross_up
        has_short_candidate = trend == "down" and ema_cross_down
        location = self._neutral_location_context()
        sideways = (self._sideways_context(candles) if self.use_sideways_filter
                    else {"is_sideways": False, "signals": 0, "detail": {}})
        regime = self._regime_context(candles, trend, q15, sideways)
        l2_thr = base_l2_thr
        if self.use_regime_router:
            if regime["state"] == "STRONG_TREND":
                l2_thr = max(0.0, l2_thr - self.regime_strong_threshold_discount)
            elif regime["state"] == "TRANSITION":
                l2_thr += self.regime_transition_threshold_penalty
        if quality_fallback:
            l2_thr += self.single_tf_quality_penalty
        location_adjusted_l2_thr = l2_thr
        engine_status: dict = {}
        candidate_reviews: list[dict] = []

        def dbg(entry_status: str) -> dict:
            return {"trend_confirm": {
                "sma_trend": l1["sma_dir"], "ema10_20_trend": l1["ema1020_dir"],
                "ema20_slope": l1["slope_dir"], "macd_trend": l1["macd_dir"],
                "confirmed": trend, "layer1_up_votes": l1.get("up_votes"),
                "layer1_down_votes": l1.get("down_votes"),
                "layer1_required_votes": l1.get("required_votes"),
                "q15": round(q15["score"], 1) if q15 else None,
                "q1h": round(q1h["score"], 1) if q1h else None,
                "q15_breakdown": q15["breakdown"] if q15 else None,
                "q1h_breakdown": q1h["breakdown"] if q1h else None,
                "quality_source": "15m_fallback" if quality_fallback else "15m_plus_1h",
                "layer2_score": round(l2_score, 1) if l2_score is not None else None,
                "layer2_threshold": location_adjusted_l2_thr,
                "base_layer2_threshold": base_l2_thr,
                "regime_adjusted_threshold": l2_thr,
                "regime": regime,
                "location": location,
                "sideways": sideways,
                "data_quality": data_quality,
                "open_position": self._open_position,
                "entry_status": entry_status,
                "entry_engines": engine_status,
                "candidate_reviews": candidate_reviews,
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
                f"Layer1: direction vote not confirmed ({l1.get('up_votes', 0)} up / "
                f"{l1.get('down_votes', 0)} down; need {self.layer1_min_agreement}/4 with EMA alignment)",
                metadata=dbg("no_trend"))

        if q15 is None or l2_score is None:
            return self._hold(current_price, "Layer2: quality indicators still warming up (15m; 1H fallback unavailable)",
                              metadata=dbg("no_trend"))

        if bar_ts_5 is not None and self._last_entry_attempt_bar_ts == bar_ts_5:
            return self._hold(current_price,
                "Layer3: an entry was already attempted on this 5m bar — waiting for the next closed bar",
                metadata=dbg("entry_already_attempted"))

        pending_cross = has_long_candidate or has_short_candidate
        score_note = ((f"(15m={q15['score']:.0f} x{self.tf_weight_15m:.2f} + "
                       f"1H={q1h['score']:.0f} x{self.tf_weight_1h:.2f})")
                      if q1h is not None else f"(15m fallback={q15['score']:.0f})")

        def _spend_cross_if_early() -> bool:
            """An early EMA cross is consumed when Layer2 definitively rejects it."""
            if not (is_early_trend and pending_cross):
                return False
            if trend == "up":
                self._last_ema_cross_up_ts = None
            else:
                self._last_ema_cross_down_ts = None
            return True

        # ── Layer 2 — sideways / range veto (hard) ────────────────────────
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

        # ── Layer 2 — regime gate ─────────────────────────────────────────
        # Optional hard block (off by default):
        if self.require_trend_regime and regime.get("state") not in ("TREND", "STRONG_TREND"):
            _spend_cross_if_early()
            return self._hold(current_price,
                f"Layer2 REGIME veto — need TREND/STRONG_TREND, got {regime.get('state')}",
                metadata=dbg("regime_veto"))

        # TRANSITION confirmation gate. Unlike the previous implementation,
        # this does not inspect the placeholder `location` object before a
        # candidate exists. It validates actual directional development:
        # momentum, volume, EMA compression/expansion and recent price action.
        if regime.get("state") == "TRANSITION":
            trans = self._transition_confirmation_context(candles, trend, q15)
            trans_bar = l2_thr + self.transition_extra_threshold
            fails = []
            if l2_score < trans_bar:
                fails.append(f"quality {l2_score:.0f} < {trans_bar:.0f}")
            if not trans.get("valid"):
                fails.extend(trans.get("fails", []))
            if fails:
                _spend_cross_if_early()
                return self._hold(
                    current_price,
                    "Layer2 TRANSITION confirmation veto: " + "; ".join(fails),
                    metadata=dbg("transition_extra_veto"),
                )

        # ── Layer 2a: base trend quality ──────────────────────────────────
        if l2_score <= l2_thr:
            spent = _spend_cross_if_early()
            tag = "FAIL (early trend, cross spent): " if spent else ""
            return self._hold(current_price,
                f"Layer2 {tag}trend quality {l2_score:.0f} <= {l2_thr:.0f} {score_note}",
                metadata=dbg("early_quality_fail" if spent else "quality_fail"))

        # ── Layer 3: Multi-entry router (5m) ──────────────────────────────
        if l3 is None:
            return self._hold(current_price, "Layer3: 5m indicators still warming up", metadata=dbg("no_trend"))

        dist_ok = dist_atr is not None and dist_atr <= self.max_dist_atr_mult
        breakout_dist_ok = dist_atr is not None and dist_atr <= self.breakout_max_dist_atr_mult
        structure_dist_ok = dist_atr is not None and dist_atr <= self.structure_max_dist_atr_mult
        dist_disp = f"{dist_atr:.2f}" if dist_atr is not None else "n/a"
        direction = "long" if trend == "up" else "short"
        candidates: list[dict] = []

        # Engine A — EMA cross. Preserve the original behavior, but consume a
        # cross that reaches Layer3 and fails the current-bar side/distance gate.
        ema_pending = ema_cross_up if direction == "long" else ema_cross_down
        if not self.use_ema_cross_entry:
            engine_status["EMA_CROSS"] = "disabled"
        elif not ema_pending:
            engine_status["EMA_CROSS"] = "waiting"
        else:
            side_ok = l3["above_ema_ref"] if direction == "long" else l3["below_ema_ref"]
            if not side_ok:
                if direction == "long":
                    self._last_ema_cross_up_ts = None
                else:
                    self._last_ema_cross_down_ts = None
                engine_status["EMA_CROSS"] = f"rejected: price wrong side of EMA{self.entry_ema_ref}"
            elif not dist_ok:
                if direction == "long":
                    self._last_ema_cross_up_ts = None
                else:
                    self._last_ema_cross_down_ts = None
                engine_status["EMA_CROSS"] = f"rejected: {dist_disp}xATR from chase EMA"
            elif not ((direction == "long" and l3["sl_ema_val"] < close_price)
                      or (direction == "short" and l3["sl_ema_val"] > close_price)):
                # Do not mirror an EMA50 that is on the wrong side of entry.
                # Mirroring hid a bad trend/location read and produced a stop
                # that was no longer actually anchored to EMA50.
                if direction == "long":
                    self._last_ema_cross_up_ts = None
                else:
                    self._last_ema_cross_down_ts = None
                engine_status["EMA_CROSS"] = f"rejected: EMA{self.sl_ema_ref} is on wrong side of entry"
            else:
                cross_ts = self._last_ema_cross_up_ts if direction == "long" else self._last_ema_cross_down_ts
                candidates.append({
                    "entry_type": "EMA_CROSS", "direction": direction,
                    "trigger_ts": cross_ts, "raw_stop": l3["sl_ema_val"],
                    "stop_mode": "ema", "base_edge": 68.0, "min_quality": l2_thr,
                    "detail": {"cross_bars_ago": ema_up_ago if direction == "long" else ema_down_ago},
                })
                engine_status["EMA_CROSS"] = "candidate"

        # Engine B — breakout + retest. Direct breakout chasing is deliberately
        # disabled: a breakout must first return to the broken level and reclaim.
        if not self.use_breakout_retest_entry:
            engine_status["BREAKOUT_RETEST"] = "disabled"
        elif not breakout_dist_ok:
            engine_status["BREAKOUT_RETEST"] = (f"rejected: {dist_disp}xATR from chase EMA "
                                                  f"> {self.breakout_max_dist_atr_mult:.2f}")
        elif not (l3["ema_bull_aligned"] if direction == "long" else l3["ema_bear_aligned"]):
            engine_status["BREAKOUT_RETEST"] = "waiting: EMA alignment"
        else:
            bo = self._breakout_retest_setup(c5m, direction, l3["atr_val"])
            last_used = (self._last_breakout_trigger_up_ts if direction == "long"
                         else self._last_breakout_trigger_down_ts)
            if bo is None:
                engine_status["BREAKOUT_RETEST"] = "waiting"
            elif last_used is not None and bo["trigger_ts"] <= last_used:
                engine_status["BREAKOUT_RETEST"] = "trigger already consumed"
            else:
                candidates.append({
                    "entry_type": "BREAKOUT_RETEST", "direction": direction,
                    "trigger_ts": bo["trigger_ts"], "raw_stop": bo["raw_stop"],
                    "stop_mode": "structure", "base_edge": bo["edge_score"],
                    "min_quality": max(l2_thr, self.breakout_entry_min_quality),
                    "detail": bo,
                })
                engine_status["BREAKOUT_RETEST"] = "candidate"

        # Engine C — confirmed market-structure retest. It only works with
        # HH/HL for longs or LH/LL for shorts and needs a reclaim, engulf/pin,
        # or micro-BOS after the level is tested.
        if not self.use_structure_retest_entry:
            engine_status["STRUCTURE_RETEST"] = "disabled"
        elif (self.use_regime_router and regime["state"] == "TRANSITION"
              and not self.allow_structure_entry_in_transition):
            engine_status["STRUCTURE_RETEST"] = "disabled in TRANSITION regime"
        elif not structure_dist_ok:
            engine_status["STRUCTURE_RETEST"] = (f"rejected: {dist_disp}xATR from chase EMA "
                                                   f"> {self.structure_max_dist_atr_mult:.2f}")
        elif not (l3["ema_bull_aligned"] if direction == "long" else l3["ema_bear_aligned"]):
            engine_status["STRUCTURE_RETEST"] = "waiting: EMA alignment"
        else:
            st = self._structure_retest_setup(c5m, direction, l3["atr_val"])
            last_used = (self._last_structure_trigger_up_ts if direction == "long"
                         else self._last_structure_trigger_down_ts)
            if st is None:
                engine_status["STRUCTURE_RETEST"] = "waiting"
            elif last_used is not None and st["trigger_ts"] <= last_used:
                engine_status["STRUCTURE_RETEST"] = "trigger already consumed"
            else:
                candidates.append({
                    "entry_type": "STRUCTURE_RETEST", "direction": direction,
                    "trigger_ts": st["trigger_ts"], "raw_stop": st["raw_stop"],
                    "stop_mode": "structure", "base_edge": st["edge_score"],
                    "min_quality": max(l2_thr, self.structure_entry_min_quality),
                    "detail": st,
                })
                engine_status["STRUCTURE_RETEST"] = "candidate"

        # Engine D — Price Action Market Structure Confirm. This is distinct
        # from STRUCTURE_RETEST: it can enter after a fresh BOS/CHOCH or a
        # liquidity sweep + reclaim, so it does not require a mature HH/HL or
        # LH/LL sequence. It is deliberately stricter in TRANSITION.
        if not self.use_price_action_structure_entry:
            engine_status["PA_STRUCTURE_CONFIRM"] = "disabled"
        elif not structure_dist_ok:
            engine_status["PA_STRUCTURE_CONFIRM"] = (
                f"rejected: {dist_disp}xATR from chase EMA > {self.structure_max_dist_atr_mult:.2f}"
            )
        else:
            pa = self._price_action_structure_setup(c5m, direction, l3["atr_val"], regime.get("state"))
            last_used = (self._last_pa_structure_trigger_up_ts if direction == "long"
                         else self._last_pa_structure_trigger_down_ts)
            if pa is None:
                engine_status["PA_STRUCTURE_CONFIRM"] = "waiting"
            elif last_used is not None and pa["trigger_ts"] <= last_used:
                engine_status["PA_STRUCTURE_CONFIRM"] = "trigger already consumed"
            else:
                candidates.append({
                    "entry_type": "PA_STRUCTURE_CONFIRM", "direction": direction,
                    "trigger_ts": pa["trigger_ts"], "raw_stop": pa["raw_stop"],
                    "stop_mode": "structure", "base_edge": pa["edge_score"],
                    "min_quality": max(l2_thr, self.pa_structure_entry_min_quality),
                    "detail": pa,
                })
                engine_status["PA_STRUCTURE_CONFIRM"] = "candidate"

        if not candidates:
            enabled = [name for name, status in engine_status.items() if status != "disabled"]
            return self._hold(current_price,
                "Layer1+2 passed — waiting for one of: " + ", ".join(enabled),
                metadata=dbg("waiting_entry_router"))

        # Candidate-specific stop, quality, and HTF room validation. This avoids
        # measuring every entry with EMA50 risk when breakout/structure stops are
        # naturally tied to their retest low/high.
        valid_candidates: list[dict] = []
        for candidate in candidates:
            risk_plan = self._compute_entry_sl_tp(
                direction=direction, price=close_price, raw_stop=candidate["raw_stop"],
                atr_val=l3["atr_val"], mirror_raw_stop=candidate["stop_mode"] == "ema",
            )
            review = {"entry_type": candidate["entry_type"]}
            if risk_plan is None:
                review["status"] = "rejected_stop_distance"
                candidate_reviews.append(review)
                continue
            sl, tp, risk_distance, risk_atr = risk_plan

            candidate_location = self._neutral_location_context()
            if self.use_location_filter:
                candidate_location = self._location_context(
                    direction=trend, entry_price=close_price,
                    atr_15m=self._last_atr(candles, self.atr_period) or l3["atr_val"],
                    estimated_risk=risk_distance, candles_15m=candles,
                    candles_1h=c1h, candles_4h=c4h, q15=q15,
                )
            adjusted_thr = candidate["min_quality"] + float(
                candidate_location.get("threshold_penalty",
                                       self.location_threshold_penalty if candidate_location.get("penalize") else 0.0)
            )
            review.update({
                "min_quality": round(adjusted_thr, 1),
                "risk_atr": round(risk_atr, 2),
                "location_valid": candidate_location["valid"],
                "location_reason": candidate_location["reason"],
                "structure_1h": candidate_location.get("structure_1h"),
                "structure_4h": candidate_location.get("structure_4h"),
                "macro_alignment": candidate_location.get("macro_alignment"),
                "supportive_zone": candidate_location.get("supportive_zone"),
            })
            if not candidate_location["valid"]:
                review["status"] = "rejected_location"
                candidate_reviews.append(review)
                continue
            if l2_score <= adjusted_thr:
                review["status"] = "rejected_quality"
                candidate_reviews.append(review)
                continue

            room_r = candidate_location.get("structure_room_r")
            edge_score = candidate["base_edge"] + min(8.0, max(0.0, (l2_score - adjusted_thr) * 0.35))
            if room_r is not None and room_r >= self.preferred_structure_room_r:
                edge_score += min(4.0, room_r)
            if candidate_location.get("macro_alignment") == "ALIGNED":
                edge_score += self.htf_alignment_edge_bonus
            supportive_zone = candidate_location.get("supportive_zone")
            if supportive_zone is not None:
                edge_score += float(supportive_zone.get("freshness", 0.0)) * self.sd_supportive_zone_edge_bonus
                if supportive_zone.get("role_reversal"):
                    edge_score += self.sd_role_reversal_edge_bonus
            if regime.get("state") == "STRONG_TREND":
                edge_score += 1.0
            candidate.update({
                "sl": sl, "tp": tp, "risk_distance": risk_distance,
                "risk_atr": risk_atr, "location": candidate_location,
                "adjusted_threshold": adjusted_thr, "edge_score": round(edge_score, 2),
            })
            review.update({"status": "valid", "edge_score": round(edge_score, 2)})
            candidate_reviews.append(review)
            valid_candidates.append(candidate)

        if not valid_candidates:
            reasons = ", ".join(f"{r['entry_type']}={r['status']}" for r in candidate_reviews)
            return self._hold(current_price,
                f"Layer3 candidates rejected after risk/location checks: {reasons}",
                metadata=dbg("candidate_rejected"))

        selected = max(valid_candidates, key=lambda x: x["edge_score"])
        location = selected["location"]
        location_adjusted_l2_thr = selected["adjusted_threshold"]
        entry_type = selected["entry_type"]
        sl, tp = selected["sl"], selected["tp"]

        # Consume the selected trigger and block duplicate attempts on this bar.
        self._last_entry_attempt_bar_ts = bar_ts_5
        if entry_type == "EMA_CROSS":
            if direction == "long":
                self._last_ema_cross_up_ts = None
            else:
                self._last_ema_cross_down_ts = None
        elif entry_type == "BREAKOUT_RETEST":
            if direction == "long":
                self._last_breakout_trigger_up_ts = selected["trigger_ts"]
            else:
                self._last_breakout_trigger_down_ts = selected["trigger_ts"]
        elif entry_type == "STRUCTURE_RETEST":
            if direction == "long":
                self._last_structure_trigger_up_ts = selected["trigger_ts"]
            else:
                self._last_structure_trigger_down_ts = selected["trigger_ts"]
        elif entry_type == "PA_STRUCTURE_CONFIRM":
            if direction == "long":
                self._last_pa_structure_trigger_up_ts = selected["trigger_ts"]
            else:
                self._last_pa_structure_trigger_down_ts = selected["trigger_ts"]

        self._open_position = direction
        self._entry_price, self._entry_sl, self._tp1_done = close_price, sl, False
        self._be_trailed = False
        meta = dbg("entered")
        meta.update({
            "entry_type": entry_type, "entry_detail": selected["detail"],
            "entry_edge_score": selected["edge_score"],
            "stop_loss": round(sl, 8),
            # No hard TP in the pure cross-back system — ride until EMA8/13
            # crosses back. Only the SL (EMA50) is placed on the exchange.
            "take_profit": round(tp, 8) if self.use_hard_tp else None,
            "risk_atr": round(selected["risk_atr"], 3), "rr_ratio": self.rr_ratio,
            "sizing_mode": self.sizing_mode, "margin_pct": self.margin_pct,
            "structure_room_r": location.get("structure_room_r"),
            "nearest_opposing_zone": location.get("nearest_opposing_zone"),
            "supportive_zone": location.get("supportive_zone"),
            "structure_1h": location.get("structure_1h"),
            "structure_4h": location.get("structure_4h"),
            "macro_alignment": location.get("macro_alignment"),
            "regime": regime,
        })
        side_label = "Uptrend" if direction == "long" else "Downtrend"
        trigger_summary = self._entry_reason_summary(selected)
        return Signal(
            type=SignalType.BUY if direction == "long" else SignalType.SELL,
            symbol=self.symbol, price=current_price, amount=0.0,
            reason=f"{side_label} confirmed (Layer1 30m) + quality {l2_score:.0f}"
                   f"{' [early]' if is_early_trend else ''} >{location_adjusted_l2_thr:.0f} + "
                   f"location OK (Layer2) + {entry_type}: {trigger_summary} (Layer3 5m)",
            confidence=1.0,
            metadata=meta,
        )

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Position management, evaluated every tick:

        1. TP1 partial (price-based, checked every tick): when price reaches
           tp1_r (0.75R, halfway to the 1.5R final TP), close tp1_close_pct
           (50%) and move SL to break-even + be_offset_r (BE + 0.1R). Once.
        2. Exit the runner (5m bar-based): EMA10/20 cross-back (a genuine trend
           reversal) OR a 5m close past EMA20 (long: close below EMA20; short:
           close above EMA20). The final TP (1.5R) and the trailed SL are the
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

        # ── 1b) Break-even TRAIL (no close): at +be_trail_trigger_r, ratchet SL
        #        up to entry +/- be_trail_sl_r and keep the full position. ──────
        if (self.use_be_trail and not self._be_trailed
                and self._entry_price is not None and self._entry_sl is not None):
            r = abs(self._entry_price - self._entry_sl)
            if r > 0:
                hit_long = (self._open_position == "long"
                            and current_price >= self._entry_price + self.be_trail_trigger_r * r)
                hit_short = (self._open_position == "short"
                             and current_price <= self._entry_price - self.be_trail_trigger_r * r)
                if hit_long or hit_short:
                    self._be_trailed = True
                    new_sl = (self._entry_price + self.be_trail_sl_r * r if hit_long
                              else self._entry_price - self.be_trail_sl_r * r)
                    return PositionUpdate(
                        action="move_sl", new_sl=new_sl,
                        reason=f"Target +{self.be_trail_trigger_r:.1f}R hit — SL -> "
                               f"BE+{self.be_trail_sl_r:.1f}R (locked), riding to cross-back")

        # ── 2) Runner exit — EMA10/20 cross-back OR 5m close past EMA20 ──────
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

        pos = self._open_position
        cb = self.exit_close_confirm_bars

        # (a) TREND-FLIP exit — the trend we entered on (trend_tf/1h Layer1) has
        #     actually reversed. This is the primary "the trend changed" exit.
        if self.exit_on_trend_flip and self._trend_state is not None:
            flipped = ((pos == "long" and self._trend_state == "down")
                       or (pos == "short" and self._trend_state == "up"))
            if flipped:
                self._reset_position_state()
                return PositionUpdate(action="close", close_pct=1.0,
                    reason=f"Exit {pos.upper()}: {self.trend_tf} trend flipped to {self._trend_state.upper()}")

        # (b) STRUCTURAL exit — price closed past EMA50 (the trend structure), a
        #     real break, not a fast EMA8/13 wiggle.
        if self.use_structural_exit and self._closes_past_ema(
                candles, pos, self.exit_structural_confirm_bars, self.sl_ema_ref):
            self._reset_position_state()
            return PositionUpdate(action="close", close_pct=1.0,
                reason=f"Exit {pos.upper()}: closed past EMA{self.sl_ema_ref} structure ({self.entry_tf})")

        # (c) Fast EMA8/13 cross-back exit — ARMED once we're in profit or past
        #     the +0.8R target (be_trailed). Before that it stays sticky (only
        #     the trend-flip / EMA50 exits above), so a losing wiggle can't churn
        #     us out; but once green, an EMA reversal locks the gain immediately.
        in_profit = ((pos == "long" and current_price > self._entry_price)
                     or (pos == "short" and current_price < self._entry_price)) \
                    if self._entry_price is not None else False
        crossback_armed = self.use_ema_crossback_exit or (
            self.crossback_exit_after_target and (self._be_trailed or in_profit))
        if crossback_armed:
            xback = (pos == "long" and l3["ema_cross_down"]) or (pos == "short" and l3["ema_cross_up"])
            close_past = self.use_close_past_exit and self._closes_past_ema_slow(candles, pos, cb)
            if xback or close_past:
                self._reset_position_state()
                tag = "in profit" if in_profit else "post-0.8R"
                return PositionUpdate(action="close", close_pct=1.0,
                    reason=f"Exit {pos.upper()}: EMA{self.ema_fast}/{self.ema_slow} cross-back ({tag}, {self.entry_tf})")

        return PositionUpdate(action="hold",
                              reason=f"Holding {pos.upper()} — {self.trend_tf} trend intact")

    def _closes_past_ema(self, candles: list, side: str, n: int, period: int) -> bool:
        """True if the last `n` bars ALL closed on the wrong side of EMA{period}
        for the position (long: below, short: above). Used for the structural
        EMA50 exit — a genuine break of the trend structure."""
        closes = [c.close for c in candles]
        ema_p = self.ema(closes, period)
        if len(closes) < n:
            return False
        for k in range(1, n + 1):
            if np.isnan(ema_p[-k]):
                return False
            if side == "long" and not (closes[-k] < ema_p[-k]):
                return False
            if side == "short" and not (closes[-k] > ema_p[-k]):
                return False
        return True

    def _closes_past_ema_slow(self, candles: list, side: str, n: int) -> bool:
        """True if the last `n` 5m bars ALL closed on the wrong side of EMA20 (ema_slow)
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
        self._be_trailed = False

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

    def _data_quality_context(self, candles: list, expected_ms: int) -> dict:
        """Validate the newest candle window without rejecting normal weekend
        or session gaps. Gaps are reported for diagnostics; malformed OHLC,
        duplicate/non-monotonic timestamps and non-finite data are hard fails."""
        if not candles:
            return {"valid": False, "reason": "missing candle series", "bars": 0}

        window = candles[-300:]
        previous_ts: Optional[int] = None
        max_gap_mult = 0.0
        for i, candle in enumerate(window):
            try:
                ts = int(candle.timestamp)
                values = [float(candle.open), float(candle.high), float(candle.low),
                          float(candle.close), float(candle.volume)]
            except (AttributeError, TypeError, ValueError, OverflowError):
                return {"valid": False, "reason": f"invalid candle fields at index {i}",
                        "bars": len(candles)}
            if not all(math.isfinite(v) for v in values):
                return {"valid": False, "reason": f"non-finite OHLCV at index {i}",
                        "bars": len(candles)}
            o, h, l, c, volume = values
            if h < l or h < max(o, c) or l > min(o, c):
                return {"valid": False, "reason": f"invalid OHLC geometry at index {i}",
                        "bars": len(candles)}
            if volume < 0:
                return {"valid": False, "reason": f"negative volume at index {i}",
                        "bars": len(candles)}
            if previous_ts is not None:
                if ts <= previous_ts:
                    return {"valid": False, "reason": "duplicate or non-monotonic timestamps",
                            "bars": len(candles)}
                if expected_ms > 0:
                    max_gap_mult = max(max_gap_mult, (ts - previous_ts) / expected_ms)
            previous_ts = ts

        gap_warning = max_gap_mult > self.max_recent_gap_mult
        return {
            "valid": True,
            "reason": "ok_with_session_gap" if gap_warning else "ok",
            "bars": len(candles),
            "max_gap_mult": round(max_gap_mult, 2),
            "gap_warning": gap_warning,
        }

    def _regime_context(self, candles: list, trend: Optional[str], q15: Optional[dict],
                        sideways: dict) -> dict:
        if sideways.get("is_sideways"):
            return {"state": "CHOP", "reason": "sideways veto", "ema_gap_atr": None}
        if not self.use_regime_router:
            return {"state": "TREND", "reason": "router disabled", "ema_gap_atr": None}
        if not trend or q15 is None or len(candles) < self.quality_ema_slow + 2:
            return {"state": "UNKNOWN", "reason": "warming up", "ema_gap_atr": None}

        closes = [c.close for c in candles]
        ema_fast = self.ema(closes, self.quality_ema_fast)
        ema_slow = self.ema(closes, self.quality_ema_slow)
        atr_val = self._last_atr(candles, self.atr_period)
        if atr_val is None or np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
            return {"state": "UNKNOWN", "reason": "indicators unavailable", "ema_gap_atr": None}

        up = trend == "up"
        aligned = ((closes[-1] > ema_fast[-1] > ema_slow[-1]) if up
                   else (closes[-1] < ema_fast[-1] < ema_slow[-1]))
        gap_atr = abs(float(ema_fast[-1]) - float(ema_slow[-1])) / atr_val
        breakdown = q15.get("breakdown", {})
        adx_val = float(breakdown.get("adx_val", 0.0) or 0.0)
        chop_val = float(breakdown.get("chop_val", 100.0) or 100.0)

        if (aligned and adx_val >= self.regime_strong_adx
                and chop_val <= self.regime_strong_chop_max
                and gap_atr >= self.regime_min_ema_gap_atr):
            state = "STRONG_TREND"
        elif aligned and adx_val >= self.regime_trend_adx and chop_val <= self.regime_trend_chop_max:
            state = "TREND"
        else:
            state = "TRANSITION"
        return {
            "state": state, "aligned": aligned, "adx": round(adx_val, 1),
            "chop": round(chop_val, 1), "ema_gap_atr": round(gap_atr, 2),
            "reason": f"aligned={aligned}, ADX={adx_val:.1f}, CHOP={chop_val:.1f}",
        }

    def _neutral_location_context(self) -> dict:
        return {
            "valid": True, "penalize": False, "threshold_penalty": 0.0,
            "reason": "filter_off_or_no_context", "structure_room_r": None,
            "nearest_opposing_zone": None, "supportive_zone": None,
            "location_type": "UNKNOWN", "structure_1h": "UNKNOWN",
            "structure_4h": "UNKNOWN", "macro_alignment": "UNKNOWN",
            "bearish_sweep": False, "bullish_sweep": False,
            "active_zone_count": 0,
        }

    def _location_context(self, direction: str, entry_price: float, atr_15m: float,
                          estimated_risk: float, candles_15m: list, candles_1h: list,
                          candles_4h: list, q15: Optional[dict] = None) -> dict:
        """HTF context gate using confirmed 1H/4H swing structure, active
        pivots and systematic supply/demand zones.

        `direction` is Layer-1's "up" or "down". Opposing zones define room
        to target; same-side zones are confluence, not mandatory. 4H/1H
        confirmed opposite structure is a hard conflict. Transition structure,
        mid-range chop or limited room raises the threshold instead of blindly
        deleting every setup.
        """
        if atr_15m <= 0 or estimated_risk <= 0:
            return self._neutral_location_context()

        p1h = self._confirmed_pivots(candles_1h, self.structure_pivot_left, self.structure_pivot_right)
        p4h = self._confirmed_pivots(candles_4h, self.structure_pivot_left, self.structure_pivot_right)
        structure_1h = self._swing_structure(p1h)
        structure_4h = self._swing_structure(p4h)
        wanted_structure = "bull" if direction == "up" else "bear"
        opposite_structure = "bear" if direction == "up" else "bull"

        context_penalty = 0.0
        context_reasons: list[str] = []
        one_h_conflict = structure_1h == opposite_structure
        four_h_conflict = self.use_htf_macro_filter and structure_4h == opposite_structure
        # A single HTF disagreement is common near reversals and should not
        # silence the bot. Only a confirmed conflict on BOTH 1H and 4H is a
        # hard veto; one conflicting timeframe raises the required quality.
        if one_h_conflict and four_h_conflict:
            return {**self._neutral_location_context(), "valid": False,
                    "reason": f"1H and 4H both confirmed {opposite_structure.upper()} structure",
                    "structure_1h": structure_1h, "structure_4h": structure_4h,
                    "macro_alignment": "CONFLICT"}
        if one_h_conflict:
            context_penalty += self.htf_single_conflict_penalty
            context_reasons.append(f"1H {opposite_structure} structure conflict")
        if four_h_conflict:
            context_penalty += self.htf_single_conflict_penalty
            context_reasons.append(f"4H {opposite_structure} structure conflict")

        levels: list[dict] = []
        for tf, pivots, bars, mult in (
            ("1h", p1h, candles_1h, self.zone_width_atr_1h),
            ("4h", p4h, candles_4h, self.zone_width_atr_4h),
        ):
            tf_atr = self._last_atr(bars, self.atr_period)
            width = max((tf_atr or atr_15m) * mult, atr_15m * 0.15)
            wanted_pivot = "high" if direction == "up" else "low"
            for pivot in pivots:
                if pivot["type"] != wanted_pivot:
                    continue
                if not self._pivot_still_active(bars, pivot, direction, width):
                    continue
                levels.append({
                    "kind": "pivot", "timeframe": tf, "price": pivot["price"],
                    "width": width, "lower": pivot["price"] - width,
                    "upper": pivot["price"] + width, "freshness": None,
                    "role_reversal": False,
                })

        zones = self._get_htf_zones(candles_1h, candles_4h) if self.use_supply_demand_zones else []
        active_zones = [z for z in zones if z.get("active") and z.get("freshness", 0.0) >= self.sd_min_freshness]
        opposing_type = "supply" if direction == "up" else "demand"
        supportive_type = "demand" if direction == "up" else "supply"

        for zone in active_zones:
            if zone["zone_type"] != opposing_type:
                continue
            proximal = zone["lower"] if direction == "up" else zone["upper"]
            levels.append({
                **zone, "kind": "supply_demand", "price": proximal,
                "width": max(zone["upper"] - zone["lower"], atr_15m * 0.05),
            })

        def room_to(level: dict) -> float:
            if direction == "up":
                return float(level["lower"]) - entry_price
            return entry_price - float(level["upper"])

        if direction == "up":
            eligible = [z for z in levels if z["upper"] > entry_price]
        else:
            eligible = [z for z in levels if z["lower"] < entry_price]
        nearest = min(eligible, key=room_to, default=None)
        room = room_to(nearest) if nearest is not None else None
        room_r = room / estimated_risk if room is not None else None
        zone_distance_atr = room / atr_15m if room is not None else None

        support_tolerance = self.sd_supportive_zone_distance_atr * atr_15m
        supportive_candidates: list[tuple[float, dict]] = []
        for zone in active_zones:
            if zone["zone_type"] != supportive_type:
                continue
            if direction == "up":
                if entry_price < zone["lower"] - support_tolerance:
                    continue
                distance = max(0.0, entry_price - zone["upper"])
            else:
                if entry_price > zone["upper"] + support_tolerance:
                    continue
                distance = max(0.0, zone["lower"] - entry_price)
            if distance <= support_tolerance:
                supportive_candidates.append((distance, zone))
        supportive = min(supportive_candidates, key=lambda x: x[0], default=(None, None))[1]

        sweep = self._wrong_side_liquidity_sweep(candles_15m, direction, nearest)
        if sweep:
            return {**self._neutral_location_context(), "valid": False,
                    "reason": "wrong-side liquidity sweep/rejection at opposing HTF level",
                    "structure_1h": structure_1h, "structure_4h": structure_4h,
                    "macro_alignment": "ALIGNED" if structure_4h == wanted_structure else "NEUTRAL",
                    "nearest_opposing_zone": nearest, "supportive_zone": supportive,
                    "structure_room_r": round(room_r, 2) if room_r is not None else None,
                    "active_zone_count": len(active_zones),
                    "bearish_sweep": direction == "up", "bullish_sweep": direction == "down"}

        if (nearest is not None and zone_distance_atr is not None
                and zone_distance_atr < self.hard_zone_distance_atr
                and (room_r is None or room_r < self.min_structure_room_r)):
            source = nearest.get("timeframe", "HTF")
            kind = nearest.get("kind", "zone")
            return {**self._neutral_location_context(), "valid": False,
                    "reason": f"entry directly into {source} opposing {kind}",
                    "structure_1h": structure_1h, "structure_4h": structure_4h,
                    "macro_alignment": "ALIGNED" if structure_4h == wanted_structure else "NEUTRAL",
                    "nearest_opposing_zone": nearest, "supportive_zone": supportive,
                    "structure_room_r": round(room_r, 2) if room_r is not None else None,
                    "active_zone_count": len(active_zones)}

        if room_r is not None and room_r < self.min_structure_room_r:
            return {**self._neutral_location_context(), "valid": False,
                    "reason": f"structure room {room_r:.2f}R below {self.min_structure_room_r:.2f}R",
                    "structure_1h": structure_1h, "structure_4h": structure_4h,
                    "macro_alignment": "ALIGNED" if structure_4h == wanted_structure else "NEUTRAL",
                    "nearest_opposing_zone": nearest, "supportive_zone": supportive,
                    "structure_room_r": round(room_r, 2), "active_zone_count": len(active_zones)}

        midrange = self._is_midrange(candles_1h, entry_price)
        chop_val = (q15 or {}).get("breakdown", {}).get("chop_val")
        choppy_midrange = bool(self.reject_midrange_when_choppy and midrange and
                               chop_val is not None and chop_val >= self.chop_threshold - 3.0)

        penalty = context_penalty
        reasons: list[str] = list(context_reasons)
        if choppy_midrange:
            penalty += self.location_threshold_penalty
            reasons.append("mid-range in choppy conditions")
        if room_r is not None and room_r < self.preferred_structure_room_r:
            penalty += self.location_threshold_penalty
            reasons.append(f"limited room {room_r:.2f}R")
        if structure_4h == "transition":
            penalty += self.htf_transition_threshold_penalty
            reasons.append("4H structure transition")
        penalty = min(penalty, self.max_location_threshold_penalty)

        macro_alignment = ("ALIGNED" if structure_4h == wanted_structure
                           else "TRANSITION" if structure_4h == "transition" else "NEUTRAL")
        location_type = "MID_RANGE" if midrange else "EDGE_OR_TREND_LOCATION"
        if supportive is not None:
            location_type = "SUPPORTIVE_HTF_ZONE"
            label = supportive.get("pattern", supportive.get("zone_type", "zone"))
            reasons.append(f"near {supportive.get('timeframe')} {label}")

        return {
            "valid": True, "penalize": penalty > 0, "threshold_penalty": round(penalty, 2),
            "reason": "; ".join(reasons) if reasons else "acceptable HTF location",
            "structure_room_r": round(room_r, 2) if room_r is not None else None,
            "nearest_opposing_zone": nearest, "supportive_zone": supportive,
            "location_type": location_type, "structure_1h": structure_1h,
            "structure_4h": structure_4h, "macro_alignment": macro_alignment,
            "bearish_sweep": False, "bullish_sweep": False,
            "active_zone_count": len(active_zones),
        }

    def _get_htf_zones(self, candles_1h: list, candles_4h: list) -> list[dict]:
        key = (
            candles_1h[-1].timestamp if candles_1h else None, len(candles_1h),
            candles_4h[-1].timestamp if candles_4h else None, len(candles_4h),
        )
        if key == self._zone_cache_key:
            return self._zone_cache
        zones = self._supply_demand_zones(candles_1h, "1h")
        zones.extend(self._supply_demand_zones(candles_4h, "4h"))
        self._zone_cache_key = key
        self._zone_cache = zones
        return zones

    def _supply_demand_zones(self, candles: list, timeframe: str) -> list[dict]:
        """Detect RBR/DBR demand and RBD/DBD supply from a compact base and
        ATR-qualified departure. This is intentionally numeric and conservative;
        it does not attempt visual discretionary zone drawing."""
        minimum = self.atr_period + self.sd_base_max_bars + self.sd_departure_bars + 5
        if len(candles) < minimum:
            return []
        atr_arr = self.atr(candles, self.atr_period)
        n = len(candles)
        scan_start = max(self.atr_period + self.sd_base_max_bars,
                         n - self.sd_scan_lookback_bars)
        last_base_end = n - self.sd_departure_bars - 1
        detected: list[dict] = []

        for base_end in range(scan_start, last_base_end + 1):
            atr_ref = float(atr_arr[base_end]) if not np.isnan(atr_arr[base_end]) else 0.0
            if atr_ref <= 0:
                continue
            best: Optional[dict] = None
            for base_len in range(1, self.sd_base_max_bars + 1):
                base_start = base_end - base_len + 1
                if base_start < 1:
                    continue
                base = candles[base_start:base_end + 1]
                base_high = max(c.high for c in base)
                base_low = min(c.low for c in base)
                base_range_atr = (base_high - base_low) / atr_ref
                avg_body_atr = float(np.mean([abs(c.close - c.open) for c in base])) / atr_ref
                if (base_range_atr > self.sd_base_max_range_atr
                        or avg_body_atr > self.sd_base_max_body_atr):
                    continue

                departure = candles[base_end + 1:base_end + 1 + self.sd_departure_bars]
                if len(departure) < self.sd_departure_bars:
                    continue
                rally_extent = (max(c.close for c in departure) - base_high) / atr_ref
                drop_extent = (base_low - min(c.close for c in departure)) / atr_ref
                departure_type: Optional[str] = None
                departure_atr = 0.0
                if rally_extent >= self.sd_min_departure_atr and rally_extent > drop_extent:
                    departure_type, departure_atr = "demand", rally_extent
                elif drop_extent >= self.sd_min_departure_atr:
                    departure_type, departure_atr = "supply", drop_extent
                if departure_type is None:
                    continue

                displacement_bodies = [abs(c.close - c.open) / atr_ref for c in departure]
                if max(displacement_bodies, default=0.0) < 0.35:
                    continue
                pre_start = max(0, base_start - 3)
                pre_delta = candles[base_start].open - candles[pre_start].close
                if departure_type == "demand":
                    pattern = "DBR" if pre_delta < -0.20 * atr_ref else "RBR"
                    lower = float(base_low)
                    upper = float(max(max(c.open, c.close) for c in base))
                else:
                    pattern = "RBD" if pre_delta > 0.20 * atr_ref else "DBD"
                    lower = float(min(min(c.open, c.close) for c in base))
                    upper = float(base_high)
                if upper <= lower:
                    lower, upper = float(base_low), float(base_high)
                width_atr = (upper - lower) / atr_ref
                if width_atr <= 0.02 or width_atr > 1.25:
                    continue

                candidate = {
                    "zone_id": f"{timeframe}:{candles[base_end].timestamp}:{departure_type}",
                    "timeframe": timeframe, "zone_type": departure_type,
                    "original_zone_type": departure_type, "pattern": pattern,
                    "lower": lower, "upper": upper,
                    "created_ts": candles[base_end].timestamp,
                    "departure_end_ts": departure[-1].timestamp,
                    "departure_atr": round(float(departure_atr), 3),
                    "base_bars": base_len, "base_range_atr": round(base_range_atr, 3),
                    "avg_base_body_atr": round(avg_body_atr, 3),
                    "quality": round(min(1.0, departure_atr / max(1.0, self.sd_min_departure_atr * 2.0)), 3),
                    "_departure_end_index": base_end + self.sd_departure_bars,
                    "_atr_ref": atr_ref,
                }
                if best is None or (candidate["quality"], -candidate["base_range_atr"]) > (
                        best["quality"], -best["base_range_atr"]):
                    best = candidate
            if best is not None:
                lifecycle = self._zone_lifecycle(candles, best)
                best.update(lifecycle)
                best.pop("_departure_end_index", None)
                best.pop("_atr_ref", None)
                if best.get("active"):
                    detected.append(best)

        return self._dedupe_zones(detected)

    def _zone_lifecycle(self, candles: list, zone: dict) -> dict:
        lower, upper = float(zone["lower"]), float(zone["upper"])
        atr_ref = float(zone["_atr_ref"])
        start = int(zone["_departure_end_index"]) + 1
        tolerance = self.sd_touch_tolerance_atr * atr_ref
        break_buffer = self.sd_break_buffer_atr * atr_ref
        original_type = zone["zone_type"]

        touch_count = 0
        in_touch = False
        break_idx: Optional[int] = None
        for i in range(start, len(candles)):
            candle = candles[i]
            intersects = candle.high >= lower - tolerance and candle.low <= upper + tolerance
            if intersects and not in_touch:
                touch_count += 1
            in_touch = intersects
            broken = ((original_type == "demand" and candle.close < lower - break_buffer)
                      or (original_type == "supply" and candle.close > upper + break_buffer))
            if broken:
                break_idx = i
                break

        role_reversal = break_idx is not None
        invalidated = False
        if role_reversal:
            flipped_type = "supply" if original_type == "demand" else "demand"
            touch_count = 0
            in_touch = False
            for candle in candles[break_idx + 1:]:
                intersects = candle.high >= lower - tolerance and candle.low <= upper + tolerance
                if intersects and not in_touch:
                    touch_count += 1
                in_touch = intersects
                invalidated = ((flipped_type == "demand" and candle.close < lower - break_buffer)
                               or (flipped_type == "supply" and candle.close > upper + break_buffer))
                if invalidated:
                    break
            active_type = flipped_type
            base_freshness = 0.75
        else:
            active_type = original_type
            base_freshness = 1.0

        freshness = max(0.0, base_freshness - touch_count * self.sd_touch_freshness_decay)
        active = (not invalidated and touch_count <= self.sd_max_touches
                  and freshness >= self.sd_min_freshness)
        return {
            "zone_type": active_type, "active": active,
            "broken_original": role_reversal, "role_reversal": role_reversal,
            "touch_count": touch_count, "freshness": round(freshness, 3),
            "invalidated": invalidated,
            "break_ts": candles[break_idx].timestamp if break_idx is not None else None,
        }

    @staticmethod
    def _dedupe_zones(zones: list[dict]) -> list[dict]:
        """Keep the strongest representative when same-type zones overlap."""
        ranked = sorted(
            zones,
            key=lambda z: (z.get("freshness", 0.0) * z.get("quality", 0.0),
                           z.get("created_ts", 0)),
            reverse=True,
        )
        kept: list[dict] = []
        for zone in ranked:
            width = max(zone["upper"] - zone["lower"], 1e-12)
            duplicate = False
            for existing in kept:
                if zone["timeframe"] != existing["timeframe"] or zone["zone_type"] != existing["zone_type"]:
                    continue
                overlap = max(0.0, min(zone["upper"], existing["upper"]) -
                              max(zone["lower"], existing["lower"]))
                denominator = min(width, max(existing["upper"] - existing["lower"], 1e-12))
                if overlap / denominator >= 0.60:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(zone)
        kept.sort(key=lambda z: z.get("created_ts", 0), reverse=True)
        return kept[:16]

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
        """Detect rejection after probing an opposing pivot/zone.

        For a long, price must actually reach the opposing zone's lower edge
        (not merely come within a full zone width) and close back below it with
        a meaningful upper wick. Shorts mirror the rule at the upper edge.
        """
        if nearest is None or len(candles_15m) < 3:
            return False
        level = float(nearest.get("price", 0.0))
        width = max(float(nearest.get("width", 0.0)), 1e-12)
        lower = float(nearest.get("lower", level - width))
        upper = float(nearest.get("upper", level + width))
        rejection_buffer = width * 0.05

        def is_sweep(c) -> bool:
            body = abs(c.close - c.open)
            rng = max(c.high - c.low, 1e-12)
            if direction == "up":
                # Probe resistance/supply then close back below its proximal edge.
                return (c.high >= lower and c.close < lower - rejection_buffer
                        and (c.high - max(c.open, c.close)) >= max(body * 1.2, rng * 0.20)
                        and body / rng >= 0.10)
            # Probe support/demand then close back above its proximal edge.
            return (c.low <= upper and c.close > upper + rejection_buffer
                    and (min(c.open, c.close) - c.low) >= max(body * 1.2, rng * 0.20)
                    and body / rng >= 0.10)

        last = candles_15m[-1]
        for idx in (-1, -2):
            candle = candles_15m[idx]
            if not is_sweep(candle):
                continue
            if idx == -2:
                reclaimed = (last.close > upper if direction == "up" else last.close < lower)
                if reclaimed:
                    continue
            return True
        return False

    def _transition_confirmation_context(self, candles: list, trend: str,
                                         q15: Optional[dict]) -> dict:
        """Directional confirmation used only in TRANSITION.

        Passes when at least three of five independent observations support the
        Layer-1 direction, while momentum and volume retain minimum floors.
        This prevents both extremes: trading every transition and blocking all
        early trends because ADX has not caught up yet.
        """
        if not candles or q15 is None or len(candles) < max(30, self.quality_ema_slow + 3):
            return {"valid": False, "fails": ["transition indicators warming up"], "score": 0}
        closes = [c.close for c in candles]
        fast = self.ema(closes, self.quality_ema_fast)
        slow = self.ema(closes, self.quality_ema_slow)
        macd = self.ema(closes, self.macd_fast) - self.ema(closes, self.macd_slow)
        macd_sig = self.ema(macd.tolist(), self.macd_signal)
        up = trend == "up"
        last = candles[-1]
        prev = candles[-2]
        atr = self._last_atr(candles, self.atr_period) or 0.0
        qb = q15.get("breakdown", {})
        vol_ratio = float(qb.get("vol_ratio", 0.0) or 0.0)
        momentum_points = float(qb.get("momentum", 0.0) or 0.0)
        checks = {
            "price_fast_side": last.close > fast[-1] if up else last.close < fast[-1],
            "fast_slow_alignment": fast[-1] > slow[-1] if up else fast[-1] < slow[-1],
            "fast_slope": fast[-1] > fast[-3] if up else fast[-1] < fast[-3],
            "macd_hist_direction": (macd[-1] - macd_sig[-1]) > 0 if up else (macd[-1] - macd_sig[-1]) < 0,
            "directional_close": (last.close > prev.high if up else last.close < prev.low)
                                 or (last.close > last.open if up else last.close < last.open),
        }
        votes = sum(bool(v) for v in checks.values())
        fails = []
        if votes < 3:
            fails.append(f"directional confirmation {votes}/5 < 3/5")
        if vol_ratio < self.transition_min_vol_ratio:
            fails.append(f"volume {vol_ratio:.2f}x < {self.transition_min_vol_ratio:.2f}x")
        min_mom = self.momentum_weight * self.transition_momentum_frac
        if self.transition_require_momentum and momentum_points < min_mom:
            fails.append(f"momentum {momentum_points:.1f} < {min_mom:.1f}")
        # Reject a large impulse candle that is already too extended; the entry
        # router can wait for a retest instead of chasing it.
        if atr > 0 and abs(last.close - last.open) / atr > 1.6:
            fails.append("transition impulse candle > 1.6 ATR; wait for retest")
        return {"valid": not fails, "fails": fails, "votes": votes, "checks": checks}

    def _price_action_structure_setup(self, candles: list, direction: str,
                                      atr_val: float, regime_state: Optional[str]) -> Optional[dict]:
        """Fresh price-action structure entry using BOS/CHOCH or sweep/reclaim.

        The setup requires displacement, a quality close, acceptable volume and
        a micro break in the trend direction. In TRANSITION it additionally
        requires both a structural event and stronger confirmation, preventing
        ordinary inside-bar noise from becoming an entry.
        """
        if atr_val <= 0 or len(candles) < self.pa_structure_lookback + 4:
            return None
        n = len(candles)
        start = max(2, n - self.entry_trigger_valid_bars)
        for idx in range(n - 1, start - 1, -1):
            c = candles[idx]
            prev = candles[idx - 1]
            hist_start = max(0, idx - self.pa_structure_lookback)
            history = candles[hist_start:idx]
            if len(history) < self.pa_structure_swing_lookback + 2:
                continue
            rng = max(c.high - c.low, 1e-12)
            body_atr = abs(c.close - c.open) / atr_val
            close_quality = ((c.close - c.low) / rng if direction == "long"
                             else (c.high - c.close) / rng)
            vol_hist = [x.volume for x in history[-self.volume_sma_period:]]
            vol_base = float(np.mean(vol_hist)) if vol_hist else 0.0
            vol_ratio = c.volume / vol_base if vol_base > 0 else 0.0

            recent = history[-self.pa_structure_swing_lookback:]
            swing_high = max(x.high for x in recent)
            swing_low = min(x.low for x in recent)
            prior_high = max(x.high for x in history[:-1])
            prior_low = min(x.low for x in history[:-1])
            buffer = self.pa_break_buffer_atr * atr_val
            tol = self.pa_sweep_tolerance_atr * atr_val

            if direction == "long":
                bos = c.close > swing_high + buffer and prev.close <= swing_high + buffer
                sweep = c.low <= prior_low + tol and c.close > prior_low and c.close > c.open
                micro_break = c.close > max(prev.high, candles[idx - 2].high)
                choch = sweep and micro_break
                directional = c.close > c.open
                raw_stop = min(c.low, prev.low, prior_low) - self.entry_stop_buffer_atr * atr_val
            else:
                bos = c.close < swing_low - buffer and prev.close >= swing_low - buffer
                sweep = c.high >= prior_high - tol and c.close < prior_high and c.close < c.open
                micro_break = c.close < min(prev.low, candles[idx - 2].low)
                choch = sweep and micro_break
                directional = c.close < c.open
                raw_stop = max(c.high, prev.high, prior_high) + self.entry_stop_buffer_atr * atr_val

            structural_event = bos or choch
            if self.pa_require_sweep_or_choch and not structural_event:
                continue
            min_body = self.pa_min_body_atr + (0.05 if regime_state == "TRANSITION" else 0.0)
            min_close = self.pa_min_close_quality + (0.05 if regime_state == "TRANSITION" else 0.0)
            min_vol = self.pa_min_volume_ratio + (0.10 if regime_state == "TRANSITION" else 0.0)
            if not (directional and micro_break and body_atr >= min_body
                    and close_quality >= min_close and vol_ratio >= min_vol):
                continue
            labels = []
            if bos: labels.append("bos")
            if choch: labels.append("choch_sweep_reclaim")
            if sweep and "choch_sweep_reclaim" not in labels: labels.append("liquidity_sweep")
            labels.append("micro_structure_break")
            edge = 76.0 + (3.0 if choch else 0.0) + (2.0 if bos else 0.0)
            edge += min(3.0, max(0.0, (vol_ratio - min_vol) * 3.0))
            return {
                "trigger_ts": c.timestamp, "raw_stop": float(raw_stop),
                "structure_level": round(float(swing_high if direction == "long" else swing_low), 8),
                "body_atr": round(body_atr, 3), "close_quality": round(close_quality, 3),
                "volume_ratio": round(vol_ratio, 3), "confirmations": labels,
                "regime": regime_state, "edge_score": round(edge, 2),
            }
        return None

    def _breakout_retest_setup(self, candles: list, direction: str, atr_val: float) -> Optional[dict]:
        """Return the newest qualified breakout-retest trigger within the
        validity window. A direct breakout is never returned: price must revisit
        the broken level and close back on the trend side."""
        min_needed = self.breakout_lookback + self.breakout_arm_bars + 3
        if atr_val <= 0 or len(candles) < min_needed:
            return None

        n = len(candles)
        trigger_start = max(1, n - self.entry_trigger_valid_bars)
        buffer = self.breakout_buffer_atr * atr_val
        tolerance = self.breakout_retest_tolerance_atr * atr_val
        invalidation = self.breakout_invalidation_atr * atr_val

        for trigger_idx in range(n - 1, trigger_start - 1, -1):
            trigger = candles[trigger_idx]
            bo_start = max(self.breakout_lookback, trigger_idx - self.breakout_arm_bars)
            for bo_idx in range(trigger_idx - 1, bo_start - 1, -1):
                prior = candles[bo_idx - self.breakout_lookback:bo_idx]
                if len(prior) < self.breakout_lookback:
                    continue
                breakout = candles[bo_idx]
                prev = candles[bo_idx - 1]
                level = (max(c.high for c in prior) if direction == "long"
                         else min(c.low for c in prior))
                body_atr = abs(breakout.close - breakout.open) / atr_val
                rng = max(breakout.high - breakout.low, 1e-12)
                close_quality = ((breakout.close - breakout.low) / rng if direction == "long"
                                 else (breakout.high - breakout.close) / rng)
                vol_hist = [c.volume for c in candles[max(0, bo_idx - self.volume_sma_period):bo_idx]]
                vol_base = float(np.mean(vol_hist)) if vol_hist else 0.0
                vol_ratio = breakout.volume / vol_base if vol_base > 0 else 0.0

                if direction == "long":
                    broke = breakout.close > level + buffer and prev.close <= level + buffer
                else:
                    broke = breakout.close < level - buffer and prev.close >= level - buffer
                if not (broke and body_atr >= self.breakout_min_body_atr
                        and close_quality >= self.breakout_min_close_quality
                        and vol_ratio >= self.breakout_min_volume_ratio):
                    continue

                segment = candles[bo_idx + 1:trigger_idx + 1]
                if not segment:
                    continue
                if direction == "long":
                    touched = any(c.low <= level + tolerance for c in segment)
                    held = all(c.close >= level - invalidation for c in segment)
                    reclaimed = trigger.close > level and candles[-1].close > level
                else:
                    touched = any(c.high >= level - tolerance for c in segment)
                    held = all(c.close <= level + invalidation for c in segment)
                    reclaimed = trigger.close < level and candles[-1].close < level
                confirm, labels = self._entry_candle_confirmation(
                    candles, trigger_idx, direction, level, atr_val, allow_micro_bos=True)
                if not (touched and held and reclaimed and confirm):
                    continue

                if direction == "long":
                    raw_stop = min(c.low for c in candles[bo_idx + 1:]) - self.entry_stop_buffer_atr * atr_val
                else:
                    raw_stop = max(c.high for c in candles[bo_idx + 1:]) + self.entry_stop_buffer_atr * atr_val
                edge = 75.0 + min(5.0, max(0.0, (vol_ratio - self.breakout_min_volume_ratio) * 5.0))
                if "micro_bos" in labels:
                    edge += 2.0
                return {
                    "trigger_ts": trigger.timestamp, "breakout_ts": breakout.timestamp,
                    "breakout_level": round(float(level), 8), "raw_stop": float(raw_stop),
                    "body_atr": round(body_atr, 3), "close_quality": round(close_quality, 3),
                    "volume_ratio": round(vol_ratio, 3), "confirmations": labels,
                    "edge_score": round(edge, 2),
                }
        return None

    def _structure_retest_setup(self, candles: list, direction: str, atr_val: float) -> Optional[dict]:
        """Return a HH/HL or LH/LL retest entry. The level is the latest
        confirmed HL (long) or LH (short); a touch alone is insufficient — a
        reclaim, engulf/pin, or micro-BOS must follow."""
        if atr_val <= 0 or len(candles) < max(12, self.structure_level_max_age_bars // 2):
            return None
        pivots = self._confirmed_pivots(candles, self.structure_pivot_left, self.structure_pivot_right)
        structure = self._swing_structure(pivots)
        required = "bull" if direction == "long" else "bear"
        if structure != required:
            return None

        wanted = "low" if direction == "long" else "high"
        levels = [p for p in pivots if p["type"] == wanted]
        if len(levels) < 2:
            return None
        level_pivot = levels[-1]
        index_by_ts = {c.timestamp: i for i, c in enumerate(candles)}
        pivot_idx = index_by_ts.get(level_pivot["timestamp"])
        if pivot_idx is None or len(candles) - 1 - pivot_idx > self.structure_level_max_age_bars:
            return None

        level = level_pivot["price"]
        tolerance = self.structure_retest_tolerance_atr * atr_val
        invalidation = self.structure_invalidation_atr * atr_val
        n = len(candles)
        trigger_start = max(pivot_idx + self.structure_pivot_right + 1, n - self.entry_trigger_valid_bars)

        for trigger_idx in range(n - 1, trigger_start - 1, -1):
            touch_start = max(pivot_idx + 1, trigger_idx - self.structure_retest_window_bars + 1)
            touch_idx = None
            for i in range(trigger_idx, touch_start - 1, -1):
                c = candles[i]
                touched = (c.low <= level + tolerance if direction == "long"
                           else c.high >= level - tolerance)
                if touched:
                    touch_idx = i
                    break
            if touch_idx is None:
                continue
            segment = candles[touch_idx:trigger_idx + 1]
            if direction == "long":
                held = all(c.close >= level - invalidation for c in segment)
                current_side = candles[trigger_idx].close > level and candles[-1].close > level
            else:
                held = all(c.close <= level + invalidation for c in segment)
                current_side = candles[trigger_idx].close < level and candles[-1].close < level
            confirm, labels = self._entry_candle_confirmation(
                candles, trigger_idx, direction, level, atr_val, allow_micro_bos=True)
            if not (held and current_side and confirm):
                continue

            if direction == "long":
                raw_stop = min(c.low for c in candles[touch_idx:]) - self.entry_stop_buffer_atr * atr_val
            else:
                raw_stop = max(c.high for c in candles[touch_idx:]) + self.entry_stop_buffer_atr * atr_val
            edge = 78.0
            if "micro_bos" in labels:
                edge += 3.0
            if "reclaim" in labels:
                edge += 2.0
            return {
                "trigger_ts": candles[trigger_idx].timestamp,
                "structure": structure, "retest_level": round(float(level), 8),
                "level_pivot_ts": level_pivot["timestamp"], "touch_ts": candles[touch_idx].timestamp,
                "raw_stop": float(raw_stop), "confirmations": labels,
                "edge_score": round(edge, 2),
            }
        return None

    def _entry_candle_confirmation(self, candles: list, idx: int, direction: str,
                                   level: float, atr_val: float,
                                   allow_micro_bos: bool = True) -> tuple[bool, list[str]]:
        if idx <= 0 or idx >= len(candles) or atr_val <= 0:
            return False, []
        c = candles[idx]
        prev = candles[idx - 1]
        rng = max(c.high - c.low, 1e-12)
        body = abs(c.close - c.open)
        labels: list[str] = []

        if direction == "long":
            close_quality = (c.close - c.low) / rng
            reclaim = c.low <= level + self.structure_retest_tolerance_atr * atr_val and c.close > level and c.close > c.open
            engulf = c.close > prev.open and c.open <= prev.close and c.close > c.open
            lower_wick = min(c.open, c.close) - c.low
            pin = lower_wick >= max(body * 1.5, atr_val * 0.05) and close_quality >= 0.60
            micro_bos = False
            if allow_micro_bos and idx >= self.structure_micro_bos_lookback:
                prior_high = max(x.high for x in candles[idx - self.structure_micro_bos_lookback:idx])
                micro_bos = c.close > prior_high and body >= atr_val * 0.10
        else:
            close_quality = (c.high - c.close) / rng
            reclaim = c.high >= level - self.structure_retest_tolerance_atr * atr_val and c.close < level and c.close < c.open
            engulf = c.close < prev.open and c.open >= prev.close and c.close < c.open
            upper_wick = c.high - max(c.open, c.close)
            pin = upper_wick >= max(body * 1.5, atr_val * 0.05) and close_quality >= 0.60
            micro_bos = False
            if allow_micro_bos and idx >= self.structure_micro_bos_lookback:
                prior_low = min(x.low for x in candles[idx - self.structure_micro_bos_lookback:idx])
                micro_bos = c.close < prior_low and body >= atr_val * 0.10

        if reclaim:
            labels.append("reclaim")
        if engulf:
            labels.append("engulfing")
        if pin:
            labels.append("pin_bar")
        if micro_bos:
            labels.append("micro_bos")
        return bool(labels), labels

    def _compute_entry_sl_tp(self, direction: str, price: float, raw_stop: float,
                             atr_val: float, mirror_raw_stop: bool = False
                             ) -> Optional[tuple[float, float, float, float]]:
        """Normalize a candidate stop. EMA stops keep the legacy absolute
        distance behavior; structure stops must genuinely sit beyond the retest
        low/high. Tiny stops are widened, overly large stops are rejected."""
        if price <= 0 or atr_val <= 0 or raw_stop is None or np.isnan(raw_stop):
            return None
        if mirror_raw_stop:
            raw_distance = abs(price - raw_stop)
        elif direction == "long":
            raw_distance = price - raw_stop
        else:
            raw_distance = raw_stop - price
        if raw_distance <= 0:
            return None

        min_distance = max(self.min_sl_pct * price, self.entry_min_stop_atr * atr_val)
        max_distance = max(min_distance, self.entry_max_stop_atr * atr_val)
        distance = max(raw_distance, min_distance)
        if distance > max_distance:
            return None
        if direction == "long":
            sl = price - distance
            tp = price + distance * self.rr_ratio
        else:
            sl = price + distance
            tp = price - distance * self.rr_ratio
        return sl, tp, distance, distance / atr_val

    @staticmethod
    def _entry_reason_summary(candidate: dict) -> str:
        detail = candidate.get("detail", {})
        et = candidate.get("entry_type")
        if et == "EMA_CROSS":
            return f"EMA cross {detail.get('cross_bars_ago')} bar(s) ago"
        if et == "BREAKOUT_RETEST":
            return (f"breakout level {detail.get('breakout_level')} retested; "
                    f"volume {detail.get('volume_ratio')}x; "
                    f"confirm {','.join(detail.get('confirmations', []))}")
        if et == "STRUCTURE_RETEST":
            return (f"{detail.get('structure')} structure retest at {detail.get('retest_level')}; "
                    f"confirm {','.join(detail.get('confirmations', []))}")
        return et or "entry"

    def _compute_sl_tp(self, direction: str, price: float, sl_ema_val: float) -> tuple[float, float]:
        # SL sits at the 5m EMA50 (sl_ema_val). Its distance from entry defines
        # R; TP2 = R x rr_ratio. If EMA50 sits right on price the raw SL
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
        signals; `sideways_min_signals` (default 3) of them = veto.

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

        up_votes = sum((sma_up, ema1020_up, slope_up, macd_up))
        down_votes = sum((sma_down, ema1020_down, slope_down, macd_down))
        up_core = ema1020_up if self.layer1_require_ema_alignment else True
        down_core = ema1020_down if self.layer1_require_ema_alignment else True
        trend_up = up_core and up_votes >= self.layer1_min_agreement and up_votes > down_votes
        trend_down = down_core and down_votes >= self.layer1_min_agreement and down_votes > up_votes

        return {
            "trend": "up" if trend_up else "down" if trend_down else None,
            "sma_dir": "up" if sma_up else "down" if sma_down else "flat",
            "ema1020_dir": "up" if ema1020_up else "down" if ema1020_down else "flat",
            "slope_dir": "up" if slope_up else "down" if slope_down else "flat",
            "macd_dir": "up" if macd_up else "down" if macd_down else "flat",
            "up_votes": int(up_votes), "down_votes": int(down_votes),
            "required_votes": self.layer1_min_agreement,
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
        """Entry/exit indicators on the 5m series: EMA10/20 cross, price vs the
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
            "ema_fast_val": float(ema_f[-1]),
            "ema_slow_val": float(ema_s[-1]),
            "ema_bull_aligned": ema_f[-1] > ema_s[-1] and last.close > ema_ref[-1],
            "ema_bear_aligned": ema_f[-1] < ema_s[-1] and last.close < ema_ref[-1],
            "close_below_ema_slow": last.close < ema_s[-1],
            "close_above_ema_slow": last.close > ema_s[-1],
            "above_ema_ref": last.close > ema_ref[-1],
            "below_ema_ref": last.close < ema_ref[-1],
            "sl_ema_val": float(sl_ema[-1]),
            "dist_ema_val": float(chase_ema[-1]),   # chase-guard reference
            "atr_val": float(atr_arr[-1]),
        }

    @staticmethod
    def _resample_timeframe(candles: list, bucket_ms: int, source_ms: int) -> list:
        """Resample closed OHLCV candles into a higher timeframe.

        The final bucket is dropped only when the newest source candle has not
        reached that bucket's end. This provides a safe 1H/4H fallback when the
        bot does not request those timeframes directly.
        """
        if not candles or bucket_ms <= 0 or source_ms <= 0:
            return []
        buckets: dict[int, list] = {}
        for candle in candles:
            key = (int(candle.timestamp) // bucket_ms) * bucket_ms
            buckets.setdefault(key, []).append(candle)

        class _Bar:
            __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
            def __init__(self, ts, o, h, l, cl, v):
                self.timestamp = ts; self.open = o; self.high = h
                self.low = l; self.close = cl; self.volume = v

        keys = sorted(buckets)
        out: list = []
        for key in keys:
            group = sorted(buckets[key], key=lambda c: c.timestamp)
            out.append(_Bar(
                key, group[0].open, max(c.high for c in group),
                min(c.low for c in group), group[-1].close,
                sum(c.volume for c in group),
            ))
        if keys:
            last_source_end = int(candles[-1].timestamp) + source_ms
            if last_source_end < keys[-1] + bucket_ms:
                out = out[:-1]
        return out

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
