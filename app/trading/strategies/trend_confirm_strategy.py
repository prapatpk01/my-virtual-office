"""
Trend Confirm — 4H Bias Score / 1H Context / 15M EMA Cross V4.1.

Live architecture is intentionally simple and time-frame separated:
  • 4H builds a soft 0-100 macro bias score and chooses direction only when
    the score leaves the neutral band.
  • 1H is always evaluated for context and trend quality, even while 4H is
    neutral, so diagnostics show the real 1H state instead of a false WARMUP.
  • 15M filters clear CHOP/sideways conditions and provides the only entry
    trigger: a confirmed EMA8/EMA13 cross on a CLOSED 15-minute candle.

There are no breakout/structure/price-action OR entry engines in the live path.
Once a position is open, the initial structure/ATR stop and one hard TP remain
active. The target is quality-aware (1.5R / 2.0R / maximum 2.5R). At +0.8R,
the SL moves to +0.5R while the full position stays open. A confirmed reverse
EMA8/EMA13 cross on a CLOSED 15-minute candle closes the position immediately,
even after the +0.5R lock, then the strategy waits for a genuinely new cross in
the currently allowed 4H/1H trend direction.

The public class name, constructor compatibility, signal metadata and strategy
callbacks remain compatible with the existing bot.
"""
from __future__ import annotations

import math
import time
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
        layer2_threshold: float = 55.0,        # balanced live threshold; HTF context still protects location
        layer2_threshold_early: float = 64.0,  # early setups remain stricter without becoming practically unreachable
        allow_15m_quality_fallback: bool = True,
        single_tf_quality_penalty: float = 4.0,
        # Layer 3 — entry (5m): EMA10/20 cross, price above/below EMA20, within 1.5xATR of EMA50
        entry_tf: str = "15m",       # 5m execution: HTF context filters noise while micro-structure gives earlier entries
        trend_tf: str = "4h",       # Layer1 trend timeframe: "1h" (uses mtf 1h) or "30m" (15m resample)
        ema_fast: int = 8,          # entry-cross fast EMA (on entry_tf)
        ema_slow: int = 13,         # entry-cross slow EMA (on entry_tf); also the cross-back exit reference
        entry_ema_ref: int = 13,    # price must be above (long) / below (short) this EMA — same line the cross + exit use
        sl_ema_ref: int = 50,       # SL sits at this EMA (5m)
        chase_ema_ref: int = 13,    # chase-guard distance is measured vs this EMA (5m); decoupled from sl_ema_ref
        fresh_trend_bars: int = 2,  # EMA-cross lookback (in 5m bars) when the trend just confirmed (early trend)
        cross_valid_bars: int = 2,  # how many 5m bars a cross stays usable while Layer2 gates settle —
                                    #   without this, a cross was only good on the exact bar every gate was
                                    #   already open (quality/location often clear 1-2 bars AFTER the cross,
                                    #   which silently wasted almost every signal)
        max_dist_atr_mult: float = 1.20,  # EMA-cross chase limit in ATR(5m)
        breakout_max_dist_atr_mult: float = 1.80,
        structure_max_dist_atr_mult: float = 1.60,
        # Layer 3 Entry Router — five independent context-aware triggers. Breakout uses
        # retest-only execution (no direct chasing) and structure entry requires
        # a confirmed HH/HL or LH/LL sequence plus a reclaim / micro-BOS trigger.
        use_ema_cross_entry: bool = True,
        use_breakout_retest_entry: bool = False,
        use_structure_retest_entry: bool = False,
        use_price_action_structure_entry: bool = False,
        use_early_structure_entry: bool = False,
        # Precision EMA-only execution.  When enabled, stale runtime config
        # cannot silently re-enable the experimental entry engines.
        ema_only_mode: bool = True,
        use_closed_entry_bars: bool = True,
        closed_bar_grace_ms: int = 1500,
        ema_slow_slope_lookback: int = 2,
        ema_min_gap_atr: float = 0.02,
        ema_min_body_atr: float = 0.05,
        ema_min_close_quality: float = 0.55,
        ema_min_volume_ratio: float = 0.60,
        ema_impulse_body_atr: float = 1.20,
        ema_max_extension_slow_atr: float = 0.90,
        ema_retest_tolerance_atr: float = 0.22,
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
        pa_entry_min_quality: float = 58.0,
        pa_min_body_atr: float = 0.18,
        pa_min_close_quality: float = 0.62,
        pa_min_volume_ratio: float = 0.80,
        pa_transition_quality_bonus: float = 3.0,
        early_structure_min_quality: float = 56.0,
        early_structure_sweep_lookback: int = 6,
        early_structure_micro_bos_lookback: int = 3,
        early_structure_max_signal_age_bars: int = 3,
        early_structure_min_rejection_wick_atr: float = 0.08,
        early_structure_transition_quality_bonus: float = 4.0,
        entry_stop_buffer_atr: float = 0.10,
        entry_min_stop_atr: float = 0.60,
        entry_max_stop_atr: float = 2.00,
        # Position sizing (emitted in the signal so bot.py sizes live orders the
        # same way the paper account does): margin = margin_pct of balance,
        # notional = margin x leverage. e.g. $100 x 5% = $5 x 20 = $100 notional.
        sizing_mode: str = "margin",
        margin_pct: float = 0.05,
        # Location & structure-room filter (lightweight; avoids late/blocked entries)
        use_location_filter: bool = False,
        structure_pivot_left: int = 2,
        structure_pivot_right: int = 2,
        zone_width_atr_1h: float = 0.18,
        zone_width_atr_4h: float = 0.22,
        hard_zone_distance_atr: float = 0.25,
        min_structure_room_r: float = 1.20,   # hard-reject below this many R of room to the opposing zone
        preferred_structure_room_r: float = 1.50,  # below this (but >= min) just penalizes the quality gate
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
        use_supply_demand_zones: bool = False,
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
        regime_transition_threshold_penalty: float = 2.0,
        allow_structure_entry_in_transition: bool = True,
        require_trend_regime: bool = False,     # if True, HARD-block non-TREND regimes. Default off:
                                                #   TRANSITION is allowed but must pass the extra gate below.
        # TRANSITION extra-analysis gate — these regimes CAN trade, but need
        # real confirmation. Kept LIGHT on the quality axis (a fresh cross is
        # usually early-trend, which already carries the stricter early
        # threshold) — the extra confidence comes from momentum + volume, not a
        # sky-high score bar that would block essentially every cross.
        transition_extra_threshold: float = 4.0,    # no quality stacking (6 blocked ~every cross). The extra
                                                     #   analysis is momentum + volume below, not a higher bar.
        transition_min_vol_ratio: float = 0.80,      # only block genuinely dead volume (< 0.5x its SMA)
        transition_require_momentum: bool = True,   # need MACD histogram pushing in the trend's direction
        transition_momentum_frac: float = 0.3,      # momentum dimension must reach this fraction of its weight
        transition_min_directional_votes: int = 4,    # 3-of-5 directional evidence; avoids indicator-stacking vetoes
        transition_require_clean_location: bool = True,  # applied candidate-by-candidate after the stop/room is known
        # Sideways / range veto (Layer 2) — hard-block entries when the 15m
        # context looks like a range, not a trend. Designed NOT to kill early
        # trends: it leans on EMA compression + high chop (which stay range-y
        # even as ADX lags), and only counts "really weak" ADX (< sideways_adx_max,
        # stricter than adx_threshold) so a fresh trend at ADX ~18 isn't vetoed.
        use_sideways_filter: bool = True,
        sideways_ema_compression_atr: float = 0.5,  # |EMA20-EMA50| < this x ATR = tangled/flat
        sideways_adx_max: float = 15.0,             # ADX below this = "really weak" (< adx_threshold on purpose)
        sideways_range_atr: float = 1.2,            # last-20-bar high-low range < this x ATR = tight consolidation
        sideways_min_signals: int = 4,              # how many of the 4 signals must fire to veto (clear ranges only)
        # Exit (5m): EMA10/20 cross-back OR a 5m close past EMA20 closes the runner
        use_close_past_exit: bool = False,   # faster exit: also close when price closes past EMA_slow (before
                                            #   the full EMA8/13 cross-back) — more responsive, protects profit
        exit_close_confirm_bars: int = 2,   # N consecutive closes past EMA_slow required (1 = fastest)
        signal_exit_requires_tp1: bool = False,  # cross-back exit works immediately (no TP1 to wait for now)
                                                 #   bounds manage the trade until then. On 5m the single-close
                                                 #   slow-EMA exits killed 75% of trades at ~-0.3R before TP1; arming
                                                 #   them only on the runner nearly doubled WR (25->62% BTC,
                                                 #   41->60% SOL) and cut losses ~2-3x in backtest
        # Take-profit scheme. use_hard_tp=False + use_partial_tp=False = a pure
        # trend-follow cross system: no TP at all, ride the position until the
        # EMA8/13 cross-back (the SL at EMA50 is only a disaster stop).
        use_hard_tp: bool = True,          # emit a fixed TP2 (1.5R) with the entry? off = hold to cross-back
        use_partial_tp: bool = False,       # TP1 -> take tp1_close_pct, move SL to BE+be_offset_r; runner rides on
        # Break-even TRAIL (no partial close): once price reaches be_trail_trigger_r
        # of profit, ratchet the SL up to entry + be_trail_sl_r (locks a minimum
        # profit) and keep riding the full position until the EMA8/13 cross-back.
        use_be_trail: bool = True,
        be_trail_trigger_r: float = 0.8,    # target: move the SL once +0.8R is reached
        be_trail_sl_r: float = 0.5,         # new SL sits at entry +/- 0.5R (BE + 0.5R locked)
        runner_ignore_signal_exit_after_be: bool = False,
        exit_min_hold_bars: int = 0,
        exit_failure_confirmations: int = 2,
        exit_structure_lookback: int = 3,
        same_direction_rearm_bars: int = 3,
        tp1_r: float = 0.75,                # TP1 at 0.75R (halfway to the 1.5R final TP)
        tp1_close_pct: float = 0.5,         # fraction closed at TP1
        be_offset_r: float = 0.1,           # after TP1, SL -> entry +/- this many R (BE + 0.1R, a small locked profit)
        # Risk
        atr_period: int = 14,               # ATR(5m) for the chase guard / min-distance sanity
        rr_ratio: float = 1.5,              # fallback RR; live TP is entry-type + regime aware below
        ema_rr_transition: float = 1.5,
        ema_rr_trend: float = 2.0,
        ema_rr_strong: float = 2.5,
        breakout_rr_transition: float = 1.5,
        breakout_rr_trend: float = 1.8,
        breakout_rr_strong: float = 2.0,
        structure_rr_transition: float = 1.5,
        structure_rr_trend: float = 1.8,
        structure_rr_strong: float = 2.0,
        pa_rr_transition: float = 1.3,
        pa_rr_trend: float = 1.5,
        pa_rr_strong: float = 1.8,
        early_rr_transition: float = 1.3,
        early_rr_trend: float = 1.6,
        early_rr_strong: float = 1.8,
        min_sl_pct: float = 0.005,          # legacy compatibility only; 15M live SL uses ATR floor, not fixed %
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
        self.use_early_structure_entry = use_early_structure_entry
        self.ema_only_mode = bool(ema_only_mode)
        self.use_closed_entry_bars = bool(use_closed_entry_bars)
        self.closed_bar_grace_ms = max(0, int(closed_bar_grace_ms))
        self.ema_slow_slope_lookback = max(1, int(ema_slow_slope_lookback))
        self.ema_min_gap_atr = max(0.0, float(ema_min_gap_atr))
        self.ema_min_body_atr = max(0.0, float(ema_min_body_atr))
        self.ema_min_close_quality = min(0.95, max(0.50, float(ema_min_close_quality)))
        self.ema_min_volume_ratio = max(0.0, float(ema_min_volume_ratio))
        self.ema_impulse_body_atr = max(self.ema_min_body_atr, float(ema_impulse_body_atr))
        self.ema_max_extension_slow_atr = max(0.20, float(ema_max_extension_slow_atr))
        self.ema_retest_tolerance_atr = max(0.0, float(ema_retest_tolerance_atr))
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
        self.pa_entry_min_quality = pa_entry_min_quality
        self.pa_min_body_atr = max(0.05, pa_min_body_atr)
        self.pa_min_close_quality = min(0.95, max(0.50, pa_min_close_quality))
        self.pa_min_volume_ratio = max(0.0, pa_min_volume_ratio)
        self.pa_transition_quality_bonus = max(0.0, pa_transition_quality_bonus)
        self.early_structure_min_quality = early_structure_min_quality
        self.early_structure_sweep_lookback = max(3, early_structure_sweep_lookback)
        self.early_structure_micro_bos_lookback = max(2, early_structure_micro_bos_lookback)
        self.early_structure_max_signal_age_bars = max(1, early_structure_max_signal_age_bars)
        self.early_structure_min_rejection_wick_atr = max(0.02, early_structure_min_rejection_wick_atr)
        self.early_structure_transition_quality_bonus = max(0.0, early_structure_transition_quality_bonus)
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
        self.transition_min_directional_votes = max(2, min(5, int(transition_min_directional_votes)))
        self.transition_require_clean_location = transition_require_clean_location
        self.use_sideways_filter = use_sideways_filter
        self.sideways_ema_compression_atr = sideways_ema_compression_atr
        self.sideways_adx_max = sideways_adx_max
        self.sideways_range_atr = sideways_range_atr
        self.sideways_min_signals = sideways_min_signals

        self.use_close_past_exit = use_close_past_exit
        self.exit_close_confirm_bars = exit_close_confirm_bars
        self.signal_exit_requires_tp1 = signal_exit_requires_tp1

        self.use_hard_tp = use_hard_tp
        self.use_partial_tp = use_partial_tp
        self.use_be_trail = use_be_trail
        self.be_trail_trigger_r = be_trail_trigger_r
        self.be_trail_sl_r = be_trail_sl_r
        self.runner_ignore_signal_exit_after_be = runner_ignore_signal_exit_after_be
        self.exit_min_hold_bars = max(0, int(exit_min_hold_bars))
        self.exit_failure_confirmations = max(2, min(3, int(exit_failure_confirmations)))
        self.exit_structure_lookback = max(2, int(exit_structure_lookback))
        self.same_direction_rearm_bars = max(0, int(same_direction_rearm_bars))
        self._diag_context = {
            "regime": "WARMUP", "trend_4h": "WARMUP", "trend_1h": "WARMUP",
            "aligned": False, "mtf": "WARMUP", "strategy": "EMA_CROSS_15M",
            "entry_tf": "15m", "direction_15m": "WAIT_CROSS", "entry_state": "INIT",
        }
        self._be_trailed: bool = False
        self._entry_threshold_bonus: float = 0.0  # bot raises this during post-cooldown strict window
        self.tp1_r = tp1_r
        self.tp1_close_pct = tp1_close_pct
        self.be_offset_r = be_offset_r

        self.atr_period = atr_period
        self.rr_ratio = rr_ratio
        self.ema_rr_transition = ema_rr_transition
        self.ema_rr_trend = ema_rr_trend
        self.ema_rr_strong = ema_rr_strong
        self.breakout_rr_transition = breakout_rr_transition
        self.breakout_rr_trend = breakout_rr_trend
        self.breakout_rr_strong = breakout_rr_strong
        self.structure_rr_transition = structure_rr_transition
        self.structure_rr_trend = structure_rr_trend
        self.structure_rr_strong = structure_rr_strong
        self.pa_rr_transition = pa_rr_transition
        self.pa_rr_trend = pa_rr_trend
        self.pa_rr_strong = pa_rr_strong
        self.early_rr_transition = early_rr_transition
        self.early_rr_trend = early_rr_trend
        self.early_rr_strong = early_rr_strong
        self.min_sl_pct = min_sl_pct

        # Apply params passed by the bot/config. Older versions forwarded params
        # to BaseStrategy but silently ignored them in this class.
        self._apply_runtime_params(params)
        if self.ema_only_mode:
            # V4 live core is deliberately fixed: 4H macro -> 1H quality ->
            # 15M closed-bar EMA8/13 cross. Stale runtime config cannot turn
            # the strategy back into the old 5M/multi-entry engine.
            self.entry_tf = "15m"
            self.trend_tf = "4h"
            self.chase_ema_ref = self.ema_slow
            self.use_ema_cross_entry = True
            self.use_breakout_retest_entry = False
            self.use_structure_retest_entry = False
            self.use_price_action_structure_entry = False
            self.use_early_structure_entry = False
            self.use_location_filter = False
            self.use_supply_demand_zones = False
            self.require_trend_regime = False
            self.layer2_threshold = 52.0  # normal 4H baseline; strong 4H adapts to 48
            self.sideways_min_signals = 3
            self.max_dist_atr_mult = 1.20
            self.entry_min_stop_atr = 0.60
            self.entry_max_stop_atr = 2.00
            self.use_hard_tp = True
            self.use_partial_tp = False
            self.use_be_trail = True
            self.be_trail_trigger_r = 0.8
            self.be_trail_sl_r = 0.5
            self.runner_ignore_signal_exit_after_be = False

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
        self._last_pa_trigger_up_ts: Optional[int] = None
        self._last_pa_trigger_down_ts: Optional[int] = None
        self._last_early_trigger_up_ts: Optional[int] = None
        self._last_early_trigger_down_ts: Optional[int] = None
        self._last_entry_attempt_bar_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None  # owned by tick_open_position()
        self._latest_candles: list = []
        self._latest_15m: list = []      # closed 15m series for entry and reverse-cross exit
        self._latest_5m: list = []       # compatibility alias; V4 does not use 5m execution
        # Partial-TP tracking for the open position (owned by tick_open_position())
        self._entry_price: Optional[float] = None
        self._entry_sl: Optional[float] = None
        self._entry_bar_ts: Optional[int] = None
        # Reverse-cross freshness guard.  A cross may close a position only on
        # a CLOSED 15M bar strictly newer than the entry/reference bar.  On
        # restart, attach_existing_position() arms this to the latest already
        # closed bar so historical crosses can never immediately close an
        # adopted live position.
        self._reverse_cross_arm_after_ts: Optional[int] = None
        self._adopted_after_restart: bool = False
        self._entry_regime: Optional[str] = None
        self._last_signal_exit_ts: Optional[int] = None
        self._last_signal_exit_direction: Optional[str] = None
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
        self.cross_valid_bars = max(1, min(3, int(self.cross_valid_bars)))
        self.ema_slow_slope_lookback = max(1, int(self.ema_slow_slope_lookback))
        self.ema_min_close_quality = min(0.95, max(0.50, float(self.ema_min_close_quality)))
        self.ema_min_volume_ratio = max(0.0, float(self.ema_min_volume_ratio))
        self.ema_max_extension_slow_atr = max(0.20, float(self.ema_max_extension_slow_atr))
        self.exit_failure_confirmations = max(2, min(3, int(self.exit_failure_confirmations)))
        self.exit_structure_lookback = max(2, int(self.exit_structure_lookback))
        self.same_direction_rearm_bars = max(0, int(self.same_direction_rearm_bars))

    def _diag_reset(self) -> None:
        """Reset per-scan diagnostics with the V4 4H/1H/15M schema."""
        self._diag_context = {
            "regime": "WARMUP",
            "trend_4h": "WARMUP",
            "trend_1h": "WARMUP",
            "aligned": False,
            "mtf": "WARMUP",
            "strategy": "EMA_CROSS_15M",
            "entry_tf": "15m",
            "direction_15m": "WAIT_CROSS",
            "entry_state": "SCANNING",
        }

    def _diag_update(self, **values) -> None:
        if not hasattr(self, "_diag_context") or not isinstance(self._diag_context, dict):
            self._diag_reset()
        self._diag_context.update(values)

    def _diagnostic_4h_trend(self, candles_4h: list) -> str:
        """Compatibility wrapper around the live 4H macro engine."""
        ctx = self._macro_trend_4h(candles_4h)
        return ctx.get("state", "WARMUP") if ctx else "WARMUP"

    def _macro_trend_4h(self, candles_4h: list) -> dict:
        """Soft 4H macro-bias score (0=strong bear, 100=strong bull).

        The previous implementation required ``close > EMA20 > EMA50`` (or
        the bearish mirror) as a hard core condition before the market could
        even be called a trend.  That made normal pullbacks and early trend
        transitions look NEUTRAL for too long.

        V4.1 scores six *independent-ish* pieces of evidence instead:
          EMA20/50 direction  30
          EMA20 slope         20
          price location      15
          MACD direction      15
          DMI direction       10
          simple 4H structure 10

        Each factor contributes bullish=1, neutral=0.5, bearish=0.  The
        resulting bullness score maps to STRONG_BULL/BULL/NEUTRAL/BEAR/
        STRONG_BEAR.  NEUTRAL still blocks new entries, but it no longer stops
        the 1H analysis/diagnostics from running.
        """
        lb = max(2, int(self.ema_slope_lookback))
        # Structure uses two 6-bar windows; the indicator warmup usually
        # dominates this minimum anyway.
        min_needed = max(self.quality_ema_slow + lb + 2,
                         2 * self.adx_period + 2,
                         self.macd_slow + self.macd_signal + 2,
                         14)
        if not candles_4h or len(candles_4h) < min_needed:
            return {"state": "WARMUP", "direction": None, "score": None,
                    "adx": None, "factors": {}, "bars": len(candles_4h or [])}

        closes = [float(c.close) for c in candles_4h]
        e20 = self.ema(closes, self.quality_ema_fast)
        e50 = self.ema(closes, self.quality_ema_slow)
        macd_line, macd_sig, _ = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        adx_arr, plus_di, minus_di = self.adx(candles_4h, self.adx_period)
        atr_arr = self.atr(candles_4h, self.atr_period)
        needed = (e20[-1], e50[-1], e20[-1-lb], macd_line[-1], macd_sig[-1],
                  adx_arr[-1], plus_di[-1], minus_di[-1], atr_arr[-1])
        if any(np.isnan(v) for v in needed):
            return {"state": "WARMUP", "direction": None, "score": None,
                    "adx": None, "factors": {}, "bars": len(candles_4h)}

        def ternary(bull: bool, bear: bool) -> float:
            if bull and not bear:
                return 1.0
            if bear and not bull:
                return 0.0
            return 0.5

        close = closes[-1]
        atr4 = max(float(atr_arr[-1]), 1e-12)

        # Small ATR-scaled deadbands stop microscopic EMA/MACD differences in
        # a flat market from becoming a false STRONG_BULL/STRONG_BEAR score.
        ema_gap = float(e20[-1] - e50[-1])
        slope_delta = float(e20[-1] - e20[-1-lb])
        price_delta = float(close - e20[-1])
        macd_hist = float(macd_line[-1] - macd_sig[-1])
        dmi_gap = float(plus_di[-1] - minus_di[-1])

        # 1) EMA20/50 direction — 30 points.
        ema_align = ternary(ema_gap > 0.05 * atr4, ema_gap < -0.05 * atr4)
        # 2) EMA20 slope — 20 points.
        ema_slope = ternary(slope_delta > 0.03 * atr4, slope_delta < -0.03 * atr4)
        # 3) Price location vs EMA20 — 15 points. A shallow pullback around the
        #    average is neutral, not an automatic bearish vote.
        price_loc = ternary(price_delta > 0.25 * atr4, price_delta < -0.25 * atr4)
        # 4) MACD direction — 15 points.
        macd_dir = ternary(macd_hist > 0.05 * atr4, macd_hist < -0.05 * atr4)
        # 5) DMI direction — 10 points; ignore tiny DI spreads.
        dmi_dir = ternary(dmi_gap > 2.0, dmi_gap < -2.0)

        # 6) Simple confirmed-ish structure — compare the latest 6 closed 4H
        #    bars with the preceding 6.  HH+HL = bull, LH+LL = bear, mixed =
        #    neutral. This avoids making a fragile pivot detector a hard gate.
        recent = candles_4h[-6:]
        prior = candles_4h[-12:-6]
        recent_hi = max(float(c.high) for c in recent)
        recent_lo = min(float(c.low) for c in recent)
        prior_hi = max(float(c.high) for c in prior)
        prior_lo = min(float(c.low) for c in prior)
        hh = recent_hi > prior_hi
        hl = recent_lo > prior_lo
        lh = recent_hi < prior_hi
        ll = recent_lo < prior_lo
        structure = ternary(hh and hl, lh and ll)

        weighted = {
            "ema_align": (30.0, ema_align),
            "ema20_slope": (20.0, ema_slope),
            "price_location": (15.0, price_loc),
            "macd": (15.0, macd_dir),
            "dmi": (10.0, dmi_dir),
            "structure": (10.0, structure),
        }
        score = sum(w * v for w, v in weighted.values())
        score = round(max(0.0, min(100.0, score)), 1)

        if score >= 75.0:
            state, direction = "STRONG_BULL", "long"
        elif score >= 60.0:
            state, direction = "BULL", "long"
        elif score <= 24.0:
            state, direction = "STRONG_BEAR", "short"
        elif score <= 39.0:
            state, direction = "BEAR", "short"
        else:
            state, direction = "NEUTRAL", None

        factor_labels = {
            k: ("BULL" if v > 0.5 else "BEAR" if v < 0.5 else "NEUTRAL")
            for k, (_w, v) in weighted.items()
        }
        return {
            "state": state, "direction": direction, "score": score,
            "adx": round(float(adx_arr[-1]), 1), "factors": factor_labels,
            "structure": "HH_HL" if structure > 0.5 else "LH_LL" if structure < 0.5 else "MIXED",
            "ema20": round(float(e20[-1]), 8), "ema50": round(float(e50[-1]), 8),
            "atr": round(atr4, 8),
        }

    def _best_context_1h(self, candles_1h: list) -> Optional[dict]:
        """Evaluate BOTH 1H directions for diagnostics when 4H is neutral.

        This is intentionally informational only.  It never authorizes a trade
        while 4H is NEUTRAL/WARMUP; it just prevents misleading ``1H=WARMUP``
        logs when enough 1H data actually exists.
        """
        long_ctx = self._context_1h(candles_1h, "long")
        short_ctx = self._context_1h(candles_1h, "short")
        if long_ctx is None and short_ctx is None:
            return None
        if short_ctx is None or (long_ctx is not None and long_ctx.get("score", 0) >= short_ctx.get("score", 0)):
            return {"direction": "long", "context": long_ctx}
        return {"direction": "short", "context": short_ctx}

    def _context_1h(self, candles_1h: list, direction: str) -> Optional[dict]:
        """1H context + quality. Soft score plus 2/3 structural agreement."""
        trend = "up" if direction == "long" else "down"
        q = self._tf_quality(candles_1h, trend)
        if q is None:
            return None
        closes = [float(c.close) for c in candles_1h]
        e20 = self.ema(closes, self.quality_ema_fast)
        e50 = self.ema(closes, self.quality_ema_slow)
        lb = max(2, int(self.ema_slope_lookback))
        vals = (e20[-1], e50[-1], e50[-1-lb])
        if any(np.isnan(v) for v in vals):
            return None
        up = direction == "long"
        structural = [
            (closes[-1] > e20[-1]) == up,
            (e20[-1] > e50[-1]) == up,
            (e50[-1] > e50[-1-lb]) == up,
        ]
        votes = sum(bool(x) for x in structural)
        br = q.get("breakdown", {})
        adx_val = float(br.get("adx_val", 0.0) or 0.0)
        chop_val = float(br.get("chop_val", 100.0) or 100.0)
        score = float(q.get("score", 0.0) or 0.0)
        choppy = chop_val >= 65.0 and adx_val < 18.0
        # Final readiness is decided later with the 4H macro strength.
        # Keeping this object descriptive avoids a fixed 55 + 2/3 gate from
        # blocking otherwise valid strong-macro continuation setups.
        ready = False
        label = "VERY_STRONG" if score >= 85.0 else "STRONG" if score >= 70.0 else "TREND"
        return {"ready": ready, "score": round(score, 1), "label": label,
                "votes": votes, "adx": round(adx_val, 1), "chop": round(chop_val, 1),
                "quality": q, "choppy": choppy}

    def _rr_from_quality(self, quality_score: float) -> float:
        """Single hard TP, capped at 2.5R."""
        score = float(quality_score)
        if score >= 85.0:
            return 2.5
        if score >= 70.0:
            return 2.0
        return 1.5

    def _structure_stop_15m(self, candles: list, direction: str, atr_val: float) -> float:
        """Recent confirmed 15M swing stop with ATR buffer; fallback to 6-bar structure."""
        if not candles:
            return float("nan")
        n = len(candles)
        pivot = None
        scan_start = max(2, n - 24)
        if direction == "long":
            for i in range(n - 3, scan_start - 1, -1):
                if (float(candles[i].low) <= float(candles[i-1].low)
                        and float(candles[i].low) < float(candles[i-2].low)
                        and float(candles[i].low) <= float(candles[i+1].low)
                        and float(candles[i].low) < float(candles[i+2].low)):
                    pivot = float(candles[i].low)
                    break
            if pivot is None:
                pivot = min(float(c.low) for c in candles[-6:])
            return pivot - self.entry_stop_buffer_atr * atr_val
        for i in range(n - 3, scan_start - 1, -1):
            if (float(candles[i].high) >= float(candles[i-1].high)
                    and float(candles[i].high) > float(candles[i-2].high)
                    and float(candles[i].high) >= float(candles[i+1].high)
                    and float(candles[i].high) > float(candles[i+2].high)):
                pivot = float(candles[i].high)
                break
        if pivot is None:
            pivot = max(float(c.high) for c in candles[-6:])
        return pivot + self.entry_stop_buffer_atr * atr_val

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        """4H macro -> 1H context/quality -> 15M CHOP/range -> EMA8/13 cross."""
        self._diag_reset()
        mtf = mtf_candles or {}
        if not candles:
            return self._hold(current_price, "Data Quality: empty 15M candle series")

        # Only CLOSED candles may create a cross signal.
        c15 = self._closed_candle_series(candles, 15 * 60_000, self.closed_bar_grace_ms)
        c1h_raw = mtf.get("1h", []) or self._resample_timeframe(c15, 60 * 60_000, 15 * 60_000)
        c4h_raw = mtf.get("4h", []) or self._resample_timeframe(c15, 4 * 60 * 60_000, 15 * 60_000)
        c1h = self._closed_candle_series(c1h_raw, 60 * 60_000, self.closed_bar_grace_ms)
        c4h = self._closed_candle_series(c4h_raw, 4 * 60 * 60_000, self.closed_bar_grace_ms)
        self._latest_candles = c15
        self._latest_15m = c15
        self._latest_5m = c15  # compatibility only

        min15 = max(self.quality_ema_slow + self.ema_slope_lookback + 3,
                    2 * self.adx_period + 3, self.macd_slow + self.macd_signal + 3)
        if len(c15) < min15:
            return self._hold(current_price, f"15M warm-up: need {min15}+ closed bars, have {len(c15)}")

        # Data quality is safety, not a signal filter.
        data_quality = {}
        if self.use_data_quality_gate:
            for tf_name, series, expected_ms in (
                ("15m", c15, 15 * 60_000), ("1h", c1h, 60 * 60_000), ("4h", c4h, 4 * 60 * 60_000)
            ):
                qd = self._data_quality_context(series, expected_ms) if series else {
                    "valid": False, "reason": "missing", "bars": 0
                }
                data_quality[tf_name] = qd
                if not qd.get("valid"):
                    return self._hold(current_price, f"Data Quality FAIL {tf_name}: {qd.get('reason')}",
                                      metadata={"data_quality": data_quality})

        # Layer 1 — 4H macro bias is the directional authority, but 1H is
        # ALWAYS evaluated before any macro HOLD so the log shows the real
        # context instead of a misleading WARMUP placeholder.
        macro = self._macro_trend_4h(c4h)
        macro_state = macro.get("state", "WARMUP")
        macro_score = macro.get("score")
        direction = macro.get("direction")
        macro_display = (f"{macro_state}({macro_score:.0f})"
                         if isinstance(macro_score, (int, float)) else macro_state)

        if direction in ("long", "short"):
            ctx = self._context_1h(c1h, direction)
            ctx_direction = direction
        else:
            best = self._best_context_1h(c1h)
            ctx_direction = best.get("direction") if best else None
            ctx = best.get("context") if best else None

        if ctx is None:
            one_h_label = "WARMUP"
            self._diag_update(
                trend_4h=macro_display, trend_1h=one_h_label, regime="WARMUP",
                aligned=False, mtf=f"4H={macro_display} | 1H=WARMUP",
                entry_state="1H_WARMUP",
            )
            return self._hold(current_price, "1H context/quality genuinely warming up",
                              metadata={"macro_4h": macro, "data_quality": data_quality})

        one_h_label = f"{'LONG' if ctx_direction == 'long' else 'SHORT'}_{ctx['label']}"

        # Adaptive 1H gate:
        # Strong 4H trends already provide substantial directional evidence, so
        # 1H only needs to confirm that the move is not weak/choppy. Normal 4H
        # trends require a little more 1H quality. Neutral 4H still cannot trade.
        strong_macro = macro_state in ("STRONG_BULL", "STRONG_BEAR")
        if strong_macro:
            ctx_min_quality = 48.0
            ctx_min_votes = 1
            ctx_regime_ok = (ctx["adx"] >= 17.0 and ctx["chop"] <= 58.0)
        else:
            ctx_min_quality = 52.0
            ctx_min_votes = 1
            ctx_regime_ok = (ctx["adx"] >= 18.0 or ctx["chop"] <= 52.0)

        ctx["min_quality"] = ctx_min_quality
        ctx["min_votes"] = ctx_min_votes
        ctx["regime_ok"] = bool(ctx_regime_ok)
        ctx["ready"] = bool(
            ctx["score"] >= ctx_min_quality
            and ctx["votes"] >= ctx_min_votes
            and ctx_regime_ok
            and not ctx["choppy"]
        )

        macro_aligned = bool(direction in ("long", "short") and ctx_direction == direction and ctx["ready"])
        self._diag_update(
            trend_4h=macro_display,
            trend_1h=one_h_label,
            regime=ctx["label"],
            aligned=macro_aligned,
            mtf=f"4H={macro_display} | 1H={one_h_label} {ctx['score']:.0f}",
            entry_state="1H_CONTEXT",
        )

        # A neutral/warming 4H still blocks entries, but only AFTER 1H has been
        # computed and logged.
        if direction not in ("long", "short"):
            return self._hold(
                current_price,
                f"4H macro {macro_display} — 1H={one_h_label} quality {ctx['score']:.0f}; waiting for 4H score to leave NEUTRAL",
                metadata={"macro_4h": macro, "quality_1h": ctx, "data_quality": data_quality},
            )

        # Layer 2 — 1H context and trend quality in the 4H-selected direction.
        if not ctx["ready"]:
            reason = (
                f"1H context not ready: quality={ctx['score']:.0f} (min {ctx['min_quality']:.0f}), "
                f"structure={ctx['votes']}/3 (min {ctx['min_votes']}), "
                f"ADX={ctx['adx']:.1f}, CHOP={ctx['chop']:.1f}, "
                f"trend_quality={'OK' if ctx['regime_ok'] else 'WEAK'}"
            )
            return self._hold(current_price, reason,
                              metadata={"macro_4h": macro, "quality_1h": ctx, "data_quality": data_quality})

        # Layer 3 — clear 15M CHOP/sideways veto. Do not stack more indicators.
        trend_key = "up" if direction == "long" else "down"
        q15 = self._tf_quality(c15, trend_key)
        sideways = self._sideways_context(c15) if self.use_sideways_filter else {
            "is_sideways": False, "signals": 0, "detail": {}
        }
        br15 = (q15 or {}).get("breakdown", {})
        chop15 = float(br15.get("chop_val", 100.0) or 100.0)
        adx15 = float(br15.get("adx_val", 0.0) or 0.0)
        # 15M safety veto: block genuinely poor trend conditions, not merely
        # middling CHOP. A clear range still requires >=3/4 independent signals.
        hard_chop = chop15 >= 61.0 and adx15 < 18.0
        clear_sideways = sideways.get("signals", 0) >= 3
        self._diag_update(direction_15m="WAIT_CROSS", entry_state="15M_FILTER")
        if hard_chop or clear_sideways:
            return self._hold(
                current_price,
                f"15M CHOP/SIDEWAY block: CHOP={chop15:.1f}, ADX={adx15:.1f}, "
                f"range_signals={sideways.get('signals', 0)}/4",
                metadata={"macro_4h": macro, "quality_1h": ctx, "quality_15m": q15,
                          "sideways_15m": sideways, "data_quality": data_quality},
            )

        # Existing position: keep refreshing closed 15M candles for the reverse-cross exit.
        if self._open_position is not None:
            return self._hold(current_price, f"Holding {self._open_position.upper()} — 15M cross-back/SL/TP management active",
                              metadata={"macro_4h": macro, "quality_1h": ctx, "quality_15m": q15,
                                        "sideways_15m": sideways, "data_quality": data_quality})

        # Layer 4 — the only entry trigger: fresh CLOSED 15M EMA8/13 cross.
        l15 = self._layer3_indicators(c15)
        if l15 is None:
            return self._hold(current_price, "15M EMA8/13 indicators warming up")
        bar_ts = int(c15[-1].timestamp)
        cross_ok = l15["ema_cross_up"] if direction == "long" else l15["ema_cross_down"]
        cross_label = "UP" if l15["ema_cross_up"] else "DOWN" if l15["ema_cross_down"] else "WAIT"
        self._diag_update(direction_15m=f"CROSS_{cross_label}", entry_state="WAIT_15M_CROSS",
                          mtf=f"4H={macro_display} | 1H={one_h_label} {ctx['score']:.0f} | 15M={cross_label}")
        if not cross_ok:
            return self._hold(current_price, f"15M: waiting EMA{self.ema_fast}/{self.ema_slow} cross {direction.upper()}",
                              metadata={"macro_4h": macro, "quality_1h": ctx, "quality_15m": q15,
                                        "sideways_15m": sideways, "cross_15m": cross_label,
                                        "data_quality": data_quality})
        if self._last_entry_attempt_bar_ts == bar_ts:
            return self._hold(current_price, "15M cross already processed — waiting for a new cross",
                              metadata={"cross_15m": cross_label})

        atr_arr = self.atr(c15, self.atr_period)
        atr15 = float(atr_arr[-1]) if len(atr_arr) and not np.isnan(atr_arr[-1]) else 0.0
        if atr15 <= 0:
            return self._hold(current_price, "15M ATR unavailable")
        ema13 = float(l15["ema_slow_val"])
        if direction == "long" and current_price <= ema13:
            return self._hold(current_price, f"15M cross occurred but price fell back below EMA{self.ema_slow}")
        if direction == "short" and current_price >= ema13:
            return self._hold(current_price, f"15M cross occurred but price reclaimed above EMA{self.ema_slow}")
        dist_atr = abs(float(current_price) - ema13) / atr15
        if dist_atr > self.max_dist_atr_mult:
            return self._hold(current_price, f"15M anti-chase: {dist_atr:.2f}ATR from EMA{self.ema_slow} > {self.max_dist_atr_mult:.2f}")

        rr = min(2.5, self._rr_from_quality(ctx["score"]))
        raw_stop = self._structure_stop_15m(c15, direction, atr15)
        risk_plan = self._compute_entry_sl_tp(direction, float(current_price), raw_stop, atr15,
                                              mirror_raw_stop=False, rr_ratio=rr)
        if risk_plan is None:
            return self._hold(current_price, "15M structure SL outside 0.6–2.0 ATR safety range")
        sl, tp, _risk_distance, risk_atr = risk_plan

        self._last_entry_attempt_bar_ts = bar_ts
        self._open_position = direction
        self._entry_price = float(current_price)
        self._entry_sl = float(sl)
        self._entry_bar_ts = bar_ts
        self._reverse_cross_arm_after_ts = bar_ts
        self._adopted_after_restart = False
        self._entry_regime = ctx["label"]
        self._tp1_done = False
        self._be_trailed = False
        self._last_exit_bar_ts = None

        self._diag_update(entry_state="ENTRY_READY", direction_15m=f"CROSS_{cross_label}")
        metadata = {
            **self._diag_context,
            "strategy": "EMA_CROSS_15M",
            "entry_type": "EMA_CROSS_15M",
            "entry_tf": "15m",
            "stop_loss": round(float(sl), 8),
            "take_profit": round(float(tp), 8),
            "rr_ratio": round(float(rr), 2),
            "risk_atr": round(float(risk_atr), 3),
            "be_trigger_r": 0.8,
            "be_lock_r": 0.5,
            "sizing_mode": self.sizing_mode,
            "margin_pct": self.margin_pct,
            "macro_4h": macro,
            "quality_1h": ctx,
            "quality_15m": q15,
            "sideways_15m": sideways,
            "cross_15m": cross_label,
            "distance_from_ema13_atr": round(dist_atr, 3),
            "data_quality": data_quality,
        }
        reason = (f"4H {macro_display} + 1H {ctx['label']} quality {ctx['score']:.0f} + "
                  f"15M EMA{self.ema_fast}/{self.ema_slow} cross {cross_label}; TP={rr:.1f}R")
        return Signal(
            type=SignalType.BUY if direction == "long" else SignalType.SELL,
            symbol=self.symbol, price=float(current_price), amount=0.0,
            reason=reason, confidence=min(1.0, max(0.55, ctx["score"] / 100.0)), metadata=metadata,
        )

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Manage one hard TP/SL, +0.8R -> SL +0.5R, and CLOSED 15M reverse-cross exit."""
        if self._open_position is None:
            return None
        from ..engines.position_manager import PositionUpdate

        # Reverse-cross exit uses CLOSED 15M candles only and must be FRESH:
        # the cross bar must be strictly newer than the entry/restart reference
        # bar.  This prevents a historical cross from closing a freshly opened
        # or just-reconciled position.  We also require the close to finish on
        # the wrong side of EMA13 to reject tiny touch/cross whipsaws.
        candles = self._latest_15m or self._latest_candles
        if candles:
            bar_ts = int(candles[-1].timestamp)
            arm_after = self._reverse_cross_arm_after_ts
            fresh_bar = arm_after is None or bar_ts > int(arm_after)
            if fresh_bar and bar_ts != self._last_exit_bar_ts:
                l15 = self._layer3_indicators(candles)
                if l15 is not None:
                    self._last_exit_bar_ts = bar_ts
                    crossback = (l15["ema_cross_down"] if self._open_position == "long"
                                 else l15["ema_cross_up"])
                    close_px = float(candles[-1].close)
                    ema13 = float(l15["ema_slow_val"])
                    close_confirm = (close_px < ema13 if self._open_position == "long"
                                     else close_px > ema13)
                    if crossback and close_confirm:
                        side = self._open_position
                        self._last_signal_exit_ts = bar_ts
                        self._last_signal_exit_direction = side
                        self._reset_position_state()
                        return PositionUpdate(
                            action="close", close_pct=1.0,
                            reason=(f"15M EMA{self.ema_fast}/{self.ema_slow} fresh reverse cross "
                                    f"+ close past EMA{self.ema_slow} — close {side.upper()} and "
                                    "wait for new trend-aligned cross"),
                        )

        # No partial close. Once +0.8R is touched, lock +0.5R and keep full size.
        if (self.use_be_trail and not self._be_trailed
                and self._entry_price is not None and self._entry_sl is not None):
            r = abs(float(self._entry_price) - float(self._entry_sl))
            if r > 0:
                hit = ((self._open_position == "long"
                        and current_price >= self._entry_price + self.be_trail_trigger_r * r)
                       or (self._open_position == "short"
                           and current_price <= self._entry_price - self.be_trail_trigger_r * r))
                if hit:
                    self._be_trailed = True
                    new_sl = (self._entry_price + self.be_trail_sl_r * r
                              if self._open_position == "long"
                              else self._entry_price - self.be_trail_sl_r * r)
                    return PositionUpdate(
                        action="move_sl", new_sl=float(new_sl),
                        reason=f"+{self.be_trail_trigger_r:.1f}R reached — SL -> +{self.be_trail_sl_r:.1f}R; full position remains open",
                    )

        return PositionUpdate(
            action="hold",
            reason=f"Holding {self._open_position.upper()} — hard TP/SL active; waiting for 15M reverse cross",
        )

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
        self._entry_bar_ts = None
        self._reverse_cross_arm_after_ts = None
        self._adopted_after_restart = False
        self._entry_regime = None
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
        """Seed in-memory state for a live position recovered after restart."""
        self._open_position = direction
        self._entry_price = entry_price
        self._entry_sl = stop_loss
        bars = self._latest_15m or self._latest_candles
        # We do not know the original signal-bar timestamp after a process
        # restart.  Use the latest ALREADY CLOSED 15M bar as the arming
        # baseline.  Reverse-cross exit becomes eligible only when a newer
        # 15M bar closes, so the bot cannot immediately act on old history.
        restart_ref_ts = int(bars[-1].timestamp) if bars else None
        self._entry_bar_ts = restart_ref_ts
        self._reverse_cross_arm_after_ts = restart_ref_ts
        self._adopted_after_restart = True
        self._entry_regime = None
        self._tp1_done = False
        self._be_trailed = False
        self._last_exit_bar_ts = restart_ref_ts

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
        # A confirmed 4H conflict is a macro hard veto.  The previous version
        # only rejected when 1H and 4H disagreed simultaneously, which allowed
        # repeated longs inside a confirmed 4H bear structure.  A lone 1H pivot
        # conflict remains a quality penalty because the 1H EMA trend can lead
        # the slower pivot sequence during a legitimate transition.
        if four_h_conflict:
            return {**self._neutral_location_context(), "valid": False,
                    "reason": f"4H confirmed {opposite_structure.upper()} structure conflicts with entry",
                    "structure_1h": structure_1h, "structure_4h": structure_4h,
                    "macro_alignment": "CONFLICT"}
        if one_h_conflict:
            context_penalty += self.htf_single_conflict_penalty
            context_reasons.append(f"1H {opposite_structure} structure conflict")

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

    def _price_action_structure_setup(self, candles: list, direction: str, atr_val: float,
                                      regime_state: Optional[str] = None) -> Optional[dict]:
        """Fast but selective price-action structure entry.

        Looks for a liquidity sweep/reclaim and/or micro BOS/CHOCH on the newest
        few bars. A directional candle with close-quality and participation is
        required. TRANSITION raises the bar rather than disabling the engine.
        """
        if atr_val <= 0 or len(candles) < max(12, self.volume_sma_period + 2):
            return None
        n = len(candles)
        start = max(2, n - self.entry_trigger_valid_bars)
        transition = regime_state == "TRANSITION"
        body_min = self.pa_min_body_atr + (0.05 if transition else 0.0)
        close_min = self.pa_min_close_quality + (0.05 if transition else 0.0)
        vol_min = self.pa_min_volume_ratio + (0.10 if transition else 0.0)
        for idx in range(n - 1, start - 1, -1):
            c = candles[idx]
            prev = candles[idx - 1]
            rng = max(c.high - c.low, 1e-12)
            body_atr = abs(c.close - c.open) / atr_val
            close_q = ((c.close - c.low) / rng if direction == "long" else (c.high - c.close) / rng)
            hist = [x.volume for x in candles[max(0, idx - self.volume_sma_period):idx]]
            base_vol = float(np.mean(hist)) if hist else 0.0
            vol_ratio = c.volume / base_vol if base_vol > 0 else 0.0
            lb = min(5, idx)
            prior = candles[idx-lb:idx]
            if len(prior) < 2:
                continue
            prior_high = max(x.high for x in prior)
            prior_low = min(x.low for x in prior)
            events = []
            if direction == "long":
                sweep = prev.low < min(x.low for x in candles[max(0, idx-6):idx-1]) if idx >= 3 else False
                reclaim = prev.low < prior_low + 0.20 * atr_val and c.close > prev.high
                micro_bos = c.close > prior_high and c.close > c.open
                directional = c.close > c.open
                raw_stop = min(prev.low, c.low) - self.entry_stop_buffer_atr * atr_val
            else:
                sweep = prev.high > max(x.high for x in candles[max(0, idx-6):idx-1]) if idx >= 3 else False
                reclaim = prev.high > prior_high - 0.20 * atr_val and c.close < prev.low
                micro_bos = c.close < prior_low and c.close < c.open
                directional = c.close < c.open
                raw_stop = max(prev.high, c.high) + self.entry_stop_buffer_atr * atr_val
            if sweep: events.append("liquidity_sweep")
            if reclaim: events.append("reclaim")
            if micro_bos: events.append("micro_bos")
            structural = micro_bos or (sweep and reclaim)
            if not (structural and directional and body_atr >= body_min and close_q >= close_min and vol_ratio >= vol_min):
                continue
            edge = 72.0 + (3.0 if micro_bos else 0.0) + (2.0 if sweep and reclaim else 0.0)
            return {"trigger_ts": c.timestamp, "raw_stop": float(raw_stop), "events": events,
                    "body_atr": round(body_atr, 3), "close_quality": round(close_q, 3),
                    "volume_ratio": round(vol_ratio, 3), "edge_score": round(edge, 2)}
        return None

    def _early_structure_setup(self, candles: list, direction: str, atr_val: float,
                               regime_state: Optional[str] = None) -> Optional[dict]:
        """ARM -> TRIGGER execution: sweep/rejection first, then the first
        micro structure break. This is intentionally independent of EMA cross,
        so it can enter near the first HL/LH while HTF context still controls
        trade direction."""
        look = self.early_structure_sweep_lookback
        if atr_val <= 0 or len(candles) < look + self.early_structure_micro_bos_lookback + 3:
            return None
        n = len(candles)
        trig_start = max(2, n - self.early_structure_max_signal_age_bars)
        for idx in range(n - 1, trig_start - 1, -1):
            trigger = candles[idx]
            bos_lb = min(self.early_structure_micro_bos_lookback, idx)
            prior_bos = candles[idx-bos_lb:idx]
            if not prior_bos:
                continue
            micro_break = (trigger.close > max(x.high for x in prior_bos) if direction == "long"
                           else trigger.close < min(x.low for x in prior_bos))
            if not micro_break:
                continue
            # Find the most recent sweep/rejection in the arm window before trigger.
            arm_lo = max(look, idx - look)
            for arm_idx in range(idx - 1, arm_lo - 1, -1):
                arm = candles[arm_idx]
                prior = candles[arm_idx-look:arm_idx]
                if len(prior) < look:
                    continue
                body = abs(arm.close - arm.open)
                if direction == "long":
                    swept = arm.low < min(x.low for x in prior)
                    wick = min(arm.open, arm.close) - arm.low
                    rejected = arm.close > min(x.low for x in prior) and wick >= max(body, self.early_structure_min_rejection_wick_atr * atr_val)
                    raw_stop = arm.low - self.entry_stop_buffer_atr * atr_val
                    trigger_dir = trigger.close > trigger.open
                else:
                    swept = arm.high > max(x.high for x in prior)
                    wick = arm.high - max(arm.open, arm.close)
                    rejected = arm.close < max(x.high for x in prior) and wick >= max(body, self.early_structure_min_rejection_wick_atr * atr_val)
                    raw_stop = arm.high + self.entry_stop_buffer_atr * atr_val
                    trigger_dir = trigger.close < trigger.open
                if not (swept and rejected and trigger_dir):
                    continue
                events = ["liquidity_sweep", "rejection", "micro_choch"]
                edge = 74.0 + (2.0 if regime_state == "STRONG_TREND" else 0.0)
                return {"trigger_ts": trigger.timestamp, "arm_ts": arm.timestamp,
                        "raw_stop": float(raw_stop), "events": events,
                        "edge_score": round(edge, 2)}
        return None

    def _entry_tf_ms(self) -> int:
        return {"5m": 5 * 60_000, "15m": 15 * 60_000, "30m": 30 * 60_000,
                "1h": 60 * 60_000}.get(self.entry_tf, 5 * 60_000)

    @staticmethod
    def _closed_candle_series(candles: list, timeframe_ms: int, grace_ms: int = 1500) -> list:
        """Drop a currently-forming final candle when timestamps are bar-open times.

        Historical/backtest candles are retained.  This prevents intrabar EMA
        crosses that disappear before the 5M close from becoming live orders.
        """
        if not candles:
            return []
        last_ts = int(candles[-1].timestamp)
        last_ms = last_ts * 1000 if last_ts < 10_000_000_000 else last_ts
        now_ms = int(time.time() * 1000)
        if last_ms + int(timeframe_ms) > now_ms - max(0, int(grace_ms)):
            return candles[:-1]
        return candles

    @staticmethod
    def _candle_close_quality(candle, direction: str) -> float:
        rng = max(float(candle.high) - float(candle.low), 1e-12)
        if direction == "long":
            return max(0.0, min(1.0, (float(candle.close) - float(candle.low)) / rng))
        return max(0.0, min(1.0, (float(candle.high) - float(candle.close)) / rng))

    def _precision_ema_cross_setup(self, candles: list, direction: str, cross_ts: Optional[int],
                                   current_price: float, atr_val: float,
                                   regime_state: Optional[str]) -> dict:
        """Validate a fresh closed-bar EMA cross or its first retest/reclaim."""
        if not candles or cross_ts is None or atr_val <= 0:
            return {"valid": False, "reason": "waiting for fresh confirmed EMA cross"}
        idx = next((i for i in range(len(candles) - 1, -1, -1)
                    if candles[i].timestamp == cross_ts), None)
        if idx is None:
            return {"valid": False, "reason": "cross timestamp not present in closed candle series"}
        age = len(candles) - 1 - idx
        if age < 0 or age > self.cross_valid_bars:
            return {"valid": False, "reason": f"cross stale ({age} bars old)"}

        closes = [float(c.close) for c in candles]
        volumes = [max(0.0, float(getattr(c, "volume", 0.0) or 0.0)) for c in candles]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        atrs = self.atr(candles, self.atr_period)
        if any(np.isnan(x) for x in (ema_f[-1], ema_s[-1], atrs[-1])):
            return {"valid": False, "reason": "EMA execution indicators warming up"}

        slope_lb = min(self.ema_slow_slope_lookback, len(candles) - 1)
        slow_slope = (float(ema_s[-1]) - float(ema_s[-1 - slope_lb])) / atr_val
        gap_atr = abs(float(ema_f[-1]) - float(ema_s[-1])) / atr_val
        gap_prev = abs(float(ema_f[-2]) - float(ema_s[-2])) / atr_val
        gap_expanding = gap_atr >= gap_prev
        side_ok = (current_price > float(ema_s[-1]) and closes[-1] > float(ema_s[-1])
                   if direction == "long" else
                   current_price < float(ema_s[-1]) and closes[-1] < float(ema_s[-1]))
        slope_ok = slow_slope > 0 if direction == "long" else slow_slope < 0
        if not side_ok:
            return {"valid": False, "reason": f"price not holding the trend side of EMA{self.ema_slow}"}
        if not slope_ok:
            return {"valid": False, "reason": f"EMA{self.ema_slow} slope has not turned with the cross"}
        if gap_atr < self.ema_min_gap_atr and age > 0:
            return {"valid": False, "reason": f"EMA gap too weak ({gap_atr:.2f}ATR)"}
        if age > 0 and not gap_expanding:
            return {"valid": False, "reason": "EMA separation is contracting after the cross"}

        last = candles[-1]
        body_atr = abs(float(last.close) - float(last.open)) / atr_val
        close_quality = self._candle_close_quality(last, direction)
        vol_window = volumes[max(0, len(volumes) - self.volume_sma_period - 1):-1]
        vol_base = float(np.mean(vol_window)) if vol_window else 0.0
        vol_ratio = volumes[-1] / vol_base if vol_base > 0 else 1.0
        candle_direction_ok = (last.close > last.open if direction == "long" else last.close < last.open)
        min_volume = self.ema_min_volume_ratio + (0.15 if regime_state == "TRANSITION" else 0.0)
        min_close_quality = self.ema_min_close_quality + (0.05 if regime_state == "TRANSITION" else 0.0)
        if body_atr < self.ema_min_body_atr:
            return {"valid": False, "reason": f"cross confirmation body too small ({body_atr:.2f}ATR)"}
        if close_quality < min_close_quality:
            return {"valid": False, "reason": f"weak candle close quality ({close_quality:.2f})"}
        if vol_ratio < min_volume:
            return {"valid": False, "reason": f"participation too low ({vol_ratio:.2f}x volume)"}
        if regime_state == "TRANSITION" and not candle_direction_ok:
            return {"valid": False, "reason": "TRANSITION requires a directional confirmation candle"}

        cross_bar = candles[idx]
        cross_atr = float(atrs[idx]) if idx < len(atrs) and not np.isnan(atrs[idx]) else atr_val
        cross_body_atr = abs(float(cross_bar.close) - float(cross_bar.open)) / max(cross_atr, 1e-12)
        cross_extension = abs(float(cross_bar.close) - float(ema_s[idx])) / max(cross_atr, 1e-12)
        current_extension = abs(float(current_price) - float(ema_s[-1])) / atr_val
        impulse_cross = (cross_body_atr > self.ema_impulse_body_atr or
                         cross_extension > self.ema_max_extension_slow_atr)
        if current_extension > self.ema_max_extension_slow_atr:
            return {"valid": False, "reason": f"price extended {current_extension:.2f}ATR from EMA{self.ema_slow}; wait retest"}

        retest = False
        if age > 0:
            for j in range(idx + 1, len(candles)):
                a = float(atrs[j]) if not np.isnan(atrs[j]) else atr_val
                tol = self.ema_retest_tolerance_atr * a
                q = self._candle_close_quality(candles[j], direction)
                if direction == "long":
                    touched = float(candles[j].low) <= float(ema_s[j]) + tol
                    reclaimed = float(candles[j].close) > float(ema_s[j]) and candles[j].close > candles[j].open
                else:
                    touched = float(candles[j].high) >= float(ema_s[j]) - tol
                    reclaimed = float(candles[j].close) < float(ema_s[j]) and candles[j].close < candles[j].open
                if touched and reclaimed and q >= min_close_quality:
                    retest = True
                    break
        if (impulse_cross or age > 0) and not retest:
            return {"valid": False, "reason": "waiting for first EMA retest + reclaim after cross"}

        execution = "first retest + reclaim" if retest else "direct closed-bar cross"
        edge = 70.0
        edge += min(4.0, gap_atr * 8.0)
        edge += 2.0 if gap_expanding else 0.0
        edge += min(3.0, max(0.0, vol_ratio - 0.8) * 3.0)
        edge += 2.0 if retest else 0.0
        return {
            "valid": True, "reason": "confirmed", "cross_bars_ago": age,
            "execution": execution, "ema_gap_atr": round(gap_atr, 3),
            "ema_slow_slope_atr": round(slow_slope, 3),
            "body_atr": round(body_atr, 3),
            "close_quality": round(close_quality, 3),
            "volume_ratio": round(vol_ratio, 3),
            "price_extension_atr": round(current_extension, 3),
            "edge_score": round(edge, 2),
        }

    @staticmethod
    def _micro_structure_failure(candles: list, side: str, lookback: int) -> bool:
        if len(candles) < lookback + 2:
            return False
        last = candles[-1]
        prior = candles[-lookback - 1:-1]
        if side == "long":
            return float(last.close) < min(float(c.low) for c in prior)
        return float(last.close) > max(float(c.high) for c in prior)

    def _target_rr(self, entry_type: str, regime_state: Optional[str]) -> float:
        state = regime_state if regime_state in ("TRANSITION", "TREND", "STRONG_TREND") else "TREND"
        table = {
            "EMA_CROSS": {"TRANSITION": self.ema_rr_transition, "TREND": self.ema_rr_trend, "STRONG_TREND": self.ema_rr_strong},
            "BREAKOUT_RETEST": {"TRANSITION": self.breakout_rr_transition, "TREND": self.breakout_rr_trend, "STRONG_TREND": self.breakout_rr_strong},
            "STRUCTURE_RETEST": {"TRANSITION": self.structure_rr_transition, "TREND": self.structure_rr_trend, "STRONG_TREND": self.structure_rr_strong},
            "PA_STRUCTURE_CONFIRM": {"TRANSITION": self.pa_rr_transition, "TREND": self.pa_rr_trend, "STRONG_TREND": self.pa_rr_strong},
            "EARLY_STRUCTURE": {"TRANSITION": self.early_rr_transition, "TREND": self.early_rr_trend, "STRONG_TREND": self.early_rr_strong},
        }
        return max(0.5, float(table.get(entry_type, {}).get(state, self.rr_ratio)))

    def _compute_entry_sl_tp(self, direction: str, price: float, raw_stop: float,
                             atr_val: float, mirror_raw_stop: bool = False,
                             rr_ratio: Optional[float] = None
                             ) -> Optional[tuple[float, float, float, float]]:
        """Normalize a 15M structure stop using ATR only.

        The raw stop comes from the latest confirmed 15M swing high/low plus
        ``entry_stop_buffer_atr``.  A fixed percentage-of-price floor is
        intentionally NOT used because 0.5% represents very different noise
        across XAU, BTC, XRP, HYPE, etc.

        Stops narrower than ``entry_min_stop_atr`` are widened to the ATR floor;
        stops wider than ``entry_max_stop_atr`` are rejected instead of forcing
        an oversized risk distance.
        """
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

        # 15M volatility-normalized risk bounds.  Keep min_sl_pct only as a
        # backwards-compatible config field; it no longer changes live stops.
        min_distance = self.entry_min_stop_atr * atr_val
        max_distance = self.entry_max_stop_atr * atr_val
        distance = max(raw_distance, min_distance)
        if distance > max_distance:
            return None

        rr = max(0.5, float(self.rr_ratio if rr_ratio is None else rr_ratio))
        if direction == "long":
            sl = price - distance
            tp = price + distance * rr
        else:
            sl = price + distance
            tp = price - distance * rr
        return sl, tp, distance, distance / atr_val

    @staticmethod
    def _entry_reason_summary(candidate: dict) -> str:
        detail = candidate.get("detail", {})
        et = candidate.get("entry_type")
        if et == "EMA_CROSS":
            mode = detail.get("execution", "direct")
            return (f"confirmed EMA cross {detail.get('cross_bars_ago')} bar(s) ago; "
                    f"{mode}; gap {detail.get('ema_gap_atr')}ATR; "
                    f"volume {detail.get('volume_ratio')}x")
        if et == "BREAKOUT_RETEST":
            return (f"breakout level {detail.get('breakout_level')} retested; "
                    f"volume {detail.get('volume_ratio')}x; "
                    f"confirm {','.join(detail.get('confirmations', []))}")
        if et == "STRUCTURE_RETEST":
            return (f"{detail.get('structure')} structure retest at {detail.get('retest_level')}; "
                    f"confirm {','.join(detail.get('confirmations', []))}")
        if et == "PA_STRUCTURE_CONFIRM":
            return (f"PA structure {','.join(detail.get('events', []))}; "
                    f"body {detail.get('body_atr')}ATR, volume {detail.get('volume_ratio')}x")
        if et == "EARLY_STRUCTURE":
            return (f"sweep/rejection -> micro structure shift; "
                    f"confirm {','.join(detail.get('events', []))}")
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
        slope_lb = min(self.ema_slow_slope_lookback, len(candles) - 1)
        gap_now = abs(float(ema_f[-1]) - float(ema_s[-1])) / float(atr_arr[-1])
        gap_prev = abs(float(ema_f[-2]) - float(ema_s[-2])) / float(atr_arr[-1])
        slow_slope_atr = (float(ema_s[-1]) - float(ema_s[-1 - slope_lb])) / float(atr_arr[-1])
        return {
            "ema_cross_up":   ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1],
            "ema_cross_down": ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1],
            "ema_fast_val": float(ema_f[-1]),
            "ema_slow_val": float(ema_s[-1]),
            "ema_gap_atr": gap_now,
            "ema_gap_expanding": gap_now >= gap_prev,
            "ema_slow_slope_atr": slow_slope_atr,
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
        # Every HOLD carries a complete, stable diagnostic envelope. The
        # dashboard/logger can therefore render meaningful values instead of
        # "?" even when the scan returns early during warm-up or a gate.
        merged = dict(getattr(self, "_diag_context", {}) or {})
        incoming = dict(metadata or {})
        merged.update(incoming)

        # Existing callers sometimes pass the full regime dict. Preserve it
        # under regime_detail but expose a short scalar under regime, which is
        # what compact scan logs expect.
        regime_value = merged.get("regime")
        if isinstance(regime_value, dict):
            merged["regime_detail"] = regime_value
            merged["regime"] = regime_value.get("state", "UNKNOWN")

        merged.setdefault("regime", "WARMUP")
        merged.setdefault("trend_4h", "N/A")
        merged.setdefault("trend_1h", "WARMUP")
        merged.setdefault("aligned", False)
        merged.setdefault("mtf", "WARMUP")
        merged.setdefault("strategy", "EMA_CROSS_15M")
        merged.setdefault("entry_tf", "15m")
        merged.setdefault("direction_15m", "WARMUP")
        merged.setdefault("entry_state", "HOLD")
        merged["hold_reason"] = reason

        # Compatibility aliases for older TradingBot scan formatters.
        # Informational only; never used by the trading decision path.
        _t4 = str(merged.get("trend_4h", "N/A"))
        _t1 = str(merged.get("trend_1h", "WARMUP"))
        merged.setdefault("macro_trend", {"bias": _t4, "stage": "INFO", "score": 0.0})
        merged.setdefault("context_1h", {
            "dominant_bias": _t1,
            "stage": str(merged.get("direction_15m", "WARMUP")),
        })
        _a14 = (_t4 == _t1) if _t4 in ("UP", "DOWN") and _t1 in ("UP", "DOWN") else None
        merged.setdefault("mtf_combined", {
            "aligned_1h_4h": _a14,
            "pct": 100.0 if _a14 is True else (-100.0 if _a14 is False else 0.0),
        })
        merged.setdefault("selected_strategy", merged.get("strategy", "EMA_CROSS_15M"))

        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata=merged,
        )
