"""
Configuration — every tunable lives here. Only the ENV vars listed in the
project spec are read from the environment; everything else is a sane
hardcoded default (same "minimal Railway surface" philosophy as the
TrendContV2 bot).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def _env_list(key: str, default: str) -> list[str]:
    return [s.strip() for s in os.environ.get(key, default).split(",") if s.strip()]


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    return default if not val else val.lower() in ("1", "true", "yes")


def _env_first(*keys: str, default: str = "") -> str:
    """First non-empty env var among `keys`, in order — lets a Railway service
    already configured under the old bot's variable names (EXCHANGE_API_KEY
    etc.) keep working without the user having to rename anything."""
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            return v
    return default


@dataclass
class Config:
    # ── Exchange (ENV) ────────────────────────────────────────────────────────
    # OKX_* is the primary name; EXCHANGE_* is accepted as a fallback alias
    # (the name the previous bot on this Railway service used).
    okx_api_key: str        = field(default_factory=lambda: _env_first("OKX_API_KEY", "EXCHANGE_API_KEY"))
    okx_secret: str          = field(default_factory=lambda: _env_first("OKX_SECRET", "EXCHANGE_API_SECRET"))
    okx_passphrase: str      = field(default_factory=lambda: _env_first("OKX_PASSPHRASE", "EXCHANGE_PASSPHRASE"))
    paper: bool              = field(default_factory=lambda: _env_bool("PAPER_TRADING", False))

    # ── Telegram (ENV) ────────────────────────────────────────────────────────
    telegram_bot_token: str  = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str    = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))

    # ── Trading universe (ENV) ───────────────────────────────────────────────
    symbols: list[str]       = field(default_factory=lambda: _env_list(
        "SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT"))
    leverage: int             = field(default_factory=lambda: _env_int("LEVERAGE", 20))
    # Fixed 5% risk per trade (spec). The regime/context size_multiplier scales
    # DOWN from here in weaker conditions; it never scales up past this.
    risk_per_trade: float     = field(default_factory=lambda: _env_float("RISK_PER_TRADE", 0.05))
    # Hard cap on TOTAL concurrent positions across all symbols (not per-symbol).
    # Same env var name the previous bot on this Railway service used.
    max_positions_env: int   = field(default_factory=lambda: _env_int("MAX_POSITIONS", 2))

    # ── Timeframes (ENV, default per spec) ───────────────────────────────────
    # tf_entry/TIMEFRAME_ENTRY = Entry Layer 3.1's quality pre-filter
    # timeframe; the actual timing trigger (Layer 3.3) runs on tf_fast (15M).
    tf_entry: str  = field(default_factory=lambda: os.environ.get("TIMEFRAME_ENTRY", "30m"))
    tf_bias: str   = field(default_factory=lambda: os.environ.get("TIMEFRAME_BIAS", "1h"))
    tf_regime: str = field(default_factory=lambda: os.environ.get("TIMEFRAME_REGIME", "4h"))
    tf_fast: str    = "15m"   # Bias secondary + Entry timeframe
    tf_micro: str   = "5m"    # Bias tertiary timeframe

    # ── Hardcoded strategy constants (not exposed as ENV) ───────────────────
    market_type: str = "swap"
    margin_mode: str = "isolated"
    exchange_id: str = "okx"

    min_bars: int = 100          # skip trading a symbol if any TF has fewer closed bars

    # Regime engine (4h)
    regime_ema_fast: int = 10   # was 20 — faster pair (10/20) per user request
    regime_ema_slow: int = 20   # was 50
    regime_ema_slope_lookback: int = 5
    regime_adx_period: int = 14
    regime_chop_period: int = 14
    regime_atr_period: int = 14
    regime_atr_pct_lookback: int = 100
    regime_trend_score_min: float = 70.0
    regime_early_trend_score_min: float = 55.0
    regime_adx_trend_lo: float = 18.0
    regime_adx_trend_hi: float = 40.0
    regime_chop_range_min: float = 60.0
    regime_atr_pct_high_vol: float = 85.0
    regime_atr_pct_compression: float = 20.0
    regime_chop_compression_min: float = 55.0
    regime_score_min_to_trade: float = 55.0
    regime_compression_breakout_lookback: int = 20
    regime_high_vol_size_cut: float = 0.30      # reduce position size 30% in HIGH_VOLATILITY
    regime_transition_entry_relax: float = 0.0  # TRANSITION must NEVER relax entry (user note #3)
    regime_transition_bias_tighten: float = 10.0  # TRANSITION: bias threshold +10

    # Bias engine (1h)
    bias_ema_fast: int = 20
    bias_ema_slow: int = 50
    bias_roc_period: int = 9
    bias_structure_left: int = 3
    bias_structure_right: int = 3
    bias_score_min: float = 60.0

    # ══ Entry Engine — 3 sequential sub-layers, per user spec ═══════════════
    #   Layer 3.1  15M, 5-category quality pre-filter (>= entry_min_categories,
    #              Momentum+Structure mandatory) — NOT a timing trigger.
    #   Layer 3.2  15M+5M, prior-acceleration wait-rounds — holds (not
    #              rejects) a pending Layer 3.3 trigger if the market just
    #              moved too violently, until follow-through confirms it.
    #   Layer 3.3  15M, HMA10/HMA16 fresh-cross timing trigger + anti-chase +
    #              one-entry-per-cross state machine. Decides WHEN, once 3.1
    #              and 3.2 clear.
    # ── Entry (Layer 3) — 3-layer multi-timeframe cross confluence ──────────
    # Regime (4H+1H) and Bias (1H+15M+5M) fix the SIDE. Three cross layers,
    # each on its own timeframe, watch for a cross in the bias direction:
    #   L3a — HMA10/16 cross on 30M
    #   L3b — EMA5/9  cross on 15M
    #   L3c — EMA10/20 cross on 5M   (also the hard-exit layer)
    # Any single cross ARMS a setup; the entry fires once >= entry_confluence_min
    # (2) of the three layers have crossed the SAME (bias) direction within
    # entry_confluence_window_min (45) minutes of each other. If a second layer
    # doesn't confirm within the window, the first cross ages out -> setup fails,
    # wait for a new one. One entry per setup (needs a genuinely newer cross to
    # re-arm after a fill/close).
    entry_confluence_min: int = 2
    entry_confluence_window_min: int = 45
    # L3a — HMA cross, 30M
    l3a_tf: str = "30m"
    l3a_hma_fast: int = 10
    l3a_hma_slow: int = 16
    # L3b — EMA cross, 15M
    l3b_tf: str = "15m"
    l3b_ema_fast: int = 5
    l3b_ema_slow: int = 9
    # L3c — EMA cross, 5M (entry confluence member AND the hard-exit gate)
    l3c_tf: str = "5m"
    l3c_ema_fast: int = 10
    l3c_ema_slow: int = 20
    # Early-exit grace: the L3c hard exit (EMA10/20 cross-back / open past
    # EMA20 on 5M) is SUPPRESSED for this many closed 5M bars after entry so
    # the EMAs can separate. SL/TP stay active throughout. 0 disables.
    exit_grace_bars: int = 3
    one_entry_per_cross: bool = True
    require_new_cross_after_exit: bool = True
    entry_on_closed_candle_only: bool = True
    exit_on_closed_candle_price_break: bool = True

    # Retained field names from earlier entry systems — no longer used by the
    # entry logic, kept as inert defaults so env overrides, the chart HMA
    # overlay, and the dead context/style/booster modules don't error.
    entry_timeframe: str = "5m"
    hma_fast_length: int = 10
    hma_slow_length: int = 16
    entry_ema_fast: int = 5
    entry_ema_slow: int = 9
    entry_macd_fast: int = 12
    entry_macd_slow: int = 26
    entry_macd_signal: int = 9
    entry_ema_ref: int = 15
    entry_min_categories: int = 3
    entry_roc_period: int = 9
    entry_sweep_lookback: int = 10
    entry_wick_reject_frac: float = 0.5
    entry_vol_expansion_mult: float = 1.5
    entry_rel_vol_min: float = 1.2

    # ── Bias confidence (1H) ─────────────────────────────────────────────────
    # Confidence 0-100 from: 1H ADX, RSI slope, EMA20 slope, volume confirm,
    # pullback zone. High confidence relaxes the entry threshold slightly;
    # low confidence tightens it.
    bias_conf_adx_strong: float = 25.0   # ADX >= this -> full ADX points
    bias_conf_adx_ok: float = 18.0       # ADX >= this -> partial ADX points
    bias_conf_high: float = 70.0         # conf >= this -> entry_thr - 5
    bias_conf_low: float = 40.0          # conf <  this -> entry_thr + 10
    bias_conf_high_adj: float = -5.0
    bias_conf_low_adj: float = 10.0

    # Risk manager
    risk_min_pct: float = 0.05
    risk_max_pct: float = 0.10
    # Disabled by user request: at risk_per_trade=5%, one SL loss is already
    # ~5% of the day-start balance — past this 3% limit on the FIRST loss of
    # the day, freezing all new entries until next UTC day (up to ~24h idle
    # off a single trade). The loss-streak cooldown below is the intended
    # brake instead.
    daily_loss_limit_enabled: bool = False
    daily_loss_limit_pct: float = 0.03    # halt new entries: day PnL <= -3% (only if enabled above)
    # Disabled by user request ("trade continuously, no waiting for the next
    # UTC day") — a +10.8% day tripped the +8% lock and idled the bot ~13h.
    # Loss-streak cooldown below remains the only pause.
    daily_profit_lock_enabled: bool = False
    daily_profit_lock_pct: float = 0.08   # halt new entries: day PnL >= +8% (only if enabled above)
    loss_streak_limit: int = 3            # 3 consecutive losses ->
    loss_streak_cooldown_min: int = 180   # -> 3-hour cooldown (was 30 min)
    max_open_positions: int = 0  # set in __post_init__ from MAX_POSITIONS (default 2)

    # ── Fees ─────────────────────────────────────────────────────────────────
    # OKX charges 0.10% per fill on this account — open, close, TP and SL all
    # pay it. Used by live/paper PnL accounting AND the backtest, so the two
    # can never disagree on fee drag.
    fee_rate: float = 0.001

    # ── Commodity market hours (XAU / XAG) ──────────────────────────────────
    # The underlying metals market is closed on weekends — the OKX perp still
    # quotes, but it's illiquid/frozen and signals are garbage. Block NEW
    # entries for symbols matching these keywords from Friday
    # commodity_halt_hour_utc (17:00 UTC = Sat 00:00 Asia/Bangkok) until
    # Sunday commodity_resume_hour_utc (21:00 UTC = Mon 04:00 ICT — exactly 3
    # hours before the Mon 07:00 ICT market open the user referenced).
    # Open positions keep being managed (SL/TP/exits) throughout.
    commodity_weekend_block_enabled: bool = True
    commodity_symbol_keywords: tuple = ("XAU", "XAG")
    commodity_halt_hour_utc: int = 17     # Friday >= this hour UTC -> halted
    commodity_resume_hour_utc: int = 21   # Sunday < this hour UTC -> still halted

    # Stop loss / take profit
    sl_atr_period: int = 14
    sl_atr_mult: float = 1.5
    # Floor/ceiling on SL distance as % of entry price. The floor exists so a
    # quiet-candle ATR stop can never come out so tight that TP1/TP2 R-multiple
    # profit targets fail to clear round-trip fees (see calc_stop_loss docstring).
    sl_min_pct: float = 0.004   # 0.4%
    sl_max_pct: float = 0.035  # 3.5%
    # Pulls the final SL distance in to this fraction of the ATR/swing/floor
    # calc. MEASURED on the local BTC/XAU set (Jan-May 2026): 0.85 made every
    # metric WORSE (PF 0.647->0.520, WR 64.4%->61.9%, net -17828->-19324,
    # trades/month 106->117) — a tighter stop gets hit by ordinary noise more
    # often (SL-count rose 189->223) since R-based position sizing keeps
    # dollar-risk-per-trade constant regardless of stop width, so tightening
    # only adds re-entry churn and fee drag without reducing risk. Reverted
    # to 1.0 (no tightening).
    sl_tighten_mult: float = 1.0
    tp1_r: float = 0.6   # was 0.5 (per user request)
    # Fraction of the position closed at TP1 (remainder rides to TP2/SL-at-BE).
    # Swept against the live 6-symbol backtest (BTC/ETH/SOL/XAU/XAG, Jan-Jun
    # 2026): the TP2 bucket is the strategy's only real profit source, so a
    # SMALLER TP1 take (bigger runner) improves expected value — 40% beat
    # both 50% (previous default) and 70%/60%@0.6R.
    tp1_fraction: float = 0.6
    # tp2_r = 1.5R (per user request). Paired with TP1 0.6R / 60% closed then
    # SL to breakeven — the 40% runner now targets 1.5R instead of 1.2R.
    tp2_r: float = 1.5
    swing_lookback_left: int = 3
    swing_lookback_right: int = 3

    symbol_cooldown_min: int = 30  # no new entry on a symbol for this long after it closes

    # ── SpikeGuard (fast 5m/15m reversal-spike protection) ───────────────────
    # Runs EVERY poll tick while a position is open — the slow 30m health
    # monitor cannot react to a V-reversal that eats the SL in minutes.
    spike_guard_enabled: bool = True
    spike_5m_atr_mult: float = 2.5     # 5m bar range >= this x ATR14(5m) = spike
    spike_15m_atr_mult: float = 2.0    # 15m bar range >= this x ATR14(15m) = spike
    spike_15m_cum_atr_mult: float = 2.5  # 3-bar cumulative 15m thrust vs ATR
    spike_live_atr_mult: float = 1.8   # live ticker move beyond last 5m close vs ATR
    spike_close_frac: float = 0.7      # spike bar must close in the extreme 30% (momentum, not wick)
    spike_min_adverse_r: float = 0.3   # arm CLOSE only once this deep into the stop (in R)
    spike_hard_atr_mult: float = 3.5   # a spike this big closes regardless of depth
    spike_vol_mult: float = 2.0        # volume >= this x avg20 -> soften ATR bars 20%
    spike_tf_fast: str = "5m"
    spike_tf_slow: str = "15m"
    spike_fetch_limit: int = 60

    # ══ Hard Gate + Soft Score + Adaptive Threshold pipeline ════════════════
    # Layer 1 — Regime (HARD GATE, 4H + 1H)
    regime_block_below_score: float = 60.0     # regime score < this -> block trade
    regime_extension_atr_max: float = 1.5      # |close-EMA20|/ATR above this -> anti-chase block
    regime_weight_4h: float = 0.6              # blend of 4H/1H into the combined regime score
    regime_weight_1h: float = 0.4
    regime_strong_score: float = 85.0          # >= this -> adaptive_threshold_adj = -5
    regime_normal_score: float = 70.0          # 70-84 -> 0 ; 60-69 -> +5
    regime_transition_relax: float = 0.0       # TRANSITION must NEVER relax entry (spec)
    # per-quality size multipliers
    size_mult_strong: float = 1.0
    size_mult_normal: float = 0.85
    size_mult_weak: float = 0.6
    size_mult_transition: float = 0.5

    # ── 6-regime classification (user note #2) ──────────────────────────────
    # Combined 4H+1H score bands -> regime type -> trade STYLE + entry adj.
    regime_strong_trend_min: float = 80.0      # STRONG_TREND  (entry -5)
    regime_healthy_trend_min: float = 65.0     # HEALTHY_TREND (entry 0)
    regime_early_trend_min: float = 55.0       # EARLY_TREND   (needs context/entry confirm)
    regime_range_adx_max: float = 18.0         # RANGE: ADX below this + high chop
    regime_range_chop_min: float = 55.0
    regime_compression_atrpct_max: float = 25.0  # COMPRESSION: low ATR percentile
    regime_strong_trend_adj: float = -5.0
    regime_early_trend_adj: float = 3.0        # EARLY_TREND slightly stricter entry

    # ── Trade styles (user request: 3 styles routed by regime) ──────────────
    # STRONG_TREND / HEALTHY_TREND -> TREND     (with-trend continuation)
    # EARLY_TREND                  -> SWING      (early continuation, needs confirm)
    # RANGE                        -> MEANREV    (fade extremes back to mean)
    # COMPRESSION                  -> BREAKOUT   (wait for expansion breakout)
    # TRANSITION                   -> blocked
    style_range_enabled: bool = True           # allow mean-reversion in RANGE
    style_compression_enabled: bool = True     # allow breakout in COMPRESSION
    # Mean-reversion (RANGE): fade when price is stretched from the mean and RSI
    # is at an extreme, expecting reversion. Direction is COUNTER to the stretch.
    meanrev_rsi_long_max: float = 30.0         # RSI <= this -> oversold, fade up (LONG)
    meanrev_rsi_short_min: float = 70.0        # RSI >= this -> overbought, fade down (SHORT)
    meanrev_ext_atr_min: float = 1.2           # price must be at least this many ATR from EMA20
    meanrev_bias_relax: float = 999.0          # MEANREV ignores the trend-momentum bias gate
    meanrev_entry_threshold: float = 60.0      # its own entry-score bar (reversal trigger)
    meanrev_size_mult: float = 0.6             # smaller size (counter-trend is riskier)
    # Breakout (COMPRESSION): enter on a range break WITH volume expansion.
    breakout_lookback: int = 20
    breakout_vol_mult: float = 1.5
    breakout_entry_threshold: float = 60.0
    breakout_size_mult: float = 0.7

    # Layer 2 — Bias (SOFT confirmation + min gate, 1H + 15M)
    bias_weight_1h: float = 0.7
    bias_weight_15m: float = 0.3
    # Soft confirmation gate, calibrated to the momentum scorer's real scale.
    # On regime-aligned bars the trade-side momentum score has a ~35 base (RSI
    # past the midline + one slope), rising to 60+ only when ROC or MACD also
    # lean the trade's way. 40 admits exactly those "momentum actually aligned"
    # bars and rejects the bounce bars where structure says short but momentum
    # is counter. (The spec's 65 assumed a differently-scaled scorer and blocked
    # everything.) The Entry layer's HMA trigger does the final selection.
    bias_min_threshold: float = 40.0           # weighted trade-side momentum must clear this
    bias_strong_opposite: float = 70.0         # opposite side >= this -> hard veto (NEUTRAL)

    # Layer 3 — Context (SOFT SCORE, 30M) — reweighted per user note #1.
    # CHOCH and Volume Expansion are FUSED into one component: CHOCH alone
    # scores partial, CHOCH + volume expansion scores full. Total = 100.
    context_base_threshold: float = 45.0
    context_thr_strong: float = 40.0           # strong trend -> easiest context bar
    context_thr_normal: float = 45.0
    context_thr_weak: float = 50.0
    context_thr_transition: float = 75.0       # transition is blocked anyway
    context_w_choch_vol: float = 25.0          # CHOCH + volume expansion (fused)
    context_w_choch_partial: float = 12.0      # CHOCH with weak/no volume -> partial
    context_w_vwap: float = 10.0
    context_w_pullback: float = 10.0           # EMA pullback / bounce
    context_w_sweep: float = 15.0              # liquidity sweep
    context_w_retest: float = 10.0             # retest quality
    context_w_session: float = 10.0            # session quality (scaled 0..1)
    context_w_vol_cont: float = 10.0           # volume continuation (expansion in trend dir)
    context_w_breakout: float = 10.0           # breakout quality (break prev extreme)
    context_vol_confirm_mult: float = 1.2      # volume > vol_ma20 x this -> "confirmed"

    # Layer 4 — Entry (30M setup + score + adaptive threshold)
    entry_base_threshold: float = 70.0
    entry_threshold_floor: float = 65.0        # adaptive threshold can never fall below this
    entry_near_miss_floor: float = 65.0        # score in [floor, threshold) -> Early Booster
    entry_context_strong: float = 75.0         # context >= this -> entry_threshold -3
    entry_context_weak: float = 60.0           # context < this  -> entry_threshold +5
    entry_context_strong_adj: float = -3.0
    entry_context_weak_adj: float = 5.0

    # Layer 5 — Early Entry Booster (15M, never trades alone)
    booster_max_bonus_strong: float = 10.0     # regime >= 85
    booster_max_bonus_normal: float = 8.0      # 70-84
    booster_max_bonus_weak: float = 5.0        # 60-69
    booster_max_bonus_transition: float = 4.0
    booster_score_to_bonus: float = 0.5        # early_bonus = min(early_score * this, max_bonus)

    # Loop timing
    poll_interval_sec: int = 30    # how often main.py checks for a newly-closed 30m bar
    reconcile_interval_sec: int = 60   # how often to sweep OKX for untracked positions and adopt them
    status_log_interval_sec: int = 300  # per-symbol regime/bias/entry status log cadence
    fetch_limit_entry: int = 300
    fetch_limit_bias: int = 300
    fetch_limit_regime: int = 300
    fetch_limit_context: int = 300
    fetch_limit_fast: int = 300
    fetch_limit_micro: int = 300

    # ══ Regime -> Bias -> Entry (strict 3-layer "Directional Trading         ══
    # ══ Architecture") ════════════════════════════════════════════════════
    # Layer 1 — Regime classification thresholds (7-way label, 4H+1H).
    rg_adx_trend_min: float = 15.0          # ADX > this (or rising) counts toward Strong Bull/Bear
    rg_compression_pctile_max: float = 25.0 # ATR/BBwidth percentile <= this -> compression signal
    rg_range_adx_max: float = 18.0          # ADX < this -> range/compression signal
    rg_range_flat_slope_pct: float = 0.15   # |EMA20 slope| < this% -> "EMA flat" (range signal)
    rg_highvol_atr_pctile_min: float = 85.0 # ATR percentile >= this -> volatility-expansion signal
    rg_highvol_vol_mult: float = 2.0        # volume >= this x vol_ma20 -> volume-expansion signal
    rg_highvol_range_mult: float = 2.0      # bar range >= this x range_ma20 -> candle-expansion signal

    # Layer 2 — Bias: Dynamic Combined Bias Score (1H + 15M + 5M), weights and
    # pass threshold depend on the Regime tier (Confirmed/Strong trend wants
    # more 1H weight for continuity; Early trend wants more 15M/5M weight to
    # react faster). Every TF must ALSO individually clear a floor and not be
    # flagged the opposite direction — the weighted average alone can't pass.
    bias_rel_vol_min: float = 1.0           # current bar volume / vol_ma20 >= this -> "relative volume" point
    bias_direction_bull_min: float = 55.0   # per-TF score >= this -> that TF's Direction = BULL
    bias_direction_bear_max: float = 45.0   # per-TF score <= this -> that TF's Direction = BEAR
    bias_tf_floor_1h: float = 55.0          # 1H Bull Bias must clear this for LONG (mirror: 100-x for SHORT)
    bias_tf_floor_15m: float = 55.0         # 15M Bull Bias floor
    bias_tf_floor_5m: float = 40.0          # 5M Bull Bias floor (loosest — lowest weight, noisiest TF)
    # weight profile + combined-score pass bar, by Regime tier
    bias_w1h_confirmed: float = 0.50        # STRONG_BULL/BEAR_TREND ("Confirmed Trend") — 1H-heavy for continuity
    bias_w15m_confirmed: float = 0.35
    bias_w5m_confirmed: float = 0.15
    bias_threshold_confirmed: float = 65.0
    bias_w1h_early: float = 0.35            # EARLY_BULL/BEAR_TREND ("Early Trend") — faster-reacting TFs weighted up
    bias_w15m_early: float = 0.45
    bias_w5m_early: float = 0.20
    bias_threshold_early: float = 60.0
    bias_w1h_default: float = 0.45          # fallback weight profile (regime not a recognized trend tier)
    bias_w15m_default: float = 0.40
    bias_w5m_default: float = 0.15
    bias_threshold_default: float = 60.0

    def __post_init__(self):
        self.risk_per_trade = max(self.risk_min_pct, min(self.risk_max_pct, self.risk_per_trade))
        # Hard cap from MAX_POSITIONS, independent of how many symbols are
        # configured — trading 5 symbols with MAX_POSITIONS=2 still means at
        # most 2 concurrent positions total, not 5.
        self.max_open_positions = max(1, self.max_positions_env)

    def validate_live(self) -> list[str]:
        """Returns a list of missing/invalid settings that block LIVE trading."""
        problems = []
        if not self.paper:
            if not self.okx_api_key:      problems.append("OKX_API_KEY missing")
            if not self.okx_secret:       problems.append("OKX_SECRET missing")
            if not self.okx_passphrase:   problems.append("OKX_PASSPHRASE missing")
        if not self.symbols:
            problems.append("SYMBOLS is empty")
        if not (0 < self.leverage <= 125):
            problems.append(f"LEVERAGE out of range: {self.leverage}")
        return problems


def load_config() -> Config:
    return Config()
