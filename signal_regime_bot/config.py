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
    tf_entry: str  = field(default_factory=lambda: os.environ.get("TIMEFRAME_ENTRY", "30m"))
    tf_bias: str   = field(default_factory=lambda: os.environ.get("TIMEFRAME_BIAS", "1h"))
    tf_regime: str = field(default_factory=lambda: os.environ.get("TIMEFRAME_REGIME", "4h"))
    tf_context: str = "30m"   # Context Engine timeframe
    tf_fast: str    = "15m"   # Bias-secondary + Early Booster timeframe

    # ── Hardcoded strategy constants (not exposed as ENV) ───────────────────
    market_type: str = "swap"
    margin_mode: str = "isolated"
    exchange_id: str = "okx"

    min_bars: int = 100          # skip trading a symbol if any TF has fewer closed bars

    # Regime engine (4h)
    regime_ema_fast: int = 20
    regime_ema_slow: int = 50
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
    regime_transition_entry_relax: float = 5.0  # TRANSITION: entry threshold -5
    regime_transition_bias_tighten: float = 10.0  # TRANSITION: bias threshold +10

    # Bias engine (1h)
    bias_ema_fast: int = 20
    bias_ema_slow: int = 50
    bias_roc_period: int = 9
    bias_structure_left: int = 3
    bias_structure_right: int = 3
    bias_score_min: float = 60.0

    # Entry engine (30m)
    entry_hma_fast: int = 10
    entry_hma_slow: int = 20
    entry_hma_slope_lookback: int = 2
    entry_roc_period: int = 9
    entry_ema_ref: int = 15
    entry_macd_fast: int = 12
    entry_macd_slow: int = 26
    entry_macd_signal: int = 9
    entry_score_min: float = 70.0
    # ── Signal Round Gate (HMA-cross state machine) ─────────────────────────
    # HMA10 x HMA20 cross = start of a NEW signal round (the ONLY round
    # trigger). After the cross, the entry score gets `entry_setup_window_bars`
    # closed bars (setup_age 1..5, cross bar = age 1) to reach threshold:
    #   - reaches threshold within window -> enter
    #   - window passes without reaching  -> round cancelled, wait for next cross
    #   - cross back the other way        -> old round dies, new round starts
    # One round = at most one entry (consumed on entry, no re-entry same round).
    entry_setup_window_bars: int = 3   # HMA-cross setup window (spec Layer 4)
    # Anti-chase: block entry if price is already this many ATRs away from
    # EMA15 — a spike/capitulation candle already happened and the "meat" of
    # the move is behind us, not ahead.
    entry_max_ext_atr: float = 1.8
    # Market-structure confirmation (reduce false triggers): at least this many
    # of {break prev candle high/low, volume expansion, wick rejection,
    # liquidity sweep} must be present in the trade direction. 0 disables.
    entry_structure_confirm_min: int = 1
    entry_vol_expansion_mult: float = 1.5   # volume > this x vol_ma20 = expansion
    entry_wick_reject_frac: float = 0.5     # wick >= this fraction of bar range
    entry_sweep_lookback: int = 10          # bars for liquidity-sweep prior low/high

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
    daily_loss_limit_pct: float = 0.03    # halt new entries: day PnL <= -3%
    daily_profit_lock_pct: float = 0.08   # halt new entries: day PnL >= +8%
    loss_streak_limit: int = 3
    loss_streak_cooldown_min: int = 30
    max_open_positions: int = 0  # set in __post_init__ from MAX_POSITIONS (default 2)

    # Stop loss / take profit
    sl_atr_period: int = 14
    sl_atr_mult: float = 1.5
    # Floor/ceiling on SL distance as % of entry price. The floor exists so a
    # quiet-candle ATR stop can never come out so tight that TP1/TP2 R-multiple
    # profit targets fail to clear round-trip fees (see calc_stop_loss docstring).
    sl_min_pct: float = 0.004   # 0.4%
    sl_max_pct: float = 0.035  # 3.5%
    tp1_r: float = 0.5
    # Fraction of the position closed at TP1 (remainder rides to TP2/SL-at-BE).
    # Swept against the live 6-symbol backtest (BTC/ETH/SOL/XAU/XAG, Jan-Jun
    # 2026): the TP2 bucket is the strategy's only real profit source, so a
    # SMALLER TP1 take (bigger runner) improves expected value — 40% beat
    # both 50% (previous default) and 70%/60%@0.6R.
    tp1_fraction: float = 0.4
    tp2_r: float = 1.2
    swing_lookback_left: int = 3
    swing_lookback_right: int = 3

    # Position / health monitor
    weak_confirm_bars: int = 3     # consecutive WEAK 30m-bar-close reads before force exit
    health_score_min: float = 55.0
    health_check_on_closed_bar_only: bool = True
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

    # Layer 2 — Bias (SOFT confirmation + min gate, 1H + 15M)
    bias_weight_1h: float = 0.7
    bias_weight_15m: float = 0.3
    # Soft confirmation gate. Calibrated to the momentum scorer's real scale on
    # BTC/6mo: the regime-side bias score centers at ~50 (p50=50), so 65 (the
    # spec's number, which assumed a differently-scaled scorer) admitted almost
    # nothing. 50 keeps ~half of regime-passing bars — the Entry layer does the
    # real selection; this only rejects momentum-against-the-regime setups.
    bias_min_threshold: float = 50.0           # weighted regime-side score must clear this
    bias_strong_opposite: float = 70.0         # opposite side >= this -> hard veto (NEUTRAL)

    # Layer 3 — Context (SOFT SCORE, 30M)
    context_base_threshold: float = 60.0
    context_thr_strong: float = 55.0           # regime strong -> easier context bar
    context_thr_normal: float = 60.0
    context_thr_weak: float = 70.0
    context_thr_transition: float = 75.0
    context_w_sweep: float = 15.0
    context_w_bos: float = 20.0
    context_w_vwap: float = 15.0
    context_w_pullback: float = 15.0
    context_w_volume: float = 15.0
    context_w_retest: float = 10.0
    context_w_session: float = 10.0

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
    status_log_interval_sec: int = 300  # per-symbol regime/bias/entry status log cadence
    fetch_limit_entry: int = 300
    fetch_limit_bias: int = 300
    fetch_limit_regime: int = 300
    fetch_limit_context: int = 300
    fetch_limit_fast: int = 300

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
