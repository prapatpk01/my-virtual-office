"""Config for DUAL ENTRY PRECISION V1.4 — section 35 defaults, env-overridable.

Nothing is hard-coded elsewhere: symbols, risk, leverage and exchange rules
all flow from here (exchange rules themselves come from the exchange at
runtime). validate() runs at startup and refuses to boot on nonsense.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("dual_entry.config")


def _env(name: str, default, cast=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    cast = cast or type(default)
    try:
        if cast is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return cast(raw)
    except (ValueError, TypeError):
        logger.warning("[CONFIG] bad env %s=%r — using default %r", name, raw, default)
        return default


@dataclass
class Config:
    # ── Universe / account ──────────────────────────────────────────────────
    symbols: list = field(default_factory=list)          # from SYMBOLS env
    paper: bool = True
    leverage: int = 10
    margin_mode: str = "isolated"
    position_mode: str = "hedge"                          # "hedge" | "net"

    okx_api_key: str = ""
    okx_secret: str = ""
    okx_passphrase: str = ""

    telegram_token: str = ""
    telegram_chat_id: str = ""

    # ── Timeframes ──────────────────────────────────────────────────────────
    macro_timeframe: str = "4h"
    bias_timeframe: str = "1h"
    entry_timeframe: str = "15m"

    # ── Data quality ────────────────────────────────────────────────────────
    min_15m_candles: int = 300
    min_1h_candles: int = 200
    min_4h_candles: int = 200
    fetch_15m: int = 350
    fetch_1h: int = 250
    fetch_4h: int = 250
    max_price_staleness_sec: int = 180
    max_spread_pct: float = 0.0015          # 0.15%
    max_clock_skew_sec: int = 30

    # volatility shock
    shock_range_median_mult: float = 2.5
    shock_atr_expansion: float = 1.40
    shock_lockout_bars: int = 1
    severe_shock_lockout_bars: int = 2
    severe_shock_atr_expansion: float = 1.80
    allow_high_quality_shock_breakout: bool = True

    # ── Portfolio ───────────────────────────────────────────────────────────
    max_positions: int = 2
    max_total_open_risk: float = 0.06
    correlated_risk_factor: float = 0.6     # 2nd same-direction correlated trade risk multiplier

    # ── Indicators ──────────────────────────────────────────────────────────
    hma_fast: int = 10
    hma_slow: int = 16
    roc_length: int = 5
    atr_length: int = 14
    adx_length: int = 12
    adx_smoothing: int = 12
    chop_length: int = 14
    volume_sma: int = 20
    roc_1h: int = 9

    min_adx: float = 11.0
    momentum_min_adx: float = 13.0
    strong_adx: float = 20.0
    max_chop: float = 62.0
    strong_chop: float = 52.0

    # ── Swings / structure ──────────────────────────────────────────────────
    swing_left_bars: int = 3
    swing_right_bars: int = 3
    bos_min_body_atr: float = 0.18
    zone_break_body_atr: float = 0.30
    false_bos_lookback: int = 2

    # ── Zones ───────────────────────────────────────────────────────────────
    zone_width_15m_atr: float = 0.15
    zone_width_1h_atr: float = 0.20
    zone_width_4h_atr: float = 0.25
    zone_min_score: float = 45.0
    zone_hq_score: float = 75.0
    zone_valid_score: float = 60.0
    max_zones_per_tf: int = 12

    # supply/demand
    min_departure_atr: float = 0.70
    max_base_bars: int = 6
    max_hq_mitigations: int = 1

    # ── Regime ──────────────────────────────────────────────────────────────
    regime_confirmation_bars: int = 2
    allow_candidate_regime_entry: bool = True
    chop_adx: float = 11.0
    chop_chop: float = 62.0
    hma_flip_window: int = 8
    hma_flip_count: int = 3
    hma_flip_spread_atr: float = 0.12
    di_flip_count: int = 3
    di_flip_min_spread: float = 3.0

    # ── Pullback engine ─────────────────────────────────────────────────────
    hma_pullback_zone_upper_atr: float = 0.15
    hma_pullback_zone_lower_atr: float = 0.20
    pullback_window_bars: int = 3
    pullback_max_window_bars: int = 4
    pullback_threshold: float = 64.0
    early_pullback_threshold: float = 70.0
    deep_pullback_threshold: float = 70.0
    pullback_threshold_min: float = 62.0
    pullback_threshold_max: float = 74.0
    pullback_volume_ratio: float = 0.90
    pullback_max_extension_atr: float = 0.90
    pullback_min_structure_room_r: float = 1.00
    pullback_hq_room_r: float = 1.10          # 1.00-1.10 needs high-quality setup
    pullback_bonus_room_r: float = 1.40
    prior_context_displacement_atr: float = 0.65
    prior_context_body_atr: float = 0.28
    same_bar_zone_score: float = 65.0
    same_bar_body_atr: float = 0.15
    same_bar_close_quality: float = 0.62
    early_trigger_zone_score: float = 70.0

    # ── Momentum engine ─────────────────────────────────────────────────────
    momentum_threshold: float = 70.0
    strong_breakout_threshold: float = 66.0
    momentum_threshold_min: float = 68.0
    momentum_threshold_max: float = 80.0
    momentum_volume_ratio: float = 1.05
    momentum_max_extension_atr: float = 1.05
    strong_trend_extension_atr: float = 1.25
    momentum_min_structure_room_r: float = 1.10
    momentum_hq_room_r: float = 1.25
    breakout_lookback: int = 4
    std_breakout_body_atr: float = 0.15
    std_breakout_close_quality: float = 0.65
    strong_breakout_body_atr: float = 0.25
    strong_breakout_close_quality: float = 0.75
    compression_recent_mult: float = 0.85
    compression_atr_ratio: float = 0.88
    momentum_expiry_bars: int = 1               # breakout candle or the next bar only

    # ── Risk manager ────────────────────────────────────────────────────────
    # 5% per user request (matches the legacy regime bot). The >2% warning
    # still fires at startup — one stop-out costs this much equity.
    risk_per_trade: float = 0.05
    risk_warning_level: float = 0.02
    min_stop_atr: float = 0.45
    max_stop_atr: float = 1.60
    stop_buffer_atr: float = 0.08
    risk_reward: float = 1.20
    min_acceptable_rr: float = 1.08
    target_buffer_atr: float = 0.10
    fee_rate: float = 0.001                     # per fill
    expected_slippage_atr: float = 0.03

    # ── Execution quality ───────────────────────────────────────────────────
    max_entry_slippage_atr: float = 0.12
    max_entry_deviation_atr: float = 0.15
    momentum_max_deviation_atr: float = 0.10
    pending_order_timeout_sec: int = 90
    signal_expiry_bars: int = 1                 # candidate valid this bar + next

    # ── Position management ─────────────────────────────────────────────────
    pullback_be_trigger_r: float = 0.65
    momentum_be_trigger_r: float = 0.75
    be_lock_r: float = 0.05
    min_hold_bars_soft_exit: int = 2

    # ── Cooldown ────────────────────────────────────────────────────────────
    cooldown_bars: int = 1
    sl_cooldown_bars: int = 2
    tp_cooldown_bars: int = 0
    early_exit_cooldown_bars: int = 1
    hard_flip_cooldown_bars: int = 1
    false_breakout_cooldown_bars: int = 2
    loss_streak_limit: int = 3
    loss_streak_cooldown_minutes: int = 30

    # ── Module performance gate ─────────────────────────────────────────────
    module_min_trades_reduced: int = 30
    module_min_trades_paused: int = 50
    module_pf_reduced_low: float = 0.90
    module_pf_reduced_high: float = 1.05
    module_reduced_risk_factor: float = 0.6
    module_shadow_reopen_signals: int = 15

    # ── Dynamic context adjustment (threshold modifiers) ────────────────────
    mod_alignment: float = -2.0
    mod_mild_conflict: float = 3.0
    mod_high_vol: float = 2.0
    mod_low_liquidity: float = 4.0
    mod_hq_zone: float = -2.0
    mod_opposing_near: float = 3.0
    mod_deep_pullback: float = 4.0
    mod_major_breakout: float = -2.0
    mod_reduced_module: float = 2.0

    # ── Commodity market hours (XAU/XAG) ────────────────────────────────────
    # Underlying metals market closes on weekends — the OKX perp still quotes
    # but it's illiquid. Block NEW entries for matching symbols from Friday
    # commodity_halt_hour_utc (17:00 UTC = Sat 00:00 Asia/Bangkok) until
    # Sunday commodity_resume_hour_utc (21:00 UTC = Mon 04:00 ICT, 3h before
    # the Mon 07:00 ICT open). Open positions keep being managed throughout.
    commodity_weekend_block: bool = True
    commodity_symbol_keywords: tuple = ("XAU", "XAG")
    commodity_halt_hour_utc: int = 17
    commodity_resume_hour_utc: int = 21

    # ── Ops ─────────────────────────────────────────────────────────────────
    poll_interval_sec: int = 20
    reconcile_every_loops: int = 1
    state_dir: str = "state"
    enable_advanced_patterns_live: bool = False
    enable_advanced_patterns_shadow: bool = True
    volume_quality_mode: str = "AUTO"           # AUTO -> REAL_VOLUME on OKX

    def validate(self) -> None:
        errs = []
        if not self.symbols:
            errs.append("SYMBOLS empty")
        if not (0 < self.risk_per_trade <= 0.10):
            errs.append(f"risk_per_trade {self.risk_per_trade} out of (0, 0.10]")
        if self.risk_per_trade > self.risk_warning_level:
            logger.warning("[CONFIG] risk_per_trade=%.1f%% is above the %.0f%% warning level — "
                           "one stop-out costs this much of equity",
                           self.risk_per_trade * 100, self.risk_warning_level * 100)
        if self.min_acceptable_rr > self.risk_reward:
            errs.append("min_acceptable_rr > risk_reward")
        if self.max_positions < 1:
            errs.append("max_positions < 1")
        if self.min_stop_atr >= self.max_stop_atr:
            errs.append("min_stop_atr >= max_stop_atr")
        if not (self.pullback_threshold_min <= self.pullback_threshold <= self.pullback_threshold_max):
            errs.append("pullback threshold outside its dynamic range")
        if not (self.momentum_threshold_min <= self.momentum_threshold <= self.momentum_threshold_max):
            errs.append("momentum threshold outside its dynamic range")
        if self.entry_timeframe != "15m":
            errs.append("V1.4 entry timeframe must be 15m")
        if errs:
            raise ValueError("Config invalid: " + "; ".join(errs))


def load_config() -> Config:
    symbols_raw = os.environ.get("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT")
    cfg = Config(
        symbols=[s.strip() for s in symbols_raw.split(",") if s.strip()],
        paper=_env("PAPER_TRADING", True, bool),
        leverage=_env("LEVERAGE", 10, int),
        okx_api_key=os.environ.get("OKX_API_KEY", ""),
        okx_secret=os.environ.get("OKX_SECRET_KEY", ""),
        okx_passphrase=os.environ.get("OKX_PASSPHRASE", ""),
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        risk_per_trade=_env("RISK_PER_TRADE", 0.05, float),
        max_positions=_env("MAX_POSITIONS", 2, int),
        state_dir=os.environ.get("STATE_DIR", "state"),
        poll_interval_sec=_env("POLL_INTERVAL_SEC", 20, int),
    )
    cfg.validate()
    return cfg


TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
         "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
