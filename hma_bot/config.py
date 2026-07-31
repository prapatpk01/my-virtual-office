"""MTF Structure Trend bot configuration (MODE=hma).

Deployment knobs + the strategy's own tunables (mirrored from strategy.py's
StrategyConfig so live and backtest share one source). Reuses the regime bot's
ExchangeClient / TelegramNotifier / chart engine via sys.path, exactly like the
HTF bot.

⚠️ Backtest note (BTC+XAU, Jan–May 2026, fee 0.05%): this strategy was
net-NEGATIVE (BTC −6.2R PF 0.76, XAU −13.0R PF 0.31) — HMA16-flip whipsaw plus
fee drag on ~130 trades/symbol. Shipped at the user's explicit direction; run
small risk and forward-test before trusting it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import strategy as S

logger = logging.getLogger("hma.config")


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
    v = os.environ.get(key, "")
    return default if not v else v.lower() in ("1", "true", "yes")


def _env_first(*keys: str) -> str:
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            return v
    return ""


@dataclass
class Config:
    okx_api_key: str = field(default_factory=lambda: _env_first("OKX_API_KEY", "EXCHANGE_API_KEY"))
    okx_secret: str = field(default_factory=lambda: _env_first("OKX_SECRET", "EXCHANGE_API_SECRET"))
    okx_passphrase: str = field(default_factory=lambda: _env_first("OKX_PASSPHRASE", "EXCHANGE_PASSPHRASE"))
    paper: bool = field(default_factory=lambda: _env_bool("PAPER_TRADING", False))
    telegram_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))

    symbols: list = field(default_factory=lambda: [
        s.strip() for s in os.environ.get(
            "SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT").split(",") if s.strip()])
    leverage: int = field(default_factory=lambda: _env_int("LEVERAGE", 20))
    # Fixed-margin sizing: each new position uses $20 margin at x20 leverage
    # by default, i.e. about $400 notional per position.
    margin_per_position_usd: float = field(
        default_factory=lambda: _env_float("MARGIN_PER_POSITION_USD", 20.0))
    max_positions: int = field(default_factory=lambda: _env_int("MAX_POSITIONS", 2))

    fee_rate: float = 0.0005      # verified 0.05% OKX taker per fill
    margin_mode: str = "isolated"

    # ── MTF strategy tunables ─────────────────────────────────────────────
    trend_tf: str = "4h"
    quality_tf: str = "1h"
    entry_tf: str = "15m"

    min_trend_quality: float = field(default_factory=lambda: _env_float("MIN_TREND_QUALITY", 55.0))
    min_entry_score: float = field(default_factory=lambda: _env_float("MIN_ENTRY_SCORE", 60.0))
    max_chase_atr: float = field(default_factory=lambda: _env_float("MAX_CHASE_ATR", 0.75))

    stop_loss_pct: float = field(default_factory=lambda: _env_float("SL_PCT", 0.015))
    take_profit_pct: float = field(default_factory=lambda: _env_float("TP_PCT", 0.015))

    target1_trigger_pct: float = field(default_factory=lambda: _env_float("T1_TRIGGER_PCT", 0.006))
    target1_lock_pct: float = field(default_factory=lambda: _env_float("T1_LOCK_PCT", 0.003))
    target2_trigger_pct: float = field(default_factory=lambda: _env_float("T2_TRIGGER_PCT", 0.010))
    target2_lock_pct: float = field(default_factory=lambda: _env_float("T2_LOCK_PCT", 0.007))

    reentry_cooldown_bars: int = field(default_factory=lambda: _env_int("REENTRY_COOLDOWN_BARS", 0))

    poll_interval_sec: int = 30
    status_log_interval_sec: int = field(default_factory=lambda: _env_int("STATUS_LOG_INTERVAL_SEC", 300))
    state_dir: str = field(default_factory=lambda: os.environ.get("STATE_DIR", "state"))
    stats_since_date: str = field(default_factory=lambda: os.environ.get("STATS_SINCE_DATE", "2026-07-30"))

    def strategy_config(self) -> "S.StrategyConfig":
        return S.StrategyConfig(
            trend_tf=self.trend_tf,
            quality_tf=self.quality_tf,
            entry_tf=self.entry_tf,
            min_trend_quality=self.min_trend_quality,
            min_entry_score=self.min_entry_score,
            max_chase_atr=self.max_chase_atr,
            stop_loss_pct=self.stop_loss_pct,
            final_take_profit_pct=self.take_profit_pct,
            target1_trigger_pct=self.target1_trigger_pct,
            target1_lock_pct=self.target1_lock_pct,
            target2_trigger_pct=self.target2_trigger_pct,
            target2_lock_pct=self.target2_lock_pct,
        )

    def validate_live(self) -> list:
        problems = []
        if not self.paper:
            for k, v in (("OKX_API_KEY", self.okx_api_key),
                         ("OKX_SECRET", self.okx_secret),
                         ("OKX_PASSPHRASE", self.okx_passphrase)):
                if not v:
                    problems.append(f"{k} missing")
        if not self.symbols:
            problems.append("SYMBOLS empty")
        if self.margin_per_position_usd <= 0:
            problems.append("MARGIN_PER_POSITION_USD must be > 0")
        if self.leverage <= 0:
            problems.append("LEVERAGE must be > 0")
        if self.max_positions <= 0:
            problems.append("MAX_POSITIONS must be > 0")
        return problems

    def stats_since_ms(self) -> int:
        import datetime
        try:
            y, m, d = (int(x) for x in self.stats_since_date.split("-"))
            return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp() * 1000)
        except (ValueError, TypeError):
            return 0
