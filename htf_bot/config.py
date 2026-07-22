"""HTF bot configuration — deliberately tiny. Strategy constants live in
strategy.py signatures; only deployment-level knobs are configurable here."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("htf.config")


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

    # Backtested-positive symbols only (6mo, net of real 0.05% fee):
    # XAU PF 1.45 / XRP 1.19 / XAG 1.14 / BTC 1.09.
    symbols: list = field(default_factory=lambda: [
        s.strip() for s in os.environ.get(
            "SYMBOLS",
            "XAU/USDT:USDT,XRP/USDT:USDT,XAG/USDT:USDT,BTC/USDT:USDT",
        ).split(",") if s.strip()])
    leverage: int = field(default_factory=lambda: _env_int("LEVERAGE", 20))
    # 5% per trade — same default as the dual/regime bots (per user request).
    # Note: the backtested portfolio maxDD is ~12R, so at 5% risk a bad
    # stretch can draw down very deeply; the startup warning below stays.
    risk_per_trade: float = field(default_factory=lambda: _env_float("RISK_PER_TRADE", 0.05))
    max_positions: int = field(default_factory=lambda: _env_int("MAX_POSITIONS", 2))

    fee_rate: float = 0.0005      # verified from an actual OKX fill (0.05% taker)
    margin_mode: str = "isolated"

    # Strategy knobs — the exact values the backtest validated.
    tp_r: float = 3.0
    be_at_r: float = 1.0
    swing_n: int = 6
    sl_buf_atr: float = 0.25
    min_sl_atr: float = 1.0
    min_sl_pct: float = 0.008

    poll_interval_sec: int = 30
    status_log_interval_sec: int = 300
    state_dir: str = field(default_factory=lambda: os.environ.get("STATE_DIR", "state"))
    stats_since_date: str = field(default_factory=lambda: os.environ.get("STATS_SINCE_DATE", "2026-07-22"))

    def validate_live(self) -> list:
        problems = []
        if not self.paper:
            if not self.okx_api_key:
                problems.append("OKX_API_KEY missing")
            if not self.okx_secret:
                problems.append("OKX_SECRET missing")
            if not self.okx_passphrase:
                problems.append("OKX_PASSPHRASE missing")
        if not self.symbols:
            problems.append("SYMBOLS empty")
        if self.risk_per_trade > 0.02:
            logger.warning("[CONFIG] RISK_PER_TRADE=%.1f%% — backtested portfolio "
                           "maxDD is ~12R; above 2%% one bad stretch is a very "
                           "deep drawdown", self.risk_per_trade * 100)
        return problems

    def stats_since_ms(self) -> int:
        import datetime
        try:
            y, m, d = (int(x) for x in self.stats_since_date.split("-"))
            return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp() * 1000)
        except (ValueError, TypeError):
            return 0
