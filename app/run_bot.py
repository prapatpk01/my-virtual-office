"""
OKX Perpetual Futures trading bot — TrendContV2 (Fast Entry Architecture).

Railway env vars (only these need to be set):
  EXCHANGE_API_KEY      OKX API key
  EXCHANGE_API_SECRET   OKX secret
  EXCHANGE_PASSPHRASE   OKX passphrase
  PAPER_TRADING         true = paper mode (default: false = live)
  SYMBOLS               comma-separated symbols (default: BTC/USDT:USDT,XAU/USDT:USDT)
  MAX_POSITIONS         max open positions at once (default: 2)
  TELEGRAM_BOT_TOKEN    Telegram bot token
  TELEGRAM_CHAT_ID      Telegram chat ID

Everything else is hardcoded as sensible defaults.
"""
import asyncio
import logging
import os
import signal


# ── .env loader (local dev only) ─────────────────────────────────────────────

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

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_bot")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _env_list(key: str, default: str) -> list[str]:
    return [s.strip() for s in os.environ.get(key, default).split(",") if s.strip()]

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    return default if not val else val.lower() in ("1", "true", "yes")

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ── Config (only user-facing env vars) ───────────────────────────────────────

def build_config() -> dict:
    return {
        # ── Exchange credentials ───────────────────────────────────────────────
        "api_key":        os.environ.get("EXCHANGE_API_KEY",    ""),
        "api_secret":     os.environ.get("EXCHANGE_API_SECRET", ""),
        "api_passphrase": os.environ.get("EXCHANGE_PASSPHRASE", ""),
        "paper":          _env_bool("PAPER_TRADING", False),

        # ── Trading ───────────────────────────────────────────────────────────
        "symbols":       _env_list("SYMBOLS", "BTC/USDT:USDT,XAU/USDT:USDT"),
        "max_positions": _env_int("MAX_POSITIONS", 2),

        # ── Telegram ──────────────────────────────────────────────────────────
        "telegram_token":   os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID",   ""),
    }


# ── Strategy builder ─────────────────────────────────────────────────────────

def build_strategies(symbols: list[str]) -> list:
    from trading.strategies.trend_cont_v2_strategy import TrendContV2Strategy

    strategies = []
    for sym in symbols:
        # All strategy params use DEFAULTS from TrendContV2Strategy.
        # Only override what is meaningful per-symbol (none needed currently).
        strategies.append(TrendContV2Strategy(sym))

    if not strategies:
        raise RuntimeError("No strategies built — check SYMBOLS env var is not empty")
    logger.info("Strategies: %s  symbols: %s", [s.name for s in strategies], symbols)
    return strategies


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    cfg = build_config()

    # ── Hardcoded constants ───────────────────────────────────────────────────
    EXCHANGE       = "okx"
    MARKET_TYPE    = "swap"
    LEVERAGE       = 20
    MAX_DRAWDOWN   = 0.20   # halt new entries if portfolio drops 20% from peak
    DAILY_CB       = 0.05   # halt for the day if daily PnL ≤ -5%
    SL_PCT         = 0.015  # fallback SL (dynamic SL from ATR used in practice)
    TP_PCT         = 0.025  # fallback TP
    MONITOR_SECS   = 60     # health monitor interval
    HEALTH_CONFIRM = 3      # WEAK-health cycles before closing
    INTERVAL_SECS  = 30     # main loop interval
    TG_MIN_CONF    = 0.5    # min confidence to send Telegram signal alert

    # ── Live trading safety check ─────────────────────────────────────────────
    if not cfg["paper"]:
        missing = []
        if not cfg["api_key"]:        missing.append("EXCHANGE_API_KEY")
        if not cfg["api_secret"]:     missing.append("EXCHANGE_API_SECRET")
        if not cfg["api_passphrase"]: missing.append("EXCHANGE_PASSPHRASE")
        if missing:
            raise RuntimeError(
                f"LIVE TRADING: missing required credentials: {', '.join(missing)}"
            )

    mode = "PAPER" if cfg["paper"] else "LIVE"
    logger.info(
        "=== Bot starting [%s]  exchange=%s  symbols=%s  leverage=%dx  max_pos=%d ===",
        mode, EXCHANGE, cfg["symbols"], LEVERAGE, cfg["max_positions"],
    )

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram = None
    if cfg["telegram_token"] and cfg["telegram_chat_id"]:
        from trading.telegram_notifier import TelegramNotifier
        telegram = TelegramNotifier(
            token=cfg["telegram_token"],
            chat_id=cfg["telegram_chat_id"],
            min_confidence=TG_MIN_CONF,
        )
        logger.info("Telegram configured (chat_id=%s)", cfg["telegram_chat_id"])
    else:
        logger.warning("Telegram NOT configured — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")

    # ── Connector ─────────────────────────────────────────────────────────────
    from trading.connectors.binance_conn import BinanceConnector
    connector = BinanceConnector(
        api_key=cfg["api_key"],
        api_secret=cfg["api_secret"],
        paper=cfg["paper"],
        exchange_id=EXCHANGE,
        passphrase=cfg["api_passphrase"],
        market_type=MARKET_TYPE,
        leverage=LEVERAGE,
        symbol_leverage={},
    )

    # ── Strategies ────────────────────────────────────────────────────────────
    strategies = build_strategies(cfg["symbols"])

    # ── Risk manager ──────────────────────────────────────────────────────────
    from trading.risk_manager import RiskManager
    risk = RiskManager(
        stop_loss_pct=SL_PCT,
        take_profit_pct=TP_PCT,
        max_open_positions=cfg["max_positions"],
        max_drawdown_pct=MAX_DRAWDOWN,
        fixed_trade_usdt=0.0,   # disabled — dynamic sizing via risk_pct used instead
        leverage=LEVERAGE,
        daily_loss_limit_pct=DAILY_CB,
    )
    logger.info(
        "Risk: confidence-scaled 5%%–10%%  lev=%dx  max_pos=%d  drawdown=%.0f%%  dailyCB=%.0f%%",
        LEVERAGE, cfg["max_positions"], MAX_DRAWDOWN * 100, DAILY_CB * 100,
    )

    # ── Bot ───────────────────────────────────────────────────────────────────
    from trading.bot import TradingBot
    bot = TradingBot(
        connector=connector,
        strategies=strategies,
        risk_manager=risk,
        interval_seconds=INTERVAL_SECS,
        telegram=telegram,
        fixed_sl_pct=SL_PCT,
        fixed_tp_pct=TP_PCT,
        dynamic_sizing=True,
        monitor_interval=MONITOR_SECS,
    )
    bot._health_weak_confirm = HEALTH_CONFIRM

    # ── Telegram hooks ────────────────────────────────────────────────────────
    if telegram:
        stop_event_ref: list = []

        def _tg_stop():
            if stop_event_ref:
                stop_event_ref[0].set()
            return {"message": "Bot stopping..."}

        telegram.get_state_fn = bot.get_state
        telegram.get_stats_fn = bot.get_stats
        telegram.stop_bot_fn  = _tg_stop
        telegram.start_bot_fn = lambda: {"message": "Bot is already running"}

    # ── Run ───────────────────────────────────────────────────────────────────
    stop_signal = asyncio.Event()
    if telegram:
        stop_event_ref.append(stop_signal)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_signal.set)
        except (NotImplementedError, RuntimeError):
            pass

    await bot.start()
    await stop_signal.wait()

    logger.info("Shutdown signal received — stopping bot...")
    await bot.stop()
    try:
        await connector.close()
    except Exception:
        pass
    logger.info("Bot stopped cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
