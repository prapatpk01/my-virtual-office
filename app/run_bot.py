"""
Standalone Trading Bot runner — no web UI required.
Runs the bot + Telegram integration 24/7 on any cloud platform.

Usage:
    python run_bot.py

Config via environment variables (see .env.example) or a .env file.
"""
import asyncio
import logging
import os
import signal
import sys

# ---------------------------------------------------------------------------
# Load .env if present (optional)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_bot")

# ---------------------------------------------------------------------------
# Build config from env
# ---------------------------------------------------------------------------

def _env_list(key: str, default: str) -> list[str]:
    val = os.environ.get(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")

def build_config() -> dict:
    return {
        # Exchange
        "exchange":     os.environ.get("EXCHANGE", "binance"),
        "api_key":      os.environ.get("EXCHANGE_API_KEY", ""),
        "api_secret":   os.environ.get("EXCHANGE_API_SECRET", ""),
        "paper":        _env_bool("PAPER_TRADING", True),

        # Symbols and timing
        "symbols":      _env_list("SYMBOLS", "BTC/USDT"),
        "interval":     int(os.environ.get("INTERVAL_SECONDS", "60")),

        # Strategies
        "strategies": {
            "wt_adx":   _env_bool("STRATEGY_WT_ADX",   False),
            "macd_ema": _env_bool("STRATEGY_MACD_EMA",  True),
        },

        # Risk management
        "risk_per_trade":  float(os.environ.get("RISK_PER_TRADE",  "0.02")),
        "stop_loss_pct":   float(os.environ.get("STOP_LOSS_PCT",   "0.03")),
        "take_profit_pct": float(os.environ.get("TAKE_PROFIT_PCT", "0.06")),
        "max_positions":   int(os.environ.get("MAX_POSITIONS",     "5")),

        # Anthropic (AI Signal strategy)
        "anthropic_key":   os.environ.get("ANTHROPIC_API_KEY", ""),

        # Telegram
        "telegram_token":   os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID",   ""),
        "tg_min_confidence": float(os.environ.get("TG_MIN_CONFIDENCE", "0.5")),
    }

# ---------------------------------------------------------------------------
# Bot factory (same logic as trading_api._build_bot, standalone version)
# ---------------------------------------------------------------------------

def build_bot(config: dict):
    from trading.connectors.binance_conn import BinanceConnector
    from trading.connectors.alpaca_conn import AlpacaConnector
    from trading.strategies.wt_adx_strategy import WTADXStrategy
    from trading.strategies.macd_ema_strategy import MACDEMAStrategy
    from trading.risk_manager import RiskManager
    from trading.bot import TradingBot
    from trading.telegram_notifier import TelegramNotifier

    exchange = config["exchange"]
    paper    = config["paper"]
    symbols  = config["symbols"]
    flags    = config["strategies"]

    # Connector
    if exchange in ("binance", "bybit", "okx"):
        connector = BinanceConnector(
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            paper=paper,
            exchange_id=exchange,
        )
    else:
        connector = AlpacaConnector(
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            paper=paper,
        )

    # Strategies
    strategies = []
    for sym in symbols:
        if flags.get("wt_adx"):   strategies.append(WTADXStrategy(sym))
        if flags.get("macd_ema"): strategies.append(MACDEMAStrategy(sym))

    if not strategies:
        strategies.append(WTADXStrategy(symbols[0]))

    risk = RiskManager(
        max_risk_per_trade_pct=config["risk_per_trade"],
        stop_loss_pct=config["stop_loss_pct"],
        take_profit_pct=config["take_profit_pct"],
        max_open_positions=config["max_positions"],
    )

    # Telegram
    telegram = None
    tg_token = config.get("telegram_token", "").strip()
    tg_chat  = config.get("telegram_chat_id", "").strip()
    if tg_token and tg_chat:
        telegram = TelegramNotifier(
            token=tg_token,
            chat_id=tg_chat,
            min_confidence=config["tg_min_confidence"],
        )
        logger.info("Telegram notifications enabled (chat_id=%s)", tg_chat)
    else:
        logger.warning("Telegram NOT configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

    bot = TradingBot(
        connector=connector,
        strategies=strategies,
        risk_manager=risk,
        interval_seconds=config["interval"],
        broadcast_fn=None,   # no WebSocket in standalone mode
        telegram=telegram,
    )

    if telegram:
        telegram.get_state_fn = bot.get_state
        telegram.start_bot_fn = lambda: _restart_signal.set()
        telegram.stop_bot_fn  = lambda: _stop_signal.set()

    return bot

# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

_stop_signal  = asyncio.Event()
_restart_signal = asyncio.Event()


async def main():
    config = build_config()
    mode = "PAPER" if config["paper"] else "LIVE"
    logger.info("=== Trading Bot starting [%s] exchange=%s symbols=%s ===",
                mode, config["exchange"], config["symbols"])

    enabled = [k for k, v in config["strategies"].items() if v]
    logger.info("Enabled strategies: %s", enabled)

    if not config["paper"] and not (config["api_key"] and config["api_secret"]):
        logger.error("LIVE mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET — aborting")
        sys.exit(1)

    bot = build_bot(config)

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_event_loop()

    def _handle_signal():
        logger.info("Shutdown signal received")
        _stop_signal.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, RuntimeError):
            pass  # Windows / some container environments

    await bot.start()
    logger.info("Bot running. Send /stop_bot in Telegram or Ctrl+C to stop.")

    # Wait until stopped or restart requested
    await _stop_signal.wait()

    logger.info("Stopping bot...")
    await bot.stop()
    logger.info("Bot stopped cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
