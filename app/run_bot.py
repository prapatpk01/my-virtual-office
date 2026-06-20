"""
Trading bot entry point. Load config from env vars and start the bot.

Usage:
    python run_bot.py

Config via environment variables or a .env file in the same directory.
"""
import asyncio
import logging
import os
import signal
import sys


# ---------------------------------------------------------------------------
# Load .env if present
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
# Helpers
# ---------------------------------------------------------------------------

def _env_list(key: str, default: str) -> list:
    val = os.environ.get(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")


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


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------

def build_config() -> dict:
    return {
        "exchange":        os.environ.get("EXCHANGE", "okx"),
        "api_key":         os.environ.get("EXCHANGE_API_KEY", ""),
        "api_secret":      os.environ.get("EXCHANGE_API_SECRET", ""),
        "api_passphrase":  os.environ.get("EXCHANGE_PASSPHRASE", ""),
        "paper":           _env_bool("PAPER_TRADING", False),
        "leverage":        _env_int("LEVERAGE", 1),
        "symbols":         _env_list("SYMBOLS", "BTC/USDT"),
        "candle_tf":       os.environ.get("CANDLE_TF", "1h"),
        "candle_limit":    _env_int("CANDLE_LIMIT", 250),
        "interval":        _env_int("INTERVAL_SECONDS", 3600),
        "trade_amount_usdt": _env_float("TRADE_AMOUNT_USDT", 100.0),
        "max_positions":   _env_int("MAX_POSITIONS", 3),
        "max_drawdown":    _env_float("MAX_DRAWDOWN_PCT", 0.30),
        "risk_per_trade":  _env_float("RISK_PER_TRADE", 0.02),
        "strategies": {
            "swing_reversal":   _env_bool("STRATEGY_SWING_REVERSAL",   True),
            "scalp_trend":      _env_bool("STRATEGY_SCALP_TREND",      True),
            "profitable_bot":   _env_bool("STRATEGY_PROFITABLE_BOT",   True),
        },
        # Swing v5 Wide config — tune via env vars if needed
        "sr_params": {
            "sl_atr":        _env_float("SR_SL",      2.0),
            "tp_atr":        _env_float("SR_TP",      2.0),
            "rsi_lo":        _env_float("SR_RSI_LO", 30.0),
            "rsi_hi":        _env_float("SR_RSI_HI", 65.0),
            "vol_mult":      _env_float("SR_VOL",      1.0),
            "max_hold_days": _env_float("SR_HOLD",     3.0),
        },
        "profitable_params": {
            "sl_atr":   _env_float("PB_SL",     2.0),
            "tp_atr":   _env_float("PB_TP",     1.8),
            "cooldown": _env_int("PB_COOLDOWN", 2),
        },
        "scalp_params": {
            "sl_atr":   _env_float("SC_SL",     2.0),
            "tp_atr":   _env_float("SC_TP",     2.0),
            "cooldown": _env_int("SC_COOLDOWN", 8),
        },
        "telegram_token":      os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id":    os.environ.get("TELEGRAM_CHAT_ID", ""),
    }


# ---------------------------------------------------------------------------
# Build strategies
# ---------------------------------------------------------------------------

def _make_strategies(symbols: list, flags: dict, cfg: dict,
                     connector=None) -> list:
    from trading.strategies.swing_strategy import SwingReversalStrategy
    from trading.strategies.scalp_strategy import ScalpTrendBot
    from trading.strategies.profitable_strategy import ProfitableBot

    strategies = []
    for sym in symbols:
        if flags.get("swing_reversal"):
            strategies.append(SwingReversalStrategy(sym, params=cfg["sr_params"]))
        if flags.get("scalp_trend"):
            strategies.append(ScalpTrendBot(sym, params=cfg["scalp_params"]))
        if flags.get("profitable_bot"):
            strategies.append(ProfitableBot(sym, params=cfg["profitable_params"]))
    return strategies


# ---------------------------------------------------------------------------
# Build connector
# ---------------------------------------------------------------------------

def _make_connector(cfg: dict):
    from trading.connectors.binance_conn import BinanceConnector

    return BinanceConnector(
        api_key=cfg["api_key"],
        api_secret=cfg["api_secret"],
        paper=cfg["paper"],
        exchange_id=cfg["exchange"],
        passphrase=cfg.get("api_passphrase", ""),
        leverage=cfg.get("leverage", 1),
        margin_mode="cross",
    )


# ---------------------------------------------------------------------------
# Build telegram
# ---------------------------------------------------------------------------

def _make_telegram(cfg: dict):
    from trading.telegram_notifier import TelegramNotifier

    token = cfg.get("telegram_token", "").strip()
    chat = cfg.get("telegram_chat_id", "").strip()
    if not token or not chat:
        logger.warning("Telegram NOT configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing)")
        return None
    return TelegramNotifier(token=token, chat_id=chat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    cfg = build_config()

    # Pre-flight: refuse to start live mode with missing credentials
    if not cfg["paper"]:
        required = ["api_key", "api_secret"]
        if cfg["exchange"] == "okx":
            required.append("api_passphrase")  # OKX mandates passphrase
        missing = [k for k in required if not cfg.get(k, "").strip()]
        if missing:
            logger.critical("LIVE MODE but %s not set — set env vars and restart", missing)
            sys.exit(1)
        if not cfg.get("telegram_token"):
            logger.warning("⚠ Telegram not configured — no trade alerts in LIVE mode")

    logger.info(
        "=== Bot starting [%s] exchange=%s symbols=%s ===",
        "PAPER" if cfg["paper"] else "LIVE",
        cfg["exchange"],
        cfg["symbols"],
    )

    connector = _make_connector(cfg)
    strategies = _make_strategies(cfg["symbols"], cfg["strategies"], cfg,
                                  connector=connector)
    if not strategies:
        logger.error("No strategies enabled — exiting")
        sys.exit(1)

    from trading.risk_manager import RiskManager
    from trading.bot import TradingBot

    risk = RiskManager(
        max_risk_per_trade_pct=cfg["risk_per_trade"],
        max_open_positions=cfg["max_positions"],
        max_drawdown_pct=cfg["max_drawdown"],
    )

    telegram = _make_telegram(cfg)

    bot = TradingBot(
        connector=connector,
        strategies=strategies,
        risk_manager=risk,
        interval_seconds=cfg["interval"],
        telegram=telegram,
        trade_amount_usdt=cfg["trade_amount_usdt"],
        max_positions=cfg["max_positions"],
        candle_tf=cfg["candle_tf"],
        candle_limit=cfg["candle_limit"],
        mtf_gate=False,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    if telegram:
        telegram.bot = bot
        telegram.stop_bot_fn = lambda: stop_event.set()

    def _handle_signal():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, RuntimeError):
            pass

    await bot.start()
    await stop_event.wait()

    logger.info("Stopping bot...")
    await bot.stop()

    try:
        await connector.close()
    except Exception:
        pass

    logger.info("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
