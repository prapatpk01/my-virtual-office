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
# Helpers
# ---------------------------------------------------------------------------

def _env_list(key: str, default: str) -> list[str]:
    val = os.environ.get(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def build_config() -> dict:
    return {
        "exchange":     os.environ.get("EXCHANGE", "binance"),
        "api_key":      os.environ.get("EXCHANGE_API_KEY", ""),
        "api_secret":   os.environ.get("EXCHANGE_API_SECRET", ""),
        "paper":        _env_bool("PAPER_TRADING", True),
        "symbols":      _env_list("SYMBOLS", "BTC/USDT,ETH/USDT"),
        "interval":     int(os.environ.get("INTERVAL_SECONDS", "60")),
        "strategies": {
            "wt_adx":   _env_bool("STRATEGY_WT_ADX",   False),
            "macd_ema": _env_bool("STRATEGY_MACD_EMA",  True),
        },
        "risk_per_trade":  float(os.environ.get("RISK_PER_TRADE",  "0.02")),
        "stop_loss_pct":   float(os.environ.get("STOP_LOSS_PCT",   "0.03")),
        "take_profit_pct": float(os.environ.get("TAKE_PROFIT_PCT", "0.06")),
        "max_positions":   int(os.environ.get("MAX_POSITIONS",     "5")),
        "telegram_token":   os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID",   ""),
        "tg_min_confidence": float(os.environ.get("TG_MIN_CONFIDENCE", "0.5")),
        # Forex / Gold signal-only symbols (Yahoo Finance, no order execution)
        "forex_symbols": _env_list("FOREX_SYMBOLS", "XAUUSD,EURUSD,USDJPY"),
        "forex_enabled": _env_bool("FOREX_SIGNALS", True),
        "forex_interval": int(os.environ.get("FOREX_INTERVAL_SECONDS", "60")),
    }

# ---------------------------------------------------------------------------
# Bot factories
# ---------------------------------------------------------------------------

def _make_strategies(symbols: list, flags: dict):
    from trading.strategies.wt_adx_strategy import WTADXStrategy
    from trading.strategies.macd_ema_strategy import MACDEMAStrategy
    strategies = []
    for sym in symbols:
        if flags.get("wt_adx"):   strategies.append(WTADXStrategy(sym))
        if flags.get("macd_ema"): strategies.append(MACDEMAStrategy(sym))
    return strategies


def _make_telegram(config: dict):
    from trading.telegram_notifier import TelegramNotifier
    token = config.get("telegram_token", "").strip()
    chat  = config.get("telegram_chat_id", "").strip()
    if not token or not chat:
        logger.warning("Telegram NOT configured")
        return None
    return TelegramNotifier(
        token=token, chat_id=chat,
        min_confidence=config["tg_min_confidence"],
    )


def build_crypto_bot(config: dict, telegram):
    from trading.connectors.binance_conn import BinanceConnector
    from trading.connectors.alpaca_conn import AlpacaConnector
    from trading.risk_manager import RiskManager
    from trading.bot import TradingBot

    exchange = config["exchange"]
    if exchange in ("binance", "bybit", "okx"):
        connector = BinanceConnector(
            api_key=config["api_key"], api_secret=config["api_secret"],
            paper=config["paper"], exchange_id=exchange,
        )
    else:
        connector = AlpacaConnector(
            api_key=config["api_key"], api_secret=config["api_secret"],
            paper=config["paper"],
        )

    strategies = _make_strategies(config["symbols"], config["strategies"])
    if not strategies:
        from trading.strategies.macd_ema_strategy import MACDEMAStrategy
        strategies = [MACDEMAStrategy(config["symbols"][0])]

    risk = RiskManager(
        max_risk_per_trade_pct=config["risk_per_trade"],
        stop_loss_pct=config["stop_loss_pct"],
        take_profit_pct=config["take_profit_pct"],
        max_open_positions=config["max_positions"],
    )
    return TradingBot(
        connector=connector, strategies=strategies,
        risk_manager=risk, interval_seconds=config["interval"],
        broadcast_fn=None, telegram=telegram,
    )


def build_forex_bot(config: dict, telegram):
    """Signal-only bot for Gold / FX using Yahoo Finance data."""
    from trading.connectors.yahoo_conn import YahooConnector
    from trading.risk_manager import RiskManager
    from trading.bot import TradingBot

    connector = YahooConnector()
    strategies = _make_strategies(config["forex_symbols"], config["strategies"])
    if not strategies:
        from trading.strategies.macd_ema_strategy import MACDEMAStrategy
        strategies = [MACDEMAStrategy(config["forex_symbols"][0])]

    # max_open_positions=0 → strategies run + Telegram alerts sent, but no real orders
    risk = RiskManager(max_open_positions=0)
    bot = TradingBot(
        connector=connector, strategies=strategies,
        risk_manager=risk, interval_seconds=config["forex_interval"],
        broadcast_fn=None, telegram=telegram,
    )
    # Mark as signal-only so bot.start() skips Telegram polling (already started by crypto bot)
    bot._skip_telegram_polling = True
    return bot

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_stop_signal = asyncio.Event()


async def _run_backtest(crypto_bot, config: dict, telegram):
    """Fetch 500 candles on first symbol, run backtest, apply best SL/TP to all strategies."""
    from trading.strategies.macd_ema_strategy import MACDEMAStrategy
    macd_strats = [s for s in crypto_bot.strategies if isinstance(s, MACDEMAStrategy)]
    if not macd_strats:
        return

    symbol = macd_strats[0].symbol
    logger.info("Running SL/TP backtest on %s (500 candles 15m)…", symbol)
    try:
        candles = await crypto_bot.connector.fetch_ohlcv(symbol, timeframe="15m", limit=500)
        stats, best = await macd_strats[0].backtest(candles)

        if not stats:
            logger.warning("Backtest returned no results")
            return

        # Log full stats table
        header = f"{'Config':<22} {'Trades':>6} {'WR%':>6} {'PF':>6} {'R total':>8}"
        logger.info("Backtest results for %s:\n%s", symbol, header)
        for key, v in sorted(stats.items(), key=lambda x: -x[1]["total_r"]):
            logger.info("  %-22s  %6d  %5.1f%%  %5.2f  %+7.1fR",
                        key, v["trades"], v["win_rate"], v["profit_factor"], v["total_r"])

        if best:
            sl_m, rr = best
            logger.info("Best config: SL=%.1fxATR  RR=1:%.1f — applying to all strategies", sl_m, rr)
            for s in macd_strats:
                s.sl_atr_mult = sl_m
                s.rr_ratio    = rr

            if telegram:
                best_stat = stats.get(f"SL={sl_m}xATR  RR=1:{rr}", {})
                telegram.notify(
                    f"📊 *Backtest complete* ({symbol} 500×15m)\n"
                    f"Best: SL=`{sl_m}×ATR`  R:R=`1:{rr}`\n"
                    f"WR: `{best_stat.get('win_rate',0):.1f}%` | "
                    f"PF: `{best_stat.get('profit_factor',0):.2f}` | "
                    f"Trades: `{best_stat.get('trades',0)}`\n"
                    f"_Applied to live bot_"
                )
        else:
            logger.warning("Backtest: not enough trades to pick best config, using defaults")

    except Exception as e:
        logger.warning("Backtest failed (non-fatal): %s", e)


async def main():
    config = build_config()
    logger.info("=== Bot starting [%s] crypto=%s forex=%s ===",
                "PAPER" if config["paper"] else "LIVE",
                config["symbols"], config["forex_symbols"])

    telegram = _make_telegram(config)

    # Crypto bot (Binance)
    crypto_bot = build_crypto_bot(config, telegram)

    # Forex / Gold signal bot (Yahoo Finance)
    forex_bot = build_forex_bot(config, telegram) if config["forex_enabled"] else None

    loop = asyncio.get_event_loop()

    def _handle_signal():
        logger.info("Shutdown signal received")
        _stop_signal.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, RuntimeError):
            pass

    if telegram:
        telegram.get_state_fn = crypto_bot.get_state
        telegram.stop_bot_fn  = lambda: _stop_signal.set()

    # Auto-optimize SL/TP via backtest on first symbol
    await _run_backtest(crypto_bot, config, telegram)

    # Start both bots concurrently
    tasks = [asyncio.create_task(crypto_bot.start())]
    if forex_bot:
        tasks.append(asyncio.create_task(forex_bot.start()))
        logger.info("Forex signal bot started: %s", config["forex_symbols"])

    await _stop_signal.wait()

    logger.info("Stopping all bots...")
    await crypto_bot.stop()
    if forex_bot:
        await forex_bot.stop()

    # Close exchange sessions to prevent unclosed resource warnings
    try:
        await crypto_bot.connector.close()
    except Exception:
        pass

    logger.info("All bots stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
