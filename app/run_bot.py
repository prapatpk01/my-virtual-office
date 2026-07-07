"""
Standalone Trading Bot runner — no web UI required.
Runs the bot + Telegram integration 24/7 on any cloud platform.

Usage:
    python run_bot.py

Config via environment variables (see .env.example) or a .env file.

Strategy modes (STRATEGY env var):
  ai_expert  — Full 9-layer AI Expert analysis (default, institutional grade)
  mcdx       — Legacy MCDX strategy
  wt_adx     — Legacy WaveTrend + ADX strategy
"""
import asyncio
import logging
import os
import signal
import sys


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

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_bot")


def _env_list(key: str, default: str) -> list[str]:
    val = os.environ.get(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]

def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _env_bool(key: str, default: bool) -> bool:
    return _parse_bool(os.environ.get(key), default)


def validate_config(config: dict) -> None:
    """Fail fast on unsafe live configuration before importing trading deps."""
    exchange = str(config.get("exchange", "")).lower()
    if config.get("paper", True):
        return

    if not _env_bool("LIVE_TRADING_CONFIRMED", False):
        raise ValueError(
            "Live trading requires LIVE_TRADING_CONFIRMED=true. "
            "Keep PAPER_TRADING=true on Railway until you are ready for real orders."
        )

    if exchange in {"binance", "bybit", "okx", "alpaca"}:
        if not config.get("api_key") or not config.get("api_secret"):
            raise ValueError(f"Live trading on {exchange} requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET")
    if exchange == "okx" and not config.get("api_passphrase"):
        raise ValueError("Live trading on OKX requires EXCHANGE_PASSPHRASE")
    if exchange == "oanda" and (not config.get("oanda_api_key") or not config.get("oanda_account_id")):
        raise ValueError("Live trading on OANDA requires OANDA_API_KEY and OANDA_ACCOUNT_ID")


def build_config() -> dict:
    return {
        "exchange":        os.environ.get("EXCHANGE", "binance"),
        "api_key":         os.environ.get("EXCHANGE_API_KEY", ""),
        "api_secret":      os.environ.get("EXCHANGE_API_SECRET", ""),
        "api_passphrase":  os.environ.get("EXCHANGE_PASSPHRASE", ""),
        "paper":           _env_bool("PAPER_TRADING", True),
        "oanda_api_key":   os.environ.get("OANDA_API_KEY", ""),
        "oanda_account_id":os.environ.get("OANDA_ACCOUNT_ID", ""),
        "oanda_env":       os.environ.get("OANDA_ENV", "practice"),
        "symbols":      _env_list("SYMBOLS", "BTC/USDT"),
        "candle_tf":    os.environ.get("CANDLE_TF", "15m"),
        "candle_limit": int(os.environ.get("CANDLE_LIMIT", "300")),
        "interval":     int(os.environ.get("INTERVAL_SECONDS", "60")),
        # Strategy selection: ai_expert | mcdx | wt_adx
        "strategy_mode": os.environ.get("STRATEGY", "ai_expert"),
        "strategies": {
            "mcdx":      _env_bool("STRATEGY_MCDX",      False),
            "wt_adx":    _env_bool("STRATEGY_WT_ADX",    False),
            "ai_expert": _env_bool("STRATEGY_AI_EXPERT", True),
        },
        "ai_expert_min_confidence": float(os.environ.get("AI_EXPERT_MIN_CONFIDENCE", "70")),
        "ai_expert_strict":         _env_bool("AI_EXPERT_STRICT", False),
        "risk_per_trade":  float(os.environ.get("RISK_PER_TRADE",  "0.02")),
        "stop_loss_pct":   float(os.environ.get("STOP_LOSS_PCT",   "0.03")),
        "take_profit_pct": float(os.environ.get("TAKE_PROFIT_PCT", "0.06")),
        "max_positions":   int(os.environ.get("MAX_POSITIONS",     "3")),
        "max_drawdown":    float(os.environ.get("MAX_DRAWDOWN_PCT", "0.30")),
        "telegram_token":   os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID",   ""),
        "tg_min_confidence": float(os.environ.get("TG_MIN_CONFIDENCE", "0.7")),
        "forex_symbols": _env_list("FOREX_SYMBOLS", "XAUUSD"),
        "forex_enabled": _env_bool("FOREX_SIGNALS", True),
        "forex_interval": int(os.environ.get("FOREX_INTERVAL_SECONDS", "60")),
        "railway_environment": os.environ.get("RAILWAY_ENVIRONMENT_NAME", ""),
        "railway_service": os.environ.get("RAILWAY_SERVICE_NAME", ""),
    }


def _make_strategies(symbols: list, config: dict):
    mode   = config.get("strategy_mode", "ai_expert")
    flags  = config.get("strategies", {})
    strategies = []

    for sym in symbols:
        if mode == "ai_expert" or flags.get("ai_expert", True):
            from trading.strategies.ai_expert_strategy import AIExpertStrategy
            strategies.append(AIExpertStrategy(
                sym,
                min_confidence=config.get("ai_expert_min_confidence", 70.0),
                require_all_checks=config.get("ai_expert_strict", False),
            ))
        elif flags.get("mcdx", False):
            from trading.strategies.mcdx_strategy import MCDXStrategy
            strategies.append(MCDXStrategy(sym))
        elif flags.get("wt_adx", False):
            from trading.strategies.wt_adx_strategy import WTADXStrategy
            strategies.append(WTADXStrategy(sym))
        else:
            from trading.strategies.ai_expert_strategy import AIExpertStrategy
            strategies.append(AIExpertStrategy(sym))

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
    from trading.connectors.oanda_conn import OANDAConnector
    from trading.connectors.yahoo_conn import YahooConnector
    from trading.risk_manager import RiskManager
    from trading.bot import TradingBot

    exchange = config["exchange"]

    if config["paper"] and exchange in ("binance", "bybit", "okx"):
        connector = YahooConnector()
        logger.info("Paper trading: using Yahoo Finance (exchange=%s)", exchange)
    elif exchange == "oanda":
        connector = OANDAConnector(
            api_key=config["oanda_api_key"],
            account_id=config["oanda_account_id"],
            paper=config["paper"],
            env=config["oanda_env"],
        )
    elif exchange in ("binance", "bybit", "okx"):
        connector = BinanceConnector(
            api_key=config["api_key"], api_secret=config["api_secret"],
            paper=config["paper"], exchange_id=exchange,
            passphrase=config.get("api_passphrase", ""),
        )
    else:
        connector = AlpacaConnector(
            api_key=config["api_key"], api_secret=config["api_secret"],
            paper=config["paper"],
        )

    strategies = _make_strategies(config["symbols"], config)
    if not strategies:
        from trading.strategies.ai_expert_strategy import AIExpertStrategy
        strategies = [AIExpertStrategy(config["symbols"][0])]

    risk = RiskManager(
        max_risk_per_trade_pct=config["risk_per_trade"],
        stop_loss_pct=config["stop_loss_pct"],
        take_profit_pct=config["take_profit_pct"],
        max_open_positions=config["max_positions"],
        max_drawdown_pct=config["max_drawdown"],
    )
    return TradingBot(
        connector=connector, strategies=strategies,
        risk_manager=risk, interval_seconds=config["interval"],
        broadcast_fn=None, telegram=telegram,
    )


def build_forex_bot(config: dict, telegram):
    from trading.connectors.yahoo_conn import YahooConnector
    from trading.risk_manager import RiskManager
    from trading.bot import TradingBot

    connector  = YahooConnector()
    strategies = _make_strategies(config["forex_symbols"], config)
    if not strategies:
        from trading.strategies.ai_expert_strategy import AIExpertStrategy
        strategies = [AIExpertStrategy(config["forex_symbols"][0])]

    risk = RiskManager(max_open_positions=0)  # signal-only
    bot  = TradingBot(
        connector=connector, strategies=strategies,
        risk_manager=risk, interval_seconds=config["forex_interval"],
        broadcast_fn=None, telegram=telegram,
    )
    bot._skip_telegram_polling = True
    return bot


_stop_signal = asyncio.Event()


async def _run_backtest(crypto_bot, config: dict, telegram):
    backtestable = [s for s in crypto_bot.strategies if hasattr(s, "backtest")]
    if not backtestable:
        return
    strat = backtestable[0]
    symbol = strat.symbol
    tf     = config.get("candle_tf", "15m")
    bt_limit = 1500 if tf == "15m" else 500
    logger.info("Running SL/TP backtest on %s (%d candles %s)…", symbol, bt_limit, tf)
    try:
        candles = await crypto_bot.connector.fetch_ohlcv(symbol, timeframe=tf, limit=bt_limit)
        stats, best = await strat.backtest(candles)
        if not stats:
            logger.warning("Backtest returned no results")
            return
        header = f"{'Config':<22} {'Trades':>6} {'WR%':>6} {'PF':>6} {'R total':>8}"
        logger.info("Backtest results for %s:\n%s", symbol, header)
        for key, v in sorted(stats.items(), key=lambda x: -x[1]["total_r"]):
            logger.info("  %-22s  %6d  %5.1f%%  %5.2f  %+7.1fR",
                        key, v["trades"], v["win_rate"], v["profit_factor"], v["total_r"])
        if best:
            sl_m, rr = best
            logger.info("Best: SL=%.1fxATR RR=1:%.1f", sl_m, rr)
            for s in backtestable:
                if hasattr(s, "sl_atr_mult"):
                    s.sl_atr_mult = sl_m
                if hasattr(s, "rr_ratio"):
                    s.rr_ratio = rr
    except Exception as e:
        logger.warning("Backtest failed (non-fatal): %s", e)


async def main():
    config = build_config()
    logger.info(
        "=== AI Expert Bot starting [%s] mode=%s symbols=%s railway_env=%s ===",
        "PAPER" if config["paper"] else "LIVE",
        config["strategy_mode"], config["symbols"],
        config.get("railway_environment") or "local",
    )

    validate_config(config)
    telegram    = _make_telegram(config)
    if telegram:
        telegram.notify_service_event(
            "🚀",
            "Railway Trading Bot Deploy Started",
            [
                f"Mode: `{'PAPER' if config['paper'] else 'LIVE'}`",
                f"Exchange: `{config['exchange']}`",
                f"Symbols: `{', '.join(config['symbols'])}`",
                f"Railway env: `{config.get('railway_environment') or 'local'}`",
            ],
        )
    crypto_bot  = build_crypto_bot(config, telegram)
    forex_bot   = build_forex_bot(config, telegram) if config["forex_enabled"] else None

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
        telegram.get_state_fn    = crypto_bot.get_state
        telegram.get_stats_fn    = crypto_bot.get_stats
        telegram.get_insights_fn = crypto_bot.get_learning_insights
        telegram.stop_bot_fn     = lambda: _stop_signal.set()
        telegram.start_bot_fn    = lambda: {"message": "Bot is already running"}

    # Auto-optimize SL/TP via backtest (for legacy strategies)
    await _run_backtest(crypto_bot, config, telegram)

    tasks = [asyncio.create_task(crypto_bot.start())]
    if forex_bot:
        tasks.append(asyncio.create_task(forex_bot.start()))
        logger.info("Forex signal bot started: %s", config["forex_symbols"])

    await _stop_signal.wait()

    logger.info("Stopping all bots...")
    await crypto_bot.stop()
    if forex_bot:
        await forex_bot.stop()

    try:
        await crypto_bot.connector.close()
    except Exception:
        pass

    if telegram:
        telegram.notify_service_event("🛑", "Railway Trading Bot Stopped", ["Shutdown completed cleanly."])
    logger.info("All bots stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("Fatal startup/runtime error")
        try:
            config = build_config()
            telegram = _make_telegram(config)
            if telegram:
                telegram.notify_service_event("🚨", "Railway Trading Bot Error", [f"`{type(exc).__name__}: {exc}`"])
        except Exception:
            logger.exception("Failed to send fatal Telegram notification")
        raise
