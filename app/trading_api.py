"""
Trading Bot API handlers.
Called from server.py's do_GET / do_POST to manage the global bot instance.
"""
import asyncio
import json
import logging
import os
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger("trading_api")

# Global bot registry: exchange_key -> TradingBot instance
_bots: dict[str, Any] = {}
_bot_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        t = threading.Thread(target=_loop.run_forever, daemon=True, name="trading-event-loop")
        t.start()
    return _loop


def _run_async(coro):
    """Run a coroutine on the trading event loop and block until done."""
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


# -----------------------------------------------------------------------
# Bot factory
# -----------------------------------------------------------------------

def _build_bot(config: dict, broadcast_fn: Callable):
    from trading.connectors.binance_conn import BinanceConnector
    from trading.connectors.alpaca_conn import AlpacaConnector
    from trading.strategies.ma_crossover import MACrossoverStrategy
    from trading.strategies.rsi_macd import RSIMACDStrategy
    from trading.strategies.grid_trading import GridTradingStrategy
    from trading.strategies.ai_signal import AISignalStrategy
    from trading.strategies.mcdx_strategy import MCDXStrategy
    from trading.strategies.sentinel_strategy import SentinelStrategy
    from trading.strategies.rvol_strategy import RVolStrategy
    from trading.risk_manager import RiskManager
    from trading.bot import TradingBot
    from trading.telegram_notifier import TelegramNotifier

    exchange = config.get("exchange", "binance")
    symbols = config.get("symbols", ["BTC/USDT"])
    paper = config.get("paper", True)
    api_key = config.get("api_key", "")
    api_secret = config.get("api_secret", "")
    interval = int(config.get("interval", 60))
    strategy_flags = config.get("strategies", {})

    # Set Anthropic key if provided
    if config.get("anthropic_key"):
        os.environ["ANTHROPIC_API_KEY"] = config["anthropic_key"]

    # Connector
    if exchange in ("binance", "bybit", "okx"):
        connector = BinanceConnector(api_key=api_key, api_secret=api_secret,
                                     paper=paper, exchange_id=exchange)
    else:
        connector = AlpacaConnector(api_key=api_key, api_secret=api_secret, paper=paper)

    # Strategies — one instance per symbol per enabled strategy
    strategies = []
    for symbol in symbols:
        if strategy_flags.get("ma_crossover", True):
            strategies.append(MACrossoverStrategy(symbol))
        if strategy_flags.get("rsi_macd", True):
            strategies.append(RSIMACDStrategy(symbol))
        if strategy_flags.get("grid_trading", False):
            strategies.append(GridTradingStrategy(symbol))
        if strategy_flags.get("ai_signal", False):
            strategies.append(AISignalStrategy(symbol))
        if strategy_flags.get("mcdx", False):
            strategies.append(MCDXStrategy(symbol))
        if strategy_flags.get("sentinel", False):
            strategies.append(SentinelStrategy(symbol))
        if strategy_flags.get("rvol", True):
            strategies.append(RVolStrategy(symbol))

    if not strategies:
        strategies.append(MACrossoverStrategy(symbols[0]))

    risk = RiskManager(
        max_risk_per_trade_pct=float(config.get("risk_per_trade", 0.02)),
        stop_loss_pct=float(config.get("stop_loss_pct", 0.03)),
        take_profit_pct=float(config.get("take_profit_pct", 0.06)),
        max_open_positions=int(config.get("max_positions", 5)),
    )

    # Telegram (optional)
    tg_token = config.get("telegram_token", "").strip() or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat  = config.get("telegram_chat_id", "").strip() or os.environ.get("TELEGRAM_CHAT_ID", "")
    telegram = None
    if tg_token and tg_chat:
        telegram = TelegramNotifier(
            token=tg_token,
            chat_id=tg_chat,
            min_confidence=float(config.get("tg_min_confidence", 0.5)),
        )

    bot = TradingBot(
        connector=connector,
        strategies=strategies,
        risk_manager=risk,
        interval_seconds=interval,
        broadcast_fn=broadcast_fn,
        telegram=telegram,
    )

    # Wire command callbacks so /start_bot and /stop_bot work from Telegram
    if telegram:
        telegram.get_state_fn  = bot.get_state
        telegram.start_bot_fn  = lambda: handle_start(config, broadcast_fn)
        telegram.stop_bot_fn   = lambda: handle_stop()

    return bot


# -----------------------------------------------------------------------
# Public API handlers
# -----------------------------------------------------------------------

def handle_get_state(bot_key: str = "default") -> dict:
    with _bot_lock:
        bot = _bots.get(bot_key)
    if not bot:
        return {"running": False, "paper": True, "balance": 0, "equity": 0,
                "pnl_total": 0, "positions": [], "recent_trades": [], "signals": [],
                "error": "Bot not started"}
    return bot.get_state()


def handle_start(config: dict, broadcast_fn: Callable, bot_key: str = "default") -> dict:
    with _bot_lock:
        existing = _bots.get(bot_key)
        if existing and existing.state.running:
            return {"ok": False, "error": "Bot is already running"}

        try:
            bot = _build_bot(config, broadcast_fn)
            _bots[bot_key] = bot
        except Exception as e:
            logger.exception("Failed to build bot")
            return {"ok": False, "error": str(e)}

    try:
        _run_async(bot.start())
        return {"ok": True, "message": f"Bot started (paper={bot.connector.paper})"}
    except Exception as e:
        logger.exception("Failed to start bot")
        return {"ok": False, "error": str(e)}


def handle_stop(bot_key: str = "default") -> dict:
    with _bot_lock:
        bot = _bots.get(bot_key)
    if not bot:
        return {"ok": False, "error": "No bot running"}
    try:
        _run_async(bot.stop())
        return {"ok": True, "message": "Bot stopped"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_telegram_test(bot_key: str = "default") -> dict:
    """Send a test message via Telegram to verify the connection."""
    with _bot_lock:
        bot = _bots.get(bot_key)
    if not bot or not bot.telegram:
        # Try env vars directly
        import os
        from trading.telegram_notifier import TelegramNotifier
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return {"ok": False, "error": "No Telegram token/chat_id configured"}
        tg = TelegramNotifier(token=token, chat_id=chat_id)
        tg.notify("✅ *Trading Bot* — Telegram test message OK!")
        return {"ok": True, "message": "Test message sent"}
    bot.telegram.notify("✅ *Trading Bot* — Telegram test message OK!")
    return {"ok": True, "message": "Test message sent"}


def handle_manual_trade(body: dict, bot_key: str = "default") -> dict:
    with _bot_lock:
        bot = _bots.get(bot_key)
    if not bot:
        return {"ok": False, "error": "Bot not running"}
    symbol = body.get("symbol", "")
    side = body.get("side", "")
    amount = float(body.get("amount", 0))
    if not symbol or side not in ("buy", "sell") or amount <= 0:
        return {"ok": False, "error": "Invalid trade params"}
    try:
        _run_async(bot.manual_signal(symbol, side, amount, reason="manual-ui"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
