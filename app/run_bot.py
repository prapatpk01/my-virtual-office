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
        "exchange":        os.environ.get("EXCHANGE", "binance"),
        "api_key":         os.environ.get("EXCHANGE_API_KEY", ""),
        "api_secret":      os.environ.get("EXCHANGE_API_SECRET", ""),
        "api_passphrase":  os.environ.get("EXCHANGE_PASSPHRASE", ""),  # OKX required
        "paper":           _env_bool("PAPER_TRADING", True),
        # ── Margin / leverage ──────────────────────────────────────────────────
        # MARGIN_MODE: "cross" | "isolated" | "" (spot)
        # MARKET_TYPE: "swap" for OKX Perpetual Futures, "" for spot/margin
        "margin_mode": os.environ.get("MARGIN_MODE", ""),
        "market_type": os.environ.get("MARKET_TYPE", ""),   # "swap" = futures
        "leverage":    int(os.environ.get("LEVERAGE", "20")),
        # ── OANDA ─────────────────────────────────────────────────────────────
        "oanda_api_key":   os.environ.get("OANDA_API_KEY", ""),
        "oanda_account_id":os.environ.get("OANDA_ACCOUNT_ID", ""),
        "oanda_env":       os.environ.get("OANDA_ENV", "practice"),
        # ── Candles / timing ──────────────────────────────────────────────────
        "symbols":      _env_list("SYMBOLS", "BTC/USDT"),
        "candle_tf":    os.environ.get("CANDLE_TF", "1h"),        # 1h: proven WR for MCDX + Sentinel
        "candle_limit": int(os.environ.get("CANDLE_LIMIT", "300")),
        "interval":     int(os.environ.get("INTERVAL_SECONDS", "60")),
        # ── Strategies ────────────────────────────────────────────────────────
        # RSI+MACD disabled by default — BTC 15m uses MCDX + Sentinel only
        "strategies": {
            "mcdx":      _env_bool("STRATEGY_MCDX",      True),
            "sentinel":  _env_bool("STRATEGY_SENTINEL",   True),
            "rsi_macd":  _env_bool("STRATEGY_RSI_MACD",   False),
            "utbot_wt":  _env_bool("STRATEGY_UTBOT_WT",   False),
            "sjutbot":   _env_bool("STRATEGY_SJUTBOT",    False),
        },
        # ── Risk / SL / TP ────────────────────────────────────────────────────
        "risk_per_trade":    float(os.environ.get("RISK_PER_TRADE",    "0.02")),
        # Fixed USDT margin per trade — overrides risk_per_trade when > 0
        "fixed_trade_usdt":  float(os.environ.get("FIXED_TRADE_USDT",  "20")),
        # BTC 1H 10x margin: SL 1.5%, TP 3.0%
        # Net RR ~1:1.83 after 0.25% fees | break-even WR = 35%
        # At 10x: loss ~15% capital | win ~27.5% capital per trade
        "stop_loss_pct":   float(os.environ.get("STOP_LOSS_PCT",   "0.015")),   # 1.5%
        "take_profit_pct": float(os.environ.get("TAKE_PROFIT_PCT", "0.030")),   # 3.0%
        # Max 2 positions: 1 per strategy (MCDX + Sentinel)
        "max_positions":   int(os.environ.get("MAX_POSITIONS",     "2")),
        "max_drawdown":    float(os.environ.get("MAX_DRAWDOWN_PCT", "0.30")),
        # ── Telegram ──────────────────────────────────────────────────────────
        "telegram_token":   os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID",   ""),
        "tg_min_confidence": float(os.environ.get("TG_MIN_CONFIDENCE", "0.5")),
        # ── Forex / Gold signal-only ───────────────────────────────────────────
        "forex_symbols": _env_list("FOREX_SYMBOLS", "XAUUSD"),
        "forex_enabled": _env_bool("FOREX_SIGNALS", True),
        "forex_interval": int(os.environ.get("FOREX_INTERVAL_SECONDS", "60")),
    }

# ---------------------------------------------------------------------------
# Bot factories
# ---------------------------------------------------------------------------

def _make_strategies(symbols: list, flags: dict):
    from trading.strategies.mcdx_strategy import MCDXStrategy
    from trading.strategies.sentinel_strategy import SentinelStrategy
    from trading.strategies.rsi_macd import RSIMACDStrategy
    from trading.strategies.utbot_wt_strategy import UTBotWTStrategy
    from trading.strategies.sjutbot_strategy import SJUTBotStrategy

    # ── Dual-param MCDX mode: P1(rv≥0.8) + P2(rv≥1.2), same SL/TP ────────
    # Activate with MCDX_DUAL=true in .env.  Each param set gets its own
    # position slot so both can hold simultaneously (max 2 positions total).
    mcdx_dual = _env_bool("MCDX_DUAL", False)
    mcdx_dwcs_buy  = int(os.environ.get("MCDX_DWCS_BUY", "57"))
    mcdx_p1_rvol   = float(os.environ.get("MCDX_P1_RVOL", "0.8"))
    mcdx_p2_rvol   = float(os.environ.get("MCDX_P2_RVOL", "1.2"))
    mtf_fast_ema   = int(os.environ.get("MTF_FAST_EMA", "21"))
    mtf_slow_ema   = int(os.environ.get("MTF_SLOW_EMA", "50"))

    strategies = []
    for sym in symbols:
        if flags.get("mcdx"):
            if mcdx_dual:
                # P1 — aggressive entry (rv≥0.8, more trades)
                strategies.append(MCDXStrategy(sym, params={
                    "name":         "MCDX-P1",
                    "dwcs_buy":     mcdx_dwcs_buy,
                    "dwcs_sell":    100 - mcdx_dwcs_buy,
                    "rvol_min":     mcdx_p1_rvol,
                    "mtf_fast_ema": mtf_fast_ema,
                    "mtf_slow_ema": mtf_slow_ema,
                }))
                # P2 — selective entry (rv≥1.2, higher WR)
                strategies.append(MCDXStrategy(sym, params={
                    "name":         "MCDX-P2",
                    "dwcs_buy":     mcdx_dwcs_buy,
                    "dwcs_sell":    100 - mcdx_dwcs_buy,
                    "rvol_min":     mcdx_p2_rvol,
                    "mtf_fast_ema": mtf_fast_ema,
                    "mtf_slow_ema": mtf_slow_ema,
                }))
                logger.info(
                    "MCDX Dual-Param mode: P1(rv≥%.1f) + P2(rv≥%.1f) "
                    "dwcs_buy=%d  MTF EMA(%d/%d)",
                    mcdx_p1_rvol, mcdx_p2_rvol, mcdx_dwcs_buy,
                    mtf_fast_ema, mtf_slow_ema,
                )
            else:
                strategies.append(MCDXStrategy(sym))
        if flags.get("sentinel"):  strategies.append(SentinelStrategy(sym))
        if flags.get("rsi_macd"):  strategies.append(RSIMACDStrategy(sym))
        if flags.get("utbot_wt"):  strategies.append(UTBotWTStrategy(sym))
        if flags.get("sjutbot"):   strategies.append(SJUTBotStrategy(sym))
    if not strategies:
        logger.warning("No strategies enabled — defaulting to MCDXStrategy")
        from trading.strategies.mcdx_strategy import MCDXStrategy
        for sym in symbols:
            strategies.append(MCDXStrategy(sym))
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

    # Paper trading: OKX uses real market data (OHLCV is public, no auth needed)
    # so we keep BinanceConnector with paper=True — simulates orders but gets
    # real prices. Other exchanges fall back to Yahoo Finance.
    if config["paper"] and exchange in ("binance", "bybit"):
        connector = YahooConnector()
        logger.info("Paper trading: using Yahoo Finance for market data (exchange=%s)", exchange)
    elif exchange == "oanda":
        connector = OANDAConnector(
            api_key=config["oanda_api_key"],
            account_id=config["oanda_account_id"],
            paper=config["paper"],
            env=config["oanda_env"],
        )
        logger.info("OANDA connector: env=%s paper=%s account=%s",
                    config["oanda_env"], config["paper"],
                    config["oanda_account_id"][:8] + "..." if config["oanda_account_id"] else "—")
    elif exchange in ("binance", "bybit", "okx"):
        connector = BinanceConnector(
            api_key=config["api_key"], api_secret=config["api_secret"],
            paper=config["paper"], exchange_id=exchange,
            passphrase=config.get("api_passphrase", ""),
            margin_mode=config.get("margin_mode", ""),
            market_type=config.get("market_type", ""),
            leverage=config.get("leverage", 1),
        )
    else:
        connector = AlpacaConnector(
            api_key=config["api_key"], api_secret=config["api_secret"],
            paper=config["paper"],
        )

    strategies = _make_strategies(config["symbols"], config["strategies"])

    risk = RiskManager(
        max_risk_per_trade_pct=config["risk_per_trade"],
        stop_loss_pct=config["stop_loss_pct"],
        take_profit_pct=config["take_profit_pct"],
        max_open_positions=config["max_positions"],
        max_drawdown_pct=config["max_drawdown"],
        fixed_trade_usdt=config["fixed_trade_usdt"],
        leverage=config["leverage"],
    )
    mkt = config.get("market_type", "") or config.get("margin_mode", "") or "spot"
    size_info = (f"fixed={config['fixed_trade_usdt']:.0f}USDT"
                 if config["fixed_trade_usdt"] > 0
                 else f"risk={config['risk_per_trade']*100:.1f}%")
    logger.info(
        "Bot config: SL=%.2f%%  TP=%.2f%%  size=%s  lev=x%d  mode=%s  positions=%d  strategies=%s",
        config["stop_loss_pct"] * 100, config["take_profit_pct"] * 100,
        size_info, config["leverage"], mkt.upper(),
        config["max_positions"], [s.name for s in strategies],
    )
    return TradingBot(
        connector=connector, strategies=strategies,
        risk_manager=risk, interval_seconds=config["interval"],
        broadcast_fn=None, telegram=telegram,
        fixed_sl_pct=config["stop_loss_pct"],
        fixed_tp_pct=config["take_profit_pct"],
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
    """Fetch candles on first symbol, run backtest on strategies that support it."""
    # MCDX / Sentinel / AISignal don't have a backtest() method — skip silently
    backtestable = [s for s in crypto_bot.strategies if hasattr(s, "backtest")]
    if not backtestable:
        logger.info("Startup backtest skipped (active strategies use fixed SL/TP from config)")
        return
    strat = backtestable[0]
    symbol = strat.symbol
    tf = config.get("candle_tf", "15m")
    bt_limit = 1500 if tf == "15m" else 500
    logger.info("Running SL/TP backtest on %s (%d candles %s)…", symbol, bt_limit, tf)
    try:
        candles = await crypto_bot.connector.fetch_ohlcv(symbol, timeframe=tf, limit=bt_limit)
        stats, best = await strat.backtest(candles)

        if not stats:
            logger.warning("Backtest returned no results")
            return

        # Log full stats table
        header = f"{'Config':<22} {'Trades':>6} {'WR%':>6} {'PF':>6} {'R total':>8}"
        logger.info("Backtest results for %s:\n%s", symbol, header)
        for key, v in sorted(stats.items(), key=lambda x: -x[1]["total_r"]):
            logger.info("  %-22s  %6d  %5.1f%%  %5.2f  %+7.1fR",
                        key, v["trades"], v["win_rate"], v["profit_factor"], v["total_r"])

        n_candles = len(candles)
        if best:
            sl_m, rr = best
            logger.info("Best config: SL=%.1fxATR  RR=1:%.1f — applying to all strategies", sl_m, rr)
            for s in backtestable:
                s.sl_atr_mult = sl_m
                s.rr_ratio    = rr

            if telegram:
                best_stat = stats.get(f"SL={sl_m}xATR  RR=1:{rr}", {})
                telegram.notify(
                    f"📊 *Backtest complete* ({symbol} {n_candles}×{tf})\n"
                    f"Best: SL=`{sl_m}×ATR`  R:R=`1:{rr}`\n"
                    f"WR: `{best_stat.get('win_rate',0):.1f}%` | "
                    f"PF: `{best_stat.get('profit_factor',0):.2f}` | "
                    f"Trades: `{best_stat.get('trades',0)}`\n"
                    f"_Applied to live bot_"
                )
        else:
            logger.warning("Backtest: not enough trades to pick best config, using defaults")
            if telegram:
                top = sorted(stats.items(), key=lambda x: -x[1]["total_r"])[:2]
                lines = "\n".join(
                    f"`{k}` WR:{v['win_rate']:.0f}% T:{v['trades']}"
                    for k, v in top
                ) if top else "_no signals found_"
                telegram.notify(
                    f"📊 *Backtest complete* ({symbol} {n_candles}×{tf})\n"
                    f"Not enough trades — using default SL/TP\n{lines}"
                )

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
        telegram.get_stats_fn = crypto_bot.get_stats
        telegram.stop_bot_fn  = lambda: _stop_signal.set()
        telegram.start_bot_fn = lambda: {"message": "Bot is already running"}

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
