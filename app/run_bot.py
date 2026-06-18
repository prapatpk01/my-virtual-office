"""
OKX Perpetual Futures trading bot runner.

Strategies  : MCDX (15m) + SJUTBot (1H) + UTBot (15m)
Symbols     : BTC/USDT:USDT, XAU/USDT:USDT  (configurable via SYMBOLS)
Exchange    : OKX, swap market, hedge mode

Config via Railway / environment variables — see the variable table below.
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

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_bot")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env_list(key: str, default: str) -> list[str]:
    return [s.strip() for s in os.environ.get(key, default).split(",") if s.strip()]

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    return default if not val else val.lower() in ("1", "true", "yes")

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


# ── Config ────────────────────────────────────────────────────────────────────

def build_config() -> dict:
    return {
        # ── Exchange credentials ───────────────────────────────────────────────
        "exchange":       os.environ.get("EXCHANGE",           "okx"),
        "api_key":        os.environ.get("EXCHANGE_API_KEY",   ""),
        "api_secret":     os.environ.get("EXCHANGE_API_SECRET",""),
        "api_passphrase": os.environ.get("EXCHANGE_PASSPHRASE",""),  # OKX required

        # ── Market mode ───────────────────────────────────────────────────────
        "paper":       _env_bool("PAPER_TRADING", False),
        "market_type": os.environ.get("MARKET_TYPE", "swap"),  # "swap" = perpetual futures
        "leverage":    _env_int("LEVERAGE", 20),

        # ── Symbols ───────────────────────────────────────────────────────────
        "symbols": _env_list("SYMBOLS", "BTC/USDT:USDT,XAU/USDT:USDT"),

        # ── Strategies ────────────────────────────────────────────────────────
        "strategy_mcdx":   _env_bool("STRATEGY_MCDX",   True),
        "strategy_sjutbot":_env_bool("STRATEGY_SJUTBOT", True),
        "strategy_utbot":  _env_bool("STRATEGY_UTBOT",   True),

        # ── MCDX tuning ───────────────────────────────────────────────────────
        "mcdx_dwcs_buy": _env_int("MCDX_DWCS_BUY", 57),      # DWCS buy threshold
        "mcdx_rvol":     _env_float("MCDX_RVOL",   0.8),     # Relative volume min

        # ── SJUTBot tuning ────────────────────────────────────────────────────
        # SL = SJUTBOT_SL_MULT × ATR,  TP = SJUTBOT_SL_MULT × SJUTBOT_RR × ATR
        # Default: SL=1.2×ATR, TP=1.62×ATR → R:R 1:1.35 (recommended ratio)
        "sjutbot_sl_mult": _env_float("SJUTBOT_SL_MULT", 1.2),
        "sjutbot_rr":      _env_float("SJUTBOT_RR",      1.35),

        # ── Risk / sizing ─────────────────────────────────────────────────────
        # FIXED_TRADE_USDT: margin reserved per trade (before leverage).
        #   BTC/USDT:USDT @ $100k, 20x: min 1 contract = 0.01 BTC = $1000 notional = $50 margin.
        #   XAU/USDT:USDT @ $3300, 20x: depends on OKX contract size (check live).
        "fixed_trade_usdt": _env_float("FIXED_TRADE_USDT", 50.0),

        # SL/TP percentages — applied when strategy metadata doesn't provide levels.
        # MCDX optimised: SL=1.5% TP=2.5% → +$43.60 / 5mo on $50 margin 20x.
        # SJUTBot / UTBot use ATR-based levels from their own metadata (these are fallback only).
        "stop_loss_pct":   _env_float("STOP_LOSS_PCT",   0.015),  # 1.5%
        "take_profit_pct": _env_float("TAKE_PROFIT_PCT", 0.025),  # 2.5%

        # Max open positions across all strategies + symbols.
        # 3 strategies × 2 symbols × 2 sides = 12 theoretical max.
        # Set lower to limit capital exposure (default: 6 = 3 strats × 2 symbols, 1 side each).
        "max_positions":  _env_int("MAX_POSITIONS",  6),
        "max_drawdown":   _env_float("MAX_DRAWDOWN_PCT", 0.30),  # 30% drawdown halt

        # ── Timing ────────────────────────────────────────────────────────────
        # Tick interval: how often the bot loops (fetches data + evaluates signals).
        # MCDX/UTBot work on 15m candles → 60s tick is fine.
        # SJUTBot works on 1H → signal fires once per hour at most.
        "interval": _env_int("INTERVAL_SECONDS", 60),

        # ── Telegram ──────────────────────────────────────────────────────────
        "telegram_token":    os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id":  os.environ.get("TELEGRAM_CHAT_ID",   ""),
        "tg_min_confidence": _env_float("TG_MIN_CONFIDENCE", 0.5),
    }


# ── Strategy builder ─────────────────────────────────────────────────────────

def build_strategies(symbols: list[str], cfg: dict) -> list:
    from trading.strategies.mcdx_strategy    import MCDXStrategy
    from trading.strategies.sjutbot_strategy import SJUTBotStrategy
    from trading.strategies.utbot_wt_strategy import UTBotWTStrategy

    strategies = []
    for sym in symbols:

        # ── MCDX Plus v3 (15m) ────────────────────────────────────────────────
        # Composite DWCS momentum + ADX + Supertrend + WaveTrend regime.
        # Optimised: SL=1.5%, TP=2.5%, dwcs_buy=57, rvol≥0.8  → +$43.60/5mo
        if cfg["strategy_mcdx"]:
            strategies.append(MCDXStrategy(sym, params={
                "tf":        "15m",
                "limit":     300,
                "dwcs_buy":  cfg["mcdx_dwcs_buy"],
                "dwcs_sell": 100 - cfg["mcdx_dwcs_buy"],
                "rvol_min":  cfg["mcdx_rvol"],
            }))

        # ── SJ-UTBot v2 (1H) ──────────────────────────────────────────────────
        # Heikin-Ashi × ATR Trailing Stop crossover.
        # SL = sl_mult × ATR,  TP = sl_mult × rr × ATR  (R:R = 1:rr)
        if cfg["strategy_sjutbot"]:
            strategies.append(SJUTBotStrategy(sym, params={
                "tf":      "1h",
                "limit":   200,
                "ut_mult": 0.30,                     # TSL ATR multiplier
                "ut_len":  14,                       # TSL ATR period
                "sl_len":  14,                       # SL/TP ATR period
                "sl_mult": cfg["sjutbot_sl_mult"],   # SL = 1.2 × ATR (recommended)
                "rr":      cfg["sjutbot_rr"],         # TP = 1.62 × ATR → R:R 1:1.35
            }))

        # ── UT Bot + WaveTrend (15m) ───────────────────────────────────────────
        # ATR trailing stop with WaveTrend momentum gate.
        if cfg["strategy_utbot"]:
            strategies.append(UTBotWTStrategy(sym, params={
                "tf":    "15m",
                "limit": 300,
            }))

    if not strategies:
        raise RuntimeError(
            "No strategies enabled — set STRATEGY_MCDX, STRATEGY_SJUTBOT, "
            "or STRATEGY_UTBOT to true"
        )

    logger.info("Strategies loaded: %s", [s.name for s in strategies])
    return strategies


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    cfg = build_config()

    # ── Live trading safety checks ────────────────────────────────────────────
    if not cfg["paper"]:
        missing = []
        if not cfg["api_key"]:        missing.append("EXCHANGE_API_KEY")
        if not cfg["api_secret"]:     missing.append("EXCHANGE_API_SECRET")
        if not cfg["api_passphrase"]: missing.append("EXCHANGE_PASSPHRASE")
        if missing:
            raise RuntimeError(
                f"LIVE TRADING: missing required credentials: {', '.join(missing)}"
            )
        logger.warning(
            "⚠️  LIVE TRADING — exchange=%s  symbols=%s  leverage=%dx  fixed_usdt=%.0f",
            cfg["exchange"], cfg["symbols"], cfg["leverage"], cfg["fixed_trade_usdt"],
        )

    mode = "PAPER" if cfg["paper"] else "LIVE"
    logger.info("=== Bot starting [%s] exchange=%s symbols=%s ===",
                mode, cfg["exchange"], cfg["symbols"])

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram = None
    if cfg["telegram_token"] and cfg["telegram_chat_id"]:
        from trading.telegram_notifier import TelegramNotifier
        telegram = TelegramNotifier(
            token=cfg["telegram_token"],
            chat_id=cfg["telegram_chat_id"],
            min_confidence=cfg["tg_min_confidence"],
        )
        logger.info("Telegram configured (chat_id=%s)", cfg["telegram_chat_id"])
    else:
        logger.warning("Telegram NOT configured — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")

    # ── Connector (OKX via CCXT) ──────────────────────────────────────────────
    from trading.connectors.binance_conn import BinanceConnector
    connector = BinanceConnector(
        api_key=cfg["api_key"],
        api_secret=cfg["api_secret"],
        paper=cfg["paper"],
        exchange_id=cfg["exchange"],
        passphrase=cfg["api_passphrase"],
        market_type=cfg["market_type"],
        leverage=cfg["leverage"],
    )

    # ── Strategies ────────────────────────────────────────────────────────────
    strategies = build_strategies(cfg["symbols"], cfg)

    # ── Risk manager ─────────────────────────────────────────────────────────
    from trading.risk_manager import RiskManager
    risk = RiskManager(
        stop_loss_pct=cfg["stop_loss_pct"],
        take_profit_pct=cfg["take_profit_pct"],
        max_open_positions=cfg["max_positions"],
        max_drawdown_pct=cfg["max_drawdown"],
        fixed_trade_usdt=cfg["fixed_trade_usdt"],
        leverage=cfg["leverage"],
    )
    logger.info(
        "Risk: SL=%.1f%%  TP=%.1f%%  fixed=%gUSDT  lev=%dx  max_pos=%d  drawdown=%.0f%%",
        cfg["stop_loss_pct"] * 100, cfg["take_profit_pct"] * 100,
        cfg["fixed_trade_usdt"], cfg["leverage"],
        cfg["max_positions"], cfg["max_drawdown"] * 100,
    )

    # ── Trading bot ───────────────────────────────────────────────────────────
    from trading.bot import TradingBot
    bot = TradingBot(
        connector=connector,
        strategies=strategies,
        risk_manager=risk,
        interval_seconds=cfg["interval"],
        telegram=telegram,
        fixed_sl_pct=cfg["stop_loss_pct"],
        fixed_tp_pct=cfg["take_profit_pct"],
    )

    # ── Backtest function (called by /backtest Telegram command) ─────────────
    async def _run_backtest() -> str:
        from trading.backtester import run_full_backtest, format_backtest_telegram
        from trading.strategies.mcdx_strategy     import MCDXStrategy
        from trading.strategies.sjutbot_strategy  import SJUTBotStrategy
        from trading.strategies.utbot_wt_strategy import UTBotWTStrategy

        bt_configs = []
        for sym in cfg["symbols"]:
            if cfg["strategy_mcdx"]:
                bt_configs.append({"cls": MCDXStrategy, "symbol": sym, "tf": "15m",
                                   "limit": 3000,
                                   "params": {"dwcs_buy": cfg["mcdx_dwcs_buy"],
                                              "dwcs_sell": 100 - cfg["mcdx_dwcs_buy"],
                                              "rvol_min": cfg["mcdx_rvol"]}})
            if cfg["strategy_sjutbot"]:
                bt_configs.append({"cls": SJUTBotStrategy, "symbol": sym, "tf": "1h",
                                   "limit": 1500,
                                   "params": {"ut_mult": 0.30, "ut_len": 14, "sl_len": 14,
                                              "sl_mult": cfg["sjutbot_sl_mult"],
                                              "rr": cfg["sjutbot_rr"]}})
            if cfg["strategy_utbot"]:
                bt_configs.append({"cls": UTBotWTStrategy, "symbol": sym, "tf": "15m",
                                   "limit": 3000, "params": {}})

        results = await run_full_backtest(
            connector=connector,
            strategy_configs=bt_configs,
            fixed_trade_usdt=cfg["fixed_trade_usdt"],
            leverage=cfg["leverage"],
            sl_pct=cfg["stop_loss_pct"],
            tp_pct=cfg["take_profit_pct"],
        )
        return format_backtest_telegram(
            results, cfg["fixed_trade_usdt"], cfg["leverage"], 3000
        )

    # ── Wire Telegram callbacks ───────────────────────────────────────────────
    if telegram:
        stop_event_ref: list = []  # forward reference

        def _tg_stop():
            if stop_event_ref:
                stop_event_ref[0].set()
            return {"message": "Bot stopping..."}

        telegram.get_state_fn = bot.get_state
        telegram.get_stats_fn = bot.get_stats
        telegram.stop_bot_fn  = _tg_stop
        telegram.start_bot_fn = lambda: {"message": "Bot is already running"}
        telegram.backtest_fn  = _run_backtest

    # ── Event loop + signal handlers ─────────────────────────────────────────
    stop_signal = asyncio.Event()
    if telegram:
        stop_event_ref.append(stop_signal)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_signal.set)
        except (NotImplementedError, RuntimeError):
            pass

    # ── Run ───────────────────────────────────────────────────────────────────
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
