"""
OKX Perpetual Futures trading bot runner.

Active strategies (4):
  MeanReversion  — 1H entry / 4H MTF  — R:R 1:1.25
  TrendCont      — 15m entry / 1H+4H MTF — R:R 1:2.0
  SmartMoney     — 15m entry / 1H+4H MTF — R:R 1:1.25
  SJUTBotV3      — 30m entry            — R:R 1:1.8

Symbols     : BTC/USDT:USDT  (configurable via SYMBOLS)
Exchange    : OKX, swap market, isolated margin, hedge mode
Max positions: 4 (1 per strategy at a time)

Config via Railway environment variables.
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
        "symbols": _env_list("SYMBOLS", "BTC/USDT:USDT"),

        # ── Active strategies ─────────────────────────────────────────────────
        "strategy_mean_reversion": _env_bool("STRATEGY_MEAN_REVERSION", True),
        "strategy_trend_cont":     _env_bool("STRATEGY_TREND_CONT",     True),
        "strategy_smart_money":    _env_bool("STRATEGY_SMART_MONEY",    True),
        "strategy_sjutbot_v3":     _env_bool("STRATEGY_SJUTBOT_V3",     True),

        # ── Mean Reversion (1H entry / 4H MTF) ───────────────────────────────
        # sl=1.2 tp=1.5 → R:R 1:1.25  BE WR 44%  backtest: WR 55.1% +$161/5mo
        "mr_sl_mult":         _env_float("MR_SL_MULT",        1.2),
        "mr_tp_mult":         _env_float("MR_TP_MULT",        1.5),
        "mr_rsi_oversold":    _env_float("MR_RSI_OVERSOLD",  45.0),
        "mr_rsi_overbought":  _env_float("MR_RSI_OVERBOUGHT",55.0),
        "mr_min_conditions":  _env_int("MR_MIN_CONDITIONS",     2),

        # ── Trend Continuation (15m entry / 1H+4H MTF) ───────────────────────
        # sl=1.2 tp=1.5 → R:R 1:1.25  BE WR 44%  backtest: WR 59.8% +$185/5mo
        # bias_gate=70 + cond=4/4 + pull=1% → selective, high WR
        "tc_sl_mult":         _env_float("TC_SL_MULT",        1.2),
        "tc_tp_mult":         _env_float("TC_TP_MULT",        1.5),
        "tc_rsi_min":         _env_float("TC_RSI_MIN",       35.0),
        "tc_rsi_max":         _env_float("TC_RSI_MAX",       75.0),
        "tc_pullback_pct":    _env_float("TC_PULLBACK_PCT",   0.010),
        "tc_vol_mult":        _env_float("TC_VOL_MULT",       1.0),
        "tc_bias_gate":       _env_float("TC_BIAS_GATE",     70.0),

        # ── Smart Money (15m entry / 1H+4H MTF) ──────────────────────────────
        # sl=1.2 tp=1.5 → R:R 1:1.25  BE WR 44%  backtest: WR 52.2% +$118/5mo
        "sm_sl_mult":         _env_float("SM_SL_MULT",        1.2),
        "sm_tp_mult":         _env_float("SM_TP_MULT",        1.5),
        "sm_bias_threshold":  _env_float("SM_BIAS_THRESHOLD",15.0),

        # ── SJUTBot v3 (30m) ─────────────────────────────────────────────────
        # filter_threshold=4: ALL 4 components must agree → very selective longs
        # Higher ADX, tighter sync_window → avoids choppy false signals
        "sjv3_sl_mult":      _env_float("SJV3_SL_MULT",     1.5),
        "sjv3_rr":           _env_float("SJV3_RR",          1.8),
        "sjv3_filter_thr":   _env_int("SJV3_FILTER_THR",    4),    # need 4/4 for long
        "sjv3_adx_min":      _env_int("SJV3_ADX_MIN",       25),   # strong trend only
        "sjv3_sync_window":  _env_int("SJV3_SYNC_WINDOW",   3),    # tighter EMA sync
        "sjv3_hma_len":      _env_int("SJV3_HMA_LEN",       14),   # faster HMA for 30m
        "sjv3_ut_mult":      _env_float("SJV3_UT_MULT",     0.25), # tighter TSL
        "sjv3_ut_len":       _env_int("SJV3_UT_LEN",        10),   # faster ATR
        "sjv3_allow_long":   _env_bool("SJV3_ALLOW_LONG",   True),
        "sjv3_allow_short":  _env_bool("SJV3_ALLOW_SHORT",  True),

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
        "max_positions":  _env_int("MAX_POSITIONS",  4),
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
    from trading.strategies.sjutbot_v3_strategy          import SJUTBotV3Strategy
    from trading.strategies.mean_reversion_strategy      import MeanReversionStrategy
    from trading.strategies.trend_continuation_strategy  import TrendContinuationStrategy
    from trading.strategies.smart_money_strategy         import SmartMoneyStrategy

    strategies = []
    for sym in symbols:

        # ── SJUTBot v3 (30m) ──────────────────────────────────────────────────
        if cfg["strategy_sjutbot_v3"]:
            strategies.append(SJUTBotV3Strategy(sym, params={
                "tf":               "30m",
                "limit":            500,
                "ut_mult":          cfg["sjv3_ut_mult"],
                "ut_len":           cfg["sjv3_ut_len"],
                "filter_threshold": cfg["sjv3_filter_thr"],
                "adx_min":          cfg["sjv3_adx_min"],
                "hma_len":          cfg["sjv3_hma_len"],
                "sync_window":      cfg["sjv3_sync_window"],
                "sl_mult":          cfg["sjv3_sl_mult"],
                "rr":               cfg["sjv3_rr"],
                "allow_long":       cfg["sjv3_allow_long"],
                "allow_short":      cfg["sjv3_allow_short"],
            }))

        # ── Mean Reversion (1H entry / 4H MTF) ────────────────────────────────
        if cfg["strategy_mean_reversion"]:
            strategies.append(MeanReversionStrategy(sym, params={
                "name":           "MeanReversion",
                "tf":             "1h",
                "limit":          300,
                "sl_mult":        cfg["mr_sl_mult"],
                "tp_mult":        cfg["mr_tp_mult"],
                "rsi_oversold":   cfg["mr_rsi_oversold"],
                "rsi_overbought": cfg["mr_rsi_overbought"],
                "min_conditions": cfg["mr_min_conditions"],
            }))

        # ── Trend Continuation (15m entry / 1H+4H MTF) ────────────────────────
        if cfg["strategy_trend_cont"]:
            strategies.append(TrendContinuationStrategy(sym, params={
                "name":         "TrendCont",
                "tf":           "15m",
                "limit":        300,
                "sl_mult":      cfg["tc_sl_mult"],
                "tp_mult":      cfg["tc_tp_mult"],
                "rsi_min":      cfg["tc_rsi_min"],
                "rsi_max":      cfg["tc_rsi_max"],
                "pullback_pct": cfg["tc_pullback_pct"],
                "vol_mult":     cfg["tc_vol_mult"],
                "bias_gate":    cfg["tc_bias_gate"],
            }))

        # ── Smart Money (15m entry / 1H+4H MTF) ───────────────────────────────
        if cfg["strategy_smart_money"]:
            strategies.append(SmartMoneyStrategy(sym, params={
                "name":           "SmartMoney",
                "tf":             "15m",
                "limit":          300,
                "sl_mult":        cfg["sm_sl_mult"],
                "tp_mult":        cfg["sm_tp_mult"],
                "bias_threshold": cfg["sm_bias_threshold"],
            }))

    if not strategies:
        raise RuntimeError(
            "No strategies enabled — set STRATEGY_MEAN_REVERSION, "
            "STRATEGY_TREND_CONT, STRATEGY_SMART_MONEY, or STRATEGY_SJUTBOT_V3 to true"
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
        from trading.backtester import backtest_strategy_mtf, summarise
        from trading.strategies.sjutbot_v3_strategy          import SJUTBotV3Strategy
        from trading.strategies.mean_reversion_strategy      import MeanReversionStrategy
        from trading.strategies.trend_continuation_strategy  import TrendContinuationStrategy
        from trading.strategies.smart_money_strategy         import SmartMoneyStrategy

        syms     = cfg["symbols"]
        sl_pct   = cfg["stop_loss_pct"]
        tp_pct   = cfg["take_profit_pct"]
        notional = cfg["fixed_trade_usdt"] * cfg["leverage"]

        cache: dict = {}
        results: dict[str, dict] = {}

        for sym in syms:
            for tf, lim in [("15m", 2000), ("1h", 700), ("4h", 180), ("30m", 1000)]:
                k = (sym, tf)
                logger.info("[BT] Fetching %s %s %d bars...", sym, tf, lim)
                try:
                    cache[k] = await connector.fetch_ohlcv(sym, timeframe=tf, limit=lim)
                except Exception as e:
                    logger.error("[BT] fetch %s %s failed: %s", sym, tf, e)
                    cache[k] = []

            c15m = cache.get((sym, "15m"), [])
            c1h  = cache.get((sym, "1h"),  [])
            c4h  = cache.get((sym, "4h"),  [])
            c30m = cache.get((sym, "30m"), [])
            tag  = sym.split('/')[0]

            if c1h and c4h:
                mr = MeanReversionStrategy(sym, params={
                    "name": "MeanReversion", "sl_mult": cfg["mr_sl_mult"],
                    "tp_mult": cfg["mr_tp_mult"], "rsi_oversold": cfg["mr_rsi_oversold"],
                    "rsi_overbought": cfg["mr_rsi_overbought"], "min_conditions": cfg["mr_min_conditions"],
                })
                trades = await backtest_strategy_mtf(mr, c1h, {"4h": c4h}, notional, sl_pct, tp_pct, primary_window=200)
                results[f"MeanRev/{tag}"] = summarise(trades)

            if c15m and c1h and c4h:
                tc = TrendContinuationStrategy(sym, params={
                    "name": "TrendCont", "sl_mult": cfg["tc_sl_mult"], "tp_mult": cfg["tc_tp_mult"],
                    "rsi_min": cfg["tc_rsi_min"], "rsi_max": cfg["tc_rsi_max"],
                    "pullback_pct": cfg["tc_pullback_pct"], "vol_mult": cfg["tc_vol_mult"],
                    "bias_gate": cfg["tc_bias_gate"],
                })
                trades = await backtest_strategy_mtf(tc, c15m, {"1h": c1h, "4h": c4h}, notional, sl_pct, tp_pct)
                results[f"TrendCont/{tag}"] = summarise(trades)

            if c15m and c1h and c4h:
                sm = SmartMoneyStrategy(sym, params={
                    "name": "SmartMoney", "sl_mult": cfg["sm_sl_mult"], "tp_mult": cfg["sm_tp_mult"],
                    "bias_threshold": cfg["sm_bias_threshold"],
                })
                trades = await backtest_strategy_mtf(sm, c15m, {"1h": c1h, "4h": c4h}, notional, sl_pct, tp_pct)
                results[f"SmartMoney/{tag}"] = summarise(trades)

            if c30m:
                sv3 = SJUTBotV3Strategy(sym, params={
                    "tf": "30m", "limit": 500, "ut_mult": cfg["sjv3_ut_mult"], "ut_len": cfg["sjv3_ut_len"],
                    "filter_threshold": cfg["sjv3_filter_thr"], "adx_min": cfg["sjv3_adx_min"],
                    "hma_len": cfg["sjv3_hma_len"], "sync_window": cfg["sjv3_sync_window"],
                    "sl_mult": cfg["sjv3_sl_mult"], "rr": cfg["sjv3_rr"],
                    "allow_long": cfg["sjv3_allow_long"], "allow_short": cfg["sjv3_allow_short"],
                })
                trades = await backtest_strategy_mtf(sv3, c30m, {}, notional, sl_pct, tp_pct)
                results[f"SJUTBotV3/{tag}"] = summarise(trades)

        lines = [f"📊 *Backtest — 4 Strategies* (${cfg['fixed_trade_usdt']}×{cfg['leverage']}x=${notional:.0f})"]
        total_net = 0.0
        for label, s in results.items():
            if s.get("trades", 0) == 0:
                lines.append(f"  ⚪ `{label}` — no signals")
                continue
            net = s.get("net_usdt", 0); wr = s.get("win_rate", 0)
            pf = s.get("profit_factor", 0); t = s["trades"]
            icon = "✅" if net >= 0 else "❌"
            sign = "+" if net >= 0 else ""
            lines.append(f"  {icon} `{label}` T={t} WR={wr:.0f}% PF={pf:.2f} Net=`{sign}{net:.2f}$`")
            total_net += net
        icon = "✅" if total_net >= 0 else "❌"
        sign = "+" if total_net >= 0 else ""
        lines.append(f"\n{icon} *Total: {sign}{total_net:.2f}$*")
        return "\n".join(lines)

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
