"""
OKX Perpetual Futures trading bot runner.

Active strategies (5):
  SJUTBotV4      — 30m entry / 1H+4H MTF — 8/9 confirmation, 70% TP1, regime TP2
  MeanReversion  — 1H entry  / 4H MTF    — fade extremes, TP1+TP2 partial
  TrendCont      — 15m entry / 1H+4H MTF — trend dip buy, TP1+TP2 partial
  SmartMoney     — 15m entry / 1H+4H MTF — MTF momentum + OBV flow
  SJUTBotV3      — 30m entry             — UTBot + multi-filter confirmation

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

        # ── Strategy variant: "v2" = new partial-close (TP1/TP2 + ADX/ST/OBV),
        #    "legacy" = original single-TP system (backtest +$149, most robust).
        #    Flip STRATEGY_VARIANT=legacy to revert without redeploying code.
        "strategy_variant": os.environ.get("STRATEGY_VARIANT", "v2").strip().lower(),

        # ── Active strategies ─────────────────────────────────────────────────
        "strategy_mean_reversion": _env_bool("STRATEGY_MEAN_REVERSION", True),
        "strategy_trend_cont":     _env_bool("STRATEGY_TREND_CONT",     True),
        "strategy_smart_money":    _env_bool("STRATEGY_SMART_MONEY",    True),
        "strategy_sjutbot_v3":     _env_bool("STRATEGY_SJUTBOT_V3",     True),

        # ── Global Risk multiples (shared by MR/TC/SM) ───────────────────────
        # Grid-tuned: Initial SL 1.2×ATR, TP1 1.5×ATR (close 50% → SL→breakeven), TP2 3.0×ATR.
        "g_sl_atr":   _env_float("GLOBAL_SL_ATR",   1.2),
        "g_tp1_atr":  _env_float("GLOBAL_TP1_ATR",  1.5),
        "g_tp2_atr":  _env_float("GLOBAL_TP2_ATR",  3.0),
        "g_partial":  _env_float("GLOBAL_PARTIAL_PCT", 0.5),

        # ── Mean Reversion (1H entry / 4H MTF) — fade extremes, avoid strong trends
        "mr_rsi_oversold":    _env_float("MR_RSI_OVERSOLD",  45.0),
        "mr_rsi_overbought":  _env_float("MR_RSI_OVERBOUGHT",55.0),
        "mr_min_conditions":  _env_int("MR_MIN_CONDITIONS",     3),   # ↑ from 2
        "mr_adx_cap":         _env_float("MR_ADX_CAP",       50.0),   # no fade if 4H ADX > 50

        # ── Trend Continuation (15m entry / 1H+4H MTF) — ride trend, buy dips ─
        "tc_rsi_min":         _env_float("TC_RSI_MIN",       35.0),
        "tc_rsi_max":         _env_float("TC_RSI_MAX",       75.0),
        "tc_pullback_atr":    _env_float("TC_PULLBACK_ATR",   0.5),   # EMA20 ± 0.5×ATR(1H)
        "tc_vol_mult":        _env_float("TC_VOL_MULT",       1.0),
        "tc_bias_gate":       _env_float("TC_BIAS_GATE",     55.0),   # ↓ from 70
        "tc_st_period":       _env_int("TC_ST_PERIOD",         10),   # Supertrend(4H)
        "tc_st_mult":         _env_float("TC_ST_MULT",        3.0),

        # ── Smart Money (15m entry / 1H+4H MTF) — MTF momentum + OBV flow ─────
        "sm_bias_threshold":  _env_float("SM_BIAS_THRESHOLD",40.0),   # ↑ from 15
        "sm_obv_ema":         _env_int("SM_OBV_EMA",           20),

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

        # ── SJUTBot v4 (30m primary + 1h + 4h MTF) ──────────────────────────
        # High-WR (68%) selective system: 8/9 confirmations required.
        # TP1 = 1.2R (close 70%), SL→BE, TP2 = regime-aware (1.5R-3.0R).
        # Backtest Jan-May 2026: 22 trades, WR 68.2%, Net +$23.37, MaxDD -2.28%.
        "strategy_sjutbot_v4":  _env_bool("STRATEGY_SJUTBOT_V4", True),
        "sjv4_ut_mult":         _env_float("SJV4_UT_MULT",      0.30),
        "sjv4_ut_len":          _env_int("SJV4_UT_LEN",         14),
        "sjv4_adx_min":         _env_int("SJV4_ADX_MIN",        35),
        "sjv4_min_score":       _env_int("SJV4_MIN_SCORE",       8),
        "sjv4_sl_mult":         _env_float("SJV4_SL_MULT",      1.0),
        "sjv4_tp1_r":           _env_float("SJV4_TP1_R",        1.2),
        "sjv4_tp1_fraction":    _env_float("SJV4_TP1_FRACTION", 0.70),
        "sjv4_tp2_strong":      _env_float("SJV4_TP2_STRONG",   3.0),
        "sjv4_tp2_weak":        _env_float("SJV4_TP2_WEAK",     2.0),
        "sjv4_tp2_break":       _env_float("SJV4_TP2_BREAK",    2.5),
        "sjv4_tp2_range":       _env_float("SJV4_TP2_RANGE",    1.5),
        "sjv4_tp2_chop":        _env_float("SJV4_TP2_CHOP",     1.5),

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
        # Daily circuit breaker: pause NEW entries once realized PnL ≤ -X% for the UTC day.
        "daily_loss_limit": _env_float("DAILY_LOSS_LIMIT_PCT", 0.05),  # -5%
        # Per-trade risk budget for dynamic sizing (strategies pass risk_pct in metadata).
        "risk_per_trade":   _env_float("RISK_PER_TRADE_PCT", 0.02),    # 2%
        # DYNAMIC_SIZING=false → use fixed FIXED_TRADE_USDT margin per trade even on v2
        # (required for small accounts below the 2%-risk → 1-contract minimum).
        "dynamic_sizing":   _env_bool("DYNAMIC_SIZING", True),

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
    from trading.strategies.sjutbot_v3_strategy import SJUTBotV3Strategy
    from trading.strategies.sjutbot_v4_strategy import SJUTBotV4Strategy

    legacy = cfg["strategy_variant"] == "legacy"
    if legacy:
        logger.info("STRATEGY_VARIANT=legacy → original single-TP strategies (backtest +$149)")
        from trading.strategies.mean_reversion_legacy     import MeanReversionStrategy
        from trading.strategies.trend_continuation_legacy import TrendContinuationStrategy
        from trading.strategies.smart_money_legacy        import SmartMoneyStrategy
    else:
        from trading.strategies.mean_reversion_strategy      import MeanReversionStrategy
        from trading.strategies.trend_continuation_strategy  import TrendContinuationStrategy
        from trading.strategies.smart_money_strategy         import SmartMoneyStrategy

    strategies = []
    for sym in symbols:

        # ── SJUTBot v4 (30m + 1h + 4h MTF) ──────────────────────────────────
        if cfg["strategy_sjutbot_v4"]:
            strategies.append(SJUTBotV4Strategy(sym, params={
                "name":         "SJUTBotV4",
                "tf":           "30m",
                "limit":        500,
                "ut_mult":      cfg["sjv4_ut_mult"],
                "ut_len":       cfg["sjv4_ut_len"],
                "adx_min":      cfg["sjv4_adx_min"],
                "min_score":    cfg["sjv4_min_score"],
                "sl_mult":      cfg["sjv4_sl_mult"],
                "tp1_r":        cfg["sjv4_tp1_r"],
                "tp1_fraction": cfg["sjv4_tp1_fraction"],
                "tp2_strong":   cfg["sjv4_tp2_strong"],
                "tp2_weak":     cfg["sjv4_tp2_weak"],
                "tp2_break":    cfg["sjv4_tp2_break"],
                "tp2_range":    cfg["sjv4_tp2_range"],
                "tp2_chop":     cfg["sjv4_tp2_chop"],
            }))

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

        _grisk = {
            "sl_atr":      cfg["g_sl_atr"],
            "tp1_atr":     cfg["g_tp1_atr"],
            "tp2_atr":     cfg["g_tp2_atr"],
            "partial_pct": cfg["g_partial"],
            "risk_pct":    cfg["risk_per_trade"],
        }

        # ── Mean Reversion (1H entry / 4H MTF) ────────────────────────────────
        if cfg["strategy_mean_reversion"]:
            strategies.append(MeanReversionStrategy(sym, params={
                "name":           "MeanReversion",
                "tf":             "1h",
                "limit":          300,
                "rsi_oversold":   cfg["mr_rsi_oversold"],
                "rsi_overbought": cfg["mr_rsi_overbought"],
                "min_conditions": cfg["mr_min_conditions"],
                "adx_cap":        cfg["mr_adx_cap"],
                **_grisk,
            }))

        # ── Trend Continuation (15m entry / 1H+4H MTF) ────────────────────────
        if cfg["strategy_trend_cont"]:
            strategies.append(TrendContinuationStrategy(sym, params={
                "name":         "TrendCont",
                "tf":           "15m",
                "limit":        300,
                "rsi_min":      cfg["tc_rsi_min"],
                "rsi_max":      cfg["tc_rsi_max"],
                "pullback_atr": cfg["tc_pullback_atr"],
                "vol_mult":     cfg["tc_vol_mult"],
                "bias_gate":    cfg["tc_bias_gate"],
                "st_period":    cfg["tc_st_period"],
                "st_mult":      cfg["tc_st_mult"],
                **_grisk,
            }))

        # ── Smart Money (15m entry / 1H+4H MTF) ───────────────────────────────
        if cfg["strategy_smart_money"]:
            strategies.append(SmartMoneyStrategy(sym, params={
                "name":           "SmartMoney",
                "tf":             "15m",
                "limit":          300,
                "bias_threshold": cfg["sm_bias_threshold"],
                "obv_ema":        cfg["sm_obv_ema"],
                **_grisk,
            }))

    if not strategies:
        raise RuntimeError(
            "No strategies enabled — set STRATEGY_SJUTBOT_V4, STRATEGY_MEAN_REVERSION, "
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
        daily_loss_limit_pct=cfg["daily_loss_limit"],
    )
    logger.info(
        "Risk: SL=%.1f%%  TP=%.1f%%  fixed=%gUSDT  lev=%dx  max_pos=%d  drawdown=%.0f%%  dailyCB=%.0f%%",
        cfg["stop_loss_pct"] * 100, cfg["take_profit_pct"] * 100,
        cfg["fixed_trade_usdt"], cfg["leverage"],
        cfg["max_positions"], cfg["max_drawdown"] * 100, cfg["daily_loss_limit"] * 100,
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
        dynamic_sizing=cfg["dynamic_sizing"],
    )

    # ── Backtest function (called by /backtest Telegram command) ─────────────
    async def _run_backtest() -> str:
        from trading.backtester import backtest_strategy_mtf, backtest_strategy_mtf_v2, summarise
        from trading.strategies.sjutbot_v3_strategy          import SJUTBotV3Strategy
        from trading.strategies.mean_reversion_strategy      import MeanReversionStrategy
        from trading.strategies.trend_continuation_strategy  import TrendContinuationStrategy
        from trading.strategies.smart_money_strategy         import SmartMoneyStrategy

        syms     = cfg["symbols"]
        sl_pct   = cfg["stop_loss_pct"]
        tp_pct   = cfg["take_profit_pct"]
        notional = cfg["fixed_trade_usdt"] * cfg["leverage"]
        _grisk = {
            "sl_atr":  cfg["g_sl_atr"],  "tp1_atr": cfg["g_tp1_atr"],
            "tp2_atr": cfg["g_tp2_atr"], "partial_pct": cfg["g_partial"],
            "risk_pct": cfg["risk_per_trade"],
        }

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
                    "name": "MeanReversion", "rsi_oversold": cfg["mr_rsi_oversold"],
                    "rsi_overbought": cfg["mr_rsi_overbought"], "min_conditions": cfg["mr_min_conditions"],
                    "adx_cap": cfg["mr_adx_cap"], **_grisk,
                })
                trades, _ = await backtest_strategy_mtf_v2(mr, c1h, {"4h": c4h}, notional, primary_window=200)
                results[f"MeanRev/{tag}"] = summarise(trades)

            if c15m and c1h and c4h:
                tc = TrendContinuationStrategy(sym, params={
                    "name": "TrendCont", "rsi_min": cfg["tc_rsi_min"], "rsi_max": cfg["tc_rsi_max"],
                    "pullback_atr": cfg["tc_pullback_atr"], "vol_mult": cfg["tc_vol_mult"],
                    "bias_gate": cfg["tc_bias_gate"], "st_period": cfg["tc_st_period"],
                    "st_mult": cfg["tc_st_mult"], **_grisk,
                })
                trades, _ = await backtest_strategy_mtf_v2(tc, c15m, {"1h": c1h, "4h": c4h}, notional)
                results[f"TrendCont/{tag}"] = summarise(trades)

            if c15m and c1h and c4h:
                sm = SmartMoneyStrategy(sym, params={
                    "name": "SmartMoney", "bias_threshold": cfg["sm_bias_threshold"],
                    "obv_ema": cfg["sm_obv_ema"], **_grisk,
                })
                trades, _ = await backtest_strategy_mtf_v2(sm, c15m, {"1h": c1h, "4h": c4h}, notional)
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
