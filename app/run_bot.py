"""
OKX Perpetual Futures trading bot runner.

Strategies  : SJUTBot v2 (1H) + SJUTBot v3.1 (1H) + UTBot (15m)  [MCDX off by default]
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
        "strategy_mcdx":         _env_bool("STRATEGY_MCDX",         True),   # MCDX p-1
        "strategy_mcdx_dual":    _env_bool("STRATEGY_MCDX_DUAL",    True),   # MCDX p-2
        "strategy_sjutbot":      _env_bool("STRATEGY_SJUTBOT",      False),  # v2
        "strategy_sjutbot_v3":   _env_bool("STRATEGY_SJUTBOT_V3",   True),   # v3.1
        "strategy_sjutbot_v2rev":_env_bool("STRATEGY_SJUTBOT_V2REV",False),  # v2 reversed
        "strategy_utbot":        _env_bool("STRATEGY_UTBOT",        False),  # off
        "strategy_scalp_trend":  _env_bool("STRATEGY_SCALP_TREND",  False),  # scalp trend
        "strategy_prof_bot":     _env_bool("STRATEGY_PROF_BOT",     False),  # profitable bot

        # ── MCDX p-1 tuning — selective (30m, WR 60%+ target) ───────────────────
        "mcdx_dwcs_buy":     _env_int("MCDX_DWCS_BUY",     62),   # high threshold → fewer, better signals
        "mcdx_rvol":         _env_float("MCDX_RVOL",       1.2),  # require above-avg volume
        "mcdx_adx_thr":      _env_int("MCDX_ADX_THR",      25),   # ADX gate: only in trending market
        "mcdx_rsi_min":      _env_int("MCDX_RSI_MIN",      40),   # RSI lower bound
        "mcdx_rsi_max":      _env_int("MCDX_RSI_MAX",      70),   # RSI upper bound

        # ── MCDX p-2 tuning — moderate (wider threshold, still 60%+ target) ────
        "mcdx2_dwcs_buy":    _env_int("MCDX2_DWCS_BUY",    55),
        "mcdx2_rvol":        _env_float("MCDX2_RVOL",      0.9),
        "mcdx2_adx_thr":     _env_int("MCDX2_ADX_THR",     22),
        "mcdx2_rsi_min":     _env_int("MCDX2_RSI_MIN",     38),
        "mcdx2_rsi_max":     _env_int("MCDX2_RSI_MAX",     72),

        # ── Scalp Trend tuning (15m entry / 1H+4H MTF) ───────────────────────
        # sl_mult=0.8, tp_mult=1.0 → TP1-only (high hit-rate scalp style)
        "scalp_sl_mult":     _env_float("SCALP_SL_MULT",      0.8),
        "scalp_tp_mult":     _env_float("SCALP_TP_MULT",      1.0),
        "scalp_rsi_min":     _env_float("SCALP_RSI_MIN",     42.0),
        "scalp_rsi_max":     _env_float("SCALP_RSI_MAX",     65.0),
        "scalp_pullback_pct":_env_float("SCALP_PULLBACK_PCT",  0.015),

        # ── Profitable Bot tuning (1H entry / 4H MTF) ────────────────────────
        # sl_mult=1.5, tp_mult=2.5 → R:R ≈ 1:1.67 (ATR-based, no fixed %)
        "prof_sl_mult":       _env_float("PROF_SL_MULT",      1.5),
        "prof_tp_mult":       _env_float("PROF_TP_MULT",      2.5),
        "prof_rsi_oversold":  _env_float("PROF_RSI_OVERSOLD", 35.0),
        "prof_rsi_overbought":_env_float("PROF_RSI_OVERBOUGHT",65.0),
        "prof_min_cond":      _env_int("PROF_MIN_COND",         3),

        # ── SJUTBot v2 tuning ─────────────────────────────────────────────────
        "sjutbot_sl_mult": _env_float("SJUTBOT_SL_MULT", 1.2),
        "sjutbot_rr":      _env_float("SJUTBOT_RR",      1.35),

        # ── SJUTBot v3.1 tuning — 30m, WR 60%+ target ────────────────────────
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
        "max_positions":  _env_int("MAX_POSITIONS",  3),
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
    from trading.strategies.mcdx_strategy                import MCDXStrategy
    from trading.strategies.sjutbot_strategy             import SJUTBotStrategy
    from trading.strategies.sjutbot_v2_reversed_strategy import SJUTBotV2ReversedStrategy
    from trading.strategies.sjutbot_v3_strategy          import SJUTBotV3Strategy
    from trading.strategies.utbot_wt_strategy            import UTBotWTStrategy
    from trading.strategies.scalp_trend_strategy         import ScalpTrendStrategy
    from trading.strategies.profitable_bot_strategy      import ProfitableBotStrategy

    strategies = []
    for sym in symbols:

        # ── MCDX p-1 (30m, selective — WR 60%+ target) ───────────────────────
        if cfg["strategy_mcdx"]:
            strategies.append(MCDXStrategy(sym, params={
                "name":          "MCDXStrategy_p1",
                "tf":            "30m",
                "limit":         500,
                "dwcs_buy":      cfg["mcdx_dwcs_buy"],
                "dwcs_sell":     100 - cfg["mcdx_dwcs_buy"],
                "rvol_min":      cfg["mcdx_rvol"],
                "adx_threshold": cfg["mcdx_adx_thr"],
                "rsi_min_gate":  cfg["mcdx_rsi_min"],
                "rsi_max_gate":  cfg["mcdx_rsi_max"],
                "dwcs_stability":True,
            }))

        # ── MCDX p-2 (30m, moderate) ──────────────────────────────────────────
        if cfg["strategy_mcdx_dual"]:
            strategies.append(MCDXStrategy(sym, params={
                "name":          "MCDXStrategy_p2",
                "tf":            "30m",
                "limit":         500,
                "dwcs_buy":      cfg["mcdx2_dwcs_buy"],
                "dwcs_sell":     100 - cfg["mcdx2_dwcs_buy"],
                "rvol_min":      cfg["mcdx2_rvol"],
                "adx_threshold": cfg["mcdx2_adx_thr"],
                "rsi_min_gate":  cfg["mcdx2_rsi_min"],
                "rsi_max_gate":  cfg["mcdx2_rsi_max"],
                "dwcs_stability":True,
            }))

        # ── SJ-UTBot v2 (30m) ─────────────────────────────────────────────────
        if cfg["strategy_sjutbot"]:
            strategies.append(SJUTBotStrategy(sym, params={
                "tf":      "30m",
                "limit":   400,
                "ut_mult": 0.30,
                "ut_len":  14,
                "sl_len":  14,
                "sl_mult": cfg["sjutbot_sl_mult"],
                "rr":      cfg["sjutbot_rr"],
            }))

        # ── SJ-UTBot v3.1 (30m, WR 60%+ target) ──────────────────────────────
        # filter_threshold=4: ALL 4 components (HMA, ADX, SwingHigh, Resistance)
        # must agree before going long → naturally achieves 60%+ WR.
        # Short side unchanged (UT↓ only, fast hedge).
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

        # ── SJ-UTBot v2 Reversed (30m) — contrarian ───────────────────────────
        if cfg["strategy_sjutbot_v2rev"]:
            strategies.append(SJUTBotV2ReversedStrategy(sym, params={
                "tf":      "30m",
                "limit":   400,
                "ut_mult": 0.30,
                "ut_len":  14,
                "sl_len":  14,
                "sl_mult": cfg["sjutbot_sl_mult"],
                "rr":      cfg["sjutbot_rr"],
            }))

        # ── UT Bot + WaveTrend (15m) ───────────────────────────────────────────
        if cfg["strategy_utbot"]:
            strategies.append(UTBotWTStrategy(sym, params={
                "tf":    "15m",
                "limit": 300,
            }))

        # ── Scalp Trend (15m entry / 1H+4H MTF) ───────────────────────────────
        if cfg["strategy_scalp_trend"]:
            strategies.append(ScalpTrendStrategy(sym, params={
                "name":        "ScalpTrend",
                "tf":          "15m",
                "limit":       200,
                "sl_mult":     cfg["scalp_sl_mult"],
                "tp_mult":     cfg["scalp_tp_mult"],
                "rsi_min":     cfg["scalp_rsi_min"],
                "rsi_max":     cfg["scalp_rsi_max"],
                "pullback_pct":cfg["scalp_pullback_pct"],
            }))

        # ── Profitable Bot (1H entry / 4H MTF) ────────────────────────────────
        if cfg["strategy_prof_bot"]:
            strategies.append(ProfitableBotStrategy(sym, params={
                "name":           "ProfBot",
                "tf":             "1h",
                "limit":          200,
                "sl_mult":        cfg["prof_sl_mult"],
                "tp_mult":        cfg["prof_tp_mult"],
                "rsi_oversold":   cfg["prof_rsi_oversold"],
                "rsi_overbought": cfg["prof_rsi_overbought"],
                "min_conditions": cfg["prof_min_cond"],
            }))

    if not strategies:
        raise RuntimeError(
            "No strategies enabled — set one of: STRATEGY_MCDX, STRATEGY_MCDX_DUAL, "
            "STRATEGY_SJUTBOT, STRATEGY_SJUTBOT_V3, STRATEGY_SJUTBOT_V2REV, "
            "STRATEGY_UTBOT, STRATEGY_SCALP_TREND, STRATEGY_PROF_BOT to true"
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
        from trading.backtester import (
            run_full_backtest, backtest_with_signals,
            signal_overlap, format_comparison_telegram,
        )
        from trading.strategies.mcdx_strategy                import MCDXStrategy
        from trading.strategies.sjutbot_strategy             import SJUTBotStrategy
        from trading.strategies.sjutbot_v2_reversed_strategy import SJUTBotV2ReversedStrategy
        from trading.strategies.sjutbot_v3_strategy          import SJUTBotV3Strategy
        from trading.strategies.utbot_wt_strategy            import UTBotWTStrategy

        syms     = cfg["symbols"]
        sl_pct   = cfg["stop_loss_pct"]
        tp_pct   = cfg["take_profit_pct"]
        notional = cfg["fixed_trade_usdt"] * cfg["leverage"]

        v2_params   = {"ut_mult": 0.30, "ut_len": 14, "sl_len": 14,
                       "sl_mult": cfg["sjutbot_sl_mult"], "rr": cfg["sjutbot_rr"]}
        v3_params   = {"ut_mult": 0.30, "ut_len": 14, "filter_threshold": 3,
                       "sl_mult": cfg["sjv3_sl_mult"], "rr": cfg["sjv3_rr"],
                       "allow_long": True, "allow_short": True}
        mcdx_p1 = {"name": "MCDXStrategy_p1",
                   "dwcs_buy": cfg["mcdx_dwcs_buy"], "dwcs_sell": 100-cfg["mcdx_dwcs_buy"],
                   "rvol_min": cfg["mcdx_rvol"], "adx_threshold": cfg["mcdx_adx_thr"],
                   "rsi_min_gate": cfg["mcdx_rsi_min"], "rsi_max_gate": cfg["mcdx_rsi_max"],
                   "dwcs_stability": True}
        mcdx_p2 = {"name": "MCDXStrategy_p2",
                   "dwcs_buy": cfg["mcdx2_dwcs_buy"], "dwcs_sell": 100-cfg["mcdx2_dwcs_buy"],
                   "rvol_min": cfg["mcdx2_rvol"], "adx_threshold": cfg["mcdx2_adx_thr"],
                   "rsi_min_gate": cfg["mcdx2_rsi_min"], "rsi_max_gate": cfg["mcdx2_rsi_max"],
                   "dwcs_stability": True}
        v3_30m  = {**v3_params,
                   "ut_mult": cfg["sjv3_ut_mult"], "ut_len": cfg["sjv3_ut_len"],
                   "filter_threshold": cfg["sjv3_filter_thr"],
                   "adx_min": cfg["sjv3_adx_min"], "hma_len": cfg["sjv3_hma_len"],
                   "sync_window": cfg["sjv3_sync_window"]}

        # ── Case 1: SJUTBot v2 + v3.1 (30m) ──────────────────────────────
        c1_configs = []
        for sym in syms:
            c1_configs += [
                {"cls": SJUTBotStrategy,   "symbol": sym, "tf": "30m", "limit": 4000, "params": v2_params},
                {"cls": SJUTBotV3Strategy, "symbol": sym, "tf": "30m", "limit": 4000, "params": v3_30m},
            ]

        # ── Case 2: MCDX dual + SJUTBot v3.1 (30m, WR 60%+ target) ──────
        c2_configs = []
        for sym in syms:
            c2_configs += [
                {"cls": MCDXStrategy,      "symbol": sym, "tf": "30m", "limit": 4000, "params": mcdx_p1},
                {"cls": MCDXStrategy,      "symbol": sym, "tf": "30m", "limit": 4000, "params": mcdx_p2},
                {"cls": SJUTBotV3Strategy, "symbol": sym, "tf": "30m", "limit": 4000, "params": v3_30m},
            ]

        # ── Case 3: Full (MCDX dual + UTBot + v2 + v3.1, 30m) ────────────
        c3_configs = list(c2_configs)
        for sym in syms:
            c3_configs += [
                {"cls": SJUTBotStrategy, "symbol": sym, "tf": "30m", "limit": 4000, "params": v2_params},
                {"cls": UTBotWTStrategy, "symbol": sym, "tf": "30m", "limit": 4000, "params": {}},
            ]

        # ── Case 4: v2 Reversed solo (30m) ────────────────────────────────
        c4_configs = []
        for sym in syms:
            c4_configs += [
                {"cls": SJUTBotV2ReversedStrategy, "symbol": sym, "tf": "30m",
                 "limit": 4000, "params": {**v2_params, "name": "SJUTBotV2ReversedStrategy"}},
            ]

        logger.info("[BT] Fetching candles for 4-case comparison backtest...")
        cache: dict = {}
        case1 = await run_full_backtest(connector, c1_configs,
                                        cfg["fixed_trade_usdt"], cfg["leverage"],
                                        sl_pct, tp_pct, _cache=cache)
        case2 = await run_full_backtest(connector, c2_configs,
                                        cfg["fixed_trade_usdt"], cfg["leverage"],
                                        sl_pct, tp_pct, _cache=cache)
        case3 = await run_full_backtest(connector, c3_configs,
                                        cfg["fixed_trade_usdt"], cfg["leverage"],
                                        sl_pct, tp_pct, _cache=cache)
        case4 = await run_full_backtest(connector, c4_configs,
                                        cfg["fixed_trade_usdt"], cfg["leverage"],
                                        sl_pct, tp_pct, _cache=cache)

        # ── Signal correlation: v2 vs v2-reversed (should be perfect inverse)
        correlation: dict = {}
        for sym in syms:
            candles = cache.get((sym, "30m"), [])
            if not candles:
                continue
            sv2  = SJUTBotStrategy(sym, params=v2_params)
            srev = SJUTBotV2ReversedStrategy(sym, params={**v2_params, "name": "SJUTBotV2ReversedStrategy"})
            _, sigs_v2  = await backtest_with_signals(sv2,  candles, notional, sl_pct, tp_pct)
            _, sigs_rev = await backtest_with_signals(srev, candles, notional, sl_pct, tp_pct)
            correlation[sym] = signal_overlap(sigs_v2, sigs_rev)

        return format_comparison_telegram(
            {
                "C1 v2 + v3.1":           case1,
                "C2 MCDXdual + v3.1":     case2,
                "C3 Full":                case3,
                "C4 v2-REVERSED (solo)":  case4,
            },
            correlation,
            cfg["fixed_trade_usdt"], cfg["leverage"],
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
