"""
OKX Perpetual Futures trading bot runner.

Active strategy (1):
  TrendContImproved v2 — 15m entry / 1H+4H MTF
  FAST v2 (default):   1H EMA20 ±2.5% + bias>60 + ADX>15 rising + 5-bar cooldown
                       TP1=0.5R, TP2=2.5R, SJ hybrid scoring enabled
  STRICT:              15m swing pullback (4-bar low/high ±0.6%) + ADX(15m)>30

Symbols      : BTC/USDT:USDT, XAU/USDT:USDT  (configurable via SYMBOLS)
Exchange     : OKX, swap market, isolated margin, hedge mode
Max positions: 2 (one per symbol at a time)

Position Health Monitor:
  Re-evaluates every open position every MONITOR_INTERVAL seconds (default 180 = 3 min).
  Uses 5m + 1h + 4h candles with RELAXED indicator thresholds:
    BULL    (score ≥ 85%): TP2 ladder extended (1.2→1.5→2.0→2.5→3.0R)
    NEUTRAL (score 45–84%): hold — radar green, stay in trade
    WEAK    (score < 45%): 2-cycle confirm then close — radar red, exit and wait for next setup
  Crash-guard: closes immediately if ≥0.7R underwater + 5m momentum strongly reversed.

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

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_bot")


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Config ───────────────────────────────────────────────────────────────────

def build_config() -> dict:
    # min_score is on the 100-point Confidence Scale (≥90 strong, 70-89 normal, 60-69 small×0.5).
    # Default 60 = gate; override via TCI_MIN_SCORE env var. An explicit env var always wins.
    _sj_scoring  = _env_bool("TCI_SJ_SCORING", True)
    _sj_roc9     = _env_bool("TCI_SJ_ROC9",    True)
    _min_default = 60.0
    return {
        # ── Exchange credentials ───────────────────────────────────────────────
        "exchange":       os.environ.get("EXCHANGE",            "okx"),
        "api_key":        os.environ.get("EXCHANGE_API_KEY",    ""),
        "api_secret":     os.environ.get("EXCHANGE_API_SECRET", ""),
        "api_passphrase": os.environ.get("EXCHANGE_PASSPHRASE", ""),  # OKX required

        # ── Market mode ───────────────────────────────────────────────────────
        "paper":       _env_bool("PAPER_TRADING", False),
        "market_type": os.environ.get("MARKET_TYPE", "swap"),  # "swap" = perpetual futures
        "leverage":    _env_int("LEVERAGE", 20),
        # Per-symbol leverage override. Default is EMPTY → both BTC and XAU use the
        # global LEVERAGE (20x). To cap a symbol, set e.g. "XAU/USDT:USDT=10,BTC/USDT:USDT=20".
        # (Note: XAU max gap ~4.6% means 20x carries higher liquidation risk than BTC.)
        "symbol_leverage": {
            k.strip(): int(v.strip())
            for pair in os.environ.get("SYMBOL_LEVERAGE", "").split(",")
            if "=" in pair
            for k, v in [pair.split("=", 1)]
        },

        # ── Symbols ───────────────────────────────────────────────────────────
        # Both BTC and XAU active by default.
        # OKX minimums: BTC 0.01 contract (~$800 notional), XAU 1 oz contract (~$4,300).
        "symbols": _env_list("SYMBOLS", "BTC/USDT:USDT,XAU/USDT:USDT"),

        # ── TrendCont Improved v2 (15m primary + 1h + 4h MTF) ─────────────────
        # STRICT: 15m swing pullback (4-bar low/high ±0.6%) + ADX>30.
        # FAST v2: 1H EMA20 ±2.5% + bias>60 + ADX>15 rising + 5-bar cooldown.
        # Exit (both modes): TP1=0.5R, TP2=2.5R.
        "tci_bias_gate":        _env_float("TCI_BIAS_GATE",         70.0),  # STRICT only
        "tci_adx_min":          _env_int("TCI_ADX_MIN",             30),    # STRICT only
        "tci_sl_mult":          _env_float("TCI_SL_MULT",           1.2),
        "tci_sl_min_pct":       _env_float("TCI_SL_MIN_PCT",        0.012),
        "tci_sl_max_pct":       _env_float("TCI_SL_MAX_PCT",        0.035),
        "tci_tp1_r":            _env_float("TCI_TP1_R",             0.5),
        "tci_tp1_fraction":     _env_float("TCI_TP1_FRACTION",      0.60),
        "tci_tp2_r":            _env_float("TCI_TP2_R",             2.5),
        "tci_min_score":        _env_float("TCI_MIN_SCORE",         _min_default),
        "tci_vol_mult":         _env_float("TCI_VOL_MULT",          1.0),
        # Strict: 15m swing pullback parameters
        "tci_swing_lookback":   _env_int("TCI_SWING_LOOKBACK",      4),
        "tci_swing_pct":        _env_float("TCI_SWING_PCT",         0.006),
        # Fast mode v2 — default ON; bias lowered 60→55 to reduce entry lag
        "tci_fast_mode":        _env_bool("TCI_FAST_MODE",          True),
        "tci_fast_bias_gate":   _env_float("TCI_FAST_BIAS_GATE",    55.0),  # lowered 60→55 to enter earlier
        "tci_fast_adx_min":     _env_int("TCI_ADX_MIN_FAST",        15),
        "tci_fast_adx_max":     _env_int("TCI_ADX_MAX_FAST",        44),
        "tci_fast_tp2_r":       _env_float("TCI_FAST_TP2_R",        2.5),
        "tci_fast_pullback_pct":_env_float("TCI_PULLBACK_PCT_FAST", 0.025),
        "tci_adx_rising":       _env_bool("TCI_ADX_RISING",         True),
        "tci_cooldown_bars":    _env_int("TCI_COOLDOWN_BARS",       5),
        # Crash-guard (both modes)
        "tci_health_guard":     _env_bool("TCI_HEALTH_GUARD",       True),
        "tci_health_uw_frac":   _env_float("TCI_HEALTH_UW_FRAC",    0.7),
        "reversal_spike_enabled": _env_bool("REVERSAL_SPIKE_ENABLED", True),
        "reversal_spike_atr":   _env_float("REVERSAL_SPIKE_ATR",    1.5),
        "reversal_spike_bars":  _env_int("REVERSAL_SPIKE_BARS",     4),
        # Trend-Fade cut: ≥0.6R underwater + ADX(15m) falling + lost EMA20(5m) → close
        "trend_fade_enabled":   _env_bool("TREND_FADE_ENABLED",     True),
        "trend_fade_uw_frac":   _env_float("TREND_FADE_UW_FRAC",    0.6),
        "tci_health_bias_flip": _env_float("TCI_HEALTH_BIAS_FLIP",  50.0),
        # SJ Hybrid scoring (FAST mode only)
        "tci_sj_scoring":       _sj_scoring,
        "tci_sj_roc9":          _sj_roc9,
        "tci_chop_filter":      _env_bool("CHOP_FILTER_ENABLED",    True),
        # MACD peak/slope filter: disabled — was blocking valid early entries (entry lag fix)
        "tci_macd_peak_filter": _env_bool("TCI_MACD_PEAK_FILTER",   False),
        # Pullback zone: compare live 15m close vs 1H EMA20 (vs stale 1H bar close)
        "tci_pullback_live_15m": _env_bool("TCI_PULLBACK_LIVE_15M", True),

        # ── Position health monitor ───────────────────────────────────────────
        "monitor_interval":      _env_int("MONITOR_INTERVAL",       60),
        "health_weak_confirm":   _env_int("HEALTH_WEAK_CONFIRM",   3),

        # ── Risk / sizing ─────────────────────────────────────────────────────
        # RISK_PER_TRADE: fraction of free balance lost if SL is hit (8-12% recommended).
        # Dynamic sizing: notional = (balance × risk_pct) / sl_dist_pct.
        "risk_per_trade":   _env_float("RISK_PER_TRADE",   0.10),  # 10% of balance
        "fixed_trade_usdt": _env_float("FIXED_TRADE_USDT", 0.0),   # 0 = use risk_per_trade
        "stop_loss_pct":   _env_float("STOP_LOSS_PCT",   0.015),
        "take_profit_pct": _env_float("TAKE_PROFIT_PCT", 0.025),
        "max_positions":  _env_int("MAX_POSITIONS", 2),
        "max_drawdown":   _env_float("MAX_DRAWDOWN_PCT", 0.20),
        "daily_loss_limit": _env_float("DAILY_LOSS_LIMIT_PCT", 0.05),

        # ── Timing ────────────────────────────────────────────────────────────
        "interval": _env_int("INTERVAL_SECONDS", 30),

        # ── Telegram ──────────────────────────────────────────────────────────
        "telegram_token":    os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id":  os.environ.get("TELEGRAM_CHAT_ID",   ""),
        "tg_min_confidence": _env_float("TG_MIN_CONFIDENCE", 0.5),
    }


# ── Strategy builder ────────────────────────────────────────────────────────

def build_strategies(symbols: list[str], cfg: dict) -> list:
    from trading.strategies.trend_cont_v2_strategy import TrendContV2Strategy

    strategies = []
    for sym in symbols:
        strategies.append(TrendContV2Strategy(sym, params={
            "name":                   "TrendContV2",
            "tf":                     "15m",
            "limit":                  500,
            # Risk
            "sl_mult":                cfg["tci_sl_mult"],
            "sl_min_pct":             cfg["tci_sl_min_pct"],
            "sl_max_pct":             cfg["tci_sl_max_pct"],
            "tp1_r":                  cfg["tci_tp1_r"],
            "tp2_r":                  cfg["tci_fast_tp2_r"],
            "sl_ladder_enabled":      True,
            # Health / crash guard
            "health_guard_enabled":   cfg["tci_health_guard"],
            "reversal_spike_enabled": cfg["reversal_spike_enabled"],
            "reversal_spike_atr":     cfg["reversal_spike_atr"],
            "reversal_spike_bars":    cfg["reversal_spike_bars"],
            "reversal_spike_uw_frac": 0.7,
            "trend_fade_enabled":     cfg["trend_fade_enabled"],
            "trend_fade_uw_frac":     cfg["trend_fade_uw_frac"],
            # Cooldown
            "cooldown_bars":          cfg["tci_cooldown_bars"],
            # Sizing
            "risk_per_trade":         cfg["risk_per_trade"],
        }))

    if not strategies:
        raise RuntimeError("No strategies built — check SYMBOLS env var is not empty")
    logger.info("Strategies loaded: %s for symbols %s", [s.name for s in strategies], symbols)
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

    from trading.connectors.binance_conn import BinanceConnector
    connector = BinanceConnector(
        api_key=cfg["api_key"],
        api_secret=cfg["api_secret"],
        paper=cfg["paper"],
        exchange_id=cfg["exchange"],
        passphrase=cfg["api_passphrase"],
        market_type=cfg["market_type"],
        leverage=cfg["leverage"],
        symbol_leverage=cfg["symbol_leverage"],
    )

    strategies = build_strategies(cfg["symbols"], cfg)

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
        "Risk: risk_per_trade=%.0f%%  lev=%dx  max_pos=%d  drawdown=%.0f%%  dailyCB=%.0f%%",
        cfg["risk_per_trade"] * 100, cfg["leverage"],
        cfg["max_positions"], cfg["max_drawdown"] * 100, cfg["daily_loss_limit"] * 100,
    )

    from trading.bot import TradingBot
    bot = TradingBot(
        connector=connector,
        strategies=strategies,
        risk_manager=risk,
        interval_seconds=cfg["interval"],
        telegram=telegram,
        fixed_sl_pct=cfg["stop_loss_pct"],
        fixed_tp_pct=cfg["take_profit_pct"],
        dynamic_sizing=True,
        monitor_interval=cfg["monitor_interval"],
    )
    bot._health_weak_confirm = cfg["health_weak_confirm"]

    async def _run_backtest() -> str:
        from trading.backtester import backtest_strategy_mtf_v2, summarise
        from trading.strategies.trend_cont_improved_strategy import TrendContImprovedStrategy

        syms     = cfg["symbols"]
        notional = cfg["fixed_trade_usdt"] * cfg["leverage"]
        results: dict[str, dict] = {}

        for sym in syms:
            cache: dict = {}
            for tf, lim in [("15m", 2000), ("1h", 500), ("4h", 200)]:
                logger.info("[BT] Fetching %s %s %d bars...", sym, tf, lim)
                try:
                    cache[(sym, tf)] = await connector.fetch_ohlcv(sym, timeframe=tf, limit=lim)
                except Exception as e:
                    logger.error("[BT] fetch %s %s failed: %s", sym, tf, e)
                    cache[(sym, tf)] = []

            c15m = cache.get((sym, "15m"), [])
            c1h  = cache.get((sym, "1h"),  [])
            c4h  = cache.get((sym, "4h"),  [])
            tag  = sym.split("/")[0]

            if c15m and c1h and c4h:
                tci = TrendContImprovedStrategy(sym, params={
                    "name": "TrendContImproved", "tf": "15m", "limit": 500,
                    "bias_gate": cfg["tci_bias_gate"], "adx_min": cfg["tci_adx_min"],
                    "sl_mult": cfg["tci_sl_mult"], "sl_min_pct": cfg["tci_sl_min_pct"],
                    "sl_max_pct": cfg["tci_sl_max_pct"],
                    "tp1_r": cfg["tci_tp1_r"], "tp1_fraction": cfg["tci_tp1_fraction"],
                    "tp2_r": cfg["tci_tp2_r"], "min_score": cfg["tci_min_score"],
                    "vol_mult": cfg["tci_vol_mult"],
                    "swing_lookback": cfg["tci_swing_lookback"],
                    "swing_pct": cfg["tci_swing_pct"],
                    "fast_mode": cfg["tci_fast_mode"],
                    "bias_gate_fast": cfg["tci_fast_bias_gate"],
                    "adx_min_fast": cfg["tci_fast_adx_min"],
                    "adx_max_fast": cfg["tci_fast_adx_max"],
                    "tp2_r_fast": cfg["tci_fast_tp2_r"],
                    "pullback_pct_fast": cfg["tci_fast_pullback_pct"],
                    "adx_rising_fast": cfg["tci_adx_rising"],
                    "cooldown_bars": cfg["tci_cooldown_bars"],
                    "sj_scoring": cfg["tci_sj_scoring"],
                    "sj_roc9": cfg["tci_sj_roc9"],
                })
                trades, _ = await backtest_strategy_mtf_v2(
                    tci, c15m, {"1h": c1h, "4h": c4h}, notional,
                    warmup=900, primary_window=500, mtf_windows={"1h": 200, "4h": 120},
                )
                results[f"TCI/{tag}"] = summarise(trades)

        lines = [f"📊 *Backtest — TrendContImproved* (${cfg['fixed_trade_usdt']}×{cfg['leverage']}x=${notional:.0f})"]
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

    if telegram:
        stop_event_ref: list = []

        def _tg_stop():
            if stop_event_ref:
                stop_event_ref[0].set()
            return {"message": "Bot stopping..."}

        telegram.get_state_fn = bot.get_state
        telegram.get_stats_fn = bot.get_stats
        telegram.stop_bot_fn  = _tg_stop
        telegram.start_bot_fn = lambda: {"message": "Bot is already running"}
        telegram.backtest_fn  = _run_backtest

    stop_signal = asyncio.Event()
    if telegram:
        stop_event_ref.append(stop_signal)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_signal.set)
        except (NotImplementedError, RuntimeError):
            pass

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
