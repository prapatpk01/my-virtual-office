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
        "leverage":        _env_int("LEVERAGE", 10),
        "symbols":         _env_list("SYMBOLS", "BTC/USDT:USDT"),
        "candle_tf":       os.environ.get("CANDLE_TF", "15m"),
        "candle_limit":    _env_int("CANDLE_LIMIT", 400),
        "interval":        _env_int("INTERVAL_SECONDS", 60),
        "trade_amount_usdt": _env_float("TRADE_AMOUNT_USDT", 100.0),
        "max_positions":   _env_int("MAX_POSITIONS", 4),
        "max_drawdown":    _env_float("MAX_DRAWDOWN_PCT", 0.20),
        "risk_per_trade":  _env_float("RISK_PER_TRADE", 0.01),
        "strategies": {
            "swing_reversal_pro": _env_bool("STRATEGY_SWING_REVERSAL", True),
            "mean_reversion":     _env_bool("STRATEGY_MEAN_REVERSION", True),
        },
        "swing_reversal_params": {
            "risk_pct":        _env_float("SR_RISK",        0.01),
            "l1_min_score":    _env_int("SR_L1",               5),   # Mode A quality gate (5/7)
            "l2_min_pass":     _env_int("SR_L2",               4),   # Mode A context (4/6)
            "sl_atr_mult":     _env_float("SR_SL",            1.5),  # SL = 1.5×ATR
            "tp_mult":         _env_float("SR_TP",            2.0),  # TP = 2.0×SL dist (1:2 R:R)
            "adx_no_trade":    _env_float("SR_ADX_MIN",      10.0),  # 15m ADX min threshold
            "adx_must_rise":   _env_bool("SR_ADX_RISING",   False),  # require ADX rising over N bars
            "adx_rise_bars":   _env_int("SR_ADX_RISE_BARS",     3),  # lookback for rising check
            "mtf_bias_limit":  _env_float("SR_MTF_LIMIT",    50.0),
            "max_bars":        _env_int("SR_MAXBARS",           48), # 12h max hold (48×15m bars)
            "rsi_entry_long":  _env_float("SR_RSI_LONG",      42.0), # Mode A RSI gate long
            "rsi_entry_short": _env_float("SR_RSI_SHORT",     58.0), # Mode A RSI gate short
        },
        "telegram_token":      os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id":    os.environ.get("TELEGRAM_CHAT_ID", ""),
        "use_adaptive": _env_bool("USE_ADAPTIVE", False),
        "adaptive_balance": _env_float("ADAPTIVE_BALANCE", 10000.0),
        "adaptive_risk_pct": _env_float("ADAPTIVE_RISK_PCT", 0.01),
        "adaptive_daily_loss": _env_float("ADAPTIVE_DAILY_LOSS_PCT", -3.0),
        "adaptive_daily_profit": _env_float("ADAPTIVE_DAILY_PROFIT_PCT", 8.0),
        "adaptive_cooldown_min": _env_int("ADAPTIVE_COOLDOWN_MIN", 30),
        "adaptive_max_loss_streak": _env_int("ADAPTIVE_MAX_LOSS_STREAK", 3),
        "adaptive_max_positions": _env_int("ADAPTIVE_MAX_POSITIONS", 2),
        # TP geometry dial (None = use bot class defaults 0.7 / 1.5). Lower
        # ADAPTIVE_TP1_R raises win-rate at the cost of average win size.
        "adaptive_tp1_r": (_env_float("ADAPTIVE_TP1_R", 0.0) or None),
        "adaptive_tp2_r": (_env_float("ADAPTIVE_TP2_R", 0.0) or None),
        # Fake-signal chop-zone filter (None = default 0.8). Higher = stricter,
        # higher WR, fewer trades (~1.2 pushes WR toward 56%).
        "adaptive_min_ema_dist_atr": (_env_float("ADAPTIVE_MIN_EMA_DIST_ATR", 0.0) or None),
    }


# ---------------------------------------------------------------------------
# Build strategies
# ---------------------------------------------------------------------------

def _make_strategies(symbols: list, flags: dict, cfg: dict,
                     connector=None) -> list:
    from trading.strategies.swing_reversal_pro import SwingReversalPro

    strategies = []
    p = cfg["swing_reversal_params"]
    for sym in symbols:
        if flags.get("swing_reversal_pro"):
            strategies.append(SwingReversalPro(sym, params={**p, "direction": "long"}))
            strategies.append(SwingReversalPro(sym, params={**p, "direction": "short"}))
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
# Adaptive mode runner
# ---------------------------------------------------------------------------

async def _run_adaptive(cfg, connector, telegram, stop_event):
    from trading.adaptive_trading_bot import TradingBot as AdaptiveBot
    from trading.indicator_engine import IndicatorEngine

    symbols = cfg["symbols"]
    logger.info("=== ADAPTIVE MODE: %d symbols ===", len(symbols))

    # FIX-#2: choose execution adapter based on configured exchange
    exchange_id = cfg.get("exchange", "okx").lower()
    if exchange_id == "binance":
        from trading.connectors.binance_conn import BinanceConnector as _ExecAdapter
        okx = _ExecAdapter(
            api_key=cfg["api_key"],
            api_secret=cfg["api_secret"],
            paper=cfg["paper"],
            exchange_id="binance",
            passphrase="",
            leverage=cfg.get("leverage", 10),
        )
    else:
        from trading.connectors.okx_adapter import OKXAdapter
        okx = OKXAdapter(
            api_key=cfg["api_key"],
            api_secret=cfg["api_secret"],
            api_passphrase=cfg.get("api_passphrase", ""),
            paper=cfg["paper"],
            leverage=cfg.get("leverage", 10),
        )

    ind_engine = IndicatorEngine()

    # One adaptive bot per symbol
    bots: dict = {}
    last_bar_ts: dict = {}  # symbol → last processed 15m bar timestamp

    # FIX-#6: use BOT_STATE_DIR env var so state survives container restarts
    import os as _os
    _state_dir = _os.environ.get("BOT_STATE_DIR", "/tmp")
    _os.makedirs(_state_dir, exist_ok=True)

    for sym in symbols:
        safe_sym = sym.replace("/", "_").replace(":", "_")
        state_file = _os.path.join(_state_dir, f"adaptive_{safe_sym}.json")

        def _make_callback(s, t):
            def cb(order_type, trade_info):
                result = t.execute(order_type, {**trade_info, "symbol": s})
                if telegram:
                    try:
                        if "OPEN" in order_type:
                            msg = (
                                f"🟢 [Adaptive] {order_type} {s}\n"
                                f"entry={trade_info.get('entry', '?')} "
                                f"sl={trade_info.get('sl', '?')}\n"
                                f"TP1={trade_info.get('tp1', '?')} "
                                f"TP2={trade_info.get('tp2', '?')} "
                                f"size={float(trade_info.get('size') or 0):.4f}"
                            )
                        else:
                            pnl = float(trade_info.get("pnl") or 0)
                            emoji = "✅" if pnl > 0 else ("⚪" if pnl == 0 else "❌")
                            msg = (
                                f"{emoji} [Adaptive] {order_type} {s} "
                                f"({trade_info.get('reason', '')})\n"
                                f"direction={trade_info.get('direction', '?')} "
                                f"price={trade_info.get('price', '?')}\n"
                                f"pnl={pnl:+.2f} size={float(trade_info.get('size') or 0):.4f}"
                            )
                        telegram.send(msg)
                    except Exception:
                        pass
                return result
            return cb

        bot = AdaptiveBot(
            account_balance=cfg["adaptive_balance"],
            base_risk_pct=cfg["adaptive_risk_pct"],
            daily_loss_limit_pct=cfg["adaptive_daily_loss"],
            daily_profit_limit_pct=cfg["adaptive_daily_profit"],
            cooldown_minutes=cfg["adaptive_cooldown_min"],
            max_loss_streak=cfg["adaptive_max_loss_streak"],
            state_file=state_file,
            execution_callback=_make_callback(sym, okx),
            enable_swing_reversal=cfg["strategies"].get("swing_reversal_pro", True),
            enable_mean_reversion=cfg["strategies"].get("mean_reversion", False),
            tp1_r=cfg.get("adaptive_tp1_r"),
            tp2_r=cfg.get("adaptive_tp2_r"),
            min_ema_dist_atr=cfg.get("adaptive_min_ema_dist_atr"),
        )
        bot.load_state(state_file)
        bot.reconcile_with_exchange(sym, okx)
        bots[sym] = (bot, state_file)
        last_bar_ts[sym] = 0

    if telegram:
        # Wire bots dict so /stats and /log commands work
        telegram.bots_dict = bots
        # Wire /stop command so it reaches the adaptive runner's stop_event
        telegram.stop_bot_fn = lambda: stop_event.set()
        # FIX-#8: start Telegram command polling in adaptive mode
        loop = asyncio.get_event_loop()
        telegram.start_polling(loop)
        telegram.send(
            f"Adaptive Bot Started\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Mode: {'PAPER' if cfg['paper'] else 'LIVE'}\n"
            f"Warmup: 45m — no new entries until indicators stabilize"
        )

    import time as _time
    import datetime as _dt

    loop = asyncio.get_event_loop()
    max_pos = cfg.get("adaptive_max_positions", 2)

    # 5-minute health log: tracks last log time and cached indicators per symbol
    HEALTH_LOG_SECS = 300
    last_health_log: dict = {sym: 0.0 for sym in symbols}
    last_ind_cache:  dict = {sym: {}  for sym in symbols}
    last_price_cache: dict = {sym: 0.0 for sym in symbols}

    # Cooldown expiry check — independent of new-candle ticks (on_tick only
    # runs per closed 15m bar, so without this a cooldown could sit expired
    # for up to 15 min before the bot resumes SCANNING). Default: every 5 min.
    COOLDOWN_CHECK_SECS = _env_int("COOLDOWN_CHECK_SECONDS", 300)
    last_cooldown_check = _time.time()

    # Entry-filter rejection summary — INFO level so it's visible without
    # LOG_LEVEL=DEBUG. Shows which gate (Bias/Health/Confidence) blocks most.
    FILTER_STATS_LOG_SECS = _env_int("FILTER_STATS_LOG_SECONDS", 300)
    last_filter_stats_log = _time.time()

    # Real balance sync — bots are constructed with a fixed ADAPTIVE_BALANCE
    # (default $10k) that never reflected the actual exchange account, so
    # position sizing could exceed real available margin and get rejected
    # ("available margin too low"). Refresh from the exchange periodically.
    BALANCE_SYNC_SECS = _env_int("BALANCE_SYNC_SECONDS", 300)
    last_balance_sync = 0.0  # force an immediate sync on the first loop tick

    # Exchange reconciliation — on_tick only updates position state from bot
    # logic, so a position closed externally (manually, liquidated, or via an
    # exchange-side TP/SL) would keep showing as open in the bot until restart.
    # reconcile_with_exchange already handles "local=open, exchange=flat" —
    # just needs to run periodically, not only at startup.
    RECONCILE_SYNC_SECS = _env_int("RECONCILE_SYNC_SECONDS", 300)
    last_reconcile_sync = _time.time()

    # Adaptive state scan log — the [Adaptive][sym] state=... line otherwise
    # only fires when a new 15m candle closes (up to 15 min gap). Mirror the
    # Health log's independent 5-min cadence so state is visible that often
    # even when no new bar has closed yet.
    SCAN_LOG_SECS = _env_int("SCAN_LOG_SECONDS", 300)
    last_scan_log = _time.time()

    _HEALTH_EMOJI = {
        "STRONG":   "✅",
        "GOOD":     "🟢",
        "WARN":     "🟡",
        "POOR":     "🟠",
        "CRITICAL": "🔴",
    }

    # Track which symbols had a position open on the previous health-log cycle,
    # so we can reset the timer when a new position opens (fires the first
    # health log immediately rather than waiting up to 5 min).
    _had_position: dict = {sym: False for sym in symbols}

    def _log_health(sym: str, bot, price: float, ind_15m: dict):
        """Emit a 5-minute health log — only when a position is open."""
        if not bot.position_open or not ind_15m:
            return   # no position → nothing to report
        report = bot.get_position_health_report(price, ind_15m)
        level   = report.get("health_level", "?")
        emoji   = _HEALTH_EMOJI.get(level, "?")
        rev     = report.get("reversal_signals", {})
        rev_str = f" | REVERSAL={','.join(rev)}" if rev else ""
        tp1_str = " TP1✓" if report.get("tp1_hit") else ""
        logger.info(
            "[Health][%s] %s %s(%.0f)%s | dir=%s entry=%.2f cur=%.2f"
            " pnl=%+.2f R=%.2f | SL=%.2f TP1=%.2f TP2=%.2f"
            " | ADX=%.1f RSI=%.1f bars=%d%s",
            sym, emoji, level, report["health_score"], tp1_str,
            report["direction"], report["entry"], price,
            report["pnl"], report["current_r"],
            report["sl"], report["tp1"], report["tp2"],
            report["adx"], report["rsi"], report["holding_bars"],
            rev_str,
        )

    # Note: periodic Telegram stats removed — too noisy.
    # Use /stats command to check on demand. Railway logs show state every tick.
    # 5-minute health logs are written to Railway log automatically below.

    while not stop_event.is_set():
        for sym in symbols:
            try:
                # Fetch candles for all 3 timeframes
                c15m = await connector.fetch_ohlcv(sym, "15m", 300)
                c1h  = await connector.fetch_ohlcv(sym, "1h",  200)
                c4h  = await connector.fetch_ohlcv(sym, "4h",  200)

                if not c15m or not c1h or not c4h:
                    logger.warning("[Adaptive][%s] Empty candles — skipping", sym)
                    continue

                latest_ts = c15m[-1].timestamp if c15m else 0
                is_new_bar = latest_ts > last_bar_ts[sym]
                bot, state_file = bots[sym]

                if is_new_bar:
                    last_bar_ts[sym] = latest_ts

                    # Global position cap — bots without open position skip when full.
                    # Bots that already hold a position always run (SL/TP management).
                    should_process = True
                    if not bot.position_open:
                        open_count = sum(
                            1 for b, _ in bots.values() if b.position_open
                        )
                        if open_count >= max_pos:
                            logger.debug(
                                "[Adaptive][%s] SKIP — global positions %d/%d full",
                                sym, open_count, max_pos,
                            )
                            should_process = False

                    if should_process:
                        # Compute indicators
                        candle_15m, candle_1h, candle_4h, ind_15m, ind_1h, ind_4h = \
                            ind_engine.compute(c15m, c1h, c4h)

                        price = candle_15m.get("close", 0.0)

                        # Cache for health logs between new bars
                        last_ind_cache[sym]   = ind_15m
                        last_price_cache[sym] = price

                        extras = {"symbol": sym, "session": "", "funding_rate": 0.0, "oi": 0}

                        bar_dt = _dt.datetime.fromtimestamp(
                            latest_ts / 1000, tz=_dt.timezone.utc
                        )

                        # Run on_tick in thread (sync order execution inside)
                        await loop.run_in_executor(
                            None,
                            lambda b=bot, sf=state_file, bdt=bar_dt: b.on_tick(
                                candle_15m, candle_1h, candle_4h,
                                ind_15m, ind_1h, ind_4h,
                                extras, price,
                                bar_dt=bdt,
                            )
                        )

                        # Log state on every new bar
                        status = bot.get_status()
                        logger.info(
                            "[Adaptive][%s] state=%s pos=%s market=%s regime=%.0f",
                            sym, status["state"], status["position_open"],
                            status["market_state"], status["regime_score"],
                        )

                # 5-minute health log — only fires when a position is open.
                # Timer resets the moment a new position opens so the first
                # health report fires immediately (not after 5 min).
                _ts = _time.time()
                pos_now = bot.position_open
                if pos_now and not _had_position[sym]:
                    # Position just opened — fire immediately and start timer
                    last_health_log[sym] = _ts
                    if last_ind_cache.get(sym):
                        _log_health(sym, bot, last_price_cache.get(sym, 0.0),
                                    last_ind_cache[sym])
                elif (pos_now
                        and _ts - last_health_log.get(sym, 0) >= HEALTH_LOG_SECS
                        and last_ind_cache.get(sym)):
                    last_health_log[sym] = _ts
                    _log_health(sym, bot, last_price_cache.get(sym, 0.0),
                                last_ind_cache[sym])
                _had_position[sym] = pos_now

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("[Adaptive][%s] tick error: %s", sym, e, exc_info=True)

        if _time.time() - last_cooldown_check >= COOLDOWN_CHECK_SECS:
            last_cooldown_check = _time.time()
            for sym, (bot, _sf) in bots.items():
                if bot.check_cooldown_expiry():
                    logger.info("[Adaptive][%s] COOLDOWN expired (5-min check) → SCANNING", sym)

        if _time.time() - last_filter_stats_log >= FILTER_STATS_LOG_SECS:
            last_filter_stats_log = _time.time()
            for sym, (bot, _sf) in bots.items():
                fs = bot.get_filter_stats()
                if fs["checked"] == 0:
                    continue
                logger.info(
                    "[FilterStats][%s] checked=%d passed=%d | "
                    "bias_fail=%d health_fail=%d confidence_fail=%d",
                    sym, fs["checked"], fs["passed"],
                    fs["bias_fail"], fs["health_fail"], fs["confidence_fail"],
                )

        if _time.time() - last_balance_sync >= BALANCE_SYNC_SECS:
            last_balance_sync = _time.time()
            try:
                balances = await okx.fetch_balance()
                usdt = next((b for b in balances if b.asset == "USDT"), None)
                real_balance = usdt.free if usdt else 0.0
                if real_balance and real_balance > 0:
                    for _sym, (bot, _sf) in bots.items():
                        bot.account_balance = real_balance
                    logger.info("[Balance] synced from exchange: %.2f USDT", real_balance)
                else:
                    logger.warning("[Balance] exchange returned %.2f — keeping previous value", real_balance)
            except Exception as e:
                logger.warning("[Balance] sync failed (non-fatal): %s", e)

        if _time.time() - last_reconcile_sync >= RECONCILE_SYNC_SECS:
            last_reconcile_sync = _time.time()
            for sym, (bot, _sf) in bots.items():
                try:
                    was_open = bot.position_open
                    await loop.run_in_executor(None, bot.reconcile_with_exchange, sym, okx)
                    if was_open and not bot.position_open:
                        logger.info(
                            "[Reconcile][%s] position closed externally — local state cleared",
                            sym,
                        )
                except Exception as e:
                    logger.warning("[Reconcile][%s] periodic sync failed: %s", sym, e)

        if _time.time() - last_scan_log >= SCAN_LOG_SECS:
            last_scan_log = _time.time()
            for sym, (bot, _sf) in bots.items():
                try:
                    status = bot.get_status()
                    logger.info(
                        "[Adaptive][%s] state=%s pos=%s market=%s regime=%.0f",
                        sym, status["state"], status["position_open"],
                        status["market_state"], status["regime_score"],
                    )
                except Exception as e:
                    logger.warning("[Adaptive][%s] scan log failed: %s", sym, e)

        # Stats are on-demand only (via the /stats Telegram command) — no
        # periodic auto-digest, to avoid spamming the chat.

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=cfg["interval"])
        except asyncio.TimeoutError:
            pass

    # FIX-#6/#8: persist state for all bots on clean shutdown (SIGTERM/stop)
    for sym, (bot, sf) in bots.items():
        try:
            bot.save_state(sf)
            logger.info("[Adaptive] State saved for %s → %s", sym, sf)
        except Exception as e:
            logger.warning("[Adaptive] Could not save state for %s: %s", sym, e)

    # The execution adapter created here (okx) is a SEPARATE ccxt async client
    # from the outer `connector` — _cleanup_connector() in main() only closes
    # `connector`, so okx's aiohttp session was never closed, producing the
    # "Unclosed client session" warning on every shutdown/redeploy.
    try:
        await okx.close()
    except Exception as e:
        logger.warning("[Adaptive] okx connector close error (non-fatal): %s", e)


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

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, RuntimeError):
            pass

    telegram = _make_telegram(cfg)

    async def _cleanup_connector():
        logger.info("Stopping bot...")
        try:
            await connector.close()
        except Exception as e:
            logger.warning("Connector close error (non-fatal): %s", e)

        # Belt-and-suspenders: force-clear CCXT/aiohttp objects so their __del__
        # never emits "Unclosed connector" or "requires .close()" during GC.
        # This runs whether the bot exits cleanly OR crashes on startup.
        _exc = getattr(connector, "_exchange", None)
        if _exc is not None:
            try:
                _sess = getattr(_exc, "session", None)
                if _sess is not None:
                    _aio_conn = getattr(_sess, "_connector", None)
                    if _aio_conn is not None and not getattr(_aio_conn, "_closed", True):
                        await _aio_conn.close()
                    _exc.session = None
                _exc.socks_proxy_sessions = None
                _tcp = getattr(_exc, "tcp_connector", None)
                if _tcp is not None:
                    if not getattr(_tcp, "_closed", True):
                        await _tcp.close()
                    _exc.tcp_connector = None
            except Exception:
                pass

        await asyncio.sleep(0.5)
        logger.info("Done.")

    if cfg.get("use_adaptive"):
        logger.info("Starting ADAPTIVE trading mode")
        try:
            await _run_adaptive(cfg, connector, telegram, stop_event)
        finally:
            await _cleanup_connector()
        return

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

    if telegram:
        telegram.stop_bot_fn = lambda: stop_event.set()

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

    if telegram:
        telegram.bot = bot

    try:
        await bot.start()
        await stop_event.wait()
    finally:
        try:
            await bot.stop()
        except Exception as e:
            logger.warning("Bot stop error (non-fatal): %s", e)
        await _cleanup_connector()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
