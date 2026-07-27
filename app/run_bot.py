"""
Standalone Trading Bot runner — no web UI required.
Runs the bot + Telegram integration 24/7 on any cloud platform.

Usage:
    python run_bot.py

Config via environment variables (see .env.example) or a .env file.

Strategy modes (STRATEGY env var):
  ai_expert  — Layer 0-8 AI Decision Engine (default, institutional grade)
  ema_macd   — EMA12/26 cross + SMA50 filter (15m) + MACD 30m confirmation
  ema_sma    — EMA12/26 cross + SMA50 filter, pure 15m (no MACD confirmation)
  hma_macd_roc — HMA10/20 cross gate + MACD + ROC 3-bar confirmation (30m)
  trend_confirm — SMA30+MACD trend gate + EMA5/10 cross entry (30m)
  mcdx       — Legacy MCDX strategy
  wt_adx     — Legacy WaveTrend + ADX strategy
"""
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, time as dt_time, timezone
from zoneinfo import ZoneInfo


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
# Send ALL logs to stdout (not the default stderr). Railway/most log viewers
# render stderr as red "error" lines, which made ordinary INFO logs look like
# failures. stream=sys.stdout keeps INFO/WARNING/ERROR on the normal stream.
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_bot")


def _env_list(key: str, default: str) -> list[str]:
    val = os.environ.get(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")


def build_config() -> dict:
    return {
        # ── Exchange ──────────────────────────────────────────────────────
        "exchange":         os.environ.get("EXCHANGE",          "okx"),
        "api_key":          os.environ.get("EXCHANGE_API_KEY",  ""),
        "api_secret":       os.environ.get("EXCHANGE_API_SECRET", ""),
        "api_passphrase":   os.environ.get("EXCHANGE_PASSPHRASE", ""),
        "paper":            _env_bool("PAPER_TRADING", False),
        # OANDA (forex only — leave blank for crypto)
        "oanda_api_key":    os.environ.get("OANDA_API_KEY", ""),
        "oanda_account_id": os.environ.get("OANDA_ACCOUNT_ID", ""),
        "oanda_env":        os.environ.get("OANDA_ENV", "practice"),

        # ── Futures / leverage / hedge mode ──────────────────────────────
        "futures":    _env_bool("FUTURES",    True),   # use perpetual swaps
        "leverage":   int(os.environ.get("LEVERAGE",   "20")),
        "hedge_mode": _env_bool("HEDGE_MODE", True),   # hold LONG + SHORT simultaneously

        # ── Symbols & candles ─────────────────────────────────────────────
        # Perpetual format: BTC/USDT:USDT (OKX/Bybit), BTC/USDT (Binance perp)
        "symbols":      _env_list("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT"),
        "candle_tf":    os.environ.get("CANDLE_TF",         "15m"),
        "candle_limit": int(os.environ.get("CANDLE_LIMIT",  "300")),
        "interval":     300,  # fixed 5-minute main scan cadence

        # ── Global FX-week Sleep Mode ─────────────────────────────────────
        # Apply the FX 24/5 weekly calendar to ALL trading symbols, including
        # crypto/perpetual symbols.  During sleep mode the bot keeps running
        # and manages already-open positions, but RiskManager.can_open() is
        # blocked so no NEW position can be created.
        #
        # Standard FX week: Sunday 17:00 -> Friday 17:00 New York time.
        # We intentionally allow entries FX_EARLY_OPEN_HOURS before the Sunday
        # open (default 4h -> Sunday 13:00 New York). ZoneInfo handles DST.
        "fx_sleep_mode":       _env_bool("FX_SLEEP_MODE", True),
        "fx_early_open_hours": float(os.environ.get("FX_EARLY_OPEN_HOURS", "4")),
        "fx_market_open_hour": int(os.environ.get("FX_MARKET_OPEN_HOUR_NY", "17")),
        "fx_market_close_hour": int(os.environ.get("FX_MARKET_CLOSE_HOUR_NY", "17")),
        # Optional broker-specific FULL-DAY closures in New York calendar
        # dates, e.g. FX_MARKET_HOLIDAYS=2026-12-25,2027-01-01
        # (FX itself trades through many public holidays, so these are opt-in).
        "fx_market_holidays": _env_list("FX_MARKET_HOLIDAYS", ""),

        # ── Strategy ──────────────────────────────────────────────────────
        "strategy_mode":            os.environ.get("STRATEGY", "ai_expert"),
        "strategies": {
            "mcdx":      _env_bool("STRATEGY_MCDX",      False),
            "wt_adx":    _env_bool("STRATEGY_WT_ADX",    False),
            "ai_expert": _env_bool("STRATEGY_AI_EXPERT", True),
        },
        "ai_expert_min_confidence": float(os.environ.get("AI_EXPERT_MIN_CONFIDENCE", "70")),
        "ai_expert_strict":         _env_bool("AI_EXPERT_STRICT", False),

        # ── Position sizing (confidence-based) ───────────────────────────
        # Actual sizes come from bot._confidence_size_pct() using
        # SIZE_WEAK / SIZE_GOOD / SIZE_HIGH env vars (defaults 8/10/12%).
        # risk_per_trade is only the RiskManager fallback (non-AI paths).
        "risk_per_trade":  float(os.environ.get("RISK_PER_TRADE",  "0.08")),

        # ── Stop-loss / take-profit (fallback when AI SL/TP absent) ──────
        "stop_loss_pct":   float(os.environ.get("STOP_LOSS_PCT",   "0.03")),   # 3 %
        "take_profit_pct": float(os.environ.get("TAKE_PROFIT_PCT", "0.036")),  # 1.2R

        # ── Risk limits ───────────────────────────────────────────────────
        "max_positions": int(os.environ.get("MAX_POSITIONS",    "2")),
        "max_drawdown":  float(os.environ.get("MAX_DRAWDOWN_PCT", "0.15")),    # 15 %
        "max_consecutive_sl": int(os.environ.get("MAX_CONSECUTIVE_SL", "3")),
        "cooldown_hours":     float(os.environ.get("COOLDOWN_HOURS", "3")),   # per-symbol pause
        "post_cooldown_strict_trades": int(os.environ.get("POST_COOLDOWN_STRICT_TRADES", "5")),
        "post_cooldown_threshold_bonus": float(os.environ.get("POST_COOLDOWN_THRESHOLD_BONUS", "6")),

        # ── Telegram ──────────────────────────────────────────────────────
        "telegram_token":    os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id":  os.environ.get("TELEGRAM_CHAT_ID",   ""),
        "tg_min_confidence": float(os.environ.get("TG_MIN_CONFIDENCE", "0.7")),

        # ── Forex signals (disabled by default for pure-crypto setup) ─────
        "forex_symbols":  _env_list("FOREX_SYMBOLS", "XAUUSD"),
        "forex_enabled":  _env_bool("FOREX_SIGNALS", False),
        "forex_interval": int(os.environ.get("FOREX_INTERVAL_SECONDS", "60")),
    }



_NY_TZ = ZoneInfo("America/New_York")


def _parse_fx_holidays(raw_dates: list[str]) -> set:
    """Parse optional YYYY-MM-DD full-day closures in New York time.

    FX is open on many public holidays, and broker holiday hours can differ.
    For that reason the runner only applies dates explicitly supplied through
    FX_MARKET_HOLIDAYS rather than guessing a universal holiday calendar.
    """
    holidays = set()
    for raw in raw_dates or []:
        raw = str(raw).strip()
        if not raw:
            continue
        try:
            holidays.add(datetime.strptime(raw, "%Y-%m-%d").date())
        except ValueError:
            logger.warning(
                "Ignoring invalid FX_MARKET_HOLIDAYS date %r (expected YYYY-MM-DD)",
                raw,
            )
    return holidays


def _fx_entry_session_status(config: dict, now_utc: datetime = None) -> dict:
    """Return global new-entry status using the FX weekly calendar.

    The underlying bots NEVER stop.  This function controls only whether new
    positions may be opened.  Existing positions therefore continue to receive
    normal SL/TP, trailing, BE and strategy-exit management during sleep mode.

    Weekly schedule (America/New_York, DST aware):
      * FX week closes Friday at 17:00 NY by default.
      * FX officially reopens Sunday at 17:00 NY by default.
      * This bot resumes EARLY by FX_EARLY_OPEN_HOURS (default 4 hours), so
        entries are allowed from Sunday 13:00 NY through Friday 17:00 NY.
    """
    if not config.get("fx_sleep_mode", True):
        return {
            "entry_allowed": True,
            "sleep": False,
            "reason": "FX sleep mode disabled",
            "now_ny": (now_utc or datetime.now(timezone.utc)).astimezone(_NY_TZ),
            "next_transition": None,
        }

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_ny = now.astimezone(_NY_TZ)

    open_hour = int(config.get("fx_market_open_hour", 17))
    close_hour = int(config.get("fx_market_close_hour", 17))
    early_hours = max(0.0, float(config.get("fx_early_open_hours", 4.0)))
    early_delta = timedelta(hours=early_hours)
    holidays = _parse_fx_holidays(config.get("fx_market_holidays", []))

    # Optional explicit full-day broker closures.  We still calculate the next
    # normal weekly transition below; monitor output will state the holiday.
    if now_ny.date() in holidays:
        tomorrow = (now_ny + timedelta(days=1)).date()
        next_transition = datetime.combine(tomorrow, dt_time(0, 0), tzinfo=_NY_TZ)
        return {
            "entry_allowed": False,
            "sleep": True,
            "reason": f"FX market holiday {now_ny.date().isoformat()} (NY)",
            "now_ny": now_ny,
            "next_transition": next_transition,
        }

    # Build this week's Friday close and Sunday early-resume times from the
    # current NY calendar date. weekday(): Mon=0 ... Sun=6.
    weekday = now_ny.weekday()

    # Entry window is open Monday-Thursday all day.
    if weekday <= 3:
        days_to_friday = 4 - weekday
        friday = (now_ny + timedelta(days=days_to_friday)).date()
        next_transition = datetime.combine(
            friday, dt_time(close_hour, 0), tzinfo=_NY_TZ
        )
        return {
            "entry_allowed": True,
            "sleep": False,
            "reason": "FX-week entry window open",
            "now_ny": now_ny,
            "next_transition": next_transition,
        }

    # Friday: entries allowed until the weekly close.
    if weekday == 4:
        close_dt = datetime.combine(
            now_ny.date(), dt_time(close_hour, 0), tzinfo=_NY_TZ
        )
        if now_ny < close_dt:
            return {
                "entry_allowed": True,
                "sleep": False,
                "reason": "FX-week entry window open",
                "now_ny": now_ny,
                "next_transition": close_dt,
            }
        # Closed Friday evening -> resume next Sunday 4h before official open.
        sunday = now_ny.date() + timedelta(days=2)
        official_open = datetime.combine(
            sunday, dt_time(open_hour, 0), tzinfo=_NY_TZ
        )
        resume = official_open - early_delta
        return {
            "entry_allowed": False,
            "sleep": True,
            "reason": "FX weekend closed",
            "now_ny": now_ny,
            "next_transition": resume,
        }

    # Saturday: always sleeping; resume Sunday early.
    if weekday == 5:
        sunday = now_ny.date() + timedelta(days=1)
        official_open = datetime.combine(
            sunday, dt_time(open_hour, 0), tzinfo=_NY_TZ
        )
        resume = official_open - early_delta
        return {
            "entry_allowed": False,
            "sleep": True,
            "reason": "FX weekend closed",
            "now_ny": now_ny,
            "next_transition": resume,
        }

    # Sunday: sleep until early-resume time, then allow every symbol to trade.
    official_open = datetime.combine(
        now_ny.date(), dt_time(open_hour, 0), tzinfo=_NY_TZ
    )
    resume = official_open - early_delta
    if now_ny < resume:
        return {
            "entry_allowed": False,
            "sleep": True,
            "reason": "FX weekend closed",
            "now_ny": now_ny,
            "next_transition": resume,
        }

    # Sunday early window is deliberately active even though spot FX may not
    # yet be open; this lets the configured crypto/perpetual universe start
    # four hours before the FX week officially opens, as requested.
    friday = now_ny.date() + timedelta(days=5)
    next_transition = datetime.combine(
        friday, dt_time(close_hour, 0), tzinfo=_NY_TZ
    )
    return {
        "entry_allowed": True,
        "sleep": False,
        "reason": f"Early-open window ({early_hours:g}h before FX open)",
        "now_ny": now_ny,
        "next_transition": next_transition,
    }


def _install_fx_sleep_entry_gate(bot, config: dict) -> None:
    """Block only NEW entries while preserving all open-position management.

    Every real crypto/futures entry path in TradingBot calls
    RiskManager.can_open().  Wrapping that method is therefore safer than
    stopping TradingBot.start() or setting max_open_positions=0 (which would
    activate the bot's signal-only mode).  Exits, hard SL/TP and trailing logic
    continue unchanged.
    """
    risk = getattr(bot, "risk", None)
    if risk is None or getattr(risk, "_fx_sleep_gate_installed", False):
        return

    original_can_open = risk.can_open

    def _session_aware_can_open(symbol: str, strategy: str = ""):
        status = _fx_entry_session_status(config)
        if not status["entry_allowed"]:
            nxt = status.get("next_transition")
            resume_text = nxt.strftime("%Y-%m-%d %H:%M %Z") if nxt else "unknown"
            return False, (
                f"SLEEP MODE: {status['reason']} — no new positions for any symbol; "
                f"next entry window {resume_text}"
            )
        return original_can_open(symbol, strategy=strategy)

    risk.can_open = _session_aware_can_open
    risk._fx_sleep_gate_installed = True
    logger.info("FX-week Sleep Mode entry gate installed for ALL symbols")
    logger.info("TrendConfirm diagnostic metadata enabled: 1H/15M/5M context will be present on HOLD signals")


async def _fx_sleep_monitor(config: dict):
    """Log sleep/active state changes without stopping the trading loop."""
    last_state = None
    while not _stop_signal.is_set():
        status = _fx_entry_session_status(config)
        state = "SLEEP" if status["sleep"] else "ACTIVE"
        if state != last_state:
            now_ny = status["now_ny"]
            nxt = status.get("next_transition")
            next_text = nxt.strftime("%Y-%m-%d %H:%M %Z") if nxt else "n/a"
            if status["sleep"]:
                logger.warning(
                    "🌙 SLEEP MODE | ALL symbols: NEW positions BLOCKED | "
                    "open positions still managed normally | reason=%s | "
                    "NY=%s | entries resume=%s",
                    status["reason"], now_ny.strftime("%Y-%m-%d %H:%M %Z"), next_text,
                )
            else:
                logger.info(
                    "🟢 ACTIVE MODE | ALL symbols: new entries ALLOWED | "
                    "reason=%s | NY=%s | next sleep=%s",
                    status["reason"], now_ny.strftime("%Y-%m-%d %H:%M %Z"), next_text,
                )
            last_state = state

        try:
            await asyncio.wait_for(_stop_signal.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass


def _install_trendconfirm_scan_logger(bot) -> None:
    """Compact ViewLog for 4H macro -> 1H quality -> 15M EMA cross."""
    original = getattr(bot, "_log_scan", None)
    if original is None or getattr(bot, "_trendconfirm_scan_logger_installed", False):
        return

    scan_logger = logging.getLogger("trading_bot")

    def _text(value, default="N/A"):
        if value is None or value == "":
            return default
        return str(value)

    def _patched(symbol, strategy_name, price, sig):
        meta = getattr(sig, "metadata", None) or {}
        is_tc = (
            str(strategy_name).startswith("TrendConfirm")
            or meta.get("strategy") == "EMA_CROSS_15M"
            or "macro_4h" in meta
            or "quality_1h" in meta
        )
        if not is_tc:
            return original(symbol, strategy_name, price, sig)

        macro = meta.get("macro_4h", {}) or {}
        t4 = _text(meta.get("trend_4h") or macro.get("state"), "WARMUP").upper()
        ctx = meta.get("quality_1h", {}) or {}
        t1 = _text(meta.get("trend_1h"), "WARMUP").upper()
        q1 = ctx.get("score")
        q1s = "--" if q1 is None else f"{float(q1):.0f}"
        adx1 = ctx.get("adx")
        chop1 = ctx.get("chop")
        qctx = f"Q={q1s} ADX={adx1 if adx1 is not None else '--'} CHOP={chop1 if chop1 is not None else '--'}"

        side = meta.get("sideways_15m", {}) or {}
        side_n = int(side.get("signals", 0) or 0)
        q15 = meta.get("quality_15m", {}) or {}
        b15 = q15.get("breakdown", {}) or {}
        chop15 = b15.get("chop_val")
        cross = _text(meta.get("cross_15m") or meta.get("direction_15m"), "WAIT")
        state = _text(meta.get("entry_state"), "HOLD")
        rr = meta.get("rr_ratio")
        rrs = "--" if rr is None else f"{float(rr):.1f}R"
        reason = (getattr(sig, "reason", "") or meta.get("hold_reason", "") or "")[:170]
        sig_type = getattr(getattr(sig, "type", None), "value", "HOLD")

        scan_logger.info(
            "[SCAN] %-16s %-22s px=%-12.4f sig=%-4s "
            "4H=%-12s 1H=%-16s %-24s 15M=%-12s CHOP15=%-5s side=%d/4 "
            "state=%-16s TP=%-5s | %s",
            strategy_name, symbol, price, str(sig_type).upper(),
            t4, t1, qctx, cross,
            "--" if chop15 is None else f"{float(chop15):.1f}", side_n,
            state, rrs, reason,
        )

    bot._log_scan = _patched
    bot._trendconfirm_scan_logger_installed = True
    logger.info("TrendConfirm ViewLog formatter installed — 4H/1H/15M cross schema")

def _make_strategies(symbols: list, config: dict):
    mode   = config.get("strategy_mode", "ai_expert")
    flags  = config.get("strategies", {})
    strategies = []

    for sym in symbols:
        if mode == "ema_macd":
            from trading.strategies.ema_macd_strategy import EMAMacdStrategy
            strategies.append(EMAMacdStrategy(sym))
        elif mode == "ema_sma":
            from trading.strategies.ema_sma_strategy import EMASMAStrategy
            strategies.append(EMASMAStrategy(sym))
        elif mode == "hma_macd_roc":
            from trading.strategies.hma_macd_roc_strategy import HMAMacdROCStrategy
            strategies.append(HMAMacdROCStrategy(sym))
        elif mode == "trend_confirm":
            from trading.strategies.trend_confirm_strategy import TrendConfirmStrategy
            strategies.append(TrendConfirmStrategy(sym))
        elif mode == "mcdx" or flags.get("mcdx", False):
            from trading.strategies.mcdx_strategy import MCDXStrategy
            strategies.append(MCDXStrategy(sym))
        elif mode == "wt_adx" or flags.get("wt_adx", False):
            from trading.strategies.wt_adx_strategy import WTADXStrategy
            strategies.append(WTADXStrategy(sym))
        elif mode == "ai_expert" or flags.get("ai_expert", True):
            from trading.strategies.ai_expert_strategy import AIExpertStrategy
            strategies.append(AIExpertStrategy(
                sym,
                min_confidence=config.get("ai_expert_min_confidence", 70.0),
                require_all_checks=config.get("ai_expert_strict", False),
            ))
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

    exchange   = config["exchange"]
    futures    = config.get("futures", True)
    leverage   = config.get("leverage", 20)
    hedge_mode = config.get("hedge_mode", True)

    if exchange == "oanda":
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
            futures=futures, leverage=leverage, hedge_mode=hedge_mode,
        )
        logger.info(
            "Connector: %s | paper=%s | futures=%s | leverage=%dx | hedge=%s",
            exchange, config["paper"], futures, leverage, hedge_mode,
        )
        if futures and leverage < 5:
            logger.warning(
                "LEVERAGE=%dx is unusually low for futures trading — position "
                "sizing (margin x leverage) will size trades ~%dx SMALLER than "
                "intended if the exchange itself has a higher leverage set "
                "(e.g. LEVERAGE env says %d but the OKX UI shows 20x). Check "
                "the LEVERAGE env var if this isn't deliberate.",
                leverage, max(1, 20 // max(leverage, 1)), leverage,
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
        max_consecutive_sl=config["max_consecutive_sl"],
        cooldown_hours=config["cooldown_hours"],
        post_cooldown_strict_trades=config["post_cooldown_strict_trades"],
        post_cooldown_threshold_bonus=config["post_cooldown_threshold_bonus"],
    )
    bot = TradingBot(
        connector=connector, strategies=strategies,
        risk_manager=risk, interval_seconds=config["interval"],
        broadcast_fn=None, telegram=telegram,
    )
    _install_trendconfirm_scan_logger(bot)
    return bot


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
    _install_trendconfirm_scan_logger(bot)
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
    if config.get("strategy_mode") == "trend_confirm":
        old_tf = os.environ.get("CANDLE_TF", "15m")
        if old_tf != "15m":
            logger.warning("TrendConfirm V4 forces CANDLE_TF=15m (ignoring stale %s)", old_tf)
        os.environ["CANDLE_TF"] = "15m"
        config["candle_tf"] = "15m"
    logger.info(
        "=== AI Expert Bot starting [%s] mode=%s symbols=%s ===",
        "PAPER" if config["paper"] else "LIVE",
        config["strategy_mode"], config["symbols"],
    )

    telegram    = _make_telegram(config)
    crypto_bot  = build_crypto_bot(config, telegram)
    forex_bot   = build_forex_bot(config, telegram) if config["forex_enabled"] else None

    # Global 24/5 gate: block NEW positions across the whole symbol universe
    # during the FX weekend. Existing positions continue normal management.
    _install_fx_sleep_entry_gate(crypto_bot, config)

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
        telegram.get_okx_stats_fn = crypto_bot.get_okx_stats
        telegram.get_insights_fn = crypto_bot.get_learning_insights
        telegram.stop_bot_fn     = lambda: _stop_signal.set()
        telegram.start_bot_fn    = lambda: {"message": "Bot is already running"}

    # Auto-optimize SL/TP via backtest (for legacy strategies)
    await _run_backtest(crypto_bot, config, telegram)

    tasks = [
        asyncio.create_task(crypto_bot.start()),
        asyncio.create_task(_fx_sleep_monitor(config)),
    ]
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

    logger.info("All bots stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
