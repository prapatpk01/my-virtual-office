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


def _fmt_px(v) -> str:
    """Price for Telegram: thousands separator, ≤4 decimals, no trailing zeros
    (61770.6 not 61770.6000, and never a raw 61542.753621112424)."""
    try:
        return f"{float(v):,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(v)


def _open_fill_figures(trade_info: dict, result, leverage: float = 10.0) -> tuple:
    """Real post-fill size/order-value/margin/fee when the exchange returned
    them (result = the execution_callback's return dict — has _filled_coins /
    _entry_avg_px / _entry_order_value / _entry_margin / _entry_fee); falls
    back to the bot's pre-fill request otherwise (paper/backtest, or the
    adapter didn't return fill data — margin is then approximated from
    `leverage` since there's no real fill to derive it from). The exchange's
    1-contract minimum can round the requested size up a lot on a small
    balance, so the real fill is what should be shown, not the request."""
    entry = float(trade_info.get("entry") or 0)
    req_size = float(trade_info.get("size") or 0)
    r = result if isinstance(result, dict) else {}
    size = float(r.get("_filled_coins") or 0) or req_size
    avg_px = float(r.get("_entry_avg_px") or 0) or entry
    order_value = float(r.get("_entry_order_value") or 0) or (size * avg_px)
    margin = float(r.get("_entry_margin") or 0) or (order_value / max(leverage, 1))
    fee = float(r.get("_entry_fee") or 0.0)
    return size, order_value, margin, fee


def _format_open_msg(order_type: str, sym: str, trade_info: dict, result=None, leverage: float = 10.0) -> str:
    """OPEN notification: entry / size / SL / full T1-T2 target structure
    (last level = the TP actually attached on the exchange). Falls back to
    the old TP1/TP2 line if the ladder isn't in the payload."""
    entry = trade_info.get("entry")
    size, order_value, margin, fee = _open_fill_figures(trade_info, result, leverage=leverage)
    lines = [
        f"🟢 [Adaptive] {order_type} {sym}",
        f"Entry : {_fmt_px(entry)}",
        f"Size : {size:.4f} (≈${order_value:,.2f})",
        f"Margin : ${margin:,.2f}  Fee : ${fee:,.4f}",
        f"SL : {_fmt_px(trade_info.get('sl'))}",
    ]
    ladder = trade_info.get("ladder") or []
    for i, (label, price, r) in enumerate(ladder):
        tag = f"{label} (TP)" if i == len(ladder) - 1 else label
        lines.append(f"{tag} : {_fmt_px(price)} ({r}R)")
    if not ladder:
        lines.append(f"TP1={trade_info.get('tp1', '?')} TP2={trade_info.get('tp2', '?')}")
    return "\n".join(lines)


def _format_fill_block(fill) -> str:
    """OKX post-fill ground truth (avgPx/fillSz/fee/realizedPnl) + the
    computed Net PnL, for appending to a close/target-hit alert. Empty
    string when `fill` is None (paper/backtest executors, or the OKX call
    didn't return fill data) — callers show only the pre-fill estimate
    they already had in that case.
    Net PnL = Realized Trading PnL (OKX) - Entry Fee Allocation - Exit Fee."""
    if not fill:
        return ""
    return (
        f"\n\nOKX Fill\n"
        f"avgPx : {fill['avg_px']:.4f}\n"
        f"fillSz : {fill['fill_sz']:.6f}\n"
        f"fee : {fill['fee']:.4f}\n"
        f"Realized PnL : {fill['realized_pnl']:+.4f}\n"
        f"Entry fee alloc : -{fill['entry_fee_alloc']:.4f}\n"
        f"Exit fee : -{fill['fee']:.4f}\n"
        f"Net PnL : {fill['net_pnl']:+.4f}"
    )


def _format_close_msg(order_type: str, sym: str, trade_info: dict, fill=None) -> str:
    # Prefer the OKX-verified Net PnL over the bot's own pre-fill estimate
    # (computed from the SL/target trigger price before the market order
    # actually executed) whenever a real fill is available.
    pnl   = float(fill["net_pnl"]) if fill else float(trade_info.get("pnl") or 0)
    emoji = "✅" if pnl > 0 else ("⚪" if pnl == 0 else "❌")
    return (
        f"{emoji} [Adaptive] {order_type} {sym} "
        f"({trade_info.get('reason', '')})\n"
        f"direction={trade_info.get('direction', '?')} "
        f"price={_fmt_px(trade_info.get('price'))}\n"
        f"pnl={pnl:+.2f} size={float(trade_info.get('size') or 0):.4f}"
        f"{_format_fill_block(fill)}"
    )


def _translate_scan_reason(reason: str) -> str:
    """Turn one direction's raw _scan_info string into a plain-English
    pass/fail/pending line for the [View] log block. Same underlying data
    TradingBot._generate_signal already writes — this only relabels it."""
    if not reason:
        return "no signal evaluated yet"
    if reason.startswith("veto:"):
        return f"✗ blocked — {reason[len('veto:'):]}"
    if reason.startswith("strategy_fail"):
        return f"✗ blocked — no strategy scored high enough ({reason})"
    if reason.startswith("total "):
        if "→ SIGNAL" in reason:
            return f"✅ READY TO ENTER — {reason.replace(' → SIGNAL', '')}"
        return f"⏳ passed all vetoes, score not high enough yet — {reason}"
    return reason


def _view_waiting_reason(status: dict, tradeable_regimes) -> str:
    """Why a symbol produced no per-direction verdict this cycle — derived
    from bot status (the bot was blocked before signal generation, so
    _scan_info is empty). Checked in the same order the gates actually fire
    in TradingBot._check_global_gates so the FIRST real blocker is shown."""
    if status.get("position_open"):
        d = status.get("direction") or "?"
        return f"in a {d} position — managing it (no new entry while open)"
    warm = status.get("warmup_remaining_m", 0)
    if warm and warm > 0:
        return f"⏳ startup warmup — {warm}m left before entries are allowed"
    st = status.get("state")
    if st == "COOLDOWN":
        return "cooldown after a loss streak (waits for the cooldown to expire)"
    if st == "BLOCKED":
        return "daily PnL limit hit — blocked until the next trading day"
    if not status.get("session_gate_open", True):
        return (f"session {status.get('session_state')} — new entries paused "
                f"(commodity weekend; crypto is unaffected)")
    regime = status.get("market_state")
    if regime not in tradeable_regimes:
        return (f"regime is {regime}, not tradeable — the bot only opens in "
                f"{'/'.join(sorted(tradeable_regimes))} (waiting for an enabled setup)")
    if regime == "Trend" and status.get("regime_bias") == "NEUTRAL":
        return (f"regime is Trend but the 4H macro is NEUTRAL "
                f"({status.get('regime_score', 50):.0f}/100) — no higher-timeframe "
                f"direction behind it yet")
    return "evaluating — no direction scored this cycle yet"


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
        "leverage":        _env_int("LEVERAGE", 20),
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
        "adaptive_risk_pct": _env_float("ADAPTIVE_RISK_PCT", 0.05),
        "adaptive_daily_loss": _env_float("ADAPTIVE_DAILY_LOSS_PCT", -3.0),
        "adaptive_daily_profit": _env_float("ADAPTIVE_DAILY_PROFIT_PCT", 8.0),
        "adaptive_cooldown_min": _env_int("ADAPTIVE_COOLDOWN_MIN", 30),
        "adaptive_max_loss_streak": _env_int("ADAPTIVE_MAX_LOSS_STREAK", 3),
        "adaptive_max_positions": _env_int("ADAPTIVE_MAX_POSITIONS", 2),
        # TP geometry dial (None = use bot class defaults 0.7 / 1.5). Lower
        # ADAPTIVE_TP1_R raises win-rate at the cost of average win size.
        "adaptive_tp1_r": (_env_float("ADAPTIVE_TP1_R", 0.70) or None),
        "adaptive_tp2_r": (_env_float("ADAPTIVE_TP2_R", 1.20) or None),
        # T1's SL-move locks in this many R of profit instead of a flat
        # scratch at exactly entry (0R) — e.g. 0.15 means the worst case
        # after T1 is +0.15R, not breakeven.
        "adaptive_breakeven_lock_r": _env_float("ADAPTIVE_BREAKEVEN_LOCK_R", 0.15),
        # Fraction of the position closed at T1 (rest runs to T2). Was
        # documented as ADAPTIVE_TP1_CLOSE_PCT in .env.example but never
        # actually wired here — the live bot always used the class default
        # (75%) regardless of that env var. Now genuinely configurable;
        # default lowered to 60% per request (leaves a bigger 40% runner).
        "adaptive_tp1_close_pct": _env_float("ADAPTIVE_TP1_CLOSE_PCT", 0.50),
        # Fake-signal chop-zone filter (None = default 0.8). Higher = stricter,
        # higher WR, fewer trades (~1.2 pushes WR toward 56%).
        "adaptive_min_ema_dist_atr": (_env_float("ADAPTIVE_MIN_EMA_DIST_ATR", 0.0) or None),
        # Whipsaw guard: minutes since the last OPEN on a symbol before a new
        # entry is allowed there (0 = disabled).
        "adaptive_entry_spacing_min": _env_int("ADAPTIVE_ENTRY_SPACING_MIN", 60),
        # Startup warmup — no new entries until this many minutes after
        # process start (indicators need a few closed bars to stabilize
        # after a fresh restart). Was hardcoded 45; lowered to 10 by request.
        "adaptive_warmup_min": _env_int("ADAPTIVE_WARMUP_MIN", 10),
        # /stats now sources trade count/win-rate/PnL straight from OKX's own
        # post-fill positions-history (realizedPnl already nets OKX's trading
        # + funding fee — no local re-derivation, so the numbers always match
        # OKX exactly). Only trades closed on/after this date are counted.
        "adaptive_stats_since": os.environ.get("ADAPTIVE_STATS_SINCE_DATE", "2026-07-16"),
        # [SIZING MODE] Back to classic risk-%-of-balance (live default,
        # ADAPTIVE_RISK_PCT above = 5%): position size is derived from the
        # SL distance so that a full stop-out loses exactly risk_pct of
        # current balance, regardless of leverage/notional. This is the
        # ORIGINAL sizing mode — active whenever both of the following are
        # 0 (the default). Set ADAPTIVE_MARGIN_USDT>0 for fixed-$ sizing
        # instead, or both ADAPTIVE_MARGIN_PCT_MIN/MAX>0 for confidence-
        # weighted %-of-balance sizing (Level 1 Adaptive Risk).
        "adaptive_margin_usdt": _env_float("ADAPTIVE_MARGIN_USDT", 0.0),
        "adaptive_margin_pct_min": _env_float("ADAPTIVE_MARGIN_PCT_MIN", 0.0),
        "adaptive_margin_pct_max": _env_float("ADAPTIVE_MARGIN_PCT_MAX", 0.0),
        # [MTF-CONFLUENCE] "adaptive" (default) = V9.2 L1/L2/L3/StrategyScorer
        # pipeline. "mtf_confluence" = deterministic 4H+1H trend-alignment +
        # 15m 3-signal confluence entry engine (see mtf_confluence_engine.py)
        # — exit management (T1/T2/SL/post-T1 protection) is identical
        # either way, only entry direction+timing changes.
        "adaptive_entry_engine": os.environ.get("ADAPTIVE_ENTRY_ENGINE", "adaptive").strip().lower(),
        # [EARLY TREND] fast dual-TF (4H+1H) HMA/MACD/ROC lean folded into L1's
        # score when confirmed. Backtested MIXED on real data (helps OOS,
        # hurts in-sample, net ~breakeven across both) — off by default,
        # not currently recommended.
        "adaptive_early_trend": _env_bool("ADAPTIVE_EARLY_TREND", False),
        # [FAST MACRO EMA] override L1's EMA20/50 cross component with a
        # faster pair. 12/26 (the classic MACD periods) is now the DEFAULT —
        # backtested on real Jan-Jun 2026 data at +$209 combined vs EMA20/50,
        # improving BOTH the in-sample and out-of-sample splits independently.
        # Set ADAPTIVE_MACRO_EMA_FAST=0 (or leave *_SLOW unset) to restore the
        # old EMA20/50-only behavior.
        "adaptive_macro_ema_fast": (_env_int("ADAPTIVE_MACRO_EMA_FAST", 12) or None),
        "adaptive_macro_ema_slow": (_env_int("ADAPTIVE_MACRO_EMA_SLOW", 26) or None),
        # [PULLBACK FILTER] Max 15m ADX for a Trend entry — the single biggest
        # entry-frequency knob (blocked 41-55% of Trend-regime checks at the
        # old 22). Raised to 26 to trade more often; higher = more trades but
        # risks chasing extended legs (backtest: ADX>30 ran only ~47% WR).
        "adaptive_max_15m_adx_trend": _env_float("ADAPTIVE_MAX_15M_ADX_TREND", 26.0),
        # [RANGE ADD-ON] Dedicated half-risk engine. These settings affect
        # Range only; Trend and Breakout keep their existing risk and gates.
        "adaptive_range_risk_multiplier": _env_float(
            "ADAPTIVE_RANGE_RISK_MULTIPLIER", 0.50),
        "adaptive_range_cooldown_min": _env_int(
            "ADAPTIVE_RANGE_COOLDOWN_MIN", 90),
        "adaptive_max_range_positions": _env_int(
            "ADAPTIVE_MAX_RANGE_POSITIONS", 1),
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
# Shared ccxt-async shutdown helper
# ---------------------------------------------------------------------------

async def _force_close_ccxt_async(exchange, label: str = "") -> None:
    """
    Belt-and-suspenders shutdown for a ccxt.async_support exchange instance.
    ccxt's own `.close()` doesn't reliably prevent aiohttp's "Unclosed
    connector" warning on process exit (seen on every Railway redeploy) —
    force-close the underlying aiohttp session/connector directly so their
    __del__ never fires against a live socket during interpreter shutdown.
    Shared by every ccxt-async client this process creates (the market-data
    connector and each exchange adapter's own async client) so a fix here
    covers all of them, not just whichever one was noticed first.
    """
    if exchange is None:
        return
    try:
        await exchange.close()
    except Exception as e:
        logger.warning("[Shutdown] %s ccxt close() error (non-fatal): %s", label, e)
    try:
        session = getattr(exchange, "session", None)
        if session is not None:
            connector = getattr(session, "_connector", None)
            if connector is not None and not getattr(connector, "_closed", True):
                await connector.close()
            exchange.session = None
        exchange.socks_proxy_sessions = None
        tcp = getattr(exchange, "tcp_connector", None)
        if tcp is not None:
            if not getattr(tcp, "_closed", True):
                await tcp.close()
            exchange.tcp_connector = None
    except Exception:
        pass


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
    from trading.adaptive_trading_bot import (
        TradingBot as AdaptiveBot, ExpectancyEngine, _TRADEABLE_REGIMES)
    from trading.indicator_engine import IndicatorEngine
    from trading.chart_renderer import render_entry_chart

    symbols = cfg["symbols"]
    logger.info("=== ADAPTIVE MODE: %d symbols ===", len(symbols))

    # [SHARED-LEARNING] One ExpectancyEngine pooled across every symbol bot —
    # unlike the sequential-per-symbol backtest, live symbols advance in real
    # wall-clock lockstep, so pooling here has no look-ahead concern at all;
    # it's strictly an improvement over each symbol learning in isolation.
    shared_expectancy = ExpectancyEngine()

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

    # [CHART ALERTS] latest fetched 15m candles per symbol — the OPEN-alert
    # chart renderer reads from here at execution-callback time (the callback
    # itself only receives the order payload, not market data).
    last_c15m_cache: dict = {}

    def _send_open_chart_alert(sym: str, order_type: str, trade_info: dict, result=None) -> bool:
        """Render an entry chart + HTML caption and send as a Telegram photo.
        Returns True when the photo was queued (send_photo itself falls back
        to a text message if the upload fails); False → caller sends the
        plain-text alert instead."""
        candles = last_c15m_cache.get(sym) or []
        if len(candles) < 30:
            return False
        direction = "LONG" if "LONG" in order_type else "SHORT"
        entry = float(trade_info.get("entry") or 0)
        sl    = float(trade_info.get("sl") or 0)
        tp1   = float(trade_info.get("tp1") or 0)
        tp2   = float(trade_info.get("tp2") or 0)
        size, order_value, margin, fee = _open_fill_figures(
            trade_info, result, leverage=cfg.get("leverage", 10))

        # Signal context lives on the bot (current_trade is set before the
        # OPEN order is sent), not in the order payload.
        bot_ref = bots.get(sym, (None, None))[0] if sym in bots else None
        strategy = regime = l1_level = ""
        l1_score = 0.0
        close_pct = 0.75
        if bot_ref is not None:
            t = getattr(bot_ref, "current_trade", {}) or {}
            strategy  = t.get("strategy", "") or ""
            regime    = t.get("e_state", "") or getattr(bot_ref, "current_market_state", "")
            l1_level  = getattr(bot_ref, "current_regime_bias", "")
            l1_score  = float(getattr(bot_ref, "regime_score", 0.0) or 0.0)
            close_pct = float(getattr(bot_ref, "tp1_close_pct", 0.75) or 0.75)

        chart_path = render_entry_chart(
            candles, sym, direction, entry, sl, tp1, tp2,
            strategy=strategy, regime=regime,
        )
        if not chart_path:
            return False

        emoji    = "🟢" if direction == "LONG" else "🔴"
        sl_pct   = abs(sl - entry) / entry * 100 if entry else 0.0
        lines = [
            f"{emoji} <b>OPEN {direction}</b>  {sym}",
            "━━━━━━━━━━━━━━━",
            f"📍 Entry : <code>{_fmt_px(entry)}</code>",
            f"🛑 SL : <code>{_fmt_px(sl)}</code> (-{sl_pct:.2f}%)",
            f"🎯 T1 : <code>{_fmt_px(tp1)}</code> (0.5R · close {close_pct:.0%} · SL→BE)",
            f"🏁 T2 : <code>{_fmt_px(tp2)}</code> (1.0R · close rest)",
            f"💰 Size : {size:.4f} (≈${order_value:,.2f})",
            f"📥 Margin : ${margin:,.2f}  Fee : ${fee:,.4f}",
        ]
        if strategy or regime:
            lines.append("━━━━━━━━━━━━━━━")
            ctx = []
            if strategy:
                ctx.append(f"Strategy: <b>{strategy}</b>")
            if regime:
                ctx.append(f"Regime: <b>{regime}</b>")
            lines.append("📊 " + " | ".join(ctx))
        if l1_level:
            lines.append(f"🧭 4H Macro: <b>{l1_level}</b> ({l1_score:.0f}/100)")
        telegram.send_photo(chart_path, "\n".join(lines), parse_mode="HTML")
        return True

    # FIX-#6: use BOT_STATE_DIR env var so state survives container restarts
    import os as _os
    _state_dir = _os.environ.get("BOT_STATE_DIR", "/tmp")
    _os.makedirs(_state_dir, exist_ok=True)

    # Approximate, manually-maintained classification (this codebase has no
    # richer per-symbol asset-class metadata) — shared by min-SL-floor sizing
    # below and the session-control crypto-bypass check further down.
    _commodity_symbols = {"XAU", "XAG", "CL"}
    # [MIN-SL FLOOR] Originally split crypto vs commodity on the theory that
    # gold/silver/oil rarely move 2% intra-trade — but 0.8% for XAU/XAG
    # backtested WORSE (-$470 combined on real data) than the old uniform 2%:
    # every XAU loss under 0.8% was a full -1R stop, consistent with 0.8%
    # being tighter than gold's actual 15m ATR often calls for, getting
    # stopped by normal noise before the trade develops. Both buckets now use
    # the same 1.2% (the value that DID backtest better, +$343 on crypto) —
    # see TradingBot.min_sl_pct.
    _min_sl_pct_crypto     = _env_float("ADAPTIVE_MIN_SL_PCT_CRYPTO", 0.012)
    _min_sl_pct_commodity  = _env_float("ADAPTIVE_MIN_SL_PCT_COMMODITY", 0.012)

    for sym in symbols:
        safe_sym = sym.replace("/", "_").replace(":", "_")
        state_file = _os.path.join(_state_dir, f"adaptive_{safe_sym}.json")

        def _make_callback(s, t):
            def cb(order_type, trade_info):
                result = t.execute(order_type, {**trade_info, "symbol": s})
                # AMEND_SL's user-facing notification is already sent by
                # _send_target_alerts (the exact "Target N Hit / SL moved"
                # format) — the generic OPEN/CLOSE-shaped message below would
                # just be misleading noise for this order type.
                if telegram and order_type != "AMEND_SL":
                    try:
                        if "OPEN" in order_type:
                            # [CHART ALERTS] photo (candles + EMA + entry/SL/
                            # T1/T2 levels) with an HTML caption; plain-text
                            # fallback when rendering isn't possible.
                            sent = False
                            try:
                                sent = _send_open_chart_alert(s, order_type, trade_info, result=result)
                            except Exception as ce:
                                logger.warning("[Chart][%s] alert failed (%s) — text fallback", s, ce)
                            if not sent:
                                telegram.send(_format_open_msg(
                                    order_type, s, trade_info, result=result,
                                    leverage=cfg.get("leverage", 10)))
                        else:
                            # [OKX FILL] current_trade still holds this
                            # close's "last_fill" (set in _close_position,
                            # synchronously just before this callback runs) —
                            # None on paper/backtest executors, or if OKX
                            # didn't return fill data for some reason.
                            bot_ref = bots.get(s, (None, None))[0]
                            fill = (getattr(bot_ref, "current_trade", {}) or {}).get("last_fill") \
                                   if bot_ref else None
                            telegram.send(_format_close_msg(order_type, s, trade_info, fill=fill))
                    except Exception:
                        pass
                return result
            return cb

        _base_sym = sym.split("/")[0].upper()
        _min_sl_pct = _min_sl_pct_commodity if _base_sym in _commodity_symbols else _min_sl_pct_crypto
        # [PER-SYMBOL SIZING] Fixed-$ margin defaults to ADAPTIVE_MARGIN_USDT
        # for every symbol (0 = disabled, the current default — see the
        # [SIZING MODE] comment above; risk_pct-of-balance is active
        # instead). ADAPTIVE_MARGIN_USDT_<BASE_SYM> still overrides just one
        # symbol if fixed-$ sizing is ever turned back on.
        _margin_usdt = _env_float(f"ADAPTIVE_MARGIN_USDT_{_base_sym}",
                                  cfg.get("adaptive_margin_usdt", 0.0))

        # Build constructor kwargs once, then keep only arguments supported by
        # the strategy class actually imported in this deployment.  This makes
        # run_bot forward/backward compatible with strategy-file revisions and
        # prevents Railway restart loops from a single keyword mismatch.
        _bot_kwargs = {
            "account_balance": cfg["adaptive_balance"],
            "min_sl_pct": _min_sl_pct,
            "tp1_close_pct": cfg.get("adaptive_tp1_close_pct", 0.50),
            "base_risk_pct": cfg["adaptive_risk_pct"],
            "daily_loss_limit_pct": cfg["adaptive_daily_loss"],
            "daily_profit_limit_pct": cfg["adaptive_daily_profit"],
            "cooldown_minutes": cfg["adaptive_cooldown_min"],
            "max_loss_streak": cfg["adaptive_max_loss_streak"],
            "startup_warmup_minutes": cfg.get("adaptive_warmup_min", 10),
            "state_file": state_file,
            "execution_callback": _make_callback(sym, okx),
            "enable_swing_reversal": cfg["strategies"].get("swing_reversal_pro", True),
            "enable_mean_reversion": cfg["strategies"].get("mean_reversion", False),
            "tp1_r": cfg.get("adaptive_tp1_r"),
            "tp2_r": cfg.get("adaptive_tp2_r"),
            "breakeven_lock_r": cfg.get("adaptive_breakeven_lock_r"),
            "min_ema_dist_atr": cfg.get("adaptive_min_ema_dist_atr"),
            "entry_spacing_min": cfg.get("adaptive_entry_spacing_min", 60),
            "margin_pct_min": cfg.get("adaptive_margin_pct_min", 0.08),
            "margin_pct_max": cfg.get("adaptive_margin_pct_max", 0.15),
            "margin_usdt": _margin_usdt,
            "sizing_leverage": cfg.get("leverage", 10),
            "expectancy_engine": shared_expectancy,
            "entry_engine": cfg.get("adaptive_entry_engine", "adaptive"),
            "enable_early_trend": cfg.get("adaptive_early_trend", False),
            "macro_ema_fast": cfg.get("adaptive_macro_ema_fast"),
            "macro_ema_slow": cfg.get("adaptive_macro_ema_slow"),
            "max_15m_adx_trend": cfg.get("adaptive_max_15m_adx_trend"),
            "range_risk_multiplier": cfg.get("adaptive_range_risk_multiplier", 0.50),
            "range_cooldown_minutes": cfg.get("adaptive_range_cooldown_min", 90),
        }
        import inspect as _inspect
        _sig = _inspect.signature(AdaptiveBot.__init__)
        _accepts_var_kw = any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
        )
        if not _accepts_var_kw:
            _supported = set(_sig.parameters) - {"self"}
            _dropped = sorted(set(_bot_kwargs) - _supported)
            if _dropped:
                logger.warning(
                    "[Adaptive][%s] strategy constructor does not support %s; ignoring them",
                    sym, ", ".join(_dropped),
                )
            _bot_kwargs = {k: v for k, v in _bot_kwargs.items() if k in _supported}
        bot = AdaptiveBot(**_bot_kwargs)
        bot.load_state(state_file)
        bot.reconcile_with_exchange(sym, okx)
        # [MIN-LOT TP1] tell the bot the smallest closable size (1 contract in
        # coins) so an unsplittable TP1 becomes a breakeven-move instead of
        # accidentally closing the whole position.
        try:
            if hasattr(okx, "_get_ct_val"):
                bot.min_close_size = float(okx._get_ct_val(sym))
                logger.info("[Adaptive][%s] min close size = %.6f (1 contract)",
                            sym, bot.min_close_size)
        except Exception as e:
            logger.warning("[Adaptive][%s] could not fetch contract size: %s", sym, e)
        bots[sym] = (bot, state_file)
        last_bar_ts[sym] = 0

    if telegram:
        # Wire bots dict so /stats and /log commands work
        telegram.bots_dict = bots
        # Wire the shared exchange adapter so /stats can pull real post-fill
        # trade history straight from OKX (see fetch_closed_positions_history)
        # instead of relying only on each bot's local trade_journal.
        telegram.exchange = okx
        try:
            import datetime as _dt_stats
            _since_str = cfg.get("adaptive_stats_since", "2026-07-16")
            _since_dt = _dt_stats.datetime.strptime(_since_str, "%Y-%m-%d").replace(
                tzinfo=_dt_stats.timezone.utc)
            telegram.stats_since_ms = int(_since_dt.timestamp() * 1000)
        except Exception as e:
            logger.warning("[Stats] bad ADAPTIVE_STATS_SINCE_DATE (%s): %s — /stats will show all history", cfg.get("adaptive_stats_since"), e)
            telegram.stats_since_ms = 0
        # Wire /stop command so it reaches the adaptive runner's stop_event
        telegram.stop_bot_fn = lambda: stop_event.set()
        # FIX-#8: start Telegram command polling in adaptive mode
        loop = asyncio.get_event_loop()
        telegram.start_polling(loop)
        telegram.send(
            f"Adaptive Bot Started\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Mode: {'PAPER' if cfg['paper'] else 'LIVE'}\n"
            f"Regimes: {', '.join(sorted(_TRADEABLE_REGIMES))}\n"
            f"Range: risk×{cfg.get('adaptive_range_risk_multiplier', 0.50):.2f}, "
            f"max positions={max(0, cfg.get('adaptive_max_range_positions', 1))}, "
            f"cooldown={cfg.get('adaptive_range_cooldown_min', 90)}m\n"
            f"Warmup: {cfg.get('adaptive_warmup_min', 10)}m — "
            f"no new entries until indicators stabilize"
        )

    import time as _time
    import datetime as _dt

    loop = asyncio.get_event_loop()
    max_pos = cfg.get("adaptive_max_positions", 2)
    max_range_pos = max(0, cfg.get("adaptive_max_range_positions", 1))

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

    # [SESSION CONTROL] Extended commodity-market session gate — see
    # session_engine.py. ONE shared engine (the weekly session is a single
    # global fact, not per-symbol); every bot's session_gate_open is set from
    # its output right before that bot's on_tick call. Position management,
    # protective orders, Telegram alerts, and exchange heartbeat are never
    # touched by this — only the SCANNING->FILTERING (new entry) transition.
    from trading.session_engine import TradingSessionEngine
    session_engine = TradingSessionEngine(
        reference_market=os.environ.get("REFERENCE_MARKET", "XAU"),
        market_tz_name=os.environ.get("MARKET_SESSION_TIMEZONE", "America/New_York"),
        open_weekday=os.environ.get("WEEKLY_OPEN_WEEKDAY", "SUNDAY"),
        open_hour=_env_int("WEEKLY_OPEN_HOUR", 18),
        close_weekday=os.environ.get("WEEKLY_CLOSE_WEEKDAY", "FRIDAY"),
        close_hour=_env_int("WEEKLY_CLOSE_HOUR", 17),
        pre_open_extension_hours=_env_float("PRE_OPEN_EXTENSION_HOURS", 3.0),
        post_close_extension_hours=_env_float("POST_CLOSE_EXTENSION_HOURS", 3.0),
    )
    # Default FALSE: the weekend/session gate applies ONLY to commodities
    # (XAU/XAG/CL); crypto trades 24/7 as usual. Set
    # FOLLOW_REFERENCE_SESSION_FOR_CRYPTO=true to also pause crypto on the
    # commodity weekend.
    _follow_session_for_crypto = _env_bool("FOLLOW_REFERENCE_SESSION_FOR_CRYPTO", False)
    # _commodity_symbols defined earlier (shared with the min-SL-floor split
    # above) — with the default above, this is the ONLY set the session gate
    # touches; crypto's session_gate_open stays True regardless of session.
    last_session_state = None
    last_allow_new_positions = None

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

    def _send_target_alerts(sym: str, bot):
        """Pop and forward any queued target-hit alerts (T1/T2) to Telegram,
        in the exact format requested: symbol / "TargetN Hit" / price /
        partial-close % + SL move to breakeven (or "Take Profit" + close for
        the final level), plus the OKX-verified fill/PnL block when available."""
        for alert in bot.pop_target_alerts():
            fill_block = _format_fill_block(alert.get("fill"))
            if alert["final"]:
                msg = (
                    f"🎯 {sym}\n"
                    f"Target {alert['label'][1:]} Hit — Take Profit\n\n"
                    f"Price : {alert['price']:.4f}\n"
                    f"Position closed"
                    f"{fill_block}"
                )
            else:
                # Arrow reflects the actual SL move direction: tighter-for-LONG
                # moves the number UP, tighter-for-SHORT moves it DOWN — a
                # hardcoded "↓" would show the wrong direction for LONG trades.
                arrow = "↑" if alert["new_sl"] > alert["old_sl"] else "↓"
                close_pct = alert.get("close_pct") or 0.0
                close_line = f"Closed {close_pct:.0%} of position\n\n" if close_pct > 0 else ""
                msg = (
                    f"✅ {sym}\n"
                    f"Target {alert['label'][1:]} Hit\n\n"
                    f"Price : {alert['price']:.4f}\n\n"
                    f"{close_line}"
                    f"SL moved\n"
                    f"{alert['old_sl']:.4f}\n"
                    f"{arrow}\n"
                    f"{alert['new_sl']:.4f}"
                    f"{fill_block}"
                )
            logger.info("[Target][%s] %s", sym, alert["label"])
            if telegram:
                try:
                    telegram.send(msg)
                except Exception:
                    pass

    # Note: periodic Telegram stats removed — too noisy.
    # Use /stats command to check on demand. Railway logs show state every tick.
    # 5-minute health logs are written to Railway log automatically below.

    while not stop_event.is_set():
        # [SESSION CONTROL] One evaluation per loop pass, shared by every
        # symbol this iteration — the weekly session is a single global fact.
        # check_price_protection/on_tick consult session_gate_open every
        # tick regardless of whether anything gets logged/sent here.
        #
        # Telegram is reserved for the two moments the user actually cares
        # about — allow_new_positions flipping True->False (entering
        # SLEEP_MODE for the weekend) or False->True (trading resumes,
        # PRE_OPEN_EXTENSION starting) — not every internal state-label
        # change (PRE_OPEN_EXTENSION->ACTIVE and ACTIVE->POST_CLOSE_EXTENSION
        # both keep allow_new_positions=True the whole time, so neither is a
        # user-visible change). The Railway log line is more permissive
        # (every state-label transition) since that's for debugging, not
        # a push notification.
        session = session_engine.evaluate()
        if session.allow_new_positions != last_allow_new_positions:
            if last_allow_new_positions is not None and telegram:
                # Scope note: with crypto exempt (the default), this gate
                # only pauses commodities — say so, otherwise the message's
                # "New Positions: DISABLED" reads as a full stop.
                if _follow_session_for_crypto:
                    scope = "Applies to: ALL symbols (crypto + commodities)"
                else:
                    _commodity_syms_here = sorted(
                        s.split("/")[0].upper() for s in symbols
                        if s.split("/")[0].upper() in _commodity_symbols)
                    scope = ("Applies to: COMMODITIES ONLY "
                             f"({', '.join(_commodity_syms_here) or 'none configured'}) "
                             "— crypto keeps trading 24/7")
                telegram.send(session.view_log_message + "\n" + scope)
            last_allow_new_positions = session.allow_new_positions
        if session.state.value != last_session_state:
            last_session_state = session.state.value
            logger.info(session.view_log_message)

        for sym in symbols:
            try:
                # Fetch candles for all 3 timeframes
                c15m = await connector.fetch_ohlcv(sym, "15m", 300)
                c1h  = await connector.fetch_ohlcv(sym, "1h",  200)
                c4h  = await connector.fetch_ohlcv(sym, "4h",  200)

                if not c15m or not c1h or not c4h:
                    logger.warning("[Adaptive][%s] Empty candles — skipping", sym)
                    continue

                # [CHART ALERTS] keep the freshest candles available for the
                # OPEN-alert renderer (fires inside on_tick below).
                last_c15m_cache[sym] = c15m

                latest_ts = c15m[-1].timestamp if c15m else 0
                is_new_bar = latest_ts > last_bar_ts[sym]
                bot, state_file = bots[sym]

                # [RANGE PORTFOLIO CAP] At most one Range position by default.
                # This gate is read only by the dedicated Range engine; Trend
                # and Breakout are never blocked by it and retain priority for
                # the remaining global position slots.
                range_open_count = sum(
                    1 for b, _ in bots.values()
                    if b.position_open and
                    (getattr(b, "current_trade", {}) or {}).get("e_state") == "Range"
                )
                bot.range_entry_allowed = (
                    max_range_pos > 0 and range_open_count < max_range_pos
                )

                # [SESSION CONTROL] gate only new entries (never position
                # management — _check_global_gates is the only reader of
                # session_gate_open). Crypto symbols follow the same
                # commodity-market schedule by default; set
                # FOLLOW_REFERENCE_SESSION_FOR_CRYPTO=false to exempt them.
                base_sym = sym.split("/")[0].upper()
                applies = _follow_session_for_crypto or base_sym in _commodity_symbols
                bot.session_gate_open = session.allow_new_positions if applies else True
                bot.session_state = session.state.value if applies else "ACTIVE"

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
                        elif getattr(bot, "margin_usdt", 0) > 0:
                            # [SHARED-MARGIN GUARD] Each bot's account_balance
                            # is its own private copy, resynced only every
                            # ~5 min — it does NOT reflect margin already
                            # committed by a SIBLING symbol's position opened
                            # moments ago on the same real account. Reserve
                            # margin across all currently-open positions
                            # before letting another symbol attempt one.
                            reserved = sum(
                                getattr(b, "margin_usdt", 0)
                                for b, _ in bots.values() if b.position_open
                            )
                            if reserved + bot.margin_usdt > bot.account_balance:
                                logger.debug(
                                    "[Adaptive][%s] SKIP — reserved margin "
                                    "$%.2f + $%.2f > balance $%.2f",
                                    sym, reserved, bot.margin_usdt, bot.account_balance,
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
                                # [MTF-CONFLUENCE] needs whole OHLCV series
                                # (current+previous bar) for HMA/ROC/MACD
                                # cross detection — no-op when entry_engine
                                # is "adaptive" (default).
                                raw_candles={"15m": c15m, "1h": c1h, "4h": c4h},
                            )
                        )

                        # Log state on every new bar
                        status = bot.get_status()
                        logger.info(
                            "[Adaptive][%s] state=%s pos=%s market=%s regime=%.0f session=%s",
                            sym, status["state"], status["position_open"],
                            status["market_state"], status["regime_score"],
                            status["session_state"],
                        )

                        # [TARGET LADDER] forward T1/T2 hit alerts to Telegram
                        _send_target_alerts(sym, bot)

                        # [LESSON] forward loss-cluster post-mortem to Telegram
                        lesson = bot.pop_lesson_alert()
                        if lesson:
                            logger.warning("[Lesson][%s]\n%s", sym, lesson)
                            if telegram:
                                try:
                                    telegram.send(f"[{sym}]\n{lesson}")
                                except Exception:
                                    pass

                        # [LEVEL 3] forward adaptive-strategy activation/expiry
                        for alert in bot.pop_strategy_alerts():
                            logger.warning("[Strategy][%s] %s", sym, alert)
                            if telegram:
                                try:
                                    telegram.send(f"[{sym}]\n{alert}")
                                except Exception:
                                    pass

                else:
                    # [INTRABAR PROTECT] Between bar closes, check price-level
                    # exits (SL/TP1/TP2) against the forming candle's latest
                    # price every poll (~INTERVAL_SECONDS), instead of leaving
                    # bot-side protection blind for up to 15 minutes. Full
                    # indicator-based management still runs on bar close.
                    if bot.position_open and c15m:
                        live_px = float(c15m[-1].close)
                        try:
                            action = await loop.run_in_executor(
                                None, bot.check_price_protection, live_px)
                            if action:
                                logger.info("[Protect][%s] intrabar %s (px=%.4f)",
                                            sym, action, live_px)
                            _send_target_alerts(sym, bot)
                        except Exception as e:
                            logger.warning("[Protect][%s] intrabar check failed: %s",
                                           sym, e)

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
                # [CRASH-FIX] This block previously had no try/except — a
                # stats-schema mismatch (e.g. get_filter_stats() key rename)
                # raised uncaught here, killing the whole _run_adaptive
                # coroutine (and therefore the process) on the next 5-min
                # tick after any symbol had a signal evaluation, which is
                # exactly when positions tend to open/close. Every other
                # periodic block in this loop (balance sync, reconcile,
                # scan log) already guards itself the same way.
                try:
                    fs = bot.get_filter_stats()
                    if fs.get("checked", 0) == 0:
                        continue
                    logger.info(
                        "[FilterStats][%s] checked=%d passed=%d | "
                        "veto_chop=%d veto_climax=%d veto_1h_chop=%d "
                        "veto_chase=%d veto_macro=%d "
                        "strategy_fail=%d threshold_fail=%d",
                        sym, fs.get("checked", 0), fs.get("passed", 0),
                        fs.get("veto_chop", 0), fs.get("veto_climax", 0),
                        fs.get("veto_1h_chop", 0), fs.get("veto_chase", 0),
                        fs.get("veto_macro", 0),
                        fs.get("strategy_fail", 0), fs.get("threshold_fail", 0),
                    )
                except Exception as e:
                    logger.warning("[FilterStats][%s] log failed (non-fatal): %s", sym, e)

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
                    # [VIEW LOG] ONE single-line entry per symbol per scan
                    # cycle. Kept to a single line ON PURPOSE: Railway's log
                    # viewer splits any "\n" into separate timestamped rows,
                    # so a multi-line block showed up as 5 disconnected
                    # entries. The trend/macro/session header plus the verdict
                    # (per-direction score when the bot reached scoring, else
                    # _view_waiting_reason's derived gate) all go on one line
                    # joined by " · " so it stays a single log record.
                    head = (f"[View][{sym}] regime={status['market_state']} "
                            f"macro={status['regime_bias']}({status['regime_score']:.0f}) "
                            f"session={status['session_state']}")
                    scan = status.get("scan_info") or {}
                    scored = [d for d in ("LONG", "SHORT") if d in scan]
                    if scored and not status["position_open"] \
                            and status["state"] not in ("COOLDOWN", "BLOCKED"):
                        verdict = " · ".join(
                            f"{d}: {_translate_scan_reason(scan[d])}" for d in scored)
                    else:
                        verdict = f"waiting: {_view_waiting_reason(status, _TRADEABLE_REGIMES)}"
                    logger.info("%s · %s", head, verdict)
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
    # `connector`, so okx's own async client needs the same force-close
    # treatment (see _force_close_ccxt_async) or its aiohttp connector logs
    # "Unclosed connector" on every shutdown/redeploy.
    await _force_close_ccxt_async(getattr(okx, "_aex", None), label="okx-adapter")


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
        # This runs whether the bot exits cleanly OR crashes on startup, so
        # __del__ never fires "Unclosed connector"/"requires .close()"
        # warnings against a live socket during interpreter shutdown.
        await _force_close_ccxt_async(getattr(connector, "_exchange", None), label="market-data")
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
