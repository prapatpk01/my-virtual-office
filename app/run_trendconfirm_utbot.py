"""Production runner: Trend Confirm + optional XAU-only UT Bot v2.

This wrapper preserves the existing enhanced Trend Confirm production runner and
adds one independent strategy family:

    UTBotXAU(XAU/USDT:USDT)

Railway toggles:
    ENABLE_TREND_CONFIRM=true|false
    ENABLE_UTBOT_XAU=true|false

Supported modes:
    true / true   -> Trend Confirm + UT Bot XAU
    true / false  -> Trend Confirm only
    false / true  -> UT Bot XAU only
    false / false -> startup error (no strategy enabled)

XAU coexistence policy:
- Trend Confirm and UT Bot may hold XAU simultaneously when their sides are
  opposite (true OKX hedge: one LONG + one SHORT).
- They may NOT stack two independently-owned positions on the same XAU side.
  OKX aggregates same-side hedge positions, which would make Trend Confirm's
  exchange-side SL/TP ownership ambiguous. Blocking same-side stacking keeps
  close/reconcile/accounting deterministic.

UT Bot itself is a faithful port of the supplied Pine strategy:
- source=close, ATR(10), multiplier=1 by default
- confirmed 15m crossover with recursive ATR trailing stop
- no trend/EMA/WT/structure filter
- no fixed SL/TP
- opposite UT cross closes the UT position and reverses on that confirmed bar
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar

# Importing this module installs all existing Trend Confirm production patches,
# including USE_LAYER1_4H, trigger-owner exits, Telegram and trigger-aware charts.
import run_enhanced_dual_bot as trend_runner
import run_bot
import trading.chart_renderer as chart_renderer
from trading.bot import TradingBot
from trading.risk_manager import RiskManager
from trading.telegram_notifier import TelegramNotifier
from trading.utbot_chart_renderer import render_utbot_entry_chart


logger = logging.getLogger("run_trendconfirm_utbot")
UT_SYMBOL = "XAU/USDT:USDT"
UT_FAMILY_PREFIX = "UTBotXAU("
TC_FAMILY_PREFIX = "TrendConfirm("
_UT_ENTRY_CONTEXT: ContextVar[dict | None] = ContextVar(
    "utbot_xau_entry_context",
    default=None,
)

# Capture the enhanced Trend Confirm hooks before this wrapper replaces them.
_TREND_BUILD_CONFIG = run_bot.build_config
_TREND_MAKE_STRATEGIES = run_bot._make_strategies


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"{name} must be true/false, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _strip_side_suffix(strategy_key: str) -> str:
    key = str(strategy_key or "")
    return key[:-2] if key.endswith((":L", ":S")) else key


def _family(strategy_key: str) -> str:
    key = _strip_side_suffix(strategy_key)
    if key.startswith(TC_FAMILY_PREFIX):
        return "trend_confirm"
    if key.startswith(UT_FAMILY_PREFIX):
        return "utbot_xau"
    return "other"


def _position_side(strategy_key: str, position=None) -> str:
    key = str(strategy_key or "")
    if key.endswith(":L"):
        return "long"
    if key.endswith(":S"):
        return "short"
    side = str(getattr(position, "side", "") or "").lower()
    if side in {"buy", "long"}:
        return "long"
    if side in {"sell", "short"}:
        return "short"
    return ""


def _install_combined_risk_policy() -> None:
    """Give TC and UT independent quotas while permitting true XAU hedging."""
    if getattr(RiskManager, "_tc_utbot_risk_policy_installed", False):
        return

    def _can_open(self: RiskManager, symbol: str, strategy: str = ""):
        if self._halted:
            return False, "Trading halted: max drawdown reached"

        in_cd, remaining = self.in_cooldown(symbol)
        if in_cd:
            return False, (
                f"{symbol} cooldown after {self.max_consecutive_sl} consecutive losing closes "
                f"— resumes in {remaining/60:.0f} min"
            )

        candidate_family = _family(strategy)
        candidate_side = _position_side(strategy)
        tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
        ut_enabled = _env_bool("ENABLE_UTBOT_XAU", False)

        if candidate_family == "trend_confirm" and not tc_enabled:
            return False, "Trend Confirm disabled by ENABLE_TREND_CONFIRM=false"
        if candidate_family == "utbot_xau" and not ut_enabled:
            return False, "UT Bot XAU disabled by ENABLE_UTBOT_XAU=false"
        if candidate_family == "utbot_xau" and symbol != UT_SYMBOL:
            return False, f"UT Bot is hard-locked to {UT_SYMBOL}"

        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return False, f"{strategy} already has an open position for {symbol}"

        positions_for_symbol = []
        family_count = 0
        for position_key, position in self._positions.items():
            tracked_strategy = (
                position_key.split("||", 1)[1]
                if "||" in position_key
                else ""
            )
            tracked_symbol = str(getattr(position, "symbol", "") or "")
            tracked_family = _family(tracked_strategy)
            if tracked_family == candidate_family:
                family_count += 1
            if tracked_symbol == symbol:
                positions_for_symbol.append((tracked_strategy, position))

        family_limit = 1
        if candidate_family == "trend_confirm":
            family_limit = max(0, _env_int("TREND_CONFIRM_MAX_POSITIONS", 2))
        elif candidate_family == "utbot_xau":
            family_limit = 1

        if candidate_family in {"trend_confirm", "utbot_xau"} and family_count >= family_limit:
            return False, (
                f"{candidate_family} position quota reached "
                f"({family_count}/{family_limit})"
            )

        # XAU may be owned by both strategies at once ONLY as opposite sides.
        # Same-side OKX hedge positions aggregate into one exchange position,
        # which would destroy deterministic per-strategy SL/TP ownership.
        if symbol == UT_SYMBOL and positions_for_symbol:
            for tracked_strategy, position in positions_for_symbol:
                tracked_family = _family(tracked_strategy)
                if tracked_family not in {"trend_confirm", "utbot_xau"}:
                    continue
                if tracked_family == candidate_family:
                    return False, f"{candidate_family} already owns XAU exposure"
                existing_side = _position_side(tracked_strategy, position)
                if candidate_side not in {"long", "short"}:
                    return False, f"Cannot determine hedge side for {strategy}"
                if existing_side == candidate_side:
                    return False, (
                        f"XAU {candidate_side.upper()} already owned by {tracked_family}; "
                        "same-side TC+UT stacking blocked to preserve separate ownership"
                    )

        per_symbol_limit = 2 if symbol == UT_SYMBOL else max(
            1, _env_int("MAX_POSITIONS_PER_SYMBOL", 2)
        )
        if len(positions_for_symbol) >= per_symbol_limit:
            return False, (
                f"{symbol} per-symbol position limit reached "
                f"({len(positions_for_symbol)}/{per_symbol_limit})"
            )

        if len(self._positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"
        return True, "ok"

    RiskManager.can_open = _can_open
    RiskManager._tc_utbot_risk_policy_installed = True


def _make_combined_strategies(symbols: list, config: dict):
    tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
    ut_enabled = _env_bool("ENABLE_UTBOT_XAU", False)
    if not tc_enabled and not ut_enabled:
        raise RuntimeError(
            "No strategy enabled. Set ENABLE_TREND_CONFIRM=true and/or "
            "ENABLE_UTBOT_XAU=true"
        )

    strategies = []
    if tc_enabled:
        # Crucial: UT may append XAU to config symbols for its own data, but TC
        # must still trade exactly the user's original SYMBOLS universe.
        tc_symbols = list(config.get("trend_confirm_symbols") or symbols)
        strategies.extend(_TREND_MAKE_STRATEGIES(tc_symbols, config))

    if ut_enabled:
        from trading.strategies.utbot_xau_strategy import UTBotXAUStrategy

        strategies.append(
            UTBotXAUStrategy(
                symbol=UT_SYMBOL,
                multiplier=_env_float("UTBOT_MULTIPLIER", 1.0),
                atr_period=_env_int("UTBOT_ATR_PERIOD", 10),
                timeframe="15m",
                use_date_filter=_env_bool("UTBOT_USE_DATE_FILTER", True),
            )
        )
    return strategies


def _build_combined_config() -> dict:
    tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
    ut_enabled = _env_bool("ENABLE_UTBOT_XAU", False)
    if not tc_enabled and not ut_enabled:
        raise RuntimeError(
            "No strategy enabled. Set ENABLE_TREND_CONFIRM=true and/or "
            "ENABLE_UTBOT_XAU=true"
        )

    if tc_enabled:
        config = _TREND_BUILD_CONFIG()
        tc_symbols = list(config.get("symbols") or [])
    else:
        # Enhanced TC intentionally refuses TC=false. UT-only starts from the
        # original baseline config but instantiates no baseline strategies.
        config = trend_runner.run_dual_bot._ORIGINAL_BUILD_CONFIG()
        tc_symbols = []

    config["trend_confirm_symbols"] = tc_symbols

    if tc_enabled:
        symbols = list(tc_symbols)
    else:
        symbols = []
    if ut_enabled and UT_SYMBOL not in symbols:
        symbols.append(UT_SYMBOL)

    config["symbols"] = symbols
    config["candle_tf"] = "15m"
    os.environ["CANDLE_TF"] = "15m"
    config["enable_trend_confirm"] = tc_enabled
    config["enable_utbot_xau"] = ut_enabled
    config["strategy_mode"] = (
        "trend_confirm_plus_utbot_xau"
        if tc_enabled and ut_enabled
        else "trend_confirm"
        if tc_enabled
        else "utbot_xau"
    )

    tc_limit = (
        max(0, _env_int("TREND_CONFIRM_MAX_POSITIONS", 2))
        if tc_enabled
        else 0
    )
    ut_limit = 1 if ut_enabled else 0
    required_slots = max(1, tc_limit + ut_limit)
    requested = max(1, _env_int("MAX_POSITIONS", required_slots))
    # Give every enabled family enough room to use its own quota. Set the TC
    # quota lower if a smaller combined total is desired.
    config["max_positions"] = max(requested, required_slots)

    # True hedge mode is mandatory for TC LONG + UT SHORT (or vice versa).
    config["hedge_mode"] = True
    config["futures"] = True
    return config


def _ut_caption(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    strategy: str,
    paper: bool,
    direction: str = None,
    fee: float = 0.0,
    notional: float = None,
    margin: float = None,
) -> str:
    context = _UT_ENTRY_CONTEXT.get() or {}
    meta = context.get("metadata") or {}
    is_long = direction == "long" or (direction is None and side == "buy")
    emoji = "🟢" if is_long else "🔴"
    dir_label = "LONG" if is_long else "SHORT"
    mode = "📄 PAPER" if paper else "💰 LIVE"
    sep = "—" * 16

    lines = [
        f"{emoji} *OPEN {dir_label}*  `{symbol}`  {mode}",
        sep,
        f"📍 Entry Fill : `{price:,.4f}`",
        "⚡ Entry Trigger : `UT Bot ATR Trailing Stop Cross`",
    ]
    if meta.get("bar_close") is not None:
        lines.append(f"🕯 Signal Close : `{float(meta['bar_close']):,.4f}`")
    if meta.get("tsl_price") is not None:
        lines.append(f"〰️ ATR Trail : `{float(meta['tsl_price']):,.4f}`")
    if meta.get("atr") is not None:
        lines.append(
            f"📏 ATR({int(meta.get('atr_period', 10))}) : "
            f"`{float(meta['atr']):,.4f}` × "
            f"`{float(meta.get('atr_multiplier', 1.0)):g}`"
        )
    lines += [
        "🛑 Fixed SL : `None`",
        "🎯 Fixed TP : `None`",
        "🏁 Exit / Reverse : `Opposite confirmed UT cross`",
        f"💰 Size : `{amount:.6g}`"
        + (f"  (≈`${notional:,.2f}`)" if notional is not None else ""),
    ]
    costs = []
    if margin is not None:
        costs.append(f"Margin `${margin:,.2f}`")
    if fee:
        costs.append(f"Fee `${fee:,.4f}`")
    if costs:
        lines.append("📥 " + "   ".join(costs))
    lines += [
        sep,
        "📊 Strategy : `UT Bot v2 — ATR Trailing Stop`",
        "⏱ Timeframe : `15m confirmed close`",
        f"🔒 Symbol : `XAU only` ({UT_SYMBOL})",
    ]
    return "\n".join(lines)


def _install_ut_telegram_and_chart_patch() -> None:
    """Layer UT-specific UI on top of the installed Trend Confirm UI."""
    if getattr(TradingBot, "_combined_utbot_ui_installed", False):
        return

    original_execute = TradingBot._execute_signal
    original_caption = TelegramNotifier.build_order_caption
    original_notify = TelegramNotifier.notify
    original_renderer = chart_renderer.render_entry_chart

    async def _execute(self, signal, *args, **kwargs):
        meta = getattr(signal, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        is_ut = str(meta.get("strategy", "")).startswith("UTBOT_")
        token = _UT_ENTRY_CONTEXT.set({"is_ut": is_ut, "metadata": dict(meta)})
        try:
            return await original_execute(self, signal, *args, **kwargs)
        finally:
            _UT_ENTRY_CONTEXT.reset(token)

    def _caption(self, symbol, side, amount, price, strategy, paper, **kwargs):
        context = _UT_ENTRY_CONTEXT.get() or {}
        if context.get("is_ut") or _strip_side_suffix(strategy).startswith(UT_FAMILY_PREFIX):
            return _ut_caption(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                strategy=strategy,
                paper=paper,
                direction=kwargs.get("direction"),
                fee=kwargs.get("fee", 0.0),
                notional=kwargs.get("notional"),
                margin=kwargs.get("margin"),
            )
        return original_caption(
            self, symbol, side, amount, price, strategy, paper, **kwargs
        )

    def _renderer(*args, **kwargs):
        context = _UT_ENTRY_CONTEXT.get() or {}
        if context.get("is_ut"):
            return render_utbot_entry_chart(*args, **kwargs)
        return original_renderer(*args, **kwargs)

    def _notify(self, text: str):
        context = _UT_ENTRY_CONTEXT.get() or {}
        if context.get("is_ut") and "Order Executed" in str(text):
            text = (
                f"{text}\n⚡ Entry Trigger: UT Bot ATR Cross\n"
                "🏁 Exit/Reverse: opposite confirmed UT cross"
            )
        return original_notify(self, text)

    TradingBot._execute_signal = _execute
    TelegramNotifier.build_order_caption = _caption
    TelegramNotifier.notify = _notify
    chart_renderer.render_entry_chart = _renderer
    TradingBot._combined_utbot_ui_installed = True


def _install_ut_scan_log() -> None:
    if getattr(TradingBot, "_combined_utbot_scan_log_installed", False):
        return
    original_log_scan = TradingBot._log_scan

    def _log_scan(self, symbol, strategy_name, price, signal):
        if _strip_side_suffix(strategy_name).startswith(UT_FAMILY_PREFIX):
            meta = getattr(signal, "metadata", None) or {}
            logger.info(
                "[SCAN] UTBotXAU %-20s px=%-12.4f sig=%-5s close=%-11s TSL=%-11s ATR=%-9s | %s",
                symbol,
                price,
                getattr(getattr(signal, "type", None), "value", "hold").upper(),
                "--" if meta.get("bar_close") is None else f"{float(meta['bar_close']):.4f}",
                "--" if meta.get("tsl_price") is None else f"{float(meta['tsl_price']):.4f}",
                "--" if meta.get("atr") is None else f"{float(meta['atr']):.4f}",
                getattr(signal, "reason", ""),
            )
            return
        return original_log_scan(self, symbol, strategy_name, price, signal)

    TradingBot._log_scan = _log_scan
    TradingBot._combined_utbot_scan_log_installed = True


_install_combined_risk_policy()
_install_ut_telegram_and_chart_patch()
_install_ut_scan_log()
run_bot._make_strategies = _make_combined_strategies
run_bot.build_config = _build_combined_config


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
