"""Production runner: Trend Confirm + Adaptive Multi-Trigger + XAU UT Bot.

Railway toggles:
    ENABLE_TREND_CONFIRM=true|false
    ENABLE_ADAPTIVE_MULTI_TRIGGER=true|false
    ENABLE_UTBOT_XAU=true|false

The three strategy families may be enabled independently. XAU is always
reserved for UTBotXAU. Trend Confirm and Adaptive Multi-Trigger use the normal
SYMBOLS universe with XAU removed.

Safety/ownership rules:
- XAU: UT Bot only.
- Trend Confirm and Adaptive may scan the same non-XAU symbols, but only ONE of
  those two families may own an active position on a symbol at a time. This
  prevents duplicate entries when both engines identify the same setup.
- Each family has its own quota while MAX_POSITIONS remains the true global cap.
- UT Bot keeps its original opposite-cross lifecycle and no fixed SL/TP.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar

# Import installs Trend Confirm production patches (4H toggle, owner-aware exits,
# Telegram trigger labels and trigger-aware charts).
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
ADAPTIVE_FAMILY_PREFIX = "AdaptiveMultiTriggerV1("

_UT_ENTRY_CONTEXT: ContextVar[dict | None] = ContextVar(
    "utbot_xau_entry_context", default=None
)

# Capture Trend Confirm hooks before replacing run_bot public hooks.
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
    if key.startswith(ADAPTIVE_FAMILY_PREFIX):
        return "adaptive_multi_trigger"
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


def _family_enabled(family: str) -> bool:
    if family == "trend_confirm":
        return _env_bool("ENABLE_TREND_CONFIRM", True)
    if family == "adaptive_multi_trigger":
        return _env_bool("ENABLE_ADAPTIVE_MULTI_TRIGGER", False)
    if family == "utbot_xau":
        return _env_bool("ENABLE_UTBOT_XAU", False)
    return True


def _family_limit(family: str, global_limit: int) -> int:
    if family == "trend_confirm":
        return max(0, _env_int("TREND_CONFIRM_MAX_POSITIONS", 2))
    if family == "adaptive_multi_trigger":
        return max(0, _env_int("ADAPTIVE_MULTI_TRIGGER_MAX_POSITIONS", 2))
    if family == "utbot_xau":
        return 1
    return max(1, global_limit)


def _install_combined_risk_policy() -> None:
    """Independent family quotas, hard XAU ownership and anti-duplicate rule."""
    if getattr(RiskManager, "_tc_utbot_risk_policy_installed", False):
        return

    original_open_position = RiskManager.open_position

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
        if not _family_enabled(candidate_family):
            return False, f"{candidate_family} disabled by Railway toggle"

        # XAU belongs only to UT Bot. Adaptive and TC never open it.
        if candidate_family in {"trend_confirm", "adaptive_multi_trigger"} and symbol == UT_SYMBOL:
            return False, "XAU is reserved exclusively for UT Bot"
        if candidate_family == "utbot_xau" and symbol != UT_SYMBOL:
            return False, f"UT Bot is hard-locked to {UT_SYMBOL}"

        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return False, f"{strategy} already has an open position for {symbol}"

        positions_for_symbol = []
        family_count = 0
        for position_key, position in self._positions.items():
            tracked_strategy = position_key.split("||", 1)[1] if "||" in position_key else ""
            tracked_symbol = str(getattr(position, "symbol", "") or "")
            tracked_family = _family(tracked_strategy)
            if tracked_family == candidate_family:
                family_count += 1
            if tracked_symbol == symbol:
                positions_for_symbol.append((tracked_strategy, position))

        limit = _family_limit(candidate_family, self.max_open_positions)
        if candidate_family in {
            "trend_confirm", "adaptive_multi_trigger", "utbot_xau"
        } and family_count >= limit:
            return False, f"{candidate_family} position quota reached ({family_count}/{limit})"

        # Anti-conflict: TC + Adaptive may both scan a symbol, but they may not
        # create two independent positions on that same non-XAU symbol.
        if symbol != UT_SYMBOL and candidate_family in {
            "trend_confirm", "adaptive_multi_trigger"
        }:
            for tracked_strategy, _position in positions_for_symbol:
                tracked_family = _family(tracked_strategy)
                if tracked_family in {"trend_confirm", "adaptive_multi_trigger"}:
                    return False, (
                        f"{symbol} already owned by {tracked_family}; one symbol = one active "
                        "Trend/Adaptive position"
                    )

        # Legacy TC-XAU guard retained for safe migration.
        if symbol == UT_SYMBOL and candidate_family == "utbot_xau":
            for tracked_strategy, position in positions_for_symbol:
                tracked_family = _family(tracked_strategy)
                if tracked_family == "utbot_xau":
                    return False, "UT Bot already owns an XAU position"
                if tracked_family != "trend_confirm":
                    continue
                existing_side = _position_side(tracked_strategy, position)
                if candidate_side == existing_side:
                    return False, (
                        f"Legacy Trend Confirm XAU {existing_side.upper()} is still open; "
                        "same-side UT entry blocked to avoid OKX aggregation"
                    )

        # MAX_POSITIONS remains the real account-wide ceiling.
        if len(self._positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"
        return True, "ok"

    def _open_position(
        self: RiskManager,
        symbol: str,
        side: str,
        entry_price: float,
        amount: float,
        strategy: str = "",
        stop_loss: float = None,
        take_profit: float = None,
    ):
        position = original_open_position(
            self, symbol, side, entry_price, amount,
            strategy=strategy, stop_loss=stop_loss, take_profit=take_profit,
        )
        if _family(strategy) == "utbot_xau":
            position.stop_loss = None
            position.take_profit = None
        return position

    RiskManager.can_open = _can_open
    RiskManager.open_position = _open_position
    RiskManager._tc_utbot_risk_policy_installed = True


def _normal_symbols(config: dict) -> list[str]:
    """The normal Railway SYMBOLS universe with XAU removed."""
    source = list(config.get("base_symbols") or config.get("symbols") or [])
    return list(dict.fromkeys(s for s in source if s != UT_SYMBOL))


def _make_combined_strategies(symbols: list, config: dict):
    tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
    adaptive_enabled = _env_bool("ENABLE_ADAPTIVE_MULTI_TRIGGER", False)
    ut_enabled = _env_bool("ENABLE_UTBOT_XAU", False)

    if not any((tc_enabled, adaptive_enabled, ut_enabled)):
        raise RuntimeError(
            "No strategy enabled. Set ENABLE_TREND_CONFIRM=true, "
            "ENABLE_ADAPTIVE_MULTI_TRIGGER=true and/or ENABLE_UTBOT_XAU=true"
        )

    strategies = []
    normal_symbols = list(config.get("normal_symbols") or [])

    if tc_enabled:
        if not normal_symbols:
            logger.warning("Trend Confirm enabled but no non-XAU symbols are configured")
        else:
            strategies.extend(_TREND_MAKE_STRATEGIES(normal_symbols, config))

    if adaptive_enabled:
        from trading.strategies.adaptive_multi_trigger_strategy import (
            AdaptiveMultiTriggerStrategy,
        )
        for symbol in normal_symbols:
            strategies.append(
                AdaptiveMultiTriggerStrategy(
                    symbol=symbol,
                    entry_quality_threshold=_env_float("ADAPTIVE_ENTRY_QUALITY_THRESHOLD", 60.0),
                    weak_context_threshold=_env_float("ADAPTIVE_WEAK_CONTEXT_THRESHOLD", 70.0),
                    trigger_freshness_bars=_env_int("ADAPTIVE_TRIGGER_FRESHNESS_BARS", 3),
                    breakout_rvol_preferred=_env_float("ADAPTIVE_BREAKOUT_RVOL", 1.20),
                    strong_rvol=_env_float("ADAPTIVE_STRONG_RVOL", 1.50),
                    max_ema20_extension_atr=_env_float("ADAPTIVE_MAX_EMA20_EXTENSION_ATR", 1.50),
                    minimum_structure_room_r=_env_float("ADAPTIVE_MIN_STRUCTURE_ROOM_R", 1.20),
                    preferred_structure_room_r=_env_float("ADAPTIVE_PREFERRED_STRUCTURE_ROOM_R", 1.50),
                )
            )

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

    if not strategies:
        raise RuntimeError("Strategy toggles are enabled but no strategy instances were created")
    return strategies


def _build_combined_config() -> dict:
    tc_enabled = _env_bool("ENABLE_TREND_CONFIRM", True)
    adaptive_enabled = _env_bool("ENABLE_ADAPTIVE_MULTI_TRIGGER", False)
    ut_enabled = _env_bool("ENABLE_UTBOT_XAU", False)

    if not any((tc_enabled, adaptive_enabled, ut_enabled)):
        raise RuntimeError("No strategy enabled by Railway toggles")

    # Use TC config when TC is enabled because it installs the complete current
    # Trend Confirm production configuration. Otherwise start from baseline so
    # Adaptive-only / UT-only modes do not depend on TC being enabled.
    if tc_enabled:
        config = _TREND_BUILD_CONFIG()
    else:
        config = trend_runner.run_dual_bot._ORIGINAL_BUILD_CONFIG()

    base_symbols = list(config.get("symbols") or [])
    config["base_symbols"] = base_symbols
    normal_symbols = list(dict.fromkeys(s for s in base_symbols if s != UT_SYMBOL))
    config["normal_symbols"] = normal_symbols
    config["trend_confirm_symbols"] = normal_symbols if tc_enabled else []
    config["adaptive_multi_trigger_symbols"] = normal_symbols if adaptive_enabled else []
    config["trend_confirm_xau_disabled"] = True
    config["adaptive_xau_disabled"] = True
    config["trend_confirm_excluded_symbols"] = [UT_SYMBOL]

    market_data_symbols = list(normal_symbols) if (tc_enabled or adaptive_enabled) else []
    if ut_enabled:
        market_data_symbols.append(UT_SYMBOL)
    config["symbols"] = list(dict.fromkeys(market_data_symbols))

    config["candle_tf"] = "15m"
    os.environ["CANDLE_TF"] = "15m"
    config["enable_trend_confirm"] = tc_enabled
    config["enable_adaptive_multi_trigger"] = adaptive_enabled
    config["enable_utbot_xau"] = ut_enabled

    enabled_names = []
    if tc_enabled:
        enabled_names.append("trend_confirm")
    if adaptive_enabled:
        enabled_names.append("adaptive_multi_trigger")
    if ut_enabled:
        enabled_names.append("utbot_xau")
    config["strategy_mode"] = "+".join(enabled_names)

    # Respect Railway MAX_POSITIONS as the hard global risk ceiling. Only when
    # it is absent do we choose a sensible default from enabled family quotas.
    tc_limit = _env_int("TREND_CONFIRM_MAX_POSITIONS", 2) if tc_enabled else 0
    adaptive_limit = _env_int("ADAPTIVE_MULTI_TRIGGER_MAX_POSITIONS", 2) if adaptive_enabled else 0
    ut_limit = 1 if ut_enabled else 0
    default_global = max(1, tc_limit + adaptive_limit + ut_limit)
    if os.getenv("MAX_POSITIONS") is None:
        config["max_positions"] = default_global
    else:
        config["max_positions"] = max(1, _env_int("MAX_POSITIONS", default_global))

    config["hedge_mode"] = True
    config["futures"] = True
    return config


def _ut_caption(
    symbol: str, side: str, amount: float, price: float, strategy: str, paper: bool,
    direction: str = None, fee: float = 0.0, notional: float = None,
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
        f"{emoji} *OPEN {dir_label}*  `{symbol}`  {mode}", sep,
        f"📍 Entry Fill : `{price:,.4f}`",
        "⚡ Entry Trigger : `UT Bot ATR Trailing Stop Cross`",
    ]
    if meta.get("bar_close") is not None:
        lines.append(f"🕯 Signal Close : `{float(meta['bar_close']):,.4f}`")
    if meta.get("tsl_price") is not None:
        lines.append(f"〰️ ATR Trail : `{float(meta['tsl_price']):,.4f}`")
    if meta.get("atr") is not None:
        lines.append(
            f"📏 ATR({int(meta.get('atr_period', 10))}) : `{float(meta['atr']):,.4f}` × "
            f"`{float(meta.get('atr_multiplier', 1.0)):g}`"
        )
    lines += [
        "🛑 Fixed SL : `None`", "🎯 Fixed TP : `None`",
        "🏁 Exit / Reverse : `Opposite confirmed UT cross`",
        f"💰 Size : `{amount:.6g}`" + (f"  (≈`${notional:,.2f}`)" if notional is not None else ""),
    ]
    costs = []
    if margin is not None:
        costs.append(f"Margin `${margin:,.2f}`")
    if fee:
        costs.append(f"Fee `${fee:,.4f}`")
    if costs:
        lines.append("📥 " + "   ".join(costs))
    lines += [sep, "📊 Strategy : `UT Bot v2 — ATR Trailing Stop`",
              "⏱ Timeframe : `15m confirmed close`", f"🔒 Symbol : `XAU only` ({UT_SYMBOL})"]
    return "\n".join(lines)


def _install_ut_telegram_and_chart_patch() -> None:
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
                symbol=symbol, side=side, amount=amount, price=price,
                strategy=strategy, paper=paper, direction=kwargs.get("direction"),
                fee=kwargs.get("fee", 0.0), notional=kwargs.get("notional"),
                margin=kwargs.get("margin"),
            )
        return original_caption(self, symbol, side, amount, price, strategy, paper, **kwargs)

    def _renderer(*args, **kwargs):
        context = _UT_ENTRY_CONTEXT.get() or {}
        if context.get("is_ut"):
            return render_utbot_entry_chart(*args, **kwargs)
        return original_renderer(*args, **kwargs)

    def _notify(self, text: str):
        context = _UT_ENTRY_CONTEXT.get() or {}
        if context.get("is_ut") and "Order Executed" in str(text):
            text = f"{text}\n⚡ Entry Trigger: UT Bot ATR Cross\n🏁 Exit/Reverse: opposite confirmed UT cross"
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
                symbol, price,
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
