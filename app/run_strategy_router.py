"""Canonical production router for Sentinel V8 — Responsive Price Action Core.

Railway starts this file. Older Sentinel versions remain in the repository for
rollback, but production instantiates only V8.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import run_bot
from trading.bot import TradingBot, TradeRecord
from trading.telegram_notifier import TelegramNotifier
from trading import signal_state as signal_state_module
from trading.strategies.sentinel_v8_strategy import SentinelV8Strategy

logger = logging.getLogger("run_strategy_router")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_symbols(name: str, fallback: list[str]) -> list[str]:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return list(dict.fromkeys(fallback))
    return list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))


_BASE_BUILD_CONFIG = run_bot.build_config


def _build_config() -> dict:
    config = _BASE_BUILD_CONFIG()
    legacy_symbols = _env_symbols("SIMPLE_PRECISION_SYMBOLS", config.get("symbols") or [])
    config["symbols"] = _env_symbols("SENTINEL_SYMBOLS", legacy_symbols)
    config["strategy_mode"] = "simple_precision"
    config["candle_tf"] = "15m"
    config["interval"] = _env_int("SENTINEL_SCAN_SECONDS", 60)
    os.environ["CANDLE_TF"] = "15m"
    logger.warning(
        "[PRODUCTION CONFIG] Sentinel V%s | symbols=%s | scan=%ss closed-bars-only | "
        "15M bias=2of3(price/EMA20, EMA20 slope, RSI50) | no 1H hard gate | "
        "5M gate ADX>=10 CHOP<68 ATRx>=0.55 | triggers=PULLBACK_RECLAIM/MICRO_BREAKOUT/SWEEP_RECLAIM | "
        "anti-chase<=0.35ATR | SL=5M structure+0.15ATR min0.85 max2.00ATR | "
        "TP1=1R close50%% lock+0.15R | TP2=2R",
        SentinelV8Strategy.VERSION,
        config["symbols"],
        config["interval"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV8Strategy]:
    strategies = [SentinelV8Strategy(symbol) for symbol in symbols]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


# ---------------------------------------------------------------------------
# Compact V8 scan log — every layer visible without ARM/cross-state clutter.
# ---------------------------------------------------------------------------
_ORIGINAL_LOG_SCAN = TradingBot._log_scan
_LAST_SENTINEL_SCAN: dict[str, dict] = {}


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V8":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    reason = str(getattr(signal, "reason", "") or "")
    if reason != "5M bar already evaluated" and (meta.get("bias_15m") or meta.get("setup_5m")):
        _LAST_SENTINEL_SCAN[symbol] = meta
    view = _LAST_SENTINEL_SCAN.get(symbol, meta) if reason == "5M bar already evaluated" else meta

    bias = view.get("bias_15m") or {}
    setup = view.get("setup_5m") or {}
    gate = "PASS" if setup.get("market_ready") else "BLOCK" if setup else "-"
    blocks = setup.get("blocks", []) or setup.get("gate_blocks", []) or []
    repeat_tag = " | cached=same-5M-bar" if reason == "5M bar already evaluated" else ""

    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL V8] %s px=%.4f sig=%s | "
        "15M bias=%s votesL=%s votesS=%s close=%s EMA20=%s slopeATR=%s RSI=%s | "
        "5M gate=%s ADX=%s CHOP=%s ATRx=%s candle=%s EMA20=%s | "
        "PB=%s BO=%s SWEEP=%s trigger=%s chaseATR=%s slATR=%s | "
        "TP1=1R/50%% lock+0.15R TP2=2R | blocks=%s | %s%s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        bias.get("direction", "NEUTRAL"),
        bias.get("long_votes", "-"),
        bias.get("short_votes", "-"),
        bias.get("close", "-"),
        bias.get("ema20", "-"),
        bias.get("ema20_slope_atr", "-"),
        bias.get("rsi", "-"),
        gate,
        setup.get("adx", "-"),
        setup.get("chop", "-"),
        setup.get("atr_ratio", "-"),
        setup.get("candle", "-"),
        setup.get("ema20", "-"),
        "Y" if setup.get("pullback") else "N",
        "Y" if setup.get("breakout") else "N",
        "Y" if setup.get("sweep") else "N",
        setup.get("trigger", "-") or "-",
        setup.get("chase_atr", "-"),
        setup.get("sl_atr", setup.get("raw_sl_atr", "-")),
        ",".join(blocks) or "none",
        reason,
        repeat_tag,
    )


# ---------------------------------------------------------------------------
# Preserve lifecycle management for V7/V7.1 positions open during deployment.
# ---------------------------------------------------------------------------
_ORIGINAL_RESOLVE_STRATEGY = TradingBot._resolve_strategy_inst


def _sentinel_resolve_strategy(self, strategy_name: str):
    inst = _ORIGINAL_RESOLVE_STRATEGY(self, strategy_name)
    if inst is not None:
        return inst
    raw = str(strategy_name or "")
    bare = raw[:-2] if raw.endswith((":L", ":S")) else raw
    prefixes = ("SentinelV7(", "SentinelV7.1(")
    for prefix in prefixes:
        if bare.startswith(prefix) and bare.endswith(")"):
            symbol = bare[len(prefix):-1]
            return self._strategy_map.get(f"SentinelV8({symbol})")
    return None


# ---------------------------------------------------------------------------
# Telegram lifecycle description
# ---------------------------------------------------------------------------
_ORIGINAL_BUILD_ORDER_CAPTION = TelegramNotifier.build_order_caption


def _sentinel_build_order_caption(self, *args, **kwargs):
    text = _ORIGINAL_BUILD_ORDER_CAPTION(self, *args, **kwargs)
    strategy = kwargs.get("strategy")
    if strategy is None and len(args) >= 5:
        strategy = args[4]
    if "SentinelV8" in str(strategy or "") or "SentinelV8" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🛑 SL : 5M trigger structure + 0.15 ATR (0.85–2.00 ATR risk band)\n"
            "🎯 TP1 : +1.0R close 50% → runner SL +0.15R\n"
            "🎯 TP2 : +2.0R close remaining 50%\n"
            "🏁 Early exit : confirmed opposite 15M 2-of-3 bias flip",
        )
    return text


# ---------------------------------------------------------------------------
# /stats reason labels
# ---------------------------------------------------------------------------
_ORIGINAL_CLASSIFY_EXIT_REASON = signal_state_module.classify_exit_reason


def _sentinel_classify_exit_reason(reason: str, won: bool):
    r = (reason or "").lower()
    if "bias_flip_exit" in r:
        return ("15M Bias Flip Exit", "↩️")
    return _ORIGINAL_CLASSIFY_EXIT_REASON(reason, won)


# ---------------------------------------------------------------------------
# PAPER Sentinel only: hard SL/TP must win before technical bias-flip exit.
# LIVE already has exchange-side SL/TP orders.
# ---------------------------------------------------------------------------
_ORIGINAL_TICK = TradingBot._tick


async def _sentinel_tick_hard_stop_first(self):
    if getattr(self.connector, "paper", False):
        for pos_info in list(self.risk.get_positions()):
            strategy_name = pos_info.get("strategy", "")
            if not str(strategy_name).startswith("SentinelV"):
                continue

            sym = pos_info["symbol"]
            try:
                ticker = await self.connector.fetch_ticker(sym)
                price = ticker["last"]
                trigger = self.risk.check_stops(sym, price, strategy=strategy_name)
                if not trigger:
                    continue

                side = "sell" if pos_info["side"] == "long" else "buy"
                pos_side = pos_info["side"] if self._hedge_mode else None
                close_order = await self.connector.create_order(
                    sym, side, pos_info["amount"], pos_side=pos_side
                )
                fill = self._close_fill_info(
                    f"{sym}||{strategy_name}", close_order, price,
                    pos_info["amount"], 1.0, final=True,
                )
                exit_px = fill["exit_avg_px"]
                pnl = (
                    fill["net_pnl"] if fill["net_pnl"] is not None else
                    ((exit_px - pos_info["entry"]) * pos_info["amount"]
                     if pos_info["side"] == "long"
                     else (pos_info["entry"] - exit_px) * pos_info["amount"])
                )

                self._record_trade(TradeRecord(
                    timestamp=int(time.time() * 1000),
                    symbol=sym,
                    side=side,
                    price=exit_px,
                    amount=fill["exit_sz"],
                    pnl=round(pnl, 4),
                    strategy=strategy_name,
                    reason=trigger,
                    paper=True,
                ))
                outcome = self._sig.record_outcome(
                    symbol=sym,
                    side=pos_info["side"],
                    entry=pos_info["entry"],
                    exit_price=exit_px,
                    sl=pos_info.get("stop_loss"),
                    tp=pos_info.get("take_profit"),
                    reason=trigger,
                    strategy=strategy_name,
                    fill=fill,
                )
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                strategy_inst = self._resolve_strategy_inst(strategy_name)
                self._on_position_closed(sym, strategy_name, exit_px, trigger, strategy_inst)
                logging.getLogger("trading_bot").info(
                    "[SENTINEL V8 PAPER RISK-FIRST] Position closed by %s before technical exit: %s [%s]",
                    trigger, sym, strategy_name,
                )
                if self.telegram:
                    self.telegram.notify_trade_closed(sym, outcome, self._sig.summary())
                self._check_cooldown_trigger(pnl, sym)
            except Exception as e:
                logging.getLogger("trading_bot").error(
                    "[SENTINEL V8 PAPER RISK-FIRST] hard-stop precheck failed [%s %s]: %s",
                    strategy_name, sym, e,
                )

    return await _ORIGINAL_TICK(self)


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _sentinel_log_scan
TradingBot._resolve_strategy_inst = _sentinel_resolve_strategy
TradingBot._tick = _sentinel_tick_hard_stop_first
TelegramNotifier.build_order_caption = _sentinel_build_order_caption
signal_state_module.classify_exit_reason = _sentinel_classify_exit_reason

logger.warning(
    "[PRODUCTION] Sentinel V%s installed; scan=60s closed-bars-only; "
    "15M 2-of-3 bias -> 5M multi price-action trigger; no 1H/ARM hard gate; "
    "gate ADX10/CHOP68/ATRx0.55; structure SL 0.85-2.00ATR; "
    "TP1=1R close50%%+lock0.15R; TP2=2R",
    SentinelV8Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
