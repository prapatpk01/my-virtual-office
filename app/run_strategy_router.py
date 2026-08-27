"""Canonical production router for Sentinel V5 — Simple RSI14/SMA14 Cross.

Railway starts this file. V4.x remains in the repository for rollback and
comparison, but production instantiates only the deliberately small V5 engine.
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
from trading.strategies.sentinel_v5_strategy import SentinelV5Strategy

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
    os.environ["CANDLE_TF"] = "15m"
    logger.warning(
        "[PRODUCTION CONFIG] Sentinel V%s | symbols=%s | 15M closed bars | "
        "gate ADX>=12 CHOP<65 ATRx>=0.65 | LONG RSI/SMA cross<55 | "
        "SHORT RSI/SMA cross>=65 | SL=1.2ATR | T1=.8R->+.25R | "
        "T2=1.3R->+.70R | TP=1.8R",
        SentinelV5Strategy.VERSION,
        config["symbols"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV5Strategy]:
    strategies = [
        SentinelV5Strategy(
            symbol,
            exit_cooldown_bars=_env_int("SENTINEL_EXIT_COOLDOWN_BARS", 2),
        )
        for symbol in symbols
    ]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


# ---------------------------------------------------------------------------
# Compact V5 scan log
# ---------------------------------------------------------------------------
_ORIGINAL_LOG_SCAN = TradingBot._log_scan
_LAST_SENTINEL_SCAN: dict[str, dict] = {}


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V5":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    reason = str(getattr(signal, "reason", "") or "")
    market = meta.get("market_15m") or {}
    entry = meta.get("entry_15m") or {}
    if market and reason != "15M bar already evaluated":
        _LAST_SENTINEL_SCAN[symbol] = meta

    repeated_bar = reason == "15M bar already evaluated"
    view = _LAST_SENTINEL_SCAN.get(symbol, meta) if repeated_bar else meta
    market = view.get("market_15m") or market
    entry = view.get("entry_15m") or entry

    gate = "PASS" if market.get("market_ready") else "BLOCK" if market else "-"
    cross = market.get("cross", "-")
    long_zone = "Y" if market.get("long_zone") else "N"
    short_zone = "Y" if market.get("short_zone") else "N"
    blocks = entry.get("blocks", []) or market.get("blocks", []) or []
    repeat_tag = " | cached=same-15M-bar" if repeated_bar and view is not meta else ""

    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL V5] %s px=%.4f sig=%s | gate=%s ADX=%s CHOP=%s ATRx=%s | "
        "RSI=%s SMA14=%s prev=%s/%s cross=%s zoneL<55=%s zoneS>=65=%s | "
        "SL=1.2ATR T1=.8R->+.25R T2=1.3R->+.70R TP=1.8R | blocks=%s | %s%s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        gate,
        market.get("adx", "-"),
        market.get("chop", "-"),
        market.get("atr_ratio", "-"),
        market.get("rsi", "-"),
        market.get("rsi_sma", "-"),
        market.get("prev_rsi", "-"),
        market.get("prev_rsi_sma", "-"),
        cross,
        long_zone,
        short_zone,
        ",".join(blocks) or "none",
        reason,
        repeat_tag,
    )


# ---------------------------------------------------------------------------
# Telegram: describe the actual V5 lifecycle instead of the legacy EMA caption.
# ---------------------------------------------------------------------------
_ORIGINAL_BUILD_ORDER_CAPTION = TelegramNotifier.build_order_caption


def _sentinel_build_order_caption(self, *args, **kwargs):
    text = _ORIGINAL_BUILD_ORDER_CAPTION(self, *args, **kwargs)
    strategy = kwargs.get("strategy")
    if strategy is None and len(args) >= 5:
        strategy = args[4]
    if "SentinelV5" in str(strategy or "") or "SentinelV5" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🔒 Locks : +0.8R → SL +0.25R | +1.3R → SL +0.70R\n"
            "🏁 Exit : opposite RSI14/SMA14 cross (closed 15M) / hard SL / TP",
        )
    return text


# ---------------------------------------------------------------------------
# /stats reason label: V5's early exit is an RSI/SMA cross, not an EMA cross.
# ---------------------------------------------------------------------------
_ORIGINAL_CLASSIFY_EXIT_REASON = signal_state_module.classify_exit_reason


def _sentinel_classify_exit_reason(reason: str, won: bool):
    r = (reason or "").lower()
    if "rsi14/sma14 opposite cross" in r:
        return ("RSI/SMA Cross Exit", "↩️")
    return _ORIGINAL_CLASSIFY_EXIT_REASON(reason, won)


# ---------------------------------------------------------------------------
# PAPER Sentinel only: hard SL/TP must win before the closed-bar RSI exit.
# LIVE keeps exchange-side SL/TP as the primary hard-risk enforcement.
# ---------------------------------------------------------------------------
_ORIGINAL_TICK = TradingBot._tick


async def _sentinel_tick_hard_stop_first(self):
    if getattr(self.connector, "paper", False):
        for pos_info in list(self.risk.get_positions()):
            strategy_name = pos_info.get("strategy", "")
            if not str(strategy_name).startswith("SentinelV5"):
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
                    "[SENTINEL V5 PAPER RISK-FIRST] closed by %s before RSI exit: %s [%s]",
                    trigger, sym, strategy_name,
                )
                if self.telegram:
                    self.telegram.notify_trade_closed(sym, outcome, self._sig.summary())
                self._check_cooldown_trigger(pnl, sym)
            except Exception as e:
                logging.getLogger("trading_bot").error(
                    "[SENTINEL V5 PAPER RISK-FIRST] precheck failed [%s %s]: %s",
                    strategy_name, sym, e,
                )

    return await _ORIGINAL_TICK(self)


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _sentinel_log_scan
TradingBot._tick = _sentinel_tick_hard_stop_first
TelegramNotifier.build_order_caption = _sentinel_build_order_caption
signal_state_module.classify_exit_reason = _sentinel_classify_exit_reason

logger.warning(
    "[PRODUCTION] Sentinel V%s installed — SIMPLE RSI14/SMA14 CROSS; no structure/BOS/sweep/1H-SR/room/chase; "
    "SL=1.2ATR, T1=.8R lock +.25R, T2=1.3R lock +.70R, TP=1.8R, opposite RSI/SMA cross exits early",
    SentinelV5Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
