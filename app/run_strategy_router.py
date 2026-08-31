"""Canonical production router for Sentinel V7.1 — RSI Rotation Memory + 5M Price Trigger.

Railway starts this file. Older Sentinel versions remain in the repository for
rollback, but production instantiates only V7.1.
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
from trading.strategies.sentinel_v71_strategy import SentinelV71Strategy

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
    # Base series remains 15M. entry_tf='5m' requests a closed 5M execution
    # series. Poll every 60s so a newly closed 5M bar is normally seen within
    # about one minute instead of waiting for the old 300s cadence.
    config["candle_tf"] = "15m"
    config["interval"] = _env_int("SENTINEL_SCAN_SECONDS", 60)
    os.environ["CANDLE_TF"] = "15m"
    logger.warning(
        "[PRODUCTION CONFIG] Sentinel V%s | symbols=%s | scan=%ss closed-bars-only | "
        "1H EMA20/50+slope direction | 15M ADX>=12 CHOP<65 ATRx>=0.65 | "
        "RSI rotation memory: LONG min4<=45 crossUp curr<=55; SHORT max4>=55 crossDn curr>=45; "
        "RSI slope + |spread|>=0.30 hard, SMA slope/spread accel/divergence soft | "
        "ARM=4x5M | 5M micro breakout anti-chase<=0.25ATR | "
        "SL=structure+0.20ATR min0.80 max1.60ATR | TP1=1R close50%% lock+0.15R | TP2=2R",
        SentinelV71Strategy.VERSION,
        config["symbols"],
        config["interval"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV71Strategy]:
    strategies = [
        SentinelV71Strategy(
            symbol,
            exit_cooldown_bars=_env_int("SENTINEL_EXIT_COOLDOWN_BARS", 2),
        )
        for symbol in symbols
    ]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


# ---------------------------------------------------------------------------
# Compact V7.1 scan log with persistent 15M ARM-decision visibility.
# ---------------------------------------------------------------------------
_ORIGINAL_LOG_SCAN = TradingBot._log_scan


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V7_1":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    reason = str(getattr(signal, "reason", "") or "")
    trend = meta.get("trend_1h") or {}
    market = meta.get("market_15m") or {}
    arm = meta.get("arm") or {}
    trig5 = meta.get("trigger_5m") or {}
    divergence = market.get("divergence") or {}
    decision = meta.get("arm_decision_cached") or {}

    gate = "PASS" if market.get("market_ready") else "BLOCK" if market else "-"
    arm_state = (str(arm.get("side") or "-").upper() if arm.get("active") else "OFF")
    current_arm_blocks = meta.get("arm_blocks", []) or []
    cached_blocks = decision.get("blocks", []) or []
    blocks = trig5.get("blocks", []) or current_arm_blocks or cached_blocks or market.get("blocks", []) or []
    cached_tag = "Y" if decision and not meta.get("arm_event") else "N"

    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL V7.1] %s px=%.4f sig=%s | "
        "1H=%s EMA20=%s EMA50=%s slope=%s | "
        "15M gate=%s ADX=%s CHOP=%s ATRx=%s RSI=%s SMA14=%s cross=%s "
        "rSlope=%s spread=%s dSpread=%s rotMin4=%s rotMax4=%s div=%s counterDiv=%s | "
        "ARM=%s decision=%s cached15=%s | "
        "5M=%s candle=%s close=%s prev2H=%s prev2L=%s chaseATR=%s slATR=%s | "
        "TP1=1R/50%% lock+0.15R TP2=2R | blocks=%s | %s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        trend.get("direction", "-"),
        trend.get("ema20", "-"),
        trend.get("ema50", "-"),
        trend.get("slope", "-"),
        gate,
        market.get("adx", "-"),
        market.get("chop", "-"),
        market.get("atr_ratio", "-"),
        market.get("rsi", "-"),
        market.get("rsi_sma", "-"),
        market.get("cross", "-"),
        market.get("rsi_slope", "-"),
        market.get("spread", "-"),
        market.get("spread_delta", "-"),
        market.get("rotation_recent_min", "-"),
        market.get("rotation_recent_max", "-"),
        divergence.get("label", "NONE"),
        "Y" if meta.get("counter_divergence_soft") else "N",
        arm_state,
        decision.get("decision", "-"),
        cached_tag,
        "TRIGGER" if trig5.get("trigger") else trig5.get("reason", "-"),
        trig5.get("candle", "-"),
        trig5.get("close", "-"),
        trig5.get("prev2_high", "-"),
        trig5.get("prev2_low", "-"),
        trig5.get("chase_atr", "-"),
        trig5.get("sl_atr", trig5.get("raw_sl_atr", "-")),
        ",".join(blocks) or "none",
        reason,
    )


# ---------------------------------------------------------------------------
# Preserve lifecycle management for any V7 position that was already open
# during the V7 -> V7.1 deployment.
# ---------------------------------------------------------------------------
_ORIGINAL_RESOLVE_STRATEGY = TradingBot._resolve_strategy_inst


def _sentinel_resolve_strategy(self, strategy_name: str):
    inst = _ORIGINAL_RESOLVE_STRATEGY(self, strategy_name)
    if inst is not None:
        return inst
    raw = str(strategy_name or "")
    bare = raw[:-2] if raw.endswith((":L", ":S")) else raw
    if bare.startswith("SentinelV7(") and bare.endswith(")"):
        symbol = bare[len("SentinelV7("):-1]
        return self._strategy_map.get(f"SentinelV7.1({symbol})")
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
    if "SentinelV7.1" in str(strategy or "") or "SentinelV7.1" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🛑 SL : 5M structure + 0.20 ATR (0.80–1.60 ATR risk band)\n"
            "🎯 TP1 : +1.0R close 50% → runner SL +0.15R\n"
            "🎯 TP2 : +2.0R close remaining 50%\n"
            "🏁 Early exit : opposite RSI14/SMA14 on closed 15M; neutral confirm before TP1",
        )
    return text


# ---------------------------------------------------------------------------
# /stats reason labels
# ---------------------------------------------------------------------------
_ORIGINAL_CLASSIFY_EXIT_REASON = signal_state_module.classify_exit_reason


def _sentinel_classify_exit_reason(reason: str, won: bool):
    r = (reason or "").lower()
    if "rsi_reversal_exit" in r:
        return ("RSI Reversal Exit", "↩️")
    if "rsi_runner_exit" in r:
        return ("Runner RSI Exit", "🏃")
    return _ORIGINAL_CLASSIFY_EXIT_REASON(reason, won)


# ---------------------------------------------------------------------------
# PAPER Sentinel only: hard SL/TP must win before the closed-bar RSI exit.
# LIVE already has exchange-side SL/TP orders.
# ---------------------------------------------------------------------------
_ORIGINAL_TICK = TradingBot._tick


async def _sentinel_tick_hard_stop_first(self):
    if getattr(self.connector, "paper", False):
        for pos_info in list(self.risk.get_positions()):
            strategy_name = pos_info.get("strategy", "")
            if not str(strategy_name).startswith("SentinelV7"):
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
                    "[SENTINEL V7.1 PAPER RISK-FIRST] Position closed by %s before RSI exit: %s [%s]",
                    trigger, sym, strategy_name,
                )
                if self.telegram:
                    self.telegram.notify_trade_closed(sym, outcome, self._sig.summary())
                self._check_cooldown_trigger(pnl, sym)
            except Exception as e:
                logging.getLogger("trading_bot").error(
                    "[SENTINEL V7.1 PAPER RISK-FIRST] hard-stop precheck failed [%s %s]: %s",
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
    "1H direction -> 15M RSI rotation-memory ARM -> 5M price trigger; "
    "ARM=4 bars anti-chase=0.25ATR; structure SL +0.20ATR (0.80-1.60ATR); "
    "TP1=1R close50%%+lock0.15R; TP2=2R",
    SentinelV71Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
