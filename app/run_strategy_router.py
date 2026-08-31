"""Canonical production router for Sentinel V8.1 — Quality Price Action Core.

Railway starts this file. Older Sentinel versions remain available for rollback,
but production instantiates only V8.1.
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
from trading.strategies.sentinel_v81_strategy import SentinelV81Strategy

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
        "15M slope-anchor bias + price/RSI confirm | 5M gate ADX>=12 CHOP<64 ATRx>=0.65 | "
        "PA=PULLBACK/MICRO_BREAK/SWEEP + quality close/location | raw structure risk >=0.40%% fee-edge | "
        "one fill/15M; hard-SL reentry wait=3x5M | anti-chase<=0.30ATR | "
        "SL structure+0.18ATR min0.90 max1.80ATR | TP1=1R close50%% lock+0.15R | TP2=2R",
        SentinelV81Strategy.VERSION,
        config["symbols"],
        config["interval"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV81Strategy]:
    strategies = [SentinelV81Strategy(symbol) for symbol in symbols]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


# ---------------------------------------------------------------------------
# V8.1 scan log — show the quality/fee decision, not only the raw trigger.
# ---------------------------------------------------------------------------
_ORIGINAL_LOG_SCAN = TradingBot._log_scan
_LAST_SENTINEL_SCAN: dict[str, dict] = {}


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V8_1":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    reason = str(getattr(signal, "reason", "") or "")
    if reason != "5M bar already evaluated" and (meta.get("bias_15m") or meta.get("setup_5m")):
        _LAST_SENTINEL_SCAN[symbol] = meta
    view = _LAST_SENTINEL_SCAN.get(symbol, meta) if reason == "5M bar already evaluated" else meta

    bias = view.get("bias_15m") or {}
    setup = view.get("setup_5m") or {}
    gate = "PASS" if setup.get("market_ready") else "BLOCK" if setup else "-"
    blocks = setup.get("blocks", []) or setup.get("gate_blocks", []) or []
    candidate = setup.get("trigger_candidate") or setup.get("trigger") or "-"
    repeat_tag = " | cached=same-5M-bar" if reason == "5M bar already evaluated" else ""

    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL V8.1] %s px=%.4f sig=%s | "
        "15M bias=%s strength=%s EMA20=%s slopeATR=%s RSI=%s | "
        "5M gate=%s ADX=%s CHOP=%s ATRx=%s | "
        "PB=%s BO=%s SWEEP=%s candidate=%s trigger=%s | "
        "bodyATR=%s closePos=%s distEMA=%s volx=%s slope5=%s chaseATR=%s rawRisk=%s%% slATR=%s strict=%s | "
        "TP1=1R/50%% lock+0.15R TP2=2R | blocks=%s | %s%s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        bias.get("direction", "NEUTRAL"),
        bias.get("strength", "-"),
        bias.get("ema20", "-"),
        bias.get("ema20_slope_atr", "-"),
        bias.get("rsi", "-"),
        gate,
        setup.get("adx", "-"),
        setup.get("chop", "-"),
        setup.get("atr_ratio", "-"),
        "Y" if setup.get("pullback") else "N",
        "Y" if setup.get("breakout") else "N",
        "Y" if setup.get("sweep") else "N",
        candidate,
        setup.get("trigger", "-") or "-",
        setup.get("body_atr", "-"),
        setup.get("close_pos", "-"),
        setup.get("dist_ema_atr", "-"),
        setup.get("volume_ratio", "-"),
        setup.get("ema20_slope_atr", "-"),
        setup.get("chase_atr", "-"),
        setup.get("raw_risk_pct", "-"),
        setup.get("sl_atr", setup.get("raw_sl_atr", "-")),
        "Y" if setup.get("strict_mode") else "N",
        ",".join(blocks) or "none",
        reason,
        repeat_tag,
    )


# ---------------------------------------------------------------------------
# Preserve lifecycle management for older Sentinel positions across deploys.
# ---------------------------------------------------------------------------
_ORIGINAL_RESOLVE_STRATEGY = TradingBot._resolve_strategy_inst


def _sentinel_resolve_strategy(self, strategy_name: str):
    inst = _ORIGINAL_RESOLVE_STRATEGY(self, strategy_name)
    if inst is not None:
        return inst
    raw = str(strategy_name or "")
    bare = raw[:-2] if raw.endswith((":L", ":S")) else raw
    prefixes = ("SentinelV7(", "SentinelV7.1(", "SentinelV8(")
    for prefix in prefixes:
        if bare.startswith(prefix) and bare.endswith(")"):
            symbol = bare[len(prefix):-1]
            return self._strategy_map.get(f"SentinelV8.1({symbol})")
    return None


# ---------------------------------------------------------------------------
# Fill synchronization for V8.1.
# Keep the structure SL absolute, but rebuild TP from the freshest pre-order
# price and then sync the strategy to the ACTUAL post-fill entry afterwards.
# This prevents a fill shift from turning a planned 2R target into 2.3R/1.7R.
# ---------------------------------------------------------------------------
_ORIGINAL_EXECUTE_SIGNAL = TradingBot._execute_signal


async def _sentinel_execute_signal(self, signal, strategy_name: str,
                                   direction: str = "long", candles=None):
    is_v81 = str(strategy_name).startswith("SentinelV8.1")
    if is_v81:
        meta = signal.metadata or {}
        sl = meta.get("stop_loss")
        if sl:
            try:
                px = float((await self.connector.fetch_ticker(signal.symbol))["last"])
                risk = abs(px - float(sl))
                if risk > 0:
                    rr = float(meta.get("rr_ratio") or SentinelV81Strategy.TP2_R)
                    tp1r = float(meta.get("tp1_r") or SentinelV81Strategy.TP1_R)
                    meta["take_profit"] = px + rr * risk if direction == "long" else px - rr * risk
                    meta["tp1_price"] = px + tp1r * risk if direction == "long" else px - tp1r * risk
                    signal.metadata = meta
            except Exception as e:
                logging.getLogger("trading_bot").warning(
                    "[SENTINEL V8.1] pre-fill TP rebase skipped [%s]: %s", signal.symbol, e
                )

    result = await _ORIGINAL_EXECUTE_SIGNAL(
        self, signal, strategy_name, direction=direction, candles=candles
    )

    if not is_v81:
        return result

    # Exact post-fill synchronization. The order is already open, so use the
    # RiskManager's recorded fill entry and keep the original structure SL.
    key = f"{signal.symbol}||{strategy_name}"
    pos_obj = getattr(self.risk, "_positions", {}).get(key)
    if pos_obj is None or pos_obj.stop_loss is None:
        return result

    try:
        entry = float(pos_obj.entry_price)
        sl = float(pos_obj.stop_loss)
        actual_risk = abs(entry - sl)
        if actual_risk <= 0:
            return result
        exact_tp = (
            entry + SentinelV81Strategy.TP2_R * actual_risk
            if direction == "long"
            else entry - SentinelV81Strategy.TP2_R * actual_risk
        )
        pos_obj.take_profit = exact_tp

        meta = signal.metadata or {}
        meta["take_profit"] = exact_tp
        meta["tp1_price"] = (
            entry + SentinelV81Strategy.TP1_R * actual_risk
            if direction == "long"
            else entry - SentinelV81Strategy.TP1_R * actual_risk
        )
        signal.metadata = meta

        strategy_inst = self._resolve_strategy_inst(strategy_name)
        if hasattr(strategy_inst, "attach_existing_position"):
            strategy_inst.attach_existing_position(direction, entry, sl, exact_tp)

        try:
            await self.connector.set_position_tpsl(
                signal.symbol, direction, float(pos_obj.amount), sl=sl, tp=exact_tp
            )
        except Exception as e:
            logging.getLogger("trading_bot").warning(
                "[SENTINEL V8.1] exact post-fill TP replace failed [%s]: %s", signal.symbol, e
            )

        logging.getLogger("trading_bot").info(
            "[SENTINEL V8.1 FILL-SYNC] %s %s entry=%.4f SL=%.4f TP=%.4f exact=2.00R",
            signal.symbol, direction.upper(), entry, sl, exact_tp,
        )
    except Exception as e:
        logging.getLogger("trading_bot").warning(
            "[SENTINEL V8.1] post-fill synchronization failed [%s]: %s", signal.symbol, e
        )
    return result


# ---------------------------------------------------------------------------
# Telegram lifecycle description
# ---------------------------------------------------------------------------
_ORIGINAL_BUILD_ORDER_CAPTION = TelegramNotifier.build_order_caption


def _sentinel_build_order_caption(self, *args, **kwargs):
    text = _ORIGINAL_BUILD_ORDER_CAPTION(self, *args, **kwargs)
    strategy = kwargs.get("strategy")
    if strategy is None and len(args) >= 5:
        strategy = args[4]
    if "SentinelV8.1" in str(strategy or "") or "SentinelV8.1" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🛑 SL : 5M structure + 0.18 ATR (0.90–1.80 ATR; structure risk must be ≥0.40%)\n"
            "🎯 TP1 : +1.0R close 50% → runner SL +0.15R\n"
            "🎯 TP2 : +2.0R close remaining 50%\n"
            "🏁 Early exit : confirmed opposite 15M slope-anchor bias flip",
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
# PAPER Sentinel: simulate exchange-side hard SL/TP at the trigger level rather
# than at the next 60-second poll price. LIVE already has OKX algo TP/SL orders.
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
                price = float((await self.connector.fetch_ticker(sym))["last"])
                trigger = self.risk.check_stops(sym, price, strategy=strategy_name)
                if not trigger:
                    continue

                trigger_px = (
                    pos_info.get("stop_loss") if trigger == "stop_loss"
                    else pos_info.get("take_profit") if trigger == "take_profit"
                    else price
                )
                trigger_px = float(trigger_px or price)
                side = "sell" if pos_info["side"] == "long" else "buy"
                pos_side = pos_info["side"] if self._hedge_mode else None

                # Paper connector honors explicit price for a limit simulation.
                # This models an exchange-side trigger fill at the configured
                # level instead of waiting for the next polling price.
                close_order = await self.connector.create_order(
                    sym, side, pos_info["amount"], order_type="limit",
                    price=trigger_px, pos_side=pos_side,
                )
                fill = self._close_fill_info(
                    f"{sym}||{strategy_name}", close_order, trigger_px,
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
                    symbol=sym, side=side, price=exit_px,
                    amount=fill["exit_sz"], pnl=round(pnl, 4),
                    strategy=strategy_name, reason=trigger, paper=True,
                ))
                outcome = self._sig.record_outcome(
                    symbol=sym, side=pos_info["side"], entry=pos_info["entry"],
                    exit_price=exit_px, sl=pos_info.get("stop_loss"),
                    tp=pos_info.get("take_profit"), reason=trigger,
                    strategy=strategy_name, fill=fill,
                )
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                strategy_inst = self._resolve_strategy_inst(strategy_name)
                self._on_position_closed(sym, strategy_name, exit_px, trigger, strategy_inst)
                logging.getLogger("trading_bot").info(
                    "[SENTINEL PAPER TRIGGER-FILL] %s closed by %s @ %.4f [%s]",
                    sym, trigger, exit_px, strategy_name,
                )
                if self.telegram:
                    self.telegram.notify_trade_closed(sym, outcome, self._sig.summary())
                self._check_cooldown_trigger(pnl, sym)
            except Exception as e:
                logging.getLogger("trading_bot").error(
                    "[SENTINEL V8.1 PAPER] hard-stop precheck failed [%s %s]: %s",
                    strategy_name, sym, e,
                )

    return await _ORIGINAL_TICK(self)


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _sentinel_log_scan
TradingBot._resolve_strategy_inst = _sentinel_resolve_strategy
TradingBot._execute_signal = _sentinel_execute_signal
TradingBot._tick = _sentinel_tick_hard_stop_first
TelegramNotifier.build_order_caption = _sentinel_build_order_caption
signal_state_module.classify_exit_reason = _sentinel_classify_exit_reason

logger.warning(
    "[PRODUCTION] Sentinel V%s installed; fee-aware quality PA; scan=60s; "
    "15M slope-anchor bias -> 5M PA; economic structure risk>=0.40%%; one fill/15M; "
    "hard-SL wait=3x5M; TP1=1R/50%% lock+0.15R; TP2 exact 2R from actual fill",
    SentinelV81Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
