"""Canonical production router for Sentinel V9 — Scored Setup Execution.

Railway starts this file. Production instantiates only Sentinel V9 while older
strategy files remain available for rollback.
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
from trading.strategies.sentinel_v9_strategy import SentinelV9Strategy

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
        "15M Pine-v6.2 scored analysis PB/LQ/BO/REV | "
        "score mins PB6 LQ6 BOdirect7 BOretest6.5 REV7.5 | "
        "5M V8.1 execution gate ADX>=12 CHOP<64 ATRx>=0.65 + quality/fee edge | "
        "anti-chase<=0.30ATR | SL structure+0.18ATR min0.90 max1.80ATR risk>=0.40%% | "
        "TP1=1R close50%% lock+0.15R | TP2 dynamic1.5-2.5R fallback2R",
        SentinelV9Strategy.VERSION,
        config["symbols"],
        config["interval"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV9Strategy]:
    strategies = [SentinelV9Strategy(symbol) for symbol in symbols]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


# ---------------------------------------------------------------------------
# Compact V9 scan log: 15M thesis/score + 5M execution/edge.
# ---------------------------------------------------------------------------
_ORIGINAL_LOG_SCAN = TradingBot._log_scan
_LAST_SENTINEL_SCAN: dict[str, dict] = {}


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V9":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)

    reason = str(getattr(signal, "reason", "") or "")
    if reason != "5M bar already evaluated" and (meta.get("analysis_15m") or meta.get("setup_5m")):
        _LAST_SENTINEL_SCAN[symbol] = meta
    view = _LAST_SENTINEL_SCAN.get(symbol, meta) if reason == "5M bar already evaluated" else meta

    a = view.get("analysis_15m") or {}
    s = view.get("setup_5m") or {}
    gate = "PASS" if s.get("market_ready") else "BLOCK" if s else "-"
    blocks = s.get("blocks", []) or s.get("gate_blocks", []) or []
    candidate = s.get("trigger_candidate") or s.get("trigger") or "-"
    comp = a.get("components") or {}
    repeat_tag = " | cached=same-5M-bar" if reason == "5M bar already evaluated" else ""

    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL V9] %s px=%.4f sig=%s | "
        "15M setup=%s side=%s score=%s/%s L=%s S=%s | "
        "T=%s QL/QS=%s/%s Struct=%s Loc=%s roomL/S=%s/%s "
        "ADX15=%s CHOP15=%s RSI=%s/%s HMA=%s | "
        "pts[T/Q/S/L/M]=%s/%s/%s/%s/%s | "
        "5M gate=%s ADX=%s CHOP=%s ATRx=%s candidate=%s trigger=%s "
        "body=%s closePos=%s distEMA=%s volx=%s chase=%s rawRisk=%s%% slATR=%s strict=%s | "
        "TP2R=%s source=%s | blocks=%s | %s%s",
        symbol,
        price,
        getattr(getattr(signal, "type", None), "value", "hold").upper(),
        a.get("selected_setup", "-") or "-",
        a.get("direction", "NEUTRAL") or "NEUTRAL",
        a.get("selected_score", "-"),
        a.get("score_threshold", "-"),
        a.get("score_long", "-"),
        a.get("score_short", "-"),
        a.get("trend", "-"),
        a.get("trend_quality_long", "-"),
        a.get("trend_quality_short", "-"),
        a.get("structure", "-"),
        a.get("location", "-"),
        a.get("room_long_atr", "-"),
        a.get("room_short_atr", "-"),
        a.get("adx", "-"),
        a.get("chop", "-"),
        a.get("rsi", "-"),
        a.get("rsi_sma", "-"),
        a.get("hma_slope_atr", "-"),
        comp.get("trend", "-"),
        comp.get("quality", "-"),
        comp.get("structure", "-"),
        comp.get("location", "-"),
        comp.get("momentum", "-"),
        gate,
        s.get("adx", "-"),
        s.get("chop", "-"),
        s.get("atr_ratio", "-"),
        candidate,
        s.get("trigger", "-") or "-",
        s.get("body_atr", "-"),
        s.get("close_pos", "-"),
        s.get("dist_ema_atr", "-"),
        s.get("volume_ratio", "-"),
        s.get("chase_atr", "-"),
        s.get("raw_risk_pct", "-"),
        s.get("sl_atr", s.get("raw_sl_atr", "-")),
        "Y" if s.get("strict_mode") else "N",
        view.get("tp2_r_dynamic", "-"),
        view.get("tp2_source", "-"),
        ",".join(blocks) or "none",
        reason,
        repeat_tag,
    )


# ---------------------------------------------------------------------------
# Preserve lifecycle management for positions opened by older Sentinel versions.
# ---------------------------------------------------------------------------
_ORIGINAL_RESOLVE_STRATEGY = TradingBot._resolve_strategy_inst


def _sentinel_resolve_strategy(self, strategy_name: str):
    inst = _ORIGINAL_RESOLVE_STRATEGY(self, strategy_name)
    if inst is not None:
        return inst
    raw = str(strategy_name or "")
    bare = raw[:-2] if raw.endswith((":L", ":S")) else raw
    prefixes = ("SentinelV7(", "SentinelV7.1(", "SentinelV8(", "SentinelV8.1(")
    for prefix in prefixes:
        if bare.startswith(prefix) and bare.endswith(")"):
            symbol = bare[len(prefix):-1]
            return self._strategy_map.get(f"SentinelV9({symbol})")
    return None


# ---------------------------------------------------------------------------
# Fill synchronization.
# Keep the structure SL absolute. Rebuild TP1/TP2 from freshest pre-order price,
# then from ACTUAL post-fill entry. V9's rr_ratio can be dynamic 1.5..2.5R.
# ---------------------------------------------------------------------------
_ORIGINAL_EXECUTE_SIGNAL = TradingBot._execute_signal


async def _sentinel_execute_signal(self, signal, strategy_name: str,
                                   direction: str = "long", candles=None):
    is_v9 = str(strategy_name).startswith("SentinelV9")
    if is_v9:
        meta = signal.metadata or {}
        sl = meta.get("stop_loss")
        if sl:
            try:
                px = float((await self.connector.fetch_ticker(signal.symbol))["last"])
                risk = abs(px - float(sl))
                if risk > 0:
                    rr = float(meta.get("rr_ratio") or SentinelV9Strategy.DYNAMIC_TP_FALLBACK_R)
                    rr = max(SentinelV9Strategy.DYNAMIC_TP_MIN_R,
                             min(SentinelV9Strategy.DYNAMIC_TP_MAX_R, rr))
                    tp1r = float(meta.get("tp1_r") or SentinelV9Strategy.TP1_R)
                    meta["rr_ratio"] = rr
                    meta["take_profit"] = px + rr * risk if direction == "long" else px - rr * risk
                    meta["tp1_price"] = px + tp1r * risk if direction == "long" else px - tp1r * risk
                    signal.metadata = meta
            except Exception as e:
                logging.getLogger("trading_bot").warning(
                    "[SENTINEL V9] pre-fill target rebase skipped [%s]: %s", signal.symbol, e
                )

    result = await _ORIGINAL_EXECUTE_SIGNAL(
        self, signal, strategy_name, direction=direction, candles=candles
    )

    if not is_v9:
        return result

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

        meta = signal.metadata or {}
        rr = float(meta.get("rr_ratio") or SentinelV9Strategy.DYNAMIC_TP_FALLBACK_R)
        rr = max(SentinelV9Strategy.DYNAMIC_TP_MIN_R,
                 min(SentinelV9Strategy.DYNAMIC_TP_MAX_R, rr))
        exact_tp = entry + rr * actual_risk if direction == "long" else entry - rr * actual_risk
        exact_tp1 = (
            entry + SentinelV9Strategy.TP1_R * actual_risk
            if direction == "long"
            else entry - SentinelV9Strategy.TP1_R * actual_risk
        )

        pos_obj.take_profit = exact_tp
        meta["take_profit"] = exact_tp
        meta["tp1_price"] = exact_tp1
        meta["rr_ratio"] = rr
        meta["tp2_r_dynamic"] = rr
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
                "[SENTINEL V9] exact post-fill TP replace failed [%s]: %s", signal.symbol, e
            )

        logging.getLogger("trading_bot").info(
            "[SENTINEL V9 FILL-SYNC] %s %s entry=%.4f SL=%.4f TP=%.4f exact=%.2fR source=%s",
            signal.symbol, direction.upper(), entry, sl, exact_tp, rr,
            meta.get("tp2_source", "FALLBACK_2R"),
        )
    except Exception as e:
        logging.getLogger("trading_bot").warning(
            "[SENTINEL V9] post-fill synchronization failed [%s]: %s", signal.symbol, e
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
    if "SentinelV9" in str(strategy or "") or "SentinelV9" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🧠 15M : scored PB/LQ/BO/REV analysis (Trend/Quality/Structure/Location/Momentum)\n"
            "⚡ Entry : confirmed 5M price action\n"
            "🛑 SL : 5M structure + 0.18 ATR (0.90–1.80 ATR; natural risk ≥0.40%)\n"
            "🎯 TP1 : +1.0R close 50% → runner SL +0.15R\n"
            "🎯 TP2 : dynamic 1.5–2.5R from 15M structure/Fib; fallback 2R\n"
            "🏁 Early exit : confirmed opposite qualified 15M scored setup",
        )
    return text


# ---------------------------------------------------------------------------
# /stats reason labels
# ---------------------------------------------------------------------------
_ORIGINAL_CLASSIFY_EXIT_REASON = signal_state_module.classify_exit_reason


def _sentinel_classify_exit_reason(reason: str, won: bool):
    r = (reason or "").lower()
    if "bias_flip_exit" in r:
        return ("15M Scored Setup Flip Exit", "↩️")
    return _ORIGINAL_CLASSIFY_EXIT_REASON(reason, won)


# ---------------------------------------------------------------------------
# PAPER Sentinel: hard SL/TP first, filled at configured trigger level.
# LIVE already has exchange-side OKX algo TP/SL.
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
                    "[SENTINEL V9 PAPER] hard-stop precheck failed [%s %s]: %s",
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
    "[PRODUCTION] Sentinel V%s installed; 15M Pine-v6.2 scored analysis -> 5M V8.1 execution/risk; "
    "PB/LQ/BO/REV setup-specific scores; fee-aware structure SL; one fill/15M; "
    "hard-SL wait=3x5M; TP1=1R/50%% lock+0.15R; TP2 dynamic1.5-2.5R fallback2R",
    SentinelV9Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
