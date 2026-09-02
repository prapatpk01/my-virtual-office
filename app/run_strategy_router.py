"""Canonical production router for Sentinel V10 — Momentum + Location Forecast Core.

Railway starts this file. Production instantiates V10 only. V9 and all older
strategy files remain untouched for rollback.
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
from trading.strategies.sentinel_v10_strategy import SentinelV10Strategy

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
        "15M EMA20 + RSI/SMA14 + MACD + KDJ + S/R | regime ADX/CHOP/ATRx=2of3 | "
        "setups=PB>=6 BO-RETEST>=6.5 SWEEP>=7 | Forecast advisory/runner target only | "
        "5M PA execution | anti-chase<=0.30ATR | SL structure+0.20ATR min0.90 max1.80 risk>=0.40%% | "
        "TP1=1R close50%% lock+0.15R | TP2 S/R 1.5-2.5R fallback2R",
        SentinelV10Strategy.VERSION, config["symbols"], config["interval"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV10Strategy]:
    strategies = [SentinelV10Strategy(symbol) for symbol in symbols]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


_ORIGINAL_LOG_SCAN = TradingBot._log_scan
_LAST_SENTINEL_SCAN: dict[str, dict] = {}


def _sentinel_log_scan(self, symbol, strategy_name, price, signal):
    meta = getattr(signal, "metadata", None) or {}
    if meta.get("strategy") != "SENTINEL_V10":
        return _ORIGINAL_LOG_SCAN(self, symbol, strategy_name, price, signal)
    reason = str(getattr(signal, "reason", "") or "")
    if reason != "5M bar already evaluated" and (meta.get("analysis_15m") or meta.get("setup_5m")):
        _LAST_SENTINEL_SCAN[symbol] = meta
    view = _LAST_SENTINEL_SCAN.get(symbol, meta) if reason == "5M bar already evaluated" else meta
    a = view.get("analysis_15m") or {}
    s = view.get("setup_5m") or {}
    f = view.get("forecast") or a.get("forecast") or {}
    reg = a.get("regime") or {}
    comp = a.get("components") or {}
    blocks = s.get("blocks", []) or []
    candidate = s.get("trigger_candidate") or s.get("candidate") or s.get("trigger") or "-"
    repeat_tag = " | cached=same-5M-bar" if reason == "5M bar already evaluated" else ""
    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL V10] %s px=%.4f sig=%s | 15M setup=%s side=%s score=%s/%s L/S=%s/%s | "
        "EMA20=%s slope=%s RSI/SMA=%s/%s MACDhist=%s dHist=%s KDJ=%s/%s/%s | "
        "S/R=%s/%s loc=%s regime=%s/3 ADX=%s CHOP=%s ATRx=%s | pts[E/R/M/K/L/G]=%s/%s/%s/%s/%s/%s | "
        "FC=%s raw=%s conf=%s%% | 5M candidate=%s trigger=%s reg=%s/3 body=%s closePos=%s distEMA=%s volx=%s chase=%s rawRisk=%s%% slATR=%s | "
        "TP2R=%s source=%s | blocks=%s | %s%s",
        symbol, price, getattr(getattr(signal, "type", None), "value", "hold").upper(),
        a.get("selected_setup", "-") or "-", a.get("direction", "NEUTRAL") or "NEUTRAL", a.get("selected_score", "-"), a.get("score_threshold", "-"), a.get("score_long", "-"), a.get("score_short", "-"),
        a.get("ema20", "-"), a.get("ema20_slope_atr", "-"), a.get("rsi", "-"), a.get("rsi_sma", "-"), a.get("macd_hist", "-"), a.get("macd_hist_delta", "-"), a.get("kdj_k", "-"), a.get("kdj_d", "-"), a.get("kdj_j", "-"),
        a.get("support", "-"), a.get("resistance", "-"), a.get("location", "-"), reg.get("pass_count", "-"), reg.get("adx", "-"), reg.get("chop", "-"), reg.get("atr_ratio", "-"),
        comp.get("ema20", "-"), comp.get("rsi_sma", "-"), comp.get("macd", "-"), comp.get("kdj", "-"), comp.get("location", "-"), comp.get("regime", "-"),
        f.get("side", "-"), f.get("raw", "-"), f.get("confidence", "-"), candidate, s.get("trigger", "-") or "-", s.get("regime_pass", "-"), s.get("body_atr", "-"), s.get("close_pos", "-"), s.get("dist_ema_atr", "-"), s.get("volume_ratio", "-"), s.get("chase_atr", "-"), s.get("raw_risk_pct", "-"), s.get("sl_atr", s.get("raw_sl_atr", "-")),
        view.get("tp2_r_dynamic", "-"), view.get("tp2_source", "-"), ",".join(blocks) or "none", reason, repeat_tag,
    )


_ORIGINAL_RESOLVE_STRATEGY = TradingBot._resolve_strategy_inst


def _sentinel_resolve_strategy(self, strategy_name: str):
    inst = _ORIGINAL_RESOLVE_STRATEGY(self, strategy_name)
    if inst is not None:
        return inst
    raw = str(strategy_name or "")
    bare = raw[:-2] if raw.endswith((":L", ":S")) else raw
    prefixes = ("SentinelV7(", "SentinelV7.1(", "SentinelV8(", "SentinelV8.1(", "SentinelV9(")
    for prefix in prefixes:
        if bare.startswith(prefix) and bare.endswith(")"):
            symbol = bare[len(prefix):-1]
            return self._strategy_map.get(f"SentinelV10({symbol})")
    return None


_ORIGINAL_EXECUTE_SIGNAL = TradingBot._execute_signal


async def _sentinel_execute_signal(self, signal, strategy_name: str, direction: str = "long", candles=None):
    is_v10 = str(strategy_name).startswith("SentinelV10")
    if is_v10:
        meta = signal.metadata or {}
        sl = meta.get("stop_loss")
        if sl:
            try:
                px = float((await self.connector.fetch_ticker(signal.symbol))["last"])
                risk = abs(px - float(sl))
                if risk > 0:
                    rr = float(meta.get("rr_ratio") or SentinelV10Strategy.DYNAMIC_TP_FALLBACK_R)
                    rr = max(SentinelV10Strategy.DYNAMIC_TP_MIN_R, min(SentinelV10Strategy.DYNAMIC_TP_MAX_R, rr))
                    meta["rr_ratio"] = rr
                    meta["take_profit"] = px + rr * risk if direction == "long" else px - rr * risk
                    meta["tp1_price"] = px + SentinelV10Strategy.TP1_R * risk if direction == "long" else px - SentinelV10Strategy.TP1_R * risk
                    signal.metadata = meta
            except Exception as e:
                logging.getLogger("trading_bot").warning("[SENTINEL V10] pre-fill target rebase skipped [%s]: %s", signal.symbol, e)
    result = await _ORIGINAL_EXECUTE_SIGNAL(self, signal, strategy_name, direction=direction, candles=candles)
    if not is_v10:
        return result
    key = f"{signal.symbol}||{strategy_name}"
    pos_obj = getattr(self.risk, "_positions", {}).get(key)
    if pos_obj is None or pos_obj.stop_loss is None:
        return result
    try:
        entry = float(pos_obj.entry_price)
        sl = float(pos_obj.stop_loss)
        risk = abs(entry - sl)
        if risk <= 0:
            return result
        meta = signal.metadata or {}
        rr = float(meta.get("rr_ratio") or SentinelV10Strategy.DYNAMIC_TP_FALLBACK_R)
        rr = max(SentinelV10Strategy.DYNAMIC_TP_MIN_R, min(SentinelV10Strategy.DYNAMIC_TP_MAX_R, rr))
        exact_tp = entry + rr * risk if direction == "long" else entry - rr * risk
        exact_tp1 = entry + SentinelV10Strategy.TP1_R * risk if direction == "long" else entry - SentinelV10Strategy.TP1_R * risk
        pos_obj.take_profit = exact_tp
        meta["take_profit"] = exact_tp
        meta["tp1_price"] = exact_tp1
        meta["rr_ratio"] = rr
        meta["tp2_r_dynamic"] = rr
        signal.metadata = meta
        strategy_inst = self._resolve_strategy_inst(strategy_name)
        if hasattr(strategy_inst, "attach_existing_position"):
            strategy_inst.attach_existing_position(direction, entry, sl, exact_tp)
        if hasattr(strategy_inst, "_tp2_rr_active"):
            strategy_inst._tp2_rr_active = rr
        try:
            await self.connector.set_position_tpsl(signal.symbol, direction, float(pos_obj.amount), sl=sl, tp=exact_tp)
        except Exception as e:
            logging.getLogger("trading_bot").warning("[SENTINEL V10] exact post-fill TP replace failed [%s]: %s", signal.symbol, e)
        logging.getLogger("trading_bot").info(
            "[SENTINEL V10 FILL-SYNC] %s %s entry=%.4f SL=%.4f TP=%.4f exact=%.2fR source=%s FC=%s/%s%%",
            signal.symbol, direction.upper(), entry, sl, exact_tp, rr, meta.get("tp2_source", "FALLBACK_2R"), meta.get("forecast_side", "-"), meta.get("forecast_confidence", "-"),
        )
    except Exception as e:
        logging.getLogger("trading_bot").warning("[SENTINEL V10] post-fill synchronization failed [%s]: %s", signal.symbol, e)
    return result


_ORIGINAL_BUILD_ORDER_CAPTION = TelegramNotifier.build_order_caption


def _sentinel_build_order_caption(self, *args, **kwargs):
    text = _ORIGINAL_BUILD_ORDER_CAPTION(self, *args, **kwargs)
    strategy = kwargs.get("strategy")
    if strategy is None and len(args) >= 5:
        strategy = args[4]
    if "SentinelV10" in str(strategy or "") or "SentinelV10" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🧠 15M : EMA20 + RSI/SMA14 + MACD + KDJ + S/R\n"
            "🌡 Regime : ADX / CHOP / ATR Activity, pass 2-of-3\n"
            "🔭 Forecast : advisory bias/confidence + S/R runner target (not an entry gate)\n"
            "⚡ Entry : confirmed 5M price action\n"
            "🛑 SL : 5M structure + 0.20 ATR (0.90–1.80 ATR; natural risk ≥0.40%)\n"
            "🎯 TP1 : +1.0R close 50% → runner SL +0.15R\n"
            "🎯 TP2 : S/R + Forecast 1.5–2.5R; fallback 2R\n"
            "🏁 Exit : 15M EMA/RSI/MACD 2-of-3; runner may exit on RSI+KDJ/forecast reversal",
        )
    return text


_ORIGINAL_CLASSIFY_EXIT_REASON = signal_state_module.classify_exit_reason


def _sentinel_classify_exit_reason(reason: str, won: bool):
    r = (reason or "").lower()
    if "v10_exit_2of3" in r:
        return ("V10 Momentum Exit 2/3", "↩️")
    if "v10_runner_exit" in r:
        return ("V10 Runner Forecast Exit", "🏁")
    return _ORIGINAL_CLASSIFY_EXIT_REASON(reason, won)


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
                trigger_px = pos_info.get("stop_loss") if trigger == "stop_loss" else pos_info.get("take_profit") if trigger == "take_profit" else price
                trigger_px = float(trigger_px or price)
                side = "sell" if pos_info["side"] == "long" else "buy"
                pos_side = pos_info["side"] if self._hedge_mode else None
                close_order = await self.connector.create_order(sym, side, pos_info["amount"], order_type="limit", price=trigger_px, pos_side=pos_side)
                fill = self._close_fill_info(f"{sym}||{strategy_name}", close_order, trigger_px, pos_info["amount"], 1.0, final=True)
                exit_px = fill["exit_avg_px"]
                pnl = fill["net_pnl"] if fill["net_pnl"] is not None else ((exit_px-pos_info["entry"])*pos_info["amount"] if pos_info["side"] == "long" else (pos_info["entry"]-exit_px)*pos_info["amount"])
                self._record_trade(TradeRecord(timestamp=int(time.time()*1000), symbol=sym, side=side, price=exit_px, amount=fill["exit_sz"], pnl=round(pnl,4), strategy=strategy_name, reason=trigger, paper=True))
                outcome = self._sig.record_outcome(symbol=sym, side=pos_info["side"], entry=pos_info["entry"], exit_price=exit_px, sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"), reason=trigger, strategy=strategy_name, fill=fill)
                self._sig.unlock_strategy(sym, strategy_name)
                self.risk.close_position(sym, strategy=strategy_name)
                strategy_inst = self._resolve_strategy_inst(strategy_name)
                self._on_position_closed(sym, strategy_name, exit_px, trigger, strategy_inst)
                logging.getLogger("trading_bot").info("[SENTINEL PAPER TRIGGER-FILL] %s closed by %s @ %.4f [%s]", sym, trigger, exit_px, strategy_name)
                if self.telegram:
                    self.telegram.notify_trade_closed(sym, outcome, self._sig.summary())
                self._check_cooldown_trigger(pnl, sym)
            except Exception as e:
                logging.getLogger("trading_bot").error("[SENTINEL V10 PAPER] hard-stop precheck failed [%s %s]: %s", strategy_name, sym, e)
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
    "[PRODUCTION] Sentinel V%s installed | V9 retained for rollback | 15M EMA20+RSI/SMA+MACD+KDJ+S/R, regime 2of3 -> 5M PA | "
    "Forecast advisory/runner targeting | fee-aware structure SL | one fill/15M | hard-SL wait=3x5M+fresh event | TP1=1R/50%% lock+0.15R | TP2 S/R 1.5-2.5R fallback2R",
    SentinelV10Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
