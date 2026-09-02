"""Canonical production router for Sentinel V10.1 — Bias-Separated Momentum + Location Core.

Railway starts this file. Production instantiates V10.1 only. V10, V9 and all
older strategy source files remain available for rollback.

V10.1 entry policy:
- Exact Setup Engine keeps PULLBACK / BREAKOUT_RETEST / SWEEP_REVERSAL.
- If no exact setup exists, a qualified Bias Engine can expose
  MOMENTUM_CONTINUATION to the 5M Execution Engine.
- Forecast Engine remains advisory for normal exact setups and is only a
  strong-opposition veto for the fallback bias path.
- Entry engine context is persisted in SignalState and follows the position to
  TP1, TP2, SL and technical/runner exits, including across restarts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time

import run_bot
from trading.bot import TradingBot, TradeRecord
from trading.telegram_notifier import TelegramNotifier
from trading import signal_state as signal_state_module
from trading.strategies.sentinel_v101_strategy import SentinelV101Strategy

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
        "exact=PB>=6 BO-RETEST>=6.5 SWEEP>=7 | bias fallback=score>=7 edge>=1.5 mom>=2/3 room>=0.75ATR | "
        "Forecast advisory; opposing>=65%% veto only fallback | 5M PA execution | anti-chase<=0.30ATR | "
        "SL structure+0.20ATR min0.90 max1.80 risk>=0.40%% | TP1=1R close50%% lock+0.15R | "
        "TP2 S/R 1.5-2.5R fallback2R | engine attribution persisted",
        SentinelV101Strategy.VERSION, config["symbols"], config["interval"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV101Strategy]:
    strategies = [SentinelV101Strategy(symbol) for symbol in symbols]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


# ---------------------------------------------------------------------------
# Stable per-position engine context.
# ---------------------------------------------------------------------------
_V10_ENTRY_CONTEXT: dict[str, dict] = {}


def _v10_context_from_meta(meta: dict) -> dict:
    meta = meta or {}
    fc = meta.get("forecast") or {}
    analysis = meta.get("analysis_15m") or {}
    setup = meta.get("setup_5m") or {}
    setup_engine = meta.get("setup_family") or analysis.get("selected_setup") or "UNKNOWN"
    execution_engine = meta.get("entry_trigger") or setup.get("trigger") or "UNKNOWN"
    setup_mode = meta.get("setup_mode") or analysis.get("setup_mode") or "EXACT_SETUP"
    fc_side = meta.get("forecast_side") or fc.get("side") or "NEUTRAL"
    fc_conf = meta.get("forecast_confidence")
    if fc_conf is None:
        fc_conf = fc.get("confidence")
    try:
        fc_conf = round(float(fc_conf), 1) if fc_conf is not None else None
    except (TypeError, ValueError):
        fc_conf = None
    return {
        "entry_engine": "BIAS_ENGINE+5M_EXECUTION" if setup_mode == "BIAS_FALLBACK" else "SETUP_ENGINE+5M_EXECUTION",
        "setup_mode": str(setup_mode),
        "setup_engine": str(setup_engine),
        "execution_engine": str(execution_engine),
        "forecast_engine": str(fc_side),
        "forecast_confidence": fc_conf,
        "setup_score": meta.get("setup_score") or analysis.get("selected_score"),
        "tp2_r": meta.get("tp2_r_dynamic") or meta.get("rr_ratio"),
        "tp2_source": meta.get("tp2_source") or "FALLBACK_2R",
    }


def _v10_context_lines(ctx: dict) -> list[str]:
    if not ctx:
        return []
    fc_conf = ctx.get("forecast_confidence")
    fc_txt = str(ctx.get("forecast_engine") or "NEUTRAL")
    if fc_conf is not None:
        fc_txt += f" {fc_conf:.1f}%"
    score = ctx.get("setup_score")
    setup_txt = str(ctx.get("setup_engine") or "UNKNOWN")
    if score is not None:
        try:
            setup_txt += f" · score {float(score):.2f}"
        except (TypeError, ValueError):
            pass
    return [
        f"🧭 Entry Path : `{ctx.get('setup_mode') or 'EXACT_SETUP'}`",
        f"🧩 Setup/Bias Engine : `{setup_txt}`",
        f"⚡ Execution Engine : `{ctx.get('execution_engine') or 'UNKNOWN'}`",
        f"🔭 Forecast Engine : `{fc_txt}` _(advisory; fallback veto only if strongly opposite)_",
    ]


def _v10_load_context(sig_state, key: str) -> dict:
    active = getattr(sig_state, "_active", {}).get(key) or {}
    ctx = active.get("engine_context")
    if ctx:
        return dict(ctx)
    pp = getattr(sig_state, "_paper_positions", {}).get(key) or {}
    ctx = pp.get("engine_context")
    if ctx:
        return dict(ctx)
    return dict(_V10_ENTRY_CONTEXT.get(key) or {})


def _v10_persist_context(bot, key: str, ctx: dict, signal=None) -> None:
    if not ctx:
        return
    _V10_ENTRY_CONTEXT[key] = dict(ctx)
    sig = getattr(bot, "_sig", None)
    if sig is None:
        return

    active = getattr(sig, "_active", {}).get(key)
    if isinstance(active, dict):
        active["engine_context"] = dict(ctx)

    pp = getattr(sig, "_paper_positions", {}).get(key)
    if isinstance(pp, dict):
        pp["engine_context"] = dict(ctx)

    if signal is not None:
        fired = getattr(sig, "_fired", [])
        for item in reversed(fired[-20:]):
            if item.get("symbol") == signal.symbol and item.get("strategy") == key.split("||", 1)[1]:
                item.update({
                    "entry_engine": ctx.get("entry_engine"),
                    "setup_mode": ctx.get("setup_mode"),
                    "setup_engine": ctx.get("setup_engine"),
                    "execution_engine": ctx.get("execution_engine"),
                    "forecast_engine": ctx.get("forecast_engine"),
                    "forecast_confidence": ctx.get("forecast_confidence"),
                    "setup_score": ctx.get("setup_score"),
                })
                break
    try:
        sig._save()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scan logging.
# ---------------------------------------------------------------------------
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
    bias_rejects = a.get("bias_rejects") or []
    repeat_tag = " | cached=same-5M-bar" if reason == "5M bar already evaluated" else ""
    logging.getLogger("trading_bot").info(
        "[SCAN SENTINEL V10.1] %s px=%.4f sig=%s | 15M setup=%s mode=%s side=%s score=%s/%s L/S=%s/%s "
        "edge=%s mom=%s/3 room=%s rejects=%s | EMA20=%s slope=%s RSI/SMA=%s/%s MACDhist=%s dHist=%s KDJ=%s/%s/%s | "
        "S/R=%s/%s loc=%s regime=%s/3 ADX=%s CHOP=%s ATRx=%s | pts[E/R/M/K/L/G]=%s/%s/%s/%s/%s/%s | "
        "FC=%s raw=%s conf=%s%% | 5M candidate=%s trigger=%s reg=%s/3 body=%s closePos=%s distEMA=%s volx=%s chase=%s rawRisk=%s%% slATR=%s | "
        "TP2R=%s source=%s | blocks=%s | %s%s",
        symbol, price, getattr(getattr(signal, "type", None), "value", "hold").upper(),
        a.get("selected_setup", "-") or "-", a.get("setup_mode", "NONE"), a.get("direction", "NEUTRAL") or "NEUTRAL",
        a.get("selected_score", "-"), a.get("score_threshold", "-"), a.get("score_long", "-"), a.get("score_short", "-"),
        a.get("bias_score_edge", "-"), a.get("bias_momentum_votes", "-"), a.get("bias_room_atr", "-"), ",".join(bias_rejects) or "none",
        a.get("ema20", "-"), a.get("ema20_slope_atr", "-"), a.get("rsi", "-"), a.get("rsi_sma", "-"), a.get("macd_hist", "-"), a.get("macd_hist_delta", "-"), a.get("kdj_k", "-"), a.get("kdj_d", "-"), a.get("kdj_j", "-"),
        a.get("support", "-"), a.get("resistance", "-"), a.get("location", "-"), reg.get("pass_count", "-"), reg.get("adx", "-"), reg.get("chop", "-"), reg.get("atr_ratio", "-"),
        comp.get("ema20", "-"), comp.get("rsi_sma", "-"), comp.get("macd", "-"), comp.get("kdj", "-"), comp.get("location", "-"), comp.get("regime", "-"),
        f.get("side", "-"), f.get("raw", "-"), f.get("confidence", "-"), candidate, s.get("trigger", "-") or "-", s.get("regime_pass", "-"), s.get("body_atr", "-"), s.get("close_pos", "-"), s.get("dist_ema_atr", "-"), s.get("volume_ratio", "-"), s.get("chase_atr", "-"), s.get("raw_risk_pct", "-"), s.get("sl_atr", s.get("raw_sl_atr", "-")),
        view.get("tp2_r_dynamic", "-"), view.get("tp2_source", "-"), ",".join(blocks) or "none", reason, repeat_tag,
    )


# ---------------------------------------------------------------------------
# Strategy compatibility across deploys.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# V10.1 entry execution + exact fill-sync + context persistence.
# ---------------------------------------------------------------------------
_ORIGINAL_EXECUTE_SIGNAL = TradingBot._execute_signal


async def _sentinel_execute_signal(self, signal, strategy_name: str, direction: str = "long", candles=None):
    is_v10 = str(strategy_name).startswith("SentinelV10")
    ctx = {}
    key = f"{signal.symbol}||{strategy_name}"
    if is_v10:
        meta = signal.metadata or {}
        ctx = _v10_context_from_meta(meta)
        _V10_ENTRY_CONTEXT[key] = dict(ctx)
        sl = meta.get("stop_loss")
        if sl:
            try:
                px = float((await self.connector.fetch_ticker(signal.symbol))["last"])
                risk = abs(px - float(sl))
                if risk > 0:
                    rr = float(meta.get("rr_ratio") or SentinelV101Strategy.DYNAMIC_TP_FALLBACK_R)
                    rr = max(SentinelV101Strategy.DYNAMIC_TP_MIN_R, min(SentinelV101Strategy.DYNAMIC_TP_MAX_R, rr))
                    meta["rr_ratio"] = rr
                    meta["take_profit"] = px + rr * risk if direction == "long" else px - rr * risk
                    meta["tp1_price"] = px + SentinelV101Strategy.TP1_R * risk if direction == "long" else px - SentinelV101Strategy.TP1_R * risk
                    signal.metadata = meta
            except Exception as e:
                logging.getLogger("trading_bot").warning("[SENTINEL V10.1] pre-fill target rebase skipped [%s]: %s", signal.symbol, e)

    result = await _ORIGINAL_EXECUTE_SIGNAL(self, signal, strategy_name, direction=direction, candles=candles)
    if not is_v10:
        return result

    pos_obj = getattr(self.risk, "_positions", {}).get(key)
    if pos_obj is None or pos_obj.stop_loss is None:
        return result

    _v10_persist_context(self, key, ctx, signal=signal)

    try:
        entry = float(pos_obj.entry_price)
        sl = float(pos_obj.stop_loss)
        risk = abs(entry - sl)
        if risk <= 0:
            return result
        meta = signal.metadata or {}
        rr = float(meta.get("rr_ratio") or SentinelV101Strategy.DYNAMIC_TP_FALLBACK_R)
        rr = max(SentinelV101Strategy.DYNAMIC_TP_MIN_R, min(SentinelV101Strategy.DYNAMIC_TP_MAX_R, rr))
        exact_tp = entry + rr * risk if direction == "long" else entry - rr * risk
        exact_tp1 = entry + SentinelV101Strategy.TP1_R * risk if direction == "long" else entry - SentinelV101Strategy.TP1_R * risk
        pos_obj.take_profit = exact_tp
        meta["take_profit"] = exact_tp
        meta["tp1_price"] = exact_tp1
        meta["rr_ratio"] = rr
        meta["tp2_r_dynamic"] = rr
        signal.metadata = meta
        ctx["tp2_r"] = rr
        ctx["tp2_source"] = meta.get("tp2_source") or ctx.get("tp2_source")
        _v10_persist_context(self, key, ctx, signal=signal)

        strategy_inst = self._resolve_strategy_inst(strategy_name)
        if hasattr(strategy_inst, "attach_existing_position"):
            strategy_inst.attach_existing_position(direction, entry, sl, exact_tp)
        if hasattr(strategy_inst, "_tp2_rr_active"):
            strategy_inst._tp2_rr_active = rr
        try:
            await self.connector.set_position_tpsl(signal.symbol, direction, float(pos_obj.amount), sl=sl, tp=exact_tp)
        except Exception as e:
            logging.getLogger("trading_bot").warning("[SENTINEL V10.1] exact post-fill TP replace failed [%s]: %s", signal.symbol, e)
        logging.getLogger("trading_bot").info(
            "[SENTINEL V10.1 FILL-SYNC] %s %s entry=%.4f SL=%.4f TP=%.4f exact=%.2fR source=%s FC=%s/%s%% path=%s setup=%s trigger=%s",
            signal.symbol, direction.upper(), entry, sl, exact_tp, rr, meta.get("tp2_source", "FALLBACK_2R"),
            ctx.get("forecast_engine", "-"), ctx.get("forecast_confidence", "-"), ctx.get("setup_mode", "-"),
            ctx.get("setup_engine", "-"), ctx.get("execution_engine", "-"),
        )
    except Exception as e:
        logging.getLogger("trading_bot").warning("[SENTINEL V10.1] post-fill synchronization failed [%s]: %s", signal.symbol, e)
    return result


# ---------------------------------------------------------------------------
# Persist engine attribution into each final journal outcome.
# ---------------------------------------------------------------------------
_ORIGINAL_RECORD_OUTCOME = signal_state_module.SignalState.record_outcome


def _sentinel_record_outcome(self, *args, **kwargs):
    symbol = kwargs.get("symbol") if "symbol" in kwargs else (args[0] if len(args) > 0 else "")
    strategy = kwargs.get("strategy") if "strategy" in kwargs else (args[7] if len(args) > 7 else "")
    key = f"{symbol}||{strategy}"
    ctx = _v10_load_context(self, key) if str(strategy).startswith("SentinelV10") else {}
    outcome = _ORIGINAL_RECORD_OUTCOME(self, *args, **kwargs)
    if ctx:
        outcome["engine_context"] = dict(ctx)
        outcome["entry_engine"] = ctx.get("entry_engine")
        outcome["setup_mode"] = ctx.get("setup_mode")
        outcome["setup_engine"] = ctx.get("setup_engine")
        outcome["execution_engine"] = ctx.get("execution_engine")
        outcome["forecast_engine"] = ctx.get("forecast_engine")
        outcome["forecast_confidence"] = ctx.get("forecast_confidence")
        outcome["setup_score"] = ctx.get("setup_score")
        try:
            self._save()
        except Exception:
            pass
    return outcome


_ORIGINAL_SUMMARY = signal_state_module.SignalState.summary


def _sentinel_summary(self):
    data = _ORIGINAL_SUMMARY(self)
    groups: dict[str, dict] = {}
    for o in getattr(self, "_outcomes", []):
        if not str(o.get("strategy") or "").startswith("SentinelV10"):
            continue
        setup = o.get("setup_engine") or (o.get("engine_context") or {}).get("setup_engine") or "UNKNOWN"
        d = groups.setdefault(setup, {"trades": 0, "wins": 0, "losses": 0, "total_r": 0.0, "pnl_usd": 0.0})
        d["trades"] += 1
        won = bool(o.get("won", o.get("pnl_r", 0) > 0))
        d["wins" if won else "losses"] += 1
        d["total_r"] += float(o.get("pnl_r") or 0.0)
        d["pnl_usd"] += float(o.get("pnl_usd") or 0.0)
    for d in groups.values():
        d["win_rate"] = round(d["wins"] / d["trades"] * 100.0, 1) if d["trades"] else 0.0
        d["total_r"] = round(d["total_r"], 2)
        d["pnl_usd"] = round(d["pnl_usd"], 2)
    data["v10_engine_stats"] = groups
    return data


# ---------------------------------------------------------------------------
# Telegram: rich attribution at OPEN, TP1, TP2/SL and technical closes.
# ---------------------------------------------------------------------------
_ORIGINAL_BUILD_ORDER_CAPTION = TelegramNotifier.build_order_caption


def _sentinel_build_order_caption(self, *args, **kwargs):
    text = _ORIGINAL_BUILD_ORDER_CAPTION(self, *args, **kwargs)
    strategy = kwargs.get("strategy")
    symbol = kwargs.get("symbol")
    if strategy is None and len(args) >= 5:
        strategy = args[4]
    if symbol is None and len(args) >= 1:
        symbol = args[0]
    if "SentinelV10" in str(strategy or "") or "SentinelV10" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🧠 15M : exact Setup Engine or qualified Bias Fallback\n"
            "📈 Core : EMA20 + RSI/SMA14 + MACD + KDJ + S/R\n"
            "🌡 Regime : ADX / CHOP / ATR Activity, pass 2-of-3\n"
            "🔭 Forecast : advisory; strong opposite veto only for Bias Fallback\n"
            "⚡ Entry : confirmed 5M price action\n"
            "🛑 SL : 5M structure + 0.20 ATR (0.90–1.80 ATR; natural risk ≥0.40%)\n"
            "🎯 TP1 : +1.0R close 50% → runner SL +0.15R\n"
            "🎯 TP2 : S/R + Forecast 1.5–2.5R; fallback 2R\n"
            "🏁 Exit : 15M EMA/RSI/MACD 2-of-3; runner may exit on RSI+KDJ/forecast reversal",
        )
        key = f"{symbol}||{strategy}"
        ctx = _V10_ENTRY_CONTEXT.get(key) or {}
        if ctx:
            text += "\n" + "\n".join(_v10_context_lines(ctx))
    return text


_ORIGINAL_NOTIFY = TelegramNotifier.notify


def _sentinel_notify(self, text: str):
    if text and "SentinelV10" in text and ("Partial Take-Profit" in text or "SL moved to lock profit" in text):
        strat_m = re.search(r"\[(SentinelV10[^\]]+)\]", text)
        sym_m = re.search(r"`([^`]+)`", text)
        if strat_m and sym_m:
            key = f"{sym_m.group(1)}||{strat_m.group(1)}"
            ctx = _V10_ENTRY_CONTEXT.get(key) or {}
            if ctx:
                text += "\n" + "\n".join(_v10_context_lines(ctx))
    return _ORIGINAL_NOTIFY(self, text)


_ORIGINAL_NOTIFY_TRADE_CLOSED = TelegramNotifier.notify_trade_closed


def _sentinel_notify_trade_closed(self, symbol: str, outcome: dict, stats: dict):
    strategy = str(outcome.get("strategy") or "")
    if not strategy.startswith("SentinelV10"):
        return _ORIGINAL_NOTIFY_TRADE_CLOSED(self, symbol, outcome, stats)

    ctx = outcome.get("engine_context") or {}
    won = outcome.get("won", outcome.get("pnl_usd", 0) > 0)
    result_emoji = "✅" if won else "❌"
    label = outcome.get("reason_label", "Position Closed")
    reason_emoji = outcome.get("emoji", "☑️")
    side = str(outcome.get("side") or "").upper()
    entry = float(outcome.get("entry") or 0.0)
    exit_px = float(outcome.get("exit") or 0.0)
    sl = outcome.get("sl")
    tp = outcome.get("tp")
    pnl_r = float(outcome.get("pnl_r") or 0.0)
    pnl_usd = float(outcome.get("pnl_usd") or 0.0)
    fill = outcome.get("fill") or {}
    if fill.get("net_pnl") is not None:
        pnl_usd = float(fill.get("net_pnl") or 0.0)
    sign = "+" if pnl_usd >= 0 else "-"
    sl_txt = f"`{float(sl):,.4f}`" if sl else "—"
    tp_txt = f"`{float(tp):,.4f}`" if tp else "—"
    lines = [
        f"{result_emoji} *{label}* {reason_emoji}",
        f"`{symbol}` {side} · `{strategy}`",
        f"Entry `{entry:,.4f}` → Exit `{exit_px:,.4f}`",
        f"SL {sl_txt} | TP {tp_txt}",
        f"💵 Net P&L `{sign}${abs(pnl_usd):,.4f}` | `{pnl_r:+.2f}R`",
    ]
    lines.extend(_v10_context_lines(ctx))
    lines.append(f"📌 Reason : `{outcome.get('reason', 'closed')}`")
    lines.append("_Engine attribution is stored with this trade for /stats._")
    self.notify("\n".join(lines))


_ORIGINAL_RENDER_STATS = TelegramNotifier._render_stats


def _sentinel_render_stats(self, s: dict) -> str:
    text = _ORIGINAL_RENDER_STATS(self, s)
    groups = s.get("v10_engine_stats") or {}
    if groups:
        lines = [text, "", "—" * 16, "V10.1 · BY SETUP/BIAS ENGINE", "—" * 16]
        for name, d in sorted(groups.items(), key=lambda kv: -kv[1].get("total_r", 0.0)):
            sign = "+" if d.get("pnl_usd", 0.0) >= 0 else "-"
            lines.append(
                f"`{name}` {d.get('trades', 0)} trades · {d.get('win_rate', 0):.1f}%WR · "
                f"`{d.get('total_r', 0):+.2f}R` · `{sign}${abs(d.get('pnl_usd', 0.0)):.2f}`"
            )
        return "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Exit labels.
# ---------------------------------------------------------------------------
_ORIGINAL_CLASSIFY_EXIT_REASON = signal_state_module.classify_exit_reason


def _sentinel_classify_exit_reason(reason: str, won: bool):
    r = (reason or "").lower()
    if "v10_exit_2of3" in r:
        return ("V10.1 Momentum Exit 2/3", "↩️")
    if "v10_runner_exit" in r:
        return ("V10.1 Runner Forecast Exit", "🏁")
    return _ORIGINAL_CLASSIFY_EXIT_REASON(reason, won)


# ---------------------------------------------------------------------------
# PAPER: hard SL/TP first and always notify the final lifecycle event.
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
                key = f"{sym}||{strategy_name}"
                persisted = _v10_load_context(self._sig, key)
                if persisted:
                    _V10_ENTRY_CONTEXT[key] = persisted

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
                _V10_ENTRY_CONTEXT.pop(key, None)
            except Exception as e:
                logging.getLogger("trading_bot").error("[SENTINEL V10.1 PAPER] hard-stop precheck failed [%s %s]: %s", strategy_name, sym, e)
    return await _ORIGINAL_TICK(self)


run_bot.build_config = _build_config
run_bot._make_strategies = _make_strategies
TradingBot._log_scan = _sentinel_log_scan
TradingBot._resolve_strategy_inst = _sentinel_resolve_strategy
TradingBot._execute_signal = _sentinel_execute_signal
TradingBot._tick = _sentinel_tick_hard_stop_first
TelegramNotifier.build_order_caption = _sentinel_build_order_caption
TelegramNotifier.notify = _sentinel_notify
TelegramNotifier.notify_trade_closed = _sentinel_notify_trade_closed
TelegramNotifier._render_stats = _sentinel_render_stats
signal_state_module.SignalState.record_outcome = _sentinel_record_outcome
signal_state_module.SignalState.summary = _sentinel_summary
signal_state_module.classify_exit_reason = _sentinel_classify_exit_reason

logger.warning(
    "[PRODUCTION] Sentinel V%s installed | V10/V9 retained for rollback | exact setup OR qualified bias fallback -> 5M PA | "
    "Forecast advisory except strong-opposition fallback veto | engine attribution persisted | TG OPEN/TP1/TP2/SL/CLOSE enabled | "
    "fee-aware structure SL | one fill/15M | hard-SL wait=3x5M+fresh event | TP1=1R/50%% lock+0.15R | TP2 S/R 1.5-2.5R fallback2R",
    SentinelV101Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
