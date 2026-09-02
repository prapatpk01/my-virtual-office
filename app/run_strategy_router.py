"""Canonical production router for Sentinel V9 — Scored Setup Execution.

Production is rolled back to the original V9 trading logic. V10/V10.1/V10.2
strategy sources remain untouched for later comparison or rollback.

V9 entry path:
- 15M scored PB/LQ/BO/REV analysis (Pine-v6.2-inspired).
- 5M Sentinel price-action execution.
- V8.1-derived fee-aware structure risk / anti-chase.
- TP1 +1R close 50%, runner SL +0.15R.
- TP2 dynamic 1.5..2.5R from structure/Fib, fallback 2R.

Telegram lifecycle attribution is retained for new V9 trades. Positions opened
by V10 or older Sentinel versions are mapped to the V9 lifecycle manager so a
deploy cannot orphan an existing position.
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
        "[PRODUCTION CONFIG] Sentinel V%s ROLLBACK | symbols=%s | scan=%ss closed-bars-only | "
        "15M scored PB/LQ/BO/REV | mins PB6 LQ6 BOdirect7 BOretest6.5 REV7.5 | "
        "5M execution ADX>=12 CHOP<64 ATRx>=0.65 | anti-chase<=0.30ATR | "
        "SL structure+0.18ATR min0.90 max1.80ATR natural-risk>=0.40%% | "
        "TP1=1R close50%% lock+0.15R | TP2 dynamic1.5-2.5R fallback2R",
        SentinelV9Strategy.VERSION, config["symbols"], config["interval"],
    )
    return config


def _make_strategies(symbols: list[str], config: dict) -> list[SentinelV9Strategy]:
    strategies = [SentinelV9Strategy(symbol) for symbol in symbols]
    if not strategies:
        raise RuntimeError("SENTINEL_SYMBOLS/SIMPLE_PRECISION_SYMBOLS/SYMBOLS is empty")
    return strategies


# ---------------------------------------------------------------------------
# Stable per-position V9 setup attribution.
# ---------------------------------------------------------------------------
_V9_ENTRY_CONTEXT: dict[str, dict] = {}


def _context_from_v9_meta(meta: dict) -> dict:
    meta = meta or {}
    a = meta.get("analysis_15m") or {}
    s = meta.get("setup_5m") or {}
    return {
        "entry_engine": "V9_SETUP_ENGINE+5M_EXECUTION",
        "setup_engine": str(meta.get("setup_family") or a.get("selected_setup") or "UNKNOWN"),
        "execution_engine": str(meta.get("entry_trigger") or s.get("trigger") or "UNKNOWN"),
        "setup_score": meta.get("setup_score") if meta.get("setup_score") is not None else a.get("selected_score"),
        "tp2_r": meta.get("tp2_r_dynamic") or meta.get("rr_ratio"),
        "tp2_source": meta.get("tp2_source") or "FALLBACK_2R",
    }


def _context_lines(ctx: dict) -> list[str]:
    if not ctx:
        return []
    setup = str(ctx.get("setup_engine") or "UNKNOWN")
    score = ctx.get("setup_score")
    if score is not None:
        try:
            setup += f" · score {float(score):.2f}"
        except (TypeError, ValueError):
            pass
    return [
        f"🧩 V9 Setup Engine : `{setup}`",
        f"⚡ 5M Execution : `{ctx.get('execution_engine') or 'UNKNOWN'}`",
    ]


def _load_context(sig_state, key: str) -> dict:
    active = getattr(sig_state, "_active", {}).get(key) or {}
    if active.get("engine_context"):
        return dict(active["engine_context"])
    pp = getattr(sig_state, "_paper_positions", {}).get(key) or {}
    if pp.get("engine_context"):
        return dict(pp["engine_context"])
    return dict(_V9_ENTRY_CONTEXT.get(key) or {})


def _persist_context(bot, key: str, ctx: dict, signal=None) -> None:
    if not ctx:
        return
    _V9_ENTRY_CONTEXT[key] = dict(ctx)
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
                    "setup_engine": ctx.get("setup_engine"),
                    "execution_engine": ctx.get("execution_engine"),
                    "setup_score": ctx.get("setup_score"),
                })
                break
    try:
        sig._save()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# V9 scan diagnostics.
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
        "[SCAN SENTINEL V9] %s px=%.4f sig=%s | 15M setup=%s side=%s score=%s/%s L/S=%s/%s | "
        "T=%s QL/QS=%s/%s Struct=%s Loc=%s roomL/S=%s/%s ADX15=%s CHOP15=%s RSI=%s/%s HMA=%s | "
        "pts[T/Q/S/L/M]=%s/%s/%s/%s/%s | 5M gate=%s ADX=%s CHOP=%s ATRx=%s candidate=%s trigger=%s "
        "body=%s closePos=%s distEMA=%s volx=%s chase=%s rawRisk=%s%% slATR=%s strict=%s | "
        "TP2R=%s source=%s | blocks=%s | %s%s",
        symbol, price, getattr(getattr(signal, "type", None), "value", "hold").upper(),
        a.get("selected_setup", "-") or "-", a.get("direction", "NEUTRAL") or "NEUTRAL",
        a.get("selected_score", "-"), a.get("score_threshold", "-"), a.get("score_long", "-"), a.get("score_short", "-"),
        a.get("trend", "-"), a.get("trend_quality_long", "-"), a.get("trend_quality_short", "-"),
        a.get("structure", "-"), a.get("location", "-"), a.get("room_long_atr", "-"), a.get("room_short_atr", "-"),
        a.get("adx", "-"), a.get("chop", "-"), a.get("rsi", "-"), a.get("rsi_sma", "-"), a.get("hma_slope_atr", "-"),
        comp.get("trend", "-"), comp.get("quality", "-"), comp.get("structure", "-"), comp.get("location", "-"), comp.get("momentum", "-"),
        gate, s.get("adx", "-"), s.get("chop", "-"), s.get("atr_ratio", "-"), candidate, s.get("trigger", "-") or "-",
        s.get("body_atr", "-"), s.get("close_pos", "-"), s.get("dist_ema_atr", "-"), s.get("volume_ratio", "-"),
        s.get("chase_atr", "-"), s.get("raw_risk_pct", "-"), s.get("sl_atr", s.get("raw_sl_atr", "-")),
        "Y" if s.get("strict_mode") else "N", view.get("tp2_r_dynamic", "-"), view.get("tp2_source", "-"),
        ",".join(blocks) or "none", reason, repeat_tag,
    )


# ---------------------------------------------------------------------------
# Lifecycle compatibility across Sentinel versions.
# ---------------------------------------------------------------------------
_ORIGINAL_RESOLVE_STRATEGY = TradingBot._resolve_strategy_inst


def _sentinel_resolve_strategy(self, strategy_name: str):
    inst = _ORIGINAL_RESOLVE_STRATEGY(self, strategy_name)
    if inst is not None:
        return inst
    raw = str(strategy_name or "")
    bare = raw[:-2] if raw.endswith((":L", ":S")) else raw
    prefixes = (
        "SentinelV7(", "SentinelV7.1(", "SentinelV8(", "SentinelV8.1(",
        "SentinelV10(", "SentinelV10.1(", "SentinelV10.2(",
    )
    for prefix in prefixes:
        if bare.startswith(prefix) and bare.endswith(")"):
            symbol = bare[len(prefix):-1]
            return self._strategy_map.get(f"SentinelV9({symbol})")
    return None


# ---------------------------------------------------------------------------
# V9 fill synchronization + context persistence.
# ---------------------------------------------------------------------------
_ORIGINAL_EXECUTE_SIGNAL = TradingBot._execute_signal


async def _sentinel_execute_signal(self, signal, strategy_name: str, direction: str = "long", candles=None):
    is_v9 = str(strategy_name).startswith("SentinelV9")
    key = f"{signal.symbol}||{strategy_name}"
    ctx = {}
    if is_v9:
        meta = signal.metadata or {}
        ctx = _context_from_v9_meta(meta)
        _V9_ENTRY_CONTEXT[key] = dict(ctx)
        sl = meta.get("stop_loss")
        if sl:
            try:
                px = float((await self.connector.fetch_ticker(signal.symbol))["last"])
                risk = abs(px - float(sl))
                if risk > 0:
                    rr = float(meta.get("rr_ratio") or SentinelV9Strategy.DYNAMIC_TP_FALLBACK_R)
                    rr = max(SentinelV9Strategy.DYNAMIC_TP_MIN_R, min(SentinelV9Strategy.DYNAMIC_TP_MAX_R, rr))
                    tp1r = float(meta.get("tp1_r") or SentinelV9Strategy.TP1_R)
                    meta["rr_ratio"] = rr
                    meta["take_profit"] = px + rr * risk if direction == "long" else px - rr * risk
                    meta["tp1_price"] = px + tp1r * risk if direction == "long" else px - tp1r * risk
                    signal.metadata = meta
            except Exception as e:
                logging.getLogger("trading_bot").warning("[SENTINEL V9] pre-fill target rebase skipped [%s]: %s", signal.symbol, e)

    result = await _ORIGINAL_EXECUTE_SIGNAL(self, signal, strategy_name, direction=direction, candles=candles)
    if not is_v9:
        return result

    pos_obj = getattr(self.risk, "_positions", {}).get(key)
    if pos_obj is None or pos_obj.stop_loss is None:
        return result
    _persist_context(self, key, ctx, signal=signal)

    try:
        entry = float(pos_obj.entry_price)
        sl = float(pos_obj.stop_loss)
        actual_risk = abs(entry - sl)
        if actual_risk <= 0:
            return result
        meta = signal.metadata or {}
        rr = float(meta.get("rr_ratio") or SentinelV9Strategy.DYNAMIC_TP_FALLBACK_R)
        rr = max(SentinelV9Strategy.DYNAMIC_TP_MIN_R, min(SentinelV9Strategy.DYNAMIC_TP_MAX_R, rr))
        exact_tp = entry + rr * actual_risk if direction == "long" else entry - rr * actual_risk
        exact_tp1 = entry + SentinelV9Strategy.TP1_R * actual_risk if direction == "long" else entry - SentinelV9Strategy.TP1_R * actual_risk
        pos_obj.take_profit = exact_tp
        meta["take_profit"] = exact_tp
        meta["tp1_price"] = exact_tp1
        meta["rr_ratio"] = rr
        meta["tp2_r_dynamic"] = rr
        signal.metadata = meta
        ctx["tp2_r"] = rr
        ctx["tp2_source"] = meta.get("tp2_source") or ctx.get("tp2_source")
        _persist_context(self, key, ctx, signal=signal)

        strategy_inst = self._resolve_strategy_inst(strategy_name)
        if hasattr(strategy_inst, "attach_existing_position"):
            strategy_inst.attach_existing_position(direction, entry, sl, exact_tp)
        try:
            await self.connector.set_position_tpsl(signal.symbol, direction, float(pos_obj.amount), sl=sl, tp=exact_tp)
        except Exception as e:
            logging.getLogger("trading_bot").warning("[SENTINEL V9] exact post-fill TP replace failed [%s]: %s", signal.symbol, e)
        logging.getLogger("trading_bot").info(
            "[SENTINEL V9 FILL-SYNC] %s %s entry=%.4f SL=%.4f TP=%.4f exact=%.2fR source=%s setup=%s trigger=%s",
            signal.symbol, direction.upper(), entry, sl, exact_tp, rr, meta.get("tp2_source", "FALLBACK_2R"),
            ctx.get("setup_engine", "-"), ctx.get("execution_engine", "-"),
        )
    except Exception as e:
        logging.getLogger("trading_bot").warning("[SENTINEL V9] post-fill synchronization failed [%s]: %s", signal.symbol, e)
    return result


# ---------------------------------------------------------------------------
# Persist attribution in final journal outcomes / paper stats.
# ---------------------------------------------------------------------------
_ORIGINAL_RECORD_OUTCOME = signal_state_module.SignalState.record_outcome


def _sentinel_record_outcome(self, *args, **kwargs):
    symbol = kwargs.get("symbol") if "symbol" in kwargs else (args[0] if len(args) > 0 else "")
    strategy = kwargs.get("strategy") if "strategy" in kwargs else (args[7] if len(args) > 7 else "")
    key = f"{symbol}||{strategy}"
    ctx = _load_context(self, key) if str(strategy).startswith("SentinelV") else {}
    outcome = _ORIGINAL_RECORD_OUTCOME(self, *args, **kwargs)
    if ctx:
        outcome["engine_context"] = dict(ctx)
        for field in ("entry_engine", "setup_engine", "execution_engine", "setup_score"):
            outcome[field] = ctx.get(field)
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
        if not str(o.get("strategy") or "").startswith("SentinelV9"):
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
    data["v9_engine_stats"] = groups
    return data


# ---------------------------------------------------------------------------
# Telegram lifecycle — OPEN / TP1 / TP2 / SL / technical close.
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
    if "SentinelV9" in str(strategy or "") or "SentinelV9" in text:
        text = text.replace(
            "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
            "🧠 15M : V9 scored PB/LQ/BO/REV (Trend/Quality/Structure/Location/Momentum)\n"
            "⚡ Entry : confirmed 5M price action\n"
            "🛑 SL : 5M structure + 0.18 ATR (0.90–1.80 ATR; natural risk ≥0.40%)\n"
            "🎯 TP1 : +1.0R close 50% → runner SL +0.15R\n"
            "🎯 TP2 : dynamic 1.5–2.5R from 15M structure/Fib; fallback 2R\n"
            "🏁 Early exit : confirmed opposite qualified 15M scored setup",
        )
        ctx = _V9_ENTRY_CONTEXT.get(f"{symbol}||{strategy}") or {}
        if ctx:
            text += "\n" + "\n".join(_context_lines(ctx))
    return text


_ORIGINAL_NOTIFY = TelegramNotifier.notify


def _sentinel_notify(self, text: str):
    if text and "SentinelV9" in text and ("Partial Take-Profit" in text or "SL moved to lock profit" in text):
        strat_m = re.search(r"\[(SentinelV9[^\]]+)\]", text)
        sym_m = re.search(r"`([^`]+)`", text)
        if strat_m and sym_m:
            ctx = _V9_ENTRY_CONTEXT.get(f"{sym_m.group(1)}||{strat_m.group(1)}") or {}
            if ctx:
                text += "\n" + "\n".join(_context_lines(ctx))
    return _ORIGINAL_NOTIFY(self, text)


_ORIGINAL_NOTIFY_TRADE_CLOSED = TelegramNotifier.notify_trade_closed


def _sentinel_notify_trade_closed(self, symbol: str, outcome: dict, stats: dict):
    strategy = str(outcome.get("strategy") or "")
    if not strategy.startswith("SentinelV"):
        return _ORIGINAL_NOTIFY_TRADE_CLOSED(self, symbol, outcome, stats)
    ctx = outcome.get("engine_context") or {}
    won = outcome.get("won", outcome.get("pnl_usd", 0) > 0)
    result_emoji = "✅" if won else "❌"
    label = outcome.get("reason_label", "Position Closed")
    side = str(outcome.get("side") or "").upper()
    entry = float(outcome.get("entry") or 0.0)
    exit_px = float(outcome.get("exit") or 0.0)
    pnl_r = float(outcome.get("pnl_r") or 0.0)
    pnl_usd = float(outcome.get("pnl_usd") or 0.0)
    fill = outcome.get("fill") or {}
    if fill.get("net_pnl") is not None:
        pnl_usd = float(fill.get("net_pnl") or 0.0)
    sign = "+" if pnl_usd >= 0 else "-"
    lines = [
        f"{result_emoji} *{label}*",
        f"`{symbol}` {side} · `{strategy}`",
        f"Entry `{entry:,.4f}` → Exit `{exit_px:,.4f}`",
        f"💵 Net P&L `{sign}${abs(pnl_usd):,.4f}` | `{pnl_r:+.2f}R`",
    ]
    if ctx:
        lines.extend(_context_lines(ctx))
    lines.append(f"📌 Reason : `{outcome.get('reason', 'closed')}`")
    self.notify("\n".join(lines))


_ORIGINAL_RENDER_STATS = TelegramNotifier._render_stats


def _sentinel_render_stats(self, s: dict) -> str:
    text = _ORIGINAL_RENDER_STATS(self, s)
    groups = s.get("v9_engine_stats") or {}
    if not groups:
        return text
    lines = [text, "", "—" * 16, "V9 · BY SETUP ENGINE", "—" * 16]
    for name, d in sorted(groups.items(), key=lambda kv: -kv[1].get("total_r", 0.0)):
        sign = "+" if d.get("pnl_usd", 0.0) >= 0 else "-"
        lines.append(
            f"`{name}` {d.get('trades', 0)} trades · {d.get('win_rate', 0):.1f}%WR · "
            f"`{d.get('total_r', 0):+.2f}R` · `{sign}${abs(d.get('pnl_usd', 0.0)):.2f}`"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exit labels.
# ---------------------------------------------------------------------------
_ORIGINAL_CLASSIFY_EXIT_REASON = signal_state_module.classify_exit_reason


def _sentinel_classify_exit_reason(reason: str, won: bool):
    if "bias_flip_exit" in (reason or "").lower():
        return ("15M Scored Setup Flip Exit", "↩️")
    return _ORIGINAL_CLASSIFY_EXIT_REASON(reason, won)


# ---------------------------------------------------------------------------
# PAPER: hard SL/TP has priority and fills at the configured trigger level.
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
                persisted = _load_context(self._sig, key)
                if persisted:
                    _V9_ENTRY_CONTEXT[key] = persisted
                price = float((await self.connector.fetch_ticker(sym))["last"])
                trigger = self.risk.check_stops(sym, price, strategy=strategy_name)
                if not trigger:
                    continue
                trigger_px = pos_info.get("stop_loss") if trigger == "stop_loss" else pos_info.get("take_profit") if trigger == "take_profit" else price
                trigger_px = float(trigger_px or price)
                side = "sell" if pos_info["side"] == "long" else "buy"
                pos_side = pos_info["side"] if self._hedge_mode else None
                close_order = await self.connector.create_order(
                    sym, side, pos_info["amount"], order_type="limit", price=trigger_px, pos_side=pos_side
                )
                fill = self._close_fill_info(key, close_order, trigger_px, pos_info["amount"], 1.0, final=True)
                exit_px = fill["exit_avg_px"]
                pnl = fill["net_pnl"] if fill["net_pnl"] is not None else (
                    (exit_px - pos_info["entry"]) * pos_info["amount"] if pos_info["side"] == "long"
                    else (pos_info["entry"] - exit_px) * pos_info["amount"]
                )
                self._record_trade(TradeRecord(
                    timestamp=int(time.time() * 1000), symbol=sym, side=side, price=exit_px,
                    amount=fill["exit_sz"], pnl=round(pnl, 4), strategy=strategy_name,
                    reason=trigger, paper=True,
                ))
                outcome = self._sig.record_outcome(
                    symbol=sym, side=pos_info["side"], entry=pos_info["entry"], exit_price=exit_px,
                    sl=pos_info.get("stop_loss"), tp=pos_info.get("take_profit"), reason=trigger,
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
                _V9_ENTRY_CONTEXT.pop(key, None)
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
TelegramNotifier.notify = _sentinel_notify
TelegramNotifier.notify_trade_closed = _sentinel_notify_trade_closed
TelegramNotifier._render_stats = _sentinel_render_stats
signal_state_module.SignalState.record_outcome = _sentinel_record_outcome
signal_state_module.SignalState.summary = _sentinel_summary
signal_state_module.classify_exit_reason = _sentinel_classify_exit_reason

logger.warning(
    "[PRODUCTION] Sentinel V%s restored | V10.x retained for rollback/comparison | "
    "15M PB/LQ/BO/REV score -> 5M PA | TG OPEN/TP1/TP2/SL/CLOSE enabled | "
    "existing older/V10 positions mapped to V9 lifecycle manager",
    SentinelV9Strategy.VERSION,
)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
