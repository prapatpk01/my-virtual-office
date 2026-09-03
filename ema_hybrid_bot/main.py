"""EMA Hybrid Pro Advanced runtime using the existing OKX execution infrastructure."""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HMA = os.path.join(ROOT, "hma_bot")
if HMA not in sys.path:
    sys.path.insert(0, HMA)

import main_v15 as base

_strategy_path = os.path.join(HERE, "strategy.py")
_strategy_spec = importlib.util.spec_from_file_location("ema_hybrid_bot_strategy", _strategy_path)
if _strategy_spec is None or _strategy_spec.loader is None:
    raise ImportError(f"Cannot load EMA Hybrid strategy: {_strategy_path}")
_strategy_module = importlib.util.module_from_spec(_strategy_spec)
sys.modules[_strategy_spec.name] = _strategy_module
_strategy_spec.loader.exec_module(_strategy_module)
EMAHybridProStrategy = _strategy_module.EMAHybridProStrategy

_LOG = base._LOG


def _ohlcv_to_df(raw):
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = [list(r[:6]) for r in raw if len(r) >= 6]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    return df


def _setup_key_from_trigger(trigger: str) -> str:
    t = str(trigger or "").upper()
    if "EMA5M_CROSS_" in t:
        return "A_EMA_CROSS"
    if "RSI_SMA_" in t:
        return "C_RSI_SMA_REVERSAL"
    if "PULLBACK_" in t:
        return "B_PULLBACK_RECLAIM"
    return "UNKNOWN"


def _setup_label_from_trigger(trigger: str) -> str:
    key = _setup_key_from_trigger(trigger)
    return {
        "A_EMA_CROSS": "A · EMA8/13 CROSS",
        "B_PULLBACK_RECLAIM": "B · PULLBACK RECLAIM/BOS",
        "C_RSI_SMA_REVERSAL": "C · RSI/SMA OB-OS REVERSAL",
    }.get(key, "UNKNOWN")


def _setup_label_from_text(text: str) -> str | None:
    upper = str(text or "").upper()
    if "EMA5M_CROSS_" in upper:
        return "A · EMA8/13 CROSS"
    if "RSI_SMA_" in upper:
        return "C · RSI/SMA OB-OS REVERSAL"
    if "PULLBACK_RECLAIM_" in upper or "PULLBACK_MICRO_BOS_" in upper:
        return "B · PULLBACK RECLAIM/BOS"
    return None


def _ema_runtime_text(text: str) -> str:
    """Rewrite inherited HMA labels and expose the active EMA setup engine."""
    if not isinstance(text, str):
        return text
    replacements = (
        (" | Final TP `", " | TP2 Liquidity/Swing `"),
        ("T1 `+0.6%` → lock `+0.3%` | T2 `+1.0%` → lock `+0.7%`", "TP1 `+1R` → trim `60%` + SL `BE+0.15R` | TP2 `Liquidity/Swing`"),
        ("Stage 1 `+0.7%` → lock `+0.4%` | Stage 2 `+1.1%` → lock `+0.75%`", "TP1 `+1R` → trim `60%` + SL `BE+0.15R` | TP2 `Liquidity/Swing`"),
        ("15M `PULLBACK` → 5M `", "5M Trigger `"),
    )
    for old, new in replacements:
        text = text.replace(old, new)

    setup_label = _setup_label_from_text(text)
    if setup_label and "Setup Engine:" not in text:
        lines = text.splitlines()
        if lines:
            lines.insert(1, f"Setup Engine: `{setup_label}`")
            text = "\n".join(lines)
    return text


async def _await_if_needed(value):
    """Compatibility guard for legacy notifier wrappers that may return None/bool."""
    if inspect.isawaitable(value):
        return await value
    return value


class Bot(base.Bot):
    TP1_R = float(os.getenv("EMA_ADV_TP1_R", "1.0"))
    TP1_TRIM_PCT = float(os.getenv("EMA_ADV_TP1_TRIM_PCT", "0.60"))
    TP1_LOCK_R = float(os.getenv("EMA_ADV_TP1_LOCK_R", "0.15"))

    def __init__(self):
        super().__init__()
        self.strat = EMAHybridProStrategy(self.cfg.strategy_config())
        self._okx_reconnect_lock = asyncio.Lock()
        self._ema_journal_path = os.path.join(self.cfg.state_dir, "ema_hybrid_trades.jsonl")
        self.ema_journal = self._load_ema_journal()

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def ema_text(text: str) -> bool:
            result = await _await_if_needed(previous_send_text(_ema_runtime_text(text)))
            return bool(result)

        async def ema_photo(path: str, caption: str) -> bool:
            result = await _await_if_needed(previous_send_photo(path, _ema_runtime_text(caption)))
            return bool(result)

        self.tg.send_text = ema_text
        self.tg._send_photo = ema_photo

    def _load_ema_journal(self) -> list[dict]:
        rows: list[dict] = []
        try:
            with open(self._ema_journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except (FileNotFoundError, OSError):
            pass
        except json.JSONDecodeError as exc:
            _LOG.warning("[EMA STATS] malformed journal row skipped: %s", exc)
        return rows

    def _append_ema_trade(self, row: dict) -> None:
        try:
            os.makedirs(self.cfg.state_dir, exist_ok=True)
            with open(self._ema_journal_path, "a") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.ema_journal.append(row)
        except OSError as exc:
            _LOG.error("[EMA STATS] journal write failed: %s", exc)

    def _paper_trade_finalize(self, symbol: str, pos: dict, final_order, exit_reason: str) -> dict | None:
        """Write exactly one completed EMA Hybrid trade, including both 2TP legs."""
        if not self.cfg.paper or pos.get("ema_trade_journaled"):
            return None
        entry = float(pos.get("entry") or 0.0)
        initial_amount = float(pos.get("initial_amount") or 0.0)
        if initial_amount <= 0:
            initial_amount = float(pos.get("amount") or 0.0) + float(pos.get("tp1_trimmed") or 0.0)
        tp1_net = float(pos.get("tp1_net_pnl") or 0.0)
        final_net = float(getattr(final_order, "realized_pnl", 0.0) or 0.0) - float(getattr(final_order, "fee_cost", 0.0) or 0.0)
        open_fee = initial_amount * entry * float(getattr(self.cfg, "fee_rate", 0.0) or 0.0)
        net = tp1_net + final_net - open_fee
        trigger = str(pos.get("trigger") or "")
        setup_key = _setup_key_from_trigger(trigger)
        row = {
            "close_ms": int(time.time() * 1000),
            "open_ms": int(pos.get("opened_ms") or 0),
            "symbol": symbol,
            "side": str(pos.get("side") or ""),
            "entry": entry,
            "initial_amount": initial_amount,
            "tp1_done": bool(pos.get("tp1_done")),
            "tp1_trimmed": float(pos.get("tp1_trimmed") or 0.0),
            "tp1_net_pnl": round(tp1_net, 8),
            "final_net_pnl": round(final_net, 8),
            "open_fee": round(open_fee, 8),
            "pnl": round(net, 8),
            "exit_type": exit_reason,
            "setup": setup_key,
            "trigger": trigger,
        }
        self._append_ema_trade(row)
        pos["ema_trade_journaled"] = True
        _LOG.info(
            "[EMA STATS] completed %s %s setup=%s exit=%s net=%.6f",
            symbol, row["side"], setup_key, exit_reason, net,
        )
        return row

    async def _send_exit_alert(self, symbol: str, pos: dict, exit_reason: str, price: float, row: dict | None = None) -> None:
        if pos.get("ema_exit_alerted"):
            return
        setup = _setup_label_from_trigger(str(pos.get("trigger") or ""))
        side = str(pos.get("side") or "?").upper()
        pnl = float((row or {}).get("pnl") or 0.0)
        if exit_reason == "TP2":
            head = "🏁 *TP2 HIT — POSITION CLOSED*"
        elif exit_reason == "TP1_LOCK":
            head = "🔒 *TP1 LOCK HIT — POSITION CLOSED*"
        elif exit_reason == "SL":
            head = "🛑 *STOP LOSS HIT — POSITION CLOSED*"
        else:
            head = "📤 *POSITION CLOSED*"
        pnl_line = f"Net PnL: `{pnl:+.4f}` USDT\n" if row is not None else ""
        await self.tg.send_text(
            f"{head}\n"
            f"Symbol: `{symbol.split('/')[0]}` | Side: `{side}`\n"
            f"Setup Engine: `{setup}`\n"
            f"Trigger: `{str(pos.get('trigger') or '—')}`\n"
            f"Exit: `{exit_reason}` @ `{price:.8g}`\n"
            f"{pnl_line}"
            f"TP1 status: `{'DONE' if pos.get('tp1_done') else 'NOT HIT'}`"
        )
        pos["ema_exit_alerted"] = True

    async def _recover_okx_session(self, reason: str = "") -> None:
        async with self._okx_reconnect_lock:
            exchange = getattr(self.client, "_exchange", None)
            if exchange is None:
                return
            session = getattr(exchange, "session", None)
            if session is not None and not getattr(session, "closed", False):
                return
            _LOG.warning("[OKX] async session closed; reopening%s", f" ({reason})" if reason else "")
            await exchange.open()
            _LOG.info("[OKX] async session reopened")

    @staticmethod
    def _is_closed_session_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "instance was closed by the user" in text
            or "session is closed" in text
            or "client session is closed" in text
        )

    async def _frame(self, symbol, tf, minutes, limit=300):
        now_ms = int(time.time() * 1000)
        try:
            raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)
        except Exception as exc:
            if not self._is_closed_session_error(exc):
                raise
            await self._recover_okx_session(f"{symbol} {tf}")
            raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)

        df = _ohlcv_to_df(raw)
        if df.empty:
            return df
        close_ms = (df.index.astype("int64") // 1_000_000) + minutes * 60_000
        return df[close_ms <= now_ms]

    async def _reconcile_startup(self):
        await super()._reconcile_startup()
        changed = False
        for symbol in self.cfg.symbols:
            pos = (self.state.get(symbol) or {}).get("pos") or {}
            side = str(pos.get("side") or "")
            entry = float(pos.get("entry") or 0.0)
            sl = float(pos.get("sl") or 0.0)
            if entry <= 0 or side not in ("long", "short"):
                continue
            profit_side = (side == "long" and sl > entry) or (side == "short" and 0 < sl < entry)
            if profit_side and not pos.get("tp1_done"):
                pos["tp1_done"] = True
                pos["tp1_recovered"] = True
                changed = True
        if changed:
            self._save_state()

    async def _manage(self, symbol: str, st: dict):
        """EMA 2TP manager with explicit Telegram lifecycle alerts."""
        pos = st.get("pos") or {}
        side = str(pos.get("side") or "").lower()
        entry = float(pos.get("entry") or 0.0)
        risk = float(pos.get("risk") or 0.0)
        tp2 = float(pos.get("tp") or 0.0)

        if side not in ("long", "short") or entry <= 0:
            return await super()._manage(symbol, st)

        live_amount = await self.client.fetch_position_amount(symbol, side)
        if live_amount <= 0:
            if not pos.get("ema_exit_alerted"):
                await self._send_exit_alert(symbol, pos, "EXCHANGE_CLOSED", float(pos.get("best_price") or entry), None)
            return await self._report_close(symbol, st)

        ticker = await self.client.fetch_ticker(symbol)
        price = float((ticker or {}).get("last") or 0.0)
        if price <= 0:
            return await super()._manage(symbol, st)

        active_sl = float(pos.get("sl") or 0.0)
        if self.cfg.paper:
            sl_hit = (side == "long" and active_sl > 0 and price <= active_sl) or (side == "short" and active_sl > 0 and price >= active_sl)
            tp2_hit = (side == "long" and tp2 > 0 and price >= tp2) or (side == "short" and tp2 > 0 and price <= tp2)
            if sl_hit or tp2_hit:
                close_side = "sell" if side == "long" else "buy"
                final_order = await self.client.create_order(symbol, close_side, live_amount, pos_side=side, reduce_only=True)
                exit_reason = "TP2" if tp2_hit else ("TP1_LOCK" if pos.get("tp1_done") else "SL")
                row = self._paper_trade_finalize(symbol, pos, final_order, exit_reason)
                await self._send_exit_alert(symbol, pos, exit_reason, price, row)
                self._save_state()
                return await self._report_close(symbol, st, hint=exit_reason)

        if risk > 0 and not bool(pos.get("tp1_done")):
            tp1 = entry + self.TP1_R*risk if side == "long" else entry - self.TP1_R*risk
            hit = price >= tp1 if side == "long" else price <= tp1
            if hit:
                requested_trim = live_amount * self.TP1_TRIM_PCT
                trim_contracts, trim_base = await self.client.quantize_amount(symbol, requested_trim)
                rem_contracts, _ = await self.client.quantize_amount(symbol, max(live_amount-trim_base, 0.0))
                trimmed = 0.0
                tp1_net = 0.0

                if trim_contracts > 0 and rem_contracts > 0:
                    close_side = "sell" if side == "long" else "buy"
                    order = await self.client.create_order(symbol, close_side, trim_base, pos_side=side, reduce_only=True)
                    trimmed = float(order.amount or trim_base)
                    tp1_net = float(order.realized_pnl or 0.0) - float(order.fee_cost or 0.0)
                    await asyncio.sleep(0.15)
                    live_amount = await self.client.fetch_position_amount(symbol, side)
                else:
                    _LOG.info("[%s] TP1 hit but 60%% partial is below contract-lot constraints; keeping full runner", symbol)

                if live_amount > 0:
                    lock_sl = entry + self.TP1_LOCK_R*risk if side == "long" else entry - self.TP1_LOCK_R*risk
                    synced = await self.client.move_sl_to_breakeven(symbol, side, lock_sl, live_amount, tp_price=(tp2 or None))
                    pos["initial_amount"] = float(pos.get("initial_amount") or (live_amount + trimmed))
                    pos["tp1_done"] = True
                    pos["tp1_price"] = tp1
                    pos["tp1_trimmed"] = trimmed
                    pos["tp1_net_pnl"] = float(pos.get("tp1_net_pnl") or 0.0) + tp1_net
                    pos["amount"] = live_amount
                    pos["sl"] = lock_sl
                    pos["lock_stage"] = 1
                    pos["native_sl_synced_stage"] = 1 if synced else 0
                    self._save_state()
                    trim_msg = f"trimmed `{trimmed:.8g}`" if trimmed > 0 else "partial trim skipped (minimum lot)"
                    setup = _setup_label_from_trigger(str(pos.get("trigger") or ""))
                    await self.tg.send_text(
                        f"✅ *{symbol.split('/')[0]} TP1 reached* `{self.TP1_R:.1f}R`\n"
                        f"Setup Engine: `{setup}`\n"
                        f"Trigger: `{str(pos.get('trigger') or '—')}`\n"
                        f"{trim_msg} | remaining `{live_amount:.8g}`\n"
                        f"SL → `BE+{self.TP1_LOCK_R:.2f}R` = `{lock_sl:.6g}`\n"
                        f"TP2 Liquidity/Swing = `{tp2:.6g}`"
                    )
                    _LOG.info("[%s] TP1 %.2fR hit px=%.8g trim=%.8g remain=%.8g tp1_net=%.6f lock=%.8g tp2=%.8g synced=%s", symbol, self.TP1_R, price, trimmed, live_amount, tp1_net, lock_sl, tp2, synced)
                    return

        self._view[symbol] = (
            f"OPEN {side.upper()} | setup={_setup_key_from_trigger(str(pos.get('trigger') or ''))} | px={price:.8g} | "
            f"SL={float(pos.get('sl') or 0):.8g} | TP1={'DONE' if pos.get('tp1_done') else f'{self.TP1_R:.1f}R'} | "
            f"TP2={tp2:.8g} LIQ/SWING"
        )
        return await super()._manage(symbol, st)

    async def _build_stats_report(self) -> str:
        """EMA-native stats with by-symbol and by-setup attribution."""
        if not self.cfg.paper:
            inherited = await super()._build_stats_report()
            lines = inherited.splitlines()
            if lines and lines[0].startswith("📊"):
                lines = lines[1:]
            return "📊 EMA Hybrid Advanced Stats\n" + "\n".join(lines)

        balance = await self.client.fetch_balance_usdt()
        now = datetime.now(timezone.utc)
        month_start = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
        since_ms = int(self.cfg.stats_since_ms() or 0)
        rows = [r for r in self.ema_journal if int(r.get("close_ms") or 0) >= since_ms]
        month_rows = [r for r in rows if int(r.get("close_ms") or 0) >= month_start]

        def metrics(block: list[dict]):
            wins = sum(1 for r in block if float(r.get("pnl") or 0.0) > 0)
            losses = sum(1 for r in block if float(r.get("pnl") or 0.0) < 0)
            net = sum(float(r.get("pnl") or 0.0) for r in block)
            gross_win = sum(float(r.get("pnl") or 0.0) for r in block if float(r.get("pnl") or 0.0) > 0)
            gross_loss = abs(sum(float(r.get("pnl") or 0.0) for r in block if float(r.get("pnl") or 0.0) < 0))
            pf = gross_win / gross_loss if gross_loss > 1e-12 else (float("inf") if gross_win > 0 else 0.0)
            wr = (wins / len(block) * 100.0) if block else 0.0
            return wins, losses, net, pf, wr

        def row_setup_key(r: dict) -> str:
            explicit = str(r.get("setup") or "")
            if explicit in {"A_EMA_CROSS", "B_PULLBACK_RECLAIM", "C_RSI_SMA_REVERSAL"}:
                return explicit
            return _setup_key_from_trigger(str(r.get("trigger") or ""))

        mw, ml, mnet, mpf, mwr = metrics(month_rows)
        sw, sl, snet, spf, swr = metrics(rows)
        open_lines = []
        for symbol in self.cfg.symbols:
            pos = (self.state.get(symbol) or {}).get("pos") or {}
            if pos:
                setup = _setup_label_from_trigger(str(pos.get("trigger") or ""))
                open_lines.append(
                    f"📌 {symbol.split('/')[0]} {str(pos.get('side') or '?').upper()} @ {float(pos.get('entry') or 0):.6g} | "
                    f"{setup} | TP1 {'DONE' if pos.get('tp1_done') else 'WAIT'} | TP2 {float(pos.get('tp') or 0):.6g}"
                )

        def pf_text(v: float) -> str:
            return "∞" if v == float("inf") else f"{v:.2f}"

        month_label = now.strftime("%b %Y")
        since_label = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if since_ms else "ALL"
        sep = "――――――――――――――――"
        by_symbol = []
        for symbol in self.cfg.symbols:
            block = [r for r in month_rows if r.get("symbol") == symbol]
            if not block:
                continue
            w, l, net, pf, wr = metrics(block)
            by_symbol.append(f"{symbol.split('/')[0]:5s} {len(block)} trades | {wr:.0f}% WR | PF {pf_text(pf)} | ${net:+.2f}")

        by_setup = []
        setup_names = (
            ("A_EMA_CROSS", "A EMA CROSS"),
            ("B_PULLBACK_RECLAIM", "B PULLBACK"),
            ("C_RSI_SMA_REVERSAL", "C RSI/SMA REV"),
        )
        for key, label in setup_names:
            block = [r for r in month_rows if row_setup_key(r) == key]
            if not block:
                continue
            w, l, net, pf, wr = metrics(block)
            by_setup.append(f"{label:14s} {len(block)} | {wr:.0f}% WR | PF {pf_text(pf)} | ${net:+.2f}")

        return "\n".join([
            "📊 EMA Hybrid Advanced Stats",
            "",
            f"💰 Balance: ${balance:.2f}",
            *(open_lines or ["📌 No open positions"]),
            "Source: PAPER EMA Hybrid 2TP journal",
            "",
            sep,
            f"OVERALL — {month_label}",
            sep,
            f"Trades   : {len(month_rows)} ({mw}W / {ml}L)",
            f"Win rate : {mwr:.0f}%" if month_rows else "Win rate : —",
            f"Profit Factor : {pf_text(mpf)}" if month_rows else "Profit Factor : —",
            f"Net PnL  : ${mnet:+.2f}",
            "Exit tags : " + (", ".join(f"{k} {sum(1 for r in month_rows if r.get('exit_type') == k)}" for k in ("TP2", "TP1_LOCK", "SL")) if month_rows else "—"),
            "",
            sep,
            f"BY SYMBOL — {month_label}",
            sep,
            *(by_symbol or ["(no completed EMA Hybrid trades this month)"]),
            "",
            sep,
            f"BY SETUP — {month_label}",
            sep,
            *(by_setup or ["(no completed A/B/C setup trades this month)"]),
            "",
            sep,
            f"SINCE {since_label}",
            sep,
            f"Trades   : {len(rows)} ({sw}W / {sl}L)",
            f"Win rate : {swr:.0f}%" if rows else "Win rate : —",
            f"Profit Factor : {pf_text(spf)}" if rows else "Profit Factor : —",
            f"Net PnL  : ${snet:+.2f}",
        ])

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"EMA HYBRID POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df15["close"].iloc[-1]) if len(df15) else 0.0
            self._view[symbol] = f"15M px={px:.6g} | {self.strat.entry_status(df4h, df1h, df15, df5)}"
        except Exception as exc:
            self._view[symbol] = f"EMA Hybrid view error: {str(exc)[:140]}"

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== EMA HYBRID PRO MULTI-SETUP MTF [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===",
            "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols,
            self.cfg.margin_per_position_usd, self.cfg.leverage, self.cfg.max_positions, balance,
        )
        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            mode = "PAPER" if self.cfg.paper else "LIVE"
            await self.tg.send_text(
                f"📈 *EMA Hybrid Pro Multi-Setup — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                "15M Bias → 5M Multi-Setup Execution\n"
                f"15M Bias: Close vs SMA{self.strat.SMA_LEN} + RSI{self.strat.RSI_LEN} side of `{self.strat.BIAS_RSI_MID:.0f}`\n"
                f"A: EMA{self.strat.EMA_FAST}/{self.strat.EMA_SLOW} fresh cross\n"
                "B: Pullback to EMA13 → reclaim / micro BOS\n"
                f"C: RSI{self.strat.RSI5_LEN}/SMA{self.strat.RSI5_SMA_LEN} reversal from OS `≤{self.strat.RSI_OS:.0f}` / OB `≥{self.strat.RSI_OB:.0f}`\n"
                f"5M Quality: ADX `≥{self.strat.ADX_MIN:.0f}` | CHOP `≤{self.strat.CHOP_MAX:.0f}`\n"
                f"SL: `5M structure + {self.strat.SL_BUFFER_ATR:.2f} ATR`\n"
                f"TP1: `+{self.TP1_R:.1f}R` → trim `{self.TP1_TRIM_PCT*100:.0f}%` → SL `BE+{self.TP1_LOCK_R:.2f}R`\n"
                f"TP2: `next 5M liquidity/swing target` with room `≥{self.strat.TP2_MIN_RR:.1f}R`\n"
                "Telegram: Entry + Setup Engine + TP1 + TP2/SL/TP1_LOCK alerts\n"
                "PAPER entries: `24/7` | LIVE entries: `24/5` | Open positions managed: `24/7`"
            )
        _LOG.info("EMA Hybrid Multi-Setup startup complete: A/B/C attribution + lifecycle Telegram alerts active")


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
