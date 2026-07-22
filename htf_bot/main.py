"""HTF pullback bot — live entry point (MODE=htf).

One strategy, one file: 4H EMA20/50 trend -> 1H EMA20 pullback-reclaim entry
-> swing/ATR stop -> TP 3R with break-even at +1R. All signal math lives in
strategy.py, which the backtest imports too — live and backtest cannot
diverge.

Reuses signal_regime_bot's battle-tested ExchangeClient (OKX hedge-mode,
fill resolution, native SL/TP attach, BE re-arm) and TelegramNotifier via
sys.path — both are self-contained modules with no config import.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as _signal
import sys
import time

import pandas as pd

# htf_bot's own modules FIRST (both trees have a config.py — ours must win),
# THEN append signal_regime_bot for its self-contained infra modules only.
from config import Config                           # noqa: E402  (htf_bot's own)
import strategy as S                                # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "signal_regime_bot"))
from exchange_client import ExchangeClient          # noqa: E402
from telegram_notifier import TelegramNotifier      # noqa: E402
try:
    from chart_engine import build_entry_chart       # noqa: E402  (regime bot's)
except Exception:  # matplotlib/mplfinance missing -> alerts fall back to text
    build_entry_chart = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout, force=True)
logger = logging.getLogger("htf")


def _sym(symbol: str) -> str:
    return symbol.split("/")[0]


class Bot:
    def __init__(self):
        self.cfg = Config()
        self.client = ExchangeClient(
            api_key=self.cfg.okx_api_key, api_secret=self.cfg.okx_secret,
            passphrase=self.cfg.okx_passphrase, paper=self.cfg.paper,
            leverage=self.cfg.leverage, margin_mode=self.cfg.margin_mode,
            fee_rate=self.cfg.fee_rate)
        self.tg = TelegramNotifier(self.cfg.telegram_token, self.cfg.telegram_chat_id)
        self._state_path = os.path.join(self.cfg.state_dir, "htf_state.json")
        self._journal_path = os.path.join(self.cfg.state_dir, "trade_journal.jsonl")
        self.state: dict = self._load_state()   # symbol -> {last_bar, pos{...}}
        # persistent close-journal: one line per closed trade with the exit
        # type (TP/BE/SL) the bot observed — survives restarts, and is the
        # ONLY source of the TP/BE/SL breakdown (OKX has no "which target"
        # concept). Trade COUNT / WR / PnL always come from OKX, never here.
        self.journal: list = self._load_journal()
        self._journaled_close_ms = {(e["symbol"], int(e["close_ms"] // 60000))
                                    for e in self.journal}   # dedup key: symbol+minute
        self._view: dict = {s: "starting…" for s in self.cfg.symbols}  # per-symbol view line
        self._tg_offset = 0
        self._running = False
        self._last_status_ts = 0.0

    # ── state persistence (restart-safe) ─────────────────────────────────────

    def _load_state(self) -> dict:
        try:
            with open(self._state_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_state(self) -> None:
        os.makedirs(self.cfg.state_dir, exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self._state_path)

    def _sym_state(self, symbol: str) -> dict:
        return self.state.setdefault(symbol, {"last_bar": None, "pos": None})

    def open_position_count(self) -> int:
        return sum(1 for st in self.state.values() if st.get("pos"))

    # ── close journal (exit-type breakdown source) ───────────────────────────

    def _load_journal(self) -> list:
        out = []
        try:
            with open(self._journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except (FileNotFoundError, OSError):
            pass
        except json.JSONDecodeError:
            logger.warning("[JOURNAL] corrupt line(s) skipped")
        return out

    def _journal_add(self, symbol: str, side: str, pnl: float, exit_type: str,
                     close_ms: int) -> None:
        """Append one closed trade. Deduped by (symbol, close-minute) so a
        restart-time backfill can't double-count a trade already journaled."""
        key = (symbol, int(close_ms // 60000))
        if key in self._journaled_close_ms:
            return
        entry = {"close_ms": int(close_ms), "symbol": symbol, "side": side,
                 "pnl": round(float(pnl), 4), "exit_type": exit_type}
        try:
            os.makedirs(self.cfg.state_dir, exist_ok=True)
            with open(self._journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning("[JOURNAL] write failed: %s", e)
        self.journal.append(entry)
        self._journaled_close_ms.add(key)

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper:
            if not await self.client.ensure_hedge_mode():
                raise RuntimeError("Could not confirm OKX hedge mode.")
        balance = await self.client.fetch_balance_usdt()
        logger.info("=== HTF pullback bot [%s] symbols=%s risk=%.1f%% balance=%.2f ===",
                    "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols,
                    self.cfg.risk_per_trade * 100, balance)
        await self._reconcile_startup()
        self._running = True
        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            await self.tg.send_text(
                f"🤖 *HTF bot started* [{'PAPER' if self.cfg.paper else 'LIVE'}]\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Risk `{self.cfg.risk_per_trade*100:.1f}%`/trade\n"
                f"Strategy: 4H EMA20/50 trend → 1H EMA20 pullback → TP 3R, BE @1R")

    async def _reconcile_startup(self):
        """A tracked position that no longer exists on OKX -> clear it; an
        OKX position we don't track -> alert (it still has its native SL/TP)."""
        for symbol in self.cfg.symbols:
            st = self._sym_state(symbol)
            pos = st.get("pos")
            for side in ("long", "short"):
                amt = await self.client.fetch_position_amount(symbol, side)
                tracked = pos is not None and pos["side"] == side
                if amt > 0 and not tracked:
                    await self.tg.send_text(
                        f"⚠️ `{_sym(symbol)}` {side.upper()} position exists on OKX but "
                        f"isn't tracked (opened before a restart?). Its exchange-side "
                        f"SL/TP still protect it; the bot won't double-enter this symbol.")
                    st["pos"] = {"side": side, "entry": 0.0, "sl": 0.0, "tp": 0.0,
                                 "risk": 0.0, "amount": amt, "be_done": True,
                                 "opened_ms": int(time.time() * 1000), "adopted": True}
                if tracked and amt <= 0:
                    logger.info("[%s] tracked %s position gone from OKX — clearing", symbol, side)
                    st["pos"] = None
        self._save_state()

    async def run_forever(self):
        while self._running:
            for symbol in self.cfg.symbols:
                try:
                    await self._process(symbol)
                except Exception as e:
                    logger.error("[%s] unhandled: %s", symbol, e, exc_info=True)
                    try:
                        await self.tg.send_text(f"❌ `{_sym(symbol)}` error: {str(e)[:150]}")
                    except Exception:
                        pass
            self._maybe_status_log()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def stop(self):
        self._running = False
        await self.client.close()

    # ── per-symbol tick ──────────────────────────────────────────────────────

    async def _process(self, symbol: str):
        st = self._sym_state(symbol)
        if st.get("pos"):
            await self._manage(symbol, st)
        else:
            await self._look_for_entry(symbol, st)

    async def _frames(self, symbol: str):
        now_ms = int(time.time() * 1000)
        h1 = S.drop_unclosed(S.ohlcv_to_df(
            await self.client.fetch_ohlcv(symbol, "1h", limit=300)), 1, now_ms)
        h4 = S.drop_unclosed(S.ohlcv_to_df(
            await self.client.fetch_ohlcv(symbol, "4h", limit=250)), 4, now_ms)
        return h1, h4

    def _set_view(self, symbol: str, h1, h4):
        """One-line 'what is the bot watching' per symbol — refreshed every
        poll from the same frames the entry check uses."""
        trend = S.trend_direction(h4)
        tlabel = {1: "4H↑LONG-only", -1: "4H↓SHORT-only", 0: "4H no-trend"}[trend]
        e20 = float(S.ema(h1["close"], 20).iloc[-1])
        av = float(S.atr(h1, 14).iloc[-1])
        px = float(h1["close"].iloc[-1])
        dist_atr = (px - e20) / av if av > 0 else 0.0
        if S.commodity_halted(symbol, pd.Timestamp.now(tz="UTC")):
            why = "HALT (metal weekend)"
        elif trend == 0:
            why = "no 4H trend"
        elif self.open_position_count() >= self.cfg.max_positions:
            why = f"max {self.cfg.max_positions} positions open"
        else:
            # what the pullback trigger is waiting for
            side_ok = "above" if trend == 1 else "below"
            near = abs(dist_atr) <= 0.15
            if (trend == 1 and px > e20) or (trend == -1 and px < e20):
                why = (f"waiting pullback→EMA20 (px {side_ok} EMA20 by {abs(dist_atr):.2f}ATR)"
                       if not near else "at EMA20 — waiting close-back confirm")
            else:
                why = f"px on wrong side of EMA20 ({dist_atr:+.2f}ATR) — waiting reclaim"
        self._view[symbol] = f"{tlabel} | 1H px={px:.6g} EMA20={e20:.6g} | {why}"

    async def _look_for_entry(self, symbol: str, st: dict):
        h1, h4 = await self._frames(symbol)
        if len(h1) < 60 or len(h4) < 55:
            self._view[symbol] = f"warming up ({len(h1)}×1H, {len(h4)}×4H)"
            return
        self._set_view(symbol, h1, h4)   # refresh view every poll (pre-gate)

        bar_key = h1.index[-1].isoformat()
        if st.get("last_bar") == bar_key:
            return
        st["last_bar"] = bar_key      # one evaluation per closed 1H bar
        self._save_state()

        if S.commodity_halted(symbol, pd.Timestamp.now(tz="UTC")):
            return
        if self.open_position_count() >= self.cfg.max_positions:
            return
        sig = S.entry_signal(h1, S.trend_direction(h4),
                             min_body_atr=self.cfg.min_body_atr)
        if sig is None:
            return

        ticker = await self.client.fetch_ticker(symbol)
        entry_ref = float(ticker["last"])
        sl, tp, dist = S.plan_stop_target(
            h1, sig.direction, entry_ref, sig.atr1h,
            swing_n=self.cfg.swing_n, sl_buf_atr=self.cfg.sl_buf_atr,
            min_sl_atr=self.cfg.min_sl_atr, min_sl_pct=self.cfg.min_sl_pct,
            tp_r=self.cfg.tp_r)
        balance = await self.client.fetch_balance_usdt()
        amount = (balance * self.cfg.risk_per_trade) / dist
        if amount * entry_ref < 5:
            logger.info("[%s] size too small (%.2f USDT notional) — skip", symbol, amount * entry_ref)
            return

        side = "buy" if sig.direction == S.LONG else "sell"
        try:
            order = await self.client.create_order(
                symbol, side, amount, pos_side=sig.direction,
                tp_price=tp, sl_price=sl)
        except Exception as e:
            logger.error("[%s] order failed: %s", symbol, e)
            await self.tg.send_text(f"❌ `{_sym(symbol)}` entry order failed: {str(e)[:150]}")
            return

        fill = order.avg_price or entry_ref
        # re-anchor SL/TP to the actual fill so R stays exact
        sl_f, tp_f, dist_f = S.plan_stop_target(
            h1, sig.direction, fill, sig.atr1h,
            swing_n=self.cfg.swing_n, sl_buf_atr=self.cfg.sl_buf_atr,
            min_sl_atr=self.cfg.min_sl_atr, min_sl_pct=self.cfg.min_sl_pct,
            tp_r=self.cfg.tp_r)
        st["pos"] = {"side": sig.direction, "entry": fill, "sl": sl_f, "tp": tp_f,
                     "risk": dist_f, "amount": order.amount or amount,
                     "be_done": False, "opened_ms": int(time.time() * 1000),
                     "entry_fee": order.fee_cost}
        self._save_state()
        logger.info("[%s] OPEN %s @ %.6g sl=%.6g tp=%.6g risk/unit=%.6g",
                    symbol, sig.direction.upper(), fill, sl_f, tp_f, dist_f)
        caption = (
            f"🟢 *{_sym(symbol)} {sig.direction.upper()}* @ `{fill:.6g}`\n"
            f"SL `{sl_f:.6g}` (−1R)  TP `{tp_f:.6g}` (+{self.cfg.tp_r:.0f}R)\n"
            f"Size `{st['pos']['amount']:.6g}` | BE lock at +{self.cfg.be_at_r:.0f}R\n"
            f"4H {'↑LONG' if sig.direction==S.LONG else '↓SHORT'}-trend → 1H EMA20 pullback")
        # technical chart like the regime bot: 1H candles + EMA20/50 + entry/SL/TP
        chart = self._build_chart(symbol, h1, sig.direction, fill, sl_f, tp_f)
        if chart:
            await self.tg._send_photo(chart, caption)
        else:
            await self.tg.send_text(caption)

    def _build_chart(self, symbol, h1, direction, entry, sl, tp):
        """1H candlestick + EMA20/50 + entry/SL/TP lines (reuses the regime
        bot's chart engine). Single-target here, so tp1==tp2==tp. Returns a
        file path or None; never raises."""
        if build_entry_chart is None:
            return None
        try:
            return build_entry_chart(
                symbol, h1, direction.upper(), entry, sl, tp, tp,
                ema_fast_len=20, ema_slow_len=50, tf_label="1H")
        except Exception as e:
            logger.warning("[%s] chart build failed: %s", symbol, e)
            return None

    async def _manage(self, symbol: str, st: dict):
        pos = st["pos"]
        side = pos["side"]
        amt = await self.client.fetch_position_amount(symbol, side)
        if amt <= 0:
            await self._report_close(symbol, st)
            return
        ticker = await self.client.fetch_ticker(symbol)
        price = float(ticker["last"])
        longp = side == S.LONG

        # safety net: if price is beyond SL/TP but the position still exists
        # (native algo failed/detached), close at market.
        if pos["sl"] and ((price <= pos["sl"]) if longp else (price >= pos["sl"])):
            await self._close_market(symbol, st, "SL")
            return
        if pos["tp"] and ((price >= pos["tp"]) if longp else (price <= pos["tp"])):
            await self._close_market(symbol, st, "TP")
            return

        if not pos["be_done"] and pos["risk"] > 0:
            r_now = ((price - pos["entry"]) if longp else (pos["entry"] - price)) / pos["risk"]
            if r_now >= self.cfg.be_at_r:
                pos["sl"] = pos["entry"]
                pos["be_done"] = True
                self._save_state()
                ok = await self.client.move_sl_to_breakeven(
                    symbol, side, pos["entry"], amt, tp_price=pos["tp"])
                await self.tg.send_text(
                    f"🔒 `{_sym(symbol)}` +{self.cfg.be_at_r:.0f}R reached — SL moved to "
                    f"entry `{pos['entry']:.6g}`{'' if ok else ' (exchange re-arm FAILED — watching locally)'}")

    async def _close_market(self, symbol: str, st: dict, why: str):
        pos = st["pos"]
        side = "sell" if pos["side"] == S.LONG else "buy"
        try:
            await self.client.create_order(symbol, side, pos["amount"],
                                           pos_side=pos["side"], reduce_only=True)
        except Exception as e:
            if "position" not in str(e).lower():
                logger.warning("[%s] market close failed: %s", symbol, e)
        await self._report_close(symbol, st, hint=why)

    def _classify_exit(self, pos: dict, hint: str, close_px: float, pnl: float) -> str:
        """TP / BE / SL / UNTRACKED for this single-TP trade. A bot-driven
        close carries a `hint`; an exchange-side (native TP/SL) or offline
        close is classified by comparing the ACTUAL close price to the stored
        tp/entry/sl (single-TP ⇒ close price is clean, no partial to skew it).
        Adopted positions with no known levels stay UNTRACKED."""
        if hint in ("TP", "SL", "BE"):
            return hint
        if pos.get("adopted") or not pos.get("entry") or not pos.get("risk"):
            return "UNTRACKED"
        entry, sl, tp, risk = pos["entry"], pos["sl"], pos["tp"], pos["risk"]
        tol = 0.30 * risk
        if close_px > 0:
            if abs(close_px - tp) <= tol:
                return "TP"
            if pos.get("be_done") and abs(close_px - entry) <= tol:
                return "BE"
            if abs(close_px - sl) <= tol:
                return "SL"
        # fall back to sign of realized pnl
        return "TP" if pnl > 0 else "SL"

    async def _report_close(self, symbol: str, st: dict, hint: str = ""):
        pos = st["pos"]
        st["pos"] = None
        self._save_state()
        pnl, close_px, close_ms = None, 0.0, int(time.time() * 1000)
        if not self.cfg.paper:
            try:
                rows = await self.client.fetch_trade_history(pos["opened_ms"] - 60_000, [symbol])
                rows = [r for r in rows if r["symbol"] == symbol]
                if rows:
                    r = rows[-1]
                    pnl = r["pnl"]
                    close_px = r.get("close_avg_px", 0.0)
                    close_ms = r.get("close_time_ms", close_ms)
            except Exception as e:
                logger.warning("[%s] pnl fetch failed: %s", symbol, e)
        if pnl is None:
            # paper mode / history briefly lagging: estimate from last price
            ticker = await self.client.fetch_ticker(symbol)
            close_px = float(ticker["last"])
            mult = 1 if pos["side"] == S.LONG else -1
            pnl = mult * (close_px - pos["entry"]) * pos["amount"] if pos.get("entry") else 0.0
        exit_type = self._classify_exit(pos, hint, close_px, pnl)
        self._journal_add(symbol, pos["side"], pnl, exit_type, close_ms)
        r_mult = (pnl / (pos["risk"] * pos["amount"])) if (pos.get("risk") and pos.get("amount")) else None
        emoji = "✅" if pnl > 0 else "❌"
        r_txt = f" ({r_mult:+.2f}R)" if r_mult is not None else ""
        src = "from OKX" if not self.cfg.paper else "est"
        await self.tg.send_text(
            f"{emoji} *{_sym(symbol)} closed* [{exit_type}]\n"
            f"PnL `{pnl:+.2f}` USDT{r_txt} ({src})")

    # ── status / telegram ────────────────────────────────────────────────────

    def _view_line(self, symbol: str) -> str:
        """OPEN position summary if in a trade, else the 'what am I watching'
        view line (trend + distance-to-EMA20 + what it's waiting for)."""
        st = self.state.get(symbol) or {}
        pos = st.get("pos")
        if pos:
            return (f"OPEN {pos['side'].upper()} @ {pos['entry']:.6g} "
                    f"SL {pos['sl']:.6g} TP {pos['tp']:.6g}"
                    f"{' 🔒BE' if pos.get('be_done') else ''}")
        return self._view.get(symbol, "flat")

    def _maybe_status_log(self):
        now = time.time()
        if now - self._last_status_ts < self.cfg.status_log_interval_sec:
            return
        self._last_status_ts = now
        logger.info("── VIEW %d symbols ──", len(self.cfg.symbols))
        for symbol in self.cfg.symbols:
            logger.info("  %-16s %s", _sym(symbol), self._view_line(symbol))

    async def _command_loop(self):
        while self._running:
            try:
                updates = await self.tg.get_updates(self._tg_offset + 1)
                for u in updates:
                    self._tg_offset = max(self._tg_offset, int(u.get("update_id", 0)))
                    msg = u.get("message") or {}
                    if str((msg.get("chat") or {}).get("id", "")) != str(self.tg.chat_id):
                        continue
                    text = (msg.get("text") or "").strip().lower().split("@")[0]
                    if text.startswith("/"):
                        await self._handle_cmd(text)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[TG] %s", e)
                await asyncio.sleep(5)

    async def _handle_cmd(self, cmd: str):
        if cmd in ("/help", "/start"):
            await self.tg.send_text(
                "🤖 *HTF Pullback Bot*\n/status — สถานะทุก symbol\n"
                "/stats — ผลเทรด (จาก OKX, post-fee)\n/help — เมนูนี้")
        elif cmd == "/status":
            lines = [f"`{_sym(symbol)}` {self._view_line(symbol)}"
                     for symbol in self.cfg.symbols]
            await self.tg.send_text("📡 *Status*\n\n" + "\n".join(lines))
        elif cmd == "/stats":
            # plain text (no markdown) per spec: separators + emoji, so
            # symbols/numbers never get mangled by Telegram's Markdown parser.
            await self.tg._send_message(await self._build_stats_report(), _markdown=False)
        else:
            await self.tg.send_text(f"unknown: {cmd} — /help")

    @staticmethod
    def _month_bounds(now_ms: int) -> tuple:
        """(this_month_start_ms, prev_month_start_ms, prev_month_label) in UTC."""
        import datetime as dt
        now = dt.datetime.fromtimestamp(now_ms / 1000, tz=dt.timezone.utc)
        m0 = dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
        py, pm = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        p0 = dt.datetime(py, pm, 1, tzinfo=dt.timezone.utc)
        return int(m0.timestamp() * 1000), int(p0.timestamp() * 1000), p0.strftime("%b")

    def _match_exit_types(self, okx_rows: list) -> dict:
        """One-to-one match OKX-closed trades to journal entries by nearest
        close time (same symbol, within 3 min), each journal entry consumed at
        most once — so two same-symbol trades minutes apart can't both borrow
        one journal row. Returns {id(row): exit_type} for matched rows."""
        pool = list(self.journal)
        used = [False] * len(pool)
        out = {}
        # nearest-first so tight pairs bind before looser ones steal a match
        for row in sorted(okx_rows, key=lambda r: r.get("close_time_ms", 0)):
            cms = row.get("close_time_ms", 0)
            best_j, best_d = -1, 3 * 60_000 + 1
            for j, e in enumerate(pool):
                if used[j] or e["symbol"] != row["symbol"]:
                    continue
                d = abs(int(e["close_ms"]) - cms)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                used[best_j] = True
                out[id(row)] = pool[best_j]["exit_type"]
        return out

    async def _build_stats_report(self) -> str:
        """Three sections, plain text, exactly per spec:
        OVERALL (current UTC month, resets on the 1st) with TP/BE/SL/Untracked
        broken out — counts/WR/PnL from OKX, breakdown from the local journal,
        denominators = OKX total (never the journal's); BY SYMBOL (all-time);
        LAST 5. Falls back to the journal with a warning if OKX is unreachable."""
        since = self.cfg.stats_since_ms()
        now_ms = int(time.time() * 1000)
        m0, p0, prev_lbl = self._month_bounds(now_ms)

        okx_ok = True
        rows = []
        if not self.cfg.paper:
            try:
                rows = await self.client.fetch_trade_history(since, self.cfg.symbols)
            except Exception as e:
                logger.warning("[STATS] OKX history fetch failed: %s", e)
                okx_ok = False
        if not rows and (self.cfg.paper or not okx_ok):
            rows = [{"symbol": e["symbol"], "side": e.get("side", ""), "pnl": e["pnl"],
                     "close_time_ms": e["close_ms"], "_journal": True}
                    for e in self.journal if e["close_ms"] >= since]

        balance = await self.client.fetch_balance_usdt()
        open_lines = [f"📌 {_sym(s)} {p['side'].upper()} @ {p['entry']:.6g}"
                      for s in self.cfg.symbols if (p := (self.state.get(s) or {}).get("pos"))]
        header = (f"💰 Balance: ${balance:.2f}\n"
                  + ("\n".join(open_lines) if open_lines else "📌 No open positions"))
        if not okx_ok:
            header = "⚠️ OKX history unavailable — showing local journal\n" + header
        sep = "――――――――――――――――――"
        if not rows:
            return header + f"\n\n(no closed trades since {self.cfg.stats_since_date})"

        # ── ช่อง 1: OVERALL — current month only ──
        month = [r for r in rows if r.get("close_time_ms", 0) >= m0]
        total = len(month)
        wins = sum(1 for r in month if r["pnl"] > 0)
        net = sum(r["pnl"] for r in month)
        prev_net = sum(r["pnl"] for r in rows if p0 <= r.get("close_time_ms", 0) < m0)
        cnt = {"TP": 0, "BE": 0, "SL": 0}
        matched = self._match_exit_types([r for r in month if not r.get("_journal")])
        tracked = 0
        for r in month:
            et = matched.get(id(r))
            if et in cnt:
                cnt[et] += 1
                tracked += 1
        untracked = total - tracked

        def pct(n):
            return f"{n}/{total} ({n / total * 100:.0f}%)" if total else "0/0"
        lines = [header, "", sep, "OVERALL (this month, OKX)", sep,
                 f"Trades   : {total}  ({wins}W / {total - wins}L)",
                 f"Win rate : {wins / total * 100:.0f}%" if total else "Win rate : —",
                 f"TP hit   : {pct(cnt['TP'])}   BE : {pct(cnt['BE'])}   SL : {pct(cnt['SL'])}"]
        if untracked:
            lines.append(f"Untracked: {untracked}/{total} (closed while bot offline)")
        lines.append(f"Net PnL  : ${net:+.2f}  (post-fee, from OKX)")
        lines.append(f"{prev_lbl} PnL  : ${prev_net:+.2f}")

        # ── ช่อง 2: BY SYMBOL — all-time ──
        lines += ["", sep, "BY SYMBOL (all-time)", sep]
        by = {}
        for r in rows:
            by.setdefault(r["symbol"], []).append(r["pnl"])
        for s in self.cfg.symbols:
            ps = by.get(s, [])
            if not ps:
                lines.append(f"{_sym(s):5s} 0 trades")
                continue
            w = sum(1 for p in ps if p > 0)
            lines.append(f"{_sym(s):5s} {len(ps)} trades  {w / len(ps) * 100:.0f}%WR  ${sum(ps):+.2f}")
        allp = [p for ps in by.values() for p in ps]
        if allp:
            wa = sum(1 for p in allp if p > 0)
            lines += ["――――――――",
                      f"TOTAL {len(allp)} trades  {wa / len(allp) * 100:.0f}%WR  ${sum(allp):+.2f}"]

        # ── ช่อง 3: LAST 5 TRADES ──
        lines += ["", sep, "LAST 5 TRADES", sep]
        now = time.time()
        for i, r in enumerate(sorted(rows, key=lambda x: -x.get("close_time_ms", 0))[:5], 1):
            age = now - r.get("close_time_ms", now_ms) / 1000
            age_lbl = f"{age / 3600:.1f}h ago" if age < 86400 else f"{age / 86400:.1f}d ago"
            e = "✅" if r["pnl"] > 0 else "❌"
            side = (r.get("side") or "").upper()
            lines.append(f"{i}. {e} {_sym(r['symbol'])} {side} ${r['pnl']:+.2f} — {age_lbl}")
        return "\n".join(lines)


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(_signal, sig_name),
                                    lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
