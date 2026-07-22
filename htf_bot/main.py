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
        self.state: dict = self._load_state()   # symbol -> {last_bar, pos{...}}
        self.trade_log: list = []               # closed trades this process
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
        await self.tg.send_text(
            f"🟢 *{_sym(symbol)} {sig.direction.upper()}* @ `{fill:.6g}`\n"
            f"SL `{sl_f:.6g}` (−1R)  TP `{tp_f:.6g}` (+{self.cfg.tp_r:.0f}R)\n"
            f"Size `{st['pos']['amount']:.6g}` | BE lock at +{self.cfg.be_at_r:.0f}R")

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

    async def _report_close(self, symbol: str, st: dict, hint: str = ""):
        pos = st["pos"]
        st["pos"] = None
        self._save_state()
        pnl = None
        if not self.cfg.paper:
            try:
                rows = await self.client.fetch_trade_history(pos["opened_ms"] - 60_000, [symbol])
                rows = [r for r in rows if r["symbol"] == symbol]
                if rows:
                    pnl = rows[-1]["pnl"]
            except Exception as e:
                logger.warning("[%s] pnl fetch failed: %s", symbol, e)
        if pnl is None and pos.get("risk"):
            # estimate from last known levels (paper mode / history briefly lagging)
            ticker = await self.client.fetch_ticker(symbol)
            px = float(ticker["last"])
            mult = 1 if pos["side"] == S.LONG else -1
            pnl = mult * (px - pos["entry"]) * pos["amount"] if pos["entry"] else 0.0
        r_mult = (pnl / (pos["risk"] * pos["amount"])) if (pnl is not None and pos.get("risk") and pos.get("amount")) else None
        self.trade_log.append({"time": time.time(), "symbol": symbol, "side": pos["side"],
                               "pnl": pnl or 0.0, "hint": hint})
        emoji = "✅" if (pnl or 0) > 0 else "❌"
        r_txt = f" ({r_mult:+.2f}R)" if r_mult is not None else ""
        await self.tg.send_text(
            f"{emoji} *{_sym(symbol)} closed* {hint}\n"
            f"PnL `{(pnl or 0):+.2f}` USDT{r_txt} (from OKX)" if not self.cfg.paper else
            f"{emoji} *{_sym(symbol)} closed* {hint} PnL est `{(pnl or 0):+.2f}` USDT{r_txt}")

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
            since = self.cfg.stats_since_ms()
            rows = []
            if not self.cfg.paper:
                try:
                    rows = await self.client.fetch_trade_history(since, self.cfg.symbols)
                except Exception as e:
                    logger.warning("[STATS] %s", e)
            if not rows:
                rows = [{"symbol": t["symbol"], "pnl": t["pnl"]}
                        for t in self.trade_log if t["time"] * 1000 >= since]
            if not rows:
                await self.tg.send_text(f"_no closed trades since {self.cfg.stats_since_date}_")
                return
            wins = [r for r in rows if r["pnl"] > 0]
            net = sum(r["pnl"] for r in rows)
            by = {}
            for r in rows:
                by.setdefault(r["symbol"], []).append(r["pnl"])
            lines = [f"Trades `{len(rows)}` | WR `{len(wins)/len(rows)*100:.0f}%` | "
                     f"Net `{net:+.2f}` USDT (post-fee, OKX)", ""]
            for s, ps in by.items():
                w = sum(1 for p in ps if p > 0)
                lines.append(f"`{_sym(s)}` {len(ps)} tr {w/len(ps)*100:.0f}%WR `{sum(ps):+.2f}`")
            await self.tg.send_text("📈 *Stats*\n\n" + "\n".join(lines))
        else:
            await self.tg.send_text(f"unknown: {cmd} — /help")


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
