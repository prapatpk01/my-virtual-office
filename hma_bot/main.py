"""MTF Structure Trend bot — live entry point (MODE=hma).

One 15m strategy (strategy.py, the user's HMA16TrendFollowStrategy) driving the
regime bot's battle-tested infrastructure: OKX ExchangeClient (hedge mode, native
SL/TP), TelegramNotifier, chart engine, and the OKX-accurate /stats + persistent
close-journal. 4H EMA20/50+HMA16 selects direction; 1H ADX+CHOP Q confirms quality; 15M market structure/micro-BOS triggers entry. Profit locks: +0.6%->SL +0.3%, +1.0%->SL +0.7%, runner to +1.5%; initial SL -1.5%.

⚠️ Backtested NEGATIVE on BTC+XAU (see config.py); shipped at user direction.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as _signal
import sys
import time

import numpy as np
import pandas as pd

from config import Config                              # noqa: E402  (hma_bot's own)
import strategy as S                                   # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "signal_regime_bot"))
from exchange_client import ExchangeClient             # noqa: E402
from telegram_notifier import TelegramNotifier         # noqa: E402
try:
    from chart_engine import build_entry_chart          # noqa: E402
except Exception:
    build_entry_chart = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout, force=True)
logger = logging.getLogger("hma")

_TF_MIN = 15


def _sym(symbol: str) -> str:
    return symbol.split("/")[0]


def _ohlcv_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype(float)


def _drop_unclosed(df: pd.DataFrame, now_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    close_ms = (df.index.as_unit("ns").asi8 // 1_000_000) + _TF_MIN * 60_000
    return df[close_ms <= now_ms]


def _is_metal(symbol: str) -> bool:
    u = symbol.upper()
    return any(k in u for k in ("XAU", "XAG", "GOLD", "SILVER", "XPT", "XPD"))


def _metal_halted(symbol: str, ts: pd.Timestamp) -> bool:
    """Weekend halt for metals (Fri 21:00 UTC → Sun 21:00 UTC), same window the
    regime bot uses. Crypto trades 24/7."""
    if not _is_metal(symbol):
        return False
    wd, hr = ts.weekday(), ts.hour
    return (wd == 4 and hr >= 21) or wd == 5 or (wd == 6 and hr < 21)


class Bot:
    def __init__(self):
        self.cfg = Config()
        self.strat = S.MTFStructureStrategy(self.cfg.strategy_config())
        self.client = ExchangeClient(
            api_key=self.cfg.okx_api_key, api_secret=self.cfg.okx_secret,
            passphrase=self.cfg.okx_passphrase, paper=self.cfg.paper,
            leverage=self.cfg.leverage, margin_mode=self.cfg.margin_mode,
            fee_rate=self.cfg.fee_rate)
        self.tg = TelegramNotifier(self.cfg.telegram_token, self.cfg.telegram_chat_id)
        self._state_path = os.path.join(self.cfg.state_dir, "hma_state.json")
        self._journal_path = os.path.join(self.cfg.state_dir, "trade_journal.jsonl")
        self.state: dict = self._load_state()
        self.journal: list = self._load_journal()
        self._journaled_close_ms = {(e["symbol"], int(e["close_ms"] // 60000))
                                    for e in self.journal}
        self._view: dict = {s: "starting…" for s in self.cfg.symbols}
        self._cooldown_until: dict = {}
        self._tg_offset = 0
        self._running = False
        self._last_status_ts = 0.0

    # ── state / journal ──────────────────────────────────────────────────────

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

    def _journal_add(self, symbol, side, pnl, exit_type, close_ms):
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
        logger.info("=== HMA16 bot [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===",
                    "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols,
                    self.cfg.margin_per_position_usd, self.cfg.leverage,
                    self.cfg.max_positions, balance)
        await self._reconcile_startup()
        self._running = True
        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            await self.tg.send_text(
                f"🤖 *MTF Structure Trend bot started* [{'PAPER' if self.cfg.paper else 'LIVE'}]\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n"
                f"Strategy: 15M EMA20/50 trend → HMA16 flip → TP/SL {self.cfg.take_profit_pct*100:.1f}% "
                f"+ HMA flip exit")

    async def _reconcile_startup(self):
        for symbol in self.cfg.symbols:
            st = self._sym_state(symbol)
            pos = st.get("pos")
            for side in ("long", "short"):
                amt = await self.client.fetch_position_amount(symbol, side)
                tracked = pos is not None and pos["side"] == side
                if amt > 0 and not tracked:
                    await self.tg.send_text(
                        f"⚠️ `{_sym(symbol)}` {side.upper()} exists on OKX but isn't tracked "
                        f"(pre-restart?). Its exchange SL/TP still protect it; no double-entry.")
                    st["pos"] = {"side": side, "entry": 0.0, "sl": 0.0, "tp": 0.0,
                                 "risk": 0.0, "amount": amt, "opened_ms": int(time.time() * 1000),
                                 "adopted": True}
                if tracked and amt <= 0:
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

    async def _frame(self, symbol: str, tf: str, minutes: int, limit: int = 300) -> pd.DataFrame:
        now_ms = int(time.time() * 1000)
        raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)
        df = _ohlcv_to_df(raw)
        if df.empty:
            return df
        close_ms = (df.index.as_unit("ns").asi8 // 1_000_000) + minutes * 60_000
        return df[close_ms <= now_ms]

    async def _frames(self, symbol: str):
        df15, df1h, df4h = await asyncio.gather(
            self._frame(symbol, "15m", 15, 320),
            self._frame(symbol, "1h", 60, 240),
            self._frame(symbol, "4h", 240, 220),
        )
        return df15, df1h, df4h

    def _set_view(self, symbol: str, df15, df1h, df4h):
        try:
            t = self.strat.trend_state_4h(df4h)
            q = self.strat.quality_state_1h(df1h)
            dmi = "DMI+" if q.plus_di > q.minus_di else "DMI-"
            if _metal_halted(symbol, pd.Timestamp.now(tz="UTC")):
                why = "HALT"
            elif self.open_position_count() >= self.cfg.max_positions:
                why = f"MAX {self.cfg.max_positions}"
            elif t.trend == S.Trend.NEUTRAL:
                why = "WAIT 4H trend"
            elif q.q < self.cfg.min_trend_quality:
                why = f"WAIT Q<{self.cfg.min_trend_quality:.0f}"
            else:
                why = "WAIT 15M structure BOS"
            px = float(df15["close"].iloc[-1]) if len(df15) else 0.0
            self._view[symbol] = (
                f"4H={t.trend.value} HMA={'UP' if t.hma_state>0 else 'DOWN' if t.hma_state<0 else 'FLAT'} "
                f"| 1H Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f} {dmi} "
                f"| 15M px={px:.6g} | {why}"
            )
        except Exception as exc:
            self._view[symbol] = f"view error: {str(exc)[:80]}"

    async def _look_for_entry(self, symbol: str, st: dict):
        df15, df1h, df4h = await self._frames(symbol)
        if len(df15) < 80 or len(df1h) < 60 or len(df4h) < 60:
            self._view[symbol] = f"warming up 15M={len(df15)} 1H={len(df1h)} 4H={len(df4h)}"
            return

        self._set_view(symbol, df15, df1h, df4h)

        bar_key = df15.index[-1].isoformat()
        if st.get("last_bar") == bar_key:
            return
        st["last_bar"] = bar_key
        self._save_state()

        if _metal_halted(symbol, pd.Timestamp.now(tz="UTC")):
            return
        if self.open_position_count() >= self.cfg.max_positions:
            return
        if time.time() < self._cooldown_until.get(symbol, 0):
            return

        sig = self.strat.generate_entry(df4h, df1h, df15, has_open_position=False)
        if sig is None:
            return

        direction = "long" if sig.side == S.Side.LONG else "short"
        ticker = await self.client.fetch_ticker(symbol)
        fill_ref = float(ticker["last"])

        sl = fill_ref * (1 - self.cfg.stop_loss_pct) if direction == "long" else fill_ref * (1 + self.cfg.stop_loss_pct)
        tp = fill_ref * (1 + self.cfg.take_profit_pct) if direction == "long" else fill_ref * (1 - self.cfg.take_profit_pct)

        balance = await self.client.fetch_balance_usdt()
        required_margin = float(self.cfg.margin_per_position_usd)
        notional = required_margin * float(self.cfg.leverage)
        if balance < required_margin:
            self._view[symbol] = f"insufficient balance ${balance:.2f} for ${required_margin:.2f} margin"
            return

        amount = notional / fill_ref if fill_ref > 0 else 0.0
        if amount <= 0 or amount * fill_ref < 5:
            return

        side = "buy" if direction == "long" else "sell"
        try:
            order = await self.client.create_order(
                symbol, side, amount, pos_side=direction,
                tp_price=tp, sl_price=sl
            )
        except Exception as exc:
            logger.error("[%s] order failed: %s", symbol, exc)
            await self.tg.send_text(f"❌ `{_sym(symbol)}` entry order failed: {str(exc)[:150]}")
            return

        fill = order.avg_price or fill_ref
        sl = fill * (1 - self.cfg.stop_loss_pct) if direction == "long" else fill * (1 + self.cfg.stop_loss_pct)
        tp = fill * (1 + self.cfg.take_profit_pct) if direction == "long" else fill * (1 - self.cfg.take_profit_pct)

        st["pos"] = {
            "side": direction,
            "entry": fill,
            "sl": sl,
            "initial_sl": sl,
            "tp": tp,
            "risk": abs(fill - sl),
            "amount": order.amount or amount,
            "margin_usd": required_margin,
            "leverage": self.cfg.leverage,
            "notional_usd": notional,
            "opened_ms": int(time.time() * 1000),
            "exit_bar": None,
            "best_price": fill,
            "lock_stage": 0,
            "setup": sig.setup,
            "q_1h": sig.q_1h,
            "entry_score": sig.entry_score,
        }
        self._save_state()

        logger.info(
            "[%s] OPEN %s @ %.6g sl=%.6g tp=%.6g 4H=%s Q=%.0f Score=%.0f %s",
            symbol, direction.upper(), fill, sl, tp, sig.trend_4h.value,
            sig.q_1h, sig.entry_score, sig.setup
        )
        caption = (
            f"🟢 *{_sym(symbol)} {direction.upper()}* @ `{fill:.6g}`\n"
            f"4H `{sig.trend_4h.value}` | 1H Q `{sig.q_1h:.0f}` "
            f"(ADX {sig.adx_1h:.1f}, CHOP {sig.chop_1h:.1f})\n"
            f"15M `{sig.setup}` | Score `{sig.entry_score:.0f}` | {sig.reason}\n"
            f"SL `{sl:.6g}` (−1.5%) | Final TP `{tp:.6g}` (+1.5%)\n"
            f"T1 `+0.6%` → lock `+0.3%` | T2 `+1.0%` → lock `+0.7%`\n"
            f"Margin `${required_margin:.2f}` × `x{self.cfg.leverage}` ≈ `${notional:.2f}` notional"
        )
        chart = self._build_chart(symbol, df15, direction, fill, sl, tp)
        if chart:
            await self.tg._send_photo(chart, caption)
        else:
            await self.tg.send_text(caption)

    def _build_chart(self, symbol, df, direction, entry, sl, tp):
        if build_entry_chart is None:
            return None
        try:
            return build_entry_chart(symbol, df, direction.upper(), entry, sl, tp, tp,
                                     ema_fast_len=20, ema_slow_len=50, tf_label="15M")
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
        longp = side == "long"
        entry = float(pos["entry"])

        # Best favorable price since entry.
        if longp:
            pos["best_price"] = max(float(pos.get("best_price", entry)), price)
            strat_side = S.Side.LONG
        else:
            pos["best_price"] = min(float(pos.get("best_price", entry)), price)
            strat_side = S.Side.SHORT

        # Two-stage profit lock. Stop can only move toward profit, never backward.
        desired_sl, desired_stage = self.strat.locked_stop(
            strat_side, entry, float(pos["best_price"])
        )
        current_stage = int(pos.get("lock_stage", 0))

        if desired_stage > current_stage:
            old_sl = float(pos["sl"])
            if longp:
                pos["sl"] = max(old_sl, desired_sl)
            else:
                pos["sl"] = min(old_sl, desired_sl)
            pos["lock_stage"] = desired_stage
            self._save_state()

            if desired_stage == 1:
                msg = (
                    f"🔒 *{_sym(symbol)} T1 reached* `+0.6%`\n"
                    f"SL moved to lock `+0.3%` → `{pos['sl']:.6g}`"
                )
            else:
                msg = (
                    f"🔒 *{_sym(symbol)} T2 reached* `+1.0%`\n"
                    f"SL moved to lock `+0.7%` → `{pos['sl']:.6g}`\n"
                    f"Runner active to final TP `+1.5%`"
                )
            logger.info("[%s] profit lock stage %d: SL %.6g -> %.6g",
                        symbol, desired_stage, old_sl, pos["sl"])
            await self.tg.send_text(msg)

        # Local dynamic SL enforcement. The original exchange-native -1.5% SL
        # remains the disaster/offline safety net; these profit locks are managed
        # by the running bot unless the exchange adapter supports SL amendment.
        if pos["sl"] and ((price <= pos["sl"]) if longp else (price >= pos["sl"])):
            why = "LOCK_SL" if int(pos.get("lock_stage", 0)) > 0 else "SL"
            await self._close_market(symbol, st, why)
            return

        if pos["tp"] and ((price >= pos["tp"]) if longp else (price <= pos["tp"])):
            await self._close_market(symbol, st, "TP")
            return

        if pos.get("adopted"):
            return

        # 15M structure invalidation + opposite 4H trend invalidation.
        df15, _, df4h = await self._frames(symbol)
        if len(df15) < 40 or len(df4h) < 60:
            return

        bar_key = df15.index[-1].isoformat()
        if pos.get("exit_bar") == bar_key:
            return
        pos["exit_bar"] = bar_key
        self._save_state()

        ex = self.strat.evaluate_structure_exit(strat_side, df4h, df15)
        if ex.should_exit:
            reason = "HTF_FLIP" if ex.reason == S.ExitReason.HTF_TREND_INVALIDATION else "STRUCTURE"
            logger.info("[%s] early exit: %s", symbol, reason)
            await self._close_market(symbol, st, reason)

    async def _close_market(self, symbol: str, st: dict, why: str):
        pos = st["pos"]
        side = "sell" if pos["side"] == "long" else "buy"
        try:
            await self.client.create_order(symbol, side, pos["amount"],
                                           pos_side=pos["side"], reduce_only=True)
        except Exception as e:
            if "position" not in str(e).lower():
                logger.warning("[%s] market close failed: %s", symbol, e)
        if why in ("STRUCTURE", "HTF_FLIP") and self.cfg.reentry_cooldown_bars > 0:
            self._cooldown_until[symbol] = time.time() + self.cfg.reentry_cooldown_bars * _TF_MIN * 60
        await self._report_close(symbol, st, hint=why)

    def _classify_exit(self, pos, hint, close_px, pnl):
        """TP / SL / FLIP / UNTRACKED. Bot-driven closes carry a hint; a native
        (exchange TP/SL) or offline close is classified by the actual close price
        vs the stored tp/sl."""
        if hint in ("TP", "SL", "LOCK_SL", "STRUCTURE", "HTF_FLIP"):
            return hint
        if pos.get("adopted") or not pos.get("entry") or not pos.get("risk"):
            return "UNTRACKED"
        tol = 0.35 * pos["risk"]
        if close_px > 0:
            if abs(close_px - pos["tp"]) <= tol:
                return "TP"
            if abs(close_px - pos["sl"]) <= tol:
                return "SL"
        return "STRUCTURE" if pnl >= 0 else "SL"   # otherwise it was an in-between close

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
            ticker = await self.client.fetch_ticker(symbol)
            close_px = float(ticker["last"])
            mult = 1 if pos["side"] == "long" else -1
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
        st = self.state.get(symbol) or {}
        pos = st.get("pos")
        if pos:
            return (f"OPEN {pos['side'].upper()} @ {pos['entry']:.6g} "
                    f"SL {pos['sl']:.6g} TP {pos['tp']:.6g}")
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
                "🤖 *HMA16 Trend-Follow Bot*\n/status — สถานะทุก symbol\n"
                "/stats — ผลเทรด (จาก OKX, post-fee)\n/help — เมนูนี้")
        elif cmd == "/status":
            lines = [f"`{_sym(s)}` {self._view_line(s)}" for s in self.cfg.symbols]
            await self.tg.send_text("📡 *Status*\n\n" + "\n".join(lines))
        elif cmd == "/stats":
            await self.tg._send_message(await self._build_stats_report(), _markdown=False)
        else:
            await self.tg.send_text(f"unknown: {cmd} — /help")

    @staticmethod
    def _month_bounds(now_ms: int) -> tuple:
        import datetime as dt
        now = dt.datetime.fromtimestamp(now_ms / 1000, tz=dt.timezone.utc)
        m0 = dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
        py, pm = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        p0 = dt.datetime(py, pm, 1, tzinfo=dt.timezone.utc)
        return int(m0.timestamp() * 1000), int(p0.timestamp() * 1000), p0.strftime("%b")

    def _match_journal(self, okx_rows: list) -> dict:
        pool = list(self.journal)
        used = [False] * len(pool)
        out = {}
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
        """Same OKX-accurate layout as the regime/Adaptive bots — monthly OVERALL
        (resets on the 1st), BY SYMBOL, LAST 5. Counts/WR/PnL from OKX; the
        TP/FLIP/SL breakdown comes from the local journal matched 1-to-1 to OKX
        rows, denominators = OKX total (iron rule)."""
        import datetime as _dt
        since = self.cfg.stats_since_ms()
        now_ms = int(time.time() * 1000)
        m0, p0, _ = self._month_bounds(now_ms)
        cur_lbl = _dt.datetime.fromtimestamp(m0 / 1000, tz=_dt.timezone.utc).strftime("%b %Y")
        prev_lbl = _dt.datetime.fromtimestamp(p0 / 1000, tz=_dt.timezone.utc).strftime("%b %Y")

        okx_ok, rows = True, []
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
        sep = "――――――――――――――――"
        header = (f"📊 HMA16 Bot Stats\n\n💰 Balance: ${balance:.2f}\n"
                  + ("\n".join(open_lines) if open_lines else "📌 No open positions"))
        if not okx_ok:
            header += "\n⚠️ OKX history unavailable — showing local journal"
        if not rows:
            since_lbl = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(since / 1000))
            return header + f"\n\n(no closed trades since {since_lbl})"

        month = [r for r in rows if r.get("close_time_ms", 0) >= m0]
        total = len(month)
        wins = sum(1 for r in month if r["pnl"] > 0)
        net = sum(r["pnl"] for r in month)
        prev_net = sum(r["pnl"] for r in rows if p0 <= r.get("close_time_ms", 0) < m0)
        matched = self._match_journal([r for r in month if not r.get("_journal")])
        cnt = {"TP": 0, "FLIP": 0, "SL": 0}
        tracked = 0
        for r in month:
            et = matched.get(id(r))
            if et in cnt:
                cnt[et] += 1
                tracked += 1
        untracked = total - tracked

        def pct(n):
            return f"{n}/{total} ({n / total * 100:.0f}%)" if total else "0/0"
        lines = [header, "", sep, f"OVERALL (OKX) — {cur_lbl}", sep,
                 f"Trades   : {total}  ({wins}W / {total - wins}L)",
                 f"Win rate : {wins / total * 100:.0f}%" if total else "Win rate : —",
                 f"TP hit   : {pct(cnt['TP'])}   FLIP : {pct(cnt['FLIP'])}   SL : {pct(cnt['SL'])}"]
        if untracked:
            lines.append(f"Untracked: {untracked}/{total} (closed while bot was offline)")
        lines.append(f"Net PnL  : ${net:+.2f}  (post-fee, from OKX)")
        lines.append(f"{prev_lbl} PnL : ${prev_net:+.2f}")

        lines += ["", sep, "BY SYMBOL", sep]
        by = {}
        for r in rows:
            by.setdefault(r["symbol"], []).append(r["pnl"])
        ordered = [s for s in self.cfg.symbols if s in by] + [s for s in by if s not in self.cfg.symbols]
        for s in ordered:
            ps = by[s]
            w = sum(1 for p in ps if p > 0)
            lines.append(f"{_sym(s):<5} {len(ps)} trades  {w / len(ps) * 100:.0f}%WR  ${sum(ps):+.2f}")
        allp = [p for ps in by.values() for p in ps]
        if allp:
            wa = sum(1 for p in allp if p > 0)
            lines += [sep, f"TOTAL   {len(allp)} trades  {wa / len(allp) * 100:.0f}%WR  ${sum(allp):+.2f}"]

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
