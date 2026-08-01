"""HMA Expert MTF V3.6 — OKX restart recovery and auditable stats.

Adds two production guarantees:
1) A live position found after restart is rebuilt from OKX position details and
   pending attached SL/TP orders, including inference of the active lock stage.
2) /stats uses OKX positions-history for trade counts and realizedPnl, while
   clearly separating exchange truth from locally classified exit reasons.
"""
from __future__ import annotations

import asyncio
import time

import main_v7 as v7


_LOG = v7.v5.v4.v3.base.logger


def _v36_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace(
            "HMA Expert MTF V3.5 Balanced Two-Stage",
            "HMA Expert MTF V3.6 OKX-Recovery Two-Stage",
        )
        .replace(
            "HMA Expert MTF V3.5",
            "HMA Expert MTF V3.6",
        )
    )


class Bot(v7.Bot):
    def __init__(self):
        super().__init__()

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v36_text(text: str) -> bool:
            return await previous_send_text(_v36_text(text))

        async def v36_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v36_text(caption))

        self.tg.send_text = v36_text
        self.tg._send_photo = v36_photo

    @staticmethod
    def _valid_price(value) -> float:
        try:
            value = float(value or 0.0)
            return value if value > 0.0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _infer_lock_stage(self, side: str, entry: float, sl: float) -> int:
        """Infer stage from the actual OKX stop, allowing a small tick tolerance."""
        entry = self._valid_price(entry)
        sl = self._valid_price(sl)
        if entry <= 0.0 or sl <= 0.0:
            return 0

        tol = entry * 0.00020  # 2 bp: rounding/tick-size tolerance
        if side == "long":
            stage2 = entry * (1.0 + self.strat.stage2_lock_pct)
            stage1 = entry * (1.0 + self.strat.stage1_lock_pct)
            if sl >= stage2 - tol:
                return 2
            if sl >= stage1 - tol:
                return 1
            return 0

        stage2 = entry * (1.0 - self.strat.stage2_lock_pct)
        stage1 = entry * (1.0 - self.strat.stage1_lock_pct)
        if sl <= stage2 + tol:
            return 2
        if sl <= stage1 + tol:
            return 1
        return 0

    def _best_price_for_recovery(
        self, side: str, entry: float, current_price: float, stage: int
    ) -> float:
        """Seed best price so a recovered stage can never regress after restart."""
        if side == "long":
            floor = entry
            if stage >= 2:
                floor = entry * (1.0 + self.strat.stage2_trigger_pct)
            elif stage >= 1:
                floor = entry * (1.0 + self.strat.stage1_trigger_pct)
            return max(floor, current_price or entry)

        ceiling = entry
        if stage >= 2:
            ceiling = entry / (1.0 + self.strat.stage2_trigger_pct)
        elif stage >= 1:
            ceiling = entry / (1.0 + self.strat.stage1_trigger_pct)
        return min(ceiling, current_price or entry)

    async def _reconcile_startup(self):
        """Rebuild open positions from OKX instead of adopting them at entry=0."""
        recovered = 0

        for symbol in self.cfg.symbols:
            st = self._sym_state(symbol)
            local = st.get("pos") or {}
            live_legs: list[tuple[str, dict]] = []

            for side in ("long", "short"):
                details = await self.client.fetch_position_details(symbol, side)
                if details and float(details.get("amount") or 0.0) > 0.0:
                    live_legs.append((side, details))

            if not live_legs:
                if local:
                    st["pos"] = None
                continue

            if len(live_legs) > 1:
                preferred = str(local.get("side") or "")
                selected = next(
                    ((side, details) for side, details in live_legs if side == preferred),
                    live_legs[0],
                )
                await self.tg.send_text(
                    f"⚠️ `{v7.v5.v4.v3.base._sym(symbol)}` has both LONG and SHORT "
                    f"legs on OKX. HMA will track `{selected[0].upper()}` only; "
                    f"manual review is required."
                )
            else:
                selected = live_legs[0]

            side, details = selected
            entry = self._valid_price(details.get("entry_price"))
            amount = float(details.get("amount") or 0.0)

            if entry <= 0.0 or amount <= 0.0:
                await self.tg.send_text(
                    f"⚠️ `{v7.v5.v4.v3.base._sym(symbol)}` {side.upper()} exists "
                    f"but OKX entry/amount recovery failed. No double-entry; "
                    f"exchange-native protection remains authoritative."
                )
                st["pos"] = {
                    "side": side,
                    "entry": 0.0,
                    "sl": 0.0,
                    "tp": 0.0,
                    "risk": 0.0,
                    "amount": amount,
                    "opened_ms": int(time.time() * 1000),
                    "adopted": True,
                }
                continue

            okx_sl, okx_tp = await self.client.fetch_attached_stops(symbol, side)
            sl = self._valid_price(okx_sl)
            tp = self._valid_price(okx_tp)
            used_fallback = False

            if sl <= 0.0:
                used_fallback = True
                sl = (
                    entry * (1.0 - self.cfg.stop_loss_pct)
                    if side == "long"
                    else entry * (1.0 + self.cfg.stop_loss_pct)
                )
            if tp <= 0.0:
                used_fallback = True
                tp = (
                    entry * (1.0 + self.cfg.take_profit_pct)
                    if side == "long"
                    else entry * (1.0 - self.cfg.take_profit_pct)
                )

            # If one native leg was missing, restore a complete exchange OCO using
            # the recovered leg plus the configured disaster-stop/final-TP fallback.
            protection_ok = True
            if used_fallback and not self.cfg.paper:
                protection_ok = await self.client.move_sl_to_breakeven(
                    symbol, side, sl, amount, tp_price=tp
                )

            stage = self._infer_lock_stage(side, entry, sl)
            ticker = await self.client.fetch_ticker(symbol)
            current_price = self._valid_price((ticker or {}).get("last"))
            best_price = self._best_price_for_recovery(
                side, entry, current_price, stage
            )

            initial_risk = 0.0
            loss_side_sl = (
                side == "long" and sl < entry
            ) or (
                side == "short" and sl > entry
            )
            if stage == 0 and loss_side_sl:
                initial_risk = abs(entry - sl)

            st["pos"] = {
                "side": side,
                "entry": entry,
                "sl": sl,
                "initial_sl": sl if stage == 0 else 0.0,
                "tp": tp,
                "risk": initial_risk,
                "amount": amount,
                "margin_usd": (amount * entry / max(float(self.cfg.leverage), 1.0)),
                "leverage": self.cfg.leverage,
                "notional_usd": amount * entry,
                "opened_ms": int(time.time() * 1000),
                "exit_bar": None,
                "best_price": best_price,
                "lock_stage": stage,
                "setup": "RECOVERED_OKX",
                "q_1h": 0.0,
                "room_pct": 0.0,
                "trigger": "RESTART_RECOVERY",
                "recovered": True,
                "adopted": False,
                "recovery_source": (
                    "OKX_POSITION+OKX_ALGO"
                    if not used_fallback
                    else "OKX_POSITION+CONFIG_FALLBACK"
                ),
            }
            recovered += 1

            source = (
                "OKX position + native SL/TP"
                if not used_fallback
                else "OKX position + restored fallback protection"
            )
            extra = "" if protection_ok else "\n⚠️ Native OCO restore failed; check Railway log/OKX."
            await self.tg.send_text(
                f"♻️ *{v7.v5.v4.v3.base._sym(symbol)} {side.upper()} recovered after restart*\n"
                f"Entry `{entry:.6g}` | SL `{sl:.6g}` | Final TP `{tp:.6g}`\n"
                f"Lock stage `{stage}` | Amount `{amount:.8g}`\n"
                f"Source: `{source}`{extra}"
            )

        self._save_state()
        _LOG.info("HMA V3.6 startup reconciliation recovered=%d", recovered)

    async def _manage(self, symbol: str, st: dict):
        """Mirror every local stage lock to OKX-native OCO protection."""
        before = st.get("pos") or {}
        old_stage = int(before.get("lock_stage", 0))
        await super()._manage(symbol, st)

        pos = st.get("pos") or {}
        if not pos:
            return

        new_stage = int(pos.get("lock_stage", 0))
        if new_stage <= old_stage:
            return

        side = str(pos.get("side") or "")
        sl = self._valid_price(pos.get("sl"))
        tp = self._valid_price(pos.get("tp"))
        amount = float(pos.get("amount") or 0.0)
        if side not in ("long", "short") or sl <= 0.0 or amount <= 0.0:
            return

        ok = await self.client.move_sl_to_breakeven(
            symbol, side, sl, amount, tp_price=(tp or None)
        )
        if ok:
            pos["native_sl_synced_stage"] = new_stage
            self._save_state()
            _LOG.info(
                "[%s] stage %d lock synced to OKX native SL %.8g TP %.8g",
                symbol, new_stage, sl, tp,
            )
        else:
            await self.tg.send_text(
                f"⚠️ `{v7.v5.v4.v3.base._sym(symbol)}` Stage {new_stage} local "
                f"lock is active at `{sl:.6g}`, but OKX-native SL amendment failed. "
                f"Check Railway log and OKX orders."
            )

    async def _live_open_lines(self) -> list[str]:
        lines: list[str] = []
        for symbol in self.cfg.symbols:
            for side in ("long", "short"):
                details = await self.client.fetch_position_details(symbol, side)
                if not details:
                    continue
                amount = float(details.get("amount") or 0.0)
                entry = self._valid_price(details.get("entry_price"))
                if amount > 0.0:
                    sl, tp = await self.client.fetch_attached_stops(symbol, side)
                    sl_txt = f"{float(sl):.6g}" if sl else "—"
                    tp_txt = f"{float(tp):.6g}" if tp else "—"
                    lines.append(
                        f"📌 {v7.v5.v4.v3.base._sym(symbol)} {side.upper()} "
                        f"@ {entry:.6g} | SL {sl_txt} | TP {tp_txt}"
                    )
        return lines

    async def _build_stats_report(self) -> str:
        """Auditable OKX stats with consistent monthly and since-date scopes."""
        import datetime as dt

        since = self.cfg.stats_since_ms()
        now_ms = int(time.time() * 1000)
        month_start, previous_start, _ = self._month_bounds(now_ms)
        current_label = dt.datetime.fromtimestamp(
            month_start / 1000, tz=dt.timezone.utc
        ).strftime("%b %Y")
        previous_label = dt.datetime.fromtimestamp(
            previous_start / 1000, tz=dt.timezone.utc
        ).strftime("%b %Y")
        since_label = dt.datetime.fromtimestamp(
            since / 1000, tz=dt.timezone.utc
        ).strftime("%Y-%m-%d") if since else "configured baseline"

        okx_ok = True
        rows: list[dict] = []
        if not self.cfg.paper:
            try:
                rows = await self.client.fetch_trade_history(since, self.cfg.symbols)
            except Exception as exc:
                okx_ok = False
                _LOG.warning("[STATS] OKX positions-history failed: %s", exc)

        if self.cfg.paper or not okx_ok:
            rows = [
                {
                    "symbol": event["symbol"],
                    "side": event.get("side", ""),
                    "pnl": event["pnl"],
                    "close_time_ms": event["close_ms"],
                    "_journal": True,
                }
                for event in self.journal
                if event["close_ms"] >= since
            ]

        balance = await self.client.fetch_balance_usdt()
        open_lines = await self._live_open_lines()
        sep = "――――――――――――――――"
        header = [
            "📊 HMA V3.6 Bot Stats",
            "",
            f"💰 Balance: ${balance:.2f}",
            *(open_lines or ["📌 No open positions"]),
        ]

        if not okx_ok:
            header.append("⚠️ OKX history unavailable — local journal fallback")
        elif not self.cfg.paper:
            header.append(
                "Source: OKX positions-history / realizedPnl "
                "(includes fees and funding)"
            )
        else:
            header.append("Source: PAPER local journal")

        month_rows = [
            row for row in rows
            if int(row.get("close_time_ms", 0)) >= month_start
        ]
        previous_rows = [
            row for row in rows
            if previous_start <= int(row.get("close_time_ms", 0)) < month_start
        ]

        def summary(block: list[dict]) -> tuple[int, int, float]:
            total = len(block)
            wins = sum(1 for row in block if float(row.get("pnl", 0.0)) > 0.0)
            net = sum(float(row.get("pnl", 0.0)) for row in block)
            return total, wins, net

        month_total, month_wins, month_net = summary(month_rows)
        all_total, all_wins, all_net = summary(rows)
        previous_net = sum(float(row.get("pnl", 0.0)) for row in previous_rows)

        lines = header + [
            "",
            sep,
            f"OVERALL (OKX) — {current_label}",
            sep,
            f"Trades   : {month_total}  ({month_wins}W / {month_total - month_wins}L)",
            (
                f"Win rate : {month_wins / month_total * 100:.0f}%"
                if month_total else "Win rate : —"
            ),
            f"Net PnL  : ${month_net:+.2f}  (OKX realizedPnl)",
            f"{previous_label} PnL : ${previous_net:+.2f}",
        ]

        # Exit labels are bot metadata, not exchange PnL truth. Keep them separate.
        matched = self._match_journal(
            [row for row in month_rows if not row.get("_journal")]
        )
        tags = {"TP": 0, "LOCK": 0, "EARLY": 0, "SL": 0}
        classified = 0
        for row in month_rows:
            exit_type = matched.get(id(row))
            bucket = None
            if exit_type == "TP":
                bucket = "TP"
            elif exit_type == "LOCK_SL":
                bucket = "LOCK"
            elif exit_type in ("STRUCTURE", "HTF_FLIP"):
                bucket = "EARLY"
            elif exit_type == "SL":
                bucket = "SL"
            if bucket:
                tags[bucket] += 1
                classified += 1

        unknown = month_total - classified
        lines += [
            f"Exit tags: TP {tags['TP']} | LOCK {tags['LOCK']} | "
            f"EARLY {tags['EARLY']} | SL {tags['SL']}",
            f"Unknown exit reason: {unknown}/{month_total}",
        ]
        if unknown:
            lines.append(
                "Note: trade/PnL are still real OKX data; only the bot exit label "
                "was unavailable after restart."
            )

        lines += ["", sep, f"BY SYMBOL — {current_label}", sep]
        month_by: dict[str, list[float]] = {}
        for row in month_rows:
            month_by.setdefault(row["symbol"], []).append(float(row["pnl"]))

        if month_by:
            ordered = [
                symbol for symbol in self.cfg.symbols if symbol in month_by
            ] + [
                symbol for symbol in month_by if symbol not in self.cfg.symbols
            ]
            for symbol in ordered:
                pnl_values = month_by[symbol]
                wins = sum(1 for pnl in pnl_values if pnl > 0.0)
                lines.append(
                    f"{v7.v5.v4.v3.base._sym(symbol):<5} {len(pnl_values)} trades  "
                    f"{wins / len(pnl_values) * 100:.0f}%WR  "
                    f"${sum(pnl_values):+.2f}"
                )
        else:
            lines.append("(no closed trades this month)")

        lines += [
            "",
            sep,
            f"SINCE {since_label}",
            sep,
            f"Trades   : {all_total}  ({all_wins}W / {all_total - all_wins}L)",
            (
                f"Win rate : {all_wins / all_total * 100:.0f}%"
                if all_total else "Win rate : —"
            ),
            f"Net PnL  : ${all_net:+.2f}",
        ]

        lines += ["", sep, "LAST 5 TRADES (OKX)", sep]
        if not rows:
            lines.append("(no closed trades)")
        else:
            now = time.time()
            for index, row in enumerate(
                sorted(
                    rows,
                    key=lambda item: -int(item.get("close_time_ms", 0)),
                )[:5],
                1,
            ):
                age = now - int(row.get("close_time_ms", now_ms)) / 1000
                age_label = (
                    f"{age / 3600:.1f}h ago"
                    if age < 86400
                    else f"{age / 86400:.1f}d ago"
                )
                pnl = float(row.get("pnl", 0.0))
                emoji = "✅" if pnl > 0 else "❌"
                side = str(row.get("side") or "").upper()
                lines.append(
                    f"{index}. {emoji} "
                    f"{v7.v5.v4.v3.base._sym(row['symbol'])} {side} "
                    f"${pnl:+.2f} — {age_label}"
                )

        return "\n".join(lines)

    async def start(self):
        await super().start()
        _LOG.info(
            "HMA V3.6 active: OKX entry/SL/TP restart recovery, "
            "native stage-lock sync, and auditable OKX stats"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v7.v5.v4.v3.base._signal, sig_name),
                lambda: asyncio.ensure_future(bot.stop()),
            )
        except (NotImplementedError, AttributeError):
            pass

    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
