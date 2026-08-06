"""HMA Fast Structure V6 production runtime.

Keeps the proven OKX execution, restart reconciliation, native SL/TP and
Telegram/statistics infrastructure.  Only the active strategy and runtime
guards are replaced.  One path only; the counter-trend fallback is removed.
"""
from __future__ import annotations

import asyncio
import os
import time

import main_v15 as v15
import strategy_v12 as S

_LOG = v15._LOG


class Bot(v15.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV12(self.cfg.strategy_config())
        self._closed_seen = {symbol: not bool((self.state.get(symbol) or {}).get("pos")) for symbol in self.cfg.symbols}
        self.post_close_cooldown_sec = 45 * 60
        self.risk_per_trade_pct = float(os.environ.get("FAST_RISK_PER_TRADE_PCT", "0.02"))
        self.min_dynamic_margin = float(os.environ.get("FAST_MIN_MARGIN_USD", "5.0"))
        self._shutdown_requested = False
        self._client_closed = False

    def request_shutdown(self) -> None:
        """Stop scheduling work; keep the OKX client alive for in-flight work."""
        self._shutdown_requested = True
        self._running = False
        _LOG.info("FAST-V6 graceful shutdown requested")

    async def run_forever(self):
        """Finish the current symbol safely before closing exchange access."""
        while self._running and not self._shutdown_requested:
            for symbol in self.cfg.symbols:
                if self._shutdown_requested:
                    break
                try:
                    await self._process(symbol)
                except Exception as exc:
                    if self._shutdown_requested and "closed by the user" in str(exc).lower():
                        break
                    _LOG.error("[%s] unhandled: %s", symbol, exc, exc_info=True)
            if self._shutdown_requested:
                break
            self._maybe_status_log()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def stop(self):
        """Close the shared OKX client exactly once."""
        self._running = False
        self._shutdown_requested = True
        if self._client_closed:
            return
        self._client_closed = True
        await self.client.close()
        _LOG.info("FAST-V6 shutdown complete")

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"FAST-V6 POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            remaining = max(0, self._cooldown_until.get(symbol, 0) - time.time())
            if remaining > 0:
                self._view[symbol] = f"FAST-V6 COOLDOWN | {remaining / 60:.0f}m"
                return
            px = float(df15["close"].iloc[-1]) if len(df15) else 0.0
            self._view[symbol] = f"15M px={px:.6g} | {self.strat.entry_status(df4h, df1h, df15, df5)}"
        except Exception as exc:
            self._view[symbol] = f"FAST-V6 view error: {str(exc)[:140]}"

    async def _manage(self, symbol: str, st: dict):
        had_position = bool(st.get("pos"))
        pos = st.get("pos") or {}
        if pos.get("recovery_quarantine"):
            side = str(pos.get("side") or "")
            native_sl, native_tp = await self.client.fetch_attached_stops(
                symbol, side
            )
            if native_sl and native_tp:
                pos.update({
                    "sl": float(native_sl),
                    "initial_sl": float(native_sl),
                    "tp": float(native_tp),
                    "risk": abs(float(pos.get("entry") or 0) - float(native_sl)),
                    "recovery_quarantine": False,
                })
                self._save_state()
                _LOG.info("[%s] recovery quarantine cleared read-only", symbol)
        await super()._manage(symbol, st)
        has_position = bool(st.get("pos"))
        if had_position and not has_position:
            self._cooldown_until[symbol] = time.time() + self.post_close_cooldown_sec
            self._closed_seen[symbol] = True
            _LOG.info("[%s] FAST-V6 post-close cooldown 45 minutes", symbol)

    async def _reconcile_startup(self):
        """Recover positions without cancelling or replacing existing TP/SL."""
        await super()._reconcile_startup()
        for symbol in self.cfg.symbols:
            st = self.state.get(symbol) or {}
            pos = st.get("pos") or {}
            side = str(pos.get("side") or "")
            entry = float(pos.get("entry") or 0.0)
            amount = float(pos.get("amount") or 0.0)
            if side not in ("long", "short") or entry <= 0 or amount <= 0:
                continue

            native_sl = native_tp = None
            for _ in range(6):
                native_sl, native_tp = await self.client.fetch_attached_stops(
                    symbol, side
                )
                if native_sl and native_tp:
                    break
                await asyncio.sleep(1.0)

            if native_sl and native_tp:
                sl = float(native_sl)
                tp = float(native_tp)
                pos.update({
                    "sl": sl,
                    "initial_sl": sl,
                    "tp": tp,
                    "risk": abs(entry - sl),
                    "recovery_quarantine": False,
                })
                _LOG.info(
                    "[%s] recovered position kept existing protection: SL %.8g TP %.8g",
                    symbol, sl, tp,
                )
            else:
                # Read failure is not proof that protection is absent. Keep the
                # position quarantined and leave all OKX orders untouched.
                pos["recovery_quarantine"] = True
                _LOG.warning("[%s] recovered protection read unavailable", symbol)
                await self.tg.send_text(
                    f"⚠️ `{symbol}` recovered TP/SL could not be read yet. "
                    "No OKX order was cancelled, replaced or added."
                )
        self._save_state()

    async def _look_for_entry(self, symbol: str, st: dict):
        """Risk-size the inherited entry, then verify protection read-only."""
        had_position = bool(st.get("pos"))
        configured_margin = float(self.cfg.margin_per_position_usd)
        try:
            try:
                df5, df15, df1h, df4h = await self._entry_frames(symbol)
                preview = self.strat.generate_entry(
                    df4h, df1h, df15, df5, has_open_position=False
                )
                if preview is not None and preview.entry_price > 0:
                    stop_pct = abs(
                        float(preview.entry_price) - float(preview.stop_loss)
                    ) / float(preview.entry_price)
                    balance = await self.client.fetch_balance_usdt()
                    risk_budget = max(0.0, balance * self.risk_per_trade_pct)
                    margin = risk_budget / max(
                        stop_pct * float(self.cfg.leverage), 1e-12
                    )
                    self.cfg.margin_per_position_usd = min(
                        configured_margin,
                        max(self.min_dynamic_margin, margin),
                    )
                    _LOG.info(
                        "[%s] dynamic risk size: balance=%.2f risk=$%.2f "
                        "SL=%.2f%% margin=$%.2f cap=$%.2f",
                        symbol, balance, risk_budget, stop_pct * 100,
                        self.cfg.margin_per_position_usd, configured_margin,
                    )
            except Exception as exc:
                _LOG.warning("[%s] dynamic sizing preflight failed: %s", symbol, exc)
            await super()._look_for_entry(symbol, st)
        finally:
            self.cfg.margin_per_position_usd = configured_margin
        pos = st.get("pos") or {}
        if had_position or not pos:
            return

        side = str(pos.get("side") or "")
        sl = float(pos.get("sl") or 0.0)
        tp = float(pos.get("tp") or 0.0)
        amount = float(pos.get("amount") or 0.0)
        if side not in ("long", "short") or sl <= 0 or tp <= 0 or amount <= 0:
            _LOG.error("[%s] invalid local protection plan after entry", symbol)
            await self._close_market(symbol, st, "PROTECTION_PLAN_INVALID")
            return

        # The market-order request already attached TP/SL. Verification is
        # read-only and delayed because OKX may expose the child algo later.
        native_sl = native_tp = None
        for _ in range(8):
            native_sl, native_tp = await self.client.fetch_attached_stops(
                symbol, side
            )
            if native_sl and native_tp:
                break
            await asyncio.sleep(1.0)
        if native_sl and native_tp:
            _LOG.info(
                "[%s] attached protection visible: SL %.8g TP %.8g",
                symbol, native_sl, native_tp,
            )
            return

        # Never cancel, add, replace or close solely because the read API is
        # lagging. Local management continues and the next cycle can read it.
        _LOG.warning("[%s] attached protection not visible yet; no action taken", symbol)
        await self.tg.send_text(
            f"⚠️ `{symbol}` TP/SL is not visible through the read API yet. "
            "The position remains open and no OKX order was changed."
        )

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info("=== HMA FAST STRUCTURE V6 [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===",
                  "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols,
                  self.cfg.margin_per_position_usd, self.cfg.leverage,
                  self.cfg.max_positions, balance)
        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            mode = "PAPER" if self.cfg.paper else "LIVE"
            await self.tg.send_text(
                f"⚡ *HMA Fast Structure V6 — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin cap `${self.cfg.margin_per_position_usd:.2f}` "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}`\n\n"
                "Pipeline: `1H direction → relaxed Q guard → direct closed-15M entry`\n"
                "Entry: `EMA8/13 cross OR HMA16 flip OR EMA13 pullback reclaim`\n"
                "4H: `soft context only` | No mandatory Sentinel S/R | No counter-trend fallback\n"
                "Quality defaults: `Q≥42`; only severe ADX/CHOP or opposing DMI blocks\n"
                "Anti-chase: `≤1.10 ATR from EMA13`\n"
                "Risk: structure SL `0.60–1.00%` and `≥1.35 ATR15`; TP `1.2%`\n"
                "Sizing: dynamic margin targets `2% balance risk`; `$20` is the cap\n"
                "Re-entry: `45-minute cooldown after every close`\n"
                "Recovery: existing positions and native SL/TP reconciled after restart"
            )
        _LOG.info("HMA Fast Structure V6 startup complete")


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            signal_module = v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal
            loop.add_signal_handler(
                getattr(signal_module, sig_name), bot.request_shutdown
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
