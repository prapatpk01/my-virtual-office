"""HMA Fast Structure V6 production runtime.

Keeps the proven OKX execution, restart reconciliation, native SL/TP and
Telegram/statistics infrastructure.  Only the active strategy and runtime
guards are replaced.  One path only; the counter-trend fallback is removed.
"""
from __future__ import annotations

import asyncio
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
        await super()._manage(symbol, st)
        has_position = bool(st.get("pos"))
        if had_position and not has_position:
            self._cooldown_until[symbol] = time.time() + self.post_close_cooldown_sec
            self._closed_seen[symbol] = True
            _LOG.info("[%s] FAST-V6 post-close cooldown 45 minutes", symbol)

    async def _look_for_entry(self, symbol: str, st: dict):
        """Use the inherited entry flow, then verify native protection."""
        had_position = bool(st.get("pos"))
        await super()._look_for_entry(symbol, st)
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

        native_sl, native_tp = await self.client.fetch_attached_stops(symbol, side)
        if native_sl and native_tp:
            return
        repaired = await self.client.move_sl_to_breakeven(
            symbol, side, sl, amount, tp_price=tp
        )
        if repaired:
            _LOG.warning("[%s] missing attached SL/TP repaired immediately", symbol)
            return

        _LOG.error("[%s] native SL/TP verification failed; closing position", symbol)
        await self.tg.send_text(
            f"🚨 `{symbol}` native SL/TP could not be verified. "
            "FAST-V6 is closing the position for safety."
        )
        await self._close_market(symbol, st, "NATIVE_PROTECTION_FAILED")

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
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}` "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}`\n\n"
                "Pipeline: `1H direction → relaxed Q guard → direct closed-15M entry`\n"
                "Entry: `EMA8/13 cross OR HMA16 flip OR EMA13 pullback reclaim`\n"
                "4H: `soft context only` | No mandatory Sentinel S/R | No counter-trend fallback\n"
                "Quality defaults: `Q≥42`; only severe ADX/CHOP or opposing DMI blocks\n"
                "Anti-chase: `≤1.10 ATR from EMA13`\n"
                "Risk: native OKX `SL≤1.0%` and `TP 1.2%`; minimum `1.05R`\n"
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
            loop.add_signal_handler(getattr(signal_module, sig_name), lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
