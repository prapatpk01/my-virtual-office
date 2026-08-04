"""HMA Expert MTF V5.1 — smoother weighted Sentinel runtime.

Startup is intentionally implemented here instead of chaining every inherited
``start()`` method.  The strategy classes still inherit the proven execution,
recovery and position-management code, but Telegram receives one authoritative
V5.1 startup message only.
"""
from __future__ import annotations

import asyncio

import main_v14 as v14
import strategy_v11 as S

_LOG = v14._LOG


def _v51_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace("HMA Expert MTF V5.0 1H-Led Sentinel", "HMA Expert MTF V5.1 Smooth Sentinel")
        .replace("HMA Expert MTF V5.0", "HMA Expert MTF V5.1")
        .replace("HMA V5.0 Bot Stats", "HMA V5.1 Bot Stats")
    )


class Bot(v14.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV11(self.cfg.strategy_config())

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v51_text(text: str) -> bool:
            return await previous_send_text(_v51_text(text))

        async def v51_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v51_text(caption))

        self.tg.send_text = v51_text
        self.tg._send_photo = v51_photo

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"V5.1 POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            self._view[symbol] = f"5M px={px:.6g} | {self.strat.entry_status(df4h, df1h, df15, df5)}"
        except Exception as exc:
            self._view[symbol] = f"V5.1 view error: {str(exc)[:140]}"

    async def start(self):
        """Initialize production services once and emit one V5.1 startup alert."""
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))

        if not self.cfg.paper:
            if not await self.client.ensure_hedge_mode():
                raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== HMA V5.1 SMOOTH SENTINEL [%s] symbols=%s margin=$%.2f "
            "leverage=x%d max_pos=%d balance=%.2f ===",
            "PAPER" if self.cfg.paper else "LIVE",
            self.cfg.symbols,
            self.cfg.margin_per_position_usd,
            self.cfg.leverage,
            self.cfg.max_positions,
            balance,
        )

        # Restore/synchronise any live OKX position before scans begin.
        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())

            mode = "PAPER" if self.cfg.paper else "LIVE"
            await self.tg.send_text(
                f"⚡ *HMA V5.1 Smooth Sentinel — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                "Pipeline: `1H Direction + Quality → 15M Sentinel S/R → 5M Trigger`\n"
                "4H: `soft macro bias only` — never a hard entry gate\n"
                "Trade Score: `1H 40% · Location 30% · Execution 20% · 4H 10%`\n"
                "Location: `S1/R1=65 · corridor=75 · S2/R2=90`\n"
                f"Normal entry: `≥{self.strat.v5_entry_score:.0f}`\n"
                f"Conditional: `≥{self.strat.v5_conditional_score:.0f}` with Location "
                f"`≥{self.strat.v5_conditional_location:.0f}` at corridor, S2/R2 or sweep\n"
                "Risk: `15M structure + ATR buffer` | Stage 1 `+0.7%→lock +0.4%` "
                "| Stage 2 `+1.1%→lock +0.75%` | Final TP `+1.5%`\n"
                "Schedule: `FX 24/5 new entries` | Existing positions managed `24/7`\n"
                "Recovery: `OKX-native SL/TP synchronised after restart`"
            )

        _LOG.info(
            "HMA V5.1 startup complete: one notification only; "
            "weights 1H=40 location=30 execution=20 macro=10"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal, sig_name),
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
