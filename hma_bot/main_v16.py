"""HMA Simple Sentinel production runtime.

Startup is standalone so Telegram receives one authoritative message only.
"""
from __future__ import annotations

import asyncio

import main_v15 as v15
import strategy_v12 as S

_LOG = v15._LOG


class Bot(v15.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV12(self.cfg.strategy_config())

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            self._view[symbol] = (
                f"5M px={px:.6g} | "
                f"{self.strat.entry_status(df4h, df1h, df15, df5)}"
            )
        except Exception as exc:
            self._view[symbol] = f"view error: {str(exc)[:140]}"

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))

        if not self.cfg.paper:
            if not await self.client.ensure_hedge_mode():
                raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== HMA SIMPLE SENTINEL [%s] symbols=%s margin=$%.2f "
            "leverage=x%d max_pos=%d balance=%.2f ===",
            "PAPER" if self.cfg.paper else "LIVE",
            self.cfg.symbols,
            self.cfg.margin_per_position_usd,
            self.cfg.leverage,
            self.cfg.max_positions,
            balance,
        )

        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            mode = "PAPER" if self.cfg.paper else "LIVE"
            await self.tg.send_text(
                f"⚡ *HMA Simple Sentinel — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                "Simplified entry logic:\n"
                f"`Layer 1 · 1H Direction` Trend `≥{self.strat.one_h_early_min:.0f}` · edge `≥{self.strat.one_h_direction_edge_min:.0f}` · Q `≥{self.strat.quality_min:.0f}`\n"
                f"`Layer 2 · 15M Setup` S/R within `{self.strat.setup_proximity_atr:.2f} ATR` or aligned EMA20 pullback\n"
                f"`Layer 3 · 5M Trigger` EMA8/13, BOS/CHOCH or continuation within `{self.strat.exec_trigger_lookback}` closed bars\n"
                f"`Risk Check` Room `≥{self.strat.min_room_atr:.2f} ATR` and actual R:R `≥{self.strat.min_actual_rr:.2f}`\n"
                "4H and Confidence are diagnostic only; neither can block an entry.\n"
                "Risk management: Stage 1 `+0.7%→lock +0.4%` | Stage 2 `+1.1%→lock +0.75%` | Final TP `+1.5%`\n"
                "Schedule: `FX 24/5 new entries` | Existing positions managed `24/7`\n"
                "Recovery: `OKX-native SL/TP synchronised after restart`"
            )

        _LOG.info(
            "HMA Simple Sentinel startup complete: 3 layers + 1 risk check; "
            "status and order creation use the same decision"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(
                    v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal,
                    sig_name,
                ),
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
