"""HMA Expert MTF V5.0 — 1H-led Sentinel S/R production runtime."""
from __future__ import annotations

import asyncio

import main_v13 as v13
import strategy_v10 as S

_LOG = v13._LOG


def _v50_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace(
            "HMA Expert MTF V4.3 Sentinel S/R Entry",
            "HMA Expert MTF V5.0 1H-Led Sentinel",
        )
        .replace("HMA Expert MTF V4.3", "HMA Expert MTF V5.0")
        .replace("HMA V4.3 Bot Stats", "HMA V5.0 Bot Stats")
    )


class Bot(v13.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV10(self.cfg.strategy_config())

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v50_text(text: str) -> bool:
            return await previous_send_text(_v50_text(text))

        async def v50_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v50_text(caption))

        self.tg.send_text = v50_text
        self.tg._send_photo = v50_photo

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"V5 POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            self._view[symbol] = (
                f"5M px={px:.6g} | "
                f"{self.strat.entry_status(df4h, df1h, df15, df5)}"
            )
        except Exception as exc:
            self._view[symbol] = f"V5 view error: {str(exc)[:140]}"

    async def start(self):
        await super().start()
        _LOG.info(
            "HMA V5 active: 1H direction/quality -> 15M Sentinel S/R -> "
            "5M execution; 4H soft bias only; entry>=%.0f conditional>=%.0f",
            self.strat.v5_entry_score,
            self.strat.v5_conditional_score,
        )
        await self.tg.send_text(
            "⚡ *HMA V5.0 1H-Led Sentinel — ACTIVE*\n"
            "Primary pipeline: `1H Direction + Quality → 15M S/R → 5M Trigger`\n"
            "4H is now `soft macro bias only` and can no longer block a valid entry.\n"
            "Trade Score: `1H 35% · Location 35% · Execution 20% · 4H 10%`\n"
            f"Normal entry: `≥{self.strat.v5_entry_score:.0f}`\n"
            f"Conditional entry: `≥{self.strat.v5_conditional_score:.0f}` only at `S2/R2` or sweep with Location `≥{self.strat.v5_conditional_location:.0f}`\n"
            "EARLY 1H uses only S2/R2; STRONG 1H can use S1/S2 or R1/R2."
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal, sig_name),
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
