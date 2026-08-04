"""HMA Expert MTF V5.1 — smoother weighted Sentinel runtime."""
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
        await super().start()
        _LOG.info(
            "HMA V5.1 active: weights 1H=40 location=30 execution=20 macro=10; "
            "graduated S/R scores S1/R1=65 corridor=75 S2/R2=90"
        )
        await self.tg.send_text(
            "⚡ *HMA V5.1 Smooth Sentinel — ACTIVE*\n"
            "Trade Score: `1H 40% · Location 30% · Execution 20% · 4H 10%`\n"
            "Graduated Location: `S1/R1=65 · corridor=75 · S2/R2=90`\n"
            "The bot still requires a valid S/R setup and a recent closed-5M trigger.\n"
            f"Normal entry: `≥{self.strat.v5_entry_score:.0f}`\n"
            f"Conditional entry: `≥{self.strat.v5_conditional_score:.0f}` with Location `≥{self.strat.v5_conditional_location:.0f}` at corridor, S2/R2 or sweep."
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
