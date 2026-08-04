"""HMA Expert MTF V4.1 — adaptive Sentinel trend production runtime.

Inherits V4.0/V3.7 production infrastructure:
- FX 24/5 new-entry gate
- OKX restart recovery and native SL/TP sync
- auditable OKX stats
- 5M chart alerts
- balanced two-stage profit locks

V4.1 replaces the legacy discrete 4H neutral gate with independent Sentinel
LONG/SHORT trend scores and emits full decision diagnostics in Railway logs.
"""
from __future__ import annotations

import asyncio

import main_v10 as v10
import strategy_v7 as S

_LOG = v10._LOG


def _v41_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace(
            "HMA Expert MTF V4.0 Sentinel Location-Aware",
            "HMA Expert MTF V4.1 Adaptive Sentinel",
        )
        .replace("HMA Expert MTF V4.0", "HMA Expert MTF V4.1")
        .replace("HMA V4.0 Bot Stats", "HMA V4.1 Bot Stats")
    )


class Bot(v10.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV7(self.cfg.strategy_config())

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v41_text(text: str) -> bool:
            return await previous_send_text(_v41_text(text))

        async def v41_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v41_text(caption))

        self.tg.send_text = v41_text
        self.tg._send_photo = v41_photo

    async def start(self):
        await super().start()
        _LOG.info(
            "HMA V4.1 adaptive Sentinel active: EARLY>=%.0f MODERATE>=%.0f "
            "STRONG>=%.0f direction-edge>=%.0f trade-score>=%.0f",
            self.strat.early_trend_min,
            self.strat.sentinel_moderate_trend_min,
            self.strat.sentinel_strong_trend_min,
            self.strat.direction_edge_min,
            self.strat.trade_score_min,
        )
        await self.tg.send_text(
            "🧭 *HMA V4.1 Adaptive Sentinel — ACTIVE*\n"
            "4H direction now uses independent `LONG/SHORT Trend Scores` instead of a hard neutral gate.\n"
            f"EARLY `≥{self.strat.early_trend_min:.0f}`: only `S2/R2`, Location `≥{self.strat.early_location_min:.0f}`, Q `≥{self.strat.early_quality_min:.0f}`\n"
            f"MODERATE `≥{self.strat.sentinel_moderate_trend_min:.0f}`: only `S2/R2`\n"
            f"STRONG `≥{self.strat.sentinel_strong_trend_min:.0f}`: `S1/S2` or `R1/R2`\n"
            f"Direction edge `≥{self.strat.direction_edge_min:.0f}` | Trade Score `≥{self.strat.trade_score_min:.0f}`\n"
            "Railway status now shows Trend, Q, Location, Room, Execution and Trade Score."
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v10.v9.v8.v7.v5.v4.v3.base._signal, sig_name),
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
