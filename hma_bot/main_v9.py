"""HMA Expert MTF V3.7 — FX 24/5 new-entry schedule.

New positions:
- allowed Monday through Thursday,
- allowed Friday until 17:00 New York,
- paused after the FX weekly close,
- enabled again Sunday at 13:00 New York, four hours before the conventional
  Sunday 17:00 New York FX open.

Open positions continue to be monitored and managed 24/7.  This gate never
pauses native OKX SL/TP, stage locks, close detection, or restart recovery.
"""
from __future__ import annotations

import asyncio
import os

import main_v8 as v8
from fx_session import fx_entry_session, format_next_open

_LOG = v8._LOG


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _v37_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return (
        text.replace(
            "HMA Expert MTF V3.6 OKX-Recovery Two-Stage",
            "HMA Expert MTF V3.7 FX-24/5 Two-Stage",
        )
        .replace("HMA Expert MTF V3.6", "HMA Expert MTF V3.7")
        .replace("HMA V3.6 Bot Stats", "HMA V3.7 Bot Stats")
    )


class Bot(v8.Bot):
    def __init__(self):
        super().__init__()
        self.fx_24_5_enabled = _env_bool("FX_24_5_ENABLED", True)
        self.fx_preopen_hours = max(0, min(_env_int("FX_PREOPEN_HOURS", 4), 23))
        self._fx_session_status: str | None = None

        previous_send_text = self.tg.send_text
        previous_send_photo = self.tg._send_photo

        async def v37_text(text: str) -> bool:
            return await previous_send_text(_v37_text(text))

        async def v37_photo(path: str, caption: str) -> bool:
            return await previous_send_photo(path, _v37_text(caption))

        self.tg.send_text = v37_text
        self.tg._send_photo = v37_photo

    async def _announce_session_transition(self, state) -> None:
        if self._fx_session_status == state.status:
            return
        self._fx_session_status = state.status

        if state.entries_allowed:
            _LOG.info(
                "[FX SESSION] new entries enabled now_utc=%s now_ny=%s",
                state.now_utc.isoformat(), state.now_ny.isoformat(),
            )
            await self.tg.send_text(
                "🟢 *FX 24/5 Entry Mode — ACTIVE*\n"
                "New positions are enabled. Existing-position management remains active 24/7."
            )
            return

        next_open = format_next_open(state)
        _LOG.info(
            "[FX SESSION] sleep mode; new entries paused until %s",
            next_open,
        )
        await self.tg.send_text(
            "😴 *FX Weekend Sleep Mode*\n"
            "New positions are paused while the FX market is closed.\n"
            "Open positions continue normal SL/TP, stage-lock and close management.\n"
            f"New entries resume: `{next_open}` — {self.fx_preopen_hours}h before FX open."
        )

    async def _look_for_entry(self, symbol: str, st: dict):
        """Apply the FX calendar only to NEW entries, never position management."""
        if not self.fx_24_5_enabled:
            return await super()._look_for_entry(symbol, st)

        state = fx_entry_session(preopen_hours=self.fx_preopen_hours)
        await self._announce_session_transition(state)

        if not state.entries_allowed:
            self._view[symbol] = (
                f"SLEEP MODE | FX market closed | no new entries | "
                f"resume {format_next_open(state)} | open positions managed 24/7"
            )
            return

        return await super()._look_for_entry(symbol, st)

    async def start(self):
        await super().start()
        state = fx_entry_session(preopen_hours=self.fx_preopen_hours)
        await self._announce_session_transition(state)
        schedule = (
            f"Sunday {17 - self.fx_preopen_hours:02d}:00 NY → Friday 17:00 NY"
            if self.fx_24_5_enabled
            else "disabled"
        )
        _LOG.info(
            "HMA V3.7 FX 24/5 gate active=%s schedule=%s; open-position management remains 24/7",
            self.fx_24_5_enabled, schedule,
        )
        await self.tg.send_text(
            "🕒 *Trading Schedule*\n"
            f"FX 24/5 new-entry gate: `{'ON' if self.fx_24_5_enabled else 'OFF'}`\n"
            f"Entry window: `{schedule}`\n"
            "Existing positions: `managed 24/7`"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(v8.v7.v5.v4.v3.base._signal, sig_name),
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
