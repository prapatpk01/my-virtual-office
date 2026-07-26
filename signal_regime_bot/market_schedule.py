"""FX-style weekly sleep schedule for NEW entries.

The bot trades OKX instruments, but the user wants every symbol to follow the
standard FX weekly rhythm.  Existing positions are NEVER disabled by this
module; it only controls whether a new position may be opened.

Baseline FX session: Sunday 17:00 -> Friday 17:00 America/New_York.
User preference: wake 4 hours before the normal Sunday open, so new entries
resume Sunday 13:00 New York time.  ZoneInfo makes this DST-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class FxSleepStatus:
    sleeping: bool
    phase: str  # SLEEP, PREOPEN, OPEN
    now_market: datetime
    next_entry_resume: datetime | None
    next_regular_open: datetime | None
    reason: str

    @property
    def mode_label(self) -> str:
        if self.sleeping:
            return "SLEEP MODE"
        if self.phase == "PREOPEN":
            return "PRE-OPEN MODE"
        return "TRADING MODE"


class FxMarketSleepSchedule:
    """Weekly FX schedule applied globally to all configured symbols."""

    def __init__(self, cfg):
        self.enabled = bool(getattr(cfg, "fx_sleep_mode_enabled", True))
        self.market_tz_name = str(getattr(cfg, "fx_market_timezone", "America/New_York"))
        self.close_hour = int(getattr(cfg, "fx_weekly_close_hour", 17))
        self.open_hour = int(getattr(cfg, "fx_weekly_open_hour", 17))
        self.preopen_hours = max(0, int(getattr(cfg, "fx_preopen_hours", 4)))
        try:
            self.tz = ZoneInfo(self.market_tz_name)
        except ZoneInfoNotFoundError:
            # UTC fallback is fail-safe for startup, but the Docker image ships
            # Python tzdata so production should use America/New_York.
            self.tz = timezone.utc
            self.market_tz_name = "UTC"

    def status(self, now: datetime | None = None) -> FxSleepStatus:
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        market_now = now.astimezone(self.tz)

        if not self.enabled:
            return FxSleepStatus(False, "OPEN", market_now, None, None,
                                 "FX sleep mode disabled")

        weekday = market_now.weekday()  # Mon=0 ... Sun=6
        t_minutes = market_now.hour * 60 + market_now.minute
        close_minutes = self.close_hour * 60
        open_minutes = self.open_hour * 60
        preopen_minutes = (open_minutes - self.preopen_hours * 60) % (24 * 60)

        sleeping = False
        phase = "OPEN"

        if weekday == 4 and t_minutes >= close_minutes:  # Friday after weekly close
            sleeping = True
            phase = "SLEEP"
        elif weekday == 5:  # Saturday
            sleeping = True
            phase = "SLEEP"
        elif weekday == 6:  # Sunday
            if t_minutes < preopen_minutes:
                sleeping = True
                phase = "SLEEP"
            elif t_minutes < open_minutes:
                sleeping = False
                phase = "PREOPEN"
            else:
                sleeping = False
                phase = "OPEN"

        next_resume = None
        next_regular = None
        if sleeping:
            days_to_sunday = (6 - weekday) % 7
            sunday = (market_now + timedelta(days=days_to_sunday)).date()
            # If somehow evaluated on Sunday after the resume threshold while
            # still sleeping, choose next week's Sunday defensively.
            resume = datetime(sunday.year, sunday.month, sunday.day,
                              self.open_hour, tzinfo=self.tz) - timedelta(hours=self.preopen_hours)
            regular = datetime(sunday.year, sunday.month, sunday.day,
                               self.open_hour, tzinfo=self.tz)
            if resume <= market_now:
                resume += timedelta(days=7)
                regular += timedelta(days=7)
            next_resume = resume
            next_regular = regular
            reason = (
                f"FX weekly closure; new entries resume {resume.strftime('%a %Y-%m-%d %H:%M %Z')} "
                f"({self.preopen_hours}h before regular FX open {regular.strftime('%H:%M %Z')})"
            )
        elif phase == "PREOPEN":
            regular = datetime(market_now.year, market_now.month, market_now.day,
                               self.open_hour, tzinfo=self.tz)
            next_regular = regular
            reason = (
                f"pre-open window active; new entries enabled {self.preopen_hours}h before "
                f"regular FX open at {regular.strftime('%H:%M %Z')}"
            )
        else:
            reason = "FX-style new-entry window open"

        return FxSleepStatus(sleeping, phase, market_now, next_resume, next_regular, reason)

    def format_resume(self, status: FxSleepStatus) -> str:
        if status.next_entry_resume is None:
            return "now"
        local = status.next_entry_resume
        utc = local.astimezone(timezone.utc)
        return (
            f"{local.strftime('%a %Y-%m-%d %H:%M %Z')} "
            f"({utc.strftime('%Y-%m-%d %H:%M UTC')})"
        )
