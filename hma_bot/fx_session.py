"""FX-style 24/5 entry session gate.

New positions are allowed from Sunday 13:00 New York time (four hours before
the conventional FX weekly open at Sunday 17:00 NY) through Friday 17:00 NY.

Existing positions are never paused by this module: SL/TP, stage locks, recovery,
and position monitoring continue 24/7.  ZoneInfo handles US daylight-saving time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

# Conventional weekly FX boundary in New York local time.
FRIDAY_CLOSE_HOUR_NY = 17
SUNDAY_OPEN_HOUR_NY = 17
DEFAULT_PREOPEN_HOURS = 4


@dataclass(frozen=True)
class FxSessionState:
    entries_allowed: bool
    status: str
    now_utc: datetime
    now_ny: datetime
    next_entry_open_utc: datetime | None = None
    next_entry_open_ny: datetime | None = None


def _as_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(tz=UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _next_sunday_preopen(now_ny: datetime, preopen_hours: int) -> datetime:
    """Return next Sunday (17:00 - preopen_hours) in NY local time."""
    days_until_sunday = (6 - now_ny.weekday()) % 7
    candidate_date = (now_ny + timedelta(days=days_until_sunday)).date()
    candidate = datetime(
        candidate_date.year,
        candidate_date.month,
        candidate_date.day,
        SUNDAY_OPEN_HOUR_NY,
        tzinfo=NY,
    ) - timedelta(hours=preopen_hours)
    if candidate <= now_ny:
        candidate += timedelta(days=7)
    return candidate


def fx_entry_session(
    now: datetime | None = None,
    *,
    preopen_hours: int = DEFAULT_PREOPEN_HOURS,
) -> FxSessionState:
    """Return whether NEW entries are allowed under the FX 24/5 schedule.

    Allowed window in New York time:
      Sunday (17:00 - preopen_hours) <= time < Friday 17:00.

    With the default four-hour pre-open this is Sunday 13:00 NY through
    Friday 17:00 NY.  Saturday is always closed.
    """
    preopen_hours = max(0, min(int(preopen_hours), 23))
    now_utc = _as_utc(now)
    now_ny = now_utc.astimezone(NY)
    weekday = now_ny.weekday()  # Monday=0 ... Sunday=6

    allowed = False
    if 0 <= weekday <= 3:  # Mon-Thu
        allowed = True
    elif weekday == 4:  # Friday until 17:00 NY
        allowed = now_ny.hour < FRIDAY_CLOSE_HOUR_NY
    elif weekday == 6:  # Sunday from pre-open time
        preopen_hour = SUNDAY_OPEN_HOUR_NY - preopen_hours
        allowed = now_ny.hour >= preopen_hour

    if allowed:
        return FxSessionState(
            entries_allowed=True,
            status="OPEN_24_5",
            now_utc=now_utc,
            now_ny=now_ny,
        )

    next_ny = _next_sunday_preopen(now_ny, preopen_hours)
    return FxSessionState(
        entries_allowed=False,
        status="SLEEP_FX_WEEKEND",
        now_utc=now_utc,
        now_ny=now_ny,
        next_entry_open_utc=next_ny.astimezone(UTC),
        next_entry_open_ny=next_ny,
    )


def format_next_open(state: FxSessionState) -> str:
    if state.entries_allowed or state.next_entry_open_utc is None:
        return "entries open"
    utc_text = state.next_entry_open_utc.strftime("%a %Y-%m-%d %H:%M UTC")
    ny_text = state.next_entry_open_ny.strftime("%a %Y-%m-%d %H:%M NY")
    return f"{utc_text} ({ny_text})"
