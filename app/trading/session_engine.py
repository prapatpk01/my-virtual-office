"""
Extended Commodity Market Session Control
==========================================
Gates NEW entries to a 24/5-ish schedule anchored on the weekly XAU/
commodity market session (extended by a configurable buffer on each
side), applied uniformly to every configured symbol — including
cryptocurrencies, so the bot skips the low-volume weekend gap rather
than trading it 24/7.

This module owns ONE decision: is the effective trading window open
right now. It never touches position management — existing positions,
protective orders (SL/TP/trailing), Telegram alerts, and exchange
heartbeat all keep running regardless of session state (wired at the
call site: run_bot.py sets `bot.session_gate_open` from this engine's
output *before* on_tick; TradingBot._check_global_gates() only reads
that flag to block the SCANNING->FILTERING transition — every other
state in the machine, and check_price_protection's intrabar polling,
never consults it at all).

Weekly boundary math is done in the market's local timezone (session
hours are defined in local wall-clock time, e.g. "Sunday 18:00 New
York") via zoneinfo, so DST transitions shift the UTC instant of open/
close correctly across the year; the pre/post-open extension buffers
are then added as fixed-duration timedeltas once converted to UTC, so
"3 hours" always means 3 real hours regardless of any DST change that
happens to fall inside the buffer.
"""

import datetime as _dt
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger("session_engine")

_WEEKDAYS = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}


class TradingSessionState(str, Enum):
    PRE_OPEN_EXTENSION    = "PRE_OPEN_EXTENSION"
    ACTIVE                = "ACTIVE"
    POST_CLOSE_EXTENSION  = "POST_CLOSE_EXTENSION"
    SLEEP_MODE            = "SLEEP_MODE"
    SESSION_ERROR         = "SESSION_ERROR"


@dataclass
class TradingSessionDecision:
    state: TradingSessionState

    current_time_utc: _dt.datetime

    official_open_time: Optional[_dt.datetime]
    official_close_time: Optional[_dt.datetime]

    effective_open_time: Optional[_dt.datetime]
    effective_close_time: Optional[_dt.datetime]

    allow_new_positions: bool
    allow_position_management: bool

    next_official_open_time: Optional[_dt.datetime]
    next_effective_open_time: Optional[_dt.datetime]

    seconds_until_trading_resumes: int

    reason: str
    view_log_message: str


def _fmt(dt: Optional[_dt.datetime], tz: Optional[ZoneInfo]) -> str:
    if dt is None:
        return "unknown"
    local = dt.astimezone(tz) if tz else dt
    return local.strftime("%a %Y-%m-%d %H:%M %Z")


def _fmt_countdown(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _   = divmod(rem, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h or d: parts.append(f"{h}h")
    parts.append(f"{m}m")
    return " ".join(parts)


class TradingSessionEngine:
    """
    One instance shared across every symbol's bot (see run_bot.py) — the
    weekly session is a single global fact, not a per-symbol one.
    """

    def __init__(self,
                 reference_market: str = "XAU",
                 market_tz_name: str = "America/New_York",
                 open_weekday: str = "SUNDAY", open_hour: int = 18,
                 close_weekday: str = "FRIDAY", close_hour: int = 17,
                 pre_open_extension_hours: float = 3.0,
                 post_close_extension_hours: float = 3.0):
        self.reference_market = reference_market
        self.market_tz_name   = market_tz_name
        self.open_weekday     = _WEEKDAYS[open_weekday.strip().upper()]
        self.open_hour        = int(open_hour)
        self.close_weekday    = _WEEKDAYS[close_weekday.strip().upper()]
        self.close_hour       = int(close_hour)
        self.pre_open_extension_hours   = float(pre_open_extension_hours)
        self.post_close_extension_hours = float(post_close_extension_hours)

    # ── Weekly boundary math (market-local time) ─────────────────────────────

    def _floor_open_local(self, now_local: _dt.datetime) -> _dt.datetime:
        """Most recent official-open instant <= now_local, in market-local time."""
        tz = now_local.tzinfo
        days_back = (now_local.weekday() - self.open_weekday) % 7
        d = now_local.date() - _dt.timedelta(days=days_back)
        candidate = _dt.datetime(d.year, d.month, d.day, self.open_hour, 0, 0, tzinfo=tz)
        if candidate > now_local:
            d = d - _dt.timedelta(days=7)
            candidate = _dt.datetime(d.year, d.month, d.day, self.open_hour, 0, 0, tzinfo=tz)
        return candidate

    def _matching_close_local(self, official_open_local: _dt.datetime) -> _dt.datetime:
        """The close instant for the SAME weekly cycle as official_open_local."""
        tz = official_open_local.tzinfo
        offset_days = (self.close_weekday - self.open_weekday) % 7
        d = official_open_local.date() + _dt.timedelta(days=offset_days)
        return _dt.datetime(d.year, d.month, d.day, self.close_hour, 0, 0, tzinfo=tz)

    @staticmethod
    def _shift_weeks_local(local_dt: _dt.datetime, weeks: int) -> _dt.datetime:
        """Same wall-clock hour/minute, `weeks` calendar weeks later, in the
        same tz — NOT `local_dt + timedelta(weeks=weeks)`. Adding a timedelta
        to an already-resolved aware instant advances real elapsed time, which
        silently drifts the wall-clock hour by the DST offset whenever a
        transition falls inside that span (confirmed: this bug reproduced a
        real off-by-one-hour miss on the PRE_OPEN_EXTENSION window bracketing
        the 2026-03-08 US spring-forward). Rebuilding from the calendar date
        instead lets zoneinfo pick the correct UTC offset for that new date."""
        tz = local_dt.tzinfo
        d = local_dt.date() + _dt.timedelta(weeks=weeks)
        return _dt.datetime(d.year, d.month, d.day, local_dt.hour, local_dt.minute,
                            local_dt.second, tzinfo=tz)

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(self, now_utc: Optional[_dt.datetime] = None) -> TradingSessionDecision:
        if now_utc is None:
            now_utc = _dt.datetime.now(_dt.timezone.utc)
        if now_utc.tzinfo is None:
            raise ValueError("TradingSessionEngine.evaluate() requires a timezone-aware datetime")
        now_utc = now_utc.astimezone(_dt.timezone.utc)

        try:
            tz = ZoneInfo(self.market_tz_name)
        except ZoneInfoNotFoundError as e:
            return self._session_error(now_utc, f"unknown MARKET_SESSION_TIMEZONE '{self.market_tz_name}': {e}")

        try:
            now_local = now_utc.astimezone(tz)
            # `prev_*` = the most recently STARTED weekly cycle (floor-search —
            # this week's, or last week's if this week's open hasn't happened
            # yet). `next_*` = exactly one cycle (7 days) after that.
            #
            # [PRE-OPEN FIX] A naive 4-branch check against `prev_*` alone
            # mishandles the pre-open window: once `now` is past
            # prev_close+extension but still before THIS week's official
            # open, `prev_*` refers to a cycle that's fully closed out days
            # ago — none of the ACTIVE/POST_CLOSE/PRE_OPEN conditions against
            # it can ever match, so everything up to the official open
            # (including the 3h pre-open window itself) incorrectly fell
            # through to SLEEP_MODE. The pre-open check below must test
            # against `next_*` (the cycle that's about to start), not `prev_*`.
            prev_open_local  = self._floor_open_local(now_local)
            prev_close_local = self._matching_close_local(prev_open_local)
        except Exception as e:  # pragma: no cover — defensive, see fail-safe rule
            return self._session_error(now_utc, f"session schedule computation failed: {e}")

        next_open_local = self._shift_weeks_local(prev_open_local, 1)

        prev_open  = prev_open_local.astimezone(_dt.timezone.utc)
        prev_close = prev_close_local.astimezone(_dt.timezone.utc)
        prev_effective_open  = prev_open  - _dt.timedelta(hours=self.pre_open_extension_hours)
        prev_effective_close = prev_close + _dt.timedelta(hours=self.post_close_extension_hours)

        next_open  = next_open_local.astimezone(_dt.timezone.utc)
        next_effective_open = next_open - _dt.timedelta(hours=self.pre_open_extension_hours)

        if prev_open <= now_utc < prev_close:
            state, allow_new, reason = (
                TradingSessionState.ACTIVE, True,
                f"{self.reference_market} weekly session is open",
            )
        elif prev_close <= now_utc < prev_effective_close:
            remaining = _fmt_countdown((prev_effective_close - now_utc).total_seconds())
            state, allow_new, reason = (
                TradingSessionState.POST_CLOSE_EXTENSION, True,
                f"Trading remains enabled for {self.post_close_extension_hours:g}h after the "
                f"official {self.reference_market} weekly close ({remaining} left)",
            )
        elif next_effective_open <= now_utc < next_open:
            state, allow_new, reason = (
                TradingSessionState.PRE_OPEN_EXTENSION, True,
                f"Trading starts {self.pre_open_extension_hours:g}h before the official "
                f"{self.reference_market} weekly opening",
            )
        else:
            state, allow_new, reason = (
                TradingSessionState.SLEEP_MODE, False,
                f"{self.reference_market} weekly session is outside the extended trading window",
            )

        # official_*/effective_* always describe the most-recently-STARTED
        # cycle (matches ACTIVE/POST_CLOSE_EXTENSION exactly); next_* is
        # always exactly one cycle later — the log picks whichever pair is
        # actually relevant to the current state (see _render_log).
        seconds_until_resume = (
            0 if allow_new else max(int((next_effective_open - now_utc).total_seconds()), 0)
        )

        msg = self._render_log(
            state, tz, prev_open, prev_close, prev_effective_open, prev_effective_close,
            next_open, next_effective_open, now_utc, seconds_until_resume,
        )

        return TradingSessionDecision(
            state=state,
            current_time_utc=now_utc,
            official_open_time=prev_open,
            official_close_time=prev_close,
            effective_open_time=prev_effective_open,
            effective_close_time=prev_effective_close,
            allow_new_positions=allow_new,
            allow_position_management=True,
            next_official_open_time=next_open,
            next_effective_open_time=next_effective_open,
            seconds_until_trading_resumes=seconds_until_resume,
            reason=reason,
            view_log_message=msg,
        )

    # ── Fail-safe (section 10) ───────────────────────────────────────────

    def _session_error(self, now_utc: _dt.datetime, reason: str) -> TradingSessionDecision:
        logger.error("[SESSION] SESSION_ERROR: %s — new entries disabled, retrying next tick", reason)
        msg = (
            f"[SESSION] SESSION_ERROR\n"
            f"Reason                   : {reason}\n"
            f"New Positions            : DISABLED\n"
            f"Existing Positions       : ACTIVE (managed normally)\n"
            f"Protective Orders        : PRESERVED\n"
            f"Action                   : retrying session schedule load every tick"
        )
        return TradingSessionDecision(
            state=TradingSessionState.SESSION_ERROR,
            current_time_utc=now_utc,
            official_open_time=None, official_close_time=None,
            effective_open_time=None, effective_close_time=None,
            allow_new_positions=False, allow_position_management=True,
            next_official_open_time=None, next_effective_open_time=None,
            seconds_until_trading_resumes=0,
            reason=reason,
            view_log_message=msg,
        )

    # ── View log (section 7) ──────────────────────────────────────────────

    def _render_log(self, state, tz, official_open, official_close,
                    effective_open, effective_close, next_official_open,
                    next_effective_open, now_utc, seconds_until_resume) -> str:
        rm = self.reference_market

        if state == TradingSessionState.PRE_OPEN_EXTENSION:
            return (
                f"[SESSION] PRE-OPEN EXTENSION\n"
                f"Reference Market         : {rm} / Commodity\n"
                f"Official Market Status   : CLOSED\n"
                f"New Positions            : ENABLED\n"
                f"Existing Positions       : ACTIVE\n"
                f"Entry Pipeline           : ENABLED\n"
                f"Position Management      : ACTIVE\n"
                f"Reason                   : Trading starts {self.pre_open_extension_hours:g} hours "
                f"before the official {rm} weekly opening\n"
                f"Official Market Open     : {_fmt(next_official_open, tz)}\n"
                f"Extended Trading Start   : {_fmt(next_effective_open, tz)}"
            )

        if state == TradingSessionState.ACTIVE:
            return (
                f"[SESSION] ACTIVE\n"
                f"Reference Market         : {rm} / Commodity\n"
                f"Official Market Status   : OPEN\n"
                f"New Positions            : ENABLED\n"
                f"Existing Positions       : ACTIVE\n"
                f"Entry Pipeline           : ENABLED\n"
                f"Position Management      : ACTIVE"
            )

        if state == TradingSessionState.POST_CLOSE_EXTENSION:
            return (
                f"[SESSION] POST-CLOSE EXTENSION\n"
                f"Reference Market         : {rm} / Commodity\n"
                f"Official Market Status   : CLOSED\n"
                f"New Positions            : ENABLED\n"
                f"Existing Positions       : ACTIVE\n"
                f"Entry Pipeline           : ENABLED\n"
                f"Position Management      : ACTIVE\n"
                f"Reason                   : Trading remains enabled for "
                f"{self.post_close_extension_hours:g} hours after the official {rm} weekly close\n"
                f"Official Market Close    : {_fmt(official_close, tz)}\n"
                f"Extended Trading End     : {_fmt(effective_close, tz)}\n"
                f"Remaining Extension Time : {_fmt_countdown((effective_close - now_utc).total_seconds())}"
            )

        # SLEEP_MODE
        return (
            f"[SESSION] SLEEP MODE\n"
            f"Status                   : Temporarily paused\n"
            f"Reason                   : {rm} / commodity market weekend session is outside "
            f"the extended trading window\n"
            f"New Positions            : DISABLED\n"
            f"New Entry Orders         : DISABLED\n"
            f"Existing Positions       : ACTIVE\n"
            f"Stop-Loss Management     : ACTIVE\n"
            f"Take-Profit Management   : ACTIVE\n"
            f"Trailing-Stop Management : ACTIVE\n"
            f"Emergency Exit           : ACTIVE\n"
            f"Next Official Market Open: {_fmt(next_official_open, tz)}\n"
            f"Trading Resumes At       : {_fmt(next_effective_open, tz)}\n"
            f"Time Remaining           : {_fmt_countdown(seconds_until_resume)}"
        )
