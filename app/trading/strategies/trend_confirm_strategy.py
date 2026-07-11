"""
TF30M Trend-Confirmed EMA5/10 Cross Strategy.

Two-step, in order — trend must be confirmed BEFORE an entry is even
looked for, and the entry that gets acted on must occur at-or-after the
bar the trend confirmed, never before:

Step 1 — Trend gate (SMA30 and MACD must agree, or no trade at all):
  SMA30 trend : candle opens above SMA30 -> uptrend, opens below -> downtrend
  MACD trend  : raw MACD LINE (fast EMA - slow EMA, the "MACD 4C" zero-line
                read, not the line-vs-signal read) above zero -> uptrend,
                below zero -> downtrend. Matches the user-supplied Pine
                script (macd() line only, colored purely by sign +
                accelerating/decelerating — the trend read here only uses
                the sign, i.e. currMacd > 0 / < 0).
  Every 30m bar, the confirmed direction ("up" / "down" / None-if-undecided
  or conflicting) is tracked in self._trend_state, purely for logging/the
  grace window below — it does NOT limit how many trades a single trend
  streak can produce. After ANY close, the next entry just needs Step 1 to
  currently read confirmed (in either the same or a new direction) before
  Step 2 looks for a fresh cross — "close, reconfirm the trend, find the
  next entry in that trend," not "one trade per streak."

Step 2 — Entry (whenever the trend gate above is confirmed and no
position is open):
  1. EMA5 crosses EMA10 the matching way             (30m)
  2. Price is within 1.5xATR of SMA30                (30m)
  Because cross_up/cross_down are one-shot transition events, they're only
  ever True on their own bar. self._last_cross_up_ts/_last_cross_down_ts
  record when each last happened, regardless of trend state, so a cross
  that occurred UP TO 3 BARS before the trend confirms is still eligible —
  it doesn't have to occur on the confirmation bar itself, just within that
  grace window (0 = same bar). This is exactly the "EMA cross that closed
  the previous trend's position now also becomes the trigger for the new
  trend's entry" case: a long's exit cross often IS a fresh cross for the
  opposite direction, just a bar or two ahead of SMA30/MACD catching up
  and confirming the new trend. Older than 3 bars, or a cross from the
  SAME direction already traded this streak, doesn't count — the strategy
  keeps waiting for the next fresh cross instead.
  If the cross (same-bar or within the grace window) fires but price is
  already more than 1.5xATR from SMA30, the setup fails outright — no
  chase, and a later pullback + fresh cross within the same confirmed
  trend can still try again.

Exit — OR logic, whichever happens first while SL/TP hasn't been hit yet:
  Exit LONG:  candle opens below SMA30 (30m)  OR  EMA5 crosses below EMA10
  Exit SHORT: candle opens above SMA30 (30m)  OR  EMA5 crosses above EMA10
  SMA30 alone let chop churn the position less (a live chart showed a
  cluster of EMA crosses all within one still-uptrending swing that a
  pure-cross exit would've closed on repeatedly) — but the EMA cross is
  added back as a second, faster trigger: if either condition fires, close
  immediately. SL/TP (ATR(14,30m)x1.5, 1:1 R:R) remains the hard-stop
  safety net checked by bot.py's risk-manager fallback underneath both.

Once closed (by either exit condition, or the hard SL/TP stop), the very
next bar where Step 1's trend gate reads confirmed — same direction as
before or a fresh flip, doesn't matter — is eligible for a new entry via
Step 2. No cooldown, no "used up this streak."

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze() — in
hedge mode a SELL Signal always OPENS a new short rather than closing an
open long. tick_open_position()'s PositionUpdate("close") always closes
whichever position is actually open, regardless of hedge mode.

TF30m candles aren't fetched separately from the exchange — they're
resampled from the 15m series this strategy already receives, and the
last (possibly still-forming) 30m bucket is dropped so entries/exits only
ever evaluate against a genuinely closed 30m bar.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


class TrendConfirmStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        ema_fast: int = 5,
        ema_slow: int = 10,
        sma_trend: int = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        max_dist_atr_mult: float = 1.5,
        sl_atr_mult: float = 1.5,
        rr_ratio: float = 1.0,
        cross_grace_bars: int = 3,
    ):
        super().__init__(symbol, params)
        self.name = f"TrendConfirm({symbol})"
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sma_trend = sma_trend
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.max_dist_atr_mult = max_dist_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio
        self.cross_grace_bars = cross_grace_bars

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._trend_state: Optional[str] = None      # "up" | "down" | None
        self._last_cross_up_ts: Optional[int] = None
        self._last_cross_down_ts: Optional[int] = None
        self._last_bar_ts: Optional[int] = None       # owned by analyze(): indicator/cross/trend tracking
        self._last_exit_bar_ts: Optional[int] = None   # owned by tick_open_position(): exit evaluation only
        self._latest_candles: list = []

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles  # cached for tick_open_position()

        c30 = self._closed_30m_bars(candles)
        min_needed = max(self.ema_slow, self.sma_trend,
                         self.macd_slow + self.macd_signal, self.atr_period) + 5
        if len(c30) < min_needed:
            return self._hold(current_price, f"Need {min_needed}+ closed 30m bars, have {len(c30)}")

        bar_ts = c30[-1].timestamp
        is_new_bar = bar_ts != self._last_bar_ts

        ind = self._indicators(c30)
        if ind is None:
            return self._hold(current_price, "Indicators still warming up (30m)")

        def dbg(entry_status: str, dist_atr: Optional[float] = None) -> dict:
            sma_trend = "up" if ind["sma_up"] else "down" if ind["sma_down"] else "flat"
            macd_trend = "up" if ind["macd_up"] else "down" if ind["macd_down"] else "flat"
            return {"trend_confirm": {
                "sma_trend": sma_trend,
                "macd_trend": macd_trend,
                "confirmed": self._trend_state,
                "open_position": self._open_position,
                "entry_status": entry_status,
                "dist_atr": round(dist_atr, 2) if dist_atr is not None else None,
                "max_dist_atr": self.max_dist_atr_mult,
            }}

        # Cross/trend tracking must run on every new bar REGARDLESS of
        # whether a position is currently open — otherwise a cross that
        # happens while holding (e.g. the very cross that closes the
        # position via tick_open_position) would never get recorded, and
        # the grace window below would never see it once the position
        # closes and a fresh trend confirms a bar or two later.
        if is_new_bar:
            self._last_bar_ts = bar_ts
            if ind["cross_up"]:
                self._last_cross_up_ts = bar_ts
            if ind["cross_down"]:
                self._last_cross_down_ts = bar_ts

            # ── Step 1: Trend gate — SMA30 and MACD must agree, or no trade ─
            sma_up, sma_down = ind["sma_up"], ind["sma_down"]
            macd_up, macd_down = ind["macd_up"], ind["macd_down"]
            trend_up   = sma_up and macd_up
            trend_down = sma_down and macd_down
            self._trend_state = "up" if trend_up else "down" if trend_down else None

        new_trend = self._trend_state

        if self._open_position is not None:
            return self._hold(current_price, f"Holding {self._open_position.upper()} — managed via tick_open_position()",
                              metadata=dbg("position_open"))

        if not is_new_bar:
            return self._hold(current_price, "Waiting for the next 30m bar close", metadata=dbg("waiting_next_bar"))

        if new_trend is None:
            return self._hold(current_price, "No trade: SMA30/MACD trend not confirmed or conflicting",
                              metadata=dbg("no_trend"))

        # ── Step 2: Entry — EMA5/10 cross in the confirmed direction only,
        # occurring at-or-after the bar the trend confirmed (guaranteed by
        # cross_up/cross_down being one-shot: a cross from a prior trend
        # streak can never read True again on a later bar) ─────────────────
        close_price = c30[-1].close
        dist_atr = abs(close_price - ind["sma_val"]) / ind["atr_val"] if ind["atr_val"] > 0 else 999.0
        bucket_ms = 30 * 60_000

        def _bars_ago(ts: Optional[int]) -> Optional[int]:
            return (bar_ts - ts) // bucket_ms if ts is not None else None

        if new_trend == "up":
            up_ago = _bars_ago(self._last_cross_up_ts)
            cross_ok = up_ago is not None and 0 <= up_ago <= self.cross_grace_bars
            if ind["cross_down"] and not cross_ok:
                return self._hold(current_price, "Trend UP confirmed but EMA5/10 crossed down — no counter-trend entry",
                                  metadata=dbg("counter_cross_blocked", dist_atr))
            if cross_ok:
                if dist_atr > self.max_dist_atr_mult:
                    return self._hold(current_price,
                        f"Long setup FAILED: price {dist_atr:.2f}xATR above SMA30 (max {self.max_dist_atr_mult}x) "
                        f"— waiting for pullback + fresh EMA cross",
                        metadata=dbg("cross_pass_distance_fail", dist_atr))
                sl, tp = self._compute_sl_tp("long", close_price, ind["atr_val"])
                self._open_position = "long"
                self._last_cross_up_ts = None
                grace_note = f" ({up_ago} bar(s) before trend confirmed)" if up_ago > 0 else ""
                meta = dbg("entered", dist_atr)
                meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
                return Signal(
                    type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                    reason=f"Uptrend confirmed (SMA30+MACD 30m) + EMA5↑EMA10 within {dist_atr:.2f}xATR of SMA30{grace_note}",
                    confidence=1.0,
                    metadata=meta,
                )
            return self._hold(current_price, "Trend UP confirmed — waiting for EMA5/10 cross",
                              metadata=dbg("waiting_cross"))

        # new_trend == "down"
        down_ago = _bars_ago(self._last_cross_down_ts)
        cross_ok = down_ago is not None and 0 <= down_ago <= self.cross_grace_bars
        if ind["cross_up"] and not cross_ok:
            return self._hold(current_price, "Trend DOWN confirmed but EMA5/10 crossed up — no counter-trend entry",
                              metadata=dbg("counter_cross_blocked", dist_atr))
        if cross_ok:
            if dist_atr > self.max_dist_atr_mult:
                return self._hold(current_price,
                    f"Short setup FAILED: price {dist_atr:.2f}xATR below SMA30 (max {self.max_dist_atr_mult}x) "
                    f"— waiting for pullback + fresh EMA cross",
                    metadata=dbg("cross_pass_distance_fail", dist_atr))
            sl, tp = self._compute_sl_tp("short", close_price, ind["atr_val"])
            self._open_position = "short"
            self._last_cross_down_ts = None
            grace_note = f" ({down_ago} bar(s) before trend confirmed)" if down_ago > 0 else ""
            meta = dbg("entered", dist_atr)
            meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
                reason=f"Downtrend confirmed (SMA30+MACD 30m) + EMA5↓EMA10 within {dist_atr:.2f}xATR of SMA30{grace_note}",
                confidence=1.0,
                metadata=meta,
            )
        return self._hold(current_price, "Trend DOWN confirmed — waiting for EMA5/10 cross",
                          metadata=dbg("waiting_cross"))

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Exit = OR logic: candle opens on the wrong side of SMA30, OR
        EMA5/10 crosses back — whichever fires first, evaluated only on a
        newly-closed 30m bar. Hedge-mode-safe: always closes whichever
        position is actually open, never relies on signal.type semantics."""
        if self._open_position is None or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate

        c30 = self._closed_30m_bars(self._latest_candles)
        min_needed = max(self.ema_slow, self.sma_trend,
                         self.macd_slow + self.macd_signal, self.atr_period) + 5
        if len(c30) < min_needed:
            return PositionUpdate(action="hold", reason="Indicators warming up (30m)")

        bar_ts = c30[-1].timestamp
        if bar_ts == self._last_exit_bar_ts:
            return PositionUpdate(action="hold", reason="Waiting for the next 30m bar close")

        ind = self._indicators(c30)
        if ind is None:
            return PositionUpdate(action="hold", reason="Indicators warming up (30m)")
        self._last_exit_bar_ts = bar_ts

        if self._open_position == "long" and (ind["sma_down"] or ind["cross_down"]):
            reason = "candle opened below SMA30" if ind["sma_down"] else "EMA5 crossed below EMA10"
            self._open_position = None
            return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit LONG: {reason} (30m)")
        if self._open_position == "short" and (ind["sma_up"] or ind["cross_up"]):
            reason = "candle opened above SMA30" if ind["sma_up"] else "EMA5 crossed above EMA10"
            self._open_position = None
            return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit SHORT: {reason} (30m)")

        return PositionUpdate(action="hold", reason=f"Holding {self._open_position.upper()}")

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Called by bot.py after ANY close, including the risk-manager's
        hard SL/TP fallback firing before EMA5/10 crosses back — without
        this, _open_position would stay set forever and analyze() would
        refuse all future entries."""
        self._open_position = None

    def cancel_pending_entry(self, reason: str = "") -> None:
        """Called by bot.py when a signal this strategy just emitted failed
        to actually open (rejected by risk/portfolio gates, insufficient
        balance, or an order error)."""
        self._open_position = None

    def attach_existing_position(self, direction: str, entry_price: float,
                                  stop_loss: Optional[float] = None,
                                  take_profit: Optional[float] = None) -> None:
        """Called once on bot startup when a position for this symbol is
        already open on the exchange (from before a restart) — nothing in
        this strategy's in-memory state would otherwise know about it, so
        analyze() would try to open a duplicate and tick_open_position()
        would never manage the exit."""
        self._open_position = direction

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_sl_tp(self, direction: str, price: float, atr: float) -> tuple[float, float]:
        dist = self.sl_atr_mult * atr
        if direction == "long":
            sl = price - dist
            tp = price + dist * self.rr_ratio
        else:
            sl = price + dist
            tp = price - dist * self.rr_ratio
        return sl, tp

    def _indicators(self, c30: list) -> Optional[dict]:
        closes = [c.close for c in c30]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        sma_t = self.sma(closes, self.sma_trend)
        macd_line, macd_signal_line, hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        atr_arr = self.atr(c30, self.atr_period)

        needed = [ema_f[-1], ema_f[-2], ema_s[-1], ema_s[-2], sma_t[-1],
                 macd_line[-1], macd_signal_line[-1], hist[-1], atr_arr[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        last = c30[-1]
        return {
            "cross_up":   ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1],
            "cross_down": ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1],
            "sma_up":   last.open > sma_t[-1],
            "sma_down": last.open < sma_t[-1],
            "sma_val":  float(sma_t[-1]),
            # "MACD 4C" trend read: raw MACD line vs zero, not vs its
            # signal line — matches the user-supplied Pine script, which
            # only plots/colors currMacd by sign (+ momentum direction).
            "macd_up":   macd_line[-1] > 0,
            "macd_down": macd_line[-1] < 0,
            "atr_val":  float(atr_arr[-1]),
        }

    @staticmethod
    def _closed_30m_bars(candles_15m: list) -> list:
        """Resample 15m -> 30m, dropping the last bucket if it isn't closed
        yet (i.e. the input series hasn't reached its 30m boundary)."""
        if not candles_15m:
            return []
        bucket_ms = 30 * 60_000
        buckets: dict[int, list] = {}
        for c in candles_15m:
            key = (c.timestamp // bucket_ms) * bucket_ms
            buckets.setdefault(key, []).append(c)

        class _Bar:
            __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
            def __init__(self, ts, o, h, l, cl, v):
                self.timestamp = ts; self.open = o; self.high = h
                self.low = l; self.close = cl; self.volume = v

        keys = sorted(buckets)
        out = []
        for key in keys:
            grp = buckets[key]
            out.append(_Bar(
                key, grp[0].open, max(g.high for g in grp), min(g.low for g in grp),
                grp[-1].close, sum(g.volume for g in grp),
            ))

        last_c15_end = candles_15m[-1].timestamp + 15 * 60_000
        last_bucket_end = keys[-1] + bucket_ms
        if last_c15_end < last_bucket_end:
            out = out[:-1]

        return out

    def _hold(self, price: float, reason: str = "", metadata: Optional[dict] = None) -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata=metadata or {},
        )
