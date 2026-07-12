"""
3-Layer Trend-Confirmed Multi-TF Strategy.

Layer 1 — Trend confirmation (TF30m):
  SMA30      : candle opens above SMA30 -> up, below -> down
  EMA10/20   : EMA10 > EMA20 -> up, EMA10 < EMA20 -> down
  EMA20 slope: EMA20 higher than it was `ema_slope_lookback` bars ago -> up,
               lower -> down
  MACD       : raw MACD LINE (fast EMA - slow EMA) vs zero, the "MACD 4C"
               zero-line read (not line-vs-signal) — matches the
               user-supplied Pine script, sign only.
  ALL FOUR must agree on the same direction, or there is no confirmed
  trend at all (self._trend_state = None) and nothing below runs.
  Tracked once per new 30m bar in self._trend_state, purely for
  logging/gating — a confirmed trend does not expire on its own; it just
  gets re-evaluated fresh every 30m bar.

Layer 2 — Bias filter (TF15m + TF1H):
  BaseStrategy.compute_mtf_bias() (same formula ai_expert uses: per-TF
  score from close>EMA20 / EMA20>EMA50 / RSI lean, weighted 15m x1 + 1H
  x2 — 4H excluded here via w3=0) is converted from its native -100..+100
  scale to 0-100 (50 = neutral) via `50 + pct/2`. Must AGREE with and
  exceed Layer 1's confirmed direction:
    uptrend   needs bias_score > bias_threshold        (default 60)
    downtrend needs bias_score < 100 - bias_threshold  (default 40)
  Layer 1 confirmed but Layer 2 disagreeing/weak = no trade.

Layer 3 — Entry (TF15m):
  EMA5/EMA10 cross AND HMA10/HMA20 cross must BOTH fire, in the
  confirmed direction, within `cross_grace_bars` (default 2) bars of
  each other — HMA tends to cross a bar or two before/after EMA on the
  same swing, so whichever fires first still counts as long as the
  other follows within the grace window (0 = same bar, up to
  cross_grace_bars apart either order). Long only when Layer1+Layer2
  both say "up"; short only when both say "down". Each side's cross
  timestamp is consumed (reset to None) once it triggers an entry, so a
  stale old cross can't silently satisfy a future setup.

Exit — OR logic, whichever fires first while SL/TP hasn't been hit yet:
  EMA5/10 cross-back  OR  HMA10/20 cross-back  ->  close 100% immediately.
  SL/TP (ATR(14, 15m) x1.5, 1:1 R:R by default) remains the hard-stop
  safety net checked by bot.py's risk-manager fallback underneath this.

Once closed (by either exit condition or the hard SL/TP stop), the very
next bar where Layer1+Layer2 read confirmed again — same direction as
before or a fresh flip — is eligible for a new Layer3 entry. No
cooldown, no "used up this streak."

Exits are evaluated in tick_open_position() (called by bot.py every tick for
an open position), NOT by returning a SELL/BUY Signal from analyze() — in
hedge mode a SELL Signal always OPENS a new short rather than closing an
open long. tick_open_position()'s PositionUpdate("close") always closes
whichever position is actually open, regardless of hedge mode.

TF30m candles for Layer 1 aren't fetched separately from the exchange —
they're resampled from the 15m series this strategy already receives,
and the last (possibly still-forming) 30m bucket is dropped so Layer 1
only ever evaluates against a genuinely closed 30m bar. Layer 2/3 run
directly on the raw 15m candles (and the 1H candles passed in via
mtf_candles), matching how every other pure-TF15m strategy in this
codebase evaluates crosses (no separate "closed 15m" bookkeeping).
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
        # Layer 1 — trend confirmation (30m)
        sma_trend: int = 30,
        ema1_period: int = 10,
        ema2_period: int = 20,
        ema_slope_lookback: int = 5,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # Layer 2 — bias filter (15m + 1H)
        bias_threshold: float = 60.0,
        # Layer 3 — entry (15m)
        ema_fast: int = 5,
        ema_slow: int = 10,
        hma_fast: int = 10,
        hma_slow: int = 20,
        cross_grace_bars: int = 2,
        # Risk
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        rr_ratio: float = 1.0,
    ):
        super().__init__(symbol, params)
        self.name = f"TrendConfirm({symbol})"

        self.sma_trend = sma_trend
        self.ema1_period = ema1_period
        self.ema2_period = ema2_period
        self.ema_slope_lookback = ema_slope_lookback
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        self.bias_threshold = bias_threshold

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.hma_fast = hma_fast
        self.hma_slow = hma_slow
        self.cross_grace_bars = cross_grace_bars

        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._trend_state: Optional[str] = None      # "up" | "down" | None — Layer 1 result
        self._last_bar_ts_30: Optional[int] = None    # Layer 1 new-bar tracking
        self._last_bar_ts_15: Optional[int] = None     # Layer 3 new-bar tracking
        self._last_ema_cross_up_ts: Optional[int] = None
        self._last_ema_cross_down_ts: Optional[int] = None
        self._last_hma_cross_up_ts: Optional[int] = None
        self._last_hma_cross_down_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None  # owned by tick_open_position()
        self._latest_candles: list = []

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles  # cached for tick_open_position()
        mtf = mtf_candles or {}

        # ── Layer 1: Trend confirmation (30m) ──────────────────────────────
        c30 = self._closed_30m_bars(candles)
        min_needed_30 = max(self.ema2_period + self.ema_slope_lookback, self.sma_trend,
                            self.macd_slow + self.macd_signal) + 5
        if len(c30) < min_needed_30:
            return self._hold(current_price, f"Layer1: need {min_needed_30}+ closed 30m bars, have {len(c30)}")

        bar_ts_30 = c30[-1].timestamp
        is_new_bar_30 = bar_ts_30 != self._last_bar_ts_30

        l1 = self._layer1_indicators(c30)
        if l1 is None:
            return self._hold(current_price, "Layer1: indicators still warming up (30m)")

        if is_new_bar_30:
            self._last_bar_ts_30 = bar_ts_30
            self._trend_state = l1["trend"]

        trend = self._trend_state

        # ── Layer 3 cross tracking — runs on every new 15m bar REGARDLESS of
        # Layer1/Layer2 gating below, so a cross that fires before the trend
        # confirms (or while a position is open) is still remembered within
        # the grace window once everything else lines up. HMA tends to cross
        # a bar or two before/after EMA on the same swing — cross_grace_bars
        # lets whichever one fires first count, as long as the other follows
        # within that many bars, instead of requiring the exact same bar. ──
        bar_ts_15 = candles[-1].timestamp
        is_new_bar_15 = bar_ts_15 != self._last_bar_ts_15
        l3 = self._layer3_indicators(candles)
        if is_new_bar_15:
            self._last_bar_ts_15 = bar_ts_15
            if l3 is not None:
                if l3["ema_cross_up"]:
                    self._last_ema_cross_up_ts = bar_ts_15
                if l3["ema_cross_down"]:
                    self._last_ema_cross_down_ts = bar_ts_15
                if l3["hma_cross_up"]:
                    self._last_hma_cross_up_ts = bar_ts_15
                if l3["hma_cross_down"]:
                    self._last_hma_cross_down_ts = bar_ts_15

        def _bars_ago_15(ts: Optional[int]) -> Optional[int]:
            return (bar_ts_15 - ts) // (15 * 60_000) if ts is not None else None

        ema_up_ago    = _bars_ago_15(self._last_ema_cross_up_ts)
        ema_down_ago  = _bars_ago_15(self._last_ema_cross_down_ts)
        hma_up_ago    = _bars_ago_15(self._last_hma_cross_up_ts)
        hma_down_ago  = _bars_ago_15(self._last_hma_cross_down_ts)
        gb = self.cross_grace_bars
        dual_cross_up   = (ema_up_ago is not None and ema_up_ago <= gb and
                           hma_up_ago is not None and hma_up_ago <= gb)
        dual_cross_down = (ema_down_ago is not None and ema_down_ago <= gb and
                           hma_down_ago is not None and hma_down_ago <= gb)

        # ── Layer 2: Bias filter (15m + 1H) ────────────────────────────────
        bias_pct, bias_label = self.compute_mtf_bias(candles, {"1h": mtf.get("1h", [])}, w3=0.0)
        bias_score = 50.0 + bias_pct / 2.0

        def dbg(entry_status: str) -> dict:
            return {"trend_confirm": {
                "sma_trend": l1["sma_dir"], "ema10_20_trend": l1["ema1020_dir"],
                "ema20_slope": l1["slope_dir"], "macd_trend": l1["macd_dir"],
                "confirmed": trend,
                "bias_score": round(bias_score, 1), "bias_threshold": self.bias_threshold,
                "open_position": self._open_position,
                "entry_status": entry_status,
                "cross_grace_bars": gb,
                "ema_cross_up_ago": ema_up_ago, "ema_cross_down_ago": ema_down_ago,
                "hma_cross_up_ago": hma_up_ago, "hma_cross_down_ago": hma_down_ago,
            }}

        if self._open_position is not None:
            return self._hold(current_price, f"Holding {self._open_position.upper()} — managed via tick_open_position()",
                              metadata=dbg("position_open"))

        if trend is None:
            return self._hold(current_price,
                "Layer1: SMA30/EMA10-20/EMA20 slope/MACD not confirmed or conflicting",
                metadata=dbg("no_trend"))

        layer2_pass = (
            (trend == "up" and bias_score > self.bias_threshold) or
            (trend == "down" and bias_score < (100.0 - self.bias_threshold))
        )
        if not layer2_pass:
            return self._hold(current_price,
                f"Layer2: bias score {bias_score:.0f} doesn't confirm {trend} "
                f"(need {'>' if trend == 'up' else '<'}"
                f"{self.bias_threshold if trend == 'up' else 100 - self.bias_threshold:.0f})",
                metadata=dbg("bias_fail"))

        # ── Layer 3: Entry (15m) — EMA5/10 AND HMA10/20 must BOTH have
        # crossed in the confirmed direction within cross_grace_bars of
        # each other (whichever fired first still counts) ─────────────────
        if l3 is None:
            return self._hold(current_price, "Layer3: indicators still warming up (15m)", metadata=dbg("no_trend"))

        close_price = candles[-1].close

        if trend == "up":
            if dual_cross_up:
                sl, tp = self._compute_sl_tp("long", close_price, l3["atr_val"])
                self._open_position = "long"
                self._last_ema_cross_up_ts = None
                self._last_hma_cross_up_ts = None
                meta = dbg("entered")
                meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
                return Signal(
                    type=SignalType.BUY, symbol=self.symbol, price=current_price, amount=0.0,
                    reason=f"Uptrend confirmed (Layer1 30m) + bias {bias_score:.0f} (Layer2) + "
                           f"EMA5↑EMA10 & HMA10↑HMA20 crossed within {gb} bar(s) of each other (Layer3)",
                    confidence=1.0,
                    metadata=meta,
                )
            return self._hold(current_price, "Layer1+2 confirmed UP — waiting for EMA5/10 + HMA10/20 cross "
                              f"(within {gb} bars of each other)",
                              metadata=dbg("waiting_cross"))

        # trend == "down"
        if dual_cross_down:
            sl, tp = self._compute_sl_tp("short", close_price, l3["atr_val"])
            self._open_position = "short"
            self._last_ema_cross_down_ts = None
            self._last_hma_cross_down_ts = None
            meta = dbg("entered")
            meta.update({"stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio})
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=current_price, amount=0.0,
                reason=f"Downtrend confirmed (Layer1 30m) + bias {bias_score:.0f} (Layer2) + "
                       f"EMA5↓EMA10 & HMA10↓HMA20 crossed within {gb} bar(s) of each other (Layer3)",
                confidence=1.0,
                metadata=meta,
            )
        return self._hold(current_price, "Layer1+2 confirmed DOWN — waiting for EMA5/10 + HMA10/20 cross "
                          f"(within {gb} bars of each other)",
                          metadata=dbg("waiting_cross"))

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Exit = OR logic: EMA5/10 cross-back OR HMA10/20 cross-back,
        whichever fires first, evaluated once per newly-formed 15m bar.
        Hedge-mode-safe: always closes whichever position is actually
        open, never relies on signal.type semantics."""
        if self._open_position is None or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate

        candles = self._latest_candles
        bar_ts = candles[-1].timestamp
        if bar_ts == self._last_exit_bar_ts:
            return PositionUpdate(action="hold", reason="Waiting for the next 15m bar close")

        l3 = self._layer3_indicators(candles)
        if l3 is None:
            return PositionUpdate(action="hold", reason="Indicators warming up (15m)")
        self._last_exit_bar_ts = bar_ts

        if self._open_position == "long" and (l3["ema_cross_down"] or l3["hma_cross_down"]):
            reason = "EMA5 crossed below EMA10" if l3["ema_cross_down"] else "HMA10 crossed below HMA20"
            self._open_position = None
            return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit LONG: {reason} (15m)")
        if self._open_position == "short" and (l3["ema_cross_up"] or l3["hma_cross_up"]):
            reason = "EMA5 crossed above EMA10" if l3["ema_cross_up"] else "HMA10 crossed above HMA20"
            self._open_position = None
            return PositionUpdate(action="close", close_pct=1.0, reason=f"Exit SHORT: {reason} (15m)")

        return PositionUpdate(action="hold", reason=f"Holding {self._open_position.upper()}")

    def record_closed_trade(self, exit_price: float, exit_reason: str, duration_min: float = 0.0) -> None:
        """Called by bot.py after ANY close, including the risk-manager's
        hard SL/TP fallback firing before EMA5/10 or HMA10/20 crosses back —
        without this, _open_position would stay set forever and analyze()
        would refuse all future entries."""
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

    def _layer1_indicators(self, c30: list) -> Optional[dict]:
        closes = [c.close for c in c30]
        ema1 = self.ema(closes, self.ema1_period)
        ema2 = self.ema(closes, self.ema2_period)
        sma_t = self.sma(closes, self.sma_trend)
        macd_line, macd_signal_line, hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)

        lb = self.ema_slope_lookback
        needed = [ema1[-1], ema2[-1], ema2[-1 - lb], sma_t[-1], macd_line[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        last = c30[-1]
        sma_up, sma_down = last.open > sma_t[-1], last.open < sma_t[-1]
        ema1020_up, ema1020_down = ema1[-1] > ema2[-1], ema1[-1] < ema2[-1]
        slope_up, slope_down = ema2[-1] > ema2[-1 - lb], ema2[-1] < ema2[-1 - lb]
        # "MACD 4C" trend read: raw MACD line vs zero, not vs its signal
        # line — matches the user-supplied Pine script (sign only).
        macd_up, macd_down = macd_line[-1] > 0, macd_line[-1] < 0

        trend_up = sma_up and ema1020_up and slope_up and macd_up
        trend_down = sma_down and ema1020_down and slope_down and macd_down

        return {
            "trend": "up" if trend_up else "down" if trend_down else None,
            "sma_dir": "up" if sma_up else "down" if sma_down else "flat",
            "ema1020_dir": "up" if ema1020_up else "down" if ema1020_down else "flat",
            "slope_dir": "up" if slope_up else "down" if slope_down else "flat",
            "macd_dir": "up" if macd_up else "down" if macd_down else "flat",
        }

    def _layer3_indicators(self, candles: list) -> Optional[dict]:
        closes = [c.close for c in candles]
        ema_f = self.ema(closes, self.ema_fast)
        ema_s = self.ema(closes, self.ema_slow)
        hma_f = self.hma(closes, self.hma_fast)
        hma_s = self.hma(closes, self.hma_slow)
        atr_arr = self.atr(candles, self.atr_period)

        needed = [ema_f[-1], ema_f[-2], ema_s[-1], ema_s[-2],
                 hma_f[-1], hma_f[-2], hma_s[-1], hma_s[-2], atr_arr[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        return {
            "ema_cross_up":   ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1],
            "ema_cross_down": ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1],
            "hma_cross_up":   hma_f[-2] <= hma_s[-2] and hma_f[-1] > hma_s[-1],
            "hma_cross_down": hma_f[-2] >= hma_s[-2] and hma_f[-1] < hma_s[-1],
            "atr_val": float(atr_arr[-1]),
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
