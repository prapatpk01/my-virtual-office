"""
TF30M HMA-MACD-ROC Momentum Strategy.

Instant AND-gated entry, OR-based exit:

  Entry = AND logic, single bar, no confirmation window: the moment a
          30m bar closes with HMA10/HMA20 crossing AND MACD line
          already on the same side of its signal line AND ROC(9)
          agreeing in sign — enter immediately. (An earlier version
          used a 3-bar confirmation window requiring a fresh MACD
          cross; that added ~1-2 bars of entry lag waiting for MACD/ROC
          to catch up to the HMA cross, so it was dropped in favor of
          this instant check — MACD/ROC just need to already agree,
          not freshly cross, on the same bar as the HMA gate.)
  Exit  = OR logic: any one of HMA reverse / MACD reverse / ROC sign
          flip closes the position 100% immediately, no partial, no
          runner, no same-bar reversal into the opposite side.

  SL = ATR(14, 30m) x 1.5, TP = same distance (1:1 R:R).
  Position margin = 5% of balance x leverage (opt-in via
  signal.metadata["sizing_mode"]="margin", read by bot.py).
  Only one position per symbol; a new setup can't start while one is open.

Entries are returned as a Signal from analyze(). Exits are NOT — a
SELL/BUY Signal in hedge mode always OPENS the opposite side rather
than closing the current one, so exits go through tick_open_position()
instead, which always closes whichever position is actually open.

TF30m candles aren't fetched separately from the exchange — they're
resampled from the 15m series this strategy already receives, and the
last (possibly still-forming) 30m bucket is dropped so entries/exits
only ever evaluate against a genuinely closed 30m bar.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


class HMAMacdROCStrategy(BaseStrategy):

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        hma_fast: int = 10,
        hma_slow: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        roc_period: int = 9,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        rr_ratio: float = 1.0,
        margin_pct: float = 0.05,
    ):
        super().__init__(symbol, params)
        self.name = f"HMAMacdROC({symbol})"
        self.hma_fast = hma_fast
        self.hma_slow = hma_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.roc_period = roc_period
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio
        self.margin_pct = margin_pct

        self._open_position: Optional[str] = None   # "long" | "short" | None
        self._last_bar_ts: Optional[int] = None       # last 30m bar this strategy has acted on
        self._latest_candles: list = []

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        self._latest_candles = candles  # cached for tick_open_position()

        c30 = self._closed_30m_bars(candles)
        min_needed = max(self.hma_slow, self.macd_slow + self.macd_signal,
                         self.roc_period, self.atr_period) + 5
        if len(c30) < min_needed:
            return self._hold(current_price, f"Need {min_needed}+ closed 30m bars, have {len(c30)}")

        ind = self._indicators(c30)
        if ind is None:
            return self._hold(current_price, "Indicators still warming up (30m)")

        bar_ts = c30[-1].timestamp
        is_new_bar = bar_ts != self._last_bar_ts

        # Position management is entirely delegated to tick_open_position();
        # analyze() just refuses new entries while one is open.
        if self._open_position is not None:
            return self._hold(current_price, f"Holding {self._open_position.upper()} — managed via tick_open_position()")

        if not is_new_bar:
            return self._hold(current_price, "Waiting for the next 30m bar close")
        self._last_bar_ts = bar_ts

        hma_cross_up, hma_cross_down = ind["hma_cross_up"], ind["hma_cross_down"]
        macd_bullish, macd_bearish = ind["macd_bullish"], ind["macd_bearish"]
        roc_val = ind["roc_val"]
        atr_val = ind["atr_val"]
        close_price = c30[-1].close

        # ── Instant AND gate: HMA cross + MACD already agreeing + ROC sign ────
        if hma_cross_up and macd_bullish and roc_val > 0:
            return self._open_trade("long", close_price, atr_val,
                                    "Long: HMA10 crossed above HMA20, MACD bullish, ROC9>0 (30m)", current_price)
        if hma_cross_down and macd_bearish and roc_val < 0:
            return self._open_trade("short", close_price, atr_val,
                                    "Short: HMA10 crossed below HMA20, MACD bearish, ROC9<0 (30m)", current_price)

        if hma_cross_up or hma_cross_down:
            gate = "Long" if hma_cross_up else "Short"
            return self._hold(current_price, f"{gate} Gate fired but MACD/ROC didn't agree on the same bar — no entry")

        return self._hold(current_price, "Waiting for HMA10/HMA20 cross (30m)")

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Exit check — OR logic across HMA/MACD/ROC reversal, evaluated only
        when a new 30m bar closes. Hedge-mode-safe: always closes whichever
        position is actually open, never relies on signal.type semantics."""
        if self._open_position is None or not self._latest_candles:
            return None

        from ..engines.position_manager import PositionUpdate

        c30 = self._closed_30m_bars(self._latest_candles)
        min_needed = max(self.hma_slow, self.macd_slow + self.macd_signal, self.roc_period) + 5
        if len(c30) < min_needed:
            return PositionUpdate(action="hold", reason="Indicators warming up (30m)")

        bar_ts = c30[-1].timestamp
        if bar_ts == self._last_bar_ts:
            return PositionUpdate(action="hold", reason="Waiting for the next 30m bar close")

        ind = self._indicators(c30)
        if ind is None:
            return PositionUpdate(action="hold", reason="Indicators warming up (30m)")
        self._last_bar_ts = bar_ts

        roc_val = ind["roc_val"]

        if self._open_position == "long":
            if ind["hma_cross_down"] or ind["macd_cross_down"] or roc_val < 0:
                reasons = []
                if ind["hma_cross_down"]: reasons.append("HMA10 crossed below HMA20")
                if ind["macd_cross_down"]: reasons.append("MACD crossed below signal")
                if roc_val < 0: reasons.append(f"ROC9={roc_val:.2f}<0")
                self._reset("Long position closed: " + " | ".join(reasons))
                return PositionUpdate(action="close", close_pct=1.0, reason="Exit LONG: " + " | ".join(reasons))
            return PositionUpdate(action="hold", reason="Holding LONG (30m)")

        if self._open_position == "short":
            if ind["hma_cross_up"] or ind["macd_cross_up"] or roc_val > 0:
                reasons = []
                if ind["hma_cross_up"]: reasons.append("HMA10 crossed above HMA20")
                if ind["macd_cross_up"]: reasons.append("MACD crossed above signal")
                if roc_val > 0: reasons.append(f"ROC9={roc_val:.2f}>0")
                self._reset("Short position closed: " + " | ".join(reasons))
                return PositionUpdate(action="close", close_pct=1.0, reason="Exit SHORT: " + " | ".join(reasons))
            return PositionUpdate(action="hold", reason="Holding SHORT (30m)")

        return PositionUpdate(action="hold")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _reset(self, reason: str) -> None:
        self._open_position = None

    def _open_trade(self, direction: str, entry_price: float, atr_val: float, reason: str, current_price: float) -> Signal:
        sl_distance = atr_val * self.sl_atr_mult
        if direction == "long":
            sl = entry_price - sl_distance
            tp = entry_price + sl_distance * self.rr_ratio
            sig_type = SignalType.BUY
        else:
            sl = entry_price + sl_distance
            tp = entry_price - sl_distance * self.rr_ratio
            sig_type = SignalType.SELL

        self._open_position = direction

        return Signal(
            type=sig_type, symbol=self.symbol, price=current_price, amount=0.0,
            reason=reason, confidence=1.0,
            metadata={
                "stop_loss": round(sl, 8), "take_profit": round(tp, 8), "rr_ratio": self.rr_ratio,
                "sizing_mode": "margin", "margin_pct": self.margin_pct,
            },
        )

    # ── Indicators (all computed on the closed 30m series) ──────────────────

    def _indicators(self, c30: list) -> Optional[dict]:
        closes = [c.close for c in c30]
        hma10 = self.hma(closes, self.hma_fast)
        hma20 = self.hma(closes, self.hma_slow)
        macd_line, macd_signal_line, _hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        roc_arr = self._roc(closes, self.roc_period)
        atr_arr = self.atr(c30, self.atr_period)

        needed = [hma10[-1], hma10[-2], hma20[-1], hma20[-2],
                 macd_line[-1], macd_line[-2], macd_signal_line[-1], macd_signal_line[-2],
                 atr_arr[-1]]
        if any(np.isnan(x) for x in needed):
            return None

        return {
            "hma_cross_up": hma10[-2] <= hma20[-2] and hma10[-1] > hma20[-1],
            "hma_cross_down": hma10[-2] >= hma20[-2] and hma10[-1] < hma20[-1],
            "hma10_last": float(hma10[-1]), "hma20_last": float(hma20[-1]),
            # Cross events (transition) — used for the OR-based exit, per spec
            "macd_cross_up": macd_line[-2] <= macd_signal_line[-2] and macd_line[-1] > macd_signal_line[-1],
            "macd_cross_down": macd_line[-2] >= macd_signal_line[-2] and macd_line[-1] < macd_signal_line[-1],
            # Current state (not necessarily a fresh cross) — used for the
            # instant entry gate, so MACD just needs to already agree with
            # the HMA cross's direction on the same bar.
            "macd_bullish": macd_line[-1] > macd_signal_line[-1],
            "macd_bearish": macd_line[-1] < macd_signal_line[-1],
            "roc_val": float(roc_arr[-1]),
            "atr_val": float(atr_arr[-1]),
        }

    @staticmethod
    def _roc(closes: list, period: int) -> np.ndarray:
        arr = np.array(closes, dtype=float)
        out = np.full(len(arr), np.nan)
        for i in range(period, len(arr)):
            prev = arr[i - period]
            if prev != 0:
                out[i] = (arr[i] - prev) / prev * 100.0
        return out

    # ── 30m resample, closed bars only ──────────────────────────────────────

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

        # Drop the last bar if its 30m bucket hasn't fully elapsed yet —
        # entries/exits must only ever see a genuinely closed 30m candle.
        last_c15_end = candles_15m[-1].timestamp + 15 * 60_000
        last_bucket_end = keys[-1] + bucket_ms
        if last_c15_end < last_bucket_end:
            out = out[:-1]

        return out

    def _hold(self, price: float, reason: str = "") -> Signal:
        return Signal(
            type=SignalType.HOLD, symbol=self.symbol, price=price, amount=0.0,
            reason=reason, confidence=0.0, metadata={},
        )
