"""UT Bot v2 — XAU-only ATR trailing-stop strategy.

Faithful live port of the supplied Pine Script:
- source = CLOSE
- ATR = Wilder RMA (Pine ta.atr compatible seed)
- multiplier = 1.0 by default
- ATR period = 10 by default
- BUY  = source crosses ABOVE the recursive ATR trailing stop on a CLOSED bar
- SELL = trailing stop crosses ABOVE source on a CLOSED bar
- no extra trend/quality/EMA/WT/structure filters
- no fixed SL or TP: an open position is held until the opposite UT signal

The strategy is intended to be instantiated only for XAU/USDT:USDT by the
production runner.  It maintains an optimistic local position flag so the
existing TradingBot can pair opposite-cross exits with same-bar reversals.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


class UTBotXAUStrategy(BaseStrategy):
    """Exact-direction UT Bot v2 implementation for one XAU market."""

    LOCKED_SYMBOL = "XAU/USDT:USDT"
    ENTRY_OWNER = "UTBOT"

    def __init__(
        self,
        symbol: str = LOCKED_SYMBOL,
        params: Optional[dict] = None,
        multiplier: float = 1.0,
        atr_period: int = 10,
        timeframe: str = "15m",
        use_date_filter: bool = True,
        start_time_ms: int = 1577836800000,   # 2020-01-01 00:00 UTC
        end_time_ms: int = 1893456000000,     # 2030-01-01 00:00 UTC
        closed_bar_grace_ms: int = 1500,
    ):
        if symbol != self.LOCKED_SYMBOL:
            raise ValueError(
                f"UTBotXAUStrategy is locked to {self.LOCKED_SYMBOL}; got {symbol!r}"
            )
        super().__init__(symbol=symbol, params=params)
        self.name = f"UTBotXAU({symbol})"

        self.multiplier = max(0.1, float(multiplier))
        self.atr_period = max(1, int(atr_period))
        self.entry_tf = str(timeframe or "15m").lower()
        self.timeframe = self.entry_tf
        self.closed_bar_grace_ms = max(0, int(closed_bar_grace_ms))
        self.use_date_filter = bool(use_date_filter)
        self.start_time_ms = int(start_time_ms)
        self.end_time_ms = int(end_time_ms)

        self._open_position: Optional[str] = None
        self._entry_bar_ts: Optional[int] = None
        self._last_signal_bar_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None
        self._pending_entry = False

        # TradingBot refreshes this before position management for 15M entries.
        self._latest_15m: list = []
        self._latest_candles: list = []

    # ------------------------------------------------------------------
    # Pine-compatible series helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp_ms(timestamp: int) -> int:
        value = int(timestamp)
        return value * 1000 if value < 10_000_000_000 else value

    @staticmethod
    def _timeframe_ms(timeframe: str) -> int:
        tf = str(timeframe or "15m").strip().lower()
        if tf.endswith("m"):
            return max(1, int(tf[:-1])) * 60_000
        if tf.endswith("h"):
            return max(1, int(tf[:-1])) * 60 * 60_000
        if tf.endswith("d"):
            return max(1, int(tf[:-1])) * 24 * 60 * 60_000
        raise ValueError(f"Unsupported UTBOT timeframe: {timeframe!r}")

    def _closed_candles(self, candles: list) -> list:
        if not candles:
            return []
        tf_ms = self._timeframe_ms(self.timeframe)
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - self.closed_bar_grace_ms
        return [
            candle
            for candle in candles
            if self._timestamp_ms(candle.timestamp) + tf_ms <= cutoff
        ]

    def _pine_atr(self, candles: list) -> np.ndarray:
        """Pine ta.atr(): TR including bar-0 high-low, then Wilder RMA."""
        n = len(candles)
        tr = np.full(n, np.nan, dtype=float)
        if n == 0:
            return tr

        tr[0] = float(candles[0].high) - float(candles[0].low)
        for index in range(1, n):
            high = float(candles[index].high)
            low = float(candles[index].low)
            previous_close = float(candles[index - 1].close)
            tr[index] = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        atr = np.full(n, np.nan, dtype=float)
        period = self.atr_period
        if n < period:
            return atr

        # Pine RMA seeds with SMA of the first `period` non-na TR values.
        atr[period - 1] = float(np.mean(tr[:period]))
        alpha = 1.0 / period
        for index in range(period, n):
            atr[index] = (
                alpha * tr[index]
                + (1.0 - alpha) * atr[index - 1]
            )
        return atr

    def _ut_series(self, candles: list) -> dict:
        n = len(candles)
        source = np.asarray([float(c.close) for c in candles], dtype=float)
        atr = self._pine_atr(candles)
        tsl = np.full(n, np.nan, dtype=float)

        for index in range(n):
            if not np.isfinite(atr[index]):
                continue

            sl_value = self.multiplier * float(atr[index])
            current_source = float(source[index])
            previous_tsl = tsl[index - 1] if index > 0 else np.nan
            previous_source = source[index - 1] if index > 0 else np.nan

            # Exact branch order from the supplied Pine recurrence.
            if np.isfinite(previous_tsl):
                if (
                    current_source > previous_tsl
                    and previous_source > previous_tsl
                ):
                    tsl[index] = max(
                        float(previous_tsl),
                        current_source - sl_value,
                    )
                elif (
                    current_source < previous_tsl
                    and previous_source < previous_tsl
                ):
                    tsl[index] = min(
                        float(previous_tsl),
                        current_source + sl_value,
                    )
                elif current_source > previous_tsl:
                    tsl[index] = current_source - sl_value
                else:
                    tsl[index] = current_source + sl_value
            else:
                # When tsl_price[1] is na in Pine, all comparisons are false,
                # so the final `source + sl_value` branch is selected.
                tsl[index] = current_source + sl_value

        buy = False
        sell = False
        if n >= 2 and all(
            np.isfinite(value)
            for value in (
                source[-2], source[-1], tsl[-2], tsl[-1]
            )
        ):
            buy = bool(source[-2] <= tsl[-2] and source[-1] > tsl[-1])
            sell = bool(tsl[-2] <= source[-2] and tsl[-1] > source[-1])

        return {
            "source": source,
            "atr": atr,
            "tsl": tsl,
            "buy": buy,
            "sell": sell,
        }

    def _in_date_range(self, bar_timestamp: int) -> bool:
        if not self.use_date_filter:
            return True
        bar_ms = self._timestamp_ms(bar_timestamp)
        return self.start_time_ms <= bar_ms <= self.end_time_ms

    # ------------------------------------------------------------------
    # Live signal + reversal lifecycle
    # ------------------------------------------------------------------

    async def analyze(
        self,
        candles: list,
        current_price: float,
        mtf_candles: dict = None,
    ) -> Signal:
        closed = self._closed_candles(candles)
        self._latest_15m = closed
        self._latest_candles = closed

        minimum = self.atr_period + 3
        if len(closed) < minimum:
            return self._hold(
                current_price,
                f"UT Bot warming up ({len(closed)}/{minimum} closed {self.timeframe} bars)",
            )

        values = self._ut_series(closed)
        bar = closed[-1]
        bar_ts = int(bar.timestamp)
        source_now = float(values["source"][-1])
        atr_now = float(values["atr"][-1])
        tsl_now = float(values["tsl"][-1])
        buy = bool(values["buy"])
        sell = bool(values["sell"])

        metadata = {
            "strategy": "UTBOT_V2_ATR_TRAILING_STOP",
            "selected_strategy": "UT Bot v2",
            "entry_trigger_owner": self.ENTRY_OWNER,
            "entry_trigger": "UTBOT_ATR_CROSS",
            "signal_exit_rule": "OPPOSITE_UTBOT_ATR_CROSS",
            "entry_tf": self.timeframe,
            "source": "close",
            "atr_period": self.atr_period,
            "atr_multiplier": self.multiplier,
            "atr": round(atr_now, 8),
            "tsl_price": round(tsl_now, 8),
            "bar_close": round(source_now, 8),
            "buy_cross": buy,
            "sell_cross": sell,
            # The Pine script has no fixed stop-loss or take-profit. Zero is
            # intentional: it prevents the shared RiskManager from inventing
            # fallback SL/TP levels and prevents exchange TPSL algo creation.
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "utbot_no_fixed_sl_tp": True,
            "locked_symbol": self.LOCKED_SYMBOL,
        }

        if not self._in_date_range(bar_ts):
            return self._hold(
                current_price,
                "UT Bot outside configured backtest/live date range",
                metadata=metadata,
            )

        # While a live position exists, reversal is owned by tick_open_position
        # so it closes first; analyze() will then open the opposite side later
        # in the SAME bot cycle from this same confirmed bar.
        if self._open_position is not None:
            return self._hold(
                current_price,
                (
                    f"UT Bot holding {self._open_position.upper()} — "
                    "waiting for opposite confirmed ATR trailing-stop cross"
                ),
                metadata=metadata,
            )

        direction = "long" if buy else "short" if sell else None
        if direction is None:
            trend = "BULL" if source_now > tsl_now else "BEAR"
            return self._hold(
                current_price,
                f"UT Bot {trend}: close={source_now:.4f} TSL={tsl_now:.4f}; waiting cross",
                metadata=metadata,
            )

        if self._last_signal_bar_ts == bar_ts:
            return self._hold(
                current_price,
                "UT Bot signal bar already processed — waiting for a new confirmed cross",
                metadata=metadata,
            )

        self._last_signal_bar_ts = bar_ts
        self._open_position = direction
        self._entry_bar_ts = bar_ts
        self._pending_entry = True

        side_text = "BUY / LONG" if direction == "long" else "SELL / SHORT"
        metadata.update({
            "utbot_signal": "BUY" if direction == "long" else "SELL",
            "direction": direction,
        })
        return Signal(
            type=SignalType.BUY if direction == "long" else SignalType.SELL,
            symbol=self.symbol,
            price=float(current_price),
            amount=0.0,
            reason=(
                f"UT Bot v2 {side_text}: confirmed {self.timeframe} close "
                f"crossed {'above' if direction == 'long' else 'below'} "
                f"ATR trailing stop | close={source_now:.4f} TSL={tsl_now:.4f} "
                f"ATR({self.atr_period})={atr_now:.4f} x{self.multiplier:g}"
            ),
            confidence=1.0,
            metadata=metadata,
        )

    def tick_open_position(
        self,
        current_price: float,
        position_key: Optional[str] = None,
    ):
        """Close on the opposite confirmed UT cross; analyze() then reverses."""
        if self._open_position is None:
            return None

        from ..engines.position_manager import PositionUpdate

        candles = self._latest_15m or self._latest_candles
        if len(candles) < self.atr_period + 3:
            return PositionUpdate(
                action="hold",
                reason="UT Bot position active — waiting for enough confirmed bars",
            )

        values = self._ut_series(candles)
        bar_ts = int(candles[-1].timestamp)
        if self._last_exit_bar_ts == bar_ts:
            return PositionUpdate(
                action="hold",
                reason="UT Bot opposite-cross bar already processed",
            )

        opposite = (
            bool(values["sell"])
            if self._open_position == "long"
            else bool(values["buy"])
        )
        if not opposite:
            return PositionUpdate(
                action="hold",
                reason=(
                    f"UT Bot holding {self._open_position.upper()} — "
                    "no opposite confirmed ATR trailing-stop cross"
                ),
            )

        side = self._open_position
        self._last_exit_bar_ts = bar_ts
        self._open_position = None
        self._entry_bar_ts = None
        self._pending_entry = False
        return PositionUpdate(
            action="close",
            close_pct=1.0,
            reason=(
                f"UT Bot opposite confirmed {self.timeframe} cross — "
                f"close {side.upper()} and allow reversal"
            ),
        )

    def attach_existing_position(
        self,
        direction: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Recover XAU side after a Railway process restart."""
        side = str(direction or "").lower()
        if side in ("long", "short"):
            self._open_position = side
            self._pending_entry = False

    def cancel_pending_entry(self, reason: str = "") -> None:
        """Rollback optimistic state when the execution/risk gate rejects it."""
        if self._pending_entry:
            self._open_position = None
            self._entry_bar_ts = None
            self._pending_entry = False

    def record_closed_trade(
        self,
        exit_price: float,
        reason: str,
        duration_min: float,
    ) -> None:
        # If the exchange closes externally, make sure the next fresh UT cross
        # is allowed to open a new position.
        self._open_position = None
        self._entry_bar_ts = None
        self._pending_entry = False

    def _hold(
        self,
        current_price: float,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> Signal:
        return Signal(
            type=SignalType.HOLD,
            symbol=self.symbol,
            price=float(current_price),
            amount=0.0,
            reason=reason,
            confidence=0.0,
            metadata=metadata or {
                "strategy": "UTBOT_V2_ATR_TRAILING_STOP",
                "selected_strategy": "UT Bot v2",
                "entry_trigger_owner": self.ENTRY_OWNER,
                "entry_tf": self.timeframe,
                "locked_symbol": self.LOCKED_SYMBOL,
            },
        )
