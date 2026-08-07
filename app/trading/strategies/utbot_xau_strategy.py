"""UT Bot v2 — XAU-only ATR trailing-stop strategy.

Core logic remains the supplied Pine Script:
- source = CLOSE
- ATR = Wilder RMA (Pine ta.atr compatible seed)
- multiplier = 1.0 by default
- ATR period = 10 by default
- BUY  = source crosses ABOVE recursive ATR trailing stop on a CLOSED bar
- SELL = trailing stop crosses ABOVE source on a CLOSED bar
- no fixed SL/TP: open positions exit only on the opposite UT signal

Optional sideway quality gate (ENTRY ONLY):
- UTBOT_SIDEWAY_FILTER=true enables ADX/CHOP filtering.
- UTBOT_ADX_MIN defaults to 18.
- UTBOT_CHOP_MAX defaults to 58.
- ADX/CHOP can only block a NEW UT cross entry. They never create an entry,
  never close an existing position, and never alter the opposite-cross exit.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType


class UTBotXAUStrategy(BaseStrategy):
    """UT Bot v2 for XAU, with an optional entry-only sideway filter."""

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
        start_time_ms: int = 1577836800000,
        end_time_ms: int = 1893456000000,
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

        # Optional entry-only sideway filter. Default OFF so the original UT
        # behaviour is preserved until Railway explicitly enables it.
        self.sideway_filter = os.getenv("UTBOT_SIDEWAY_FILTER", "false").lower() in (
            "1", "true", "yes", "on"
        )
        self.adx_period = max(2, int(os.getenv("UTBOT_ADX_PERIOD", "14")))
        self.chop_period = max(2, int(os.getenv("UTBOT_CHOP_PERIOD", "14")))
        self.adx_min = float(os.getenv("UTBOT_ADX_MIN", "18"))
        self.chop_max = float(os.getenv("UTBOT_CHOP_MAX", "58"))

        self._open_position: Optional[str] = None
        self._entry_bar_ts: Optional[int] = None
        self._last_signal_bar_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None
        self._pending_entry = False
        self._latest_15m: list = []
        self._latest_candles: list = []

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
        cutoff = int(time.time() * 1000) - self.closed_bar_grace_ms
        return [
            candle for candle in candles
            if self._timestamp_ms(candle.timestamp) + tf_ms <= cutoff
        ]

    @staticmethod
    def _true_range(candles: list) -> np.ndarray:
        n = len(candles)
        tr = np.full(n, np.nan, dtype=float)
        if n == 0:
            return tr
        tr[0] = float(candles[0].high) - float(candles[0].low)
        for i in range(1, n):
            high = float(candles[i].high)
            low = float(candles[i].low)
            prev_close = float(candles[i - 1].close)
            tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))
        return tr

    @staticmethod
    def _wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        if len(values) < period:
            return out
        seed = values[:period]
        if not np.all(np.isfinite(seed)):
            return out
        out[period - 1] = float(np.mean(seed))
        alpha = 1.0 / float(period)
        for i in range(period, len(values)):
            if np.isfinite(values[i]) and np.isfinite(out[i - 1]):
                out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _pine_atr(self, candles: list) -> np.ndarray:
        return self._wilder_rma(self._true_range(candles), self.atr_period)

    def _adx(self, candles: list) -> np.ndarray:
        """Wilder ADX compatible with the standard DMI construction."""
        n = len(candles)
        result = np.full(n, np.nan, dtype=float)
        if n < self.adx_period * 2:
            return result

        tr = self._true_range(candles)
        plus_dm = np.zeros(n, dtype=float)
        minus_dm = np.zeros(n, dtype=float)
        for i in range(1, n):
            up = float(candles[i].high) - float(candles[i - 1].high)
            down = float(candles[i - 1].low) - float(candles[i].low)
            plus_dm[i] = up if up > down and up > 0 else 0.0
            minus_dm[i] = down if down > up and down > 0 else 0.0

        atr = self._wilder_rma(tr, self.adx_period)
        plus_sm = self._wilder_rma(plus_dm, self.adx_period)
        minus_sm = self._wilder_rma(minus_dm, self.adx_period)
        dx = np.full(n, np.nan, dtype=float)

        for i in range(n):
            if not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            plus_di = 100.0 * plus_sm[i] / atr[i]
            minus_di = 100.0 * minus_sm[i] / atr[i]
            denom = plus_di + minus_di
            if denom > 0:
                dx[i] = 100.0 * abs(plus_di - minus_di) / denom

        finite_idx = np.flatnonzero(np.isfinite(dx))
        if len(finite_idx) < self.adx_period:
            return result
        first = int(finite_idx[self.adx_period - 1])
        window = dx[finite_idx[:self.adx_period]]
        result[first] = float(np.mean(window))
        alpha = 1.0 / float(self.adx_period)
        for i in range(first + 1, n):
            if np.isfinite(dx[i]) and np.isfinite(result[i - 1]):
                result[i] = alpha * dx[i] + (1.0 - alpha) * result[i - 1]
            elif np.isfinite(result[i - 1]):
                result[i] = result[i - 1]
        return result

    def _choppiness(self, candles: list) -> np.ndarray:
        """CHOP = 100*log10(sum(TR,n)/(HH(n)-LL(n)))/log10(n)."""
        n = len(candles)
        out = np.full(n, np.nan, dtype=float)
        period = self.chop_period
        if n < period:
            return out
        tr = self._true_range(candles)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        denom_log = np.log10(float(period))
        for i in range(period - 1, n):
            start = i - period + 1
            tr_sum = float(np.sum(tr[start:i + 1]))
            price_range = float(np.max(highs[start:i + 1]) - np.min(lows[start:i + 1]))
            if tr_sum > 0 and price_range > 0:
                out[i] = 100.0 * np.log10(tr_sum / price_range) / denom_log
        return out

    def _quality_snapshot(self, candles: list) -> tuple[float, float, bool]:
        if not self.sideway_filter:
            return float("nan"), float("nan"), True
        adx_arr = self._adx(candles)
        chop_arr = self._choppiness(candles)
        adx = float(adx_arr[-1]) if len(adx_arr) and np.isfinite(adx_arr[-1]) else float("nan")
        chop = float(chop_arr[-1]) if len(chop_arr) and np.isfinite(chop_arr[-1]) else float("nan")
        passed = (
            np.isfinite(adx) and np.isfinite(chop)
            and adx >= self.adx_min and chop <= self.chop_max
        )
        return adx, chop, bool(passed)

    def _ut_series(self, candles: list) -> dict:
        n = len(candles)
        source = np.asarray([float(c.close) for c in candles], dtype=float)
        atr = self._pine_atr(candles)
        tsl = np.full(n, np.nan, dtype=float)

        for i in range(n):
            if not np.isfinite(atr[i]):
                continue
            sl_value = self.multiplier * float(atr[i])
            current = float(source[i])
            prev_tsl = tsl[i - 1] if i > 0 else np.nan
            prev_source = source[i - 1] if i > 0 else np.nan
            if np.isfinite(prev_tsl):
                if current > prev_tsl and prev_source > prev_tsl:
                    tsl[i] = max(float(prev_tsl), current - sl_value)
                elif current < prev_tsl and prev_source < prev_tsl:
                    tsl[i] = min(float(prev_tsl), current + sl_value)
                elif current > prev_tsl:
                    tsl[i] = current - sl_value
                else:
                    tsl[i] = current + sl_value
            else:
                tsl[i] = current + sl_value

        buy = sell = False
        if n >= 2 and all(np.isfinite(v) for v in (source[-2], source[-1], tsl[-2], tsl[-1])):
            buy = bool(source[-2] <= tsl[-2] and source[-1] > tsl[-1])
            sell = bool(tsl[-2] <= source[-2] and tsl[-1] > source[-1])
        return {"source": source, "atr": atr, "tsl": tsl, "buy": buy, "sell": sell}

    def _in_date_range(self, bar_timestamp: int) -> bool:
        if not self.use_date_filter:
            return True
        bar_ms = self._timestamp_ms(bar_timestamp)
        return self.start_time_ms <= bar_ms <= self.end_time_ms

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        closed = self._closed_candles(candles)
        self._latest_15m = closed
        self._latest_candles = closed

        minimum = max(self.atr_period + 3, self.adx_period * 2 + 2 if self.sideway_filter else 0)
        if len(closed) < minimum:
            return self._hold(current_price, f"UT Bot warming up ({len(closed)}/{minimum} closed {self.timeframe} bars)")

        values = self._ut_series(closed)
        bar = closed[-1]
        bar_ts = int(bar.timestamp)
        source_now = float(values["source"][-1])
        atr_now = float(values["atr"][-1])
        tsl_now = float(values["tsl"][-1])
        buy = bool(values["buy"])
        sell = bool(values["sell"])
        adx_now, chop_now, quality_ok = self._quality_snapshot(closed)

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
            "sideway_filter_enabled": self.sideway_filter,
            "adx_period": self.adx_period,
            "adx": round(adx_now, 2) if np.isfinite(adx_now) else None,
            "adx_min": self.adx_min,
            "chop_period": self.chop_period,
            "chop": round(chop_now, 2) if np.isfinite(chop_now) else None,
            "chop_max": self.chop_max,
            "quality_gate_pass": quality_ok,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "utbot_no_fixed_sl_tp": True,
            "locked_symbol": self.LOCKED_SYMBOL,
        }

        if not self._in_date_range(bar_ts):
            return self._hold(current_price, "UT Bot outside configured backtest/live date range", metadata)

        # Sideway quality never interferes with management/exits of an existing position.
        if self._open_position is not None:
            return self._hold(
                current_price,
                f"UT Bot holding {self._open_position.upper()} — waiting for opposite confirmed ATR trailing-stop cross",
                metadata,
            )

        direction = "long" if buy else "short" if sell else None
        if direction is None:
            trend = "BULL" if source_now > tsl_now else "BEAR"
            quality_text = ""
            if self.sideway_filter:
                quality_text = f" | ADX={adx_now:.1f}/{self.adx_min:g} CHOP={chop_now:.1f}/{self.chop_max:g} {'PASS' if quality_ok else 'SIDEWAY'}"
            return self._hold(
                current_price,
                f"UT Bot {trend}: close={source_now:.4f} TSL={tsl_now:.4f}; waiting cross{quality_text}",
                metadata,
            )

        if self._last_signal_bar_ts == bar_ts:
            return self._hold(current_price, "UT Bot signal bar already processed — waiting for a new confirmed cross", metadata)

        # Mark every fresh cross as consumed, including filtered crosses. A blocked
        # cross must never be entered later as a stale signal when quality improves.
        self._last_signal_bar_ts = bar_ts

        if self.sideway_filter and not quality_ok:
            side_text = "BUY/LONG" if direction == "long" else "SELL/SHORT"
            metadata.update({"utbot_signal": "BUY" if direction == "long" else "SELL", "direction": direction, "entry_blocked": "SIDEWAY_FILTER"})
            return self._hold(
                current_price,
                f"UT Bot {side_text} cross BLOCKED by sideway filter | ADX={adx_now:.1f} (need >= {self.adx_min:g}) CHOP={chop_now:.1f} (need <= {self.chop_max:g})",
                metadata,
            )

        self._open_position = direction
        self._entry_bar_ts = bar_ts
        self._pending_entry = True
        side_text = "BUY / LONG" if direction == "long" else "SELL / SHORT"
        metadata.update({"utbot_signal": "BUY" if direction == "long" else "SELL", "direction": direction})
        quality_suffix = ""
        if self.sideway_filter:
            quality_suffix = f" | ADX={adx_now:.1f} CHOP={chop_now:.1f} PASS"
        return Signal(
            type=SignalType.BUY if direction == "long" else SignalType.SELL,
            symbol=self.symbol,
            price=float(current_price),
            amount=0.0,
            reason=(
                f"UT Bot v2 {side_text}: confirmed {self.timeframe} close crossed "
                f"{'above' if direction == 'long' else 'below'} ATR trailing stop | "
                f"close={source_now:.4f} TSL={tsl_now:.4f} ATR({self.atr_period})={atr_now:.4f} x{self.multiplier:g}{quality_suffix}"
            ),
            confidence=1.0,
            metadata=metadata,
        )

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        """Exit only on opposite confirmed UT cross; sideway filter is ignored here."""
        if self._open_position is None:
            return None
        from ..engines.position_manager import PositionUpdate
        candles = self._latest_15m or self._latest_candles
        if len(candles) < self.atr_period + 3:
            return PositionUpdate(action="hold", reason="UT Bot position active — waiting for enough confirmed bars")
        values = self._ut_series(candles)
        bar_ts = int(candles[-1].timestamp)
        if self._last_exit_bar_ts == bar_ts:
            return PositionUpdate(action="hold", reason="UT Bot opposite-cross bar already processed")
        opposite = bool(values["sell"]) if self._open_position == "long" else bool(values["buy"])
        if not opposite:
            return PositionUpdate(action="hold", reason=f"UT Bot holding {self._open_position.upper()} — no opposite confirmed ATR trailing-stop cross")
        side = self._open_position
        self._last_exit_bar_ts = bar_ts
        self._open_position = None
        self._entry_bar_ts = None
        self._pending_entry = False
        return PositionUpdate(
            action="close",
            close_pct=1.0,
            reason=f"UT Bot opposite confirmed {self.timeframe} cross — close {side.upper()} and allow reversal",
        )

    def attach_existing_position(self, direction: str, entry_price: float, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> None:
        side = str(direction or "").lower()
        if side in ("long", "short"):
            self._open_position = side
            self._pending_entry = False

    def cancel_pending_entry(self, reason: str = "") -> None:
        if self._pending_entry:
            self._open_position = None
            self._entry_bar_ts = None
            self._pending_entry = False

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float) -> None:
        self._open_position = None
        self._entry_bar_ts = None
        self._pending_entry = False

    def _hold(self, current_price: float, reason: str, metadata: Optional[dict] = None) -> Signal:
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
                "sideway_filter_enabled": self.sideway_filter,
                "locked_symbol": self.LOCKED_SYMBOL,
            },
        )
