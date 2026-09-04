"""EMA Hybrid A+B Quality V2.1.

15M bias:
- LONG: close > SMA14, RSI14 >= 52, SMA14 slope > 0
- SHORT: close < SMA14, RSI14 <= 48, SMA14 slope < 0

5M setup A:
- fresh EMA8/13 cross
- candle confirms cross direction
- EMA spread is expanding
- ADX >= 15, CHOP <= 60

5M setup B1 RECLAIM:
- EMA8/13 trend intact
- EMA13 slope agrees with direction
- recent candle overlaps the true EMA13 +/- 0.20 ATR pullback zone
- fresh EMA13 reclaim
- ADX >= 15, CHOP <= 60

5M setup B2 MICRO BOS:
- same pullback/trend requirements as B1
- close must clear the prior micro structure by >= 0.10 ATR
- EMA spread must be expanding
- ADX >= 18 and rising
- CHOP <= 55

Risk:
- SL = recent 5M structure +/- 0.25 ATR
- reject if initial SL distance < 0.35% or > 1.00%
- TP1 = +1R; runtime trims 60% and moves remaining SL to BE+0.15R
- TP2 = nearest 5M liquidity/swing target >= 1.3R; fallback 2R
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import strategy_v12 as base

Side = base.Side
EntrySignal = base.EntrySignal
Trend = base.Trend
SetupType = base.SetupType


@dataclass(frozen=True)
class BiasView:
    side: Optional[Side]
    rsi: float
    sma14: float
    close: float
    slope: float
    reason: str


@dataclass(frozen=True)
class TriggerView:
    side: Optional[Side]
    ready: bool
    adx: float
    chop: float
    setup: str
    trigger: str
    reason: str


class EMAHybridProStrategy(base.PrecisionTrendStructureV12):
    """15M quality bias + Setup A cross + Setup B reclaim/strict micro BOS."""

    SL_BUFFER_ATR = float(os.getenv("EMA_ADV_SL_BUFFER_ATR", "0.25"))
    SL_MIN_PCT = float(os.getenv("EMA_5M_SL_MIN_PCT", "0.0035"))
    SL_MAX_PCT = float(os.getenv("EMA_5M_SL_MAX_PCT", "0.0100"))
    TP2_MIN_RR = float(os.getenv("EMA_ADV_TP2_MIN_RR", "1.30"))
    TP2_FALLBACK_R = float(os.getenv("EMA_ADV_TP2_FALLBACK_R", "2.0"))

    EMA_FAST = int(os.getenv("EMA_5M_FAST", "8"))
    EMA_SLOW = int(os.getenv("EMA_5M_SLOW", "13"))

    RSI_LEN = int(os.getenv("EMA_15M_RSI_LEN", "14"))
    SMA_LEN = int(os.getenv("EMA_15M_SMA_LEN", "14"))
    BIAS_RSI_MID = float(os.getenv("EMA_15M_RSI_MID", "50"))
    BIAS_RSI_LONG_MIN = float(os.getenv("EMA_15M_RSI_LONG_MIN", "52"))
    BIAS_RSI_SHORT_MAX = float(os.getenv("EMA_15M_RSI_SHORT_MAX", "48"))

    PULLBACK_LOOKBACK = max(3, int(os.getenv("EMA_5M_PULLBACK_LOOKBACK", "6")))
    PULLBACK_TOUCH_ATR = float(os.getenv("EMA_5M_PULLBACK_TOUCH_ATR", "0.20"))
    BOS_LOOKBACK = max(2, int(os.getenv("EMA_5M_BOS_LOOKBACK", "3")))

    ADX_MIN = float(os.getenv("EMA_5M_ADX_MIN", "15"))
    CHOP_MAX = float(os.getenv("EMA_5M_CHOP_MAX", "60"))
    MICRO_BOS_ADX_MIN = float(os.getenv("EMA_5M_MICRO_BOS_ADX_MIN", "18"))
    MICRO_BOS_CHOP_MAX = float(os.getenv("EMA_5M_MICRO_BOS_CHOP_MAX", "55"))
    MICRO_BOS_BREAK_ATR = float(os.getenv("EMA_5M_MICRO_BOS_BREAK_ATR", "0.10"))

    SWING_LOOKBACK = max(20, int(os.getenv("EMA_5M_SWING_LOOKBACK", "48")))
    STRUCTURE_LOOKBACK = max(6, int(os.getenv("EMA_5M_STRUCTURE_LOOKBACK", "12")))

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.stage_locks_enabled = False
        self.correlation_guard: Optional[Callable[[str, Side], bool]] = None
        tz = os.getenv("LIVE_SCHEDULE_TIMEZONE", "Asia/Seoul").strip() or "Asia/Seoul"
        self._live_tz = ZoneInfo(tz)

    def _entry_schedule_open(self) -> bool:
        if bool(getattr(self.cfg, "paper", False)):
            return True
        return datetime.now(timezone.utc).astimezone(self._live_tz).weekday() < 5

    @staticmethod
    def _ema(series: pd.Series, n: int) -> pd.Series:
        return series.astype(float).ewm(span=n, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, n: int) -> pd.Series:
        s = series.astype(float)
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0.0, 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    def _prep5(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = frame.copy()
        c = d["close"].astype(float)
        d["ema_fast"] = self._ema(c, self.EMA_FAST)
        d["ema_slow"] = self._ema(c, self.EMA_SLOW)
        d["atr"] = self._atr(d, self.cfg.atr_len)
        return d

    def _prep15(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = frame.copy()
        c = d["close"].astype(float)
        d["sma14"] = c.rolling(self.SMA_LEN, min_periods=self.SMA_LEN).mean()
        d["rsi14"] = self._rsi(c, self.RSI_LEN)
        return d

    def _quality_values_5m(self, d: pd.DataFrame) -> tuple[float, float]:
        try:
            q = self.quality_state_1h(d)
            return float(q.adx), float(q.chop)
        except Exception:
            return 0.0, 100.0

    def _bias15(self, df15: pd.DataFrame) -> BiasView:
        if len(df15) < 30:
            return BiasView(None, 0.0, 0.0, 0.0, 0.0, "15M WARMUP")

        d = self._prep15(df15)
        r, p = d.iloc[-1], d.iloc[-2]
        if pd.isna(r.sma14) or pd.isna(r.rsi14) or pd.isna(p.sma14):
            return BiasView(None, 0.0, 0.0, float(r.close), 0.0, "15M WARMUP")

        close = float(r.close)
        sma = float(r.sma14)
        prev_sma = float(p.sma14)
        slope = sma - prev_sma
        rsi = float(r.rsi14)

        if close > sma and rsi >= self.BIAS_RSI_LONG_MIN and slope > 0:
            return BiasView(
                Side.LONG, rsi, sma, close, slope,
                f"15M BULL: Close>SMA{self.SMA_LEN} RSI={rsi:.1f}>={self.BIAS_RSI_LONG_MIN:.0f} SMA slope UP",
            )
        if close < sma and rsi <= self.BIAS_RSI_SHORT_MAX and slope < 0:
            return BiasView(
                Side.SHORT, rsi, sma, close, slope,
                f"15M BEAR: Close<SMA{self.SMA_LEN} RSI={rsi:.1f}<={self.BIAS_RSI_SHORT_MAX:.0f} SMA slope DOWN",
            )
        return BiasView(
            None, rsi, sma, close, slope,
            f"15M NEUTRAL: need RSI>={self.BIAS_RSI_LONG_MIN:.0f}+UP slope or RSI<={self.BIAS_RSI_SHORT_MAX:.0f}+DOWN slope",
        )

    def _trigger5(self, df5: pd.DataFrame, bias_side: Optional[Side]) -> TriggerView:
        if len(df5) < 80 or bias_side is None:
            return TriggerView(None, False, 0.0, 100.0, "NONE", "NONE", "5M WAIT: no valid 15M bias")

        d = self._prep5(df5)
        r, p = d.iloc[-1], d.iloc[-2]
        adx, chop = self._quality_values_5m(d)
        prev_adx, _ = self._quality_values_5m(d.iloc[:-1])
        quality_ok = adx >= self.ADX_MIN and chop <= self.CHOP_MAX
        micro_bos_quality_ok = (
            adx >= self.MICRO_BOS_ADX_MIN
            and adx > prev_adx
            and chop <= self.MICRO_BOS_CHOP_MAX
        )

        fast_now = float(r.ema_fast)
        slow_now = float(r.ema_slow)
        fast_prev = float(p.ema_fast)
        slow_prev = float(p.ema_slow)

        # Setup A — fresh cross + directional candle + expanding EMA spread.
        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now
        bull_candle = float(r.close) > float(r.open)
        bear_candle = float(r.close) < float(r.open)
        long_spread_now = fast_now - slow_now
        long_spread_prev = fast_prev - slow_prev
        short_spread_now = slow_now - fast_now
        short_spread_prev = slow_prev - fast_prev
        long_spread_expanding = long_spread_now > 0 and long_spread_now > long_spread_prev
        short_spread_expanding = short_spread_now > 0 and short_spread_now > short_spread_prev
        setup_a_long = cross_up and bull_candle and long_spread_expanding
        setup_a_short = cross_down and bear_candle and short_spread_expanding

        # Setup B — true EMA13 +/- ATR zone + trend/slope + reclaim or strict micro BOS.
        atr = max(float(r.atr), 1e-12)
        pb_window = d.iloc[-(self.PULLBACK_LOOKBACK + 1):-1]
        prior_high = float(d.high.iloc[-(self.BOS_LOOKBACK + 1):-1].max())
        prior_low = float(d.low.iloc[-(self.BOS_LOOKBACK + 1):-1].min())

        pb_atr = pb_window["atr"].astype(float).clip(lower=1e-12)
        zone_band = self.PULLBACK_TOUCH_ATR * pb_atr
        zone_low = pb_window["ema_slow"].astype(float) - zone_band
        zone_high = pb_window["ema_slow"].astype(float) + zone_band
        overlap = (
            pb_window["low"].astype(float).le(zone_high)
            & pb_window["high"].astype(float).ge(zone_low)
        )
        true_zone_touch = bool(overlap.any())

        ema13_slope = slow_now - slow_prev
        long_reclaim = float(p.close) <= slow_prev and float(r.close) > slow_now
        short_reclaim = float(p.close) >= slow_prev and float(r.close) < slow_now

        raw_long_bos = float(r.close) > prior_high and float(p.close) <= prior_high
        raw_short_bos = float(r.close) < prior_low and float(p.close) >= prior_low
        break_buffer = self.MICRO_BOS_BREAK_ATR * atr
        strict_long_bos = (
            raw_long_bos
            and float(r.close) >= prior_high + break_buffer
            and long_spread_expanding
        )
        strict_short_bos = (
            raw_short_bos
            and float(r.close) <= prior_low - break_buffer
            and short_spread_expanding
        )

        long_trend_intact = fast_now > slow_now and ema13_slope > 0
        short_trend_intact = fast_now < slow_now and ema13_slope < 0

        reclaim_long = long_trend_intact and true_zone_touch and long_reclaim
        reclaim_short = short_trend_intact and true_zone_touch and short_reclaim
        micro_bos_long = long_trend_intact and true_zone_touch and strict_long_bos
        micro_bos_short = short_trend_intact and true_zone_touch and strict_short_bos

        if bias_side == Side.LONG:
            if setup_a_long and quality_ok:
                return TriggerView(
                    Side.LONG, True, adx, chop, "A_EMA_CROSS", "EMA5M_CROSS_UP",
                    "5M READY LONG: A cross + bullish candle + spread expanding",
                )
            if reclaim_long and quality_ok:
                return TriggerView(
                    Side.LONG, True, adx, chop, "B_PULLBACK_RECLAIM", "PULLBACK_RECLAIM_LONG",
                    "5M READY LONG: B1 true-zone RECLAIM + EMA13 slope UP",
                )
            if micro_bos_long and micro_bos_quality_ok:
                return TriggerView(
                    Side.LONG, True, adx, chop, "B_PULLBACK_RECLAIM", "PULLBACK_MICRO_BOS_LONG",
                    f"5M READY LONG: B2 MICRO_BOS break>={self.MICRO_BOS_BREAK_ATR:.2f}ATR + spread expanding + ADX rising",
                )

            if raw_long_bos and true_zone_touch and long_trend_intact:
                return TriggerView(
                    Side.LONG, False, adx, chop, "NONE", "NONE",
                    f"5M MICRO_BOS_FILTERED LONG | ADX={adx:.1f} prev={prev_adx:.1f} CHOP={chop:.1f}",
                )
            raw = "EMA_CROSS" if cross_up else "PULLBACK_RECLAIM" if reclaim_long else "WAIT"
            if cross_up and not setup_a_long:
                raw = "EMA_CROSS_CONFIRM"
            suffix = " FILTERED" if raw != "WAIT" and not quality_ok else ""
            return TriggerView(
                Side.LONG, False, adx, chop, "NONE", "NONE",
                f"5M {raw}{suffix} LONG",
            )

        if setup_a_short and quality_ok:
            return TriggerView(
                Side.SHORT, True, adx, chop, "A_EMA_CROSS", "EMA5M_CROSS_DOWN",
                "5M READY SHORT: A cross + bearish candle + spread expanding",
            )
        if reclaim_short and quality_ok:
            return TriggerView(
                Side.SHORT, True, adx, chop, "B_PULLBACK_RECLAIM", "PULLBACK_RECLAIM_SHORT",
                "5M READY SHORT: B1 true-zone RECLAIM + EMA13 slope DOWN",
            )
        if micro_bos_short and micro_bos_quality_ok:
            return TriggerView(
                Side.SHORT, True, adx, chop, "B_PULLBACK_RECLAIM", "PULLBACK_MICRO_BOS_SHORT",
                f"5M READY SHORT: B2 MICRO_BOS break>={self.MICRO_BOS_BREAK_ATR:.2f}ATR + spread expanding + ADX rising",
            )

        if raw_short_bos and true_zone_touch and short_trend_intact:
            return TriggerView(
                Side.SHORT, False, adx, chop, "NONE", "NONE",
                f"5M MICRO_BOS_FILTERED SHORT | ADX={adx:.1f} prev={prev_adx:.1f} CHOP={chop:.1f}",
            )
        raw = "EMA_CROSS" if cross_down else "PULLBACK_RECLAIM" if reclaim_short else "WAIT"
        if cross_down and not setup_a_short:
            raw = "EMA_CROSS_CONFIRM"
        suffix = " FILTERED" if raw != "WAIT" and not quality_ok else ""
        return TriggerView(
            Side.SHORT, False, adx, chop, "NONE", "NONE",
            f"5M {raw}{suffix} SHORT",
        )

    def _structure_stop(self, d: pd.DataFrame, side: Side, entry: float) -> tuple[float, float]:
        atr = max(float(d.atr.iloc[-1]), 1e-12)
        recent = d.iloc[-self.STRUCTURE_LOOKBACK:]
        if side == Side.LONG:
            structure = float(recent.low.min())
            sl = structure - self.SL_BUFFER_ATR * atr
            risk = entry - sl
        else:
            structure = float(recent.high.max())
            sl = structure + self.SL_BUFFER_ATR * atr
            risk = sl - entry
        return sl, risk

    def _pivot_levels(self, d: pd.DataFrame):
        span = 2
        start = max(span, len(d) - self.SWING_LOOKBACK)
        highs, lows = [], []
        for i in range(start, len(d) - span):
            w = d.iloc[i-span:i+span+1]
            if float(d.high.iloc[i]) >= float(w.high.max()):
                highs.append(float(d.high.iloc[i]))
            if float(d.low.iloc[i]) <= float(w.low.min()):
                lows.append(float(d.low.iloc[i]))
        return highs, lows

    def _tp2(self, d: pd.DataFrame, side: Side, entry: float, risk: float) -> tuple[float, float, str]:
        highs, lows = self._pivot_levels(d)
        if side == Side.LONG:
            for target in sorted(set(x for x in highs if x > entry)):
                rr = (target - entry) / risk
                if rr >= self.TP2_MIN_RR:
                    return target, rr, "5M_SWING_HIGH"
            return entry + self.TP2_FALLBACK_R * risk, self.TP2_FALLBACK_R, "FALLBACK_R"

        for target in sorted(set((x for x in lows if x < entry)), reverse=True):
            rr = (entry - target) / risk
            if rr >= self.TP2_MIN_RR:
                return target, rr, "5M_SWING_LOW"
        return entry - self.TP2_FALLBACK_R * risk, self.TP2_FALLBACK_R, "FALLBACK_R"

    def _sl_gate(self, entry: float, risk: float) -> tuple[bool, float]:
        risk_pct = risk / max(abs(entry), 1e-12)
        return self.SL_MIN_PCT <= risk_pct <= self.SL_MAX_PCT, risk_pct

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False):
        if has_open_position or not self._entry_schedule_open():
            return None

        bias = self._bias15(df15)
        trigger = self._trigger5(df5, bias.side)
        if not trigger.ready or trigger.side is None:
            return None

        symbol = str(getattr(df15, "attrs", {}).get("symbol") or getattr(df5, "attrs", {}).get("symbol") or "")
        if symbol and callable(self.correlation_guard) and self.correlation_guard(symbol, trigger.side):
            return None

        d5 = self._prep5(df5)
        entry = float(d5.close.iloc[-1])
        sl, risk = self._structure_stop(d5, trigger.side, entry)
        if risk <= 0:
            return None

        sl_ok, risk_pct = self._sl_gate(entry, risk)
        if not sl_ok:
            return None

        tp2, rr, target_type = self._tp2(d5, trigger.side, entry, risk)
        atr = max(float(d5.atr.iloc[-1]), 1e-12)
        trend = Trend.BULL if trigger.side == Side.LONG else Trend.BEAR
        room_pct = abs(tp2 - entry) / max(entry, 1e-12)
        structure_px = (
            sl + self.SL_BUFFER_ATR * atr
            if trigger.side == Side.LONG
            else sl - self.SL_BUFFER_ATR * atr
        )

        reason = (
            f"EMA Hybrid A+B QUALITY V2.1 | {bias.reason} | {trigger.setup} | {trigger.trigger} | "
            f"ADX5M={trigger.adx:.1f} CHOP5M={trigger.chop:.1f} | "
            f"SL={risk_pct*100:.2f}% ({self.SL_MIN_PCT*100:.2f}-{self.SL_MAX_PCT*100:.2f}% gate) | "
            f"TP1=1R trim60% + SL BE+0.15R | TP2={target_type} {rr:.2f}R"
        )

        return EntrySignal(
            trigger.side, entry, sl, tp2, trend,
            bias.rsi, trigger.adx, trigger.chop,
            SetupType.PULLBACK, trigger.trigger, room_pct, atr, structure_px, reason,
        )

    def entry_status(self, df4h, df1h, df15, df5):
        bias = self._bias15(df15)
        trigger = self._trigger5(df5, bias.side)
        paper = bool(getattr(self.cfg, "paper", False))
        open_ = self._entry_schedule_open()
        sched = "24/7 PAPER(OPEN)" if paper else f"24/5 LIVE({'OPEN' if open_ else 'WEEKEND_CLOSED'})"
        side = bias.side.value if bias.side is not None else "NEUTRAL"

        sl_note = ""
        if trigger.ready and trigger.side is not None and len(df5) >= 80:
            d5 = self._prep5(df5)
            entry = float(d5.close.iloc[-1])
            _, risk = self._structure_stop(d5, trigger.side, entry)
            sl_ok, risk_pct = self._sl_gate(entry, risk)
            sl_note = f" | SL={'PASS' if sl_ok else 'BLOCK'} {risk_pct*100:.2f}%"

        return (
            f"A+B QUALITY V2.1 | 15M Bias={side} Close={bias.close:.6g} SMA{self.SMA_LEN}={bias.sma14:.6g} "
            f"RSI{self.RSI_LEN}={bias.rsi:.1f} Slope={bias.slope:.6g} | "
            f"5M={trigger.reason} ADX={trigger.adx:.1f} CHOP={trigger.chop:.1f}{sl_note} | "
            f"A=EMA{self.EMA_FAST}/{self.EMA_SLOW} Cross+confirm+spread | "
            f"B1=EMA13 reclaim ADX>={self.ADX_MIN:.0f}/CHOP<={self.CHOP_MAX:.0f} | "
            f"B2=MicroBOS break>={self.MICRO_BOS_BREAK_ATR:.2f}ATR + ADX>={self.MICRO_BOS_ADX_MIN:.0f} rising + CHOP<={self.MICRO_BOS_CHOP_MAX:.0f} + spread | "
            f"SL {self.SL_MIN_PCT*100:.2f}-{self.SL_MAX_PCT*100:.2f}% | Schedule={sched}"
        )
