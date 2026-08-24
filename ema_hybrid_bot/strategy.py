"""EMA Hybrid 5M Cross-Filter System.

Signal model (closed 5-minute candles only):
- Trigger: fresh EMA8/EMA13 cross.
- Filters: RSI14, SMA14, ADX and Choppiness.
- LONG: EMA8 crosses above EMA13, close > SMA14, RSI in bullish zone,
  ADX >= minimum and CHOP <= maximum.
- SHORT: inverse rules.

Risk model remains compatible with the EMA Hybrid runtime:
- SL = recent 5M structure +/- 0.25 ATR buffer.
- TP1 = +1R; runtime trims 60% and moves remaining SL to BE+0.15R.
- TP2 = nearest 5M liquidity/swing target; fallback 2R.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import strategy_v12 as base

Side = base.Side
EntrySignal = base.EntrySignal
Trend = base.Trend
SetupType = base.SetupType


@dataclass(frozen=True)
class SignalView:
    side: Optional[Side]
    ready: bool
    reason: str
    rsi: float = 50.0
    sma14: float = 0.0
    adx: float = 0.0
    chop: float = 100.0


class EMAHybridProStrategy(base.PrecisionTrendStructureV12):
    """5M EMA8/13 cross trigger with RSI14/SMA14/ADX/CHOP filters."""

    SL_BUFFER_ATR = float(os.getenv("EMA_ADV_SL_BUFFER_ATR", "0.25"))
    TP2_MIN_RR = float(os.getenv("EMA_ADV_TP2_MIN_RR", "1.30"))
    TP2_FALLBACK_R = float(os.getenv("EMA_ADV_TP2_FALLBACK_R", "2.0"))

    EMA_FAST = int(os.getenv("EMA_5M_FAST", "8"))
    EMA_SLOW = int(os.getenv("EMA_5M_SLOW", "13"))
    RSI_LEN = int(os.getenv("EMA_5M_RSI_LEN", "14"))
    SMA_LEN = int(os.getenv("EMA_5M_SMA_LEN", "14"))

    RSI_LONG_MIN = float(os.getenv("EMA_5M_RSI_LONG_MIN", "52"))
    RSI_LONG_MAX = float(os.getenv("EMA_5M_RSI_LONG_MAX", "70"))
    RSI_SHORT_MIN = float(os.getenv("EMA_5M_RSI_SHORT_MIN", "30"))
    RSI_SHORT_MAX = float(os.getenv("EMA_5M_RSI_SHORT_MAX", "48"))
    ADX_MIN = float(os.getenv("EMA_5M_ADX_MIN", "12"))
    CHOP_MAX = float(os.getenv("EMA_5M_CHOP_MAX", "65"))

    SWING_LOOKBACK = max(20, int(os.getenv("EMA_5M_SWING_LOOKBACK", "48")))
    STRUCTURE_LOOKBACK = max(6, int(os.getenv("EMA_5M_STRUCTURE_LOOKBACK", "12")))

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.stage_locks_enabled = False
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

    def _prep(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = frame.copy()
        c = d["close"].astype(float)
        d["ema_fast"] = self._ema(c, self.EMA_FAST)
        d["ema_slow"] = self._ema(c, self.EMA_SLOW)
        d["sma14"] = c.rolling(self.SMA_LEN, min_periods=self.SMA_LEN).mean()
        d["rsi14"] = self._rsi(c, self.RSI_LEN)
        d["atr"] = self._atr(d, self.cfg.atr_len)
        return d

    def _quality_values(self, d: pd.DataFrame) -> tuple[float, float]:
        """Reuse the production ADX/CHOP implementation on the 5M frame."""
        try:
            q = self.quality_state_1h(d)
            return float(q.adx), float(q.chop)
        except Exception:
            return 0.0, 100.0

    def _cross_signal(self, df5: pd.DataFrame) -> tuple[Optional[SignalView], dict]:
        if len(df5) < 80:
            return None, {"state": "WARMUP", "rsi": 0.0, "sma": 0.0, "adx": 0.0, "chop": 0.0}

        d = self._prep(df5)
        r, p = d.iloc[-1], d.iloc[-2]
        if pd.isna(r.sma14) or pd.isna(r.rsi14):
            return None, {"state": "WARMUP", "rsi": 0.0, "sma": 0.0, "adx": 0.0, "chop": 0.0}

        adx, chop = self._quality_values(d)
        rsi = float(r.rsi14)
        sma = float(r.sma14)
        close = float(r.close)

        cross_up = float(p.ema_fast) <= float(p.ema_slow) and float(r.ema_fast) > float(r.ema_slow)
        cross_down = float(p.ema_fast) >= float(p.ema_slow) and float(r.ema_fast) < float(r.ema_slow)

        quality_ok = adx >= self.ADX_MIN and chop <= self.CHOP_MAX
        long_filters = (
            close > sma
            and self.RSI_LONG_MIN <= rsi <= self.RSI_LONG_MAX
            and quality_ok
        )
        short_filters = (
            close < sma
            and self.RSI_SHORT_MIN <= rsi <= self.RSI_SHORT_MAX
            and quality_ok
        )

        meta = {
            "state": "WAIT",
            "rsi": rsi,
            "sma": sma,
            "adx": adx,
            "chop": chop,
            "cross_up": cross_up,
            "cross_down": cross_down,
            "close": close,
        }

        if cross_up and long_filters:
            reason = (
                f"EMA{self.EMA_FAST}/{self.EMA_SLOW} CROSS_UP | "
                f"RSI14={rsi:.1f} | Close>SMA14 | ADX={adx:.1f} | CHOP={chop:.1f}"
            )
            meta["state"] = "LONG_READY"
            return SignalView(Side.LONG, True, reason, rsi, sma, adx, chop), meta

        if cross_down and short_filters:
            reason = (
                f"EMA{self.EMA_FAST}/{self.EMA_SLOW} CROSS_DOWN | "
                f"RSI14={rsi:.1f} | Close<SMA14 | ADX={adx:.1f} | CHOP={chop:.1f}"
            )
            meta["state"] = "SHORT_READY"
            return SignalView(Side.SHORT, True, reason, rsi, sma, adx, chop), meta

        if cross_up:
            meta["state"] = "CROSS_UP_FILTERED"
        elif cross_down:
            meta["state"] = "CROSS_DOWN_FILTERED"
        return None, meta

    def _structure_stop(self, d: pd.DataFrame, side: Side, entry: float) -> tuple[float, float]:
        atr = max(float(d.atr.iloc[-1]), 1e-12)
        recent = d.iloc[-self.STRUCTURE_LOOKBACK:]
        if side == Side.LONG:
            base_sl = float(recent.low.min())
            sl = base_sl - self.SL_BUFFER_ATR * atr
            risk = entry - sl
        else:
            base_sl = float(recent.high.max())
            sl = base_sl + self.SL_BUFFER_ATR * atr
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
                    return target, rr, "SWING_HIGH"
            target = entry + self.TP2_FALLBACK_R * risk
        else:
            for target in sorted(set((x for x in lows if x < entry)), reverse=True):
                rr = (entry - target) / risk
                if rr >= self.TP2_MIN_RR:
                    return target, rr, "SWING_LOW"
            target = entry - self.TP2_FALLBACK_R * risk
        return target, self.TP2_FALLBACK_R, "FALLBACK_R"

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False):
        if has_open_position or not self._entry_schedule_open():
            return None

        sig, meta = self._cross_signal(df5)
        if sig is None or sig.side is None:
            return None

        d = self._prep(df5)
        entry = float(d.close.iloc[-1])
        sl, risk = self._structure_stop(d, sig.side, entry)
        if risk <= 0:
            return None

        tp2, rr, target_type = self._tp2(d, sig.side, entry, risk)
        atr = max(float(d.atr.iloc[-1]), 1e-12)
        trend = Trend.BULL if sig.side == Side.LONG else Trend.BEAR
        trigger = f"EMA5M_CROSS_{'UP' if sig.side == Side.LONG else 'DOWN'}"
        room_pct = abs(tp2 - entry) / max(entry, 1e-12)
        structure_px = sl + self.SL_BUFFER_ATR * atr if sig.side == Side.LONG else sl - self.SL_BUFFER_ATR * atr

        reason = (
            f"EMA Hybrid 5M CROSS | {sig.reason} | "
            f"SL=5M structure+{self.SL_BUFFER_ATR:.2f}ATR | "
            f"TP1=1R trim60% + SL BE+0.15R | TP2={target_type} {rr:.2f}R"
        )

        # q_1h field is retained for runtime compatibility; RSI is surfaced there
        # until the legacy alert formatter is fully renamed.
        return EntrySignal(
            sig.side, entry, sl, tp2, trend,
            sig.rsi, sig.adx, sig.chop,
            SetupType.PULLBACK, trigger, room_pct, atr, structure_px, reason,
        )

    def entry_status(self, df4h, df1h, df15, df5):
        sig, meta = self._cross_signal(df5)
        paper = bool(getattr(self.cfg, "paper", False))
        open_ = self._entry_schedule_open()
        sched = "24/7 PAPER(OPEN)" if paper else f"24/5 LIVE({'OPEN' if open_ else 'WEEKEND_CLOSED'})"

        if sig is not None:
            return (
                f"EMA-CROSS READY | {sig.side.value} | RSI14={sig.rsi:.1f} | SMA14={sig.sma14:.6g} | "
                f"ADX={sig.adx:.1f} CHOP={sig.chop:.1f} | Schedule={sched} | "
                f"SLBuf={self.SL_BUFFER_ATR:.2f}ATR"
            )

        return (
            f"EMA-CROSS WAIT | State={meta.get('state','?')} | RSI14={meta.get('rsi',0):.1f} | "
            f"SMA14={meta.get('sma',0):.6g} | ADX={meta.get('adx',0):.1f} "
            f"CHOP={meta.get('chop',0):.1f} | "
            f"Rules: LONG RSI {self.RSI_LONG_MIN:.0f}-{self.RSI_LONG_MAX:.0f}, "
            f"SHORT RSI {self.RSI_SHORT_MIN:.0f}-{self.RSI_SHORT_MAX:.0f}, "
            f"ADX>={self.ADX_MIN:.0f}, CHOP<={self.CHOP_MAX:.0f} | Schedule={sched}"
        )
