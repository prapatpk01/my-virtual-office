"""EMA Hybrid balanced MTF execution with a 15M regime score.

15M = directional bias + soft trend-regime gate:
- Direction foundation: Close vs SMA14 + RSI14 side of 50.
- Regime score (0-6): direction, RSI, SMA14 slope, ADX>=18,
  ADX rising, CHOP<=58.
- Score >= 4 enables 5M execution.
- Score 5-6 = STRONG_TREND, 4 = TREND, 0-3 = CHOP/BLOCK.

5M = actual execution trigger:
- LONG: fresh EMA8/13 cross up + ADX >= 12 + CHOP <= 65.
- SHORT: fresh EMA8/13 cross down + ADX >= 12 + CHOP <= 65.

Risk model:
- SL = recent 5M structure +/- 0.25 ATR.
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
class BiasView:
    side: Optional[Side]
    enabled: bool
    score: int
    regime: str
    rsi: float
    sma14: float
    close: float
    adx: float
    prev_adx: float
    chop: float
    slope: float
    reason: str


@dataclass(frozen=True)
class TriggerView:
    side: Optional[Side]
    ready: bool
    adx: float
    chop: float
    reason: str


class EMAHybridProStrategy(base.PrecisionTrendStructureV12):
    """15M regime score + 5M EMA8/13 cross execution."""

    SL_BUFFER_ATR = float(os.getenv("EMA_ADV_SL_BUFFER_ATR", "0.25"))
    TP2_MIN_RR = float(os.getenv("EMA_ADV_TP2_MIN_RR", "1.30"))
    TP2_FALLBACK_R = float(os.getenv("EMA_ADV_TP2_FALLBACK_R", "2.0"))

    EMA_FAST = int(os.getenv("EMA_5M_FAST", "8"))
    EMA_SLOW = int(os.getenv("EMA_5M_SLOW", "13"))
    RSI_LEN = int(os.getenv("EMA_15M_RSI_LEN", "14"))
    SMA_LEN = int(os.getenv("EMA_15M_SMA_LEN", "14"))
    BIAS_RSI_MID = float(os.getenv("EMA_15M_RSI_MID", "50"))

    REGIME_MIN_SCORE = max(1, min(6, int(os.getenv("EMA_15M_REGIME_MIN_SCORE", "4"))))
    REGIME_ADX_MIN = float(os.getenv("EMA_15M_REGIME_ADX_MIN", "18"))
    REGIME_CHOP_MAX = float(os.getenv("EMA_15M_REGIME_CHOP_MAX", "58"))

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

    def _quality_values(self, d: pd.DataFrame) -> tuple[float, float]:
        try:
            q = self.quality_state_1h(d)
            return float(q.adx), float(q.chop)
        except Exception:
            return 0.0, 100.0

    def _quality_values_5m(self, d: pd.DataFrame) -> tuple[float, float]:
        return self._quality_values(d)

    def _bias15(self, df15: pd.DataFrame) -> BiasView:
        if len(df15) < 80:
            return BiasView(None, False, 0, "WARMUP", 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, "15M WARMUP")

        d = self._prep15(df15)
        r, p = d.iloc[-1], d.iloc[-2]
        if pd.isna(r.sma14) or pd.isna(r.rsi14) or pd.isna(p.sma14):
            return BiasView(None, False, 0, "WARMUP", 0.0, 0.0, float(r.close), 0.0, 0.0, 100.0, 0.0, "15M WARMUP")

        close = float(r.close)
        sma = float(r.sma14)
        prev_sma = float(p.sma14)
        slope = sma - prev_sma
        rsi = float(r.rsi14)
        adx, chop = self._quality_values(d)
        prev_adx, _ = self._quality_values(d.iloc[:-1])

        long_foundation = close > sma and rsi >= self.BIAS_RSI_MID
        short_foundation = close < sma and rsi <= self.BIAS_RSI_MID

        if long_foundation:
            side = Side.LONG
            checks = (
                close > sma,
                rsi >= self.BIAS_RSI_MID,
                slope > 0,
                adx >= self.REGIME_ADX_MIN,
                adx > prev_adx,
                chop <= self.REGIME_CHOP_MAX,
            )
        elif short_foundation:
            side = Side.SHORT
            checks = (
                close < sma,
                rsi <= self.BIAS_RSI_MID,
                slope < 0,
                adx >= self.REGIME_ADX_MIN,
                adx > prev_adx,
                chop <= self.REGIME_CHOP_MAX,
            )
        else:
            return BiasView(
                None, False, 0, "MIXED", rsi, sma, close, adx, prev_adx, chop, slope,
                "15M MIXED: Close/SMA14 and RSI14 do not agree",
            )

        score = sum(1 for ok in checks if ok)
        enabled = score >= self.REGIME_MIN_SCORE
        regime = "STRONG_TREND" if score >= 5 else ("TREND" if enabled else "CHOP")
        direction = "BULL" if side == Side.LONG else "BEAR"
        reason = (
            f"15M {direction} {regime} Score={score}/6 | RSI14={rsi:.1f} | "
            f"SMA14Slope={'UP' if slope > 0 else 'DOWN' if slope < 0 else 'FLAT'} | "
            f"ADX={adx:.1f}({'RISING' if adx > prev_adx else 'FALLING'}) | CHOP={chop:.1f}"
        )
        return BiasView(side, enabled, score, regime, rsi, sma, close, adx, prev_adx, chop, slope, reason)

    def _trigger5(self, df5: pd.DataFrame, bias: BiasView) -> TriggerView:
        if len(df5) < 80 or bias.side is None or not bias.enabled:
            why = f"15M {bias.regime} Score={bias.score}/6" if bias.side is not None else "no valid 15M direction"
            return TriggerView(None, False, 0.0, 100.0, f"5M BLOCK: {why}")

        d = self._prep5(df5)
        r, p = d.iloc[-1], d.iloc[-2]
        adx, chop = self._quality_values_5m(d)
        quality_ok = adx >= self.ADX_MIN and chop <= self.CHOP_MAX

        cross_up = float(p.ema_fast) <= float(p.ema_slow) and float(r.ema_fast) > float(r.ema_slow)
        cross_down = float(p.ema_fast) >= float(p.ema_slow) and float(r.ema_fast) < float(r.ema_slow)

        if bias.side == Side.LONG:
            ready = cross_up and quality_ok
            state = "READY LONG" if ready else ("CROSS_UP_FILTERED" if cross_up else "WAIT CROSS_UP")
            return TriggerView(Side.LONG, ready, adx, chop, f"5M {state}")

        ready = cross_down and quality_ok
        state = "READY SHORT" if ready else ("CROSS_DOWN_FILTERED" if cross_down else "WAIT CROSS_DOWN")
        return TriggerView(Side.SHORT, ready, adx, chop, f"5M {state}")

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

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False):
        if has_open_position or not self._entry_schedule_open():
            return None

        bias = self._bias15(df15)
        trigger = self._trigger5(df5, bias)
        if not trigger.ready or trigger.side is None:
            return None

        d5 = self._prep5(df5)
        entry = float(d5.close.iloc[-1])
        sl, risk = self._structure_stop(d5, trigger.side, entry)
        if risk <= 0:
            return None

        tp2, rr, target_type = self._tp2(d5, trigger.side, entry, risk)
        atr = max(float(d5.atr.iloc[-1]), 1e-12)
        trend = Trend.BULL if trigger.side == Side.LONG else Trend.BEAR
        trigger_name = f"EMA5M_CROSS_{'UP' if trigger.side == Side.LONG else 'DOWN'}_R{bias.score}"
        room_pct = abs(tp2 - entry) / max(entry, 1e-12)
        structure_px = sl + self.SL_BUFFER_ATR * atr if trigger.side == Side.LONG else sl - self.SL_BUFFER_ATR * atr

        reason = (
            f"EMA Hybrid REGIME | {bias.reason} | EMA{self.EMA_FAST}/{self.EMA_SLOW} 5M CROSS | "
            f"ADX5M={trigger.adx:.1f} CHOP5M={trigger.chop:.1f} | "
            f"SL=5M structure+{self.SL_BUFFER_ATR:.2f}ATR | "
            f"TP1=1R trim60% + SL BE+0.15R | TP2={target_type} {rr:.2f}R"
        )

        return EntrySignal(
            trigger.side, entry, sl, tp2, trend,
            bias.rsi, trigger.adx, trigger.chop,
            SetupType.PULLBACK, trigger_name, room_pct, atr, structure_px, reason,
        )

    def entry_status(self, df4h, df1h, df15, df5):
        bias = self._bias15(df15)
        trigger = self._trigger5(df5, bias)
        paper = bool(getattr(self.cfg, "paper", False))
        open_ = self._entry_schedule_open()
        sched = "24/7 PAPER(OPEN)" if paper else f"24/5 LIVE({'OPEN' if open_ else 'WEEKEND_CLOSED'})"
        side = bias.side.value if bias.side is not None else "NEUTRAL"

        return (
            f"REGIME MTF | 15M={side} {bias.regime} Score={bias.score}/6 "
            f"Close={bias.close:.6g} SMA14={bias.sma14:.6g} RSI14={bias.rsi:.1f} "
            f"Slope={bias.slope:.6g} ADX={bias.adx:.1f} PrevADX={bias.prev_adx:.1f} CHOP={bias.chop:.1f} | "
            f"5M={trigger.reason} ADX={trigger.adx:.1f} CHOP={trigger.chop:.1f} | "
            f"Gate: Score>={self.REGIME_MIN_SCORE}/6 (15M ADX>={self.REGIME_ADX_MIN:.0f}, "
            f"CHOP<={self.REGIME_CHOP_MAX:.0f}); Trigger: EMA{self.EMA_FAST}/{self.EMA_SLOW} cross, "
            f"ADX5M>={self.ADX_MIN:.0f}, CHOP5M<={self.CHOP_MAX:.0f} | Schedule={sched}"
        )
