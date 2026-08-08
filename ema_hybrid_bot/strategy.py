"""EMA Hybrid Pro — Triple EMA + Fibonacci + Liquidity + Price Action.

Primary timeframe: M15
Trend confirmation: H1
4H is diagnostic only, not an entry gate.

Entry checklist:
1) EMA20/50/200 aligned on H1 and M15.
2) Pullback to EMA20 or EMA50 (EMA200 is diagnostic/deep pullback only).
3) Price in the 50%-61.8% Fibonacci retracement of the latest confirmed impulse.
4) Closed-M15 price action confirms direction.
5) Liquidity sweep in the entry direction.
6) Volume is supportive when available (soft confirmation, not a hard gate).
7) Initial R:R must be >= 1:2.

Schedule:
- PAPER: entries allowed 24/7 for uninterrupted strategy testing.
- LIVE: new entries allowed Monday-Friday (24/5); existing positions remain managed 24/7.
- LIVE weekday is evaluated in LIVE_SCHEDULE_TIMEZONE (default Asia/Seoul).

Risk:
- SL beyond the confirmed swing that anchors the Fibonacci impulse plus ATR buffer.
- 2R is TP1 milestone: lock +1R.
- 3R is final TP.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import strategy_v12 as base

Side = base.Side
EntrySignal = base.EntrySignal
Trend = base.Trend
SetupType = base.SetupType


@dataclass(frozen=True)
class HybridView:
    side: Optional[Side]
    stage: str
    reason: str
    fib_low: float = 0.0
    fib_high: float = 0.0
    ema_touch: str = "NONE"
    pa: str = "NONE"
    sweep: str = "NONE"
    volume_ratio: float = 0.0
    rr: float = 0.0


class EMAHybridProStrategy(base.PrecisionTrendStructureV12):
    MIN_RR = 2.0
    FINAL_RR = 3.0
    SWING_LOOKBACK = 90
    PIVOT_SPAN = 2
    EMA_TOUCH_ATR = 0.18
    FIB_TOL_ATR = 0.10
    SL_BUFFER_ATR = 0.15
    SWEEP_LOOKBACK = 12
    VOLUME_SUPPORT = 1.00

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.stage_locks_enabled = False
        self.live_schedule_timezone = os.getenv("LIVE_SCHEDULE_TIMEZONE", "Asia/Seoul").strip() or "Asia/Seoul"
        try:
            self._live_tz = ZoneInfo(self.live_schedule_timezone)
        except Exception as exc:
            raise ValueError(
                f"Invalid LIVE_SCHEDULE_TIMEZONE={self.live_schedule_timezone!r}; "
                "use an IANA timezone such as Asia/Seoul or UTC"
            ) from exc

    def _entry_schedule_open(self) -> bool:
        if bool(getattr(self.cfg, "paper", False)):
            return True
        return datetime.now(timezone.utc).astimezone(self._live_tz).weekday() < 5

    @staticmethod
    def _ema(series: pd.Series, n: int) -> pd.Series:
        return series.astype(float).ewm(span=n, adjust=False).mean()

    def _prep(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = frame.copy()
        c = d["close"].astype(float)
        d["ema20"] = self._ema(c, 20)
        d["ema50"] = self._ema(c, 50)
        d["ema200"] = self._ema(c, 200)
        d["atr"] = self._atr(d, self.cfg.atr_len)
        return d

    @staticmethod
    def _ema_side(d: pd.DataFrame) -> Optional[Side]:
        if d.empty or len(d) < 200:
            return None
        r = d.iloc[-1]
        if float(r.ema20) > float(r.ema50) > float(r.ema200):
            return Side.LONG
        if float(r.ema20) < float(r.ema50) < float(r.ema200):
            return Side.SHORT
        return None

    @staticmethod
    def _trend_label(d: pd.DataFrame) -> str:
        if d.empty:
            return "DATA_ERROR"
        if len(d) < 200:
            return f"WARMUP({len(d)}/200)"
        side = EMAHybridProStrategy._ema_side(d)
        if side == Side.LONG:
            return "BULL"
        if side == Side.SHORT:
            return "BEAR"
        return "NEUTRAL"

    def _trend_diagnostics(self, df4h, df1h, df15):
        """Return explicit HTF/MID/entry trend state; never use '?' placeholders."""
        return {
            "4H": self._trend_label(self._prep(df4h)) if len(df4h) else "DATA_ERROR",
            "1H": self._trend_label(self._prep(df1h)) if len(df1h) else "DATA_ERROR",
            "15M": self._trend_label(self._prep(df15)) if len(df15) else "DATA_ERROR",
        }

    def _pivot_pairs(self, d: pd.DataFrame):
        span = self.PIVOT_SPAN
        start = max(span, len(d) - self.SWING_LOOKBACK)
        end = len(d) - span - 1
        highs, lows = [], []
        for i in range(start, end):
            w = d.iloc[i-span:i+span+1]
            if float(d.high.iloc[i]) >= float(w.high.max()):
                highs.append((i, float(d.high.iloc[i])))
            if float(d.low.iloc[i]) <= float(w.low.min()):
                lows.append((i, float(d.low.iloc[i])))
        return highs, lows

    def _impulse(self, d: pd.DataFrame, side: Side):
        highs, lows = self._pivot_pairs(d)
        if side == Side.LONG:
            pairs = [(li, lv, hi, hv) for li, lv in lows for hi, hv in highs if li < hi]
            if not pairs:
                return None
            li, low, hi, high = max(pairs, key=lambda x: x[2])
            if high <= low:
                return None
            fib50 = high - 0.500 * (high - low)
            fib618 = high - 0.618 * (high - low)
        else:
            pairs = [(hi, hv, li, lv) for hi, hv in highs for li, lv in lows if hi < li]
            if not pairs:
                return None
            hi, high, li, low = max(pairs, key=lambda x: x[2])
            if high <= low:
                return None
            fib50 = low + 0.500 * (high - low)
            fib618 = low + 0.618 * (high - low)
        return {"low": low, "high": high, "fib_low": min(fib50, fib618), "fib_high": max(fib50, fib618)}

    def _ema_touch(self, d: pd.DataFrame, side: Side) -> str:
        r = d.iloc[-1]
        atr = max(float(r.atr), 1e-12)
        tolerance = self.EMA_TOUCH_ATR * atr
        if side == Side.LONG:
            if float(r.low) <= float(r.ema20) + tolerance and float(r.close) >= float(r.ema20) - tolerance:
                return "EMA20"
            if float(r.low) <= float(r.ema50) + tolerance and float(r.close) >= float(r.ema50) - tolerance:
                return "EMA50"
        else:
            if float(r.high) >= float(r.ema20) - tolerance and float(r.close) <= float(r.ema20) + tolerance:
                return "EMA20"
            if float(r.high) >= float(r.ema50) - tolerance and float(r.close) <= float(r.ema50) + tolerance:
                return "EMA50"
        return "NONE"

    @staticmethod
    def _price_action(d: pd.DataFrame, side: Side) -> str:
        if len(d) < 4:
            return "NONE"
        r, p, p2 = d.iloc[-1], d.iloc[-2], d.iloc[-3]
        ro, rh, rl, rc = map(float, (r.open, r.high, r.low, r.close))
        po, ph, pl, pc = map(float, (p.open, p.high, p.low, p.close))
        body = abs(rc - ro)
        rng = max(rh - rl, 1e-12)
        upper = rh - max(ro, rc)
        lower = min(ro, rc) - rl
        if side == Side.LONG:
            if rc > ro and pc < po and rc >= po and ro <= pc:
                return "BULL_ENGULF"
            if lower >= max(body * 1.8, rng * 0.45) and rc > rl + 0.55 * rng:
                return "BULL_PIN"
            inside = ph < float(p2.high) and pl > float(p2.low)
            if inside and rc > ph:
                return "INSIDE_BREAK_UP"
            if float(p.low) < float(p2.low) and rc > ph:
                return "BREAK_RETEST_UP"
        else:
            if rc < ro and pc > po and rc <= po and ro >= pc:
                return "BEAR_ENGULF"
            if upper >= max(body * 1.8, rng * 0.45) and rc < rh - 0.55 * rng:
                return "BEAR_PIN"
            inside = ph < float(p2.high) and pl > float(p2.low)
            if inside and rc < pl:
                return "INSIDE_BREAK_DOWN"
            if float(p.high) > float(p2.high) and rc < pl:
                return "BREAK_RETEST_DOWN"
        return "NONE"

    def _liquidity_sweep(self, d: pd.DataFrame, side: Side) -> str:
        if len(d) < self.SWEEP_LOOKBACK + 3:
            return "NONE"
        r = d.iloc[-1]
        prior = d.iloc[-self.SWEEP_LOOKBACK-1:-1]
        if side == Side.LONG:
            prior_low = float(prior.low.min())
            if float(r.low) < prior_low and float(r.close) > prior_low:
                return "SWEEP_LOW"
        else:
            prior_high = float(prior.high.max())
            if float(r.high) > prior_high and float(r.close) < prior_high:
                return "SWEEP_HIGH"
        return "NONE"

    @staticmethod
    def _volume_ratio(d: pd.DataFrame) -> float:
        if "volume" not in d.columns or len(d) < 20:
            return 1.0
        avg = float(d.volume.astype(float).iloc[-20:].mean())
        return float(d.volume.iloc[-1]) / max(avg, 1e-12)

    def _view(self, df1h, df15) -> HybridView:
        if len(df1h) < 220 or len(df15) < 220:
            return HybridView(None, "WARMUP", "need >=220 H1/M15 candles")
        h1, m15 = self._prep(df1h), self._prep(df15)
        side_h1, side_m15 = self._ema_side(h1), self._ema_side(m15)
        if side_h1 is None:
            return HybridView(None, "H1_TREND", "H1 EMA20/50/200 not aligned")
        if side_m15 != side_h1:
            return HybridView(side_h1, "M15_TREND", "M15 triple EMA not aligned with H1")
        side = side_h1
        impulse = self._impulse(m15, side)
        if not impulse:
            return HybridView(side, "FIB", "no confirmed impulse swing")
        r = m15.iloc[-1]
        atr = max(float(r.atr), 1e-12)
        fib_lo = impulse["fib_low"] - self.FIB_TOL_ATR * atr
        fib_hi = impulse["fib_high"] + self.FIB_TOL_ATR * atr
        in_fib = fib_lo <= float(r.close) <= fib_hi or not (float(r.high) < fib_lo or float(r.low) > fib_hi)
        if not in_fib:
            return HybridView(side, "FIB", "waiting 50%-61.8% retracement", impulse["fib_low"], impulse["fib_high"])
        touch = self._ema_touch(m15, side)
        if touch == "NONE":
            return HybridView(side, "EMA_PULLBACK", "Fibo reached; waiting EMA20/50 touch", impulse["fib_low"], impulse["fib_high"])
        sweep = self._liquidity_sweep(m15, side)
        if sweep == "NONE":
            return HybridView(side, "LIQUIDITY", "waiting liquidity sweep", impulse["fib_low"], impulse["fib_high"], touch)
        pa = self._price_action(m15, side)
        if pa == "NONE":
            return HybridView(side, "PRICE_ACTION", "waiting closed-M15 confirmation", impulse["fib_low"], impulse["fib_high"], touch, "NONE", sweep)
        entry = float(r.close)
        if side == Side.LONG:
            sl = min(float(impulse["low"]), float(m15.low.iloc[-8:].min())) - self.SL_BUFFER_ATR * atr
            risk = entry - sl
        else:
            sl = max(float(impulse["high"]), float(m15.high.iloc[-8:].max())) + self.SL_BUFFER_ATR * atr
            risk = sl - entry
        if risk <= 0 or not math.isfinite(risk):
            return HybridView(side, "RISK", "invalid swing stop", impulse["fib_low"], impulse["fib_high"], touch, pa, sweep)
        rr = self.FINAL_RR
        vr = self._volume_ratio(m15)
        return HybridView(side, "READY", "all mandatory gates passed", impulse["fib_low"], impulse["fib_high"], touch, pa, sweep, vr, rr)

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position: bool = False) -> Optional[EntrySignal]:
        if has_open_position:
            return None
        if not self._entry_schedule_open():
            return None
        v = self._view(df1h, df15)
        if v.stage != "READY" or v.side is None:
            return None
        m15 = self._prep(df15)
        impulse = self._impulse(m15, v.side)
        r = m15.iloc[-1]
        entry = float(r.close)
        atr = max(float(r.atr), 1e-12)
        if v.side == Side.LONG:
            sl = min(float(impulse["low"]), float(m15.low.iloc[-8:].min())) - self.SL_BUFFER_ATR * atr
            risk = entry - sl
            tp = entry + self.FINAL_RR * risk
            trend = Trend.BULL
        else:
            sl = max(float(impulse["high"]), float(m15.high.iloc[-8:].max())) + self.SL_BUFFER_ATR * atr
            risk = sl - entry
            tp = entry - self.FINAL_RR * risk
            trend = Trend.BEAR
        if risk <= 0:
            return None
        q = self.quality_state_1h(df1h)
        reason = (
            f"EMA Hybrid Pro {v.side.value} | H1+M15 EMA20/50/200 aligned | "
            f"Fib 50-61.8 {v.fib_low:.6g}-{v.fib_high:.6g} | {v.ema_touch} | "
            f"{v.sweep} | {v.pa} | Vol {v.volume_ratio:.2f}x | "
            f"SL swing+0.15ATR | TP1 2R milestone | TP2 3R final"
        )
        trigger = f"EMA_HYBRID_{v.pa}_{v.sweep}"
        room_pct = abs(tp - entry) / max(entry, 1e-12)
        structure = float(impulse["low"] if v.side == Side.LONG else impulse["high"])
        return EntrySignal(v.side, entry, sl, tp, trend, q.q, q.adx, q.chop, SetupType.PULLBACK, trigger, room_pct, atr, structure, reason)

    def locked_stop(self, side: Side, entry: float, best_price: float):
        r = max(entry * float(self.cfg.stop_loss_pct), entry * 0.003)
        move = (best_price - entry) if side == Side.LONG else (entry - best_price)
        if move >= 2.0 * r:
            locked = entry + r if side == Side.LONG else entry - r
            return locked, 1
        default_stop = entry * (1.0 - self.cfg.stop_loss_pct) if side == Side.LONG else entry * (1.0 + self.cfg.stop_loss_pct)
        return default_stop, 0

    def entry_status(self, df4h, df1h, df15, df5) -> str:
        v = self._view(df1h, df15)
        diag = self._trend_diagnostics(df4h, df1h, df15)
        paper = bool(getattr(self.cfg, "paper", False))
        schedule = "24/7 PAPER" if paper else "24/5 LIVE"
        if paper:
            schedule_state = "OPEN"
        else:
            local_now = datetime.now(timezone.utc).astimezone(self._live_tz)
            schedule_state = "OPEN" if local_now.weekday() < 5 else "WEEKEND_CLOSED"
        aligned = "YES" if diag["1H"] != "NEUTRAL" and diag["1H"] == diag["15M"] else "NO"
        return (
            f"EMA-HYBRID {'READY' if v.stage == 'READY' and schedule_state == 'OPEN' else 'WAIT'} | "
            f"4H={diag['4H']} | 1H={diag['1H']} | 15M={diag['15M']} | MTF={aligned} | "
            f"Side={(v.side.value if v.side else 'NONE')} | Schedule={schedule}({schedule_state}) | "
            f"Stage={v.stage} | Fib={v.fib_low:.6g}-{v.fib_high:.6g} | EMA={v.ema_touch} | "
            f"Sweep={v.sweep} | PA={v.pa} | Vol={v.volume_ratio:.2f}x | Reason={v.reason}"
        )
