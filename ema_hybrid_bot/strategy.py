"""EMA Hybrid Advanced V3 — higher-frequency, quality-controlled execution.

Goal: produce materially more valid paper/live setups without returning to the
old failure mode where a liquidity sweep alone could trigger a trade.

Pipeline
    H1 directional bias -> M15 trend/pullback location -> M5 execution -> 2TP

V3 changes
- H1 direction uses EMA20/50 momentum + price vs EMA50; EMA200 becomes a score
  bonus instead of a hard gate.
- Quality gate remains hard, but balanced: Q>=40, ADX>=10, CHOP<=68.
- M15 EMA20/50 agreement remains mandatory.
- Location accepts either Fib value or a genuine EMA20/50 pullback; Fib is no
  longer mandatory for every trade.
- Recent M5 trigger window expands to 6 closed bars (30 minutes).
- Three entry paths: LIQUIDITY, CONFIRM, TREND_PULLBACK.
- TP2 minimum room defaults to 1.30R instead of 1.50R while TP1 remains 1R.
- SL remains structure + 0.25 ATR; runtime still trims 60% at TP1 and moves the
  remaining stop to BE+0.15R.
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
class HybridView:
    side: Optional[Side]
    stage: str
    reason: str
    fib_low: float = 0.0
    fib_high: float = 0.0
    ema_touch: str = "NONE"
    pa: str = "NONE"
    sweep: str = "NONE"
    structure: str = "NONE"
    zone: str = "NONE"
    m5_trigger: str = "NONE"
    volume_ratio: float = 0.0
    rr: float = 0.0
    score: int = 0
    grade: str = "BUILDING"
    location: str = "NONE"
    entry_path: str = "NONE"


class EMAHybridProStrategy(base.PrecisionTrendStructureV12):
    MIN_SCORE = int(os.getenv("EMA_ADV_MIN_SCORE", "7"))
    STRONG_CONFIRM_SCORE = int(os.getenv("EMA_ADV_STRONG_CONFIRM_SCORE", "7"))
    TP2_MIN_RR = float(os.getenv("EMA_ADV_TP2_MIN_RR", "1.30"))
    QUALITY_MIN = float(os.getenv("EMA_ADV_QUALITY_MIN", "40"))
    HARD_ADX_MIN = float(os.getenv("EMA_ADV_HARD_ADX_MIN", "10"))
    HARD_CHOP_MAX = float(os.getenv("EMA_ADV_HARD_CHOP_MAX", "68"))
    TREND_PATH_Q_MIN = float(os.getenv("EMA_ADV_TREND_PATH_Q_MIN", "60"))
    TREND_PATH_ADX_MIN = float(os.getenv("EMA_ADV_TREND_PATH_ADX_MIN", "15"))
    M5_TRIGGER_LOOKBACK = max(1, int(os.getenv("EMA_ADV_M5_TRIGGER_LOOKBACK", "6")))
    SWING_LOOKBACK = max(50, int(os.getenv("EMA_ADV_SWING_LOOKBACK", "90")))
    PIVOT_SPAN = 2
    EMA_TOUCH_ATR = float(os.getenv("EMA_ADV_EMA_TOUCH_ATR", "0.35"))
    FIB_TOL_ATR = float(os.getenv("EMA_ADV_FIB_TOL_ATR", "0.25"))
    NEAR_FIB_ATR = float(os.getenv("EMA_ADV_NEAR_FIB_ATR", "0.65"))
    SL_BUFFER_ATR = float(os.getenv("EMA_ADV_SL_BUFFER_ATR", "0.25"))
    SWEEP_LOOKBACK = max(6, int(os.getenv("EMA_ADV_SWEEP_LOOKBACK", "10")))

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

    def _prep(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = frame.copy()
        c = d["close"].astype(float)
        d["ema20"] = self._ema(c, 20)
        d["ema50"] = self._ema(c, 50)
        d["ema200"] = self._ema(c, 200)
        d["atr"] = self._atr(d, self.cfg.atr_len)
        return d

    @staticmethod
    def _h1_side(d: pd.DataFrame) -> Optional[Side]:
        """Fast H1 bias: EMA20/50 + price structure; EMA200 is soft context."""
        if len(d) < 60:
            return None
        r = d.iloc[-1]
        if float(r.ema20) > float(r.ema50) and float(r.close) >= float(r.ema50):
            return Side.LONG
        if float(r.ema20) < float(r.ema50) and float(r.close) <= float(r.ema50):
            return Side.SHORT
        return None

    @staticmethod
    def _ema200_aligned(d: pd.DataFrame, side: Side) -> bool:
        if len(d) < 200:
            return False
        r = d.iloc[-1]
        if side == Side.LONG:
            return float(r.close) >= float(r.ema200) and float(r.ema50) >= float(r.ema200)
        return float(r.close) <= float(r.ema200) and float(r.ema50) <= float(r.ema200)

    @staticmethod
    def _m15_agrees(d: pd.DataFrame, side: Side) -> bool:
        if len(d) < 60:
            return False
        r = d.iloc[-1]
        if side == Side.LONG:
            return float(r.ema20) > float(r.ema50) and float(r.close) >= float(r.ema50)
        return float(r.ema20) < float(r.ema50) and float(r.close) <= float(r.ema50)

    @staticmethod
    def _trend_label(d: pd.DataFrame) -> str:
        if d.empty:
            return "DATA_ERROR"
        if len(d) < 60:
            return f"WARMUP({len(d)}/60)"
        side = EMAHybridProStrategy._h1_side(d)
        return "BULL" if side == Side.LONG else "BEAR" if side == Side.SHORT else "NEUTRAL"

    def _trend_diagnostics(self, df4h, df1h, df15):
        return {
            "4H": self._trend_label(self._prep(df4h)) if len(df4h) else "DATA_ERROR",
            "1H": self._trend_label(self._prep(df1h)) if len(df1h) else "DATA_ERROR",
            "15M": self._trend_label(self._prep(df15)) if len(df15) else "DATA_ERROR",
        }

    def _ema_slope_ok(self, d, side):
        if len(d) < 8:
            return False
        s20 = float(d.ema20.iloc[-1] - d.ema20.iloc[-5])
        s50 = float(d.ema50.iloc[-1] - d.ema50.iloc[-5])
        return (s20 > 0 and s50 >= 0) if side == Side.LONG else (s20 < 0 and s50 <= 0)

    def _pivot_pairs(self, d):
        span = self.PIVOT_SPAN
        start, end = max(span, len(d)-self.SWING_LOOKBACK), len(d)-span-1
        highs, lows = [], []
        for i in range(start, end):
            w = d.iloc[i-span:i+span+1]
            if float(d.high.iloc[i]) >= float(w.high.max()):
                highs.append((i, float(d.high.iloc[i])))
            if float(d.low.iloc[i]) <= float(w.low.min()):
                lows.append((i, float(d.low.iloc[i])))
        return highs, lows

    def _impulse(self, d, side):
        highs, lows = self._pivot_pairs(d)
        if side == Side.LONG:
            pairs = [(li, lv, hi, hv) for li, lv in lows for hi, hv in highs if li < hi]
            if not pairs:
                return None
            _, low, _, high = max(pairs, key=lambda x: x[2])
        else:
            pairs = [(hi, hv, li, lv) for hi, hv in highs for li, lv in lows if hi < li]
            if not pairs:
                return None
            _, high, _, low = max(pairs, key=lambda x: x[2])
        if high <= low:
            return None
        if side == Side.LONG:
            a, b = high-.500*(high-low), high-.618*(high-low)
        else:
            a, b = low+.500*(high-low), low+.618*(high-low)
        return {"low": low, "high": high, "fib_low": min(a, b), "fib_high": max(a, b)}

    def _ema_touch(self, d, side):
        r, atr = d.iloc[-1], max(float(d.atr.iloc[-1]), 1e-12)
        tol = self.EMA_TOUCH_ATR * atr
        if side == Side.LONG:
            if float(r.low) <= float(r.ema20)+tol and float(r.close) >= float(r.ema20)-tol:
                return "EMA20"
            if float(r.low) <= float(r.ema50)+tol and float(r.close) >= float(r.ema50)-tol:
                return "EMA50"
        else:
            if float(r.high) >= float(r.ema20)-tol and float(r.close) <= float(r.ema20)+tol:
                return "EMA20"
            if float(r.high) >= float(r.ema50)-tol and float(r.close) <= float(r.ema50)+tol:
                return "EMA50"
        return "NONE"

    def _liquidity_sweep(self, d, side):
        if len(d) < self.SWEEP_LOOKBACK+3:
            return "NONE"
        r, prior = d.iloc[-1], d.iloc[-self.SWEEP_LOOKBACK-1:-1]
        if side == Side.LONG:
            lv = float(prior.low.min())
            return "SWEEP_LOW" if float(r.low) < lv and float(r.close) > lv else "NONE"
        hv = float(prior.high.max())
        return "SWEEP_HIGH" if float(r.high) > hv and float(r.close) < hv else "NONE"

    @staticmethod
    def _price_action(d, side):
        if len(d) < 4:
            return "NONE"
        r, p, p2 = d.iloc[-1], d.iloc[-2], d.iloc[-3]
        ro, rh, rl, rc = map(float, (r.open, r.high, r.low, r.close))
        po, ph, pl, pc = map(float, (p.open, p.high, p.low, p.close))
        body = abs(rc-ro)
        rng = max(rh-rl, 1e-12)
        upper = rh-max(ro, rc)
        lower = min(ro, rc)-rl
        if side == Side.LONG:
            if rc > ro and pc < po and rc >= po and ro <= pc: return "BULL_ENGULF"
            if lower >= max(body*1.8, rng*.45) and rc > rl+.55*rng: return "BULL_PIN"
            if ph < float(p2.high) and pl > float(p2.low) and rc > ph: return "INSIDE_BREAK_UP"
            if rc > ro and body/rng >= .55: return "STRONG_BULL_CLOSE"
        else:
            if rc < ro and pc > po and rc <= po and ro >= pc: return "BEAR_ENGULF"
            if upper >= max(body*1.8, rng*.45) and rc < rh-.55*rng: return "BEAR_PIN"
            if ph < float(p2.high) and pl > float(p2.low) and rc < pl: return "INSIDE_BREAK_DOWN"
            if rc < ro and body/rng >= .55: return "STRONG_BEAR_CLOSE"
        return "NONE"

    def _structure_confirm(self, d, side):
        if len(d) < 12:
            return "NONE"
        recent, r = d.iloc[-8:-1], d.iloc[-1]
        if side == Side.LONG:
            if float(r.close) > float(recent.high.max()): return "BOS_UP"
            if float(r.close) > float(d.high.iloc[-3]): return "CHOCH_UP"
        else:
            if float(r.close) < float(recent.low.min()): return "BOS_DOWN"
            if float(r.close) < float(d.low.iloc[-3]): return "CHOCH_DOWN"
        return "NONE"

    def _ob_fvg(self, d, side):
        if len(d) < 5:
            return "NONE"
        a, b, c = d.iloc[-3], d.iloc[-2], d.iloc[-1]
        if side == Side.LONG:
            if float(a.high) < float(c.low): return "BULL_FVG"
            if float(b.close) < float(b.open) and float(c.close) > float(b.high): return "BULL_OB"
        else:
            if float(a.low) > float(c.high): return "BEAR_FVG"
            if float(b.close) > float(b.open) and float(c.close) < float(b.low): return "BEAR_OB"
        return "NONE"

    def _m5_trigger(self, df5, side):
        """Recent closed-M5 execution window. Six bars = up to 30 minutes."""
        if len(df5) < 30:
            return "NONE"
        d = df5.copy()
        c = d.close.astype(float)
        d["ema8"] = self._ema(c, 8)
        d["ema13"] = self._ema(c, 13)
        start = max(1, len(d)-self.M5_TRIGGER_LOOKBACK)
        for i in range(len(d)-1, start-1, -1):
            r, p = d.iloc[i], d.iloc[i-1]
            prior = d.iloc[max(0, i-5):i]
            if side == Side.LONG:
                if float(p.ema8) <= float(p.ema13) and float(r.ema8) > float(r.ema13): return "EMA8_13_CROSS_UP"
                if len(prior) and float(r.close) > float(prior.high.max()): return "M5_BOS_UP"
                if float(r.close) > float(r.ema13) and float(r.ema8) >= float(r.ema13) and float(r.close) > float(r.open): return "M5_RECLAIM_UP"
            else:
                if float(p.ema8) >= float(p.ema13) and float(r.ema8) < float(r.ema13): return "EMA8_13_CROSS_DOWN"
                if len(prior) and float(r.close) < float(prior.low.min()): return "M5_BOS_DOWN"
                if float(r.close) < float(r.ema13) and float(r.ema8) <= float(r.ema13) and float(r.close) < float(r.open): return "M5_RECLAIM_DOWN"
        return "NONE"

    @staticmethod
    def _volume_ratio(d):
        if "volume" not in d.columns or len(d) < 20:
            return 1.0
        avg = float(d.volume.astype(float).iloc[-20:].mean())
        return float(d.volume.iloc[-1]) / max(avg, 1e-12)

    @staticmethod
    def _grade(score):
        return "A+" if score >= 10 else "A" if score >= 7 else "B" if score >= 5 else "BUILDING"

    def _location_state(self, m15, impulse, touch):
        """Fib value is premium; EMA pullback is also a valid location in V3."""
        r = m15.iloc[-1]
        atr = max(float(r.atr), 1e-12)
        lo = impulse["fib_low"] - self.FIB_TOL_ATR*atr
        hi = impulse["fib_high"] + self.FIB_TOL_ATR*atr
        if lo <= float(r.close) <= hi or not (float(r.high) < lo or float(r.low) > hi):
            return True, "FIB_VALUE"
        distance = min(abs(float(r.close)-lo), abs(float(r.close)-hi)) / atr
        if touch != "NONE":
            if distance <= self.NEAR_FIB_ATR:
                return True, "EMA_NEAR_FIB"
            return True, "EMA_PULLBACK"
        return False, f"WAIT_LOCATION({distance:.2f}ATR)"

    def _score(self, h1, m15, df5, side, location_ok, touch, sweep, pa):
        score = 2  # valid H1 EMA20/50 directional bias
        if self._ema200_aligned(h1, side): score += 1
        if self._m15_agrees(m15, side): score += 1
        if location_ok: score += 1
        if touch != "NONE": score += 1
        if sweep != "NONE": score += 2
        structure = self._structure_confirm(m15, side)
        zone = self._ob_fvg(m15, side)
        m5 = self._m5_trigger(df5, side)
        if structure != "NONE": score += 2
        if zone != "NONE": score += 1
        if pa != "NONE": score += 1
        if m5 != "NONE": score += 1
        if self._ema_slope_ok(h1, side): score += 1
        vr = self._volume_ratio(m15)
        return score, self._grade(score), structure, zone, m5, vr

    def _view(self, df1h, df15, df5):
        if len(df1h) < 220 or len(df15) < 120:
            return HybridView(None, "WARMUP", "need >=220 H1 and >=120 M15 candles")
        h1, m15 = self._prep(df1h), self._prep(df15)
        side = self._h1_side(h1)
        if side is None:
            return HybridView(None, "H1_TREND", "H1 EMA20/50 bias unclear")

        q = self.quality_state_1h(df1h)
        if q.adx < self.HARD_ADX_MIN or q.chop > self.HARD_CHOP_MAX or q.q < self.QUALITY_MIN:
            return HybridView(side, "QUALITY", f"1H quality blocked Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f}")

        if not self._m15_agrees(m15, side):
            score = 2 + (1 if self._ema200_aligned(h1, side) else 0)
            return HybridView(side, "M15_TREND", "M15 EMA20/50 momentum opposes H1", score=score, grade=self._grade(score))

        impulse = self._impulse(m15, side)
        if not impulse:
            return HybridView(side, "FIB", "no confirmed impulse swing", score=3, grade="BUILDING")

        touch = self._ema_touch(m15, side)
        location_ok, location = self._location_state(m15, impulse, touch)
        sweep = self._liquidity_sweep(m15, side)
        pa = self._price_action(m15, side)
        score, grade, structure, zone, m5, vr = self._score(h1, m15, df5, side, location_ok, touch, sweep, pa)
        common = dict(
            fib_low=impulse["fib_low"], fib_high=impulse["fib_high"], ema_touch=touch,
            pa=pa, sweep=sweep, structure=structure, zone=zone, m5_trigger=m5,
            volume_ratio=vr, score=score, grade=grade, location=location,
        )

        if not location_ok:
            return HybridView(side, "LOCATION", f"waiting pullback/value: {location}", **common)
        if m5 == "NONE":
            return HybridView(side, "M5_EXEC", f"setup armed; waiting M5 trigger ({self.M5_TRIGGER_LOOKBACK} bars)", **common)

        # Path A: classic liquidity sweep + execution.
        if sweep != "NONE" and score >= self.MIN_SCORE:
            return HybridView(side, "READY", "liquidity + M5 execution passed", entry_path="LIQUIDITY", **common)

        # Path B: structure/PA confirmation + execution, sweep not mandatory.
        confirm_ok = (structure != "NONE" or pa != "NONE") and score >= self.STRONG_CONFIRM_SCORE
        if confirm_ok:
            return HybridView(side, "READY", "confirmation + M5 execution passed", entry_path="CONFIRM", **common)

        # Path C: strong trend pullback. This is the main frequency expansion,
        # but requires substantially stronger 1H quality and H1 slope.
        trend_path = (
            touch != "NONE"
            and q.q >= self.TREND_PATH_Q_MIN
            and q.adx >= self.TREND_PATH_ADX_MIN
            and self._ema_slope_ok(h1, side)
            and score >= self.MIN_SCORE
        )
        if trend_path:
            return HybridView(side, "READY", "strong trend EMA pullback + M5 execution passed", entry_path="TREND_PULLBACK", **common)

        return HybridView(side, "SCORE", f"execution present but confirmation incomplete score={score}", **common)

    def _liquidity_target(self, m15, side, entry, risk, impulse):
        if risk <= 0:
            return None, 0.0
        highs, lows = self._pivot_pairs(m15)
        # Add recent range extremes as practical liquidity pools in addition to
        # confirmed pivots and the impulse endpoint.
        recent = m15.iloc[-40:]
        candidates = []
        if side == Side.LONG:
            candidates = [v for _, v in highs if v > entry]
            candidates += [float(impulse["high"]), float(recent.high.max())]
            for target in sorted(set(candidates)):
                rr = (target-entry)/risk
                if rr >= self.TP2_MIN_RR:
                    return target, rr
        else:
            candidates = [v for _, v in lows if v < entry]
            candidates += [float(impulse["low"]), float(recent.low.min())]
            for target in sorted(set(candidates), reverse=True):
                rr = (entry-target)/risk
                if rr >= self.TP2_MIN_RR:
                    return target, rr
        return None, 0.0

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False):
        if has_open_position or not self._entry_schedule_open():
            return None
        v = self._view(df1h, df15, df5)
        if v.stage != "READY" or v.side is None:
            return None

        m15 = self._prep(df15)
        impulse = self._impulse(m15, v.side)
        if not impulse:
            return None
        r = m15.iloc[-1]
        entry = float(r.close)
        atr = max(float(r.atr), 1e-12)

        if v.side == Side.LONG:
            sl = min(float(impulse["low"]), float(m15.low.iloc[-8:].min())) - self.SL_BUFFER_ATR*atr
            risk = entry-sl
            trend = Trend.BULL
        else:
            sl = max(float(impulse["high"]), float(m15.high.iloc[-8:].max())) + self.SL_BUFFER_ATR*atr
            risk = sl-entry
            trend = Trend.BEAR
        if risk <= 0:
            return None

        tp2, target_rr = self._liquidity_target(m15, v.side, entry, risk, impulse)
        if tp2 is None:
            return None

        q = self.quality_state_1h(df1h)
        reason = (
            f"EMA Hybrid Advanced V3 {v.side.value} | Path={v.entry_path} | Score {v.score}/14 {v.grade} | "
            f"Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f} | Location={v.location} | {v.ema_touch} | {v.sweep} | "
            f"Structure={v.structure} | Zone={v.zone} | M5={v.m5_trigger} | PA={v.pa} | Vol={v.volume_ratio:.2f}x | "
            f"SL=structure+{self.SL_BUFFER_ATR:.2f}ATR | TP1=1R trim60% + SL BE+0.15R | "
            f"TP2=liquidity/swing {target_rr:.2f}R"
        )
        trigger = f"EMA_ADV_V3_{v.entry_path}_{v.grade}_{v.m5_trigger}_{v.structure}_{v.pa}"
        room_pct = abs(tp2-entry)/max(entry, 1e-12)
        structure_px = float(impulse["low"] if v.side == Side.LONG else impulse["high"])
        return EntrySignal(
            v.side, entry, sl, tp2, trend, q.q, q.adx, q.chop,
            SetupType.PULLBACK, trigger, room_pct, atr, structure_px, reason,
        )

    def entry_status(self, df4h, df1h, df15, df5):
        v = self._view(df1h, df15, df5)
        d = self._trend_diagnostics(df4h, df1h, df15)
        paper = bool(getattr(self.cfg, "paper", False))
        open_ = self._entry_schedule_open()
        sched = "24/7 PAPER(OPEN)" if paper else f"24/5 LIVE({'OPEN' if open_ else 'WEEKEND_CLOSED'})"
        side = v.side.value if v.side else "NONE"
        ready = v.stage == "READY" and open_
        return (
            f"EMA-HYBRID-V3 {'READY' if ready else 'WAIT'} | {side} | 4H={d['4H']} | 1H={d['1H']} | 15M={d['15M']} | "
            f"Schedule={sched} | Stage={v.stage} | Score={v.score}/14({v.grade}) | Location={v.location} | Path={v.entry_path} | "
            f"Fib={v.fib_low:.6g}-{v.fib_high:.6g} | EMA={v.ema_touch} | Sweep={v.sweep} | Structure={v.structure} | Zone={v.zone} | "
            f"M5={v.m5_trigger} | PA={v.pa} | Vol={v.volume_ratio:.2f}x | QMin={self.QUALITY_MIN:.0f} ADXMin={self.HARD_ADX_MIN:.0f} "
            f"ChopMax={self.HARD_CHOP_MAX:.0f} | M5Window={self.M5_TRIGGER_LOOKBACK} | SLBuf={self.SL_BUFFER_ATR:.2f}ATR | "
            f"TP2Min={self.TP2_MIN_RR:.2f}R | Reason={v.reason}"
        )
