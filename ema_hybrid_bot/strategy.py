"""EMA Hybrid Pro Advanced — Direction -> Location -> Liquidity -> Structure -> Execution.

Progressive scoring with two entry paths:
A) Liquidity path: value-zone + sweep + score >= 7.
B) Strong-confirmation path: value-zone + structure + M5 trigger + PA + score >= 8.

Risk/exit model:
- SL = structure beyond swing + 0.25 ATR (configurable).
- TP1 = +1R, runtime trims 60% and moves remaining SL to BE + 0.15R.
- TP2 = next eligible M15 liquidity/swing target with at least 1.5R room.
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
    STRONG_CONFIRM_SCORE = int(os.getenv("EMA_ADV_STRONG_CONFIRM_SCORE", "8"))
    TP2_MIN_RR = float(os.getenv("EMA_ADV_TP2_MIN_RR", "1.5"))
    SWING_LOOKBACK = 90
    PIVOT_SPAN = 2
    EMA_TOUCH_ATR = float(os.getenv("EMA_ADV_EMA_TOUCH_ATR", "0.22"))
    FIB_TOL_ATR = float(os.getenv("EMA_ADV_FIB_TOL_ATR", "0.15"))
    NEAR_FIB_ATR = float(os.getenv("EMA_ADV_NEAR_FIB_ATR", "0.35"))
    SL_BUFFER_ATR = float(os.getenv("EMA_ADV_SL_BUFFER_ATR", "0.25"))
    SWEEP_LOOKBACK = 12

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.stage_locks_enabled = False
        self.live_schedule_timezone = os.getenv("LIVE_SCHEDULE_TIMEZONE", "Asia/Seoul").strip() or "Asia/Seoul"
        self._live_tz = ZoneInfo(self.live_schedule_timezone)

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
        if len(d) < 200:
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
        return "BULL" if side == Side.LONG else "BEAR" if side == Side.SHORT else "NEUTRAL"

    def _trend_diagnostics(self, df4h, df1h, df15):
        return {
            "4H": self._trend_label(self._prep(df4h)) if len(df4h) else "DATA_ERROR",
            "1H": self._trend_label(self._prep(df1h)) if len(df1h) else "DATA_ERROR",
            "15M": self._trend_label(self._prep(df15)) if len(df15) else "DATA_ERROR",
        }

    def _ema_slope_ok(self, d: pd.DataFrame, side: Side) -> bool:
        if len(d) < 8:
            return False
        s20 = float(d.ema20.iloc[-1] - d.ema20.iloc[-5])
        s50 = float(d.ema50.iloc[-1] - d.ema50.iloc[-5])
        return (s20 > 0 and s50 >= 0) if side == Side.LONG else (s20 < 0 and s50 <= 0)

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
            _, low, _, high = max(pairs, key=lambda x: x[2])
        else:
            pairs = [(hi, hv, li, lv) for hi, hv in highs for li, lv in lows if hi < li]
            if not pairs:
                return None
            _, high, _, low = max(pairs, key=lambda x: x[2])
        if high <= low:
            return None
        if side == Side.LONG:
            fib50 = high - 0.500 * (high-low)
            fib618 = high - 0.618 * (high-low)
        else:
            fib50 = low + 0.500 * (high-low)
            fib618 = low + 0.618 * (high-low)
        return {"low": low, "high": high, "fib_low": min(fib50, fib618), "fib_high": max(fib50, fib618)}

    def _ema_touch(self, d, side):
        r = d.iloc[-1]
        atr = max(float(r.atr), 1e-12)
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
        r = d.iloc[-1]
        prior = d.iloc[-self.SWEEP_LOOKBACK-1:-1]
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
            if lower >= max(body*1.8, rng*0.45) and rc > rl+0.55*rng: return "BULL_PIN"
            if ph < float(p2.high) and pl > float(p2.low) and rc > ph: return "INSIDE_BREAK_UP"
            if rc > ro and body/rng >= 0.60: return "STRONG_BULL_CLOSE"
        else:
            if rc < ro and pc > po and rc <= po and ro >= pc: return "BEAR_ENGULF"
            if upper >= max(body*1.8, rng*0.45) and rc < rh-0.55*rng: return "BEAR_PIN"
            if ph < float(p2.high) and pl > float(p2.low) and rc < pl: return "INSIDE_BREAK_DOWN"
            if rc < ro and body/rng >= 0.60: return "STRONG_BEAR_CLOSE"
        return "NONE"

    def _structure_confirm(self, d, side):
        if len(d) < 12:
            return "NONE"
        recent = d.iloc[-8:-1]
        r = d.iloc[-1]
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
        if len(df5) < 25:
            return "NONE"
        d = df5.copy()
        c = d.close.astype(float)
        d["ema8"] = self._ema(c, 8)
        d["ema13"] = self._ema(c, 13)
        r, p = d.iloc[-1], d.iloc[-2]
        if side == Side.LONG:
            if float(p.ema8) <= float(p.ema13) and float(r.ema8) > float(r.ema13): return "EMA8_13_CROSS_UP"
            if float(r.close) > float(d.high.iloc[-6:-1].max()): return "M5_BOS_UP"
            if float(r.close) > float(r.ema13) and float(r.close) > float(r.open): return "M5_RECLAIM_UP"
        else:
            if float(p.ema8) >= float(p.ema13) and float(r.ema8) < float(r.ema13): return "EMA8_13_CROSS_DOWN"
            if float(r.close) < float(d.low.iloc[-6:-1].min()): return "M5_BOS_DOWN"
            if float(r.close) < float(r.ema13) and float(r.close) < float(r.open): return "M5_RECLAIM_DOWN"
        return "NONE"

    @staticmethod
    def _volume_ratio(d):
        if "volume" not in d.columns or len(d) < 20:
            return 1.0
        avg = float(d.volume.astype(float).iloc[-20:].mean())
        return float(d.volume.iloc[-1]) / max(avg, 1e-12)

    @staticmethod
    def _grade(score):
        return "A+" if score >= 9 else "A" if score >= 7 else "B" if score >= 5 else "BUILDING"

    def _location_state(self, m15, impulse, touch):
        r = m15.iloc[-1]
        atr = max(float(r.atr), 1e-12)
        fib_lo = impulse["fib_low"] - self.FIB_TOL_ATR*atr
        fib_hi = impulse["fib_high"] + self.FIB_TOL_ATR*atr
        bar_overlaps = not (float(r.high) < fib_lo or float(r.low) > fib_hi)
        close_inside = fib_lo <= float(r.close) <= fib_hi
        if close_inside or bar_overlaps:
            return True, "FIB_VALUE"
        distance = min(abs(float(r.close)-fib_lo), abs(float(r.close)-fib_hi)) / atr
        if touch != "NONE" and distance <= self.NEAR_FIB_ATR:
            return True, "EMA_NEAR_FIB"
        return False, f"WAIT_FIB({distance:.2f}ATR)"

    def _score(self, h1, m15, df5, side, location_ok, touch, sweep, pa):
        score = 0
        if self._ema_side(h1) == side: score += 2
        if self._ema_side(m15) == side: score += 1
        if location_ok: score += 1
        if touch != "NONE": score += 1
        if sweep != "NONE": score += 2
        structure = self._structure_confirm(m15, side)
        if structure != "NONE": score += 2
        zone = self._ob_fvg(m15, side)
        if zone != "NONE": score += 1
        if pa != "NONE": score += 1
        m5 = self._m5_trigger(df5, side)
        if m5 != "NONE": score += 1
        if self._ema_slope_ok(h1, side): score += 1
        vr = self._volume_ratio(m15)
        if vr >= 1.10: score += 1
        return score, self._grade(score), structure, zone, m5, vr

    def _view(self, df1h, df15, df5):
        if len(df1h) < 220 or len(df15) < 220:
            return HybridView(None, "WARMUP", "need >=220 H1/M15 candles")
        h1, m15 = self._prep(df1h), self._prep(df15)
        side = self._ema_side(h1)
        if side is None:
            return HybridView(None, "H1_TREND", "H1 EMA20/50/200 not aligned")
        if self._ema_side(m15) != side:
            score = 2 + (1 if self._ema_slope_ok(h1, side) else 0)
            return HybridView(side, "M15_TREND", "M15 triple EMA not aligned with H1", score=score, grade=self._grade(score))
        impulse = self._impulse(m15, side)
        if not impulse:
            score = 3 + (1 if self._ema_slope_ok(h1, side) else 0)
            return HybridView(side, "FIB", "no confirmed impulse swing", score=score, grade=self._grade(score))
        touch = self._ema_touch(m15, side)
        location_ok, location = self._location_state(m15, impulse, touch)
        sweep = self._liquidity_sweep(m15, side)
        pa = self._price_action(m15, side)
        score, grade, structure, zone, m5, vr = self._score(h1, m15, df5, side, location_ok, touch, sweep, pa)
        common = dict(fib_low=impulse["fib_low"], fib_high=impulse["fib_high"], ema_touch=touch,
                      pa=pa, sweep=sweep, structure=structure, zone=zone, m5_trigger=m5,
                      volume_ratio=vr, score=score, grade=grade, location=location)
        if not location_ok:
            return HybridView(side, "LOCATION", f"waiting value zone: {location}", **common)
        if sweep != "NONE" and score >= self.MIN_SCORE:
            return HybridView(side, "READY", "liquidity path passed", entry_path="LIQUIDITY", **common)
        strong_confirm = structure != "NONE" and m5 != "NONE" and pa != "NONE"
        if strong_confirm and score >= self.STRONG_CONFIRM_SCORE:
            return HybridView(side, "READY", "strong confirmation path passed", entry_path="CONFIRM", **common)
        if sweep == "NONE":
            return HybridView(side, "CONFIRM", f"need sweep OR Structure+M5+PA; score {score}/{self.STRONG_CONFIRM_SCORE}", **common)
        return HybridView(side, "SCORE", f"setup score {score} < {self.MIN_SCORE}", **common)

    def _liquidity_target(self, m15, side: Side, entry: float, risk: float, impulse) -> tuple[Optional[float], float]:
        """Nearest confirmed M15 swing/liquidity level that leaves >= TP2_MIN_RR room."""
        if risk <= 0:
            return None, 0.0
        highs, lows = self._pivot_pairs(m15)
        candidates = []
        if side == Side.LONG:
            candidates.extend(v for _, v in highs if v > entry)
            candidates.append(float(impulse["high"]))
            candidates = sorted(set(candidates))
            for target in candidates:
                rr = (target-entry)/risk
                if rr >= self.TP2_MIN_RR:
                    return target, rr
        else:
            candidates.extend(v for _, v in lows if v < entry)
            candidates.append(float(impulse["low"]))
            candidates = sorted(set(candidates), reverse=True)
            for target in candidates:
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
            f"EMA Hybrid Advanced {v.side.value} | Path={v.entry_path} | Score {v.score}/14 {v.grade} | "
            f"Location={v.location} | {v.ema_touch} | {v.sweep} | Structure={v.structure} | Zone={v.zone} | "
            f"M5={v.m5_trigger} | PA={v.pa} | Vol={v.volume_ratio:.2f}x | "
            f"SL=structure+{self.SL_BUFFER_ATR:.2f}ATR | TP1=1R trim60% + SL BE+0.15R | "
            f"TP2=liquidity/swing {target_rr:.2f}R"
        )
        trigger = f"EMA_ADV_{v.entry_path}_{v.grade}_{v.m5_trigger}_{v.structure}_{v.pa}"
        room_pct = abs(tp2-entry)/max(entry, 1e-12)
        structure_px = float(impulse["low"] if v.side == Side.LONG else impulse["high"])
        return EntrySignal(v.side, entry, sl, tp2, trend, q.q, q.adx, q.chop,
                           SetupType.PULLBACK, trigger, room_pct, atr, structure_px, reason)

    def entry_status(self, df4h, df1h, df15, df5):
        v = self._view(df1h, df15, df5)
        d = self._trend_diagnostics(df4h, df1h, df15)
        paper = bool(getattr(self.cfg, "paper", False))
        schedule_open = self._entry_schedule_open()
        sched = "24/7 PAPER(OPEN)" if paper else f"24/5 LIVE({'OPEN' if schedule_open else 'WEEKEND_CLOSED'})"
        mtf = "YES" if d["1H"] in ("BULL", "BEAR") and d["1H"] == d["15M"] else "NO"
        side = v.side.value if v.side else "NONE"
        ready = v.stage == "READY" and schedule_open
        needed = max(0, (self.MIN_SCORE if v.sweep != "NONE" else self.STRONG_CONFIRM_SCORE) - v.score)
        return (
            f"EMA-HYBRID-ADV {'READY' if ready else 'WAIT'} | {side} | 4H={d['4H']} | 1H={d['1H']} | "
            f"15M={d['15M']} | MTF={mtf} | Schedule={sched} | Stage={v.stage} | Score={v.score}/14({v.grade}) "
            f"Need={needed} | Location={v.location} | Path={v.entry_path} | Fib={v.fib_low:.6g}-{v.fib_high:.6g} | "
            f"EMA={v.ema_touch} | Sweep={v.sweep} | Structure={v.structure} | Zone={v.zone} | M5={v.m5_trigger} | "
            f"PA={v.pa} | Vol={v.volume_ratio:.2f}x | SLBuf={self.SL_BUFFER_ATR:.2f}ATR | TP2Min={self.TP2_MIN_RR:.1f}R | Reason={v.reason}"
        )
