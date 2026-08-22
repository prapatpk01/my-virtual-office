"""EMA Hybrid 5M Five-Engine System.

All signal generation is executed on closed 5-minute candles.
Five independent entry engines:
1) EMA pullback reclaim
2) EMA8/13 momentum cross
3) Liquidity sweep reversal
4) BOS breakout with volume
5) Trend continuation

Risk model remains compatible with the EMA Hybrid runtime:
- SL = recent 5M structure + 0.25 ATR buffer
- TP1 = +1R, runtime trims 60% and moves remaining SL to BE+0.15R
- TP2 = nearest 5M liquidity/swing target; fallback 2R if no clean target exists
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
    engine: str
    ready: bool
    score: int
    reason: str
    regime: str = "NEUTRAL"
    pa: str = "NONE"
    volume_ratio: float = 1.0
    structure: str = "NONE"


class EMAHybridProStrategy(base.PrecisionTrendStructureV12):
    """Fast 5M-only strategy with five independent signal engines."""

    SL_BUFFER_ATR = float(os.getenv("EMA_ADV_SL_BUFFER_ATR", "0.25"))
    TP2_MIN_RR = float(os.getenv("EMA_ADV_TP2_MIN_RR", "1.30"))
    TP2_FALLBACK_R = float(os.getenv("EMA_ADV_TP2_FALLBACK_R", "2.0"))
    MIN_SIGNAL_SCORE = int(os.getenv("EMA_5M_MIN_SIGNAL_SCORE", "5"))
    ADX_MIN = float(os.getenv("EMA_5M_ADX_MIN", "9"))
    CHOP_MAX = float(os.getenv("EMA_5M_CHOP_MAX", "70"))
    SWING_LOOKBACK = max(20, int(os.getenv("EMA_5M_SWING_LOOKBACK", "48")))
    STRUCTURE_LOOKBACK = max(6, int(os.getenv("EMA_5M_STRUCTURE_LOOKBACK", "12")))
    SWEEP_LOOKBACK = max(5, int(os.getenv("EMA_5M_SWEEP_LOOKBACK", "10")))
    EMA_TOUCH_ATR = float(os.getenv("EMA_5M_EMA_TOUCH_ATR", "0.35"))
    VOLUME_EXPANSION = float(os.getenv("EMA_5M_VOLUME_EXPANSION", "1.10"))

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
        d["ema8"] = self._ema(c, 8)
        d["ema13"] = self._ema(c, 13)
        d["ema20"] = self._ema(c, 20)
        d["ema50"] = self._ema(c, 50)
        d["ema200"] = self._ema(c, 200)
        d["atr"] = self._atr(d, self.cfg.atr_len)
        return d

    def _regime(self, d: pd.DataFrame) -> tuple[Optional[Side], str]:
        if len(d) < 60:
            return None, "WARMUP"
        r = d.iloc[-1]
        slope50 = float(d.ema50.iloc[-1] - d.ema50.iloc[-6])
        if float(r.close) > float(r.ema50) and float(r.ema8) > float(r.ema13) and slope50 >= 0:
            return Side.LONG, "BULL"
        if float(r.close) < float(r.ema50) and float(r.ema8) < float(r.ema13) and slope50 <= 0:
            return Side.SHORT, "BEAR"
        return None, "NEUTRAL"

    @staticmethod
    def _volume_ratio(d: pd.DataFrame) -> float:
        if "volume" not in d.columns or len(d) < 20:
            return 1.0
        avg = float(d.volume.astype(float).iloc[-20:].mean())
        return float(d.volume.iloc[-1]) / max(avg, 1e-12)

    @staticmethod
    def _pa(d: pd.DataFrame, side: Side) -> str:
        if len(d) < 3:
            return "NONE"
        r, p = d.iloc[-1], d.iloc[-2]
        ro, rh, rl, rc = map(float, (r.open, r.high, r.low, r.close))
        po, ph, pl, pc = map(float, (p.open, p.high, p.low, p.close))
        rng = max(rh-rl, 1e-12)
        body = abs(rc-ro)
        upper = rh-max(ro, rc)
        lower = min(ro, rc)-rl
        if side == Side.LONG:
            if rc > ro and pc < po and rc >= po and ro <= pc:
                return "BULL_ENGULF"
            if lower >= max(body*1.7, rng*0.40) and rc > rl + 0.55*rng:
                return "BULL_PIN"
            if rc > ro and body/rng >= 0.55:
                return "STRONG_BULL"
        else:
            if rc < ro and pc > po and rc <= po and ro >= pc:
                return "BEAR_ENGULF"
            if upper >= max(body*1.7, rng*0.40) and rc < rh - 0.55*rng:
                return "BEAR_PIN"
            if rc < ro and body/rng >= 0.55:
                return "STRONG_BEAR"
        return "NONE"

    def _structure(self, d: pd.DataFrame, side: Side) -> str:
        if len(d) < self.STRUCTURE_LOOKBACK + 2:
            return "NONE"
        r = d.iloc[-1]
        prior = d.iloc[-self.STRUCTURE_LOOKBACK-1:-1]
        if side == Side.LONG and float(r.close) > float(prior.high.max()):
            return "BOS_UP"
        if side == Side.SHORT and float(r.close) < float(prior.low.min()):
            return "BOS_DOWN"
        return "NONE"

    def _sweep(self, d: pd.DataFrame, side: Side) -> bool:
        if len(d) < self.SWEEP_LOOKBACK + 2:
            return False
        r = d.iloc[-1]
        prior = d.iloc[-self.SWEEP_LOOKBACK-1:-1]
        if side == Side.LONG:
            low = float(prior.low.min())
            return float(r.low) < low and float(r.close) > low
        high = float(prior.high.max())
        return float(r.high) > high and float(r.close) < high

    def _quality(self, d: pd.DataFrame):
        try:
            return self.quality_state_1h(d)
        except Exception:
            return None

    def _quality_ok(self, d: pd.DataFrame) -> tuple[bool, float, float, float]:
        q = self._quality(d)
        if q is None:
            return True, 50.0, 15.0, 50.0
        qv, adx, chop = float(q.q), float(q.adx), float(q.chop)
        return adx >= self.ADX_MIN and chop <= self.CHOP_MAX, qv, adx, chop

    def _engine_pullback_reclaim(self, d: pd.DataFrame, side: Side) -> SignalView:
        r, p = d.iloc[-1], d.iloc[-2]
        atr = max(float(r.atr), 1e-12)
        tol = self.EMA_TOUCH_ATR * atr
        touched = False
        reclaimed = False
        if side == Side.LONG:
            touched = float(p.low) <= float(p.ema13) + tol or float(p.low) <= float(p.ema20) + tol
            reclaimed = float(r.close) > float(r.ema13) and float(r.close) > float(r.open)
        else:
            touched = float(p.high) >= float(p.ema13) - tol or float(p.high) >= float(p.ema20) - tol
            reclaimed = float(r.close) < float(r.ema13) and float(r.close) < float(r.open)
        pa = self._pa(d, side)
        score = 2 + (2 if touched else 0) + (2 if reclaimed else 0) + (1 if pa != "NONE" else 0)
        return SignalView(side, "EMA_PULLBACK", touched and reclaimed and score >= self.MIN_SIGNAL_SCORE,
                          score, f"touch={touched} reclaim={reclaimed}", pa=pa, volume_ratio=self._volume_ratio(d))

    def _engine_cross(self, d: pd.DataFrame, side: Side) -> SignalView:
        r, p = d.iloc[-1], d.iloc[-2]
        if side == Side.LONG:
            cross = float(p.ema8) <= float(p.ema13) and float(r.ema8) > float(r.ema13)
            momentum = float(r.close) > float(r.open) and float(r.close) > float(r.ema20)
        else:
            cross = float(p.ema8) >= float(p.ema13) and float(r.ema8) < float(r.ema13)
            momentum = float(r.close) < float(r.open) and float(r.close) < float(r.ema20)
        vr = self._volume_ratio(d)
        score = 2 + (3 if cross else 0) + (1 if momentum else 0) + (1 if vr >= 1.0 else 0)
        return SignalView(side, "EMA_CROSS", cross and momentum and score >= self.MIN_SIGNAL_SCORE,
                          score, f"cross={cross} momentum={momentum}", volume_ratio=vr)

    def _engine_sweep(self, d: pd.DataFrame, side: Side) -> SignalView:
        sweep = self._sweep(d, side)
        pa = self._pa(d, side)
        vr = self._volume_ratio(d)
        score = 2 + (3 if sweep else 0) + (2 if pa != "NONE" else 0) + (1 if vr >= 1.0 else 0)
        return SignalView(side, "LIQUIDITY_SWEEP", sweep and pa != "NONE" and score >= self.MIN_SIGNAL_SCORE,
                          score, f"sweep={sweep} pa={pa}", pa=pa, volume_ratio=vr)

    def _engine_bos(self, d: pd.DataFrame, side: Side) -> SignalView:
        structure = self._structure(d, side)
        vr = self._volume_ratio(d)
        r = d.iloc[-1]
        if side == Side.LONG:
            body_ok = float(r.close) > float(r.open) and float(r.close) > float(r.ema13)
        else:
            body_ok = float(r.close) < float(r.open) and float(r.close) < float(r.ema13)
        score = 2 + (3 if structure != "NONE" else 0) + (2 if vr >= self.VOLUME_EXPANSION else 0) + (1 if body_ok else 0)
        ready = structure != "NONE" and body_ok and vr >= self.VOLUME_EXPANSION and score >= self.MIN_SIGNAL_SCORE
        return SignalView(side, "BOS_BREAKOUT", ready, score,
                          f"structure={structure} vol={vr:.2f}x", volume_ratio=vr, structure=structure)

    def _engine_continuation(self, d: pd.DataFrame, side: Side) -> SignalView:
        r, p = d.iloc[-1], d.iloc[-2]
        if side == Side.LONG:
            stack = float(r.ema8) > float(r.ema13) > float(r.ema50)
            shallow = float(p.low) <= float(p.ema13) and float(p.close) >= float(p.ema50)
            continue_bar = float(r.close) > float(p.high) and float(r.close) > float(r.open)
        else:
            stack = float(r.ema8) < float(r.ema13) < float(r.ema50)
            shallow = float(p.high) >= float(p.ema13) and float(p.close) <= float(p.ema50)
            continue_bar = float(r.close) < float(p.low) and float(r.close) < float(r.open)
        vr = self._volume_ratio(d)
        score = 2 + (2 if stack else 0) + (2 if shallow else 0) + (2 if continue_bar else 0) + (1 if vr >= 1.0 else 0)
        return SignalView(side, "TREND_CONTINUATION", stack and shallow and continue_bar and score >= self.MIN_SIGNAL_SCORE,
                          score, f"stack={stack} shallow={shallow} continuation={continue_bar}", volume_ratio=vr)

    def _best_signal(self, df5: pd.DataFrame) -> tuple[Optional[SignalView], dict]:
        if len(df5) < 80:
            return None, {"regime": "WARMUP", "q": 0.0, "adx": 0.0, "chop": 0.0}
        d = self._prep(df5)
        side, regime = self._regime(d)
        quality_ok, qv, adx, chop = self._quality_ok(d)
        meta = {"regime": regime, "q": qv, "adx": adx, "chop": chop}
        if side is None or not quality_ok:
            return None, meta
        engines = [
            self._engine_pullback_reclaim(d, side),
            self._engine_cross(d, side),
            self._engine_sweep(d, side),
            self._engine_bos(d, side),
            self._engine_continuation(d, side),
        ]
        ready = [x for x in engines if x.ready]
        if not ready:
            meta["engines"] = engines
            return None, meta
        best = max(ready, key=lambda x: x.score)
        meta["engines"] = engines
        return best, meta

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
        start = max(span, len(d)-self.SWING_LOOKBACK)
        highs, lows = [], []
        for i in range(start, len(d)-span):
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
                rr = (target-entry)/risk
                if rr >= self.TP2_MIN_RR:
                    return target, rr, "SWING_HIGH"
            target = entry + self.TP2_FALLBACK_R*risk
        else:
            for target in sorted(set((x for x in lows if x < entry)), reverse=True):
                rr = (entry-target)/risk
                if rr >= self.TP2_MIN_RR:
                    return target, rr, "SWING_LOW"
            target = entry - self.TP2_FALLBACK_R*risk
        return target, self.TP2_FALLBACK_R, "FALLBACK_R"

    def generate_entry(self, df4h, df1h, df15, df5, has_open_position=False):
        if has_open_position or not self._entry_schedule_open():
            return None
        best, meta = self._best_signal(df5)
        if best is None or best.side is None:
            return None
        d = self._prep(df5)
        entry = float(d.close.iloc[-1])
        sl, risk = self._structure_stop(d, best.side, entry)
        if risk <= 0:
            return None
        tp2, rr, target_type = self._tp2(d, best.side, entry, risk)
        atr = max(float(d.atr.iloc[-1]), 1e-12)
        trend = Trend.BULL if best.side == Side.LONG else Trend.BEAR
        reason = (
            f"EMA Hybrid 5M | Engine={best.engine} | Score={best.score} | Regime={meta['regime']} | "
            f"Q={meta['q']:.0f} ADX={meta['adx']:.1f} CHOP={meta['chop']:.1f} | {best.reason} | "
            f"SL=5M structure+{self.SL_BUFFER_ATR:.2f}ATR | TP1=1R trim60% + SL BE+0.15R | "
            f"TP2={target_type} {rr:.2f}R"
        )
        trigger = f"EMA5M_{best.engine}_{best.score}"
        room_pct = abs(tp2-entry)/max(entry, 1e-12)
        structure_px = sl + self.SL_BUFFER_ATR*atr if best.side == Side.LONG else sl - self.SL_BUFFER_ATR*atr
        return EntrySignal(best.side, entry, sl, tp2, trend, meta['q'], meta['adx'], meta['chop'],
                           SetupType.PULLBACK, trigger, room_pct, atr, structure_px, reason)

    def entry_status(self, df4h, df1h, df15, df5):
        best, meta = self._best_signal(df5)
        paper = bool(getattr(self.cfg, "paper", False))
        open_ = self._entry_schedule_open()
        sched = "24/7 PAPER(OPEN)" if paper else f"24/5 LIVE({'OPEN' if open_ else 'WEEKEND_CLOSED'})"
        if best is not None:
            return (f"EMA-5M READY | {best.side.value} | Engine={best.engine} | Score={best.score} | "
                    f"Regime={meta['regime']} | Q={meta['q']:.0f} ADX={meta['adx']:.1f} CHOP={meta['chop']:.1f} | "
                    f"Schedule={sched} | SLBuf={self.SL_BUFFER_ATR:.2f}ATR | TP2Min={self.TP2_MIN_RR:.2f}R")
        engines = meta.get("engines") or []
        eng = ", ".join(f"{x.engine}:{x.score}" for x in engines) if engines else "NONE"
        return (f"EMA-5M WAIT | Regime={meta.get('regime','?')} | Q={meta.get('q',0):.0f} "
                f"ADX={meta.get('adx',0):.1f} CHOP={meta.get('chop',0):.1f} | Engines={eng} | "
                f"Schedule={sched} | MinScore={self.MIN_SIGNAL_SCORE} | SLBuf={self.SL_BUFFER_ATR:.2f}ATR")
