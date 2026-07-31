"""HMA Expert MTF V3.

4H chooses direction, 1H validates trend quality + DMI direction,
15M finds location/structure, and 5M decides execution timing.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

import strategy as legacy

Side = legacy.Side
Trend = legacy.Trend
SetupType = legacy.SetupType
ExitReason = legacy.ExitReason
EntrySignal = legacy.EntrySignal
StructureExit = legacy.StructureExit
TrendState = legacy.TrendState
QualityState = legacy.QualityState


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PrecisionTrendStructureV3(legacy.PrecisionTrendStructureV2):
    """Keeps V2's proven HTF/location model and upgrades execution quality."""

    def __init__(self, config: Optional[legacy.StrategyConfig] = None) -> None:
        super().__init__(config)
        self.exec_max_chase_atr = _env_float("EXEC_MAX_CHASE_ATR", 0.90)
        self.exec_min_body_atr = _env_float("EXEC_MIN_BODY_ATR", 0.22)
        self.exec_min_close_location = _env_float("EXEC_MIN_CLOSE_LOCATION", 0.62)
        self.sl_min_atr = _env_float("STRUCTURE_SL_MIN_ATR", 0.60)
        self.sl_max_atr = _env_float("STRUCTURE_SL_MAX_ATR", 1.50)
        self.sl_buffer_atr = _env_float("STRUCTURE_SL_BUFFER_ATR", 0.15)

    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    def _dmi_aligned(self, q: QualityState, side: Side) -> bool:
        return q.plus_di > q.minus_di if side == Side.LONG else q.minus_di > q.plus_di

    def _execution_trigger(self, df5: pd.DataFrame, side: Side, setup_type: SetupType):
        if len(df5) < 70:
            return None
        d = df5.copy()
        d["atr"] = self._atr(d, self.cfg.atr_len)
        d["ema8"] = self._ema(d["close"], 8)
        d["ema13"] = self._ema(d["close"], 13)
        r, p = d.iloc[-1], d.iloc[-2]
        atr = float(r["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None

        ctx = self._structure(d)
        body = abs(float(r["close"] - r["open"])) / atr
        close_loc = self._close_location(r, side == Side.LONG)
        if body > 1.50 or close_loc < 0.50:
            return None

        if side == Side.LONG:
            aligned = float(r["ema8"]) > float(r["ema13"])
            cross = aligned and float(p["ema8"]) <= float(p["ema13"])
            reclaim = aligned and float(p["close"]) <= float(p["ema8"]) and float(r["close"]) > float(r["ema8"])
            micro = ctx["mic_h"][-1][1] if ctx["mic_h"] else None
            bos = micro is not None and float(r["close"]) > micro and float(p["close"]) <= micro
            displacement = (
                float(r["close"]) > float(r["open"]) and
                body >= self.exec_min_body_atr and
                close_loc >= self.exec_min_close_location and
                float(r["close"]) > float(p["high"])
            )
            chase = (float(r["close"]) - float(r["ema13"])) / atr
        else:
            aligned = float(r["ema8"]) < float(r["ema13"])
            cross = aligned and float(p["ema8"]) >= float(p["ema13"])
            reclaim = aligned and float(p["close"]) >= float(p["ema8"]) and float(r["close"]) < float(r["ema8"])
            micro = ctx["mic_l"][-1][1] if ctx["mic_l"] else None
            bos = micro is not None and float(r["close"]) < micro and float(p["close"]) >= micro
            displacement = (
                float(r["close"]) < float(r["open"]) and
                body >= self.exec_min_body_atr and
                close_loc >= self.exec_min_close_location and
                float(r["close"]) < float(p["low"])
            )
            chase = (float(r["ema13"]) - float(r["close"])) / atr

        if chase > self.exec_max_chase_atr:
            return None

        if setup_type == SetupType.PULLBACK:
            if bos:
                return "5M_MICRO_BOS", atr
            if reclaim or cross:
                return "5M_EMA8_13_RECLAIM", atr
        elif setup_type == SetupType.SWEEP:
            if bos:
                return "5M_CHOCH_AFTER_SWEEP", atr
            if aligned and displacement:
                return "5M_DISPLACEMENT_RECLAIM", atr
        elif setup_type == SetupType.BREAKOUT_RETEST:
            if bos:
                return "5M_REBREAK_AFTER_RETEST", atr
            if aligned and displacement:
                return "5M_DISPLACEMENT_REBREAK", atr
        return None

    def _structure_stop(self, entry: float, side: Side, atr15: float, ctx: dict, structure_level: float) -> float:
        if side == Side.LONG:
            refs = [float(structure_level)] if structure_level < entry else []
            refs += [v for _, v in ctx["maj_l"][-3:] if v < entry]
            refs += [v for _, v in ctx["mic_l"][-3:] if v < entry]
            reference = max(refs) if refs else entry - atr15
            raw = reference - self.sl_buffer_atr * atr15
            dist = entry - raw
        else:
            refs = [float(structure_level)] if structure_level > entry else []
            refs += [v for _, v in ctx["maj_h"][-3:] if v > entry]
            refs += [v for _, v in ctx["mic_h"][-3:] if v > entry]
            reference = min(refs) if refs else entry + atr15
            raw = reference + self.sl_buffer_atr * atr15
            dist = raw - entry

        dist = float(np.clip(dist, self.sl_min_atr * atr15, self.sl_max_atr * atr15))
        return entry - dist if side == Side.LONG else entry + dist

    def generate_entry(
        self,
        df4h: pd.DataFrame,
        df1h: pd.DataFrame,
        df15: pd.DataFrame,
        df5: pd.DataFrame,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        if has_open_position or len(df15) < 90 or len(df5) < 70:
            return None

        trend = self.trend_state_4h(df4h)
        if trend.trend == Trend.NEUTRAL:
            return None

        quality = self.quality_state_1h(df1h)
        if quality.q < self.cfg.min_trend_quality:
            return None

        side = Side.LONG if trend.trend == Trend.BULL else Side.SHORT
        if not self._dmi_aligned(quality, side):
            return None

        d15 = df15.copy()
        d15["atr"] = self._atr(d15, self.cfg.atr_len)
        d15["ema20"] = self._ema(d15["close"], 20)
        atr15 = float(d15["atr"].iloc[-1])
        if not np.isfinite(atr15) or atr15 <= 0:
            return None

        ctx15 = self._structure(d15)
        setup = self._find_setup(d15, side, atr15, ctx15)
        if setup is None:
            return None
        setup_type, structure_level, _ = setup

        exec_sig = self._execution_trigger(df5, side, setup_type)
        if exec_sig is None:
            return None
        trigger_name, _ = exec_sig

        entry = float(df5["close"].iloc[-1])
        room = self._room_pct(entry, side, ctx15)
        if room < self.cfg.min_room_pct:
            return None

        sl = self._structure_stop(entry, side, atr15, ctx15, structure_level)
        if side == Side.LONG:
            tp = entry * (1.0 + self.cfg.final_take_profit_pct)
        else:
            tp = entry * (1.0 - self.cfg.final_take_profit_pct)

        risk = abs(entry - sl)
        rr = abs(tp - entry) / max(risk, 1e-12)
        if rr < 1.05:
            return None

        return EntrySignal(
            side=side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=trend.trend,
            q_1h=quality.q,
            adx_1h=quality.adx,
            chop_1h=quality.chop,
            setup=setup_type,
            trigger=trigger_name,
            room_pct=room,
            atr15=atr15,
            structure_level=structure_level,
            reason=(
                f"{setup_type.value} -> {trigger_name} | 1H DMI aligned | "
                f"room={room*100:.2f}% | structure SL={risk/entry*100:.2f}% | RR={rr:.2f}"
            ),
        )


MTFStructureStrategy = PrecisionTrendStructureV3
