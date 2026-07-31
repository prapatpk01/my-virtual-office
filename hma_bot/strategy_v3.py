"""HMA Expert MTF V3.1 Balanced.

4H chooses direction, 1H validates trend quality with a soft DMI veto,
15M finds location/structure, and 5M decides execution timing.

V3.1 fixes the main V3 failure mode: a valid 5M trigger could happen inside
the 15M setup candle and be gone by the time that 15M candle closed.  The
execution layer therefore accepts a *recent* 5M trigger (default last 3 closed
bars) as long as EMA alignment is still valid and price has not been chased.
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PrecisionTrendStructureV3(legacy.PrecisionTrendStructureV2):
    """V2 HTF/location model + balanced 5M execution."""

    def __init__(self, config: Optional[legacy.StrategyConfig] = None) -> None:
        super().__init__(config)

        # 5M execution.  The old V3 only inspected the latest 5M bar, which was
        # too easy to miss after waiting for a 15M candle to close.
        self.exec_trigger_lookback = max(1, _env_int("EXEC_TRIGGER_LOOKBACK_BARS", 3))
        self.exec_max_chase_atr = _env_float("EXEC_MAX_CHASE_ATR", 1.10)
        self.exec_min_body_atr = _env_float("EXEC_MIN_BODY_ATR", 0.16)
        self.exec_min_close_location = _env_float("EXEC_MIN_CLOSE_LOCATION", 0.55)
        self.exec_break_body_atr = _env_float("EXEC_BREAK_BODY_ATR", 0.12)

        # 1H DMI is a veto only when it is materially opposite. Q/ADX/CHOP and
        # 4H direction remain the primary gates. A tiny DI crossover no longer
        # silences an otherwise valid setup.
        self.dmi_opposite_tolerance = _env_float("DMI_OPPOSITE_TOLERANCE", 4.0)

        # Room remains setup-aware. Breakout retests need less pre-existing room
        # because the setup itself is a confirmed break/retest of that level.
        self.room_pullback = _env_float("ROOM_PULLBACK_PCT", 0.0060)
        self.room_sweep = _env_float("ROOM_SWEEP_PCT", 0.0050)
        self.room_breakout = _env_float("ROOM_BREAKOUT_RETEST_PCT", 0.0040)

        self.sl_min_atr = _env_float("STRUCTURE_SL_MIN_ATR", 0.60)
        self.sl_max_atr = _env_float("STRUCTURE_SL_MAX_ATR", 1.50)
        self.sl_buffer_atr = _env_float("STRUCTURE_SL_BUFFER_ATR", 0.15)
        self.min_actual_rr = _env_float("MIN_ACTUAL_RR", 1.05)

    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    def _dmi_edge(self, q: QualityState, side: Side) -> float:
        return (q.plus_di - q.minus_di) if side == Side.LONG else (q.minus_di - q.plus_di)

    def _dmi_aligned(self, q: QualityState, side: Side) -> bool:
        return self._dmi_edge(q, side) >= -self.dmi_opposite_tolerance

    def _room_floor(self, setup_type: SetupType) -> float:
        if setup_type == SetupType.BREAKOUT_RETEST:
            return self.room_breakout
        if setup_type == SetupType.SWEEP:
            return self.room_sweep
        return self.room_pullback

    def _execution_trigger(self, df5: pd.DataFrame, side: Side, setup_type: SetupType):
        """Return a recent closed-5M trigger while current alignment remains valid.

        We deliberately scan a short recent window because 15M location is only
        confirmed on a closed 15M candle. The best 5M trigger often occurs in the
        last 5-15 minutes of that candle; latest-bar-only logic missed it.
        """
        if len(df5) < 70:
            return None

        d = df5.copy()
        d["atr"] = self._atr(d, self.cfg.atr_len)
        d["ema8"] = self._ema(d["close"], 8)
        d["ema13"] = self._ema(d["close"], 13)

        cur = d.iloc[-1]
        cur_atr = float(cur["atr"])
        if not np.isfinite(cur_atr) or cur_atr <= 0:
            return None

        if side == Side.LONG:
            current_aligned = float(cur["ema8"]) > float(cur["ema13"])
            current_chase = (float(cur["close"]) - float(cur["ema13"])) / cur_atr
        else:
            current_aligned = float(cur["ema8"]) < float(cur["ema13"])
            current_chase = (float(cur["ema13"]) - float(cur["close"])) / cur_atr

        if not current_aligned or current_chase > self.exec_max_chase_atr:
            return None

        max_back = min(self.exec_trigger_lookback, len(d) - 8)
        for back in range(max_back):
            i = len(d) - 1 - back
            if i < 7:
                break
            r = d.iloc[i]
            p = d.iloc[i - 1]
            atr = float(r["atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue

            body = abs(float(r["close"] - r["open"])) / atr
            close_loc = self._close_location(r, side == Side.LONG)
            if body > 1.60 or close_loc < 0.50:
                continue

            local = d.iloc[max(0, i - 6):i]
            if local.empty:
                continue

            if side == Side.LONG:
                aligned = float(r["ema8"]) > float(r["ema13"])
                cross = aligned and float(p["ema8"]) <= float(p["ema13"])
                reclaim = (
                    aligned and float(p["close"]) <= float(p["ema8"])
                    and float(r["close"]) > float(r["ema8"])
                )
                local_high = float(local["high"].max())
                bos = float(r["close"]) > local_high and float(p["close"]) <= local_high
                prev_break = float(r["close"]) > float(p["high"])
                displacement = (
                    aligned and float(r["close"]) > float(r["open"])
                    and body >= self.exec_min_body_atr
                    and close_loc >= self.exec_min_close_location
                    and prev_break
                )
                continuation = (
                    aligned and float(r["close"]) > float(r["open"])
                    and body >= self.exec_break_body_atr and prev_break
                )
            else:
                aligned = float(r["ema8"]) < float(r["ema13"])
                cross = aligned and float(p["ema8"]) >= float(p["ema13"])
                reclaim = (
                    aligned and float(p["close"]) >= float(p["ema8"])
                    and float(r["close"]) < float(r["ema8"])
                )
                local_low = float(local["low"].min())
                bos = float(r["close"]) < local_low and float(p["close"]) >= local_low
                prev_break = float(r["close"]) < float(p["low"])
                displacement = (
                    aligned and float(r["close"]) < float(r["open"])
                    and body >= self.exec_min_body_atr
                    and close_loc >= self.exec_min_close_location
                    and prev_break
                )
                continuation = (
                    aligned and float(r["close"]) < float(r["open"])
                    and body >= self.exec_break_body_atr and prev_break
                )

            trigger = None
            if setup_type == SetupType.PULLBACK:
                if bos:
                    trigger = "5M_MICRO_BOS"
                elif reclaim or cross:
                    trigger = "5M_EMA8_13_RECLAIM"
                elif continuation:
                    trigger = "5M_PULLBACK_CONTINUATION"
            elif setup_type == SetupType.SWEEP:
                if bos:
                    trigger = "5M_CHOCH_AFTER_SWEEP"
                elif displacement:
                    trigger = "5M_DISPLACEMENT_RECLAIM"
                elif continuation and close_loc >= 0.55:
                    trigger = "5M_SWEEP_CONTINUATION"
            elif setup_type == SetupType.BREAKOUT_RETEST:
                if bos:
                    trigger = "5M_REBREAK_AFTER_RETEST"
                elif displacement:
                    trigger = "5M_DISPLACEMENT_REBREAK"
                elif continuation and close_loc >= 0.55:
                    trigger = "5M_RETEST_CONTINUATION"

            if trigger:
                suffix = "" if back == 0 else f"_RECENT{back}"
                return trigger + suffix, atr

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

    def entry_status(self, df4h: pd.DataFrame, df1h: pd.DataFrame, df15: pd.DataFrame, df5: pd.DataFrame) -> str:
        """Human-readable first blocking stage for Railway /status diagnostics."""
        if len(df15) < 90 or len(df5) < 70:
            return "WARMUP"
        trend = self.trend_state_4h(df4h)
        if trend.trend == Trend.NEUTRAL:
            return "WAIT 4H direction"
        side = Side.LONG if trend.trend == Trend.BULL else Side.SHORT
        quality = self.quality_state_1h(df1h)
        if quality.q < self.cfg.min_trend_quality:
            return f"WAIT Q {quality.q:.0f}<{self.cfg.min_trend_quality:.0f}"
        edge = self._dmi_edge(quality, side)
        if not self._dmi_aligned(quality, side):
            return f"WAIT DMI edge {edge:+.1f}"

        d15 = df15.copy()
        d15["atr"] = self._atr(d15, self.cfg.atr_len)
        d15["ema20"] = self._ema(d15["close"], 20)
        atr15 = float(d15["atr"].iloc[-1])
        if not np.isfinite(atr15) or atr15 <= 0:
            return "WAIT 15M ATR"
        ctx15 = self._structure(d15)
        setup = self._find_setup(d15, side, atr15, ctx15)
        if setup is None:
            return "WAIT 15M location"
        setup_type, _, _ = setup
        trig = self._execution_trigger(df5, side, setup_type)
        if trig is None:
            return f"ARMED {setup_type.value} | WAIT 5M trigger"
        room = self._room_pct(float(df5["close"].iloc[-1]), side, ctx15)
        floor = self._room_floor(setup_type)
        if room < floor:
            return f"BLOCK room {room*100:.2f}%<{floor*100:.2f}%"
        return f"READY {setup_type.value} | {trig[0]}"

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
        if room < self._room_floor(setup_type):
            return None

        sl = self._structure_stop(entry, side, atr15, ctx15, structure_level)
        if side == Side.LONG:
            tp = entry * (1.0 + self.cfg.final_take_profit_pct)
        else:
            tp = entry * (1.0 - self.cfg.final_take_profit_pct)

        risk = abs(entry - sl)
        rr = abs(tp - entry) / max(risk, 1e-12)
        if rr < self.min_actual_rr:
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
                f"{setup_type.value} -> {trigger_name} | 1H DMI edge={self._dmi_edge(quality, side):+.1f} | "
                f"room={room*100:.2f}% | structure SL={risk/entry*100:.2f}% | RR={rr:.2f}"
            ),
        )


MTFStructureStrategy = PrecisionTrendStructureV3
