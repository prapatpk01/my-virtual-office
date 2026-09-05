"""EMA Hybrid Setup D — MA5/MA20 trend pullback confirmation.

Extends Quality V2.3 A/B/C without changing those engines.
Setup D follows the MA5+MA20 workflow from the supplied reference:
1) establish MA20 direction,
2) require a recent MA5/MA20 cross in the 15M-bias direction,
3) wait for a pullback toward the MA corridor,
4) require a confirming candle + volume + trend-quality filters,
5) avoid flat/sideways and late/chasing entries.

Priority: A > C > D > B on the same closed 5M candle.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(HERE, "strategy_c.py")
_SPEC = importlib.util.spec_from_file_location("ema_hybrid_quality_v23_abc", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load EMA Hybrid A/B/C strategy: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

Side = _BASE.Side
TriggerView = _BASE.TriggerView


class EMAHybridProStrategy(_BASE.EMAHybridProStrategy):
    """Quality V2.4: A + precision B + C triple-confirm + D MA5/MA20 pullback."""

    D_ENABLED = os.getenv("EMA_5M_SETUP_D_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }

    D_MA_FAST = max(2, int(os.getenv("EMA_5M_D_MA_FAST", "5")))
    D_MA_SLOW = max(D_MA_FAST + 1, int(os.getenv("EMA_5M_D_MA_SLOW", "20")))
    D_CROSS_LOOKBACK = max(4, int(os.getenv("EMA_5M_D_CROSS_LOOKBACK", "12")))
    D_PULLBACK_LOOKBACK = max(1, int(os.getenv("EMA_5M_D_PULLBACK_LOOKBACK", "4")))
    D_MA20_SLOPE_LOOKBACK = max(1, int(os.getenv("EMA_5M_D_MA20_SLOPE_LOOKBACK", "3")))

    D_PULLBACK_ZONE_ATR = float(os.getenv("EMA_5M_D_PULLBACK_ZONE_ATR", "0.15"))
    D_MAX_DEPTH_ATR = float(os.getenv("EMA_5M_D_MAX_DEPTH_ATR", "0.35"))
    D_MAX_ENTRY_ATR = float(os.getenv("EMA_5M_D_MAX_ENTRY_ATR", "1.00"))
    D_MIN_SPREAD_ATR = float(os.getenv("EMA_5M_D_MIN_SPREAD_ATR", "0.05"))

    D_ADX_MIN = float(os.getenv("EMA_5M_D_ADX_MIN", "16"))
    D_CHOP_MAX = float(os.getenv("EMA_5M_D_CHOP_MAX", "58"))

    D_VOLUME_LEN = max(5, int(os.getenv("EMA_5M_D_VOLUME_LEN", "20")))
    D_VOLUME_RATIO_MIN = float(os.getenv("EMA_5M_D_VOLUME_RATIO_MIN", "1.00"))

    def _prep5(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = super()._prep5(frame)
        c = d["close"].astype(float)
        d["d_ma_fast"] = c.rolling(self.D_MA_FAST, min_periods=self.D_MA_FAST).mean()
        d["d_ma_slow"] = c.rolling(self.D_MA_SLOW, min_periods=self.D_MA_SLOW).mean()
        d["d_vol_avg"] = d["volume"].astype(float).rolling(
            self.D_VOLUME_LEN, min_periods=self.D_VOLUME_LEN
        ).mean()
        return d

    def _recent_d_cross(self, d: pd.DataFrame, side) -> int | None:
        """Return position of the most recent MA5/20 cross, leaving >=1 bar for pullback."""
        start = max(1, len(d) - self.D_CROSS_LOOKBACK - 2)
        end = len(d) - 2  # cross must be at least two bars before current confirmation
        found = None
        for i in range(start, end):
            f0 = float(d.d_ma_fast.iloc[i - 1])
            s0 = float(d.d_ma_slow.iloc[i - 1])
            f1 = float(d.d_ma_fast.iloc[i])
            s1 = float(d.d_ma_slow.iloc[i])
            if any(pd.isna(x) for x in (f0, s0, f1, s1)):
                continue
            if side == Side.LONG and f0 <= s0 and f1 > s1:
                found = i
            elif side == Side.SHORT and f0 >= s0 and f1 < s1:
                found = i
        return found

    def _trigger_d(self, df5: pd.DataFrame, bias_side):
        if not self.D_ENABLED or bias_side is None or len(df5) < 80:
            return None

        d = self._prep5(df5)
        r, p = d.iloc[-1], d.iloc[-2]
        required = (
            r.d_ma_fast, r.d_ma_slow, r.d_vol_avg, r.atr,
        )
        if any(pd.isna(x) for x in required):
            return None

        cross_i = self._recent_d_cross(d, bias_side)
        if cross_i is None:
            return None

        atr = max(float(r.atr), 1e-12)
        fast_now = float(r.d_ma_fast)
        slow_now = float(r.d_ma_slow)
        close_now = float(r.close)
        close_prev = float(p.close)

        slope_ref_i = max(0, len(d) - 1 - self.D_MA20_SLOPE_LOOKBACK)
        slow_ref = float(d.d_ma_slow.iloc[slope_ref_i])
        slow_slope = slow_now - slow_ref

        pb_start = max(cross_i + 1, len(d) - self.D_PULLBACK_LOOKBACK - 1)
        pb = d.iloc[pb_start:-1]
        if pb.empty:
            return None

        pb_atr = pb["atr"].astype(float).clip(lower=1e-12)
        corridor_low = pd.concat(
            [pb["d_ma_fast"].astype(float), pb["d_ma_slow"].astype(float)], axis=1
        ).min(axis=1) - self.D_PULLBACK_ZONE_ATR * pb_atr
        corridor_high = pd.concat(
            [pb["d_ma_fast"].astype(float), pb["d_ma_slow"].astype(float)], axis=1
        ).max(axis=1) + self.D_PULLBACK_ZONE_ATR * pb_atr
        touched_corridor = bool(
            (
                pb["low"].astype(float).le(corridor_high)
                & pb["high"].astype(float).ge(corridor_low)
            ).any()
        )

        adx, chop = self._quality_values_5m(d)
        quality_ok = adx >= self.D_ADX_MIN and chop <= self.D_CHOP_MAX

        vol_avg = max(float(r.d_vol_avg), 1e-12)
        vol_ratio = float(r.volume) / vol_avg
        volume_ok = vol_ratio >= self.D_VOLUME_RATIO_MIN

        spread_atr = abs(fast_now - slow_now) / atr
        spread_ok = spread_atr >= self.D_MIN_SPREAD_ATR
        entry_dist_atr = abs(close_now - slow_now) / atr
        anti_chase_ok = entry_dist_atr <= self.D_MAX_ENTRY_ATR

        if bias_side == Side.LONG:
            trend_ok = fast_now > slow_now and slow_slope > 0 and close_now > slow_now
            depth_floor = pb["d_ma_slow"].astype(float) - self.D_MAX_DEPTH_ATR * pb_atr
            depth_ok = bool(pb["low"].astype(float).ge(depth_floor).all())
            candle_ok = close_now > float(r.open) and close_now > close_prev and close_now > fast_now
            shape = trend_ok and touched_corridor and depth_ok and candle_ok
            if shape:
                if quality_ok and volume_ok and spread_ok and anti_chase_ok:
                    return TriggerView(
                        Side.LONG, True, adx, chop,
                        "D_MA5_MA20", "MA5_MA20_PULLBACK_LONG",
                        f"5M READY LONG: D MA{self.D_MA_FAST}>MA{self.D_MA_SLOW} recent cross + pullback + bounce "
                        f"| MA20 slope UP | Vol={vol_ratio:.2f}x | Spread={spread_atr:.2f}ATR | Entry={entry_dist_atr:.2f}ATR",
                    )
                return TriggerView(
                    Side.LONG, False, adx, chop, "NONE", "NONE",
                    f"5M D_FILTERED LONG | ADX={adx:.1f} CHOP={chop:.1f} Vol={vol_ratio:.2f}x "
                    f"Spread={spread_atr:.2f}ATR Entry={entry_dist_atr:.2f}ATR",
                )
            return None

        trend_ok = fast_now < slow_now and slow_slope < 0 and close_now < slow_now
        depth_ceiling = pb["d_ma_slow"].astype(float) + self.D_MAX_DEPTH_ATR * pb_atr
        depth_ok = bool(pb["high"].astype(float).le(depth_ceiling).all())
        candle_ok = close_now < float(r.open) and close_now < close_prev and close_now < fast_now
        shape = trend_ok and touched_corridor and depth_ok and candle_ok
        if shape:
            if quality_ok and volume_ok and spread_ok and anti_chase_ok:
                return TriggerView(
                    Side.SHORT, True, adx, chop,
                    "D_MA5_MA20", "MA5_MA20_PULLBACK_SHORT",
                    f"5M READY SHORT: D MA{self.D_MA_FAST}<MA{self.D_MA_SLOW} recent cross + pullback + rejection "
                    f"| MA20 slope DOWN | Vol={vol_ratio:.2f}x | Spread={spread_atr:.2f}ATR | Entry={entry_dist_atr:.2f}ATR",
                )
            return TriggerView(
                Side.SHORT, False, adx, chop, "NONE", "NONE",
                f"5M D_FILTERED SHORT | ADX={adx:.1f} CHOP={chop:.1f} Vol={vol_ratio:.2f}x "
                f"Spread={spread_atr:.2f}ATR Entry={entry_dist_atr:.2f}ATR",
            )
        return None

    def _trigger5(self, df5: pd.DataFrame, bias_side):
        # Existing strategy_c already resolves A > C > B.
        base_view = super()._trigger5(df5, bias_side)
        if base_view.ready and base_view.setup in {"A_EMA_CROSS", "C_BOLL_MACD_KDJ"}:
            return base_view

        d_view = self._trigger_d(df5, bias_side)
        if d_view is not None and d_view.ready:
            return d_view

        # D is more specific than generic B, so it owns the trade before B.
        if base_view.ready:
            return base_view
        if d_view is not None:
            return d_view
        return base_view

    def entry_status(self, df4h, df1h, df15, df5):
        base_text = super().entry_status(df4h, df1h, df15, df5)
        return (
            base_text
            + f" | D=MA{self.D_MA_FAST}/{self.D_MA_SLOW} recent-cross+pullback+confirm"
            + f" Vol>={self.D_VOLUME_RATIO_MIN:.2f}x ADX>={self.D_ADX_MIN:.0f} CHOP<={self.D_CHOP_MAX:.0f}"
            + f" Spread>={self.D_MIN_SPREAD_ATR:.2f}ATR Entry<={self.D_MAX_ENTRY_ATR:.2f}ATR"
        )
