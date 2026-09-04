"""EMA Hybrid Setup C — Bollinger + MACD + KDJ triple-confirmation reversal.

This module extends the deployed Quality V2.1 A/B strategy without changing A or B.
Priority is A > C > B when multiple setups are valid on the same closed 5M bar.

Setup C is trend-aligned with the existing 15M bias. It is intentionally a
pullback/reversal timing engine, not a counter-trend mean-reversion system.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(HERE, "strategy.py")
_SPEC = importlib.util.spec_from_file_location("ema_hybrid_quality_v21_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load EMA Hybrid base strategy: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

Side = _BASE.Side
TriggerView = _BASE.TriggerView


class EMAHybridProStrategy(_BASE.EMAHybridProStrategy):
    """Quality V2.1 A/B plus Setup C BOLL+MACD+KDJ reversal confirmation."""

    C_ENABLED = os.getenv("EMA_5M_SETUP_C_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }

    BOLL_LEN = max(10, int(os.getenv("EMA_5M_C_BOLL_LEN", "20")))
    BOLL_STD = float(os.getenv("EMA_5M_C_BOLL_STD", "2.0"))
    C_BAND_LOOKBACK = max(1, int(os.getenv("EMA_5M_C_BAND_LOOKBACK", "3")))
    C_BOLL_MIN_WIDTH_PCT = float(os.getenv("EMA_5M_C_BOLL_MIN_WIDTH_PCT", "0.004"))

    MACD_FAST = max(2, int(os.getenv("EMA_5M_C_MACD_FAST", "12")))
    MACD_SLOW = max(MACD_FAST + 1, int(os.getenv("EMA_5M_C_MACD_SLOW", "26")))
    MACD_SIGNAL = max(2, int(os.getenv("EMA_5M_C_MACD_SIGNAL", "9")))

    KDJ_LEN = max(3, int(os.getenv("EMA_5M_C_KDJ_LEN", "9")))
    KDJ_SMOOTH_K = max(1, int(os.getenv("EMA_5M_C_KDJ_SMOOTH_K", "3")))
    KDJ_SMOOTH_D = max(1, int(os.getenv("EMA_5M_C_KDJ_SMOOTH_D", "3")))
    KDJ_ZONE_LOOKBACK = max(2, int(os.getenv("EMA_5M_C_KDJ_ZONE_LOOKBACK", "4")))
    KDJ_OS = float(os.getenv("EMA_5M_C_KDJ_OS", "20"))
    KDJ_OB = float(os.getenv("EMA_5M_C_KDJ_OB", "80"))

    def _prep5(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = super()._prep5(frame)
        c = d["close"].astype(float)

        # Bollinger Bands (20,2 by default).
        d["boll_mid"] = c.rolling(self.BOLL_LEN, min_periods=self.BOLL_LEN).mean()
        sigma = c.rolling(self.BOLL_LEN, min_periods=self.BOLL_LEN).std(ddof=0)
        d["boll_upper"] = d["boll_mid"] + self.BOLL_STD * sigma
        d["boll_lower"] = d["boll_mid"] - self.BOLL_STD * sigma
        d["boll_width_pct"] = (
            (d["boll_upper"] - d["boll_lower"])
            / d["boll_mid"].abs().clip(lower=1e-12)
        )

        # MACD (12,26,9 by default).
        macd_fast = c.ewm(span=self.MACD_FAST, adjust=False).mean()
        macd_slow = c.ewm(span=self.MACD_SLOW, adjust=False).mean()
        d["macd"] = macd_fast - macd_slow
        d["macd_signal"] = d["macd"].ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        d["macd_hist"] = d["macd"] - d["macd_signal"]

        # KDJ (9,3,3 by default). EWM smoothing gives stable K/D and an expressive J.
        low_n = d["low"].astype(float).rolling(self.KDJ_LEN, min_periods=self.KDJ_LEN).min()
        high_n = d["high"].astype(float).rolling(self.KDJ_LEN, min_periods=self.KDJ_LEN).max()
        span = (high_n - low_n).replace(0.0, 1e-12)
        rsv = ((c - low_n) / span * 100.0).clip(lower=0.0, upper=100.0)
        d["kdj_k"] = rsv.ewm(alpha=1.0 / self.KDJ_SMOOTH_K, adjust=False).mean()
        d["kdj_d"] = d["kdj_k"].ewm(alpha=1.0 / self.KDJ_SMOOTH_D, adjust=False).mean()
        d["kdj_j"] = 3.0 * d["kdj_k"] - 2.0 * d["kdj_d"]
        return d

    def _trigger_c(self, df5: pd.DataFrame, bias_side):
        """Return a ready/filtered Setup-C view, or None when no C pattern exists."""
        if not self.C_ENABLED or bias_side is None or len(df5) < 80:
            return None

        d = self._prep5(df5)
        r, p, p2 = d.iloc[-1], d.iloc[-2], d.iloc[-3]
        required = (
            r.boll_mid, r.boll_upper, r.boll_lower, r.boll_width_pct,
            r.macd, r.macd_signal, r.macd_hist,
            r.kdj_k, r.kdj_d,
        )
        if any(pd.isna(x) for x in required):
            return None

        adx, chop = self._quality_values_5m(d)
        quality_ok = adx >= self.ADX_MIN and chop <= self.CHOP_MAX
        width_ok = float(r.boll_width_pct) >= self.C_BOLL_MIN_WIDTH_PCT

        touch_window = d.iloc[-(self.C_BAND_LOOKBACK + 1):-1]
        kdj_window = d.iloc[-(self.KDJ_ZONE_LOOKBACK + 1):]

        lower_touch = bool(
            (touch_window["low"].astype(float) <= touch_window["boll_lower"].astype(float)).any()
        )
        upper_touch = bool(
            (touch_window["high"].astype(float) >= touch_window["boll_upper"].astype(float)).any()
        )

        close_now = float(r.close)
        close_prev = float(p.close)
        mid = float(r.boll_mid)
        lower = float(r.boll_lower)
        upper = float(r.boll_upper)

        # Re-entry must still be in the outer half of the band so Setup C does not chase.
        long_reentry = lower_touch and lower < close_now <= mid and close_now > close_prev
        short_reentry = upper_touch and mid <= close_now < upper and close_now < close_prev

        k_now, d_now = float(r.kdj_k), float(r.kdj_d)
        k_prev, d_prev = float(p.kdj_k), float(p.kdj_d)
        k_cross_up = k_prev <= d_prev and k_now > d_now
        k_cross_down = k_prev >= d_prev and k_now < d_now
        was_os = bool(
            ((kdj_window["kdj_k"] <= self.KDJ_OS) | (kdj_window["kdj_d"] <= self.KDJ_OS)).any()
        )
        was_ob = bool(
            ((kdj_window["kdj_k"] >= self.KDJ_OB) | (kdj_window["kdj_d"] >= self.KDJ_OB)).any()
        )

        hist_now = float(r.macd_hist)
        hist_prev = float(p.macd_hist)
        hist_prev2 = float(p2.macd_hist)
        macd_now = float(r.macd)
        macd_prev = float(p.macd)
        sig_now = float(r.macd_signal)
        sig_prev = float(p.macd_signal)

        macd_cross_up = macd_prev <= sig_prev and macd_now > sig_now
        macd_cross_down = macd_prev >= sig_prev and macd_now < sig_now
        macd_long = (
            hist_now > hist_prev
            and macd_now > macd_prev
            and (macd_cross_up or hist_now >= 0.0 or hist_prev > hist_prev2)
        )
        macd_short = (
            hist_now < hist_prev
            and macd_now < macd_prev
            and (macd_cross_down or hist_now <= 0.0 or hist_prev < hist_prev2)
        )

        long_shape = long_reentry and k_cross_up and was_os and macd_long
        short_shape = short_reentry and k_cross_down and was_ob and macd_short

        if bias_side == Side.LONG and long_shape:
            if quality_ok and width_ok:
                return TriggerView(
                    Side.LONG, True, adx, chop,
                    "C_BOLL_MACD_KDJ", "BOLL_MACD_KDJ_LONG",
                    f"5M READY LONG: C lower-band reentry + MACD improving + KDJ cross from OS "
                    f"| K={k_now:.1f} D={d_now:.1f} Hist={hist_now:.6g} BBW={float(r.boll_width_pct)*100:.2f}%",
                )
            return TriggerView(
                Side.LONG, False, adx, chop, "NONE", "NONE",
                f"5M C_FILTERED LONG | ADX={adx:.1f} CHOP={chop:.1f} "
                f"BBW={float(r.boll_width_pct)*100:.2f}%",
            )

        if bias_side == Side.SHORT and short_shape:
            if quality_ok and width_ok:
                return TriggerView(
                    Side.SHORT, True, adx, chop,
                    "C_BOLL_MACD_KDJ", "BOLL_MACD_KDJ_SHORT",
                    f"5M READY SHORT: C upper-band reentry + MACD weakening + KDJ cross from OB "
                    f"| K={k_now:.1f} D={d_now:.1f} Hist={hist_now:.6g} BBW={float(r.boll_width_pct)*100:.2f}%",
                )
            return TriggerView(
                Side.SHORT, False, adx, chop, "NONE", "NONE",
                f"5M C_FILTERED SHORT | ADX={adx:.1f} CHOP={chop:.1f} "
                f"BBW={float(r.boll_width_pct)*100:.2f}%",
            )

        return None

    def _trigger5(self, df5: pd.DataFrame, bias_side):
        # Preserve Setup A as first priority.
        base_view = super()._trigger5(df5, bias_side)
        if base_view.ready and base_view.setup == "A_EMA_CROSS":
            return base_view

        c_view = self._trigger_c(df5, bias_side)
        if c_view is not None and c_view.ready:
            return c_view

        # Setup B remains valid when C is absent or filtered.
        if base_view.ready:
            return base_view
        if c_view is not None:
            return c_view
        return base_view

    def entry_status(self, df4h, df1h, df15, df5):
        base_text = super().entry_status(df4h, df1h, df15, df5)
        return (
            base_text
            + f" | C=BOLL{self.BOLL_LEN},{self.BOLL_STD:g}+MACD{self.MACD_FAST}/{self.MACD_SLOW}/{self.MACD_SIGNAL}"
            + f"+KDJ{self.KDJ_LEN}/{self.KDJ_SMOOTH_K}/{self.KDJ_SMOOTH_D}"
            + f" OS<={self.KDJ_OS:.0f} OB>={self.KDJ_OB:.0f} BBW>={self.C_BOLL_MIN_WIDTH_PCT*100:.2f}%"
        )
