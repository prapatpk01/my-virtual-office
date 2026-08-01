"""
Chart Engine — renders a candlestick PNG (mplfinance) with EMA overlays,
recent swing support/resistance, and entry/SL/TP markers, for Telegram alerts.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
import pandas as pd

import indicators as ind

logger = logging.getLogger("chart_engine")


def _mplfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean frame using the canonical mplfinance column names.

    Exchange frames in this project use lowercase OHLCV names. mplfinance's
    default column resolver expects Open/High/Low/Close/Volume, so passing the
    raw frame can silently make chart creation fall back to a text alert.
    """
    required = ("open", "high", "low", "close")
    if df is None or df.empty or any(col not in df.columns for col in required):
        return pd.DataFrame()

    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df.loc[:, cols].copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    out = out.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def build_entry_chart(symbol: str, df: pd.DataFrame, direction: str,
                      entry: float, sl: float, tp1: float, tp2: float,
                      out_dir: Optional[str] = None,
                      ema_fast_len: int = 10, ema_slow_len: int = 20,
                      tf_label: str = "5M") -> Optional[str]:
    """Save a candlestick entry chart and return its file path.

    The input dataframe may use lowercase exchange-style OHLCV columns. The
    function normalizes those names before calling mplfinance.
    """
    try:
        raw = df.tail(120).copy() if df is not None else pd.DataFrame()
        if raw.empty:
            return None

        closes = pd.to_numeric(raw["close"], errors="coerce")
        ema_fast = ind.ema(closes, ema_fast_len)
        ema_slow = ind.ema(closes, ema_slow_len)
        swing_high, swing_low = ind.recent_swing_levels(raw["high"], raw["low"], 3, 3)

        plot_df = _mplfinance_frame(raw)
        if plot_df.empty:
            logger.warning("[CHART] no valid OHLC rows for %s", symbol)
            return None

        # Align overlays to the cleaned/sorted plotting index.
        ema_fast = ema_fast.reindex(plot_df.index)
        ema_slow = ema_slow.reindex(plot_df.index)
        addplots = [
            mpf.make_addplot(ema_fast, color="#00d4ff", width=1.1),
            mpf.make_addplot(ema_slow, color="#ff5d8f", width=1.1),
        ]

        levels = [float(entry), float(sl), float(tp1), float(tp2)]
        colors = ["#ffffff", "#ff4d4f", "#3fb950", "#3fb950"]
        linestyles = ["--", "-", "-", "-"]
        linewidths = [1.0, 1.2, 1.0, 1.2]

        if pd.notna(swing_high):
            levels.append(float(swing_high))
            colors.append("#8b949e")
            linestyles.append(":")
            linewidths.append(0.8)
        if pd.notna(swing_low):
            levels.append(float(swing_low))
            colors.append("#8b949e")
            linestyles.append(":")
            linewidths.append(0.8)

        hlines = dict(
            hlines=levels,
            colors=colors,
            linestyle=linestyles,
            linewidths=linewidths,
        )

        mc = mpf.make_marketcolors(up="#3fb950", down="#f85149", edge="inherit",
                                   wick="inherit", volume="in")
        style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc,
                                   gridcolor="#30363d", facecolor="#0d1117",
                                   figcolor="#0d1117", edgecolor="#30363d")

        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{symbol.replace('/', '_').replace(':', '_')}_{int(time.time())}.png"
        path = os.path.join(out_dir, fname)

        title = (
            f"{symbol}  {direction}  {tf_label} EMA{ema_fast_len}/{ema_slow_len}  "
            f"entry={entry:.4f}"
        )
        mpf.plot(
            plot_df,
            type="candle",
            style=style,
            addplot=addplots,
            hlines=hlines,
            volume="Volume" in plot_df.columns,
            title=title,
            savefig=dict(fname=path, dpi=130, bbox_inches="tight"),
        )
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            logger.warning("[CHART] output file missing/empty for %s", symbol)
            return None
        return path
    except Exception as e:
        logger.warning("[CHART] build_entry_chart failed for %s: %s", symbol, e, exc_info=True)
        return None
