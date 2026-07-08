"""
Chart Engine — renders a candlestick PNG (mplfinance) with EMA/HMA overlays,
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


def build_entry_chart(symbol: str, df_30m: pd.DataFrame, direction: str,
                      entry: float, sl: float, tp1: float, tp2: float,
                      out_dir: Optional[str] = None) -> Optional[str]:
    """Save a candlestick chart with overlays; return the file path, or None on failure."""
    try:
        plot_df = df_30m.tail(120).copy()
        if plot_df.empty:
            return None

        closes = plot_df["close"]
        ema15 = ind.ema(closes, 15)
        hma10 = ind.hma(closes, 10)
        hma20 = ind.hma(closes, 20)
        swing_high, swing_low = ind.recent_swing_levels(plot_df["high"], plot_df["low"], 3, 3)

        addplots = [
            mpf.make_addplot(ema15, color="#f0b90b", width=1.1),
            mpf.make_addplot(hma10, color="#00d4ff", width=1.0),
            mpf.make_addplot(hma20, color="#ff5d8f", width=1.0),
        ]

        hlines = dict(
            hlines=[entry, sl, tp1, tp2],
            colors=["#ffffff", "#ff4d4f", "#3fb950", "#3fb950"],
            linestyle=["--", "-", "-", "-"],
            linewidths=[1.0, 1.2, 1.0, 1.2],
        )
        if swing_high == swing_high:  # not NaN
            hlines["hlines"].append(swing_high)
            hlines["colors"].append("#8b949e")
            hlines["linestyle"].append(":")
            hlines["linewidths"].append(0.8)
        if swing_low == swing_low:
            hlines["hlines"].append(swing_low)
            hlines["colors"].append("#8b949e")
            hlines["linestyle"].append(":")
            hlines["linewidths"].append(0.8)

        mc = mpf.make_marketcolors(up="#3fb950", down="#f85149", edge="inherit",
                                   wick="inherit", volume="in")
        style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc,
                                   gridcolor="#30363d", facecolor="#0d1117",
                                   figcolor="#0d1117", edgecolor="#30363d")

        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{symbol.replace('/', '_').replace(':', '_')}_{int(time.time())}.png"
        path = os.path.join(out_dir, fname)

        title = f"{symbol}  {direction}  entry={entry:.4f}"
        mpf.plot(plot_df, type="candle", style=style, addplot=addplots, hlines=hlines,
                 volume=True, title=title, savefig=dict(fname=path, dpi=130, bbox_inches="tight"))
        return path
    except Exception as e:
        logger.warning("[CHART] build_entry_chart failed for %s: %s", symbol, e)
        return None
