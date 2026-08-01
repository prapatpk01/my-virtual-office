"""HMA alert chart renderer.

This local module intentionally shadows the shared chart engine when HMA mode
runs from ``hma_bot/``. It accepts the exchange's lowercase OHLCV columns and
renders the actual 5M execution chart with EMA8/13 plus Entry, Structure SL,
Final TP, and recent confirmed swing levels.
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
import numpy as np
import pandas as pd

logger = logging.getLogger("hma_chart_engine")


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ("open", "high", "low", "close")
    if df is None or df.empty or any(c not in df.columns for c in required):
        return pd.DataFrame()

    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df.loc[:, cols].tail(120).copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })


def _recent_confirmed_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[float, float]:
    if len(df) < left + right + 3:
        return float("nan"), float("nan")
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    swing_high = float("nan")
    swing_low = float("nan")
    for i in range(left, len(df) - right):
        hw = highs[i-left:i+right+1]
        lw = lows[i-left:i+right+1]
        if np.isfinite(highs[i]) and highs[i] >= np.nanmax(hw):
            swing_high = float(highs[i])
        if np.isfinite(lows[i]) and lows[i] <= np.nanmin(lw):
            swing_low = float(lows[i])
    return swing_high, swing_low


def build_entry_chart(
    symbol: str,
    df: pd.DataFrame,
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    out_dir: Optional[str] = None,
    ema_fast_len: int = 8,
    ema_slow_len: int = 13,
    tf_label: str = "5M",
) -> Optional[str]:
    """Render a Telegram-ready PNG and return its local path."""
    try:
        plot_df = _clean_frame(df)
        if len(plot_df) < max(ema_slow_len + 2, 20):
            logger.warning("[HMA CHART] insufficient candles for %s: %d", symbol, len(plot_df))
            return None

        ema_fast = plot_df["Close"].ewm(span=max(1, int(ema_fast_len)), adjust=False).mean()
        ema_slow = plot_df["Close"].ewm(span=max(1, int(ema_slow_len)), adjust=False).mean()
        addplots = [
            mpf.make_addplot(ema_fast, color="#00d4ff", width=1.1),
            mpf.make_addplot(ema_slow, color="#ffb000", width=1.1),
        ]

        levels = [float(entry), float(sl), float(tp1)]
        colors = ["#ffffff", "#ff4d4f", "#3fb950"]
        linestyles = ["--", "-", "-"]
        linewidths = [1.0, 1.2, 1.2]
        if float(tp2) != float(tp1):
            levels.append(float(tp2))
            colors.append("#3fb950")
            linestyles.append("-")
            linewidths.append(1.0)

        swing_high, swing_low = _recent_confirmed_swings(plot_df)
        for level in (swing_high, swing_low):
            if np.isfinite(level):
                levels.append(float(level))
                colors.append("#8b949e")
                linestyles.append(":")
                linewidths.append(0.8)

        hlines = dict(
            hlines=levels,
            colors=colors,
            linestyle=linestyles,
            linewidths=linewidths,
        )
        market_colors = mpf.make_marketcolors(
            up="#3fb950", down="#f85149", edge="inherit", wick="inherit", volume="in"
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=market_colors,
            gridcolor="#30363d",
            facecolor="#0d1117",
            figcolor="#0d1117",
            edgecolor="#30363d",
        )

        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        path = os.path.join(out_dir, f"hma_{safe_symbol}_{int(time.time() * 1000)}.png")
        title = (
            f"{symbol}  {direction.upper()}  {tf_label} EMA{ema_fast_len}/{ema_slow_len}  "
            f"entry={float(entry):.6g}"
        )
        mpf.plot(
            plot_df,
            type="candle",
            style=style,
            addplot=addplots,
            hlines=hlines,
            volume="Volume" in plot_df.columns,
            title=title,
            warn_too_much_data=1000,
            savefig=dict(fname=path, dpi=130, bbox_inches="tight"),
        )
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            logger.warning("[HMA CHART] output missing/empty for %s", symbol)
            return None
        return path
    except Exception as exc:
        logger.warning("[HMA CHART] failed for %s: %s", symbol, exc, exc_info=True)
        return None
