"""
Entry chart renderer — draws a candlestick chart with EMA20/EMA50,
recent support/resistance, and entry/SL/TP markers, then saves it
as a PNG for sending to Telegram.

Uses mplfinance for the candlestick plot (matplotlib under the hood).
Falls back gracefully (returns None) if the plotting libraries are
unavailable, so the trading loop never breaks because of a chart error.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger("chart_renderer")

try:
    import matplotlib
    matplotlib.use("Agg")  # headless — no display server on the server/container
    import mplfinance as mpf
    import pandas as pd
    from .strategies.base import BaseStrategy
    _CHARTS_AVAILABLE = True
except ImportError:
    _CHARTS_AVAILABLE = False


def charts_available() -> bool:
    return _CHARTS_AVAILABLE


def _to_dataframe(candles: list):
    rows = [{
        "Date":   pd.to_datetime(c.timestamp, unit="ms"),
        "Open":   float(c.open),
        "High":   float(c.high),
        "Low":    float(c.low),
        "Close":  float(c.close),
        "Volume": float(c.volume),
    } for c in candles]
    df = pd.DataFrame(rows).set_index("Date")
    return df


def _swing_levels(df, lookback: int = 40) -> tuple[float, float]:
    """Recent swing high / low used as support-resistance reference lines."""
    window = df.tail(min(lookback, len(df)))
    return float(window["High"].max()), float(window["Low"].min())


def render_entry_chart(
    candles: list,
    symbol: str,
    direction: str,       # "long" | "short"
    entry: float,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    strategy: str = "",
    macro_bias: str = "",
    lookback: int = 100,
    out_dir: Optional[str] = None,
    ma_type: str = "ema",           # "ema" | "hma" — matches the strategy that fired
    ema_fast: int = 20,
    ema_slow: int = 50,
    sma_period: Optional[int] = None,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    dir_label: Optional[str] = None,        # overrides the default "LONG (+)"/"SHORT (-)" title text
    indicator_status: Optional[str] = None,  # e.g. "SMA30=up  MACD=down  EMA5/10=no_cross" — shown in the legend box
    entry_label: str = "Entry",             # e.g. "Price" when there's no actual entry (status-only chart)
) -> Optional[str]:
    """
    Render a candlestick chart using the SAME moving-average/MACD periods
    the firing strategy actually trades on (defaults are ai_expert's
    EMA20/EMA50 — callers should pass their own strategy's periods, e.g.
    trend_confirm's EMA5/EMA10+SMA30, ema_sma's EMA12/EMA26+SMA50, or
    hma_macd_roc's HMA10/HMA20). S/R lines and entry/SL/TP markers as
    before. Returns the path to the saved PNG, or None if charting is
    unavailable or fails (caller should degrade to text-only notification).
    """
    if not _CHARTS_AVAILABLE:
        logger.debug("mplfinance/matplotlib not installed — skipping chart render")
        return None
    if not candles or len(candles) < 20:
        return None

    try:
        df = _to_dataframe(candles[-lookback:])
        closes = df["Close"].tolist()

        ma_fn = BaseStrategy.hma if ma_type == "hma" else BaseStrategy.ema
        ma1 = pd.Series(ma_fn(closes, ema_fast), index=df.index)
        ma2 = pd.Series(ma_fn(closes, ema_slow), index=df.index)
        ma_label = "HMA" if ma_type == "hma" else "EMA"
        res_level, sup_level = _swing_levels(df, lookback=40)

        addplots = [
            mpf.make_addplot(ma1, color="#3b82f6", width=1.1),
            mpf.make_addplot(ma2, color="#f59e0b", width=1.1),
        ]
        if sma_period:
            sma_line = pd.Series(BaseStrategy.sma(closes, sma_period), index=df.index)
            addplots.append(mpf.make_addplot(sma_line, color="#a855f7", width=1.1, linestyle="dashed"))

        macd_line_arr, macd_signal_arr, macd_hist_arr = BaseStrategy.macd(
            closes, macd_fast, macd_slow, macd_signal_period,
        )
        macd_line = pd.Series(macd_line_arr, index=df.index)
        macd_sig  = pd.Series(macd_signal_arr, index=df.index)
        macd_hist = pd.Series(macd_hist_arr, index=df.index)
        hist_colors = ["#22c55e" if (v == v and v >= 0) else "#ef4444" for v in macd_hist]

        addplots += [
            mpf.make_addplot(macd_hist, type="bar", panel=2, color=hist_colors,
                             width=0.7, ylabel="MACD"),
            mpf.make_addplot(macd_line, panel=2, color="#38bdf8", width=1.0),
            mpf.make_addplot(macd_sig, panel=2, color="#f97316", width=1.0),
        ]

        hlines_prices  = []
        hlines_colors  = []
        hlines_styles  = []
        hlines_widths  = []

        # Entry
        hlines_prices.append(entry)
        hlines_colors.append("#e5e7eb")
        hlines_styles.append("solid")
        hlines_widths.append(1.3)

        if sl:
            hlines_prices.append(sl)
            hlines_colors.append("#ef4444")
            hlines_styles.append("dashed")
            hlines_widths.append(1.3)
        if tp:
            hlines_prices.append(tp)
            hlines_colors.append("#22c55e")
            hlines_styles.append("dashed")
            hlines_widths.append(1.3)

        # Support / resistance (thin dotted reference lines)
        hlines_prices += [res_level, sup_level]
        hlines_colors += ["#9ca3af", "#9ca3af"]
        hlines_styles += ["dotted", "dotted"]
        hlines_widths += [0.8, 0.8]

        resolved_dir_label = dir_label or ("LONG (+)" if direction == "long" else "SHORT (-)")
        title = f"{symbol}  {resolved_dir_label}  |  {strategy or ''}".strip()

        mc = mpf.make_marketcolors(
            up="#22c55e", down="#ef4444",
            edge="inherit", wick="inherit", volume="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            gridstyle="",
            facecolor="#111827",
            figcolor="#111827",
            edgecolor="#374151",
            rc={"axes.labelcolor": "#e5e7eb", "xtick.color": "#9ca3af",
                "ytick.color": "#9ca3af", "text.color": "#e5e7eb"},
        )

        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"chart_{symbol.replace('/', '').replace(':', '')}_{int(time.time())}.png"
        out_path = os.path.join(out_dir, fname)

        fig, axes = mpf.plot(
            df,
            type="candle",
            style=style,
            addplot=addplots,
            hlines=dict(hlines=hlines_prices, colors=hlines_colors,
                        linestyle=hlines_styles, linewidths=hlines_widths),
            volume=True,
            panel_ratios=(3, 1, 1.2),
            title=title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(10, 8.5),
            returnfig=True,
        )

        # Legend: line meaning + exact price levels (mplfinance hlines has no
        # native legend, so annotate the price axis directly).
        ax = axes[0]
        legend_lines = [
            f"{entry_label} {entry:,.2f}",
            f"{ma_label}{ema_fast} / {ma_label}{ema_slow}" + (f" / SMA{sma_period}" if sma_period else ""),
            f"MACD {macd_fast}/{macd_slow}/{macd_signal_period}",
        ]
        if sl:
            legend_lines.append(f"SL {sl:,.2f}")
        if tp:
            legend_lines.append(f"TP {tp:,.2f}")
        if macro_bias:
            legend_lines.append(f"Macro: {macro_bias}")
        if indicator_status:
            legend_lines.append(indicator_status)
        ax.text(
            0.01, 0.99, "\n".join(legend_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, color="#e5e7eb",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#1f2937", edgecolor="#374151", alpha=0.85),
        )

        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return out_path
    except Exception as e:
        logger.warning("Chart render failed for %s: %s", symbol, e)
        return None
