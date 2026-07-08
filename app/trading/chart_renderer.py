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
) -> Optional[str]:
    """
    Render a candlestick chart with EMA20/EMA50, S/R lines, and entry/SL/TP
    markers. Returns the path to the saved PNG, or None if charting is
    unavailable or fails (caller should degrade to text-only notification).
    """
    if not _CHARTS_AVAILABLE:
        logger.debug("mplfinance/matplotlib not installed — skipping chart render")
        return None
    if not candles or len(candles) < 20:
        return None

    try:
        df = _to_dataframe(candles[-lookback:])

        ema20 = df["Close"].ewm(span=20, adjust=False).mean()
        ema50 = df["Close"].ewm(span=50, adjust=False).mean()
        res_level, sup_level = _swing_levels(df, lookback=40)

        addplots = [
            mpf.make_addplot(ema20, color="#3b82f6", width=1.1),
            mpf.make_addplot(ema50, color="#f59e0b", width=1.1),
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

        dir_label = "LONG (+)" if direction == "long" else "SHORT (-)"
        title = f"{symbol}  {dir_label}  |  {strategy or ''}".strip()

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
            title=title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(10, 7),
            returnfig=True,
        )

        # Legend: line meaning + exact price levels (mplfinance hlines has no
        # native legend, so annotate the price axis directly).
        ax = axes[0]
        legend_lines = [
            f"Entry {entry:,.2f}",
            f"EMA20 / EMA50",
        ]
        if sl:
            legend_lines.append(f"SL {sl:,.2f}")
        if tp:
            legend_lines.append(f"TP {tp:,.2f}")
        if macro_bias:
            legend_lines.append(f"Macro: {macro_bias}")
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
