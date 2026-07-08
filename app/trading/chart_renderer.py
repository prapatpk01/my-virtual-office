"""
Entry-alert chart renderer — candlestick chart PNG for Telegram photos.

Draws the last ~96 15m candles (1 day) with:
  - EMA5 / EMA20 / EMA50 overlays
  - Entry price marker + horizontal line
  - SL (red) / T1 / T2 (green) horizontal levels with labels
  - Recent swing-high resistance / swing-low support (dashed)
  - Volume panel

Pure function of (candles, trade levels) — no bot/exchange access. Renders
to a PNG file under /tmp and returns the path; caller sends + deletes it.
Uses the Agg backend explicitly so it is safe from a worker thread (the
execution callback runs in run_in_executor, not the main thread).
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger("chart_renderer")

# Import guard: chart rendering is a nice-to-have — the bot must run fine
# (text-only alerts) when mplfinance/matplotlib aren't installed.
try:
    import matplotlib
    matplotlib.use("Agg")                     # thread-safe, headless
    import matplotlib.pyplot as plt           # noqa: E402
    import mplfinance as mpf                  # noqa: E402
    import pandas as pd                       # noqa: E402
    CHARTS_AVAILABLE = True
except Exception as _e:                       # pragma: no cover
    CHARTS_AVAILABLE = False
    logger.warning("chart rendering disabled (import failed: %s)", _e)


def _swing_levels(df, lookback: int = 60, wing: int = 3) -> tuple:
    """Most recent swing-high (resistance) and swing-low (support):
    a bar whose high/low is the extreme of `wing` bars on each side."""
    highs, lows = df["High"], df["Low"]
    res = sup = None
    n = len(df)
    start = max(wing, n - lookback)
    for i in range(n - wing - 1, start - 1, -1):
        window_h = highs.iloc[i - wing: i + wing + 1]
        if highs.iloc[i] == window_h.max() and res is None:
            res = float(highs.iloc[i])
        window_l = lows.iloc[i - wing: i + wing + 1]
        if lows.iloc[i] == window_l.min() and sup is None:
            sup = float(lows.iloc[i])
        if res is not None and sup is not None:
            break
    return sup, res


def render_entry_chart(candles: List, symbol: str, direction: str,
                       entry: float, sl: float, tp1: float, tp2: float,
                       strategy: str = "", regime: str = "",
                       bars: int = 96) -> Optional[str]:
    """
    Render an entry chart PNG. `candles` is the connector's OHLCV list
    (objects with .timestamp/.open/.high/.low/.close/.volume, ms epoch).
    Returns the PNG path, or None when rendering is unavailable/fails.
    """
    if not CHARTS_AVAILABLE:
        return None
    try:
        rows = candles[-bars:]
        if len(rows) < 30:
            return None

        df = pd.DataFrame({
            "Date":   [pd.Timestamp(int(c.timestamp), unit="ms", tz="UTC") for c in rows],
            "Open":   [float(c.open) for c in rows],
            "High":   [float(c.high) for c in rows],
            "Low":    [float(c.low) for c in rows],
            "Close":  [float(c.close) for c in rows],
            "Volume": [float(c.volume) for c in rows],
        }).set_index("Date")

        ema5  = df["Close"].ewm(span=5,  adjust=False).mean()
        ema20 = df["Close"].ewm(span=20, adjust=False).mean()
        ema50 = df["Close"].ewm(span=50, adjust=False).mean()
        sup, res = _swing_levels(df)

        addplots = [
            mpf.make_addplot(ema5,  color="#f0b90b", width=1.0),
            mpf.make_addplot(ema20, color="#2962ff", width=1.2),
            mpf.make_addplot(ema50, color="#9c27b0", width=1.2),
        ]

        # Horizontal levels: entry / SL / T1 / T2 (+ optional S/R)
        hlines, colors, styles, widths = [], [], [], []

        def _add(price, color, style, width=1.2):
            if price and price > 0:
                hlines.append(float(price))
                colors.append(color)
                styles.append(style)
                widths.append(width)

        _add(entry, "#ffffff", "-",  1.4)
        _add(sl,    "#f6465d", "--", 1.4)
        _add(tp1,   "#0ecb81", "--", 1.2)
        _add(tp2,   "#0ecb81", "-",  1.4)
        _add(sup,   "#787b86", ":",  1.0)
        _add(res,   "#787b86", ":",  1.0)

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mpf.make_marketcolors(
                up="#0ecb81", down="#f6465d",
                edge="inherit", wick="inherit", volume="in",
            ),
            gridcolor="#2a2e39", gridstyle=":",
            facecolor="#131722", edgecolor="#131722", figcolor="#131722",
            rc={"axes.labelcolor": "#d1d4dc", "xtick.color": "#787b86",
                "ytick.color": "#787b86", "font.size": 9},
        )

        arrow = "▲" if direction == "LONG" else "▼"
        title = f"\n{symbol}  {direction} {arrow}  15m" \
                + (f"  |  {strategy}" if strategy else "") \
                + (f"  |  {regime}" if regime else "")

        # mplfinance sizes the y-axis to the CANDLES only — SL/T1/T2 levels
        # beyond the visible price range (common right after entry: targets
        # sit 1-2.4% away) would be silently clipped off-chart. Expand the
        # axis to cover every level.
        level_prices = [p for p in (entry, sl, tp1, tp2) if p and p > 0]
        y_lo = min(float(df["Low"].min()),  *level_prices)
        y_hi = max(float(df["High"].max()), *level_prices)
        y_pad = (y_hi - y_lo) * 0.04 or y_hi * 0.001

        fig, axes = mpf.plot(
            df, type="candle", style=style, addplot=addplots,
            volume=True, returnfig=True, figsize=(12, 7),
            ylim=(y_lo - y_pad, y_hi + y_pad),
            hlines=dict(hlines=hlines, colors=colors,
                        linestyle=styles, linewidths=widths, alpha=0.9),
            title=dict(title=title, color="#d1d4dc"),
            tight_layout=True, xrotation=0,
        )

        # Right-edge labels for each level (plotted in axes coordinates)
        ax = axes[0]
        x_right = len(df) - 1
        label_map = [
            (entry, f" ENTRY {entry:,.4f}".rstrip("0").rstrip("."), "#ffffff"),
            (sl,    f" SL {sl:,.4f}".rstrip("0").rstrip("."),       "#f6465d"),
            (tp1,   f" T1 {tp1:,.4f}".rstrip("0").rstrip("."),      "#0ecb81"),
            (tp2,   f" T2 {tp2:,.4f}".rstrip("0").rstrip("."),      "#0ecb81"),
        ]
        if sup:
            label_map.append((sup, " S", "#787b86"))
        if res:
            label_map.append((res, " R", "#787b86"))
        for price, text, color in label_map:
            if price and price > 0:
                ax.annotate(text, xy=(x_right, price),
                            xytext=(4, 0), textcoords="offset points",
                            color=color, fontsize=8, va="center",
                            fontweight="bold")

        # Entry marker on the last candle
        marker = "^" if direction == "LONG" else "v"
        ax.scatter([x_right], [entry], marker=marker, s=140,
                   color="#f0b90b", zorder=5, edgecolors="#131722")

        fd, path = tempfile.mkstemp(prefix=f"entry_{symbol.split('/')[0]}_",
                                    suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning("chart render failed (falling back to text alert): %s", e)
        try:
            plt.close("all")
        except Exception:
            pass
        return None
