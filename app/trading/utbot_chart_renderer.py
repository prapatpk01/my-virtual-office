"""Dedicated Telegram chart for the XAU-only UT Bot v2 strategy."""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger("utbot_chart_renderer")

try:
    import matplotlib
    matplotlib.use("Agg")
    import mplfinance as mpf
    import numpy as np
    import pandas as pd
    from .strategies.utbot_xau_strategy import UTBotXAUStrategy
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def _dataframe(candles: list):
    rows = [
        {
            "Date": pd.to_datetime(UTBotXAUStrategy._timestamp_ms(c.timestamp), unit="ms"),
            "Open": float(c.open),
            "High": float(c.high),
            "Low": float(c.low),
            "Close": float(c.close),
            "Volume": float(c.volume),
        }
        for c in candles
    ]
    return pd.DataFrame(rows).set_index("Date")


def render_utbot_entry_chart(
    candles: list,
    symbol: str,
    direction: str,
    entry: float,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    strategy: str = "",
    macro_bias: str = "",
    lookback: int = 100,
    out_dir: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """Render candles + the exact recursive UT ATR trailing-stop series."""
    if not _AVAILABLE or not candles:
        return None

    try:
        multiplier = float(os.getenv("UTBOT_MULTIPLIER", "1.0"))
        atr_period = int(os.getenv("UTBOT_ATR_PERIOD", "10"))
        timeframe = os.getenv("UTBOT_TIMEFRAME", "15m").strip().lower() or "15m"

        calc = UTBotXAUStrategy(
            symbol=UTBotXAUStrategy.LOCKED_SYMBOL,
            multiplier=multiplier,
            atr_period=atr_period,
            timeframe=timeframe,
            use_date_filter=False,
        )
        closed = calc._closed_candles(candles)
        if len(closed) < atr_period + 3:
            return None
        closed = closed[-max(30, int(lookback)):]
        values = calc._ut_series(closed)
        df = _dataframe(closed)
        tsl = pd.Series(values["tsl"], index=df.index, dtype=float)

        addplots = [
            mpf.make_addplot(tsl, panel=0, width=1.7, color="#f59e0b"),
        ]

        # Mark the confirmed cross that caused this entry when it is the latest bar.
        marker = pd.Series(np.nan, index=df.index, dtype=float)
        marker.iloc[-1] = (
            float(df["Low"].iloc[-1]) * 0.998
            if direction == "long"
            else float(df["High"].iloc[-1]) * 1.002
        )
        addplots.append(
            mpf.make_addplot(
                marker,
                panel=0,
                type="scatter",
                marker="^" if direction == "long" else "v",
                markersize=110,
                color="#facc15",
            )
        )

        hlines = [float(entry)]
        hcolors = ["#e5e7eb"]
        hstyles = ["solid"]
        hwidths = [1.2]

        market_colors = mpf.make_marketcolors(
            up="#22c55e",
            down="#ef4444",
            edge="inherit",
            wick="inherit",
            volume="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=market_colors,
            gridstyle="",
            facecolor="#111827",
            figcolor="#111827",
            edgecolor="#374151",
            rc={
                "font.family": "DejaVu Sans",
                "axes.labelcolor": "#e5e7eb",
                "xtick.color": "#9ca3af",
                "ytick.color": "#9ca3af",
                "text.color": "#e5e7eb",
            },
        )

        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"utbot_xau_{direction}_{int(time.time())}.png",
        )

        title = (
            f"XAU  [{timeframe}]  {direction.upper()}  |  "
            f"UT Bot v2 ATR({atr_period}) x{multiplier:g}"
        )
        fig, axes = mpf.plot(
            df,
            type="candle",
            style=style,
            addplot=addplots,
            hlines=dict(
                hlines=hlines,
                colors=hcolors,
                linestyle=hstyles,
                linewidths=hwidths,
            ),
            volume=True,
            panel_ratios=(4, 1),
            title=title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(10, 7.2),
            returnfig=True,
        )

        atr_now = float(values["atr"][-1])
        tsl_now = float(values["tsl"][-1])
        close_now = float(values["source"][-1])
        signal_text = "BUY cross" if direction == "long" else "SELL cross"
        info = [
            f"Entry {entry:,.4f}",
            f"{signal_text} — confirmed {timeframe} close",
            f"Close {close_now:,.4f}",
            f"ATR({atr_period}) {atr_now:,.4f}  x{multiplier:g}",
            f"ATR Trailing Stop {tsl_now:,.4f}",
            "Exit: opposite UT cross -> reverse",
            "Fixed SL/TP: none",
        ]
        axes[0].text(
            0.01,
            0.99,
            "\n".join(info),
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            color="#e5e7eb",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#1f2937",
                edgecolor="#374151",
                alpha=0.88,
            ),
        )

        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.warning("UT Bot chart render failed for %s: %s", symbol, exc)
        return None
