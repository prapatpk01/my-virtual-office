"""
Entry chart renderer for Telegram order notifications.

The upper panels always show price, the strategy moving averages, volume and
entry-risk levels. The lower panel is selected by the entry trigger:
- EMA entry: MACD panel (legacy chart behaviour).
- WT entry: WaveTrend WT1/WT2 panel with the production -42/+45 extreme levels
  and a marker on the fresh cross that opened the trade.

Chart failures are non-fatal: the caller falls back to a text-only Telegram
notification, so rendering can never interrupt live order management.
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
    matplotlib.use("Agg")  # headless server/container
    import mplfinance as mpf
    import numpy as np
    import pandas as pd
    from .strategies.base import BaseStrategy
    _CHARTS_AVAILABLE = True
except ImportError:
    _CHARTS_AVAILABLE = False


def charts_available() -> bool:
    return _CHARTS_AVAILABLE


def _to_dataframe(candles: list):
    rows = [{
        "Date": pd.to_datetime(c.timestamp, unit="ms"),
        "Open": float(c.open),
        "High": float(c.high),
        "Low": float(c.low),
        "Close": float(c.close),
        "Volume": float(c.volume),
    } for c in candles]
    return pd.DataFrame(rows).set_index("Date")


def _swing_levels(df, lookback: int = 40) -> tuple[float, float]:
    """Recent swing high/low used as support-resistance references."""
    window = df.tail(min(lookback, len(df)))
    return float(window["High"].max()), float(window["Low"].min())


def _ema_finite(values, period: int):
    """EMA seeded after ``period`` finite samples, matching live WT logic."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    period = max(1, int(period))
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size < period:
        return out

    seed_idx = int(finite_idx[period - 1])
    out[seed_idx] = float(np.mean(arr[finite_idx[:period]]))
    alpha = 2.0 / (period + 1.0)
    prev = float(out[seed_idx])
    for i in range(seed_idx + 1, len(arr)):
        if np.isfinite(arr[i]):
            prev = alpha * float(arr[i]) + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _sma_finite(values, period: int):
    """SMA requiring a complete finite rolling window."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    period = max(1, int(period))
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        if np.all(np.isfinite(window)):
            out[i] = float(np.mean(window))
    return out


def _wave_trend_series(
    df,
    channel_length: int = 10,
    average_length: int = 21,
    signal_length: int = 4,
):
    """Calculate WT1/WT2 exactly like TrendConfirmWTFixedStrategy."""
    open_values = df["Open"].to_numpy(dtype=float)
    high_values = df["High"].to_numpy(dtype=float)
    low_values = df["Low"].to_numpy(dtype=float)
    close_values = df["Close"].to_numpy(dtype=float)

    # The strategy uses Heikin-Ashi close in its HLC3-style source.
    ha_close = (open_values + high_values + low_values + close_values) / 4.0
    source = (high_values + low_values + ha_close) / 3.0

    esa = _ema_finite(source, channel_length)
    deviation = _ema_finite(np.abs(source - esa), channel_length)
    ci = np.full(source.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(source)
        & np.isfinite(esa)
        & np.isfinite(deviation)
        & (deviation > 1e-12)
    )
    ci[valid] = (source[valid] - esa[valid]) / (0.015 * deviation[valid])

    wt1 = _ema_finite(ci, average_length)
    wt2 = _sma_finite(wt1, signal_length)
    return (
        pd.Series(wt1, index=df.index, dtype=float),
        pd.Series(wt2, index=df.index, dtype=float),
    )


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
    ma_type: str = "ema",
    ema_fast: int = 20,
    ema_slow: int = 50,
    extra_ema: Optional[int] = None,
    sma_period: Optional[int] = None,
    tf_label: Optional[str] = None,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    dir_label: Optional[str] = None,
    indicator_status: Optional[str] = None,
    entry_label: str = "Entry",
    # Trigger-aware chart options.
    lower_panel: str = "macd",             # "macd" | "wt"
    entry_trigger: Optional[str] = None,    # "EMA8/13 Cross" | "WT Cross"
    wt_channel_length: int = 10,
    wt_average_length: int = 21,
    wt_signal_length: int = 4,
    wt_oversold: float = -42.0,
    wt_overbought: float = 45.0,
    t1_pct: Optional[float] = None,         # decimal, e.g. 0.006
    t1_trim_pct: float = 0.40,
    t1_lock_pct: float = 0.003,
) -> Optional[str]:
    """Render the entry chart and return its PNG path, or ``None`` on failure."""
    if not _CHARTS_AVAILABLE:
        logger.debug("mplfinance/matplotlib unavailable — skipping chart render")
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
        if extra_ema:
            ema3 = pd.Series(BaseStrategy.ema(closes, extra_ema), index=df.index)
            addplots.append(mpf.make_addplot(ema3, color="#a855f7", width=1.1))
        if sma_period:
            sma_line = pd.Series(BaseStrategy.sma(closes, sma_period), index=df.index)
            addplots.append(
                mpf.make_addplot(
                    sma_line, color="#a855f7", width=1.1, linestyle="dashed"
                )
            )

        use_wt_panel = str(lower_panel).strip().lower() in {"wt", "wavetrend"}
        wt1 = wt2 = None
        wt_cross_label = None

        if use_wt_panel:
            wt1, wt2 = _wave_trend_series(
                df,
                channel_length=wt_channel_length,
                average_length=wt_average_length,
                signal_length=wt_signal_length,
            )
            addplots += [
                mpf.make_addplot(
                    wt1, panel=2, color="#38bdf8", width=1.25, ylabel="WaveTrend"
                ),
                mpf.make_addplot(wt2, panel=2, color="#f97316", width=1.15),
                mpf.make_addplot(
                    pd.Series(float(wt_oversold), index=df.index),
                    panel=2, color="#22c55e", width=0.8, linestyle="dashed",
                ),
                mpf.make_addplot(
                    pd.Series(float(wt_overbought), index=df.index),
                    panel=2, color="#ef4444", width=0.8, linestyle="dashed",
                ),
                mpf.make_addplot(
                    pd.Series(0.0, index=df.index),
                    panel=2, color="#6b7280", width=0.7, linestyle="dotted",
                ),
            ]

            # Mark the actual final-bar WT cross used by the live entry.
            if len(wt1) >= 2 and all(
                np.isfinite(v) for v in (wt1.iloc[-2], wt1.iloc[-1], wt2.iloc[-2], wt2.iloc[-1])
            ):
                cross_up = wt1.iloc[-2] <= wt2.iloc[-2] and wt1.iloc[-1] > wt2.iloc[-1]
                cross_down = wt1.iloc[-2] >= wt2.iloc[-2] and wt1.iloc[-1] < wt2.iloc[-1]
                expected_cross = cross_up if direction == "long" else cross_down
                if expected_cross:
                    marker = pd.Series(np.nan, index=df.index, dtype=float)
                    marker.iloc[-1] = float(wt1.iloc[-1])
                    addplots.append(
                        mpf.make_addplot(
                            marker,
                            panel=2,
                            type="scatter",
                            marker="^" if direction == "long" else "v",
                            markersize=90,
                            color="#facc15",
                        )
                    )
                    wt_cross_label = "CROSS UP" if direction == "long" else "CROSS DOWN"
        else:
            macd_line_arr, macd_signal_arr, macd_hist_arr = BaseStrategy.macd(
                closes, macd_fast, macd_slow, macd_signal_period,
            )
            macd_line = pd.Series(macd_line_arr, index=df.index)
            macd_sig = pd.Series(macd_signal_arr, index=df.index)
            macd_hist = pd.Series(macd_hist_arr, index=df.index)
            hist_colors = [
                "#22c55e" if (v == v and v >= 0) else "#ef4444" for v in macd_hist
            ]
            addplots += [
                mpf.make_addplot(
                    macd_hist, type="bar", panel=2, color=hist_colors,
                    width=0.7, ylabel="MACD",
                ),
                mpf.make_addplot(macd_line, panel=2, color="#38bdf8", width=1.0),
                mpf.make_addplot(macd_sig, panel=2, color="#f97316", width=1.0),
            ]

        hlines_prices = [float(entry)]
        hlines_colors = ["#e5e7eb"]
        hlines_styles = ["solid"]
        hlines_widths = [1.3]

        t1_price = None
        if t1_pct is not None and float(t1_pct) > 0:
            t1_price = (
                float(entry) * (1.0 + float(t1_pct))
                if direction == "long"
                else float(entry) * (1.0 - float(t1_pct))
            )
            hlines_prices.append(t1_price)
            hlines_colors.append("#38bdf8")
            hlines_styles.append("dashdot")
            hlines_widths.append(1.1)

        if sl:
            hlines_prices.append(float(sl))
            hlines_colors.append("#ef4444")
            hlines_styles.append("dashed")
            hlines_widths.append(1.3)
        if tp:
            hlines_prices.append(float(tp))
            hlines_colors.append("#22c55e")
            hlines_styles.append("dashed")
            hlines_widths.append(1.3)

        hlines_prices += [res_level, sup_level]
        hlines_colors += ["#9ca3af", "#9ca3af"]
        hlines_styles += ["dotted", "dotted"]
        hlines_widths += [0.8, 0.8]

        resolved_dir_label = dir_label or (
            "LONG (+)" if direction == "long" else "SHORT (-)"
        )
        tf_text = f"  [{tf_label}]" if tf_label else ""
        trigger_text = f"  |  {entry_trigger}" if entry_trigger else ""
        title = (
            f"{symbol}{tf_text}  {resolved_dir_label}  |  {strategy or ''}{trigger_text}"
        ).strip()

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
            rc={
                "font.family": "DejaVu Sans",
                "font.weight": "normal",
                "axes.titleweight": "bold",
                "axes.labelweight": "normal",
                "axes.labelcolor": "#e5e7eb",
                "xtick.color": "#9ca3af",
                "ytick.color": "#9ca3af",
                "text.color": "#e5e7eb",
            },
        )

        output_dir = out_dir or tempfile.gettempdir()
        os.makedirs(output_dir, exist_ok=True)
        filename = (
            f"chart_{symbol.replace('/', '').replace(':', '')}_{int(time.time())}.png"
        )
        out_path = os.path.join(output_dir, filename)

        fig, axes = mpf.plot(
            df,
            type="candle",
            style=style,
            addplot=addplots,
            hlines=dict(
                hlines=hlines_prices,
                colors=hlines_colors,
                linestyle=hlines_styles,
                linewidths=hlines_widths,
            ),
            volume=True,
            panel_ratios=(3, 1, 1.25),
            title=title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(10, 8.7),
            returnfig=True,
        )

        ax = axes[0]
        legend_lines = [
            f"{entry_label} {entry:,.4f}",
            f"Trigger: {entry_trigger or 'n/a'}",
            f"{ma_label}{ema_fast} / {ma_label}{ema_slow}"
            + (f" / EMA{extra_ema}" if extra_ema else "")
            + (f" / SMA{sma_period}" if sma_period else ""),
        ]

        if use_wt_panel:
            legend_lines.append(
                f"WT {wt_channel_length}/{wt_average_length}/{wt_signal_length}  "
                f"levels {wt_oversold:g}/{wt_overbought:g}"
            )
            if wt1 is not None and wt2 is not None and np.isfinite(wt1.iloc[-1]) and np.isfinite(wt2.iloc[-1]):
                wt_status = f"WT1 {wt1.iloc[-1]:.1f} / WT2 {wt2.iloc[-1]:.1f}"
                if wt_cross_label:
                    wt_status += f"  {wt_cross_label}"
                legend_lines.append(wt_status)
        else:
            legend_lines.append(
                f"MACD {macd_fast}/{macd_slow}/{macd_signal_period}"
            )

        if sl:
            legend_lines.append(f"SL {sl:,.4f}")
        if t1_price is not None:
            legend_lines.append(
                f"T1 {t1_price:,.4f} (+{float(t1_pct) * 100:.1f}%, trim {t1_trim_pct * 100:.0f}%)"
            )
            legend_lines.append(
                f"Runner {100 - t1_trim_pct * 100:.0f}% locks SL +{t1_lock_pct * 100:.1f}%"
            )
        if tp:
            legend_lines.append(f"TP Final {tp:,.4f}")
        if macro_bias:
            legend_lines.append(f"Macro: {macro_bias}")
        if indicator_status:
            legend_lines.append(indicator_status)

        ax.text(
            0.01, 0.99, "\n".join(legend_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
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
        logger.warning("Chart render failed for %s: %s", symbol, exc)
        return None
