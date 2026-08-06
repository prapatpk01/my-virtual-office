"""Trigger-aware entry charts for Telegram order notifications.

The price panel always shows candles, strategy moving averages and risk levels.
The trigger owner selects the supporting visualization:
- EMA entry: MACD lower panel.
- WT entry: WT1/WT2 lower panel with -42/+45 levels and cross marker.
- Structure entry: price-only structure view with BOS level, breakout marker and
  retest marker; unrelated MACD/WT panels are intentionally omitted.

Chart failure is non-fatal and falls back to the text Telegram notification.
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
    matplotlib.use("Agg")
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
    window = df.tail(min(lookback, len(df)))
    return float(window["High"].max()), float(window["Low"].min())


def _timestamp_ms(timestamp: int) -> int:
    value = int(timestamp)
    return value * 1000 if value < 10_000_000_000 else value


def _index_position_for_timestamp(df, timestamp: Optional[int]) -> Optional[int]:
    if timestamp is None or len(df.index) == 0:
        return None
    target = pd.to_datetime(_timestamp_ms(timestamp), unit="ms")
    matches = np.flatnonzero(df.index == target)
    if matches.size:
        return int(matches[-1])
    return None


def _ema_finite(values, period: int):
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
    for index in range(seed_idx + 1, len(arr)):
        if np.isfinite(arr[index]):
            prev = alpha * float(arr[index]) + (1.0 - alpha) * prev
        out[index] = prev
    return out


def _sma_finite(values, period: int):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    period = max(1, int(period))
    for index in range(period - 1, len(arr)):
        window = arr[index - period + 1:index + 1]
        if np.all(np.isfinite(window)):
            out[index] = float(np.mean(window))
    return out


def _wave_trend_series(
    df,
    channel_length: int = 10,
    average_length: int = 21,
    signal_length: int = 4,
):
    open_values = df["Open"].to_numpy(dtype=float)
    high_values = df["High"].to_numpy(dtype=float)
    low_values = df["Low"].to_numpy(dtype=float)
    close_values = df["Close"].to_numpy(dtype=float)

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
    direction: str,
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
    lower_panel: str = "macd",  # macd | wt | structure
    entry_trigger: Optional[str] = None,
    wt_channel_length: int = 10,
    wt_average_length: int = 21,
    wt_signal_length: int = 4,
    wt_oversold: float = -42.0,
    wt_overbought: float = 45.0,
    structure_level: Optional[float] = None,
    structure_breakout_ts: Optional[int] = None,
    structure_retest_ts: Optional[int] = None,
    t1_pct: Optional[float] = None,
    t1_trim_pct: float = 0.40,
    t1_lock_pct: float = 0.003,
) -> Optional[str]:
    if not _CHARTS_AVAILABLE:
        logger.debug("mplfinance/matplotlib unavailable — skipping chart render")
        return None
    if not candles or len(candles) < 20:
        return None

    try:
        df = _to_dataframe(candles[-lookback:])
        closes = df["Close"].tolist()
        mode = str(lower_panel).strip().lower()
        use_wt_panel = mode in {"wt", "wavetrend"}
        use_structure_view = mode in {"structure", "bos", "retest"}

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
                    sma_line,
                    color="#a855f7",
                    width=1.1,
                    linestyle="dashed",
                )
            )

        wt1 = wt2 = None
        wt_cross_label = None
        breakout_marker_drawn = False
        retest_marker_drawn = False

        if use_wt_panel:
            wt1, wt2 = _wave_trend_series(
                df,
                channel_length=wt_channel_length,
                average_length=wt_average_length,
                signal_length=wt_signal_length,
            )
            addplots += [
                mpf.make_addplot(
                    wt1,
                    panel=2,
                    color="#38bdf8",
                    width=1.25,
                    ylabel="WaveTrend",
                ),
                mpf.make_addplot(wt2, panel=2, color="#f97316", width=1.15),
                mpf.make_addplot(
                    pd.Series(float(wt_oversold), index=df.index),
                    panel=2,
                    color="#22c55e",
                    width=0.8,
                    linestyle="dashed",
                ),
                mpf.make_addplot(
                    pd.Series(float(wt_overbought), index=df.index),
                    panel=2,
                    color="#ef4444",
                    width=0.8,
                    linestyle="dashed",
                ),
                mpf.make_addplot(
                    pd.Series(0.0, index=df.index),
                    panel=2,
                    color="#6b7280",
                    width=0.7,
                    linestyle="dotted",
                ),
            ]
            if len(wt1) >= 2 and all(
                np.isfinite(value)
                for value in (
                    wt1.iloc[-2], wt1.iloc[-1], wt2.iloc[-2], wt2.iloc[-1]
                )
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
                    wt_cross_label = (
                        "CROSS UP" if direction == "long" else "CROSS DOWN"
                    )

        elif use_structure_view:
            breakout_position = _index_position_for_timestamp(
                df, structure_breakout_ts
            )
            if breakout_position is not None:
                marker = pd.Series(np.nan, index=df.index, dtype=float)
                marker.iloc[breakout_position] = (
                    float(df["Low"].iloc[breakout_position]) * 0.998
                    if direction == "long"
                    else float(df["High"].iloc[breakout_position]) * 1.002
                )
                addplots.append(
                    mpf.make_addplot(
                        marker,
                        panel=0,
                        type="scatter",
                        marker="^" if direction == "long" else "v",
                        markersize=100,
                        color="#facc15",
                    )
                )
                breakout_marker_drawn = True

            retest_position = _index_position_for_timestamp(df, structure_retest_ts)
            if retest_position is not None:
                marker = pd.Series(np.nan, index=df.index, dtype=float)
                marker.iloc[retest_position] = (
                    float(df["Low"].iloc[retest_position]) * 0.996
                    if direction == "long"
                    else float(df["High"].iloc[retest_position]) * 1.004
                )
                addplots.append(
                    mpf.make_addplot(
                        marker,
                        panel=0,
                        type="scatter",
                        marker="o",
                        markersize=70,
                        color="#38bdf8",
                    )
                )
                retest_marker_drawn = True

        else:
            macd_line_arr, macd_signal_arr, macd_hist_arr = BaseStrategy.macd(
                closes,
                macd_fast,
                macd_slow,
                macd_signal_period,
            )
            macd_line = pd.Series(macd_line_arr, index=df.index)
            macd_sig = pd.Series(macd_signal_arr, index=df.index)
            macd_hist = pd.Series(macd_hist_arr, index=df.index)
            hist_colors = [
                "#22c55e" if (value == value and value >= 0) else "#ef4444"
                for value in macd_hist
            ]
            addplots += [
                mpf.make_addplot(
                    macd_hist,
                    type="bar",
                    panel=2,
                    color=hist_colors,
                    width=0.7,
                    ylabel="MACD",
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
        if use_structure_view and structure_level is not None:
            hlines_prices.append(float(structure_level))
            hlines_colors.append("#facc15")
            hlines_styles.append("dashdot")
            hlines_widths.append(1.25)

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
            f"{symbol}{tf_text}  {resolved_dir_label}  |  "
            f"{strategy or ''}{trigger_text}"
        ).strip()

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
            f"chart_{symbol.replace('/', '').replace(':', '')}_"
            f"{int(time.time())}.png"
        )
        out_path = os.path.join(output_dir, filename)

        panel_ratios = (3, 1) if use_structure_view else (3, 1, 1.25)
        figure_size = (10, 7.4) if use_structure_view else (10, 8.7)
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
            panel_ratios=panel_ratios,
            title=title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=figure_size,
            returnfig=True,
        )

        price_axis = axes[0]
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
            if (
                wt1 is not None
                and wt2 is not None
                and np.isfinite(wt1.iloc[-1])
                and np.isfinite(wt2.iloc[-1])
            ):
                wt_status = (
                    f"WT1 {wt1.iloc[-1]:.1f} / WT2 {wt2.iloc[-1]:.1f}"
                )
                if wt_cross_label:
                    wt_status += f"  {wt_cross_label}"
                legend_lines.append(wt_status)
        elif use_structure_view:
            if structure_level is not None:
                legend_lines.append(f"BOS / Retest level {structure_level:,.4f}")
            marker_status = []
            if breakout_marker_drawn:
                marker_status.append("Breakout marked")
            if retest_marker_drawn:
                marker_status.append("Retest marked")
            if marker_status:
                legend_lines.append(" | ".join(marker_status))
            legend_lines.append("Exit: level invalidation / opposite CHOCH")
        else:
            legend_lines.append(
                f"MACD {macd_fast}/{macd_slow}/{macd_signal_period}"
            )

        if sl:
            legend_lines.append(f"SL {sl:,.4f}")
        if t1_price is not None:
            legend_lines.append(
                f"T1 {t1_price:,.4f} "
                f"(+{float(t1_pct) * 100:.1f}%, trim {t1_trim_pct * 100:.0f}%)"
            )
            legend_lines.append(
                f"Runner {100 - t1_trim_pct * 100:.0f}% "
                f"locks SL +{t1_lock_pct * 100:.1f}%"
            )
        if tp:
            legend_lines.append(f"TP Final {tp:,.4f}")
        if macro_bias:
            legend_lines.append(f"Macro: {macro_bias}")
        if indicator_status:
            legend_lines.append(indicator_status)

        price_axis.text(
            0.01,
            0.99,
            "\n".join(legend_lines),
            transform=price_axis.transAxes,
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
