"""Reliable HMA entry-chart renderer for Telegram.

The previous renderer used mplfinance and could silently return ``None`` on
some production frames/configurations. This implementation uses Matplotlib
directly, accepts lowercase exchange OHLCV columns, and always logs the exact
failure reason.

Chart contents:
- Actual 5M execution candles
- EMA8 / EMA13
- Entry, Structure SL, Final TP
- Recent confirmed swing high / swing low
- Volume
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

logger = logging.getLogger("hma_chart_engine")


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ("open", "high", "low", "close")
    if df is None or df.empty or any(c not in df.columns for c in required):
        return pd.DataFrame()

    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df.loc[:, cols].tail(96).copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"])


def _recent_confirmed_swings(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> tuple[float, float]:
    if len(df) < left + right + 3:
        return float("nan"), float("nan")

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    swing_high = float("nan")
    swing_low = float("nan")

    for i in range(left, len(df) - right):
        high_window = highs[i - left : i + right + 1]
        low_window = lows[i - left : i + right + 1]
        if np.isfinite(highs[i]) and highs[i] >= np.nanmax(high_window):
            swing_high = float(highs[i])
        if np.isfinite(lows[i]) and lows[i] <= np.nanmin(low_window):
            swing_low = float(lows[i])

    return swing_high, swing_low


def _price_label(ax, value: float, label: str, color: str, linestyle: str, width: float) -> None:
    if not np.isfinite(value) or value <= 0:
        return
    ax.axhline(value, color=color, linestyle=linestyle, linewidth=width, alpha=0.95)
    ax.text(
        1.002,
        value,
        f" {label} {value:.6g}",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=8,
        color=color,
        clip_on=False,
    )


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
    """Create a Telegram-ready PNG and return its local path."""
    fig = None
    try:
        plot_df = _clean_frame(df)
        minimum = max(int(ema_slow_len) + 2, 20)
        if len(plot_df) < minimum:
            logger.warning(
                "[HMA CHART] insufficient candles for %s: got=%d need=%d",
                symbol,
                len(plot_df),
                minimum,
            )
            return None

        direction = str(direction or "").upper()
        entry = float(entry)
        sl = float(sl)
        final_tp = float(tp2 if tp2 else tp1)

        closes = plot_df["close"]
        ema_fast = closes.ewm(span=max(1, int(ema_fast_len)), adjust=False).mean()
        ema_slow = closes.ewm(span=max(1, int(ema_slow_len)), adjust=False).mean()
        swing_high, swing_low = _recent_confirmed_swings(plot_df)

        fig, (ax, ax_vol) = plt.subplots(
            2,
            1,
            figsize=(10.5, 7.0),
            dpi=130,
            sharex=True,
            gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.05},
        )
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax_vol.set_facecolor("#0d1117")

        x = np.arange(len(plot_df), dtype=float)
        opens = plot_df["open"].to_numpy(dtype=float)
        highs = plot_df["high"].to_numpy(dtype=float)
        lows = plot_df["low"].to_numpy(dtype=float)
        close_values = plot_df["close"].to_numpy(dtype=float)
        candle_width = 0.62

        for i in range(len(plot_df)):
            bullish = close_values[i] >= opens[i]
            color = "#3fb950" if bullish else "#f85149"
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=0.85, alpha=0.95)
            body_low = min(opens[i], close_values[i])
            body_height = abs(close_values[i] - opens[i])
            if body_height <= 0:
                body_height = max(abs(highs[i] - lows[i]) * 0.01, 1e-12)
            ax.add_patch(
                Rectangle(
                    (x[i] - candle_width / 2.0, body_low),
                    candle_width,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.7,
                )
            )

        ax.plot(x, ema_fast.to_numpy(dtype=float), color="#00d4ff", linewidth=1.25, label=f"EMA{ema_fast_len}")
        ax.plot(x, ema_slow.to_numpy(dtype=float), color="#ffb000", linewidth=1.25, label=f"EMA{ema_slow_len}")

        _price_label(ax, entry, "ENTRY", "#f0f6fc", "--", 1.0)
        _price_label(ax, sl, "SL", "#ff4d4f", "-", 1.25)
        _price_label(ax, final_tp, "TP", "#3fb950", "-", 1.25)
        if np.isfinite(swing_high):
            _price_label(ax, swing_high, "SWING H", "#8b949e", ":", 0.85)
        if np.isfinite(swing_low):
            _price_label(ax, swing_low, "SWING L", "#8b949e", ":", 0.85)

        if "volume" in plot_df.columns:
            volume = plot_df["volume"].fillna(0.0).to_numpy(dtype=float)
            volume_colors = ["#3fb950" if c >= o else "#f85149" for o, c in zip(opens, close_values)]
            ax_vol.bar(x, volume, width=0.68, color=volume_colors, alpha=0.72)
        else:
            ax_vol.text(0.5, 0.5, "Volume unavailable", transform=ax_vol.transAxes,
                        ha="center", va="center", color="#8b949e")

        marker = "LONG" if direction == "LONG" else "SHORT"
        title_color = "#3fb950" if direction == "LONG" else "#f85149"
        ax.set_title(
            f"{symbol}  {marker}  {tf_label} EXECUTION",
            color=title_color,
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor="#c9d1d9")
        ax.grid(True, color="#30363d", linewidth=0.45, alpha=0.55)
        ax_vol.grid(True, axis="y", color="#30363d", linewidth=0.4, alpha=0.45)
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        ax_vol.tick_params(colors="#8b949e", labelsize=7)
        ax.set_ylabel("Price", color="#c9d1d9")
        ax_vol.set_ylabel("Vol", color="#8b949e")

        for axis in (ax, ax_vol):
            for spine in axis.spines.values():
                spine.set_color("#30363d")

        tick_count = min(8, len(plot_df))
        tick_positions = np.linspace(0, len(plot_df) - 1, tick_count, dtype=int)
        tick_labels = [plot_df.index[i].strftime("%m-%d\n%H:%M") for i in tick_positions]
        ax_vol.set_xticks(tick_positions)
        ax_vol.set_xticklabels(tick_labels, color="#8b949e", fontsize=7)
        ax.set_xlim(-1.0, len(plot_df) + 7.0)

        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        path = os.path.join(out_dir, f"hma_{safe_symbol}_{int(time.time() * 1000)}.png")
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)

        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            logger.warning("[HMA CHART] output missing/empty for %s path=%s", symbol, path)
            return None

        logger.info("[HMA CHART] created %s bytes=%d", path, os.path.getsize(path))
        return path
    except Exception as exc:
        logger.exception("[HMA CHART] failed for %s: %s", symbol, exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
