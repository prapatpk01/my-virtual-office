"""Entry chart for Telegram (Dual mode).

Renders the 15M entry timeframe with HMA10/16 + EMA50 overlays, entry/SL/TP
lines and the active + opposing zones shaded — the same technical context
the regime bot's chart carried. Built straight from the EntryIndicators
bundle (it already holds OHLCV + HMA arrays), so no extra recompute.

Fails soft: returns None on any error (a missing chart must never cost the
alert — the notifier falls back to text).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger("dual_entry.chart")

try:
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    import mplfinance as mpf
    from .indicator_engine import ema
    _AVAILABLE = True
except Exception as _e:                       # pragma: no cover
    logger.warning("[CHART] charting unavailable: %s", _e)
    _AVAILABLE = False


def build_entry_chart(symbol: str, ind, direction: str, entry: float,
                      sl: float, tp: float, active_zone=None, opposing_zone=None,
                      out_dir: Optional[str] = None, bars: int = 120) -> Optional[str]:
    if not _AVAILABLE or ind is None or len(ind.closes) < 30:
        return None
    try:
        n = min(bars, len(ind.closes))
        idx = pd.to_datetime(np.asarray(ind.timestamps[-n:], dtype="int64"), unit="ms", utc=True)
        df = pd.DataFrame({
            "Open": ind.opens[-n:], "High": ind.highs[-n:],
            "Low": ind.lows[-n:], "Close": ind.closes[-n:], "Volume": ind.volumes[-n:],
        }, index=idx)

        ema50 = ema(ind.closes, 50)
        addplots = [
            mpf.make_addplot(ind.hma_fast[-n:], color="#00d4ff", width=1.0),   # HMA10
            mpf.make_addplot(ind.hma_slow[-n:], color="#ff5d8f", width=1.0),   # HMA16
            mpf.make_addplot(ema50[-n:], color="#f0b90b", width=1.0),          # EMA50
        ]

        hlines = dict(hlines=[entry, sl, tp],
                      colors=["#ffffff", "#ff4d4f", "#3fb950"],
                      linestyle=["--", "-", "-"], linewidths=[1.0, 1.3, 1.3])
        # active zone (where we enter) + opposing zone (TP ceiling) as dotted bands
        for z, color in ((active_zone, "#58a6ff"), (opposing_zone, "#8b949e")):
            if z is not None:
                for edge in (getattr(z, "upper_price", None), getattr(z, "lower_price", None)):
                    if edge is not None:
                        hlines["hlines"].append(float(edge))
                        hlines["colors"].append(color)
                        hlines["linestyle"].append(":")
                        hlines["linewidths"].append(0.8)

        mc = mpf.make_marketcolors(up="#3fb950", down="#f85149", edge="inherit",
                                   wick="inherit", volume="in")
        style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc,
                                   gridcolor="#30363d", facecolor="#0d1117",
                                   figcolor="#0d1117", edgecolor="#30363d")

        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"dev14_{symbol.replace('/', '_').replace(':', '_')}_{int(time.time())}.png"
        path = os.path.join(out_dir, fname)
        title = f"{symbol}  {direction}  15M  entry={entry:.6g}"
        mpf.plot(df, type="candle", style=style, addplot=addplots, hlines=hlines,
                 volume=True, title=title, tight_layout=True,
                 savefig=dict(fname=path, dpi=130, bbox_inches="tight"))
        return path
    except Exception as e:
        logger.warning("[CHART] build failed for %s: %s", symbol, e)
        return None
