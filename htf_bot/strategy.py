"""HTF pullback strategy — the ENTIRE trading logic, as pure functions.

Rules (validated on 6 months of 1H/4H data BEFORE this bot was written —
net of the verified 0.05% taker fee: portfolio XAU/XRP/XAG/BTC +42R,
XAU alone PF 1.45, maxDD 6.7R):

  Trend (4H): EMA20 > EMA50 -> LONG-only. EMA20 < EMA50 -> SHORT-only.
  Entry (1H): the closed bar TOUCHES EMA20(1H) (low<=EMA20 for long) and
              CLOSES back on the trend side -> enter at next bar open.
  Stop:  beyond the last `swing_n` 1H bars' extreme, buffered 0.25 ATR,
         never tighter than 1.0 ATR(1H) or 0.8% of price (fee floor).
  Target: fixed 3R. Break-even: stop to entry once price reaches +1R.

Both live (main.py) and the backtest import THESE functions — they cannot
diverge. No scoring, no multi-engine arbitration, no dynamic thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

LONG, SHORT = "long", "short"


# ── indicators ───────────────────────────────────────────────────────────────

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def ohlcv_to_df(raw: list) -> pd.DataFrame:
    """ccxt fetch_ohlcv rows -> UTC-indexed OHLCV frame."""
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype(float)


def drop_unclosed(df: pd.DataFrame, tf_hours: int, now_ms: int) -> pd.DataFrame:
    """Keep only bars whose close time has passed."""
    if df.empty:
        return df
    close_ms = (df.index.as_unit("ns").asi8 // 1_000_000) + tf_hours * 3_600_000
    return df[close_ms <= now_ms]


# ── signal ───────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    direction: str          # "long" | "short"
    bar_ts: pd.Timestamp    # the closed 1H bar the signal fired on
    ema20: float
    atr1h: float


def trend_direction(df_4h: pd.DataFrame, fast: int = 20, slow: int = 50) -> int:
    """+1 long-only, -1 short-only, 0 no trend (not enough history)."""
    if len(df_4h) < slow + 5:
        return 0
    f = ema(df_4h["close"], fast).iloc[-1]
    s = ema(df_4h["close"], slow).iloc[-1]
    return 1 if f > s else -1


def entry_signal(df_1h: pd.DataFrame, trend: int,
                 min_body_atr: float = 0.5) -> Optional[Signal]:
    """Signal on the LAST CLOSED 1H bar, or None.

    min_body_atr: the reclaim bar's BODY must be at least this many ATRs —
    a decisive close-back, not a doji graze. Swept 0.0→1.2 on the 6-month
    set: monotone improvement into a 0.5–1.0 plateau (portfolio +22R at 0.0
    → +57R at 0.5, 7/8 symbols positive, drawdowns lower). 0.5 is the
    middle of the plateau, not the tail spike, to avoid curve-fitting."""
    if trend == 0 or len(df_1h) < 60:
        return None
    e20 = ema(df_1h["close"], 20)
    a = atr(df_1h, 14)
    e, av = float(e20.iloc[-1]), float(a.iloc[-1])
    if not np.isfinite(e) or not np.isfinite(av) or av <= 0:
        return None
    bar = df_1h.iloc[-1]
    if abs(float(bar["close"]) - float(bar["open"])) < min_body_atr * av:
        return None
    if trend == 1 and bar["low"] <= e and bar["close"] > e:
        return Signal(LONG, df_1h.index[-1], e, av)
    if trend == -1 and bar["high"] >= e and bar["close"] < e:
        return Signal(SHORT, df_1h.index[-1], e, av)
    return None


def plan_stop_target(df_1h: pd.DataFrame, direction: str, entry: float,
                     atr1h: float, swing_n: int = 6, sl_buf_atr: float = 0.25,
                     min_sl_atr: float = 1.0, min_sl_pct: float = 0.008,
                     tp_r: float = 3.0) -> tuple[float, float, float]:
    """(sl, tp, risk_dist) for a fill at `entry`. Uses the last `swing_n`
    CLOSED 1H bars (the signal bar and earlier)."""
    lows = df_1h["low"].values[-swing_n:]
    highs = df_1h["high"].values[-swing_n:]
    if direction == LONG:
        dist = entry - (float(np.min(lows)) - sl_buf_atr * atr1h)
    else:
        dist = (float(np.max(highs)) + sl_buf_atr * atr1h) - entry
    dist = max(dist, min_sl_atr * atr1h, min_sl_pct * entry)
    if direction == LONG:
        return entry - dist, entry + tp_r * dist, dist
    return entry + dist, entry - tp_r * dist, dist


def commodity_halted(symbol: str, now_utc: pd.Timestamp,
                     keywords: tuple = ("XAU", "XAG"),
                     halt_hour: int = 17, resume_hour: int = 21) -> bool:
    """No NEW metal entries Fri 17:00 UTC -> Sun 21:00 UTC (market closed)."""
    if not any(k in symbol.upper() for k in keywords):
        return False
    wd, hr = now_utc.weekday(), now_utc.hour
    return (wd == 4 and hr >= halt_hour) or wd == 5 or (wd == 6 and hr < resume_hour)
