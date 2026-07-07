"""
TrendContV2 Backtest — Comprehensive Realistic Simulation
==========================================================
Reads Binance OHLCV CSV zip files and runs a full backtest of the
TrendContV2 entry/exit logic with realistic trade simulation.

Usage:
    python app/backtest_v2.py [--data-dir /tmp/backtest_data] [--out report.html]
    python app/backtest_v2.py --symbols BTC,XAU  (subset)
    python app/backtest_v2.py --balance 10000 --leverage 20

Output: self-contained HTML report with embedded charts.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed — charts will be skipped.")


# ═══════════════════════════════════════════════════════════════════════════════
# Constants / Config
# ═══════════════════════════════════════════════════════════════════════════════

LEVERAGE       = 20
INITIAL_BAL    = 10_000.0
RISK_MIN       = 0.08       # 8%  of balance at risk per trade (min confidence)
RISK_MAX       = 0.12       # 12% of balance at risk per trade (max confidence)
SL_ATR_MULT    = 1.2
SL_MIN_PCT     = 0.012      # 1.2% minimum stop distance
SL_MAX_PCT     = 0.035      # 3.5% maximum stop distance
TP1_R          = 0.5        # TP1 at 0.5× the SL distance (close 50%, SL → BE)
TP2_R          = 1.2        # TP2 at 1.2× the SL distance (close remainder)
MAKER_FEE      = 0.0002     # 0.02% maker; we use taker for entries
TAKER_FEE      = 0.0005     # 0.05% taker
SLIPPAGE       = 0.0003     # 0.03% slippage on entry
MAX_POSITIONS  = 2
ADX_MIN        = 18.0       # minimum ADX to enter
COOLDOWN_BARS  = 3          # bars between exits and next entry (same symbol)

# Signal score thresholds
SIGNAL_THRESHOLD = 60.0     # minimum score to enter

COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


# ═══════════════════════════════════════════════════════════════════════════════
# Symbol Definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymbolConfig:
    name: str
    ticker: str
    data_dir: str
    nested_tf: dict = field(default_factory=dict)   # {"15m": "15m.zip", "1h": "1h.zip"}
    pip_size: float = 0.01

ALL_SYMBOLS: list[SymbolConfig] = [
    SymbolConfig("BTC",  "BTCUSDT",  "/tmp/backtest_data/crypto/Crypto/BTC_data/BTC",
                 nested_tf={"15m": "15m.zip", "1h": "1h.zip"}),
    SymbolConfig("ETH",  "ETHUSDT",  "/tmp/backtest_data/crypto/Crypto/ETH_data/ETH"),
    SymbolConfig("SOL",  "SOLUSDT",  "/tmp/backtest_data/crypto/Crypto/SOL_data/SOL"),
    SymbolConfig("XRP",  "XRPUSDT",  "/tmp/backtest_data/crypto/Crypto/XRP_data/XRP"),
    SymbolConfig("HYPE", "HYPEUSDT", "/tmp/backtest_data/crypto/Crypto/HYPE_data/HYPE"),
    SymbolConfig("XAU",  "XAUUSDT",  "/tmp/backtest_data/commodity/Commodity/XAU_data/XAU"),
    SymbolConfig("XAG",  "XAGUSDT",  "/tmp/backtest_data/commodity/Commodity/XAG_data/XAG"),
    SymbolConfig("CL",   "CLUSDT",   "/tmp/backtest_data/commodity/Commodity/CL_data/CL"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Read one CSV file from an open ZipFile."""
    raw = zf.open(name).read().decode()
    lines = raw.strip().split("\n")
    # detect header
    if lines[0].startswith("open_time") or lines[0].startswith("Open"):
        lines = lines[1:]
    rows = [l.split(",") for l in lines if l.strip()]
    df = pd.DataFrame(rows, columns=COLS[:len(rows[0])] if rows else COLS)
    return df


def _normalize_ts(ts_series: pd.Series) -> pd.Series:
    """Convert any timestamp to milliseconds (13-digit Unix ms)."""
    ts = ts_series.astype(np.int64)
    # 16-digit → microseconds; 13-digit → milliseconds
    mask_us = ts > 9_999_999_999_999  # > year 2286 in ms, must be µs
    ts_ms = ts.copy()
    ts_ms[mask_us] = ts[mask_us] // 1000
    return ts_ms


def load_ohlcv(sym: SymbolConfig, tf: str = "15m") -> pd.DataFrame:
    """Load and merge all monthly OHLCV files for one symbol and timeframe."""
    frames = []
    data_dir = Path(sym.data_dir)

    # ── Nested zip (BTC style) ──────────────────────────────────────────────
    nested_name = sym.nested_tf.get(tf)
    if nested_name:
        nested_path = data_dir / nested_name
        if nested_path.exists():
            outer = zipfile.ZipFile(nested_path)
            for inner_name in outer.namelist():
                if inner_name.startswith("__") or not inner_name.endswith(".zip"):
                    continue
                inner_bytes = outer.open(inner_name).read()
                inner_zip = zipfile.ZipFile(io.BytesIO(inner_bytes))
                csv_names = [n for n in inner_zip.namelist() if n.endswith(".csv")]
                for csv_name in csv_names:
                    frames.append(_read_csv_from_zip(inner_zip, csv_name))

    # ── Direct monthly zips ─────────────────────────────────────────────────
    for zpath in sorted(data_dir.glob(f"{sym.ticker}-{tf}-*.zip")):
        if zpath.name == nested_name:
            continue
        with zipfile.ZipFile(zpath) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            for csv_name in csv_names:
                frames.append(_read_csv_from_zip(zf, csv_name))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # ── Normalize ────────────────────────────────────────────────────────────
    df["open_time"] = _normalize_ts(pd.to_numeric(df["open_time"], errors="coerce"))
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open_time", "close"])
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# Technical Indicators (vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx_indicators(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)

    up_move = high - prev_high
    dn_move = prev_low - low

    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean()
    atr_s      = tr.ewm(span=period, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_s.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val  = dx.ewm(span=period, adjust=False).mean()
    return adx_val, plus_di, minus_di


def macd(series: pd.Series, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series]:
    """Returns (MACD line, histogram)."""
    line = ema(series, fast) - ema(series, slow)
    sig  = ema(line, signal)
    return line, line - sig


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Generation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df15: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to 15m DataFrame."""
    d = df15.copy()

    # Trend structure
    d["ema20"]  = ema(d["close"], 20)
    d["ema50"]  = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)

    # EMA50 slope (5-bar % change)
    d["ema50_slope"] = (d["ema50"] - d["ema50"].shift(5)) / d["ema50"].shift(5) * 100

    # ATR
    d["atr"] = atr(d, 14)

    # ADX / DI
    d["adx"], d["plus_di"], d["minus_di"] = adx_indicators(d, 14)

    # MACD
    _macd_line, d["macd_hist"] = macd(d["close"])

    # Volume
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma20"].replace(0, np.nan)

    # ── 1h Higher-Timeframe (HTF) ────────────────────────────────────────────
    if not df1h.empty:
        h = df1h.copy()
        h["h1_ema20"] = ema(h["close"], 20)
        h["h1_ema50"] = ema(h["close"], 50)
        h["h1_adx"], _, _ = adx_indicators(h, 14)
        # Forward-fill to 15m bars
        h_cols = h[["h1_ema20", "h1_ema50", "h1_adx"]]
        d = d.join(h_cols, how="left")
        d[["h1_ema20", "h1_ema50", "h1_adx"]] = d[["h1_ema20", "h1_ema50", "h1_adx"]].ffill()
    else:
        d["h1_ema20"] = np.nan
        d["h1_ema50"] = np.nan
        d["h1_adx"]   = np.nan

    return d


def _dynamic_threshold(adx_series: pd.Series) -> pd.Series:
    """ADX-based dynamic entry threshold (mirrors TrendContV2Strategy)."""
    thr = pd.Series(95.0, index=adx_series.index)
    thr = thr.where(adx_series < 15,  95.0)
    thr = thr.where(adx_series < 20,  88.0)
    thr = thr.where(adx_series < 25,  82.0)
    thr = thr.where(adx_series < 30,  78.0)
    thr = thr.where(adx_series < 40,  75.0)
    thr = thr.where(adx_series >= 40, 70.0)
    return thr


def score_signals(d: pd.DataFrame) -> pd.DataFrame:
    """
    Compute entry scores for LONG and SHORT directions.
    Dynamic threshold: ADX>=40→70, >=30→75, >=25→78, >=20→82, >=15→88, <15→95.

    Components (mirrors TrendContV2Strategy):
      Trend structure  (0-12): EMA20 vs EMA50 — stable through pullbacks
      DI bonus         (0-4):  DI alignment bonus (not gate)
      ADX strength     (0-8):  higher ADX → higher score
      Pullback zone    (0-20): price relative to EMA20/EMA50
      MACD momentum    (0-15): histogram direction and magnitude
      Volume           (0-10): vol ratio vs 20-bar MA
      HTF alignment    (0-12): 1h EMA structure confirmation
      EMA50 slope      (0-8):  slope confirms trend direction
      Acceleration     (0-11): MACD histogram acceleration
    Total max: 100
    """
    # ── Structural conditions ────────────────────────────────────────────────
    bull_struct = d["ema20"] > d["ema50"]
    bear_struct = d["ema20"] < d["ema50"]

    # Pullback zone definitions:
    # LONG pullback: price pulled back BELOW EMA20 but still ABOVE EMA50
    in_pb_long  = (d["close"] < d["ema20"]) & (d["close"] > d["ema50"])
    near_e20_long = (d["close"] >= d["ema20"]) & (d["close"] < d["ema20"] * 1.003)

    # SHORT pullback: price bounced ABOVE EMA20 but still BELOW EMA50
    in_pb_short = (d["close"] > d["ema20"]) & (d["close"] < d["ema50"])
    near_e20_short = (d["close"] <= d["ema20"]) & (d["close"] > d["ema20"] * 0.997)

    # ── Trend: EMA20 vs EMA50 (12 pts) ──────────────────────────────────────
    trend_long  = pd.Series(np.where(bull_struct, 12.0, 0.0), index=d.index)
    trend_short = pd.Series(np.where(bear_struct, 12.0, 0.0), index=d.index)

    # ── DI bonus (0-4 pts) ──────────────────────────────────────────────────
    di_total  = (d["plus_di"] + d["minus_di"]).replace(0, np.nan)
    di_margin = (d["plus_di"] - d["minus_di"]).abs() / di_total
    di_bonus  = (di_margin * 10).clip(0, 4).fillna(0)
    di_long_bonus  = di_bonus.where(d["plus_di"]  > d["minus_di"], 0.0)
    di_short_bonus = di_bonus.where(d["minus_di"] > d["plus_di"],  0.0)

    # ── ADX strength (0-8 pts) ───────────────────────────────────────────────
    adx_score = ((d["adx"] - ADX_MIN) / (50.0 - ADX_MIN) * 8).clip(0, 8).fillna(0)

    # ── Pullback zone (0-20 pts) ─────────────────────────────────────────────
    pb_long  = pd.Series(0.0, index=d.index)
    pb_long  = pb_long.where(~in_pb_long,  20.0)
    pb_long  = pb_long.where(~(~in_pb_long & near_e20_long),  pb_long).where(
                              ~in_pb_long & near_e20_long, 14.0)
    pb_short = pd.Series(0.0, index=d.index)
    pb_short = pb_short.where(~in_pb_short, 20.0)
    pb_short = pb_short.where(~(~in_pb_short & near_e20_short), pb_short).where(
                               ~in_pb_short & near_e20_short, 14.0)

    # Simplified vectorised version (avoids chained-where confusion):
    pb_long_vals  = np.where(in_pb_long, 20.0,
                    np.where(near_e20_long, 14.0, 0.0))
    pb_short_vals = np.where(in_pb_short, 20.0,
                    np.where(near_e20_short, 14.0, 0.0))
    pb_long  = pd.Series(pb_long_vals,  index=d.index)
    pb_short = pd.Series(pb_short_vals, index=d.index)

    # ── MACD momentum (0-15 pts) ─────────────────────────────────────────────
    h  = d["macd_hist"]
    hp = h.shift(1)
    hist_long  = pd.Series(np.where(
        (h > 0) & (h > hp), 15.0,
        np.where((h > 0) & (h <= hp), 8.0,
        np.where((h <= 0) & (h > hp), 5.0, 0.0))), index=d.index)
    hist_short = pd.Series(np.where(
        (h < 0) & (h < hp), 15.0,
        np.where((h < 0) & (h >= hp), 8.0,
        np.where((h >= 0) & (h < hp), 5.0, 0.0))), index=d.index)

    # ── Volume (0-10 pts) ────────────────────────────────────────────────────
    vol_score = (d["vol_ratio"] * 5).clip(0, 10).fillna(5.0)

    # ── HTF alignment (0-12 pts) ─────────────────────────────────────────────
    has_htf = d["h1_ema20"].notna() & d["h1_ema50"].notna()
    htf_long  = pd.Series(np.where(
        has_htf & (d["h1_ema20"] > d["h1_ema50"]), 12.0,
        np.where(~has_htf, 6.0, 0.0)), index=d.index)
    htf_short = pd.Series(np.where(
        has_htf & (d["h1_ema20"] < d["h1_ema50"]), 12.0,
        np.where(~has_htf, 6.0, 0.0)), index=d.index)

    # ── EMA50 slope (0-8 pts) ────────────────────────────────────────────────
    s = d["ema50_slope"]
    slope_long  = pd.Series(np.where(s > 0.05, 8.0,
                             np.where(s > -0.02, 3.0, 0.0)), index=d.index)
    slope_short = pd.Series(np.where(s < -0.05, 8.0,
                             np.where(s < 0.02,  3.0, 0.0)), index=d.index)

    # ── Acceleration (0-11 pts) ──────────────────────────────────────────────
    dh  = h - h.shift(1)
    dh2 = dh - dh.shift(1)
    acc_long  = ((dh2 > 0).astype(float) * 7 + (dh > 0).astype(float) * 4).clip(0, 11)
    acc_short = ((dh2 < 0).astype(float) * 7 + (dh < 0).astype(float) * 4).clip(0, 11)

    # ── Total scores ─────────────────────────────────────────────────────────
    d["score_long"]  = (trend_long  + di_long_bonus  + adx_score + pb_long  +
                        hist_long   + vol_score + htf_long  + slope_long  + acc_long ).round(1)
    d["score_short"] = (trend_short + di_short_bonus + adx_score + pb_short +
                        hist_short  + vol_score + htf_short + slope_short + acc_short).round(1)

    # ── Dynamic threshold (ADX-based, matches real strategy) ─────────────────
    d["threshold"] = _dynamic_threshold(d["adx"].fillna(0))

    # ── MACD histogram momentum cross (entry trigger) ─────────────────────────
    # Only trigger on the REVERSAL BAR (MACD hist turns in our direction).
    # This prevents generating signals on every bar inside a pullback zone.
    macd_cross_long  = (d["macd_hist"] > d["macd_hist"].shift(1)) & (d["macd_hist"].shift(1) <= 0)
    macd_cross_short = (d["macd_hist"] < d["macd_hist"].shift(1)) & (d["macd_hist"].shift(1) >= 0)

    # ── Required conditions ───────────────────────────────────────────────────
    # 1. ADX >= minimum (trending, not chop)
    # 2. EMA20/50 structural alignment (stable indicator)
    # 3. Price in pullback zone (touched EMA20 area) — required, not optional
    # 4. MACD histogram just turned in our direction (entry trigger)
    # 5. Score >= dynamic threshold (strength confirmation)
    adx_ok       = d["adx"] >= ADX_MIN
    pb_long_req  = in_pb_long | near_e20_long
    pb_short_req = in_pb_short | near_e20_short

    d["sig_long"]  = (
        (d["score_long"]  >= d["threshold"]) &
        adx_ok & bull_struct & pb_long_req & macd_cross_long
    )
    d["sig_short"] = (
        (d["score_short"] >= d["threshold"]) &
        adx_ok & bear_struct & pb_short_req & macd_cross_short
    )

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Simulation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol:     str
    direction:  str          # "long" or "short"
    entry_time: pd.Timestamp
    exit_time:  Optional[pd.Timestamp] = None
    entry_px:   float = 0.0
    sl_px:      float = 0.0
    tp1_px:     float = 0.0
    tp2_px:     float = 0.0
    size:       float = 0.0  # contracts / coins
    notional:   float = 0.0
    risk_pct:   float = 0.0
    score:      float = 0.0
    exit_px:    float = 0.0
    pnl_usd:    float = 0.0
    pnl_pct:    float = 0.0  # % of balance at entry
    exit_reason: str = ""
    tp1_hit:    bool = False
    balance_before: float = 0.0
    balance_after:  float = 0.0


def _entry_fee(notional: float) -> float:
    return notional * TAKER_FEE


def _exit_fee(notional: float) -> float:
    return notional * MAKER_FEE


def _calc_sl_dist(entry_px: float, atr_val: float) -> float:
    sl_dist = atr_val * SL_ATR_MULT / entry_px
    return float(np.clip(sl_dist, SL_MIN_PCT, SL_MAX_PCT))


def _confidence_risk_pct(score: float) -> float:
    raw = (score - SIGNAL_THRESHOLD) / max(100.0 - SIGNAL_THRESHOLD, 1.0)
    conf = float(np.clip(raw, 0.0, 1.0))
    return RISK_MIN + (RISK_MAX - RISK_MIN) * conf


def simulate(sym_name: str, df: pd.DataFrame, balance_ref: list[float]) -> list[Trade]:
    """
    Bar-by-bar simulation for one symbol, sharing a balance pool.
    balance_ref[0] is updated in place so multi-symbol simulation shares equity.
    Returns list of completed trades.
    """
    trades: list[Trade] = []
    open_trade: Optional[Trade] = None
    tp1_hit = False
    cooldown = 0

    arr_open  = df["open"].values
    arr_high  = df["high"].values
    arr_low   = df["low"].values
    arr_close = df["close"].values
    arr_atr   = df["atr"].values
    arr_sl    = df["score_long"].values   # reused as score placeholder below
    arr_score_l = df["score_long"].values
    arr_score_s = df["score_short"].values
    arr_sig_l   = df["sig_long"].values
    arr_sig_s   = df["sig_short"].values
    idx = df.index

    n = len(df)
    warmup = 60  # skip first 60 bars (indicator warmup)

    for i in range(warmup, n - 1):
        balance = balance_ref[0]
        bar_time = idx[i]

        # ── Manage open trade ────────────────────────────────────────────────
        if open_trade is not None:
            hi  = arr_high[i]
            lo  = arr_low[i]
            t   = open_trade

            if t.direction == "long":
                # Check SL then TP (conservative: SL first if same bar)
                sl_hit = lo <= t.sl_px
                tp1_trigger = hi >= t.tp1_px and not tp1_hit
                tp2_hit = hi >= t.tp2_px

                if sl_hit:
                    # SL exit (full remaining position)
                    exit_px = t.sl_px
                    remaining = 0.5 if tp1_hit else 1.0
                    pnl = (exit_px - t.entry_px) / t.entry_px * t.notional * remaining
                    pnl -= _exit_fee(t.notional * remaining)
                    t.pnl_usd += pnl
                    t.exit_px = exit_px
                    t.exit_time = bar_time
                    t.exit_reason = "SL"
                    t.balance_after = balance + t.pnl_usd
                    balance_ref[0] = max(1.0, t.balance_after)
                    trades.append(t)
                    open_trade = None
                    tp1_hit = False
                    cooldown = COOLDOWN_BARS
                    continue

                if tp1_trigger and not tp1_hit:
                    # Partial exit at TP1 (50%)
                    exit_px = t.tp1_px
                    half = t.notional * 0.5
                    pnl = (exit_px - t.entry_px) / t.entry_px * half
                    pnl -= _exit_fee(half)
                    t.pnl_usd += pnl
                    tp1_hit = True
                    t.tp1_hit = True
                    # Move SL to breakeven
                    t.sl_px = t.entry_px

                if tp2_hit and tp1_hit:
                    exit_px = t.tp2_px
                    half = t.notional * 0.5
                    pnl = (exit_px - t.entry_px) / t.entry_px * half
                    pnl -= _exit_fee(half)
                    t.pnl_usd += pnl
                    t.exit_px = exit_px
                    t.exit_time = bar_time
                    t.exit_reason = "TP2"
                    t.balance_after = balance + t.pnl_usd
                    balance_ref[0] = max(1.0, t.balance_after)
                    trades.append(t)
                    open_trade = None
                    tp1_hit = False
                    cooldown = COOLDOWN_BARS
                    continue

            else:  # short
                sl_hit = hi >= t.sl_px
                tp1_trigger = lo <= t.tp1_px and not tp1_hit
                tp2_hit = lo <= t.tp2_px

                if sl_hit:
                    exit_px = t.sl_px
                    remaining = 0.5 if tp1_hit else 1.0
                    pnl = (t.entry_px - exit_px) / t.entry_px * t.notional * remaining
                    pnl -= _exit_fee(t.notional * remaining)
                    t.pnl_usd += pnl
                    t.exit_px = exit_px
                    t.exit_time = bar_time
                    t.exit_reason = "SL"
                    t.balance_after = balance + t.pnl_usd
                    balance_ref[0] = max(1.0, t.balance_after)
                    trades.append(t)
                    open_trade = None
                    tp1_hit = False
                    cooldown = COOLDOWN_BARS
                    continue

                if tp1_trigger and not tp1_hit:
                    exit_px = t.tp1_px
                    half = t.notional * 0.5
                    pnl = (t.entry_px - exit_px) / t.entry_px * half
                    pnl -= _exit_fee(half)
                    t.pnl_usd += pnl
                    tp1_hit = True
                    t.tp1_hit = True
                    t.sl_px = t.entry_px

                if tp2_hit and tp1_hit:
                    exit_px = t.tp2_px
                    half = t.notional * 0.5
                    pnl = (t.entry_px - exit_px) / t.entry_px * half
                    pnl -= _exit_fee(half)
                    t.pnl_usd += pnl
                    t.exit_px = exit_px
                    t.exit_time = bar_time
                    t.exit_reason = "TP2"
                    t.balance_after = balance + t.pnl_usd
                    balance_ref[0] = max(1.0, t.balance_after)
                    trades.append(t)
                    open_trade = None
                    tp1_hit = False
                    cooldown = COOLDOWN_BARS
                    continue

        # ── Cooldown ─────────────────────────────────────────────────────────
        if cooldown > 0:
            cooldown -= 1
            continue

        # ── New entry ────────────────────────────────────────────────────────
        if open_trade is not None:
            continue  # already in a trade for this symbol

        # Signal is on current bar; entry at next bar open
        sig_l = bool(arr_sig_l[i])
        sig_s = bool(arr_sig_s[i])
        if not sig_l and not sig_s:
            continue

        direction = "long" if sig_l else "short"
        score = float(arr_score_l[i]) if sig_l else float(arr_score_s[i])

        entry_px = float(arr_open[i + 1])
        if direction == "long":
            entry_px *= (1 + SLIPPAGE)
        else:
            entry_px *= (1 - SLIPPAGE)

        atr_val = float(arr_atr[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        sl_dist = _calc_sl_dist(entry_px, atr_val)
        risk_pct = _confidence_risk_pct(score)
        risk_usd = balance * risk_pct

        # notional sized so that (notional × sl_dist = risk_usd)
        notional = risk_usd / sl_dist
        # cap to available margin × leverage
        notional = min(notional, balance * LEVERAGE)

        entry_fee = _entry_fee(notional)
        if entry_fee > balance * 0.05:
            continue  # sanity check

        balance_before = balance

        if direction == "long":
            sl_px  = entry_px * (1 - sl_dist)
            tp1_px = entry_px * (1 + sl_dist * TP1_R)
            tp2_px = entry_px * (1 + sl_dist * TP2_R)
        else:
            sl_px  = entry_px * (1 + sl_dist)
            tp1_px = entry_px * (1 - sl_dist * TP1_R)
            tp2_px = entry_px * (1 - sl_dist * TP2_R)

        t = Trade(
            symbol=sym_name,
            direction=direction,
            entry_time=idx[i + 1],
            entry_px=entry_px,
            sl_px=sl_px,
            tp1_px=tp1_px,
            tp2_px=tp2_px,
            size=notional / entry_px,
            notional=notional,
            risk_pct=risk_pct,
            score=score,
            balance_before=balance_before,
        )
        t.pnl_usd = -entry_fee  # pre-deduct entry fee
        open_trade = t
        tp1_hit = False

    # Close any open trade at last bar
    if open_trade is not None:
        close_px = float(arr_close[-1])
        t = open_trade
        remaining = 0.5 if tp1_hit else 1.0
        if t.direction == "long":
            pnl = (close_px - t.entry_px) / t.entry_px * t.notional * remaining
        else:
            pnl = (t.entry_px - close_px) / t.entry_px * t.notional * remaining
        pnl -= _exit_fee(t.notional * remaining)
        t.pnl_usd += pnl
        t.exit_px = close_px
        t.exit_time = idx[-1]
        t.exit_reason = "END"
        balance_ref[0] = max(1.0, balance_ref[0] + t.pnl_usd)
        trades.append(t)

    # Recalculate balance_before/after sequentially using starting INITIAL_BAL
    running = INITIAL_BAL
    for tr in trades:
        tr.balance_before = running
        running = max(1.0, running + tr.pnl_usd)
        tr.balance_after = running
        tr.pnl_pct = tr.pnl_usd / tr.balance_before * 100 if tr.balance_before > 0 else 0.0

    balance_ref[0] = running
    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio Simulation (each symbol independent, equal-weight portfolio)
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    symbols: list[SymbolConfig],
    data_dir_override: Optional[str] = None,
) -> tuple[list[Trade], pd.DataFrame]:
    """
    Simulate each symbol independently with INITIAL_BAL.
    Portfolio equity = equal-weight combination:
      each symbol contributes PnL × (1/N_symbols).

    Returns (all_trades, portfolio_equity_df).
    """
    all_trades: list[Trade] = []
    active_syms: list[str] = []

    for sym in symbols:
        if data_dir_override:
            sym.data_dir = sym.data_dir.replace("/tmp/backtest_data", data_dir_override)

        print(f"  Loading {sym.name} 15m...", end=" ", flush=True)
        df15 = load_ohlcv(sym, "15m")
        if df15.empty:
            print("NO DATA — skipped")
            continue
        print(f"{len(df15):,} bars", end=", ", flush=True)

        print("1h...", end=" ", flush=True)
        df1h = load_ohlcv(sym, "1h")
        print(f"{len(df1h):,} bars", end=" ", flush=True)

        print("indicators...", end=" ", flush=True)
        df = compute_indicators(df15, df1h)
        df = score_signals(df)

        long_signals  = df["sig_long"].sum()
        short_signals = df["sig_short"].sum()
        print(f"signals L={long_signals} S={short_signals}", end=" ", flush=True)

        # Each symbol gets its own fresh balance simulation
        sym_balance = [INITIAL_BAL]
        trades = simulate(sym.name, df, sym_balance)
        all_trades.extend(trades)
        active_syms.append(sym.name)
        total_pnl = sym_balance[0] - INITIAL_BAL
        ret_pct   = total_pnl / INITIAL_BAL * 100
        print(f"→ {len(trades)} trades  PnL={total_pnl:+,.0f} ({ret_pct:+.1f}%)")

    # ── Portfolio equity (equal-weight combination) ───────────────────────────
    if not all_trades:
        equity = pd.DataFrame({"equity": [INITIAL_BAL]},
                              index=pd.to_datetime(["2026-01-01"], utc=True))
        return all_trades, equity

    n = max(len(active_syms), 1)
    weight = 1.0 / n   # equal weight

    trade_rows = []
    running = INITIAL_BAL
    for tr in sorted(all_trades, key=lambda x: x.exit_time or pd.Timestamp.max.tz_localize("UTC")):
        portfolio_pnl = tr.pnl_usd * weight
        running += portfolio_pnl
        trade_rows.append({"time": tr.exit_time, "equity": running, "symbol": tr.symbol})

    equity = (pd.DataFrame(trade_rows)
                .set_index("time")
                .sort_index()[["equity"]])
    return all_trades, equity


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(trades: list[Trade], equity: pd.DataFrame) -> dict:
    if not trades:
        return {}

    pnl = [t.pnl_usd for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]

    eq = equity["equity"].values
    drawdowns = []
    peak = INITIAL_BAL
    for e in np.concatenate([[INITIAL_BAL], eq]):
        if e > peak:
            peak = e
        drawdowns.append((peak - e) / peak * 100)
    max_dd = max(drawdowns) if drawdowns else 0.0

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_return = (equity["equity"].iloc[-1] - INITIAL_BAL) / INITIAL_BAL * 100

    # Sharpe (daily returns)
    eq_full = pd.Series([INITIAL_BAL] + list(eq))
    daily_ret = eq_full.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0.0

    # Calmar
    n_months = len(set(
        (t.exit_time.year, t.exit_time.month)
        for t in trades if t.exit_time
    )) or 1
    ann_return = total_return / n_months * 12
    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    return {
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       len(wins) / len(trades) * 100,
        "avg_win":        np.mean(wins) if wins else 0.0,
        "avg_loss":       np.mean(losses) if losses else 0.0,
        "profit_factor":  profit_factor,
        "total_pnl":      sum(pnl),
        "total_return":   total_return,
        "max_drawdown":   max_dd,
        "sharpe":         sharpe,
        "calmar":         calmar,
        "ann_return":     ann_return,
        "tp1_hit":        sum(1 for t in trades if t.tp1_hit),
        "sl_exits":       sum(1 for t in trades if t.exit_reason == "SL"),
        "tp2_exits":      sum(1 for t in trades if t.exit_reason == "TP2"),
        "longs":          sum(1 for t in trades if t.direction == "long"),
        "shorts":         sum(1 for t in trades if t.direction == "short"),
        "avg_score":      np.mean([t.score for t in trades]),
    }


def per_symbol_stats(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for sym in sorted(set(t.symbol for t in trades)):
        st = [t for t in trades if t.symbol == sym]
        pnl = [t.pnl_usd for t in st]
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        rows.append({
            "Symbol":       sym,
            "Trades":       len(st),
            "Win Rate %":   round(len(wins) / len(st) * 100, 1),
            "Total PnL $":  round(sum(pnl), 2),
            "Avg Win $":    round(np.mean(wins) if wins else 0, 2),
            "Avg Loss $":   round(np.mean(losses) if losses else 0, 2),
            "Profit Factor": round(sum(wins) / abs(sum(losses)) if losses else float("inf"), 2),
            "Best Trade $": round(max(pnl) if pnl else 0, 2),
            "Worst Trade $": round(min(pnl) if pnl else 0, 2),
        })
    return pd.DataFrame(rows)


def monthly_pnl(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        if t.exit_time:
            rows.append({
                "year":   t.exit_time.year,
                "month":  t.exit_time.month,
                "symbol": t.symbol,
                "pnl":    t.pnl_usd,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = (df.groupby(["year", "month"])["pnl"]
               .sum()
               .reset_index()
               .pivot(index="year", columns="month", values="pnl")
               .fillna(0))
    pivot.columns = [f"M{m}" for m in pivot.columns]
    return pivot


# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_equity_curve(trades: list[Trade], equity: pd.DataFrame) -> str:
    if not HAS_MPL or equity.empty:
        return ""

    # Build drawdown series
    eq_vals = np.concatenate([[INITIAL_BAL], equity["equity"].values])
    peaks = np.maximum.accumulate(eq_vals)
    dd    = (peaks - eq_vals) / peaks * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=False)
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")

    # Equity
    times = [pd.Timestamp("2026-01-01", tz="UTC")] + list(equity.index)
    ax1.plot(times, eq_vals, color="#58a6ff", lw=1.8, label="Portfolio Equity")
    ax1.axhline(INITIAL_BAL, color="#484f58", lw=0.8, ls="--", label="Initial Capital")
    ax1.fill_between(times, INITIAL_BAL, eq_vals,
                     where=[e >= INITIAL_BAL for e in eq_vals],
                     alpha=0.15, color="#3fb950")
    ax1.fill_between(times, INITIAL_BAL, eq_vals,
                     where=[e < INITIAL_BAL for e in eq_vals],
                     alpha=0.15, color="#f85149")
    ax1.set_ylabel("Balance (USD)", color="#e6edf3")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor="#21262d", labelcolor="#e6edf3")
    ax1.set_title("Portfolio Equity Curve", color="#e6edf3", pad=12)

    # Drawdown
    ax2.fill_between(range(len(dd)), 0, -dd, color="#f85149", alpha=0.6)
    ax2.set_ylabel("Drawdown %", color="#e6edf3")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.set_title("Drawdown", color="#8b949e", pad=6)
    ax2.invert_yaxis()

    plt.tight_layout(pad=1.5)
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def chart_per_symbol(trades: list[Trade]) -> str:
    if not HAS_MPL or not trades:
        return ""

    symbols = sorted(set(t.symbol for t in trades))
    ncols = min(3, len(symbols))
    nrows = (len(symbols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    fig.patch.set_facecolor("#0d1117")
    if len(symbols) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, sym in enumerate(symbols):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")

        sym_trades = sorted([t for t in trades if t.symbol == sym],
                            key=lambda x: x.exit_time or pd.Timestamp.max.tz_localize("UTC"))
        cumulative = np.cumsum([0] + [t.pnl_usd for t in sym_trades])
        color = "#3fb950" if cumulative[-1] >= 0 else "#f85149"
        ax.plot(cumulative, color=color, lw=1.5)
        ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.15, color=color)
        ax.axhline(0, color="#484f58", lw=0.8, ls="--")
        ax.set_title(sym, color="#e6edf3")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        wr = sum(1 for t in sym_trades if t.pnl_usd > 0) / max(len(sym_trades), 1) * 100
        ax.set_xlabel(f"{len(sym_trades)} trades | WR {wr:.0f}%", color="#8b949e", fontsize=9)

    # Hide unused subplots
    for idx in range(len(symbols), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Per-Symbol Cumulative PnL", color="#e6edf3", y=1.01)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def chart_monthly_heatmap(trades: list[Trade]) -> str:
    if not HAS_MPL or not trades:
        return ""

    pivot = monthly_pnl(trades)
    if pivot.empty:
        return ""

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 1.2)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    cmap = LinearSegmentedColormap.from_list("pnl", ["#f85149", "#21262d", "#3fb950"])
    vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 1)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color="#e6edf3")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str), color="#e6edf3")

    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            ax.text(c, r, f"${val:,.0f}", ha="center", va="center",
                    color="#e6edf3", fontsize=8,
                    fontweight="bold" if abs(val) > vmax * 0.5 else "normal")

    plt.colorbar(im, ax=ax, label="PnL (USD)")
    ax.set_title("Monthly PnL Heatmap", color="#e6edf3", pad=10)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def chart_pnl_distribution(trades: list[Trade]) -> str:
    if not HAS_MPL or not trades:
        return ""

    pnl = [t.pnl_usd for t in trades]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")

    # Histogram
    bins = min(40, len(pnl) // 2 + 1)
    ax1.hist([p for p in pnl if p > 0], bins=bins, color="#3fb950", alpha=0.7, label="Wins")
    ax1.hist([p for p in pnl if p <= 0], bins=bins, color="#f85149", alpha=0.7, label="Losses")
    ax1.axvline(0, color="#ffffff", lw=0.8, ls="--")
    ax1.set_title("Trade PnL Distribution", color="#e6edf3")
    ax1.set_xlabel("PnL (USD)", color="#8b949e")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor="#21262d", labelcolor="#e6edf3")

    # Win/Loss by symbol
    syms = sorted(set(t.symbol for t in trades))
    x = np.arange(len(syms))
    wins_count  = [sum(1 for t in trades if t.symbol == s and t.pnl_usd > 0) for s in syms]
    loss_count  = [sum(1 for t in trades if t.symbol == s and t.pnl_usd <= 0) for s in syms]
    ax2.bar(x - 0.2, wins_count,  0.4, label="Wins",   color="#3fb950", alpha=0.8)
    ax2.bar(x + 0.2, loss_count,  0.4, label="Losses", color="#f85149", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(syms, color="#e6edf3")
    ax2.set_title("Win vs Loss by Symbol", color="#e6edf3")
    ax2.legend(facecolor="#21262d", labelcolor="#e6edf3")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def chart_score_vs_pnl(trades: list[Trade]) -> str:
    if not HAS_MPL or not trades:
        return ""

    scores = [t.score for t in trades]
    pnl    = [t.pnl_usd for t in trades]
    colors = ["#3fb950" if p > 0 else "#f85149" for p in pnl]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")

    ax.scatter(scores, pnl, c=colors, alpha=0.6, s=30)
    ax.axhline(0, color="#484f58", lw=0.8, ls="--")
    ax.axvline(SIGNAL_THRESHOLD, color="#d29922", lw=0.8, ls="--", label=f"Threshold={SIGNAL_THRESHOLD}")
    ax.set_xlabel("Signal Score", color="#8b949e")
    ax.set_ylabel("PnL (USD)", color="#8b949e")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Signal Score vs Trade PnL", color="#e6edf3")
    ax.legend(facecolor="#21262d", labelcolor="#e6edf3")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════════

def _metric_card(label: str, value: str, color: str = "#e6edf3") -> str:
    return f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="color:{color}">{value}</div>
    </div>"""


def build_html_report(
    trades: list[Trade],
    equity: pd.DataFrame,
    stats: dict,
    sym_stats: pd.DataFrame,
) -> str:
    chart_eq  = chart_equity_curve(trades, equity)
    chart_sym = chart_per_symbol(trades)
    chart_mth = chart_monthly_heatmap(trades)
    chart_pnl = chart_pnl_distribution(trades)
    chart_sc  = chart_score_vs_pnl(trades)

    def img(b64: str, title: str = "") -> str:
        if not b64:
            return f'<p style="color:#8b949e">Chart unavailable (matplotlib not installed)</p>'
        return f'<figure><figcaption>{title}</figcaption><img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px"/></figure>'

    # ── Stats cards ──────────────────────────────────────────────────────────
    total_return = stats.get("total_return", 0)
    cards_html = "".join([
        _metric_card("Total Return", f"{total_return:+.1f}%",
                     "#3fb950" if total_return >= 0 else "#f85149"),
        _metric_card("Total PnL", f"${stats.get('total_pnl', 0):+,.0f}",
                     "#3fb950" if stats.get("total_pnl", 0) >= 0 else "#f85149"),
        _metric_card("Win Rate", f"{stats.get('win_rate', 0):.1f}%"),
        _metric_card("Profit Factor", f"{stats.get('profit_factor', 0):.2f}",
                     "#3fb950" if stats.get("profit_factor", 1) >= 1.5 else "#d29922"),
        _metric_card("Max Drawdown", f"{stats.get('max_drawdown', 0):.1f}%",
                     "#f85149" if stats.get("max_drawdown", 0) > 20 else "#d29922"),
        _metric_card("Sharpe Ratio", f"{stats.get('sharpe', 0):.2f}"),
        _metric_card("Calmar Ratio",  f"{stats.get('calmar', 0):.2f}"),
        _metric_card("Total Trades",  f"{stats.get('total_trades', 0)}"),
        _metric_card("Avg Win",  f"${stats.get('avg_win', 0):+,.0f}", "#3fb950"),
        _metric_card("Avg Loss", f"${stats.get('avg_loss', 0):+,.0f}", "#f85149"),
        _metric_card("TP1 Hits", f"{stats.get('tp1_hit', 0)} ({stats.get('tp1_hit',0)/max(stats.get('total_trades',1),1)*100:.0f}%)"),
        _metric_card("Avg Signal Score", f"{stats.get('avg_score', 0):.1f}"),
    ])

    # ── Per-symbol table ──────────────────────────────────────────────────────
    if not sym_stats.empty:
        thr = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in sym_stats.columns) + "</tr></thead>"
        tbody = "<tbody>"
        for _, row in sym_stats.iterrows():
            pnl_col = row["Total PnL $"]
            color = "#3fb950" if pnl_col >= 0 else "#f85149"
            cells = "".join(
                f'<td style="color:{color}">{row[c]}</td>' if c == "Total PnL $"
                else f"<td>{row[c]}</td>"
                for c in sym_stats.columns
            )
            tbody += f"<tr>{cells}</tr>"
        tbody += "</tbody>"
        sym_table = f'<table class="data-table">{thr}{tbody}</table>'
    else:
        sym_table = "<p>No per-symbol data.</p>"

    # ── Trade log (last 50) ───────────────────────────────────────────────────
    recent = sorted(trades, key=lambda x: x.exit_time or pd.Timestamp.min.tz_localize("UTC"),
                    reverse=True)[:50]
    trade_rows_html = ""
    for t in recent:
        color = "#3fb950" if t.pnl_usd > 0 else "#f85149"
        entry_str = t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "-"
        exit_str  = t.exit_time.strftime("%Y-%m-%d %H:%M")  if t.exit_time  else "-"
        trade_rows_html += f"""
        <tr>
          <td>{t.symbol}</td>
          <td class="{"long" if t.direction=="long" else "short"}">{t.direction.upper()}</td>
          <td>{entry_str}</td>
          <td>{exit_str}</td>
          <td>${t.entry_px:,.4f}</td>
          <td>${t.exit_px:,.4f}</td>
          <td>{t.exit_reason}</td>
          <td>{t.score:.0f}</td>
          <td>{t.risk_pct*100:.1f}%</td>
          <td style="color:{color}">${t.pnl_usd:+,.2f}</td>
          <td style="color:{color}">{t.pnl_pct:+.2f}%</td>
        </tr>"""

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 24px; }
    h1 { font-size: 1.8rem; margin-bottom: 4px; color: #58a6ff; }
    h2 { font-size: 1.2rem; margin: 28px 0 12px; color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
    .subtitle { color: #8b949e; margin-bottom: 24px; font-size: 0.9rem; }
    .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; min-width: 140px; flex: 1; }
    .card-label { font-size: 0.75rem; color: #8b949e; margin-bottom: 4px; text-transform: uppercase; letter-spacing: .05em; }
    .card-value { font-size: 1.4rem; font-weight: 700; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .data-table th, .data-table td { padding: 8px 12px; border: 1px solid #21262d; text-align: right; }
    .data-table th { background: #161b22; color: #8b949e; font-weight: 600; text-align: center; }
    .data-table tr:hover { background: #161b22; }
    .long { color: #3fb950; }
    .short { color: #f85149; }
    figure { margin: 0 0 24px; }
    figcaption { color: #8b949e; font-size: 0.8rem; margin-bottom: 6px; }
    .section { margin-bottom: 32px; }
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TrendContV2 Backtest Report</title>
<style>{css}</style>
</head>
<body>
<h1>TrendContV2 Strategy — Backtest Report</h1>
<p class="subtitle">
  Period: Jan–Jun 2026 &nbsp;|&nbsp;
  Symbols: {", ".join(sorted(set(t.symbol for t in trades)))} &nbsp;|&nbsp;
  Initial Balance: ${INITIAL_BAL:,.0f} &nbsp;|&nbsp;
  Leverage: {LEVERAGE}× &nbsp;|&nbsp;
  Risk/Trade: {RISK_MIN*100:.0f}–{RISK_MAX*100:.0f}%
</p>

<h2>Performance Summary</h2>
<div class="cards">{cards_html}</div>

<div class="section">
{img(chart_eq)}
</div>

<h2>Per-Symbol Performance</h2>
<div class="section">{sym_table}</div>

<div class="section">
{img(chart_sym)}
</div>

<h2>Monthly PnL Heatmap</h2>
<div class="section">
{img(chart_mth)}
</div>

<h2>Analytics</h2>
<div class="section">
{img(chart_pnl)}
</div>
<div class="section">
{img(chart_sc)}
</div>

<h2>Trade Log (last 50 trades)</h2>
<div style="overflow-x:auto">
<table class="data-table">
<thead>
  <tr>
    <th>Symbol</th><th>Dir</th><th>Entry Time</th><th>Exit Time</th>
    <th>Entry $</th><th>Exit $</th><th>Reason</th>
    <th>Score</th><th>Risk%</th><th>PnL $</th><th>PnL %Bal</th>
  </tr>
</thead>
<tbody>{trade_rows_html}</tbody>
</table>
</div>

<p style="color:#484f58;font-size:0.75rem;margin-top:32px">
  Generated by backtest_v2.py &mdash; TrendContV2 strategy simulation.
  All results are hypothetical and do not guarantee future performance.
</p>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TrendContV2 Backtest")
    parser.add_argument("--data-dir", default="/tmp/backtest_data",
                        help="Root directory of extracted backtest data")
    parser.add_argument("--out", default=None,
                        help="Output HTML file path (default: scratchpad dir)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated subset e.g. BTC,XAU,ETH")
    parser.add_argument("--balance", type=float, default=INITIAL_BAL,
                        help="Initial balance in USD")
    parser.add_argument("--leverage", type=int, default=LEVERAGE)
    args = parser.parse_args()

    # Update module constants
    globals().update({"INITIAL_BAL": args.balance, "LEVERAGE": args.leverage})

    # Select symbols
    syms = ALL_SYMBOLS
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",")}
        syms = [s for s in ALL_SYMBOLS if s.name in wanted]
        if not syms:
            print(f"No matching symbols found. Available: {[s.name for s in ALL_SYMBOLS]}")
            sys.exit(1)

    # Update data_dir if overridden
    if args.data_dir != "/tmp/backtest_data":
        for s in syms:
            s.data_dir = s.data_dir.replace("/tmp/backtest_data", args.data_dir)

    print(f"=== TrendContV2 Backtest ===")
    print(f"Symbols: {[s.name for s in syms]}")
    print(f"Balance: ${INITIAL_BAL:,.0f}  Leverage: {LEVERAGE}x  Risk: {RISK_MIN*100:.0f}-{RISK_MAX*100:.0f}%")
    print()

    trades, equity = run_backtest(syms)

    if not trades:
        print("\nNo trades generated — check data paths and signal parameters.")
        sys.exit(0)

    stats     = compute_stats(trades, equity)
    sym_stats = per_symbol_stats(trades)

    print("\n" + "=" * 55)
    print(f"  Total Trades:     {stats['total_trades']}")
    print(f"  Win Rate:         {stats['win_rate']:.1f}%")
    print(f"  Profit Factor:    {stats['profit_factor']:.2f}")
    print(f"  Total PnL:        ${stats['total_pnl']:+,.2f}")
    print(f"  Total Return:     {stats['total_return']:+.1f}%")
    print(f"  Max Drawdown:     {stats['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio:     {stats['sharpe']:.2f}")
    print(f"  Calmar Ratio:     {stats['calmar']:.2f}")
    print(f"  TP1 Hit Rate:     {stats['tp1_hit']}/{stats['total_trades']}")
    print(f"  SL Exits:         {stats['sl_exits']}")
    print("=" * 55)

    print("\nPer-symbol:")
    print(sym_stats.to_string(index=False))

    # Write report
    if args.out:
        out_path = args.out
    else:
        scratchpad = "/tmp/claude-0/-home-user-my-virtual-office/a7bf9841-f7c4-5fca-9bbf-0f57f984e072/scratchpad"
        os.makedirs(scratchpad, exist_ok=True)
        out_path = os.path.join(scratchpad, "backtest_report.html")

    html = build_html_report(trades, equity, stats, sym_stats)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
