#!/usr/bin/env python3
"""
5-Month Multi-Strategy Backtest  (vectorized)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategies  : SwingReversal · CPKRegime · HybridSwing · Intern
Symbols     : BTC/USDT · ETH/USDT  (configurable via BT_SYMBOLS env)
Slots       : max 3 open positions; max 1 per symbol
Chief       : simulated — require ≥CHIEF_N_AGREE strategies to agree
Data source : CCXT (binance default); falls back to synthetic GBM
Timeframes  : 1h (primary), 30m + 15m (Intern MTF bias)
Fees        : 0.10% entry + 0.10% exit

Performance: indicators precomputed once on full dataset (O(n));
             simulation loop uses O(1) array lookups per bar.

Usage:
    cd app && python backtest_all.py           # run (fetches & caches data)
    cd app && python backtest_all.py --clear   # delete cache and re-fetch
"""

import asyncio
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import ccxt.async_support as ccxt

from trading.connectors.base import OHLCV, to_heikin_ashi
from trading.strategies.base import BaseStrategy

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOLS       = os.environ.get("BT_SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
EXCHANGE_ID   = os.environ.get("EXCHANGE",   "binance")
MONTHS        = int(os.environ.get("BT_MONTHS", "5"))
TRADE_USDT    = float(os.environ.get("TRADE_AMOUNT_USDT", "100"))
MAX_POSITIONS = 3
FEE_RT        = 0.0020   # 0.10% in + 0.10% out
CACHE_FILE    = Path(__file__).parent / ".bt_cache_5m.json"

CHIEF_N_AGREE   = int(os.environ.get("CHIEF_N_AGREE",  "2"))
CHIEF_LOOKFWD   = int(os.environ.get("CHIEF_LOOKFWD", "100"))
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY",    "")
BINANCE_SECRET  = os.environ.get("BINANCE_SECRET",     "")


# ── Data fetching ─────────────────────────────────────────────────────────────
async def _paginate(ex, symbol: str, tf: str, since_ms: int) -> list[OHLCV]:
    result, cur = [], since_ms
    while True:
        try:
            batch = await ex.fetch_ohlcv(symbol, tf, since=cur, limit=1000)
        except Exception as e:
            print(f"    WARN fetch {symbol} {tf}: {e}")
            break
        if not batch:
            break
        result.extend(batch)
        if len(batch) < 1000:
            break
        cur = batch[-1][0] + 1
        await asyncio.sleep(0.3)
    seen = {}
    for row in sorted(result, key=lambda r: r[0]):
        seen[row[0]] = row
    return [OHLCV(ts, o, h, l, c, v) for ts, o, h, l, c, v in seen.values()]


def _gen_synthetic(sym: str, tf_seconds: int, n_bars: int,
                   start_price: float, mu: float, sigma: float,
                   since_ms: int, rng_seed: int) -> list[OHLCV]:
    """Realistic GBM price series with trending + ranging regimes."""
    import random
    rng = random.Random(rng_seed)
    ms_step = tf_seconds * 1000
    candles: list[OHLCV] = []
    price = start_price
    trend_bias = 0.0
    regime_len = 0

    for i in range(n_bars):
        if regime_len <= 0:
            regime_len = rng.randint(24, 96)
            trend_bias = rng.gauss(0, sigma * 0.3)
        regime_len -= 1
        ret   = trend_bias + sigma * rng.gauss(0, 1)
        open_ = price
        close = open_ * math.exp(ret)
        hi    = max(open_, close) * (1 + abs(rng.gauss(0, sigma * 0.4)))
        lo    = min(open_, close) * (1 - abs(rng.gauss(0, sigma * 0.4)))
        vol   = start_price * 10 * abs(rng.gauss(1, 0.5))
        candles.append(OHLCV(
            timestamp=since_ms + i * ms_step,
            open=round(open_, 2), high=round(hi, 2),
            low=round(lo, 2), close=round(close, 2),
            volume=round(max(vol, 0.01), 4),
        ))
        price = close
    return candles


def _gen_all_synthetic() -> dict[str, dict[str, list[OHLCV]]]:
    since_ms = int((time.time() - (MONTHS + 0.5) * 30 * 86400) * 1000)
    tf_cfg = {
        "BTC/USDT": {"start": 95_000.0, "mu": 0.00010, "sigma": 0.0090},
        "ETH/USDT": {"start":  3_200.0, "mu": 0.00012, "sigma": 0.0110},
    }
    tf_secs = {"1h": 3600, "30m": 1800, "15m": 900}
    data: dict[str, dict[str, list[OHLCV]]] = {}
    for i, sym in enumerate(SYMBOLS):
        cfg = tf_cfg.get(sym, {"start": 1.0, "mu": 0.0001, "sigma": 0.01})
        data[sym] = {}
        for tf, secs in tf_secs.items():
            n = (MONTHS + 1) * 30 * 86400 // secs
            data[sym][tf] = _gen_synthetic(
                sym, secs, n, cfg["start"], cfg["mu"], cfg["sigma"],
                since_ms, rng_seed=hash(sym + tf) % 10000,
            )
    return data


async def fetch_or_load() -> tuple[dict, bool]:
    if CACHE_FILE.exists():
        print("  Loading cached OHLCV data...")
        raw = json.loads(CACHE_FILE.read_text())
        is_syn = raw.get("__synthetic__", False)
        data = {
            sym: {tf: [OHLCV(*row) for row in raw[sym][tf]] for tf in raw[sym]
                  if tf != "__synthetic__"}
            for sym in raw if sym != "__synthetic__"
        }
        return data, is_syn

    since_ms = int((time.time() - (MONTHS + 0.5) * 30 * 86400) * 1000)
    print(f"  Trying to fetch {MONTHS}-month OHLCV from {EXCHANGE_ID}...")
    data: dict[str, dict[str, list[OHLCV]]] = {}
    is_synthetic = False
    fetch_ok = False

    try:
        ex_cls = getattr(ccxt, EXCHANGE_ID)
        ex_cfg: dict = {"enableRateLimit": True}
        if BINANCE_API_KEY:
            ex_cfg["apiKey"] = BINANCE_API_KEY
            ex_cfg["secret"] = BINANCE_SECRET
        ex = ex_cls(ex_cfg)
        if EXCHANGE_ID == "okx":
            ex.options["defaultType"] = "margin"

        for sym in SYMBOLS:
            data[sym] = {}
            for tf in ("1h", "30m", "15m"):
                print(f"    {sym} {tf} ... ", end="", flush=True)
                bars = await _paginate(ex, sym, tf, since_ms)
                data[sym][tf] = bars
                print(f"{len(bars)} bars")
        await ex.close()

        if all(len(data.get(s, {}).get("1h", [])) >= 1000 for s in SYMBOLS):
            fetch_ok = True
    except Exception as e:
        print(f"  Exchange fetch failed: {e}")

    if not fetch_ok:
        print("  ⚠ Exchange unreachable — using SYNTHETIC price data (GBM)")
        print("    Run locally with internet access for real results.")
        data = _gen_all_synthetic()
        is_synthetic = True

    raw: dict = {"__synthetic__": is_synthetic}
    for sym in data:
        raw[sym] = {tf: [[c.timestamp, c.open, c.high, c.low, c.close, c.volume]
                         for c in data[sym][tf]]
                    for tf in data[sym]}
    CACHE_FILE.write_text(json.dumps(raw))
    label = "synthetic" if is_synthetic else "real"
    print(f"  Data cached ({label}) → {CACHE_FILE.name}")
    return data, is_synthetic


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol:      str
    strategy:    str
    entry_price: float
    entry_bar:   int
    sl:          float
    tp:          float
    amount:      float


@dataclass
class Trade:
    symbol:      str
    strategy:    str
    entry_price: float
    exit_price:  float
    sl:          float
    tp:          float
    amount:      float
    pnl_usdt:    float
    pnl_pct:     float
    reason:      str     # "tp" | "sl" | "sell_signal" | "end"
    bars_held:   int


# ── Indicator precomputation ──────────────────────────────────────────────────
def _precompute(sym: str, ha: dict) -> dict:
    """Compute all indicators on the full dataset once.  O(n) per indicator."""
    ha_1h  = ha[sym]["1h"]
    ha_15m = ha[sym].get("15m", [])
    ha_30m = ha[sym].get("30m", [])
    n = len(ha_1h)

    closes  = np.array([c.close  for c in ha_1h], dtype=float)
    opens   = np.array([c.open   for c in ha_1h], dtype=float)
    highs   = np.array([c.high   for c in ha_1h], dtype=float)
    lows    = np.array([c.low    for c in ha_1h], dtype=float)
    volumes = np.array([c.volume for c in ha_1h], dtype=float)

    ema80       = BaseStrategy.ema(closes.tolist(), 80)
    ema200      = BaseStrategy.ema(closes.tolist(), 200)
    rsi14       = BaseStrategy.rsi(closes.tolist(), 14)
    vol_ma      = BaseStrategy.sma(volumes.tolist(), 20)
    atr14       = BaseStrategy.atr(ha_1h, 14)
    _, _, mhist = BaseStrategy.macd(closes.tolist(), 12, 26, 9)
    hma15       = BaseStrategy.hma(closes.tolist(), 15)

    # MACD cross-up: hist[j-1]<0 → hist[j]>0; 5-bar lookback window
    mh    = mhist.astype(float)
    valid = ~np.isnan(mh)
    prev_valid = np.concatenate([[False], valid[:-1]])
    prev_mh    = np.concatenate([[0.0], mh[:-1]])
    cross_evt  = valid & prev_valid & (prev_mh < 0) & (mh > 0)
    macd_cross = cross_evt.copy()
    for k in range(1, 5):           # OR the event forward up to 5 bars
        if n > k:
            macd_cross[k:] |= cross_evt[:-k]

    # Candle bullish: (close − low) / (high − low) ≥ 0.50
    rng        = highs - lows
    candle_bull = np.where(rng > 1e-6,
                           (closes - lows) / np.where(rng > 0, rng, 1.0) >= 0.50,
                           False)

    # MTF bias arrays (InternStrategy) aligned to 1h timestamps
    ts_1h = np.array([c.timestamp for c in ha_1h], dtype=np.int64)

    def _bias_at_1h(ha_tf: list) -> np.ndarray:
        """Per-1h-bar bias score from a lower TF dataset."""
        if len(ha_tf) < 52:
            return np.zeros(n, dtype=float)
        c_tf  = np.array([x.close for x in ha_tf], dtype=float)
        ts_tf = np.array([x.timestamp for x in ha_tf], dtype=np.int64)
        e20   = BaseStrategy.ema(c_tf.tolist(), 20)
        e50   = BaseStrategy.ema(c_tf.tolist(), 50)
        r14   = BaseStrategy.rsi(c_tf.tolist(), 14)
        score = (np.where(c_tf > e20, 1.0, -1.0) +
                 np.where(e20 > e50,  1.0, -1.0) +
                 np.where(r14 > 55,   1.0, np.where(r14 < 45, -1.0, 0.0)))
        score[np.isnan(e20) | np.isnan(e50) | np.isnan(r14)] = 0.0
        # For each 1h bar find the latest tf bar at-or-before that timestamp
        idx      = np.searchsorted(ts_tf, ts_1h + 1) - 1
        valid_tf = idx >= 51
        safe_idx = np.clip(idx, 0, len(score) - 1)
        return np.where(valid_tf, score[safe_idx], 0.0)

    bias_15m = _bias_at_1h(ha_15m)
    bias_30m = _bias_at_1h(ha_30m)

    return {
        "n": n,
        "closes": closes, "opens": opens, "highs": highs,
        "lows": lows, "volumes": volumes,
        "ema80": ema80, "ema200": ema200, "rsi14": rsi14,
        "vol_ma": vol_ma, "atr14": atr14,
        "macd_cross": macd_cross, "hma15": hma15,
        "candle_bull": candle_bull,
        "bias_15m": bias_15m, "bias_30m": bias_30m,
        "ts_1h": ts_1h,
    }


def _precompute_ext(sym: str, ha: dict) -> dict:
    """Additional indicators for ProfitableBot (1H/4H) and ScalpTrendBot (15m/1H/4H)."""
    ha_1h  = ha[sym]["1h"]
    ha_15m = ha[sym].get("15m", [])
    n1h    = len(ha_1h)

    closes_1h  = np.array([c.close  for c in ha_1h], dtype=float)
    volumes_1h = np.array([c.volume for c in ha_1h], dtype=float)
    ts_1h      = np.array([c.timestamp for c in ha_1h], dtype=np.int64)

    # ── 1H indicators ──────────────────────────────────────────────────────────
    ema20_1h = BaseStrategy.ema(closes_1h.tolist(), 20)
    ema50_1h = BaseStrategy.ema(closes_1h.tolist(), 50)
    rsi14_1h = BaseStrategy.rsi(closes_1h.tolist(), 14)
    atr14_1h = BaseStrategy.atr(ha_1h, 14)
    volma20_1h = BaseStrategy.sma(volumes_1h.tolist(), 20)
    _, _, mhist_1h = BaseStrategy.macd(closes_1h.tolist(), 12, 26, 9)
    mhist_1h = mhist_1h.astype(float)
    bb_lower_1h = _bb_lower(closes_1h, 20, 2.0)

    # ── 4H data (resample: group every 4 1H bars) ─────────────────────────────
    n4h = n1h // 4
    closes_4h = np.array([ha_1h[k * 4 + 3].close for k in range(n4h)], dtype=float)
    ema20_4h_raw = BaseStrategy.ema(closes_4h.tolist(), 20)
    ema50_4h_raw = BaseStrategy.ema(closes_4h.tolist(), 50)

    # Expand 4H EMAs back to 1H index (bar i → 4H index i//4)
    ema20_4h = np.full(n1h, np.nan)
    ema50_4h = np.full(n1h, np.nan)
    for i in range(n1h):
        k = i // 4
        if k < n4h:
            ema20_4h[i] = ema20_4h_raw[k]
            ema50_4h[i] = ema50_4h_raw[k]

    # 4H EMA20 slope: change over 1 4H bar (= 4 1H bars)
    ema20_4h_slope = np.full(n1h, np.nan)
    for i in range(4, n1h):
        if not (math.isnan(ema20_4h[i]) or math.isnan(ema20_4h[i - 4])):
            ema20_4h_slope[i] = ema20_4h[i] - ema20_4h[i - 4]

    # ── 15m indicators ─────────────────────────────────────────────────────────
    if len(ha_15m) < 30:
        return {
            "ema20_1h": ema20_1h, "ema50_1h": ema50_1h,
            "rsi14_1h": rsi14_1h, "atr14_1h": atr14_1h,
            "volma20_1h": volma20_1h, "mhist_1h": mhist_1h,
            "bb_lower_1h": bb_lower_1h,
            "ema20_4h": ema20_4h, "ema50_4h": ema50_4h,
            "ema20_4h_slope": ema20_4h_slope,
            # 15m stubs
            "c15": np.array([]), "h15": np.array([]), "l15": np.array([]),
            "v15": np.array([]), "ts_15m": np.array([], dtype=np.int64),
            "ema9_15m": np.array([]), "ema20_15m": np.array([]),
            "ema50_15m": np.array([]),
            "rsi14_15m": np.array([]), "volma20_15m": np.array([]),
            "atr14_15m": np.array([]),
            "ema20_1h_at15m": np.array([]), "ema50_1h_at15m": np.array([]),
            "ema20_4h_at15m": np.array([]), "ema50_4h_at15m": np.array([]),
        }

    c15   = np.array([b.close  for b in ha_15m], dtype=float)
    h15   = np.array([b.high   for b in ha_15m], dtype=float)
    l15   = np.array([b.low    for b in ha_15m], dtype=float)
    v15   = np.array([b.volume for b in ha_15m], dtype=float)
    ts_15m = np.array([b.timestamp for b in ha_15m], dtype=np.int64)

    ema9_15m   = BaseStrategy.ema(c15.tolist(), 9)
    ema20_15m  = BaseStrategy.ema(c15.tolist(), 20)
    ema50_15m  = BaseStrategy.ema(c15.tolist(), 50)
    rsi14_15m  = BaseStrategy.rsi(c15.tolist(), 14)
    volma20_15m = BaseStrategy.sma(v15.tolist(), 20)
    atr14_15m  = BaseStrategy.atr(ha_15m, 14)

    # Align 1H and 4H indicators to 15m timestamps
    n15 = len(ha_15m)
    ema20_1h_at15m = np.full(n15, np.nan)
    ema50_1h_at15m = np.full(n15, np.nan)
    ema20_4h_at15m = np.full(n15, np.nan)
    ema50_4h_at15m = np.full(n15, np.nan)

    for j in range(n15):
        # Find latest 1H bar where ts_1h[idx] <= ts_15m[j]
        idx = np.searchsorted(ts_1h, ts_15m[j] + 1) - 1
        if 0 <= idx < n1h:
            ema20_1h_at15m[j] = ema20_1h[idx]
            ema50_1h_at15m[j] = ema50_1h[idx]
            ema20_4h_at15m[j] = ema20_4h[idx]
            ema50_4h_at15m[j] = ema50_4h[idx]

    return {
        "ema20_1h": ema20_1h, "ema50_1h": ema50_1h,
        "rsi14_1h": rsi14_1h, "atr14_1h": atr14_1h,
        "volma20_1h": volma20_1h, "mhist_1h": mhist_1h,
        "bb_lower_1h": bb_lower_1h,
        "ema20_4h": ema20_4h, "ema50_4h": ema50_4h,
        "ema20_4h_slope": ema20_4h_slope,
        "c15": c15, "h15": h15, "l15": l15,
        "v15": v15, "ts_15m": ts_15m,
        "ema9_15m": ema9_15m, "ema20_15m": ema20_15m,
        "ema50_15m": ema50_15m,
        "rsi14_15m": rsi14_15m, "volma20_15m": volma20_15m,
        "atr14_15m": atr14_15m,
        "ema20_1h_at15m": ema20_1h_at15m, "ema50_1h_at15m": ema50_1h_at15m,
        "ema20_4h_at15m": ema20_4h_at15m, "ema50_4h_at15m": ema50_4h_at15m,
    }


# ── Standalone per-strategy config ────────────────────────────────────────────
# Each SJ runs on its OWN slot (1 position per symbol). Tuned so every SJ is
# active (10+ trades/month) and the TP/SL geometry yields a healthy win rate.
#   sl/tp  = ATR multiples (tp < sl → WR > 50% on mean-reverting/random moves)
#   rlo/rhi= RSI entry window   vol = volume-vs-SMA20 multiple
#   cd     = bars to wait after an exit   hold = max bars to hold a position
SWING_WARMUP  = 210
INTERN_WARMUP = 60

STANDALONE_CFG: dict[str, dict] = {
    "SwingReversalStrategy":   dict(sl=2.5, tp=1.5, rlo=30.0, rhi=68.0, vol=1.0, cd=1, hold=72),
    "CPKRegimeStrategy":       dict(sl=2.5, tp=1.5, rlo=36.0, rhi=70.0, vol=1.0, cd=1, hold=96),
    "HybridSwingStrategy":     dict(sl=2.5, tp=1.6, rlo=34.0, rhi=70.0, vol=1.0, cd=1, hold=48),
    # Intern — shown twice:
    #   (entry-only): sell disabled — isolates entry quality (TP/SL geometry)
    #   (faithful)  : sell fires on HMA-cross-down, identical to live behaviour
    "InternStrategy":          dict(sl=1.5, tp=3.0, cd=3, hold=60,
                                    bias_lo=1.0, bias_mid=1.0, sell_in_bt=False),
    "InternStrategy(faithful)": dict(sl=1.5, tp=1.5, cd=3, hold=60,
                                     bias_lo=1.0, bias_mid=1.0, sell_in_bt=True),
}
# STRAT_ORDER: key in STANDALONE_CFG; display_name maps to InternStrategy class
STRAT_ORDER = [
    "SwingReversalStrategy",
    "CPKRegimeStrategy",
    "HybridSwingStrategy",
    "InternStrategy",
    "InternStrategy(faithful)",
]
# Map config-key → strategy class name used in _simulate_standalone
STRAT_CLASS = {k: k.split("(")[0] for k in STRAT_ORDER}


def _mk_trade(sym, name, pos, exit_price, reason, bar_i) -> Trade:
    gross = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["amount"] * pos["entry_price"]
    fee   = pos["amount"] * pos["entry_price"] * FEE_RT
    pct   = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
    return Trade(
        symbol=sym, strategy=name,
        entry_price=pos["entry_price"], exit_price=exit_price,
        sl=pos["sl"], tp=pos["tp"], amount=pos["amount"],
        pnl_usdt=gross - fee, pnl_pct=pct,
        reason=reason, bars_held=bar_i - pos["entry_bar"],
    )


def _simulate_standalone(sym: str, name: str, ind: dict, ha: dict) -> list[Trade]:
    """Run ONE strategy on ONE symbol with its own position slot."""
    cfg       = STANDALONE_CFG[name]
    cls_name  = STRAT_CLASS[name]          # "InternStrategy" or swing name
    is_intern = cls_name == "InternStrategy"
    d         = ind[sym]
    bars      = ha[sym]["1h"]
    n         = len(bars)
    warm      = INTERN_WARMUP if is_intern else SWING_WARMUP

    trades: list[Trade] = []
    pos: dict | None = None
    cd = 0

    for i in range(1, n):
        cp  = float(d["closes"][i])
        bar = bars[i]

        # ── Exit handling ─────────────────────────────────────────────
        if pos is not None and i > pos["entry_bar"]:
            exit_r = exit_p = None
            h_sl = bar.low  <= pos["sl"]
            h_tp = bar.high >= pos["tp"]
            if h_tp and not h_sl:
                exit_r, exit_p = "tp", pos["tp"]
            elif h_sl and not h_tp:
                exit_r, exit_p = "sl", pos["sl"]
            elif h_sl and h_tp:
                exit_r, exit_p = (("tp", pos["tp"])
                                  if abs(bar.open - pos["tp"]) < abs(bar.open - pos["sl"])
                                  else ("sl", pos["sl"]))
            # Intern early exit: HMA-cross-down closes position on live.
            # Disabled in backtest when sell_in_bt=False (GBM whipsaws too much).
            if exit_r is None and is_intern and cfg.get("sell_in_bt", True):
                hp = float(d["hma15"][i - 1]); hc = float(d["hma15"][i])
                c_p = float(d["closes"][i - 1]); o_c = float(d["opens"][i])
                if not (math.isnan(hp) or math.isnan(hc)) and c_p < hp and o_c < hc:
                    exit_r, exit_p = "sell_signal", cp
            # Max-hold timeout
            if exit_r is None and (i - pos["entry_bar"]) >= cfg["hold"]:
                exit_r, exit_p = "max_hold", cp
            if exit_r is not None:
                trades.append(_mk_trade(sym, name, pos, exit_p, exit_r, i))
                pos = None
                cd  = cfg["cd"]

        # ── Entry handling ────────────────────────────────────────────
        if pos is None:
            if cd > 0:
                cd -= 1
                continue
            if i < warm:
                continue
            av = float(d["atr14"][i])
            if math.isnan(av):
                av = cp * 0.015

            entered = False
            if is_intern:
                hp = float(d["hma15"][i - 1]); hc = float(d["hma15"][i])
                c_p = float(d["closes"][i - 1])
                o_c = float(d["opens"][i]);    o_p = float(d["opens"][i - 1])
                b_lo = cfg.get("bias_lo", 1.0); b_mid = cfg.get("bias_mid", 1.0)
                if (not (math.isnan(hp) or math.isnan(hc))
                        and c_p > hp and o_c > hc and o_c > o_p
                        and d["bias_15m"][i] >= b_lo and d["bias_30m"][i] >= b_mid):
                    entered = True
            else:
                e80 = float(d["ema80"][i]);  e200  = float(d["ema200"][i])
                rsi_v = float(d["rsi14"][i]); vol_v = float(d["volumes"][i])
                vma   = float(d["vol_ma"][i])
                cb    = bool(d["candle_bull"][i]); mc = bool(d["macd_cross"][i])
                if not any(math.isnan(v) for v in [e80, e200, rsi_v, vma]):
                    if (e80 >= e200 * 0.98 and cb and mc
                            and cfg["rlo"] <= rsi_v <= cfg["rhi"]
                            and vol_v >= vma * cfg["vol"]):
                        entered = True

            if entered:
                sl_p = round(cp - cfg["sl"] * av, 4)
                tp_p = round(cp + cfg["tp"] * av, 4)
                if sl_p < cp < tp_p:
                    pos = dict(entry_price=cp, entry_bar=i, sl=sl_p, tp=tp_p,
                               amount=round(TRADE_USDT / cp, 6))

    if pos is not None:
        trades.append(_mk_trade(sym, name, pos, float(bars[-1].close), "end", n - 1))
    return trades


_MAJOR_COINS = {"BTC/USDT", "ETH/USDT", "BNB/USDT"}


def _simulate_profitable(sym: str, d_ext: dict, ha: dict) -> list[Trade]:
    """ProfitableBot: 1H trend-following with relaxed entry (2/4 conditions)."""
    bars_1h = ha[sym]["1h"]
    n = len(bars_1h)
    if n < 50:
        return []

    closes  = np.array([b.close  for b in bars_1h], dtype=float)
    volumes = np.array([b.volume for b in bars_1h], dtype=float)

    rsi14    = d_ext["rsi14_1h"]
    mhist    = d_ext["mhist_1h"]
    bb_lower = d_ext["bb_lower_1h"]
    volma20  = d_ext["volma20_1h"]
    atr14    = d_ext["atr14_1h"]
    ema20_1h = d_ext["ema20_1h"]
    ema50_1h = d_ext["ema50_1h"]
    ema20_4h = d_ext["ema20_4h"]
    ema50_4h = d_ext["ema50_4h"]
    ema20_4h_slope = d_ext["ema20_4h_slope"]

    # TP < SL → high WR on trending moves; R:R is ~0.6 so BEP ~62%
    sl_mult    = 2.0
    tp_mult    = 1.2
    sl_min_pct = 0.015
    sl_max_pct = 0.07

    trades: list[Trade] = []
    pos: dict | None = None
    cooldown = 0
    WARMUP = 60

    for i in range(1, n):
        cp  = float(closes[i])
        bar = bars_1h[i]

        # Exit handling
        if pos is not None and i > pos["entry_bar"]:
            h_sl = bar.low  <= pos["sl"]
            h_tp = bar.high >= pos["tp"]
            exit_r = exit_p = None

            if h_tp and not h_sl:
                exit_r, exit_p = "tp", pos["tp"]
            elif h_sl and not h_tp:
                exit_r, exit_p = "sl", pos["sl"]
            elif h_sl and h_tp:
                exit_r, exit_p = (("tp", pos["tp"])
                                  if abs(bar.open - pos["tp"]) < abs(bar.open - pos["sl"])
                                  else ("sl", pos["sl"]))

            # Breakeven lock when profit reaches 1×ATR
            if exit_r is None:
                if not pos["be_locked"] and (cp - pos["entry_price"]) >= pos["atr_entry"]:
                    new_sl = pos["entry_price"] * 1.002
                    if new_sl > pos["sl"]:
                        pos["sl"] = new_sl
                        pos["be_locked"] = True

            if exit_r is None and (i - pos["entry_bar"]) >= 72:
                exit_r, exit_p = "max_hold", cp

            if exit_r is not None:
                trades.append(_mk_trade(sym, "ProfitableBot", pos, exit_p, exit_r, i))
                pos = None
                cooldown = 2
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        if i < WARMUP:
            continue

        rsi_cur   = float(rsi14[i])
        rsi_prev  = float(rsi14[i - 1])
        hist_cur  = float(mhist[i])
        hist_prev = float(mhist[i - 1])
        bb_l_cur  = float(bb_lower[i])
        bb_l_prev = float(bb_lower[i - 1])
        close_prev = float(closes[i - 1])
        vol_cur   = float(volumes[i])
        vma_cur   = float(volma20[i])
        atr_cur   = float(atr14[i])
        e20_1h    = float(ema20_1h[i])
        e50_1h    = float(ema50_1h[i])
        e20_4h    = float(ema20_4h[i])
        e50_4h    = float(ema50_4h[i])
        slope_4h  = float(ema20_4h_slope[i]) if not math.isnan(ema20_4h_slope[i]) else 0.0

        if any(math.isnan(v) for v in [rsi_cur, rsi_prev, hist_cur, hist_prev,
                                        bb_l_cur, bb_l_prev, vma_cur, atr_cur,
                                        e20_1h, e50_1h]):
            continue

        # Mandatory base: 1H uptrend
        if e20_1h <= e50_1h:
            continue

        # Soft 4H guard: only block sharply bearish 4H (both below AND slope strongly negative)
        if not math.isnan(e20_4h) and not math.isnan(e50_4h):
            if e20_4h < e50_4h and slope_4h < -0.001:
                continue

        # Entry conditions (need 2/4)
        conds = 0
        # c1: RSI ≤ 48 AND turning up (relaxed from ≤35 over 2 bars)
        if rsi_cur <= 48.0 and rsi_cur > rsi_prev:
            conds += 1
        # c2: MACD histogram positive or crossing up
        if (hist_prev < 0 and hist_cur > 0) or (hist_cur > 0 and hist_cur > hist_prev):
            conds += 1
        # c3: BB lower bounce
        if close_prev <= bb_l_prev and cp > bb_l_cur:
            conds += 1
        # c4: Volume above average (relaxed from 1.3× to 1.1×)
        if vol_cur >= vma_cur * 1.1:
            conds += 1

        if conds < 2:
            continue

        sl_dist = sl_mult * atr_cur
        tp_dist = tp_mult * atr_cur
        sl_dist = max(sl_dist, cp * sl_min_pct)
        sl_dist = min(sl_dist, cp * sl_max_pct)

        sl_p = round(cp - sl_dist, 4)
        tp_p = round(cp + tp_dist, 4)
        if sl_p >= cp or tp_p <= cp:
            continue

        pos = {
            "entry_price": cp,
            "entry_bar":   i,
            "sl":          sl_p,
            "tp":          tp_p,
            "amount":      round(TRADE_USDT / cp, 6),
            "atr_entry":   atr_cur,
            "be_locked":   False,
        }

    if pos is not None:
        ep = float(bars_1h[-1].close)
        trades.append(_mk_trade(sym, "ProfitableBot", pos, ep, "end", n - 1))
    return trades


def _simulate_scalp(sym: str, d_ext: dict, ha: dict) -> list[Trade]:
    """ScalpTrendBot: 15m entry — 1H uptrend + pullback to EMA (simplified single TP)."""
    ha_15m = ha[sym].get("15m", [])
    n15 = len(ha_15m)
    if n15 < 30:
        return []

    c15   = d_ext["c15"]
    h15   = d_ext["h15"]
    l15   = d_ext["l15"]
    ema9  = d_ext["ema9_15m"]
    rsi14 = d_ext["rsi14_15m"]
    atr14 = d_ext["atr14_15m"]
    e20_1h = d_ext["ema20_1h_at15m"]
    e50_1h = d_ext["ema50_1h_at15m"]

    # TP < SL → high WR on momentum continuation
    sl_mult    = 1.5
    tp_mult    = 1.0
    sl_min_pct = 0.008
    sl_max_pct = 0.04
    WARMUP    = 50
    TIME_STOP = 32   # 15m bars ≈ 8h
    COOLDOWN  = 8

    trades: list[Trade] = []
    pos: dict | None = None
    cooldown = 0

    for j in range(1, n15):
        cp = float(c15[j])
        hj = float(h15[j])
        lj = float(l15[j])

        # Exit
        if pos is not None and j > pos["entry_bar"]:
            bars_held = j - pos["entry_bar"]
            sl_hit = lj <= pos["sl"]
            tp_hit = hj >= pos["tp"]
            exit_r = exit_p = None

            if tp_hit and not sl_hit:
                exit_r, exit_p = "tp", pos["tp"]
            elif sl_hit and not tp_hit:
                exit_r, exit_p = "sl", pos["sl"]
            elif sl_hit and tp_hit:
                op = float(ha_15m[j].open)
                exit_r, exit_p = (("tp", pos["tp"]) if abs(op - pos["tp"]) < abs(op - pos["sl"])
                                   else ("sl", pos["sl"]))
            elif bars_held >= TIME_STOP:
                exit_r, exit_p = "max_hold", cp

            if exit_r is not None:
                gross = (exit_p - pos["entry_price"]) / pos["entry_price"] * pos["amount"] * pos["entry_price"]
                fee   = pos["amount"] * pos["entry_price"] * FEE_RT
                pct   = (exit_p - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append(Trade(
                    symbol=sym, strategy="ScalpTrendBot",
                    entry_price=pos["entry_price"], exit_price=exit_p,
                    sl=pos["sl"], tp=pos["tp"], amount=pos["amount"],
                    pnl_usdt=gross - fee, pnl_pct=pct,
                    reason=exit_r, bars_held=bars_held,
                ))
                pos = None
                cooldown = COOLDOWN
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        if j < WARMUP:
            continue

        e9j   = float(ema9[j])
        rsi_j = float(rsi14[j])
        atr_j = float(atr14[j])
        e201h = float(e20_1h[j])
        e501h = float(e50_1h[j])

        if any(math.isnan(v) for v in [e9j, rsi_j, atr_j, e201h, e501h]):
            continue

        # Cond 1: 1H uptrend (dropped 4H requirement)
        if not (e201h > e501h):
            continue
        # Cond 2: Pullback within ±4% of 1H EMA20 (was ±1.5%)
        if abs(cp - e201h) / e201h > 0.04:
            continue
        # Cond 3: 15m momentum resuming — close above EMA9
        if cp <= e9j:
            continue
        # Cond 4: RSI not overbought (35–70, widened from 42–65)
        if not (35.0 <= rsi_j <= 70.0):
            continue

        sl_dist = sl_mult * atr_j
        sl_dist = max(sl_dist, cp * sl_min_pct)
        sl_dist = min(sl_dist, cp * sl_max_pct)
        sl_p    = round(cp - sl_dist, 6)
        tp_p    = round(cp + tp_mult * atr_j, 6)
        if sl_p >= cp or tp_p <= cp:
            continue

        pos = {
            "entry_price": cp,
            "entry_bar":   j,
            "sl":          sl_p,
            "tp":          tp_p,
            "amount":      round(TRADE_USDT / cp, 6),
        }

    if pos is not None:
        ep = float(c15[-1])
        bars_held = n15 - 1 - pos["entry_bar"]
        gross = (ep - pos["entry_price"]) / pos["entry_price"] * pos["amount"] * pos["entry_price"]
        fee   = pos["amount"] * pos["entry_price"] * FEE_RT
        pct   = (ep - pos["entry_price"]) / pos["entry_price"] * 100
        trades.append(Trade(
            symbol=sym, strategy="ScalpTrendBot",
            entry_price=pos["entry_price"], exit_price=ep,
            sl=pos["sl"], tp=pos["tp"], amount=pos["amount"],
            pnl_usdt=gross - fee, pnl_pct=pct,
            reason="end", bars_held=bars_held,
        ))
    return trades


def run_standalone(ind: dict, ha: dict, ind_ext: dict | None = None) -> dict[str, list[Trade]]:
    """Per-strategy standalone backtest (each SJ on its own slot)."""
    out: dict[str, list[Trade]] = {name: [] for name in STRAT_ORDER}
    for name in STRAT_ORDER:
        for sym in SYMBOLS:
            out[name].extend(_simulate_standalone(sym, name, ind, ha))
    if ind_ext is not None:
        out["ProfitableBot"] = []
        out["ScalpTrendBot"] = []
        for sym in SYMBOLS:
            out["ProfitableBot"].extend(_simulate_profitable(sym, ind_ext[sym], ha))
            out["ScalpTrendBot"].extend(_simulate_scalp(sym, ind_ext[sym], ha))
    return out


# ── Data prep — build HA + indicators once ────────────────────────────────────
def prepare(data: dict) -> tuple[dict, dict]:
    ha: dict[str, dict[str, list[OHLCV]]] = {
        sym: {tf: to_heikin_ashi(data[sym][tf]) for tf in data[sym]}
        for sym in SYMBOLS
    }
    print("  Precomputing indicators...", end="", flush=True)
    ind = {sym: _precompute(sym, ha) for sym in SYMBOLS}
    print(" done")
    return ha, ind


# ── Simulation (vectorized — O(n) per bar) ────────────────────────────────────
def run_backtest(ha: dict, ind: dict, ind_ext: dict | None = None) -> tuple[list[Trade], dict]:
    master_1h = ha[SYMBOLS[0]]["1h"]
    n_bars    = len(master_1h)

    ts_map: dict[str, dict[int, int]] = {
        sym: {int(c.timestamp): i for i, c in enumerate(ha[sym]["1h"])}
        for sym in SYMBOLS
    }

    # Swing strategy configs: (name, sl_mult, tp_mult, rsi_lo, rsi_hi, vol_mult, cooldown_bars)
    _SWING = [
        ("SwingReversalStrategy", 2.5, 1.5, 34.0, 62.0, 1.2, 4),
        ("CPKRegimeStrategy",     2.0, 1.2, 44.0, 60.0, 1.5, 4),
    ]
    ALL_STRAT_NAMES = [sn for sn, *_ in _SWING] + ["ProfitableBot", "ScalpTrendBot"]
    SWING_WARMUP = 210
    NEW_WARMUP   = 60

    positions:      list[Position] = []
    trades:         list[Trade]    = []
    blocked:        dict[str, int] = defaultdict(int)
    chief_approved: list[dict]     = []
    chief_filtered: list[dict]     = []
    maxpos_blocked: int             = 0

    cooldown: dict[tuple, int] = {
        (sym, sn): 0 for sym in SYMBOLS for sn in ALL_STRAT_NAMES
    }

    def sym_pos(sym: str) -> Position | None:
        return next((p for p in positions if p.symbol == sym), None)

    def close_pos(pos: Position, exit_price: float, reason: str, bar_i: int):
        gross = (exit_price - pos.entry_price) / pos.entry_price * pos.amount * pos.entry_price
        fee   = pos.amount * pos.entry_price * FEE_RT
        pct   = (exit_price - pos.entry_price) / pos.entry_price * 100
        trades.append(Trade(
            symbol=pos.symbol, strategy=pos.strategy,
            entry_price=pos.entry_price, exit_price=exit_price,
            sl=pos.sl, tp=pos.tp, amount=pos.amount,
            pnl_usdt=gross - fee, pnl_pct=pct,
            reason=reason, bars_held=bar_i - pos.entry_bar,
        ))
        positions.remove(pos)

    print(f"  [Case 2] Simulating {n_bars} bars × {len(SYMBOLS)} symbols × "
          f"{len(ALL_STRAT_NAMES)} strategies (Swing+CPK+Profitable+Scalp)...")

    for bar_i, master_bar in enumerate(master_1h):
        ts = int(master_bar.timestamp)

        # ── 1. Price-based SL/TP exits ────────────────────────────────
        for pos in list(positions):
            if pos.entry_bar == bar_i:
                continue
            idx = ts_map[pos.symbol].get(ts)
            if idx is None:
                continue
            bar  = ha[pos.symbol]["1h"][idx]
            h_sl = bar.low  <= pos.sl
            h_tp = bar.high >= pos.tp
            if h_sl or h_tp:
                if h_tp and not h_sl:
                    r, p = "tp", pos.tp
                elif h_sl and not h_tp:
                    r, p = "sl", pos.sl
                else:
                    r, p = (("tp", pos.tp)
                             if abs(bar.open - pos.tp) < abs(bar.open - pos.sl)
                             else ("sl", pos.sl))
                close_pos(pos, p, r, bar_i)

        # ── 2. Strategy signals per symbol ────────────────────────────
        for sym in SYMBOLS:
            idx = ts_map[sym].get(ts)
            if idx is None or idx < 1:
                continue
            i  = idx
            d  = ind[sym]
            d_e = ind_ext[sym] if ind_ext else None
            cp = float(d["closes"][i])
            av = float(d["atr14"][i])
            if math.isnan(av):
                av = cp * 0.015

            pos = sym_pos(sym)
            if pos:
                continue   # no new entry while in a position

            # ── BUY signals ────────────────────────────────────────────
            buy_sigs: list[tuple[str, float, float]] = []

            # Swing variants (shared base conditions)
            if i >= SWING_WARMUP:
                e80   = float(d["ema80"][i]);   e200  = float(d["ema200"][i])
                rsi_v = float(d["rsi14"][i]);   vol_v = float(d["volumes"][i])
                vma_v = float(d["vol_ma"][i])
                cb    = bool(d["candle_bull"][i]); mc = bool(d["macd_cross"][i])
                data_valid = not any(math.isnan(v) for v in [e80, e200, rsi_v, vma_v])
                regime_ok  = data_valid and e80 >= e200 * 0.98 and cb and mc

                for sn, sl_m, tp_m, rlo, rhi, vm, cd_len in _SWING:
                    key = (sym, sn)
                    if cooldown[key] > 0:
                        cooldown[key] -= 1
                        continue
                    if regime_ok and rlo <= rsi_v <= rhi and vol_v >= vma_v * vm:
                        sl_p = round(cp - sl_m * av, 4)
                        tp_p = round(cp + tp_m * av, 4)
                        if sl_p < cp < tp_p:
                            buy_sigs.append((sn, sl_p, tp_p))
                            cooldown[key] = cd_len

            # ProfitableBot BUY (1H trend-following, 2/4 conditions)
            key_pb = (sym, "ProfitableBot")
            if cooldown[key_pb] > 0:
                cooldown[key_pb] -= 1
            elif i >= NEW_WARMUP and d_e is not None:
                rsi_c  = float(d_e["rsi14_1h"][i])
                rsi_p  = float(d_e["rsi14_1h"][i - 1])
                hc     = float(d_e["mhist_1h"][i])
                hp     = float(d_e["mhist_1h"][i - 1])
                bbl_c  = float(d_e["bb_lower_1h"][i])
                bbl_p  = float(d_e["bb_lower_1h"][i - 1])
                cp_p   = float(d["closes"][i - 1])
                vma_pb = float(d_e["volma20_1h"][i])
                atr_pb = float(d_e["atr14_1h"][i])
                e20_1h = float(d_e["ema20_1h"][i])
                e50_1h = float(d_e["ema50_1h"][i])
                e20_4h = float(d_e["ema20_4h"][i])
                e50_4h = float(d_e["ema50_4h"][i])
                slope_4h = float(d_e["ema20_4h_slope"][i]) if not math.isnan(d_e["ema20_4h_slope"][i]) else 0.0
                if not any(math.isnan(v) for v in [rsi_c, rsi_p, hc, hp,
                                                    bbl_c, bbl_p, vma_pb, atr_pb, e20_1h, e50_1h]):
                    if e20_1h > e50_1h:
                        trend_ok = math.isnan(e20_4h) or math.isnan(e50_4h) or not (e20_4h < e50_4h and slope_4h < -0.001)
                        if trend_ok:
                            conds = 0
                            if rsi_c <= 48.0 and rsi_c > rsi_p:         conds += 1
                            if (hp < 0 and hc > 0) or (hc > 0 and hc > hp): conds += 1
                            if cp_p <= bbl_p and cp > bbl_c:             conds += 1
                            if float(d["volumes"][i]) >= vma_pb * 1.1:  conds += 1
                            if conds >= 2:
                                sl_dist = max(2.0 * atr_pb, cp * 0.015)
                                sl_dist = min(sl_dist, cp * 0.07)
                                sl_p = round(cp - sl_dist, 4)
                                tp_p = round(cp + 1.2 * atr_pb, 4)
                                if sl_p < cp < tp_p:
                                    buy_sigs.append(("ProfitableBot", sl_p, tp_p))
                                    cooldown[key_pb] = 2

            # ScalpTrendBot BUY (1H proxy — EMA20 cross-above in uptrend)
            key_sc = (sym, "ScalpTrendBot")
            if cooldown[key_sc] > 0:
                cooldown[key_sc] -= 1
            elif i >= NEW_WARMUP + 1 and d_e is not None:
                e20_1h_sc = float(d_e["ema20_1h"][i]);   e50_1h_sc = float(d_e["ema50_1h"][i])
                e20_p_sc  = float(d_e["ema20_1h"][i - 1])
                rsi_sc    = float(d_e["rsi14_1h"][i]);   atr_sc    = float(d_e["atr14_1h"][i])
                cp_prev   = float(d["closes"][i - 1])
                if not any(math.isnan(v) for v in [e20_1h_sc, e50_1h_sc, rsi_sc, atr_sc, e20_p_sc]):
                    if (e20_1h_sc > e50_1h_sc
                            and cp_prev < e20_p_sc
                            and cp > e20_1h_sc
                            and 40.0 <= rsi_sc <= 65.0):
                        sl_dist = max(2.0 * atr_sc, cp * 0.008)
                        sl_dist = min(sl_dist, cp * 0.05)
                        sl_p = round(cp - sl_dist, 4)
                        tp_p = round(cp + 1.5 * atr_sc, 4)
                        if sl_p < cp < tp_p:
                            buy_sigs.append(("ScalpTrendBot", sl_p, tp_p))
                            cooldown[key_sc] = 8

            if not buy_sigs:
                continue

            n_agree = len(buy_sigs)

            # 3-slot limit
            if len(positions) >= MAX_POSITIONS:
                for sn, _, _ in buy_sigs:
                    blocked[sn] += 1
                maxpos_blocked += 1
                continue

            sn0, sl0, tp0 = buy_sigs[0]
            sig_rec = {
                "bar_i":    bar_i,
                "symbol":   sym,
                "strategy": sn0,
                "n_agree":  n_agree,
                "price":    cp,
                "sl":       sl0,
                "tp":       tp0,
                "strategies_agreeing": [sn for sn, _, _ in buy_sigs],
            }

            if n_agree >= CHIEF_N_AGREE:
                chief_approved.append(sig_rec)
                positions.append(Position(
                    symbol=sym, strategy=sn0,
                    entry_price=cp, entry_bar=bar_i,
                    sl=sl0, tp=tp0,
                    amount=round(TRADE_USDT / cp, 6),
                ))
            else:
                chief_filtered.append(sig_rec)

    # Close any still-open positions at last bar price
    last_ts = int(master_1h[-1].timestamp)
    for pos in list(positions):
        idx = ts_map[pos.symbol].get(last_ts)
        ep  = (float(ha[pos.symbol]["1h"][idx].close)
               if idx is not None else pos.entry_price)
        close_pos(pos, ep, "end", n_bars - 1)

    # Retrospective outcome for Chief-filtered signals
    for ms in chief_filtered:
        future = ha[ms["symbol"]]["1h"][ms["bar_i"] + 1: ms["bar_i"] + 1 + CHIEF_LOOKFWD]
        outcome = "unknown"
        for bar in future:
            if bar.low  <= ms["sl"]: outcome = "sl"; break
            if bar.high >= ms["tp"]: outcome = "tp"; break
        ms["outcome"] = outcome
        ep  = ms["price"]
        exp = ms["tp"] if outcome == "tp" else ms["sl"] if outcome == "sl" else ep
        ms["pnl_usdt"] = (exp - ep) / ep * TRADE_USDT - TRADE_USDT * FEE_RT

    # Match Chief-approved signals to actual trade outcomes
    for ap in chief_approved:
        match = next(
            (t for t in trades
             if t.symbol == ap["symbol"] and t.entry_price == ap["price"]),
            None,
        )
        ap["outcome"]  = match.reason   if match else "unknown"
        ap["pnl_usdt"] = match.pnl_usdt if match else 0.0

    stats: dict = {
        "bars_total":     n_bars,
        "blocked":        dict(blocked),
        "maxpos_blocked": maxpos_blocked,
        "chief_approved": chief_approved,
        "chief_filtered": chief_filtered,
    }
    return trades, stats


def run_portfolio_case3(ha: dict, ind: dict, ind_ext: dict) -> tuple[list[Trade], dict]:
    """Portfolio: 4 strategies independent — no Chief, 1 pos/strategy, max 3 total slots."""
    master_1h = ha[SYMBOLS[0]]["1h"]
    n_bars    = len(master_1h)

    ts_map: dict[str, dict[int, int]] = {
        sym: {int(c.timestamp): i for i, c in enumerate(ha[sym]["1h"])}
        for sym in SYMBOLS
    }

    positions:     list[Position] = []
    trades:        list[Trade]    = []
    blocked:       dict[str, int] = defaultdict(int)
    maxpos_blocked = 0

    cooldown: dict[tuple, int] = {}
    for sym in SYMBOLS:
        for sn in ["CPKRegimeStrategy", "ProfitableBot", "ScalpTrendBot"]:
            cooldown[(sym, sn)] = 0

    be_locked: set = set()

    def strat_pos(sn: str) -> Position | None:
        """Return the open position for a strategy (any symbol), or None."""
        return next((p for p in positions if p.strategy == sn), None)

    def close_pos(pos: Position, exit_price: float, reason: str, bar_i: int):
        gross = (exit_price - pos.entry_price) / pos.entry_price * pos.amount * pos.entry_price
        fee   = pos.amount * pos.entry_price * FEE_RT
        pct   = (exit_price - pos.entry_price) / pos.entry_price * 100
        trades.append(Trade(
            symbol=pos.symbol, strategy=pos.strategy,
            entry_price=pos.entry_price, exit_price=exit_price,
            sl=pos.sl, tp=pos.tp, amount=pos.amount,
            pnl_usdt=gross - fee, pnl_pct=pct,
            reason=reason, bars_held=bar_i - pos.entry_bar,
        ))
        be_locked.discard(id(pos))
        positions.remove(pos)

    SWING_WARMUP = 210
    NEW_WARMUP   = 60

    print(f"  [Case 3] No-Chief portfolio — {n_bars} bars × {len(SYMBOLS)} symbols "
          f"× 3 strategies (CPK+Profitable+Scalp), 1 pos/SJ, max {MAX_POSITIONS} slots...")

    for bar_i, master_bar in enumerate(master_1h):
        ts = int(master_bar.timestamp)

        # ── 1. SL/TP exits + breakeven + time stop ────────────────────
        for pos in list(positions):
            if pos.entry_bar == bar_i:
                continue
            idx = ts_map[pos.symbol].get(ts)
            if idx is None:
                continue
            bar = ha[pos.symbol]["1h"][idx]
            i   = idx
            d_e = ind_ext[pos.symbol]
            cp  = float(bar.close)

            # Breakeven lock
            if id(pos) not in be_locked:
                atr_1h = float(d_e["atr14_1h"][i]) if i < len(d_e["atr14_1h"]) else 0.0
                if not math.isnan(atr_1h) and atr_1h > 0 and (cp - pos.entry_price) >= atr_1h:
                    new_sl = pos.entry_price * 1.002
                    if new_sl > pos.sl:
                        pos.sl = new_sl
                        be_locked.add(id(pos))

            h_sl = bar.low  <= pos.sl
            h_tp = bar.high >= pos.tp
            if h_sl or h_tp:
                if h_tp and not h_sl:
                    r, p = "tp", pos.tp
                elif h_sl and not h_tp:
                    r, p = "sl", pos.sl
                else:
                    r, p = (("tp", pos.tp)
                             if abs(bar.open - pos.tp) < abs(bar.open - pos.sl)
                             else ("sl", pos.sl))
                close_pos(pos, p, r, bar_i)
                continue

            if pos.strategy == "ScalpTrendBot":        time_limit = 20
            elif pos.strategy == "CPKRegimeStrategy": time_limit = 96
            else:                                      time_limit = 72
            if (bar_i - pos.entry_bar) >= time_limit:
                close_pos(pos, cp, "max_hold", bar_i)

        # ── 2. Each strategy fires independently ──────────────────────
        for sym in SYMBOLS:
            idx = ts_map[sym].get(ts)
            if idx is None or idx < 1:
                continue
            i    = idx
            d    = ind[sym]
            d_e  = ind_ext[sym]
            cp   = float(d["closes"][i])
            atr_v = float(d["atr14"][i])
            if math.isnan(atr_v):
                atr_v = cp * 0.015

            def _try_open(sn: str, sl_p: float, tp_p: float, cd: int) -> bool:
                nonlocal maxpos_blocked
                if strat_pos(sn) is not None:
                    return False  # strategy already in a position
                if len(positions) >= MAX_POSITIONS:
                    blocked[sn] += 1
                    maxpos_blocked += 1
                    return False
                positions.append(Position(
                    symbol=sym, strategy=sn,
                    entry_price=cp, entry_bar=bar_i,
                    sl=sl_p, tp=tp_p,
                    amount=round(TRADE_USDT / cp, 6),
                ))
                cooldown[(sym, sn)] = cd
                return True

            # CPKRegimeStrategy
            key_cpk = (sym, "CPKRegimeStrategy")
            if cooldown[key_cpk] > 0:
                cooldown[key_cpk] -= 1
            elif i >= SWING_WARMUP:
                e80_c   = float(d["ema80"][i]);   e200_c  = float(d["ema200"][i])
                rsi_c   = float(d["rsi14"][i]);   vol_c   = float(d["volumes"][i])
                vma_c   = float(d["vol_ma"][i])
                cb_c    = bool(d["candle_bull"][i]); mc_c = bool(d["macd_cross"][i])
                dv_c    = not any(math.isnan(v) for v in [e80_c, e200_c, rsi_c, vma_c])
                if (dv_c and e80_c >= e200_c * 0.98 and cb_c and mc_c
                        and 44.0 <= rsi_c <= 60.0 and vol_c >= vma_c * 1.5):
                    sl_p = round(cp - 2.5 * atr_v, 4)
                    tp_p = round(cp + 1.5 * atr_v, 4)
                    if sl_p < cp < tp_p:
                        _try_open("CPKRegimeStrategy", sl_p, tp_p, 2)

            # ProfitableBot (1H trend-following, 2/4 conditions)
            key_pb = (sym, "ProfitableBot")
            if cooldown[key_pb] > 0:
                cooldown[key_pb] -= 1
            elif i >= NEW_WARMUP:
                rsi_c3  = float(d_e["rsi14_1h"][i]);   rsi_p3  = float(d_e["rsi14_1h"][i - 1])
                hc3     = float(d_e["mhist_1h"][i]);    hp3     = float(d_e["mhist_1h"][i - 1])
                bbl_c3  = float(d_e["bb_lower_1h"][i]); bbl_p3  = float(d_e["bb_lower_1h"][i - 1])
                cp_p3   = float(d["closes"][i - 1])
                vma3    = float(d_e["volma20_1h"][i]);   atr3    = float(d_e["atr14_1h"][i])
                e20_1h3 = float(d_e["ema20_1h"][i]);    e50_1h3 = float(d_e["ema50_1h"][i])
                e20_4h3 = float(d_e["ema20_4h"][i]);    e50_4h3 = float(d_e["ema50_4h"][i])
                sl4h3   = float(d_e["ema20_4h_slope"][i]) if not math.isnan(d_e["ema20_4h_slope"][i]) else 0.0
                if not any(math.isnan(v) for v in [rsi_c3, rsi_p3, hc3, hp3,
                                                    bbl_c3, bbl_p3, vma3, atr3, e20_1h3, e50_1h3]):
                    if e20_1h3 > e50_1h3:
                        trend_ok = math.isnan(e20_4h3) or math.isnan(e50_4h3) or not (e20_4h3 < e50_4h3 and sl4h3 < -0.001)
                        if trend_ok:
                            conds = 0
                            if rsi_c3 <= 48.0 and rsi_c3 > rsi_p3:                 conds += 1
                            if (hp3 < 0 and hc3 > 0) or (hc3 > 0 and hc3 > hp3):  conds += 1
                            if cp_p3 <= bbl_p3 and cp > bbl_c3:                    conds += 1
                            if float(d["volumes"][i]) >= vma3 * 1.1:               conds += 1
                            if conds >= 2:
                                sl_dist = max(2.0 * atr3, cp * 0.015)
                                sl_dist = min(sl_dist, cp * 0.07)
                                sl_p = round(cp - sl_dist, 4)
                                tp_p = round(cp + 1.2 * atr3, 4)
                                if sl_p < cp < tp_p:
                                    _try_open("ProfitableBot", sl_p, tp_p, 2)

            # ScalpTrendBot (1H proxy — EMA20 cross-above in uptrend)
            key_sc = (sym, "ScalpTrendBot")
            if cooldown[key_sc] > 0:
                cooldown[key_sc] -= 1
            elif i >= NEW_WARMUP + 1:
                e20_sc  = float(d_e["ema20_1h"][i]);   e50_sc  = float(d_e["ema50_1h"][i])
                e20_p   = float(d_e["ema20_1h"][i - 1])
                rsi_sc  = float(d_e["rsi14_1h"][i]);   atr_sc  = float(d_e["atr14_1h"][i])
                cp_prev = float(d["closes"][i - 1])
                if not any(math.isnan(v) for v in [e20_sc, e50_sc, rsi_sc, atr_sc, e20_p]):
                    if (e20_sc > e50_sc                       # 1H uptrend
                            and cp_prev < e20_p               # previous bar below EMA20
                            and cp > e20_sc                   # current bar crossed above EMA20
                            and 40.0 <= rsi_sc <= 65.0):
                        sl_dist = max(2.0 * atr_sc, cp * 0.008)
                        sl_dist = min(sl_dist, cp * 0.05)
                        sl_p = round(cp - sl_dist, 4)
                        tp_p = round(cp + 1.5 * atr_sc, 4)
                        if sl_p < cp < tp_p:
                            _try_open("ScalpTrendBot", sl_p, tp_p, 8)

    # Close remaining open positions at last bar price
    last_ts = int(master_1h[-1].timestamp)
    for pos in list(positions):
        idx = ts_map[pos.symbol].get(last_ts)
        ep  = (float(ha[pos.symbol]["1h"][idx].close)
               if idx is not None else pos.entry_price)
        close_pos(pos, ep, "end", n_bars - 1)

    stats: dict = {
        "bars_total":     n_bars,
        "blocked":        dict(blocked),
        "maxpos_blocked": maxpos_blocked,
    }
    return trades, stats


# ── Report ────────────────────────────────────────────────────────────────────
def _equity_curve(trade_list: list[Trade]) -> tuple[float, float]:
    if not trade_list:
        return 0.0, 0.0
    equity = peak = 0.0
    max_dd = 0.0
    for t in trade_list:
        equity += t.pnl_usdt
        peak    = max(peak, equity)
        max_dd  = min(max_dd, equity - peak)
    return equity, max_dd


def _bep(trade_list: list[Trade]) -> float:
    """Break-even WR from actual avg win / avg loss amounts (includes fees & early exits)."""
    wins   = [t.pnl_usdt for t in trade_list if t.pnl_usdt > 0]
    losses = [t.pnl_usdt for t in trade_list if t.pnl_usdt <= 0]
    if not wins or not losses:
        return 50.0
    avg_w = sum(wins) / len(wins)
    avg_l = abs(sum(losses) / len(losses))
    return avg_l / (avg_w + avg_l) * 100  # = % WR needed to break even


def _bb_lower(closes: np.ndarray, period: int = 20, std_mult: float = 2.0) -> np.ndarray:
    """Rolling Bollinger Band lower = SMA(period) - std_mult * StdDev(period)."""
    n = len(closes)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        out[i] = window.mean() - std_mult * window.std(ddof=0)
    return out


_INTERN_KEYS   = {"InternStrategy", "InternStrategy(faithful)"}
_DISPLAY_NAMES = {
    "InternStrategy":           "Intern(entry-only TP/SL)",
    "InternStrategy(faithful)": "Intern(faithful+HMA-exit)",
}


def print_standalone(standalone: dict[str, list[Trade]], bars: int):
    """Per-SJ standalone performance — each strategy on its own slot."""
    months = bars / 24 / 30.0
    W = 86
    print("\n" + "═" * W)
    print(f"{'  STANDALONE PER-STRATEGY REPORT  (each SJ trades on its own slot)':^{W}}")
    print(f"{'  ~%.1f months  |  %d symbols  |  $%.0f / trade' % (months, len(SYMBOLS), TRADE_USDT):^{W}}")
    print("═" * W)
    print(f"\n{'Strategy':<30} {'#':>4} {'/mo':>5} {'WR%':>6} {'BEP%':>6} "
          f"{'PnL$':>9} {'AvgPnL%':>8} {'MaxDD$':>8} {'Hold':>5}")
    print("─" * W)

    all_pass = True
    intern_separator_done = False
    for name in STRAT_ORDER:
        # Print a blank separator line before the two Intern rows
        if name in _INTERN_KEYS and not intern_separator_done:
            print("  ┄ Intern shown twice: entry-only (TP/SL geometry) vs faithful (live logic)")
            intern_separator_done = True

        display = _DISPLAY_NAMES.get(name, name)
        ts = standalone.get(name, [])
        if not ts:
            print(f"{display:<30} {'—':>4}")
            if name not in _INTERN_KEYS:
                all_pass = False
            continue
        wins    = sum(1 for t in ts if t.pnl_usdt > 0)
        wr      = wins / len(ts) * 100
        bep     = _bep(ts)
        pnl, dd = _equity_curve(ts)
        avg_pct = sum(t.pnl_pct for t in ts) / len(ts)
        avg_h   = sum(t.bars_held for t in ts) / len(ts)
        per_mo  = len(ts) / months if months else 0
        ok      = pnl > 0 and per_mo >= 10
        # For overall pass/fail only count non-Intern (Intern's pass is informational)
        if name not in _INTERN_KEYS and not ok:
            all_pass = False
        flag    = "  ✓" if ok else "  ⚠"
        print(f"{display:<30} {len(ts):>4} {per_mo:>5.1f} {wr:>5.1f}% {bep:>5.1f}% "
              f"{pnl:>+9.2f} {avg_pct:>+7.2f}% {dd:>+8.2f} {avg_h:>4.1f}h{flag}")

    # Summary — exclude faithful Intern from "ALL" sum (would double-count entries)
    main_keys  = [n for n in STRAT_ORDER if n != "InternStrategy(faithful)"]
    all_t      = [t for n in main_keys for t in standalone.get(n, [])]
    if all_t:
        wins    = sum(1 for t in all_t if t.pnl_usdt > 0)
        pnl, dd = _equity_curve(all_t)
        print("─" * W)
        print(f"{'ALL excl. faithful Intern':<30} {len(all_t):>4} {len(all_t)/months:>5.1f} "
              f"{wins/len(all_t)*100:>5.1f}%        {pnl:>+9.2f}")

    # New bots section
    new_bot_keys = ["ProfitableBot", "ScalpTrendBot"]
    if any(k in standalone for k in new_bot_keys):
        print("\n" + "─" * W)
        print("  New Bots (standalone — no Chief gate)")
        print("─" * W)
        for name in new_bot_keys:
            ts = standalone.get(name, [])
            if not ts:
                print(f"{name:<30} {'—':>4}")
                continue
            wins    = sum(1 for t in ts if t.pnl_usdt > 0)
            wr      = wins / len(ts) * 100
            bep     = _bep(ts)
            pnl, dd = _equity_curve(ts)
            avg_pct = sum(t.pnl_pct for t in ts) / len(ts)
            if name == "ScalpTrendBot":
                avg_h = sum(t.bars_held for t in ts) / len(ts) / 4  # 15m bars → hours
            else:
                avg_h = sum(t.bars_held for t in ts) / len(ts)
            per_mo  = len(ts) / months if months else 0
            ok      = pnl > 0 and per_mo >= 5
            flag    = "  ✓" if ok else "  ⚠"
            print(f"{name:<30} {len(ts):>4} {per_mo:>5.1f} {wr:>5.1f}% {bep:>5.1f}% "
                  f"{pnl:>+9.2f} {avg_pct:>+7.2f}% {dd:>+8.2f} {avg_h:>4.1f}h{flag}")

    print(f"\n  BEP  = break-even WR computed from actual avg-win / avg-loss (fees included)")
    print(f"  ✓ = PnL > 0  AND  ≥10 trades/month   (Intern rows are informational)")
    print(f"  Core 3 SJs: {'✓ ALL PROFITABLE' if all_pass else '⚠ some not profitable'}")


def print_report(trades: list[Trade], stats: dict):
    if not trades:
        print("\nNo trades generated — warmup period may cover entire simulation.")
        return

    bars = stats.get("bars_total", 0)
    days = bars / 24
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    is_syn     = stats.get("is_synthetic", False)
    data_label = "⚠ SYNTHETIC GBM DATA" if is_syn else "REAL MARKET DATA"
    print("\n" + "═" * 74)
    print(f"{'  BACKTEST REPORT — 5-Month Multi-Strategy  ' + data_label:^74}")
    print(f"{'  Period: ~%.0f days  (%d bars 1h)  Data to: %s' % (days, bars, now_str):^74}")
    print(f"{'  Symbols: %s  |  Slots: %d  |  Trade: $%.0f USDT' % (', '.join(SYMBOLS), MAX_POSITIONS, TRADE_USDT):^74}")
    print(f"{'  Chief gate: require ≥%d strategies to agree' % CHIEF_N_AGREE:^74}")
    print("═" * 74)

    by_strat: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_strat[t.strategy].append(t)

    print(f"\n{'Strategy':<28} {'#':>4} {'Win%':>6} {'PnL$':>8} {'AvgPnL%':>8} "
          f"{'MaxDD$':>8} {'AvgHold':>8}")
    print("─" * 74)

    strat_order = [
        "SwingReversalStrategy", "CPKRegimeStrategy",
        "ProfitableBot",         "ScalpTrendBot",
    ]
    all_pnl = 0.0
    for name in strat_order:
        ts = by_strat.get(name, [])
        if not ts:
            print(f"{name:<28} {'—':>4}")
            continue
        wins    = sum(1 for t in ts if t.pnl_usdt > 0)
        pnl, dd = _equity_curve(ts)
        avg_pct = sum(t.pnl_pct for t in ts) / len(ts)
        avg_h   = sum(t.bars_held for t in ts) / len(ts)
        print(f"{name:<28} {len(ts):>4} {wins/len(ts)*100:>5.1f}% {pnl:>+8.2f} "
              f"{avg_pct:>+7.2f}% {dd:>+8.2f} {avg_h:>6.1f}h")
        all_pnl += pnl

    print("─" * 74)
    all_wins     = sum(1 for t in trades if t.pnl_usdt > 0)
    all_pnl2, dd = _equity_curve(trades)
    avg_h_all    = sum(t.bars_held for t in trades) / len(trades)
    print(f"{'TOTAL':<28} {len(trades):>4} {all_wins/len(trades)*100:>5.1f}% "
          f"{all_pnl2:>+8.2f} {sum(t.pnl_pct for t in trades)/len(trades):>+7.2f}% "
          f"{dd:>+8.2f} {avg_h_all:>6.1f}h")

    # By symbol
    print(f"\n{'Symbol':<14} {'#':>4} {'Win%':>6} {'PnL$':>8}")
    print("─" * 36)
    by_sym: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_sym[t.symbol].append(t)
    for sym, ts in sorted(by_sym.items()):
        wins = sum(1 for t in ts if t.pnl_usdt > 0)
        pnl  = sum(t.pnl_usdt for t in ts)
        print(f"{sym:<14} {len(ts):>4} {wins/len(ts)*100:>5.1f}% {pnl:>+8.2f}")

    # Exit reasons
    print("\nExit reason breakdown:")
    print(f"  {'Reason':<14} {'Count':>5} {'%':>6} {'Avg PnL$':>10}")
    print("  " + "─" * 38)
    reasons = Counter(t.reason for t in trades)
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pct     = cnt / len(trades) * 100
        avg_pnl = sum(t.pnl_usdt for t in trades if t.reason == r) / cnt
        print(f"  {r:<14} {cnt:>5} {pct:>5.1f}% {avg_pnl:>+10.2f}")

    # Blocking stats
    blocked = stats.get("blocked", {})
    if blocked:
        print("\nBlocked signals (3-slot full — signal ignored):")
        for name, cnt in sorted(blocked.items(), key=lambda x: -x[1]):
            print(f"  {name:<30} {cnt:>5}×")

    maxpos_blocked = stats.get("maxpos_blocked", 0)
    if maxpos_blocked:
        print(f"\n  Max-position blocks: {maxpos_blocked}×")

    # ── Chief simulation stats ────────────────────────────────────────────────
    approved       = stats.get("chief_approved", [])
    filtered       = stats.get("chief_filtered", [])
    total_signals  = len(approved) + len(filtered)

    print(f"\n{'─'*74}")
    print(f"Chief Simulation  (require ≥{CHIEF_N_AGREE} strategies to agree)")
    print(f"{'─'*74}")
    print(f"  Total BUY signals generated : {total_signals}")
    print(f"  Chief APPROVED              : {len(approved)} "
          f"({len(approved)/max(total_signals,1)*100:.1f}%)")
    print(f"  Chief FILTERED              : {len(filtered)} "
          f"({len(filtered)/max(total_signals,1)*100:.1f}%)")

    if approved:
        ap_tp  = sum(1 for a in approved if a.get("outcome") == "tp")
        ap_sl  = sum(1 for a in approved if a.get("outcome") == "sl")
        ap_pnl = sum(a.get("pnl_usdt", 0) for a in approved)
        print(f"\n  Approved outcomes:")
        print(f"    TP hit  : {ap_tp} ({ap_tp/len(approved)*100:.1f}%)  → correct entry")
        print(f"    SL hit  : {ap_sl} ({ap_sl/len(approved)*100:.1f}%)  → loss accepted")
        print(f"    Net P&L : ${ap_pnl:+.2f}")

    if filtered:
        fi_tp  = sum(1 for f in filtered if f.get("outcome") == "tp")
        fi_sl  = sum(1 for f in filtered if f.get("outcome") == "sl")
        fi_unk = sum(1 for f in filtered if f.get("outcome") == "unknown")
        fi_pnl = sum(f.get("pnl_usdt", 0) for f in filtered)
        print(f"\n  Filtered signal outcomes (what would have happened):")
        print(f"    TP hit  : {fi_tp} ({fi_tp/len(filtered)*100:.1f}%)  ⚠ MISSED PROFIT")
        print(f"    SL hit  : {fi_sl} ({fi_sl/len(filtered)*100:.1f}%)  ✓ GOOD FILTER (saved loss)")
        print(f"    Unknown : {fi_unk}  (lookfwd {CHIEF_LOOKFWD}h expired)")
        if fi_tp + fi_sl > 0:
            print(f"    Net delta: ${fi_pnl:+.2f}  (negative = Chief saved money)")
        if fi_tp > 0:
            avg_miss = sum(f["pnl_usdt"] for f in filtered if f.get("outcome") == "tp") / fi_tp
            print(f"    Avg missed profit / filtered signal: ${avg_miss:+.2f}")

    print(f"\n  Chief verdict: ", end="")
    if filtered:
        fi_pnl = sum(f.get("pnl_usdt", 0) for f in filtered)
        if fi_pnl < 0:
            print(f"✓ Filtering IMPROVED result by ${-fi_pnl:.2f}  (blocked bad trades)")
        else:
            print(f"⚠ Filtering MISSED ${fi_pnl:.2f} in potential profit")
    else:
        print("N/A (no filtered signals)")

    # Top / worst trades
    def _fmt(t: Trade) -> str:
        return (f"  {t.strategy:<28}  {t.symbol:<10}  "
                f"{t.reason:<12}  {t.pnl_usdt:>+8.2f}$  "
                f"({t.pnl_pct:>+6.2f}%)  {t.bars_held}h")

    print("\nTop 5 trades by P&L:")
    for t in sorted(trades, key=lambda x: x.pnl_usdt, reverse=True)[:5]:
        print(_fmt(t))

    print("\nWorst 5 trades:")
    for t in sorted(trades, key=lambda x: x.pnl_usdt)[:5]:
        print(_fmt(t))

    # Consecutive wins/losses
    results = [1 if t.pnl_usdt > 0 else -1 for t in trades]
    max_w = max_l = cur_w = cur_l = 0
    for r in results:
        if r > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w); max_l = max(max_l, cur_l)
    print(f"\nMax consecutive wins: {max_w}  |  Max consecutive losses: {max_l}")
    print(f"Total P&L: ${all_pnl2:+.2f}  |  Max drawdown: ${dd:.2f}")
    print("═" * 74)


def print_case3_report(trades: list[Trade], stats: dict, bars: int):
    if not trades:
        print("\nNo trades generated in Case 3.")
        return

    days = bars / 24
    months = bars / 24 / 30.0
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    is_syn     = stats.get("is_synthetic", False)
    data_label = "⚠ SYNTHETIC GBM DATA" if is_syn else "REAL MARKET DATA"
    W = 86
    print("\n" + "═" * W)
    print(f"{'  CASE 3: No-Chief Portfolio  ' + data_label:^{W}}")
    print(f"{'  CPKRegime · ProfitableBot · ScalpTrend':^{W}}")
    print(f"{'  1 pos/strategy · max %d slots total · ~%.0f months · %s' % (MAX_POSITIONS, months, now_str):^{W}}")
    print("═" * W)

    by_strat: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_strat[t.strategy].append(t)

    print(f"\n{'Strategy':<30} {'#':>4} {'/mo':>5} {'WR%':>6} {'BEP%':>6} "
          f"{'PnL$':>9} {'AvgPnL%':>8} {'MaxDD$':>8} {'Hold':>5}")
    print("─" * W)

    strat_order = ["CPKRegimeStrategy", "ProfitableBot", "ScalpTrendBot"]
    all_pnl = 0.0
    for name in strat_order:
        ts = by_strat.get(name, [])
        if not ts:
            print(f"{name:<30} {'—':>4}")
            continue
        wins    = sum(1 for t in ts if t.pnl_usdt > 0)
        wr      = wins / len(ts) * 100
        bep     = _bep(ts)
        pnl, dd = _equity_curve(ts)
        avg_pct = sum(t.pnl_pct for t in ts) / len(ts)
        avg_h   = sum(t.bars_held for t in ts) / len(ts)
        per_mo  = len(ts) / months if months else 0
        ok      = pnl > 0 and per_mo >= 5
        flag    = "  ✓" if ok else "  ⚠"
        print(f"{name:<30} {len(ts):>4} {per_mo:>5.1f} {wr:>5.1f}% {bep:>5.1f}% "
              f"{pnl:>+9.2f} {avg_pct:>+7.2f}% {dd:>+8.2f} {avg_h:>4.1f}h{flag}")
        all_pnl += pnl

    print("─" * W)
    all_wins     = sum(1 for t in trades if t.pnl_usdt > 0)
    all_pnl2, dd = _equity_curve(trades)
    avg_h_all    = sum(t.bars_held for t in trades) / len(trades)
    print(f"{'TOTAL':<30} {len(trades):>4} {len(trades)/months:>5.1f} "
          f"{all_wins/len(trades)*100:>5.1f}%        "
          f"{all_pnl2:>+9.2f} {sum(t.pnl_pct for t in trades)/len(trades):>+7.2f}% "
          f"{dd:>+8.2f} {avg_h_all:>4.1f}h")

    # By symbol
    print(f"\n{'Symbol':<14} {'#':>4} {'WR%':>6} {'PnL$':>9} {'BEP%':>6}")
    print("─" * 44)
    by_sym: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_sym[t.symbol].append(t)
    for sym, ts in sorted(by_sym.items()):
        wins = sum(1 for t in ts if t.pnl_usdt > 0)
        pnl  = sum(t.pnl_usdt for t in ts)
        bep  = _bep(ts)
        print(f"{sym:<14} {len(ts):>4} {wins/len(ts)*100:>5.1f}% {pnl:>+9.2f} {bep:>5.1f}%")

    # Exit reasons
    print("\nExit reason breakdown:")
    print(f"  {'Reason':<14} {'Count':>5} {'%':>6} {'Avg PnL$':>10}")
    print("  " + "─" * 38)
    reasons = Counter(t.reason for t in trades)
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pct     = cnt / len(trades) * 100
        avg_pnl = sum(t.pnl_usdt for t in trades if t.reason == r) / cnt
        print(f"  {r:<14} {cnt:>5} {pct:>5.1f}% {avg_pnl:>+10.2f}")

    # Blocking stats
    blocked = stats.get("blocked", {})
    maxpos_blocked = stats.get("maxpos_blocked", 0)
    if blocked or maxpos_blocked:
        print(f"\n{'─'*W}")
        print(f"Slot contention  (max {MAX_POSITIONS} positions — no Chief gate, each SJ independent)")
        print(f"{'─'*W}")
        print(f"  Max-position blocks total: {maxpos_blocked}×")
        if blocked:
            print("  Blocked by strategy (signal ready, but all slots taken):")
            for name, cnt in sorted(blocked.items(), key=lambda x: -x[1]):
                print(f"    {name:<30} {cnt:>5}×")

    # Top / worst trades
    def _fmt(t: Trade) -> str:
        return (f"  {t.strategy:<30}  {t.symbol:<10}  "
                f"{t.reason:<12}  {t.pnl_usdt:>+8.2f}$  "
                f"({t.pnl_pct:>+6.2f}%)  {t.bars_held}h")

    print("\nTop 5 trades by P&L:")
    for t in sorted(trades, key=lambda x: x.pnl_usdt, reverse=True)[:5]:
        print(_fmt(t))

    print("\nWorst 5 trades:")
    for t in sorted(trades, key=lambda x: x.pnl_usdt)[:5]:
        print(_fmt(t))

    print(f"\nTotal P&L: ${all_pnl2:+.2f}  |  Max drawdown: ${dd:.2f}")
    print("═" * W)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    if "--clear" in sys.argv and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print("Cache cleared.")

    print("\n" + "━" * 50)
    print("  Multi-Strategy Backtest — 5 months")
    print("━" * 50)

    print("\n[1/3] Data")
    data, is_synthetic = await fetch_or_load()

    print("\n[2/3] Simulation")
    t0 = time.time()
    ha, ind = prepare(data)

    # Compute extended indicators
    print("  Precomputing extended indicators...", end="", flush=True)
    ind_ext = {sym: _precompute_ext(sym, ha) for sym in SYMBOLS}
    print(" done")

    # Case 1: All strategies standalone
    standalone = run_standalone(ind, ha, ind_ext)

    # Case 2: Swing + CPK + Profitable + Scalp + Chief portfolio
    trades_c2, stats_c2 = run_backtest(ha, ind, ind_ext)
    stats_c2["is_synthetic"] = is_synthetic

    # Case 3: No-Chief portfolio (1 pos/strategy, max 3 slots)
    trades_c3, stats_c3 = run_portfolio_case3(ha, ind, ind_ext)
    stats_c3["is_synthetic"] = is_synthetic
    stats_c3["bars_total"]   = stats_c2["bars_total"]

    elapsed = time.time() - t0
    n_standalone = sum(len(v) for v in standalone.values())
    print(f"  Done: {n_standalone} standalone + {len(trades_c2)} case2 + {len(trades_c3)} case3 portfolio trades in {elapsed:.1f}s")

    print("\n[3/3] Results")
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + "  CASE 1 — All strategies standalone  (no Chief, no slot limit)".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print_standalone(standalone, stats_c2.get("bars_total", 0))

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + "  CASE 2 — Chief (SwingReversal · CPK · ProfitableBot · ScalpTrend)".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print_report(trades_c2, stats_c2)

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + "  CASE 3 — No-Chief · CPK + Profitable + Scalp · 1 pos/SJ · max 3 slots".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print_case3_report(trades_c3, stats_c3, stats_c3["bars_total"])


if __name__ == "__main__":
    asyncio.run(main())
