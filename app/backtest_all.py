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
MONTHS        = 5
TRADE_USDT    = float(os.environ.get("TRADE_AMOUNT_USDT", "100"))
MAX_POSITIONS = 3
FEE_RT        = 0.0020   # 0.10% in + 0.10% out
CACHE_FILE    = Path(__file__).parent / ".bt_cache_5m.json"

CHIEF_N_AGREE = int(os.environ.get("CHIEF_N_AGREE",  "2"))
CHIEF_LOOKFWD = int(os.environ.get("CHIEF_LOOKFWD", "100"))


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
        ex = ex_cls({"enableRateLimit": True})
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


def run_standalone(ind: dict, ha: dict) -> dict[str, list[Trade]]:
    """Per-strategy standalone backtest (each SJ on its own slot)."""
    out: dict[str, list[Trade]] = {name: [] for name in STRAT_ORDER}
    for name in STRAT_ORDER:
        for sym in SYMBOLS:
            out[name].extend(_simulate_standalone(sym, name, ind, ha))
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
def run_backtest(ha: dict, ind: dict) -> tuple[list[Trade], dict]:
    master_1h = ha[SYMBOLS[0]]["1h"]
    n_bars    = len(master_1h)

    # O(1) timestamp → bar-index lookup per symbol
    ts_map: dict[str, dict[int, int]] = {
        sym: {int(c.timestamp): i for i, c in enumerate(ha[sym]["1h"])}
        for sym in SYMBOLS
    }

    # Swing strategy configs: (name, sl_mult, tp_mult, rsi_lo, rsi_hi, vol_mult, cooldown_bars)
    _SWING = [
        ("SwingReversalStrategy", 2.5, 1.5, 34.0, 62.0, 1.2, 4),
        ("CPKRegimeStrategy",     2.5, 1.5, 40.0, 64.0, 1.2, 5),
        ("HybridSwingStrategy",   2.5, 2.0, 38.0, 64.0, 1.2, 3),
    ]
    ALL_STRAT_NAMES = [sn for sn, *_ in _SWING] + ["InternStrategy"]
    SWING_WARMUP  = 210
    INTERN_WARMUP = 60

    positions:      list[Position] = []
    trades:         list[Trade]    = []
    blocked:        dict[str, int] = defaultdict(int)
    chief_approved: list[dict]     = []
    chief_filtered: list[dict]     = []

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

    print(f"  Simulating {n_bars} bars × {len(SYMBOLS)} symbols × "
          f"{len(ALL_STRAT_NAMES)} strategies...")

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
            cp = float(d["closes"][i])
            av = float(d["atr14"][i])
            if math.isnan(av):
                av = cp * 0.015

            # ── Intern SELL (checked before BUY, no cooldown on exits) ──
            sell = False
            if i >= INTERN_WARMUP:
                hp = float(d["hma15"][i - 1]); hc = float(d["hma15"][i])
                c_p = float(d["closes"][i - 1]); o_c = float(d["opens"][i])
                if not (math.isnan(hp) or math.isnan(hc)):
                    sell = c_p < hp and o_c < hc

            pos = sym_pos(sym)
            if pos:
                if sell:
                    close_pos(pos, cp, "sell_signal", bar_i)
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

            # InternStrategy BUY
            key_in = (sym, "InternStrategy")
            if cooldown[key_in] > 0:
                cooldown[key_in] -= 1
            elif i >= INTERN_WARMUP:
                hp  = float(d["hma15"][i - 1]); hc  = float(d["hma15"][i])
                c_p = float(d["closes"][i - 1])
                o_c = float(d["opens"][i]);      o_p = float(d["opens"][i - 1])
                if not (math.isnan(hp) or math.isnan(hc)):
                    if (c_p > hp and o_c > hc and o_c > o_p
                            and d["bias_15m"][i] >= 1.0
                            and d["bias_30m"][i] >= 1.0):
                        sl_p = round(cp - 2.5 * av, 4)
                        tp_p = round(cp + 2.0 * av, 4)
                        if sl_p < cp < tp_p:
                            buy_sigs.append(("InternStrategy", sl_p, tp_p))
                            cooldown[key_in] = 3

            if not buy_sigs:
                continue

            n_agree = len(buy_sigs)

            # 3-slot limit
            if len(positions) >= MAX_POSITIONS:
                for sn, _, _ in buy_sigs:
                    blocked[sn] += 1
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
        "chief_approved": chief_approved,
        "chief_filtered": chief_filtered,
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
        "HybridSwingStrategy",   "InternStrategy",
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

    # (a) standalone per-SJ
    standalone = run_standalone(ind, ha)
    # (b) portfolio with Chief consensus gate
    trades, stats = run_backtest(ha, ind)
    stats["is_synthetic"] = is_synthetic
    elapsed = time.time() - t0
    n_standalone = sum(len(v) for v in standalone.values())
    print(f"  Done: {n_standalone} standalone + {len(trades)} portfolio trades "
          f"in {elapsed:.1f}s")

    print("\n[3/3] Results")
    print_standalone(standalone, stats.get("bars_total", 0))
    print_report(trades, stats)


if __name__ == "__main__":
    asyncio.run(main())
