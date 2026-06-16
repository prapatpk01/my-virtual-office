"""
TRX Parameter Optimizer
Grid search: SL/TP tiers × strategy param variants
Target: WR ≥ 66% AND Trades ≥ 10 per month (750-bar window)

Usage:
    python backtest_optimize_trx.py
"""
import asyncio
import sys
import os
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy
from app.trading.strategies.sentinel_strategy import SentinelStrategy
from app.trading.strategies.rsi_macd import RSIMACDStrategy
from app.trading.strategies.base import SignalType

# ── TRX synthetic data ────────────────────────────────────────────────────────
WARMUP     = 200
STAGE_BARS = 250
N_STAGES   = 3
CTX_WINDOW = 250
FETCH_BARS = WARMUP + STAGE_BARS * N_STAGES
LOOKFWD    = 48

TRX_START  = 0.1050
TRX_VOL    = 0.75   # scale vs ETH baseline
TRX_SEED   = 77


def generate_synthetic(n: int, start: float, vol_scale: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    closes = [start]
    regime_len = 168
    phases = [
        (+0.0003, 0.014 * vol_scale),
        (-0.0002, 0.014 * vol_scale),
        ( 0.0001, 0.008 * vol_scale),
        (+0.0004, 0.018 * vol_scale),
        (-0.0003, 0.016 * vol_scale),
        ( 0.0000, 0.007 * vol_scale),
    ]
    for i in range(1, n):
        phase_idx = (i // regime_len) % len(phases)
        drift, vol = phases[phase_idx]
        closes.append(max(closes[-1] * (1 + rng.normal(drift, vol)), 0.0001))
    bar_ms  = 60 * 60 * 1000
    base_ts = 1_700_000_000_000
    candles = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        sp = abs(rng.normal(0, c * 0.003))
        candles.append(OHLCV(
            timestamp=base_ts + i * bar_ms,
            open=round(o, 6), high=round(max(o, c) + sp, 6),
            low=round(min(o, c) - sp, 6), close=round(c, 6),
            volume=float(rng.uniform(500_000, 5_000_000)),
        ))
    return candles


def find_exit(candles, entry_idx, direction, sl_p, tp_p):
    for j in range(entry_idx + 1, min(entry_idx + LOOKFWD + 1, len(candles))):
        h, l = candles[j].high, candles[j].low
        if direction == 1:
            if l <= sl_p: return j, -1
            if h >= tp_p: return j, +1
        else:
            if h >= sl_p: return j, -1
            if l <= tp_p: return j, +1
    return min(entry_idx + LOOKFWD, len(candles) - 1), 0


async def run_stage(strategy, candles, start, end, sl_pct, tp_pct):
    trades, lock_until = [], -1
    rr = tp_pct / sl_pct
    for i in range(start, end):
        if i <= lock_until:
            continue
        ctx  = candles[max(0, i - CTX_WINDOW + 1):i + 1]
        price = candles[i].close
        try:
            sig = await strategy.analyze(ctx, price, mtf_candles=None)
        except Exception:
            continue
        if sig.type == SignalType.HOLD:
            continue
        d = 1 if sig.type == SignalType.BUY else -1
        sl_p = price * (1 - sl_pct) if d == 1 else price * (1 + sl_pct)
        tp_p = price * (1 + tp_pct) if d == 1 else price * (1 - tp_pct)
        eb, oc = find_exit(candles, i, d, sl_p, tp_p)
        trades.append({"outcome": oc, "pnl_r": rr if oc == 1 else (-1.0 if oc == -1 else 0.0)})
        lock_until = eb
    return trades


def calc(trades):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "r": 0.0}
    wins = [t for t in trades if t["outcome"] ==  1]
    loss = [t for t in trades if t["outcome"] == -1]
    dec  = len(wins) + len(loss)
    rr   = trades[0]["pnl_r"] if wins else 1.0
    return {
        "n":  len(trades),
        "wr": round(len(wins) / max(dec, 1) * 100, 1),
        "pf": round((len(wins) * rr) / max(len(loss), 1e-9), 2),
        "r":  round(sum(t["pnl_r"] for t in trades), 2),
        "to": len(trades) - dec,
    }


# ── Parameter grids ───────────────────────────────────────────────────────────
SL_TP_GRID = [
    (0.025, 0.027),  # tight — fits TRX low-vol tight range
    (0.030, 0.032),  # balanced
    (0.035, 0.038),  # moderate
    (0.040, 0.043),  # wider
    (0.046, 0.048),  # original ETH params
]

MCDX_PARAMS = [
    {"dwcs_buy": 55, "dwcs_sell": 45, "rvol_min": 1.0},
    {"dwcs_buy": 57, "dwcs_sell": 43, "rvol_min": 1.1},
    {"dwcs_buy": 60, "dwcs_sell": 40, "rvol_min": 1.2},
]

SENTINEL_PARAMS = [
    {"min_conf": 50, "fresh_bos_bars": 30, "dwcs_bull_min": 45},
    {"min_conf": 55, "fresh_bos_bars": 25, "dwcs_bull_min": 50},
    {"min_conf": 58, "fresh_bos_bars": 20, "dwcs_bull_min": 52},
]

RSIMACD_PARAMS = [
    {"rsi_oversold": 38, "rsi_overbought": 60},
    {"rsi_oversold": 40, "rsi_overbought": 58},
    {"rsi_oversold": 42, "rsi_overbought": 56},
]


async def main():
    W = 74
    print("\n" + "═" * W)
    print("  TRX/USDT PARAMETER OPTIMIZER  |  1H  |  3 × 250 bars")
    print(f"  Target: WR ≥ 66%  AND  Trades ≥ 10/month")
    print("═" * W)

    candles = generate_synthetic(FETCH_BARS, TRX_START, TRX_VOL, TRX_SEED)
    stages  = [(WARMUP + s * STAGE_BARS,
                min(WARMUP + (s + 1) * STAGE_BARS, len(candles)))
               for s in range(N_STAGES)]

    print(f"  Synthetic TRX: {len(candles)} bars  "
          f"vol×{TRX_VOL}  seed={TRX_SEED}")
    print(f"  Price: ${min(c.low for c in candles):.5f} – "
          f"${max(c.high for c in candles):.5f}\n")

    results = []  # (strategy_name, label, sl_pct, tp_pct, stats)

    # ── MCDX sweep ────────────────────────────────────────────────────────────
    print(f"  [1/3] MCDX sweep ({len(SL_TP_GRID)} SL/TP × {len(MCDX_PARAMS)} param sets)...")
    best_mcdx = None
    for (sl, tp), params in product(SL_TP_GRID, MCDX_PARAMS):
        strat = MCDXStrategy("TRX/USDT", params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl, tp)
        s = calc(all_trades)
        label = f"dwcs_buy={params['dwcs_buy']} rvol≥{params['rvol_min']}"
        results.append(("MCDX", label, sl, tp, s))
        if s["n"] >= 10 and s["wr"] >= 66.0:
            if best_mcdx is None or s["r"] > best_mcdx[4]["r"]:
                best_mcdx = ("MCDX", label, sl, tp, s)

    # ── Sentinel sweep ────────────────────────────────────────────────────────
    print(f"  [2/3] Sentinel sweep ({len(SL_TP_GRID)} SL/TP × {len(SENTINEL_PARAMS)} param sets)...")
    best_sentinel = None
    for (sl, tp), params in product(SL_TP_GRID, SENTINEL_PARAMS):
        strat = SentinelStrategy("TRX/USDT", params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl, tp)
        s = calc(all_trades)
        label = f"conf={params['min_conf']} fresh={params['fresh_bos_bars']}"
        results.append(("Sentinel", label, sl, tp, s))
        if s["n"] >= 10 and s["wr"] >= 66.0:
            if best_sentinel is None or s["r"] > best_sentinel[4]["r"]:
                best_sentinel = ("Sentinel", label, sl, tp, s)

    # ── RSI+MACD sweep ────────────────────────────────────────────────────────
    print(f"  [3/3] RSI+MACD sweep ({len(SL_TP_GRID)} SL/TP × {len(RSIMACD_PARAMS)} param sets)...")
    best_rsimacd = None
    for (sl, tp), params in product(SL_TP_GRID, RSIMACD_PARAMS):
        strat = RSIMACDStrategy("TRX/USDT", params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl, tp)
        s = calc(all_trades)
        label = f"oversold={params['rsi_oversold']} overbought={params['rsi_overbought']}"
        results.append(("RSI+MACD", label, sl, tp, s))
        if s["n"] >= 10 and s["wr"] >= 66.0:
            if best_rsimacd is None or s["r"] > best_rsimacd[4]["r"]:
                best_rsimacd = ("RSI+MACD", label, sl, tp, s)

    # ── Print top results per strategy ────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Strategy':<10}  {'Params':<36}  {'SL%':>4}  {'TP%':>4}  "
          f"{'N':>4}  {'WR%':>6}  {'PF':>5}  {'R':>6}")
    print(f"  {'─' * 70}")

    for strat_name in ["MCDX", "Sentinel", "RSI+MACD"]:
        subset = [(r) for r in results if r[0] == strat_name]
        # Sort by: target met first, then by total_r
        subset.sort(key=lambda x: (
            -(1 if x[4]["n"] >= 10 and x[4]["wr"] >= 66 else 0),
            -x[4]["r"]
        ))
        for i, (sn, lbl, sl, tp, s) in enumerate(subset[:5]):
            star = " ★" if s["n"] >= 10 and s["wr"] >= 66.0 else "  "
            print(f"  {sn:<10}  {lbl:<36}  {sl*100:.1f}%  {tp*100:.1f}%  "
                  f"{s['n']:>4}  {s['wr']:>5.1f}%{star}  {s['pf']:>5.2f}  {s['r']:>+6.2f}R")
        print()

    # ── Best configs summary ───────────────────────────────────────────────────
    print(f"{'═' * W}")
    print("  BEST CONFIG PER STRATEGY (WR ≥ 66% AND Trades ≥ 10)")
    print(f"  {'─' * 70}")

    best_list = [b for b in [best_mcdx, best_sentinel, best_rsimacd] if b]
    if best_list:
        for sn, lbl, sl, tp, s in best_list:
            rr = round(tp / sl, 3)
            print(f"\n  ★ {sn}")
            print(f"    Params : {lbl}")
            print(f"    SL/TP  : {sl*100:.1f}% / {tp*100:.1f}%  (RR 1:{rr})")
            print(f"    Result : {s['n']} trades  WR={s['wr']}%  PF={s['pf']}  "
                  f"R={s['r']:+.2f}  (timeouts={s.get('to', 0)})")
    else:
        # Show closest if none pass both thresholds
        print("\n  ไม่มี config ผ่านทั้ง WR≥66% และ Trades≥10 พร้อมกัน")
        print("  Closest results:")
        # Best WR among those with ≥10 trades
        cands = [(sn, lbl, sl, tp, s) for sn, lbl, sl, tp, s in results if s["n"] >= 10]
        cands.sort(key=lambda x: -x[4]["wr"])
        for sn, lbl, sl, tp, s in cands[:6]:
            print(f"    {sn:<10}  {lbl:<36}  SL={sl*100:.1f}%  "
                  f"N={s['n']}  WR={s['wr']}%  R={s['r']:+.2f}R")

    print(f"\n{'═' * W}\n")


if __name__ == "__main__":
    asyncio.run(main())
