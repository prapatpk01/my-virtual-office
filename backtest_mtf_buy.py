"""
BTC/USDT 15m Buy-Only  |  MTF Bias: 1H + 4H EMA confirmation
Entry: MCDX signal on 15m  |  Filter: 1H BULL + 4H BULL  |  No shorts

Cases:
  B) SL=1.9%  TP=2.2%  → net RR 1:0.91  BE WR=52.4%
  C) SL=1.6%  TP=1.8%  → net RR 1:0.84  BE WR=54.4%

MTF bias: EMA_fast > EMA_slow on 1H AND 4H → BULL → allow entry
15m candles aggregated → 1H (×4) and 4H (×16) for bias computation.

Usage:
    python backtest_mtf_buy.py
"""
import asyncio
import sys
import os
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy
from app.trading.strategies.base import SignalType

# ── Config ────────────────────────────────────────────────────────────────────
WARMUP     = 800          # 15m bars warmup (800/16=50 complete 4H bars for EMA50)
STAGE_BARS = 480          # 5 days of 15m bars per stage (480 × 15min = 120h)
N_STAGES   = 5            # 5 stages ≈ 25 trading days
CTX_WINDOW = 300          # MCDX context window (15m bars)
FETCH_BARS = WARMUP + STAGE_BARS * N_STAGES
LOOKFWD    = 96           # 24h max hold at 15m
FEE        = 0.0025       # 0.10%+0.10%+0.05% per trade on notional
LEVERAGE   = 10
RISK_USD   = 100.0
SEEDS      = [42, 77, 99]
TARGET_WR  = 67.0
MIN_TRADES = 8

# ── SL/TP cases ───────────────────────────────────────────────────────────────
CASES = [
    ("B", 0.019, 0.022),  # SL 1.9%  TP 2.2%
    ("C", 0.016, 0.018),  # SL 1.6%  TP 1.8%
]

# ── MCDX entry params (15m) ───────────────────────────────────────────────────
MCDX_GRID = [
    {"dwcs_buy": b, "dwcs_sell": 100 - b, "rvol_min": r}
    for b in [52, 55, 57, 60]
    for r in [0.8, 1.0, 1.2]
]

# ── MTF bias params: (1H_fast, 1H_slow, 4H_fast, 4H_slow) ────────────────────
MTF_GRID = [
    ( 9, 21,  9, 21),   # aggressive: short EMAs both TFs
    ( 9, 21, 13, 34),   # 1H short / 4H medium
    (21, 50, 13, 34),   # 1H medium / 4H medium
    (21, 50, 21, 50),   # balanced
    ( 9, 21, 21, 50),   # 1H sensitive / 4H slow
]

BAR_15M = 15 * 60 * 1000


# ── Helpers ───────────────────────────────────────────────────────────────────

def ema_calc(values: list, period: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def generate_btc_15m(n: int, seed: int) -> list:
    """Synthetic BTC 15m GBM: vol ~0.25%/bar, 6 regime phases."""
    rng = np.random.default_rng(seed)
    closes = [65_000.0]
    phases = [
        (+0.00008, 0.0025),   # bull
        (-0.00005, 0.0025),   # bear
        (+0.00002, 0.0018),   # sideways
        (+0.00015, 0.0035),   # strong bull
        (-0.00012, 0.0035),   # strong bear
        (+0.00000, 0.0010),   # tight range
    ]
    regime_len = max(STAGE_BARS, 4 * 168)  # 7 days at 15m = 672 bars
    for i in range(1, n):
        drift, vol = phases[(i // regime_len) % len(phases)]
        closes.append(max(closes[-1] * (1 + rng.normal(drift, vol)), 1.0))
    candles = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        sp = abs(rng.normal(0, c * 0.0008))
        candles.append(OHLCV(
            BAR_15M * i,
            round(o, 2), round(max(o, c) + sp, 2),
            round(min(o, c) - sp, 2), round(c, 2),
            float(rng.uniform(50, 500)),
        ))
    return candles


def aggregate(candles_15m: list, factor: int) -> list:
    """Aggregate 15m → higher TF. Bar k = 15m bars [k*factor .. (k+1)*factor-1]."""
    out = []
    n = len(candles_15m)
    for k in range(n // factor):
        chunk = candles_15m[k * factor:(k + 1) * factor]
        out.append(OHLCV(
            chunk[0].timestamp,
            chunk[0].open,
            max(c.high for c in chunk),
            min(c.low  for c in chunk),
            chunk[-1].close,
            sum(c.volume for c in chunk),
        ))
    return out


def precompute_bias(candles_1h, candles_4h, f1h, s1h, f4h, s4h):
    """Pre-compute EMA arrays for both higher TFs."""
    c1h = [c.close for c in candles_1h]
    c4h = [c.close for c in candles_4h]
    return (
        ema_calc(c1h, f1h), ema_calc(c1h, s1h),
        ema_calc(c4h, f4h), ema_calc(c4h, s4h),
    )


def bias_at(ema_f: np.ndarray, ema_s: np.ndarray, idx: int) -> int:
    """Return +1 BULL / -1 BEAR / 0 neutral at HTF index idx."""
    if idx < 0 or idx >= len(ema_f) or np.isnan(ema_f[idx]) or np.isnan(ema_s[idx]):
        return 0
    return 1 if ema_f[idx] > ema_s[idx] else -1


def find_exit(candles, entry_idx, sl_p, tp_p):
    """Find exit bar for long trade (buy-only)."""
    for j in range(entry_idx + 1, min(entry_idx + LOOKFWD + 1, len(candles))):
        h, l = candles[j].high, candles[j].low
        if l <= sl_p: return j, -1
        if h >= tp_p: return j, +1
    return min(entry_idx + LOOKFWD, len(candles) - 1), 0


# ── Stage runner ──────────────────────────────────────────────────────────────

async def run_stage(strategy, candles_15m, ema1h_f, ema1h_s, ema4h_f, ema4h_s,
                    start, end, sl_pct, tp_pct):
    net_win = tp_pct - FEE
    net_los = sl_pct + FEE
    trades, lock = [], -1

    for i in range(start, end):
        if i <= lock:
            continue

        # Map 15m bar i → last completed 1H / 4H index (no lookahead)
        # 1H bar k covers 15m bars [4k .. 4k+3]; complete when i reaches 4k+3
        idx_1h = (i - 3) // 4       # last complete 1H bar index
        idx_4h = (i - 15) // 16     # last complete 4H bar index

        if idx_1h < 0 or idx_4h < 0:
            continue

        if bias_at(ema1h_f, ema1h_s, idx_1h) != 1:
            continue   # 1H not bull
        if bias_at(ema4h_f, ema4h_s, idx_4h) != 1:
            continue   # 4H not bull

        # Run MCDX on 15m context
        ctx   = candles_15m[max(0, i - CTX_WINDOW + 1):i + 1]
        price = candles_15m[i].close
        try:
            sig = await strategy.analyze(ctx, price, mtf_candles=None)
        except Exception:
            continue

        if sig.type != SignalType.BUY:   # buy-only: skip SELL signals
            continue

        sl_p = price * (1 - sl_pct)
        tp_p = price * (1 + tp_pct)
        eb, oc = find_exit(candles_15m, i, sl_p, tp_p)

        net_r = (net_win / sl_pct if oc ==  1 else
                 -(net_los / sl_pct) if oc == -1 else
                 -(FEE / sl_pct))
        trades.append({"outcome": oc, "net_r": net_r})
        lock = eb

    return trades


def calc(trades, sl_pct, tp_pct):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net_r": 0.0, "pnl_usd": 0.0,
                "wins": 0, "losses": 0, "timo": 0}
    net_win = tp_pct - FEE
    net_los = sl_pct + FEE
    wins = [t for t in trades if t["outcome"] ==  1]
    loss = [t for t in trades if t["outcome"] == -1]
    timo = [t for t in trades if t["outcome"] ==  0]
    dec  = len(wins) + len(loss)
    wr   = len(wins) / max(dec, 1) * 100
    net_r   = sum(t["net_r"] for t in trades)
    pnl_usd = net_r * sl_pct * LEVERAGE * RISK_USD
    pf      = (len(wins) * net_win) / max(len(loss) * net_los, 1e-9)
    return {
        "n": len(trades), "wins": len(wins), "losses": len(loss), "timo": len(timo),
        "wr": round(wr, 1), "pf": round(pf, 2),
        "net_r": round(net_r, 2), "pnl_usd": round(pnl_usd, 1),
    }


# ── Combo runner ──────────────────────────────────────────────────────────────

async def run_combo(mcdx_params, mtf_params, all_data, sl_pct, tp_pct):
    h1f, h1s, h4f, h4s = mtf_params
    per_seed = []
    for candles_15m, candles_1h, candles_4h in all_data:
        ema1h_f, ema1h_s, ema4h_f, ema4h_s = precompute_bias(
            candles_1h, candles_4h, h1f, h1s, h4f, h4s
        )
        stages = [(WARMUP + s * STAGE_BARS,
                   min(WARMUP + (s + 1) * STAGE_BARS, len(candles_15m)))
                  for s in range(N_STAGES)]
        strat = MCDXStrategy("BTC/USDT", mcdx_params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(
                strat, candles_15m,
                ema1h_f, ema1h_s, ema4h_f, ema4h_s,
                start, end, sl_pct, tp_pct,
            )
        per_seed.append(calc(all_trades, sl_pct, tp_pct))

    n_s = len(per_seed)
    return {
        "n":       sum(s["n"]       for s in per_seed) / n_s,
        "wr":      sum(s["wr"]      for s in per_seed) / n_s,
        "pf":      sum(s["pf"]      for s in per_seed) / n_s,
        "net_r":   sum(s["net_r"]   for s in per_seed) / n_s,
        "pnl_usd": sum(s["pnl_usd"] for s in per_seed) / n_s,
        "timo":    sum(s["timo"]    for s in per_seed) / n_s,
        "per_seed": per_seed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    W = 96
    print("\n" + "═" * W)
    print(f"  BTC/USDT 15m Buy-Only  |  MTF Bias: 1H + 4H EMA  |  Target WR≥{TARGET_WR:.0f}%")
    print(f"  Entry: MCDX on 15m  |  Seeds={SEEDS}  |  {N_STAGES} stages × {STAGE_BARS} bars ({STAGE_BARS*15//60}h each)")
    print(f"  Lookahead-free bias: use last completed 1H/4H bar before each 15m entry")
    for name, sl, tp in CASES:
        net_win = tp - FEE; net_los = sl + FEE
        be_wr = net_los / (net_win + net_los) * 100
        rr    = net_win / net_los
        print(f"    Case {name}: SL={sl*100:.1f}%  TP={tp*100:.1f}%  "
              f"net RR 1:{rr:.2f}  BE WR={be_wr:.1f}%")
    print("═" * W)

    # Generate and aggregate candles
    print(f"\n  Generating {len(SEEDS)} seeds × {FETCH_BARS} bars (15m)...")
    all_data = []
    for seed in SEEDS:
        c15 = generate_btc_15m(FETCH_BARS, seed)
        c1h = aggregate(c15, 4)
        c4h = aggregate(c15, 16)
        all_data.append((c15, c1h, c4h))
        print(f"    seed={seed}  15m={len(c15)}  1H={len(c1h)}  4H={len(c4h)}  "
              f"BTC=${c15[0].close:,.0f}→${c15[-1].close:,.0f}")

    n_combos = len(MCDX_GRID) * len(MTF_GRID)
    print(f"\n  Running {n_combos} combos × {len(CASES)} cases × {len(SEEDS)} seeds "
          f"= {n_combos * len(CASES) * len(SEEDS)} total runs...")

    # Run all combos
    all_results = {}   # case_key → [(label, avg, mcdx_p, mtf_p)]
    for case_name, sl, tp in CASES:
        results = []
        for mcdx_p, mtf_p in product(MCDX_GRID, MTF_GRID):
            avg = await run_combo(mcdx_p, mtf_p, all_data, sl, tp)
            h1f, h1s, h4f, h4s = mtf_p
            lbl = (f"buy={mcdx_p['dwcs_buy']} rv≥{mcdx_p['rvol_min']}  "
                   f"1H({h1f}/{h1s}) 4H({h4f}/{h4s})")
            results.append((lbl, avg, mcdx_p, mtf_p))
        all_results[case_name] = (sl, tp, results)

    # ── Print results per case ────────────────────────────────────────────────
    for case_name, (sl, tp, results) in all_results.items():
        net_win = tp - FEE; net_los = sl + FEE
        be_wr   = net_los / (net_win + net_los) * 100

        passed = [r for r in results
                  if r[1]["wr"] >= TARGET_WR and r[1]["n"] >= MIN_TRADES]
        passed.sort(key=lambda x: (-x[1]["wr"], -x[1]["net_r"]))
        top_all = sorted(results, key=lambda x: (-x[1]["wr"], -x[1]["net_r"]))

        print(f"\n{'═' * W}")
        status = f"★ {len(passed)} combos pass WR≥{TARGET_WR:.0f}%" if passed else f"— best below {TARGET_WR:.0f}%"
        print(f"  Case {case_name}: SL={sl*100:.1f}%  TP={tp*100:.1f}%  "
              f"BE WR={be_wr:.1f}%  |  {status}")
        print(f"  {'─' * 92}")
        print(f"  {'Label':<52}  {'N':>5}  {'WR%':>6}  {'PF':>5}  "
              f"{'NetR':>6}  {'USD':>7}  {'To%':>4}")
        print(f"  {'─' * 92}")

        shown = 0
        for lbl, s, _, _ in top_all:
            if shown >= 15:
                break
            timo_pct = s["timo"] / max(s["n"], 1) * 100
            flag = ("★" if s["wr"] >= TARGET_WR and s["n"] >= MIN_TRADES else
                    "▲" if s["wr"] >= be_wr and s["n"] >= MIN_TRADES else " ")
            print(f"  {lbl:<52}  {s['n']:>5.1f}  {s['wr']:>5.1f}%  "
                  f"{s['pf']:>5.2f}  {s['net_r']:>+6.2f}  "
                  f"${s['pnl_usd']:>+5.1f}  {timo_pct:>3.0f}% {flag}")
            shown += 1

        # Per-seed for top combo
        best = passed[0] if passed else top_all[0]
        b_lbl, b_s, b_mcdx, b_mtf = best
        print(f"\n  Per-seed breakdown → {b_lbl}")
        for seed, ps in zip(SEEDS, b_s["per_seed"]):
            timo_pct = ps["timo"] / max(ps["n"], 1) * 100
            flag = "✓" if ps["wr"] >= TARGET_WR and ps["n"] >= MIN_TRADES else "✗"
            print(f"    seed={seed}  N={ps['n']:>3}  WR={ps['wr']:>5.1f}%  "
                  f"PF={ps['pf']:>5.2f}  Net-R={ps['net_r']:>+6.2f}  "
                  f"Timo={timo_pct:.0f}%  {flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  SUMMARY  (best per case by WR, N≥{MIN_TRADES})")
    print(f"  {'─' * 92}")
    for case_name, (sl, tp, results) in all_results.items():
        net_win = tp - FEE; net_los = sl + FEE
        be_wr   = net_los / (net_win + net_los) * 100
        cands   = [r for r in results if r[1]["n"] >= MIN_TRADES]
        if not cands:
            print(f"  Case {case_name}: no combo with N≥{MIN_TRADES}")
            continue
        best = max(cands, key=lambda x: x[1]["wr"])
        lbl, s, mcdx_p, mtf_p = best
        h1f, h1s, h4f, h4s = mtf_p
        flag = ("★" if s["wr"] >= TARGET_WR else "▲" if s["wr"] >= be_wr else " ")
        print(f"\n  {flag} Case {case_name} (SL={sl*100:.1f}%/TP={tp*100:.1f}%)")
        print(f"    {lbl}")
        print(f"    Avg N={s['n']:.1f}  WR={s['wr']:.1f}%  PF={s['pf']:.2f}  "
              f"Net-R={s['net_r']:+.2f}  ~${s['pnl_usd']:+.1f} (10x $100/trade)")
        print(f"    MCDX: dwcs_buy={mcdx_p['dwcs_buy']}  rvol_min={mcdx_p['rvol_min']}")
        print(f"    MTF:  1H EMA({h1f},{h1s})  4H EMA({h4f},{h4s})")
        print(f"    .env: STOP_LOSS_PCT={sl}  TAKE_PROFIT_PCT={tp}  CANDLE_TF=15m")

    print(f"\n  ★=WR≥{TARGET_WR:.0f}%  ▲=WR≥BE  To%=timeout rate")
    print("═" * W + "\n")


if __name__ == "__main__":
    asyncio.run(main())
