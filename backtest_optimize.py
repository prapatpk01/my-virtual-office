"""
Parameter Optimization Script: Walk-Forward Grid Search
Strategies: MCDXStrategy | SentinelStrategy | RSIMACDStrategy
TRX/USDT 1H  |  950 synthetic bars  |  seed=77  |  start=0.1050

Walk-forward:
  WARMUP  = 200 bars
  3 stages × 250 bars live
  Rolling context: last 250 bars per bar

Fixed SL = 4.6%  |  TP = 4.8%  |  LOOKFWD = 48

Usage:
    python backtest_optimize.py
"""
import asyncio
import sys
import os
import math
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy
from app.trading.strategies.sentinel_strategy import SentinelStrategy
from app.trading.strategies.rsi_macd import RSIMACDStrategy
from app.trading.strategies.base import Signal, SignalType

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOL     = "TRX/USDT"
STAGE_BARS = 250
N_STAGES   = 3
WARMUP     = 200
TOTAL_BARS = WARMUP + STAGE_BARS * N_STAGES  # 950
LOOKFWD    = 48
SL_PCT     = 0.046   # 4.6%
TP_PCT     = 0.048   # 4.8%
RR         = TP_PCT / SL_PCT   # ≈ 1.043
CTX_WINDOW = 250     # rolling context window


# ── Synthetic TRX Data ────────────────────────────────────────────────────────
def generate_synthetic_trx(n: int = 950, seed: int = 77) -> list:
    """
    Synthetic TRX/USDT-like 1H candles.
    GBM + trend regimes calibrated to TRX characteristics:
      - Base price ~$0.10, moderate-high crypto volatility
      - Regime length ≈ 7 days (168 bars)
    6 phases: bull, bear, sideways, strong bull, strong bear, tight range.
    """
    rng = np.random.default_rng(seed)
    start_price = 0.1050

    closes = [start_price]
    regime_len = 168  # 7 days per regime at 1H
    phases = [
        (+0.0003, 0.014),  # bull trend
        (-0.0002, 0.014),  # bear trend
        ( 0.0001, 0.008),  # sideways
        (+0.0004, 0.018),  # strong bull
        (-0.0003, 0.016),  # strong bear
        ( 0.0000, 0.007),  # tight range
    ]
    for i in range(1, n):
        phase_idx = (i // regime_len) % len(phases)
        drift, vol = phases[phase_idx]
        shock = rng.normal(drift, vol)
        closes.append(max(closes[-1] * (1 + shock), 0.0001))

    bar_ms = 60 * 60 * 1000      # 1 hour in ms
    base_ts = 1_700_000_000_000  # ~Nov 2023 reference

    candles = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        spread = abs(rng.normal(0, c * 0.003))
        h = max(o, c) + spread
        l = min(o, c) - spread
        vol_size = float(rng.uniform(500_000, 5_000_000))
        candles.append(OHLCV(
            timestamp=base_ts + i * bar_ms,
            open=round(o, 6), high=round(h, 6),
            low=round(l, 6),  close=round(c, 6),
            volume=vol_size,
        ))
    return candles


# ── Exit Scanner ──────────────────────────────────────────────────────────────
def find_exit(candles: list, entry_idx: int, direction: int,
              sl_p: float, tp_p: float) -> tuple:
    """Returns (exit_bar, outcome): +1=TP hit, -1=SL hit, 0=timeout."""
    limit = min(entry_idx + LOOKFWD + 1, len(candles))
    for j in range(entry_idx + 1, limit):
        h, l = candles[j].high, candles[j].low
        if direction == 1:    # long
            if l <= sl_p: return j, -1
            if h >= tp_p: return j, +1
        else:                  # short
            if h >= sl_p: return j, -1
            if l <= tp_p: return j, +1
    return min(entry_idx + LOOKFWD, len(candles) - 1), 0


# ── Stage Runner ──────────────────────────────────────────────────────────────
async def run_stage(strategy, candles: list, stage_start: int,
                    stage_end: int) -> list:
    """
    Walk through bars [stage_start, stage_end).
    Each bar: rolling context of last CTX_WINDOW bars.
    Fixed SL/TP percentages. 1 position at a time.
    """
    trades = []
    lock_until = -1

    for i in range(stage_start, stage_end):
        if i <= lock_until:
            continue

        # Rolling context: last CTX_WINDOW bars up to and including bar i
        ctx_start = max(0, i - CTX_WINDOW + 1)
        context = candles[ctx_start: i + 1]
        current_price = candles[i].close

        try:
            signal = await strategy.analyze(context, current_price, mtf_candles=None)
        except Exception:
            continue

        if signal.type == SignalType.HOLD:
            continue

        direction = 1 if signal.type == SignalType.BUY else -1

        # Fixed SL / TP
        if direction == 1:   # long
            sl_p = current_price * (1.0 - SL_PCT)
            tp_p = current_price * (1.0 + TP_PCT)
        else:                 # short
            sl_p = current_price * (1.0 + SL_PCT)
            tp_p = current_price * (1.0 - TP_PCT)

        exit_bar, outcome = find_exit(candles, i, direction, sl_p, tp_p)

        # PnL in R
        pnl_r = RR if outcome == 1 else (-1.0 if outcome == -1 else 0.0)

        trades.append({
            "bar":      i,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry":    current_price,
            "sl":       sl_p,
            "tp":       tp_p,
            "exit_bar": exit_bar,
            "outcome":  outcome,
            "pnl_r":    pnl_r,
        })

        lock_until = exit_bar

    return trades


# ── Stats Calculator ──────────────────────────────────────────────────────────
def calc_stats(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                "wr": 0.0, "total_r": 0.0}
    wins     = sum(1 for t in trades if t["outcome"] ==  1)
    losses   = sum(1 for t in trades if t["outcome"] == -1)
    timeouts = sum(1 for t in trades if t["outcome"] ==  0)
    decided  = wins + losses
    wr       = wins / max(decided, 1) * 100
    total_r  = sum(t["pnl_r"] for t in trades)
    return {
        "trades":   len(trades),
        "wins":     wins,
        "losses":   losses,
        "timeouts": timeouts,
        "wr":       round(wr, 1),
        "total_r":  round(total_r, 3),
    }


# ── Grid Search Runner ────────────────────────────────────────────────────────
async def grid_search(strategy_name: str, param_grid: list,
                      candles: list, stages: list) -> list:
    """
    Run all parameter combos in param_grid.
    param_grid: list of dicts, each is one set of params.
    Returns list of result dicts sorted by WR desc.
    """
    results = []

    for params in param_grid:
        # Instantiate fresh strategy for each combo
        if strategy_name == "MCDX":
            strategy = MCDXStrategy(SYMBOL, params=params)
        elif strategy_name == "Sentinel":
            strategy = SentinelStrategy(SYMBOL, params=params)
        elif strategy_name == "RSI_MACD":
            strategy = RSIMACDStrategy(SYMBOL, params=params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        all_trades = []
        for (start, end) in stages:
            stage_trades = await run_stage(strategy, candles, start, end)
            all_trades.extend(stage_trades)

        stats = calc_stats(all_trades)
        results.append({"params": params, **stats})

    # Sort by WR desc, then total_r desc as tiebreaker
    results.sort(key=lambda r: (r["wr"], r["total_r"]), reverse=True)
    return results


# ── Report Printer ────────────────────────────────────────────────────────────
def print_results(strategy_name: str, results: list):
    W = 78
    print(f"\n{'═' * W}")
    print(f"  GRID SEARCH RESULTS  |  {strategy_name}")
    print(f"{'═' * W}")
    print(f"  {'Params':<38}  {'Trades':>6}  {'W/L/T':<12}  {'WR%':>6}  {'Total_R':>8}")
    print(f"  {'─'*70}")

    best = None
    for r in results:
        params_str = "  ".join(f"{k}={v}" for k, v in r["params"].items())
        wlt = f"{r['wins']}W/{r['losses']}L/{r['timeouts']}T"
        target = " ★ TARGET" if r["wr"] >= 67.0 else ""
        print(f"  {params_str:<38}  {r['trades']:>6}  {wlt:<12}  "
              f"{r['wr']:>5.1f}%  {r['total_r']:>+8.3f}R{target}")
        if best is None and r["trades"] >= 5:
            best = r

    if best:
        print(f"\n  >> Best (≥5 trades): params={best['params']}  "
              f"WR={best['wr']:.1f}%  Trades={best['trades']}  "
              f"Total_R={best['total_r']:+.3f}R")
    else:
        print(f"\n  >> No combo reached ≥5 trades.")

    return best


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    W = 78
    print("\n" + "═" * W)
    print("  PARAMETER OPTIMIZATION  |  TRX/USDT 1H  |  Walk-Forward Grid Search")
    print(f"  SL={SL_PCT*100:.1f}%  TP={TP_PCT*100:.1f}%  RR={RR:.3f}  "
          f"LOOKFWD={LOOKFWD}  CTX_WINDOW={CTX_WINDOW}")
    print(f"  WARMUP={WARMUP}  Stages={N_STAGES}×{STAGE_BARS}  Total={TOTAL_BARS} bars")
    print("═" * W)

    # Generate synthetic candles
    print(f"\n  Generating {TOTAL_BARS} synthetic TRX/USDT 1H bars (seed=77, start=0.1050)...")
    candles = generate_synthetic_trx(n=TOTAL_BARS, seed=77)
    price_lo = min(c.low  for c in candles)
    price_hi = max(c.high for c in candles)
    print(f"  Price range: ${price_lo:.5f} – ${price_hi:.5f}")

    # Stage boundaries
    stages = []
    for s in range(N_STAGES):
        start = WARMUP + s * STAGE_BARS
        end   = min(start + STAGE_BARS, TOTAL_BARS)
        stages.append((start, end))
    print(f"  Stages: {stages}")

    # ── MCDX Grid ────────────────────────────────────────────────────────────
    mcdx_pairs = [
        (53, 47),
        (55, 45),
        (57, 43),
        (60, 40),
        (62, 38),
    ]
    mcdx_grid = [{"dwcs_buy": b, "dwcs_sell": s} for b, s in mcdx_pairs]

    print(f"\n  Running MCDXStrategy grid ({len(mcdx_grid)} combos)...")
    mcdx_results = await grid_search("MCDX", mcdx_grid, candles, stages)
    mcdx_best = print_results("MCDXStrategy", mcdx_results)

    # ── Sentinel Grid ─────────────────────────────────────────────────────────
    sentinel_grid = [{"min_conf": mc} for mc in [60, 63, 66, 69, 72, 75]]

    print(f"\n  Running SentinelStrategy grid ({len(sentinel_grid)} combos)...")
    sentinel_results = await grid_search("Sentinel", sentinel_grid, candles, stages)
    sentinel_best = print_results("SentinelStrategy", sentinel_results)

    # ── RSI+MACD Grid ─────────────────────────────────────────────────────────
    rsi_macd_grid = [
        {"rsi_oversold": os_, "rsi_overbought": ob_, "mtf_threshold": 0}
        for os_, ob_ in itertools.product([38, 40, 42], [58, 60, 62])
    ]

    print(f"\n  Running RSIMACDStrategy grid ({len(rsi_macd_grid)} combos)...")
    rsi_macd_results = await grid_search("RSI_MACD", rsi_macd_grid, candles, stages)
    rsi_macd_best = print_results("RSIMACDStrategy", rsi_macd_results)

    # ── Final Recommendation ──────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  FINAL RECOMMENDATION")
    print(f"{'═' * W}")

    bests = [
        ("MCDXStrategy",    mcdx_best),
        ("SentinelStrategy", sentinel_best),
        ("RSIMACDStrategy",  rsi_macd_best),
    ]

    combined_wr_numerator   = 0.0
    combined_wr_denominator = 0
    all_have_data = True

    for name, b in bests:
        if b and b["trades"] >= 5:
            print(f"\n  {name}")
            print(f"    Best params : {b['params']}")
            print(f"    Trades      : {b['trades']}  ({b['wins']}W / {b['losses']}L / {b['timeouts']}T)")
            print(f"    Win Rate    : {b['wr']:.1f}%")
            print(f"    Total R     : {b['total_r']:+.3f}R")
            decided = b["wins"] + b["losses"]
            combined_wr_numerator   += b["wins"]
            combined_wr_denominator += decided
        else:
            print(f"\n  {name}: insufficient data (< 5 trades with best params)")
            all_have_data = False

    if combined_wr_denominator > 0:
        combined_wr = combined_wr_numerator / combined_wr_denominator * 100
        print(f"\n  {'─'*60}")
        print(f"  Combined expected WR (all 3 strategies, decided trades only):")
        print(f"    Total decided trades : {combined_wr_denominator}")
        print(f"    Total wins           : {int(combined_wr_numerator)}")
        print(f"    Expected WR          : {combined_wr:.1f}%")
        if combined_wr >= 67.0:
            print(f"    ★ ABOVE 67% TARGET!")
        else:
            print(f"    (Below 67% target — consider further tuning)")
    else:
        print(f"\n  No strategies have enough decided trades to compute combined WR.")

    print(f"\n{'═' * W}")
    print("  Optimization complete.")
    print("═" * W + "\n")


if __name__ == "__main__":
    asyncio.run(main())
