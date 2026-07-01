"""
Backtest + Parameter Optimization for AdaptiveTradingBot V8
============================================================
Runs baseline then grid-searches key thresholds.

Usage:
    cd app && python backtest/run_optimize.py
"""
import sys, os, copy, json, logging
from dataclasses import asdict
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP  = os.path.dirname(_HERE)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import numpy as np
import pandas as pd

from backtest.backtest_engine import (
    BacktestConfig, SymbolBacktest, compute_metrics, TradeRecord
)
import trading.adaptive_trading_bot as _bot_mod

logging.basicConfig(
    level=logging.WARNING,   # quiet during grid search
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimize")

DATA_ROOT  = os.path.join(_HERE, "backtest_data")
OUTPUT_DIR = os.path.join(_HERE, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOL = "BTC"
BASE_CFG = BacktestConfig(
    initial_balance = 10_000.0,
    risk_pct        = 0.01,
    tp1_close_pct   = 0.50,
    daily_loss_pct  = -3.0,
    daily_profit_pct= 8.0,
    cooldown_min    = 20,
    max_loss_streak = 4,
    warmup_15m      = 60,
    commission_pct  = 0.0005,
    slippage_pct    = 0.0002,
    data_root       = DATA_ROOT,
    output_dir      = OUTPUT_DIR,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def run_bt(cfg=BASE_CFG) -> dict:
    """Run one full backtest and return metrics dict."""
    runner  = SymbolBacktest(SYMBOL, cfg, DATA_ROOT)
    trades  = runner.run()
    metrics = compute_metrics(trades, cfg.initial_balance, SYMBOL)
    metrics["_trades"] = trades
    return metrics


def score(m: dict) -> float:
    """Composite objective: net_pnl weighted by profit factor, penalise drawdown."""
    if m.get("total_trades", 0) < 5:
        return -9999.0
    pf = min(m.get("profit_factor", 0), 5.0)
    dd = m.get("max_drawdown_pct", 100)
    wr = m.get("win_rate", 0)
    return m["net_pnl"] * pf * (1 - dd / 100) * (wr + 0.2)


def patch_thresholds(overrides: dict):
    """Patch ADAPTIVE_THRESHOLDS in-place for the current run."""
    for state, vals in overrides.items():
        for k, v in vals.items():
            _bot_mod.ADAPTIVE_THRESHOLDS[state][k] = v


def restore_thresholds(original: dict):
    """Restore original ADAPTIVE_THRESHOLDS."""
    for state, vals in original.items():
        for k, v in vals.items():
            _bot_mod.ADAPTIVE_THRESHOLDS[state][k] = v


def print_state_table(trades: list, label: str = ""):
    """Print per-state and per-strategy breakdown."""
    from collections import defaultdict
    by_state: dict = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    by_strat: dict = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

    for t in trades:
        s = t.market_state or "?"
        st = t.strategy or "SR"
        for d in (by_state[s], by_strat[st]):
            d["pnl"] += t.pnl
            if t.result == "WIN":
                d["wins"] += 1
            else:
                d["losses"] += 1

    if label:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")

    print(f"\n{'Market State':<18} {'Trades':>6} {'WR%':>7} {'Net PnL':>10}")
    print("-" * 45)
    for state in sorted(by_state):
        v    = by_state[state]
        tot  = v["wins"] + v["losses"]
        wr   = v["wins"] / tot * 100 if tot else 0
        print(f"{state:<18} {tot:>6} {wr:>6.0f}%  {v['pnl']:>+10.2f}")

    print(f"\n{'Strategy':<18} {'Trades':>6} {'WR%':>7} {'Net PnL':>10}")
    print("-" * 45)
    for st in sorted(by_strat):
        v   = by_strat[st]
        tot = v["wins"] + v["losses"]
        wr  = v["wins"] / tot * 100 if tot else 0
        print(f"{st:<18} {tot:>6} {wr:>6.0f}%  {v['pnl']:>+10.2f}")


# ── STEP 1: Baseline ─────────────────────────────────────────────────────────

def run_baseline():
    print("\n" + "=" * 60)
    print("  STEP 1 — BASELINE (SR + MR, current params)")
    print("=" * 60)

    m = run_bt()
    trades = m.pop("_trades")

    print(f"\n  Trades : {m['total_trades']}")
    print(f"  Win%   : {m['win_rate']*100:.1f}%")
    print(f"  Net PnL: ${m['net_pnl']:+.2f}  ({m['net_pnl_pct']:+.2f}%)")
    print(f"  PF     : {m['profit_factor']:.2f}")
    print(f"  MaxDD  : {m['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe : {m['sharpe']:.3f}")
    print_state_table(trades, "Baseline breakdown")

    return m, trades


# ── STEP 2: Grid search on health_min + confidence_min ───────────────────────

def run_grid_search(baseline_score: float):
    print("\n" + "=" * 60)
    print("  STEP 2 — GRID SEARCH (health_min / confidence_min)")
    print("=" * 60)

    orig = copy.deepcopy(_bot_mod.ADAPTIVE_THRESHOLDS)

    # Grid: delta on health_min and confidence_min across all tradeable states
    health_deltas = [-5, 0, +5]
    conf_deltas   = [-5, 0, +5]

    results = []
    total = len(health_deltas) * len(conf_deltas)
    n = 0

    for hd, cd in product(health_deltas, conf_deltas):
        n += 1
        # Build overrides
        overrides = {}
        for state in _bot_mod._TRADEABLE_STATES:
            thrs = orig[state]
            overrides[state] = {
                "health_min":     min(95, max(50, thrs["health_min"]     + hd)),
                "confidence_min": min(90, max(50, thrs["confidence_min"] + cd)),
            }
        patch_thresholds(overrides)
        m = run_bt()
        m.pop("_trades")
        s = score(m)
        print(f"  [{n:2d}/{total}] hd={hd:+d} cd={cd:+d}  "
              f"trades={m['total_trades']:3d} wr={m['win_rate']*100:.0f}% "
              f"pnl=${m['net_pnl']:+.0f} pf={m['profit_factor']:.2f}  score={s:.1f}")
        results.append({
            "health_delta": hd, "conf_delta": cd,
            "score": s, **{k: m[k] for k in
                           ("total_trades","win_rate","net_pnl","profit_factor","max_drawdown_pct","sharpe")}
        })
        restore_thresholds(orig)

    restore_thresholds(orig)
    results.sort(key=lambda r: r["score"], reverse=True)

    best = results[0]
    print(f"\n  BEST: health_delta={best['health_delta']:+d}  conf_delta={best['conf_delta']:+d}")
    print(f"        trades={best['total_trades']}  wr={best['win_rate']*100:.1f}%  "
          f"pnl=${best['net_pnl']:+.2f}  pf={best['profit_factor']:.2f}  score={best['score']:.1f}")

    if best["score"] <= baseline_score:
        print("\n  → Baseline is already optimal (or grid found no improvement).")
        return None

    return best


# ── STEP 3: Per-state RSI range tune ─────────────────────────────────────────

def run_rsi_tune(bad_states: list):
    """For states with WR < 45%, tighten RSI range by ±5 pts."""
    if not bad_states:
        print("\n  → No bad states to tune.")
        return {}

    print(f"\n  Tuning RSI for states: {bad_states}")
    orig = copy.deepcopy(_bot_mod.ADAPTIVE_THRESHOLDS)
    best_by_state = {}

    for state in bad_states:
        thrs = orig[state]
        lo_l, hi_l = thrs.get("rsi_long",  (30, 60))
        lo_s, hi_s = thrs.get("rsi_short", (40, 70))
        best_s, best_params = -9999, {}

        # Tighten lower long entry (need more OS) and upper short entry (need more OB)
        for tighten in (0, 5, 10):
            ov = {state: {
                "rsi_long":  (lo_l + tighten, hi_l - tighten // 2),
                "rsi_short": (lo_s + tighten // 2, hi_s - tighten),
            }}
            patch_thresholds(ov)
            m = run_bt()
            m.pop("_trades")
            s = score(m)
            if s > best_s:
                best_s = s
                best_params = ov[state].copy()
            restore_thresholds(orig)

        if best_params:
            best_by_state[state] = best_params
            print(f"    {state}: rsi_long={best_params.get('rsi_long')}  "
                  f"rsi_short={best_params.get('rsi_short')}  score={best_s:.1f}")

    restore_thresholds(orig)
    return best_by_state


# ── STEP 4: Apply best params and final run ───────────────────────────────────

def apply_and_final(best_grid, best_rsi):
    print("\n" + "=" * 60)
    print("  STEP 4 — FINAL RUN with tuned params")
    print("=" * 60)

    orig = copy.deepcopy(_bot_mod.ADAPTIVE_THRESHOLDS)

    # Apply grid deltas
    if best_grid:
        hd = best_grid["health_delta"]
        cd = best_grid["conf_delta"]
        overrides = {}
        for state in _bot_mod._TRADEABLE_STATES:
            thrs = orig[state]
            overrides[state] = {
                "health_min":     min(95, max(50, thrs["health_min"]     + hd)),
                "confidence_min": min(90, max(50, thrs["confidence_min"] + cd)),
            }
        patch_thresholds(overrides)

    # Apply RSI overrides
    patch_thresholds(best_rsi)

    m = run_bt()
    trades = m.pop("_trades")

    print(f"\n  Trades : {m['total_trades']}")
    print(f"  Win%   : {m['win_rate']*100:.1f}%")
    print(f"  Net PnL: ${m['net_pnl']:+.2f}  ({m['net_pnl_pct']:+.2f}%)")
    print(f"  PF     : {m['profit_factor']:.2f}")
    print(f"  MaxDD  : {m['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe : {m['sharpe']:.3f}")
    print_state_table(trades, "Tuned breakdown")

    # Print final tuned thresholds
    tuned = {s: dict(_bot_mod.ADAPTIVE_THRESHOLDS[s]) for s in _bot_mod._TRADEABLE_STATES}
    print("\n  TUNED ADAPTIVE_THRESHOLDS (copy into adaptive_trading_bot.py):")
    print("-" * 60)
    for state, vals in tuned.items():
        print(f"  '{state}': health_min={vals['health_min']}  "
              f"confidence_min={vals['confidence_min']}  "
              f"rsi_long={vals.get('rsi_long')}  rsi_short={vals.get('rsi_short')}")

    restore_thresholds(orig)

    # Save final metrics
    out_path = os.path.join(OUTPUT_DIR, "optimized_metrics_BTC.json")
    with open(out_path, "w") as f:
        json.dump({**m, "tuned_thresholds": tuned}, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}")

    return m, tuned


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: baseline
    base_m, base_trades = run_baseline()
    base_score = score(base_m)
    print(f"\n  Baseline score: {base_score:.1f}")

    # Identify bad states (WR < 45%)
    bad_states = [
        state for state, v in base_m.get("by_market_state", {}).items()
        if v.get("win_rate", 1.0) < 0.45 and v.get("total", 0) >= 3
    ]
    print(f"\n  Under-performing states (WR<45%): {bad_states or 'none'}")

    # Step 2: grid search
    best_grid = run_grid_search(base_score)

    # Step 3: RSI tune for bad states
    best_rsi = run_rsi_tune(bad_states)

    # Step 4: final run + print tuned params
    if best_grid or best_rsi:
        apply_and_final(best_grid, best_rsi)
    else:
        print("\n  No improvement found. Baseline params are best.")

    print("\nDone.")
