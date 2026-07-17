"""Monte Carlo — bootstrap resampling of the realized R sequence to estimate
drawdown/ruin distributions (order-dependence stress, not a profit promise).
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np


def monte_carlo(results_r: list, risk_per_trade: float, initial_balance: float = 10_000.0,
                n_paths: int = 2000, n_trades: Optional[int] = None,
                ruin_dd: float = 0.5, seed: Optional[int] = 42) -> dict:
    """results_r: list of per-trade R multiples (e.g. from TradeRecords).
    Resamples WITH replacement into n_paths equity paths."""
    if not results_r:
        return {"paths": 0}
    rng = random.Random(seed)
    n = n_trades or len(results_r)
    finals, max_dds, ruined = [], [], 0
    for _ in range(n_paths):
        bal = initial_balance
        peak = bal
        max_dd = 0.0
        for _ in range(n):
            r = rng.choice(results_r)
            bal += bal * risk_per_trade * r
            peak = max(peak, bal)
            max_dd = max(max_dd, (peak - bal) / peak)
            if max_dd >= ruin_dd:
                ruined += 1
                break
        finals.append(bal)
        max_dds.append(max_dd)
    finals_a = np.array(finals)
    dds = np.array(max_dds)
    return {
        "paths": n_paths, "trades_per_path": n,
        "median_final": float(np.median(finals_a)),
        "p05_final": float(np.percentile(finals_a, 5)),
        "p95_final": float(np.percentile(finals_a, 95)),
        "median_max_dd_pct": float(np.median(dds) * 100),
        "p95_max_dd_pct": float(np.percentile(dds, 95) * 100),
        "prob_ruin_pct": ruined / n_paths * 100,
        "prob_profit_pct": float(np.mean(finals_a > initial_balance) * 100),
    }
