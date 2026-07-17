"""Walk-forward harness — rolling in-sample/out-of-sample splits over the
same run_backtest used everywhere else. V1.4 does not auto-optimize
parameters (anti-overfitting stance); the harness measures OOS stability
of the fixed config, and accepts an optional param_grid for manual studies.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

from .backtest_engine import run_backtest
from .config import Config, TF_MS


def split_windows(candles_15m: list, n_windows: int, oos_frac: float = 0.25) -> list:
    """[(is_start, is_end, oos_end)] as timestamps."""
    if not candles_15m or n_windows < 1:
        return []
    ts = [c.timestamp for c in candles_15m]
    total = len(ts)
    win = total // n_windows
    out = []
    for k in range(n_windows):
        s = k * win
        e = min(s + win, total - 1)
        oos_len = int(win * oos_frac)
        is_end = max(s + win - oos_len, s + 1)
        out.append((ts[s], ts[min(is_end, total - 1)], ts[e]))
    return out


def _slice(data: dict, start_ms: int, end_ms: int, warmup_bars: int) -> dict:
    out = {}
    pre = warmup_bars * TF_MS["15m"]
    for sym, tfs in data.items():
        out[sym] = {tf: [c for c in cs if start_ms - pre * (4 if tf == "1h" else 16 if tf == "4h" else 1)
                         <= c.timestamp <= end_ms]
                    for tf, cs in tfs.items()}
    return out


async def walk_forward(cfg: Config, data: dict, n_windows: int = 4,
                       oos_frac: float = 0.25,
                       mutate: Optional[Callable[[Config, int], Config]] = None) -> dict:
    """Runs each window's OOS segment with the (optionally mutated) config.
    Returns per-window results + aggregate OOS stats."""
    any_sym = next(iter(data))
    windows = split_windows(data[any_sym]["15m"], n_windows, oos_frac)
    results = []
    for k, (is_start, is_end, oos_end) in enumerate(windows):
        c = copy.deepcopy(cfg)
        if mutate is not None:
            c = mutate(c, k)
        oos_data = _slice(data, is_end, oos_end, cfg.min_15m_candles)
        res = await run_backtest(c, oos_data)
        results.append({"window": k, "oos_start": is_end, "oos_end": oos_end,
                        "trades": res["trades"], "win_rate": round(res["win_rate"], 4),
                        "profit_factor": (round(res["profit_factor"], 3)
                                          if res["profit_factor"] != float("inf") else "inf"),
                        "net_pnl": round(res["net_pnl"], 2),
                        "max_dd_pct": round(res["max_drawdown_pct"], 2)})
    pnls = [r["net_pnl"] for r in results]
    pos = sum(1 for p in pnls if p > 0)
    return {"windows": results,
            "oos_positive_windows": f"{pos}/{len(results)}",
            "oos_total_pnl": round(sum(pnls), 2)}
