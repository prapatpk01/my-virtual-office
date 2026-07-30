"""HMA16 Trend-Follow backtest — replays 15m bars through the SAME strategy.py
the live bot uses (HMA16TrendFollowStrategy). Entry fills at the next 15m open
(+slippage); intrabar SL is checked before TP (conservative); the HMA16 opposite
flip exits on bar close; fees charged per fill.

Usage:
    python backtest.py <csv_dir> <PREFIX> [fee_per_side]

<csv_dir> holds Binance-style klines named <PREFIX>USDT-15m-*.csv (nested ok).
Reports trades / WR / PF / sumR / maxDD and the TP/SL/FLIP exit breakdown, where
1R = the fixed stop distance (stop_loss_pct of entry).
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

import strategy as S

SLIP = 0.0003


def load_15m(csv_dir: str, prefix: str) -> pd.DataFrame:
    rows = {}
    for f in glob.glob(f"{csv_dir}/**/{prefix}USDT-15m-*.csv", recursive=True):
        if "__MACOSX" in f:
            continue
        with open(f) as fh:
            for line in fh:
                p = line.strip().split(",")
                if len(p) < 6:
                    continue
                try:
                    ts = int(float(p[0]))
                    while ts > 2_000_000_000_000:
                        ts //= 1000
                    rows[ts] = tuple(float(x) for x in p[1:6])
                except ValueError:
                    continue
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    idx = pd.to_datetime(sorted(rows), unit="ms", utc=True)
    return pd.DataFrame([rows[int(t.value // 1_000_000)] for t in idx],
                        columns=["open", "high", "low", "close", "volume"], index=idx)


def run_symbol(csv_dir: str, prefix: str, fee: float = 0.0005,
               cfg: S.StrategyConfig | None = None) -> dict:
    """Fast replay: indicators are causal, so add_indicators() over the FULL
    frame equals recomputing per-window — precompute once (O(n)) and drive the
    loop with the strategy's own row-level helpers (identical logic to
    generate_entry / evaluate_exit, no divergence)."""
    cfg = cfg or S.StrategyConfig()
    strat = S.HMA16TrendFollowStrategy(cfg)
    raw = load_15m(csv_dir, prefix)
    if len(raw) < 120:
        return dict(symbol=prefix, trades=0, note="insufficient data")
    df = strat.add_indicators(raw)

    sl_pct, tp_pct = cfg.stop_loss_pct, cfg.take_profit_pct
    fee_r = 2 * fee / sl_pct     # round-trip fee expressed in R (risk = sl_pct of price)
    warm = 80
    trades = []
    pos = None

    for i in range(warm, len(df) - 1):
        prev, row = df.iloc[i - 1], df.iloc[i]
        if pos is not None:
            longp = pos["side"] == S.Side.LONG
            h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
            exit_px = why = None
            # 1) hard TP/SL intrabar — stop checked before target (conservative)
            if longp:
                if l <= pos["sl"]:
                    exit_px, why = pos["sl"] * (1 - SLIP), "SL"
                elif h >= pos["tp"]:
                    exit_px, why = pos["tp"] * (1 - SLIP), "TP"
            else:
                if h >= pos["sl"]:
                    exit_px, why = pos["sl"] * (1 + SLIP), "SL"
                elif l <= pos["tp"]:
                    exit_px, why = pos["tp"] * (1 + SLIP), "TP"
            # 2) HMA16 opposite flip on this closed bar (evaluate_exit logic)
            if exit_px is None:
                flip = (strat._flip_down(prev, row) if longp
                        else strat._flip_up(prev, row))
                if flip:
                    exit_px, why = c * (1 - SLIP if longp else 1 + SLIP), "FLIP"
            if exit_px is not None:
                gross = ((exit_px - pos["entry"]) if longp else (pos["entry"] - exit_px))
                trades.append(dict(r=gross / pos["risk"] - fee_r, why=why))
                pos = None
            continue

        # entry — mirror generate_entry() using the same row-level helpers
        if not strat._quality_gate_common(row):
            continue
        trend = strat.classify_trend(row)
        side = None
        if trend == S.Trend.BULL and strat._flip_up(prev, row):
            if strat._long_chase_ok(row)[0] and \
               strat.trend_quality_score(row, S.Side.LONG) >= cfg.min_trend_quality:
                side = S.Side.LONG
        elif trend == S.Trend.BEAR and strat._flip_down(prev, row):
            if strat._short_chase_ok(row)[0] and \
               strat.trend_quality_score(row, S.Side.SHORT) >= cfg.min_trend_quality:
                side = S.Side.SHORT
        if side is None:
            continue
        # fill at NEXT bar open with adverse slippage
        longp = side == S.Side.LONG
        fill = float(df.iloc[i + 1]["open"]) * (1 + SLIP if longp else 1 - SLIP)
        pos = dict(side=side, entry=fill, risk=fill * sl_pct,
                   sl=fill * (1 - sl_pct) if longp else fill * (1 + sl_pct),
                   tp=fill * (1 + tp_pct) if longp else fill * (1 - tp_pct))

    return _metrics(prefix, trades)


def _metrics(prefix: str, trades: list) -> dict:
    if not trades:
        return dict(symbol=prefix, trades=0, wr=0, pf=0, sumR=0, maxDD=0,
                    tp=0, sl=0, flip=0)
    r = np.array([t["r"] for t in trades], float)
    wins, losses = r[r > 0], r[r < 0]
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0, eq])[:-1]
    dd = (eq - peak).min() if len(eq) else 0.0
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    return dict(symbol=prefix, trades=len(r), wr=100 * len(wins) / len(r),
                pf=pf, sumR=r.sum(), maxDD=dd,
                tp=sum(t["why"] == "TP" for t in trades),
                sl=sum(t["why"] == "SL" for t in trades),
                flip=sum(t["why"] == "FLIP" for t in trades))


def _fmt(m: dict) -> str:
    if m.get("trades", 0) == 0:
        return f"{m['symbol']:5s} 0 trades ({m.get('note','')})"
    pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return (f"{m['symbol']:5s} trades={m['trades']:4d}  WR={m['wr']:5.1f}%  PF={pf:>5}  "
            f"sumR={m['sumR']:+7.1f}  maxDD={m['maxDD']:6.1f}  "
            f"[TP {m['tp']} / SL {m['sl']} / FLIP {m['flip']}]")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python backtest.py <csv_dir> <PREFIX> [fee_per_side]")
        sys.exit(1)
    csv_dir, prefix = sys.argv[1], sys.argv[2]
    fee = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0005
    print(_fmt(run_symbol(csv_dir, prefix, fee)))
