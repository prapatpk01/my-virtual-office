"""HTF bot backtest — replays 1H bars through the SAME strategy.py the live
bot uses. Entry fills at next 1H open (+slippage), SL checked before TP
intrabar (conservative), fees per fill.

Usage:
    python backtest.py <csv_dir> [fee_per_side] [risk_pct]

<csv_dir> must contain Binance-style kline CSVs named *USDT-1h-*.csv and
*USDT-4h-*.csv (nested dirs ok). Reports per-symbol trades/WR/PF/sumR/maxDD.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

import strategy as S

SLIP = 0.0003


def load_tf(csv_dir: str, prefix: str, tf: str) -> pd.DataFrame:
    rows = {}
    for f in glob.glob(f"{csv_dir}/**/{prefix}USDT-{tf}-*.csv", recursive=True):
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
               tp_r: float = 3.0, be_at_r: float = 1.0) -> dict:
    h1 = load_tf(csv_dir, prefix, "1h")
    h4 = load_tf(csv_dir, prefix, "4h")
    if len(h1) < 200 or len(h4) < 60:
        return dict(symbol=prefix, trades=0, note="insufficient data")

    h4_close = h4.index + pd.Timedelta(hours=4)
    trades = []
    pos = None
    for i in range(60, len(h1) - 1):
        bar_close = h1.index[i] + pd.Timedelta(hours=1)
        if pos is not None:
            bar = h1.iloc[i]
            longp = pos["dir"] == S.LONG
            fee_r = 2 * fee * pos["entry"] / pos["risk"]
            if (bar["low"] <= pos["sl"]) if longp else (bar["high"] >= pos["sl"]):
                r = (pos["sl"] - pos["entry"]) / pos["risk"] * (1 if longp else -1)
                trades.append(dict(r=r - fee_r, why="SL" if not pos["be"] else "BE"))
                pos = None
                continue
            if (bar["high"] >= pos["tp"]) if longp else (bar["low"] <= pos["tp"]):
                r = (pos["tp"] - pos["entry"]) / pos["risk"] * (1 if longp else -1)
                trades.append(dict(r=r - fee_r, why="TP"))
                pos = None
                continue
            if not pos["be"]:
                reach = (bar["high"] - pos["entry"]) if longp else (pos["entry"] - bar["low"])
                if reach >= be_at_r * pos["risk"]:
                    pos["sl"] = pos["entry"]
                    pos["be"] = True
            continue

        if S.commodity_halted(prefix, bar_close):
            continue
        window_1h = h1.iloc[max(0, i - 299):i + 1]
        cutoff = int(np.searchsorted(h4_close.values, bar_close.to_datetime64(), side="right"))
        window_4h = h4.iloc[max(0, cutoff - 250):cutoff]
        trend = S.trend_direction(window_4h)
        sig = S.entry_signal(window_1h, trend)
        if sig is None:
            continue
        nxt = h1.iloc[i + 1]
        entry = float(nxt["open"]) * (1 + SLIP if sig.direction == S.LONG else 1 - SLIP)
        sl, tp, dist = S.plan_stop_target(window_1h, sig.direction, entry, sig.atr1h,
                                          tp_r=tp_r)
        pos = dict(dir=sig.direction, entry=entry, sl=sl, tp=tp, risk=dist, be=False)

    if not trades:
        return dict(symbol=prefix, trades=0)
    rs = np.array([t["r"] for t in trades])
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    gl = rs[rs <= 0].sum()
    whys = {}
    for t in trades:
        whys[t["why"]] = whys.get(t["why"], 0) + 1
    return dict(symbol=prefix, trades=len(rs), wr=float((rs > 0).mean() * 100),
                pf=float(rs[rs > 0].sum() / abs(gl)) if gl else float("inf"),
                sum_r=float(rs.sum()), maxdd_r=float(np.max(peak - eq)), whys=whys)


if __name__ == "__main__":
    csv_dir = sys.argv[1]
    fee = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005
    print(f"=== htf_bot backtest | fee {fee*100:.3f}%/side | TP3R BE@1R ===")
    tot_r, tot_n = 0.0, 0
    for pfx in ["BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "XAG", "CL"]:
        r = run_symbol(csv_dir, pfx, fee)
        if r["trades"] == 0:
            print(f"{pfx:5s} | 0 trades {r.get('note','')}")
            continue
        tot_r += r["sum_r"]; tot_n += r["trades"]
        pf = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
        print(f"{pfx:5s} | {r['trades']:3d} tr | WR {r['wr']:4.1f}% | PF {pf:>5} | "
              f"sumR {r['sum_r']:+7.1f} | maxDD {r['maxdd_r']:4.1f}R | {r['whys']}")
    print(f"TOTAL {tot_n} trades, sumR {tot_r:+.1f}")
    print("BACKTEST_DONE")
