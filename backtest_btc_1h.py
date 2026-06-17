"""
BTC/USDT 1H Fine-tuner — 3 SL/TP pairs, Target WR ≥ 67%
Fee=0.25%/trade on notional  |  Max hold=48h  |  Seeds=[42,77,99]

SL/TP pairs tested:
  A) SL=2.5%  TP=3.0%  → net RR 1:1.00  BE WR=50.0%
  B) SL=1.9%  TP=2.2%  → net RR 1:0.91  BE WR=52.4%
  C) SL=1.6%  TP=1.8%  → net RR 1:0.84  BE WR=54.4%

Usage:
    python backtest_btc_1h.py
"""
import asyncio
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy
from app.trading.strategies.sentinel_strategy import SentinelStrategy
from app.trading.strategies.base import SignalType

# ── Backtest config ───────────────────────────────────────────────────────────
WARMUP     = 300
STAGE_BARS = 200
N_STAGES   = 6
CTX_WINDOW = 300
FETCH_BARS = WARMUP + STAGE_BARS * N_STAGES
LOOKFWD    = 48           # 48h max hold at 1H

FEE_TOTAL  = 0.0025       # 0.10% open + 0.10% close + 0.05% borrow
LEVERAGE   = 10
RISK_USD   = 100.0
SEEDS      = [42, 77, 99]
TARGET_WR  = 67.0
MIN_TRADES = 8

# ── SL/TP pairs ───────────────────────────────────────────────────────────────
SL_TP_PAIRS = [
    (0.025, 0.030),   # A: SL 2.5%  TP 3.0%
    (0.019, 0.022),   # B: SL 1.9%  TP 2.2%
    (0.016, 0.018),   # C: SL 1.6%  TP 1.8%
]

def pair_meta(sl, tp):
    net_win = tp - FEE_TOTAL
    net_los = sl + FEE_TOTAL
    be_wr   = net_los / (net_win + net_los) * 100
    net_rr  = net_win / net_los
    return net_win, net_los, be_wr, net_rr

# ── Parameter grids ───────────────────────────────────────────────────────────
MCDX_GRID = [
    {"dwcs_buy": b, "dwcs_sell": 100 - b, "rvol_min": r}
    for b in [55, 57, 60, 62, 65]
    for r in [1.0, 1.2, 1.4]
]

SENTINEL_GRID = [
    {"min_conf": c, "fresh_bos_bars": f, "dwcs_bull_min": d}
    for c in [60, 65, 70, 75]
    for f in [8, 12, 16, 20]
    for d in [52, 55]
]

BTC_START = 65_000.0
BAR_MS    = 60 * 60 * 1000


def generate_btc_1h(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    closes = [BTC_START]
    phases = [
        (+0.00030, 0.0035),
        (-0.00020, 0.0035),
        (+0.00005, 0.0025),
        (+0.00050, 0.0050),
        (-0.00040, 0.0050),
        (+0.00000, 0.0015),
    ]
    regime_len = max(STAGE_BARS, 168)
    for i in range(1, n):
        drift, vol = phases[(i // regime_len) % len(phases)]
        closes.append(max(closes[-1] * (1 + rng.normal(drift, vol)), 1.0))
    base_ts = 1_700_000_000_000
    candles = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        spread = abs(rng.normal(0, c * 0.0012))
        candles.append(OHLCV(
            timestamp=base_ts + i * BAR_MS,
            open=round(o, 2), high=round(max(o, c) + spread, 2),
            low=round(min(o, c) - spread, 2), close=round(c, 2),
            volume=float(rng.uniform(100, 2000)),
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
    net_win, net_los, _, _ = pair_meta(sl_pct, tp_pct)
    trades, lock_until = [], -1
    for i in range(start, end):
        if i <= lock_until:
            continue
        ctx   = candles[max(0, i - CTX_WINDOW + 1):i + 1]
        price = candles[i].close
        try:
            sig = await strategy.analyze(ctx, price, mtf_candles=None)
        except Exception:
            continue
        if sig.type == SignalType.HOLD:
            continue
        d    = 1 if sig.type == SignalType.BUY else -1
        sl_p = price * (1 - sl_pct) if d == 1 else price * (1 + sl_pct)
        tp_p = price * (1 + tp_pct) if d == 1 else price * (1 - tp_pct)
        eb, oc = find_exit(candles, i, d, sl_p, tp_p)
        if oc == 1:
            net_r = net_win / sl_pct
        elif oc == -1:
            net_r = -(net_los / sl_pct)
        else:
            net_r = -(FEE_TOTAL / sl_pct)
        trades.append({"outcome": oc, "net_r": net_r})
        lock_until = eb
    return trades


def calc(trades, sl_pct, tp_pct):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net_r": 0.0, "pnl_usd": 0.0, "timo": 0}
    net_win, net_los, _, _ = pair_meta(sl_pct, tp_pct)
    wins = [t for t in trades if t["outcome"] ==  1]
    loss = [t for t in trades if t["outcome"] == -1]
    timo = [t for t in trades if t["outcome"] ==  0]
    dec  = len(wins) + len(loss)
    wr   = len(wins) / max(dec, 1) * 100
    net_r   = sum(t["net_r"] for t in trades)
    pnl_usd = net_r * sl_pct * LEVERAGE * RISK_USD
    pf      = (len(wins) * net_win) / max(len(loss) * net_los, 1e-9)
    return {
        "n": len(trades), "wins": len(wins), "losses": len(loss),
        "timo": len(timo),
        "wr": round(wr, 1), "pf": round(pf, 2),
        "net_r": round(net_r, 2), "pnl_usd": round(pnl_usd, 1),
    }


async def run_combo(StratCls, symbol, params, all_candles, sl_pct, tp_pct):
    """Run one param combo across all seeds; return avg stats."""
    per_seed = []
    for candles in all_candles:
        stages = [(WARMUP + s * STAGE_BARS,
                   min(WARMUP + (s + 1) * STAGE_BARS, len(candles)))
                  for s in range(N_STAGES)]
        strat = StratCls(symbol, params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl_pct, tp_pct)
        per_seed.append(calc(all_trades, sl_pct, tp_pct))

    n_s = len(per_seed)
    return {
        "n":        sum(s["n"]       for s in per_seed) / n_s,
        "wr":       sum(s["wr"]      for s in per_seed) / n_s,
        "pf":       sum(s["pf"]      for s in per_seed) / n_s,
        "net_r":    sum(s["net_r"]   for s in per_seed) / n_s,
        "pnl_usd":  sum(s["pnl_usd"] for s in per_seed) / n_s,
        "timo":     sum(s["timo"]    for s in per_seed) / n_s,
        "per_seed": per_seed,
    }


async def main():
    W = 92
    print("\n" + "═" * W)
    print(f"  BTC/USDT 1H  |  Fee={FEE_TOTAL*100:.2f}%  |  Max hold={LOOKFWD}h  "
          f"|  Target WR≥{TARGET_WR:.0f}%  |  Seeds={SEEDS}")
    for sl, tp in SL_TP_PAIRS:
        _, _, be, rr = pair_meta(sl, tp)
        print(f"    SL={sl*100:.1f}%  TP={tp*100:.1f}%  → net RR 1:{rr:.2f}  BE WR={be:.1f}%")
    print("═" * W)

    print(f"\n  Generating candles for {len(SEEDS)} seeds × {FETCH_BARS} bars...")
    all_candles = [generate_btc_1h(FETCH_BARS, s) for s in SEEDS]
    price_lo = min(c.low  for cds in all_candles for c in cds)
    price_hi = max(c.high for cds in all_candles for c in cds)
    print(f"  BTC range: ${price_lo:,.0f} – ${price_hi:,.0f}\n")

    # ── Run all pairs ──────────────────────────────────────────────────────────
    all_results = {}   # (sl, tp) → list of (sname, label, avg, params)

    total_combos = len(MCDX_GRID) + len(SENTINEL_GRID)
    for pair_idx, (sl, tp) in enumerate(SL_TP_PAIRS):
        label_pair = f"SL={sl*100:.1f}%/TP={tp*100:.1f}%"
        print(f"  [{pair_idx+1}/{len(SL_TP_PAIRS)}] {label_pair}  ({total_combos} combos × {len(SEEDS)} seeds)")
        results = []

        for params in MCDX_GRID:
            avg = await run_combo(MCDXStrategy, "BTC/USDT", params, all_candles, sl, tp)
            lbl = f"buy={params['dwcs_buy']} sell={params['dwcs_sell']} rv≥{params['rvol_min']}"
            results.append(("MCDX", lbl, avg, params))

        for params in SENTINEL_GRID:
            avg = await run_combo(SentinelStrategy, "BTC/USDT", params, all_candles, sl, tp)
            lbl = f"conf≥{params['min_conf']} fresh≤{params['fresh_bos_bars']} dwcs≥{params['dwcs_bull_min']}"
            results.append(("Sentinel", lbl, avg, params))

        all_results[(sl, tp)] = results

    # ── Results per SL/TP pair ─────────────────────────────────────────────────
    for sl, tp in SL_TP_PAIRS:
        net_win, net_los, be_wr, net_rr = pair_meta(sl, tp)
        results = all_results[(sl, tp)]

        passed = [r for r in results if r[2]["wr"] >= TARGET_WR and r[2]["n"] >= MIN_TRADES]
        passed.sort(key=lambda x: (-x[2]["wr"], -x[2]["net_r"]))

        print(f"\n{'═' * W}")
        print(f"  SL={sl*100:.1f}%  TP={tp*100:.1f}%  |  net RR 1:{net_rr:.2f}  "
              f"|  BE WR={be_wr:.1f}%  |  {'★ '+str(len(passed))+' combos pass WR≥67%' if passed else 'no combo ≥67% WR'}")
        print(f"  {'─' * 88}")
        print(f"  {'ST':<8}  {'Params':<44}  {'N':>5}  {'WR%':>6}  {'PF':>5}  "
              f"{'NetR':>6}  {'USD':>7}  {'Timo':>5}")
        print(f"  {'─' * 88}")

        # Top 5 passing + top 5 overall
        top_pass = passed[:5]
        top_all  = sorted(results, key=lambda x: (-x[2]["wr"], -x[2]["net_r"]))

        shown_ids = set()
        for rows, marker in [(top_pass, "★"), (top_all, " ")]:
            for sn, lbl, s, _ in rows:
                if id((sn, lbl)) in shown_ids:
                    continue
                shown_ids.add(id((sn, lbl)))
                timo_pct = s["timo"] / max(s["n"], 1) * 100
                flag = "★" if s["wr"] >= TARGET_WR and s["n"] >= MIN_TRADES else ("▲" if s["wr"] >= be_wr and s["n"] >= MIN_TRADES else " ")
                print(f"  {sn:<8}  {lbl:<44}  {s['n']:>5.1f}  {s['wr']:>5.1f}%  "
                      f"{s['pf']:>5.2f}  {s['net_r']:>+6.2f}  "
                      f"${s['pnl_usd']:>+5.1f}  {timo_pct:>4.0f}% {flag}")
                if len(shown_ids) >= 10:
                    break
            if len(shown_ids) >= 10:
                break

        # Per-seed for best combo
        best = passed[0] if passed else sorted(results, key=lambda x: -x[2]["wr"])[0]
        sn, lbl, s, _ = best
        print(f"\n  Per-seed → {sn}  {lbl}")
        for seed, ps in zip(SEEDS, s["per_seed"]):
            timo_pct = ps["timo"] / max(ps["n"], 1) * 100
            flag = "✓" if ps["wr"] >= TARGET_WR and ps["n"] >= MIN_TRADES else "✗"
            print(f"    seed={seed}  N={ps['n']:>3}  WR={ps['wr']:>5.1f}%  "
                  f"PF={ps['pf']:>5.2f}  Net-R={ps['net_r']:>+6.2f}  Timo={timo_pct:.0f}%  {flag}")

    # ── Cross-pair comparison ─────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  CROSS-PAIR SUMMARY  (best combo per SL/TP by WR, N≥{MIN_TRADES})")
    print(f"  {'─' * 88}")
    print(f"  {'SL/TP':<14}  {'BE WR':>6}  {'Best Strategy & Params':<48}  "
          f"{'WR%':>6}  {'NetR':>6}  {'USD':>7}")
    print(f"  {'─' * 88}")
    for sl, tp in SL_TP_PAIRS:
        _, _, be_wr, _ = pair_meta(sl, tp)
        results = all_results[(sl, tp)]
        cands = [r for r in results if r[2]["n"] >= MIN_TRADES]
        if not cands:
            print(f"  SL={sl*100:.1f}%/TP={tp*100:.1f}%  {be_wr:>5.1f}%  — no combo with N≥{MIN_TRADES}")
            continue
        best = max(cands, key=lambda x: x[2]["wr"])
        sn, lbl, s, _ = best
        flag = "★" if s["wr"] >= TARGET_WR else ("▲" if s["wr"] >= be_wr else " ")
        print(f"  SL={sl*100:.1f}%/TP={tp*100:.1f}%  {be_wr:>5.1f}%  "
              f"{sn} {lbl:<46}  {s['wr']:>5.1f}%  {s['net_r']:>+6.2f}  ${s['pnl_usd']:>+5.1f} {flag}")

    print(f"\n  ★=WR≥67%  ▲=WR≥BE  Timo=timeout %  USD=P&L at 10x $100/trade")
    print("═" * W + "\n")


if __name__ == "__main__":
    asyncio.run(main())
