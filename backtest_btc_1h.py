"""
BTC/USDT 1H Parameter Optimizer — OKX Cross Margin x10
Fee model: 0.10% open + 0.10% close + 0.05% daily borrow = 0.25%/trade on notional
Intraday: max hold = 8 bars (8h), positions force-closed at day end

Usage:
    python backtest_btc_1h.py
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
from app.trading.strategies.base import SignalType

WARMUP     = 200
STAGE_BARS = 250          # ~10 days of 1H bars
N_STAGES   = 4            # 4 stages ≈ 40 trading days ≈ 2 months
CTX_WINDOW = 250
FETCH_BARS = WARMUP + STAGE_BARS * N_STAGES
LOOKFWD    = 8            # max 8h hold (intraday)

FEE_OPEN   = 0.0010
FEE_CLOSE  = 0.0010
FEE_BORROW = 0.0005       # avg ~4h borrow per trade
FEE_TOTAL  = FEE_OPEN + FEE_CLOSE + FEE_BORROW

LEVERAGE   = 10
RISK_USD   = 100.0

BTC_START  = 65_000.0
BTC_SEED   = 42
BAR_MS     = 60 * 60 * 1000


def generate_btc_1h(n: int, seed: int = BTC_SEED) -> list:
    """GBM 1H BTC: realistic crypto vol ~0.35%/bar, 6 regime phases."""
    rng = np.random.default_rng(seed)
    closes = [BTC_START]
    regime_len = 168   # 7 days per phase
    phases = [
        (+0.00030, 0.0035),   # bull
        (-0.00020, 0.0035),   # bear
        (+0.00005, 0.0025),   # sideways
        (+0.00050, 0.0050),   # strong bull
        (-0.00040, 0.0050),   # strong bear
        (+0.00000, 0.0015),   # tight range
    ]
    for i in range(1, n):
        phase_idx = (i // regime_len) % len(phases)
        drift, vol = phases[phase_idx]
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
            net_pct = tp_pct - FEE_TOTAL
        elif oc == -1:
            net_pct = -(sl_pct + FEE_TOTAL)
        else:
            net_pct = -FEE_TOTAL
        trades.append({"outcome": oc, "net_pct": net_pct, "net_r": net_pct / sl_pct})
        lock_until = eb
    return trades


def calc(trades, sl_pct, tp_pct):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net_r": 0.0, "pnl_usd": 0.0}
    wins = [t for t in trades if t["outcome"] ==  1]
    loss = [t for t in trades if t["outcome"] == -1]
    dec  = len(wins) + len(loss)
    wr   = len(wins) / max(dec, 1) * 100
    net_r   = sum(t["net_r"] for t in trades)
    pnl_usd = sum(t["net_pct"] * LEVERAGE * RISK_USD for t in trades)
    net_tp  = tp_pct - FEE_TOTAL
    net_sl  = sl_pct + FEE_TOTAL
    pf      = (len(wins) * net_tp) / max(len(loss) * net_sl, 1e-9)
    return {
        "n": len(trades), "wins": len(wins), "losses": len(loss),
        "timo": len(trades) - dec,
        "wr": round(wr, 1), "pf": round(pf, 2),
        "net_r": round(net_r, 2), "pnl_usd": round(pnl_usd, 1),
    }


SL_TP_GRID = [
    (0.010, 0.018),   # SL 1.0%, TP 1.8%
    (0.012, 0.020),   # SL 1.2%, TP 2.0%
    (0.015, 0.025),   # SL 1.5%, TP 2.5%  ← likely sweet spot
    (0.015, 0.030),   # SL 1.5%, TP 3.0%
    (0.020, 0.030),   # SL 2.0%, TP 3.0%
    (0.020, 0.040),   # SL 2.0%, TP 4.0%
]

MCDX_PARAMS = [
    {"dwcs_buy": 52, "dwcs_sell": 48, "rvol_min": 1.0},
    {"dwcs_buy": 55, "dwcs_sell": 45, "rvol_min": 1.0},
    {"dwcs_buy": 57, "dwcs_sell": 43, "rvol_min": 1.1},
]

SENTINEL_PARAMS = [
    {"min_conf": 50, "fresh_bos_bars": 30, "dwcs_bull_min": 45},
    {"min_conf": 55, "fresh_bos_bars": 25, "dwcs_bull_min": 50},
    {"min_conf": 58, "fresh_bos_bars": 20, "dwcs_bull_min": 52},
]


async def main():
    W = 82
    print("\n" + "═" * W)
    print("  BTC/USDT 1H OPTIMIZER  |  OKX Cross Margin x10  |  Intraday (max 8h hold)")
    print(f"  Fee: {FEE_TOTAL*100:.2f}%/trade on notional  |  4 stages × 250 bars ≈ 2 months")
    print("═" * W)

    candles = generate_btc_1h(FETCH_BARS)
    stages  = [(WARMUP + s * STAGE_BARS,
                min(WARMUP + (s + 1) * STAGE_BARS, len(candles)))
               for s in range(N_STAGES)]

    print(f"  BTC 1H: {len(candles)} bars  "
          f"${min(c.low for c in candles):,.0f} – ${max(c.high for c in candles):,.0f}  "
          f"seed={BTC_SEED}\n")

    results = []
    best_mcdx, best_sentinel = None, None

    print(f"  [1/2] MCDX  ({len(SL_TP_GRID)} SL/TP × {len(MCDX_PARAMS)} param sets)...")
    for (sl, tp), params in product(SL_TP_GRID, MCDX_PARAMS):
        strat = MCDXStrategy("BTC/USDT", params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl, tp)
        s = calc(all_trades, sl, tp)
        label = f"dwcs_buy={params['dwcs_buy']} rvol≥{params['rvol_min']}"
        results.append(("MCDX", label, sl, tp, s))
        if s["n"] >= 10 and s["net_r"] > 0 and s["wr"] >= 50:
            if best_mcdx is None or s["net_r"] > best_mcdx[4]["net_r"]:
                best_mcdx = ("MCDX", label, sl, tp, s)

    print(f"  [2/2] Sentinel  ({len(SL_TP_GRID)} SL/TP × {len(SENTINEL_PARAMS)} param sets)...")
    for (sl, tp), params in product(SL_TP_GRID, SENTINEL_PARAMS):
        strat = SentinelStrategy("BTC/USDT", params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl, tp)
        s = calc(all_trades, sl, tp)
        label = f"conf={params['min_conf']} fresh={params['fresh_bos_bars']}"
        results.append(("Sentinel", label, sl, tp, s))
        if s["n"] >= 10 and s["net_r"] > 0 and s["wr"] >= 50:
            if best_sentinel is None or s["net_r"] > best_sentinel[4]["net_r"]:
                best_sentinel = ("Sentinel", label, sl, tp, s)

    print(f"\n{'─' * W}")
    print(f"  {'Strategy':<10}  {'Params':<30}  {'SL%':>4}  {'TP%':>4}  "
          f"{'N':>4}  {'WR%':>6}  {'PF':>5}  {'NetR':>6}  {'USD($10x)':>10}")
    print(f"  {'─' * 78}")

    for strat_name in ["MCDX", "Sentinel"]:
        subset = [r for r in results if r[0] == strat_name]
        subset.sort(key=lambda x: (-(1 if x[4]["net_r"] > 0 else 0), -x[4]["net_r"]))
        for sn, lbl, sl, tp, s in subset[:6]:
            star = " ★" if s["net_r"] > 0 and s["n"] >= 10 and s["wr"] >= 50 else "  "
            print(f"  {sn:<10}  {lbl:<30}  {sl*100:.1f}%  {tp*100:.1f}%  "
                  f"{s['n']:>4}  {s['wr']:>5.1f}%{star}  {s['pf']:>5.2f}  "
                  f"{s['net_r']:>+6.2f}  ${s['pnl_usd']:>+8.1f}")
        print()

    print(f"{'═' * W}")
    print("  RECOMMENDED (Net-R > 0 AND WR ≥ 50% AND Trades ≥ 10)")
    print(f"  {'─' * 78}")

    best_list = [b for b in [best_mcdx, best_sentinel] if b]
    if best_list:
        for sn, lbl, sl, tp, s in best_list:
            eff_rr = (tp - FEE_TOTAL) / (sl + FEE_TOTAL)
            be_wr  = (sl + FEE_TOTAL) / (tp + sl) * 100
            print(f"\n  ★ {sn}")
            print(f"    Params     : {lbl}")
            print(f"    SL / TP    : {sl*100:.1f}% / {tp*100:.1f}%  (gross RR 1:{tp/sl:.2f})")
            print(f"    Net RR     : 1:{eff_rr:.2f} after fees  |  Break-even WR: {be_wr:.1f}%")
            print(f"    Result     : {s['n']} trades  WR={s['wr']}%  PF={s['pf']}  "
                  f"Net-R={s['net_r']:+.2f}  ~${s['pnl_usd']:+.1f} ({LEVERAGE}x, ${RISK_USD}/trade)")
    else:
        print("\n  ไม่มี config ผ่านเกณฑ์ — closest by Net-R:")
        cands = sorted([r for r in results if r[4]["n"] >= 5], key=lambda x: -x[4]["net_r"])
        for sn, lbl, sl, tp, s in cands[:8]:
            print(f"    {sn:<10}  {lbl:<30}  SL={sl*100:.1f}%  TP={tp*100:.1f}%  "
                  f"N={s['n']}  WR={s['wr']}%  Net-R={s['net_r']:+.2f}")

    print(f"\n  Compare: 15m WR=38-46% (fail) vs 1H WR above (strategies designed for 1H)")
    print(f"  Note: 1H intraday (max 8h hold) = วันเดียวจบ เหมือนกัน แค่ signal ใช้ 1H candle")
    print("═" * W + "\n")


if __name__ == "__main__":
    asyncio.run(main())
