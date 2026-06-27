"""
BTC/USDT 15m Parameter Optimizer — OKX Cross Margin x10
Fee model: 0.10% open + 0.10% close + 0.05% daily borrow = 0.25% per trade (on notional)
At 10x leverage: break-even price move = 0.25% per trade

Targets: Net-R > 0  AND  WR ≥ 50% (fee-adjusted)  AND  Trades ≥ 20/month
Intraday: max hold = 32 bars (8h), no overnight positions

Usage:
    python backtest_btc_15m.py
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

# ── Config ────────────────────────────────────────────────────────────────────
WARMUP     = 300                  # bars for indicator warmup
STAGE_BARS = 672                  # 1 week of 15m bars (7d × 24h × 4)
N_STAGES   = 4                    # 4 weeks ≈ 1 month
CTX_WINDOW = 300                  # rolling context
FETCH_BARS = WARMUP + STAGE_BARS * N_STAGES
LOOKFWD    = 32                   # max 8h hold (intraday constraint)

# Fee model
FEE_OPEN   = 0.0010               # 0.10% taker
FEE_CLOSE  = 0.0010               # 0.10% taker
FEE_BORROW = 0.0005               # 0.05% daily borrow (avg ~4h hold)
FEE_TOTAL  = FEE_OPEN + FEE_CLOSE + FEE_BORROW  # 0.25% of notional

LEVERAGE   = 10
RISK_USD   = 100.0                # capital per trade (for $ P&L display)

# BTC synthetic params
BTC_START  = 65_000.0
BTC_SEED   = 42
BAR_MS     = 15 * 60 * 1000      # 15-minute bars


def generate_btc_15m(n: int, seed: int = BTC_SEED) -> list:
    """GBM 15m BTC: 6 regime phases × 1 week each, calibrated ~4% daily vol."""
    rng = np.random.default_rng(seed)
    closes = [BTC_START]
    regime_len = STAGE_BARS  # 1 week per phase (4h × 168 → same calendar time as 1h version)
    phases = [
        (+0.00008, 0.0030),   # bull (≈+1.9%/day drift, 0.3% vol/bar)
        (-0.00005, 0.0030),   # bear
        (+0.00002, 0.0020),   # sideways
        (+0.00012, 0.0040),   # strong bull
        (-0.00010, 0.0040),   # strong bear
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
        spread = abs(rng.normal(0, c * 0.0008))
        candles.append(OHLCV(
            timestamp=base_ts + i * BAR_MS,
            open=round(o, 2), high=round(max(o, c) + spread, 2),
            low=round(min(o, c) - spread, 2), close=round(c, 2),
            volume=float(rng.uniform(50, 500)),
        ))
    return candles


def find_exit(candles, entry_idx, direction, sl_p, tp_p):
    """Returns (exit_bar, outcome): +1=TP, -1=SL, 0=timeout."""
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
    """Run one stage; apply fee-adjusted P&L."""
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

        # Fee-adjusted net pnl (in % of notional)
        if oc == 1:
            net_pct = tp_pct - FEE_TOTAL       # TP hit, minus fees
        elif oc == -1:
            net_pct = -(sl_pct + FEE_TOTAL)    # SL hit, plus fees
        else:
            net_pct = -FEE_TOTAL               # timeout: exit flat, pay fees

        # In R-multiples where 1R = sl_pct (price)
        net_r = net_pct / sl_pct

        trades.append({
            "outcome": oc,
            "net_pct": net_pct,
            "net_r":   net_r,
        })
        lock_until = eb
    return trades


def calc(trades, sl_pct, tp_pct):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net_r": 0.0,
                "pnl_usd": 0.0, "net_tp_pct": 0.0}
    wins = [t for t in trades if t["outcome"] ==  1]
    loss = [t for t in trades if t["outcome"] == -1]
    timo = [t for t in trades if t["outcome"] ==  0]
    dec  = len(wins) + len(loss)
    wr   = len(wins) / max(dec, 1) * 100
    net_r = sum(t["net_r"] for t in trades)
    # P&L in USD at 10x leverage (RISK_USD = capital per trade)
    pnl_usd = sum(t["net_pct"] * LEVERAGE * RISK_USD for t in trades)
    net_tp  = tp_pct - FEE_TOTAL             # what a win actually earns on price
    net_sl  = sl_pct + FEE_TOTAL             # what a loss actually costs on price
    pf      = (len(wins) * net_tp) / max(len(loss) * net_sl, 1e-9)
    return {
        "n":       len(trades),
        "wins":    len(wins), "losses": len(loss), "timo": len(timo),
        "wr":      round(wr, 1),
        "pf":      round(pf, 2),
        "net_r":   round(net_r, 2),
        "pnl_usd": round(pnl_usd, 1),
        "net_tp_pct": round(net_tp * 100, 3),
    }


# ── Grids ─────────────────────────────────────────────────────────────────────
SL_TP_GRID = [
    (0.005, 0.008),   # SL 0.5%, TP 0.8%  — tight scalp
    (0.006, 0.010),   # SL 0.6%, TP 1.0%  — 1.67:1 ratio
    (0.008, 0.012),   # SL 0.8%, TP 1.2%  — balanced
    (0.008, 0.016),   # SL 0.8%, TP 1.6%  — wider TP
    (0.010, 0.015),   # SL 1.0%, TP 1.5%  — 1.5:1 ratio
    (0.010, 0.020),   # SL 1.0%, TP 2.0%  — 2:1 ratio
    (0.012, 0.020),   # SL 1.2%, TP 2.0%  — wider SL, big TP
]

MCDX_PARAMS = [
    {"dwcs_buy": 52, "dwcs_sell": 48, "rvol_min": 1.0},
    {"dwcs_buy": 55, "dwcs_sell": 45, "rvol_min": 1.1},
    {"dwcs_buy": 57, "dwcs_sell": 43, "rvol_min": 1.2},
]

SENTINEL_PARAMS = [
    {"min_conf": 50, "fresh_bos_bars": 30, "dwcs_bull_min": 45},
    {"min_conf": 55, "fresh_bos_bars": 25, "dwcs_bull_min": 50},
    {"min_conf": 58, "fresh_bos_bars": 20, "dwcs_bull_min": 52},
]


async def main():
    W = 80
    print("\n" + "═" * W)
    print("  BTC/USDT 15m OPTIMIZER  |  OKX Cross Margin x10  |  4 × 1-week stages")
    print(f"  Fee: {FEE_OPEN*100:.2f}% open + {FEE_CLOSE*100:.2f}% close + {FEE_BORROW*100:.2f}% borrow = "
          f"{FEE_TOTAL*100:.2f}%/trade on notional")
    print(f"  Intraday: max {LOOKFWD} bars ({LOOKFWD//4}h) hold  |  "
          f"Break-even price move: {FEE_TOTAL*100:.2f}%")
    print("═" * W)

    candles = generate_btc_15m(FETCH_BARS)
    stages  = [(WARMUP + s * STAGE_BARS,
                min(WARMUP + (s + 1) * STAGE_BARS, len(candles)))
               for s in range(N_STAGES)]

    price_min = min(c.low  for c in candles)
    price_max = max(c.high for c in candles)
    print(f"  BTC 15m: {len(candles)} bars  "
          f"${price_min:,.0f} – ${price_max:,.0f}  seed={BTC_SEED}\n")

    results = []

    # ── MCDX sweep ────────────────────────────────────────────────────────────
    print(f"  [1/2] MCDX  ({len(SL_TP_GRID)} SL/TP × {len(MCDX_PARAMS)} param sets)...")
    best_mcdx = None
    for (sl, tp), params in product(SL_TP_GRID, MCDX_PARAMS):
        strat = MCDXStrategy("BTC/USDT", params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl, tp)
        s = calc(all_trades, sl, tp)
        label = f"dwcs_buy={params['dwcs_buy']} rvol≥{params['rvol_min']}"
        results.append(("MCDX", label, sl, tp, s))
        if s["n"] >= 20 and s["net_r"] > 0 and s["wr"] >= 50:
            if best_mcdx is None or s["net_r"] > best_mcdx[4]["net_r"]:
                best_mcdx = ("MCDX", label, sl, tp, s)

    # ── Sentinel sweep ────────────────────────────────────────────────────────
    print(f"  [2/2] Sentinel  ({len(SL_TP_GRID)} SL/TP × {len(SENTINEL_PARAMS)} param sets)...")
    best_sentinel = None
    for (sl, tp), params in product(SL_TP_GRID, SENTINEL_PARAMS):
        strat = SentinelStrategy("BTC/USDT", params)
        all_trades = []
        for start, end in stages:
            all_trades += await run_stage(strat, candles, start, end, sl, tp)
        s = calc(all_trades, sl, tp)
        label = f"conf={params['min_conf']} fresh={params['fresh_bos_bars']}"
        results.append(("Sentinel", label, sl, tp, s))
        if s["n"] >= 20 and s["net_r"] > 0 and s["wr"] >= 50:
            if best_sentinel is None or s["net_r"] > best_sentinel[4]["net_r"]:
                best_sentinel = ("Sentinel", label, sl, tp, s)

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Strategy':<10}  {'Params':<32}  {'SL%':>4}  {'TP%':>4}  "
          f"{'N':>4}  {'WR%':>6}  {'PF':>5}  {'NetR':>6}  {'USD($10x)':>10}")
    print(f"  {'─' * 76}")

    for strat_name in ["MCDX", "Sentinel"]:
        subset = [r for r in results if r[0] == strat_name]
        subset.sort(key=lambda x: (
            -(1 if x[4]["n"] >= 20 and x[4]["net_r"] > 0 else 0),
            -x[4]["net_r"]
        ))
        for sn, lbl, sl, tp, s in subset[:5]:
            star = " ★" if s["n"] >= 20 and s["net_r"] > 0 and s["wr"] >= 50 else "  "
            print(f"  {sn:<10}  {lbl:<32}  {sl*100:.1f}%  {tp*100:.1f}%  "
                  f"{s['n']:>4}  {s['wr']:>5.1f}%{star}  {s['pf']:>5.2f}  "
                  f"{s['net_r']:>+6.2f}  ${s['pnl_usd']:>+8.1f}")
        print()

    # ── Best configs ──────────────────────────────────────────────────────────
    print(f"{'═' * W}")
    print("  RECOMMENDED CONFIG (Net-R > 0 AND WR ≥ 50% AND Trades ≥ 20/month)")
    print(f"  {'─' * 76}")

    best_list = [b for b in [best_mcdx, best_sentinel] if b]
    if best_list:
        for sn, lbl, sl, tp, s in best_list:
            eff_rr = (tp - FEE_TOTAL) / (sl + FEE_TOTAL)
            be_wr  = (sl + FEE_TOTAL) / (tp + sl) * 100
            print(f"\n  ★ {sn}")
            print(f"    Params     : {lbl}")
            print(f"    SL / TP    : {sl*100:.1f}% / {tp*100:.1f}%  (gross RR 1:{tp/sl:.2f})")
            print(f"    Net RR     : 1:{eff_rr:.2f} after fees  "
                  f"| Break-even WR: {be_wr:.1f}%")
            print(f"    Result     : {s['n']} trades  WR={s['wr']}%  PF={s['pf']}  "
                  f"Net-R={s['net_r']:+.2f}  ~${s['pnl_usd']:+.1f} ({LEVERAGE}x, ${RISK_USD}/trade)")
    else:
        print("\n  ไม่มี config ผ่านเกณฑ์ — closest by Net-R:")
        cands = [r for r in results if r[4]["n"] >= 10]
        cands.sort(key=lambda x: -x[4]["net_r"])
        for sn, lbl, sl, tp, s in cands[:8]:
            print(f"    {sn:<10}  {lbl:<32}  SL={sl*100:.1f}%  TP={tp*100:.1f}%  "
                  f"N={s['n']}  WR={s['wr']}%  Net-R={s['net_r']:+.2f}")

    print(f"\n  หมายเหตุ:")
    print(f"   • Fee {FEE_TOTAL*100:.2f}% = win net TP-fee, loss = SL+fee (worst case, avg hold ~4h)")
    print(f"   • At 10x leverage: 1% price move = 10% capital; SL กว้างไป = risk สูงมาก")
    print(f"   • Intraday: LOOKFWD={LOOKFWD} bars = max {LOOKFWD//4}h hold")
    print(f"   • Synthetic BTC 15m ≈ WORST CASE; live WR คาดว่าสูงกว่าเมื่อ MTF + BOS จริง")
    print("═" * W + "\n")


if __name__ == "__main__":
    asyncio.run(main())
