"""
Parameter sweep for MCDXStrategy (Adaptive Trading Bot) on 1H.
Target: WR ≥ 68% at 500 bars, fixed TP=$7 / SL=$5, $50 position.
"""
import asyncio, sys, math, random
import numpy as np
sys.path.insert(0, "")

from dataclasses import dataclass
from trading.strategies.mcdx_strategy import MCDXStrategy
from trading.strategies.base import SignalType

# ── Synthetic 1H BTC data ──────────────────────────────────────────────────
@dataclass
class C:
    timestamp: int
    open: float; high: float; low: float; close: float; volume: float

def gbm_candles(n=1200, seed=99, start=100_000., mu=0.0002, sigma=0.005):
    rng = random.Random(seed)
    candles = []; price = start
    for i in range(n):
        ret   = mu + sigma * rng.gauss(0, 1)
        o = price; c = price * math.exp(ret)
        h = max(o, c) * (1 + abs(rng.gauss(0, sigma * 0.4)))
        l = min(o, c) * (1 - abs(rng.gauss(0, sigma * 0.4)))
        candles.append(C(i * 3600, o, h, l, c, rng.uniform(1, 10)))
        price = c
    return candles

CANDLES = gbm_candles()

# ── Backtest engine (fixed $7 TP / $5 SL) ────────────────────────────────
POSITION = 50.0
FEE_RT   = 0.0020
TP_USD   = 7.0
SL_USD   = 5.0
LOOKFWD  = 120   # 5 days

async def bt_mcdx(params: dict, bars=500) -> dict:
    strat   = MCDXStrategy("BTC/USDT", params)
    subset  = CANDLES[-bars:]
    n       = len(subset)
    warmup  = min(120, n // 4)
    closes  = np.array([c.close for c in subset], dtype=float)
    highs   = np.array([c.high  for c in subset], dtype=float)
    lows    = np.array([c.low   for c in subset], dtype=float)
    results = []
    locked  = -1
    for i in range(warmup, n - 1):
        if i <= locked:
            continue
        window = subset[:i + 1]
        try:
            sig = await strat.analyze(window, window[-1].close)
        except Exception:
            continue
        if sig.type == SignalType.BUY:
            entry = float(closes[i])
            amt   = POSITION / entry
            tp_p  = entry + TP_USD / amt
            sl_p  = entry - SL_USD / amt
            pnl   = None
            for j in range(i + 1, min(i + LOOKFWD, n)):
                if lows[j]  <= sl_p: pnl = -(SL_USD + POSITION * FEE_RT); break
                if highs[j] >= tp_p: pnl =   TP_USD - POSITION * FEE_RT;  break
            if pnl is None:
                pnl = amt * (float(closes[min(i + LOOKFWD, n) - 1]) - entry) - POSITION * FEE_RT
            results.append(pnl)
            locked = i + LOOKFWD
    if not results:
        return {"wr": 0, "trades": 0, "total$": 0}
    wins = sum(1 for p in results if p > 0)
    return {
        "wr":     round(wins / len(results) * 100, 1),
        "trades": len(results),
        "total$": round(sum(results), 2),
    }


# ── Parameter sweep ────────────────────────────────────────────────────────

async def sweep_mcdx():
    print("\n── MCDX (Adaptive) 1H sweep ─────────────────────────────────")
    print(f"  {'length':>6} {'sma_pc':>6} {'sma_lc':>6} {'dw_buy':>6} {'dw_sel':>6} │ {'WR%':>6} {'Trades':>7} {'Total$':>8}")
    best = None
    for length in [50, 100, 150]:
        for sma_len in [5, 10, 20]:
            for dwcs_buy, dwcs_sell in [(53, 47), (55, 45), (60, 40)]:
                params = {
                    "length": length,
                    "sma_pc_len": sma_len,
                    "sma_lc_len": sma_len,
                    "dwcs_buy": dwcs_buy,
                    "dwcs_sell": dwcs_sell,
                }
                r = await bt_mcdx(params)
                tag = " ◄ WR≥68%" if r["wr"] >= 68 and r["trades"] >= 5 else ""
                print(f"  {length:>6} {sma_len:>6} {sma_len:>6} {dwcs_buy:>6} {dwcs_sell:>6} │ "
                      f"{r['wr']:>5.1f}% {r['trades']:>7} {r['total$']:>+8.2f}{tag}")
                if r["wr"] >= 68 and r["trades"] >= 5:
                    if best is None or r["total$"] > best[1]["total$"]:
                        best = (params, r)
    return best


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    print(f"\n{'='*62}")
    print(f"  Param Sweep — MCDXStrategy, 1H BTC, TP=${TP_USD} SL=${SL_USD}, ${POSITION}/trade, 500 bars")
    print(f"{'='*62}")

    best_mcdx = await sweep_mcdx()

    print(f"\n{'='*62}")
    print("  BEST PARAMS (WR ≥ 68%, trades ≥ 5)")
    print(f"{'='*62}")
    if best_mcdx:
        p, r = best_mcdx
        print(f"  MCDX: {p}")
        print(f"    WR={r['wr']}%  trades={r['trades']}  total$={r['total$']:+.2f}")
    else:
        print("  MCDX: no config reached WR≥68% with ≥5 trades")

asyncio.run(main())
