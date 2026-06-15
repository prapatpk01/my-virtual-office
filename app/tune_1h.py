"""
Parameter sweep for UT Bot, Momentum, MACD/EMA on 1H.
Target: WR ≥ 68% at 500 bars, fixed TP=$7 / SL=$5, $50 position.
"""
import asyncio, sys, math, random
import numpy as np
sys.path.insert(0, "")

from dataclasses import dataclass
from trading.strategies.ut_bot_strategy         import UTBotStrategy
from trading.strategies.momentum_score_strategy import MomentumScoreStrategy
from trading.strategies.macd_ema_strategy       import MACDEMAStrategy

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

def bt_utbot(params: dict, bars=500) -> dict:
    strat   = UTBotStrategy("BTC/USDT", params)
    subset  = CANDLES[-bars:]
    n       = len(subset)
    min_len = strat.ut_atr_len + 14 + 5
    try:
        buy_arr, sell_arr, _, atr_a = strat._build_signals(subset)
    except Exception:
        return {"wr": 0, "trades": 0, "total$": 0}
    sell_set = {i for i in range(n) if sell_arr[i]}
    return _simulate(subset, n, buy_arr, sell_set, atr_a, min_len)

def bt_momentum(params: dict, bars=500) -> dict:
    strat   = MomentumScoreStrategy("BTC/USDT", params)
    subset  = CANDLES[-bars:]
    n       = len(subset)
    min_len = strat.macd_slow + strat.macd_sig + strat.ema_len + 5
    try:
        buy_arr, sell_arr, _, _, atr_a = strat._build_signals(subset)
    except Exception:
        return {"wr": 0, "trades": 0, "total$": 0}
    sell_set = {i for i in range(n) if sell_arr[i]}
    return _simulate(subset, n, buy_arr, sell_set, atr_a, min_len)

def bt_macdema(params: dict, bars=500) -> dict:
    strat   = MACDEMAStrategy("BTC/USDT", params)
    subset  = CANDLES[-bars:]
    n       = len(subset)
    min_len = strat.macd_slow + strat.macd_sig + strat.hma_period + 5
    try:
        (ha_o, ha_c, ha_highs, ha_lows,
         hma, ema, sma, ml, sl_line, hist, atr_a) = strat._build_arrays(subset)
    except Exception:
        return {"wr": 0, "trades": 0, "total$": 0}
    buy_arr  = np.zeros(n, dtype=bool)
    sell_arr = np.zeros(n, dtype=bool)
    for i in range(2, n):
        d = strat._signal_at(i, ha_o, ha_c, ha_highs, ha_lows, hma, ema, sma, ml, sl_line)
        if d == 1:  buy_arr[i]  = True
        if d == -1: sell_arr[i] = True
    sell_set = {i for i in range(n) if sell_arr[i]}
    return _simulate(subset, n, buy_arr, sell_set, atr_a, min_len)

def _simulate(subset, n, buy_arr, sell_set, atr_a, min_len):
    closes = np.array([c.close for c in subset])
    highs  = np.array([c.high  for c in subset])
    lows   = np.array([c.low   for c in subset])
    results = []; prev = -999
    for i in range(min_len, n - 1):
        if not buy_arr[i] or i <= prev:
            continue
        entry = float(closes[i])
        amt   = POSITION / entry
        tp_p  = entry + TP_USD / amt
        sl_p  = entry - SL_USD / amt
        pnl   = None
        for j in range(i + 1, min(i + LOOKFWD, n)):
            if j in sell_set:
                pnl = amt * (float(closes[j]) - entry) - POSITION * FEE_RT
                break
            if lows[j]  <= sl_p: pnl =  -(SL_USD + POSITION * FEE_RT); break
            if highs[j] >= tp_p: pnl =    TP_USD - POSITION * FEE_RT;  break
        if pnl is None:
            pnl = amt * (float(closes[min(i + LOOKFWD, n) - 1]) - entry) - POSITION * FEE_RT
        results.append(pnl)
        prev = i
    if not results:
        return {"wr": 0, "trades": 0, "total$": 0}
    wins = sum(1 for p in results if p > 0)
    return {
        "wr":     round(wins / len(results) * 100, 1),
        "trades": len(results),
        "total$": round(sum(results), 2),
    }

# ── Sweeps ────────────────────────────────────────────────────────────────

def sweep_utbot():
    print("\n── UT Bot 1H sweep ──────────────────────────────────────────")
    print(f"  {'mult':>5} {'atr_len':>7} │ {'WR%':>6} {'Trades':>7} {'Total$':>8}")
    best = None
    for mult in [0.5, 0.8, 1.0, 1.1, 1.5, 2.0, 2.5, 3.0]:
        for atr_len in [10, 14, 21]:
            r = bt_utbot({"ut_mult": mult, "ut_atr_len": atr_len,
                          "sl_atr_mult": 1.0, "rr_ratio": 1.4})
            tag = " ◄ WR≥68%" if r["wr"] >= 68 and r["trades"] >= 5 else ""
            print(f"  {mult:>5.1f} {atr_len:>7} │ {r['wr']:>5.1f}% {r['trades']:>7} {r['total$']:>+8.2f}{tag}")
            if r["wr"] >= 68 and r["trades"] >= 5:
                if best is None or r["total$"] > best[1]["total$"]:
                    best = ({"ut_mult": mult, "ut_atr_len": atr_len}, r)
    return best

def sweep_momentum():
    print("\n── Momentum 1H sweep ────────────────────────────────────────")
    print(f"  {'rsi':>4} {'ema_len':>7} {'thr':>4} │ {'WR%':>6} {'Trades':>7} {'Total$':>8}")
    best = None
    for rsi_len in [9, 14, 21]:
        for ema_len in [20, 50, 100]:
            for thr in [3, 4, 5]:
                r = bt_momentum({"rsi_len": rsi_len, "ema_len": ema_len,
                                 "threshold": thr, "rr_ratio": 1.4})
                tag = " ◄ WR≥68%" if r["wr"] >= 68 and r["trades"] >= 5 else ""
                print(f"  {rsi_len:>4} {ema_len:>7} {thr:>4} │ {r['wr']:>5.1f}% {r['trades']:>7} {r['total$']:>+8.2f}{tag}")
                if r["wr"] >= 68 and r["trades"] >= 5:
                    if best is None or r["total$"] > best[1]["total$"]:
                        best = ({"rsi_len": rsi_len, "ema_len": ema_len, "threshold": thr}, r)
    return best

def sweep_macdema():
    print("\n── MACD/EMA 1H sweep ────────────────────────────────────────")
    print(f"  {'hma':>4} {'ef':>4} {'ss':>4} {'mf':>4} {'ms':>4} │ {'WR%':>6} {'Trades':>7} {'Total$':>8}")
    best = None
    for hma in [20, 30, 50]:
        for ef, ss in [(10, 20), (21, 50), (9, 21)]:
            for mf, ms in [(12, 26), (8, 21)]:
                r = bt_macdema({"hma_period": hma, "ema_fast": ef, "sma_slow": ss,
                                "macd_fast": mf, "macd_slow": ms, "rr_ratio": 1.4})
                tag = " ◄ WR≥68%" if r["wr"] >= 68 and r["trades"] >= 5 else ""
                print(f"  {hma:>4} {ef:>4} {ss:>4} {mf:>4} {ms:>4} │ {r['wr']:>5.1f}% {r['trades']:>7} {r['total$']:>+8.2f}{tag}")
                if r["wr"] >= 68 and r["trades"] >= 5:
                    if best is None or r["total$"] > best[1]["total$"]:
                        best = ({"hma_period": hma, "ema_fast": ef, "sma_slow": ss,
                                 "macd_fast": mf, "macd_slow": ms}, r)
    return best

# ── Main ──────────────────────────────────────────────────────────────────

print(f"\n{'='*62}")
print(f"  Param Sweep — 1H BTC, TP=${TP_USD} SL=${SL_USD}, ${POSITION}/trade, 500 bars")
print(f"{'='*62}")

best_ut  = sweep_utbot()
best_mom = sweep_momentum()
best_mac = sweep_macdema()

print(f"\n{'='*62}")
print("  BEST PARAMS (WR ≥ 68%, trades ≥ 5)")
print(f"{'='*62}")
for name, result in [("UT Bot", best_ut), ("Momentum", best_mom), ("MACD/EMA", best_mac)]:
    if result:
        p, r = result
        print(f"  {name}: {p}")
        print(f"    WR={r['wr']}%  trades={r['trades']}  total$={r['total$']:+.2f}")
    else:
        print(f"  {name}: no config reached WR≥68% with ≥5 trades")
