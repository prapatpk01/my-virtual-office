"""
Full backtest — fixed $50 position size, OKX fees 0.20% round trip.
BTC/USDT synthetic data (GBM).  Bars: 250 / 500 / 1000.
Usage: python backtest_50usd.py [15m|1h]
"""
import asyncio, sys, math, random
import numpy as np
sys.path.insert(0, "")

TF = sys.argv[1] if len(sys.argv) > 1 else "15m"
assert TF in ("15m", "1h"), "TF must be 15m or 1h"

# 15m: sigma=0.002, bar_sec=900, bars/day=96
# 1h:  sigma=0.005, bar_sec=3600, bars/day=24
TF_CFG = {
    "15m": dict(sigma=0.002, bar_sec=900,  bars_per_day=96,  lookfwd=60),
    "1h":  dict(sigma=0.005, bar_sec=3600, bars_per_day=24,  lookfwd=30),
}
cfg = TF_CFG[TF]

from dataclasses import dataclass
from trading.strategies.ut_bot_strategy      import UTBotStrategy
from trading.strategies.wt_adx_strategy      import WTADXStrategy
from trading.strategies.momentum_score_strategy import MomentumScoreStrategy
from trading.strategies.macd_ema_strategy    import MACDEMAStrategy

# ── synthetic BTC OHLCV (GBM) ──────────────────────────────────────────────

@dataclass
class C:
    timestamp: int
    open: float; high: float; low: float; close: float; volume: float

def gbm_candles(n: int, seed: int = 42, start: float = 100_000.0,
                mu: float = 0.0001, sigma: float = None) -> list:
    if sigma is None:
        sigma = cfg["sigma"]
    rng = random.Random(seed)
    candles = []
    price = start
    for i in range(n):
        ret   = mu + sigma * (sum(rng.gauss(0,1) for _ in range(1)))
        open_ = price
        close = price * math.exp(ret)
        high  = max(open_, close) * (1 + abs(rng.gauss(0, sigma * 0.5)))
        low   = min(open_, close) * (1 - abs(rng.gauss(0, sigma * 0.5)))
        candles.append(C(i * 900, open_, high, low, close, rng.uniform(1, 10)))
        price = close
    return candles

# ── backtest engine ────────────────────────────────────────────────────────

FEE_RT    = 0.0020   # 0.10% entry + 0.10% exit
POSITION  = 50.0     # USD per trade
LOOKFWD   = cfg["lookfwd"]

async def run_strategy_backtest(strategy, candles: list, bars: int) -> dict:
    subset = candles[-bars:]
    n = len(subset)

    buy_sigs, sell_sigs = [], []

    # Call _build_signals if available (UT Bot, Momentum, MACD/EMA)
    # WaveTrend uses its own internal method
    name = type(strategy).__name__

    if name == "UTBotStrategy":
        buy_arr, sell_arr, _, atr_a = strategy._build_signals(subset)
        min_len = strategy.ut_atr_len + 14 + 5
        for i in range(min_len, n - 1):
            if np.isnan(atr_a[i]) or atr_a[i] <= 0:
                continue
            if buy_arr[i]:
                buy_sigs.append((i, float(atr_a[i]),
                                 strategy.sl_atr_mult, strategy.rr_ratio))
            elif sell_arr[i]:
                sell_sigs.append((i, "sell"))

    elif name == "MomentumScoreStrategy":
        buy_arr, sell_arr, _, _, atr_a = strategy._build_signals(subset)
        min_len = strategy.macd_slow + strategy.macd_sig + strategy.ema_len + 5
        for i in range(min_len, n - 1):
            if np.isnan(atr_a[i]) or atr_a[i] <= 0:
                continue
            if buy_arr[i]:
                buy_sigs.append((i, float(atr_a[i]),
                                 strategy.sl_atr_mult, strategy.rr_ratio))
            elif sell_arr[i]:
                sell_sigs.append((i, "sell"))

    elif name == "MACDEMAStrategy":
        (ha_o, ha_c, ha_highs, ha_lows,
         hma, ema, sma, ml, sl_line, hist, atr_a) = strategy._build_arrays(subset)
        min_len = strategy.macd_slow + strategy.macd_sig + strategy.hma_period + 5
        for i in range(min_len, n - 1):
            if np.isnan(atr_a[i]) or atr_a[i] <= 0:
                continue
            d = strategy._signal_at(i, ha_o, ha_c, ha_highs, ha_lows,
                                    hma, ema, sma, ml, sl_line)
            if d == 1:
                buy_sigs.append((i, float(atr_a[i]),
                                 strategy.sl_atr_mult, strategy.rr_ratio))
            elif d == -1:
                sell_sigs.append((i, "sell"))

    elif name == "WTADXStrategy":
        # WaveTrend uses wt1/wt2; get signals via analyze on rolling window
        # Simplified: use internal _wt_signals helper
        # Build arrays manually
        closes = np.array([c.close for c in subset], dtype=float)
        highs  = np.array([c.high  for c in subset], dtype=float)
        lows   = np.array([c.low   for c in subset], dtype=float)
        atr_a  = np.array(strategy.atr(subset, 14), dtype=float)

        # Compute WT1
        def ema_arr(arr, p):
            k = 2/(p+1); r = np.full(len(arr), np.nan)
            first = np.where(~np.isnan(arr))[0]
            if len(first) == 0: return r
            s = first[0] + p - 1
            if s >= len(arr): return r
            r[s] = float(np.nanmean(arr[first[0]:first[0]+p]))
            for i in range(s+1, len(arr)):
                r[i] = arr[i]*k + r[i-1]*(1-k)
            return r

        n_s = 9; avg_mult = 0.015
        hlc3 = (highs + lows + closes) / 3
        esa  = ema_arr(hlc3, n_s)
        diff = np.abs(hlc3 - esa)
        de   = ema_arr(diff, n_s)
        ci   = np.where(de > 0, (hlc3 - esa) / (0.015 * de), 0.0)
        wt1  = ema_arr(ci, 3)
        wt2  = np.array([np.nanmean(wt1[max(0,i-3):i+1]) if not np.isnan(wt1[i]) else np.nan
                         for i in range(len(wt1))])

        ob = strategy.params.get("ob_level", 53)
        os_ = strategy.params.get("os_level", -53)
        min_len = 30
        for i in range(min_len, n - 1):
            if np.isnan(wt1[i]) or np.isnan(atr_a[i]) or atr_a[i] <= 0:
                continue
            # Buy: wt1 crosses above wt2 from oversold
            if (wt1[i-1] <= wt2[i-1] and wt1[i] > wt2[i] and wt2[i] < os_):
                buy_sigs.append((i, float(atr_a[i]), 1.5, 1.2))
            elif (wt1[i-1] >= wt2[i-1] and wt1[i] < wt2[i] and wt2[i] > ob):
                sell_sigs.append((i, "sell"))

    # ── Simulate trades (BUY-only, exit on SELL signal or SL/TP hit) ─────────
    closes_all = np.array([c.close for c in subset], dtype=float)
    highs_all  = np.array([c.high  for c in subset], dtype=float)
    lows_all   = np.array([c.low   for c in subset], dtype=float)

    # Build sell signal set for quick lookup
    sell_set = {i for i, _ in sell_sigs}

    results = []
    prev_buy_i = -999

    for (bi, atr_val, sl_m, rr) in buy_sigs:
        if bi <= prev_buy_i:
            continue  # same bar (shouldn't happen but guard)

        entry = float(closes_all[bi])
        sl_p  = entry - sl_m * atr_val
        tp_p  = entry + sl_m * rr * atr_val
        sl_dist = sl_m * atr_val
        tp_dist = sl_m * rr * atr_val

        # $ amounts for $50 position
        amount_btc = POSITION / entry
        win_usd    = amount_btc * tp_dist - POSITION * FEE_RT
        loss_usd   = -(amount_btc * sl_dist + POSITION * FEE_RT)

        outcome = 0
        exit_reason = "timeout"
        for j in range(bi + 1, min(bi + LOOKFWD, n)):
            # Check sell signal first (exit before TP/SL)
            if j in sell_set:
                exit_price = float(closes_all[j])
                pnl_usd = amount_btc * (exit_price - entry) - POSITION * FEE_RT
                results.append(("sell_sig", pnl_usd))
                prev_buy_i = bi
                exit_reason = "sell_sig"
                break
            if lows_all[j] <= sl_p:
                results.append(("loss", loss_usd))
                prev_buy_i = bi
                exit_reason = "sl"
                break
            if highs_all[j] >= tp_p:
                results.append(("win", win_usd))
                prev_buy_i = bi
                exit_reason = "tp"
                break
        else:
            # Timeout — close at last bar close
            exit_price = float(closes_all[min(bi + LOOKFWD, n) - 1])
            pnl_usd = amount_btc * (exit_price - entry) - POSITION * FEE_RT
            results.append(("timeout", pnl_usd))
            prev_buy_i = bi

    # ── Stats ──────────────────────────────────────────────────────────────────
    wins   = [p for r,p in results if p > 0]
    losses = [p for r,p in results if p <= 0]
    total  = len(results)
    wr     = len(wins)/total*100 if total else 0
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    pf   = gross_win/gross_loss if gross_loss > 0 else float("inf")
    total_usd  = sum(p for _,p in results)
    avg_usd    = total_usd/total if total else 0
    bars_per_day = cfg["bars_per_day"]
    trades_per_day = total / (bars / bars_per_day)

    return {
        "trades": total,
        "wins":   len(wins),
        "losses": len(losses),
        "wr":     round(wr, 1),
        "pf":     round(pf, 2),
        "total$": round(total_usd, 2),
        "avg$":   round(avg_usd, 3),
        "t/day":  round(trades_per_day, 1),
    }


async def main():
    BAR_SETS = [250, 500, 1000]
    candles_max = gbm_candles(1100, seed=99)

    strategies = [
        ("UT Bot",    UTBotStrategy("BTC/USDT")),
        ("Momentum",  MomentumScoreStrategy("BTC/USDT")),
        ("MACD/EMA",  MACDEMAStrategy("BTC/USDT")),
        ("WaveTrend", WTADXStrategy("BTC/USDT")),
    ]

    print(f"\n{'='*80}")
    print(f"  Full Backtest — $50/trade fixed, OKX fee 0.20% RT, BTC/USDT {TF} (synthetic)")
    print(f"{'='*80}")

    all_rows = []

    for sname, strat in strategies:
        print(f"\n── {sname} ──────────────────────────────────────────────")
        print(f"  {'Bars':>6} │ {'Trades':>6} │ {'WR%':>6} │ {'PF':>6} │ {'Total $':>9} │ {'Avg$/T':>8} │ {'T/day':>6}")
        print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*6}")
        for bars in BAR_SETS:
            r = await run_strategy_backtest(strat, candles_max, bars)
            sign = "+" if r["total$"] >= 0 else ""
            print(f"  {bars:>6} │ {r['trades']:>6} │ {r['wr']:>5.1f}% │ {r['pf']:>6.2f} │ "
                  f"{sign}{r['total$']:>8.2f} │ {r['avg$']:>+8.3f} │ {r['t/day']:>6.1f}")
            all_rows.append({"strategy": sname, "bars": bars, **r})

    # ── summary table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY — avg across 250/500/1000 bars")
    print(f"  {'Strategy':12} │ {'Avg WR%':>8} │ {'Avg PF':>7} │ {'Total$ 1k':>10} │ {'Avg$/T':>8}")
    print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*8}")
    for sname, _ in strategies:
        rows = [r for r in all_rows if r["strategy"] == sname]
        avg_wr  = sum(r["wr"]     for r in rows) / len(rows)
        avg_pf  = sum(r["pf"]     for r in rows) / len(rows)
        tot_1k  = next((r["total$"] for r in rows if r["bars"]==1000), 0)
        avg_per = sum(r["avg$"]   for r in rows) / len(rows)
        print(f"  {sname:12} │ {avg_wr:>7.1f}% │ {avg_pf:>7.2f} │ "
              f"{tot_1k:>+10.2f} │ {avg_per:>+8.3f}")

    print(f"\n  Note: fee = $0.10/trade ($50 × 0.20%). ATR-based SL/TP.")
    print(f"  SL/TP size dictates actual $/trade — see avg$ column.")

asyncio.run(main())
