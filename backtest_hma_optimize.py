"""
Parameter optimizer for backtest_hma_filter.py
Grid search: HMA_PERIOD × TP_RR × SL_MULT × CROSS_CONDITION

Baseline: HMA=20  TP_RR=1.2  SL=2.5  CROSS=3bars
"""
import sys, os, time, math, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from trading.connectors.base import OHLCV
from trading.strategies.base import BaseStrategy
from trading.strategies.wt_adx_strategy import WTADXStrategy
from trading.strategies.ut_bot_strategy import UTBotStrategy

# ─── Fixed config ──────────────────────────────────────────────────────────────
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1h"
MONTHS      = 5
CAPITAL     = 260.0
TRADE_USD   = 100.0
FEE_RATE    = 0.001
EMA_PERIOD  = 5
SMA_PERIOD  = 9
MAX_BARS_IN = 120

# ─── Search grid ───────────────────────────────────────────────────────────────
HMA_PERIODS  = [20, 50, 100]
TP_RRS       = [1.2, 1.5, 2.0]
SL_MULTS     = [2.0, 2.5]
# CROSS_BARS: number of look-back bars for "EMA5 crossed above SMA9"
#   0 = loose: EMA5 simply > SMA9 at entry bar (no strict cross required)
#   3 = original: cross within last 3 bars
#   5 = wider:    cross within last 5 bars
CROSS_BARS   = [0, 3, 5]

BASELINE = (20, 1.2, 2.5, 3)   # (hma, tp_rr, sl_mult, cross_bars)

# ─── Fetch candles ─────────────────────────────────────────────────────────────

def fetch_candles() -> list:
    since_ms = int(time.time() * 1000) - MONTHS * 31 * 24 * 3600 * 1000
    sym_b    = SYMBOL.replace("/", "")
    import urllib.request, json as _json
    for source, url_fn in [
        ("Binance", lambda f: (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={sym_b}&interval=1h&startTime={f}&limit=1000")),
        ("OKX", lambda f: (
            f"https://www.okx.com/api/v5/market/history-candles"
            f"?instId={SYMBOL.replace('/','-')}&bar=1H&limit=300&after={f}")),
    ]:
        try:
            req = urllib.request.Request(
                url_fn(since_ms), headers={"User-Agent": "backtest/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read())
            raw = data if isinstance(data, list) else data.get("data", [])
            if raw and len(raw) > 50:
                raw.sort(key=lambda r: int(r[0]))
                candles = [OHLCV(int(r[0]), float(r[1]), float(r[2]),
                                 float(r[3]), float(r[4]), float(r[5]))
                           for r in raw]
                print(f"  {source}: {len(candles)} candles")
                return candles
        except Exception:
            pass

    print("  Synthetic BTC data (network restricted)")
    rng    = np.random.default_rng(42)
    n_bars = MONTHS * 31 * 24 + 50
    closes = [60_000.0]
    regimes = [
        (0.00, 0.15, +0.00010, 0.0050),
        (0.15, 0.30, -0.00008, 0.0070),
        (0.30, 0.50, +0.00005, 0.0040),
        (0.50, 0.65, +0.00015, 0.0055),
        (0.65, 0.80, -0.00012, 0.0080),
        (0.80, 1.00, +0.00010, 0.0045),
    ]
    for i in range(1, n_bars):
        frac = i / n_bars
        dr, vl = 0.0, 0.005
        for s, e, d, v in regimes:
            if s <= frac < e:
                dr, vl = d, v; break
        closes.append(closes[-1] * (1 + rng.normal(dr, vl)))
    candles = []
    ts0 = since_ms
    for i, c in enumerate(closes):
        o     = closes[i - 1] if i > 0 else c
        noise = abs(rng.normal(0, c * 0.0012))
        candles.append(OHLCV(ts0 + i * 3_600_000, round(o, 2),
                             round(max(o, c) + noise, 2),
                             round(min(o, c) - noise, 2),
                             round(c, 2), float(rng.uniform(200, 2000))))
    return candles


# ─── Parameterised backtest ────────────────────────────────────────────────────

def run_combo(candles, buy_sig, sell_sig, atr14, hma_a, ema5_a, sma9_a,
              sl_mult, tp_rr, cross_bars):
    n       = len(candles)
    closes  = np.array([c.close for c in candles])
    highs   = np.array([c.high  for c in candles])
    lows    = np.array([c.low   for c in candles])
    opens   = np.array([c.open  for c in candles])

    # Warmup: first valid HMA bar + buffer
    valid_hma = np.where(~np.isnan(hma_a))[0]
    warmup    = int(valid_hma[0]) + 10 if len(valid_hma) else 110

    trades: list = []
    capital   = CAPITAL
    in_pos    = False
    entry_bar = 0
    entry_px  = sl_px = tp_px = amount = 0.0
    pending_hma_exit = False

    for i in range(warmup, n - 1):
        c_close = float(closes[i])
        c_high  = float(highs[i])
        c_low   = float(lows[i])
        hma_v   = float(hma_a[i])

        if any(math.isnan(x) for x in [hma_v, float(ema5_a[i]), float(sma9_a[i])]):
            continue

        # ── In position ───────────────────────────────────────────────
        if in_pos:
            exit_px = exit_type = None

            if pending_hma_exit:
                nxt = float(opens[i])
                if nxt < float(hma_a[i]):
                    exit_px, exit_type = nxt, "hma_break"
                pending_hma_exit = False

            if exit_px is None:
                if c_high >= tp_px:
                    exit_px, exit_type = tp_px, "tp"
                elif c_low <= sl_px:
                    exit_px, exit_type = sl_px, "sl"

            if exit_px is None and bool(sell_sig[i]):
                exit_px, exit_type = c_close, "sell_signal"

            if exit_px is None and (i - entry_bar) >= MAX_BARS_IN:
                exit_px, exit_type = c_close, "timeout"

            if exit_px is None and c_close < hma_v:
                pending_hma_exit = True

            if exit_px is not None:
                pnl_net  = (exit_px - entry_px) * amount - exit_px * amount * FEE_RATE
                capital += pnl_net
                trades.append({"pnl_net": pnl_net, "type": exit_type})
                in_pos = False
                pending_hma_exit = False
            continue

        # ── Entry filters ─────────────────────────────────────────────
        if not bool(buy_sig[i]):
            continue
        if float(closes[i - 1]) <= float(hma_a[i - 1]):
            continue

        if cross_bars == 0:
            if float(ema5_a[i]) <= float(sma9_a[i]):
                continue
        else:
            crossed = False
            for k in range(max(1, i - cross_bars + 1), i + 1):
                if (float(ema5_a[k - 1]) <= float(sma9_a[k - 1]) and
                        float(ema5_a[k]) > float(sma9_a[k])):
                    crossed = True; break
            if not crossed:
                continue

        atr_v = float(atr14[i])
        if math.isnan(atr_v) or atr_v <= 0:
            continue

        entry_px = c_close
        sl_px    = round(entry_px - sl_mult * atr_v, 2)
        tp_px    = round(entry_px + sl_mult * tp_rr * atr_v, 2)
        amount   = TRADE_USD / entry_px
        capital -= entry_px * amount * FEE_RATE
        in_pos   = True
        entry_bar = i
        pending_hma_exit = False

    return trades


def summarise(trades: list) -> dict:
    if not trades:
        return {"total": 0, "wins": 0, "wr": 0.0, "net": 0.0, "pf": 0.0, "hma_pct": 0.0}
    wins   = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] <= 0]
    w_sum  = sum(t["pnl_net"] for t in wins)
    l_sum  = abs(sum(t["pnl_net"] for t in losses))
    pf     = (w_sum / l_sum) if l_sum > 0 else 9.99
    hma_n  = sum(1 for t in trades if t["type"] == "hma_break")
    return {
        "total":   len(trades),
        "wins":    len(wins),
        "wr":      len(wins) / len(trades) * 100,
        "net":     sum(t["pnl_net"] for t in trades),
        "pf":      min(pf, 9.99),
        "hma_pct": hma_n / len(trades) * 100,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    combos = list(itertools.product(HMA_PERIODS, TP_RRS, SL_MULTS, CROSS_BARS))
    total  = len(combos)

    print("\n" + "═" * 74)
    print(f"  HMA Filter Param Optimizer  |  {SYMBOL} {TIMEFRAME}  |  {MONTHS}-month")
    print(f"  Grid: HMA{HMA_PERIODS} × TP{TP_RRS} × SL{SL_MULTS} × CROSS{CROSS_BARS}")
    print(f"  {total} combos × 2 strategies = {total*2} runs")
    print("═" * 74)

    candles = fetch_candles()
    if len(candles) < 300:
        print("ERROR: not enough candles"); sys.exit(1)

    n      = len(candles)
    closes = [c.close for c in candles]
    print(f"\n  Pre-computing indicators ({n} bars)...")

    ema5_a = BaseStrategy.ema(closes, EMA_PERIOD)
    sma9_a = BaseStrategy.sma(closes, SMA_PERIOD)
    atr14  = BaseStrategy.atr(candles, 14)
    hma_cache = {p: BaseStrategy.hma(closes, p) for p in HMA_PERIODS}

    wt_strat = WTADXStrategy(SYMBOL)
    _, _, wt_buy, wt_sell, _ = wt_strat._build_signals(candles)

    ut_strat = UTBotStrategy(SYMBOL)
    ut_buy, ut_sell, _, _ = ut_strat._build_signals(candles)

    strategies = [
        ("WTADXStrategy", wt_buy.astype(bool), wt_sell.astype(bool)),
        ("UTBotStrategy",  ut_buy.astype(bool),  ut_sell.astype(bool)),
    ]

    cl = lambda cb: f"cross{cb}" if cb > 0 else "EMA>SMA"

    all_results = {}

    for strat_name, buy_sig, sell_sig in strategies:
        print(f"\n  Running {strat_name}...", flush=True)
        rows = []
        for hma_p, tp_rr, sl_mult, cross_b in combos:
            trades = run_combo(
                candles, buy_sig, sell_sig, atr14,
                hma_cache[hma_p], ema5_a, sma9_a,
                sl_mult, tp_rr, cross_b,
            )
            s = summarise(trades)
            rows.append(((hma_p, tp_rr, sl_mult, cross_b), s))

        rows.sort(key=lambda x: x[1]["net"], reverse=True)
        all_results[strat_name] = rows

    # ── Print tables ────────────────────────────────────────────────────
    for strat_name, rows in all_results.items():
        baseline_row = next((r for r in rows if r[0] == BASELINE), None)
        base_s = baseline_row[1] if baseline_row else {}
        base_rank = rows.index(baseline_row) + 1 if baseline_row else -1

        print(f"\n{'═'*74}")
        print(f"  {strat_name}  |  ranked by Net P/L  (top 15 / {total})")
        print(f"  Baseline rank: #{base_rank}  "
              f"Net=${base_s.get('net',0):+.2f}  WR={base_s.get('wr',0):.1f}%  "
              f"Trades={base_s.get('total',0)}")
        print(f"{'─'*74}")
        print(f"  {'HMA':>5} {'TP_RR':>6} {'SL':>5} {'CROSS':>8} │"
              f" {'Trades':>7} {'WR%':>6} {'Net$':>9} {'PF':>6} {'HMA%':>6}")
        print(f"  {'─'*5} {'─'*6} {'─'*5} {'─'*8} ┼"
              f" {'─'*7} {'─'*6} {'─'*9} {'─'*6} {'─'*6}")

        shown_base = False
        for rank, (combo, s) in enumerate(rows[:15], 1):
            marker = "  ◄ BASELINE" if combo == BASELINE else ""
            if combo == BASELINE:
                shown_base = True
            print(f"  {combo[0]:>5} {combo[1]:>6.1f} {combo[2]:>5.1f} {cl(combo[3]):>8} │"
                  f" {s['total']:>7} {s['wr']:>6.1f} {s['net']:>+9.2f}"
                  f" {s['pf']:>6.2f} {s['hma_pct']:>5.0f}%{marker}")

        if not shown_base and baseline_row:
            hma_p, tp_rr, sl_mult, cross_b = BASELINE
            s = base_s
            print(f"  {'─'*74}")
            print(f"  {hma_p:>5} {tp_rr:>6.1f} {sl_mult:>5.1f} {cl(cross_b):>8} │"
                  f" {s['total']:>7} {s['wr']:>6.1f} {s['net']:>+9.2f}"
                  f" {s['pf']:>6.2f} {s['hma_pct']:>5.0f}%"
                  f"  ◄ BASELINE (rank #{base_rank})")

    # ── Winner / Recommendation ────────────────────────────────────────
    print(f"\n{'═'*74}")
    print("  FINAL RECOMMENDATION")
    print(f"{'─'*74}")

    update_needed = False
    best_combos = {}
    for strat_name, rows in all_results.items():
        best_combo, best_s = rows[0]
        baseline_row = next((r for r in rows if r[0] == BASELINE), None)
        base_s = baseline_row[1] if baseline_row else {"net": 0}

        improved = best_s["net"] > base_s["net"] + 0.50   # needs >$0.50 improvement to matter
        best_combos[strat_name] = (best_combo, best_s, base_s, improved)
        if improved:
            update_needed = True

        hma_p, tp_rr, sl_mult, cross_b = best_combo
        verdict = (f"IMPROVED ${best_s['net']:+.2f} vs baseline ${base_s['net']:+.2f}"
                   if improved else
                   f"no meaningful gain — baseline ${base_s['net']:+.2f} is fine")
        print(f"\n  {strat_name}:")
        print(f"    Best: HMA={hma_p} TP_RR={tp_rr} SL={sl_mult} CROSS={cl(cross_b)}")
        print(f"    Trades={best_s['total']}  WR={best_s['wr']:.1f}%  Net=${best_s['net']:+.2f}  PF={best_s['pf']:.2f}")
        print(f"    → {verdict}")

    print(f"\n{'─'*74}")
    if update_needed:
        print("  ✓ Improvements found. Suggested changes to backtest_hma_filter.py:")
        for strat_name, (combo, best_s, base_s, improved) in best_combos.items():
            if improved:
                hma_p, tp_rr, sl_mult, cross_b = combo
                print(f"\n  {strat_name}:")
                print(f"    HMA_PERIOD = {hma_p}   # was {BASELINE[0]}")
                print(f"    TP_RR      = {tp_rr}   # was {BASELINE[1]}")
                print(f"    SL_MULT    = {sl_mult}  # was {BASELINE[2]}")
                print(f"    CROSS_BARS = {cross_b}  # was {BASELINE[3]}  "
                      f"({'EMA>SMA' if cross_b==0 else f'{cross_b}-bar window'})")
    else:
        print("  No combo meaningfully beats the baseline.")
        print("  Keep original: HMA_PERIOD=20  SL_MULT=2.5  TP_RR=1.2  CROSS_BARS=3")
    print("═" * 74 + "\n")


if __name__ == "__main__":
    main()
