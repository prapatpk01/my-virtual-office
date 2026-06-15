"""
Backtest: 15m vs 1H — 250 bars × 5 windows (1250 bars total)
Multi-position engine:
  - $500 starting capital, RISK_PER_TRADE=20% of current balance
  - Max 4 open positions total
  - WaveTrend: up to 2 concurrent positions
  - MACD/EMA, Momentum, UT Bot: 1 position each at a time
  - TP/SL: ATR-based from each strategy's built-in params
  - OKX fee 0.20% RT per trade
  - Lookforward: 60 bars (15m) / 120 bars (1H)
"""
import asyncio, sys, math, random
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.wt_adx_strategy        import WTADXStrategy
from trading.strategies.macd_ema_strategy       import MACDEMAStrategy
from trading.strategies.momentum_score_strategy import MomentumScoreStrategy
from trading.strategies.ut_bot_strategy         import UTBotStrategy

# ── constants ─────────────────────────────────────────────────────────────────
RISK_PCT   = 0.20    # 20% of balance per trade
FEE_RT     = 0.0020  # 0.10% entry + 0.10% exit
START_BAL  = 500.0
MAX_POS    = 4
WT_MAX_POS = 2

TF_CFG = {
    "15m": dict(sigma=0.002, mu=0.00005, bars_per_day=96,  lookfwd=60,  start=65_000.0),
    "1h":  dict(sigma=0.005, mu=0.0002,  bars_per_day=24,  lookfwd=120, start=65_000.0),
}

# ── candle ────────────────────────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp=ts; self.open=o; self.high=h
        self.low=l; self.close=c; self.volume=v

def gbm_candles(n, seed, cfg):
    rng = random.Random(seed)
    candles, price = [], cfg["start"]
    bar_sec = 3600 if cfg["bars_per_day"] == 24 else 900
    for i in range(n):
        ret   = cfg["mu"] + cfg["sigma"] * rng.gauss(0, 1)
        o     = price
        c     = price * math.exp(ret)
        h     = max(o, c) * (1 + abs(rng.gauss(0, cfg["sigma"] * 0.5)))
        l     = min(o, c) * (1 - abs(rng.gauss(0, cfg["sigma"] * 0.5)))
        candles.append(C(i * bar_sec, o, h, l, c, rng.uniform(1, 10)))
        price = c
    return candles

# ── signal extraction per strategy ────────────────────────────────────────────
def extract_signals(strat, candles):
    """Returns list of (bar_idx, entry_price, tp_price, sl_price, strategy_name)."""
    name = type(strat).__name__
    n    = len(candles)
    ha_c = np.array([c.close for c in candles], dtype=float)  # fallback

    if name == "WTADXStrategy":
        _, _, buy_arr, _, atr_a = strat._build_signals(candles)
        min_i = strat.n1 + strat.n2 + 14 + 10
        signals = []
        for i in range(min_i, n - 1):
            if buy_arr[i] and not np.isnan(atr_a[i]) and atr_a[i] > 0:
                entry = float(candles[i].close)
                atr   = float(atr_a[i])
                sl_p  = entry - strat.sl_atr_mult * atr
                tp_p  = entry + strat.sl_atr_mult * strat.rr_ratio * atr
                signals.append((i, entry, tp_p, sl_p))
        return signals

    elif name == "UTBotStrategy":
        buy_arr, _, _, atr_a = strat._build_signals(candles)
        min_i = strat.ut_atr_len + 14 + 5
        signals = []
        for i in range(min_i, n - 1):
            if buy_arr[i] and not np.isnan(atr_a[i]) and atr_a[i] > 0:
                entry = float(candles[i].close)
                atr   = float(atr_a[i])
                sl_p  = entry - strat.sl_atr_mult * atr
                tp_p  = entry + strat.sl_atr_mult * strat.rr_ratio * atr
                signals.append((i, entry, tp_p, sl_p))
        return signals

    elif name == "MomentumScoreStrategy":
        buy_arr, _, _, _, atr_a = strat._build_signals(candles)
        min_i = strat.macd_slow + strat.macd_sig + strat.ema_len + 5
        signals = []
        for i in range(min_i, n - 1):
            if buy_arr[i] and not np.isnan(atr_a[i]) and atr_a[i] > 0:
                entry = float(candles[i].close)
                atr   = float(atr_a[i])
                sl_p  = entry - strat.sl_atr_mult * atr
                tp_p  = entry + strat.sl_atr_mult * strat.rr_ratio * atr
                signals.append((i, entry, tp_p, sl_p))
        return signals

    elif name == "MACDEMAStrategy":
        (ha_o, ha_c_arr, ha_highs, ha_lows,
         hma, ema, sma, ml, sl_line, hist, atr_a) = strat._build_arrays(candles)
        min_i = strat.macd_slow + strat.macd_sig + strat.hma_period + 5
        signals = []
        for i in range(min_i, n - 1):
            d = strat._signal_at(i, ha_o, ha_c_arr, ha_highs, ha_lows,
                                 hma, ema, sma, ml, sl_line)
            if d == 1 and not np.isnan(atr_a[i]) and atr_a[i] > 0:
                entry = float(candles[i].close)
                atr   = float(atr_a[i])
                sl_p  = entry - strat.sl_atr_mult * atr
                tp_p  = entry + strat.sl_atr_mult * strat.rr_ratio * atr
                signals.append((i, entry, tp_p, sl_p))
        return signals

    return []

# ── multi-position simulation engine ─────────────────────────────────────────
def simulate_windows(strategies, candles_all, cfg, n_windows=5, bars_per_win=250):
    """Simulate n_windows × bars_per_win bars with multi-position management."""
    highs  = np.array([c.high  for c in candles_all], dtype=float)
    lows   = np.array([c.low   for c in candles_all], dtype=float)
    closes = np.array([c.close for c in candles_all], dtype=float)
    total_bars = n_windows * bars_per_win
    lookfwd = cfg["lookfwd"]

    # Pre-extract signals from full dataset for each strategy
    all_signals = {}
    for sname, strat in strategies:
        sigs = extract_signals(strat, candles_all[:total_bars + 50])
        # Group by bar
        sig_map = {}
        for (bi, entry, tp, sl) in sigs:
            sig_map.setdefault(bi, []).append((entry, tp, sl))
        all_signals[sname] = sig_map

    balance = START_BAL
    window_results = []

    # open_positions: list of dicts
    open_positions = []   # {sname, entry, tp_p, sl_p, amount_usd, open_bar, exit_bar}

    # All trade records
    all_trades = []

    for w in range(n_windows):
        w_start = w * bars_per_win
        w_end   = w_start + bars_per_win
        w_trades = []
        w_open_at_start = len(open_positions)

        for i in range(w_start, w_end):
            # ── 1. check exits for open positions at bar i ─────────────────
            still_open = []
            for pos in open_positions:
                if i < pos["open_bar"] + 1:
                    still_open.append(pos)
                    continue
                if i > pos["exit_bar"]:  # timeout
                    exit_p = float(closes[min(pos["exit_bar"], len(closes)-1)])
                    pct    = (exit_p - pos["entry"]) / pos["entry"]
                    pnl    = pos["amount_usd"] * pct - pos["amount_usd"] * FEE_RT
                    balance += pnl
                    t = {"outcome":"timeout","pnl":round(pnl,4),"sname":pos["sname"],"bar":i}
                    w_trades.append(t); all_trades.append(t)
                    continue
                # check SL then TP
                hit = None
                if lows[i] <= pos["sl_p"]:
                    pct  = (pos["sl_p"] - pos["entry"]) / pos["entry"]
                    pnl  = pos["amount_usd"] * pct - pos["amount_usd"] * FEE_RT
                    hit  = ("loss", pnl)
                elif highs[i] >= pos["tp_p"]:
                    pct  = (pos["tp_p"] - pos["entry"]) / pos["entry"]
                    pnl  = pos["amount_usd"] * pct - pos["amount_usd"] * FEE_RT
                    hit  = ("win", pnl)

                if hit:
                    balance += hit[1]
                    t = {"outcome":hit[0],"pnl":round(hit[1],4),"sname":pos["sname"],"bar":i}
                    w_trades.append(t); all_trades.append(t)
                else:
                    still_open.append(pos)
            open_positions = still_open

            # ── 2. collect new signals at bar i ───────────────────────────
            for sname, _ in strategies:
                if i not in all_signals[sname]:
                    continue

                # position limit checks
                total_open = len(open_positions)
                if total_open >= MAX_POS:
                    continue

                sname_open = sum(1 for p in open_positions if p["sname"] == sname)
                max_for_strat = WT_MAX_POS if sname == "WaveTrend" else 1
                if sname_open >= max_for_strat:
                    continue

                # open position with first signal at this bar
                entry, tp_p, sl_p = all_signals[sname][i][0]
                amount_usd = balance * RISK_PCT
                if amount_usd < 1.0:
                    continue

                open_positions.append({
                    "sname":      sname,
                    "entry":      entry,
                    "tp_p":       tp_p,
                    "sl_p":       sl_p,
                    "amount_usd": amount_usd,
                    "open_bar":   i,
                    "exit_bar":   i + lookfwd,
                })

        # close remaining open positions at window end (mark-to-market)
        still_open = []
        for pos in open_positions:
            if pos["open_bar"] < w_end:  # opened in this or earlier window
                exit_p = float(closes[min(w_end - 1, len(closes)-1)])
                pct    = (exit_p - pos["entry"]) / pos["entry"]
                pnl    = pos["amount_usd"] * pct - pos["amount_usd"] * FEE_RT
                balance += pnl
                t = {"outcome":"timeout","pnl":round(pnl,4),"sname":pos["sname"],"bar":w_end-1}
                w_trades.append(t); all_trades.append(t)
            else:
                still_open.append(pos)
        open_positions = still_open

        wins   = [t for t in w_trades if t["outcome"]=="win"]
        losses = [t for t in w_trades if t["outcome"] in ("loss","timeout") and t["pnl"]<0]
        net    = sum(t["pnl"] for t in w_trades)
        wr     = len(wins)/len(w_trades)*100 if w_trades else 0

        window_results.append({
            "window": w+1, "trades": len(w_trades),
            "wins": len(wins), "losses": len(losses),
            "wr": round(wr, 1), "net": round(net, 2),
            "balance": round(balance, 2),
        })

    return window_results, all_trades, balance

# ── main ─────────────────────────────────────────────────────────────────────
async def main():
    N_WINDOWS = 5
    BARS      = 250
    TOTAL     = N_WINDOWS * BARS + 60  # extra buffer for lookfwd

    print("=" * 72)
    print("  BTC/USDT — 15m vs 1H Backtest  │  250 bars × 5 windows")
    print(f"  $500 capital │ Risk 20%/trade │ Max {MAX_POS} pos │ WT max {WT_MAX_POS} │ Fee 0.20% RT")
    print("=" * 72)

    for tf in ("15m", "1h"):
        cfg = TF_CFG[tf]
        candles = gbm_candles(TOTAL, seed=99, cfg=cfg)

        strategies = [
            ("WaveTrend", WTADXStrategy("BTC/USDT")),
            ("UT Bot",    UTBotStrategy("BTC/USDT")),
            ("Momentum",  MomentumScoreStrategy("BTC/USDT")),
            ("MACD/EMA",  MACDEMAStrategy("BTC/USDT")),
        ]

        print(f"\n{'─'*72}")
        print(f"  TF: {tf.upper()}  │  {cfg['bars_per_day']} bars/day  │  "
              f"ATR≈{cfg['sigma']*100:.2f}%/bar  │  "
              f"Lookfwd {cfg['lookfwd']} bars")
        print(f"{'─'*72}")

        window_res, all_trades, final_bal = simulate_windows(
            strategies, candles, cfg, N_WINDOWS, BARS
        )

        # per-strategy breakdown
        strat_names = [s for s, _ in strategies]

        print(f"\n  {'Window':<10} │ {'Trades':>6} │ {'W':>4} │ {'L':>4} │ "
              f"{'WR%':>6} │ {'Net $':>8} │ {'Balance':>9}")
        print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*4}─┼─"
              f"{'─'*6}─┼─{'─'*8}─┼─{'─'*9}")

        for r in window_res:
            sign = "+" if r["net"] >= 0 else ""
            print(f"  Window {r['window']:<3}   │ {r['trades']:>6} │ {r['wins']:>4} │ "
                  f"{r['losses']:>4} │ {r['wr']:>5.1f}% │ "
                  f"{sign}{r['net']:>7.2f} │ ${r['balance']:>8.2f}")

        total_trades = sum(r["trades"] for r in window_res)
        total_wins   = sum(r["wins"]   for r in window_res)
        total_losses = sum(r["losses"] for r in window_res)
        total_net    = sum(r["net"]    for r in window_res)
        avg_wr       = sum(r["wr"]     for r in window_res) / N_WINDOWS
        sign = "+" if total_net >= 0 else ""

        print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*4}─┼─"
              f"{'─'*6}─┼─{'─'*8}─┼─{'─'*9}")
        print(f"  {'TOTAL':<10}   │ {total_trades:>6} │ {total_wins:>4} │ "
              f"{total_losses:>4} │ {avg_wr:>5.1f}% │ "
              f"{sign}{total_net:>7.2f} │ ${final_bal:>8.2f}")

        days = N_WINDOWS * BARS / cfg["bars_per_day"]
        tpd  = total_trades / days
        print(f"\n  → {days:.0f} days | {tpd:.1f} trades/day | "
              f"Capital ${START_BAL:.0f} → ${final_bal:.2f} "
              f"({'+'if final_bal>=START_BAL else ''}{final_bal-START_BAL:.2f} / "
              f"{(final_bal-START_BAL)/START_BAL*100:.1f}%)")

        # per-strategy breakdown
        print(f"\n  Per-strategy breakdown:")
        print(f"  {'Strategy':<12} │ {'Trades':>6} │ {'W/L':>7} │ {'WR%':>6} │ {'Net $':>8}")
        print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*8}")
        for sn in strat_names:
            st = [t for t in all_trades if t["sname"]==sn]
            sw = [t for t in st if t["outcome"]=="win"]
            sl = [t for t in st if t["outcome"] in ("loss","timeout") and t["pnl"]<0]
            if not st: continue
            wr = len(sw)/len(st)*100
            net = sum(t["pnl"] for t in st)
            sign = "+" if net >= 0 else ""
            print(f"  {sn:<12} │ {len(st):>6} │ {len(sw):>3}/{len(sl):<3} │ "
                  f"{wr:>5.1f}% │ {sign}{net:>7.2f}")

        # TP/SL size estimate
        avg_pct = cfg["sigma"] * 1.5 * 100  # rough ATR-based SL estimate
        pos_usd = START_BAL * RISK_PCT
        print(f"\n  Fee impact: position=${pos_usd:.0f} | fee/trade=${pos_usd*FEE_RT:.2f} | "
              f"ATR-SL≈${pos_usd*avg_pct/100:.2f} | fee/SL={FEE_RT/(avg_pct/100)*100:.0f}%")

    print(f"\n{'='*72}")

asyncio.run(main())
