"""
Multi-Position Full Backtest — 750 bars × 3 windows (~3 months, 1H)
5 strategies running concurrently, shared $500 capital.

Rules:
  - $500 capital, $50/trade
  - Max 4 open positions (global)
  - WaveTrend: max 2 concurrent  |  others: max 1 each
  - Fee 0.25% RT (0.10+0.10+0.05 slippage)
  - TP/SL per strategy:
      WT/UTBot/Momentum/MACD: TP=4.8%  SL=4.6%
      kNN:                    TP=2.25% SL=1.5%
  - Lookforward: 120 bars (timeout if neither hit)
"""
import sys, math, random
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.ut_bot_strategy         import UTBotStrategy
from trading.strategies.wt_adx_strategy         import WTADXStrategy
from trading.strategies.momentum_score_strategy  import MomentumScoreStrategy
from trading.strategies.macd_ema_strategy        import MACDEMAStrategy
from trading.strategies.knn_strategy             import KNNStrategy

# ── constants ─────────────────────────────────────────────────────────────────
POSITION   = 50.0
FEE_RT     = 0.0025   # 0.25% round-trip
START_BAL  = 500.0
LOOKFWD    = 120
MAX_GLOBAL = 4
MAX_WT     = 2        # WaveTrend max concurrent

STRAT_PARAMS = {
    "WaveTrend": {"tp": 0.048,  "sl": 0.046, "max_pos": MAX_WT},
    "UT Bot":    {"tp": 0.048,  "sl": 0.046, "max_pos": 1},
    "Momentum":  {"tp": 0.048,  "sl": 0.046, "max_pos": 1},
    "MACD/EMA":  {"tp": 0.048,  "sl": 0.046, "max_pos": 1},
    "kNN":       {"tp": 0.048,  "sl": 0.046, "max_pos": 1},
}

# ── candle ────────────────────────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self,ts,o,h,l,c,v):
        self.timestamp=ts;self.open=o;self.high=h;self.low=l;self.close=c;self.volume=v

def gbm(n, seed=99, start=100_000.0, mu=0.0002, sigma=0.005):
    rng=random.Random(seed); candles=[]; price=start
    for i in range(n):
        ret=mu+sigma*rng.gauss(0,1); o=price; c=price*math.exp(ret)
        h=max(o,c)*(1+abs(rng.gauss(0,sigma*0.5)))
        l=min(o,c)*(1-abs(rng.gauss(0,sigma*0.5)))
        candles.append(C(i*3600,o,h,l,c,rng.uniform(1,10))); price=c
    return candles

# ── pre-compute all buy signals for one window ────────────────────────────────
def get_buy_signals(strategies, candles):
    """Returns {sname: set(bar_idx)} for buy signals."""
    n = len(candles)
    sigs = {}
    for sname, strat in strategies:
        name = type(strat).__name__
        if name == "UTBotStrategy":
            b,_,_,_ = strat._build_signals(candles)
            mi = strat.ut_atr_len + 14 + 5
            sigs[sname] = {i for i in range(mi, n-1) if b[i]}
        elif name == "MomentumScoreStrategy":
            b,_,_,_,_ = strat._build_signals(candles)
            mi = strat.macd_slow + strat.macd_sig + strat.ema_len + 5
            sigs[sname] = {i for i in range(mi, n-1) if b[i]}
        elif name == "MACDEMAStrategy":
            (ha_o,ha_c,ha_h,ha_l,hma,ema,sma,ml,sl_l,hist,atr_a) = strat._build_arrays(candles)
            mi = strat.macd_slow + strat.macd_sig + strat.hma_period + 5
            buys = set()
            for i in range(mi, n-1):
                if strat._signal_at(i,ha_o,ha_c,ha_h,ha_l,hma,ema,sma,ml,sl_l) == 1:
                    buys.add(i)
            sigs[sname] = buys
        elif name == "WTADXStrategy":
            _,_,b,_,_ = strat._build_signals(candles)
            mi = strat.n1 + strat.n2 + 14 + 10
            sigs[sname] = {i for i in range(mi, n-1) if b[i]}
        elif name == "KNNStrategy":
            b,_,_ = strat._build_signals(candles)
            sigs[sname] = {i for i in range(n) if b[i]}
        else:
            sigs[sname] = set()
    return sigs

# ── multi-position simulation for one window ──────────────────────────────────
def simulate_multi(strategies, candles, balance):
    n   = len(candles)
    hig = np.array([c.high  for c in candles])
    low = np.array([c.low   for c in candles])
    cls = np.array([c.close for c in candles])

    sigs = get_buy_signals(strategies, candles)

    open_positions = []   # list of dicts
    all_trades     = []

    for bar in range(n):
        # ── 1. Resolve open positions at this bar ─────────────────────────────
        still_open = []
        for pos in open_positions:
            resolved = False
            # timeout check
            if bar > pos["entry_bar"] + LOOKFWD:
                ep   = float(cls[bar])
                pnl  = round((ep - pos["entry_price"]) / pos["entry_price"] * POSITION
                             - POSITION * FEE_RT, 4)
                balance += pnl
                all_trades.append({"sname": pos["sname"], "outcome": "timeout",
                                    "pnl": pnl, "bar": bar})
                resolved = True
            elif low[bar] <= pos["sl_p"]:
                balance -= pos["sl_net"]
                all_trades.append({"sname": pos["sname"], "outcome": "loss",
                                    "pnl": -pos["sl_net"], "bar": bar})
                resolved = True
            elif hig[bar] >= pos["tp_p"]:
                balance += pos["tp_net"]
                all_trades.append({"sname": pos["sname"], "outcome": "win",
                                    "pnl": pos["tp_net"], "bar": bar})
                resolved = True

            if not resolved:
                still_open.append(pos)
        open_positions = still_open

        # ── 2. Open new positions at this bar ─────────────────────────────────
        for sname, _ in strategies:
            if bar not in sigs.get(sname, set()):
                continue
            sp  = STRAT_PARAMS[sname]
            # strategy position limit
            strat_count = sum(1 for p in open_positions if p["sname"] == sname)
            if strat_count >= sp["max_pos"]:
                continue
            # global limit
            if len(open_positions) >= MAX_GLOBAL:
                continue
            # open
            entry  = float(cls[bar])
            tp_net = round(POSITION * sp["tp"] - POSITION * FEE_RT, 4)
            sl_net = round(POSITION * sp["sl"] + POSITION * FEE_RT, 4)
            open_positions.append({
                "sname":       sname,
                "entry_bar":   bar,
                "entry_price": entry,
                "tp_p":        entry * (1 + sp["tp"]),
                "sl_p":        entry * (1 - sp["sl"]),
                "tp_net":      tp_net,
                "sl_net":      sl_net,
            })

    # ── Close any still-open positions at window end ──────────────────────────
    for pos in open_positions:
        ep  = float(cls[-1])
        pnl = round((ep - pos["entry_price"]) / pos["entry_price"] * POSITION
                    - POSITION * FEE_RT, 4)
        balance += pnl
        all_trades.append({"sname": pos["sname"], "outcome": "timeout",
                            "pnl": pnl, "bar": n-1})

    return all_trades, balance

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print("="*74)
    print("  BTC/USDT 1H — Multi-Position Backtest  750 bars × 3 windows")
    print("  $500 capital | $50/trade | Fee 0.25% RT | Max 4 positions")
    print("  WaveTrend max 2 concurrent | others max 1 each")
    print("  WT/UTBot/Mom/MACD: TP=4.8% SL=4.6%  |  kNN: TP=2.25% SL=1.5%")
    print("="*74)

    strategies = [
        ("WaveTrend", WTADXStrategy("BTC/USDT")),
        ("UT Bot",    UTBotStrategy("BTC/USDT")),
        ("Momentum",  MomentumScoreStrategy("BTC/USDT")),
        ("MACD/EMA",  MACDEMAStrategy("BTC/USDT")),
        ("kNN",       KNNStrategy("BTC/USDT")),
    ]
    snames = [s[0] for s in strategies]

    # 2300 bars GBM → take last 2250, split 3×750
    candles_all = gbm(2400, seed=99)
    tail    = candles_all[-2250:]
    windows = [tail[i*750:(i+1)*750] for i in range(3)]
    wlabels = ["W1 (oldest ~month 1)", "W2 (mid ~month 2)", "W3 (newest ~month 3)"]

    print(f"\n  Data: GBM 1H (seed=99) | W1: bars 1-750 | W2: 751-1500 | W3: 1501-2250\n")

    balance = START_BAL
    window_results = []

    for wi, win_candles in enumerate(windows):
        trades, balance = simulate_multi(strategies, win_candles, balance)
        per_strat = {sn: {"w":0,"l":0,"to":0,"net":0.0} for sn in snames}
        for t in trades:
            sn = t["sname"]; p = t["pnl"]
            ps = per_strat[sn]
            ps["net"] += p
            if t["outcome"] == "win":      ps["w"]  += 1
            elif t["outcome"] == "loss":   ps["l"]  += 1
            else:                          ps["to"] += 1
        total_t = len(trades)
        total_w = sum(v["w"] for v in per_strat.values())
        total_l = sum(v["l"] for v in per_strat.values())
        total_n = sum(v["net"] for v in per_strat.values())
        wr = total_w / max(total_w + total_l, 1) * 100
        window_results.append({"wi": wi, "trades": total_t, "w": total_w,
                                "l": total_l, "net": total_n, "bal": balance,
                                "per_strat": per_strat})

        sign = "+" if total_n >= 0 else ""
        print(f"  {'─'*70}")
        print(f"  {wlabels[wi]}  |  {total_t} trades  WR {wr:.1f}%  Net {sign}${total_n:.2f}  Bal ${balance:.2f}")
        print(f"  {'Strategy':<12} {'T':>4} {'W':>4} {'L':>4} {'TO':>4} {'WR%':>6} {'Net$':>8}")
        print(f"  {'─'*12}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}")
        for sn in snames:
            ps = per_strat[sn]
            t_ = ps["w"]+ps["l"]+ps["to"]
            wr_= ps["w"]/max(ps["w"]+ps["l"],1)*100 if (ps["w"]+ps["l"])>0 else 0
            s  = "+" if ps["net"]>=0 else ""
            print(f"  {sn:<12} {t_:>4} {ps['w']:>4} {ps['l']:>4} {ps['to']:>4} "
                  f"{wr_:>5.1f}% {s}{ps['net']:>7.2f}")

    # ── Grand summary ──────────────────────────────────────────────────────────
    print(f"\n  {'='*70}")
    print(f"  GRAND SUMMARY — 3 windows × 750 bars = 2250 bars (~3 months 1H)")
    print(f"  {'Window':<24} {'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} {'Net$':>9} {'Balance':>9}")
    print(f"  {'─'*24}  {'─'*6}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*9}")
    for r in window_results:
        wr = r["w"] / max(r["w"]+r["l"], 1) * 100
        s  = "+" if r["net"] >= 0 else ""
        print(f"  {wlabels[r['wi']]:<24} {r['trades']:>6} {r['w']:>4} {r['l']:>4} "
              f"{wr:>5.1f}% {s}{r['net']:>8.2f} ${r['bal']:>8.2f}")

    all_t = sum(r["trades"] for r in window_results)
    all_w = sum(r["w"]      for r in window_results)
    all_l = sum(r["l"]      for r in window_results)
    all_n = sum(r["net"]    for r in window_results)
    all_wr= all_w / max(all_w+all_l, 1) * 100
    s     = "+" if all_n >= 0 else ""
    t_per_day = all_t / (2250 / 24)
    print(f"  {'─'*24}  {'─'*6}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*9}")
    print(f"  {'TOTAL (2250 bars)':<24} {all_t:>6} {all_w:>4} {all_l:>4} "
          f"{all_wr:>5.1f}% {s}{all_n:>8.2f} ${balance:>8.2f}")
    print(f"\n  Trades/day: {t_per_day:.1f}  |  Capital: $500 → ${balance:.2f} "
          f"({s}{balance-START_BAL:.2f}, {(balance-START_BAL)/START_BAL*100:.1f}%)")

    # per-strategy totals
    combined = {sn: {"w":0,"l":0,"to":0,"net":0.0} for sn in snames}
    for r in window_results:
        for sn in snames:
            for k in ("w","l","to"):
                combined[sn][k] += r["per_strat"][sn][k]
            combined[sn]["net"] += r["per_strat"][sn]["net"]

    print(f"\n  Per-strategy totals (2250 bars):")
    print(f"  {'Strategy':<12} {'T':>4} {'W':>4} {'L':>4} {'WR%':>6} {'Net$':>8}")
    print(f"  {'─'*12}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}")
    for sn in snames:
        cs = combined[sn]; t_=cs["w"]+cs["l"]+cs["to"]
        wr_= cs["w"]/max(cs["w"]+cs["l"],1)*100 if (cs["w"]+cs["l"])>0 else 0
        s  = "+" if cs["net"]>=0 else ""
        print(f"  {sn:<12} {t_:>4} {cs['w']:>4} {cs['l']:>4} {wr_:>5.1f}% {s}{cs['net']:>7.2f}")

    print(f"  {'='*70}\n")

main()
