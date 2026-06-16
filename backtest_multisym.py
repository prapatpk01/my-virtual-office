"""
Multi-Symbol Backtest — compare symbol combinations
5 strategies, all symbols share one global capital pool.
250 bars × 3 windows (~1 month total, 1H bars per symbol).

Symbol GBM volatility:
  BTC  σ=0.005  (normal crypto)
  TRX  σ=0.008  (higher vol altcoin)
  XAUT σ=0.002  (gold-pegged token, low vol)

Capital:  $300 | $50/trade | Max 3 global positions
Fee:      0.25% RT
TP/SL:    4.8% / 4.6%   (all strategies)
"""
import sys, math, random
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.ut_bot_strategy        import UTBotStrategy
from trading.strategies.wt_adx_strategy        import WTADXStrategy
from trading.strategies.momentum_score_strategy import MomentumScoreStrategy
from trading.strategies.macd_ema_strategy       import MACDEMAStrategy
from trading.strategies.knn_strategy            import KNNStrategy

# ── constants ─────────────────────────────────────────────────────────────────
POSITION   = 50.0
FEE_RT     = 0.0025
START_BAL  = 300.0
LOOKFWD    = 120
MAX_GLOBAL = 3
TP_PCT     = 0.048
SL_PCT     = 0.046

TP_NET = round(POSITION * TP_PCT - POSITION * FEE_RT, 4)
SL_NET = round(POSITION * SL_PCT + POSITION * FEE_RT, 4)

# ── candle ────────────────────────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self,ts,o,h,l,c,v):
        self.timestamp=ts;self.open=o;self.high=h;self.low=l;self.close=c;self.volume=v

ASSET_PARAMS = {
    "BTC":  {"start": 65_000.0, "mu": 0.0002,  "sigma": 0.005,  "seed_offset": 0},
    "ETH":  {"start":  3_000.0, "mu": 0.0002,  "sigma": 0.007,  "seed_offset": 10},
    "TRX":  {"start":      0.13, "mu": 0.00015, "sigma": 0.008,  "seed_offset": 20},
    "XAUT": {"start":  2_000.0, "mu": 0.00005,  "sigma": 0.002,  "seed_offset": 30},
}

def gbm(n, asset, seed_base=99):
    p = ASSET_PARAMS[asset]
    rng = random.Random(seed_base + p["seed_offset"])
    candles = []; price = p["start"]
    for i in range(n):
        ret = p["mu"] + p["sigma"] * rng.gauss(0, 1)
        o = price; c = price * math.exp(ret)
        h = max(o,c) * (1 + abs(rng.gauss(0, p["sigma"]*0.5)))
        l = min(o,c) * (1 - abs(rng.gauss(0, p["sigma"]*0.5)))
        candles.append(C(i*3600, o, h, l, c, rng.uniform(1,10)))
        price = c
    return candles

# ── build strategies for a given symbol ───────────────────────────────────────
def make_strategies(sym):
    pair = f"{sym}/USDT"
    return [
        ("WaveTrend", WTADXStrategy(pair)),
        ("UT Bot",    UTBotStrategy(pair)),
        ("Momentum",  MomentumScoreStrategy(pair)),
        ("MACD/EMA",  MACDEMAStrategy(pair)),
        ("kNN",       KNNStrategy(pair, {"sl_pct": SL_PCT, "tp_pct": TP_PCT})),
    ]

# ── compute buy signals ────────────────────────────────────────────────────────
def get_buy_signals(strategies, candles):
    n = len(candles); sigs = {}
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
            arrs = strat._build_arrays(candles)
            ha_o,ha_c,ha_h,ha_l,hma,ema,sma,ml,sl_l,hist,atr_a = arrs
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
            b,_,_,_ = strat._build_signals(candles)
            sigs[sname] = {i for i in range(n) if b[i]}
        else:
            sigs[sname] = set()
    return sigs

# ── simulate one window, multiple symbols ─────────────────────────────────────
def simulate_window(sym_candles_list, balance):
    """
    sym_candles_list: [(sym, strategies, candles), ...]
    All symbols share one balance + global position count.
    """
    # pre-compute signals per symbol
    sym_sigs = {}
    sym_arrays = {}
    for sym, strategies, candles in sym_candles_list:
        sym_sigs[sym] = get_buy_signals(strategies, candles)
        n = len(candles)
        sym_arrays[sym] = {
            "hig": np.array([c.high  for c in candles]),
            "low": np.array([c.low   for c in candles]),
            "cls": np.array([c.close for c in candles]),
            "n":   n,
        }

    n = sym_arrays[sym_candles_list[0][0]]["n"]
    open_positions = []
    all_trades = []

    for bar in range(n):
        # resolve open positions
        still_open = []
        for pos in open_positions:
            arr = sym_arrays[pos["sym"]]
            resolved = False
            if bar > pos["entry_bar"] + LOOKFWD:
                ep  = float(arr["cls"][bar])
                pnl = round((ep - pos["entry_price"]) / pos["entry_price"] * POSITION - POSITION*FEE_RT, 4)
                balance += pnl
                all_trades.append({"sym": pos["sym"], "sname": pos["sname"], "outcome": "timeout", "pnl": pnl})
                resolved = True
            elif arr["low"][bar] <= pos["sl_p"]:
                balance -= SL_NET
                all_trades.append({"sym": pos["sym"], "sname": pos["sname"], "outcome": "loss", "pnl": -SL_NET})
                resolved = True
            elif arr["hig"][bar] >= pos["tp_p"]:
                balance += TP_NET
                all_trades.append({"sym": pos["sym"], "sname": pos["sname"], "outcome": "win", "pnl": TP_NET})
                resolved = True
            if not resolved:
                still_open.append(pos)
        open_positions = still_open

        # open new positions
        for sym, strategies, candles in sym_candles_list:
            arr = sym_arrays[sym]
            if bar >= arr["n"]: continue
            for sname, _ in strategies:
                if bar not in sym_sigs[sym].get(sname, set()):
                    continue
                strat_count = sum(1 for p in open_positions if p["sym"]==sym and p["sname"]==sname)
                if strat_count >= 1:
                    continue
                if len(open_positions) >= MAX_GLOBAL:
                    continue
                entry = float(arr["cls"][bar])
                open_positions.append({
                    "sym": sym, "sname": sname,
                    "entry_bar": bar, "entry_price": entry,
                    "tp_p": entry*(1+TP_PCT), "sl_p": entry*(1-SL_PCT),
                })

    # close remaining positions at window end
    for pos in open_positions:
        arr = sym_arrays[pos["sym"]]
        ep  = float(arr["cls"][-1])
        pnl = round((ep - pos["entry_price"]) / pos["entry_price"] * POSITION - POSITION*FEE_RT, 4)
        balance += pnl
        all_trades.append({"sym": pos["sym"], "sname": pos["sname"], "outcome": "timeout", "pnl": pnl})

    return all_trades, balance

# ── run one symbol-combination config ─────────────────────────────────────────
def run_config(label, symbols, n_bars=760, seed=99):
    # generate enough bars for warmup + 3 windows of 250
    candles_all = {sym: gbm(n_bars, sym, seed) for sym in symbols}
    windows = []
    for w in range(3):
        win = []
        for sym in symbols:
            tail = candles_all[sym][-750:]
            win_candles = tail[w*250:(w+1)*250]
            strats = make_strategies(sym)
            win.append((sym, strats, win_candles))
        windows.append(win)

    wlabels = ["W1 (oldest)", "W2 (mid)   ", "W3 (newest)"]
    balance = START_BAL
    all_trades = []

    print(f"\n{'─'*72}")
    print(f"  Config: {label}  |  Symbols: {', '.join(symbols)}")
    print(f"  {'Window':<14} {'T':>4} {'W':>4} {'L':>4} {'TO':>4} {'WR%':>6} {'Net$':>8} {'Bal':>9}")
    print(f"  {'─'*14}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*9}")

    for wi, win in enumerate(windows):
        trades, balance = simulate_window(win, balance)
        all_trades.extend(trades)
        w_ = [t for t in trades if t["outcome"]=="win"]
        l_ = [t for t in trades if t["outcome"]=="loss"]
        to = [t for t in trades if t["outcome"]=="timeout"]
        nn = len(trades)
        wr = len(w_)/max(len(w_)+len(l_),1)*100 if (len(w_)+len(l_))>0 else 0
        net= sum(t["pnl"] for t in trades)
        s  = "+" if net>=0 else ""
        print(f"  {wlabels[wi]:<14} {nn:>4} {len(w_):>4} {len(l_):>4} {len(to):>4} "
              f"{wr:>5.1f}% {s}{net:>7.2f} ${balance:>8.2f}")

    aw = [t for t in all_trades if t["outcome"]=="win"]
    al = [t for t in all_trades if t["outcome"]=="loss"]
    at = [t for t in all_trades if t["outcome"]=="timeout"]
    an = len(all_trades)
    awr = len(aw)/max(len(aw)+len(al),1)*100 if (len(aw)+len(al))>0 else 0
    anet= sum(t["pnl"] for t in all_trades)
    s   = "+" if anet>=0 else ""
    t_per_day = an / (750/24)
    print(f"  {'─'*14}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*9}")
    print(f"  {'TOTAL':<14} {an:>4} {len(aw):>4} {len(al):>4} {len(at):>4} "
          f"{awr:>5.1f}% {s}{anet:>7.2f} ${balance:>8.2f}")
    print(f"  → {t_per_day:.1f} trades/day | $300 → ${balance:.2f} "
          f"({s}{balance-START_BAL:.2f}, {(balance-START_BAL)/START_BAL*100:.1f}%)")

    # per-symbol breakdown
    syms_seen = list(dict.fromkeys(t["sym"] for t in all_trades))
    if len(syms_seen) > 1:
        print(f"\n  Per-symbol  {'T':>4} {'W':>4} {'L':>4} {'WR%':>6} {'Net$':>8}")
        for sym in syms_seen:
            st = [t for t in all_trades if t["sym"]==sym]
            sw = [t for t in st if t["outcome"]=="win"]
            sl = [t for t in st if t["outcome"]=="loss"]
            sn = sum(t["pnl"] for t in st)
            wr = len(sw)/max(len(sw)+len(sl),1)*100 if (len(sw)+len(sl))>0 else 0
            s  = "+" if sn>=0 else ""
            print(f"  {sym:<10}  {len(st):>4} {len(sw):>4} {len(sl):>4} {wr:>5.1f}% {s}{sn:>7.2f}")

    return balance, anet, awr, an

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  Multi-Symbol Backtest — 250 bars × 3 windows (~1 month, 1H)")
    print(f"  $300 capital | ${POSITION}/trade | Max {MAX_GLOBAL} positions | Fee {FEE_RT*100:.2f}% RT")
    print(f"  TP={TP_PCT*100:.1f}% (+${TP_NET:.3f}) | SL={SL_PCT*100:.1f}% (-${SL_NET:.3f})")
    print("="*72)

    configs = [
        ("BTC only",      ["BTC"]),
        ("BTC + ETH+TRX", ["BTC","ETH","TRX"]),
        ("BTC + XAUT",    ["BTC","XAUT"]),
        ("BTC+TRX+XAUT",  ["BTC","TRX","XAUT"]),
    ]

    summary = []
    for label, symbols in configs:
        bal, net, wr, trades = run_config(label, symbols)
        summary.append((label, bal, net, wr, trades))

    be_wr = SL_NET / (TP_NET + SL_NET) * 100
    print(f"\n{'='*72}")
    print(f"  SUMMARY — Break-even WR: {be_wr:.1f}%")
    print(f"  {'Config':<20} {'Trades':>6} {'WR%':>6} {'Net$':>8} {'Final':>9} {'ROI%':>6}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*6}")
    best_roi = -999
    best_label = ""
    for label, bal, net, wr, trades in summary:
        roi = (bal - START_BAL) / START_BAL * 100
        s   = "+" if net >= 0 else ""
        print(f"  {label:<20} {trades:>6} {wr:>5.1f}% {s}{net:>7.2f} ${bal:>8.2f} {roi:>+5.1f}%")
        if roi > best_roi:
            best_roi = roi; best_label = label
    print(f"\n  Best: {best_label}  (ROI {best_roi:+.1f}%)")
    print("="*72)

main()
