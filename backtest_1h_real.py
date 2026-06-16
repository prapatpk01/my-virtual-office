"""
Full 1H Backtest — 250 bars × 3 windows (750 bars total)  [5 strategies]
Real BTC/USDT candles from Binance (falls back to GBM if unavailable).

Rules:
  - $50/trade, $500 starting capital
  - TP/SL per strategy (TP/SL are strategy-specific, see STRAT_PARAMS)
  - Pure TP/SL exits only
  - 1 active trade per strategy at a time
  - Lookforward: 120 bars max before timeout
"""
import asyncio, sys, math, random
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.ut_bot_strategy         import UTBotStrategy
from trading.strategies.wt_adx_strategy         import WTADXStrategy
from trading.strategies.momentum_score_strategy  import MomentumScoreStrategy
from trading.strategies.macd_ema_strategy        import MACDEMAStrategy
from trading.strategies.knn_strategy             import KNNStrategy

# ── constants ───────────────────────────────────────────────────────────────
POSITION   = 50.0     # USD per trade
FEE_RT     = 0.0020   # 0.10% entry + 0.10% exit
START_BAL  = 500.0
LOOKFWD    = 120      # max bars to wait for TP/SL on 1H

# per-strategy TP/SL (tuned for 1H)
STRAT_PARAMS = {
    "WaveTrend":  {"tp": 0.048, "sl": 0.046},   # TP=4.8% SL=4.6%
    "UT Bot":     {"tp": 0.048, "sl": 0.046},
    "Momentum":   {"tp": 0.048, "sl": 0.046},
    "MACD/EMA":   {"tp": 0.048, "sl": 0.046},
    "kNN":        {"tp": 0.048,  "sl": 0.046},
}

def _net(tp_pct, sl_pct):
    return (round(POSITION * tp_pct - POSITION * FEE_RT, 4),
            round(POSITION * sl_pct + POSITION * FEE_RT, 4))

# legacy aliases kept for existing simulate_window
TP_PCT = 0.048; SL_PCT = 0.046
TP_NET = round(POSITION * TP_PCT - POSITION * FEE_RT, 4)
SL_NET = round(POSITION * SL_PCT + POSITION * FEE_RT, 4)

# ── candle dataclass ─────────────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp=ts; self.open=o; self.high=h
        self.low=l; self.close=c; self.volume=v

# ── fetch real candles from Binance ──────────────────────────────────────────
async def fetch_real(limit: int = 760):
    try:
        from trading.connectors.binance_conn import BinanceConnector
        conn = BinanceConnector(paper=True)
        raw = await conn.fetch_ohlcv("BTC/USDT", "1h", limit=limit)
        await conn._exchange.close()
        candles = [C(r.timestamp, r.open, r.high, r.low, r.close, r.volume)
                   for r in raw]
        print(f"  [Binance] fetched {len(candles)} real 1H candles  "
              f"(last close: ${candles[-1].close:,.0f})")
        return candles
    except Exception as e:
        print(f"  [Binance] unavailable ({e.__class__.__name__}: {e})")
        return None

# ── GBM fallback ─────────────────────────────────────────────────────────────
def gbm_candles(n: int, seed: int = 99, start: float = 100_000.0,
                mu: float = 0.0002, sigma: float = 0.005) -> list:
    rng = random.Random(seed)
    candles, price = [], start
    for i in range(n):
        ret   = mu + sigma * rng.gauss(0, 1)
        o     = price
        c     = price * math.exp(ret)
        h     = max(o, c) * (1 + abs(rng.gauss(0, sigma * 0.5)))
        l     = min(o, c) * (1 - abs(rng.gauss(0, sigma * 0.5)))
        candles.append(C(i * 3600, o, h, l, c, rng.uniform(1, 10)))
        price = c
    return candles

# ── signal extraction ─────────────────────────────────────────────────────────
def get_signals(strategy, candles: list):
    """
    Returns (buy_idx[], sell_idx[]) — bar indices where signals fire.
    Pure array-based; does NOT simulate trades.
    """
    name = type(strategy).__name__
    n = len(candles)

    if name == "UTBotStrategy":
        buy_arr, sell_arr, _, _ = strategy._build_signals(candles)
        min_i = strategy.ut_atr_len + 14 + 5
        buys  = [i for i in range(min_i, n - 1) if buy_arr[i]]
        sells = [i for i in range(min_i, n - 1) if sell_arr[i]]

    elif name == "MomentumScoreStrategy":
        buy_arr, sell_arr, _, _, _ = strategy._build_signals(candles)
        min_i = strategy.macd_slow + strategy.macd_sig + strategy.ema_len + 5
        buys  = [i for i in range(min_i, n - 1) if buy_arr[i]]
        sells = [i for i in range(min_i, n - 1) if sell_arr[i]]

    elif name == "MACDEMAStrategy":
        (ha_o, ha_c, ha_highs, ha_lows,
         hma, ema, sma, ml, sl_line, hist, atr_a) = strategy._build_arrays(candles)
        min_i = strategy.macd_slow + strategy.macd_sig + strategy.hma_period + 5
        buys, sells = [], []
        for i in range(min_i, n - 1):
            d = strategy._signal_at(i, ha_o, ha_c, ha_highs, ha_lows,
                                    hma, ema, sma, ml, sl_line)
            if d == 1:
                buys.append(i)
            elif d == -1:
                sells.append(i)

    elif name == "WTADXStrategy":
        _, _, buy_arr, sell_arr, _ = strategy._build_signals(candles)
        min_i = strategy.n1 + strategy.n2 + 14 + 10
        buys  = [i for i in range(min_i, n - 1) if buy_arr[i]]
        sells = [i for i in range(min_i, n - 1) if sell_arr[i]]

    elif name == "KNNStrategy":
        buy_arr, sell_arr, _ = strategy._build_signals(candles)
        buys  = [i for i in range(n) if buy_arr[i]]
        sells = [i for i in range(n) if sell_arr[i]]

    else:
        buys, sells = [], []

    return buys, sells

# ── simulate one 250-bar window ───────────────────────────────────────────────
def simulate_window(strategy, sname: str, candles: list, balance: float):
    """
    Returns (trades_list, ending_balance).
    Uses per-strategy TP/SL from STRAT_PARAMS.
    """
    highs  = np.array([c.high  for c in candles], dtype=float)
    lows   = np.array([c.low   for c in candles], dtype=float)
    closes = np.array([c.close for c in candles], dtype=float)
    n = len(candles)

    sp  = STRAT_PARAMS.get(sname, {"tp": TP_PCT, "sl": SL_PCT})
    tp_pct = sp["tp"]; sl_pct = sp["sl"]
    tp_net, sl_net = _net(tp_pct, sl_pct)

    buys, sells = get_signals(strategy, candles)
    sell_set = set(sells)

    trades = []
    trade_entry_bar = -1
    trade_exit_bar  = -1
    blocked_until   = -1

    for bi in buys:
        if bi <= trade_exit_bar:
            continue
        if bi < blocked_until:
            continue
        if bi <= trade_entry_bar:
            continue

        entry = float(closes[bi])
        tp_p  = entry * (1 + tp_pct)
        sl_p  = entry * (1 - sl_pct)

        outcome  = "timeout"
        exit_bar = min(bi + LOOKFWD, n - 1)
        pnl      = 0.0

        for j in range(bi + 1, min(bi + LOOKFWD, n)):
            if j in sell_set:
                blocked_until = j + 10
            if lows[j] <= sl_p:
                outcome  = "loss";  pnl = -sl_net; exit_bar = j; break
            if highs[j] >= tp_p:
                outcome  = "win";   pnl = tp_net;  exit_bar = j; break
        else:
            exit_p = float(closes[min(bi + LOOKFWD, n) - 1])
            pnl    = round((exit_p - entry) / entry * POSITION - POSITION * FEE_RT, 4)

        balance += pnl
        trades.append({
            "entry_bar": bi, "exit_bar": exit_bar,
            "outcome": outcome, "pnl": round(pnl, 4), "balance": round(balance, 2),
        })
        trade_entry_bar = bi
        trade_exit_bar  = exit_bar

    return trades, balance

# ── run all strategies on 3 windows ──────────────────────────────────────────
async def main():
    print("\n" + "="*74)
    print("  BTC/USDT 1H — Full Backtest  250 bars × 3 windows  [5 strategies]")
    print(f"  $50/trade | $500 capital | Fee 0.20% RT | Pure TP/SL exits | Lookfwd {LOOKFWD} bars")
    print(f"  WT/UTBot/Momentum/MACD: TP=4.8% SL=4.6%  |  kNN: TP=2.25% SL=1.5%")
    print("="*74)

    # ── data ──────────────────────────────────────────────────────────────────
    candles_all = await fetch_real(760)
    data_source = "Binance 1H"
    if candles_all is None or len(candles_all) < 750:
        candles_all = gbm_candles(760, seed=99)
        data_source = "Synthetic GBM 1H (seed=99, μ=0.02%/bar, σ=0.5%/bar)"
        print(f"  [GBM] using {len(candles_all)} synthetic 1H candles")

    # take last 750 bars, split into 3 × 250
    tail = candles_all[-750:]
    windows = [tail[i*250:(i+1)*250] for i in range(3)]
    label_map = {0: "Window 1 (oldest)", 1: "Window 2 (mid)", 2: "Window 3 (newest)"}

    strategies = [
        ("WaveTrend", WTADXStrategy("BTC/USDT")),
        ("UT Bot",    UTBotStrategy("BTC/USDT")),
        ("Momentum",  MomentumScoreStrategy("BTC/USDT")),
        ("MACD/EMA",  MACDEMAStrategy("BTC/USDT")),
        ("kNN",       KNNStrategy("BTC/USDT")),
    ]

    print(f"\n  Data: {data_source}")
    print(f"  Window 1: bars 1-250  |  Window 2: bars 251-500  |  Window 3: bars 501-750\n")

    grand_summary = []

    for sname, strat in strategies:
        print(f"{'─'*74}")
        print(f"  Strategy: {sname}")
        print(f"  {'Window':<20} │ {'Trades':>6} │ {'W':>4} │ {'L':>4} │ "
              f"{'WR%':>6} │ {'Net $':>8} │ {'Balance':>9}")
        print(f"  {'─'*20}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*4}─┼─"
              f"{'─'*6}─┼─{'─'*8}─┼─{'─'*9}")

        balance = START_BAL
        all_trades = []

        for wi, win_candles in enumerate(windows):
            trades, balance = simulate_window(strat, sname, win_candles, balance)
            wins   = [t for t in trades if t["outcome"] == "win"]
            losses = [t for t in trades if t["outcome"] in ("loss", "timeout") and t["pnl"] < 0]
            total  = len(trades)
            wr     = len(wins)/total*100 if total else 0
            net    = sum(t["pnl"] for t in trades)
            all_trades.extend(trades)

            label = label_map[wi]
            sign  = "+" if net >= 0 else ""
            print(f"  {label:<20} │ {total:>6} │ {len(wins):>4} │ {len(losses):>4} │ "
                  f"{wr:>5.1f}% │ {sign}{net:>7.2f} │ ${balance:>8.2f}")

        # overall
        all_wins   = [t for t in all_trades if t["outcome"] == "win"]
        all_losses = [t for t in all_trades if t["outcome"] in ("loss","timeout") and t["pnl"] < 0]
        all_total  = len(all_trades)
        all_wr     = len(all_wins)/all_total*100 if all_total else 0
        all_net    = sum(t["pnl"] for t in all_trades)
        all_sign   = "+" if all_net >= 0 else ""
        print(f"  {'─'*20}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*4}─┼─"
              f"{'─'*6}─┼─{'─'*8}─┼─{'─'*9}")
        print(f"  {'TOTAL (750 bars)':<20} │ {all_total:>6} │ {len(all_wins):>4} │ "
              f"{len(all_losses):>4} │ {all_wr:>5.1f}% │ "
              f"{all_sign}{all_net:>7.2f} │ ${balance:>8.2f}")

        trades_per_day = all_total / (750 / 24)
        print(f"  → {trades_per_day:.1f} trades/day avg  |  "
              f"Capital: ${START_BAL:.0f} → ${balance:.2f} "
              f"({'+'if balance>=START_BAL else ''}{balance-START_BAL:.2f})")

        grand_summary.append({
            "name": sname, "trades": all_total, "wins": len(all_wins),
            "losses": len(all_losses), "wr": all_wr,
            "net": all_net, "final_bal": balance,
            "t_per_day": trades_per_day,
        })

    # ── grand summary table ────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("  GRAND SUMMARY — all 5 strategies, 750 bars, $500 starting capital")
    print(f"  {'Strategy':<12} │ {'Trades':>6} │ {'W/L':>7} │ {'WR%':>6} │ "
          f"{'Net $':>8} │ {'Final Bal':>10} │ {'T/day':>6}")
    print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}─┼─"
          f"{'─'*8}─┼─{'─'*10}─┼─{'─'*6}")
    total_net = 0.0
    for r in grand_summary:
        sign = "+" if r["net"] >= 0 else ""
        bal_sign = "+" if r["final_bal"] >= START_BAL else ""
        wl = f"{r['wins']}/{r['losses']}"
        print(f"  {r['name']:<12} │ {r['trades']:>6} │ {wl:>7} │ "
              f"{r['wr']:>5.1f}% │ {sign}{r['net']:>7.2f} │ "
              f"${r['final_bal']:>8.2f}    │ {r['t_per_day']:>5.1f}")
        total_net += r["net"]

    avg_wr = sum(r["wr"] for r in grand_summary) / len(grand_summary)
    print(f"  {'─'*12}─┴─{'─'*6}─┴─{'─'*7}─┴─{'─'*6}─┴─"
          f"{'─'*8}─┴─{'─'*10}─┴─{'─'*6}")
    print(f"  {'Avg/Total':<12}   {'':>6}   {'':>7}   {avg_wr:>5.1f}%   "
          f"{'+'if total_net>=0 else ''}{total_net:>7.2f}")
    print(f"\n  Break-even WR: {SL_NET/(TP_NET+SL_NET)*100:.1f}%  "
          f"(TP net ${TP_NET} / SL net ${SL_NET})")
    print("="*74)

asyncio.run(main())
