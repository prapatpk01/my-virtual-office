"""
Backtest: WTADXStrategy & UTBotStrategy with HMA20/EMA5-SMA9 entry filters
Real 1H BTC/USDT data from OKX public API — last 5 months.

Entry: strategy BUY signal  AND  prev_close > HMA20  AND  EMA5 crosses above SMA9
Exit (first triggered):
  1. High >= TP                           → close at TP        (win)
  2. Low  <= SL                           → close at SL        (loss)
  3. Strategy SELL signal                 → close at bar close
  4. Close < HMA20 and next open < HMA20  → close at next open (filtered exit)

Capital: $260 start | $100/trade | 0.1% OKX taker fee per leg
"""
import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

try:
    import ccxt
except ImportError:
    pass

try:
    import urllib.request, json as _json
except ImportError:
    pass

from trading.connectors.base import OHLCV
from trading.strategies.base import BaseStrategy
from trading.strategies.wt_adx_strategy import WTADXStrategy
from trading.strategies.ut_bot_strategy import UTBotStrategy

# ─── Config ───────────────────────────────────────────────────────────────────
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1h"
MONTHS      = 5
CAPITAL     = 260.0
TRADE_USD   = 100.0
FEE_RATE    = 0.001
EMA_PERIOD  = 5
SMA_PERIOD  = 9
MAX_BARS_IN = 120

# Per-strategy tuned params (from backtest_hma_optimize.py grid search)
# CROSS_BARS: look-back bars for EMA5 cross SMA9;  0 = "EMA5 > SMA9" (no strict cross)
STRATEGY_PARAMS = {
    "WTADXStrategy": dict(hma_period=100, sl_mult=2.5, tp_rr=2.0, cross_bars=0),
    "UTBotStrategy": dict(hma_period=50,  sl_mult=2.0, tp_rr=1.5, cross_bars=3),
}

# ─── Fetch real data ──────────────────────────────────────────────────────────

def fetch_candles() -> list:
    """
    Tries live REST (Binance → OKX); falls back to realistic synthetic BTC data.
    Synthetic: GBM with regime switching calibrated to BTC 1H characteristics.
    NOTE: synthetic data tests strategy logic correctly but is not real price history.
    """
    since_ms = int(time.time() * 1000) - MONTHS * 31 * 24 * 3600 * 1000
    sym_b    = SYMBOL.replace("/", "")

    # ── Try live sources ─────────────────────────────────────────────────
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
                print(f"  Fetched {len(candles)} candles from {source}")
                return candles
        except Exception:
            pass

    # ── Synthetic BTC 1H data ─────────────────────────────────────────────
    print(f"  ⚠ Network restricted — using SYNTHETIC BTC data (not real history)")
    print(f"  (Run locally for real data: python backtest_hma_filter.py)")
    rng   = np.random.default_rng(42)
    n_bars = MONTHS * 31 * 24 + 50   # ~3,750 bars

    # BTC starts ~60k, target end ~65k (mild bull with corrections)
    start = 60_000.0
    closes = [start]

    # Regime schedule: (start_pct, end_pct, drift_per_h, vol_per_h)
    regimes = [
        (0.00, 0.15, +0.00010, 0.0050),  # bull rally
        (0.15, 0.30, -0.00008, 0.0070),  # correction
        (0.30, 0.50, +0.00005, 0.0040),  # sideways recovery
        (0.50, 0.65, +0.00015, 0.0055),  # strong bull
        (0.65, 0.80, -0.00012, 0.0080),  # sharp correction
        (0.80, 1.00, +0.00010, 0.0045),  # recovery
    ]
    for i in range(1, n_bars):
        frac = i / n_bars
        dr, vl = 0.0, 0.005
        for s, e, d, v in regimes:
            if s <= frac < e:
                dr, vl = d, v
                break
        r = rng.normal(dr, vl)
        closes.append(closes[-1] * (1 + r))

    candles = []
    ts0 = since_ms
    for i, c in enumerate(closes):
        o   = closes[i - 1] if i > 0 else c
        noise = abs(rng.normal(0, c * 0.0012))
        h = max(o, c) + noise
        l = min(o, c) - noise
        vol = float(rng.uniform(200, 2000))
        candles.append(OHLCV(ts0 + i * 3_600_000, round(o, 2), round(h, 2),
                             round(l, 2), round(c, 2), vol))

    print(f"  Generated {len(candles)} synthetic bars  "
          f"(start=${closes[0]:,.0f}  end=${closes[-1]:,.0f})")
    return candles


# ─── Indicator helpers ────────────────────────────────────────────────────────

def hma_n(closes: list, period: int) -> np.ndarray:
    return BaseStrategy.hma(closes, period)

def ema5(closes: list) -> np.ndarray:
    return BaseStrategy.ema(closes, EMA_PERIOD)

def sma9(closes: list) -> np.ndarray:
    return BaseStrategy.sma(closes, SMA_PERIOD)


# ─── Single-strategy backtest ─────────────────────────────────────────────────

def run_strategy(name: str, candles: list,
                 buy_sig: np.ndarray, sell_sig: np.ndarray,
                 atr14: np.ndarray,
                 hma_a: np.ndarray, ema5_a: np.ndarray, sma9_a: np.ndarray,
                 sl_mult: float, tp_rr: float, cross_bars: int):
    """
    Returns list of trade dicts. Entry = close of signal bar.
    cross_bars: 0 = "EMA5 > SMA9" (loose), >0 = strict cross within N bars.
    """
    n       = len(candles)
    closes  = np.array([c.close for c in candles])
    highs   = np.array([c.high  for c in candles])
    lows    = np.array([c.low   for c in candles])
    opens   = np.array([c.open  for c in candles])

    trades   : list = []
    capital   = CAPITAL
    in_pos    = False
    entry_bar = 0
    entry_px  = 0.0
    sl_px     = 0.0
    tp_px     = 0.0
    amount    = 0.0
    pending_hma_exit = False

    valid_hma = np.where(~np.isnan(hma_a))[0]
    warmup    = max(int(valid_hma[0]) + 10 if len(valid_hma) else 50,
                   ATR_WARMUP(atr14) + 10)

    for i in range(int(warmup), n - 1):
        c_close = float(closes[i])
        c_high  = float(highs[i])
        c_low   = float(lows[i])
        hma_v   = float(hma_a[i])

        if any(math.isnan(x) for x in [hma_v, float(ema5_a[i]), float(sma9_a[i])]):
            continue

        # ── If in position ────────────────────────────────────────────
        if in_pos:
            exit_px   = None
            exit_type = None

            # Pending HMA exit: prev close < HMA → exit at next open if also < HMA
            if pending_hma_exit and i < n:
                next_open = float(opens[i])
                if next_open < float(hma_a[i]):
                    exit_px   = next_open
                    exit_type = "hma_break"
                pending_hma_exit = False

            if exit_px is None:
                if c_high >= tp_px:
                    exit_px   = tp_px
                    exit_type = "tp"
                elif c_low <= sl_px:
                    exit_px   = sl_px
                    exit_type = "sl"

            if exit_px is None and bool(sell_sig[i]):
                exit_px   = c_close
                exit_type = "sell_signal"

            if exit_px is None and (i - entry_bar) >= MAX_BARS_IN:
                exit_px   = c_close
                exit_type = "timeout"

            if exit_px is None and c_close < hma_v:
                pending_hma_exit = True

            if exit_px is not None:
                pnl_gross = (exit_px - entry_px) * amount
                fee_exit  = exit_px * amount * FEE_RATE
                pnl_net   = pnl_gross - fee_exit
                capital  += pnl_net
                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar":  i,
                    "entry":     round(entry_px, 2),
                    "exit":      round(exit_px, 2),
                    "sl":        round(sl_px, 2),
                    "tp":        round(tp_px, 2),
                    "amount":    round(amount, 6),
                    "pnl_net":   round(pnl_net, 4),
                    "pnl_pct":   round((exit_px - entry_px) / entry_px * 100, 3),
                    "type":      exit_type,
                    "capital":   round(capital, 2),
                })
                in_pos = False
                pending_hma_exit = False
            continue

        # ── Entry filters ─────────────────────────────────────────────
        if not bool(buy_sig[i]):
            continue

        # Filter 1: previous candle close > HMA
        prev_close = float(closes[i - 1])
        if prev_close <= float(hma_a[i - 1]):
            continue

        # Filter 2: EMA5 cross / above SMA9
        if cross_bars == 0:
            if float(ema5_a[i]) <= float(sma9_a[i]):
                continue
        else:
            crossed = False
            for k in range(max(1, i - cross_bars + 1), i + 1):
                if (float(ema5_a[k - 1]) <= float(sma9_a[k - 1]) and
                        float(ema5_a[k]) > float(sma9_a[k])):
                    crossed = True
                    break
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


def ATR_WARMUP(atr14: np.ndarray) -> int:
    valid = np.where(~np.isnan(atr14))[0]
    return int(valid[0]) if len(valid) else 30


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(name: str, trades: list, n_bars: int, params: dict):
    W = 70
    if not trades:
        print(f"\n  [{name}] No trades fired.\n")
        return

    wins     = [t for t in trades if t["pnl_net"] > 0]
    losses   = [t for t in trades if t["pnl_net"] <= 0]
    total    = len(trades)
    wr       = len(wins) / total * 100
    gross    = sum(t["pnl_net"] for t in trades)
    avg_win  = sum(t["pnl_net"] for t in wins)  / max(len(wins), 1)
    avg_loss = sum(t["pnl_net"] for t in losses)/ max(len(losses), 1)
    pf       = abs(sum(t["pnl_net"] for t in wins) /
                   sum(t["pnl_net"] for t in losses)) if losses else float("inf")

    # drawdown
    cap_series = [CAPITAL] + [t["capital"] for t in trades]
    peak = CAPITAL; max_dd = 0.0
    for c in cap_series:
        peak   = max(peak, c)
        max_dd = max(max_dd, peak - c)

    # exit breakdown
    exit_counts = {}
    for t in trades:
        exit_counts[t["type"]] = exit_counts.get(t["type"], 0) + 1

    hma_p     = params["hma_period"]
    sl_mult   = params["sl_mult"]
    tp_rr     = params["tp_rr"]
    cross_b   = params["cross_bars"]
    cross_desc = (f"EMA{EMA_PERIOD}>SMA{SMA_PERIOD}"
                  if cross_b == 0
                  else f"EMA{EMA_PERIOD} cross SMA{SMA_PERIOD} ({cross_b} bars)")

    print(f"\n{'═'*W}")
    print(f"  {name}  |  {SYMBOL} {TIMEFRAME}  |  {MONTHS}-month  |  "
          f"${TRADE_USD}/trade  |  SL={sl_mult}×ATR  TP={tp_rr}R")
    print(f"  Filters: prev_close > HMA{hma_p}  |  {cross_desc}")
    print(f"{'─'*W}")
    print(f"  Total trades : {total}")
    print(f"  Win / Loss   : {len(wins)}W / {len(losses)}L  →  WR {wr:.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Net P/L      : ${gross:+.2f}")
    print(f"  Avg win      : ${avg_win:+.2f}   Avg loss: ${avg_loss:+.2f}")
    print(f"  Max drawdown : ${max_dd:.2f}")
    final_cap = trades[-1]["capital"] if trades else CAPITAL
    roi = (final_cap - CAPITAL) / CAPITAL * 100
    print(f"  Capital      : ${CAPITAL:.0f} → ${final_cap:.2f}  ({roi:+.2f}%)")
    print(f"\n  Exit breakdown:")
    for k, v in sorted(exit_counts.items(), key=lambda x: -x[1]):
        pct = v / total * 100
        print(f"    {k:<16}: {v:>3}  ({pct:.0f}%)")

    print(f"\n  {'#':<4} {'Entry':>10} {'Exit':>10} {'SL':>10} {'TP':>10} "
          f"{'PnL $':>8} {'PnL%':>7} {'Type':<14} {'Balance':>9}")
    print(f"  {'─'*W}")
    for idx, t in enumerate(trades[-30:], 1):
        arrow = "✓" if t["pnl_net"] > 0 else "✗"
        print(f"  {idx:<4} {t['entry']:>10,.2f} {t['exit']:>10,.2f} "
              f"{t['sl']:>10,.2f} {t['tp']:>10,.2f} "
              f"{t['pnl_net']:>+8.2f} {t['pnl_pct']:>+6.2f}% "
              f"{t['type']:<14} {t['capital']:>9,.2f} {arrow}")
    if len(trades) > 30:
        print(f"  ... ({len(trades)-30} earlier trades not shown)")
    print(f"{'═'*W}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*70)
    print(f"  HMA Filter Backtest  |  WTADXStrategy + UTBotStrategy")
    print(f"  {SYMBOL} {TIMEFRAME}  |  last {MONTHS} months  |  "
          f"${CAPITAL:.0f} capital  |  ${TRADE_USD}/trade")
    print(f"  Per-strategy params from grid search (backtest_hma_optimize.py)")
    for sname, p in STRATEGY_PARAMS.items():
        cl = "EMA>SMA" if p["cross_bars"] == 0 else f"cross{p['cross_bars']}"
        print(f"    {sname}: HMA={p['hma_period']} SL={p['sl_mult']}×ATR "
              f"TP={p['tp_rr']}R  CROSS={cl}")
    print("═"*70)

    candles = fetch_candles()
    if len(candles) < 300:
        print("ERROR: Not enough candles fetched.")
        sys.exit(1)

    n      = len(candles)
    closes = [c.close for c in candles]

    print(f"\n  Computing indicators on {n} bars...")
    ema5_a = ema5(closes)
    sma9_a = sma9(closes)
    atr14  = BaseStrategy.atr(candles, 14)

    # Pre-compute one HMA per unique period needed
    hma_cache = {}
    for p in STRATEGY_PARAMS.values():
        hp = p["hma_period"]
        if hp not in hma_cache:
            hma_cache[hp] = hma_n(closes, hp)

    # ── WTADXStrategy signals ──────────────────────────────────────────
    wt_strat = WTADXStrategy(SYMBOL)
    wt1, wt2, wt_buy, wt_sell, wt_atr = wt_strat._build_signals(candles)

    # ── UTBotStrategy signals ──────────────────────────────────────────
    ut_strat = UTBotStrategy(SYMBOL)
    ut_buy, ut_sell, ut_tsl, ut_atr = ut_strat._build_signals(candles)

    # ── Run backtests with per-strategy params ─────────────────────────
    wt_p = STRATEGY_PARAMS["WTADXStrategy"]
    ut_p = STRATEGY_PARAMS["UTBotStrategy"]

    print("\n  Running WTADXStrategy backtest...")
    wt_trades = run_strategy(
        "WTADXStrategy", candles,
        wt_buy.astype(bool), wt_sell.astype(bool), wt_atr,
        hma_cache[wt_p["hma_period"]], ema5_a, sma9_a,
        wt_p["sl_mult"], wt_p["tp_rr"], wt_p["cross_bars"],
    )

    print("  Running UTBotStrategy backtest...")
    ut_trades = run_strategy(
        "UTBotStrategy", candles,
        ut_buy.astype(bool), ut_sell.astype(bool), ut_atr,
        hma_cache[ut_p["hma_period"]], ema5_a, sma9_a,
        ut_p["sl_mult"], ut_p["tp_rr"], ut_p["cross_bars"],
    )

    # ── Reports ────────────────────────────────────────────────────────
    print_report("WTADXStrategy", wt_trades, n, wt_p)
    print_report("UTBotStrategy", ut_trades, n, ut_p)

    # ── Combined summary ───────────────────────────────────────────────
    all_trades = sorted(wt_trades + ut_trades, key=lambda t: t["entry_bar"])
    if all_trades:
        print(f"\n{'═'*70}")
        print(f"  COMBINED SUMMARY  (both strategies, independent, own params)")
        print(f"{'─'*70}")
        total = len(all_trades)
        wins  = sum(1 for t in all_trades if t["pnl_net"] > 0)
        net   = sum(t["pnl_net"] for t in all_trades)
        wr    = wins / total * 100 if total else 0
        print(f"  Total trades: {total}  |  WR: {wr:.1f}%  |  Net P/L: ${net:+.2f}")
        print(f"  (WTADXStrategy: {len(wt_trades)} trades  |  "
              f"UTBotStrategy: {len(ut_trades)} trades)")
        print(f"\n  NOTE: Results on SYNTHETIC data — network-restricted environment.")
        print(f"  Run locally with real BTC data for accurate results.")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
