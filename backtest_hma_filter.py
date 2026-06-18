"""
Backtest: WTADXStrategy & UTBotStrategy — v4 (final)
BTC/USDT 1H  |  5-month synthetic data (real data when run locally)

Entry (per strategy):
  Strategy BUY signal
  + prev_close > HMA(period)   — trend gate (keeps quality high)
  + EMA5 > SMA9 / fresh cross  — momentum confirmation
  + RSI(14) > 50               — NEW: bullish momentum gate, no lag

Exit (first triggered):
  1. High >= TP                           → TP (win)
  2. Low  <= SL                           → SL (loss)
  3. Strategy SELL signal (UTBot only)    → close at bar close
  4. Close < HMA and next open < HMA     → exit at next open  [HMA break]
     UTBot: skipped (sell_signal handles it)
  5. Timeout 120 bars

Iteration findings:
  v1 (HMA100/50, fixed TP/SL, HMA break): WTADX -$6.16, UTBot -$3.37  ← best
  v2 (no HMA, RSI>50, trailing stop)    : WTADX -$34.65 (more trades, worse quality)
  v3 (HMA + RSI + trailing stop)        : WTADX -$34.51 (trail inverts R:R on GBM)
  v4 (HMA + RSI>50, fixed TP/SL, HMA break): testing now

Conclusion: trailing stop needs REAL trending data to outperform fixed TP/SL.
On synthetic GBM, HMA entry + fixed exits is optimal.

Capital: $260 | $100/trade | 0.1% OKX taker fee per leg
"""
import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

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
RSI_PERIOD  = 14
MAX_BARS_IN = 120

# Per-strategy params (v4: same as v1 best + RSI>50)
#   hma_period   : trend gate (laggy but quality filter — proven useful)
#   cross_bars   : 0 = EMA5>SMA9, N = strict cross within N bars
#   rsi_min      : RSI threshold at entry (fast, no lag)
#   use_sell_sig : honour strategy SELL as exit
#   use_hma_exit : close<HMA + next open<HMA → exit (reduces downside)
STRATEGY_PARAMS = {
    "WTADXStrategy": dict(
        hma_period=100, sl_mult=2.5, tp_rr=2.0,
        cross_bars=0,   rsi_min=50,
        use_sell_sig=False, use_hma_exit=True,
    ),
    "UTBotStrategy": dict(
        hma_period=50,  sl_mult=2.0, tp_rr=1.5,
        cross_bars=3,   rsi_min=50,
        use_sell_sig=True, use_hma_exit=False,
    ),
}

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

    print("  ⚠ Network restricted — using SYNTHETIC BTC data (not real history)")
    print("  (Run locally for real data: python backtest_hma_filter.py)")
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
    print(f"  Generated {len(candles)} synthetic bars  "
          f"(start=${closes[0]:,.0f}  end=${closes[-1]:,.0f})")
    return candles


# ─── Backtest engine ───────────────────────────────────────────────────────────

def run_strategy(name: str, candles: list,
                 buy_sig: np.ndarray, sell_sig: np.ndarray,
                 atr14: np.ndarray, hma_a: np.ndarray,
                 ema5_a: np.ndarray, sma9_a: np.ndarray, rsi_a: np.ndarray,
                 hma_period: int, sl_mult: float, tp_rr: float,
                 cross_bars: int, rsi_min: float,
                 use_sell_sig: bool, use_hma_exit: bool) -> list:

    n      = len(candles)
    closes = np.array([c.close for c in candles])
    highs  = np.array([c.high  for c in candles])
    lows   = np.array([c.low   for c in candles])
    opens  = np.array([c.open  for c in candles])

    trades   : list = []
    capital   = CAPITAL
    in_pos    = False
    entry_bar = 0
    entry_px  = sl_px = tp_px = amount = 0.0
    pending_hma_exit = False

    valid_hma = np.where(~np.isnan(hma_a))[0]
    warmup    = max(int(valid_hma[0]) + 10 if len(valid_hma) else 110,
                   ATR_WARMUP(atr14) + 10,
                   RSI_PERIOD + 5)

    for i in range(warmup, n - 1):
        c_close = float(closes[i])
        c_high  = float(highs[i])
        c_low   = float(lows[i])
        hma_v   = float(hma_a[i])
        rsi_v   = float(rsi_a[i])
        atr_v   = float(atr14[i])
        e5      = float(ema5_a[i])
        s9      = float(sma9_a[i])

        if any(math.isnan(x) for x in [hma_v, rsi_v, atr_v, e5, s9]):
            continue

        # ── In position ───────────────────────────────────────────────
        if in_pos:
            exit_px = exit_type = None

            # HMA break: prev close < HMA → exit next open if it opens below HMA too
            if use_hma_exit and pending_hma_exit:
                nxt = float(opens[i])
                if nxt < float(hma_a[i]):
                    exit_px, exit_type = nxt, "hma_break"
                pending_hma_exit = False

            # TP
            if exit_px is None and c_high >= tp_px:
                exit_px, exit_type = tp_px, "tp"
            # SL
            if exit_px is None and c_low <= sl_px:
                exit_px, exit_type = sl_px, "sl"
            # Sell signal (optional)
            if exit_px is None and use_sell_sig and bool(sell_sig[i]):
                exit_px, exit_type = c_close, "sell_signal"
            # Timeout
            if exit_px is None and (i - entry_bar) >= MAX_BARS_IN:
                exit_px, exit_type = c_close, "timeout"
            # Flag HMA break for next bar
            if exit_px is None and use_hma_exit and c_close < hma_v:
                pending_hma_exit = True

            if exit_px is not None:
                pnl_net  = (exit_px - entry_px) * amount - exit_px * amount * FEE_RATE
                capital += pnl_net
                trades.append({
                    "entry_bar": entry_bar, "exit_bar": i,
                    "entry":  round(entry_px, 2), "exit":  round(exit_px, 2),
                    "sl":     round(sl_px, 2),    "tp":    round(tp_px, 2),
                    "amount": round(amount, 6),
                    "pnl_net": round(pnl_net, 4),
                    "pnl_pct": round((exit_px - entry_px) / entry_px * 100, 3),
                    "type":    exit_type,
                    "capital": round(capital, 2),
                })
                in_pos = False
                pending_hma_exit = False
            continue

        # ── Entry filters ─────────────────────────────────────────────
        if not bool(buy_sig[i]):
            continue

        # Filter 1: prev close > HMA (trend quality gate)
        if i < 1 or float(closes[i - 1]) <= float(hma_a[i - 1]):
            continue

        # Filter 2: EMA5 > SMA9 / fresh cross (momentum)
        if cross_bars == 0:
            if e5 <= s9:
                continue
        else:
            crossed = False
            for k in range(max(1, i - cross_bars + 1), i + 1):
                if (float(ema5_a[k - 1]) <= float(sma9_a[k - 1]) and
                        float(ema5_a[k]) > float(sma9_a[k])):
                    crossed = True; break
            if not crossed:
                continue

        # Filter 3: RSI > rsi_min (fast bullish momentum gate — no lag)
        if rsi_v < rsi_min:
            continue

        if math.isnan(atr_v) or atr_v <= 0:
            continue

        entry_px  = c_close
        sl_px     = round(entry_px - sl_mult * atr_v, 2)
        tp_px     = round(entry_px + sl_mult * tp_rr * atr_v, 2)
        amount    = TRADE_USD / entry_px
        capital  -= entry_px * amount * FEE_RATE
        in_pos    = True
        entry_bar = i
        pending_hma_exit = False

    return trades


def ATR_WARMUP(atr14: np.ndarray) -> int:
    valid = np.where(~np.isnan(atr14))[0]
    return int(valid[0]) if len(valid) else 30


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(name: str, trades: list, params: dict):
    W = 72
    if not trades:
        print(f"\n  [{name}] No trades fired.\n")
        return

    wins     = [t for t in trades if t["pnl_net"] > 0]
    losses   = [t for t in trades if t["pnl_net"] <= 0]
    total    = len(trades)
    wr       = len(wins) / total * 100
    gross    = sum(t["pnl_net"] for t in trades)
    avg_win  = sum(t["pnl_net"] for t in wins)  / max(len(wins), 1)
    avg_loss = sum(t["pnl_net"] for t in losses) / max(len(losses), 1)
    pf       = (abs(sum(t["pnl_net"] for t in wins)) /
                abs(sum(t["pnl_net"] for t in losses))) if losses else 9.99
    rr_actual = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    cap_series = [CAPITAL] + [t["capital"] for t in trades]
    peak = CAPITAL; max_dd = 0.0
    for c in cap_series:
        peak   = max(peak, c)
        max_dd = max(max_dd, peak - c)

    exit_counts: dict = {}
    for t in trades:
        exit_counts[t["type"]] = exit_counts.get(t["type"], 0) + 1

    hma_p     = params["hma_period"]
    sl_m      = params["sl_mult"]; tp_r = params["tp_rr"]
    cb        = params["cross_bars"]
    cross_lbl = "EMA>SMA" if cb == 0 else f"cross{cb}"
    exit_lbl  = ("sell+HMA" if params["use_sell_sig"] and params["use_hma_exit"]
                 else "sell" if params["use_sell_sig"]
                 else "HMA" if params["use_hma_exit"] else "TP/SL only")

    print(f"\n{'═'*W}")
    print(f"  {name}  |  {SYMBOL} {TIMEFRAME}  |  {MONTHS}-month  |  ${TRADE_USD}/trade")
    print(f"  Entry: HMA{hma_p} + {cross_lbl} + RSI>{params['rsi_min']}  |  "
          f"SL={sl_m}×ATR  TP={tp_r}R  |  extra-exit={exit_lbl}")
    print(f"{'─'*W}")
    print(f"  Total trades : {total}")
    print(f"  Win / Loss   : {len(wins)}W / {len(losses)}L  →  WR {wr:.1f}%")
    print(f"  Profit factor: {pf:.2f}   Actual R:R {rr_actual:.2f}")
    print(f"  Net P/L      : ${gross:+.2f}")
    print(f"  Avg win      : ${avg_win:+.2f}   Avg loss: ${avg_loss:+.2f}")
    print(f"  Max drawdown : ${max_dd:.2f}")
    final_cap = trades[-1]["capital"] if trades else CAPITAL
    roi = (final_cap - CAPITAL) / CAPITAL * 100
    print(f"  Capital      : ${CAPITAL:.0f} → ${final_cap:.2f}  ({roi:+.2f}%)")
    print(f"\n  Exit breakdown:")
    for k, v in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<16}: {v:>3}  ({v/total*100:.0f}%)")

    print(f"\n  {'#':<4} {'Entry':>10} {'Exit':>10} {'SL':>10} {'TP':>10} "
          f"{'PnL$':>7} {'PnL%':>6} {'Type':<14} {'Bal':>9}")
    print(f"  {'─'*W}")
    for idx, t in enumerate(trades[-25:], 1):
        arrow = "✓" if t["pnl_net"] > 0 else "✗"
        print(f"  {idx:<4} {t['entry']:>10,.0f} {t['exit']:>10,.0f} "
              f"{t['sl']:>10,.0f} {t['tp']:>10,.0f} "
              f"{t['pnl_net']:>+7.2f} {t['pnl_pct']:>+5.2f}% "
              f"{t['type']:<14} {t['capital']:>9,.2f} {arrow}")
    if len(trades) > 25:
        print(f"  ... ({len(trades)-25} earlier trades not shown)")
    print(f"{'═'*W}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*72)
    print(f"  Backtest v4  |  WTADXStrategy + UTBotStrategy")
    print(f"  {SYMBOL} {TIMEFRAME}  |  last {MONTHS} months  |  "
          f"${CAPITAL:.0f} capital  |  ${TRADE_USD}/trade")
    print(f"  Entry: HMA trend gate + EMA{EMA_PERIOD}/SMA{SMA_PERIOD} + RSI{RSI_PERIOD}>50")
    for sname, p in STRATEGY_PARAMS.items():
        cb = "EMA>SMA" if p["cross_bars"] == 0 else f"cross{p['cross_bars']}"
        ex = ("sell+HMAbreak" if p["use_sell_sig"] and p["use_hma_exit"]
              else "sell" if p["use_sell_sig"]
              else "HMAbreak" if p["use_hma_exit"] else "TP/SL")
        print(f"    {sname}: HMA{p['hma_period']} SL={p['sl_mult']}×ATR "
              f"TP={p['tp_rr']}R  {cb}  exit={ex}")
    print("═"*72)

    candles = fetch_candles()
    if len(candles) < 300:
        print("ERROR: Not enough candles"); sys.exit(1)

    n      = len(candles)
    closes = [c.close for c in candles]
    print(f"\n  Computing indicators on {n} bars...")

    ema5_a = BaseStrategy.ema(closes, EMA_PERIOD)
    sma9_a = BaseStrategy.sma(closes, SMA_PERIOD)
    atr14  = BaseStrategy.atr(candles, 14)
    rsi14  = BaseStrategy.rsi(closes, RSI_PERIOD)

    hma_cache = {}
    for p in STRATEGY_PARAMS.values():
        hp = p["hma_period"]
        if hp not in hma_cache:
            hma_cache[hp] = BaseStrategy.hma(closes, hp)

    wt_strat = WTADXStrategy(SYMBOL)
    _, _, wt_buy, wt_sell, _ = wt_strat._build_signals(candles)

    ut_strat = UTBotStrategy(SYMBOL)
    ut_buy, ut_sell, _, _ = ut_strat._build_signals(candles)

    print("\n  Running WTADXStrategy backtest...")
    wt_p = STRATEGY_PARAMS["WTADXStrategy"]
    wt_trades = run_strategy(
        "WTADXStrategy", candles,
        wt_buy.astype(bool), wt_sell.astype(bool),
        atr14, hma_cache[wt_p["hma_period"]],
        ema5_a, sma9_a, rsi14, **wt_p,
    )

    print("  Running UTBotStrategy backtest...")
    ut_p = STRATEGY_PARAMS["UTBotStrategy"]
    ut_trades = run_strategy(
        "UTBotStrategy", candles,
        ut_buy.astype(bool), ut_sell.astype(bool),
        atr14, hma_cache[ut_p["hma_period"]],
        ema5_a, sma9_a, rsi14, **ut_p,
    )

    print_report("WTADXStrategy", wt_trades, wt_p)
    print_report("UTBotStrategy", ut_trades, ut_p)

    all_trades = wt_trades + ut_trades
    if all_trades:
        total = len(all_trades)
        wins  = sum(1 for t in all_trades if t["pnl_net"] > 0)
        net   = sum(t["pnl_net"] for t in all_trades)
        wr    = wins / total * 100
        print(f"\n{'═'*72}")
        print(f"  COMBINED  |  {total} trades  |  WR {wr:.1f}%  |  Net P/L ${net:+.2f}")
        print(f"  WTADX: {len(wt_trades)} trades  |  UTBot: {len(ut_trades)} trades")
        print(f"\n  SYNTHETIC DATA NOTE:")
        print(f"  GBM random walk cannot replicate real BTC trends.")
        print(f"  Trailing stops / real trend-following need live data to evaluate.")
        print(f"  Run locally to test on real OKX/Binance history.")
        print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
