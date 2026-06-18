"""
Backtest: 3 Swing Reversal Strategy Variants — 1 position each

  S1  SwingReversal  RSI+MACD+BB+Vol 3/4  4H-proxy not-down  TP=6%  SL=3%  hold≤5d
  S2  CPKRegime      RSI+MACD+BB+Vol 3/4  price>EMA200       TP=8%  SL=4%  hold≤7d
  S3  HybridSwing    RSI+MACD+BB+Vol 2/4  only block crash    TP=4%  SL=2%  hold≤3d

All 3 run independently on the same synthetic BTC data (GARCH regime-switching).
Each strategy allows exactly 1 open position at a time.

Source: swing_reversal_bot.py (OKX Spot) — backtest adaptation
"""

import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from trading.connectors.base import OHLCV
from trading.strategies.base import BaseStrategy

# ─── Config ────────────────────────────────────────────────────────────────────
SYMBOL    = "BTC/USDT"
MONTHS    = 5
CAPITAL   = 260.0   # total starting capital
TRADE_USD = 100.0   # fixed notional per trade
FEE_RATE  = 0.001   # 0.1% taker fee each side

WARMUP_BARS = 40    # skip first N bars while indicators warm up

STRATEGIES = {
    "SwingReversal": dict(
        tp_pct=0.06, sl_pct=0.03, max_hold_days=5,
        min_conds=3, rsi_os=35.0, vol_mult=1.3,
        regime="ema80_200",
    ),
    "CPKRegime": dict(
        tp_pct=0.08, sl_pct=0.04, max_hold_days=7,
        min_conds=3, rsi_os=35.0, vol_mult=1.3,
        regime="price_ema200",
    ),
    "HybridSwing": dict(
        tp_pct=0.04, sl_pct=0.02, max_hold_days=3,
        min_conds=2, rsi_os=40.0, vol_mult=1.2,
        regime="soft",
    ),
}


# ─── Synthetic data (same GARCH generator as backtest_claude_gate.py) ──────────

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

    print("  ⚠ Network restricted — using BTC-realistic synthetic data")
    print("  (GARCH + regime-switching; stationary coeff=0.02, df=6-8)")
    rng    = np.random.default_rng(42)
    n_bars = MONTHS * 31 * 24 + 50

    regimes = [
        (0.00, 0.15, +0.00012, 0.0020, 0.93, 8),
        (0.15, 0.30, +0.00022, 0.0030, 0.91, 7),
        (0.30, 0.40, +0.00030, 0.0040, 0.89, 6),
        (0.40, 0.48, +0.00003, 0.0050, 0.86, 6),
        (0.48, 0.58, -0.00015, 0.0060, 0.88, 6),
        (0.58, 0.66, +0.00002, 0.0035, 0.90, 7),
        (0.66, 0.76, +0.00008, 0.0022, 0.93, 8),
        (0.76, 0.90, +0.00025, 0.0032, 0.90, 7),
        (0.90, 1.00, +0.00010, 0.0045, 0.87, 6),
    ]

    closes = [32_000.0]
    vol    = 0.0022

    for i in range(1, n_bars):
        frac = i / n_bars
        dr, base_v, persist, df = 0.0, 0.003, 0.90, 7
        for rs, re, d, bv, p, tdf in regimes:
            if rs <= frac < re:
                dr, base_v, persist, df = d, bv, p, tdf; break
        shock = float(rng.standard_t(df)) * vol
        vol   = float(np.sqrt(persist * vol**2 + (1 - persist) * base_v**2 +
                               0.02 * shock**2))
        vol   = max(0.0010, min(vol, 0.015))
        closes.append(max(closes[-1] * (1 + dr + shock), 1000.0))

    candles = []
    ts0 = since_ms
    for i, c in enumerate(closes):
        o        = closes[i - 1] if i > 0 else c
        atr_n    = abs(float(rng.normal(0, c * 0.0018)))
        vol_mult = float(rng.lognormal(6.5, 0.6))
        candles.append(OHLCV(
            ts0 + i * 3_600_000,
            round(o, 2),
            round(max(o, c) + atr_n, 2),
            round(min(o, c) - atr_n, 2),
            round(c, 2),
            round(vol_mult, 0),
        ))

    peak    = max(closes)
    trough  = min(closes[int(n_bars * 0.38): int(n_bars * 0.72)])
    dd      = (peak - trough) / peak * 100
    print(f"  Generated {len(candles)} bars  "
          f"start=${closes[0]:,.0f}  peak=${peak:,.0f}  "
          f"trough=${trough:,.0f}  drawdown={dd:.0f}%  end=${closes[-1]:,.0f}")
    return candles


# ─── Entry condition checker ───────────────────────────────────────────────────

def check_entry(
    i: int,
    closes: np.ndarray,
    volumes: np.ndarray,
    rsi: np.ndarray,
    macd_hist: np.ndarray,
    bb_lower: np.ndarray,
    vol_ma: np.ndarray,
    ema80: np.ndarray,
    ema200: np.ndarray,
    params: dict,
) -> bool:
    if i < 2:
        return False

    def _nan(*vs): return any(math.isnan(v) for v in vs)

    # ── Trend / Regime filter ──────────────────────────────────────────────
    regime = params["regime"]

    if regime == "price_ema200":
        # S2: must be in bullish regime (price above EMA200)
        if not _nan(ema200[i]):
            if closes[i] < ema200[i]:
                return False
        if not _nan(ema80[i], ema200[i]):
            if ema80[i] < ema200[i] and ema80[i] < ema80[i - 1]:
                return False

    elif regime == "ema80_200":
        # S1: block only if 4H proxy is clearly down
        if not _nan(ema80[i], ema200[i], ema80[i - 1]):
            if ema80[i] < ema200[i] and ema80[i] < ema80[i - 1]:
                return False

    elif regime == "soft":
        # S3: only block if clearly crashing (EMA80 > 3% below EMA200)
        if not _nan(ema80[i], ema200[i]):
            if ema80[i] < ema200[i] * 0.97:
                return False

    # ── 4 Entry conditions ─────────────────────────────────────────────────
    conds = 0
    rsi_os = params["rsi_os"]

    # 1. RSI: was in oversold zone, now bouncing up
    if not _nan(rsi[i], rsi[i - 1]):
        if rsi[i - 1] <= rsi_os and rsi[i] > rsi[i - 1]:
            conds += 1

    # 2. MACD histogram improving (momentum turning)
    if not _nan(macd_hist[i], macd_hist[i - 1]):
        if macd_hist[i] > macd_hist[i - 1]:
            conds += 1

    # 3. Bollinger Band lower bounce: prev candle breached lower band, now back inside
    if not _nan(bb_lower[i], bb_lower[i - 1]):
        if closes[i - 1] <= bb_lower[i - 1] and closes[i] > bb_lower[i]:
            conds += 1

    # 4. Volume spike confirms momentum
    if not _nan(vol_ma[i]):
        if volumes[i] >= vol_ma[i] * params["vol_mult"]:
            conds += 1

    return conds >= params["min_conds"]


# ─── Backtest runner (1 position at a time per strategy) ───────────────────────

def run_strategy(
    name: str,
    candles: list,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    rsi: np.ndarray,
    macd_hist: np.ndarray,
    bb_lower: np.ndarray,
    vol_ma: np.ndarray,
    ema80: np.ndarray,
    ema200: np.ndarray,
    params: dict,
) -> list:
    """Run a single strategy variant. Returns list of closed trade dicts."""
    max_hold_bars = int(params["max_hold_days"] * 24)
    tp_pct        = params["tp_pct"]
    sl_pct        = params["sl_pct"]

    trades: list = []
    capital  = CAPITAL
    in_pos   = False
    entry_px = sl_px = tp_px = 0.0
    entry_bar = 0
    n = len(candles)

    for i in range(WARMUP_BARS, n - 1):
        c_close  = float(closes[i])
        c_high   = float(highs[i])
        c_low    = float(lows[i])

        # ── Exit logic ─────────────────────────────────────────────────────
        if in_pos:
            bars_held = i - entry_bar
            exit_px = exit_type = None

            # Check SL first (use low of bar)
            if c_low <= sl_px:
                exit_px, exit_type = sl_px, "sl"
            # TP (use high of bar)
            elif c_high >= tp_px:
                exit_px, exit_type = tp_px, "tp"
            # Time stop
            elif bars_held >= max_hold_bars:
                exit_px, exit_type = c_close, "timeout"

            if exit_px is not None:
                qty  = TRADE_USD / entry_px
                pnl  = (exit_px - entry_px) * qty - exit_px * qty * FEE_RATE
                capital += pnl
                trades.append({
                    "bar": i,
                    "entry": round(entry_px, 2),
                    "exit":  round(exit_px, 2),
                    "sl":    round(sl_px, 2),
                    "tp":    round(tp_px, 2),
                    "type":  exit_type,
                    "bars":  bars_held,
                    "pnl_net": round(pnl, 4),
                    "pnl_pct": round((exit_px - entry_px) / entry_px * 100, 3),
                    "capital": round(capital, 2),
                })
                in_pos = False

        # ── Entry logic ─────────────────────────────────────────────────────
        if not in_pos:
            if check_entry(i, closes, volumes, rsi, macd_hist,
                           bb_lower, vol_ma, ema80, ema200, params):
                entry_px = c_close
                sl_px    = entry_px * (1 - sl_pct)
                tp_px    = entry_px * (1 + tp_pct)
                qty      = TRADE_USD / entry_px
                capital -= entry_px * qty * FEE_RATE
                entry_bar = i
                in_pos    = True

    # Close any open position at end of data
    if in_pos:
        c = float(closes[-1])
        qty = TRADE_USD / entry_px
        pnl = (c - entry_px) * qty - c * qty * FEE_RATE
        capital += pnl
        trades.append({
            "bar": n - 1, "entry": round(entry_px, 2), "exit": round(c, 2),
            "sl": round(sl_px, 2), "tp": round(tp_px, 2),
            "type": "end", "bars": n - 1 - entry_bar,
            "pnl_net": round(pnl, 4),
            "pnl_pct": round((c - entry_px) / entry_px * 100, 3),
            "capital": round(capital, 2),
        })

    return trades


# ─── Stats & display ───────────────────────────────────────────────────────────

def _stats(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "net": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "pf": 0.0,
                "final": CAPITAL, "avg_bars": 0.0}
    wins   = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] <= 0]
    net    = sum(t["pnl_net"] for t in trades)
    aw     = sum(t["pnl_net"] for t in wins)   / max(len(wins),   1)
    al     = sum(t["pnl_net"] for t in losses) / max(len(losses), 1)
    gross_w = abs(sum(t["pnl_net"] for t in wins))
    gross_l = abs(sum(t["pnl_net"] for t in losses))
    pf     = gross_w / max(gross_l, 1e-9)
    avg_bars = sum(t["bars"] for t in trades) / len(trades)
    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "wr": len(wins) / len(trades) * 100,
        "net": net, "avg_win": aw, "avg_loss": al, "pf": pf,
        "final": trades[-1]["capital"],
        "avg_bars": avg_bars,
    }


def print_results(results: dict[str, tuple]):
    W = 78
    print(f"\n{'═'*W}")
    print(f"  SWING REVERSAL BACKTEST  |  {SYMBOL} 1H  |  {MONTHS}-month synthetic data")
    print(f"  Capital: ${CAPITAL:.0f}  |  Trade size: ${TRADE_USD:.0f}/trade  |  Fee: {FEE_RATE*100:.1f}%")
    print(f"{'─'*W}")

    params_info = {
        "SwingReversal": "3/4 conds | RSI<35 | 4H-not-down | TP6% SL3% hold≤5d",
        "CPKRegime":     "3/4 conds | RSI<35 | price>EMA200 | TP8% SL4% hold≤7d",
        "HybridSwing":   "2/4 conds | RSI<40 | soft-filter  | TP4% SL2% hold≤3d",
    }

    header = f"  {'Metric':<24}"
    for name in results:
        header += f"  {name:>16}"
    print(header)
    print(f"  {'─'*22}" + "  " + "  ".join(["─" * 16] * len(results)))

    def row(label, fn):
        line = f"  {label:<24}"
        for name, trades in results.items():
            s = _stats(trades)
            line += f"  {fn(s):>16}"
        print(line)

    row("Trades",          lambda s: str(s["trades"]))
    row("Win / Loss",      lambda s: f"{s['wins']}W / {s['losses']}L")
    row("Win Rate",        lambda s: f"{s['wr']:.1f}%")
    row("Profit Factor",   lambda s: f"{s['pf']:.2f}")
    row("Avg Win $",       lambda s: f"+${s['avg_win']:.2f}")
    row("Avg Loss $",      lambda s: f"${s['avg_loss']:.2f}")
    row("Net P/L",         lambda s: f"${s['net']:+.2f}")
    row("ROI",             lambda s: f"{(s['final'] - CAPITAL) / CAPITAL * 100:+.1f}%")
    row("Final Capital",   lambda s: f"${s['final']:.2f}")
    row("Avg Hold (bars)", lambda s: f"{s['avg_bars']:.0f}h")

    print(f"\n  Exit breakdown:")
    for name, trades in results.items():
        if not trades:
            continue
        ec: dict = {}
        for t in trades:
            ec[t["type"]] = ec.get(t["type"], 0) + 1
        breakdown = "  |  ".join(f"{k}:{v}" for k, v in sorted(ec.items(), key=lambda x: -x[1]))
        print(f"    {name:<16}: {breakdown}")

    print(f"\n  Strategy params:")
    for name, info in params_info.items():
        print(f"    {name:<16}: {info}")

    print(f"{'═'*W}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    W = 78
    print(f"\n{'═'*W}")
    print(f"  Backtest: SwingReversal × 3 variants  |  {SYMBOL} 1H  |  {MONTHS} months")
    print(f"{'═'*W}")

    candles = fetch_candles()
    if len(candles) < 300:
        print("ERROR: Not enough candles"); return

    n       = len(candles)
    closes  = np.array([c.close  for c in candles])
    highs   = np.array([c.high   for c in candles])
    lows    = np.array([c.low    for c in candles])
    volumes = np.array([c.volume for c in candles])

    print(f"\n  Computing indicators on {n} bars...")

    # Core indicators
    rsi14 = BaseStrategy.rsi(closes.tolist(), 14)
    _, _, macd_hist = BaseStrategy.macd(closes.tolist(), 12, 26, 9)
    _, _, bb_lower  = BaseStrategy.bollinger_bands(closes.tolist(), 20, 2.0)
    vol_ma = BaseStrategy.sma(volumes.tolist(), 20)

    # Trend / regime proxies on 1H
    # EMA(80) ≈ EMA(20) on 4H   (4H EMA20 = 80 × 1H bars)
    # EMA(200) ≈ EMA(50) on 4H  (4H EMA50 = 200 × 1H bars)
    ema80  = BaseStrategy.ema(closes.tolist(), 80)
    ema200 = BaseStrategy.ema(closes.tolist(), 200)

    print(f"  Running 3 strategies...\n")

    results: dict[str, list] = {}
    for name, params in STRATEGIES.items():
        trades = run_strategy(
            name, candles, closes, highs, lows, volumes,
            rsi14, macd_hist, bb_lower, vol_ma,
            ema80, ema200, params,
        )
        results[name] = trades
        s = _stats(trades)
        print(f"  [{name}]  trades={s['trades']}  WR={s['wr']:.0f}%  "
              f"Net=${s['net']:+.2f}  PF={s['pf']:.2f}")

    print_results(results)


if __name__ == "__main__":
    main()
