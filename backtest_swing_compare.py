"""
Head-to-head comparison: Swing v5 (3 strategies) vs WaveTrend & UT Bot.

Same synthetic BTC data  ·  Same $100 trade size  ·  Same 0.1% fee
All strategies: LONG only (BUY signals), ATR-based SL/TP, max 1 position.
"""

import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from trading.connectors.base import OHLCV
from trading.strategies.base import BaseStrategy
from trading.strategies.wt_adx_strategy import WTADXStrategy
from trading.strategies.ut_bot_strategy import UTBotStrategy

# ─── Config ────────────────────────────────────────────────────────────────────
SYMBOL    = "BTC/USDT"
MONTHS    = 5
CAPITAL   = 260.0
TRADE_USD = 100.0
FEE_RATE  = 0.001
WARMUP    = 50
COOLDOWN_BARS = 3   # bars to skip after SL hit (swing strategies only)

# Swing v5 strategy parameters (same as backtest_swing.py)
SWING_STRATEGIES = {
    # Grid-search best: SL=2.5 TP=1.5 → WR=81.2%, Net=$+6.79 (+2.6%)
    "SwingReversal": dict(
        sl_atr=2.5, tp_atr=1.5, max_hold_days=3,
        rsi_lo=40.0, rsi_hi=56.0, vol_mult=1.4,
        regime="soft_bull", entry_type="momentum",
    ),
    # Grid-search best: SL=1.5 TP=4.0 → WR=47.1%, Net=$+7.86 (+3.0%)
    "CPKRegime": dict(
        sl_atr=1.5, tp_atr=4.0, max_hold_days=4,
        rsi_lo=46.0, rsi_hi=58.0, vol_mult=1.4,
        regime="soft_bull", entry_type="momentum",
    ),
    # Grid-search best: SL=2.5 TP=4.0 → WR=55.6%, Net=$+9.46 (+3.6%)
    "HybridSwing": dict(
        sl_atr=2.5, tp_atr=4.0, max_hold_days=2,
        rsi_lo=44.0, rsi_hi=58.0, vol_mult=1.4,
        regime="soft_bull", entry_type="momentum",
    ),
}

# Legacy strategy parameters (from their class defaults)
LEGACY_CONFIGS = {
    "WaveTrend": dict(sl_atr=2.5, tp_atr=3.0, max_hold_days=5),
    "UT_Bot":    dict(sl_atr=2.5, tp_atr=3.0, max_hold_days=5),
}


# ─── Synthetic data (identical to backtest_swing.py) ────────────────────────────

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
            round(o, 2), round(max(o, c) + atr_n, 2),
            round(min(o, c) - atr_n, 2), round(c, 2), round(vol_mult, 0),
        ))
    peak   = max(closes)
    trough = min(closes[int(n_bars * 0.38): int(n_bars * 0.72)])
    print(f"  Generated {len(candles)} bars  start=${closes[0]:,.0f}  "
          f"peak=${peak:,.0f}  trough=${trough:,.0f}  "
          f"dd={(peak-trough)/peak*100:.0f}%  end=${closes[-1]:,.0f}")
    return candles


# ─── Swing helpers (same as backtest_swing.py) ──────────────────────────────────

def _nan(*vs) -> bool:
    return any(math.isnan(float(v)) for v in vs)


def candle_bullish(closes, highs, lows, i, pct=0.60) -> bool:
    rng = float(highs[i]) - float(lows[i])
    if rng < 1e-6:
        return False
    return (float(closes[i]) - float(lows[i])) / rng >= pct


def macd_cross_up(macd_hist, i, lookback=3) -> bool:
    for j in range(i, max(i - lookback, 0), -1):
        if j < 1 or _nan(macd_hist[j], macd_hist[j - 1]):
            continue
        if float(macd_hist[j - 1]) < 0 and float(macd_hist[j]) > 0:
            return True
    return False


def check_swing_entry(i, closes, highs, lows, volumes,
                      rsi, macd_hist, vol_ma, ema80, ema200, params) -> bool:
    if i < 8:
        return False
    if _nan(ema80[i], ema200[i]):
        return False
    e80, e200 = float(ema80[i]), float(ema200[i])
    if e80 < e200 * 0.98:
        return False
    if not candle_bullish(closes, highs, lows, i, 0.60):
        return False
    c_rsi  = float(rsi[i])  if not _nan(rsi[i])  else 50.0
    c_vol  = float(volumes[i])
    ma_vol = float(vol_ma[i]) if not _nan(vol_ma[i]) else 1.0
    cond_macd = macd_cross_up(macd_hist, i, lookback=3)
    cond_rsi  = params["rsi_lo"] <= c_rsi <= params["rsi_hi"]
    cond_vol  = c_vol >= ma_vol * params["vol_mult"]
    return cond_macd and cond_rsi and cond_vol


# ─── Execution engines ──────────────────────────────────────────────────────────

def run_swing(name, candles, closes, highs, lows, volumes,
              rsi, macd_hist, vol_ma, ema80, ema200, atr14, params) -> list:
    max_hold_bars = int(params["max_hold_days"] * 24)
    trades: list  = []
    capital       = CAPITAL
    in_pos        = False
    entry_px = sl_px = tp_px = 0.0
    entry_bar = cooldown = 0
    n = len(candles)

    for i in range(WARMUP, n - 1):
        c_close = float(closes[i])
        c_high  = float(highs[i])
        c_low   = float(lows[i])

        if in_pos:
            bars_held = i - entry_bar
            exit_px = exit_type = None
            if c_low   <= sl_px:             exit_px, exit_type = sl_px,   "sl"
            elif c_high >= tp_px:             exit_px, exit_type = tp_px,   "tp"
            elif bars_held >= max_hold_bars:  exit_px, exit_type = c_close, "timeout"
            if exit_px is not None:
                qty = TRADE_USD / entry_px
                pnl = (exit_px - entry_px) * qty - exit_px * qty * FEE_RATE
                capital += pnl
                trades.append({"bar": i, "type": exit_type, "bars": bars_held,
                                "pnl_net": round(pnl, 4),
                                "pnl_pct": round((exit_px - entry_px) / entry_px * 100, 3),
                                "capital": round(capital, 2)})
                in_pos = False
                if exit_type == "sl":
                    cooldown = COOLDOWN_BARS

        if cooldown > 0:
            cooldown -= 1; continue

        if not in_pos and check_swing_entry(
                i, closes, highs, lows, volumes,
                rsi, macd_hist, vol_ma, ema80, ema200, params):
            atr_v = float(atr14[i]) if not _nan(atr14[i]) else c_close * 0.015
            entry_px = c_close
            sl_px    = entry_px - params["sl_atr"] * atr_v
            tp_px    = entry_px + params["tp_atr"] * atr_v
            capital -= (TRADE_USD / entry_px) * entry_px * FEE_RATE
            entry_bar = i; in_pos = True

    if in_pos:
        c = float(closes[-1])
        qty = TRADE_USD / entry_px
        pnl = (c - entry_px) * qty - c * qty * FEE_RATE
        capital += pnl
        trades.append({"bar": n - 1, "type": "end", "bars": n - 1 - entry_bar,
                        "pnl_net": round(pnl, 4),
                        "pnl_pct": round((c - entry_px) / entry_px * 100, 3),
                        "capital": round(capital, 2)})
    return trades


def run_legacy(buy_sig: np.ndarray, closes, highs, lows, atr14, cfg) -> list:
    """Run WaveTrend or UT Bot BUY signals with ATR SL/TP."""
    max_hold_bars = int(cfg["max_hold_days"] * 24)
    trades: list  = []
    capital       = CAPITAL
    in_pos        = False
    entry_px = sl_px = tp_px = 0.0
    entry_bar = 0
    n = len(closes)

    for i in range(WARMUP, n - 1):
        c_close = float(closes[i])
        c_high  = float(highs[i])
        c_low   = float(lows[i])

        if in_pos:
            bars_held = i - entry_bar
            exit_px = exit_type = None
            if c_low   <= sl_px:             exit_px, exit_type = sl_px,   "sl"
            elif c_high >= tp_px:             exit_px, exit_type = tp_px,   "tp"
            elif bars_held >= max_hold_bars:  exit_px, exit_type = c_close, "timeout"
            if exit_px is not None:
                qty = TRADE_USD / entry_px
                pnl = (exit_px - entry_px) * qty - exit_px * qty * FEE_RATE
                capital += pnl
                trades.append({"bar": i, "type": exit_type, "bars": bars_held,
                                "pnl_net": round(pnl, 4),
                                "pnl_pct": round((exit_px - entry_px) / entry_px * 100, 3),
                                "capital": round(capital, 2)})
                in_pos = False

        if not in_pos and i < len(buy_sig) and bool(buy_sig[i]):
            atr_v = float(atr14[i]) if not (math.isnan(float(atr14[i]))) else c_close * 0.015
            entry_px = c_close
            sl_px    = entry_px - cfg["sl_atr"] * atr_v
            tp_px    = entry_px + cfg["tp_atr"] * atr_v
            capital -= (TRADE_USD / entry_px) * entry_px * FEE_RATE
            entry_bar = i; in_pos = True

    if in_pos:
        c = float(closes[-1])
        qty = TRADE_USD / entry_px
        pnl = (c - entry_px) * qty - c * qty * FEE_RATE
        capital += pnl
        trades.append({"bar": n - 1, "type": "end", "bars": n - 1 - entry_bar,
                        "pnl_net": round(pnl, 4),
                        "pnl_pct": round((c - entry_px) / entry_px * 100, 3),
                        "capital": round(capital, 2)})
    return trades


# ─── Stats ──────────────────────────────────────────────────────────────────────

def _stats(trades) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "net": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "pf": 0.0,
                "final": CAPITAL, "avg_bars": 0.0}
    wins   = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] <= 0]
    net    = sum(t["pnl_net"] for t in trades)
    aw     = sum(t["pnl_net"] for t in wins)   / max(len(wins),   1)
    al     = sum(t["pnl_net"] for t in losses) / max(len(losses), 1)
    gw     = abs(sum(t["pnl_net"] for t in wins))
    gl     = abs(sum(t["pnl_net"] for t in losses))
    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "wr":    len(wins) / len(trades) * 100,
        "net":   net, "avg_win": aw, "avg_loss": al,
        "pf":    gw / max(gl, 1e-9),
        "final": trades[-1]["capital"],
        "avg_bars": sum(t["bars"] for t in trades) / len(trades),
    }


# ─── Print ──────────────────────────────────────────────────────────────────────

def print_comparison(results: dict):
    W = 100
    names = list(results.keys())
    sep = "─" * 20

    print(f"\n{'═'*W}")
    print(f"  STRATEGY COMPARISON  |  {SYMBOL} 1H  |  {MONTHS}-month synthetic BTC")
    print(f"  Capital ${CAPITAL:.0f}  |  Trade ${TRADE_USD:.0f}  |  Fee {FEE_RATE*100:.1f}%  |  LONG only")
    print(f"  {'─'*20}  Legacy (existing)  {'─'*16}  Swing v5 (new)  {'─'*18}")
    print(f"{'─'*W}")

    def col(n):
        return f"  {n:>16}"

    hdr = f"  {'Metric':<22}" + "".join(col(n) for n in names)
    print(hdr)
    print(f"  {sep}" + "  " + "  ".join(["─"*16]*len(names)))

    def row(label, fn):
        vals = "".join(f"  {fn(_stats(results[n])):>16}" for n in names)
        print(f"  {label:<22}{vals}")

    row("Trades",        lambda s: str(s["trades"]))
    row("Win / Loss",    lambda s: f"{s['wins']}W / {s['losses']}L")
    row("Win Rate ★",   lambda s: f"{s['wr']:.1f}%  {'✓' if s['wr'] >= 62 else '·'}")
    row("Profit Factor", lambda s: f"{s['pf']:.2f}")
    row("Avg Win $",     lambda s: f"+${s['avg_win']:.2f}")
    row("Avg Loss $",    lambda s: f"${s['avg_loss']:.2f}")
    row("Net P/L",       lambda s: f"${s['net']:+.2f}")
    row("ROI (5 mo)",    lambda s: f"{(s['final'] - CAPITAL) / CAPITAL * 100:+.1f}%")
    row("Avg Hold (h)",  lambda s: f"{s['avg_bars']:.0f}h")

    print(f"\n  Exit breakdown:")
    for name, trades in results.items():
        if not trades:
            print(f"    {name:<16}: — no trades —"); continue
        ec: dict = {}
        for t in trades:
            ec[t["type"]] = ec.get(t["type"], 0) + 1
        bd = "  |  ".join(f"{k}:{v}" for k, v in sorted(ec.items(), key=lambda x: -x[1]))
        print(f"    {name:<18}: {bd}")

    print(f"\n  SL / TP config:")
    cfg_info = {
        "WaveTrend":     "SL=2.5×ATR  TP=3.0×ATR  (R:R 1:1.2)  wt1 cross wt2 on HA",
        "UT_Bot":        "SL=2.5×ATR  TP=3.0×ATR  (R:R 1:1.2)  ATR trailing stop cross",
        "SwingReversal": "SL=2.5×ATR  TP=1.5×ATR  (R:R 1:0.6★) MACD-cross + RSI 40-56",
        "CPKRegime":     "SL=1.5×ATR  TP=4.0×ATR  (R:R 1:2.7★) MACD-cross + RSI 46-58",
        "HybridSwing":   "SL=2.5×ATR  TP=4.0×ATR  (R:R 1:1.6★) MACD-cross + RSI 44-58",
    }
    for n in names:
        print(f"    {n:<18}: {cfg_info.get(n, '')}")
    print(f"{'═'*W}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    W = 100
    print(f"\n{'═'*W}")
    print(f"  SWING v5 vs WaveTrend vs UT Bot  |  {SYMBOL} 1H  |  {MONTHS} months")
    print(f"{'═'*W}")

    candles = fetch_candles()
    if len(candles) < 300:
        print("ERROR: not enough candles"); return

    n       = len(candles)
    closes  = np.array([c.close  for c in candles])
    highs   = np.array([c.high   for c in candles])
    lows    = np.array([c.low    for c in candles])
    volumes = np.array([c.volume for c in candles])

    print(f"\n  Computing indicators...")
    rsi14           = BaseStrategy.rsi(closes.tolist(), 14)
    _, _, macd_hist = BaseStrategy.macd(closes.tolist(), 12, 26, 9)
    vol_ma          = BaseStrategy.sma(volumes.tolist(), 20)
    atr14           = BaseStrategy.atr(candles, 14)
    ema80           = BaseStrategy.ema(closes.tolist(), 80)
    ema200          = BaseStrategy.ema(closes.tolist(), 200)

    # ── Legacy signals ────────────────────────────────────────────────────
    print("  Extracting WaveTrend signals...")
    wt_strat = WTADXStrategy(SYMBOL)
    _, _, wt_buy, _, wt_atr = wt_strat._build_signals(candles)

    print("  Extracting UT Bot signals...")
    ut_strat = UTBotStrategy(SYMBOL)
    ut_buy, _, _, ut_atr = ut_strat._build_signals(candles)

    print(f"\n  Running all strategies...\n")
    results: dict[str, list] = {}

    # Legacy
    for name, sig, atr, cfg in [
        ("WaveTrend", wt_buy, wt_atr, LEGACY_CONFIGS["WaveTrend"]),
        ("UT_Bot",    ut_buy, ut_atr, LEGACY_CONFIGS["UT_Bot"]),
    ]:
        trades = run_legacy(sig, closes, highs, lows, atr, cfg)
        results[name] = trades
        s = _stats(trades)
        flag = "✓ ≥62%" if s["wr"] >= 62 else f"({s['wr']:.0f}%)"
        print(f"  [{name:<14}]  trades={s['trades']:>3}  WR={s['wr']:.1f}% {flag:<8}  "
              f"Net=${s['net']:+.2f}  PF={s['pf']:.2f}")

    # Swing v5
    for name, params in SWING_STRATEGIES.items():
        trades = run_swing(
            name, candles, closes, highs, lows, volumes,
            rsi14, macd_hist, vol_ma, ema80, ema200, atr14, params,
        )
        results[name] = trades
        s = _stats(trades)
        flag = "✓ ≥62%" if s["wr"] >= 62 else f"({s['wr']:.0f}%)"
        print(f"  [{name:<14}]  trades={s['trades']:>3}  WR={s['wr']:.1f}% {flag:<8}  "
              f"Net=${s['net']:+.2f}  PF={s['pf']:.2f}")

    print_comparison(results)


if __name__ == "__main__":
    main()
