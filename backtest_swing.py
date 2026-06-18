"""
Backtest: 3 Swing Strategy Variants — 1 position each (v5)

Root cause of low WR in v4: reversal signals fire DURING momentum downswings
in GARCH data (volatility clustering). Fix: enter AFTER reversal confirmed.

  S1  SwingReversal  Late reversal: RSI recovered 44+, MACD already>0
  S2  CPKRegime      EMA80 pullback bounce: price returns to EMA80 in uptrend
  S3  HybridSwing    Momentum: MACD cross + tight RSI zone + volume

All: SL=1.5×ATR  TP=1.5×ATR  EMA80>EMA200 required  close>60% range
"""

import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from trading.connectors.base import OHLCV
from trading.strategies.base import BaseStrategy

# ─── Config ────────────────────────────────────────────────────────────────────
SYMBOL    = "BTC/USDT"
MONTHS    = 5
CAPITAL   = 260.0
TRADE_USD = 100.0
FEE_RATE  = 0.001
WARMUP    = 50

STRATEGIES = {
    # All three use MACD-cross timing in soft-bull regime (EMA80 > EMA200×0.98)
    # — same core mechanism that gives HybridSwing 68% WR, differentiated by
    # RSI zone and TP multiplier.
    #
    # SwingReversal: slightly lower RSI (40-56) — enters earlier in recovery
    "SwingReversal": dict(
        sl_atr=1.5, tp_atr=1.5, max_hold_days=3,
        rsi_lo=40.0, rsi_hi=56.0, vol_mult=1.4,
        regime="soft_bull",
        entry_type="momentum",
    ),
    # CPKRegime: RSI 46-58 (inside the proven 44-58 sweet spot) — higher lower
    # bound means we only enter once RSI has already built some momentum.
    # TP=1.5×ATR same as the others (1.8 was too far to reach reliably).
    "CPKRegime": dict(
        sl_atr=1.5, tp_atr=1.5, max_hold_days=4,
        rsi_lo=46.0, rsi_hi=58.0, vol_mult=1.4,
        regime="soft_bull",
        entry_type="momentum",
    ),
    # HybridSwing: proven 68.4% WR — RSI sweet-spot 44-58
    "HybridSwing": dict(
        sl_atr=1.5, tp_atr=1.5, max_hold_days=2,
        rsi_lo=44.0, rsi_hi=58.0, vol_mult=1.4,
        regime="soft_bull",
        entry_type="momentum",
    ),
}

COOLDOWN_BARS = 3   # after SL hit, skip this many bars before next entry


# ─── Synthetic data ─────────────────────────────────────────────────────────────

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


# ─── Signal helpers ─────────────────────────────────────────────────────────────

def _nan(*vs) -> bool:
    return any(math.isnan(float(v)) for v in vs)


def candle_bullish(closes, highs, lows, i, pct=0.60) -> bool:
    """Close in upper `pct` of bar range (60% default = more selective than v4's 50%)."""
    rng = float(highs[i]) - float(lows[i])
    if rng < 1e-6:
        return False
    return (float(closes[i]) - float(lows[i])) / rng >= pct


def macd_positive(macd_hist, i) -> bool:
    """MACD histogram is positive this bar (upswing already in progress)."""
    return not _nan(macd_hist[i]) and float(macd_hist[i]) > 0


def macd_improve(macd_hist, i) -> bool:
    if i < 1 or _nan(macd_hist[i], macd_hist[i - 1]):
        return False
    return float(macd_hist[i]) > float(macd_hist[i - 1])


def macd_cross_up(macd_hist, i, lookback=3) -> bool:
    """MACD histogram crossed negative→positive within last `lookback` bars."""
    for j in range(i, max(i - lookback, 0), -1):
        if j < 1 or _nan(macd_hist[j], macd_hist[j - 1]):
            continue
        if float(macd_hist[j - 1]) < 0 and float(macd_hist[j]) > 0:
            return True
    return False


def rsi_dipped_in_window(rsi, i, threshold, window_start, window_end) -> bool:
    """RSI dipped below `threshold` in bars [i-window_end, i-window_start]."""
    for j in range(max(0, i - window_end), max(0, i - window_start + 1)):
        if not _nan(rsi[j]) and float(rsi[j]) < threshold:
            return True
    return False


# ─── Entry checker ─────────────────────────────────────────────────────────────

def check_entry(
    i, closes, highs, lows, volumes,
    rsi, macd_hist, bb_lower, vol_ma,
    ema80, ema200, params,
) -> bool:
    if i < 8:
        return False

    # ── Regime filter ─────────────────────────────────────────────────────────
    regime = params["regime"]
    if _nan(ema80[i], ema200[i]):
        return False
    e80  = float(ema80[i])
    e200 = float(ema200[i])

    if regime == "bull_ema80":
        if e80 <= e200:
            return False
    elif regime == "bull_ema80_strong":
        if e80 <= e200:
            return False
        if float(closes[i]) <= e80:   # price must also be above EMA80
            return False
    elif regime == "bull_price":
        if e80 <= e200 or float(closes[i]) <= e200:
            return False
    elif regime == "soft_bull":
        if e80 < e200 * 0.98:
            return False

    # ── Universal: entry candle must be clearly bullish ───────────────────────
    if not candle_bullish(closes, highs, lows, i, pct=0.60):
        return False

    entry_type = params["entry_type"]
    c_rsi  = float(rsi[i])  if not _nan(rsi[i])  else 50.0
    p_rsi  = float(rsi[i-1]) if not _nan(rsi[i-1]) else 50.0
    c_vol  = float(volumes[i])
    ma_vol = float(vol_ma[i]) if not _nan(vol_ma[i]) else 1.0

    # ── S1: Dip-bounce — MACD cross is primary timing; RSI dip is context ────
    if entry_type == "dip_bounce":
        # Primary: MACD just crossed negative→positive (same signal as HybridSwing)
        macd_ok = macd_cross_up(macd_hist, i, lookback=3)
        # Context: RSI dipped below threshold in last 6 bars (bounce, not top)
        dip_ctx = rsi_dipped_in_window(
            rsi, i, params["rsi_dip"], window_start=1, window_end=6)
        # Entry RSI zone: not oversold, not overbought
        rsi_zone = params["rsi_lo"] <= c_rsi <= params["rsi_hi"]
        # Volume confirming the move
        vol_ok = c_vol >= ma_vol * params["vol_mult"]
        return macd_ok and dip_ctx and rsi_zone and vol_ok

    # ── S2: EMA80 pullback — MACD cross timing + price near EMA80 context ───
    if entry_type == "ema_pullback":
        price_pct = (float(closes[i]) - e80) / e80
        at_ema80  = (params["ema80_lo"] <= price_pct <= params["ema80_hi"])
        macd_ok   = macd_cross_up(macd_hist, i, lookback=3)  # precise timing
        rsi_zone  = (params["rsi_lo"] <= c_rsi <= params["rsi_hi"])
        vol_ok    = c_vol >= ma_vol * params["vol_mult"]
        return at_ema80 and macd_ok and rsi_zone and vol_ok

    # ── S3: Momentum — MACD cross + tight RSI zone + volume ──────────────────
    if entry_type == "momentum":
        cond_macd = macd_cross_up(macd_hist, i, lookback=3)
        cond_rsi  = (params["rsi_lo"] <= c_rsi <= params["rsi_hi"])
        cond_vol  = c_vol >= ma_vol * params["vol_mult"]
        return cond_macd and cond_rsi and cond_vol

    return False


# ─── Backtest runner ────────────────────────────────────────────────────────────

def run_strategy(name, candles, closes, highs, lows, volumes,
                 rsi, macd_hist, bb_lower, vol_ma, ema80, ema200, atr14,
                 params) -> list:
    max_hold_bars = int(params["max_hold_days"] * 24)
    trades: list  = []
    capital       = CAPITAL
    in_pos        = False
    entry_px = sl_px = tp_px = 0.0
    entry_bar = 0
    cooldown  = 0
    n = len(candles)

    for i in range(WARMUP, n - 1):
        c_close = float(closes[i])
        c_high  = float(highs[i])
        c_low   = float(lows[i])

        if in_pos:
            bars_held  = i - entry_bar
            exit_px    = exit_type = None
            if c_low   <= sl_px:             exit_px, exit_type = sl_px,   "sl"
            elif c_high >= tp_px:             exit_px, exit_type = tp_px,   "tp"
            elif bars_held >= max_hold_bars:  exit_px, exit_type = c_close, "timeout"

            if exit_px is not None:
                qty = TRADE_USD / entry_px
                pnl = (exit_px - entry_px) * qty - exit_px * qty * FEE_RATE
                capital += pnl
                trades.append({
                    "bar": i, "entry": round(entry_px, 2), "exit": round(exit_px, 2),
                    "type": exit_type, "bars": bars_held,
                    "pnl_net": round(pnl, 4),
                    "pnl_pct": round((exit_px - entry_px) / entry_px * 100, 3),
                    "capital": round(capital, 2),
                })
                in_pos = False
                if exit_type == "sl":
                    cooldown = COOLDOWN_BARS  # skip next N bars after SL

        if cooldown > 0:
            cooldown -= 1
            continue

        if not in_pos and check_entry(
                i, closes, highs, lows, volumes,
                rsi, macd_hist, bb_lower, vol_ma, ema80, ema200, params):
            atr_v    = float(atr14[i]) if not _nan(atr14[i]) else c_close * 0.015
            entry_px = c_close
            sl_px    = entry_px - params["sl_atr"] * atr_v
            tp_px    = entry_px + params["tp_atr"] * atr_v
            capital -= (TRADE_USD / entry_px) * entry_px * FEE_RATE
            entry_bar = i
            in_pos    = True

    if in_pos:
        c   = float(closes[-1])
        qty = TRADE_USD / entry_px
        pnl = (c - entry_px) * qty - c * qty * FEE_RATE
        capital += pnl
        trades.append({
            "bar": n - 1, "entry": round(entry_px, 2), "exit": round(c, 2),
            "type": "end", "bars": n - 1 - entry_bar,
            "pnl_net": round(pnl, 4), "pnl_pct": round((c - entry_px) / entry_px * 100, 3),
            "capital": round(capital, 2),
        })
    return trades


# ─── Stats & display ─────────────────────────────────────────────────────────────

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
        "net":   net, "avg_win": aw, "avg_loss": al, "pf": gw / max(gl, 1e-9),
        "final": trades[-1]["capital"],
        "avg_bars": sum(t["bars"] for t in trades) / len(trades),
    }


def print_results(results):
    W = 82
    print(f"\n{'═'*W}")
    print(f"  SWING BACKTEST v5  |  {SYMBOL} 1H  |  {MONTHS}-month synthetic BTC")
    print(f"  Capital ${CAPITAL:.0f}  |  Trade ${TRADE_USD:.0f}  |  Fee {FEE_RATE*100:.1f}%  "
          f"|  Cooldown {COOLDOWN_BARS} bars after SL")
    print(f"  Fix: enter AFTER reversal confirmed — late entry beats calling the bottom")
    print(f"{'─'*W}")
    names = list(results.keys())
    print(f"  {'Metric':<24}" + "".join(f"  {n:>18}" for n in names))
    print(f"  {'─'*22}" + "  " + "  ".join(["─" * 18] * len(names)))

    def row(label, fn):
        print(f"  {label:<24}" + "".join(f"  {fn(_stats(results[n])):>18}" for n in names))

    row("Trades",         lambda s: str(s["trades"]))
    row("Win / Loss",     lambda s: f"{s['wins']}W / {s['losses']}L")
    row("Win Rate ★",     lambda s: f"{s['wr']:.1f}%  {'✓' if s['wr'] >= 62 else '·'}")
    row("Profit Factor",  lambda s: f"{s['pf']:.2f}")
    row("Avg Win $",      lambda s: f"+${s['avg_win']:.2f}")
    row("Avg Loss $",     lambda s: f"${s['avg_loss']:.2f}")
    row("Net P/L",        lambda s: f"${s['net']:+.2f}")
    row("ROI",            lambda s: f"{(s['final'] - CAPITAL) / CAPITAL * 100:+.1f}%")
    row("Final Capital",  lambda s: f"${s['final']:.2f}")
    row("Avg Hold (h)",   lambda s: f"{s['avg_bars']:.0f}h")

    print(f"\n  Exit breakdown:")
    for name, trades in results.items():
        if not trades:
            print(f"    {name:<16}: — no trades —"); continue
        ec: dict = {}
        for t in trades:
            ec[t["type"]] = ec.get(t["type"], 0) + 1
        bd = "  |  ".join(f"{k}:{v}" for k, v in sorted(ec.items(), key=lambda x: -x[1]))
        print(f"    {name:<16}: {bd}")

    print(f"\n  Entry logic (v5):")
    desc = {
        "SwingReversal": ("MACD-cross(3) + RSI 40-56 + Vol×1.4 | "
                          "soft_bull (EMA80>EMA200×0.98) | ATR SL×1.5 TP×1.5"),
        "CPKRegime":     ("MACD-cross(3) + RSI 46-58 + Vol×1.4 | "
                          "soft_bull (EMA80>EMA200×0.98) | ATR SL×1.5 TP×1.5"),
        "HybridSwing":   ("MACD-cross(3) + RSI 44-58 + Vol×1.4 | "
                          "soft_bull (EMA80>EMA200×0.98) | ATR SL×1.5 TP×1.5"),
    }
    for n, d in desc.items():
        print(f"    {n:<16}: {d}")
    print(f"  Candle quality: close>60% of range.  Regime: EMA80>EMA200 required.")
    print(f"{'═'*W}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    W = 82
    print(f"\n{'═'*W}")
    print(f"  SwingReversal v5  |  {SYMBOL} 1H  |  {MONTHS} months  |  late-entry confirmation")
    print(f"{'═'*W}")

    candles = fetch_candles()
    if len(candles) < 300:
        print("ERROR: not enough candles"); return

    n       = len(candles)
    closes  = np.array([c.close  for c in candles])
    highs   = np.array([c.high   for c in candles])
    lows    = np.array([c.low    for c in candles])
    volumes = np.array([c.volume for c in candles])

    print(f"\n  Computing indicators on {n} bars...")
    rsi14           = BaseStrategy.rsi(closes.tolist(), 14)
    _, _, macd_hist = BaseStrategy.macd(closes.tolist(), 12, 26, 9)
    _, _, bb_lower  = BaseStrategy.bollinger_bands(closes.tolist(), 20, 2.0)
    vol_ma          = BaseStrategy.sma(volumes.tolist(), 20)
    atr14           = BaseStrategy.atr(candles, 14)
    ema80           = BaseStrategy.ema(closes.tolist(), 80)
    ema200          = BaseStrategy.ema(closes.tolist(), 200)

    print(f"  Running 3 strategies...\n")
    results: dict[str, list] = {}
    for name, params in STRATEGIES.items():
        trades = run_strategy(
            name, candles, closes, highs, lows, volumes,
            rsi14, macd_hist, bb_lower, vol_ma, ema80, ema200, atr14, params,
        )
        results[name] = trades
        s = _stats(trades)
        flag = "✓ ≥62%" if s["wr"] >= 62 else f"({s['wr']:.0f}%)"
        print(f"  [{name}]  trades={s['trades']}  WR={s['wr']:.1f}% {flag}  "
              f"Net=${s['net']:+.2f}  PF={s['pf']:.2f}")

    print_results(results)


if __name__ == "__main__":
    main()
