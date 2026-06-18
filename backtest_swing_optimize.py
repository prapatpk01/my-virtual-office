"""
Grid-search SL × TP for the 3 Swing v5 strategies.
Tries every combination of SL ∈ [1.0..3.0] × TP ∈ [1.0..4.5]
and reports the best configs by Net P/L per strategy.
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
COOLDOWN_BARS = 3

SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

STRATEGIES = {
    "SwingReversal": dict(rsi_lo=40.0, rsi_hi=56.0, vol_mult=1.4,
                          regime="soft_bull", max_hold_days=3),
    "CPKRegime":     dict(rsi_lo=46.0, rsi_hi=58.0, vol_mult=1.4,
                          regime="soft_bull", max_hold_days=4),
    "HybridSwing":   dict(rsi_lo=44.0, rsi_hi=58.0, vol_mult=1.4,
                          regime="soft_bull", max_hold_days=2),
}


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
                print(f"  {source}: {len(candles)} candles"); return candles
        except Exception:
            pass

    print("  ⚠ Network restricted — synthetic data")
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
    closes = [32_000.0]; vol = 0.0022
    for i in range(1, n_bars):
        frac = i / n_bars
        dr, base_v, persist, df = 0.0, 0.003, 0.90, 7
        for rs, re, d, bv, p, tdf in regimes:
            if rs <= frac < re:
                dr, base_v, persist, df = d, bv, p, tdf; break
        shock = float(rng.standard_t(df)) * vol
        vol   = float(np.sqrt(persist * vol**2 + (1-persist) * base_v**2 + 0.02 * shock**2))
        vol   = max(0.0010, min(vol, 0.015))
        closes.append(max(closes[-1] * (1 + dr + shock), 1000.0))
    candles = []; ts0 = since_ms
    for i, c in enumerate(closes):
        o     = closes[i-1] if i > 0 else c
        atr_n = abs(float(rng.normal(0, c * 0.0018)))
        vm    = float(rng.lognormal(6.5, 0.6))
        candles.append(OHLCV(ts0 + i*3_600_000,
            round(o,2), round(max(o,c)+atr_n,2),
            round(min(o,c)-atr_n,2), round(c,2), round(vm,0)))
    peak   = max(closes)
    trough = min(closes[int(n_bars*0.38): int(n_bars*0.72)])
    print(f"  Generated {len(candles)} bars  ${closes[0]:,.0f}→${closes[-1]:,.0f}  "
          f"peak=${peak:,.0f}  dd={(peak-trough)/peak*100:.0f}%")
    return candles


# ─── Indicators ─────────────────────────────────────────────────────────────────

def _nan(*vs):
    return any(math.isnan(float(v)) for v in vs)

def candle_bullish(closes, highs, lows, i):
    r = float(highs[i]) - float(lows[i])
    return r > 1e-6 and (float(closes[i]) - float(lows[i])) / r >= 0.60

def macd_cross_up(macd_hist, i, lookback=3):
    for j in range(i, max(i-lookback, 0), -1):
        if j < 1 or _nan(macd_hist[j], macd_hist[j-1]): continue
        if float(macd_hist[j-1]) < 0 and float(macd_hist[j]) > 0: return True
    return False

def check_entry(i, closes, highs, lows, volumes,
                rsi, macd_hist, vol_ma, ema80, ema200, p) -> bool:
    if i < 8 or _nan(ema80[i], ema200[i]): return False
    if float(ema80[i]) < float(ema200[i]) * 0.98: return False
    if not candle_bullish(closes, highs, lows, i): return False
    c_rsi = float(rsi[i]) if not _nan(rsi[i]) else 50.0
    c_vol = float(volumes[i])
    ma_v  = float(vol_ma[i]) if not _nan(vol_ma[i]) else 1.0
    return (macd_cross_up(macd_hist, i) and
            p["rsi_lo"] <= c_rsi <= p["rsi_hi"] and
            c_vol >= ma_v * p["vol_mult"])


# ─── Runner ──────────────────────────────────────────────────────────────────────

def run(closes, highs, lows, volumes, rsi, macd_hist,
        vol_ma, ema80, ema200, atr14, candles, p, sl_atr, tp_atr) -> dict:
    max_hold = int(p["max_hold_days"] * 24)
    capital = CAPITAL; in_pos = False
    entry_px = sl_px = tp_px = 0.0; entry_bar = cooldown = 0
    wins = losses = 0; gross_win = gross_loss = 0.0; net = 0.0
    n = len(closes)

    for i in range(WARMUP, n - 1):
        cc = float(closes[i]); ch = float(highs[i]); cl = float(lows[i])
        if in_pos:
            bh = i - entry_bar; ex = et = None
            if cl <= sl_px:              ex, et = sl_px,   "sl"
            elif ch >= tp_px:            ex, et = tp_px,   "tp"
            elif bh >= max_hold:         ex, et = cc,      "to"
            if ex is not None:
                qty = TRADE_USD / entry_px
                pnl = (ex - entry_px) * qty - ex * qty * FEE_RATE
                net += pnl; capital += pnl
                if pnl > 0: wins += 1; gross_win += pnl
                else:       losses += 1; gross_loss += abs(pnl)
                in_pos = False
                if et == "sl": cooldown = COOLDOWN_BARS

        if cooldown > 0: cooldown -= 1; continue
        if not in_pos and check_entry(
                i, closes, highs, lows, volumes,
                rsi, macd_hist, vol_ma, ema80, ema200, p):
            atr_v    = float(atr14[i]) if not _nan(atr14[i]) else cc * 0.015
            entry_px = cc
            sl_px    = entry_px - sl_atr * atr_v
            tp_px    = entry_px + tp_atr * atr_v
            capital -= (TRADE_USD / entry_px) * entry_px * FEE_RATE
            entry_bar = i; in_pos = True

    total = wins + losses
    wr = wins / total * 100 if total else 0.0
    pf = gross_win / max(gross_loss, 1e-9)
    return {"trades": total, "wins": wins, "losses": losses,
            "wr": wr, "net": round(net, 4), "pf": round(pf, 3)}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*80}")
    print(f"  SL × TP Grid Search  |  {SYMBOL} 1H  |  {MONTHS} months")
    print(f"  SL grid: {SL_GRID}")
    print(f"  TP grid: {TP_GRID}")
    print(f"{'═'*80}")

    candles = fetch_candles()
    n       = len(candles)
    closes  = np.array([c.close  for c in candles])
    highs   = np.array([c.high   for c in candles])
    lows    = np.array([c.low    for c in candles])
    volumes = np.array([c.volume for c in candles])

    print(f"\n  Pre-computing indicators on {n} bars...")
    rsi14           = BaseStrategy.rsi(closes.tolist(), 14)
    _, _, macd_hist = BaseStrategy.macd(closes.tolist(), 12, 26, 9)
    vol_ma          = BaseStrategy.sma(volumes.tolist(), 20)
    atr14           = BaseStrategy.atr(candles, 14)
    ema80           = BaseStrategy.ema(closes.tolist(), 80)
    ema200          = BaseStrategy.ema(closes.tolist(), 200)

    total_combos = len(SL_GRID) * len(TP_GRID)
    print(f"  Testing {total_combos} SL×TP combos per strategy ({total_combos * 3} total)...\n")

    best_by_net: dict[str, list] = {}
    best_by_pf:  dict[str, list] = {}

    for name, params in STRATEGIES.items():
        results = []
        for sl in SL_GRID:
            for tp in TP_GRID:
                if tp <= sl * 0.5:   # skip clearly bad R:R
                    continue
                r = run(closes, highs, lows, volumes,
                        rsi14, macd_hist, vol_ma, ema80, ema200, atr14,
                        candles, params, sl, tp)
                if r["trades"] >= 8:   # need enough trades
                    results.append({"sl": sl, "tp": tp, **r})

        results.sort(key=lambda x: -x["net"])
        best_by_net[name] = results[:5]

        results2 = sorted([r for r in results if r["trades"] >= 10],
                          key=lambda x: -x["pf"])
        best_by_pf[name]  = results2[:5]

    # ── Print ──────────────────────────────────────────────────────────────
    for name in STRATEGIES:
        print(f"\n  ┌─ {name} — Top 5 by Net P/L {'─'*40}")
        print(f"  │  {'SL':>5}  {'TP':>5}  {'Trades':>7}  {'WR':>7}  {'PF':>6}  {'Net':>8}  {'ROI':>7}")
        print(f"  │  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}")
        for r in best_by_net[name]:
            roi = (CAPITAL + r["net"] - CAPITAL) / CAPITAL * 100
            flag = " ★" if r["sl"] == 2.5 and r["tp"] == 3.0 else ""
            print(f"  │  {r['sl']:>5.1f}  {r['tp']:>5.1f}  "
                  f"{r['trades']:>7}  {r['wr']:>6.1f}%  {r['pf']:>6.2f}  "
                  f"${r['net']:>+7.2f}  {roi:>+6.1f}%{flag}")

        print(f"  │")
        print(f"  └─ Top 5 by Profit Factor {'─'*43}")
        print(f"     {'SL':>5}  {'TP':>5}  {'Trades':>7}  {'WR':>7}  {'PF':>6}  {'Net':>8}  {'ROI':>7}")
        print(f"     {'─'*5}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}")
        for r in best_by_pf[name]:
            roi = (CAPITAL + r["net"] - CAPITAL) / CAPITAL * 100
            flag = " ★" if r["sl"] == 2.5 and r["tp"] == 3.0 else ""
            print(f"     {r['sl']:>5.1f}  {r['tp']:>5.1f}  "
                  f"{r['trades']:>7}  {r['wr']:>6.1f}%  {r['pf']:>6.2f}  "
                  f"${r['net']:>+7.2f}  {roi:>+6.1f}%{flag}")

    # ── Recommend ──────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  RECOMMENDED PARAMS (best Net P/L with ≥10 trades)")
    print(f"{'─'*80}")
    for name in STRATEGIES:
        cands = [r for r in best_by_net[name] if r["trades"] >= 10]
        if cands:
            r = cands[0]
            print(f"  {name:<16}  SL={r['sl']}×ATR  TP={r['tp']}×ATR  "
                  f"→  trades={r['trades']}  WR={r['wr']:.1f}%  "
                  f"PF={r['pf']:.2f}  Net=${r['net']:+.2f}")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()
