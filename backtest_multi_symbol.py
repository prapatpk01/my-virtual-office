"""
Walk-Forward Backtest: 3 Strategies × 3 Symbols (DOGE / XRP / TRX)
3 Stages × 250 bars ≈ 1 month  |  Rolling 250-bar context window

Usage:
    python backtest_multi_symbol.py
"""
import asyncio
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy
from app.trading.strategies.sentinel_strategy import SentinelStrategy
from app.trading.strategies.rsi_macd import RSIMACDStrategy
from app.trading.strategies.base import SignalType

# ── Config ──────────────────────────────────────────────────────────────────
STAGE_BARS = 250
N_STAGES   = 3
WARMUP     = 200
CTX_WINDOW = 250
FETCH_BARS = WARMUP + STAGE_BARS * N_STAGES   # 950 bars
LOOKFWD    = 48
RISK_USD   = 10.0

# ── Per-symbol config ────────────────────────────────────────────────────────
# (yfinance_ticker, ccxt_pair, start_price, vol_scale, seed, sl_pct, tp_pct, mcdx_params)
# SL/TP tuned per coin via backtest_optimize_trx.py / optimizer results:
#   DOGE: wider SL needed for meme-vol bursts
#   XRP:  balanced; ETH-equivalent vol
#   TRX:  tight SL/TP — low vol (×0.75), MCDX dwcs_buy=55 rvol≥1.0 → 72.2% WR ★
SYMBOLS = {
    "DOGE": ("DOGE-USD", "DOGE/USDT", 0.0750, 1.35, 42,
             0.046, 0.048, {}),
    "XRP":  ("XRP-USD",  "XRP/USDT",  0.5500, 0.90, 99,
             0.046, 0.048, {}),
    "TRX":  ("TRX-USD",  "TRX/USDT",  0.1050, 0.75, 77,
             0.035, 0.038, {"dwcs_buy": 55, "dwcs_sell": 45, "rvol_min": 1.0}),
}


# ── Data Fetcher ─────────────────────────────────────────────────────────────
def fetch_yfinance(symbol_yf: str, period: str = "60d", interval: str = "1h") -> list:
    try:
        import yfinance as yf
        print(f"    Fetching {symbol_yf} from Yahoo Finance...", end=" ")
        ticker = yf.Ticker(symbol_yf)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("Empty dataframe")
        candles = []
        for ts, row in df.iterrows():
            candles.append(OHLCV(
                timestamp=int(ts.timestamp() * 1000),
                open=float(row["Open"]), high=float(row["High"]),
                low=float(row["Low"]),   close=float(row["Close"]),
                volume=float(row["Volume"]),
            ))
        print(f"✓ {len(candles)} bars")
        return candles
    except Exception as e:
        print(f"✗ ({e.__class__.__name__})")
        return None


def fetch_ccxt(pair: str, interval: str = "1h", limit: int = 950) -> list:
    try:
        import ccxt
        print(f"    Fetching {pair} from Binance...", end=" ")
        ex = ccxt.binance({"options": {"defaultType": "spot"}})
        bars = ex.fetch_ohlcv(pair, interval, limit=limit)
        candles = [OHLCV(
            timestamp=int(b[0]),
            open=float(b[1]), high=float(b[2]),
            low=float(b[3]),  close=float(b[4]),
            volume=float(b[5]),
        ) for b in bars]
        print(f"✓ {len(candles)} bars")
        return candles
    except Exception as e:
        print(f"✗ ({e.__class__.__name__})")
        return None


def generate_synthetic(n: int, start_price: float, vol_scale: float, seed: int) -> list:
    """GBM synthetic with 6 regime phases, calibrated per-coin."""
    rng = np.random.default_rng(seed)
    closes = [start_price]
    regime_len = 168
    phases = [
        (+0.0003, 0.014 * vol_scale),  # bull
        (-0.0002, 0.014 * vol_scale),  # bear
        ( 0.0001, 0.008 * vol_scale),  # sideways
        (+0.0004, 0.018 * vol_scale),  # strong bull
        (-0.0003, 0.016 * vol_scale),  # strong bear
        ( 0.0000, 0.007 * vol_scale),  # tight range
    ]
    for i in range(1, n):
        phase_idx = (i // regime_len) % len(phases)
        drift, vol = phases[phase_idx]
        shock = rng.normal(drift, vol)
        closes.append(max(closes[-1] * (1 + shock), 0.0001))

    bar_ms   = 60 * 60 * 1000
    base_ts  = 1_700_000_000_000
    candles  = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        spread = abs(rng.normal(0, c * 0.003))
        candles.append(OHLCV(
            timestamp=base_ts + i * bar_ms,
            open=round(o, 6), high=round(max(o, c) + spread, 6),
            low=round(min(o, c) - spread, 6), close=round(c, 6),
            volume=float(rng.uniform(500_000, 5_000_000)),
        ))
    return candles


def get_candles(name: str, symbol_yf: str, ccxt_pair: str,
                start_price: float, vol_scale: float, seed: int) -> tuple:
    """Fetch real data; fall back to calibrated synthetic."""
    candles = fetch_yfinance(symbol_yf)
    if candles and len(candles) >= 500:
        return candles[-FETCH_BARS:], "Yahoo Finance"

    candles = fetch_ccxt(ccxt_pair)
    if candles and len(candles) >= 500:
        return candles[-FETCH_BARS:], "Binance (ccxt)"

    print(f"    → Synthetic GBM ({name}, seed={seed}, vol×{vol_scale})")
    return generate_synthetic(FETCH_BARS, start_price, vol_scale, seed), "Synthetic GBM"


# ── Exit Scanner ─────────────────────────────────────────────────────────────
def find_exit(candles: list, entry_idx: int, direction: int,
              sl_p: float, tp_p: float) -> tuple:
    for j in range(entry_idx + 1, min(entry_idx + LOOKFWD + 1, len(candles))):
        h, l = candles[j].high, candles[j].low
        if direction == 1:
            if l <= sl_p: return j, -1
            if h >= tp_p: return j, +1
        else:
            if h >= sl_p: return j, -1
            if l <= tp_p: return j, +1
    return min(entry_idx + LOOKFWD, len(candles) - 1), 0


# ── Stage Runner ─────────────────────────────────────────────────────────────
async def run_stage(strategy, candles: list, start: int, end: int,
                    sl_pct: float = 0.046, tp_pct: float = 0.048) -> list:
    rr = tp_pct / sl_pct
    trades = []
    lock_until = -1
    for i in range(start, end):
        if i <= lock_until:
            continue
        ctx_start = max(0, i - CTX_WINDOW + 1)
        context   = candles[ctx_start:i + 1]
        price     = candles[i].close
        try:
            signal = await strategy.analyze(context, price, mtf_candles=None)
        except Exception:
            continue
        if signal.type == SignalType.HOLD:
            continue
        direction = 1 if signal.type == SignalType.BUY else -1
        sl_p = price * (1 - sl_pct) if direction == 1 else price * (1 + sl_pct)
        tp_p = price * (1 + tp_pct) if direction == 1 else price * (1 - tp_pct)
        exit_bar, outcome = find_exit(candles, i, direction, sl_p, tp_p)
        pnl_r = rr if outcome == 1 else (-1.0 if outcome == -1 else 0.0)
        trades.append({
            "bar": i, "direction": "LONG" if direction == 1 else "SHORT",
            "entry": price, "outcome": outcome, "pnl_r": pnl_r,
            "confidence": signal.confidence,
        })
        lock_until = exit_bar
    return trades


# ── Stats ─────────────────────────────────────────────────────────────────────
def calc_stats(trades: list, rr: float = None) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                "wr": 0.0, "pf": 0.0, "total_r": 0.0, "est_pnl": 0.0}
    wins     = [t for t in trades if t["outcome"] ==  1]
    losses   = [t for t in trades if t["outcome"] == -1]
    timeouts = [t for t in trades if t["outcome"] ==  0]
    decided  = len(wins) + len(losses)
    wr       = len(wins) / max(decided, 1) * 100
    total_r  = sum(t["pnl_r"] for t in trades)
    _rr      = rr if rr is not None else (trades[0]["pnl_r"] if wins else 1.0)
    pf       = (len(wins) * _rr) / max(len(losses), 1e-9)
    return {
        "trades": len(trades), "wins": len(wins),
        "losses": len(losses), "timeouts": len(timeouts),
        "wr": round(wr, 1), "pf": round(pf, 2),
        "total_r": round(total_r, 2),
        "est_pnl": round(total_r * RISK_USD, 2),
    }


def row(label: str, s: dict, star: bool = False):
    wl = f"{s['wins']}W/{s['losses']}L/{s['timeouts']}T"
    mk = " ★" if star else ""
    print(f"  {label:<12}  {s['trades']:>5}  {wl:<14}  "
          f"WR:{s['wr']:>5.1f}%{mk}  PF:{s['pf']:>5.2f}  "
          f"{s['total_r']:>+6.1f}R  ${s['est_pnl']:>+8.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    W = 72
    print("\n" + "═" * W)
    print("  MULTI-SYMBOL BACKTEST  |  DOGE / XRP / TRX  |  1H  |  3 × 250 bars ≈ 1 month")
    print("  Strategies: MCDX Plus  |  Sentinel (BOS)  |  RSI+MACD (Oscillator)")
    print("═" * W)

    from datetime import datetime, timezone

    grand = {}   # {coin: {strat: stats}}

    for coin, (sym_yf, ccxt_pair, start_px, vol_sc, seed,
               sl_pct, tp_pct, mcdx_p) in SYMBOLS.items():
        rr = tp_pct / sl_pct
        print(f"\n{'━' * W}")
        print(f"  ● {coin}/USDT  (start_price=${start_px:.4f}  vol×{vol_sc}  seed={seed}  "
              f"SL={sl_pct*100:.1f}%  TP={tp_pct*100:.1f}%)")
        print(f"{'━' * W}")

        candles, src = get_candles(coin, sym_yf, ccxt_pair, start_px, vol_sc, seed)
        if len(candles) > FETCH_BARS:
            candles = candles[-FETCH_BARS:]
        total = len(candles)

        ts0 = datetime.fromtimestamp(candles[0].timestamp / 1000, tz=timezone.utc)
        tsN = datetime.fromtimestamp(candles[-1].timestamp / 1000, tz=timezone.utc)
        print(f"  {total} bars  [{src}]  {ts0:%Y-%m-%d} → {tsN:%Y-%m-%d}")
        print(f"  Price: ${min(c.low for c in candles):.5f} – ${max(c.high for c in candles):.5f}")

        stages = []
        for s in range(N_STAGES):
            start = WARMUP + s * STAGE_BARS
            stages.append((start, min(start + STAGE_BARS, total)))

        strategies = [
            ("MCDX",     MCDXStrategy(f"{coin}/USDT", mcdx_p or {})),
            ("Sentinel", SentinelStrategy(f"{coin}/USDT")),
            ("RSI+MACD", RSIMACDStrategy(f"{coin}/USDT")),
        ]

        coin_results = {}

        for strat_name, strategy in strategies:
            all_trades = []
            for s_idx, (start, end) in enumerate(stages):
                trades = await run_stage(strategy, candles, start, end, sl_pct, tp_pct)
                all_trades.extend(trades)
            stats = calc_stats(all_trades, rr)
            coin_results[strat_name] = stats
            star = stats["wr"] >= 66.0 and stats["trades"] >= 10
            row(f"  {strat_name}", stats, star)

        grand[coin] = coin_results

    # ── Grand Summary ────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  GRAND SUMMARY — per coin × per strategy")
    print(f"  {'─'*68}")
    print(f"  {'':8}  {'MCDX':^22}  {'Sentinel':^22}  {'RSI+MACD':^22}")
    print(f"  {'Coin':<8}  {'Trades':>6} {'WR%':>6} {'R':>6}    {'Trades':>6} {'WR%':>6} {'R':>6}    {'Trades':>6} {'WR%':>6} {'R':>6}")
    print(f"  {'─'*68}")

    for coin, results in grand.items():
        m  = results["MCDX"]
        se = results["Sentinel"]
        rs = results["RSI+MACD"]
        def fmt(s): return f"{s['trades']:>6}  {s['wr']:>5.1f}%{'★' if s['wr']>=66 and s['trades']>=10 else ' '}  {s['total_r']:>+5.1f}R"
        print(f"  {coin:<8}  {fmt(m)}    {fmt(se)}    {fmt(rs)}")

    print(f"  {'─'*68}")

    # ── Best coin × strategy combos ──────────────────────────────────────────
    print(f"\n  TOP PICKS (WR ≥66% AND Trades ≥10):")
    found = False
    for coin, results in grand.items():
        for strat, s in results.items():
            if s["wr"] >= 66.0 and s["trades"] >= 10:
                print(f"   ★ {coin:5} × {strat:<10}  "
                      f"Trades={s['trades']}  WR={s['wr']:.1f}%  "
                      f"PF={s['pf']:.2f}  R={s['total_r']:+.1f}  ${s['est_pnl']:+.2f}")
                found = True
    if not found:
        print("   (ไม่มี combo ที่ผ่านทั้งสองเงื่อนไขพร้อมกัน — ดู WR ต่อ Coin ด้านบน)")

    print(f"\n  หมายเหตุ:")
    print(f"   • SL/TP per coin: DOGE/XRP 4.6%/4.8%  |  TRX 3.5%/3.8% (optimized via grid search)")
    print(f"   • TRX MCDX: dwcs_buy=55, rvol_min=1.0 → 72.2% WR target (optimizer result)")
    print(f"   • Synthetic: vol calibrated per coin (DOGE ×1.35 / XRP ×0.90 / TRX ×0.75)")
    print(f"   • Live MTF gate (RSI+MACD) + real BOS structure (Sentinel) → WR คาดว่าสูงกว่า")
    print(f"   • LOOKFWD={LOOKFWD} bars")
    print("═" * W + "\n")


if __name__ == "__main__":
    asyncio.run(main())
