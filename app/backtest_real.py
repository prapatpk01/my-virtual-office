"""
Real-data backtest: CPKRegimeStrategy + ProfitableBot on OKX historical data.

Fetches 1H OHLCV from OKX public API (no API key needed), converts to
Heikin-Ashi, then runs both strategies bar-by-bar.

Usage:
    python backtest_real.py                         # real OKX data, 2000 bars
    python backtest_real.py --bars 3000             # longer history
    python backtest_real.py --exchange binance      # use Binance instead
    python backtest_real.py --offline               # synthetic data (no internet needed)

Note: requires internet access to exchange API. Run on Railway or local machine.
"""
import asyncio
import argparse
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import ssl
import aiohttp
import ccxt.async_support as ccxt

from trading.connectors.base import OHLCV, to_heikin_ashi
from trading.strategies.swing_strategy import CPKRegimeStrategy
from trading.strategies.profitable_strategy import ProfitableBot
from trading.strategies.base import SignalType


# ── Data fetching ─────────────────────────────────────────────────────────────

_MS_PER_BAR = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


async def fetch_candles(exchange, symbol: str, timeframe: str, total: int) -> list[OHLCV]:
    """Paginate OKX public OHLCV endpoint until we have `total` bars."""
    ms = _MS_PER_BAR.get(timeframe, 3_600_000)
    since = exchange.milliseconds() - (total + 50) * ms

    seen: dict = {}
    per_page = 300
    while len(seen) < total:
        batch = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=per_page)
        if not batch:
            break
        for row in batch:
            seen[row[0]] = row
        if len(batch) < per_page:
            break
        since = batch[-1][0] + 1   # next page starts after last bar

    sorted_rows = sorted(seen.values(), key=lambda r: r[0])
    candles = [OHLCV(ts, o, h, l, c, v) for ts, o, h, l, c, v in sorted_rows]
    return candles[-total:]


# ── Backtest engine ───────────────────────────────────────────────────────────

async def run_strategy(name: str, strategy, candles_ha: list, candles_4h_ha: list,
                       trade_usdt: float) -> list[dict]:
    """Simulate strategy bar-by-bar. Returns list of closed trade dicts."""
    trades = []
    pos = None  # {entry_price, sl, tp, usdt, entry_bar}

    for i, bar in enumerate(candles_ha):
        # ── Check SL/TP hit on current bar ────────────────────────────────
        if pos:
            hit_tp = pos["tp"] and float(bar.high) >= pos["tp"]
            hit_sl = pos["sl"] and float(bar.low)  <= pos["sl"]
            if hit_tp or hit_sl:
                if hit_tp and hit_sl:
                    # Both levels on same bar: whichever open is closer to
                    closer_sl = abs(float(bar.open) - pos["sl"]) < abs(float(bar.open) - pos["tp"])
                    reason = "stop_loss" if closer_sl else "take_profit"
                else:
                    reason = "take_profit" if hit_tp else "stop_loss"
                exit_price = pos["tp"] if reason == "take_profit" else pos["sl"]
                pnl_usdt   = (exit_price / pos["entry_price"] - 1) * pos["usdt"]
                trades.append({
                    "entry_bar":   pos["entry_bar"],
                    "exit_bar":    i,
                    "entry_price": pos["entry_price"],
                    "exit_price":  exit_price,
                    "reason":      reason,
                    "pnl_usdt":    pnl_usdt,
                    "pnl_pct":     (exit_price / pos["entry_price"] - 1) * 100,
                    "hold_bars":   i - pos["entry_bar"],
                })
                pos = None

        # ── Generate signal (no position open) ────────────────────────────
        if pos is None:
            # 4H slice: only bars available up to current 1H timestamp
            c4h = [c for c in candles_4h_ha if c.timestamp <= bar.timestamp]
            mtf  = {"4h": c4h} if len(c4h) >= 50 else None

            sig = await strategy.analyze(candles_ha[:i + 1], float(bar.close), mtf_candles=mtf)
            if sig.type == SignalType.BUY:
                meta = sig.metadata or {}
                sl = meta.get("stop_loss")
                tp = meta.get("take_profit")
                ep = float(bar.close)
                if sl and tp and sl > 0 and tp > 0 and sl < ep < tp:
                    pos = {"entry_price": ep, "sl": sl, "tp": tp,
                           "usdt": trade_usdt, "entry_bar": i}

    return trades


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(name: str, trades: list, raw_candles: list, trade_usdt: float):
    sep = "─" * 56

    if not trades:
        print(f"\n{sep}")
        print(f"  {name}  —  0 trades (conditions not met in this period)")
        return

    wins  = [t for t in trades if t["reason"] == "take_profit"]
    loses = [t for t in trades if t["reason"] == "stop_loss"]
    wr    = len(wins) / len(trades) * 100
    avg_w = sum(t["pnl_usdt"] for t in wins)  / max(len(wins), 1)
    avg_l = sum(t["pnl_usdt"] for t in loses) / max(len(loses), 1)
    abs_l = abs(avg_l)
    bep   = abs_l / (avg_w + abs_l) * 100 if (avg_w + abs_l) > 0 else 50.0
    total_pnl = sum(t["pnl_usdt"] for t in trades)

    # Equity curve for max drawdown
    eq, peak, max_dd = trade_usdt, trade_usdt, 0.0
    for t in trades:
        eq += t["pnl_usdt"]
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    avg_hold = sum(t["hold_bars"] for t in trades) / len(trades)
    dt_fmt  = "%Y-%m-%d"
    start   = datetime.fromtimestamp(raw_candles[0].timestamp / 1000, tz=timezone.utc).strftime(dt_fmt)
    end     = datetime.fromtimestamp(raw_candles[-1].timestamp / 1000, tz=timezone.utc).strftime(dt_fmt)
    profit  = "✓ PROFITABLE" if wr > bep else "✗ BELOW BEP"

    print(f"\n{sep}")
    print(f"  {name}")
    print(f"  Period   : {start} → {end}  ({len(raw_candles)} bars)")
    print(f"  Trades   : {len(trades)}  (Wins {len(wins)} / Losses {len(loses)})")
    print(f"  Win Rate : {wr:.1f}%   BEP: {bep:.1f}%   {profit}")
    print(f"  Avg Win  : +${avg_w:,.2f}   Avg Loss: -${abs_l:,.2f}")
    print(f"  Total P&L: ${total_pnl:+,.2f}  (per ${trade_usdt:.0f}/trade)")
    print(f"  Max DD   : {max_dd:.1f}%   Avg Hold: {avg_hold:.1f} bars")

    print(f"\n  Recent trades (last 10):")
    header = f"    {'Date':>10}  {'Entry':>9}  {'Exit':>9}  {'P&L':>9}  Reason"
    print(header)
    for t in trades[-10:]:
        dt = datetime.fromtimestamp(
            raw_candles[t["entry_bar"]].timestamp / 1000, tz=timezone.utc
        ).strftime("%m-%d %H:%M")
        icon = "✓" if t["reason"] == "take_profit" else "✗"
        print(f"    {icon} {dt}  {t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}"
              f"  {t['pnl_usdt']:>+8.2f}  {t['reason']}")


# ── Synthetic fallback (offline mode) ────────────────────────────────────────

def _synthetic_candles(bars: int, seed: int = 42) -> list[OHLCV]:
    """Generate realistic BTC-like 1H synthetic candles using a regime-switching
    GBM with volatility clustering and fat tails.  Used when exchange is unreachable."""
    import math, random
    random.seed(seed)
    price  = 65_000.0
    ms     = 1_700_000_000_000   # approx Nov 2023 start timestamp
    candles = []
    vol    = 0.008              # starting hourly vol (~1.9% daily)
    regime = "bull"
    for i in range(bars):
        # Regime switch every ~200-600 bars
        if random.random() < 0.003:
            regime = "bear" if regime == "bull" else "bull"
        # Volatility clustering: vol mean-reverts to regime target
        target_vol = 0.007 if regime == "bull" else 0.012
        vol = vol * 0.97 + target_vol * 0.03 + random.gauss(0, 0.001)
        vol = max(0.003, min(vol, 0.025))
        # Fat-tail return: occasionally 3-8× normal move
        tail = 3.0 if random.random() < 0.02 else 1.0
        drift = 0.0001 if regime == "bull" else -0.0001
        ret = drift + random.gauss(0, vol * tail)
        close = price * math.exp(ret)
        # OHLC around close
        spread = abs(random.gauss(0, vol * 0.5)) * price
        high   = max(close, price) + spread * 0.5
        low    = min(close, price) - spread * 0.5
        open_  = price
        vol_v  = abs(random.gauss(1.0, 0.5)) * 3_000.0
        candles.append(OHLCV(ms + i * 3_600_000, open_, high, low, close, vol_v))
        price = close
    return candles


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser(description="Backtest CPK + ProfitableBot on real or synthetic data")
    ap.add_argument("--bars",     type=int,   default=2000,       help="1H bars (default 2000 ≈ 83 days)")
    ap.add_argument("--symbol",   type=str,   default="BTC/USDT", help="Symbol (default BTC/USDT)")
    ap.add_argument("--usdt",     type=float, default=100.0,      help="USDT per trade (default 100)")
    ap.add_argument("--exchange", type=str,   default="okx",      help="Exchange: okx or binance (default okx)")
    ap.add_argument("--offline",  action="store_true",            help="Use synthetic data (no internet needed)")
    args = ap.parse_args()

    if args.offline:
        print(f"[OFFLINE] Generating {args.bars} synthetic BTC 1H candles...")
        raw_1h = _synthetic_candles(args.bars)
        raw_4h = _synthetic_candles(args.bars // 4 + 100, seed=99)
        data_label = "SYNTHETIC (offline)"
    else:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        session   = aiohttp.ClientSession(connector=connector)
        ex_class  = ccxt.okx if args.exchange == "okx" else ccxt.binance
        exchange  = ex_class({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        exchange.session = session
        try:
            print(f"Fetching {args.bars} × 1H candles [{args.symbol}] from {args.exchange.upper()}...")
            raw_1h = await fetch_candles(exchange, args.symbol, "1h", args.bars)
            print(f"  → {len(raw_1h)} bars  "
                  f"({datetime.fromtimestamp(raw_1h[0].timestamp/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
                  f" – {datetime.fromtimestamp(raw_1h[-1].timestamp/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
            n4h = max(600, args.bars // 4 + 100)
            print(f"Fetching {n4h} × 4H candles for MTF guard...")
            raw_4h = await fetch_candles(exchange, args.symbol, "4h", n4h)
            print(f"  → {len(raw_4h)} bars")
        except Exception as e:
            print(f"\n⚠ Exchange unreachable ({e})")
            print("  → Falling back to synthetic data (use --offline to skip this message)\n")
            raw_1h = _synthetic_candles(args.bars)
            raw_4h = _synthetic_candles(args.bars // 4 + 100, seed=99)
            args.offline = True
        finally:
            await exchange.close()
        data_label = f"REAL  {args.symbol}  {args.exchange.upper()}" if not args.offline else "SYNTHETIC (fallback)"

    candles_1h = to_heikin_ashi(raw_1h)
    candles_4h = to_heikin_ashi(raw_4h)

    strategies = [
        ("CPKRegimeStrategy", CPKRegimeStrategy(args.symbol)),
        ("ProfitableBot",     ProfitableBot(args.symbol)),
    ]

    bar   = "=" * 56
    title = f"  BACKTEST  {data_label}  (${args.usdt:.0f}/trade)"
    print(f"\n{bar}\n{title}\n{bar}")

    all_trades = []
    for name, strat in strategies:
        trades = await run_strategy(name, strat, candles_1h, candles_4h, args.usdt)
        print_report(name, trades, raw_1h, args.usdt)
        all_trades.extend(trades)

    combined_pnl = sum(t["pnl_usdt"] for t in all_trades)
    print(f"\n{bar}")
    print(f"  COMBINED P&L : ${combined_pnl:+,.2f}  ({len(all_trades)} total trades)")
    print(f"{bar}\n")


if __name__ == "__main__":
    asyncio.run(main())
