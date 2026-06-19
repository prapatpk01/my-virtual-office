"""
5-Month Real-Data Backtest — ScalpTrendStrategy & ProfitableBotStrategy
=======================================================================
Fetches real OHLCV from Binance (public endpoint, read-only key used if set).
Walk-forward MTF-aware backtest with rolling 300-bar primary window.

Usage:
    cd /home/user/my-virtual-office/app
    BINANCE_KEY=<key> BINANCE_SECRET=<secret> python backtest_real.py
    # or without key (public data):
    python backtest_real.py
"""
import asyncio
import bisect
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

import ccxt

from trading.connectors.base import OHLCV
from trading.strategies.base import SignalType
from trading.strategies.scalp_trend_strategy import ScalpTrendStrategy
from trading.strategies.profitable_bot_strategy import ProfitableBotStrategy

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

BINANCE_KEY    = os.environ.get("BINANCE_KEY",    "")
BINANCE_SECRET = os.environ.get("BINANCE_SECRET", "")
OKX_KEY        = os.environ.get("EXCHANGE_API_KEY",    "")
OKX_SECRET     = os.environ.get("EXCHANGE_API_SECRET", "")
OKX_PASS       = os.environ.get("EXCHANGE_PASSPHRASE", "")

# Exchange selection: prefer OKX if configured (Railway), else Binance, else public
EXCHANGE_ID = "okx" if OKX_KEY else "binance"
SYMBOL      = "BTC/USDT:USDT" if EXCHANGE_ID == "okx" else "BTC/USDT"
NOTIONAL    = 50.0 * 20           # $50 margin × 20x leverage = $1000 notional
TAKER_FEE_RT = 0.001              # 0.05% in + 0.05% out
MONTHS       = 5
WINDOW_PRIMARY = 300                  # rolling primary candles per analyze call
WINDOW_1H      = 200                  # rolling 1H MTF candles
WINDOW_4H      = 120                  # rolling 4H MTF candles
WARMUP         = 100                  # bars to skip before evaluating signals

# ─────────────────────────────────────────────────────────────
# Data fetching (paginated)
# ─────────────────────────────────────────────────────────────

def _exchange():
    if EXCHANGE_ID == "okx":
        cfg = {
            "apiKey":    OKX_KEY,
            "secret":    OKX_SECRET,
            "password":  OKX_PASS,
            "enableRateLimit": True,
            "options":   {"defaultType": "swap"},
        }
        return ccxt.okx(cfg)
    # Binance fallback (public OHLCV, no auth needed)
    cfg = {
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }
    if BINANCE_KEY:
        cfg["apiKey"] = BINANCE_KEY
        cfg["secret"] = BINANCE_SECRET
    return ccxt.binance(cfg)


def fetch_candles_paginated(symbol: str, tf: str, months: int) -> list[OHLCV]:
    """Fetch full history for `months` months, handling Binance 1000-bar limit."""
    ex = _exchange()

    # How many bars does `months` of this TF need?
    bars_per_hour = {"15m": 4, "1h": 1, "4h": 0.25}
    bph = bars_per_hour.get(tf, 1)
    total_bars_needed = int(months * 30 * 24 * bph) + 50   # +50 buffer

    since_ms = int((time.time() - months * 30 * 24 * 3600) * 1000)

    all_raw: list[list] = []
    fetch_since = since_ms

    print(f"  Fetching {symbol} {tf} (~{total_bars_needed} bars) ...")
    while True:
        raw = ex.fetch_ohlcv(symbol, timeframe=tf, since=fetch_since, limit=1000)
        if not raw:
            break
        all_raw.extend(raw)
        if len(raw) < 1000:
            break          # reached present
        fetch_since = raw[-1][0] + 1   # next ms after last bar
        time.sleep(0.2)   # rate limit courtesy

    # Deduplicate (overlaps can happen at boundaries)
    seen_ts: set[int] = set()
    candles: list[OHLCV] = []
    for ts, o, h, l, c, v in all_raw:
        if ts not in seen_ts:
            seen_ts.add(ts)
            candles.append(OHLCV(timestamp=int(ts), open=float(o), high=float(h),
                                 low=float(l), close=float(c), volume=float(v)))
    candles.sort(key=lambda x: x.timestamp)
    print(f"    → {len(candles)} bars  "
          f"({datetime.fromtimestamp(candles[0].timestamp/1000, tz=timezone.utc).strftime('%Y-%m-%d')} "
          f"→ {datetime.fromtimestamp(candles[-1].timestamp/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
    return candles


# ─────────────────────────────────────────────────────────────
# SL/TP helper (same logic as backtester.py)
# ─────────────────────────────────────────────────────────────

@dataclass
class BTrade:
    symbol: str
    strategy: str
    side: str
    entry: float
    sl: float
    tp: float
    entry_ts: int
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_ts: int = 0
    pnl_usdt: float = 0.0


def _get_sl_tp(signal, price: float, side: str) -> tuple[float, float]:
    meta = signal.metadata or {}
    sl = meta.get("stop_loss")
    tp = meta.get("take_profit")
    if sl is not None and tp is not None:
        if side == "short":
            if sl < price: sl = None
            if tp > price: tp = None
        if sl is not None and tp is not None:
            return sl, tp
    # Fallback 1.5% / 2.5%
    if side == "long":
        return round(price * 0.985, 2), round(price * 1.025, 2)
    return round(price * 1.015, 2), round(price * 0.975, 2)


# ─────────────────────────────────────────────────────────────
# MTF-aware walk-forward backtester
# ─────────────────────────────────────────────────────────────

async def backtest_mtf(
    strategy,
    primary: list[OHLCV],
    mtf: dict[str, list[OHLCV]],
    notional: float,
    warmup: int = WARMUP,
    w_primary: int = WINDOW_PRIMARY,
    w_1h: int = WINDOW_1H,
    w_4h: int = WINDOW_4H,
) -> list[BTrade]:
    """
    Walk-forward with MTF. For bar i in primary:
      - Primary slice: primary[max(0,i-w_primary):i]
      - MTF slice    : last w_Xh bars of each MTF up to primary[i].timestamp
    Uses bisect for O(log n) MTF slice lookups.
    """
    trades: list[BTrade] = []
    open_long:  Optional[BTrade] = None
    open_short: Optional[BTrade] = None

    # Pre-index MTF timestamps for bisect
    mtf_ts = {tf: [c.timestamp for c in clist] for tf, clist in mtf.items()}

    win_map = {"1h": w_1h, "4h": w_4h}

    total = len(primary)
    report_every = max(1, total // 20)

    for i in range(warmup, total):
        bar    = primary[i]
        ts     = bar.timestamp
        price  = bar.close
        b_high = bar.high
        b_low  = bar.low

        if (i - warmup) % report_every == 0:
            pct = (i - warmup) / (total - warmup) * 100
            print(f"    {pct:5.1f}%  bar={i}/{total}  trades={len(trades)}", end="\r")

        # ── Exit checks ───────────────────────────────────────────────────
        for pos, side in [(open_long, "long"), (open_short, "short")]:
            if pos is None:
                continue
            hit_reason = hit_price = None
            if side == "long":
                if b_low  <= pos.sl:   hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_high >= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            else:
                if b_high >= pos.sl:  hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_low  <= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            if hit_reason:
                mult        = 1 if side == "long" else -1
                pnl_pct     = mult * (hit_price - pos.entry) / pos.entry
                fee         = notional * TAKER_FEE_RT
                pos.exit_price  = hit_price
                pos.exit_reason = hit_reason
                pos.exit_ts     = ts
                pos.pnl_usdt    = round(pnl_pct * notional - fee, 4)
                trades.append(pos)
                if side == "long":  open_long  = None
                else:               open_short = None

        # ── Build candle slices ───────────────────────────────────────────
        prim_slice = primary[max(0, i - w_primary): i]

        mtf_sliced: dict[str, list[OHLCV]] = {}
        for tf, ts_list in mtf_ts.items():
            idx_end = bisect.bisect_right(ts_list, ts)  # bars up to current ts
            win     = win_map.get(tf, 150)
            idx_start = max(0, idx_end - win)
            mtf_sliced[tf] = mtf[tf][idx_start:idx_end]

        # ── Strategy signal ───────────────────────────────────────────────
        try:
            signal = await strategy.analyze(prim_slice, price, mtf_candles=mtf_sliced)
        except Exception as e:
            continue

        if signal.type == SignalType.BUY and open_long is None and open_short is None:
            sl, tp = _get_sl_tp(signal, price, "long")
            open_long = BTrade(symbol=strategy.symbol, strategy=strategy.name,
                               side="long", entry=price, sl=sl, tp=tp, entry_ts=ts)
        elif signal.type == SignalType.SELL and open_short is None and open_long is None:
            sl, tp = _get_sl_tp(signal, price, "short")
            open_short = BTrade(symbol=strategy.symbol, strategy=strategy.name,
                                side="short", entry=price, sl=sl, tp=tp, entry_ts=ts)

    print()   # newline after progress
    return trades


# ─────────────────────────────────────────────────────────────
# Summary & display
# ─────────────────────────────────────────────────────────────

def summarise(trades: list[BTrade]) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "net_usdt": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "max_dd": 0.0, "longs": 0, "shorts": 0}
    wins   = [t for t in trades if t.pnl_usdt > 0]
    losses = [t for t in trades if t.pnl_usdt <= 0]
    longs  = [t for t in trades if t.side == "long"]
    shorts = [t for t in trades if t.side == "short"]
    net    = sum(t.pnl_usdt for t in trades)
    wr     = len(wins) / len(trades) * 100
    gw     = sum(t.pnl_usdt for t in wins)
    gl     = abs(sum(t.pnl_usdt for t in losses)) or 1e-8

    # Max drawdown (equity curve)
    eq = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        eq += t.pnl_usdt
        if eq > peak: peak = eq
        dd = (peak - eq) / max(peak, 1e-8) * 100
        if dd > max_dd: max_dd = dd

    return {
        "trades":         len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(wr, 1),
        "profit_factor":  round(gw / gl, 2),
        "net_usdt":       round(net, 2),
        "avg_win":        round(gw / len(wins)   if wins   else 0, 2),
        "avg_loss":       round(-gl / len(losses) if losses else 0, 2),
        "max_dd_pct":     round(max_dd, 1),
        "longs":          len(longs),
        "shorts":         len(shorts),
    }


def print_result(name: str, s: dict, notional: float):
    if s["trades"] == 0:
        print(f"  {name:30} │ NO SIGNALS")
        return
    roi = s["net_usdt"] / notional * 100 if notional else 0
    sign = "+" if s["net_usdt"] >= 0 else ""
    icon = "✅" if s["net_usdt"] >= 0 else "❌"
    print(f"  {icon} {name:28} │ "
          f"T={s['trades']:3d}  WR={s['win_rate']:5.1f}%  PF={s['profit_factor']:5.2f}  "
          f"Net={sign}{s['net_usdt']:7.2f}$  ROI={sign}{roi:.1f}%  "
          f"MaxDD={s['max_dd_pct']:.1f}%  L={s['longs']} S={s['shorts']}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main():
    print("=" * 90)
    print(f"  5-Month Real Backtest  │  {SYMBOL}  │  Notional=${NOTIONAL:.0f}  │  Fee={TAKER_FEE_RT*100:.2f}%RT")
    print(f"  Strategies: ScalpTrendStrategy (15m/1H/4H)  +  ProfitableBotStrategy (1H/4H)")
    print("=" * 90)

    # ── Fetch candles ─────────────────────────────────────────────────────────
    print("\nFetching real Binance OHLCV data ...")
    candles_15m = fetch_candles_paginated(SYMBOL, "15m", MONTHS)
    candles_1h  = fetch_candles_paginated(SYMBOL, "1h",  MONTHS)
    candles_4h  = fetch_candles_paginated(SYMBOL, "4h",  MONTHS)

    print(f"\n  15m: {len(candles_15m)} bars  "
          f"1h: {len(candles_1h)} bars  "
          f"4h: {len(candles_4h)} bars\n")

    results: dict[str, dict] = {}

    # ── ScalpTrendStrategy (15m primary, 1H+4H MTF) ──────────────────────────
    print("─" * 90)
    print("  ScalpTrendStrategy  (15m entry / 1H+4H trend filter)")
    print("─" * 90)
    strat_scalp = ScalpTrendStrategy(SYMBOL, params={
        "name":    "ScalpTrend",
        "tf":      "15m",
        "sl_mult": 0.8,
        "tp_mult": 1.0,
    })
    print(f"  Walking forward {len(candles_15m)} bars (warmup={WARMUP}) ...")
    t0 = time.time()
    trades_scalp = await backtest_mtf(
        strat_scalp, candles_15m,
        {"1h": candles_1h, "4h": candles_4h},
        NOTIONAL,
    )
    elapsed = time.time() - t0
    r = summarise(trades_scalp)
    results["ScalpTrend/BTC"] = r
    print_result("ScalpTrend/BTC (15m/1H/4H)", r, NOTIONAL)
    print(f"  [done in {elapsed:.1f}s]")

    # ── ProfitableBotStrategy (1H primary, 4H MTF) ────────────────────────────
    print()
    print("─" * 90)
    print("  ProfitableBotStrategy  (1H entry / 4H trend filter)")
    print("─" * 90)
    strat_prof = ProfitableBotStrategy(SYMBOL, params={
        "name":    "ProfBot",
        "tf":      "1h",
        "sl_mult": 1.5,
        "tp_mult": 2.5,
    })
    print(f"  Walking forward {len(candles_1h)} bars (warmup={WARMUP}) ...")
    t0 = time.time()
    trades_prof = await backtest_mtf(
        strat_prof, candles_1h,
        {"4h": candles_4h},
        NOTIONAL,
        w_primary=WINDOW_1H,
    )
    elapsed = time.time() - t0
    r = summarise(trades_prof)
    results["ProfBot/BTC"] = r
    print_result("ProfBot/BTC  (1H/4H)",       r, NOTIONAL)
    print(f"  [done in {elapsed:.1f}s]")

    # ── Combined summary ──────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  COMBINED SUMMARY")
    print("=" * 90)
    print(f"  {'Strategy':30} │ {'Trades':>6}  {'WR%':>6}  {'PF':>5}  "
          f"{'Net $':>9}  {'ROI':>7}  {'MaxDD':>7}  {'L/S':>5}")
    print(f"  {'─'*30}─┼─{'─'*6}──{'─'*6}──{'─'*5}──{'─'*9}──{'─'*7}──{'─'*7}──{'─'*5}")
    all_net = 0.0
    all_t   = 0
    for name, s in results.items():
        roi  = s["net_usdt"] / NOTIONAL * 100
        sign = "+" if s["net_usdt"] >= 0 else ""
        icon = "✅" if s["net_usdt"] >= 0 else "❌"
        print(f"  {icon} {name:28} │ {s['trades']:>6}  {s['win_rate']:>5.1f}%  "
              f"{s['profit_factor']:>5.2f}  {sign}{s['net_usdt']:>8.2f}$  "
              f"{sign}{roi:>6.1f}%  {s['max_dd_pct']:>6.1f}%  "
              f"{s['longs']:>2}L/{s['shorts']:>2}S")
        all_net += s["net_usdt"]
        all_t   += s["trades"]
    print(f"  {'─'*30}─┼─{'─'*6}──{'─'*6}──{'─'*5}──{'─'*9}──{'─'*7}──{'─'*7}──{'─'*5}")
    sign = "+" if all_net >= 0 else ""
    all_roi = all_net / (NOTIONAL * len(results)) * 100 if results else 0
    print(f"  {'TOTAL (combined)':30} │ {all_t:>6}  {'':>6}  {'':>5}  "
          f"{sign}{all_net:>8.2f}$  {sign}{all_roi:>6.1f}%")

    print()
    print(f"  Period : last {MONTHS} months of real BTC/USDT (Binance)")
    print(f"  Margin : $50 × 20x = ${NOTIONAL:.0f} notional per trade")
    print(f"  Fee    : {TAKER_FEE_RT*100:.3f}% per round-trip")
    print(f"  SL/TP  : from strategy metadata (ATR-based)")

    # ── Trade breakdown per strategy ──────────────────────────────────────────
    for trades, name in [(trades_scalp, "ScalpTrend"), (trades_prof, "ProfBot")]:
        if not trades:
            continue
        tp_wins  = [t for t in trades if t.exit_reason == "take_profit"]
        sl_hits  = [t for t in trades if t.exit_reason == "stop_loss"]
        longs_w  = [t for t in trades if t.side == "long"  and t.pnl_usdt > 0]
        shorts_w = [t for t in trades if t.side == "short" and t.pnl_usdt > 0]
        longs_t  = [t for t in trades if t.side == "long"]
        shorts_t = [t for t in trades if t.side == "short"]
        print(f"\n  ── {name} breakdown ─────────────────────")
        print(f"    TP hits : {len(tp_wins):3d}  │  SL hits : {len(sl_hits):3d}")
        if longs_t:
            print(f"    Longs   : {len(longs_t):3d}  WR {len(longs_w)/len(longs_t)*100:.0f}%")
        if shorts_t:
            print(f"    Shorts  : {len(shorts_t):3d}  WR {len(shorts_w)/len(shorts_t)*100:.0f}%")
        # Recent 5 trades
        print(f"    Last 5 trades:")
        for t in trades[-5:]:
            dt = datetime.fromtimestamp(t.entry_ts/1000, tz=timezone.utc).strftime("%m-%d %H:%M")
            sign = "+" if t.pnl_usdt > 0 else ""
            print(f"      {dt}  {t.side:5s}  entry={t.entry:.0f}  "
                  f"{t.exit_reason:12s}  {sign}{t.pnl_usdt:.2f}$")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
