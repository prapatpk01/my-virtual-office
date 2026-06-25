"""
Backtest: run SwingReversalPro (Long + Short) on historical futures data.
No real orders — simulates entry/exit from signals + SL/TP on bar high/low.

Usage (from app/ directory):
    python backtest.py
    python backtest.py --days 14
    python backtest.py --symbol BTC/USDT --exchange binance --days 30
    python backtest.py --amount 200 --fee 0.0005

Output: per-strategy breakdown + overall summary (trades, win rate, PnL, drawdown).
"""
import argparse
import asyncio
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from trading.connectors.base import OHLCV, to_heikin_ashi
from trading.strategies import SwingReversalPro
from trading.strategies.base import SignalType


# ── Exchange fetch (public API — no key needed) ───────────────────────────────

async def _fetch(exchange_id: str, symbol: str, tf: str, limit: int) -> list[OHLCV]:
    import ccxt.async_support as ccxt

    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")
    ex = cls({"enableRateLimit": True})
    try:
        rows = await ex.fetch_ohlcv(symbol, tf, limit=limit)
        return [OHLCV(int(r[0]), float(r[1]), float(r[2]),
                      float(r[3]), float(r[4]), float(r[5]))
                for r in rows]
    finally:
        await ex.close()


# ── Trade record ──────────────────────────────────────────────────────────────

class Trade:
    __slots__ = ("strategy", "direction", "entry_ts", "entry_price",
                 "sl", "tp", "tp1", "amount_usdt",
                 "exit_ts", "exit_price", "exit_reason", "pnl_usdt")

    def __init__(self, strategy, direction, entry_ts, entry_price, sl, tp, tp1, amount_usdt):
        self.strategy    = strategy
        self.direction   = direction   # "long" | "short"
        self.entry_ts    = entry_ts
        self.entry_price = entry_price
        self.sl          = sl
        self.tp          = tp
        self.tp1         = tp1         # TP1 = partial exit at 0.8R
        self.amount_usdt = amount_usdt
        self.exit_ts = self.exit_price = self.exit_reason = self.pnl_usdt = None

    def close(self, ts, price, reason, fee_pct):
        self.exit_ts     = ts
        self.exit_price  = price
        self.exit_reason = reason
        if self.direction == "short":
            raw_pnl = self.amount_usdt * (self.entry_price - price) / self.entry_price
        else:
            raw_pnl = self.amount_usdt * (price - self.entry_price) / self.entry_price
        self.pnl_usdt = raw_pnl - fee_pct * self.amount_usdt * 2


# ── Core backtest ─────────────────────────────────────────────────────────────

async def run_backtest(exchange_id: str, symbol: str, days: int,
                       amount_usdt: float, fee_pct: float) -> None:

    print(f"\n{'='*62}")
    print(f"  Backtest  {symbol}  |  {exchange_id.upper()}  |  {days} days")
    print(f"  Amount/trade: ${amount_usdt:.0f}  |  Fee: {fee_pct*100:.3f}% per side")
    print(f"  Strategies: SwingReversalPro_L + SwingReversalPro_S")
    print(f"{'='*62}\n")

    # Warmup bars needed before eval starts
    WU_15M = 60
    WU_1H  = 30
    WU_4H  = 20

    n15m = WU_15M + days * 96 + 20
    n1h  = WU_1H  + days * 24 + 10
    n4h  = WU_4H  + days * 6  + 10

    print("Fetching historical data ...")
    try:
        raw_15m, raw_1h, raw_4h = await asyncio.gather(
            _fetch(exchange_id, symbol, "15m", n15m),
            _fetch(exchange_id, symbol, "1h",  n1h),
            _fetch(exchange_id, symbol, "4h",  n4h),
        )
    except Exception as exc:
        print(f"ERROR fetching data: {exc}")
        return

    ha_15m = to_heikin_ashi(raw_15m)
    ha_1h  = to_heikin_ashi(raw_1h)
    ha_4h  = to_heikin_ashi(raw_4h)

    print(f"  15m: {len(ha_15m)} bars | 1H: {len(ha_1h)} bars | "
          f"4H: {len(ha_4h)} bars\n")

    # Default SwingReversalPro params
    params = {
        "risk_pct": 0.01, "l1_min_score": 5, "l2_min_pass": 5,
        "sl_atr_min": 1.0, "adx_4h_max": 35.0, "adx_no_trade": 15.0,
        "atr_min_ratio": 0.8, "mtf_bias_limit": 50.0,
    }

    strategies = [
        SwingReversalPro(symbol, params={**params, "direction": "long"}),
        SwingReversalPro(symbol, params={**params, "direction": "short"}),
    ]

    eval_start = max(WU_15M, len(ha_15m) - days * 96)

    open_pos: dict[str, Trade] = {}
    closed:   list[Trade]      = []

    print(f"Running {len(strategies)} strategies — "
          f"{len(ha_15m) - eval_start} 15m bars in eval window ...\n")

    for i in range(WU_15M, len(ha_15m)):
        bar = ha_15m[i]
        ts  = bar.timestamp
        cp  = float(bar.close)
        hi  = float(bar.high)
        lo  = float(bar.low)

        # Build MTF slices up to current timestamp
        c15m = ha_15m[:i + 1]
        c1h  = [c for c in ha_1h if c.timestamp <= ts]
        c4h  = [c for c in ha_4h if c.timestamp <= ts]
        mtf  = {"15m": c15m, "1h": c1h, "4h": c4h, "health_score": 75}

        # Check SL/TP on open positions (intra-bar)
        for name in list(open_pos):
            pos = open_pos[name]
            if pos.direction == "long":
                if lo <= pos.sl:
                    pos.close(ts, pos.sl, "SL", fee_pct)
                    closed.append(pos)
                    del open_pos[name]
                elif pos.tp1 and hi >= pos.tp1 and pos.exit_reason is None:
                    # TP1 hit — move SL to breakeven, continue to TP2
                    pos.sl = pos.entry_price
                    pos.tp1 = None   # don't re-trigger
                elif hi >= pos.tp:
                    pos.close(ts, pos.tp, "TP", fee_pct)
                    closed.append(pos)
                    del open_pos[name]
            else:  # short
                if hi >= pos.sl:
                    pos.close(ts, pos.sl, "SL", fee_pct)
                    closed.append(pos)
                    del open_pos[name]
                elif pos.tp1 and lo <= pos.tp1 and pos.exit_reason is None:
                    pos.sl = pos.entry_price
                    pos.tp1 = None
                elif lo <= pos.tp:
                    pos.close(ts, pos.tp, "TP", fee_pct)
                    closed.append(pos)
                    del open_pos[name]

        if i < eval_start:
            continue

        if len(c1h) < WU_1H or len(c4h) < WU_4H:
            continue

        for strat in strategies:
            name = strat.name
            sig  = await strat.analyze(c15m, cp, mtf_candles=mtf)
            meta = sig.metadata or {}
            action = meta.get("action", "")
            direction = "long" if strat.direction == "long" else "short"

            if sig.type == SignalType.BUY and action == "open" and name not in open_pos:
                sl  = meta.get("stop_loss")
                tp  = meta.get("take_profit")
                tp1 = meta.get("tp1")
                if not sl or math.isnan(sl) or sl <= 0:
                    sl = cp * 0.97 if direction == "long" else cp * 1.03
                if not tp or math.isnan(tp) or tp <= 0:
                    tp = cp * 1.015 if direction == "long" else cp * 0.985
                open_pos[name] = Trade(name, direction, ts, cp, sl, tp, tp1, amount_usdt)

            elif sig.type == SignalType.SELL and action == "close" and name in open_pos:
                pos = open_pos.pop(name)
                pos.close(ts, cp, "Signal", fee_pct)
                closed.append(pos)

    # Close any still-open positions at end of data (mark-to-market)
    last_bar = ha_15m[-1]
    for name, pos in open_pos.items():
        pos.close(last_bar.timestamp, float(last_bar.close), "EOT", fee_pct)
        closed.append(pos)

    _print_report(closed, strategies, amount_usdt)


def _print_report(closed: list[Trade], strategies, amount_usdt: float) -> None:
    print(f"{'='*62}")
    print(f"  RESULTS — {len(closed)} trades total")
    print(f"{'='*62}")

    if not closed:
        print("  No trades executed in this window.\n")
        return

    by_strat = defaultdict(list)
    for t in closed:
        by_strat[t.strategy].append(t)

    total_pnl = 0.0
    for strat in strategies:
        name   = strat.name
        trades = by_strat.get(name, [])
        if not trades:
            print(f"\n  [{name}]  — no trades")
            continue
        wins   = [t for t in trades if (t.pnl_usdt or 0) > 0]
        losses = [t for t in trades if (t.pnl_usdt or 0) <= 0]
        pnl    = sum(t.pnl_usdt or 0 for t in trades)
        total_pnl += pnl
        wr     = len(wins) / len(trades) * 100
        avg    = pnl / len(trades)
        exits  = {r: sum(1 for t in trades if t.exit_reason == r)
                  for r in ("SL", "TP", "Signal", "EOT")}
        print(f"\n  [{name}]  ({strat.direction.upper()})")
        print(f"    Trades   : {len(trades)}"
              f"  (W:{len(wins)} L:{len(losses)})  WR: {wr:.1f}%")
        print(f"    PnL      : ${pnl:+.2f}  avg ${avg:+.2f}/trade")
        print(f"    Exits    : SL={exits['SL']} TP={exits['TP']} "
              f"Sig={exits['Signal']} EOT={exits['EOT']}")

    # Overall equity curve + max drawdown
    sorted_trades = sorted(closed, key=lambda t: t.exit_ts or 0)
    equity = float(amount_usdt * len(strategies))
    peak   = equity
    max_dd = 0.0
    for t in sorted_trades:
        equity += t.pnl_usdt or 0
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    all_wins   = [t for t in closed if (t.pnl_usdt or 0) > 0]
    all_losses = [t for t in closed if (t.pnl_usdt or 0) <= 0]
    total_wr   = len(all_wins) / len(closed) * 100
    avg_win    = (sum(t.pnl_usdt or 0 for t in all_wins) / len(all_wins)
                  if all_wins else 0.0)
    avg_loss   = (sum(t.pnl_usdt or 0 for t in all_losses) / len(all_losses)
                  if all_losses else 0.0)
    gross_win  = sum(t.pnl_usdt or 0 for t in all_wins)
    gross_loss = abs(sum(t.pnl_usdt or 0 for t in all_losses))
    pf         = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    print(f"\n{'─'*62}")
    print(f"  OVERALL SUMMARY")
    print(f"{'─'*62}")
    print(f"  Total Trades   : {len(closed)}"
          f"  (W:{len(all_wins)} L:{len(all_losses)})")
    print(f"  Win Rate       : {total_wr:.1f}%")
    print(f"  Total PnL      : ${total_pnl:+.2f}")
    print(f"  Avg Win        : ${avg_win:+.2f}")
    print(f"  Avg Loss       : ${avg_loss:+.2f}")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Max Drawdown   : {max_dd:.1f}%")
    print(f"{'='*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SwingReversalPro backtest")
    p.add_argument("--symbol",   default="BTC/USDT",
                   help="Trading pair (default: BTC/USDT)")
    p.add_argument("--exchange", default="okx",
                   help="Exchange ID: okx | binance | bybit (default: okx)")
    p.add_argument("--days",     type=int,   default=7,
                   help="Backtest window in days (default: 7)")
    p.add_argument("--amount",   type=float, default=100.0,
                   help="USDT per trade (default: 100)")
    p.add_argument("--fee",      type=float, default=0.0005,
                   help="Taker fee per side, e.g. 0.0005 = 0.05%% futures (default: 0.0005)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    asyncio.run(run_backtest(
        exchange_id=args.exchange,
        symbol=args.symbol,
        days=args.days,
        amount_usdt=args.amount,
        fee_pct=args.fee,
    ))
