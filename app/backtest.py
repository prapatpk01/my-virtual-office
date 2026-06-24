"""
Backtest: run all 3 strategies on historical BTC/USDT data.
No real orders — simulates entry/exit from signals + SL/TP on bar high/low.

Usage (from app/ directory):
    python backtest.py
    python backtest.py --days 14
    python backtest.py --symbol BTC/USDT --exchange binance --days 30
    python backtest.py --amount 200 --fee 0.0008

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
from trading.strategies import SpotMaster1H, SmartGridHybrid, SwingMaster30m
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
    __slots__ = ("strategy", "entry_ts", "entry_price",
                 "sl", "tp", "amount_usdt",
                 "exit_ts", "exit_price", "exit_reason", "pnl_usdt")

    def __init__(self, strategy, entry_ts, entry_price, sl, tp, amount_usdt):
        self.strategy    = strategy
        self.entry_ts    = entry_ts
        self.entry_price = entry_price
        self.sl          = sl
        self.tp          = tp
        self.amount_usdt = amount_usdt
        self.exit_ts = self.exit_price = self.exit_reason = self.pnl_usdt = None

    def close(self, ts, price, reason, fee_pct):
        self.exit_ts     = ts
        self.exit_price  = price
        self.exit_reason = reason
        raw_pnl          = self.amount_usdt * (price - self.entry_price) / self.entry_price
        self.pnl_usdt    = raw_pnl - fee_pct * self.amount_usdt * 2   # entry + exit fee


# ── Core backtest ─────────────────────────────────────────────────────────────

async def run_backtest(exchange_id: str, symbol: str, days: int,
                       amount_usdt: float, fee_pct: float) -> None:

    print(f"\n{'='*62}")
    print(f"  Backtest  {symbol}  |  {exchange_id.upper()}  |  {days} days")
    print(f"  Amount/trade: ${amount_usdt:.0f}  |  Fee: {fee_pct*100:.2f}% per side")
    print(f"{'='*62}\n")

    # Warmup constants (bars needed before eval starts)
    WU_30M = 60   # SmartGrid / SwingMaster
    WU_1H  = 50   # SpotMaster1H internal indicators
    WU_4H  = 50   # 4H regime
    WU_1D  = 55   # 1D macro

    # Bars to fetch (warmup + eval window + buffer)
    n30m = WU_30M + days * 48 + 20
    n1h  = WU_1H  + days * 24 + 10
    n4h  = WU_4H  + days * 6  + 10
    n1d  = max(WU_1D + 5, 70)          # always 70 days of daily bars

    print("Fetching historical data ...")
    try:
        raw_30m, raw_1h, raw_4h, raw_1d = await asyncio.gather(
            _fetch(exchange_id, symbol, "30m", n30m),
            _fetch(exchange_id, symbol, "1h",  n1h),
            _fetch(exchange_id, symbol, "4h",  n4h),
            _fetch(exchange_id, symbol, "1d",  n1d),
        )
    except Exception as exc:
        print(f"ERROR fetching data: {exc}")
        return

    ha_30m = to_heikin_ashi(raw_30m)
    ha_1h  = to_heikin_ashi(raw_1h)
    ha_4h  = to_heikin_ashi(raw_4h)
    ha_1d  = to_heikin_ashi(raw_1d)

    print(f"  30m: {len(ha_30m)} bars | 1H: {len(ha_1h)} bars | "
          f"4H: {len(ha_4h)} bars | 1D: {len(ha_1d)} bars\n")

    # Build strategies (fresh instances = clean state)
    spot_p  = {"sl_atr": 1.5, "rr": 2.0, "adx_thresh": 18.0, "health_thresh": 70.0}
    grid_p  = {"sl_atr": 1.8, "health_thresh": 70.0, "adx_swing": 18.0,
                "adx_breakout": 28.0, "adx_grid_kill": 22.0,
                "max_grid_levels": 5, "grid_step_atr": 0.8}
    swing_p = {"sl_atr": 1.8, "adx_thresh": 18.0,
                "health_thresh": 70.0, "allow_short": False}

    strategies = [
        SpotMaster1H(symbol, params=spot_p),
        SmartGridHybrid(symbol, params={**grid_p, "grid_slot": 0}),
        SmartGridHybrid(symbol, params={**grid_p, "grid_slot": 1}),
        SmartGridHybrid(symbol, params={**grid_p, "grid_slot": 2}),
        SwingMaster30m(symbol, params=swing_p),
    ]

    # Evaluation window: last `days` × 48 bars (or as many as available after warmup)
    eval_start = max(WU_30M, len(ha_30m) - days * 48)

    open_pos: dict[str, Trade] = {}   # strategy_name → Trade
    closed:   list[Trade]      = []

    print(f"Running {len(strategies)} strategies — "
          f"{len(ha_30m) - eval_start} 30m bars in eval window ...\n")

    for i in range(WU_30M, len(ha_30m)):
        bar    = ha_30m[i]
        ts     = bar.timestamp
        cp     = float(bar.close)
        hi     = float(bar.high)
        lo     = float(bar.low)

        # ── Build MTF slices up to current timestamp ──────────────────────
        c30m = ha_30m[:i + 1]
        c1h  = [c for c in ha_1h if c.timestamp <= ts]
        c4h  = [c for c in ha_4h if c.timestamp <= ts]
        c1d  = [c for c in ha_1d if c.timestamp <= ts]
        mtf  = {"30m": c30m, "4h": c4h, "1d": c1d}

        # ── Check SL/TP on open positions (intra-bar, bar's high/low) ────
        for name in list(open_pos):
            pos = open_pos[name]
            if lo <= pos.sl:
                pos.close(ts, pos.sl, "SL", fee_pct)
                closed.append(pos)
                del open_pos[name]
            elif hi >= pos.tp:
                pos.close(ts, pos.tp, "TP", fee_pct)
                closed.append(pos)
                del open_pos[name]

        # ── Only generate signals in the eval window ──────────────────────
        if i < eval_start:
            continue

        # Skip if not enough 4H / 1D bars for reliable signals
        if len(c4h) < WU_4H or len(c1d) < WU_1D:
            continue

        for strat in strategies:
            name = strat.name
            # SpotMaster1H uses 1H candles; others use 30m
            primary = c1h if isinstance(strat, SpotMaster1H) else c30m
            if not primary:
                continue

            sig = await strat.analyze(primary, cp, mtf)

            if sig.type == SignalType.BUY and name not in open_pos:
                sl = sig.metadata.get("stop_loss")  if sig.metadata else None
                tp = sig.metadata.get("take_profit") if sig.metadata else None
                if not sl or math.isnan(sl) or sl <= 0:
                    sl = cp * 0.97
                if not tp or math.isnan(tp) or tp <= 0:
                    tp = cp * 1.04
                open_pos[name] = Trade(name, ts, cp, sl, tp, amount_usdt)

            elif sig.type == SignalType.SELL and name in open_pos:
                pos = open_pos.pop(name)
                pos.close(ts, cp, "Signal", fee_pct)
                closed.append(pos)

    # Close any positions still open at end of data (mark-to-market)
    last_bar = ha_30m[-1]
    for name, pos in open_pos.items():
        pos.close(last_bar.timestamp, float(last_bar.close), "EOT", fee_pct)
        closed.append(pos)

    # ── Report ────────────────────────────────────────────────────────────────
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
    for name in [s.name for s in strategies]:
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
        print(f"\n  [{name}]")
        print(f"    Trades   : {len(trades)}"
              f"  (W:{len(wins)} L:{len(losses)})  WR: {wr:.1f}%")
        print(f"    PnL      : ${pnl:+.2f}  avg ${avg:+.2f}/trade")
        print(f"    Exits    : SL={exits['SL']} TP={exits['TP']} "
              f"Sig={exits['Signal']} EOT={exits['EOT']}")

    # Overall equity curve + max drawdown
    sorted_trades = sorted(closed, key=lambda t: t.exit_ts or 0)
    equity = float(amount_usdt * len([s for s in by_strat]))
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
    p = argparse.ArgumentParser(description="7-day strategy backtest")
    p.add_argument("--symbol",   default="BTC/USDT",
                   help="Trading pair (default: BTC/USDT)")
    p.add_argument("--exchange", default="okx",
                   help="Exchange ID: okx | binance | bybit (default: okx)")
    p.add_argument("--days",     type=int,   default=7,
                   help="Backtest window in days (default: 7)")
    p.add_argument("--amount",   type=float, default=100.0,
                   help="USDT per trade (default: 100)")
    p.add_argument("--fee",      type=float, default=0.001,
                   help="Taker fee per side, e.g. 0.001 = 0.1%% (default: 0.001)")
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
