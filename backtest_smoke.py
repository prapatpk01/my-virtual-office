"""Dependency-free synthetic backtest smoke test.

This does not replace the full MCDX/AI strategy backtests, which require the
trading dependencies in requirements-trading.txt. It gives CI and constrained
sandboxes a quick sanity check for signal generation, position locking, SL/TP
accounting, and summary reporting without numpy/ccxt/aiohttp.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Trade:
    day: int
    symbol: str
    side: str
    entry: float
    exit: float
    pnl: float
    reason: str


SYMBOLS = [
    ("BTC/USDT", 65_000.0, 0.0015, 0.00030, 42),
    ("ETH/USDT",  3_500.0, 0.0018, 0.00025, 43),
]
DAYS = 7
BARS_PER_DAY = 96
WARMUP = 80
LOOKFWD = 48
START_CAPITAL = 500.0
RISK_USD = 10.0
RR = 1.5


def generate_candles(symbol: str, start: float, sigma: float, drift: float, seed: int) -> list[Candle]:
    rng = random.Random(seed)
    candles: list[Candle] = []
    price = start
    total = WARMUP + DAYS * BARS_PER_DAY + LOOKFWD
    for i in range(total):
        phase = (i // BARS_PER_DAY) % 4
        phase_drift = drift if phase in (0, 1) else -drift if phase == 2 else 0.0
        open_ = price
        ret = phase_drift + rng.gauss(0.0, sigma)
        close = max(0.0001, open_ * math.exp(ret))
        wiggle = abs(rng.gauss(0.0, sigma * 0.6))
        high = max(open_, close) * (1 + wiggle)
        low = min(open_, close) * (1 - wiggle)
        candles.append(Candle(i * 15 * 60 * 1000, open_, high, low, close, rng.uniform(10, 100)))
        price = close
    return candles


def sma(values: list[float], period: int, idx: int) -> float | None:
    if idx + 1 < period:
        return None
    return sum(values[idx - period + 1:idx + 1]) / period


def rsi(values: list[float], period: int, idx: int) -> float | None:
    if idx < period:
        return None
    gains = losses = 0.0
    for i in range(idx - period + 1, idx + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def signal(closes: list[float], idx: int) -> str:
    fast = sma(closes, 12, idx)
    slow = sma(closes, 36, idx)
    fast_prev = sma(closes, 12, idx - 1)
    slow_prev = sma(closes, 36, idx - 1)
    rsi_now = rsi(closes, 14, idx)
    if None in (fast, slow, fast_prev, slow_prev, rsi_now):
        return "hold"
    if fast_prev <= slow_prev and fast > slow and rsi_now < 72:
        return "buy"
    if fast_prev >= slow_prev and fast < slow and rsi_now > 28:
        return "sell"
    return "hold"


def simulate_symbol(symbol: str, candles: list[Candle]) -> list[Trade]:
    closes = [c.close for c in candles]
    trades: list[Trade] = []
    locked_until = -1
    for i in range(WARMUP, WARMUP + DAYS * BARS_PER_DAY):
        if i <= locked_until:
            continue
        sig = signal(closes, i)
        if sig == "hold":
            continue
        entry = candles[i].close
        stop_pct = 0.01
        if sig == "buy":
            sl = entry * (1 - stop_pct)
            tp = entry * (1 + stop_pct * RR)
        else:
            sl = entry * (1 + stop_pct)
            tp = entry * (1 - stop_pct * RR)
        exit_price = candles[min(i + LOOKFWD, len(candles) - 1)].close
        reason = "timeout"
        pnl = 0.0
        for j in range(i + 1, min(i + LOOKFWD, len(candles))):
            if sig == "buy" and candles[j].low <= sl:
                exit_price, pnl, reason, locked_until = sl, -RISK_USD, "sl", j
                break
            if sig == "buy" and candles[j].high >= tp:
                exit_price, pnl, reason, locked_until = tp, RISK_USD * RR, "tp", j
                break
            if sig == "sell" and candles[j].high >= sl:
                exit_price, pnl, reason, locked_until = sl, -RISK_USD, "sl", j
                break
            if sig == "sell" and candles[j].low <= tp:
                exit_price, pnl, reason, locked_until = tp, RISK_USD * RR, "tp", j
                break
        else:
            locked_until = min(i + LOOKFWD, len(candles) - 1)
            change = (exit_price - entry) / entry
            pnl = RISK_USD * (change / stop_pct) * (1 if sig == "buy" else -1)
        day = (i - WARMUP) // BARS_PER_DAY + 1
        trades.append(Trade(day, symbol, sig, entry, exit_price, pnl, reason))
    return trades


def main() -> None:
    all_trades: list[Trade] = []
    for symbol, start, sigma, drift, seed in SYMBOLS:
        all_trades.extend(simulate_symbol(symbol, generate_candles(symbol, start, sigma, drift, seed)))
    all_trades.sort(key=lambda t: (t.day, t.symbol))

    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    total_pnl = gross_win - gross_loss
    balance = START_CAPITAL + total_pnl
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    pf = gross_win / gross_loss if gross_loss else float("inf")

    print("=" * 72)
    print("  DEPENDENCY-FREE BACKTEST SMOKE | synthetic 15m | MA/RSI signal")
    print("=" * 72)
    print(f"  Trades      : {len(all_trades)}")
    print(f"  Wins/Losses : {len(wins)}W / {len(losses)}L")
    print(f"  Win Rate    : {win_rate:.1f}%")
    print(f"  ProfitFactor: {pf:.2f}")
    print(f"  PnL         : {total_pnl:+.2f} USD")
    print(f"  Balance     : ${balance:.2f} from ${START_CAPITAL:.2f}")
    print("-" * 72)
    for day in range(1, DAYS + 1):
        day_trades = [t for t in all_trades if t.day == day]
        day_pnl = sum(t.pnl for t in day_trades)
        print(f"  Day {day}: {len(day_trades):2d} trades | PnL {day_pnl:+7.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
