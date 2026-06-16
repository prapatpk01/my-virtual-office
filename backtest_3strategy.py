"""
Walk-Forward Backtest: MCDX + Sentinel + AI Signal (stub) vs TRX/USDT 1H
3 Stages × 250 bars ≈ 1 month of real market data

ข้อมูล: Yahoo Finance (TRX-USD 1H)
Walk-forward: แต่ละ stage ใช้ข้อมูลก่อนหน้าเป็น warmup (expanding window)
AI Signal: ใช้ stub (RSI + trend) แทน Claude API เพื่อประหยัดค่า API

Usage:
    python backtest_3strategy.py
"""
import asyncio
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy
from app.trading.strategies.sentinel_strategy import SentinelStrategy
from app.trading.strategies.base import BaseStrategy, Signal, SignalType

# ── Config ──────────────────────────────────────────────────────────────────
SYMBOL_YF   = "TRX-USD"
SYMBOL      = "TRX/USDT"
STAGE_BARS  = 250
N_STAGES    = 3
WARMUP      = 200           # indicator warmup bars before stage 1
FETCH_BARS  = WARMUP + STAGE_BARS * N_STAGES   # 950 bars total
LOOKFWD     = 48            # max bars to scan for SL/TP hit
RISK_USD    = 10.0          # fixed risk per trade in USD
DEFAULT_SL_PCT  = 0.015     # 1.5% SL if strategy doesn't provide
DEFAULT_RR      = 1.5       # RR ratio for default SL/TP


# ── AI Signal Stub ───────────────────────────────────────────────────────────
class AISignalStub(BaseStrategy):
    """
    Simplified AI Signal: RSI + EMA trend instead of calling Claude API.
    Used for backtest only — avoids 750 × API call cost.
    Real AISignalStrategy would use Claude claude-sonnet-4-6.
    """
    name = "AI Signal (Stub)"

    def __init__(self, symbol: str):
        super().__init__(symbol)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        if len(candles) < 30:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]
        rsi_arr = self.rsi(closes, 14)
        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)

        curr_rsi  = float(rsi_arr[-1])  if not math.isnan(rsi_arr[-1])  else 50.0
        prev_rsi  = float(rsi_arr[-2])  if not math.isnan(rsi_arr[-2])  else 50.0
        curr_ema20 = float(ema20[-1])   if not math.isnan(ema20[-1])    else current_price
        curr_ema50 = float(ema50[-1])   if not math.isnan(ema50[-1])    else current_price
        prev_ema20 = float(ema20[-2])   if not math.isnan(ema20[-2])    else current_price
        prev_ema50 = float(ema50[-2])   if not math.isnan(ema50[-2])    else current_price

        ema_bull_cross = prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50
        ema_bear_cross = prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50
        rsi_rising = curr_rsi > prev_rsi
        trend_bull = curr_ema20 > curr_ema50

        if (ema_bull_cross or (curr_rsi < 35 and rsi_rising and trend_bull)):
            reason = "EMA20×50 bullish cross" if ema_bull_cross else "RSI OS + bull trend"
            conf = 0.65 if ema_bull_cross else 0.55
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=0.05, reason=f"[AI-Stub] {reason} RSI={curr_rsi:.1f}",
                confidence=conf,
                metadata={"rsi": curr_rsi, "ema20": curr_ema20, "ema50": curr_ema50},
            )

        if (ema_bear_cross or (curr_rsi > 65 and not rsi_rising and not trend_bull)):
            reason = "EMA20×50 bearish cross" if ema_bear_cross else "RSI OB + bear trend"
            conf = 0.65 if ema_bear_cross else 0.55
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=0.05, reason=f"[AI-Stub] {reason} RSI={curr_rsi:.1f}",
                confidence=conf,
                metadata={"rsi": curr_rsi, "ema20": curr_ema20, "ema50": curr_ema50},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[AI-Stub] RSI={curr_rsi:.1f} EMA{'Bull' if trend_bull else 'Bear'}",
        )


# ── Data Fetcher ─────────────────────────────────────────────────────────────
def fetch_candles_yfinance(symbol_yf: str, period: str = "60d", interval: str = "1h") -> list:
    try:
        import yfinance as yf
        print(f"  Fetching {symbol_yf} {interval} ({period}) from Yahoo Finance...")
        ticker = yf.Ticker(symbol_yf)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("Empty dataframe")
        candles = []
        for ts, row in df.iterrows():
            candles.append(OHLCV(
                timestamp=int(ts.timestamp() * 1000),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            ))
        return candles
    except Exception as e:
        print(f"  ⚠ Yahoo Finance unavailable ({e.__class__.__name__}): using synthetic TRX data")
        return None


def fetch_candles_ccxt(symbol: str = "TRX/USDT", interval: str = "1h", limit: int = 950) -> list:
    try:
        import ccxt
        print(f"  Fetching {symbol} {interval} from Binance (ccxt)...")
        ex = ccxt.binance({"options": {"defaultType": "spot"}})
        bars = ex.fetch_ohlcv(symbol, interval, limit=limit)
        candles = []
        for b in bars:
            candles.append(OHLCV(
                timestamp=int(b[0]),
                open=float(b[1]), high=float(b[2]),
                low=float(b[3]),  close=float(b[4]),
                volume=float(b[5]),
            ))
        return candles
    except Exception as e:
        print(f"  ⚠ Binance unavailable ({e.__class__.__name__}): using synthetic TRX data")
        return None


def generate_synthetic_trx(n: int = 950, seed: int = 77) -> list:
    """
    Synthetic TRX/USDT-like 1H candles.
    GBM + trend regimes calibrated to TRX characteristics:
      - Base price ~$0.10, moderate-high crypto volatility
      - Regime length ≈ 7 days (168 bars)
    """
    rng = np.random.default_rng(seed)
    start_price = 0.1050

    closes = [start_price]
    regime_len = 168  # 7 days per regime at 1H
    phases = [
        (+0.0003, 0.014),  # bull trend
        (-0.0002, 0.014),  # bear trend
        ( 0.0001, 0.008),  # sideways
        (+0.0004, 0.018),  # strong bull
        (-0.0003, 0.016),  # strong bear
        ( 0.0000, 0.007),  # tight range
    ]
    for i in range(1, n):
        phase_idx = (i // regime_len) % len(phases)
        drift, vol = phases[phase_idx]
        shock = rng.normal(drift, vol)
        closes.append(max(closes[-1] * (1 + shock), 0.0001))

    bar_ms = 60 * 60 * 1000  # 1 hour in ms
    base_ts = 1_700_000_000_000  # ~Nov 2023 reference

    candles = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        spread = abs(rng.normal(0, c * 0.003))
        h = max(o, c) + spread
        l = min(o, c) - spread
        vol_size = float(rng.uniform(500_000, 5_000_000))
        candles.append(OHLCV(
            timestamp=base_ts + i * bar_ms,
            open=round(o, 6), high=round(h, 6),
            low=round(l, 6),  close=round(c, 6),
            volume=vol_size,
        ))
    return candles


def fetch_candles(symbol_yf: str = "TRX-USD", period: str = "60d",
                  interval: str = "1h", n: int = 950) -> list:
    candles = fetch_candles_yfinance(symbol_yf, period, interval)
    if candles and len(candles) >= 500:
        return candles

    candles = fetch_candles_ccxt("TRX/USDT", interval, n)
    if candles and len(candles) >= 500:
        return candles

    print(f"  ✓ Generating {n} synthetic TRX/USDT 1H bars (GBM + trend regimes)")
    return generate_synthetic_trx(n)


# ── ATR Helper ───────────────────────────────────────────────────────────────
def compute_atr(candles: list, period: int = 14) -> float:
    if len(candles) < period + 1:
        return candles[-1].close * DEFAULT_SL_PCT
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr_vals = trs[-period:]
    return float(np.mean(atr_vals)) if atr_vals else candles[-1].close * DEFAULT_SL_PCT


# ── Exit Scanner ─────────────────────────────────────────────────────────────
def find_exit(candles: list, entry_idx: int, direction: int,
              sl_p: float, tp_p: float) -> tuple:
    """Returns (exit_bar, outcome): +1=TP hit, -1=SL hit, 0=timeout."""
    for j in range(entry_idx + 1, min(entry_idx + LOOKFWD + 1, len(candles))):
        h, l = candles[j].high, candles[j].low
        if direction == 1:          # long
            if l <= sl_p: return j, -1
            if h >= tp_p: return j, +1
        else:                       # short
            if h >= sl_p: return j, -1
            if l <= tp_p: return j, +1
    return min(entry_idx + LOOKFWD, len(candles) - 1), 0


# ── Stage Backtest ────────────────────────────────────────────────────────────
async def run_stage(strategy, candles: list, stage_start: int,
                    stage_end: int) -> list:
    """
    Walk through bars [stage_start, stage_end).
    Each bar: feed candles[0..i] as context (expanding window).
    One trade at a time. Exit bar releases the lock.
    """
    trades = []
    lock_until = -1  # bar index where current trade exits

    for i in range(stage_start, stage_end):
        if i <= lock_until:
            continue

        context = candles[:i + 1]
        current_price = candles[i].close

        try:
            signal = await strategy.analyze(context, current_price)
        except Exception as e:
            continue

        if signal.type == SignalType.HOLD:
            continue

        direction = 1 if signal.type == SignalType.BUY else -1

        # SL/TP: prefer from signal metadata, fallback to ATR-based default
        meta = signal.metadata or {}
        sl_p = meta.get("sl") or meta.get("stop_loss")
        tp_p = meta.get("tp1") or meta.get("take_profit")

        if not sl_p or not tp_p or sl_p <= 0 or tp_p <= 0:
            atr = compute_atr(context[-30:], 14)
            if direction == 1:
                sl_p = current_price - 1.5 * atr
                tp_p = current_price + DEFAULT_RR * 1.5 * atr
            else:
                sl_p = current_price + 1.5 * atr
                tp_p = current_price - DEFAULT_RR * 1.5 * atr

        # Compute actual RR from this trade's SL/TP distances
        sl_dist = abs(current_price - sl_p)
        tp_dist = abs(current_price - tp_p)
        actual_rr = tp_dist / sl_dist if sl_dist > 0 else DEFAULT_RR

        exit_bar, outcome = find_exit(candles, i, direction, sl_p, tp_p)

        # PnL in R
        pnl_r = actual_rr if outcome == 1 else (-1.0 if outcome == -1 else 0.0)

        trades.append({
            "bar":       i,
            "stage_bar": i - stage_start + 1,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry":     current_price,
            "sl":        sl_p,
            "tp":        tp_p,
            "rr":        round(actual_rr, 2),
            "exit_bar":  exit_bar,
            "outcome":   outcome,
            "pnl_r":     pnl_r,
            "confidence": signal.confidence,
            "reason":    signal.reason,
        })

        lock_until = exit_bar

    return trades


# ── Stats Calculator ──────────────────────────────────────────────────────────
def calc_stats(trades: list) -> dict:
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
            "wr": 0.0, "pf": 0.0, "total_r": 0.0, "avg_rr": 0.0,
        }
    wins     = [t for t in trades if t["outcome"] ==  1]
    losses   = [t for t in trades if t["outcome"] == -1]
    timeouts = [t for t in trades if t["outcome"] ==  0]
    decided  = len(wins) + len(losses)
    wr       = len(wins) / max(decided, 1) * 100
    total_r  = sum(t["pnl_r"] for t in trades)
    win_r    = sum(t["rr"] for t in wins)
    loss_r   = len(losses) * 1.0
    pf       = win_r / max(loss_r, 1e-9)
    avg_rr   = np.mean([t["rr"] for t in trades]) if trades else 0.0
    return {
        "trades":   len(trades),
        "wins":     len(wins),
        "losses":   len(losses),
        "timeouts": len(timeouts),
        "wr":       round(wr, 1),
        "pf":       round(pf, 2),
        "total_r":  round(total_r, 2),
        "avg_rr":   round(avg_rr, 2),
        "est_pnl":  round(total_r * RISK_USD, 2),
    }


def print_stage_row(label: str, s: dict):
    t_str  = f"{s['trades']:3d}" if s["trades"] else "  0"
    wl_str = f"{s['wins']}W/{s['losses']}L/{s['timeouts']}T"
    wr_str = f"{s['wr']:.1f}%"
    pf_str = f"{s['pf']:.2f}"
    r_str  = f"{s['total_r']:+.1f}R"
    pnl_str = f"${s['est_pnl']:+.2f}"
    print(f"  {label:<12}  Trades:{t_str}  {wl_str:<14}  "
          f"WR:{wr_str:>6}  PF:{pf_str:>5}  {r_str:>7}  {pnl_str:>9}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    W = 70
    print("\n" + "═" * W)
    print("  WALK-FORWARD BACKTEST  |  TRX/USDT  1H  |  3 × 250 bars ≈ 1 month")
    print("  Strategies: MCDX Plus  |  Sentinel  |  AI Signal (stub)")
    print("═" * W)

    candles = fetch_candles(SYMBOL_YF, period="60d", interval="1h", n=FETCH_BARS)

    # Trim to FETCH_BARS from the end
    if len(candles) > FETCH_BARS:
        candles = candles[-FETCH_BARS:]
    total = len(candles)

    from datetime import datetime, timezone
    ts0 = datetime.fromtimestamp(candles[0].timestamp / 1000, tz=timezone.utc)
    tsN = datetime.fromtimestamp(candles[-1].timestamp / 1000, tz=timezone.utc)
    is_synth = ts0.year < 2024
    data_src = "Synthetic TRX/USDT (GBM)" if is_synth else "Yahoo Finance TRX-USD"
    print(f"\n  ✓ {total} bars  |  {ts0:%Y-%m-%d} → {tsN:%Y-%m-%d}  [{data_src}]")
    print(f"  Price range: ${min(c.low for c in candles):.5f} – "
          f"${max(c.high for c in candles):.5f}")
    print(f"  Warmup: {WARMUP} bars  |  Live: {STAGE_BARS * N_STAGES} bars "
          f"({N_STAGES} × {STAGE_BARS})")

    # Stage boundaries
    stages = []
    for s in range(N_STAGES):
        start = WARMUP + s * STAGE_BARS
        end   = start + STAGE_BARS
        stages.append((start, min(end, total)))

    # Strategy list
    strategies = [
        ("MCDX Plus",       MCDXStrategy(SYMBOL)),
        ("Sentinel",        SentinelStrategy(SYMBOL)),
        ("AI Signal (Stub)", AISignalStub(SYMBOL)),
    ]

    summary_rows = []

    for strat_name, strategy in strategies:
        print(f"\n{'─' * W}")
        print(f"  Strategy: {strat_name}")
        print(f"{'─' * W}")
        print(f"  {'Stage':<12}  {'Trades':>9}  {'W/L/T':<14}  "
              f"{'WR':>6}  {'PF':>5}  {'TotalR':>7}  {'Est P&L':>9}")
        print(f"  {'─'*63}")

        all_trades = []
        for s_idx, (start, end) in enumerate(stages):
            ts_s = datetime.fromtimestamp(candles[start].timestamp / 1000, tz=timezone.utc)
            ts_e = datetime.fromtimestamp(candles[end - 1].timestamp / 1000, tz=timezone.utc)
            label = f"Stage {s_idx+1}"
            print(f"  Running {label} ({ts_s:%m/%d} – {ts_e:%m/%d})...", end="\r")

            trades = await run_stage(strategy, candles, start, end)
            stats  = calc_stats(trades)
            all_trades.extend(trades)
            print_stage_row(label, stats)

        total_stats = calc_stats(all_trades)
        print(f"  {'─'*63}")
        print_stage_row("TOTAL", total_stats)
        summary_rows.append((strat_name, total_stats))

    # ── Grand Summary ────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  GRAND SUMMARY")
    print(f"  {'─'*63}")
    print(f"  {'Strategy':<20}  {'Trades':>6}  {'WR':>6}  "
          f"{'PF':>5}  {'TotalR':>7}  {'Est P&L':>9}  {'Winner'}")
    print(f"  {'─'*63}")

    best_r = max(s["total_r"] for _, s in summary_rows) if summary_rows else 0
    for name, s in summary_rows:
        star = "◀ BEST" if s["total_r"] == best_r and s["trades"] > 0 else ""
        print(f"  {name:<20}  {s['trades']:>6}  {s['wr']:>5.1f}%  "
              f"{s['pf']:>5.2f}  {s['total_r']:>+6.1f}R  "
              f"${s['est_pnl']:>+8.2f}  {star}")

    print(f"\n  ข้อสังเกต:")
    print(f"   • ทดสอบ TRX/USDT 1H [{data_src}] ({ts0:%Y-%m-%d} – {tsN:%Y-%m-%d})")
    print(f"   • AI Signal ใช้ stub (EMA cross + RSI) แทน Claude API")
    print(f"     → ผล AI จริงจะแตกต่างถ้าใช้ ANTHROPIC_API_KEY")
    print(f"   • 'T' = Timeout (ออก position หลัง {LOOKFWD} bars)")
    print(f"   • Risk: ${RISK_USD}/trade  |  Default RR: 1:{DEFAULT_RR}")
    print(f"   • ผล backtest ไม่รับประกันผล live trading")
    print("═" * W + "\n")


if __name__ == "__main__":
    asyncio.run(main())
