"""
Full 7-Day Backtest  |  MCDXStrategy (Adaptive Trading Bot) × 3 Symbols
Capital simulation: $500 start, $10 risk/trade.

Models real bot behaviour:
  - One open trade per symbol at a time (position locked until exit)
  - Exit bar determined by first SL/TP hit or timeout (60 bars)
  - SL = 1.5×ATR, TP = SL × RR (1:1.5)
"""
import asyncio
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.mcdx_strategy import MCDXStrategy
from app.trading.strategies.base import SignalType

# ─── Constants ─────────────────────────────────────────────────────────────
WARMUP   = 285
DAYS     = 7
BPD      = 96           # 15m bars per day
N_BARS   = WARMUP + DAYS * BPD
LOOKFWD  = 60           # max bars to scan for SL/TP exit
RISK     = 10.0
CAP_0    = 500.0
ATR_P    = 14
RR_RATIO = 1.5
SL_MULT  = 1.5

SYMBOLS = [
    ("BTC/USDT", 65_000.0, 0.0015, 0.0008, 0.00030, 42),
    ("XAUUSD",   2_050.0,  0.0008, 0.0004, 0.00020, 43),
    ("EURUSD",      1.085, 0.0004, 0.0002, 0.00010, 44),
]


# ─── Candle Generator ──────────────────────────────────────────────────────
def gen_candles(n, start, vt, vr, drift, seed):
    rng = np.random.default_rng(seed)
    closes = [float(start)]
    regime = 96
    for i in range(1, n):
        ph = (i // regime) % 6
        mu, sd = (+drift, vt) if ph < 2 else ((-drift, vt) if ph < 4 else (0.0, vr))
        closes.append(closes[-1] * (1 + rng.normal(mu, sd)))
    candles = []
    for i, c in enumerate(closes):
        o  = closes[i-1] if i > 0 else c
        sp = abs(rng.normal(0, c * 0.0005))
        candles.append(OHLCV(
            timestamp=i * 15 * 60 * 1000,
            open=round(o, 6), high=round(max(o, c) + sp, 6),
            low=round(min(o, c) - sp, 6), close=round(c, 6),
            volume=float(rng.uniform(5, 50)),
        ))
    return candles


# ─── ATR helper ────────────────────────────────────────────────────────────
def calc_atr(candles, period=ATR_P):
    n = len(candles)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        h = candles[i].high; l = candles[i].low; pc = candles[i-1].close
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = float(np.nanmean(tr[1:period+1]))
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


# ─── Find trade exit ───────────────────────────────────────────────────────
def find_exit(highs, lows, idx, direction, sl_p, tp_p):
    n = len(highs)
    for j in range(idx + 1, min(idx + LOOKFWD, n)):
        if direction == 1:
            if lows[j]  <= sl_p: return j, -1
            if highs[j] >= tp_p: return j,  1
        else:
            if highs[j] >= sl_p: return j, -1
            if lows[j]  <= tp_p: return j,  1
    return min(idx + LOOKFWD, n - 1), 0


# ─── Main Backtest ─────────────────────────────────────────────────────────
async def run_backtest():
    all_signals = []

    for sym, start, vt, vr, drift, seed in SYMBOLS:
        candles = gen_candles(N_BARS, start, vt, vr, drift, seed)
        highs   = np.array([c.high  for c in candles])
        lows    = np.array([c.low   for c in candles])
        atr_a   = calc_atr(candles)

        strat = MCDXStrategy(sym)
        locked_until = -1

        for i in range(WARMUP, N_BARS - 1):
            if i <= locked_until:
                continue
            atr_v = float(atr_a[i]) if not np.isnan(atr_a[i]) else 0.0
            if atr_v <= 0:
                continue

            window = candles[:i + 1]
            sig = await strat.analyze(window, window[-1].close)
            entry = window[-1].close
            day = (i - WARMUP) // BPD + 1

            if sig.type == SignalType.BUY:
                sl_p = entry - SL_MULT * atr_v
                tp_p = entry + SL_MULT * RR_RATIO * atr_v
                eb, out = find_exit(highs, lows, i, 1, sl_p, tp_p)
                all_signals.append((day, "MCDX", sym, 1, entry, sl_p, tp_p, RR_RATIO, out))
                locked_until = eb
            elif sig.type == SignalType.SELL:
                sl_p = entry + SL_MULT * atr_v
                tp_p = entry - SL_MULT * RR_RATIO * atr_v
                eb, out = find_exit(highs, lows, i, -1, sl_p, tp_p)
                all_signals.append((day, "MCDX", sym, -1, entry, sl_p, tp_p, RR_RATIO, out))
                locked_until = eb

    return all_signals


# ─── Capital Simulation (chronological) ───────────────────────────────────
def simulate_capital(signals):
    cap    = CAP_0
    peak   = CAP_0
    max_dd = 0.0
    day_cap = {d: 0.0 for d in range(1, DAYS + 1)}

    for day, strat, sym, direction, entry, sl, tp, rr, outcome in signals:
        if outcome == 1:
            cap += RISK * rr
        elif outcome == -1:
            cap -= RISK
        peak   = max(peak, cap)
        max_dd = max(max_dd, peak - cap)
        day_cap[day] += (RISK * rr if outcome == 1 else (-RISK if outcome == -1 else 0.0))

    return cap, max_dd, day_cap


# ─── Report ────────────────────────────────────────────────────────────────
def print_report(all_signals):
    W = 68
    print("\n" + "═" * W)
    print("  FULL 7-DAY BACKTEST  |  MCDXStrategy × 3 Symbols")
    print("  Position lock per symbol  |  SL=1.5×ATR  |  15m synthetic data")
    print("═" * W)

    print(f"\n  {'Day':<5}{'Signals':>8}{'Closed':>8}{'W':>5}{'L':>5}{'WR':>7}{'P/L ($)':>10}")
    print("  " + "─" * (W - 2))
    tot_sig = tot_cl = tot_w = tot_l = 0
    tot_pnl = 0.0

    for d in range(1, DAYS + 1):
        ds = [s for s in all_signals if s[0] == d]
        cl = [s for s in ds if s[8] != 0]
        w  = sum(1 for s in cl if s[8] ==  1)
        l  = sum(1 for s in cl if s[8] == -1)
        pnl = sum(RISK * s[7] for s in cl if s[8] == 1) - RISK * l
        wr  = f"{w/len(cl)*100:.0f}%" if cl else "—"
        print(f"  Day{d:<3}{len(ds):>8}{len(cl):>8}{w:>5}{l:>5}{wr:>7}{pnl:>+9.2f}")
        tot_sig += len(ds); tot_cl += len(cl); tot_w += w; tot_l += l; tot_pnl += pnl

    tot_wr = f"{tot_w/tot_cl*100:.1f}%" if tot_cl else "—"
    print("  " + "─" * (W - 2))
    print(f"  {'TOTAL':<5}{tot_sig:>8}{tot_cl:>8}{tot_w:>5}{tot_l:>5}{tot_wr:>7}{tot_pnl:>+9.2f}")
    print(f"\n  ✦ สัญญาณเฉลี่ย : {tot_sig/DAYS:.1f} signal/day (รวม 3 symbols)")

    print(f"\n  {'Symbol':<10}{'Signals':>8}{'W':>5}{'L':>5}{'WR':>7}{'P/L ($)':>10}")
    print("  " + "─" * (W - 2))
    for sy in [s[0] for s in SYMBOLS]:
        sg = [s for s in all_signals if s[2] == sy]
        cl = [s for s in sg if s[8] != 0]
        w  = sum(1 for s in cl if s[8] ==  1)
        l  = sum(1 for s in cl if s[8] == -1)
        pnl = sum(RISK * s[7] for s in cl if s[8] == 1) - RISK * l
        wr  = f"{w/len(cl)*100:.1f}%" if cl else "—"
        print(f"  {sy:<10}{len(sg):>8}{w:>5}{l:>5}{wr:>7}{pnl:>+9.2f}")

    final_cap, max_dd, day_pnl_map = simulate_capital(all_signals)
    pnl_total = final_cap - CAP_0
    roi = pnl_total / CAP_0 * 100

    print(f"\n{'═'*W}")
    print(f"  CAPITAL SIMULATION  |  เริ่ม ${CAP_0:.0f}  |  เสี่ยง ${RISK:.0f}/trade")
    print(f"{'─'*W}")
    running = CAP_0
    print(f"  {'Day':<5}{'Balance':>12}{'Daily P/L':>11}{'Cumulative':>12}")
    print("  " + "─" * 42)
    for d in range(1, DAYS + 1):
        dp = day_pnl_map[d]
        running += dp
        print(f"  Day{d:<3}${running:>11,.2f}{dp:>+10.2f}   {running-CAP_0:>+9.2f}")

    print(f"\n  ทุนสุดท้าย  : ${final_cap:,.2f}  ({pnl_total:+.2f} USD / {roi:+.2f}%)")
    print(f"  Win Rate    : {tot_w/tot_cl*100:.1f}%  ({tot_w}W / {tot_l}L)" if tot_cl else "  Win Rate    : —")
    print(f"  Max Drawdown: ${max_dd:.2f}")

    if DAYS > 0:
        monthly_roi = roi / DAYS * 30
        open_count = len(all_signals) - len([s for s in all_signals if s[8] != 0])
        print(f"\n  ─── Estimate ──────────────────────────────────────────")
        print(f"  Projected 30d ROI : {monthly_roi:+.2f}%")
        print(f"  สัญญาณ 30 วัน    : ~{tot_sig/DAYS*30:.0f} trades")
        print(f"  รอผล (timeout)   : {open_count} trades ยังไม่ถึง SL/TP")
    print("═" * W + "\n")


if __name__ == "__main__":
    print("กำลังรัน 7-day backtest พร้อม position lock (MCDXStrategy)...")
    signals = asyncio.run(run_backtest())
    print_report(signals)
