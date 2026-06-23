"""
Backtest comparison: SJ Hybrid 5-component (current) vs 6-component (+Supertrend)
BTC + XAU, Jan-May 2026, $50 margin × 20x = $1000 notional

Key fixes vs previous version:
- BTC CSVs: header detection + microsecond→ms timestamp conversion
- Intrabar HIGH/LOW checking for SL/TP (not just close price)
- Proper 5-comp vs 6-comp registry separation
- Bar-index-based cooldown (not real-time)
"""
import sys
import os
import csv
import math

# Add project root to path
sys.path.insert(0, '/home/user/my-virtual-office')
sys.path.insert(0, '/home/user/my-virtual-office/app')

import numpy as np
import pandas as pd

from trading.strategies.trend_cont_improved_strategy import (
    _compute, build_indicator_registry
)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BTC_DIR    = '/home/user/my-virtual-office/data/btc_csv'
XAU_DIR    = '/home/user/my-virtual-office/data/xau_csv'
MONTHS     = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05']
MARGIN     = 50.0      # USD margin per trade
LEVERAGE   = 20        # 20x
NOTIONAL   = MARGIN * LEVERAGE  # $1000 notional
FEE_RATE   = 0.0004   # 0.04% per side
HEADER     = 'open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore'

PARAMS = dict(
    fast_mode=True,
    sj_scoring=True,
    adx_max_fast=44,
    tp2_r_fast=2.5,
    min_score=4,
    # strategy defaults (from DEFAULTS)
    ema_fast=20, ema_slow=50, ema_micro=9, rsi_period=14,
    rsi_min_buy=35.0, rsi_max_buy=75.0,
    rsi_min_sell=28.0, rsi_max_sell=65.0,
    bias_gate=70.0,
    bias_gate_fast=60.0,
    swing_lookback=4, swing_pct=0.006,
    vol_period=20, vol_mult=1.0,
    atr_period=14, sl_mult=1.2, sl_min_pct=0.012, sl_max_pct=0.035,
    adx_len=14, adx_min=30,
    adx_min_fast=18,
    adx_rising_fast=True,
    pullback_pct_fast=0.025,
    tp1_r=0.5, tp1_fraction=0.40, tp2_r=2.5,
    st_period=10, st_mult=3.0,
    cooldown_bars=0,  # managed externally by bar index
)

COOLDOWN_BARS = 4  # external cooldown after a trade


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_csv(data_dir: str, symbol: str, tf: str) -> pd.DataFrame:
    """Load all months Jan-May 2026 for a given symbol and timeframe."""
    frames = []
    for month in MONTHS:
        fname = f'{symbol}-{tf}-{month}.csv'
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            continue

        # Detect header: if first cell is numeric → no header (BTC raw)
        first_cell = rows[0][0].strip()
        has_header = not first_cell.lstrip('-').isdigit()

        data_rows = rows[1:] if has_header else rows

        parsed = []
        for row in data_rows:
            if len(row) < 6:
                continue
            try:
                ts_raw = int(row[0])
                # Microsecond detection (16 digits)
                if ts_raw > 1_000_000_000_000_000:
                    ts_ms = ts_raw // 1000
                else:
                    ts_ms = ts_raw
                parsed.append({
                    'timestamp': ts_ms,
                    'open':   float(row[1]),
                    'high':   float(row[2]),
                    'low':    float(row[3]),
                    'close':  float(row[4]),
                    'volume': float(row[5]),
                })
            except (ValueError, IndexError):
                continue

        if parsed:
            frames.append(pd.DataFrame(parsed))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df


# ─────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────
def run_backtest(symbol: str, data_dir: str, indicator_registry: dict, label: str):
    """
    Run a full backtest for one symbol using the given indicator registry.
    Returns a dict with trade statistics.
    """
    df15 = load_csv(data_dir, symbol, '15m')
    df1h = load_csv(data_dir, symbol, '1h')
    df4h = load_csv(data_dir, symbol, '4h')

    if df15.empty or df1h.empty or df4h.empty:
        print(f"  [WARNING] Missing data for {symbol}")
        return None

    # Build params with the given registry
    p = {**PARAMS, 'indicators': indicator_registry}

    # Compute signals on full dataset
    try:
        computed = _compute(df15, df1h, df4h, p)
    except Exception as e:
        print(f"  [ERROR] _compute failed for {symbol}: {e}")
        return None

    trades    = []
    pnl_curve = [0.0]
    peak      = 0.0
    max_dd    = 0.0

    in_pos       = False
    side         = None
    entry        = 0.0
    sl           = 0.0
    tp1          = 0.0
    tp2          = 0.0
    one_r        = 0.0
    tp1_hit      = False
    entry_bar    = 0
    cooldown_end = -1  # bar index when cooldown expires

    bars = computed.reset_index()
    N    = len(bars)

    for i in range(1, N):
        bar = bars.iloc[i]
        hi  = float(bar['high'])
        lo  = float(bar['low'])
        cl  = float(bar['close'])

        if in_pos:
            # ── Intrabar SL/TP check ──────────────────────────────────
            pnl = 0.0
            exit_price = None
            exit_reason = None

            if side == 'long':
                # Conservative: check SL first (worst case)
                sl_hit  = lo <= sl
                tp2_hit = hi >= tp2

                if sl_hit:
                    exit_price  = sl
                    exit_reason = 'SL'
                elif hi >= tp1 and not tp1_hit:
                    # TP1 hit — move SL to entry (breakeven), continue
                    tp1_hit = True
                    sl      = entry  # SL → breakeven
                    # Check TP2 on same bar after TP1
                    if hi >= tp2:
                        exit_price  = tp2
                        exit_reason = 'TP2'
                elif tp2_hit:
                    exit_price  = tp2
                    exit_reason = 'TP2'

            elif side == 'short':
                sl_hit  = hi >= sl
                tp2_hit = lo <= tp2

                if sl_hit:
                    exit_price  = sl
                    exit_reason = 'SL'
                elif lo <= tp1 and not tp1_hit:
                    tp1_hit = True
                    sl      = entry
                    if lo <= tp2:
                        exit_price  = tp2
                        exit_reason = 'TP2'
                elif tp2_hit:
                    exit_price  = tp2
                    exit_reason = 'TP2'

            if exit_price is not None:
                # Calculate PnL
                # We use full notional for simplicity (partial close modeled as avg)
                # TP1 takes 40% at TP1 price (0.5R), rest at TP2 or SL
                if exit_reason == 'TP2' and tp1_hit:
                    # 40% closed at TP1 (0.5R), 60% closed at TP2 (2.5R)
                    tp1_pnl_r = 0.5 * 0.40
                    tp2_pnl_r = 2.5 * 0.60
                    gross_r   = tp1_pnl_r + tp2_pnl_r  # 1.7R total
                elif exit_reason == 'TP2' and not tp1_hit:
                    gross_r = 2.5  # full size to TP2
                elif exit_reason == 'SL' and tp1_hit:
                    # 40% closed at TP1 (0.5R), 60% closed at breakeven (0R)
                    gross_r = 0.5 * 0.40 + 0.0 * 0.60  # 0.2R net
                else:
                    gross_r = -1.0  # full SL

                # Convert R to dollar PnL
                r_dollar = one_r / entry * NOTIONAL
                gross_pnl = gross_r * r_dollar
                # Fees: entry + exit
                fee = 2 * NOTIONAL * FEE_RATE
                net_pnl = gross_pnl - fee

                pnl_curve.append(pnl_curve[-1] + net_pnl)
                if pnl_curve[-1] > peak:
                    peak = pnl_curve[-1]
                dd = peak - pnl_curve[-1]
                if dd > max_dd:
                    max_dd = dd

                trades.append({
                    'side':   side,
                    'entry':  entry,
                    'exit':   exit_price,
                    'reason': exit_reason,
                    'pnl':    net_pnl,
                    'win':    net_pnl > 0,
                })

                in_pos       = False
                cooldown_end = i + COOLDOWN_BARS

        # ── Entry check ──────────────────────────────────────────────
        if not in_pos and i > cooldown_end:
            prev = bars.iloc[i - 1]  # use previous closed bar for signal

            buy_sig  = bool(prev.get('final_buy',  False))
            sell_sig = bool(prev.get('final_sell', False))
            dist     = float(prev.get('dist', 0) or 0)

            if (buy_sig or sell_sig) and dist > 0 and not math.isnan(dist):
                entry_px = float(prev['close'])
                in_pos   = True
                tp1_hit  = False
                entry    = entry_px
                one_r    = dist

                if buy_sig:
                    side = 'long'
                    sl   = entry - dist
                    tp1  = entry + PARAMS['tp1_r'] * dist
                    tp2  = entry + PARAMS['tp2_r_fast'] * dist
                else:
                    side = 'short'
                    sl   = entry + dist
                    tp1  = entry - PARAMS['tp1_r'] * dist
                    tp2  = entry - PARAMS['tp2_r_fast'] * dist

    # Close any open position at last bar close
    if in_pos:
        last_bar = bars.iloc[-1]
        exit_px  = float(last_bar['close'])
        if side == 'long':
            gross_r = (exit_px - entry) / one_r
        else:
            gross_r = (entry - exit_px) / one_r
        r_dollar  = one_r / entry * NOTIONAL
        gross_pnl = gross_r * r_dollar
        fee       = 2 * NOTIONAL * FEE_RATE
        net_pnl   = gross_pnl - fee
        pnl_curve.append(pnl_curve[-1] + net_pnl)
        trades.append({
            'side': side, 'entry': entry, 'exit': exit_px,
            'reason': 'EOD', 'pnl': net_pnl, 'win': net_pnl > 0,
        })

    # Compute stats
    n_trades = len(trades)
    if n_trades == 0:
        return {'label': label, 'symbol': symbol, 'trades': 0,
                'wr': 0, 'pnl': 0, 'avg': 0, 'maxdd': 0}

    wins     = sum(1 for t in trades if t['win'])
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl   = total_pnl / n_trades

    return {
        'label':   label,
        'symbol':  symbol,
        'trades':  n_trades,
        'wr':      wins / n_trades * 100,
        'pnl':     total_pnl,
        'avg':     avg_pnl,
        'maxdd':   -max_dd,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\nBuilding indicator registries...")

    # 6-comp registry (with supertrend)
    reg_6 = build_indicator_registry(sj_scoring=True)

    # 5-comp registry: same as 6-comp but WITHOUT supertrend_bull
    reg_5 = {k: v for k, v in reg_6.items() if k != 'supertrend_bull'}

    print(f"  5-comp keys: {list(reg_5.keys())}")
    print(f"  6-comp keys: {list(reg_6.keys())}")

    configs = [
        ('SJ-5comp (current)',  reg_5),
        ('SJ-6comp+Supertrend', reg_6),
    ]

    symbols = [
        ('BTCUSDT', BTC_DIR),
        ('XAUUSDT', XAU_DIR),
    ]

    results = []
    for label, registry in configs:
        for symbol, data_dir in symbols:
            print(f"\nRunning: {label} / {symbol}...")
            r = run_backtest(symbol, data_dir, registry, label)
            if r:
                results.append(r)
                print(f"  Trades={r['trades']}, WR={r['wr']:.1f}%, PnL=${r['pnl']:.2f}, MaxDD=${r['maxdd']:.2f}")

    # ── Combined stats ─────────────────────────────────────────────
    combined = []
    for label, _ in configs:
        subset = [r for r in results if r['label'] == label]
        if not subset:
            continue
        total_pnl = sum(r['pnl']   for r in subset)
        total_tr  = sum(r['trades'] for r in subset)
        wins      = sum(r['wr'] / 100 * r['trades'] for r in subset)
        max_dd    = min(r['maxdd'] for r in subset)  # worst (most negative)
        wr        = wins / total_tr * 100 if total_tr else 0
        combined.append({
            'label':  label,
            'symbol': 'COMBINED',
            'trades': total_tr,
            'wr':     wr,
            'pnl':    total_pnl,
            'avg':    0,
            'maxdd':  max_dd,
        })

    # ── Print table ────────────────────────────────────────────────
    sep  = '═' * 78
    thin = '─' * 78
    hdr  = f"{'Label':<27} {'Symbol':<10} {'Trades':>7} {'WR%':>6} {'PnL$':>9} {'Avg$':>7} {'MaxDD$':>9}"

    print(f"\n{sep}")
    print(hdr)
    print(thin)

    for r in results:
        sym   = r['symbol'].replace('USDT', '')
        pnl_s = f"${r['pnl']:+.2f}"
        avg_s = f"${r['avg']:+.2f}"
        dd_s  = f"${r['maxdd']:.2f}"
        print(f"{r['label']:<27} {sym:<10} {r['trades']:>7} {r['wr']:>5.1f}% {pnl_s:>9} {avg_s:>7} {dd_s:>9}")

    print(thin)
    for r in combined:
        pnl_s = f"${r['pnl']:+.2f}"
        dd_s  = f"${r['maxdd']:.2f}"
        print(f"{r['label']:<27} {'COMBINED':<10} {r['trades']:>7} {r['wr']:>5.1f}% {pnl_s:>9} {'':>7} {dd_s:>9}")

    print(sep)

    # ── Delta ──────────────────────────────────────────────────────
    def get_combined(label):
        for r in combined:
            if r['label'] == label:
                return r
        return None

    base = get_combined('SJ-5comp (current)')
    new  = get_combined('SJ-6comp+Supertrend')

    if base and new:
        print("\nDelta: SJ-6comp+Supertrend vs SJ-5comp (current):")
        d_pnl  = new['pnl']  - base['pnl']
        d_wr   = new['wr']   - base['wr']
        d_tr   = new['trades'] - base['trades']
        d_dd   = new['maxdd'] - base['maxdd']   # positive = better (less drawdown)

        pnl_tag = '+' if d_pnl >= 0 else ''
        wr_tag  = '+' if d_wr  >= 0 else ''
        tr_tag  = '+' if d_tr  >= 0 else ''
        dd_tag  = '+' if d_dd  >= 0 else ''

        print(f"  PnL     : {pnl_tag}${d_pnl:.2f}  ({base['pnl']:.2f} → {new['pnl']:.2f})")
        print(f"  Win Rate: {wr_tag}{d_wr:.1f}%  ({base['wr']:.1f}% → {new['wr']:.1f}%)")
        print(f"  Trades  : {tr_tag}{d_tr}    ({base['trades']} → {new['trades']})")
        print(f"  MaxDD   : {dd_tag}${d_dd:.2f} ({base['maxdd']:.2f} → {new['maxdd']:.2f})")

        print()
        if d_pnl > 0 and d_wr >= 0:
            verdict = "BETTER — 6-comp+Supertrend improves both PnL and WR"
        elif d_pnl > 0 and d_wr < 0:
            verdict = "MIXED — 6-comp+Supertrend has more PnL but lower WR (fewer, bigger trades)"
        elif d_pnl < 0 and d_wr > 0:
            verdict = "MIXED — 6-comp+Supertrend has higher WR but lower overall PnL"
        else:
            verdict = "WORSE — 6-comp+Supertrend underperforms on PnL and WR"

        print(f"  Verdict : {verdict}")
    print()


if __name__ == '__main__':
    main()
