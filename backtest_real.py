"""
Real BTC/USDT 1H Backtest — 5 strategies on uploaded OKX CSV data
Usage: python backtest_real.py
"""
import csv, sys, os, math
from datetime import datetime, timezone
import numpy as np

CSV_PATH = "/root/.claude/uploads/d635e122-be1d-52e7-aade-d61ad8dfdfee/e002a0fb-btc_1h_3mo.csv"

# ── Candle object ──────────────────────────────────────────────────────────────

class Candle:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp = ts
        self.open = float(o); self.high = float(h)
        self.low = float(l);  self.close = float(c)
        self.volume = float(v)

def load_csv(path: str) -> list:
    candles = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                int(row["timestamp"]),
                row["open"], row["high"], row["low"], row["close"], row["volume"]
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles

# ── Indicator helpers (identical to BaseStrategy) ─────────────────────────────

def ema(values, period):
    arr = np.array(values, dtype=float)
    result = np.full_like(arr, np.nan)
    k = 2.0 / (period + 1)
    result[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result

def sma(values, period):
    arr = np.array(values, dtype=float)
    result = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = arr[i - period + 1:i + 1].mean()
    return result

def wma(values, period):
    arr = np.array(values, dtype=float)
    result = np.full_like(arr, np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    denom = weights.sum()
    for i in range(period - 1, len(arr)):
        result[i] = np.dot(arr[i - period + 1:i + 1], weights) / denom
    return result

def hma(values, period):
    half  = max(2, period // 2)
    sqrtn = max(2, int(round(math.sqrt(period))))
    wma_half = wma(values, half)
    wma_full = wma(values, period)
    raw = 2.0 * wma_half - wma_full
    valid = [float(x) for x in raw if not np.isnan(x)]
    if len(valid) < sqrtn:
        return np.full(len(values), np.nan)
    h = wma(valid, sqrtn)
    pad = len(values) - len(h)
    return np.concatenate([np.full(pad, np.nan), h])

def atr_arr(candles, period=14):
    n = len(candles)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        h = candles[i].high; l = candles[i].low; pc = candles[i-1].close
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    result = np.full(n, np.nan)
    if n > period:
        result[period] = float(np.nanmean(tr[1:period+1]))
        for i in range(period+1, n):
            result[i] = (result[i-1]*(period-1) + tr[i]) / period
    return result

def macd(values, fast=12, slow=26, signal=9):
    fe = ema(values, fast); se = ema(values, slow)
    ml = fe - se
    sig_vals = [v for v in ml if not np.isnan(v)]
    if len(sig_vals) < signal:
        sl = np.full(len(ml), np.nan)
    else:
        sl_raw = ema(sig_vals, signal)
        pad = len(ml) - len(sl_raw)
        sl  = np.concatenate([np.full(pad, np.nan), sl_raw])
    return ml, sl, ml - sl

def rsi_wilders(closes, period=14):
    n = len(closes); result = np.full(n, np.nan)
    if n < period + 1: return result
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = float(np.mean(gains[:period]))
    avg_l = float(np.mean(losses[:period]))
    result[period] = 100.0 if avg_l < 1e-10 else 100 - 100/(1+avg_g/avg_l)
    for i in range(period+1, n):
        avg_g = (avg_g*(period-1) + gains[i-1]) / period
        avg_l = (avg_l*(period-1) + losses[i-1]) / period
        result[i] = 100.0 if avg_l < 1e-10 else 100 - 100/(1+avg_g/avg_l)
    return result

def heikin_ashi(candles):
    n = len(candles)
    ha_o = np.zeros(n); ha_c = np.zeros(n)
    ha_h = np.zeros(n); ha_l = np.zeros(n)
    for i, c in enumerate(candles):
        ha_c[i] = (c.open + c.high + c.low + c.close) / 4.0
        ha_o[i] = ((c.open + c.close)/2.0 if i == 0
                   else (ha_o[i-1] + ha_c[i-1])/2.0)
        ha_h[i] = max(c.high, ha_o[i], ha_c[i])
        ha_l[i] = min(c.low,  ha_o[i], ha_c[i])
    class _C:
        __slots__ = ("timestamp","open","high","low","close","volume")
        def __init__(self, i):
            self.timestamp = candles[i].timestamp
            self.open = ha_o[i]; self.high = ha_h[i]
            self.low = ha_l[i];  self.close = ha_c[i]
            self.volume = candles[i].volume
    return [_C(i) for i in range(n)], ha_o, ha_c, ha_h, ha_l

# ── Universal backtest runner ──────────────────────────────────────────────────

def simulate(candles, buy_signals, sell_signals, atr_vals, sl_mult, rr_ratio,
             use_pct_sl=False, sl_pct=0.046, tp_pct=0.048, use_ha_entry=False,
             ha_c=None, ha_h=None, ha_l=None):
    """
    Returns (wins, losses, total_r, trades_list)
    trades_list: list of (entry_bar, direction, outcome, entry_price, exit_price)
    """
    LOOKFORWARD = 60
    cls = np.array([c.close for c in candles], dtype=float)
    hig = np.array([c.high  for c in candles], dtype=float)
    low = np.array([c.low   for c in candles], dtype=float)

    if use_ha_entry:
        entry_arr = ha_c
        h_arr = ha_h; l_arr = ha_l
    else:
        entry_arr = cls; h_arr = hig; l_arr = low

    wins = losses = 0
    total_r = 0.0
    trades = []
    n = len(candles)

    for i in range(n):
        direction = 0
        if buy_signals[i]:  direction =  1
        elif sell_signals[i]: direction = -1
        if direction == 0: continue

        atr_v = float(atr_vals[i]) if not np.isnan(atr_vals[i]) else 0
        if atr_v <= 0: continue
        ep = float(entry_arr[i])

        if use_pct_sl:
            sl_p = ep * (1 - sl_pct) if direction == 1 else ep * (1 + sl_pct)
            tp_p = ep * (1 + tp_pct) if direction == 1 else ep * (1 - tp_pct)
        else:
            sl_p = ep - sl_mult * atr_v if direction == 1 else ep + sl_mult * atr_v
            tp_p = ep + sl_mult * rr_ratio * atr_v if direction == 1 else ep - sl_mult * rr_ratio * atr_v

        outcome = 0
        exit_p = ep
        for j in range(i+1, min(i+LOOKFORWARD, n)):
            if direction == 1:
                if l_arr[j] <= sl_p: outcome = -1; exit_p = sl_p; break
                if h_arr[j] >= tp_p: outcome =  1; exit_p = tp_p; break
            else:
                if h_arr[j] >= sl_p: outcome = -1; exit_p = sl_p; break
                if l_arr[j] <= tp_p: outcome =  1; exit_p = tp_p; break

        if outcome ==  1: wins   += 1; total_r += rr_ratio
        elif outcome == -1: losses += 1; total_r -= 1.0
        if outcome != 0:
            trades.append((i, direction, outcome, ep, exit_p))

    return wins, losses, total_r, trades

# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — WaveTrend (WTADX)
# ════════════════════════════════════════════════════════════════════════════════

def run_wt(candles, sl_mult=2.5, rr_ratio=1.2, n1=2, n2=4):
    ha_can, _, ha_c, ha_h, ha_l = heikin_ashi(candles)
    ha_h_arr = np.array([c.high for c in ha_can]); ha_l_arr = np.array([c.low for c in ha_can])

    ap = (ha_h_arr + ha_l_arr + ha_c) / 3.0
    esa = ema(ap.tolist(), n1)
    diff_abs = np.abs(ap - esa)
    first_ok = np.where(~np.isnan(diff_abs))[0]
    if len(first_ok):
        diff_filled = diff_abs.copy(); diff_filled[:first_ok[0]] = diff_abs[first_ok[0]]
    else:
        diff_filled = diff_abs
    d   = ema(diff_filled.tolist(), n1)
    ci  = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
    wt1 = ema(ci.tolist(), n2)
    wt2 = sma(wt1.tolist(), 4)

    atr14 = atr_arr(ha_can, 14)
    n = len(candles)
    buy_s = np.zeros(n, dtype=bool); sell_s = np.zeros(n, dtype=bool)
    prev_dir = 0
    for i in range(1, n):
        if np.isnan(wt1[i]) or np.isnan(wt2[i]): continue
        cross_up   = float(wt1[i-1]) <= float(wt2[i-1]) and float(wt1[i]) > float(wt2[i])
        cross_down = float(wt1[i-1]) >= float(wt2[i-1]) and float(wt1[i]) < float(wt2[i])
        bu = cross_up; se = cross_down
        if bu and prev_dir != 1: buy_s[i] = True; prev_dir = 1
        elif se and prev_dir != -1: sell_s[i] = True; prev_dir = -1
        elif not bu and not se: prev_dir = 0

    wins, losses, total_r, trades = simulate(
        candles, buy_s, sell_s, atr14, sl_mult, rr_ratio,
        use_ha_entry=True, ha_c=ha_c, ha_h=ha_h_arr, ha_l=ha_l_arr
    )
    return wins, losses, total_r, trades

# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — MACD/EMA
# ════════════════════════════════════════════════════════════════════════════════

def run_macd_ema(candles, sl_mult=2.0, rr_ratio=1.2):
    ha_can, ha_o, ha_c, ha_h, ha_l = heikin_ashi(candles)
    cl = ha_c.tolist()
    hma9 = hma(cl, 9)
    ml, sl_line, _ = macd(cl, 5, 13, 4)
    atr14 = atr_arr(ha_can, 14)
    n = len(candles)
    buy_s = np.zeros(n, dtype=bool); sell_s = np.zeros(n, dtype=bool)
    prev_dir = 0
    for i in range(1, n):
        if any(np.isnan(v) for v in [hma9[i], ml[i], ml[i-1], sl_line[i], sl_line[i-1]]): continue
        hist_c = float(ml[i]) - float(sl_line[i])
        hist_p = float(ml[i-1]) - float(sl_line[i-1])
        d = 0
        if hist_c > 0 and hist_p <= 0 and float(ha_c[i]) > float(hma9[i]): d = 1
        elif hist_c < 0 and hist_p >= 0: d = -1
        if d == 1 and prev_dir != 1: buy_s[i] = True; prev_dir = 1
        elif d == -1 and prev_dir != -1: sell_s[i] = True; prev_dir = -1
        elif d == 0: prev_dir = 0

    wins, losses, total_r, trades = simulate(
        candles, buy_s, sell_s, atr14, sl_mult, rr_ratio,
        use_ha_entry=True, ha_c=ha_c, ha_h=np.array([c.high for c in ha_can]),
        ha_l=np.array([c.low for c in ha_can])
    )
    return wins, losses, total_r, trades

# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Momentum Score
# ════════════════════════════════════════════════════════════════════════════════

def run_momentum(candles, sl_mult=2.0, rr_ratio=1.2, threshold=2, reentry_bars=10):
    ha_can, ha_o, ha_c, ha_h, ha_l = heikin_ashi(candles)
    cl = ha_c.tolist()
    rsi_a = rsi_wilders(ha_c, 14)
    rsi_ema_arr = np.full(len(ha_c), np.nan)
    valid = np.where(~np.isnan(rsi_a))[0]
    if len(valid) >= 9:
        s = int(valid[0])
        re = ema([float(rsi_a[i]) for i in range(s, len(rsi_a))], 9)
        rsi_ema_arr[s:s+len(re)] = re[:len(rsi_a)-s]

    ema20 = ema(cl, 14)
    ml, sl_line, _ = macd(cl, 12, 26, 9)
    atr14 = atr_arr(ha_can, 14)

    n = len(candles)
    buy_s = np.zeros(n, dtype=bool); sell_s = np.zeros(n, dtype=bool)
    prev_bull = 0; prev_bear = 0
    last_buy = -999; last_sell = -999

    for i in range(1, n):
        if any(np.isnan(v) for v in [rsi_a[i], rsi_ema_arr[i], ema20[i], ml[i], sl_line[i]]): continue
        rv = float(rsi_a[i]); re_v = float(rsi_ema_arr[i])
        ho = float(ha_o[i]); hc = float(ha_c[i]); em = float(ema20[i])
        mv = float(ml[i]); sv = float(sl_line[i])
        bull = int(rv>50)+int(rv>re_v)+int(hc>ho)+int(hc>em)+int(mv>sv)
        bear = int(rv<50)+int(rv<re_v)+int(hc<ho)+int(hc<em)+int(mv<sv)
        if bull >= threshold:
            if prev_bull < threshold or (i - last_buy) >= reentry_bars:
                buy_s[i] = True; last_buy = i
        if bear >= threshold:
            if prev_bear < threshold or (i - last_sell) >= reentry_bars:
                sell_s[i] = True; last_sell = i
        prev_bull = bull; prev_bear = bear

    wins, losses, total_r, trades = simulate(
        candles, buy_s, sell_s, atr14, sl_mult, rr_ratio,
        use_ha_entry=True, ha_c=ha_c, ha_h=np.array([c.high for c in ha_can]),
        ha_l=np.array([c.low for c in ha_can])
    )
    return wins, losses, total_r, trades

# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — UT Bot
# ════════════════════════════════════════════════════════════════════════════════

def run_utbot(candles, sl_mult=2.5, rr_ratio=1.2, ut_mult=0.30, ut_atr_len=14):
    ha_can, _, ha_c, ha_h, ha_l = heikin_ashi(candles)
    ut_atr = atr_arr(ha_can, ut_atr_len)
    atr14  = atr_arr(ha_can, 14)

    n = len(candles)
    tsl = np.full(n, np.nan)
    for i in range(n):
        slval = float(ut_atr[i]) * ut_mult if not np.isnan(ut_atr[i]) else 0.0
        src = float(ha_c[i])
        if i == 0 or np.isnan(tsl[i-1]):
            tsl[i] = src + slval; continue
        tp = float(tsl[i-1]); sp = float(ha_c[i-1])
        if src > tp and sp > tp:   tsl[i] = max(tp, src - slval)
        elif src < tp and sp < tp: tsl[i] = min(tp, src + slval)
        elif src > tp:             tsl[i] = src - slval
        else:                      tsl[i] = src + slval

    buy_s = np.zeros(n, dtype=bool); sell_s = np.zeros(n, dtype=bool)
    prev_dir = 0
    for i in range(1, n):
        if np.isnan(tsl[i]) or np.isnan(tsl[i-1]): continue
        bu = float(ha_c[i-1]) < float(tsl[i-1]) and float(ha_c[i]) > float(tsl[i])
        se = float(ha_c[i-1]) > float(tsl[i-1]) and float(ha_c[i]) < float(tsl[i])
        if bu and prev_dir != 1: buy_s[i] = True; prev_dir = 1
        elif se and prev_dir != -1: sell_s[i] = True; prev_dir = -1
        elif not bu and not se: prev_dir = 0

    wins, losses, total_r, trades = simulate(
        candles, buy_s, sell_s, atr14, sl_mult, rr_ratio,
        use_ha_entry=True, ha_c=ha_c, ha_h=np.array([c.high for c in ha_can]),
        ha_l=np.array([c.low for c in ha_can])
    )
    return wins, losses, total_r, trades

# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 5 — kNN (simplified; uses % SL/TP like live bot)
# ════════════════════════════════════════════════════════════════════════════════

def _roc(closes, period):
    n = len(closes); result = np.full(n, np.nan)
    for i in range(period, n):
        if closes[i-period] != 0:
            result[i] = (closes[i]-closes[i-period])/closes[i-period]*100
    return result

def _williams_r(highs, lows, closes, period):
    n = len(closes); result = np.full(n, np.nan)
    for i in range(period-1, n):
        hh = np.max(highs[i-period+1:i+1]); ll = np.min(lows[i-period+1:i+1])
        if hh-ll > 0: result[i] = (hh-closes[i])/(hh-ll)*-100
    return result

def _cci(highs, lows, closes, period):
    n = len(closes); result = np.full(n, np.nan)
    for i in range(period-1, n):
        tp = (highs[i-period+1:i+1]+lows[i-period+1:i+1]+closes[i-period+1:i+1])/3.0
        mean_tp = np.mean(tp); mad = np.mean(np.abs(tp-mean_tp))
        if mad > 0: result[i] = ((highs[i]+lows[i]+closes[i])/3.0 - mean_tp)/(0.015*mad)
    return result

def _stoch_k(highs, lows, closes, period):
    n = len(closes); result = np.full(n, np.nan)
    for i in range(period-1, n):
        hh = np.max(highs[i-period+1:i+1]); ll = np.min(lows[i-period+1:i+1])
        if hh-ll > 0: result[i] = (closes[i]-ll)/(hh-ll)*100
    return result

def run_knn(candles, strict_ratio=0.83, confirm_n=2, cooldown=3, sl_pct=0.046, tp_pct=0.048):
    ha_can, ha_o, ha_c, ha_h_arr, ha_l_arr = heikin_ashi(candles)
    h = np.array([c.high for c in ha_can]); l = np.array([c.low for c in ha_can])
    cl = ha_c

    ml, sl_line, hist = macd(cl.tolist(), 12, 26, 9)
    rsi_a = rsi_wilders(cl, 14)
    roc_a = _roc(cl, 10)
    wr_a  = _williams_r(h, l, cl, 14)
    cci_a = _cci(h, l, cl, 20)
    sk_a  = _stoch_k(h, l, cl, 14)
    hma20 = hma(cl.tolist(), 20)
    rsi_ema_arr = np.full(len(cl), np.nan)
    valid = np.where(~np.isnan(rsi_a))[0]
    if len(valid) >= 9:
        s = int(valid[0])
        re = ema([float(rsi_a[i]) for i in range(s, len(rsi_a))], 9)
        rsi_ema_arr[s:s+len(re)] = re[:len(rsi_a)-s]

    LAGZ = 30; N_BARS = 10; K = 5; FWD = 5

    features = np.column_stack([rsi_a, hist, roc_a, wr_a, cci_a, sk_a])
    n = len(candles)
    buy_s = np.zeros(n, dtype=bool)
    last_sig = -999

    min_i = max(LAGZ, 30)

    for i in range(min_i, n-1):
        row = features[i]
        if np.any(np.isnan(row)): continue

        # Z-score normalize over lagz window
        window = features[max(0,i-LAGZ):i+1]
        valid_rows = window[~np.any(np.isnan(window), axis=1)]
        if len(valid_rows) < LAGZ//2: continue
        mu  = valid_rows.mean(axis=0); sd = valid_rows.std(axis=0)
        sd  = np.where(sd < 1e-8, 1.0, sd)
        row_z = (row - mu) / sd

        # Find N_BARS nearest neighbors
        dists = []
        for j in range(max(0,i-LAGZ), i - FWD):
            ref = features[j]
            if np.any(np.isnan(ref)): continue
            ref_z = (ref - mu) / sd
            d = float(np.sum((row_z - ref_z)**2))
            # label: price went up?
            label = 1 if candles[j+FWD].close > candles[j].close else 0
            dists.append((d, label))
        dists.sort(key=lambda x: x[0])
        neighbors = dists[:K]
        if len(neighbors) < K: continue
        bull_votes = sum(lb for _, lb in neighbors)
        if bull_votes / K < strict_ratio: continue

        # Layer 2: momentum gate
        if np.isnan(rsi_a[i]) or np.isnan(rsi_ema_arr[i]) or np.isnan(hma20[i]): continue
        gate_a = float(rsi_a[i]) > 50 and float(rsi_a[i]) > float(rsi_ema_arr[i])
        gate_b = float(hist[i]) > 0 and float(ml[i]) > float(sl_line[i])
        gate_c = float(ha_c[i]) > float(hma20[i])
        gates_passed = int(gate_a) + int(gate_b) + int(gate_c)
        if gates_passed < confirm_n: continue

        # Cooldown
        if i - last_sig < cooldown: continue
        buy_s[i] = True; last_sig = i

    atr14 = atr_arr(ha_can, 14)
    wins, losses, total_r, trades = simulate(
        candles, buy_s, np.zeros(n, dtype=bool), atr14, 2.5, 1.2,
        use_pct_sl=True, sl_pct=sl_pct, tp_pct=tp_pct,
        use_ha_entry=True, ha_c=ha_c, ha_h=ha_h_arr, ha_l=ha_l_arr
    )
    return wins, losses, total_r, trades

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def ts_to_str(ts_ms):
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

def main():
    candles = load_csv(CSV_PATH)
    n = len(candles)
    t0 = ts_to_str(candles[0].timestamp)
    t1 = ts_to_str(candles[-1].timestamp)
    days = (candles[-1].timestamp - candles[0].timestamp) / (1000*3600*24)
    months = days / 30.44

    print(f"\n{'='*60}")
    print(f"  BTC/USDT 1H Real Backtest — OKX Data")
    print(f"  Period : {t0} → {t1}")
    print(f"  Bars   : {n}  ({days:.1f} days ≈ {months:.1f} months)")
    p_lo = min(c.low for c in candles); p_hi = max(c.high for c in candles)
    print(f"  Price  : ${p_lo:,.0f} – ${p_hi:,.0f}")
    print(f"{'='*60}\n")

    # Best params from optimizer (GBM, applied as starting point)
    print("Running 5 strategies...")
    results = {}

    print("  [1/5] WaveTrend (WTADX)...")
    w, l, r, tr = run_wt(candles, sl_mult=2.5, rr_ratio=1.2, n1=2, n2=4)
    results["WaveTrend"] = (w, l, r, tr)

    print("  [2/5] MACD/EMA...")
    w, l, r, tr = run_macd_ema(candles, sl_mult=2.0, rr_ratio=1.2)
    results["MACD/EMA"] = (w, l, r, tr)

    print("  [3/5] Momentum Score...")
    w, l, r, tr = run_momentum(candles, sl_mult=2.0, rr_ratio=1.2, threshold=2, reentry_bars=10)
    results["Momentum"] = (w, l, r, tr)

    print("  [4/5] UT Bot...")
    w, l, r, tr = run_utbot(candles, sl_mult=2.5, rr_ratio=1.2, ut_mult=0.30, ut_atr_len=14)
    results["UT Bot"] = (w, l, r, tr)

    print("  [5/5] kNN ML (this may take ~20s)...")
    w, l, r, tr = run_knn(candles, strict_ratio=0.83, confirm_n=2, cooldown=3,
                           sl_pct=0.046, tp_pct=0.048)
    results["kNN ML"] = (w, l, r, tr)

    # ── Results table ──────────────────────────────────────────────────────────
    RISK_USD = 100   # $ risked per trade (1R)
    print(f"\n{'='*70}")
    print(f"  RESULTS  (Risk per trade: ${RISK_USD}  |  Period: {days:.1f} days)")
    print(f"{'='*70}")
    print(f"  {'Strategy':<16} {'Trades':>6} {'Wins':>5} {'Loss':>5} {'WR%':>6} {'Net R':>7} {'P&L $':>8} {'t/mo':>6}")
    print(f"  {'-'*64}")

    total_trades_all = 0; total_pnl_all = 0.0

    for name, (w, l, r, tr) in results.items():
        t = w + l
        wr = w/t*100 if t else 0.0
        pnl = r * RISK_USD
        tmo = t / months if months > 0 else 0
        total_trades_all += t; total_pnl_all += pnl
        star = " *" if wr >= 67 and t >= 5 else ""
        print(f"  {name:<16} {t:>6} {w:>5} {l:>5} {wr:>5.1f}% {r:>+7.1f}R {pnl:>+8.0f}$ {tmo:>5.1f}/mo{star}")

    print(f"  {'-'*64}")
    all_months = months
    print(f"  {'ALL COMBINED':<16} {total_trades_all:>6}{'':>5}{'':>5}{'':>7} {total_pnl_all/RISK_USD:>+7.1f}R {total_pnl_all:>+8.0f}$ {total_trades_all/all_months:>5.1f}/mo")
    print(f"{'='*70}")
    print(f"  * = WR ≥ 67% (target met)")
    print(f"\n  Note: {n} bars = {days:.1f} days only (not 3 months).")
    print(f"  OKX API returns max ~300 bars/request.")
    print(f"  Use /export_candles with pagination for 3-month data.\n")

    # ── Trade detail per strategy ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TRADE LOG (BUY/SELL direction, result)")
    print(f"{'='*70}")
    for name, (w, l, r, trades) in results.items():
        if not trades:
            print(f"\n  {name}: no trades")
            continue
        print(f"\n  {name}  ({len(trades)} trades)")
        print(f"  {'Bar':>4}  {'Date':>16}  {'Dir':>4}  {'Entry':>8}  {'Exit':>8}  {'Result':>8}")
        for bar_i, direction, outcome, ep, xp in trades:
            ts = ts_to_str(candles[bar_i].timestamp)
            dir_s = "BUY " if direction == 1 else "SELL"
            res_s = f"+{RISK_USD*1.2:.0f}$" if outcome == 1 else f"-{RISK_USD:.0f}$"
            pct = (xp - ep) / ep * 100 * direction
            print(f"  {bar_i:>4}  {ts}  {dir_s}  {ep:>8,.0f}  {xp:>8,.0f}  {res_s}  ({pct:+.2f}%)")

if __name__ == "__main__":
    main()
