"""
MTF Backtest — 3 months sequential trade log
Timeframes: 15m (sub-bars) · 1H (primary) · 4H (trend bias)
Filters per signal:
  1. 4H trend   : close > EMA20 on 4H  (macro direction)
  2. 4H momentum: ADX(14) on 4H ≥ 18   (market is trending, not ranging)
  3. 1H volume  : 1H volume > 0.8 × SMA20(vol)  (participation)
  4. 15m bias   : last 2 bars of 15m close above 15m EMA9  (micro momentum)

Target: WR ≥ 67%  |  ≥ 8 trades/month per strategy  |  EV > 0

Data: BTC-calibrated GBM   mu=0.0001/hr (+8.7%/yr)  sigma=0.004/hr (~62% annual vol)
      Start $100,000 · 3 months = 2184 bars · seeds averaged
"""
import sys, math, random
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.wt_adx_strategy        import WTADXStrategy
from trading.strategies.ut_bot_strategy         import UTBotStrategy
from trading.strategies.momentum_score_strategy  import MomentumScoreStrategy
from trading.strategies.macd_ema_strategy        import MACDEMAStrategy
from trading.strategies.knn_strategy             import KNNStrategy

# ── constants ─────────────────────────────────────────────────────────────────
POSITION  = 50.0
FEE_RT    = 0.0025
LOOKFWD   = 120
SEEDS     = [42, 77, 123]
BARS_3M   = 2184          # 91 days × 24H
TP_PCT    = 0.048
SL_PCT    = 0.046
START_BAL = 300.0

# ── best signal params from optimizer run ─────────────────────────────────────
BEST_PARAMS = {
    "WaveTrend": {"wt_channel_len": 14, "wt_avg_len": 30},
    "UTBot":     {"ut_mult": 0.40, "ut_atr_len": 20},
    "Momentum":  {"threshold": 2, "reentry_bars": 5, "ema_len": 9},
    "MACD/EMA":  {"macd_fast": 5, "macd_slow": 13, "hma_period": 9},
    "kNN":       {"strict_ratio": 0.75, "confirm_n": 2, "cooldown": 5},
}

# ── candle ────────────────────────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self,ts,o,h,l,c,v):
        self.timestamp=ts;self.open=o;self.high=h
        self.low=l;self.close=c;self.volume=v

# ── GBM generators ────────────────────────────────────────────────────────────
def gbm_1h(n, seed, start=100_000.0, mu=0.0001, sigma=0.004):
    rng = random.Random(seed); price = start; out = []
    for i in range(n):
        ret = mu + sigma * rng.gauss(0, 1)
        o = price; c = price * math.exp(ret)
        vol = rng.uniform(1, 10)
        # occasionally spike volume (smart-money proxy)
        if rng.random() < 0.05:
            vol *= rng.uniform(2.5, 5.0)
        h = max(o, c) * (1 + abs(rng.gauss(0, sigma * 0.5)))
        l = min(o, c) * (1 - abs(rng.gauss(0, sigma * 0.5)))
        out.append(C(i * 3600, o, h, l, c, vol)); price = c
    return out

def resample_4h(bars_1h):
    """Aggregate 1H bars into 4H bars."""
    out = []
    for i in range(0, len(bars_1h) - 3, 4):
        chunk = bars_1h[i:i+4]
        if len(chunk) < 4:
            break
        out.append(C(
            chunk[0].timestamp,
            chunk[0].open,
            max(c.high for c in chunk),
            min(c.low  for c in chunk),
            chunk[-1].close,
            sum(c.volume for c in chunk),
        ))
    return out

def gen_15m_for_bar(bar_1h, seed_extra, sigma=0.001):
    """Generate 4 × 15m sub-bars consistent with the parent 1H bar."""
    rng = random.Random(int(bar_1h.timestamp) ^ seed_extra)
    price = bar_1h.open; out = []
    for j in range(4):
        ret = rng.gauss(0, sigma)
        o = price; c = price * math.exp(ret)
        h = max(o, c) * (1 + abs(rng.gauss(0, sigma * 0.3)))
        l = min(o, c) * (1 - abs(rng.gauss(0, sigma * 0.3)))
        out.append(C(bar_1h.timestamp + j * 900, o, h, l, c, rng.uniform(0.5, 3)))
        price = c
    # Anchor last close to 1H close to avoid drift
    out[-1] = C(out[-1].timestamp, out[-1].open, out[-1].high,
                out[-1].low, bar_1h.close, out[-1].volume)
    return out

# ── technical helpers ──────────────────────────────────────────────────────────
def _ema(arr, period):
    result = np.full(len(arr), np.nan)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) == 0 or len(arr) - valid[0] < period:
        return result
    s = int(valid[0]); k = 2.0 / (period + 1)
    result[s + period - 1] = float(np.mean(arr[s:s + period]))
    for i in range(s + period, len(arr)):
        result[i] = float(arr[i]) * k + result[i - 1] * (1 - k)
    return result

def _sma(arr, period):
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = float(np.mean(arr[i - period + 1:i + 1]))
    return result

def _atr(highs, lows, closes, period=14):
    n = len(closes); tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1]))
    result = np.full(n, np.nan)
    if n > period:
        result[period] = float(np.mean(tr[1:period+1]))
        for i in range(period + 1, n):
            result[i] = (result[i-1] * (period - 1) + tr[i]) / period
    return result

def _adx(highs, lows, closes, period=14):
    n = len(closes); adx = np.full(n, np.nan)
    atr_a = _atr(highs, lows, closes, period)
    plus_dm  = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        h_diff = highs[i] - highs[i-1]
        l_diff = lows[i-1] - lows[i]
        plus_dm[i]  = max(h_diff, 0) if h_diff > l_diff else 0
        minus_dm[i] = max(l_diff, 0) if l_diff > h_diff else 0
    plus_di = np.full(n, np.nan); minus_di = np.full(n, np.nan)
    if n > period + 1:
        plus_di[period]  = 100 * np.sum(plus_dm[1:period+1])  / (atr_a[period] * period + 1e-10)
        minus_di[period] = 100 * np.sum(minus_dm[1:period+1]) / (atr_a[period] * period + 1e-10)
        for i in range(period + 1, n):
            if not np.isnan(atr_a[i]):
                plus_di[i]  = 100 * (plus_di[i-1]  * (period-1)/period + plus_dm[i])  / (atr_a[i] + 1e-10)
                minus_di[i] = 100 * (minus_di[i-1] * (period-1)/period + minus_dm[i]) / (atr_a[i] + 1e-10)
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx[2*period] = float(np.mean(dx[period:2*period+1]))
        for i in range(2*period+1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return adx

# ── MTF filter builder ────────────────────────────────────────────────────────
def build_mtf_filters(bars_1h, seed):
    """Return per-1H-bar arrays: mtf4h_ok, vol_ok."""
    n = len(bars_1h)
    bars_4h  = resample_4h(bars_1h)

    # 4H arrays
    cls4  = np.array([c.close for c in bars_4h])
    hig4  = np.array([c.high  for c in bars_4h])
    low4  = np.array([c.low   for c in bars_4h])
    ema4  = _ema(cls4, 20)
    adx4  = _adx(hig4, low4, cls4, 14)

    # Map 4H index → 1H bars (bar i falls in 4H bucket i//4)
    mtf4h_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        idx4 = i // 4
        if idx4 < len(bars_4h) and not np.isnan(ema4[idx4]) and not np.isnan(adx4[idx4]):
            if cls4[idx4] > ema4[idx4] and adx4[idx4] >= 18:
                mtf4h_ok[i] = True

    # 1H volume filter: volume > 0.8 × SMA20(vol)
    vol1  = np.array([c.volume for c in bars_1h])
    vsma  = _sma(vol1, 20)
    vol_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(vsma[i]) and vol1[i] >= 0.8 * vsma[i]:
            vol_ok[i] = True

    # 15m micro-momentum: last 2 of 4 sub-bars close above sub-EMA9
    m15_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        sub = gen_15m_for_bar(bars_1h[i], seed)
        cls_sub = np.array([c.close for c in sub])
        ema_sub = _ema(cls_sub, min(3, len(cls_sub)))
        if not np.isnan(ema_sub[-1]) and cls_sub[-1] > ema_sub[-1]:
            m15_ok[i] = True

    return mtf4h_ok, vol_ok, m15_ok

# ── signal extraction ──────────────────────────────────────────────────────────
def get_signals(sname, strat, candles):
    n = len(candles)
    try:
        if sname == "WaveTrend":
            _, _, b, _, _ = strat._build_signals(candles)
            mi = strat.n1 + strat.n2 + 14 + 10
            return {i for i in range(mi, n-1) if b[i]}
        elif sname == "UTBot":
            b, _, _, _ = strat._build_signals(candles)
            mi = strat.ut_atr_len + 14 + 5
            return {i for i in range(mi, n-1) if b[i]}
        elif sname == "Momentum":
            b, _, _, _, _ = strat._build_signals(candles)
            mi = strat.macd_slow + strat.macd_sig + strat.ema_len + 5
            return {i for i in range(mi, n-1) if b[i]}
        elif sname == "MACD/EMA":
            arrays = strat._build_arrays(candles)
            ha_o,ha_c,ha_h,ha_l,hma,ema,sma,ml,sl_l,hist,atr_a = arrays
            mi = strat.macd_slow + strat.macd_sig + strat.hma_period + 5
            return {i for i in range(mi, n-1)
                    if strat._signal_at(i,ha_o,ha_c,ha_h,ha_l,hma,ema,sma,ml,sl_l)==1}
        elif sname == "kNN":
            b, _, _, _ = strat._build_signals(candles)
            return {i for i in range(n) if b[i]}
    except Exception:
        return set()
    return set()

# ── simulation with MTF filter ────────────────────────────────────────────────
def simulate_mtf(sname, strat, bars_1h, tp_pct, sl_pct, seed,
                 use_mtf=True, max_pos=1):
    n   = len(bars_1h)
    hig = np.array([c.high  for c in bars_1h])
    low = np.array([c.low   for c in bars_1h])
    cls = np.array([c.close for c in bars_1h])

    raw_sigs = get_signals(sname, strat, bars_1h)

    if use_mtf:
        mtf4h, vol_ok, m15_ok = build_mtf_filters(bars_1h, seed)
        sigs = {i for i in raw_sigs if mtf4h[i] and vol_ok[i] and m15_ok[i]}
    else:
        sigs = raw_sigs

    tp_net = POSITION * tp_pct - POSITION * FEE_RT
    sl_net = POSITION * sl_pct + POSITION * FEE_RT
    open_pos = []; trades = []

    for bar in range(n):
        still = []
        for pos in open_pos:
            done = False
            if bar > pos["eb"] + LOOKFWD:
                ep  = float(cls[bar])
                pnl = (ep - pos["ep"]) / pos["ep"] * POSITION - POSITION * FEE_RT
                trades.append({"bar": bar, "entry": pos["ep"], "exit": ep,
                               "outcome": "timeout", "pnl": round(pnl, 2)})
                done = True
            elif low[bar] <= pos["sl"]:
                trades.append({"bar": bar, "entry": pos["ep"], "exit": pos["sl"],
                               "outcome": "loss", "pnl": round(-sl_net, 2)})
                done = True
            elif hig[bar] >= pos["tp"]:
                trades.append({"bar": bar, "entry": pos["ep"], "exit": pos["tp"],
                               "outcome": "win", "pnl": round(tp_net, 2)})
                done = True
            if not done:
                still.append(pos)
        open_pos = still

        if bar in sigs and len(open_pos) < max_pos:
            ep = float(cls[bar])
            open_pos.append({"eb": bar, "ep": ep,
                             "tp": ep * (1 + tp_pct), "sl": ep * (1 - sl_pct)})

    for pos in open_pos:
        ep  = float(cls[-1])
        pnl = (ep - pos["ep"]) / pos["ep"] * POSITION - POSITION * FEE_RT
        trades.append({"bar": n-1, "entry": pos["ep"], "exit": ep,
                       "outcome": "timeout", "pnl": round(pnl, 2)})

    wins    = sum(1 for t in trades if t["outcome"] == "win")
    losses  = sum(1 for t in trades if t["outcome"] == "loss")
    timeouts= sum(1 for t in trades if t["outcome"] == "timeout")
    total   = len(trades)
    wr      = wins / max(wins + losses, 1)
    net     = sum(t["pnl"] for t in trades)
    raw_cnt = len(raw_sigs)
    flt_cnt = len(sigs)
    return trades, total, wins, losses, timeouts, wr, net, raw_cnt, flt_cnt

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 72)
    print("  MTF Backtest — BTC/USDT 1H  |  3 months (91 days, 2184 bars)")
    print("  Filters: 4H EMA20 trend + ADX≥18 · 1H volume · 15m micro-momentum")
    print(f"  TP={TP_PCT*100:.1f}%  SL={SL_PCT*100:.1f}%  Position=${POSITION}  Seeds: {SEEDS}")
    print("=" * 72)

    strategies = [
        ("WaveTrend", WTADXStrategy,       2),
        ("UTBot",     UTBotStrategy,        1),
        ("Momentum",  MomentumScoreStrategy,1),
        ("MACD/EMA",  MACDEMAStrategy,      1),
        ("kNN",       KNNStrategy,          1),
    ]

    summary = []

    for sname, cls_s, max_pos in strategies:
        params = BEST_PARAMS[sname]
        strat  = cls_s("BTC/USDT", params)
        print(f"\n  ── {sname} ──  params: {params}")
        print(f"  {'':4} {'Seed':>6} {'Raw':>5} {'Flt':>5} {'T':>4} {'W':>4} {'L':>4} {'TO':>4} {'WR':>6} {'Net$':>7}")
        print(f"  {'':4} {'-'*6}  {'-'*5}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}")

        agg = {"total":0,"wins":0,"losses":0,"timeouts":0,"net":0.0,"raw":0,"flt":0}
        all_trades = []

        for seed in SEEDS:
            bars = gbm_1h(BARS_3M, seed)
            tr, tot, w, l_, to, wr, net, raw, flt = simulate_mtf(
                sname, strat, bars, TP_PCT, SL_PCT, seed,
                use_mtf=True, max_pos=max_pos)
            all_trades += tr
            agg["total"]   += tot;  agg["wins"]     += w
            agg["losses"]  += l_;   agg["timeouts"] += to
            agg["net"]     += net;  agg["raw"]      += raw
            agg["flt"]     += flt
            print(f"  {'':4} {seed:>6} {raw:>5} {flt:>5} {tot:>4} {w:>4} {l_:>4} {to:>4} "
                  f"{wr*100:>5.1f}% ${net:>6.1f}")

        # Averages
        ns = len(SEEDS)
        avg_t   = agg["total"] / ns
        avg_w   = agg["wins"]  / ns
        avg_l   = agg["losses"] / ns
        avg_net = agg["net"]   / ns
        avg_wr  = avg_w / max(avg_w + avg_l, 1)
        avg_tpm = avg_t / (BARS_3M / 24 / 30)   # trades per month
        avg_raw = agg["raw"] / ns
        avg_flt = agg["flt"] / ns
        filter_rate = (1 - avg_flt / max(avg_raw, 1)) * 100
        ev = (avg_wr * TP_PCT - (1 - avg_wr) * SL_PCT - FEE_RT) * POSITION

        wr_flag = "✓" if avg_wr >= 0.67 else "✗"
        tm_flag = "✓" if avg_tpm >= 8    else "✗"
        ev_flag = "✓" if ev > 0          else "✗"

        print(f"  {'':4} {'AVG':>6} {avg_raw:>5.0f} {avg_flt:>5.0f} {avg_t:>4.0f} "
              f"{avg_w:>4.0f} {avg_l:>4.0f} {agg['timeouts']/ns:>4.0f} "
              f"{avg_wr*100:>5.1f}% ${avg_net:>6.1f}")
        print(f"  T/month={avg_tpm:.1f}{tm_flag}  WR={avg_wr*100:.1f}%{wr_flag}  "
              f"EV/trade={'+' if ev>=0 else ''}${ev:.2f}{ev_flag}  "
              f"Filter cut={filter_rate:.0f}% of signals")

        summary.append({
            "sname": sname, "params": params,
            "avg_tpm": avg_tpm, "avg_wr": avg_wr,
            "avg_net": avg_net, "ev": ev,
            "filter_rate": filter_rate,
            "raw": avg_raw, "flt": avg_flt,
        })

    # ── Grand summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  GRAND SUMMARY — with MTF (4H trend + ADX + volume + 15m micro)")
    print("=" * 72)
    print(f"  {'Strategy':<12} {'T/mo':>5} {'WR%':>6} {'EV$':>6} {'Net$/mo':>8} "
          f"{'WR':>4} {'T':>4} {'Filter%':>8}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*4}  {'-'*4}  {'-'*8}")
    for r in summary:
        wr_f = "✓" if r["avg_wr"] >= 0.67 else "✗"
        tm_f = "✓" if r["avg_tpm"] >= 8    else "✗"
        ev_f = "+" if r["ev"] >= 0 else "-"
        net_mo = r["avg_net"] / (BARS_3M / 24 / 30)
        print(f"  {r['sname']:<12} {r['avg_tpm']:>5.1f}{tm_f} {r['avg_wr']*100:>5.1f}%{wr_f} "
              f"{ev_f}${abs(r['ev']):>4.2f}  ${net_mo:>+7.2f}/mo  "
              f"cut {r['filter_rate']:>4.0f}%")

    all_net = sum(r["avg_net"] for r in summary)
    all_tpm = sum(r["avg_tpm"] for r in summary)
    print(f"  {'-'*72}")
    print(f"  {'TOTAL (5 strats)':<12} {all_tpm:>5.1f}   combined "
          f"  net/3mo=${all_net:>+.2f}  net/mo=${all_net/(BARS_3M/24/30):>+.2f}")

    # ── MTF vs no-MTF comparison (Momentum as sample) ─────────────────────────
    print()
    print("  MTF Impact Demo — Momentum strategy (seed=42)")
    print(f"  {'Mode':<18} {'Raw sigs':>9} {'Filtered':>9} {'T':>4} {'WR':>6} {'Net$':>7}")
    print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*4}  {'-'*6}  {'-'*7}")
    strat_mo = MomentumScoreStrategy("BTC/USDT", BEST_PARAMS["Momentum"])
    bars42   = gbm_1h(BARS_3M, 42)
    for mode, use in [("No MTF filter", False), ("With MTF filter", True)]:
        tr,tot,w,l_,to,wr,net,raw,flt = simulate_mtf(
            "Momentum", strat_mo, bars42, TP_PCT, SL_PCT, 42, use_mtf=use)
        cut = (1-flt/max(raw,1))*100 if use else 0
        print(f"  {mode:<18} {raw:>9} {flt:>9} {tot:>4} {wr*100:>5.1f}% ${net:>6.1f}"
              + (f"  (cut {cut:.0f}%)" if use else ""))

    print()
    print("  NOTES:")
    print("  - GBM synthetic data: WR may shift ±5-10% vs real BTC")
    print("  - On real data: 4H trend filter removes sideways/bear periods")
    print("  - Sentiment (Fear&Greed) + Smart Money (on-chain) need live API")
    print("  - Add to bot: fetch alternative.me/fng daily, skip BUY if fear<30")
    print("  - Smart money proxy: skip BUY if exchange inflows spike (CryptoQuant)")
    print("=" * 72)

main()
