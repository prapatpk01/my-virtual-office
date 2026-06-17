"""
Per-Strategy Parameter Optimizer — BTC-calibrated GBM
Period 1 (P1): 16 days  × 24H = 384 bars  (≈ June 1-16)
Period 2 (P2): 91 days  × 24H = 2184 bars (≈ 3 months)
Seeds: 3 averaged for robustness

Targets: WR ≥ 65%  |  ≥ 45 trades/month per strategy
Data: GBM  mu=0.0001/hr (+8.7%/yr drift), sigma=0.004/hr (≈ 62% annual vol)
      Start $100,000 — BTC-representative
"""
import sys, math, random, itertools
import numpy as np
sys.path.insert(0, "app")

from trading.strategies.wt_adx_strategy        import WTADXStrategy
from trading.strategies.ut_bot_strategy         import UTBotStrategy
from trading.strategies.momentum_score_strategy  import MomentumScoreStrategy
from trading.strategies.macd_ema_strategy        import MACDEMAStrategy
from trading.strategies.knn_strategy             import KNNStrategy

# ── constants ─────────────────────────────────────────────────────────────────
POSITION = 50.0
FEE_RT   = 0.0025
LOOKFWD  = 120
SEEDS    = [42, 77, 123]

P1_BARS  = 384    # 16 days × 24H
P2_BARS  = 2184   # 91 days × 24H
# Target trades/month per strategy
T_TARGET = 45

# ── candle & data ──────────────────────────────────────────────────────────────
class C:
    __slots__ = ("timestamp","open","high","low","close","volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp=ts; self.open=o; self.high=h
        self.low=l; self.close=c; self.volume=v

def gbm_btc(n, seed, start=100_000.0, mu=0.0001, sigma=0.004):
    rng = random.Random(seed); price = start; out = []
    for i in range(n):
        ret = mu + sigma * rng.gauss(0, 1)
        o = price; c = price * math.exp(ret)
        h = max(o, c) * (1 + abs(rng.gauss(0, sigma * 0.5)))
        l = min(o, c) * (1 - abs(rng.gauss(0, sigma * 0.5)))
        out.append(C(i * 3600, o, h, l, c, rng.uniform(1, 10))); price = c
    return out

# Pre-generate all datasets once
ALL_DATA = {seed: {"P1": gbm_btc(P1_BARS, seed), "P2": gbm_btc(P2_BARS, seed)}
            for seed in SEEDS}

# ── signal extraction ──────────────────────────────────────────────────────────
def get_signals(sname, strat, candles):
    n = len(candles)
    try:
        if sname == "WaveTrend":
            _, _, b, _, _ = strat._build_signals(candles)
            mi = strat.n1 + strat.n2 + 14 + 10
            return {i for i in range(mi, n - 1) if b[i]}
        elif sname == "UTBot":
            b, _, _, _ = strat._build_signals(candles)
            mi = strat.ut_atr_len + 14 + 5
            return {i for i in range(mi, n - 1) if b[i]}
        elif sname == "Momentum":
            b, _, _, _, _ = strat._build_signals(candles)
            mi = strat.macd_slow + strat.macd_sig + strat.ema_len + 5
            return {i for i in range(mi, n - 1) if b[i]}
        elif sname == "MACD/EMA":
            arrays = strat._build_arrays(candles)
            ha_o, ha_c, ha_h, ha_l, hma, ema, sma, ml, sl_l, hist, atr_a = arrays
            mi = strat.macd_slow + strat.macd_sig + strat.hma_period + 5
            return {i for i in range(mi, n - 1)
                    if strat._signal_at(i, ha_o, ha_c, ha_h, ha_l, hma, ema, sma, ml, sl_l) == 1}
        elif sname == "kNN":
            b, _, _, _ = strat._build_signals(candles)
            return {i for i in range(n) if b[i]}
    except Exception:
        return set()
    return set()

# ── simulation ─────────────────────────────────────────────────────────────────
def simulate(sname, strat, candles, tp_pct, sl_pct, max_pos=1):
    n   = len(candles)
    hig = np.array([c.high  for c in candles])
    low = np.array([c.low   for c in candles])
    cls = np.array([c.close for c in candles])
    sigs = get_signals(sname, strat, candles)
    tp_net = POSITION * tp_pct - POSITION * FEE_RT
    sl_net = POSITION * sl_pct + POSITION * FEE_RT
    open_pos = []; wins = losses = 0

    for bar in range(n):
        still = []
        for pos in open_pos:
            done = False
            if bar > pos["eb"] + LOOKFWD:
                ep = float(cls[bar])
                pnl = (ep - pos["ep"]) / pos["ep"] * POSITION - POSITION * FEE_RT
                wins += 1 if pnl >= 0 else 0
                losses += 1 if pnl < 0 else 0
                done = True
            elif low[bar] <= pos["sl"]:
                losses += 1; done = True
            elif hig[bar] >= pos["tp"]:
                wins += 1; done = True
            if not done:
                still.append(pos)
        open_pos = still
        if bar in sigs and len(open_pos) < max_pos:
            ep = float(cls[bar])
            open_pos.append({"eb": bar, "ep": ep,
                             "tp": ep * (1 + tp_pct), "sl": ep * (1 - sl_pct)})

    for pos in open_pos:
        ep = float(cls[-1])
        pnl = (ep - pos["ep"]) / pos["ep"] * POSITION - POSITION * FEE_RT
        wins += 1 if pnl >= 0 else 0
        losses += 1 if pnl < 0 else 0

    total = wins + losses
    wr  = wins / total if total > 0 else 0.0
    net = wins * tp_net - losses * sl_net
    return total, wr, net

# ── multi-seed evaluator ───────────────────────────────────────────────────────
def evaluate(sname, strat, tp, sl, max_pos=1):
    """Returns (p1_tpm, p2_tpm, avg_wr, avg_net, ev_per_trade) averaged over seeds."""
    p1t = p1w = p1n = p2t = p2w = p2n = 0.0
    for seed in SEEDS:
        t1, w1, n1 = simulate(sname, strat, ALL_DATA[seed]["P1"], tp, sl, max_pos)
        t2, w2, n2 = simulate(sname, strat, ALL_DATA[seed]["P2"], tp, sl, max_pos)
        p1t += t1; p1w += w1; p1n += n1
        p2t += t2; p2w += w2; p2n += n2
    ns = len(SEEDS)
    p1t /= ns; p1w /= ns; p1n /= ns
    p2t /= ns; p2w /= ns; p2n /= ns
    p1_days = P1_BARS / 24; p2_days = P2_BARS / 24
    p1_tpm  = p1t / (p1_days / 30)
    p2_tpm  = p2t / (p2_days / 30)
    avg_wr  = (p1w + p2w) / 2
    avg_net = (p1n + p2n) / 2
    avg_tpm = (p1_tpm + p2_tpm) / 2
    ev = (avg_wr * tp - (1 - avg_wr) * sl - FEE_RT) * POSITION
    return p1_tpm, p2_tpm, p1w, p2w, avg_wr, avg_tpm, avg_net, ev

# ── grid search ───────────────────────────────────────────────────────────────
def grid_search(sname, cls, param_grid, tp_list, sl_list, max_pos=1):
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"  Searching {len(combos) * len(tp_list) * len(sl_list)} configs …", end="", flush=True)
    results = []
    for vals in combos:
        params = dict(zip(keys, vals))
        try:
            strat = cls("BTC/USDT", params)
        except Exception:
            continue
        for tp in tp_list:
            for sl in sl_list:
                r = evaluate(sname, strat, tp, sl, max_pos)
                p1_tpm, p2_tpm, p1_wr, p2_wr, avg_wr, avg_tpm, avg_net, ev = r
                results.append({
                    "params": params, "tp": tp, "sl": sl,
                    "p1_tpm": p1_tpm, "p2_tpm": p2_tpm,
                    "p1_wr": p1_wr,   "p2_wr": p2_wr,
                    "avg_wr": avg_wr, "avg_tpm": avg_tpm,
                    "avg_net": avg_net, "ev": ev,
                })
    print(f" done ({len(results)} valid)")
    return results

# ── reporter ───────────────────────────────────────────────────────────────────
def report(sname, results, top_n=5):
    print(f"\n  {'='*68}")
    print(f"  Strategy: {sname}")
    print(f"  {'='*68}")

    # Best by WR (regardless of trade count)
    by_wr = sorted(results, key=lambda x: -x["avg_wr"])
    # Best meeting both targets
    targets_met = [r for r in results
                   if r["avg_wr"] >= 0.65 and r["avg_tpm"] >= T_TARGET and r["ev"] > 0]
    targets_met_any = [r for r in results
                       if r["avg_wr"] >= 0.65 and r["avg_tpm"] >= T_TARGET]

    if targets_met:
        pool = sorted(targets_met, key=lambda x: -(x["avg_wr"] * x["avg_tpm"]))
        tag = "PROFITABLE + WR65%+ + 45t/mo"
    elif targets_met_any:
        pool = sorted(targets_met_any, key=lambda x: -(x["avg_wr"] * x["avg_tpm"]))
        tag = "WR65%+ + 45t/mo (EV may be low)"
    else:
        pool = by_wr
        tag = "Best WR (targets not fully met)"

    print(f"  Best configs — {tag}")
    print(f"  {'Params':<32} {'TP':>5} {'SL':>5} {'P1WR':>6} {'P2WR':>6} "
          f"{'WR':>6} {'T/mo':>5} {'Net$':>7} {'EV$':>6}")
    print(f"  {'-'*32}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}")
    shown = set()
    count = 0
    for r in pool:
        key = tuple(sorted(r["params"].items())) + (r["tp"], r["sl"])
        if key in shown:
            continue
        shown.add(key)
        ps = ", ".join(f"{k.replace('wt_channel_len','n1').replace('wt_avg_len','n2').replace('ut_mult','mult').replace('strict_ratio','ratio').replace('confirm_n','conf').replace('reentry_bars','re').replace('threshold','thr')}={v}"
                       for k, v in r["params"].items())
        ps_short = ps[:32]
        wr_flag = "*" if r["avg_wr"] >= 0.65 else " "
        tm_flag = "*" if r["avg_tpm"] >= T_TARGET else " "
        ev_flag = "+" if r["ev"] > 0 else "-"
        print(f"  {ps_short:<32} {r['tp']*100:>4.1f}% {r['sl']*100:>4.1f}%"
              f"  {r['p1_wr']*100:>5.1f}%  {r['p2_wr']*100:>5.1f}%"
              f"  {r['avg_wr']*100:>5.1f}%{wr_flag} {r['avg_tpm']:>5.0f}{tm_flag}"
              f"  ${r['avg_net']:>6.1f}  {ev_flag}${abs(r['ev']):>4.2f}")
        count += 1
        if count >= top_n:
            break

    # Also show current-param baseline
    print(f"\n  Current baseline:")
    base = sorted(results, key=lambda x: abs(x["avg_tpm"] - 50))[0] if results else None
    if base:
        pass  # will be shown in main
    else:
        print("  (none)")

def pick_best(results):
    """Return the single recommended config."""
    # Priority: EV>0 AND WR>=65% AND tpm>=45
    for min_wr in [0.65, 0.63, 0.60, 0.55]:
        for min_tpm in [45, 35, 25, 10]:
            pool = [r for r in results
                    if r["avg_wr"] >= min_wr and r["avg_tpm"] >= min_tpm and r["ev"] > 0]
            if pool:
                return sorted(pool, key=lambda x: -x["avg_wr"] * min(x["avg_tpm"] / 45, 1.5))[0]
    return sorted(results, key=lambda x: -x["avg_wr"])[0]

# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("  BTC 1H Parameter Optimizer  |  3 seeds avg  |  GBM BTC-calibrated")
    print(f"  P1: {P1_BARS} bars (16 days)  P2: {P2_BARS} bars (91 days)")
    print(f"  Target: WR ≥ 65%  |  ≥ {T_TARGET} trades/month  |  EV > 0")
    print("="*70)

    # ─── 1. WaveTrend ─────────────────────────────────────────────────────────
    print("\n[1/5] WaveTrend ...")
    wt_grid = {
        "wt_channel_len": [5, 8, 10, 14],
        "wt_avg_len":     [14, 21, 30],
    }
    wt_tp = [0.025, 0.030, 0.035, 0.040, 0.048]
    wt_sl = [0.035, 0.040, 0.046, 0.050]
    wt_res = grid_search("WaveTrend", WTADXStrategy, wt_grid, wt_tp, wt_sl, max_pos=2)
    wt_best = pick_best(wt_res)
    report("WaveTrend", wt_res)

    # ─── 2. UTBot ─────────────────────────────────────────────────────────────
    print("\n[2/5] UTBot ...")
    ut_grid = {
        "ut_mult":    [0.15, 0.20, 0.25, 0.30, 0.40],
        "ut_atr_len": [10, 14, 20],
    }
    ut_tp = [0.025, 0.030, 0.035, 0.040, 0.048]
    ut_sl = [0.035, 0.040, 0.046, 0.050]
    ut_res = grid_search("UTBot", UTBotStrategy, ut_grid, ut_tp, ut_sl)
    ut_best = pick_best(ut_res)
    report("UTBot", ut_res)

    # ─── 3. Momentum ──────────────────────────────────────────────────────────
    print("\n[3/5] Momentum ...")
    mo_grid = {
        "threshold":    [2, 3],
        "reentry_bars": [5, 8, 10, 15],
        "ema_len":      [9, 14, 21],
    }
    mo_tp = [0.025, 0.030, 0.035, 0.040, 0.048]
    mo_sl = [0.035, 0.040, 0.046, 0.050]
    mo_res = grid_search("Momentum", MomentumScoreStrategy, mo_grid, mo_tp, mo_sl)
    mo_best = pick_best(mo_res)
    report("Momentum", mo_res)

    # ─── 4. MACD/EMA ──────────────────────────────────────────────────────────
    print("\n[4/5] MACD/EMA ...")
    mc_grid = {
        "macd_fast": [3, 5, 7],
        "macd_slow": [9, 13, 17],
        "hma_period":[7, 9, 12],
    }
    mc_tp = [0.025, 0.030, 0.035, 0.040, 0.048]
    mc_sl = [0.035, 0.040, 0.046, 0.050]
    mc_res = grid_search("MACD/EMA", MACDEMAStrategy, mc_grid, mc_tp, mc_sl)
    mc_best = pick_best(mc_res)
    report("MACD/EMA", mc_res)

    # ─── 5. kNN ───────────────────────────────────────────────────────────────
    print("\n[5/5] kNN ...")
    kn_grid = {
        "strict_ratio": [0.60, 0.65, 0.70, 0.75, 0.80],
        "confirm_n":    [1, 2],
        "cooldown":     [3, 5, 7],
    }
    kn_tp = [0.025, 0.030, 0.035, 0.040, 0.048]
    kn_sl = [0.035, 0.040, 0.046, 0.050]
    kn_res = grid_search("kNN", KNNStrategy, kn_grid, kn_tp, kn_sl)
    kn_best = pick_best(kn_res)
    report("kNN", kn_res)

    # ─── Grand Summary ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RECOMMENDED PARAMETERS SUMMARY")
    print("="*70)
    all_best = [
        ("WaveTrend", wt_best, 2),
        ("UTBot",     ut_best, 1),
        ("Momentum",  mo_best, 1),
        ("MACD/EMA",  mc_best, 1),
        ("kNN",       kn_best, 1),
    ]
    print(f"  {'Strategy':<12} {'Key Params':<38} {'TP':>5} {'SL':>5} "
          f"{'WR':>6} {'T/mo':>5} {'EV$':>6}")
    print(f"  {'-'*12}  {'-'*38}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*6}")
    for sname, r, mp in all_best:
        ps = ", ".join(f"{k}={v}" for k, v in r["params"].items())[:38]
        ev_s = f"+${r['ev']:.2f}" if r["ev"] >= 0 else f"-${abs(r['ev']):.2f}"
        wr_flag = "*" if r["avg_wr"] >= 0.65 else ""
        tm_flag = "*" if r["avg_tpm"] >= T_TARGET else ""
        print(f"  {sname:<12} {ps:<38} {r['tp']*100:>4.1f}% {r['sl']*100:>4.1f}%"
              f"  {r['avg_wr']*100:>5.1f}%{wr_flag} {r['avg_tpm']:>5.0f}{tm_flag}"
              f"  {ev_s:>7}")

    print(f"\n  * = target met  (WR* ≥ 65%,  T/mo* ≥ {T_TARGET})")
    print(f"\n  NOTE: Results from BTC-calibrated GBM (synthetic data).")
    print(f"  On real BTC with trending conditions, WR may differ ±5-10%.")
    print("="*70 + "\n")

main()
