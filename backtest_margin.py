"""
Margin Backtest — BTC/USDT 1H | Jan–May 2026 | OKX Cross Margin x5
Logic: BUY on signal → hold until SELL signal OR TP +5.2% OR SL -2.5%
All 5 strategies | $50 collateral x5 = $250 notional | MAX_POSITIONS=4
OKX costs: taker 0.1%/side + margin interest 0.02%/day on $200 borrowed
"""
import types, numpy as np, os

# ── Reuse all signal + data helpers from live_sim ────────────────────────────
_src = open(os.path.join(os.path.dirname(__file__), "backtest_live_sim.py")).read()
_mod = types.ModuleType("_live_sim")
exec(compile(_src, "_live_sim", "exec"), _mod.__dict__)

load_tf              = _mod.load_tf
ts_str               = _mod.ts_str
compute_wt_signals   = _mod.compute_wt_signals
compute_macd_signals = _mod.compute_macd_signals
compute_mom_signals  = _mod.compute_mom_signals
compute_ut_signals   = _mod.compute_ut_signals
build_4h_bias        = _mod.build_4h_bias
build_ts_index       = _mod.build_ts_index
get_4h_bias_at       = _mod.get_4h_bias_at

# ── Config ────────────────────────────────────────────────────────────────────
TRADE_USDT      = 50.0
LEVERAGE        = 5.0
NOTIONAL        = TRADE_USDT * LEVERAGE      # $250
BORROWED        = TRADE_USDT * (LEVERAGE-1)  # $200

MAX_POSITIONS   = 4
START_CAPITAL   = 300.0
TP_PCT          = 0.052   # +5.2% from entry
SL_PCT          = 0.025   # -2.5% from entry

TAKER_FEE_PCT   = 0.001          # OKX taker 0.1% per side
BORROW_RATE_DAY = 0.0002         # 0.02%/day on borrowed amount
BORROW_RATE_HR  = BORROW_RATE_DAY / 24

# ── Cost helpers ──────────────────────────────────────────────────────────────

def commission(entry, exit_p, btc):
    return (btc * entry + btc * exit_p) * TAKER_FEE_PCT

def interest(hours):
    return BORROWED * BORROW_RATE_HR * max(1.0, hours)

# ── Simulation ────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    c1h = load_tf("1h"); c4h = load_tf("4h")
    n      = len(c1h)
    days   = (c1h[-1].ts - c1h[0].ts) / (1000*3600*24)
    months = days / 30.44
    t0 = ts_str(c1h[0].ts, "%Y-%m-%d"); t1 = ts_str(c1h[-1].ts, "%Y-%m-%d")

    print("Computing signals...")
    wt_buy,  wt_sell,  wt_atr,  wt_hc,  *_ = compute_wt_signals(c1h)
    md_buy,  md_sell,  md_atr,  md_hc,  *_ = compute_macd_signals(c1h)
    mo_buy,  mo_sell,  mo_atr,  mo_hc,  *_ = compute_mom_signals(c1h)
    ut_buy,  ut_sell,  ut_atr,  ut_hc,  *_ = compute_ut_signals(c1h)

    print("Computing 4H bias...")
    bias4h  = build_4h_bias(c4h)
    ts4map  = build_ts_index(c4h)
    bias_at = np.array([get_4h_bias_at(c1h[i].ts, c4h, ts4map, bias4h) for i in range(n)])

    print(f"  Signals — WT buy:{wt_buy.sum()} sell:{wt_sell.sum()} | "
          f"MACD buy:{md_buy.sum()} sell:{md_sell.sum()} | "
          f"Mom buy:{mo_buy.sum()} sell:{mo_sell.sum()} | "
          f"UT buy:{ut_buy.sum()} sell:{ut_sell.sum()}")

    def get_buy_sigs(i):
        return [
            ("WT#0",     bool(wt_buy[i]), float(wt_hc[i])),
            ("WT#1",     bool(wt_buy[i]), float(wt_hc[i])),
            ("MACD",     bool(md_buy[i]), float(md_hc[i])),
            ("Momentum", bool(mo_buy[i]), float(mo_hc[i])),
            ("UTBot",    bool(ut_buy[i]), float(ut_hc[i])),
        ]

    def is_sell_sig(sk, i):
        if sk in ("WT#0", "WT#1"): return bool(wt_sell[i])
        if sk == "MACD":            return bool(md_sell[i])
        if sk == "Momentum":        return bool(mo_sell[i])
        if sk == "UTBot":           return bool(ut_sell[i])
        return False

    # ── State ─────────────────────────────────────────────────────────────────
    capital    = START_CAPITAL
    open_pos   = {}
    strategies = ["WT#0", "WT#1", "MACD", "Momentum", "UTBot"]

    def zstat():
        return {"trades":0, "wins":0, "losses":0, "sigs":0,
                "gross":0., "comm":0., "int":0.,
                "blk_mtf":0, "blk_lock":0, "blk_maxpos":0,
                "dur":[]}

    stat      = {s: zstat() for s in strategies}
    monthly   = {}
    trade_log = []

    print(f"\nSimulating {n} bars ({months:.1f} months) {t0} → {t1}\n")

    for i in range(n):
        ts = c1h[i].ts
        mk = ts_str(ts, "%Y-%m")
        if mk not in monthly:
            monthly[mk] = {"gross":0., "comm":0., "int":0., "net":0., "trades":0}

        h  = float(c1h[i].high)
        l  = float(c1h[i].low)
        cl = float(c1h[i].close)

        # ── Close: TP / SL / SELL signal ─────────────────────────────────────
        to_close = []
        for key, pos in open_pos.items():
            tp_hit   = h >= pos["tp"]
            sl_hit   = l <= pos["sl"]
            sell_hit = is_sell_sig(key, i) and not tp_hit and not sl_hit

            if tp_hit or sl_hit or sell_hit:
                if tp_hit:
                    exit_p = pos["tp"]; res = "TP"
                elif sl_hit:
                    exit_p = pos["sl"]; res = "SL"
                else:
                    exit_p = cl;        res = "SIG"

                btc   = pos["btc"]
                hrs   = (ts - pos["open_ts"]) / (1000*3600)
                gross = btc * (exit_p - pos["entry"])
                comm  = commission(pos["entry"], exit_p, btc)
                intr  = interest(hrs)
                net   = gross - comm - intr
                capital += TRADE_USDT + net
                s = pos["strat"]

                if res == "TP":   stat[s]["wins"]   += 1
                elif res == "SL": stat[s]["losses"]  += 1
                else:             stat[s]["sigs"]    += 1

                stat[s]["gross"] += gross
                stat[s]["comm"]  += comm
                stat[s]["int"]   += intr
                stat[s]["dur"].append(hrs)
                monthly[mk]["gross"]  += gross
                monthly[mk]["comm"]   += comm
                monthly[mk]["int"]    += intr
                monthly[mk]["net"]    += net
                monthly[mk]["trades"] += 1
                trade_log.append({
                    "opened": ts_str(pos["open_ts"]),
                    "strat":  s,
                    "entry":  round(pos["entry"]),
                    "exit":   round(exit_p),
                    "hrs":    round(hrs, 1),
                    "res":    res,
                    "gross":  round(gross, 2),
                    "comm":   round(comm, 3),
                    "int":    round(intr, 3),
                    "net":    round(net, 2),
                    "cap":    round(capital, 2),
                })
                to_close.append(key)
        for k in to_close:
            del open_pos[k]

        # ── Open: BUY signal + MTF gate ───────────────────────────────────────
        bias    = int(bias_at[i])
        wt_used = False

        for sk, is_buy, hc_v in get_buy_sigs(i):
            if not is_buy: continue
            if sk in ("WT#0", "WT#1") and wt_used: continue

            if bias != 1:
                stat[sk]["blk_mtf"] += 1; continue
            if sk in open_pos:
                stat[sk]["blk_lock"] += 1; continue
            if len(open_pos) >= MAX_POSITIONS:
                stat[sk]["blk_maxpos"] += 1; continue
            if capital < TRADE_USDT:
                continue

            entry = hc_v
            sl    = entry * (1 - SL_PCT)
            tp    = entry * (1 + TP_PCT)
            btc   = NOTIONAL / entry
            capital -= TRADE_USDT
            open_pos[sk] = {
                "entry": entry, "sl": sl, "tp": tp, "btc": btc,
                "strat": sk, "open_ts": ts,
            }
            stat[sk]["trades"] += 1
            if sk.startswith("WT"):
                wt_used = True

    # ── Close any still-open positions at last price ──────────────────────────
    last_p  = float(c1h[-1].close)
    last_ts = c1h[-1].ts
    for key, pos in open_pos.items():
        hrs   = (last_ts - pos["open_ts"]) / (1000*3600)
        gross = pos["btc"] * (last_p - pos["entry"])
        comm  = commission(pos["entry"], last_p, pos["btc"])
        intr  = interest(hrs)
        net   = gross - comm - intr
        capital += TRADE_USDT + net
        s = pos["strat"]
        stat[s]["gross"] += gross
        stat[s]["comm"]  += comm
        stat[s]["int"]   += intr
        stat[s]["dur"].append(hrs)

    # ── Results ───────────────────────────────────────────────────────────────
    total_t = sum(s["trades"] for s in stat.values())
    total_w = sum(s["wins"]   for s in stat.values())
    total_l = sum(s["losses"] for s in stat.values())
    total_s = sum(s["sigs"]   for s in stat.values())
    total_g = sum(s["gross"]  for s in stat.values())
    total_c = sum(s["comm"]   for s in stat.values())
    total_i = sum(s["int"]    for s in stat.values())
    total_n = total_g - total_c - total_i
    closed  = total_w + total_l + total_s
    wr      = total_w / closed * 100 if closed else 0
    all_dur = [d for s in stat.values() for d in s["dur"]]
    avg_dur = sum(all_dur) / len(all_dur) if all_dur else 0
    ret     = capital - START_CAPITAL

    W = 80
    print("=" * W)
    print(f"  MARGIN BACKTEST — BTC/USDT 1H  {t0} → {t1}")
    print(f"  Capital ${START_CAPITAL:.0f}  |  Trade: ${TRADE_USDT:.0f} x{LEVERAGE:.0f} = ${NOTIONAL:.0f} notional  |  MAX_POSITIONS: {MAX_POSITIONS}")
    print(f"  Exit: TP +{TP_PCT*100:.1f}% | SL -{SL_PCT*100:.1f}% | SELL signal  |  OKX taker 0.1%/side  |  Margin 0.02%/day")
    print("=" * W)

    print(f"\n  EXECUTION SUMMARY")
    print(f"  {'─'*68}")
    print(f"  Trades executed        : {total_t}  ({total_t/months:.1f}/month)")
    print(f"  Closed (TP+SL+SIG)     : {closed}  [TP:{total_w}  SL:{total_l}  SIG:{total_s}]  open at end: {total_t-closed}")
    print(f"  Win Rate (TP only)     : {wr:.1f}%   (TP:{total_w} / closed:{closed})")
    print(f"  Avg trade duration     : {avg_dur:.1f} hours")
    print(f"\n  COST BREAKDOWN")
    print(f"  {'─'*68}")
    print(f"  Gross P&L              : {total_g:>+10.2f}$")
    print(f"  Commission (0.1%×2)    : {-total_c:>+10.2f}$   avg {total_c/total_t if total_t else 0:.3f}$/trade")
    print(f"  Margin interest        : {-total_i:>+10.2f}$   avg {total_i/total_t if total_t else 0:.3f}$/trade")
    print(f"  {'─'*44}")
    print(f"  Net P&L                : {total_n:>+10.2f}$")
    print(f"  Final capital          : ${capital:.2f}  (started ${START_CAPITAL:.2f})")
    print(f"  Return on capital      : {ret:>+10.2f}$  ({ret/START_CAPITAL*100:>+.1f}%)")

    print(f"\n  PER-STRATEGY BREAKDOWN")
    hdr = (f"  {'Strategy':<10} {'T':>4} {'TP':>4} {'SL':>4} {'SIG':>4} {'WR%':>6} {'AvgH':>5}"
           f" {'Gross':>9} {'Comm':>8} {'Int':>7} {'NET':>10}"
           f" {'MTF':>5} {'LCK':>4} {'MAX':>4}")
    print(f"  {'─'*(W-2)}")
    print(hdr)
    print(f"  {'─'*(W-2)}")
    for sk in strategies:
        s  = stat[sk]
        t  = s["trades"]; w = s["wins"]; l = s["losses"]; sg = s["sigs"]
        cls = w + l + sg
        wr_s = f"{w/cls*100:.1f}%" if cls else "—"
        ad   = sum(s["dur"]) / len(s["dur"]) if s["dur"] else 0
        g    = s["gross"]; c_ = s["comm"]; it = s["int"]; nt = g - c_ - it
        print(f"  {sk:<10} {t:>4} {w:>4} {l:>4} {sg:>4} {wr_s:>6} {ad:>5.1f}h"
              f" {g:>+9.2f}$ {c_:>8.3f}$ {it:>7.3f}$ {nt:>+10.2f}$"
              f" {s['blk_mtf']:>5} {s['blk_lock']:>4} {s['blk_maxpos']:>4}")
    print(f"  {'─'*(W-2)}")
    blk_mtf = sum(s["blk_mtf"]    for s in stat.values())
    blk_lck = sum(s["blk_lock"]   for s in stat.values())
    blk_max = sum(s["blk_maxpos"] for s in stat.values())
    print(f"  {'TOTAL':<10} {total_t:>4} {total_w:>4} {total_l:>4} {total_s:>4} {wr:.1f}% {avg_dur:>5.1f}h"
          f" {total_g:>+9.2f}$ {total_c:>8.3f}$ {total_i:>7.3f}$ {total_n:>+10.2f}$"
          f" {blk_mtf:>5} {blk_lck:>4} {blk_max:>4}")

    print(f"\n  MONTHLY BREAKDOWN")
    print(f"  {'─'*68}")
    print(f"  {'Month':>8} {'Trades':>6} {'Gross':>9} {'Comm':>8} {'Int':>6} {'Net':>9} {'Cumul':>10}")
    cumul = 0.
    for mk in sorted(monthly):
        m = monthly[mk]; cumul += m["net"]
        bar = ("+" if m["net"] >= 0 else "-") * min(20, int(abs(m["net"]) / 1.0))
        print(f"  {mk:>8} {m['trades']:>6} {m['gross']:>+9.2f}$ {m['comm']:>8.3f}$ {m['int']:>6.3f}$ {m['net']:>+9.2f}$ {cumul:>+10.2f}$  {bar}")

    print(f"\n  LAST 15 TRADES")
    print(f"  {'─'*68}")
    print(f"  {'Opened':<17} {'Strat':<10} {'Entry':>7} {'Exit':>7} {'Hrs':>5} {'Res':>3} {'Gross':>8} {'Comm':>6} {'Int':>5} {'Net':>8} {'Cap':>9}")
    for t in trade_log[-15:]:
        print(f"  {t['opened']:<17} {t['strat']:<10} {t['entry']:>7,} {t['exit']:>7,}"
              f" {t['hrs']:>5.1f} {t['res']:>3} {t['gross']:>+8.2f}$ {t['comm']:>6.3f}$ {t['int']:>5.3f}$ {t['net']:>+8.2f}$ {t['cap']:>9.2f}")
    print("=" * W)

if __name__ == "__main__":
    main()
