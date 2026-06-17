"""
Margin Backtest — BTC/USDT 1H | Jan–May 2026 | OKX Cross Margin x5
Reuses signal computation from backtest_live_sim.py (identical logic).
Adds realistic OKX trading costs:
  - Taker fee: 0.1% per side on notional ($250)
  - Margin borrow: 0.02%/day on borrowed USDT ($200)
MAX_POSITIONS: 4 | $50 collateral → $250 exposure per trade
"""
import types, numpy as np, sys, os
from datetime import datetime, timezone

# ── Reuse all signal + data helpers from live_sim ────────────────────────────
_src = open(os.path.join(os.path.dirname(__file__), "backtest_live_sim.py")).read()
_mod = types.ModuleType("_live_sim")
exec(compile(_src, "_live_sim", "exec"), _mod.__dict__)

load_tf            = _mod.load_tf
ts_str             = _mod.ts_str
compute_wt_signals = _mod.compute_wt_signals
compute_macd_signals = _mod.compute_macd_signals
compute_mom_signals  = _mod.compute_mom_signals
compute_ut_signals   = _mod.compute_ut_signals
build_4h_bias      = _mod.build_4h_bias
build_ts_index     = _mod.build_ts_index
get_4h_bias_at     = _mod.get_4h_bias_at

# ── Config ────────────────────────────────────────────────────────────────────
TRADE_USDT      = 50.0
LEVERAGE        = 5.0
NOTIONAL        = TRADE_USDT * LEVERAGE      # $250
BORROWED        = TRADE_USDT * (LEVERAGE-1)  # $200

MAX_POSITIONS   = 4
START_CAPITAL   = 300.0
SL_MULT         = 1.5
TP_MULT         = 1.5

TAKER_FEE_PCT   = 0.001          # OKX taker 0.1% per side (market order)
BORROW_RATE_DAY = 0.0002         # OKX cross USDT ~0.02%/day on borrowed amount
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
    n   = len(c1h)
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

    print(f"  WT BUY signals: {wt_buy.sum()} | MACD: {md_buy.sum()} | Momentum: {mo_buy.sum()} | UTBot: {ut_buy.sum()}")

    def get_sigs(i):
        return [
            ("WT#0",     bool(wt_buy[i]), float(wt_atr[i]), float(wt_hc[i])),
            ("WT#1",     bool(wt_buy[i]), float(wt_atr[i]), float(wt_hc[i])),
            ("MACD",     bool(md_buy[i]), float(md_atr[i]), float(md_hc[i])),
            ("Momentum", bool(mo_buy[i]), float(mo_atr[i]), float(mo_hc[i])),
            ("UTBot",    bool(ut_buy[i]), float(ut_atr[i]), float(ut_hc[i])),
        ]

    # ── State ─────────────────────────────────────────────────────────────────
    capital   = START_CAPITAL
    open_pos  = {}
    strategies = ["WT#0","WT#1","MACD","Momentum","UTBot"]

    def zstat():
        return {"trades":0,"wins":0,"losses":0,
                "gross":0.,"comm":0.,"int":0.,
                "blk_mtf":0,"blk_lock":0,"blk_maxpos":0,
                "dur":[]}

    stat     = {s: zstat() for s in strategies}
    monthly  = {}
    trade_log = []

    print(f"\nSimulating {n} bars ({months:.1f} months) {t0} → {t1}\n")

    for i in range(n):
        ts = c1h[i].ts
        mk = ts_str(ts, "%Y-%m")
        if mk not in monthly:
            monthly[mk] = {"gross":0.,"comm":0.,"int":0.,"net":0.,"trades":0}

        # ── Close TP/SL ───────────────────────────────────────────────────────
        to_close = []
        for key, pos in open_pos.items():
            h = float(c1h[i].high); l = float(c1h[i].low)
            tp_hit = h >= pos["tp"]; sl_hit = l <= pos["sl"]
            if tp_hit or sl_hit:
                exit_p  = pos["tp"] if tp_hit else pos["sl"]
                btc     = pos["btc"]
                hrs     = (ts - pos["open_ts"]) / (1000*3600)
                gross   = btc * (exit_p - pos["entry"])
                comm    = commission(pos["entry"], exit_p, btc)
                intr    = interest(hrs)
                net     = gross - comm - intr
                capital += TRADE_USDT + net
                s = pos["strat"]
                if tp_hit: stat[s]["wins"]  += 1
                else:      stat[s]["losses"] += 1
                stat[s]["gross"] += gross
                stat[s]["comm"]  += comm
                stat[s]["int"]   += intr
                stat[s]["dur"].append(hrs)
                monthly[mk]["gross"]  += gross
                monthly[mk]["comm"]   += comm
                monthly[mk]["int"]    += intr
                monthly[mk]["net"]    += net
                monthly[mk]["trades"] += 1
                res = "TP" if tp_hit else "SL"
                trade_log.append({
                    "opened": ts_str(pos["open_ts"]),
                    "strat":  s,
                    "entry":  round(pos["entry"]),
                    "exit":   round(exit_p),
                    "hrs":    round(hrs,1),
                    "res":    res,
                    "gross":  round(gross,2),
                    "comm":   round(comm,3),
                    "int":    round(intr,3),
                    "net":    round(net,2),
                    "cap":    round(capital,2),
                })
                to_close.append(key)
        for k in to_close: del open_pos[k]

        # ── Open positions ────────────────────────────────────────────────────
        bias   = int(bias_at[i])
        wt_used = False

        for sk, is_buy, atr_v, hc_v in get_sigs(i):
            if not is_buy: continue
            if sk in ("WT#0","WT#1") and wt_used: continue

            if bias != 1:
                stat[sk]["blk_mtf"] += 1; continue
            if sk in open_pos:
                stat[sk]["blk_lock"] += 1; continue
            if len(open_pos) >= MAX_POSITIONS:
                stat[sk]["blk_maxpos"] += 1; continue
            if np.isnan(atr_v) or atr_v <= 0 or capital < TRADE_USDT:
                continue

            entry   = hc_v
            sl      = entry - SL_MULT * atr_v
            tp      = entry + TP_MULT * atr_v
            btc     = NOTIONAL / entry
            capital -= TRADE_USDT
            open_pos[sk] = {"entry":entry,"sl":sl,"tp":tp,"btc":btc,
                             "strat":sk,"open_ts":ts}
            stat[sk]["trades"] += 1
            if sk.startswith("WT"): wt_used = True

    # ── Close any still-open positions at last price ──────────────────────────
    last_p = float(c1h[-1].close); last_ts = c1h[-1].ts
    for key, pos in open_pos.items():
        hrs   = (last_ts - pos["open_ts"]) / (1000*3600)
        gross = pos["btc"] * (last_p - pos["entry"])
        comm  = commission(pos["entry"], last_p, pos["btc"])
        intr  = interest(hrs)
        net   = gross - comm - intr
        capital += TRADE_USDT + net
        s = pos["strat"]
        stat[s]["gross"] += gross; stat[s]["comm"] += comm
        stat[s]["int"]   += intr;  stat[s]["dur"].append(hrs)

    # ── Results ───────────────────────────────────────────────────────────────
    total_t = sum(s["trades"] for s in stat.values())
    total_w = sum(s["wins"]   for s in stat.values())
    total_l = sum(s["losses"] for s in stat.values())
    total_g = sum(s["gross"]  for s in stat.values())
    total_c = sum(s["comm"]   for s in stat.values())
    total_i = sum(s["int"]    for s in stat.values())
    total_n = total_g - total_c - total_i
    closed  = total_w + total_l
    wr      = total_w/closed*100 if closed else 0
    all_dur = [d for s in stat.values() for d in s["dur"]]
    avg_dur = sum(all_dur)/len(all_dur) if all_dur else 0
    ret     = capital - START_CAPITAL

    W = 76
    print("="*W)
    print(f"  MARGIN BACKTEST — BTC/USDT 1H  {t0} → {t1}")
    print(f"  Capital ${START_CAPITAL:.0f}  |  Trade: ${TRADE_USDT:.0f} collateral x{LEVERAGE:.0f} = ${NOTIONAL:.0f} notional")
    print(f"  MAX_POSITIONS: {MAX_POSITIONS}  |  Borrowed/trade: ${BORROWED:.0f}  |  OKX taker {TAKER_FEE_PCT*100:.1f}%/side  |  Margin {BORROW_RATE_DAY*100:.3f}%/day")
    print("="*W)

    print(f"\n  EXECUTION SUMMARY")
    print(f"  {'─'*64}")
    print(f"  Trades executed        : {total_t}  ({total_t/months:.1f}/month)")
    print(f"  Closed (TP+SL)         : {closed}  (open at end: {total_t-closed})")
    print(f"  Wins / Losses          : {total_w}W / {total_l}L")
    print(f"  Win Rate               : {wr:.1f}%")
    print(f"  Avg trade duration     : {avg_dur:.1f} hours")
    print(f"\n  COST BREAKDOWN")
    print(f"  {'─'*64}")
    print(f"  Gross P&L              : {total_g:>+9.2f}$")
    print(f"  Commission (0.1%×2)    : {-total_c:>+9.2f}$   avg {total_c/total_t:.3f}$/trade")
    print(f"  Margin interest        : {-total_i:>+9.2f}$   avg {total_i/total_t:.3f}$/trade")
    print(f"  {'─'*40}")
    print(f"  Net P&L                : {total_n:>+9.2f}$")
    print(f"  Final capital          : ${capital:.2f}  (started ${START_CAPITAL:.2f})")
    print(f"  Return on capital      : {ret:>+9.2f}$  ({ret/START_CAPITAL*100:>+.1f}%)")

    print(f"\n  PER-STRATEGY BREAKDOWN")
    hdr = (f"  {'Strategy':<10} {'T':>4} {'W':>4} {'L':>4} {'WR%':>6} {'AvgH':>5}"
           f" {'Gross':>8} {'Comm':>8} {'Int':>7} {'NET':>9}"
           f" {'MTF':>5} {'LCK':>4} {'MAX':>4}")
    print(f"  {'─'*(W-2)}")
    print(hdr)
    print(f"  {'─'*(W-2)}")
    for sk in strategies:
        s=stat[sk]; t=s["trades"]; w=s["wins"]; l=s["losses"]; cls=w+l
        wr_s=f"{w/cls*100:.1f}%" if cls else "—"
        ad=sum(s["dur"])/len(s["dur"]) if s["dur"] else 0
        g=s["gross"]; c_=s["comm"]; it=s["int"]; nt=g-c_-it
        print(f"  {sk:<10} {t:>4} {w:>4} {l:>4} {wr_s:>6} {ad:>5.1f}h"
              f" {g:>+8.2f}$ {c_:>8.3f}$ {it:>7.3f}$ {nt:>+9.2f}$"
              f" {s['blk_mtf']:>5} {s['blk_lock']:>4} {s['blk_maxpos']:>4}")
    print(f"  {'─'*(W-2)}")
    bg=total_g; bc=total_c; bi=total_i; bn=total_n; ad=avg_dur
    blk_mtf=sum(s['blk_mtf'] for s in stat.values())
    blk_lck=sum(s['blk_lock'] for s in stat.values())
    blk_max=sum(s['blk_maxpos'] for s in stat.values())
    wr_s=f"{wr:.1f}%"
    print(f"  {'TOTAL':<10} {total_t:>4} {total_w:>4} {total_l:>4} {wr_s:>6} {ad:>5.1f}h"
          f" {bg:>+8.2f}$ {bc:>8.3f}$ {bi:>7.3f}$ {bn:>+9.2f}$"
          f" {blk_mtf:>5} {blk_lck:>4} {blk_max:>4}")

    print(f"\n  MONTHLY BREAKDOWN")
    print(f"  {'─'*64}")
    print(f"  {'Month':>8} {'Trades':>6} {'Gross':>8} {'Comm':>7} {'Int':>6} {'Net':>8} {'Cumul':>9}")
    cumul = 0.
    for mk in sorted(monthly):
        m=monthly[mk]; cumul+=m["net"]
        bar=("+" if m["net"]>=0 else "-") * min(20, int(abs(m["net"])/0.5))
        print(f"  {mk:>8} {m['trades']:>6} {m['gross']:>+8.2f}$ {m['comm']:>7.3f}$ {m['int']:>6.3f}$ {m['net']:>+8.2f}$ {cumul:>+9.2f}$  {bar}")

    print(f"\n  LAST 15 TRADES")
    print(f"  {'─'*64}")
    print(f"  {'Opened':<17} {'Strat':<10} {'Entry':>7} {'Exit':>7} {'Hrs':>5} Res {'Gross':>7} {'Comm':>6} {'Int':>5} {'Net':>7} {'Cap':>8}")
    for t in trade_log[-15:]:
        print(f"  {t['opened']:<17} {t['strat']:<10} {t['entry']:>7,} {t['exit']:>7,}"
              f" {t['hrs']:>5.1f} {t['res']:>3} {t['gross']:>+7.2f}$ {t['comm']:>6.3f}$ {t['int']:>5.3f}$ {t['net']:>+7.2f}$ {t['cap']:>8.2f}")
    print("="*W)

if __name__ == "__main__":
    main()
