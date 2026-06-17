"""
Backtest: 15M Entry Filter Comparison — EMA20 vs HMA15
Config A: WT#0 + UTBot only | n1=8 n2=12 | SL=1.5×ATR TP=1.0×ATR | 4H MTF gate
OKX Cross Margin x5 | $50 collateral × 5 = $250 notional

3 modes compared:
  (A) Immediate — enter at 1H signal bar close (current bot behaviour)
  (B) EMA20     — enter when 15M candle closes above EMA20(20)
  (C) HMA15     — enter when 15M candle closes above HMA15(15)

Pending entry window: 4 hours (16 × 15M bars).
If no confirmation within 4h → trade is skipped (MISS).
Exits: walk forward in 1H bars checking ATR-based TP/SL.
"""

import types, numpy as np, os, math

# ── Reuse helpers from live_sim ──────────────────────────────────────────────
_src = open(os.path.join(os.path.dirname(__file__), "backtest_live_sim.py")).read()
_mod = types.ModuleType("_live_sim")
exec(compile(_src, "_live_sim", "exec"), _mod.__dict__)

load_tf             = _mod.load_tf
ts_str              = _mod.ts_str
ema                 = _mod.ema
wma                 = _mod.wma
hma                 = _mod.hma
compute_wt_signals  = _mod.compute_wt_signals
compute_ut_signals  = _mod.compute_ut_signals
build_4h_bias       = _mod.build_4h_bias
build_ts_index      = _mod.build_ts_index
get_4h_bias_at      = _mod.get_4h_bias_at

# ── Config ───────────────────────────────────────────────────────────────────
TRADE_USDT    = 50.0
LEVERAGE      = 5.0
NOTIONAL      = TRADE_USDT * LEVERAGE      # $250
BORROWED      = TRADE_USDT * (LEVERAGE-1)  # $200
MAX_POSITIONS = 3
START_CAPITAL = 300.0

WT_SL_MULT = 1.5;  WT_TP_MULT = 1.0
UT_SL_MULT = 1.5;  UT_TP_MULT = 1.0

EXPIRE_BARS_15M = 16         # 4h = 16 × 15M bars
TAKER_FEE_PCT   = 0.001
BORROW_RATE_HR  = 0.0002 / 24

INTERVAL_1H  = 3_600_000    # ms
INTERVAL_15M =   900_000    # ms

def commission(entry, exit_p, btc):
    return (btc * entry + btc * exit_p) * TAKER_FEE_PCT

def interest_cost(hours):
    return BORROWED * BORROW_RATE_HR * max(1.0, hours)

# ── 15M indicator helpers ────────────────────────────────────────────────────

def ema_arr(closes_list, period):
    return ema(closes_list, period)

def hma_arr(closes_list, period):
    return hma(closes_list, period)

# ── Core simulation ──────────────────────────────────────────────────────────

def simulate(mode, c1h, c15m, ts15_map,
             wt_buy, wt_atr, wt_hc,
             ut_buy, ut_atr, ut_hc,
             bias_at,
             ema20_15, hma15_15):
    """
    mode: "immediate" | "ema20" | "hma15"
    Returns summary dict.
    """
    n1h  = len(c1h)
    n15m = len(c15m)

    capital   = START_CAPITAL
    open_pos  = {}          # strat_key → position dict
    results   = []          # closed trade records

    misses    = 0           # signals that expired without entry
    blk_mtf   = 0
    blk_lock  = 0
    blk_max   = 0

    entry_deltas = []       # (entry_15m - entry_1h) % per trade for EMA/HMA modes

    # ── helpers ──────────────────────────────────────────────────────────────

    def first_15m_at_or_after(target_ms):
        """Return 15M bar index with ts >= target_ms, or None."""
        # Round up to next 15M boundary
        aligned = ((target_ms + INTERVAL_15M - 1) // INTERVAL_15M) * INTERVAL_15M
        for delta in range(0, EXPIRE_BARS_15M + 2):
            t = aligned + delta * INTERVAL_15M
            if t in ts15_map:
                return ts15_map[t]
        # Fallback linear scan from expected area
        for idx in range(max(0, n15m - 2000), n15m):
            if c15m[idx].ts >= target_ms:
                return idx
        return None

    def find_15m_entry(signal_1h_ts, sl_mult, tp_mult, atr_v):
        """Find 15M bar for entry. Returns (entry_price, 15m_idx, sl, tp) or None."""
        # The 1H signal bar closes at signal_1h_ts + INTERVAL_1H
        start_ms = signal_1h_ts + INTERVAL_1H
        start_idx = first_15m_at_or_after(start_ms)
        if start_idx is None:
            return None

        end_idx = min(start_idx + EXPIRE_BARS_15M, n15m)

        for idx in range(start_idx, end_idx):
            close_15m = float(c15m[idx].close)
            confirmed = False

            if mode == "immediate":
                confirmed = True                           # first bar always

            elif mode == "ema20":
                e20 = ema20_15[idx]
                if not np.isnan(e20) and close_15m > float(e20):
                    confirmed = True

            elif mode == "hma15":
                h15 = hma15_15[idx]
                if not np.isnan(h15) and close_15m > float(h15):
                    confirmed = True

            if confirmed:
                entry_p = close_15m
                sl = entry_p - sl_mult * atr_v
                tp = entry_p + tp_mult * atr_v
                return (entry_p, idx, sl, tp)

        return None   # expired

    def exit_on_1h(entry_1h_idx, sl, tp, open_ts_ms, entry_price):
        """Walk forward from the 1H bar AFTER entry, checking TP/SL high/low."""
        for j in range(entry_1h_idx + 1, n1h):
            h = float(c1h[j].high)
            l = float(c1h[j].low)
            ts_j = c1h[j].ts
            if h >= tp:
                hrs = (ts_j - open_ts_ms) / 3_600_000
                btc = NOTIONAL / entry_price
                gross = btc * (tp - entry_price)
                comm  = commission(entry_price, tp, btc)
                intr  = interest_cost(hrs)
                return ("TP", tp, hrs, gross, comm, intr)
            if l <= sl:
                hrs = (ts_j - open_ts_ms) / 3_600_000
                btc = NOTIONAL / entry_price
                gross = btc * (sl - entry_price)
                comm  = commission(entry_price, sl, btc)
                intr  = interest_cost(hrs)
                return ("SL", sl, hrs, gross, comm, intr)
        # Still open at end of data — close at last price
        j = n1h - 1
        exit_p = float(c1h[j].close)
        hrs = (c1h[j].ts - open_ts_ms) / 3_600_000
        btc = NOTIONAL / entry_price
        gross = btc * (exit_p - entry_price)
        comm  = commission(entry_price, exit_p, btc)
        intr  = interest_cost(hrs)
        return ("OPEN", exit_p, hrs, gross, comm, intr)

    # ── Main loop ─────────────────────────────────────────────────────────────
    pending   = {}          # strat_key → (entry_p, 1h_idx_at_signal, sl, tp, open_ts_ms, atr)
    wt_locked = set()       # active WT slot keys (locked until exit)
    ut_locked = False

    for i in range(n1h):
        ts_i = c1h[i].ts

        # Check pending 15M entries → try to enter
        for sk in list(pending.keys()):
            if sk in open_pos:
                continue   # already entered
            pend = pending[sk]
            if pend.get("entered"):
                continue
            entry_p, sig_1h_idx, sl, tp, open_ts_ms, atr_v, base_entry = pend["data"]
            # Already entered? Check via open_pos
            pass

        # ── Close open positions: TP / SL on this 1H bar ─────────────────────
        to_close = []
        for sk, pos in open_pos.items():
            h = float(c1h[i].high)
            l = float(c1h[i].low)
            if h >= pos["tp"]:
                res = "TP"; exit_p = pos["tp"]
            elif l <= pos["sl"]:
                res = "SL"; exit_p = pos["sl"]
            else:
                continue
            hrs   = (ts_i - pos["open_ts"]) / 3_600_000
            btc   = NOTIONAL / pos["entry"]
            gross = btc * (exit_p - pos["entry"])
            comm  = commission(pos["entry"], exit_p, btc)
            intr  = interest_cost(hrs)
            net   = gross - comm - intr
            capital += TRADE_USDT + net
            results.append({"res": res, "gross": gross, "comm": comm, "int": intr, "net": net,
                             "hrs": hrs, "entry": pos["entry"], "exit": exit_p,
                             "base_entry": pos["base_entry"], "strat": pos["strat"]})
            to_close.append(sk)
        for sk in to_close:
            del open_pos[sk]
            if sk.startswith("WT"):
                wt_locked.discard(sk)
            elif sk == "UTBot":
                ut_locked = False

        # ── Open: WT signals ─────────────────────────────────────────────────
        bias = int(bias_at[i])

        # WT BUY
        if bool(wt_buy[i]):
            if bias != 1:
                blk_mtf += 1
            else:
                for slot in ("WT#0", "WT#1"):
                    if slot in open_pos:
                        continue
                    if slot in wt_locked:
                        continue
                    if len(open_pos) >= MAX_POSITIONS:
                        blk_max += 1
                        break
                    # Find entry
                    atr_v       = float(wt_atr[i]) if not np.isnan(wt_atr[i]) else 0
                    base_entry  = float(wt_hc[i])
                    if atr_v <= 0:
                        break

                    if mode == "immediate":
                        entry_p = base_entry
                        sl  = entry_p - WT_SL_MULT * atr_v
                        tp  = entry_p + WT_TP_MULT * atr_v
                        btc = NOTIONAL / entry_p
                        capital -= TRADE_USDT
                        open_pos[slot] = {"entry": entry_p, "sl": sl, "tp": tp,
                                          "btc": btc, "open_ts": ts_i,
                                          "base_entry": base_entry, "strat": slot}
                        wt_locked.add(slot)
                    else:
                        res = find_15m_entry(ts_i, WT_SL_MULT, WT_TP_MULT, atr_v)
                        if res is None:
                            misses += 1
                        else:
                            entry_p, idx_15m, sl, tp = res
                            entry_deltas.append((entry_p - base_entry) / base_entry * 100)
                            btc = NOTIONAL / entry_p
                            capital -= TRADE_USDT
                            # Use the 1H bar index that contains the 15M entry
                            entry_15m_ts = c15m[idx_15m].ts
                            # Find corresponding 1H bar (floor to 1H boundary)
                            aligned_1h = (entry_15m_ts // INTERVAL_1H) * INTERVAL_1H
                            # entry open_ts = 15M bar ts
                            open_pos[slot] = {"entry": entry_p, "sl": sl, "tp": tp,
                                              "btc": btc, "open_ts": entry_15m_ts,
                                              "base_entry": base_entry, "strat": slot}
                            wt_locked.add(slot)
                    break   # only fill one slot per bar

        # UT BUY
        if bool(ut_buy[i]):
            if bias != 1:
                blk_mtf += 1
            elif ut_locked or "UTBot" in open_pos:
                blk_lock += 1
            elif len(open_pos) >= MAX_POSITIONS:
                blk_max += 1
            else:
                atr_v      = float(ut_atr[i]) if not np.isnan(ut_atr[i]) else 0
                base_entry = float(ut_hc[i])
                if atr_v > 0:
                    if mode == "immediate":
                        entry_p = base_entry
                        sl  = entry_p - UT_SL_MULT * atr_v
                        tp  = entry_p + UT_TP_MULT * atr_v
                        btc = NOTIONAL / entry_p
                        capital -= TRADE_USDT
                        open_pos["UTBot"] = {"entry": entry_p, "sl": sl, "tp": tp,
                                             "btc": btc, "open_ts": ts_i,
                                             "base_entry": base_entry, "strat": "UTBot"}
                        ut_locked = True
                    else:
                        res = find_15m_entry(ts_i, UT_SL_MULT, UT_TP_MULT, atr_v)
                        if res is None:
                            misses += 1
                        else:
                            entry_p, idx_15m, sl, tp = res
                            entry_deltas.append((entry_p - base_entry) / base_entry * 100)
                            btc = NOTIONAL / entry_p
                            capital -= TRADE_USDT
                            entry_15m_ts = c15m[idx_15m].ts
                            open_pos["UTBot"] = {"entry": entry_p, "sl": sl, "tp": tp,
                                                 "btc": btc, "open_ts": entry_15m_ts,
                                                 "base_entry": base_entry, "strat": "UTBot"}
                            ut_locked = True

    # Close open positions at last price
    last_p  = float(c1h[-1].close)
    last_ts = c1h[-1].ts
    for sk, pos in open_pos.items():
        hrs   = (last_ts - pos["open_ts"]) / 3_600_000
        btc   = NOTIONAL / pos["entry"]
        gross = btc * (last_p - pos["entry"])
        comm  = commission(pos["entry"], last_p, btc)
        intr  = interest_cost(hrs)
        net   = gross - comm - intr
        capital += TRADE_USDT + net
        results.append({"res": "OPEN", "gross": gross, "comm": comm, "int": intr, "net": net,
                         "hrs": hrs, "entry": pos["entry"], "exit": last_p,
                         "base_entry": pos["base_entry"], "strat": sk})

    wins    = sum(1 for r in results if r["res"] == "TP")
    losses  = sum(1 for r in results if r["res"] == "SL")
    opens   = sum(1 for r in results if r["res"] == "OPEN")
    trades  = len(results)
    closed  = wins + losses
    wr      = wins / closed * 100 if closed else 0
    gross   = sum(r["gross"] for r in results)
    comm    = sum(r["comm"]  for r in results)
    intr    = sum(r["int"]   for r in results)
    net     = gross - comm - intr
    avg_hrs = sum(r["hrs"] for r in results) / trades if trades else 0
    avg_impr = sum(entry_deltas) / len(entry_deltas) if entry_deltas else 0.0

    return {
        "mode": mode,
        "trades": trades, "wins": wins, "losses": losses, "opens": opens,
        "wr": wr, "gross": gross, "comm": comm, "intr": intr, "net": net,
        "capital": capital, "avg_hrs": avg_hrs,
        "blk_mtf": blk_mtf, "blk_lock": blk_lock, "blk_max": blk_max,
        "misses": misses,
        "avg_entry_delta_pct": avg_impr,
        "results": results,
    }


def main():
    print("Loading data...")
    c1h  = load_tf("1h")
    c15m = load_tf("15m")
    c4h  = load_tf("4h")

    n   = len(c1h)
    t0  = ts_str(c1h[0].ts,  "%Y-%m-%d")
    t1  = ts_str(c1h[-1].ts, "%Y-%m-%d")
    months = (c1h[-1].ts - c1h[0].ts) / (1000 * 3600 * 24 * 30.44)
    print(f"  1H bars: {n} ({months:.1f} months) {t0} → {t1}")
    print(f"  15M bars: {len(c15m)}")

    print("Computing signals...")
    wt_buy, wt_sell, wt_atr, wt_hc, *_ = compute_wt_signals(c1h, n1=8, n2=12)
    ut_buy, ut_sell, ut_atr, ut_hc, *_ = compute_ut_signals(c1h)

    print("Computing 4H bias...")
    bias4h  = build_4h_bias(c4h)
    ts4map  = build_ts_index(c4h)
    bias_at = np.array([get_4h_bias_at(c1h[i].ts, c4h, ts4map, bias4h) for i in range(n)])
    bull_bars = int(bias_at.sum())
    print(f"  4H bullish bars: {bull_bars}/{n} ({bull_bars/n*100:.0f}%)")
    print(f"  WT BUY signals: {int(wt_buy.sum())}  |  UTBot BUY signals: {int(ut_buy.sum())}")

    print("Pre-computing 15M indicators...")
    closes_15m = [float(c.close) for c in c15m]
    ema20_15   = ema_arr(closes_15m, 20)
    hma15_15   = hma_arr(closes_15m, 15)

    print("Building 15M timestamp index...")
    ts15_map = {c.ts: i for i, c in enumerate(c15m)}

    print("\nRunning simulations...\n")
    sims = []
    for mode in ("immediate", "ema20", "hma15"):
        print(f"  [{mode}] ...", end=" ", flush=True)
        r = simulate(mode, c1h, c15m, ts15_map,
                     wt_buy, wt_atr, wt_hc,
                     ut_buy, ut_atr, ut_hc,
                     bias_at,
                     ema20_15, hma15_15)
        sims.append(r)
        print(f"trades={r['trades']}  TP={r['wins']}  SL={r['losses']}  "
              f"WR={r['wr']:.1f}%  NET={r['net']:+.2f}$  misses={r['misses']}")

    # ── Print comparison table ────────────────────────────────────────────────
    W = 82
    print()
    print("=" * W)
    print(f"  15M ENTRY FILTER COMPARISON — BTC/USDT  {t0} → {t1}")
    print(f"  Config A: WT#0 + UTBot | SL=1.5×ATR TP=1.0×ATR | 4H MTF gate | ${TRADE_USDT}×{LEVERAGE:.0f}=${NOTIONAL:.0f} notional")
    print("=" * W)
    print(f"\n  {'Metric':<28}  {'Immediate (1H close)':>20}  {'EMA20 (15M)':>15}  {'HMA15 (15M)':>15}")
    print(f"  {'─'*74}")

    lbl_to_idx = {"immediate": 0, "ema20": 1, "hma15": 2}

    def row(label, fn):
        vals = [fn(s) for s in sims]
        print(f"  {label:<28}  {vals[0]:>20}  {vals[1]:>15}  {vals[2]:>15}")

    row("Trades executed",     lambda s: str(s["trades"]))
    row("  TP hits",           lambda s: str(s["wins"]))
    row("  SL hits",           lambda s: str(s["losses"]))
    row("  Still open at end", lambda s: str(s["opens"]))
    row("  Missed (expired)",  lambda s: str(s["misses"]))
    row("Win Rate (TP/closed)", lambda s: f"{s['wr']:.1f}%" if s['wins']+s['losses'] > 0 else "—")
    row("Avg trade duration",  lambda s: f"{s['avg_hrs']:.1f}h")
    row("Avg entry vs 1H close", lambda s: (
        "baseline"
        if s["mode"] == "immediate"
        else f"{s['avg_entry_delta_pct']:+.3f}%"
    ))
    print(f"  {'─'*74}")
    row("Gross P&L",           lambda s: f"${s['gross']:+.2f}")
    row("Commission",          lambda s: f"-${s['comm']:.2f}")
    row("Margin interest",     lambda s: f"-${s['intr']:.2f}")
    row("NET P&L",             lambda s: f"${s['net']:+.2f}")
    row("Final capital",       lambda s: f"${s['capital']:.2f}")
    row("Return on $300",      lambda s: f"{(s['capital']-START_CAPITAL)/START_CAPITAL*100:+.1f}%")
    print(f"  {'─'*74}")
    row("Blocked by 4H gate",  lambda s: str(s["blk_mtf"]))
    row("Blocked (locked)",    lambda s: str(s["blk_lock"]))
    row("Blocked (max pos)",   lambda s: str(s["blk_max"]))
    print("=" * W)

    # ── Monthly breakdown for best mode ──────────────────────────────────────
    best = max(sims, key=lambda s: s["net"])
    print(f"\n  Monthly breakdown — best mode: {best['mode'].upper()}")
    print(f"  {'─'*50}")
    monthly = {}
    for r in best["results"]:
        # We don't track month per result easily; skip monthly for now
        pass

    # ── Per-strategy breakdown ────────────────────────────────────────────────
    print(f"\n  Per-strategy breakdown (NET P&L)")
    print(f"  {'─'*60}")
    print(f"  {'Strategy':<10}  {'Immediate':>12}  {'EMA20':>12}  {'HMA15':>12}")
    for strat in ("WT#0", "WT#1", "UTBot"):
        def strat_net(sim):
            rs = [r for r in sim["results"] if r["strat"] == strat]
            if not rs:
                return "—"
            n = sum(r["gross"] - r["comm"] - r["int"] for r in rs)
            return f"${n:+.2f}"
        vals = [strat_net(s) for s in sims]
        print(f"  {strat:<10}  {vals[0]:>12}  {vals[1]:>12}  {vals[2]:>12}")

    print()
    print("  NOTE: 'Immediate' = enter at 1H signal bar Heikin-Ashi close (current bot)")
    print("        'EMA20'     = enter when 15M close > EMA20(20) within 4h of signal")
    print("        'HMA15'     = enter when 15M close > HMA15(15) within 4h of signal")
    print("        Entry delta = avg % difference vs 1H close (negative = better entry)")
    print()


if __name__ == "__main__":
    main()
