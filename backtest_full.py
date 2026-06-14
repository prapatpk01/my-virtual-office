"""
Full 7-Day Backtest  |  3 Strategies × 3 Symbols  |  WT1 Gate + Position Lock
Capital simulation: $500 start, $10 risk/trade.

Models real bot behaviour:
  - One open trade per symbol at a time (all strategies locked until exit)
  - Strategies checked in priority order: WaveTrend → MACD/EMA → Momentum
  - WT1 gate applied to MACD and Momentum signals
  - Exit bar determined by first SL/TP hit or timeout (60 bars)
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.trading.connectors.base import OHLCV
from app.trading.strategies.wt_adx_strategy import WTADXStrategy
from app.trading.strategies.momentum_score_strategy import MomentumScoreStrategy
from app.trading.strategies.macd_ema_strategy import MACDEMAStrategy

# ─── Constants ─────────────────────────────────────────────────────────────
WARMUP   = 285
DAYS     = 7
BPD      = 96           # 15m bars per day
N_BARS   = WARMUP + DAYS * BPD
LOOKFWD  = 60           # max bars to scan for SL/TP exit
RISK     = 10.0
CAP_0    = 500.0
ATR_P    = 14

SYMBOLS = [
    ("BTC/USDT", 65_000.0, 0.0015, 0.0008, 0.00030, 42),
    ("XAUUSD",   2_050.0,  0.0008, 0.0004, 0.00020, 43),
    ("EURUSD",      1.085, 0.0004, 0.0002, 0.00010, 44),
]

STRAT_RR = {"WaveTrend": 1.5, "MACD/EMA": 1.5, "Momentum": 1.5}


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


# ─── Find trade exit ───────────────────────────────────────────────────────
def find_exit(highs, lows, idx, direction, sl_p, tp_p):
    """Returns (exit_bar_idx, outcome). outcome: +1=win, -1=loss, 0=timeout."""
    n = len(highs)
    for j in range(idx + 1, min(idx + LOOKFWD, n)):
        if direction == 1:
            if lows[j]  <= sl_p: return j, -1
            if highs[j] >= tp_p: return j,  1
        else:
            if highs[j] >= sl_p: return j, -1
            if lows[j]  <= tp_p: return j,  1
    return min(idx + LOOKFWD, n - 1), 0


# ─── WT1 Gate ──────────────────────────────────────────────────────────────
def wt1_gate(wt1_a, idx, direction):
    w = float(wt1_a[idx])
    if np.isnan(w):                return True
    if direction ==  1 and w >= 10: return False
    if direction == -1 and w <= -10: return False
    return True


# ─── Main Backtest ─────────────────────────────────────────────────────────
def run_backtest():
    """
    Returns list of signal tuples:
      (day, strategy, symbol, direction, entry, sl, tp, rr, outcome)
    """
    all_signals = []

    for sym, start, vt, vr, drift, seed in SYMBOLS:
        candles = gen_candles(N_BARS, start, vt, vr, drift, seed)
        closes  = np.array([c.close for c in candles])
        highs   = np.array([c.high  for c in candles])
        lows    = np.array([c.low   for c in candles])

        # Strategy instances
        wt_s = WTADXStrategy(sym)
        mo_s = MomentumScoreStrategy(sym)
        ma_s = MACDEMAStrategy(sym)

        # ── WaveTrend + ATR (dual-confirm: SJ+UT) ───────────────────────────
        (wt1_a, wt2_a, wt_tsl,
         wt_buy_f, wt_sell_f, wt_ut_buy, wt_ut_sell,
         wt_buy_sig, wt_sell_sig,
         atr_a, _) = wt_s._build_signals(candles)

        # ── MACD/EMA indicators ──────────────────────────────────────────
        (ma_closes, ma_highs, ma_lows, ma_vols,
         ma_hma, ma_ema9, ma_sma21, ma_ml, ma_sl, ma_hist,
         ma_adx_a, ma_atr2, ma_volma, ma_ha_o, ma_ha_c) = ma_s._build_arrays(candles)

        # ── Momentum (MC·DSA) indicators ─────────────────────────────────
        (mo_ef, mo_es, mo_ml, mo_sl, mo_hi,
         mo_rf, mo_rs, mo_ka, mo_da,
         mo_roc, mo_rocm, mo_mfi,
         mo_adx, mo_dip, mo_dim, mo_atr,
         mo_res, mo_sup, mo_cn, mo_hn, mo_ln) = mo_s._all_indicators(candles)

        # Min start bars
        wt_min = max(wt_s.n1 + wt_s.n2 + max(ATR_P, wt_s.ut_atr_len) + 10, WARMUP)
        ma_min = max(ma_s.macd_slow + ma_s.macd_sig + ma_s.vol_len + ma_s.hma_period + 5, WARMUP)
        mo_min = WARMUP

        def day_of(i): return (i - WARMUP) // BPD + 1

        sym_locked_until = -1
        prev = {"WaveTrend": 0, "MACD/EMA": 0, "Momentum": 0}

        for i in range(WARMUP, N_BARS - 1):
            day = day_of(i)
            if day < 1 or day > DAYS: break

            if i <= sym_locked_until:
                continue

            atr_v = float(atr_a[i]) if not np.isnan(atr_a[i]) else 0.0
            entry = float(closes[i])
            if atr_v <= 0:
                continue

            fired = False

            # ══ Check strategies in priority order ════════════════════

            # 1. WaveTrend (SJ+UT dual-confirm) ───────────────────────────
            if not fired and i >= wt_min:
                buy  = bool(wt_buy_sig[i])
                sell = bool(wt_sell_sig[i])

                if buy and prev["WaveTrend"] != 1:
                    rr   = STRAT_RR["WaveTrend"]
                    sl_p = entry - 1.5 * atr_v
                    tp_p = entry + 1.5 * rr * atr_v
                    eb, out = find_exit(highs, lows, i, 1, sl_p, tp_p)
                    all_signals.append((day, "WaveTrend", sym, 1, entry, sl_p, tp_p, rr, out))
                    sym_locked_until = eb
                    prev["WaveTrend"] = 1
                    fired = True
                elif sell and prev["WaveTrend"] != -1:
                    rr   = STRAT_RR["WaveTrend"]
                    sl_p = entry + 1.5 * atr_v
                    tp_p = entry - 1.5 * rr * atr_v
                    eb, out = find_exit(highs, lows, i, -1, sl_p, tp_p)
                    all_signals.append((day, "WaveTrend", sym, -1, entry, sl_p, tp_p, rr, out))
                    sym_locked_until = eb
                    prev["WaveTrend"] = -1
                    fired = True
                elif not buy and not sell:
                    prev["WaveTrend"] = 0

            # 2. MACD/EMA ──────────────────────────────────────────────
            if not fired and i >= ma_min:
                d = ma_s._signal_at(i, ma_closes, ma_highs, ma_lows, ma_vols,
                                    ma_hma, ma_ema9, ma_sma21, ma_ml, ma_sl, ma_hist,
                                    ma_adx_a, ma_atr2, ma_volma, ma_ha_o, ma_ha_c)
                if d == 1 and prev["MACD/EMA"] != 1:
                    if wt1_gate(wt1_a, i, 1):
                        rr   = STRAT_RR["MACD/EMA"]
                        atr_i = float(ma_atr2[i]) if not np.isnan(ma_atr2[i]) else atr_v
                        sl_p = entry - ma_s.sl_atr_mult * atr_i
                        tp_p = entry + ma_s.sl_atr_mult * rr * atr_i
                        eb, out = find_exit(highs, lows, i, 1, sl_p, tp_p)
                        all_signals.append((day, "MACD/EMA", sym, 1, entry, sl_p, tp_p, rr, out))
                        sym_locked_until = eb
                        fired = True
                    prev["MACD/EMA"] = 1
                elif d == -1 and prev["MACD/EMA"] != -1:
                    if wt1_gate(wt1_a, i, -1):
                        rr   = STRAT_RR["MACD/EMA"]
                        atr_i = float(ma_atr2[i]) if not np.isnan(ma_atr2[i]) else atr_v
                        sl_p = entry + ma_s.sl_atr_mult * atr_i
                        tp_p = entry - ma_s.sl_atr_mult * rr * atr_i
                        eb, out = find_exit(highs, lows, i, -1, sl_p, tp_p)
                        all_signals.append((day, "MACD/EMA", sym, -1, entry, sl_p, tp_p, rr, out))
                        sym_locked_until = eb
                        fired = True
                    prev["MACD/EMA"] = -1
                elif d == 0:
                    prev["MACD/EMA"] = 0

            # 3. MomentumScore (MC·DSA) ─────────────────────────────────
            if not fired and i >= mo_min:
                check = [mo_ef[i], mo_es[i], mo_ml[i], mo_sl[i], mo_hi[i],
                         mo_rf[i], mo_rs[i], mo_ka[i], mo_da[i],
                         mo_roc[i], mo_rocm[i], mo_mfi[i], mo_atr[i]]
                if not any(np.isnan(v) for v in check):
                    sr_sc = mo_s._sr_score(
                        float(mo_cn[i]), float(mo_res[i]), float(mo_sup[i]))

                    (ema_sc, macd_sc, macd_bull, macd_bear,
                     rsi_sc, stoch_sc, roc_sc, mfi_sc, adx_sc,
                     adx_bull, adx_bear, adx_v, dip_v, dim_v) = mo_s._bar_scores(
                        i, mo_ef, mo_es, mo_ml, mo_sl, mo_hi,
                        mo_rf, mo_rs, mo_ka, mo_da,
                        mo_roc, mo_rocm, mo_mfi,
                        mo_adx, mo_dip, mo_dim, mo_cn)

                    scores     = [ema_sc, sr_sc, macd_sc, rsi_sc, stoch_sc,
                                  roc_sc, mfi_sc, adx_sc]
                    bull_score = sum(1 for s in scores if s ==  1)
                    bear_score = sum(1 for s in scores if s == -1)

                    mom_bull = bull_score >= mo_s.min_bull_score and adx_bull and macd_bull
                    mom_bear = bear_score >= mo_s.min_bear_score and adx_bear and macd_bear

                    trig_bull = (
                        (mo_ka[i-1] <= mo_da[i-1] and mo_ka[i] > mo_da[i] and mo_ka[i-1] < 50) or
                        (mo_ml[i-1] <= mo_sl[i-1] and mo_ml[i] > mo_sl[i]) or
                        (mo_ef[i-1] <= mo_es[i-1] and mo_ef[i] > mo_es[i]))
                    trig_bear = (
                        (mo_ka[i-1] >= mo_da[i-1] and mo_ka[i] < mo_da[i] and mo_ka[i-1] > 50) or
                        (mo_ml[i-1] >= mo_sl[i-1] and mo_ml[i] < mo_sl[i]) or
                        (mo_ef[i-1] >= mo_es[i-1] and mo_ef[i] < mo_es[i]))

                    buy  = mom_bull and trig_bull
                    sell = mom_bear and trig_bear

                    if buy and prev["Momentum"] != 1:
                        if wt1_gate(wt1_a, i, 1):
                            rr   = STRAT_RR["Momentum"]
                            sl_p = entry - mo_s.sl_atr_mult * float(mo_atr[i])
                            tp_p = entry + mo_s.sl_atr_mult * mo_s.rr_ratio * float(mo_atr[i])
                            eb, out = find_exit(highs, lows, i, 1, sl_p, tp_p)
                            all_signals.append((day, "Momentum", sym, 1,
                                                entry, sl_p, tp_p, rr, out))
                            sym_locked_until = eb
                            fired = True
                        prev["Momentum"] = 1
                    elif sell and prev["Momentum"] != -1:
                        if wt1_gate(wt1_a, i, -1):
                            rr   = STRAT_RR["Momentum"]
                            sl_p = entry + mo_s.sl_atr_mult * float(mo_atr[i])
                            tp_p = entry - mo_s.sl_atr_mult * mo_s.rr_ratio * float(mo_atr[i])
                            eb, out = find_exit(highs, lows, i, -1, sl_p, tp_p)
                            all_signals.append((day, "Momentum", sym, -1,
                                                entry, sl_p, tp_p, rr, out))
                            sym_locked_until = eb
                            fired = True
                        prev["Momentum"] = -1
                    elif not buy and not sell:
                        prev["Momentum"] = 0

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
    print("  FULL 7-DAY BACKTEST  |  3 Strategies × 3 Symbols  |  WT1 Gate")
    print("  Position lock per symbol  |  SL=1.5×ATR  |  15m synthetic data")
    print("═" * W)

    # ── Day-by-day ──────────────────────────────────────────────────
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

    # ── Per-strategy ─────────────────────────────────────────────────
    print(f"\n  {'Strategy':<13}{'Signals':>8}{'W':>5}{'L':>5}{'WR':>7}{'Total R':>9}{'$/trade':>9}")
    print("  " + "─" * (W - 2))
    for sn in ["WaveTrend", "MACD/EMA", "Momentum"]:
        sg = [s for s in all_signals if s[1] == sn]
        cl = [s for s in sg if s[8] != 0]
        w  = sum(1 for s in cl if s[8] ==  1)
        l  = sum(1 for s in cl if s[8] == -1)
        tr = sum(s[7] for s in cl if s[8] == 1) - float(l)
        pnl = sum(RISK * s[7] for s in cl if s[8] == 1) - RISK * l
        wr  = f"{w/len(cl)*100:.1f}%" if cl else "—"
        dpt = pnl / len(cl) if cl else 0.0
        print(f"  {sn:<13}{len(sg):>8}{w:>5}{l:>5}{wr:>7}{tr:>+8.1f}R{dpt:>+8.2f}")

    # ── Per-symbol ───────────────────────────────────────────────────
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

    # ── Capital ──────────────────────────────────────────────────────
    final_cap, max_dd, day_pnl_map = simulate_capital(all_signals)
    pnl_total = final_cap - CAP_0
    roi = pnl_total / CAP_0 * 100
    closed_all = [s for s in all_signals if s[8] != 0]
    wr_all = tot_w / tot_cl * 100 if tot_cl else 0

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
    print(f"  Win Rate    : {wr_all:.1f}%  ({tot_w}W / {tot_l}L)")
    print(f"  Max Drawdown: ${max_dd:.2f}")

    monthly_roi = roi / DAYS * 30
    open_count = len(all_signals) - len(closed_all)
    print(f"\n  ─── Estimate ──────────────────────────────────────────")
    print(f"  Projected 30d ROI : {monthly_roi:+.2f}%")
    print(f"  สัญญาณ 30 วัน    : ~{tot_sig/DAYS*30:.0f} trades")
    print(f"  รอผล (timeout)   : {open_count} trades ยังไม่ถึง SL/TP")
    print("═" * W + "\n")


if __name__ == "__main__":
    print("กำลังรัน 7-day backtest พร้อม position lock...")
    signals = run_backtest()
    print_report(signals)
