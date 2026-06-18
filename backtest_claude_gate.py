"""
Backtest: WTADXStrategy + UTBotStrategy  WITH Claude AI confirmation gate

Same v4 entry filters (HMA + EMA5/SMA9 + RSI>50), but each BUY signal is
sent to Claude before executing.  Shows side-by-side comparison:

  Baseline  : v4 filters only (same as backtest_hma_filter.py)
  AI-Gated  : v4 filters + Claude must confirm the trade

Requires:  ANTHROPIC_API_KEY env var
           pip install anthropic

Cost note:  Each signal = 1 Claude call (~500 tok in + ~150 tok out).
            Typical run: 20-60 calls ≈ $0.01-0.03 total at claude-opus-4-8 rates.

Usage:
  ANTHROPIC_API_KEY=sk-ant-... python backtest_claude_gate.py
  CLAUDE_MODEL=claude-haiku-4-5 python backtest_claude_gate.py  # cheaper/faster
"""

import asyncio, sys, os, time, math, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from trading.connectors.base import OHLCV
from trading.strategies.base import BaseStrategy
from trading.strategies.wt_adx_strategy import WTADXStrategy
from trading.strategies.ut_bot_strategy import UTBotStrategy
from trading.claude_analyzer import ClaudeAnalyzer

# ─── Config ───────────────────────────────────────────────────────────────────
SYMBOL     = "BTC/USDT"
TIMEFRAME  = "1h"
MONTHS     = 5
CAPITAL    = 260.0
TRADE_USD  = 100.0
FEE_RATE   = 0.001
EMA_PERIOD = 5
SMA_PERIOD = 9
RSI_PERIOD = 14
MAX_BARS_IN = 120

STRATEGY_PARAMS = {
    "WTADXStrategy": dict(
        hma_period=100, sl_mult=2.5, tp_rr=2.0,
        cross_bars=0,   rsi_min=50,
        use_sell_sig=False, use_hma_exit=True,
    ),
    "UTBotStrategy": dict(
        hma_period=50,  sl_mult=2.0, tp_rr=1.5,
        cross_bars=3,   rsi_min=50,
        use_sell_sig=True, use_hma_exit=False,
    ),
}

# ─── Fetch candles ─────────────────────────────────────────────────────────────

def fetch_candles() -> list:
    since_ms = int(time.time() * 1000) - MONTHS * 31 * 24 * 3600 * 1000
    sym_b    = SYMBOL.replace("/", "")
    import urllib.request, json as _json
    for source, url_fn in [
        ("Binance", lambda f: (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={sym_b}&interval=1h&startTime={f}&limit=1000")),
        ("OKX", lambda f: (
            f"https://www.okx.com/api/v5/market/history-candles"
            f"?instId={SYMBOL.replace('/','-')}&bar=1H&limit=300&after={f}")),
    ]:
        try:
            req = urllib.request.Request(
                url_fn(since_ms), headers={"User-Agent": "backtest/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read())
            raw = data if isinstance(data, list) else data.get("data", [])
            if raw and len(raw) > 50:
                raw.sort(key=lambda r: int(r[0]))
                candles = [OHLCV(int(r[0]), float(r[1]), float(r[2]),
                                 float(r[3]), float(r[4]), float(r[5]))
                           for r in raw]
                print(f"  {source}: {len(candles)} candles")
                return candles
        except Exception:
            pass

    print("  ⚠ Network restricted — using SYNTHETIC BTC data")
    rng    = np.random.default_rng(42)
    n_bars = MONTHS * 31 * 24 + 50
    closes = [60_000.0]
    regimes = [
        (0.00, 0.15, +0.00010, 0.0050),
        (0.15, 0.30, -0.00008, 0.0070),
        (0.30, 0.50, +0.00005, 0.0040),
        (0.50, 0.65, +0.00015, 0.0055),
        (0.65, 0.80, -0.00012, 0.0080),
        (0.80, 1.00, +0.00010, 0.0045),
    ]
    for i in range(1, n_bars):
        frac = i / n_bars
        dr, vl = 0.0, 0.005
        for s, e, d, v in regimes:
            if s <= frac < e:
                dr, vl = d, v; break
        closes.append(closes[-1] * (1 + rng.normal(dr, vl)))
    candles = []
    ts0 = since_ms
    for i, c in enumerate(closes):
        o     = closes[i - 1] if i > 0 else c
        noise = abs(rng.normal(0, c * 0.0012))
        candles.append(OHLCV(ts0 + i * 3_600_000, round(o, 2),
                             round(max(o, c) + noise, 2),
                             round(min(o, c) - noise, 2),
                             round(c, 2), float(rng.uniform(200, 2000))))
    print(f"  Generated {len(candles)} synthetic bars  "
          f"(start=${closes[0]:,.0f}  end=${closes[-1]:,.0f})")
    return candles


# ─── Build indicator snapshot for Claude ──────────────────────────────────────

def build_snapshot(
    candles, i, price, atr_v, sl_px, tp_px, rr,
    hma_a, ema5_a, sma9_a, rsi_a, hma_period
) -> tuple[dict, list]:
    """Build indicators dict + recent_candles list for ClaudeAnalyzer."""
    e5  = float(ema5_a[i])
    s9  = float(sma9_a[i])
    hma = float(hma_a[i])
    rsi = float(rsi_a[i])

    def _safe(v): return None if math.isnan(v) else v

    indicators = {
        "rsi": _safe(rsi),
        "hma_period": hma_period,
        "price_above_hma": None if math.isnan(hma) else price > hma,
        "ema5": _safe(e5),
        "sma9": _safe(s9),
        "ema5_above_sma9": None if (math.isnan(e5) or math.isnan(s9)) else e5 > s9,
        "atr": _safe(atr_v),
        "sl_price": sl_px,
        "tp_price": tp_px,
        "rr": rr,
    }

    recent = candles[max(0, i - 9): i + 1]
    recent_candles = [
        {"open": float(c.open), "high": float(c.high),
         "low": float(c.low),  "close": float(c.close),
         "volume": float(c.volume)}
        for c in recent
    ]
    return indicators, recent_candles


# ─── Backtest engine (async — calls Claude per signal) ────────────────────────

async def run_strategy_with_ai(
    name: str,
    candles: list,
    buy_sig: np.ndarray,
    sell_sig: np.ndarray,
    atr14: np.ndarray,
    hma_a: np.ndarray,
    ema5_a: np.ndarray,
    sma9_a: np.ndarray,
    rsi_a: np.ndarray,
    analyzer: ClaudeAnalyzer,
    hma_period: int,
    sl_mult: float,
    tp_rr: float,
    cross_bars: int,
    rsi_min: float,
    use_sell_sig: bool,
    use_hma_exit: bool,
) -> tuple[list, list]:
    """
    Returns (baseline_trades, ai_trades).
    baseline_trades : all signals that pass v4 entry filters
    ai_trades       : subset that Claude also confirmed
    """
    n      = len(candles)
    closes = np.array([c.close for c in candles])
    highs  = np.array([c.high  for c in candles])
    lows   = np.array([c.low   for c in candles])
    opens  = np.array([c.open  for c in candles])

    valid_hma = np.where(~np.isnan(hma_a))[0]
    warmup    = max(int(valid_hma[0]) + 10 if len(valid_hma) else 110,
                   (np.where(~np.isnan(atr14))[0][0] if np.any(~np.isnan(atr14)) else 30) + 10,
                   RSI_PERIOD + 5)

    # We simulate two independent runs in one pass to share the AI calls.
    # base_* : tracks position for the baseline run
    # ai_*   : tracks position for the AI-gated run
    base_trades: list = []; ai_trades: list = []
    base_capital = ai_capital = CAPITAL

    base_in = ai_in = False
    base_entry = base_sl = base_tp = base_amount = base_bar = 0.0
    ai_entry   = ai_sl   = ai_tp   = ai_amount   = ai_bar   = 0.0
    base_pending_hma = ai_pending_hma = False

    ai_calls = 0
    ai_confirmations = 0
    ai_rejections = 0

    for i in range(warmup, n - 1):
        c_close = float(closes[i])
        c_high  = float(highs[i])
        c_low   = float(lows[i])
        hma_v   = float(hma_a[i])
        rsi_v   = float(rsi_a[i])
        atr_v   = float(atr14[i])
        e5      = float(ema5_a[i])
        s9      = float(sma9_a[i])

        if any(math.isnan(x) for x in [hma_v, rsi_v, atr_v, e5, s9]):
            continue

        # ── Exit: Baseline ────────────────────────────────────────────
        if base_in:
            ep, et = None, None
            if use_hma_exit and base_pending_hma:
                nxt = float(opens[i])
                if nxt < float(hma_a[i]):
                    ep, et = nxt, "hma_break"
                base_pending_hma = False
            if ep is None and c_high >= base_tp: ep, et = base_tp, "tp"
            if ep is None and c_low  <= base_sl: ep, et = base_sl, "sl"
            if ep is None and use_sell_sig and bool(sell_sig[i]):
                ep, et = c_close, "sell_signal"
            if ep is None and (i - base_bar) >= MAX_BARS_IN:
                ep, et = c_close, "timeout"
            if ep is None and use_hma_exit and c_close < hma_v:
                base_pending_hma = True
            if ep is not None:
                pnl = (ep - base_entry) * base_amount - ep * base_amount * FEE_RATE
                base_capital += pnl
                base_trades.append({
                    "bar": i, "entry": round(base_entry, 2), "exit": round(ep, 2),
                    "sl": round(base_sl, 2), "tp": round(base_tp, 2),
                    "pnl_net": round(pnl, 4),
                    "pnl_pct": round((ep - base_entry) / base_entry * 100, 3),
                    "type": et, "capital": round(base_capital, 2),
                    "ai_confirmed": None,  # not applicable for baseline
                })
                base_in = False; base_pending_hma = False

        # ── Exit: AI-gated ────────────────────────────────────────────
        if ai_in:
            ep, et = None, None
            if use_hma_exit and ai_pending_hma:
                nxt = float(opens[i])
                if nxt < float(hma_a[i]):
                    ep, et = nxt, "hma_break"
                ai_pending_hma = False
            if ep is None and c_high >= ai_tp: ep, et = ai_tp, "tp"
            if ep is None and c_low  <= ai_sl: ep, et = ai_sl, "sl"
            if ep is None and use_sell_sig and bool(sell_sig[i]):
                ep, et = c_close, "sell_signal"
            if ep is None and (i - ai_bar) >= MAX_BARS_IN:
                ep, et = c_close, "timeout"
            if ep is None and use_hma_exit and c_close < hma_v:
                ai_pending_hma = True
            if ep is not None:
                pnl = (ep - ai_entry) * ai_amount - ep * ai_amount * FEE_RATE
                ai_capital += pnl
                ai_trades.append({
                    "bar": i, "entry": round(ai_entry, 2), "exit": round(ep, 2),
                    "sl": round(ai_sl, 2), "tp": round(ai_tp, 2),
                    "pnl_net": round(pnl, 4),
                    "pnl_pct": round((ep - ai_entry) / ai_entry * 100, 3),
                    "type": et, "capital": round(ai_capital, 2),
                    "ai_confirmed": True,
                })
                ai_in = False; ai_pending_hma = False

        # ── Skip if both already in position ──────────────────────────
        if base_in and ai_in:
            continue

        # ── Entry filters (same v4 logic) ─────────────────────────────
        if not bool(buy_sig[i]):
            continue
        if i < 1 or float(closes[i - 1]) <= float(hma_a[i - 1]):
            continue
        if cross_bars == 0:
            if e5 <= s9: continue
        else:
            crossed = False
            for k in range(max(1, i - cross_bars + 1), i + 1):
                if (float(ema5_a[k - 1]) <= float(sma9_a[k - 1]) and
                        float(ema5_a[k]) > float(sma9_a[k])):
                    crossed = True; break
            if not crossed: continue
        if rsi_v < rsi_min: continue
        if math.isnan(atr_v) or atr_v <= 0: continue

        entry_px = c_close
        sl_px    = round(entry_px - sl_mult * atr_v, 2)
        tp_px    = round(entry_px + sl_mult * tp_rr * atr_v, 2)
        amount   = TRADE_USD / entry_px

        # ── Baseline: take the trade ───────────────────────────────────
        if not base_in:
            base_entry = entry_px; base_sl = sl_px; base_tp = tp_px
            base_amount = amount
            base_capital -= entry_px * amount * FEE_RATE
            base_in = True; base_bar = i; base_pending_hma = False

        # ── AI gate: ask Claude ────────────────────────────────────────
        if not ai_in:
            indicators, recent_candles = build_snapshot(
                candles, i, entry_px, atr_v, sl_px, tp_px, tp_rr,
                hma_a, ema5_a, sma9_a, rsi_a, hma_period,
            )
            ai_calls += 1
            confirmed, ai_reason = await analyzer.confirm_buy(
                symbol=SYMBOL,
                price=entry_px,
                strategy_name=name,
                signal_reason=f"bar={i} RSI={rsi_v:.1f} HMA={hma_v:.0f}",
                indicators=indicators,
                recent_candles=recent_candles,
            )
            print(f"  [AI #{ai_calls}] bar={i:>4} ${entry_px:>9,.0f} "
                  f"{'✓ CONFIRM' if confirmed else '✗ REJECT ':9s}  {ai_reason}")

            if confirmed:
                ai_confirmations += 1
                ai_entry = entry_px; ai_sl = sl_px; ai_tp = tp_px
                ai_amount = amount
                ai_capital -= entry_px * amount * FEE_RATE
                ai_in = True; ai_bar = i; ai_pending_hma = False
            else:
                ai_rejections += 1

    print(f"\n  [{name}] AI calls={ai_calls}  confirmed={ai_confirmations}  "
          f"rejected={ai_rejections}")
    return base_trades, ai_trades


# ─── Report ───────────────────────────────────────────────────────────────────

def _stats(trades: list, capital: float = CAPITAL) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "wr": 0.0, "net": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "pf": 0.0, "final": capital}
    wins   = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] <= 0]
    net    = sum(t["pnl_net"] for t in trades)
    aw     = sum(t["pnl_net"] for t in wins)   / max(len(wins),   1)
    al     = sum(t["pnl_net"] for t in losses) / max(len(losses), 1)
    pf     = abs(sum(t["pnl_net"] for t in wins)) / max(abs(sum(t["pnl_net"] for t in losses)), 1e-9)
    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "wr": len(wins) / len(trades) * 100 if trades else 0.0,
        "net": net, "avg_win": aw, "avg_loss": al, "pf": pf,
        "final": trades[-1]["capital"] if trades else capital,
    }


def print_comparison(name: str, base_trades: list, ai_trades: list, params: dict):
    W  = 72
    b  = _stats(base_trades)
    ai = _stats(ai_trades)

    hma_p = params["hma_period"]
    sl_m  = params["sl_mult"]
    tp_r  = params["tp_rr"]

    print(f"\n{'═'*W}")
    print(f"  {name}  |  {SYMBOL} {TIMEFRAME}  |  ${TRADE_USD}/trade")
    print(f"  Entry: HMA{hma_p} + EMA5>SMA9 + RSI>50 + SL={sl_m}×ATR TP={tp_r}R")
    print(f"{'─'*W}")
    print(f"  {'Metric':<22}  {'Baseline (v4)':>16}  {'With Claude Gate':>16}")
    print(f"  {'─'*20}  {'─'*16}  {'─'*16}")
    print(f"  {'Trades':<22}  {b['trades']:>16}  {ai['trades']:>16}")
    print(f"  {'Win / Loss':<22}  {b['wins']}W / {b['losses']}L  {'':>5}  "
          f"{ai['wins']}W / {ai['losses']}L")
    print(f"  {'Win Rate':<22}  {b['wr']:>15.1f}%  {ai['wr']:>15.1f}%")
    print(f"  {'Profit Factor':<22}  {b['pf']:>16.2f}  {ai['pf']:>16.2f}")
    print(f"  {'Avg Win':<22}  ${b['avg_win']:>+14.2f}  ${ai['avg_win']:>+14.2f}")
    print(f"  {'Avg Loss':<22}  ${b['avg_loss']:>+14.2f}  ${ai['avg_loss']:>+14.2f}")
    print(f"  {'Net P/L':<22}  ${b['net']:>+14.2f}  ${ai['net']:>+14.2f}")
    roi_b  = (b['final']  - CAPITAL) / CAPITAL * 100
    roi_ai = (ai['final'] - CAPITAL) / CAPITAL * 100
    print(f"  {'ROI':<22}  {roi_b:>+15.1f}%  {roi_ai:>+15.1f}%")
    improvement = ai['net'] - b['net']
    print(f"\n  AI gate improvement: ${improvement:>+.2f}  "
          f"({'better' if improvement > 0 else 'worse' if improvement < 0 else 'same'})")

    # Exit breakdown for AI trades
    if ai_trades:
        exit_counts: dict = {}
        for t in ai_trades:
            exit_counts[t["type"]] = exit_counts.get(t["type"], 0) + 1
        print(f"\n  AI-gated exit breakdown:")
        for k, v in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"    {k:<16}: {v:>3}  ({v/len(ai_trades)*100:.0f}%)")

    print(f"{'═'*W}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async():
    W = 72
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model   = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")

    print("\n" + "═"*W)
    print(f"  Backtest: Claude AI Gate  |  WTADXStrategy + UTBotStrategy")
    print(f"  {SYMBOL} {TIMEFRAME}  |  last {MONTHS} months  |  ${CAPITAL:.0f} capital")
    if api_key:
        print(f"  Claude model: {model}")
    else:
        print(f"  ⚠  ANTHROPIC_API_KEY not set — AI gate will PASS all signals through")
        print(f"     Set ANTHROPIC_API_KEY=sk-ant-... to enable real AI filtering")
    print("═"*W)

    candles = fetch_candles()
    if len(candles) < 300:
        print("ERROR: Not enough candles"); sys.exit(1)

    n      = len(candles)
    closes = [c.close for c in candles]
    print(f"\n  Computing indicators on {n} bars...")

    ema5_a = BaseStrategy.ema(closes, EMA_PERIOD)
    sma9_a = BaseStrategy.sma(closes, SMA_PERIOD)
    atr14  = BaseStrategy.atr(candles, 14)
    rsi14  = BaseStrategy.rsi(closes, RSI_PERIOD)

    hma_cache = {}
    for p in STRATEGY_PARAMS.values():
        hp = p["hma_period"]
        if hp not in hma_cache:
            hma_cache[hp] = BaseStrategy.hma(closes, hp)

    wt_strat = WTADXStrategy(SYMBOL)
    _, _, wt_buy, wt_sell, _ = wt_strat._build_signals(candles)

    ut_strat = UTBotStrategy(SYMBOL)
    ut_buy, ut_sell, _, _ = ut_strat._build_signals(candles)

    analyzer = ClaudeAnalyzer(model=model)

    print(f"\n  {'─'*W}")
    print(f"  Running WTADXStrategy + AI gate...")
    wt_p = STRATEGY_PARAMS["WTADXStrategy"]
    wt_base, wt_ai = await run_strategy_with_ai(
        "WTADXStrategy", candles,
        wt_buy.astype(bool), wt_sell.astype(bool),
        atr14, hma_cache[wt_p["hma_period"]],
        ema5_a, sma9_a, rsi14, analyzer, **wt_p,
    )

    print(f"\n  {'─'*W}")
    print(f"  Running UTBotStrategy + AI gate...")
    ut_p = STRATEGY_PARAMS["UTBotStrategy"]
    ut_base, ut_ai = await run_strategy_with_ai(
        "UTBotStrategy", candles,
        ut_buy.astype(bool), ut_sell.astype(bool),
        atr14, hma_cache[ut_p["hma_period"]],
        ema5_a, sma9_a, rsi14, analyzer, **ut_p,
    )

    # ── Results ────────────────────────────────────────────────────────
    print_comparison("WTADXStrategy", wt_base, wt_ai, wt_p)
    print_comparison("UTBotStrategy", ut_base, ut_ai, ut_p)

    # Combined summary
    all_base = wt_base + ut_base
    all_ai   = wt_ai   + ut_ai
    if all_base:
        b  = _stats(all_base)
        ai = _stats(all_ai)
        improvement = ai['net'] - b['net']
        print(f"\n{'═'*W}")
        print(f"  COMBINED SUMMARY  |  {MONTHS}-month backtest")
        print(f"  {'─'*W}")
        print(f"  Baseline:   {b['trades']:>3} trades  WR={b['wr']:.1f}%  "
              f"Net=${b['net']:>+.2f}  PF={b['pf']:.2f}")
        print(f"  AI-Gated:   {ai['trades']:>3} trades  WR={ai['wr']:.1f}%  "
              f"Net=${ai['net']:>+.2f}  PF={ai['pf']:.2f}")
        print(f"  Trade reduction: {b['trades'] - ai['trades']} fewer trades "
              f"({(b['trades'] - ai['trades'])/max(b['trades'],1)*100:.0f}% filtered)")
        print(f"  P/L improvement: ${improvement:>+.2f}")
        if not api_key:
            print(f"\n  ⚠  Results above are IDENTICAL because no API key was provided.")
            print(f"     Claude passed all signals through (fallback mode).")
            print(f"     Set ANTHROPIC_API_KEY to see real AI filtering.")
        print(f"{'═'*W}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
