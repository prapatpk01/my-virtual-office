"""
Simple walk-forward backtester for futures strategies.
Uses the same strategy.analyze() as live trading.
Supports long + short positions with fixed % or ATR-based SL/TP.
"""
import asyncio
import bisect
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .strategies.base import BaseStrategy, SignalType
from .connectors.base import OHLCV

logger = logging.getLogger("backtester")

TAKER_FEE = 0.0005   # OKX taker fee 0.05% per side


@dataclass
class BTrade:
    symbol: str
    strategy: str
    side: str           # "long" | "short"
    entry: float
    sl: float
    tp: float
    entry_ts: int
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_ts: int = 0
    pnl_usdt: float = 0.0


def _sl_tp(signal, price: float, side: str,
           sl_pct: float, tp_pct: float) -> tuple[float, float]:
    meta = signal.metadata or {}
    sl = meta.get("stop_loss")
    tp = meta.get("take_profit")
    if sl and tp:
        if side == "long":
            if sl > price: sl = round(price * (1 - sl_pct), 6)
            if tp < price: tp = round(price * (1 + tp_pct), 6)
        elif side == "short":
            if sl < price: sl = round(price * (1 + sl_pct), 6)
            if tp > price: tp = round(price * (1 - tp_pct), 6)
        if sl and tp:
            return sl, tp
    if side == "long":
        return round(price * (1 - sl_pct), 6), round(price * (1 + tp_pct), 6)
    return round(price * (1 + sl_pct), 6), round(price * (1 - tp_pct), 6)


async def backtest_strategy(
    strategy: BaseStrategy,
    candles: list[OHLCV],
    notional: float,
    sl_pct: float,
    tp_pct: float,
    warmup: int = 60,
) -> list[BTrade]:
    """
    Walk-forward backtest. For each bar, feeds all prior candles to strategy.analyze(),
    then checks if current bar's high/low triggers SL or TP on open positions.
    """
    trades: list[BTrade] = []
    open_long:  Optional[BTrade] = None
    open_short: Optional[BTrade] = None

    for i in range(warmup, len(candles)):
        bar    = candles[i]
        price  = float(bar.close)
        b_high = float(bar.high)
        b_low  = float(bar.low)
        ts     = int(bar.timestamp)

        # ── Check exits on open positions ────────────────────────────────
        for pos, side in [(open_long, "long"), (open_short, "short")]:
            if pos is None:
                continue
            hit_reason = hit_price = None
            if side == "long":
                if b_low  <= pos.sl:   hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_high >= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            else:
                if b_high >= pos.sl:  hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_low <= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            if hit_reason:
                mult    = 1 if side == "long" else -1
                pnl_pct = mult * (hit_price - pos.entry) / pos.entry
                fee     = notional * TAKER_FEE * 2   # open + close
                pos.exit_price  = hit_price
                pos.exit_reason = hit_reason
                pos.exit_ts     = ts
                pos.pnl_usdt    = round(pnl_pct * notional - fee, 4)
                trades.append(pos)
                if side == "long":  open_long  = None
                else:               open_short = None

        # ── Run strategy ─────────────────────────────────────────────────
        # Pass candles[:i+1] so that [-1] is bar i (mimics live "forming" bar)
        # and [-2] is the last closed bar, matching strategy index conventions.
        try:
            signal = await strategy.analyze(candles[:i + 1], price)
        except Exception as e:
            logger.debug("analyze error at bar %d: %s", i, e)
            continue

        # 1-position-per-strategy: open only when no position on either side
        if signal.type == SignalType.BUY and open_long is None and open_short is None:
            sl, tp = _sl_tp(signal, price, "long", sl_pct, tp_pct)
            open_long = BTrade(
                symbol=strategy.symbol, strategy=strategy.name,
                side="long", entry=price, sl=sl, tp=tp, entry_ts=ts,
            )
        elif signal.type == SignalType.SELL and open_short is None and open_long is None:
            sl, tp = _sl_tp(signal, price, "short", sl_pct, tp_pct)
            open_short = BTrade(
                symbol=strategy.symbol, strategy=strategy.name,
                side="short", entry=price, sl=sl, tp=tp, entry_ts=ts,
            )

    return trades


@dataclass
class BTradeV2:
    """Partial-close trade: TP1 closes part of the position, then SL→breakeven, TP2 closes rest."""
    symbol: str
    strategy: str
    side: str
    entry: float
    sl: float            # mutable — moves to breakeven after TP1
    tp1: float
    tp2: float
    entry_ts: int
    notional: float
    partial_pct: float
    remaining: float = 1.0
    tp1_hit: bool = False
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_ts: int = 0
    pnl_usdt: float = 0.0   # net realized PnL after all fees (compatible with summarise())


def _realize(pos: BTradeV2, frac: float, px: float) -> float:
    """Gross PnL minus this portion's close-side taker fee."""
    mult    = 1 if pos.side == "long" else -1
    pnl_pct = mult * (px - pos.entry) / pos.entry
    gross   = pnl_pct * pos.notional * frac
    fee     = pos.notional * frac * TAKER_FEE
    return gross - fee


async def backtest_strategy_mtf_v2(
    strategy: BaseStrategy,
    primary_candles: list[OHLCV],
    mtf_candles: dict[str, list[OHLCV]],
    notional: float,
    warmup: int = 100,
    primary_window: int = 300,
    mtf_windows: Optional[dict] = None,
) -> tuple[list[BTradeV2], dict]:
    """
    Walk-forward MTF backtest with TP1 / breakeven / TP2 partial-close logic.

    Reads signal.metadata: tp1, tp2, stop_loss (initial SL), breakeven, partial_pct.
    Per bar, after entry:
      • before TP1: full position stops out at initial SL, or TP1 closes partial_pct
        and the remaining SL jumps to breakeven (entry).
      • after  TP1: remaining closes at TP2, or at the breakeven SL (risk-free runner).
    Conservative intrabar rule: SL is checked before TP within the same bar.
    """
    trades: list[BTradeV2] = []
    open_long:  Optional[BTradeV2] = None
    open_short: Optional[BTradeV2] = None
    stats = {"tp1": 0, "tp2": 0, "sl_full": 0, "breakeven": 0}

    _mtf_windows = {"1h": 200, "4h": 120, "30m": 300}
    if mtf_windows:
        _mtf_windows.update(mtf_windows)
    mtf_ts_index = {tf: [c.timestamp for c in clist] for tf, clist in mtf_candles.items()}

    def _step_exit(pos: BTradeV2, b_high: float, b_low: float, ts: int) -> bool:
        """Advance one bar of exit logic. Returns True if the position is now fully closed."""
        long = pos.side == "long"
        sl_hit = (b_low <= pos.sl) if long else (b_high >= pos.sl)
        if not pos.tp1_hit:
            tp1_hit = (b_high >= pos.tp1) if long else (b_low <= pos.tp1)
            if sl_hit:                                   # full stop before any TP
                pos.pnl_usdt += _realize(pos, pos.remaining, pos.sl)
                pos.exit_reason = "stop_loss"; pos.exit_price = pos.sl; pos.exit_ts = ts
                stats["sl_full"] += 1
                return True
            if tp1_hit:                                  # bank TP1, move SL → breakeven
                pos.pnl_usdt += _realize(pos, pos.partial_pct, pos.tp1)
                pos.remaining -= pos.partial_pct
                pos.tp1_hit = True
                pos.sl = pos.entry                       # breakeven (risk-free)
                stats["tp1"] += 1
            return False
        # ── after TP1: runner protected at breakeven ──
        be_hit = (b_low <= pos.sl) if long else (b_high >= pos.sl)
        tp2_hit = (b_high >= pos.tp2) if long else (b_low <= pos.tp2)
        if be_hit:
            pos.pnl_usdt += _realize(pos, pos.remaining, pos.sl)
            pos.exit_reason = "breakeven"; pos.exit_price = pos.sl; pos.exit_ts = ts
            stats["breakeven"] += 1
            return True
        if tp2_hit:
            pos.pnl_usdt += _realize(pos, pos.remaining, pos.tp2)
            pos.exit_reason = "take_profit2"; pos.exit_price = pos.tp2; pos.exit_ts = ts
            stats["tp2"] += 1
            return True
        return False

    for i in range(warmup, len(primary_candles)):
        bar    = primary_candles[i]
        ts     = bar.timestamp
        price  = float(bar.close)
        b_high = float(bar.high)
        b_low  = float(bar.low)

        if open_long is not None and _step_exit(open_long, b_high, b_low, ts):
            trades.append(open_long); open_long = None
        if open_short is not None and _step_exit(open_short, b_high, b_low, ts):
            trades.append(open_short); open_short = None

        prim_slice = primary_candles[max(0, i - primary_window): i + 1]
        mtf_sliced: dict[str, list[OHLCV]] = {}
        for tf, ts_idx in mtf_ts_index.items():
            idx_end   = bisect.bisect_right(ts_idx, ts)
            win       = _mtf_windows.get(tf, 200)
            mtf_sliced[tf] = mtf_candles[tf][max(0, idx_end - win):idx_end]

        try:
            signal = await strategy.analyze(prim_slice, price, mtf_candles=mtf_sliced)
        except Exception as e:
            logger.debug("MTF v2 analyze error at bar %d: %s", i, e)
            continue

        if signal.type not in (SignalType.BUY, SignalType.SELL):
            continue
        if open_long is not None or open_short is not None:
            continue  # 1 position per strategy at a time

        meta = signal.metadata or {}
        sl  = meta.get("sl_init", meta.get("stop_loss"))
        tp1 = meta.get("tp1")
        tp2 = meta.get("tp2", meta.get("take_profit"))
        if sl is None or tp1 is None or tp2 is None:
            continue
        side = "long" if signal.type == SignalType.BUY else "short"
        pos = BTradeV2(
            symbol=strategy.symbol, strategy=strategy.name, side=side,
            entry=price, sl=float(sl), tp1=float(tp1), tp2=float(tp2),
            entry_ts=ts, notional=notional,
            partial_pct=float(meta.get("partial_pct", 0.5)),
        )
        # entry-side taker fee on full notional
        pos.pnl_usdt -= notional * TAKER_FEE
        if side == "long": open_long = pos
        else:              open_short = pos

    return trades, stats


async def backtest_strategy_mtf(
    strategy: BaseStrategy,
    primary_candles: list[OHLCV],
    mtf_candles: dict[str, list[OHLCV]],
    notional: float,
    sl_pct: float,
    tp_pct: float,
    warmup: int = 100,
    primary_window: int = 300,
    mtf_windows: Optional[dict] = None,
) -> list[BTrade]:
    """
    Walk-forward backtest with MTF candle support.

    For each primary bar i, passes:
      - primary_candles[i-primary_window:i]  (rolling, avoids O(n²))
      - mtf_candles[tf][:k] sliced up to primary bar's timestamp (via bisect)

    strategy.analyze() must accept mtf_candles kwarg (e.g. ScalpTrendStrategy).
    """
    trades: list[BTrade] = []
    open_long:  Optional[BTrade] = None
    open_short: Optional[BTrade] = None

    _mtf_windows = {"1h": 200, "4h": 120, "30m": 300}
    if mtf_windows:
        _mtf_windows.update(mtf_windows)

    # Pre-index MTF timestamps for O(log n) slicing
    mtf_ts_index = {tf: [c.timestamp for c in clist] for tf, clist in mtf_candles.items()}

    for i in range(warmup, len(primary_candles)):
        bar    = primary_candles[i]
        ts     = bar.timestamp
        price  = float(bar.close)
        b_high = float(bar.high)
        b_low  = float(bar.low)

        # ── Check exits ──────────────────────────────────────────────────────
        for pos, side in [(open_long, "long"), (open_short, "short")]:
            if pos is None:
                continue
            hit_reason = hit_price = None
            if side == "long":
                if b_low  <= pos.sl:   hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_high >= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            else:
                if b_high >= pos.sl:  hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_low  <= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            if hit_reason:
                mult        = 1 if side == "long" else -1
                pnl_pct     = mult * (hit_price - pos.entry) / pos.entry
                fee         = notional * TAKER_FEE * 2
                pos.exit_price  = hit_price
                pos.exit_reason = hit_reason
                pos.exit_ts     = ts
                pos.pnl_usdt    = round(pnl_pct * notional - fee, 4)
                trades.append(pos)
                if side == "long":  open_long  = None
                else:               open_short = None

        # ── Build rolling slices ─────────────────────────────────────────────
        # Include bar i at [-1] (forming bar) so strategies use [-2] = last closed,
        # matching live OKX fetch_ohlcv where [-1] is the still-forming candle.
        prim_slice = primary_candles[max(0, i - primary_window): i + 1]

        mtf_sliced: dict[str, list[OHLCV]] = {}
        for tf, ts_idx in mtf_ts_index.items():
            idx_end   = bisect.bisect_right(ts_idx, ts)
            win       = _mtf_windows.get(tf, 200)
            idx_start = max(0, idx_end - win)
            mtf_sliced[tf] = mtf_candles[tf][idx_start:idx_end]

        # ── Strategy signal ──────────────────────────────────────────────────
        try:
            signal = await strategy.analyze(prim_slice, price, mtf_candles=mtf_sliced)
        except Exception as e:
            logger.debug("MTF analyze error at bar %d: %s", i, e)
            continue

        if signal.type == SignalType.BUY and open_long is None and open_short is None:
            sl, tp = _sl_tp(signal, price, "long", sl_pct, tp_pct)
            open_long = BTrade(
                symbol=strategy.symbol, strategy=strategy.name,
                side="long", entry=price, sl=sl, tp=tp, entry_ts=ts,
            )
        elif signal.type == SignalType.SELL and open_short is None and open_long is None:
            sl, tp = _sl_tp(signal, price, "short", sl_pct, tp_pct)
            open_short = BTrade(
                symbol=strategy.symbol, strategy=strategy.name,
                side="short", entry=price, sl=sl, tp=tp, entry_ts=ts,
            )

    return trades


def summarise(trades: list[BTrade]) -> dict:
    if not trades:
        return {"trades": 0}
    wins   = [t for t in trades if t.pnl_usdt > 0]
    losses = [t for t in trades if t.pnl_usdt <= 0]
    net    = sum(t.pnl_usdt for t in trades)
    wr     = len(wins) / len(trades) * 100
    gw     = sum(t.pnl_usdt for t in wins)
    gl     = abs(sum(t.pnl_usdt for t in losses)) or 1e-8
    return {
        "trades":  len(trades),
        "wins":    len(wins),
        "losses":  len(losses),
        "win_rate": round(wr, 1),
        "profit_factor": round(gw / gl, 2),
        "net_usdt": round(net, 2),
        "avg_win":  round(gw / len(wins) if wins else 0, 2),
        "avg_loss": round(-gl / len(losses) if losses else 0, 2),
    }


async def backtest_with_signals(
    strategy: BaseStrategy,
    candles: list[OHLCV],
    notional: float,
    sl_pct: float,
    tp_pct: float,
    warmup: int = 60,
) -> tuple[list[BTrade], list[dict]]:
    """Like backtest_strategy but also returns every signal event for correlation analysis."""
    trades: list[BTrade] = []
    signals: list[dict] = []
    open_long:  Optional[BTrade] = None
    open_short: Optional[BTrade] = None

    for i in range(warmup, len(candles)):
        bar    = candles[i]
        price  = float(bar.close)
        b_high = float(bar.high)
        b_low  = float(bar.low)
        ts     = int(bar.timestamp)

        for pos, side in [(open_long, "long"), (open_short, "short")]:
            if pos is None:
                continue
            hit_reason = hit_price = None
            if side == "long":
                if b_low  <= pos.sl:   hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_high >= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            else:
                if b_high >= pos.sl:  hit_reason, hit_price = "stop_loss",  pos.sl
                elif b_low <= pos.tp: hit_reason, hit_price = "take_profit", pos.tp
            if hit_reason:
                mult    = 1 if side == "long" else -1
                pnl_pct = mult * (hit_price - pos.entry) / pos.entry
                fee     = notional * TAKER_FEE * 2
                pos.exit_price  = hit_price
                pos.exit_reason = hit_reason
                pos.exit_ts     = ts
                pos.pnl_usdt    = round(pnl_pct * notional - fee, 4)
                trades.append(pos)
                if side == "long":  open_long  = None
                else:               open_short = None

        try:
            signal = await strategy.analyze(candles[:i + 1], price)
        except Exception as e:
            logger.debug("analyze error at bar %d: %s", i, e)
            continue

        if signal.type == SignalType.BUY and open_long is None and open_short is None:
            sl, tp = _sl_tp(signal, price, "long", sl_pct, tp_pct)
            open_long = BTrade(
                symbol=strategy.symbol, strategy=strategy.name,
                side="long", entry=price, sl=sl, tp=tp, entry_ts=ts,
            )
            signals.append({"ts": ts, "side": "long", "price": price})
        elif signal.type == SignalType.SELL and open_short is None and open_long is None:
            sl, tp = _sl_tp(signal, price, "short", sl_pct, tp_pct)
            open_short = BTrade(
                symbol=strategy.symbol, strategy=strategy.name,
                side="short", entry=price, sl=sl, tp=tp, entry_ts=ts,
            )
            signals.append({"ts": ts, "side": "short", "price": price})

    return trades, signals


def signal_overlap(
    signals_a: list[dict],
    signals_b: list[dict],
    window_ms: int = 3_600_000,
) -> dict:
    """
    Count how often two strategy signal streams agree vs disagree.
    Two signals are 'concurrent' if their timestamps are within window_ms (default 1H).
    """
    same_long = same_short = opposite = 0
    for a in signals_a:
        for b in signals_b:
            if abs(a["ts"] - b["ts"]) <= window_ms:
                if a["side"] == b["side"]:
                    if a["side"] == "long": same_long  += 1
                    else:                   same_short += 1
                else:
                    opposite += 1
    return {
        "same_long":  same_long,
        "same_short": same_short,
        "opposite":   opposite,
        "total_a":    len(signals_a),
        "total_b":    len(signals_b),
    }


async def run_full_backtest(
    connector,
    strategy_configs: list[dict],
    fixed_trade_usdt: float,
    leverage: int,
    sl_pct: float,
    tp_pct: float,
    _cache: dict | None = None,   # optional pre-fetched {(sym,tf): candles}
) -> dict[str, dict]:
    """
    Fetch candles and backtest each strategy config.
    Returns {label: summary_dict}.
    """
    notional = fixed_trade_usdt * leverage
    results: dict[str, dict] = {}
    cache: dict[tuple, list] = dict(_cache) if _cache else {}

    for cfg in strategy_configs:
        sym   = cfg["symbol"]
        tf    = cfg["tf"]
        limit = cfg.get("limit", 300)
        key   = (sym, tf)

        if key not in cache:
            logger.info("Fetching %s %s %d bars for backtest...", sym, tf, limit)
            try:
                cache[key] = await connector.fetch_ohlcv(sym, timeframe=tf, limit=limit)
                logger.info("  → %d bars fetched", len(cache[key]))
            except Exception as e:
                logger.error("Failed to fetch %s %s: %s", sym, tf, e)
                cache[key] = []

        candles = cache[key]
        if not candles:
            continue

        strat  = cfg["cls"](sym, params=cfg.get("params", {}))
        label  = f"{strat.name}/{sym.split('/')[0]}"
        logger.info("Backtesting %s on %d candles (%s)...", label, len(candles), tf)

        try:
            trades = await backtest_strategy(strat, candles, notional, sl_pct, tp_pct)
            results[label] = summarise(trades)
            s = results[label]
            logger.info("  %s: T=%d W=%d L=%d WR=%.1f%% PF=%.2f Net=%+.2f$",
                        label, s["trades"], s.get("wins", 0), s.get("losses", 0),
                        s.get("win_rate", 0), s.get("profit_factor", 0), s.get("net_usdt", 0))
        except Exception as e:
            logger.error("Backtest %s failed: %s", label, e)
            results[label] = {"trades": 0, "error": str(e)}

    return results


def format_backtest_telegram(results: dict[str, dict],
                              fixed_usdt: float, leverage: int,
                              period_bars: int) -> str:
    lines = [f"📊 *Backtest Results*",
             f"Config: ${fixed_usdt}×{leverage}x = ${fixed_usdt*leverage} notional/trade\n"]

    all_net = 0.0
    all_t   = 0
    for label, s in results.items():
        if s.get("trades", 0) == 0:
            lines.append(f"`{label}` — no signals")
            continue
        net  = s.get("net_usdt", 0)
        wr   = s.get("win_rate", 0)
        pf   = s.get("profit_factor", 0)
        t    = s["trades"]
        sign = "+" if net >= 0 else ""
        icon = "✅" if net >= 0 else "❌"
        lines.append(
            f"{icon} `{label}`\n"
            f"   T={t} WR={wr:.0f}% PF={pf:.2f} Net=`{sign}{net:.2f}$`"
        )
        all_net += net
        all_t   += t

    if all_t > 0:
        sign = "+" if all_net >= 0 else ""
        lines.append(f"\n{'✅' if all_net >= 0 else '❌'} *Total: {sign}{all_net:.2f}$ ({all_t} trades)*")

    return "\n".join(lines)


def format_comparison_telegram(
    cases: dict[str, dict[str, dict]],   # {case_label: {strategy_label: summary}}
    correlation: dict[str, dict],         # {symbol: signal_overlap dict}
    fixed_usdt: float,
    leverage: int,
) -> str:
    def _case_block(results: dict, title: str) -> str:
        lines = [f"*{title}*"]
        total_net, total_t = 0.0, 0
        for label, s in results.items():
            if s.get("trades", 0) == 0:
                lines.append(f"  `{label}` — no signals")
                continue
            net  = s.get("net_usdt", 0)
            wr   = s.get("win_rate", 0)
            pf   = s.get("profit_factor", 0)
            t    = s["trades"]
            aw   = s.get("avg_win", 0)
            al   = s.get("avg_loss", 0)
            icon = "✅" if net >= 0 else "❌"
            sign = "+" if net >= 0 else ""
            lines.append(
                f"  {icon} `{label}` T={t} WR={wr:.0f}% PF={pf:.2f} "
                f"Net=`{sign}{net:.2f}$` AvgW={aw:+.1f} AvgL={al:+.1f}"
            )
            total_net += net
            total_t   += t
        icon = "✅" if total_net >= 0 else "❌"
        sign = "+" if total_net >= 0 else ""
        lines.append(f"  {icon} *Total: {sign}{total_net:.2f}$ ({total_t} trades)*")
        return "\n".join(lines)

    parts = [
        "📊 *Backtest Comparison*",
        f"${fixed_usdt}×{leverage}x = ${fixed_usdt*leverage:.0f} notional/trade\n",
    ]
    for case_label, results in cases.items():
        parts.append(_case_block(results, case_label))
        parts.append("")

    parts.append("🔄 *Signal Overlap  v2 vs v3.1*")
    for sym, ov in correlation.items():
        sym_s = sym.split("/")[0]
        ta, tb = ov["total_a"], ov["total_b"]
        sl, ss, op = ov["same_long"], ov["same_short"], ov["opposite"]
        total_concurrent = sl + ss + op
        parts.append(f"  *{sym_s}*  v2={ta} trades  v3.1={tb} trades")
        if total_concurrent:
            parts.append(
                f"    ↑↑ Long={sl}  ↓↓ Short={ss}  ↑↓ Opposite={op}"
            )
        else:
            parts.append("    (no concurrent signals)")

    return "\n".join(parts)
