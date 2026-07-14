"""
Backtest Framework.

Reuses the EXACT SAME `SignalEngine` (regime + bias + entry) and the EXACT
SAME pure SL/TP functions from position_manager.py that main.py uses live —
the only backtest-specific code is data pagination, fee/fill simulation,
and bookkeeping. This guarantees backtest results reflect what the live
bot actually does.

No-lookahead contract:
  - A 15m decision at bar i uses ONLY 15m bars [0..i], 1h/4h/30m/5m bars
    whose CLOSE time <= bar i's close time (searchsorted cutoff).
  - Entries fill at bar i+1's OPEN (next bar), never at bar i's close.
  - SL/TP/exit checks against a bar use that bar's own OHLC, evaluated
    only after the bar is complete (standard, not lookahead — the decision
    that led to entry was made on a prior bar).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config, load_config
from exchange_client import ExchangeClient
from pipeline import Pipeline as SignalEngine, LONG, SHORT
from risk_manager import RiskManager
from position_manager import calc_stop_loss, calc_take_profits, Position

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")

TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
SLIPPAGE = 0.0003

_TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000}


def _epoch_ms(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Epoch-milliseconds for a DatetimeIndex of ANY unit resolution.

    pandas 2.x stores DatetimeIndex at the source's resolution (datetime64[s/
    ms/us/ns]) and .view("int64") returns raw ints in the INDEX'S OWN unit —
    so `view // 1_000_000` is only milliseconds when the index happens to be
    ns. With mixed-vintage CSVs (ms-headered + us-headerless) this silently
    produced second- or kilosecond-scale numbers, making every HTF cutoff
    searchsorted to the END of the dataset. Normalizing to ns first makes
    the math correct for every resolution.
    """
    if hasattr(index, "as_unit"):
        return index.as_unit("ns").asi8 // 1_000_000
    return index.view("int64") // 1_000_000


# ── Historical data pagination ────────────────────────────────────────────────

async def fetch_history(client: ExchangeClient, symbol: str, timeframe: str,
                        start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows: list = []
    since = start_ms
    tf_ms = _TF_MS.get(timeframe, 900_000)
    while since < end_ms:
        try:
            batch = await client._exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                                        since=since, limit=300)
        except Exception as e:
            logger.warning("fetch_ohlcv %s %s since=%d failed: %s", symbol, timeframe, since, e)
            break
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            break
        since = last_ts + tf_ms
        if len(batch) < 300:
            break
        await asyncio.sleep(0.15)  # rate-limit courtesy

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    ms = _epoch_ms(df.index)
    df = df[(ms >= start_ms) & (ms < end_ms)]
    return df.astype(float)


# ── Trade bookkeeping ──────────────────────────────────────────────────────────

@dataclass
class BTTrade:
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp1: float
    tp2: float
    amount: float
    risk_amount: float
    regime_at_entry: str
    bias_at_entry: str
    entry_score: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    tp1_hit: bool = False
    pnl_usd: float = 0.0
    balance_before: float = 0.0
    balance_after: float = 0.0

    @property
    def r_multiple(self) -> float:
        return self.pnl_usd / self.risk_amount if self.risk_amount > 0 else 0.0


def _entry_fee(notional: float) -> float:
    return notional * TAKER_FEE


def _exit_fee(notional: float) -> float:
    return notional * MAKER_FEE


def _closed_htf_cutoff(close_ms_sorted: np.ndarray, as_of_close_ms: int) -> int:
    """
    Positional cutoff (searchsorted, O(log n)) — number of bars whose CLOSE time
    is <= as_of_close_ms. This is the no-lookahead guard for HTF data.
    """
    return int(np.searchsorted(close_ms_sorted, as_of_close_ms, side="right"))


def simulate_symbol(cfg: Config, symbol: str, df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                    df_4h: pd.DataFrame, df_30m: pd.DataFrame, df_5m: pd.DataFrame,
                    initial_balance: float) -> list[BTTrade]:
    engine = SignalEngine(cfg)
    risk = RiskManager(cfg)
    trades: list[BTTrade] = []
    balance = initial_balance
    pos: Optional[BTTrade] = None
    tp1_hit = False

    n = len(df_15m)
    warmup = max(cfg.min_bars, 60)

    # Precompute HTF close-time arrays ONCE (O(log n) lookup per bar via
    # searchsorted, instead of an O(n) boolean scan every bar).
    close_ms_1h = _epoch_ms(df_1h.index) + _TF_MS["1h"]
    close_ms_4h = _epoch_ms(df_4h.index) + _TF_MS["4h"]
    close_ms_30m = _epoch_ms(df_30m.index) + _TF_MS["30m"]
    close_ms_5m = _epoch_ms(df_5m.index) + _TF_MS["5m"]
    open_ms_15m = _epoch_ms(df_15m.index)

    # Bound every indicator computation to a trailing window matching what
    # live's DataEngine actually fetches — see the equivalent note in a
    # prior version of this file: bounding is free (EWM/rolling indicators
    # converge well within a few hundred bars) and guarantees backtest can't
    # silently diverge from live by seeing history live never had access to.
    w15 = cfg.fetch_limit_fast
    w1h = cfg.fetch_limit_bias
    w4h = cfg.fetch_limit_regime
    w30 = cfg.fetch_limit_entry
    w5 = cfg.fetch_limit_micro

    for i in range(warmup, n - 1):
        bar = df_15m.iloc[i]
        bar_close_ms = int(open_ms_15m[i]) + _TF_MS["15m"]
        hist_15m = df_15m.iloc[max(0, i + 1 - w15): i + 1]

        cutoff_1h = _closed_htf_cutoff(close_ms_1h, bar_close_ms)
        hist_1h = df_1h.iloc[max(0, cutoff_1h - w1h): cutoff_1h]
        cutoff_4h = _closed_htf_cutoff(close_ms_4h, bar_close_ms)
        hist_4h = df_4h.iloc[max(0, cutoff_4h - w4h): cutoff_4h]
        cutoff_30m = _closed_htf_cutoff(close_ms_30m, bar_close_ms)
        hist_30m = df_30m.iloc[max(0, cutoff_30m - w30): cutoff_30m]
        cutoff_5m = _closed_htf_cutoff(close_ms_5m, bar_close_ms)
        hist_5m = df_5m.iloc[max(0, cutoff_5m - w5): cutoff_5m]

        # ── Manage open position against THIS bar's OHLC ────────────────────
        if pos is not None:
            is_long = pos.direction == LONG
            hi, lo = float(bar["high"]), float(bar["low"])

            sl_hit = (lo <= pos.sl) if is_long else (hi >= pos.sl)
            if sl_hit:
                exit_px = pos.sl
                remaining = (1.0 - cfg.tp1_fraction) if tp1_hit else 1.0
                notional = pos.amount * remaining
                pnl = ((exit_px - pos.entry_price) if is_long else (pos.entry_price - exit_px)) \
                    / pos.entry_price * notional * pos.entry_price
                pnl -= _exit_fee(notional * exit_px)
                pos.pnl_usd += pnl
                pos.exit_price, pos.exit_time = exit_px, df_15m.index[i]
                pos.exit_reason = "SL" if not tp1_hit else "BE"
                pos.tp1_hit = tp1_hit
                balance += pnl
                pos.balance_after = balance
                # Register the TRADE'S TOTAL pnl (TP1 leg + this leg), not just this
                # leg — otherwise a TP1-then-SL/BE sequence silently drops the TP1
                # profit from daily-loss/loss-streak accounting.
                risk.register_trade_result(pos.pnl_usd, balance, bar_close_ms / 1000)
                trades.append(pos)
                engine.entry_engine.on_position_closed(symbol)
                pos, tp1_hit = None, False
                continue

            if not tp1_hit and pos.tp1 is not None:
                tp1_trigger = (hi >= pos.tp1) if is_long else (lo <= pos.tp1)
                if tp1_trigger:
                    part = pos.amount * cfg.tp1_fraction
                    pnl = ((pos.tp1 - pos.entry_price) if is_long else (pos.entry_price - pos.tp1)) \
                        / pos.entry_price * part * pos.entry_price
                    pnl -= _exit_fee(part * pos.tp1)
                    pos.pnl_usd += pnl
                    tp1_hit = True
                    pos.sl = pos.entry_price   # exact breakeven
                    balance += pnl

            tp2_trigger = (hi >= pos.tp2) if is_long else (lo <= pos.tp2)
            if tp2_trigger and tp1_hit:
                part = pos.amount * (1.0 - cfg.tp1_fraction)
                pnl = ((pos.tp2 - pos.entry_price) if is_long else (pos.entry_price - pos.tp2)) \
                    / pos.entry_price * part * pos.entry_price
                pnl -= _exit_fee(part * pos.tp2)
                pos.pnl_usd += pnl
                pos.exit_price, pos.exit_time = pos.tp2, df_15m.index[i]
                pos.exit_reason = "TP2"
                pos.tp1_hit = True
                balance += pnl
                pos.balance_after = balance
                risk.register_trade_result(pos.pnl_usd, balance, bar_close_ms / 1000)
                trades.append(pos)
                engine.entry_engine.on_position_closed(symbol)
                pos, tp1_hit = None, False
                continue

            # HMA early-exit check — every closed 15m bar (this loop already
            # IS the 15m cadence, so this runs once per bar naturally).
            exit_check = engine.entry_engine.check_exit(hist_15m, pos.direction.lower())
            if exit_check.should_exit:
                exit_px = float(bar["close"])
                remaining = (1.0 - cfg.tp1_fraction) if tp1_hit else 1.0
                notional = pos.amount * remaining
                pnl = ((exit_px - pos.entry_price) if is_long else
                      (pos.entry_price - exit_px)) / pos.entry_price * notional * pos.entry_price
                pnl -= _exit_fee(notional * exit_px)
                pos.pnl_usd += pnl
                pos.exit_price, pos.exit_time = exit_px, df_15m.index[i]
                pos.exit_reason = exit_check.reason
                pos.tp1_hit = tp1_hit
                balance += pnl
                pos.balance_after = balance
                risk.register_trade_result(pos.pnl_usd, balance, bar_close_ms / 1000)
                trades.append(pos)
                engine.entry_engine.on_position_closed(symbol)
                pos, tp1_hit = None, False
            continue

        # ── Look for a new entry (flat) ──────────────────────────────────────
        can_open, _ = risk.can_open_new(balance, bar_close_ms / 1000, 0)
        if not can_open:
            continue
        if len(hist_15m) < cfg.min_bars or len(hist_1h) < cfg.min_bars or len(hist_4h) < cfg.min_bars \
           or len(hist_30m) < cfg.min_bars or len(hist_5m) < cfg.min_bars:
            continue

        sig = engine.evaluate(hist_1h, hist_4h, hist_30m, hist_15m, hist_5m, symbol=symbol)
        if sig.direction not in (LONG, SHORT):
            continue

        entry_px = float(df_15m["open"].iloc[i + 1])
        entry_px *= (1 + SLIPPAGE) if sig.direction == LONG else (1 - SLIPPAGE)

        atr_val = float(ind.atr(hist_15m, cfg.sl_atr_period).iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        swing_high, swing_low = ind.recent_swing_levels(
            hist_15m["high"], hist_15m["low"], cfg.swing_lookback_left, cfg.swing_lookback_right)
        side = "long" if sig.direction == LONG else "short"
        sl = calc_stop_loss(side, entry_px, atr_val, cfg.sl_atr_mult, swing_high, swing_low,
                           cfg.sl_min_pct, cfg.sl_max_pct, cfg.sl_tighten_mult)
        tp1, tp2 = calc_take_profits(side, entry_px, sl, cfg.tp1_r, cfg.tp2_r)

        amount = risk.size_by_risk(balance, entry_px, sl, sig.regime.size_multiplier)
        if amount <= 0:
            continue
        risk_amount = amount * abs(entry_px - sl)
        entry_fee = _entry_fee(amount * entry_px)
        if entry_fee > balance * 0.05:
            continue

        pos = BTTrade(
            symbol=symbol, direction=sig.direction, entry_time=df_15m.index[i + 1],
            entry_price=entry_px, sl=sl, tp1=tp1, tp2=tp2, amount=amount,
            risk_amount=risk_amount, regime_at_entry=sig.regime.name,
            bias_at_entry=(sig.bias.bias if sig.bias is not None else sig.regime.style),
            entry_score=sig.entry_score, balance_before=balance,
        )
        pos.pnl_usd = -entry_fee
        balance -= entry_fee
        tp1_hit = False

    return trades


async def run_backtest(symbols: list[str], start_ms: int, end_ms: int,
                       initial_balance: float = 10_000.0) -> list[BTTrade]:
    cfg = load_config()
    client = ExchangeClient(api_key="", api_secret="", passphrase="", paper=True,
                            leverage=cfg.leverage)
    all_trades: list[BTTrade] = []
    try:
        for symbol in symbols:
            logger.info("Fetching history for %s...", symbol)
            df_15m = await fetch_history(client, symbol, cfg.tf_fast, start_ms, end_ms)
            df_1h = await fetch_history(client, symbol, cfg.tf_bias, start_ms, end_ms)
            df_4h = await fetch_history(client, symbol, cfg.tf_regime, start_ms, end_ms)
            df_30m = await fetch_history(client, symbol, cfg.tf_entry, start_ms, end_ms)
            df_5m = await fetch_history(client, symbol, cfg.tf_micro, start_ms, end_ms)
            logger.info("  %s: 5m=%d 15m=%d 30m=%d 1h=%d 4h=%d bars",
                       symbol, len(df_5m), len(df_15m), len(df_30m), len(df_1h), len(df_4h))
            if len(df_15m) < cfg.min_bars or len(df_1h) < cfg.min_bars or len(df_4h) < cfg.min_bars \
               or len(df_30m) < cfg.min_bars or len(df_5m) < cfg.min_bars:
                logger.warning("  %s: insufficient history — skipped", symbol)
                continue
            trades = simulate_symbol(cfg, symbol, df_15m, df_1h, df_4h, df_30m, df_5m, initial_balance)
            logger.info("  %s: %d trades  PnL=%+.2f", symbol, len(trades),
                       sum(t.pnl_usd for t in trades))
            all_trades.extend(trades)
    finally:
        await client.close()
    return all_trades


def _parse_date_ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").value // 1_000_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Regime Bias Strategy backtest")
    parser.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start_ms, end_ms = _parse_date_ms(args.start), _parse_date_ms(args.end)

    trades = asyncio.run(run_backtest(symbols, start_ms, end_ms, args.balance))

    from report import compute_stats, per_symbol_stats, regime_stats, bias_stats, build_html_report
    stats = compute_stats(trades, args.balance)
    logger.info("=== RESULTS ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    out_path = args.out or "/tmp/signal_regime_backtest_report.html"
    html = build_html_report(trades, stats, per_symbol_stats(trades),
                             regime_stats(trades), bias_stats(trades), args.balance)
    with open(out_path, "w") as f:
        f.write(html)
    logger.info("Report saved -> %s", out_path)
