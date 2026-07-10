"""
Realistic Backtest Engine — intrabar-aware extension of backtest_engine.py
============================================================================
The original engine (backtest_engine.py) only evaluates SL/target-ladder
hits at 15m bar CLOSE, via bot.on_tick(). That misses exactly what the live
runner's ~30-60s poll (bot.check_price_protection) exists to catch: a wick
or spike that touches an SL or ladder level mid-bar and reverts before the
bar closes. This engine plugs that gap by replaying 3m candles between
each 15m close and calling check_price_protection() against them — the
same function the live bot calls on every poll.

Reuses every tested building block from backtest_engine.py (config,
PaperExecutor, TradeRecord, compute_metrics, CSV loaders) rather than
duplicating them, so results stay comparable and any fix to those stays
shared.

Usage:
    python backtest_engine_realistic.py --data-root backtest_data_v2 \
        --symbols BTC,ETH,SOL,XRP,HYPE,XAU,XAG,CL --charts
"""

import os
import sys
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.dirname(_HERE)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from trading.indicator_engine import IndicatorEngine

from backtest.backtest_engine import (
    BacktestConfig, PaperExecutor, TradeRecord, SymbolBacktest,
    compute_metrics, _load_csv_dir, _df_to_ohlcv,
)
from trading.adaptive_trading_bot import TradingBot, ExpectancyEngine

logger = logging.getLogger("backtest_realistic")


class RealisticSymbolBacktest(SymbolBacktest):
    """
    Same 15m-driven simulation as SymbolBacktest, plus a 3m intrabar pass
    between each 15m bar close that mirrors the live runner's polling loop
    (on_tick on bar close, check_price_protection every poll in between).
    """

    def _load_data(self):
        sym_dir = os.path.join(self.data_root, self.symbol)
        c15m = _df_to_ohlcv(_load_csv_dir(os.path.join(sym_dir, "csv_15m")))
        c1h  = _df_to_ohlcv(_load_csv_dir(os.path.join(sym_dir, "csv_1h")))
        c4h  = _df_to_ohlcv(_load_csv_dir(os.path.join(sym_dir, "csv_4h")))
        c3m  = _df_to_ohlcv(_load_csv_dir(os.path.join(sym_dir, "csv_3m")))
        return c15m, c1h, c4h, c3m

    @staticmethod
    def _intrabar_step(bot: TradingBot, candle3m) -> Optional[str]:
        """
        One live-poll-equivalent check against a single 3m candle. Checks
        the ADVERSE extreme first (low for LONG / high for SHORT) so a
        candle whose range technically covers both the SL and a target
        level resolves as "stopped out", the standard conservative
        backtest convention — then, only if the position survived that,
        checks the FAVORABLE extreme for target-ladder hits. This mirrors
        check_price_protection's own SL-first-then-targets order, just
        applied twice (once per extreme) instead of once (at bar close).

        Passes the candle's own (simulated, historical) timestamp as `now`
        explicitly — check_price_protection defaults to real wall-clock
        time (correct for live polling), which would stamp the whipsaw
        -spacing gate with today's real date if left unset here.
        """
        if not bot.position_open or not bot.current_trade:
            return None
        direction = bot.current_trade.get("direction", "LONG")
        lo, hi = float(candle3m.low), float(candle3m.high)
        worst = lo if direction == "LONG" else hi
        best  = hi if direction == "LONG" else lo
        sim_now = datetime.fromtimestamp(candle3m.timestamp / 1000, tz=timezone.utc)

        action = bot.check_price_protection(worst, now=sim_now)
        if not bot.position_open:
            return action
        action2 = bot.check_price_protection(best, now=sim_now)
        return action2 or action

    def run(self) -> List[TradeRecord]:
        logger.info("[%s] Loading data (15m/1h/4h + 3m intrabar)...", self.symbol)
        c15m, c1h, c4h, c3m = self._load_data()
        logger.info("[%s] bars: 15m=%d 1h=%d 4h=%d 3m=%d",
                    self.symbol, len(c15m), len(c1h), len(c4h), len(c3m))

        executor   = PaperExecutor(self.cfg.commission_pct, self.cfg.slippage_pct)
        ind_engine = IndicatorEngine()

        bot = TradingBot(
            account_balance        = self.cfg.initial_balance,
            base_risk_pct          = self.cfg.risk_pct,
            daily_loss_limit_pct   = self.cfg.daily_loss_pct,
            daily_profit_limit_pct = self.cfg.daily_profit_pct,
            cooldown_minutes       = self.cfg.cooldown_min,
            max_loss_streak        = self.cfg.max_loss_streak,
            tp1_close_pct          = self.cfg.tp1_close_pct,
            tp1_r                  = self.cfg.tp1_r,
            tp2_r                  = self.cfg.tp2_r,
            state_file             = os.devnull,
            execution_callback     = executor.execute,
            startup_warmup_minutes = 0,
            enable_swing_reversal  = True,
            enable_mean_reversion  = True,
            expectancy_engine      = self.expectancy_engine,
            entry_engine           = self.cfg.entry_engine,
            enable_early_trend     = self.cfg.enable_early_trend,
            macro_ema_fast         = self.cfg.macro_ema_fast,
            macro_ema_slow         = self.cfg.macro_ema_slow,
        )

        trade_records: List[TradeRecord] = []
        prev_journal_len = 0
        bars_held = 0

        ts15m = [c.timestamp for c in c15m]
        ts1h  = [c.timestamp for c in c1h]
        ts4h  = [c.timestamp for c in c4h]
        ts3m  = [c.timestamp for c in c3m]

        def _latest_before(ts_list, target_ts, warmup):
            idx = 0
            for i, t in enumerate(ts_list):
                if t <= target_ts:
                    idx = i
                else:
                    break
            if idx < warmup:
                return None
            return idx + 1

        import bisect

        def _capture_new_trades(current_ts: int):
            nonlocal prev_journal_len
            if len(bot.trade_journal) <= prev_journal_len:
                return
            for j_entry in bot.trade_journal[prev_journal_len:]:
                pnl     = float(j_entry.get("pnl", 0))
                entry_p = float(j_entry.get("entry") or 0)
                exit_p  = float(j_entry.get("exit") or 0)
                size_est = abs(pnl / (exit_p - entry_p + 1e-12)) if exit_p and entry_p else 0
                comm = executor.pop_trade_commission()
                rec = TradeRecord(
                    symbol       = self.symbol,
                    direction    = j_entry.get("direction", ""),
                    entry_price  = entry_p,
                    exit_price   = exit_p,
                    size         = size_est,
                    pnl          = pnl - comm,
                    pnl_pct      = (pnl - comm) / self.cfg.initial_balance * 100,
                    result       = j_entry.get("win_loss", "LOSS"),
                    market_state = j_entry.get("market_state", ""),
                    regime_score = float(j_entry.get("regime_score") or 0),
                    sl           = float(j_entry.get("sl") or 0),
                    tp1          = float(j_entry.get("tp1") or 0),
                    tp2          = float(j_entry.get("tp2") or 0),
                    entry_time   = str(j_entry.get("entry_time") or ""),
                    exit_time    = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc).isoformat(),
                    bars_held    = bars_held + 1,
                    commission   = comm,
                    strategy     = j_entry.get("strategy", ""),
                    entry_type   = j_entry.get("entry_type", ""),
                    e_entry      = float(j_entry.get("e_entry") or 0),
                    e_context    = float(j_entry.get("e_context") or 0),
                    e_fit        = float(j_entry.get("e_fit") or 0),
                    e_total      = float(j_entry.get("e_total") or 0),
                    e_adx        = float(j_entry.get("e_adx") or 0),
                    e_atr_exp    = float(j_entry.get("e_atr_exp") or 0),
                    e_vol_ratio  = float(j_entry.get("e_vol_ratio") or 0),
                    e_ema_dist_atr = float(j_entry.get("e_ema_dist_atr") or 0),
                    e_rsi        = float(j_entry.get("e_rsi") or 0),
                    realized_r   = float(j_entry.get("realized_r") or 0),
                    mae          = float(j_entry.get("mae") or 0),
                    mfe          = float(j_entry.get("mfe") or 0),
                    hour_utc     = int(j_entry.get("hour_utc") if j_entry.get("hour_utc") is not None else -1),
                    e_state      = str(j_entry.get("e_state") or ""),
                )
                trade_records.append(rec)
            prev_journal_len = len(bot.trade_journal)

        logger.info("[%s] Running realistic backtest (%d 15m bars, 3m intrabar polling)...",
                    self.symbol, len(c15m))

        for i in range(self.cfg.warmup_15m, len(c15m)):
            bar15 = c15m[i]
            current_ts = bar15.timestamp
            next_ts = c15m[i + 1].timestamp if i + 1 < len(c15m) else current_ts + 15 * 60 * 1000

            n1h = _latest_before(ts1h, current_ts, self.cfg.warmup_1h)
            n4h = _latest_before(ts4h, current_ts, self.cfg.warmup_4h)
            if n1h is None or n4h is None:
                continue

            slice15m = c15m[max(0, i - 299): i + 1]
            slice1h  = c1h[:n1h][-200:]
            slice4h  = c4h[:n4h][-200:]

            try:
                candle_15m, candle_1h, candle_4h, ind_15m, ind_1h, ind_4h = \
                    ind_engine.compute(slice15m, slice1h, slice4h)
            except Exception as e:
                logger.debug("[%s] ind compute error bar %d: %s", self.symbol, i, e)
                continue

            price = float(bar15.close)
            min_sl = price * self.cfg.min_sl_pct
            if candle_15m.get("pattern_low", price) > price - min_sl:
                candle_15m["pattern_low"] = price - min_sl
            if candle_15m.get("pattern_high", price) < price + min_sl:
                candle_15m["pattern_high"] = price + min_sl

            extras = {"symbol": self.symbol, "session": "", "funding_rate": 0.0, "oi": 0}

            try:
                bar_dt = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc)
                bot.on_tick(
                    candle_15m, candle_1h, candle_4h,
                    ind_15m, ind_1h, ind_4h,
                    extras, float(bar15.close),
                    bar_dt=bar_dt,
                    raw_candles={"15m": slice15m, "1h": slice1h, "4h": slice4h},
                )
            except Exception as e:
                logger.error("[%s] on_tick error bar %d: %s", self.symbol, i, e)
                continue

            if bot.position_open:
                bars_held += 1
            else:
                bars_held = 0

            _capture_new_trades(current_ts)

            # [INTRABAR REALISM] Replay every 3m candle strictly between this
            # 15m close and the next one, calling check_price_protection —
            # exactly what the live runner's poll loop does between bar
            # closes. Catches SL/ladder hits a bar-close-only simulation
            # would miss (a wick that touches a level and reverts before
            # the 15m bar closes).
            if bot.position_open:
                lo_idx = bisect.bisect_right(ts3m, current_ts)
                hi_idx = bisect.bisect_left(ts3m, next_ts)
                for k in range(lo_idx, hi_idx):
                    if not bot.position_open:
                        break
                    c3 = c3m[k]
                    bot._bar_now = datetime.fromtimestamp(c3.timestamp / 1000, tz=timezone.utc)
                    self._intrabar_step(bot, c3)
                    _capture_new_trades(c3.timestamp)

        logger.info("[%s] Done — %d trades found", self.symbol, len(trade_records))
        return trade_records


class RealisticBacktestEngine:
    """Same interface as BacktestEngine, using RealisticSymbolBacktest."""

    SYMBOL_MAP = {
        "BTC":  "BTC/USDT:USDT",
        "ETH":  "ETH/USDT:USDT",
        "SOL":  "SOL/USDT:USDT",
        "XRP":  "XRP/USDT:USDT",
        "HYPE": "HYPE/USDT:USDT",
        "XAU":  "XAU/USDT:USDT",
        "XAG":  "XAG/USDT:USDT",
        "CL":   "CL/USDT:USDT",
    }

    def __init__(self, data_root: str = "", output_dir: str = "backtest_results_realistic",
                 cfg: Optional[BacktestConfig] = None):
        self.cfg = cfg or BacktestConfig()
        self.cfg.data_root  = data_root or self.cfg.data_root
        self.cfg.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, symbols: Optional[List[str]] = None) -> Dict:
        symbols = symbols or list(self.SYMBOL_MAP.keys())
        all_metrics: Dict[str, Dict] = {}
        all_trades: Dict[str, List] = {}

        # [SHARED-LEARNING] see BacktestEngine.run() in backtest_engine.py —
        # one ExpectancyEngine pooled across every symbol in this run.
        # CAVEAT: symbols are simulated sequentially start-to-finish here
        # (not interleaved bar-by-bar), so a combo's later months on an
        # earlier-processed symbol can influence an earlier month on a
        # later-processed symbol — a mild look-ahead versus live trading,
        # where all symbols advance in lockstep wall-clock time. Treat
        # results as directionally indicative, not a strict causal backtest.
        shared_expectancy = ExpectancyEngine()

        for sym in symbols:
            logger.info("=" * 50)
            logger.info("SYMBOL: %s", sym)
            logger.info("=" * 50)
            try:
                runner = RealisticSymbolBacktest(sym, self.cfg, self.cfg.data_root,
                                                 expectancy_engine=shared_expectancy)
                trades = runner.run()
                metrics = compute_metrics(trades, self.cfg.initial_balance, sym)
                all_metrics[sym] = metrics
                all_trades[sym] = [asdict(t) for t in trades]
            except Exception as e:
                logger.error("[%s] FAILED: %s", sym, e, exc_info=True)
                all_metrics[sym] = {"symbol": sym, "error": str(e)}

        self._save(all_metrics, all_trades)
        self._print_summary(all_metrics)
        return {"metrics": all_metrics, "trades": all_trades}

    def _save(self, metrics: Dict, trades: Dict):
        import json
        import pandas as pd
        out = self.cfg.output_dir

        for sym, m in metrics.items():
            path = os.path.join(out, f"metrics_{sym}.json")
            with open(path, "w") as f:
                m_no_equity = {k: v for k, v in m.items() if k != "equity_curve"}
                json.dump(m_no_equity, f, indent=2, default=str)
            eq = m.get("equity_curve", [])
            if eq:
                with open(os.path.join(out, f"equity_{sym}.json"), "w") as f:
                    json.dump(eq, f)

        for sym, t_list in trades.items():
            if not t_list:
                continue
            pd.DataFrame(t_list).to_csv(os.path.join(out, f"trades_{sym}.csv"), index=False)

        summary = []
        for sym, m in metrics.items():
            row = {k: v for k, v in m.items() if k not in ("by_market_state", "equity_curve")}
            summary.append(row)
        pd.DataFrame(summary).to_csv(os.path.join(out, "summary.csv"), index=False)

        master = {
            "run_at":  datetime.now(timezone.utc).isoformat(),
            "config":  asdict(self.cfg),
            "metrics": {s: {k: v for k, v in m.items() if k != "equity_curve"}
                       for s, m in metrics.items()},
        }
        with open(os.path.join(out, "backtest_results.json"), "w") as f:
            json.dump(master, f, indent=2, default=str)
        logger.info("Results saved to %s/", out)

    def _print_summary(self, metrics: Dict):
        print("\n" + "=" * 70)
        print(f"{'REALISTIC BACKTEST SUMMARY (3m intrabar polling)':^70}")
        print("=" * 70)
        print(f"{'Symbol':<8}{'Trades':>7}{'WR%':>8}{'Net PnL':>10}{'PF':>7}{'MaxDD%':>8}{'Sharpe':>8}")
        print("-" * 70)
        for sym, m in metrics.items():
            if "error" in m and "total_trades" not in m:
                print(f"{sym:<8}  ERROR: {m['error']}")
                continue
            if m.get("total_trades", 0) == 0:
                print(f"{sym:<8}  no trades")
                continue
            print(
                f"{sym:<8}{m['total_trades']:>7}{m['win_rate']*100:>7.1f}%"
                f"{m['net_pnl']:>10.2f}{m['profit_factor']:>7.2f}"
                f"{m['max_drawdown_pct']:>7.1f}%{m['sharpe']:>8.3f}"
            )
        print("=" * 70)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Realistic (3m intrabar) Adaptive Bot Backtest")
    parser.add_argument("--data-root", default=os.environ.get("BT_DATA_ROOT", "backtest_data_v2"))
    parser.add_argument("--output-dir", default="backtest_results_realistic")
    parser.add_argument("--symbols", default="BTC,ETH,SOL,XRP,HYPE,XAU,XAG,CL")
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--entry-engine", default="adaptive",
                        choices=["adaptive", "mtf_confluence"],
                        help="adaptive (default V9.2 pipeline) or mtf_confluence "
                             "(4H+1H trend-alignment + 15m 3-signal confluence)")
    parser.add_argument("--early-trend", action="store_true",
                        help="fold a fast dual-TF (4H+1H) HMA/MACD/ROC lean into "
                             "L1's score when confirmed (see compute_early_trend)")
    parser.add_argument("--macro-ema", default="",
                        help="override L1's EMA20/50 cross component with a faster "
                             "pair, e.g. --macro-ema 12,26")
    args = parser.parse_args()

    macro_ema_fast = macro_ema_slow = None
    if args.macro_ema:
        macro_ema_fast, macro_ema_slow = (int(x) for x in args.macro_ema.split(","))

    cfg = BacktestConfig(
        initial_balance=args.balance,
        risk_pct=args.risk,
        data_root=args.data_root,
        output_dir=args.output_dir,
        entry_engine=args.entry_engine,
        enable_early_trend=args.early_trend,
        macro_ema_fast=macro_ema_fast,
        macro_ema_slow=macro_ema_slow,
    )
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    engine = RealisticBacktestEngine(data_root=args.data_root, output_dir=args.output_dir, cfg=cfg)
    engine.run(syms)
