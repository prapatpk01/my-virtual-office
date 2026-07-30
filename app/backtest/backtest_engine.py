"""
Backtest Engine for Adaptive Trading Bot V7.0
==============================================
Replays historical OHLCV (15m/1h/4h) through the exact same
IndicatorEngine + TradingBot logic used in live trading.

Usage:
    python backtest_engine.py

Or import BacktestEngine and call .run() for programmatic use.
Config via BACKTEST_* env vars (see BacktestConfig below).
"""

import sys
import os
import glob
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Allow running from repo root or app/backtest/
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP  = os.path.dirname(_HERE)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from trading.indicator_engine import IndicatorEngine
from trading.adaptive_trading_bot import TradingBot, ExpectancyEngine
from trading.connectors.base import OHLCV

logger = logging.getLogger("backtest")

# Approximate, manually-maintained classification (this codebase has no
# richer per-symbol asset-class metadata) — matches run_bot.py's
# _commodity_symbols, used for the min-SL-floor split below.
_COMMODITY_SYMBOLS = {"XAU", "XAG", "CL"}


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    # Account
    initial_balance: float = 10_000.0
    risk_pct:        float = 0.01          # 1% per trade
    tp1_close_pct:   float = 0.75          # [V9.2] 75% close at TP1

    # TP/SL structure (must match live bot)
    tp1_r:           float = 0.6           # TP1 at 0.6R (widened from 0.5R)
    tp2_r:           float = 1.3           # TP2 at 1.2R (widened from 1.0R)
    breakeven_lock_r: float = 0.30         # T1's SL-move locks in this much R (was flat 0R)

    # Risk limits
    daily_loss_pct:   float = -3.0
    daily_profit_pct: float = 8.0
    cooldown_min:     int   = 20
    max_loss_streak:  int   = 4

    # Indicator warmup bars (before first signal allowed)
    warmup_15m:  int = 60    # ~15 hours
    warmup_1h:   int = 60
    warmup_4h:   int = 60

    # [MTF-CONFLUENCE] "adaptive" (default) or "mtf_confluence" — see
    # trading/mtf_confluence_engine.py and TradingBot's entry_engine param.
    entry_engine: str = "adaptive"

    # [EARLY TREND] fast dual-TF (4H+1H) HMA/MACD/ROC lean folded into L1's
    # score when confirmed — see compute_early_trend/MacroTrendEngine.
    # UNVALIDATED pending backtest; off by default.
    enable_early_trend: bool = False

    # [FAST MACRO EMA] override L1's EMA20/50 cross component with a faster
    # pair computed fresh from raw 4H candles. 12/26 is now the DEFAULT —
    # backtested at +$209 combined vs EMA20/50, improving both in-sample and
    # out-of-sample splits independently. None/None restores old EMA20/50.
    macro_ema_fast: Optional[int] = 12
    macro_ema_slow: Optional[int] = 26

    # Position sizing safety
    min_sl_pct:     float = 0.020    # minimum SL distance = 2.0% — limits notional to ≤$5000 at 1% risk
    max_position_usd: float = 10_000  # max notional per trade (safety cap)

    # [MIN-SL FLOOR] TradingBot.min_sl_pct's floor on the ATR-based SL. Tried
    # splitting commodity to 0.8% (gold/silver/oil rarely move 2% intra-
    # trade) but that backtested WORSE than the uniform 2% (-$470 combined,
    # every XAU loss a full -1R stop) — 0.8% was tighter than gold's actual
    # ATR often calls for. Both buckets now share 1.2%, the value that DID
    # backtest better on crypto (+$343). Picked per-symbol in
    # SymbolBacktest.run() via _COMMODITY_SYMBOLS, matching run_bot.py.
    min_sl_pct_crypto:    float = 0.012
    min_sl_pct_commodity: float = 0.012

    # [PULLBACK FILTER] max 15m ADX for a Trend entry (None = TradingBot's
    # class default of 22). The single biggest entry-frequency knob.
    max_15m_adx_trend: Optional[float] = None

    # Fees & slippage
    commission_pct: float = 0.0005   # 0.05% taker (OKX SWAP)
    slippage_pct:   float = 0.0002   # 0.02% slippage

    # Paths
    data_root:   str = ""    # set by runner
    output_dir:  str = "backtest_results"


# ──────────────────────────────────────────────────────────────────────────────
# Data loader
# ──────────────────────────────────────────────────────────────────────────────

_BINANCE_COLS = ["open_time","open","high","low","close","volume",
                 "close_time","quote_volume","count","taker_buy_volume",
                 "taker_buy_quote_volume","ignore"]

def _load_csv_dir(path: str) -> pd.DataFrame:
    """Load and merge all CSVs in a directory, return sorted DataFrame.
    Handles both Binance header-less format and Binance with-header format.
    """
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs in {path}")
    frames = []
    for f in files:
        try:
            # Peek first row to detect header
            with open(f) as fh:
                first = fh.readline().strip()
            has_header = first.startswith("open_time") or first.startswith("timestamp")
            if has_header:
                df = pd.read_csv(f, header=0)
                if "open_time" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "open_time"})
            else:
                # Binance headerless — use known column count
                df = pd.read_csv(f, header=None,
                                 names=_BINANCE_COLS[:len(first.split(","))])
            # Convert timestamp — auto-detect ms vs µs
            if pd.api.types.is_numeric_dtype(df["open_time"]):
                unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
                df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
            else:
                df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            frames.append(df[["open_time","open","high","low","close","volume"]])
        except Exception as e:
            logger.warning("  skip %s: %s", f, e)
    if not frames:
        raise ValueError(f"Could not parse any CSV in {path}")
    out = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
    out[["open","high","low","close","volume"]] = \
        out[["open","high","low","close","volume"]].astype(float)
    return out.reset_index(drop=True)


def _df_to_ohlcv(df: pd.DataFrame) -> List[OHLCV]:
    """Convert DataFrame rows to list of OHLCV namedtuple-like objects."""
    result = []
    for row in df.itertuples(index=False):
        ts = int(row.open_time.timestamp() * 1000)
        result.append(OHLCV(
            timestamp=ts,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        ))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Execution callback (captures orders for metrics)
# ──────────────────────────────────────────────────────────────────────────────

class PaperExecutor:
    """Paper-trading order executor for backtesting."""

    def __init__(self, commission_pct: float = 0.0005, slippage_pct: float = 0.0002):
        self.commission_pct  = commission_pct
        self.slippage_pct    = slippage_pct
        self.orders: List[Dict] = []
        self._trade_commission: float = 0.0  # running commission for current open trade

    def reset_trade_commission(self):
        """Call when a new trade opens — resets per-trade commission accumulator."""
        self._trade_commission = 0.0

    def pop_trade_commission(self) -> float:
        """Return and reset accumulated commission for the just-closed trade."""
        c = self._trade_commission
        self._trade_commission = 0.0
        return c

    def execute(self, order_type: str, trade_info: Dict) -> Dict:
        price = float(trade_info.get("price") or trade_info.get("entry") or 0)
        size  = float(trade_info.get("size", 0))
        # Apply slippage
        if "OPEN" in order_type or "BUY" in order_type:
            fill_price = price * (1 + self.slippage_pct)
            self.reset_trade_commission()   # new trade starts
        else:
            fill_price = price * (1 - self.slippage_pct)
        commission = fill_price * size * self.commission_pct
        self._trade_commission += commission

        order = {
            "type":       order_type,
            "price":      price,
            "fill_price": fill_price,
            "size":       size,
            "commission": commission,
            "pnl":        float(trade_info.get("pnl", 0)),
            "direction":  trade_info.get("direction", ""),
            "reason":     trade_info.get("reason", ""),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
        self.orders.append(order)
        return {"fill_price": fill_price, "order_id": f"bt-{len(self.orders)}"}


# ──────────────────────────────────────────────────────────────────────────────
# Backtest runner for one symbol
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol:       str
    direction:    str
    entry_price:  float
    exit_price:   float
    size:         float
    pnl:          float
    pnl_pct:      float
    result:       str          # WIN / LOSS
    market_state: str
    regime_score: float
    sl:           float
    tp1:          float
    tp2:          float
    entry_time:   str
    exit_time:    str
    bars_held:    int
    commission:   float
    strategy:     str = ""     # SwingReversal / MeanReversion
    entry_type:   str = ""     # trend_follow / breakout / reversal / mean_revert etc.
    e_entry:      float = 0.0  # entry-time 15M entry score
    e_context:    float = 0.0  # entry-time 1H context score
    e_fit:        float = 0.0  # entry-time 4H direction fit
    e_total:      float = 0.0  # entry-time composite score
    e_adx:        float = 0.0
    e_atr_exp:    float = 0.0
    e_vol_ratio:  float = 0.0
    e_ema_dist_atr: float = 0.0
    e_rsi:        float = 0.0
    realized_r:   float = 0.0  # realized R multiple
    mae:          float = 0.0  # max adverse excursion (R)
    mfe:          float = 0.0  # max favorable excursion (R)
    hour_utc:     int   = -1   # entry-tick hour
    e_state:      str   = ""   # market state AT ENTRY (market_state col is exit-time)


class SymbolBacktest:
    def __init__(self, symbol: str, cfg: BacktestConfig, data_root: str,
                 expectancy_engine=None):
        self.symbol    = symbol
        self.cfg       = cfg
        self.data_root = data_root
        # Shared across all symbols in one multi-symbol run — see TradingBot
        # docstring on expectancy_engine for why this must be pooled.
        self.expectancy_engine = expectancy_engine

    def _load_data(self):
        sym_dir = os.path.join(self.data_root, self.symbol)
        c15m = _df_to_ohlcv(_load_csv_dir(os.path.join(sym_dir, "csv_15m")))
        c1h  = _df_to_ohlcv(_load_csv_dir(os.path.join(sym_dir, "csv_1h")))
        c4h  = _df_to_ohlcv(_load_csv_dir(os.path.join(sym_dir, "csv_4h")))
        return c15m, c1h, c4h

    def run(self) -> List[TradeRecord]:
        logger.info("[%s] Loading data...", self.symbol)
        c15m, c1h, c4h = self._load_data()
        logger.info("[%s] bars: 15m=%d  1h=%d  4h=%d", self.symbol, len(c15m), len(c1h), len(c4h))

        executor   = PaperExecutor(self.cfg.commission_pct, self.cfg.slippage_pct)
        ind_engine = IndicatorEngine()
        _base_sym  = self.symbol.split("/")[0].upper()
        _min_sl_pct = (self.cfg.min_sl_pct_commodity if _base_sym in _COMMODITY_SYMBOLS
                      else self.cfg.min_sl_pct_crypto)

        bot = TradingBot(
            account_balance        = self.cfg.initial_balance,
            base_risk_pct          = self.cfg.risk_pct,
            daily_loss_limit_pct   = self.cfg.daily_loss_pct,
            daily_profit_limit_pct = self.cfg.daily_profit_pct,
            cooldown_minutes       = self.cfg.cooldown_min,
            max_loss_streak        = self.cfg.max_loss_streak,
            tp1_close_pct          = self.cfg.tp1_close_pct,
            tp1_r                  = self.cfg.tp1_r,   # was a dead config field — now actually applied
            tp2_r                  = self.cfg.tp2_r,
            breakeven_lock_r       = self.cfg.breakeven_lock_r,
            min_sl_pct             = _min_sl_pct,
            max_15m_adx_trend      = self.cfg.max_15m_adx_trend,
            state_file             = os.devnull,   # no disk I/O during backtest
            execution_callback     = executor.execute,
            startup_warmup_minutes = 0,            # backtest has no warmup — instant start
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

        # Build aligned 1h/4h index for fast lookback
        ts15m = [c.timestamp for c in c15m]
        ts1h  = [c.timestamp for c in c1h]
        ts4h  = [c.timestamp for c in c4h]

        def _latest_before(ts_list, target_ts, warmup):
            """Return slice of ts_list[:idx+1] where ts_list[idx] <= target_ts."""
            idx = 0
            for i, t in enumerate(ts_list):
                if t <= target_ts:
                    idx = i
                else:
                    break
            if idx < warmup:
                return None
            return idx + 1

        logger.info("[%s] Running backtest (%d 15m bars)...", self.symbol, len(c15m))

        for i in range(self.cfg.warmup_15m, len(c15m)):
            bar15 = c15m[i]
            current_ts = bar15.timestamp

            # Find aligned 1h / 4h slices
            n1h = _latest_before(ts1h, current_ts, self.cfg.warmup_1h)
            n4h = _latest_before(ts4h, current_ts, self.cfg.warmup_4h)
            if n1h is None or n4h is None:
                continue

            slice15m = c15m[max(0, i - 299): i + 1]   # last 300 bars
            slice1h  = c1h[:n1h][-200:]
            slice4h  = c4h[:n4h][-200:]

            try:
                candle_15m, candle_1h, candle_4h, ind_15m, ind_1h, ind_4h = \
                    ind_engine.compute(slice15m, slice1h, slice4h)
            except Exception as e:
                logger.debug("[%s] ind compute error bar %d: %s", self.symbol, i, e)
                continue

            # Enforce minimum SL distance to prevent oversized positions on tight swings
            price = float(bar15.close)
            min_sl = price * self.cfg.min_sl_pct
            if candle_15m.get("pattern_low", price) > price - min_sl:
                candle_15m["pattern_low"] = price - min_sl
            if candle_15m.get("pattern_high", price) < price + min_sl:
                candle_15m["pattern_high"] = price + min_sl

            prev_state = bot.state
            prev_pos   = bot.position_open
            bars_held_before_tick = bars_held  # snapshot before on_tick may close the trade

            extras = {
                "symbol":       self.symbol,
                "session":      "",
                "funding_rate": 0.0,
                "oi":           0,
            }

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

            # Track bars held
            if bot.position_open:
                bars_held += 1
            else:
                bars_held = 0

            # Capture closed trades from journal
            if len(bot.trade_journal) > prev_journal_len:
                for j_entry in bot.trade_journal[prev_journal_len:]:
                    pnl      = float(j_entry.get("pnl", 0))
                    entry_p  = float(j_entry.get("entry") or 0)
                    exit_p   = float(j_entry.get("exit") or 0)
                    size_est = abs(pnl / (exit_p - entry_p + 1e-12)) if exit_p and entry_p else 0

                    # Use per-trade commission accumulator (accurate — no bleeding from prior trade)
                    comm = executor.pop_trade_commission()

                    # bars_held_before_tick reflects how many bars were held entering this tick;
                    # +1 accounts for the closing tick itself (position was open going into on_tick).
                    # Using the post-tick bars_held would always be 0 since it resets on close.
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
                        exit_time    = datetime.fromtimestamp(
                            current_ts / 1000, tz=timezone.utc
                        ).isoformat(),
                        bars_held    = bars_held_before_tick + 1,
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

        logger.info("[%s] Done — %d trades found", self.symbol, len(trade_records))
        return trade_records


# ──────────────────────────────────────────────────────────────────────────────
# Metrics calculator
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: List[TradeRecord], initial_balance: float, symbol: str) -> Dict:
    if not trades:
        return {"symbol": symbol, "total_trades": 0, "error": "no trades"}

    pnls  = [t.pnl for t in trades]
    wins  = [t for t in trades if t.result == "WIN"]
    losses= [t for t in trades if t.result == "LOSS"]

    # Equity curve
    equity = [initial_balance]
    for p in pnls:
        equity.append(equity[-1] + p)

    # Max drawdown
    peak = initial_balance
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe (daily PnL)
    if len(pnls) > 1:
        pnl_arr = np.array(pnls)
        sharpe = float(pnl_arr.mean() / (pnl_arr.std() + 1e-12) * np.sqrt(252))
    else:
        sharpe = 0.0

    profit_factor = (
        sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses) + 1e-12)
        if losses else float("inf")
    )

    by_state: Dict[str, Dict] = {}
    for t in trades:
        s = t.market_state or "Unknown"
        by_state.setdefault(s, {"wins": 0, "losses": 0, "pnl": 0.0})
        by_state[s]["pnl"] += t.pnl
        if t.result == "WIN":
            by_state[s]["wins"] += 1
        else:
            by_state[s]["losses"] += 1
    for s, v in by_state.items():
        total = v["wins"] + v["losses"]
        v["win_rate"] = v["wins"] / total if total else 0
        v["total"] = total

    return {
        "symbol":           symbol,
        "total_trades":     len(trades),
        "win_rate":         len(wins) / len(trades),
        "profit_factor":    round(profit_factor, 3),
        "net_pnl":          round(sum(pnls), 2),
        "net_pnl_pct":      round(sum(pnls) / initial_balance * 100, 2),
        "avg_win":          round(np.mean([t.pnl for t in wins]),   2) if wins   else 0,
        "avg_loss":         round(np.mean([t.pnl for t in losses]), 2) if losses else 0,
        "avg_rr":           round(abs(np.mean([t.pnl for t in wins]) /
                                  (np.mean([t.pnl for t in losses]) + 1e-12)), 3) if losses else 0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe":           round(sharpe, 3),
        "final_balance":    round(equity[-1], 2),
        "avg_bars_held":    round(np.mean([t.bars_held for t in trades]), 1),
        "total_commission": round(sum(t.commission for t in trades), 2),
        "by_market_state":  by_state,
        "longs":            len([t for t in trades if t.direction == "LONG"]),
        "shorts":           len([t for t in trades if t.direction == "SHORT"]),
        "equity_curve":     [round(e, 2) for e in equity],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Master runner
# ──────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Run backtest across multiple symbols and save results.

    Quick start:
        engine = BacktestEngine(data_root="path/to/backtest_data",
                                output_dir="backtest_results")
        engine.run(["BTC", "ETH", "XAU", "XAG", "SOL"])
    """

    SYMBOL_MAP = {
        "BTC": "BTC/USDT:USDT",
        "ETH": "ETH/USDT:USDT",
        "XAU": "XAU/USDT:USDT",
        "XAG": "XAG/USDT:USDT",
        "SOL": "SOL/USDT:USDT",
    }

    def __init__(self,
                 data_root:   str = "",
                 output_dir:  str = "backtest_results",
                 cfg: Optional[BacktestConfig] = None):
        self.cfg = cfg or BacktestConfig()
        self.cfg.data_root  = data_root or self.cfg.data_root
        self.cfg.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, symbols: Optional[List[str]] = None) -> Dict:
        symbols = symbols or list(self.SYMBOL_MAP.keys())
        all_metrics: Dict[str, Dict] = {}
        all_trades:  Dict[str, List] = {}

        # [SHARED-LEARNING] One ExpectancyEngine pooled across every symbol in
        # this run — see TradingBot.__init__ docstring: a single symbol rarely
        # accumulates enough samples of one narrow (regime, strategy) combo to
        # reach ExpectancyEngine.MIN_TRADES on its own.
        shared_expectancy = ExpectancyEngine()

        for sym in symbols:
            logger.info("=" * 50)
            logger.info("SYMBOL: %s", sym)
            logger.info("=" * 50)
            try:
                runner = SymbolBacktest(sym, self.cfg, self.cfg.data_root,
                                        expectancy_engine=shared_expectancy)
                trades = runner.run()
                metrics = compute_metrics(trades, self.cfg.initial_balance, sym)
                all_metrics[sym] = metrics
                all_trades[sym]  = [asdict(t) for t in trades]
            except Exception as e:
                logger.error("[%s] FAILED: %s", sym, e, exc_info=True)
                all_metrics[sym] = {"symbol": sym, "error": str(e)}

        # Save results
        self._save(all_metrics, all_trades)
        self._print_summary(all_metrics)
        return all_metrics

    def _save(self, metrics: Dict, trades: Dict):
        out = self.cfg.output_dir

        # Per-symbol JSON
        for sym, m in metrics.items():
            path = os.path.join(out, f"metrics_{sym}.json")
            with open(path, "w") as f:
                # equity_curve can be large — save separately
                m_no_equity = {k: v for k, v in m.items() if k != "equity_curve"}
                json.dump(m_no_equity, f, indent=2, default=str)

            eq = m.get("equity_curve", [])
            if eq:
                eq_path = os.path.join(out, f"equity_{sym}.json")
                with open(eq_path, "w") as f:
                    json.dump(eq, f)

        # Per-symbol trades CSV
        for sym, t_list in trades.items():
            if not t_list:
                continue
            df = pd.DataFrame(t_list)
            df.to_csv(os.path.join(out, f"trades_{sym}.csv"), index=False)

        # Combined summary
        summary = []
        for sym, m in metrics.items():
            row = {k: v for k, v in m.items()
                   if k not in ("by_market_state", "equity_curve")}
            summary.append(row)
        pd.DataFrame(summary).to_csv(os.path.join(out, "summary.csv"), index=False)

        # Master JSON
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
        print(f"{'BACKTEST SUMMARY':^70}")
        print("=" * 70)
        hdr = f"{'Symbol':<8}{'Trades':>7}{'WR%':>8}{'Net PnL':>10}{'PF':>7}{'MaxDD%':>8}{'Sharpe':>8}"
        print(hdr)
        print("-" * 70)
        for sym, m in metrics.items():
            if "error" in m and "total_trades" not in m:
                print(f"{sym:<8}  ERROR: {m['error']}")
                continue
            if m.get("total_trades", 0) == 0:
                print(f"{sym:<8}  no trades")
                continue
            print(
                f"{sym:<8}"
                f"{m['total_trades']:>7}"
                f"{m['win_rate']*100:>7.1f}%"
                f"{m['net_pnl']:>10.2f}"
                f"{m['profit_factor']:>7.2f}"
                f"{m['max_drawdown_pct']:>7.1f}%"
                f"{m['sharpe']:>8.3f}"
            )
        print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Adaptive Bot Backtest Engine")
    parser.add_argument("--data-root",   default=os.environ.get("BT_DATA_ROOT", "backtest_data"),
                        help="Root dir containing per-symbol subdirs (csv_15m/csv_1h/csv_4h)")
    parser.add_argument("--output-dir",  default="backtest_results",
                        help="Directory for result JSON/CSV files")
    parser.add_argument("--symbols",     default="BTC,ETH,XAU,XAG,SOL",
                        help="Comma-separated list of symbols to backtest")
    parser.add_argument("--balance",     type=float, default=10_000.0)
    parser.add_argument("--risk",        type=float, default=0.01,
                        help="Risk per trade (fraction, e.g. 0.01 = 1%%)")
    args = parser.parse_args()

    cfg = BacktestConfig(
        initial_balance = args.balance,
        risk_pct        = args.risk,
        data_root       = args.data_root,
        output_dir      = args.output_dir,
    )

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    engine = BacktestEngine(
        data_root  = args.data_root,
        output_dir = args.output_dir,
        cfg        = cfg,
    )
    engine.run(syms)
