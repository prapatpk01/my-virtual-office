"""Backtest Engine — the SAME engines and Bot pipeline as live, driven by a
SimulatedExchange that replays historical candles bar by bar (no lookahead:
each tick exposes only candles whose close time <= the simulated clock, and
fills happen at the next 15m open with slippage + fees).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import numpy as np

from .config import Config, TF_MS
from .interfaces import (ExchangeInterface, ExchangeStateSnapshot, MarketRules,
                         OrderResult, PositionInfo)
from .models import Candle

logger = logging.getLogger("dual_entry.backtest")


class SimulatedExchange(ExchangeInterface):
    """Deterministic sim: market orders fill at current price +/- slippage;
    native SL/TP are honored intra-bar on subsequent candles (SL first on
    ambiguous bars — conservative)."""

    def __init__(self, cfg: Config, data: dict, initial_balance: float = 10_000.0,
                 slippage_atr: float = 0.03):
        self.cfg = cfg
        self.data = data                      # symbol -> {"15m": [Candle...], "1h": [...], "4h": [...]}
        self.balance = initial_balance
        self.slippage_atr = slippage_atr
        self.clock_ms = 0                     # simulated clock (close time of current 15m bar)
        self.positions: dict = {}             # symbol -> dict
        self.orders: dict = {}
        self.fills: list = []
        self.equity_curve: list = []

    # ── clock ────────────────────────────────────────────────────────────────

    def set_clock(self, now_ms: int) -> None:
        self.clock_ms = now_ms

    # override wall clock with the simulated clock (staleness/expiry replay)
    def now_ms(self) -> int:            # type: ignore[override]
        return self.clock_ms

    def _visible(self, symbol: str, timeframe: str) -> list:
        step = TF_MS[timeframe]
        return [c for c in self.data[symbol][timeframe] if c.timestamp + step <= self.clock_ms]

    def _price(self, symbol: str) -> float:
        vis = self._visible(symbol, "15m")
        return vis[-1].close if vis else 0.0

    def _atr(self, symbol: str) -> float:
        vis = self._visible(symbol, "15m")[-15:]
        if not vis:
            return 0.0
        return float(np.mean([c.high - c.low for c in vis]))

    # ── bar advance: process SL/TP on the newly closed bar ──────────────────

    def process_bar(self, symbol: str, bar: Candle) -> Optional[dict]:
        p = self.positions.get(symbol)
        if p is None:
            return None
        long = p["direction"] == "LONG"
        sl, tp = p.get("sl"), p.get("tp")
        hit_sl = sl is not None and (bar.low <= sl if long else bar.high >= sl)
        hit_tp = tp is not None and (bar.high >= tp if long else bar.low <= tp)
        if hit_sl:                             # conservative: SL first
            return self._settle(symbol, sl, "STOP_LOSS")
        if hit_tp:
            return self._settle(symbol, tp, "TAKE_PROFIT")
        return None

    def _settle(self, symbol: str, px: float, reason: str,
                qty: Optional[float] = None) -> dict:
        """Settle `qty` of the position (default: all). A PARTIAL settle (TP1)
        reduces the open qty and keeps the rest running; a full settle removes
        it. Matches OKX's reduce-only close semantics so the 2-TP runner
        behaves identically live and in backtest."""
        p = self.positions.get(symbol)
        long = p["direction"] == "LONG"
        close_qty = p["qty"] if qty is None else min(qty, p["qty"])
        pnl = (px - p["entry"]) * close_qty if long else (p["entry"] - px) * close_qty
        fee = close_qty * px * self.cfg.fee_rate
        self.balance += pnl - fee
        # entry fee is a whole-position cost booked at open; attribute this
        # leg's SHARE so per-trade grouping doesn't double-count it.
        entry_fee_leg = p["entry_fee"] * (close_qty / max(p["qty0"], 1e-12))
        ev = {"symbol": symbol, "reason": reason, "exit": px, "pnl": pnl - fee,
              "entry": p["entry"], "direction": p["direction"], "qty": close_qty,
              "opened_ms": p["opened_ms"], "closed_ms": self.clock_ms,
              "entry_fee": entry_fee_leg, "exit_fee": fee}
        self.fills.append(ev)
        p["qty"] -= close_qty
        if p["qty"] <= 1e-12:
            self.positions.pop(symbol, None)
        return ev

    # ── ExchangeInterface ────────────────────────────────────────────────────

    async def get_closed_candles(self, symbol: str, timeframe: str, limit: int) -> list:
        return self._visible(symbol, timeframe)[-limit:]

    async def get_market_rules(self, symbol: str) -> MarketRules:
        return MarketRules(symbol, contract_size=1e-6, lot_step=1.0, min_qty=1.0,
                           tick_size=0.0, min_notional=0.0)

    async def get_state(self, symbol: str) -> ExchangeStateSnapshot:
        snap = ExchangeStateSnapshot(equity=self.balance, free_margin=self.balance,
                                     last_price=self._price(symbol), spread_pct=0.0002)
        p = self.positions.get(symbol)
        if p:
            snap.positions = [PositionInfo(symbol, p["direction"], p["qty"], p["entry"],
                                           attached_sl=p.get("sl"), attached_tp=p.get("tp"))]
        return snap

    async def get_all_open_positions(self) -> list:
        return [PositionInfo(s, p["direction"], p["qty"], p["entry"])
                for s, p in self.positions.items()]

    async def place_market_order(self, symbol, side, contracts, direction,
                                 client_order_id, sl_price=None, tp_price=None) -> OrderResult:
        if client_order_id in self.orders:
            return self.orders[client_order_id]
        px = self._price(symbol)
        slip = self._atr(symbol) * self.slippage_atr
        fill = px + slip if side == "buy" else px - slip
        rules = await self.get_market_rules(symbol)
        qty = contracts * rules.contract_size
        fee = qty * fill * self.cfg.fee_rate
        self.balance -= fee
        self.positions[symbol] = {"direction": direction, "qty": qty, "qty0": qty,
                                  "entry": fill, "sl": sl_price, "tp": tp_price,
                                  "opened_ms": self.clock_ms, "entry_fee": fee}
        res = OrderResult(uuid.uuid4().hex, client_order_id, symbol, side, "filled",
                          filled_qty=qty, avg_price=fill, fee_cost=fee)
        self.orders[client_order_id] = res
        return res

    async def amend_protection(self, symbol, direction, quantity, sl_price, tp_price) -> bool:
        p = self.positions.get(symbol)
        if p is None:
            return True
        if sl_price is not None:
            p["sl"] = sl_price
        if tp_price is not None:
            p["tp"] = tp_price
        return True

    async def close_position(self, symbol, direction, quantity=None) -> OrderResult:
        p = self.positions.get(symbol)
        if p is None:
            return OrderResult("", "", symbol, "sell", "rejected")
        px = self._price(symbol)
        slip = self._atr(symbol) * self.slippage_atr
        px = px - slip if direction == "LONG" else px + slip
        # honor the requested quantity (TP1 partial closes only its share)
        ev = self._settle(symbol, px, "MANUAL", qty=quantity)
        return OrderResult(uuid.uuid4().hex, "", symbol,
                           "sell" if direction == "LONG" else "buy", "filled",
                           filled_qty=ev["qty"], avg_price=px, fee_cost=ev["exit_fee"],
                           realized_pnl=ev["pnl"] + ev["exit_fee"])

    async def cancel_order(self, symbol, order_id) -> bool:
        return True

    async def find_order_by_client_id(self, symbol, client_order_id):
        return self.orders.get(client_order_id)

    async def close(self) -> None:
        pass


async def run_backtest(cfg: Config, data: dict,
                       initial_balance: float = 10_000.0) -> dict:
    """Replays 15m bars through the full live Bot pipeline. `data` is
    {symbol: {"15m": [...], "1h": [...], "4h": [...]}} of Candle lists."""
    from .main import Bot
    import tempfile

    sim = SimulatedExchange(cfg, data, initial_balance)
    cfg.state_dir = tempfile.mkdtemp(prefix="dev14_bt_")
    bot = Bot(cfg)
    bot.exchange = sim
    bot.market_data.x = sim
    bot.execution.x = sim
    bot.positions.x = sim
    bot.notifier.token = ""                 # mute telegram in backtest

    symbols = [s for s in cfg.symbols if s in data]
    all_ts = sorted({c.timestamp for s in symbols for c in data[s]["15m"]})
    warmup = max(cfg.min_15m_candles, 60)
    step = TF_MS["15m"]
    closes: list = []

    for k, ts in enumerate(all_ts):
        if k < warmup:
            continue
        sim.set_clock(ts + step)
        for s in symbols:
            bar = next((c for c in data[s]["15m"] if c.timestamp == ts), None)
            if bar is None:
                continue
            ev = sim.process_bar(s, bar)     # SL/TP on the newly closed bar
            if ev is not None:
                st = bot.state_store.get(s)
                from .execution_engine import ExecutionEngine
                ExecutionEngine._clear_position_state(st)
                bar_ms = TF_MS["15m"]
                st.cooldown_until_bar = ts + (cfg.sl_cooldown_bars if ev["reason"] == "STOP_LOSS"
                                              else cfg.tp_cooldown_bars) * bar_ms
                bot.state_store.save_atomic(s, st)
                closes.append(ev)
            await bot.process_symbol(s)
        sim.equity_curve.append((ts, sim.balance))
        bot.market_data.new_tick()

    trades = sim.fills
    wins = [t for t in trades if t["pnl"] > 0]
    gross_p = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_l = sum(t["pnl"] for t in trades if t["pnl"] < 0)
    eq = [b for _, b in sim.equity_curve] or [initial_balance]
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-9))) if len(eq) else 0.0
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": (gross_p / abs(gross_l)) if gross_l else float("inf"),
        "net_pnl": sim.balance - initial_balance,
        "final_balance": sim.balance,
        "max_drawdown_pct": dd * 100,
        "fills": trades,
        "equity_curve": sim.equity_curve,
    }
