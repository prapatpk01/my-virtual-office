"""Live position lifecycle and structure-aware SL/TP planning.

The pure stop/target helpers are shared by live and backtest code.  Live state
keeps one position per symbol.  Partial TP1 does not update loss-streak state;
risk accounting is registered once, when the full trade closes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from exchange_client import ExchangeClient
from risk_manager import RiskManager
from regime_engine import RegimeResult
from bias_engine import BiasResult

logger = logging.getLogger("position_manager")

LONG = "long"
SHORT = "short"


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    amount: float
    full_amount: float
    stop_loss: float
    tp1: Optional[float]
    tp2: float
    one_r: float
    tp1_hit: bool = False
    realized_pnl: float = 0.0
    opened_at: float = field(default_factory=time.time)
    regime_at_entry: str = ""
    bias_at_entry: str = ""
    entry_score: float = 0.0
    entry_fee: float = 0.0
    entry_bar_ts: Optional[pd.Timestamp] = None
    last_exit_check_bar_ts: Optional[pd.Timestamp] = None
    setup_type: str = ""
    trigger: str = ""
    planned_rr: float = 0.0
    structure_room_r: float = 0.0


def calc_stop_loss(
    direction: str,
    entry: float,
    atr_val: float,
    atr_mult: float,
    swing_high: float,
    swing_low: float,
    sl_min_pct: float = 0.0,
    sl_max_pct: float = 1.0,
    sl_tighten_mult: float = 1.0,
) -> float:
    """Nearest meaningful swing/ATR invalidation, clamped to safe bounds.

    `atr_mult` is the fallback ATR distance.  A confirmed swing is used only if
    it is on the correct side and closer than the fallback while still outside
    the minimum stop floor.  The stop is never tightened below ordinary noise.
    """
    atr_val = max(float(atr_val), 0.0)
    if entry <= 0 or atr_val <= 0:
        return 0.0
    min_distance = max(entry * max(sl_min_pct, 0.0), atr_val * 0.55)
    max_distance = min(entry * max(sl_max_pct, sl_min_pct), atr_val * 1.50)
    fallback = max(min_distance, min(atr_val * max(atr_mult, 0.55), max_distance))
    buffer = atr_val * 0.10

    if direction == LONG:
        candidates = [entry - fallback]
        if np.isfinite(swing_low) and swing_low < entry:
            candidates.append(float(swing_low) - buffer)
        # nearest valid invalidation below entry
        stop = max(x for x in candidates if x < entry)
        distance = max(min_distance, min(entry - stop, max_distance))
        distance *= max(0.9, min(1.0, sl_tighten_mult))
        return entry - distance

    candidates = [entry + fallback]
    if np.isfinite(swing_high) and swing_high > entry:
        candidates.append(float(swing_high) + buffer)
    stop = min(x for x in candidates if x > entry)
    distance = max(min_distance, min(stop - entry, max_distance))
    distance *= max(0.9, min(1.0, sl_tighten_mult))
    return entry + distance


def calc_take_profits(
    direction: str,
    entry: float,
    sl: float,
    tp1_r: float,
    tp2_r: float,
) -> tuple[float, float]:
    one_r = abs(entry - sl)
    if direction == LONG:
        return entry + tp1_r * one_r, entry + tp2_r * one_r
    return entry - tp1_r * one_r, entry - tp2_r * one_r


class PositionManager:
    def __init__(self, cfg: Config, client: ExchangeClient, risk: RiskManager, entry_engine):
        self.cfg = cfg
        self.client = client
        self.risk = risk
        self.entry_engine = entry_engine
        self._positions: dict[str, Position] = {}
        self._recently_closed: dict[str, float] = {}

    def get(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def open_position_count(self) -> int:
        return len(self._positions)

    def _mark_closed(self, symbol: str) -> None:
        self._recently_closed[symbol] = time.time()

    async def verify_no_stale_position(self, symbol: str) -> bool:
        for side in (LONG, SHORT):
            amount = await self.client.fetch_position_amount(symbol, side)
            if amount > 0:
                logger.warning(
                    "[POS] %s has untracked OKX %s amount %.8f — entry blocked",
                    symbol,
                    side,
                    amount,
                )
                return False
        return True

    async def reconcile_with_exchange(self, symbols: list[str]) -> list[str]:
        adopted: list[str] = []
        now = time.time()
        c = self.cfg
        for symbol in symbols:
            if self.has_position(symbol):
                continue
            closed_at = self._recently_closed.get(symbol)
            if closed_at is not None and now - closed_at < c.reconcile_settle_grace_sec:
                continue
            open_sides: list[tuple[str, dict]] = []
            for side in (LONG, SHORT):
                details = await self.client.fetch_position_details(symbol, side)
                if details and details.get("amount", 0) > 0:
                    open_sides.append((side, details))
            if not open_sides:
                continue
            if len(open_sides) > 1:
                adopted.append(
                    f"{symbol} ⚠️ HEDGE conflict — only {open_sides[0][0].upper()} adopted; close other leg manually"
                )
                logger.error("[RECONCILE] %s has both long and short open", symbol)
            side, details = open_sides[0]
            entry = float(details["entry_price"])
            amount = float(details["amount"])
            sl, tp = await self.client.fetch_attached_stops(symbol, side)
            if sl is None:
                sl = entry * (0.97 if side == LONG else 1.03)
                logger.error("[RECONCILE] %s has no attached SL; using emergency 3%% stop", symbol)
            one_r = abs(entry - float(sl))
            if tp is None:
                _, tp = calc_take_profits(side, entry, float(sl), c.tp1_r, c.tp2_r)
            # Stop at/through fee-adjusted breakeven implies TP1 already banked.
            be_tolerance = max(one_r * getattr(c, "be_lock_r", 0.08), entry * c.fee_rate * 2)
            tp1_hit = (
                float(sl) >= entry - be_tolerance if side == LONG else float(sl) <= entry + be_tolerance
            )
            tp1 = None if tp1_hit else calc_take_profits(side, entry, float(sl), c.tp1_r, c.tp2_r)[0]
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=entry,
                amount=amount,
                full_amount=amount,
                stop_loss=float(sl),
                tp1=tp1,
                tp2=float(tp),
                one_r=one_r,
                tp1_hit=tp1_hit,
                regime_at_entry="ADOPTED",
                bias_at_entry="ADOPTED",
                setup_type="ADOPTED",
            )
            adopted.append(f"{symbol} {side.upper()}")
        return adopted

    async def open_position(
        self,
        symbol: str,
        direction: str,
        price: float,
        df_15m: pd.DataFrame,
        regime: RegimeResult,
        bias: BiasResult,
        entry_score: float,
        df_5m: Optional[pd.DataFrame] = None,
        entry_result=None,
    ) -> Optional[Position]:
        c = self.cfg
        if self.has_position(symbol) or not await self.verify_no_stale_position(symbol):
            return None
        side = LONG if direction == "LONG" else SHORT
        atr_value = ind.safe_float(ind.atr(df_15m, c.sl_atr_period).iloc[-1])
        if atr_value <= 0:
            logger.warning("[POS] %s invalid ATR", symbol)
            return None
        swing_high, swing_low = ind.recent_swing_levels(
            df_15m["high"],
            df_15m["low"],
            c.swing_lookback_left,
            c.swing_lookback_right,
        )

        planned_stop = getattr(entry_result, "planned_stop", None)
        planned_target = getattr(entry_result, "planned_target", None)
        if planned_stop is not None and (
            (side == LONG and 0 < planned_stop < price)
            or (side == SHORT and planned_stop > price)
        ):
            sl = float(planned_stop)
        else:
            sl = calc_stop_loss(
                side,
                price,
                atr_value,
                c.sl_atr_mult,
                swing_high,
                swing_low,
                c.sl_min_pct,
                c.sl_max_pct,
                c.sl_tighten_mult,
            )
        if sl <= 0 or (side == LONG and sl >= price) or (side == SHORT and sl <= price):
            logger.warning("[POS] %s invalid stop %.8f for %s @ %.8f", symbol, sl, side, price)
            return None

        tp1, base_tp2 = calc_take_profits(side, price, sl, c.tp1_r, c.tp2_r)
        if planned_target is not None:
            planned_target = float(planned_target)
            if side == LONG and planned_target > price:
                tp2 = min(base_tp2, planned_target)
            elif side == SHORT and planned_target < price:
                tp2 = max(base_tp2, planned_target)
            else:
                tp2 = base_tp2
        else:
            tp2 = base_tp2
        actual_rr = abs(tp2 - price) / max(abs(price - sl), ind.EPSILON)
        if actual_rr < getattr(c, "minimum_actual_rr", 1.20):
            logger.info("[POS] %s rejected: actual RR %.2f below minimum", symbol, actual_rr)
            return None

        balance = await self.client.fetch_balance_usdt()
        if balance <= 0:
            logger.warning("[POS] %s invalid balance %.2f", symbol, balance)
            return None
        amount = self.risk.size_by_risk(
            balance,
            price,
            sl,
            regime.size_multiplier,
            fee_rate=c.fee_rate,
            expected_slippage_pct=getattr(c, "expected_slippage_pct", 0.0005),
        )
        if amount <= 0:
            return None
        estimated_risk = self.risk.estimated_risk_cash(amount, price, sl)
        logger.info(
            "[POS] %s %s risk_budget=%.2f (%.1f%% balance) estimated=%.2f qty=%.8f SL=%.8f TP1=%.8f TP2=%.8f RR=%.2f",
            symbol,
            side.upper(),
            balance * c.risk_per_trade,
            c.risk_per_trade * 100,
            estimated_risk,
            amount,
            sl,
            tp1,
            tp2,
            actual_rr,
        )

        order_side = "buy" if side == LONG else "sell"
        try:
            order = await self.client.create_order(
                symbol,
                order_side,
                amount,
                pos_side=side,
                tp_price=tp2,
                sl_price=sl,
            )
        except Exception as exc:
            logger.error("[POS] %s open failed: %s", symbol, exc)
            return None

        fill = order.avg_price if order.avg_price > 0 else price
        actual_risk = abs(fill - sl)
        if actual_risk <= 0:
            logger.critical("[POS] %s filled with invalid risk; emergency close required", symbol)
            return None
        # Keep structure target but recalculate TP1 from actual fill.
        tp1 = fill + c.tp1_r * actual_risk if side == LONG else fill - c.tp1_r * actual_risk
        actual_rr = abs(tp2 - fill) / actual_risk
        filled_amount = order.amount if order.amount > 0 else amount
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=fill,
            amount=filled_amount,
            full_amount=filled_amount,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            one_r=actual_risk,
            regime_at_entry=regime.name,
            bias_at_entry=bias.bias if bias is not None else "",
            entry_score=entry_score,
            entry_fee=order.fee_cost,
            entry_bar_ts=(
                df_5m.index[-1]
                if df_5m is not None and len(df_5m)
                else df_15m.index[-1]
            ),
            setup_type=getattr(entry_result, "setup_type", ""),
            trigger=getattr(entry_result, "trigger", ""),
            planned_rr=actual_rr,
            structure_room_r=getattr(entry_result, "structure_room_r", 0.0),
        )
        self._positions[symbol] = pos
        return pos

    async def check_exits_live(self, symbol: str, current_price: float) -> Optional[dict]:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        is_long = pos.side == LONG
        if (current_price <= pos.stop_loss) if is_long else (current_price >= pos.stop_loss):
            return await self._close_full(pos, current_price, "BE_HIT" if pos.tp1_hit else "SL_HIT")
        if not pos.tp1_hit and pos.tp1 is not None:
            tp1_hit = current_price >= pos.tp1 if is_long else current_price <= pos.tp1
            if tp1_hit:
                return await self._close_partial_tp1(pos, current_price)
        tp2_hit = current_price >= pos.tp2 if is_long else current_price <= pos.tp2
        if tp2_hit:
            return await self._close_full(pos, current_price, "TP2_HIT")
        return None

    @staticmethod
    def _is_no_position_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "51169" in text or "don't have any positions" in text

    def _leg_net_pnl(self, pos: Position, order, ticker_price: float, filled: float) -> dict:
        multiplier = 1.0 if pos.side == LONG else -1.0
        if order is not None and order.avg_price > 0:
            exit_price = order.avg_price
            exit_fee = order.fee_cost
            realized = order.realized_pnl
            if realized == 0.0:
                realized = multiplier * (exit_price - pos.entry_price) * filled
        else:
            exit_price = ticker_price
            realized = multiplier * (exit_price - pos.entry_price) * filled
            exit_fee = exit_price * filled * self.cfg.fee_rate
        entry_fee_alloc = pos.entry_fee * (filled / pos.full_amount) if pos.full_amount else 0.0
        net = realized - exit_fee - entry_fee_alloc
        return {
            "net": net,
            "realized": realized,
            "exit_fee": exit_fee,
            "entry_fee_alloc": entry_fee_alloc,
            "exit_price": exit_price,
        }

    async def _sync_closed_externally(self, pos: Position, price: float, reason: str) -> Optional[dict]:
        actual = await self.client.fetch_position_amount(pos.symbol, pos.side)
        if actual > 0:
            return None
        leg = self._leg_net_pnl(pos, None, price, pos.amount)
        trade_pnl = pos.realized_pnl + leg["net"]
        balance = await self.client.fetch_balance_usdt()
        self.risk.register_trade_result(trade_pnl, balance, time.time())
        del self._positions[pos.symbol]
        self._mark_closed(pos.symbol)
        self.entry_engine.on_position_closed(pos.symbol)
        return {
            "event": reason,
            "symbol": pos.symbol,
            "side": pos.side,
            "price": price,
            "pnl": leg["net"],
            "trade_pnl": trade_pnl,
            "realized": leg["realized"],
            "exit_fee": leg["exit_fee"],
            "entry_fee_alloc": leg["entry_fee_alloc"],
            "tp1_hit": pos.tp1_hit,
            "entry_price": pos.entry_price,
            "position": pos,
            "approximate": True,
        }

    async def _close_partial_tp1(self, pos: Position, price: float) -> dict:
        close_amount = min(round(pos.full_amount * self.cfg.tp1_fraction, 8), pos.amount)
        side = "sell" if pos.side == LONG else "buy"
        try:
            order = await self.client.create_order(
                pos.symbol,
                side,
                close_amount,
                pos_side=pos.side,
                reduce_only=True,
            )
        except Exception as exc:
            if self._is_no_position_error(exc):
                synced = await self._sync_closed_externally(pos, price, "TP1_THEN_EXTERNAL_CLOSE")
                if synced is not None:
                    return synced
            return {"event": "ERROR", "symbol": pos.symbol, "detail": f"TP1 close failed: {exc}"}

        filled = min(order.amount if order.amount > 0 else close_amount, pos.amount)
        leg = self._leg_net_pnl(pos, order, price, filled)
        pos.amount = round(pos.amount - filled, 8)
        pos.tp1_hit = True
        pos.realized_pnl += leg["net"]
        # Fee-aware positive breakeven. Exact entry can still be a net loss after
        # round-trip fees, so lock at least 0.08R or the fee equivalent.
        fee_lock = pos.entry_price * self.cfg.fee_rate * 2.2
        lock_distance = max(pos.one_r * getattr(self.cfg, "be_lock_r", 0.08), fee_lock)
        pos.stop_loss = (
            pos.entry_price + lock_distance if pos.side == LONG else pos.entry_price - lock_distance
        )
        sl_ok = await self.client.move_sl_to_breakeven(
            pos.symbol,
            pos.side,
            pos.stop_loss,
            pos.amount,
            tp_price=pos.tp2,
        )
        return {
            "event": "TP1_HIT",
            "symbol": pos.symbol,
            "side": pos.side,
            "price": leg["exit_price"],
            "pnl": leg["net"],
            "realized": leg["realized"],
            "exit_fee": leg["exit_fee"],
            "entry_fee_alloc": leg["entry_fee_alloc"],
            "sl_moved": sl_ok,
            "new_sl": pos.stop_loss,
            "position": pos,
        }

    async def _close_full(self, pos: Position, price: float, reason: str) -> dict:
        side = "sell" if pos.side == LONG else "buy"
        filled = pos.amount
        try:
            order = await self.client.create_order(
                pos.symbol,
                side,
                pos.amount,
                pos_side=pos.side,
                reduce_only=True,
            )
        except Exception as exc:
            if self._is_no_position_error(exc):
                synced = await self._sync_closed_externally(pos, price, reason)
                if synced is not None:
                    return synced
            return {"event": "ERROR", "symbol": pos.symbol, "detail": f"{reason} close failed: {exc}"}

        if order.amount > 0:
            filled = min(order.amount, pos.amount)
        leg = self._leg_net_pnl(pos, order, price, filled)
        trade_pnl = pos.realized_pnl + leg["net"]
        balance = await self.client.fetch_balance_usdt()
        self.risk.register_trade_result(trade_pnl, balance, time.time())
        del self._positions[pos.symbol]
        self._mark_closed(pos.symbol)
        self.entry_engine.on_position_closed(pos.symbol)
        return {
            "event": reason,
            "symbol": pos.symbol,
            "side": pos.side,
            "price": leg["exit_price"],
            "pnl": leg["net"],
            "trade_pnl": trade_pnl,
            "realized": leg["realized"],
            "exit_fee": leg["exit_fee"],
            "entry_fee_alloc": leg["entry_fee_alloc"],
            "tp1_hit": pos.tp1_hit,
            "entry_price": pos.entry_price,
            "position": pos,
        }

    async def process_closed_bar_exit_check(
        self,
        symbol: str,
        df_5m: pd.DataFrame,
    ) -> Optional[dict]:
        pos = self._positions.get(symbol)
        if pos is None or df_5m is None or len(df_5m) == 0:
            return None
        if self.cfg.signal_exit_requires_tp1 and not pos.tp1_hit:
            return None
        bar_ts = df_5m.index[-1]
        if pos.last_exit_check_bar_ts is not None and bar_ts <= pos.last_exit_check_bar_ts:
            return None
        pos.last_exit_check_bar_ts = bar_ts
        bars_since = (
            int((df_5m.index > pos.entry_bar_ts).sum())
            if pos.entry_bar_ts is not None
            else None
        )
        check = self.entry_engine.check_exit(df_5m, pos.side, bars_since)
        if not check.should_exit:
            return None
        event = await self._close_full(pos, float(df_5m["close"].iloc[-1]), check.reason)
        event["exit_detail"] = check.detail
        return event
