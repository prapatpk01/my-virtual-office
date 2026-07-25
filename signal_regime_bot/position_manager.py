"""Live position lifecycle and structure-aware SL/TP planning.

The pure stop/target helpers are shared by live and backtest code.  Live state
keeps one position per symbol.  Partial TP1 does not update loss-streak state;
risk accounting is registered once, when the full trade closes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
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


# One-time restart recovery for the currently open ETH short reported by the
# user.  It applies only when symbol, side, entry and amount all match closely,
# so it cannot affect later trades.  Live OKX SL/TP still takes precedence.
_RECOVERY_OVERRIDES = (
    {
        "symbol": "ETH/USDT:USDT",
        "side": SHORT,
        "entry": 1857.85,
        "amount": 0.349,
        "sl": 1872.08,
        "tp1": 1846.88,
        "tp2": 1828.88,
    },
)


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




def _stop_distance_bounds(cfg: Config, entry: float, atr_value: float) -> tuple[float, float, float]:
    """Return fee-aware minimum/maximum stop distance and estimated cost distance."""
    cost_distance = entry * (
        2.0 * max(getattr(cfg, "fee_rate", 0.0), 0.0)
        + max(getattr(cfg, "expected_slippage_pct", 0.0), 0.0)
    )
    minimum = max(
        entry * max(getattr(cfg, "sl_min_pct", 0.0075), 0.0),
        atr_value * getattr(cfg, "dual_min_stop_atr", 0.80),
        cost_distance * getattr(cfg, "stop_fee_floor_mult", 3.0),
    )
    maximum = max(
        atr_value * getattr(cfg, "dual_max_stop_atr", 2.20),
        minimum,
    )
    pct_cap = entry * max(getattr(cfg, "sl_max_pct", 0.020), 0.0)
    if pct_cap > 0:
        maximum = max(minimum, min(maximum, pct_cap))
    return minimum, maximum, cost_distance


def _normalize_planned_stop(
    cfg: Config, side: str, entry: float, planned_stop: float, atr_value: float
) -> tuple[float, float, float]:
    minimum, maximum, cost_distance = _stop_distance_bounds(cfg, entry, atr_value)
    raw_distance = (entry - planned_stop) if side == LONG else (planned_stop - entry)
    if raw_distance <= 0:
        return 0.0, 0.0, cost_distance
    distance = max(raw_distance, minimum)
    if distance > maximum * (1.0 + 1e-9):
        return 0.0, distance, cost_distance
    stop = entry - distance if side == LONG else entry + distance
    return stop, distance, cost_distance


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
        self._state_path = os.path.join(self.cfg.state_dir, "open_positions.json")
        self._state_version = 1

    @staticmethod
    def _ts_to_text(value):
        if value is None:
            return None
        try:
            return pd.Timestamp(value).isoformat()
        except Exception:
            return None

    @staticmethod
    def _text_to_ts(value):
        if not value:
            return None
        try:
            return pd.Timestamp(value)
        except Exception:
            return None

    def _position_to_dict(self, pos: Position) -> dict:
        return {
            "symbol": pos.symbol, "side": pos.side,
            "entry_price": pos.entry_price, "amount": pos.amount,
            "full_amount": pos.full_amount, "stop_loss": pos.stop_loss,
            "tp1": pos.tp1, "tp2": pos.tp2, "one_r": pos.one_r,
            "tp1_hit": pos.tp1_hit, "realized_pnl": pos.realized_pnl,
            "opened_at": pos.opened_at, "regime_at_entry": pos.regime_at_entry,
            "bias_at_entry": pos.bias_at_entry, "entry_score": pos.entry_score,
            "entry_fee": pos.entry_fee,
            "entry_bar_ts": self._ts_to_text(pos.entry_bar_ts),
            "last_exit_check_bar_ts": self._ts_to_text(pos.last_exit_check_bar_ts),
            "setup_type": pos.setup_type, "trigger": pos.trigger,
            "planned_rr": pos.planned_rr, "structure_room_r": pos.structure_room_r,
        }

    def _position_from_dict(self, item: dict) -> Optional[Position]:
        try:
            side = str(item["side"]).lower()
            if side not in (LONG, SHORT):
                return None
            return Position(
                symbol=str(item["symbol"]), side=side,
                entry_price=float(item["entry_price"]), amount=float(item["amount"]),
                full_amount=float(item.get("full_amount", item["amount"])),
                stop_loss=float(item["stop_loss"]),
                tp1=(None if item.get("tp1") is None else float(item["tp1"])),
                tp2=float(item["tp2"]), one_r=float(item["one_r"]),
                tp1_hit=bool(item.get("tp1_hit", False)),
                realized_pnl=float(item.get("realized_pnl", 0.0)),
                opened_at=float(item.get("opened_at", time.time())),
                regime_at_entry=str(item.get("regime_at_entry", "")),
                bias_at_entry=str(item.get("bias_at_entry", "")),
                entry_score=float(item.get("entry_score", 0.0)),
                entry_fee=float(item.get("entry_fee", 0.0)),
                entry_bar_ts=self._text_to_ts(item.get("entry_bar_ts")),
                last_exit_check_bar_ts=self._text_to_ts(item.get("last_exit_check_bar_ts")),
                setup_type=str(item.get("setup_type", "")),
                trigger=str(item.get("trigger", "")),
                planned_rr=float(item.get("planned_rr", 0.0)),
                structure_room_r=float(item.get("structure_room_r", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("[STATE] invalid persisted position skipped: %s", exc)
            return None

    def save_state(self) -> None:
        """Atomically persist every open position and lifecycle field."""
        try:
            os.makedirs(self.cfg.state_dir, exist_ok=True)
            payload = {
                "version": self._state_version,
                "saved_at": time.time(),
                "positions": [self._position_to_dict(p) for p in self._positions.values()],
                "recently_closed": self._recently_closed,
            }
            fd, tmp = tempfile.mkstemp(prefix="open_positions.", suffix=".tmp", dir=self.cfg.state_dir)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, separators=(",", ":"), sort_keys=True)
                    f.flush(); os.fsync(f.fileno())
                os.replace(tmp, self._state_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError as exc:
            logger.error("[STATE] position state save failed: %s", exc)

    def load_state(self) -> int:
        """Restore local lifecycle state before OKX reconciliation.

        OKX remains authoritative. Reconciliation validates every restored item
        and discards stale local records.
        """
        try:
            with open(self._state_path) as f:
                payload = json.load(f)
        except FileNotFoundError:
            return 0
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("[STATE] position state load failed: %s", exc)
            return 0
        loaded = 0
        self._positions.clear()
        for item in payload.get("positions", []):
            pos = self._position_from_dict(item)
            if pos is not None:
                self._positions[pos.symbol] = pos
                loaded += 1
        now = time.time()
        self._recently_closed = {
            str(k): float(v) for k, v in payload.get("recently_closed", {}).items()
            if now - float(v) < max(getattr(self.cfg, "reconcile_settle_grace_sec", 20), 20) * 10
        }
        logger.info("[STATE] restored %d persisted open position(s)", loaded)
        return loaded

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

    @staticmethod
    def _matching_recovery_override(symbol: str, side: str, entry: float, amount: float):
        for item in _RECOVERY_OVERRIDES:
            if item["symbol"] != symbol or item["side"] != side:
                continue
            entry_ok = abs(entry - item["entry"]) <= max(0.02, item["entry"] * 0.00002)
            amount_ok = abs(amount - item["amount"]) <= max(0.0005, item["amount"] * 0.01)
            if entry_ok and amount_ok:
                return item
        return None

    async def _sync_live_protection(self, pos: Position) -> bool:
        """Refresh an already tracked position from live OKX TP/SL values."""
        sl, tp = await self.client.fetch_attached_stops(pos.symbol, pos.side)
        override = self._matching_recovery_override(
            pos.symbol, pos.side, pos.entry_price, pos.full_amount
        )
        if sl is None and override is not None:
            sl = float(override["sl"])
        if tp is None and override is not None:
            tp = float(override["tp2"])
        if sl is None or tp is None:
            logger.error(
                "[RECONCILE] %s tracked position cannot be synced: live SL=%s TP=%s",
                pos.symbol, sl, tp,
            )
            return False

        pos.stop_loss = float(sl)
        pos.tp2 = float(tp)
        pos.one_r = abs(pos.entry_price - pos.stop_loss)
        if override is not None and not pos.tp1_hit:
            pos.tp1 = float(override["tp1"])
        elif not pos.tp1_hit:
            pos.tp1 = calc_take_profits(
                pos.side, pos.entry_price, pos.stop_loss,
                self.cfg.tp1_r, self.cfg.tp2_r,
            )[0]
        return True

    async def reconcile_with_exchange(self, symbols: list[str]) -> list[str]:
        adopted: list[str] = []
        now = time.time()
        c = self.cfg
        for symbol in symbols:
            # A periodic reconciliation must also repair a tracked position
            # whose internal SL/TP was reconstructed incorrectly on startup.
            existing = self.get(symbol)
            if existing is not None:
                details = await self.client.fetch_position_details(symbol, existing.side)
                if not details or float(details.get("amount", 0.0)) <= 0:
                    logger.warning("[STATE] removing stale persisted %s %s; no live OKX position", symbol, existing.side)
                    self._positions.pop(symbol, None)
                    self.save_state()
                    continue
                # OKX is authoritative for live quantity and average entry, while
                # persisted lifecycle fields (TP1 hit, banked PnL, setup metadata) survive.
                existing.amount = float(details.get("amount", existing.amount))
                if not existing.tp1_hit:
                    existing.full_amount = max(existing.full_amount, existing.amount)
                live_entry = float(details.get("entry_price", existing.entry_price) or existing.entry_price)
                if live_entry > 0:
                    existing.entry_price = live_entry
                if await self._sync_live_protection(existing):
                    self.save_state()
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
            override = self._matching_recovery_override(symbol, side, entry, amount)
            sl, tp = await self.client.fetch_attached_stops(symbol, side)

            # Never invent a 3% emergency stop.  For the currently open ETH
            # position, use the exact verified protection values if an OKX API
            # response is temporarily incomplete.  Other positions are not
            # adopted until their live protection can be verified.
            if sl is None and override is not None:
                sl = float(override["sl"])
                logger.warning("[RECONCILE] %s using verified recovery SL %.8f", symbol, sl)
            if tp is None and override is not None:
                tp = float(override["tp2"])
                logger.warning("[RECONCILE] %s using verified recovery TP2 %.8f", symbol, tp)
            if sl is None or tp is None:
                msg = f"{symbol} {side.upper()} ⚠️ MANUAL REVIEW — live OKX SL/TP unavailable; not adopted"
                adopted.append(msg)
                logger.error("[RECONCILE] %s; SL=%s TP=%s", msg, sl, tp)
                continue

            one_r = abs(entry - float(sl))
            be_tolerance = max(one_r * getattr(c, "be_lock_r", 0.08), entry * c.fee_rate * 2)
            tp1_hit = (
                float(sl) >= entry - be_tolerance if side == LONG else float(sl) <= entry + be_tolerance
            )
            if tp1_hit:
                tp1 = None
            elif override is not None:
                tp1 = float(override["tp1"])
            else:
                tp1 = calc_take_profits(side, entry, float(sl), c.tp1_r, c.tp2_r)[0]

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
                regime_at_entry="ADOPTED_OKX",
                bias_at_entry="ADOPTED_OKX",
                setup_type="ADOPTED_OKX",
            )
            adopted.append(f"{symbol} {side.upper()}")
            self.save_state()
        return adopted

    async def _emergency_flatten_new_fill(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        fill: float,
        entry_fee: float,
        reason: str,
    ) -> bool:
        """Immediately flatten a fill that fails post-fill safety validation.

        The entry already exists on the exchange, so returning ``None`` without
        flattening would create an untracked live position.  Retry the reduce-only
        close and confirm the exchange amount is zero before reporting success.
        """
        close_side = "sell" if side == LONG else "buy"
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            close_order = None
            try:
                close_order = await self.client.create_order(
                    symbol,
                    close_side,
                    amount,
                    pos_side=side,
                    reduce_only=True,
                )
            except Exception as exc:
                last_error = exc
                if not self._is_no_position_error(exc):
                    logger.critical(
                        "[POS] %s emergency close attempt %d/3 failed: %s",
                        symbol,
                        attempt,
                        exc,
                    )
            try:
                remaining = await self.client.fetch_position_amount(symbol, side)
            except Exception as exc:
                last_error = exc
                remaining = amount
            if remaining <= 0:
                exit_price = (
                    close_order.avg_price
                    if close_order is not None and close_order.avg_price > 0
                    else fill
                )
                exit_fee = (
                    close_order.fee_cost
                    if close_order is not None
                    else exit_price * amount * self.cfg.fee_rate
                )
                gross = (exit_price - fill) * amount if side == LONG else (fill - exit_price) * amount
                net = gross - max(entry_fee, 0.0) - max(exit_fee, 0.0)
                try:
                    balance = await self.client.fetch_balance_usdt()
                    self.risk.register_trade_result(net, balance, time.time())
                except Exception as exc:
                    logger.error("[POS] %s could not register emergency-close PnL: %s", symbol, exc)
                self._mark_closed(symbol)
                self.entry_engine.on_position_closed(
                    symbol, side.upper(), reason, net
                )
                logger.critical(
                    "[POS] %s emergency-flattened after %s; estimated net PnL %.2f",
                    symbol,
                    reason,
                    net,
                )
                return True
            await asyncio.sleep(0.5 * attempt)
        logger.critical(
            "[POS] %s EMERGENCY CLOSE FAILED after %s; position remains live: %s",
            symbol,
            reason,
            last_error,
        )
        return False

    def _track_recovery_position(
        self,
        *,
        symbol: str,
        side: str,
        fill: float,
        amount: float,
        sl: float,
        tp2: float,
        atr_value: float,
        regime: RegimeResult,
        bias: BiasResult,
        entry_score: float,
        entry_fee: float,
        df_15m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame],
        entry_result,
        reason: str,
    ) -> Position:
        """Keep a failed-to-flatten exchange fill under local management."""
        min_risk = max(
            atr_value * getattr(self.cfg, "dual_min_stop_atr", 0.55),
            fill * max(self.cfg.sl_min_pct, 0.001),
        )
        valid_sl = (side == LONG and sl < fill) or (side == SHORT and sl > fill)
        if not valid_sl:
            sl = fill - min_risk if side == LONG else fill + min_risk
        one_r = max(abs(fill - sl), min_risk)
        if side == LONG:
            sl = min(sl, fill - one_r)
            if tp2 <= fill:
                tp2 = fill + one_r * max(self.cfg.minimum_actual_rr, 1.20)
            tp1 = fill + self.cfg.tp1_r * one_r
        else:
            sl = max(sl, fill + one_r)
            if tp2 >= fill:
                tp2 = fill - one_r * max(self.cfg.minimum_actual_rr, 1.20)
            tp1 = fill - self.cfg.tp1_r * one_r
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=fill,
            amount=amount,
            full_amount=amount,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            one_r=one_r,
            regime_at_entry=regime.name,
            bias_at_entry=bias.bias if bias is not None else "",
            entry_score=entry_score,
            entry_fee=entry_fee,
            entry_bar_ts=(
                df_5m.index[-1]
                if df_5m is not None and len(df_5m)
                else df_15m.index[-1]
            ),
            setup_type=f"RECOVERY:{reason}",
            trigger=getattr(entry_result, "trigger", ""),
            planned_rr=abs(tp2 - fill) / max(one_r, ind.EPSILON),
            structure_room_r=getattr(entry_result, "structure_room_r", 0.0),
        )
        self._positions[symbol] = pos
        self.save_state()
        return pos

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
            proposed_sl = float(planned_stop)
        else:
            proposed_sl = calc_stop_loss(
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
        sl, stop_distance, cost_distance = _normalize_planned_stop(
            c, side, price, proposed_sl, atr_value
        )
        if sl <= 0 or (side == LONG and sl >= price) or (side == SHORT and sl <= price):
            logger.info(
                "[POS] %s rejected: structure stop %.8f is invalid/too wide after fee-aware bounds",
                symbol, proposed_sl,
            )
            return None
        fee_drag_r = cost_distance / max(stop_distance, ind.EPSILON)
        if fee_drag_r > getattr(c, "max_fee_drag_r", 0.35):
            logger.info(
                "[POS] %s rejected: estimated fee drag %.2fR exceeds %.2fR",
                symbol, fee_drag_r, getattr(c, "max_fee_drag_r", 0.35),
            )
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
        filled_amount = order.amount if order.amount > 0 else amount
        actual_risk = (fill - sl) if side == LONG else (sl - fill)
        if actual_risk <= 0:
            reason = f"invalid post-fill stop: fill={fill:.8f} sl={sl:.8f}"
            flattened = await self._emergency_flatten_new_fill(
                symbol=symbol,
                side=side,
                amount=filled_amount,
                fill=fill,
                entry_fee=order.fee_cost,
                reason=reason,
            )
            if flattened:
                return None
            # Never leave a live exchange position untracked if emergency close fails.
            pos = self._track_recovery_position(
                symbol=symbol, side=side, fill=fill, amount=filled_amount,
                sl=sl, tp2=tp2, atr_value=atr_value, regime=regime, bias=bias,
                entry_score=entry_score, entry_fee=order.fee_cost,
                df_15m=df_15m, df_5m=df_5m, entry_result=entry_result, reason="INVALID_RISK",
            )
            try:
                await self.client.move_sl_to_breakeven(
                    symbol, side, pos.stop_loss, pos.amount, tp_price=pos.tp2
                )
            except Exception as exc:
                logger.critical("[POS] %s recovery protection update failed: %s", symbol, exc)
            return pos

        # Keep the structure target but recalculate TP1 from the actual fill.
        tp1 = fill + c.tp1_r * actual_risk if side == LONG else fill - c.tp1_r * actual_risk
        actual_rr = ((tp2 - fill) / actual_risk) if side == LONG else ((fill - tp2) / actual_risk)
        minimum_rr = getattr(c, "minimum_actual_rr", 1.20)
        if actual_rr < minimum_rr:
            reason = f"post-fill RR {actual_rr:.2f} below minimum {minimum_rr:.2f}"
            flattened = await self._emergency_flatten_new_fill(
                symbol=symbol,
                side=side,
                amount=filled_amount,
                fill=fill,
                entry_fee=order.fee_cost,
                reason=reason,
            )
            if flattened:
                return None
            # The exchange position still exists. Track and manage it rather than
            # silently returning None and losing control of a live position.
            return self._track_recovery_position(
                symbol=symbol, side=side, fill=fill, amount=filled_amount,
                sl=sl, tp2=tp2, atr_value=atr_value, regime=regime, bias=bias,
                entry_score=entry_score, entry_fee=order.fee_cost,
                df_15m=df_15m, df_5m=df_5m, entry_result=entry_result, reason="LOW_ACTUAL_RR",
            )

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
        self.save_state()
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
        self.save_state()
        self.entry_engine.on_position_closed(
            pos.symbol, pos.side.upper(), reason, trade_pnl
        )
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

    def _fee_adjusted_runner_stop(self, pos: Position, tp1_fill: float) -> tuple[float, bool, float]:
        """Solve the runner stop so the whole trade remains net-positive after fees.

        Uses the already banked TP1 net, the remaining allocated entry fee, and
        the estimated fee on the runner exit. The stop is kept on the valid side
        of the current TP1 fill to avoid an immediately-triggered stop order.
        """
        qty = max(pos.amount, ind.EPSILON)
        fee = max(getattr(self.cfg, "fee_rate", 0.0), 0.0)
        market_buffer = max(
            pos.one_r * getattr(self.cfg, "be_market_buffer_r", 0.05),
            pos.entry_price * 0.0001,
        )
        # Fixed breakeven+N·R lock (if configured) — a raw price offset, still
        # clamped to the valid side of the TP1 fill so it can't trigger instantly.
        lock = getattr(self.cfg, "runner_lock_r", None)
        if lock is not None:
            if pos.side == LONG:
                desired = pos.entry_price + lock * pos.one_r
                stop = min(desired, tp1_fill - market_buffer)
                return stop, stop + 1e-12 >= desired, lock * pos.full_amount * pos.one_r
            desired = pos.entry_price - lock * pos.one_r
            stop = max(desired, tp1_fill + market_buffer)
            return stop, stop - 1e-12 <= desired, lock * pos.full_amount * pos.one_r
        target_cash = (
            pos.full_amount
            * pos.one_r
            * getattr(self.cfg, "be_trade_lock_r", 0.05)
        )
        remaining_entry_fee = pos.entry_fee * (pos.amount / max(pos.full_amount, ind.EPSILON))
        cash_needed = target_cash - pos.realized_pnl + remaining_entry_fee
        per_unit_needed = cash_needed / qty
        if pos.side == LONG:
            required = (pos.entry_price + per_unit_needed) / max(1.0 - fee, ind.EPSILON)
            max_valid = tp1_fill - market_buffer
            stop = min(max_valid, required) if required > max_valid else required
            guarantee_ok = stop + 1e-12 >= required
        else:
            required = (pos.entry_price - per_unit_needed) / (1.0 + fee)
            min_valid = tp1_fill + market_buffer
            stop = max(min_valid, required) if required < min_valid else required
            guarantee_ok = stop - 1e-12 <= required
        return stop, guarantee_ok, target_cash

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
        # Exact fee-adjusted trade-level breakeven. This uses TP1 banked net and
        # the remaining runner fees instead of applying a raw price offset that
        # can accidentally place the stop above TP1 and still leave the trade red.
        pos.stop_loss, guarantee_ok, target_cash = self._fee_adjusted_runner_stop(
            pos, leg["exit_price"]
        )
        if not guarantee_ok:
            logger.warning(
                "[POS] %s runner stop had to be capped by live price; trade-level net lock is not guaranteed",
                pos.symbol,
            )
        sl_ok = await self.client.move_sl_to_breakeven(
            pos.symbol,
            pos.side,
            pos.stop_loss,
            pos.amount,
            tp_price=pos.tp2,
        )
        self.save_state()
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
            "cumulative_pnl": pos.realized_pnl,
            "trade_lock_target": target_cash,
            "trade_lock_guaranteed": guarantee_ok,
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
        self.save_state()
        self.entry_engine.on_position_closed(
            pos.symbol, pos.side.upper(), reason, trade_pnl
        )
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
        self.save_state()
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
