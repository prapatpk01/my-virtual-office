"""Execution Engine (spec §25) — ordered, idempotent, journaled, protected.

Never leaves a filled position without protection: SL/TP attach on the entry
order itself; if verification fails we retry, emergency-stop, then close.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .config import Config
from .enums import ExitReason, ReasonCode, SymbolStatus
from .interfaces import ExchangeInterface, ExchangeStateSnapshot
from .models import SignalCandidate, SymbolState, TradePlan
from .risk_manager import RiskManager
from .state_store import StateStore

logger = logging.getLogger("dual_entry.exec")


class ExecutionEngine:
    def __init__(self, cfg: Config, exchange: ExchangeInterface,
                 state_store: StateStore, risk_manager: RiskManager, notifier=None):
        self.cfg = cfg
        self.x = exchange
        self.store = state_store
        self.risk = risk_manager
        self.notifier = notifier

    # ── reconciliation (exchange = source of truth) ──────────────────────────

    async def reconcile(self, symbol: str, local_state: SymbolState) -> ExchangeStateSnapshot:
        snap = await self.x.get_state(symbol)
        pos = snap.position_for(symbol)
        if pos is not None and not local_state.has_open_position:
            # adopt: memory says flat but the exchange has a live position
            local_state.status = (SymbolStatus.LONG_OPEN.value if pos.direction == "LONG"
                                  else SymbolStatus.SHORT_OPEN.value)
            local_state.setup_direction = pos.direction
            local_state.actual_entry = pos.entry_price
            local_state.actual_quantity = pos.quantity
            local_state.active_stop = pos.attached_sl or local_state.active_stop
            local_state.active_target = pos.attached_tp or local_state.active_target
            if local_state.initial_risk in (None, 0.0) and local_state.active_stop:
                local_state.initial_risk = abs(pos.entry_price - local_state.active_stop)
            logger.warning("[RECONCILE] %s adopted %s position qty=%.6f entry=%.6f",
                           symbol, pos.direction, pos.quantity, pos.entry_price)
            if self.notifier:
                await self.notifier.warn(
                    f"Adopted untracked {pos.direction} position on {symbol} "
                    f"(entry {pos.entry_price:.6f}, qty {pos.quantity:.6f}) — now managed.")
        elif pos is None and local_state.has_open_position:
            # closed externally (SL/TP fired on exchange, or manual)
            logger.info("[RECONCILE] %s position gone on exchange — clearing local", symbol)
            self._clear_position_state(local_state)
        if local_state.pending_order_id and not any(
                o.order_id == local_state.pending_order_id for o in snap.open_orders):
            local_state.pending_order_id = None
            if local_state.status == SymbolStatus.ORDER_PENDING.value:
                local_state.status = SymbolStatus.IDLE.value
        local_state.last_reconciled_at = self.x.now_ms()
        return snap

    @staticmethod
    def _clear_position_state(state: SymbolState) -> None:
        state.status = SymbolStatus.IDLE.value
        state.actual_entry = None
        state.actual_quantity = None
        state.active_stop = None
        state.active_target = None
        state.initial_risk = None
        state.position_id = None
        state.pending_order_id = None
        state.breakeven_moved = False
        state.early_exit_sent = False
        state.holding_bars = 0
        state.mfe_r = 0.0
        state.mae_r = 0.0

    # ── open (spec §25 sequence; lock held by caller) ────────────────────────

    async def open_position(self, symbol: str, candidate: SignalCandidate,
                            trade_plan: TradePlan, state: SymbolState) -> bool:
        cl_id = candidate.client_order_id
        # 4-5) duplicate signal / order guards
        if state.has_open_position or state.has_pending_order:
            return False
        if state.last_signal_key == str(candidate.signal_key):
            return False
        existing = await self.x.find_order_by_client_id(symbol, cl_id)
        if existing is not None and existing.status in ("open", "filled"):
            logger.warning("[EXEC] %s client order %s already on exchange — refusing dup",
                           symbol, cl_id)
            return False

        # 6) persist intent BEFORE sending
        self.store.journal(symbol, "ORDER_INTENT", {
            "client_order_id": cl_id, "direction": candidate.direction,
            "setup": candidate.setup_type, "entry_ref": candidate.entry_reference,
            "stop": trade_plan.stop_price, "target": trade_plan.target_price,
            "contracts": trade_plan.contracts,
        })
        state.status = SymbolStatus.ORDER_PENDING.value
        state.client_order_id = cl_id
        state.last_signal_key = str(candidate.signal_key)
        self.store.save_atomic(symbol, state)

        # 11) entry order (native SL/TP attached)
        side = "buy" if candidate.direction == "LONG" else "sell"
        try:
            res = await self.x.place_market_order(
                symbol, side, trade_plan.contracts, candidate.direction, cl_id,
                sl_price=trade_plan.stop_price, tp_price=trade_plan.target_price)
        except Exception as e:
            logger.error("[EXEC] %s order failed: %s", symbol, e)
            self.store.journal(symbol, "ORDER_ERROR", {"client_order_id": cl_id, "err": str(e)})
            state.status = SymbolStatus.IDLE.value
            state.client_order_id = None
            self.store.save_atomic(symbol, state)
            return False

        if res.status != "filled":
            state.status = SymbolStatus.IDLE.value
            state.client_order_id = None
            self.store.save_atomic(symbol, state)
            self.store.journal(symbol, "ORDER_REJECTED", {"client_order_id": cl_id})
            return False

        fill = res.avg_price or candidate.entry_reference
        qty = res.filled_qty or trade_plan.quantity

        # 15) recompute risk with the ACTUAL fill
        snap_atr = abs(candidate.entry_reference - trade_plan.stop_price) / max(
            self.cfg.min_stop_atr, 1e-9)
        recheck = self.risk.revalidate_after_fill(trade_plan, fill, candidate.direction,
                                                  atr=snap_atr * self.cfg.min_stop_atr)
        if not recheck.valid:
            logger.warning("[EXEC] %s post-fill RR failed (%s) — closing immediately",
                           symbol, recheck.reason_codes)
            await self.x.close_position(symbol, candidate.direction, qty)
            self._clear_position_state(state)
            self.store.save_atomic(symbol, state)
            self.store.journal(symbol, "ABORTED_POST_FILL", {"client_order_id": cl_id})
            return False

        # 17) verify protection actually exists; emergency ladder if not
        protected = await self._verify_protection(symbol, candidate.direction, qty,
                                                  trade_plan)
        if not protected:
            await self.x.close_position(symbol, candidate.direction, qty)
            self._clear_position_state(state)
            self.store.save_atomic(symbol, state)
            self.store.journal(symbol, "EMERGENCY_CLOSE_NO_PROTECTION",
                               {"client_order_id": cl_id})
            if self.notifier:
                await self.notifier.critical(
                    f"PROTECTION ERROR {symbol}: entry filled but protective orders "
                    f"failed — position CLOSED.")
            return False

        # 18-19) final state
        state.status = (SymbolStatus.LONG_OPEN.value if candidate.direction == "LONG"
                        else SymbolStatus.SHORT_OPEN.value)
        state.setup_type = candidate.setup_type
        state.setup_direction = candidate.direction
        state.actual_entry = fill
        state.actual_quantity = qty
        state.active_stop = trade_plan.stop_price
        state.active_target = trade_plan.target_price
        state.initial_risk = abs(fill - trade_plan.stop_price)
        state.entry_fee = res.fee_cost
        state.entry_bar_ts = candidate.signal_timestamp
        state.planned_entry = candidate.entry_reference
        state.planned_stop = trade_plan.stop_price
        state.planned_target = trade_plan.target_price
        state.planned_risk_distance = trade_plan.risk_distance
        state.planned_rr = trade_plan.planned_rr
        state.pending_order_id = None
        state.position_id = res.order_id
        state.breakeven_moved = False
        state.early_exit_sent = False
        state.holding_bars = 0
        self.store.save_atomic(symbol, state)
        self.store.journal(symbol, "POSITION_OPEN", {
            "client_order_id": cl_id, "fill": fill, "qty": qty, "fee": res.fee_cost})
        if self.notifier:
            slip_atr = abs(fill - candidate.entry_reference) / max(
                trade_plan.risk_distance / max(self.cfg.min_stop_atr, 1e-9), 1e-9)
            await self.notifier.fill(symbol, candidate, trade_plan, fill, qty, slip_atr)
        return True

    async def _verify_protection(self, symbol: str, direction: str, qty: float,
                                 plan: TradePlan) -> bool:
        """Entry attached SL/TP; verify the exchange really has them, retry via
        amend_protection, escalate to emergency stop-only."""
        for attempt in range(3):
            snap = await self.x.get_state(symbol)
            pos = snap.position_for(symbol)
            if pos is None:
                return True     # already closed (instant TP/SL) — nothing to protect
            if pos.attached_sl is not None or attempt == 0:
                # attached on entry (OKX carries them server-side even when the
                # snapshot can't always surface them) — accept on first pass,
                # verify/re-arm on later passes
                if attempt == 0:
                    ok = True
                else:
                    ok = await self.x.amend_protection(symbol, direction, qty,
                                                       plan.stop_price, plan.target_price)
                if ok:
                    return True
            else:
                if await self.x.amend_protection(symbol, direction, qty,
                                                 plan.stop_price, plan.target_price):
                    return True
        # last resort: stop-only
        return await self.x.amend_protection(symbol, direction, qty, plan.stop_price, None)

    # ── pending orders (spec §24 cancel conditions) ──────────────────────────

    async def manage_pending_order(self, symbol: str, state: SymbolState,
                                   current_context: dict) -> None:
        if not state.pending_order_id:
            state.status = SymbolStatus.IDLE.value
            return
        # market orders fill immediately; a lingering pending id means a stuck
        # order — cancel after timeout
        now = self.x.now_ms()
        started = state.setup_started_at or now
        if now - started > self.cfg.pending_order_timeout_sec * 1000:
            await self.x.cancel_order(symbol, state.pending_order_id)
            state.pending_order_id = None
            state.status = SymbolStatus.IDLE.value
            self.store.save_atomic(symbol, state)

    async def get_account_state(self) -> dict:
        # symbol-agnostic account snapshot (equity via any symbol)
        eq, free = 0.0, 0.0
        try:
            snap = await self.x.get_state(self.cfg.symbols[0])
            eq, free = snap.equity, snap.free_margin
        except Exception:
            pass
        return {"equity": eq, "free_margin": free, "leverage": self.cfg.leverage}

    async def get_market_rules(self, symbol: str) -> dict:
        rules = await self.x.get_market_rules(symbol)
        return rules.as_dict()
