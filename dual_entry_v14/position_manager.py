"""Setup-specific Position Manager (spec §26) + cooldown (spec §27).

Exit priority: reconcile → exchange SL/TP → hard structure invalidation →
confirmed hard flip → setup-specific early exit → break-even → order update.
HMA cross alone NEVER closes a position.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from .config import Config, TF_MS
from .enums import ExitReason, SetupType, StructureState, SymbolStatus
from .indicator_engine import EntryIndicators
from .interfaces import ExchangeInterface
from .models import StructureView, SymbolState, TradeRecord
from .state_store import StateStore

logger = logging.getLogger("dual_entry.pos")


class PositionManager:
    def __init__(self, cfg: Config, exchange: ExchangeInterface,
                 state_store: StateStore, performance=None, notifier=None):
        self.cfg = cfg
        self.x = exchange
        self.store = state_store
        self.perf = performance
        self.notifier = notifier

    async def manage(self, symbol: str, state: SymbolState,
                     indicators_15m: Optional[EntryIndicators],
                     structure_15m: StructureView, structure_1h: StructureView,
                     macro_context, candle_context, zones: list) -> None:
        if not state.has_open_position or indicators_15m is None:
            return
        i = indicators_15m
        long = state.setup_direction == "LONG"
        price = i.val(i.closes)
        entry = state.actual_entry or price
        risk = state.initial_risk or 1e-9
        now_bar = int(i.timestamps[-1])
        state.holding_bars += 1

        r_now = ((price - entry) / risk) if long else ((entry - price) / risk)
        state.mfe_r = max(state.mfe_r, r_now)
        state.mae_r = min(state.mae_r, r_now)

        # 1-2) exchange reconcile happens in the main loop before manage()

        # 3) hard structure invalidation
        inv = state.invalidation_level
        if inv is not None and ((price < inv) if long else (price > inv)):
            await self._close(symbol, state, ExitReason.HARD_STRUCTURE_INVALIDATION, price)
            return

        # 4) confirmed hard flip: opposite 15M CHOCH + close beyond structure,
        #    or 1H confirmed opposite CHOCH, or strong displacement through the low
        opp = "SHORT" if long else "LONG"
        choch15 = (structure_15m.last_choch is not None
                   and structure_15m.last_choch.direction == opp
                   and structure_15m.last_choch.confirmed_at == now_bar)
        choch1h = structure_1h.recent_choch_against(state.setup_direction,
                                                    TF_MS["1h"] * 2, now_bar)
        disp_against = (i.body_atr >= 0.5 and
                        ((not i.closes[-1] > i.opens[-1]) if long else (i.closes[-1] > i.opens[-1])))
        broke_zone = False
        if state.active_zone_lower is not None and long:
            broke_zone = price < state.active_zone_lower
        if state.active_zone_upper is not None and not long:
            broke_zone = price > state.active_zone_upper
        if choch15 or choch1h or (disp_against and broke_zone):
            await self._close(symbol, state, ExitReason.HARD_FLIP, price)
            return

        # 5) setup-specific early exit
        if state.setup_type == SetupType.MOMENTUM.value:
            if await self._momentum_early_exit(symbol, state, i, structure_15m, price, long):
                return
        else:
            if await self._pullback_early_exit(symbol, state, i, structure_15m,
                                               candle_context, price, long):
                return

        # 6) break-even
        await self._maybe_breakeven(symbol, state, r_now, entry, risk, long)

        self.store.save_atomic(symbol, state)

    # ── setup-specific exits ─────────────────────────────────────────────────

    async def _pullback_early_exit(self, symbol, state, i, s15, candle_ctx,
                                   price, long) -> bool:
        """Soft exit (26.1): held >= 2 bars AND close beyond HMA10 AND >= 2 of 4
        weakness signals. HMA cross alone never exits."""
        c = self.cfg
        if state.holding_bars < c.min_hold_bars_soft_exit:
            return False
        hf, hs = i.val(i.hma_fast), i.val(i.hma_slow)
        beyond_hma10 = (price < hf) if long else (price > hf)
        if not beyond_hma10:
            return False
        weak = 0
        weak += 1 if ((i.val(i.roc) < 0) if long else (i.val(i.roc) > 0)) else 0
        weak += 1 if ((i.di_spread_long < 0) if long else (i.di_spread_short < 0)) else 0
        weak += 1 if ((price < hs) if long else (price > hs)) else 0
        rejection = (candle_ctx.best_bear() if long else candle_ctx.best_bull()) is not None
        weak += 1 if rejection else 0
        if weak >= 2:
            await self._close(symbol, state, ExitReason.EARLY_EXIT, price)
            return True
        return False

    async def _momentum_early_exit(self, symbol, state, i, s15, price, long) -> bool:
        """26.2: close back under breakout level + (ROC against or displacement);
        false-breakout hard exit on close back inside + displacement/CHOCH."""
        level = state.breakout_level
        if level is None:
            return False
        back_inside = (price < level) if long else (price > level)
        if not back_inside:
            return False
        roc_against = (i.val(i.roc) < 0) if long else (i.val(i.roc) > 0)
        disp_against = i.body_atr >= 0.22 and (
            (not (i.closes[-1] > i.opens[-1])) if long else (i.closes[-1] > i.opens[-1]))
        opp_choch = (s15.last_choch is not None
                     and s15.last_choch.direction == ("SHORT" if long else "LONG")
                     and s15.last_choch.confirmed_at == int(i.timestamps[-1]))
        if disp_against or opp_choch:
            await self._close(symbol, state, ExitReason.FALSE_BREAKOUT, price)
            return True
        if roc_against:
            await self._close(symbol, state, ExitReason.EARLY_EXIT, price)
            return True
        return False

    # ── break-even ───────────────────────────────────────────────────────────

    async def _maybe_breakeven(self, symbol, state, r_now, entry, risk, long) -> None:
        c = self.cfg
        if state.breakeven_moved:
            return
        trigger = (c.momentum_be_trigger_r if state.setup_type == SetupType.MOMENTUM.value
                   else c.pullback_be_trigger_r)
        if r_now < trigger:
            return
        new_sl = entry + c.be_lock_r * risk if long else entry - c.be_lock_r * risk
        ok = await self.x.amend_protection(symbol, state.setup_direction,
                                           state.actual_quantity or 0.0,
                                           new_sl, state.active_target)
        if ok:
            state.breakeven_moved = True
            state.active_stop = new_sl
            self.store.journal(symbol, "BREAKEVEN_MOVED", {"new_sl": new_sl, "r": r_now})
            if self.notifier:
                await self.notifier.breakeven(symbol, state.setup_type, new_sl, r_now)

    # ── close + cooldown + record ────────────────────────────────────────────

    async def _close(self, symbol: str, state: SymbolState, reason: ExitReason,
                     ref_price: float) -> None:
        res = await self.x.close_position(symbol, state.setup_direction,
                                          state.actual_quantity)
        exit_px = res.avg_price or ref_price
        entry = state.actual_entry or exit_px
        qty = res.filled_qty or (state.actual_quantity or 0.0)
        risk = state.initial_risk or 1e-9
        long = state.setup_direction == "LONG"
        gross = (exit_px - entry) * qty if long else (entry - exit_px) * qty
        fees = state.entry_fee + res.fee_cost
        pnl = (res.realized_pnl or gross) - res.fee_cost - state.entry_fee
        result_r = ((exit_px - entry) / risk) if long else ((entry - exit_px) / risk)

        if self.perf is not None:
            self.perf.record(TradeRecord(
                trade_id=uuid.uuid4().hex[:12], symbol=symbol,
                setup_type=state.setup_type or "?", direction=state.setup_direction,
                signal_time=state.entry_bar_ts or 0, entry_time=state.entry_bar_ts or 0,
                exit_time=self.x.now_ms(),
                signal_score=0.0, threshold=0.0, edge_score=0.0,
                entry_price=entry, stop_price=state.active_stop or 0.0,
                target_price=state.active_target or 0.0, exit_price=exit_px,
                initial_risk=risk, actual_rr=state.planned_rr or 0.0,
                pnl_cash=pnl, pnl_percent=0.0, result_r=result_r,
                exit_reason=reason.value,
                max_favorable_excursion=state.mfe_r, max_adverse_excursion=state.mae_r,
                holding_bars=state.holding_bars,
                regime_at_entry=state.previous_regime or "",
                bias_at_entry="", macro_structure_at_entry="",
                active_zone_type=state.active_zone_type, zone_score=state.active_zone_score,
                pattern_type=state.pattern_type, candle_pattern=None,
                structure_room_r=0.0, fees=fees,
            ))

        # cooldown per spec §27
        c = self.cfg
        bars = {ExitReason.STOP_LOSS: c.sl_cooldown_bars,
                ExitReason.FALSE_BREAKOUT: c.false_breakout_cooldown_bars,
                ExitReason.EARLY_EXIT: c.early_exit_cooldown_bars,
                ExitReason.TAKE_PROFIT: c.tp_cooldown_bars,
                ExitReason.HARD_FLIP: c.hard_flip_cooldown_bars,
                }.get(reason, c.cooldown_bars)
        bar_ms = TF_MS[c.entry_timeframe]
        now = self.x.now_ms()
        cur_bar = (now // bar_ms) * bar_ms
        state.last_exit_bar = cur_bar
        state.cooldown_until_bar = cur_bar + bars * bar_ms if bars > 0 else cur_bar
        if result_r < 0:
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0
        if state.consecutive_losses >= c.loss_streak_limit:
            streak_ms = max(c.loss_streak_cooldown_minutes * 60_000, 2 * bar_ms)
            state.cooldown_until = now + streak_ms
            state.consecutive_losses = 0

        from .execution_engine import ExecutionEngine
        ExecutionEngine._clear_position_state(state)
        state.status = SymbolStatus.COOLDOWN.value if bars > 0 else SymbolStatus.IDLE.value
        self.store.save_atomic(symbol, state)
        self.store.journal(symbol, "POSITION_CLOSED",
                           {"reason": reason.value, "exit": exit_px, "r": result_r, "pnl": pnl})
        if self.notifier:
            await self.notifier.exit(symbol, state.setup_type or "?", reason.value,
                                     result_r, exit_px)
