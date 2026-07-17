"""DUAL ENTRY PRECISION V1.4 — main loop (spec §34).

Per-symbol asyncio.Lock, closed candles only, exchange as source of truth,
atomic state persistence in a finally block.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .bias_engine import BiasEngine
from .candidate_selector import CandidateSelector
from .candle_engine import CandleEngine
from .config import Config, TF_MS, load_config
from .data_quality_gate import DataQualityGate
from .diagnostic_engine import DiagnosticEngine, _sym
from .enums import ReasonCode, SymbolStatus
from .execution_engine import ExecutionEngine
from .execution_quality_gate import ExecutionQualityGate
from .indicator_engine import IndicatorEngine
from .liquidity_engine import LiquidityEngine
from .macro_context_engine import MacroContextEngine
from .market_data import MarketData
from .momentum_engine import MomentumEngine
from .notifier import Notifier
from .okx_exchange import OKXExchange
from .pattern_engine import PatternEngine
from .performance_engine import PerformanceEngine
from .portfolio_risk_manager import OpenPositionInfo, PortfolioRiskManager
from .position_manager import PositionManager
from .pullback_engine import PullbackEngine
from .regime_engine import RegimeEngine
from .risk_manager import RiskManager
from .state_store import StateStore
from .structure_engine import StructureEngine
from .supply_demand_engine import SupplyDemandEngine
from .support_resistance_engine import SupportResistanceEngine
from .swing_engine import SwingEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dual_entry.main")


def build_candle_key(symbol: str, timeframe: str, candle) -> tuple:
    return (symbol, timeframe, int(candle.timestamp))


class Bot:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        c = self.cfg
        self.exchange = OKXExchange(c)
        self.market_data = MarketData(c, self.exchange)
        self.state_store = StateStore(c.state_dir)
        self.notifier = Notifier(c.telegram_token, c.telegram_chat_id)
        self.diag = DiagnosticEngine()
        self.perf = PerformanceEngine(c, c.state_dir)

        self.quality_gate = DataQualityGate(c)
        self.indicators = IndicatorEngine(c)
        self.swings = SwingEngine(c)
        self.structure = StructureEngine(c)
        self.sr_zones = SupportResistanceEngine(c)
        self.sd_zones = SupplyDemandEngine(c)
        self.patterns = PatternEngine(c)
        self.liquidity = LiquidityEngine(c)
        self.candles = CandleEngine(c)
        self.macro = MacroContextEngine(c)
        self.bias = BiasEngine(c)
        self.regime = RegimeEngine(c)
        self.pullback = PullbackEngine(c)
        self.momentum = MomentumEngine(c)
        self.selector = CandidateSelector()
        self.portfolio = PortfolioRiskManager(c)
        self.risk = RiskManager(c)
        self.exec_quality = ExecutionQualityGate(c)
        self.execution = ExecutionEngine(c, self.exchange, self.state_store,
                                         self.risk, self.notifier)
        self.positions = PositionManager(c, self.exchange, self.state_store,
                                         self.perf, self.notifier)

        self.locks = {s: asyncio.Lock() for s in c.symbols}
        self._running = False

    # ── per-symbol pipeline (spec §34 pseudocode, faithfully) ────────────────

    async def process_symbol(self, symbol: str) -> None:
        async with self.locks[symbol]:
            state = self.state_store.get(symbol)
            try:
                exchange_state = await self.execution.reconcile(symbol, state)

                candles_15m = await self.market_data.get_closed_candles(
                    symbol, "15m", self.cfg.fetch_15m)
                candles_1h = await self.market_data.get_closed_candles(
                    symbol, "1h", self.cfg.fetch_1h)
                candles_4h = await self.market_data.get_closed_candles(
                    symbol, "4h", self.cfg.fetch_4h)

                quality = self.quality_gate.evaluate(
                    symbol, candles_15m, candles_1h, candles_4h,
                    exchange_state={"spread_pct": exchange_state.spread_pct},
                    now_ms=self.exchange.now_ms())
                if not quality.valid:
                    self.diag.record_rejection(symbol, quality.reason_codes)
                    return

                candle_key = build_candle_key(symbol, "15m", candles_15m[-1])
                if candle_key == state.last_processed_candle:
                    return
                bar_ts = int(candles_15m[-1].timestamp)

                # volatility shock lockout bookkeeping
                if quality.shock:
                    bars = (self.cfg.severe_shock_lockout_bars if quality.severe_shock
                            else self.cfg.shock_lockout_bars)
                    state.shock_lockout_until_bar = bar_ts + bars * TF_MS["15m"]
                    self.diag.record_rejection(symbol, [ReasonCode.VOLATILITY_SHOCK.value])
                shock_locked = (state.shock_lockout_until_bar is not None
                                and bar_ts <= state.shock_lockout_until_bar)

                ind_15m = self.indicators.calculate_entry(candles_15m)
                ind_1h = self.indicators.calculate_context(candles_1h)
                ind_4h = self.indicators.calculate_macro(candles_4h)

                swings_15m = self.swings.calculate(candles_15m, "15m")
                swings_1h = self.swings.calculate(candles_1h, "1h")
                swings_4h = self.swings.calculate(candles_4h, "4h")

                structure_15m = self.structure.evaluate(candles_15m, swings_15m, ind_15m)
                structure_1h = self.structure.evaluate(candles_1h, swings_1h, ind_1h)
                structure_4h = self.structure.evaluate(candles_4h, swings_4h, ind_4h)

                zones = self.sr_zones.build_zones(
                    symbol,
                    {"15m": candles_15m, "1h": candles_1h, "4h": candles_4h},
                    {"15m": swings_15m, "1h": swings_1h, "4h": swings_4h},
                    {"15m": structure_15m, "1h": structure_1h, "4h": structure_4h})
                sd = self.sd_zones.evaluate(candles_1h, candles_4h,
                                            {"1h": structure_1h, "4h": structure_4h})
                patterns = self.patterns.evaluate(candles_1h, candles_4h,
                                                  swings_1h, swings_4h, zones)
                macro_ctx = self.macro.evaluate(ind_4h, structure_4h, zones, sd)
                bias = self.bias.evaluate(ind_1h, structure_1h, macro_ctx)
                regime = self.regime.classify(ind_15m, structure_15m, zones,
                                              state.previous_regime,
                                              state.candidate_regime,
                                              state.candidate_regime_count)
                candle_ctx = self.candles.evaluate(candles_15m, ind_15m, zones)
                liq = self.liquidity.evaluate(candles_15m, candles_1h,
                                              swings_15m, swings_1h, zones)

                self.diag.set_view(symbol,
                    f"{_sym(symbol)} | {state.status} | 4H={macro_ctx.classification} | "
                    f"1H={bias.bias} | 15M={regime.confirmed_regime}")

                if state.has_open_position:
                    await self.positions.manage(symbol, state, ind_15m, structure_15m,
                                                structure_1h, macro_ctx, candle_ctx, zones)
                elif state.has_pending_order:
                    await self.execution.manage_pending_order(symbol, state, {
                        "bias": bias, "regime": regime, "structure_15m": structure_15m,
                        "structure_1h": structure_1h, "macro_context": macro_ctx,
                        "zones": zones})
                elif state.cooldown_active(self.exchange.now_ms(), bar_ts):
                    self.diag.log_cooldown(symbol, state)
                else:
                    if state.status == SymbolStatus.COOLDOWN.value:
                        state.status = SymbolStatus.IDLE.value
                    await self._look_for_entry(symbol, state, ind_15m, bias, macro_ctx,
                                               regime, structure_15m, structure_1h,
                                               zones, patterns, candle_ctx, liq, sd,
                                               shock_locked, exchange_state)

                state.last_processed_candle = candle_key
                state.previous_regime = regime.confirmed_regime
                state.candidate_regime = regime.candidate_regime
                state.candidate_regime_count = regime.candidate_count
            except Exception as exc:
                self.diag.record_error(symbol, exc)
                try:
                    await self.notifier.send_critical_error(symbol, exc)
                except Exception:
                    pass
            finally:
                self.state_store.save_atomic(symbol, state)

    def _commodity_halted(self, symbol: str) -> bool:
        """XAU/XAG weekend halt: no NEW entries Fri 17:00 UTC -> Sun 21:00 UTC
        (= Sat 00:00 -> Mon 04:00 Asia/Bangkok, 3h before the Mon 07:00 open).
        Uses exchange.now_ms() so backtests replay the same gate."""
        c = self.cfg
        if not c.commodity_weekend_block:
            return False
        if not any(k in symbol.upper() for k in c.commodity_symbol_keywords):
            return False
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(self.exchange.now_ms() / 1000, tz=timezone.utc)
        wd, hr = dt.weekday(), dt.hour
        return ((wd == 4 and hr >= c.commodity_halt_hour_utc) or wd == 5
                or (wd == 6 and hr < c.commodity_resume_hour_utc))

    async def _look_for_entry(self, symbol, state, ind_15m, bias, macro_ctx, regime,
                              s15, s1h, zones, patterns, candle_ctx, liq, sd,
                              shock_locked, exchange_state) -> None:
        if self._commodity_halted(symbol):
            self.diag.record_rejection(symbol, [ReasonCode.REJECT_MARKET_CLOSED.value])
            self.diag.set_view(symbol, f"{_sym(symbol)} | NO TRADE | Reason=MARKET_CLOSED (weekend)")
            return
        pb = self.pullback.evaluate(symbol, state, ind_15m, bias, macro_ctx, regime,
                                    s15, s1h, zones, patterns, candle_ctx, liq, sd)
        mo = self.momentum.evaluate(symbol, state, ind_15m, bias, macro_ctx, regime,
                                    s15, s1h, zones, patterns, candle_ctx,
                                    shock_lockout=shock_locked)
        # shock: pullback still allowed but only at valid structure + reduced risk
        if shock_locked and pb is not None:
            pb.risk_modifier *= 0.7
        # module performance gate
        for cand_name, cand in (("pb", pb), ("mo", mo)):
            if cand is None:
                continue
            mod = self.perf.module_risk_modifier(cand.setup_type)
            if mod is None:
                self.perf.record_shadow(cand.setup_type, 0.0)   # counted; refined at close
                self.diag.record_rejection(symbol, [ReasonCode.REJECT_MODULE_PAUSED.value])
                if cand is pb:
                    pb = None
                else:
                    mo = None
            else:
                cand.risk_modifier *= mod

        candidate = self.selector.select(pb, mo, state)
        if candidate is None:
            self.diag.count("signals_evaluated")
            return

        account_state = await self.execution.get_account_state()
        market_rules = await self.execution.get_market_rules(symbol)
        open_positions = [OpenPositionInfo(p.symbol, p.direction,
                                           account_state["equity"] * self.cfg.risk_per_trade)
                          for p in await self.exchange.get_all_open_positions()]

        pr = self.portfolio.evaluate(candidate, account_state, open_positions)
        if not pr.valid:
            self.diag.record_rejection(symbol, pr.reason_codes)
            return

        atr = ind_15m.last_atr if ind_15m else 0.0
        market = {**market_rules, "atr": atr,
                  "last_price": exchange_state.last_price or candidate.entry_reference,
                  "spread_pct": exchange_state.spread_pct}
        plan = self.risk.build_trade_plan(candidate, account_state, market,
                                          portfolio_modifier=pr.risk_modifier)
        if not plan.is_valid:
            self.diag.record_rejection(symbol, plan.reason_codes)
            return

        eq = await self.exec_quality.evaluate(symbol, candidate, plan, market,
                                              now_ms=self.exchange.now_ms())
        if not eq.valid:
            self.diag.record_rejection(symbol, eq.reason_codes)
            return

        await self.notifier.signal(candidate, plan,
                                   self.cfg.risk_per_trade * plan.risk_modifier)
        opened = await self.execution.open_position(symbol, candidate, plan, state)
        if opened:
            self.diag.count("positions_opened")

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        c = self.cfg
        logger.info("DUAL ENTRY PRECISION V1.4 starting: %s (%s)", c.symbols,
                    "PAPER" if c.paper else "LIVE")
        if self.notifier.enabled:
            await self.notifier.info(
                f"🤖 *DUAL ENTRY PRECISION V1.4 started* "
                f"[{'PAPER' if c.paper else 'LIVE'}]\n"
                f"Symbols: `{', '.join(c.symbols)}`\n"
                f"Risk: `{c.risk_per_trade*100:.1f}%`  Max positions: `{c.max_positions}`\n"
                f"Engines: `FAST_PULLBACK + MOMENTUM` on 15M, 1H bias, 4H macro")

    async def run_forever(self) -> None:
        await self.start()
        while self._running:
            self.market_data.new_tick()
            for symbol in self.cfg.symbols:
                try:
                    await self.process_symbol(symbol)
                except Exception as e:      # never let one symbol kill the loop
                    logger.error("[%s] loop error: %s", symbol, e, exc_info=True)
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def stop(self) -> None:
        self._running = False
        await self.exchange.close()


def main() -> None:
    bot = Bot()
    try:
        asyncio.run(bot.run_forever())
    except KeyboardInterrupt:
        asyncio.run(bot.stop())


if __name__ == "__main__":
    main()
