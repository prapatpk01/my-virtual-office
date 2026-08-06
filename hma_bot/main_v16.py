"""HMA Fast Structure V6 production runtime.

Keeps the proven OKX execution, restart reconciliation, native SL/TP and
Telegram/statistics infrastructure.  Only the active strategy and runtime
guards are replaced.  One path only; the counter-trend fallback is removed.
"""
from __future__ import annotations

import asyncio
import time

import main_v15 as v15
import strategy_v12 as S

_LOG = v15._LOG


class Bot(v15.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV12(self.cfg.strategy_config())
        self._closed_seen = {symbol: not bool((self.state.get(symbol) or {}).get("pos")) for symbol in self.cfg.symbols}
        self.post_close_cooldown_sec = 45 * 60
        # Every protection update (entry repair, restart recovery and stage
        # lock) must replace the old algo rather than append another one.
        self.client.move_sl_to_breakeven = self._replace_native_protection

    @staticmethod
    def _valid_trigger(value) -> bool:
        return value not in (None, "", "0", "0.0")

    async def _pending_protections(self, symbol: str, pos_side: str):
        """Return every active OKX TP/SL algo for one position leg."""
        market = self.client._exchange.market(symbol)
        inst_id = market["id"]
        found = {}
        for ord_type in ("oco", "conditional", "trigger", "move_order_stop"):
            try:
                response = await self.client._exchange.privateGetTradeAlgosPending({
                    "instId": inst_id,
                    "ordType": ord_type,
                })
                for order in (response or {}).get("data", []):
                    if str(order.get("posSide") or "") != pos_side:
                        continue
                    if not (
                        self._valid_trigger(order.get("slTriggerPx"))
                        or self._valid_trigger(order.get("tpTriggerPx"))
                    ):
                        continue
                    algo_id = str(order.get("algoId") or "")
                    if algo_id:
                        found[algo_id] = order
            except Exception as exc:
                _LOG.warning(
                    "[%s] pending protection query %s failed: %s",
                    symbol, ord_type, exc,
                )
        return inst_id, list(found.values())

    async def _cancel_protections(self, inst_id: str, orders) -> bool:
        """Cancel all supplied algos; never add a replacement after failure."""
        ok = True
        for order in orders:
            algo_id = str(order.get("algoId") or "")
            if not algo_id:
                continue
            try:
                await self.client._exchange.privatePostTradeCancelAlgos(
                    [{"algoId": algo_id, "instId": inst_id}]
                )
            except Exception as exc:
                ok = False
                _LOG.error("cancel protection %s failed: %s", algo_id, exc)
        return ok

    async def _replace_native_protection(
        self, symbol: str, pos_side: str, sl_price: float,
        remaining_amount: float, tp_price=None,
    ) -> bool:
        """Atomically converge one position leg to exactly one OCO order."""
        if self.cfg.paper:
            return True
        if (
            pos_side not in ("long", "short")
            or float(sl_price or 0) <= 0
            or float(tp_price or 0) <= 0
            or float(remaining_amount or 0) <= 0
        ):
            return False
        try:
            inst_id, old_orders = await self._pending_protections(symbol, pos_side)
            if old_orders and not await self._cancel_protections(inst_id, old_orders):
                _LOG.error(
                    "[%s] protection replace aborted: stale algo cancellation failed",
                    symbol,
                )
                return False

            contract_size = await self.client.contract_size(symbol)
            if not contract_size or contract_size <= 0:
                return False
            contracts = max(1, round(float(remaining_amount) / contract_size))
            request = {
                "instId": inst_id,
                "tdMode": self.client._margin_mode,
                "side": "sell" if pos_side == "long" else "buy",
                "posSide": pos_side,
                "sz": str(contracts),
                "ordType": "oco",
                "slTriggerPx": str(round(float(sl_price), 6)),
                "slOrdPx": "-1",
                "slTriggerPxType": "last",
                "tpTriggerPx": str(round(float(tp_price), 6)),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last",
            }
            await self.client._exchange.privatePostTradeOrderAlgo(request)

            active = []
            for _ in range(3):
                _, active = await self._pending_protections(symbol, pos_side)
                if active:
                    break
                await asyncio.sleep(0.5)
            if len(active) > 1:
                # Keep only the newest order if OKX exposed an older attached
                # algo after the initial queries.
                newest = max(
                    active,
                    key=lambda order: int(order.get("cTime") or 0),
                )
                extras = [order for order in active if order is not newest]
                if not await self._cancel_protections(inst_id, extras):
                    return False
                _, active = await self._pending_protections(symbol, pos_side)

            valid = (
                len(active) == 1
                and self._valid_trigger(active[0].get("slTriggerPx"))
                and self._valid_trigger(active[0].get("tpTriggerPx"))
            )
            if valid:
                _LOG.info(
                    "[%s] native protection converged to one OCO: SL %.8g TP %.8g",
                    symbol, sl_price, tp_price,
                )
            else:
                _LOG.error(
                    "[%s] protection verification expected 1 OCO, found %d",
                    symbol, len(active),
                )
            return valid
        except Exception as exc:
            _LOG.exception("[%s] native protection replace failed: %s", symbol, exc)
            return False

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"FAST-V6 POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            remaining = max(0, self._cooldown_until.get(symbol, 0) - time.time())
            if remaining > 0:
                self._view[symbol] = f"FAST-V6 COOLDOWN | {remaining / 60:.0f}m"
                return
            px = float(df15["close"].iloc[-1]) if len(df15) else 0.0
            self._view[symbol] = f"15M px={px:.6g} | {self.strat.entry_status(df4h, df1h, df15, df5)}"
        except Exception as exc:
            self._view[symbol] = f"FAST-V6 view error: {str(exc)[:140]}"

    async def _manage(self, symbol: str, st: dict):
        had_position = bool(st.get("pos"))
        await super()._manage(symbol, st)
        has_position = bool(st.get("pos"))
        if had_position and not has_position:
            self._cooldown_until[symbol] = time.time() + self.post_close_cooldown_sec
            self._closed_seen[symbol] = True
            _LOG.info("[%s] FAST-V6 post-close cooldown 45 minutes", symbol)

    async def _reconcile_startup(self):
        """Recover positions and repair missing exchange-native protection."""
        await super()._reconcile_startup()
        for symbol in self.cfg.symbols:
            st = self.state.get(symbol) or {}
            pos = st.get("pos") or {}
            side = str(pos.get("side") or "")
            entry = float(pos.get("entry") or 0.0)
            amount = float(pos.get("amount") or 0.0)
            if side not in ("long", "short") or entry <= 0 or amount <= 0:
                continue

            native_sl, native_tp = await self.client.fetch_attached_stops(symbol, side)
            sl = float(pos.get("sl") or native_sl or 0.0)
            tp = float(pos.get("tp") or native_tp or 0.0)
            if sl <= 0:
                sl = entry * (0.990 if side == "long" else 1.010)
            if tp <= 0:
                tp = entry * (1.012 if side == "long" else 0.988)

            # Always converge on startup, even when both prices were found;
            # fetch_attached_stops returns prices, not the number of algos.
            repaired = await self._replace_native_protection(
                symbol, side, sl, amount, tp_price=tp
            )
            if repaired:
                pos.update({
                    "sl": sl,
                    "initial_sl": sl,
                    "tp": tp,
                    "risk": abs(entry - sl),
                    "recovery_quarantine": False,
                })
                _LOG.warning(
                    "[%s] recovered position protection repaired: SL %.8g TP %.8g",
                    symbol, sl, tp,
                )
                await self.tg.send_text(
                    f"🛡️ `{symbol}` recovered {side.upper()} protection repaired\n"
                    f"SL `{sl:.6g}` | TP `{tp:.6g}`"
                )
            else:
                _LOG.error("[%s] recovered position remains unprotected", symbol)
                await self.tg.send_text(
                    f"🚨 `{symbol}` recovered position has no native SL/TP and repair failed. "
                    "FAST-V6 is closing it for safety."
                )
                await self._close_market(symbol, st, "RECOVERY_PROTECTION_FAILED")
        self._save_state()

    async def _look_for_entry(self, symbol: str, st: dict):
        """Use the inherited entry flow, then verify native protection."""
        had_position = bool(st.get("pos"))
        await super()._look_for_entry(symbol, st)
        pos = st.get("pos") or {}
        if had_position or not pos:
            return

        side = str(pos.get("side") or "")
        sl = float(pos.get("sl") or 0.0)
        tp = float(pos.get("tp") or 0.0)
        amount = float(pos.get("amount") or 0.0)
        if side not in ("long", "short") or sl <= 0 or tp <= 0 or amount <= 0:
            _LOG.error("[%s] invalid local protection plan after entry", symbol)
            await self._close_market(symbol, st, "PROTECTION_PLAN_INVALID")
            return

        # Converge unconditionally. Price lookup alone cannot tell whether
        # OKX currently has one protection or several duplicates.
        repaired = await self._replace_native_protection(
            symbol, side, sl, amount, tp_price=tp
        )
        if repaired:
            _LOG.info("[%s] post-entry protection verified as one OCO", symbol)
            return

        _LOG.error("[%s] native SL/TP verification failed; closing position", symbol)
        await self.tg.send_text(
            f"🚨 `{symbol}` native SL/TP could not be verified. "
            "FAST-V6 is closing the position for safety."
        )
        await self._close_market(symbol, st, "NATIVE_PROTECTION_FAILED")

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info("=== HMA FAST STRUCTURE V6 [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===",
                  "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols,
                  self.cfg.margin_per_position_usd, self.cfg.leverage,
                  self.cfg.max_positions, balance)
        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            mode = "PAPER" if self.cfg.paper else "LIVE"
            await self.tg.send_text(
                f"⚡ *HMA Fast Structure V6 — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}` "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}`\n\n"
                "Pipeline: `1H direction → relaxed Q guard → direct closed-15M entry`\n"
                "Entry: `EMA8/13 cross OR HMA16 flip OR EMA13 pullback reclaim`\n"
                "4H: `soft context only` | No mandatory Sentinel S/R | No counter-trend fallback\n"
                "Quality defaults: `Q≥42`; only severe ADX/CHOP or opposing DMI blocks\n"
                "Anti-chase: `≤1.10 ATR from EMA13`\n"
                "Risk: native OKX `SL≤1.0%` and `TP 1.2%`; minimum `1.05R`\n"
                "Re-entry: `45-minute cooldown after every close`\n"
                "Recovery: existing positions and native SL/TP reconciled after restart"
            )
        _LOG.info("HMA Fast Structure V6 startup complete")


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            signal_module = v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal
            loop.add_signal_handler(getattr(signal_module, sig_name), lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
