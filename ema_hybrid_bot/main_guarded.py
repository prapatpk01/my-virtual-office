"""Guarded EMA Hybrid runtime.

Adds a hard one-position-per-symbol invariant on top of the existing EMA Hybrid
runtime.  Local state is not trusted as the sole source of truth: before any
new-entry scan the runtime checks both LONG and SHORT legs from the exchange
(PAPER or LIVE).  If a position exists but local state is missing, it is
adopted/reconciled and managed instead of opening another trade.
"""
from __future__ import annotations

import asyncio
import time

import main as runtime

_LOG = runtime._LOG


class Bot(runtime.Bot):
    def __init__(self):
        super().__init__()
        self._symbol_entry_locks = {symbol: asyncio.Lock() for symbol in self.cfg.symbols}

    @staticmethod
    def _f(value, default: float = 0.0) -> float:
        try:
            return float(value or default)
        except (TypeError, ValueError):
            return default

    async def _exchange_position_legs(self, symbol: str) -> list[tuple[str, float, dict]]:
        """Return confirmed open legs; retry zeros to avoid transient false-flat reads."""
        found: dict[str, tuple[float, dict]] = {}

        for attempt in range(2):
            for side in ("long", "short"):
                try:
                    amount = self._f(await self.client.fetch_position_amount(symbol, side))
                except Exception as exc:
                    _LOG.warning("[%s] position guard amount read failed side=%s: %s", symbol, side, exc)
                    amount = 0.0
                if amount > 0:
                    details = {}
                    try:
                        details = await self.client.fetch_position_details(symbol, side) or {}
                    except Exception:
                        pass
                    found[side] = (amount, details)
            if found:
                break
            if attempt == 0:
                await asyncio.sleep(0.20)

        # Final confirmation path: some adapters can report amount=0 briefly
        # while position-details already contains the live leg.
        if not found:
            for side in ("long", "short"):
                try:
                    details = await self.client.fetch_position_details(symbol, side) or {}
                except Exception:
                    continue
                amount = self._f(details.get("amount"))
                if amount > 0:
                    found[side] = (amount, details)

        return [(side, amount, details) for side, (amount, details) in found.items()]

    async def _adopt_missing_local_position(
        self, symbol: str, st: dict, legs: list[tuple[str, float, dict]]
    ) -> None:
        """Rebuild a blocking local position instead of permitting a duplicate entry."""
        if not legs or st.get("pos"):
            return

        # One-position-per-symbol policy: if hedge mode somehow has two legs,
        # quarantine the symbol and track the larger leg. No new order is allowed.
        side, amount, details = max(legs, key=lambda x: x[1])
        entry = self._f(details.get("entry_price") or details.get("entry"))
        sl = tp = 0.0
        try:
            sl_raw, tp_raw = await self.client.fetch_attached_stops(symbol, side)
            sl, tp = self._f(sl_raw), self._f(tp_raw)
        except Exception:
            pass

        st["pos"] = {
            "side": side,
            "entry": entry,
            "sl": sl,
            "initial_sl": sl,
            "tp": tp,
            "risk": abs(entry - sl) if entry > 0 and sl > 0 else 0.0,
            "amount": amount,
            "initial_amount": amount,
            "opened_ms": int(time.time() * 1000),
            "exit_bar": None,
            "best_price": entry,
            "lock_stage": 0,
            "setup": "RUNTIME_RECONCILED",
            "q_1h": 0.0,
            "room_pct": 0.0,
            "trigger": "ONE_POSITION_GUARD",
            "adopted": entry <= 0.0,
            "recovered": True,
            "runtime_reconciled": True,
            "recovery_quarantine": len(legs) > 1 or entry <= 0.0,
        }
        self._save_state()
        self._view[symbol] = (
            f"POSITION GUARD | existing {side.upper()} amount={amount:.8g} "
            f"adopted; duplicate entry blocked"
        )
        _LOG.warning(
            "[%s] ONE-POSITION GUARD adopted exchange position side=%s amount=%.8g entry=%.8g legs=%d",
            symbol, side, amount, entry, len(legs),
        )
        try:
            await self.tg.send_text(
                f"🛡️ *{symbol.split('/')[0]} duplicate-entry blocked*\n"
                f"Existing `{side.upper()}` position detected on exchange.\n"
                f"Amount `{amount:.8g}` | Entry `{entry:.6g}`\n"
                "Local state was reconciled; no additional position was opened."
            )
        except Exception:
            pass

    async def _process(self, symbol: str):
        """Serialize each symbol and verify exchange truth before every new entry."""
        lock = self._symbol_entry_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            st = self._sym_state(symbol)

            if st.get("pos"):
                await self._manage(symbol, st)
                return

            legs = await self._exchange_position_legs(symbol)
            if legs:
                await self._adopt_missing_local_position(symbol, st, legs)
                # Always manage/reconcile; never scan for a new entry in this cycle.
                if st.get("pos"):
                    await self._manage(symbol, st)
                return

            # Only a confirmed flat symbol is allowed to reach the strategy.
            await self._look_for_entry(symbol, st)

    async def start(self):
        await super().start()
        _LOG.info(
            "EMA Hybrid ONE-POSITION-PER-SYMBOL guard active: local state + exchange legs + per-symbol lock"
        )
        try:
            await self.tg.send_text(
                "🛡️ *EMA Hybrid Position Guard — ACTIVE*\n"
                "Rule: `maximum 1 open position per symbol`\n"
                "Before every entry: local state + exchange LONG/SHORT legs are checked.\n"
                "Pyramiding / duplicate BTC, DOGE, XAU, XAG, etc. is blocked."
            )
        except Exception:
            pass


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig in (runtime.signal.SIGINT, runtime.signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
